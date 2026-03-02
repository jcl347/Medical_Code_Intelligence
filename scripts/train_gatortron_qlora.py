#!/usr/bin/env python3
"""
QLoRA fine-tuning for GatorTron on biomedical NER.

Fine-tunes GatorTron (345M+ params, pre-trained on 90B words of clinical
text) using QLoRA — 4-bit quantization + LoRA adapters — reducing GPU
memory by ~75% compared to full fine-tuning.

This script is a dedicated command for GatorTron QLoRA training because
the model's size requires specialized memory management (4-bit loading,
gradient checkpointing, LoRA adapters) that differs from standard training.

Usage
-----
# Default: GatorTron-base with QLoRA on ICD NER dataset
python scripts/train_gatortron_qlora.py

# Custom LoRA rank and learning rate
python scripts/train_gatortron_qlora.py --lora-rank 32 --lora-alpha 64 --lr 1e-4

# Full GatorTron (if available on HuggingFace) with larger batch
python scripts/train_gatortron_qlora.py --model UFNLP/gatortron-large --batch-size 4

# With adversarial training
python scripts/train_gatortron_qlora.py --adversarial

# Dry run: check model loads and parameter counts
python scripts/train_gatortron_qlora.py --dry-run
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import set_seed

from configs.ner_config import NERConfig, DATASET_CONFIGS, MODEL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for GatorTron on biomedical NER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model — accepts both config keys and HuggingFace model IDs
    parser.add_argument(
        "--model", type=str, default="UFNLP/gatortron-base",
        help=(
            "HuggingFace model ID, local path, or config key. "
            "Supports shorthand keys: gatortron-base (345M), "
            "gatortron-medium (~1B), gatortron-large (~3.9B). "
            "Default: UFNLP/gatortron-base"
        ),
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="icd_ner",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to train on",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum sequence length for tokenization",
    )

    # QLoRA parameters
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (8-64)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target-modules", type=str, nargs="+", default=None,
        help="Model modules for LoRA adapters (default: query value)",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization (use full precision + LoRA only)",
    )
    parser.add_argument(
        "--compute-dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype for quantized operations",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (higher for LoRA)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=float, default=0.1, help="Warmup steps (float in [0,1) = ratio; int = exact steps)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "constant_with_warmup"])

    # Adversarial training (compatible with QLoRA)
    parser.add_argument(
        "--adversarial", action="store_true",
        help="Enable adversarial training (FGM on embeddings)",
    )
    parser.add_argument(
        "--adv-method", type=str, default="fgm", choices=["fgm", "pgd"],
        help="Adversarial method",
    )
    parser.add_argument("--adv-epsilon", type=float, default=None)

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load model and print stats without training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("GATORTRON QLoRA TRAINING")
    logger.info("=" * 60)
    logger.info("Model: %s", args.model)
    logger.info("Dataset: %s", args.dataset)
    logger.info("QLoRA: rank=%d, alpha=%d, dropout=%.2f, 4-bit=%s",
                args.lora_rank, args.lora_alpha, args.lora_dropout, not args.no_4bit)
    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)

    if not torch.cuda.is_available() and not args.no_4bit:
        logger.warning(
            "CUDA not available. 4-bit quantization requires a GPU. "
            "Use --no-4bit for CPU-only LoRA training (much slower)."
        )

    # Resolve model key to HuggingFace ID if a config key was provided
    if args.model in MODEL_CONFIGS:
        resolved = MODEL_CONFIGS[args.model]["model_name"]
        logger.info("Resolved model key '%s' -> '%s'", args.model, resolved)
        args.model = resolved

    set_seed(args.seed)

    # 1. Load dataset
    from src.data.dataset_loader import load_ner_dataset, get_label_maps

    dataset, label_list = load_ner_dataset(args.dataset)
    label2id, id2label = get_label_maps(label_list)
    logger.info("Labels (%d): %s", len(label_list), label_list)

    # 2. Build QLoRA model
    from src.training.qlora import build_qlora_model

    model, tokenizer = build_qlora_model(
        args.model,
        label_list,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        use_4bit=not args.no_4bit,
        bnb_4bit_compute_dtype=args.compute_dtype,
    )

    if args.dry_run:
        logger.info("Dry run complete. Model loaded successfully.")
        return

    # 3. Preprocess dataset
    from src.data.preprocessing import preprocess_dataset

    tokenized_dataset = preprocess_dataset(
        dataset, tokenizer, label2id,
        max_length=args.max_length,
    )

    # Determine eval split
    if "validation" in tokenized_dataset:
        eval_split = "validation"
    elif "test" in tokenized_dataset:
        eval_split = "test"
    else:
        eval_split = None

    if eval_split is not None:
        logger.info("Using '%s' split for evaluation.", eval_split)
        train_ds = tokenized_dataset["train"]
        eval_ds = tokenized_dataset[eval_split]
    else:
        logger.info("No eval split found. Auto-splitting 90/10.")
        split = tokenized_dataset["train"].train_test_split(
            test_size=0.1, seed=args.seed,
        )
        train_ds = split["train"]
        eval_ds = split["test"]
        eval_split = "auto_eval"

    # 4. Build NERConfig for trainer
    config = NERConfig(
        model_key="gatortron-base",
        model_name_or_path=args.model,
        dataset_key=args.dataset,
        max_seq_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.grad_accum,
        early_stopping_patience=args.patience,
        lr_scheduler_type=args.scheduler,
        fp16=False,  # QLoRA handles precision via BitsAndBytes
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        seed=args.seed,
        use_wandb=args.wandb,
        use_adversarial_training=args.adversarial,
        adv_method=args.adv_method,
        adv_epsilon=args.adv_epsilon,
    )

    # Override experiment name for QLoRA
    model_short = args.model.split("/")[-1]
    config.experiment_name = f"{model_short}_qlora_{args.dataset}"

    # 5. Build trainer
    from src.training.trainer import build_trainer

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        label_list=label_list,
    )

    # 6. Train
    logger.info("Starting QLoRA training...")
    train_result = trainer.train()
    logger.info("Training complete.")

    # 7. Save — save LoRA adapters (small) + full merged model
    best_model_dir = os.path.join(
        config.output_dir, config.experiment_name, "best_model",
    )

    # Save LoRA adapters (few MB)
    adapter_dir = os.path.join(
        config.output_dir, config.experiment_name, "lora_adapters",
    )
    model.save_pretrained(adapter_dir)
    logger.info("LoRA adapters saved to: %s", adapter_dir)

    # Merge adapters into base model and save full model for inference
    try:
        merged = model.merge_and_unload()
        merged.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
        logger.info("Merged model saved to: %s", best_model_dir)
    except Exception as e:
        logger.warning(
            "Could not merge LoRA adapters: %s. "
            "Use the adapter directory for inference with PEFT.",
            e,
        )
        # Save tokenizer alongside adapters as fallback
        tokenizer.save_pretrained(adapter_dir)

    # 8. Final evaluation
    logger.info("Running final evaluation on %s split...", eval_split)
    metrics = trainer.evaluate()
    logger.info("Final metrics: %s", metrics)

    if "test" in tokenized_dataset and eval_split != "test":
        logger.info("Running evaluation on test split...")
        test_metrics = trainer.evaluate(
            tokenized_dataset["test"], metric_key_prefix="test",
        )
        logger.info("Test metrics: %s", test_metrics)

    logger.info("=" * 60)
    logger.info("QLORA TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(
        "To run predictions:\n"
        "  python scripts/predict.py --model-path %s --text \"your text here\"",
        best_model_dir,
    )


if __name__ == "__main__":
    main()
