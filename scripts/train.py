#!/usr/bin/env python3
"""
Main training script for Medical/Biomedical NER.

Usage
-----
# Train PubMedBERT on NCBI Disease corpus
python scripts/train.py --model pubmedbert --dataset ncbi_disease

# Train BioBERT on BC5CDR
python scripts/train.py --model biobert --dataset bc5cdr --epochs 15

# Train with CRF layer
python scripts/train.py --model pubmedbert --dataset ncbi_disease --use-crf

# Custom model from HuggingFace
python scripts/train.py --model-path microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \\
    --dataset jnlpba --lr 3e-5
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
from src.data.dataset_loader import load_ner_dataset, get_label_maps
from src.data.preprocessing import preprocess_dataset
from src.models.ner_model import build_ner_model
from src.training.trainer import build_trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a biomedical NER model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="pubmedbert",
        choices=list(MODEL_CONFIGS.keys()),
        help="Pre-trained model key",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Override model with a HuggingFace model id or local path",
    )
    parser.add_argument(
        "--use-crf", action="store_true",
        help="Add a CRF layer on top of the transformer",
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="ncbi_disease",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to train on",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum sequence length for tokenization",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine", "constant_with_warmup"])

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Misc
    parser.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = NERConfig(
        model_key=args.model,
        model_name_or_path=args.model_path,
        dataset_key=args.dataset,
        max_seq_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.grad_accum,
        early_stopping_patience=args.patience,
        label_smoothing_factor=args.label_smoothing,
        lr_scheduler_type=args.scheduler,
        fp16=args.fp16 and not args.no_fp16 and torch.cuda.is_available(),
        use_crf=args.use_crf,
        output_dir=args.output_dir,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        seed=args.seed,
        use_wandb=args.wandb,
    )

    logger.info("=" * 60)
    logger.info("MEDICAL NER TRAINING")
    logger.info("=" * 60)
    logger.info("Model: %s (%s)", config.model_key, config.model_name_or_path)
    logger.info("Dataset: %s", config.dataset_key)
    logger.info("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)

    set_seed(config.seed)

    # 1. Load dataset
    dataset, label_list = load_ner_dataset(config.dataset_key)
    label2id, id2label = get_label_maps(label_list)
    logger.info("Labels (%d): %s", len(label_list), label_list)

    # 2. Build model and tokenizer
    model, tokenizer = build_ner_model(
        config.model_name_or_path,
        label_list,
        use_crf=config.use_crf,
    )

    # 3. Preprocess dataset
    tokenized_dataset = preprocess_dataset(
        dataset, tokenizer, label2id,
        max_length=config.max_seq_length,
    )

    # Determine eval split name
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
        # Dataset has only a 'train' split — auto-split 90/10
        logger.info(
            "No validation or test split found. Auto-splitting train into "
            "90%% train / 10%% eval."
        )
        split = tokenized_dataset["train"].train_test_split(
            test_size=0.1, seed=config.seed,
        )
        train_ds = split["train"]
        eval_ds = split["test"]
        eval_split = "auto_eval"

    # 4. Build trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        label_list=label_list,
    )

    # 5. Train
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training complete.")

    # 6. Save best model
    best_model_dir = os.path.join(config.output_dir, config.experiment_name, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info("Best model saved to: %s", best_model_dir)

    # 7. Final evaluation
    logger.info("Running final evaluation on %s split...", eval_split)
    metrics = trainer.evaluate()
    logger.info("Final metrics: %s", metrics)

    # Also evaluate on test set if separate from validation
    if "test" in tokenized_dataset and eval_split != "test":
        logger.info("Running evaluation on test split...")
        test_metrics = trainer.evaluate(tokenized_dataset["test"], metric_key_prefix="test")
        logger.info("Test metrics: %s", test_metrics)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
