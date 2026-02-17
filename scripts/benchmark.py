#!/usr/bin/env python3
"""
Benchmark script: compare multiple models across multiple datasets.

Usage
-----
# Run full benchmark (all models x all datasets)
python scripts/benchmark.py

# Benchmark specific models and datasets
python scripts/benchmark.py --models pubmedbert biobert --datasets ncbi_disease bc5cdr

# Quick benchmark with fewer epochs
python scripts/benchmark.py --epochs 3 --eval-steps 50
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

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
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark NER models across datasets")
    parser.add_argument(
        "--models", nargs="+", default=["pubmedbert", "biobert", "bio_clinicalbert"],
        choices=list(MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["ncbi_disease", "bc5cdr", "jnlpba"],
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_single_experiment(model_key, dataset_key, args):
    """Train and evaluate a single model-dataset combination."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s on %s", model_key, dataset_key)
    logger.info("=" * 60)

    set_seed(args.seed)

    config = NERConfig(
        model_key=model_key,
        dataset_key=dataset_key,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    # Load data
    dataset, label_list = load_ner_dataset(dataset_key)
    label2id, id2label = get_label_maps(label_list)

    # Build model
    model, tokenizer = build_ner_model(config.model_name_or_path, label_list)

    # Preprocess
    tokenized = preprocess_dataset(dataset, tokenizer, label2id)
    eval_split = "validation" if "validation" in tokenized else "test"

    # Train
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        label_list=label_list,
    )
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()

    # Test set evaluation if available
    test_metrics = {}
    if "test" in tokenized and eval_split != "test":
        test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    result = {
        "model": model_key,
        "dataset": dataset_key,
        "eval_precision": metrics.get("eval_precision", 0),
        "eval_recall": metrics.get("eval_recall", 0),
        "eval_f1": metrics.get("eval_f1", 0),
    }
    if test_metrics:
        result["test_precision"] = test_metrics.get("test_precision", 0)
        result["test_recall"] = test_metrics.get("test_recall", 0)
        result["test_f1"] = test_metrics.get("test_f1", 0)

    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for model_key in args.models:
        for dataset_key in args.datasets:
            try:
                result = run_single_experiment(model_key, dataset_key, args)
                results.append(result)
                logger.info("Result: %s", json.dumps(result, indent=2))
            except Exception as e:
                logger.error("FAILED: %s on %s: %s", model_key, dataset_key, e)
                results.append({
                    "model": model_key,
                    "dataset": dataset_key,
                    "error": str(e),
                })

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Dataset':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<20} {r['dataset']:<18} {'ERROR':>30}")
        else:
            key = "test_f1" if "test_f1" in r else "eval_f1"
            p_key = key.replace("f1", "precision")
            r_key = key.replace("f1", "recall")
            print(
                f"{r['model']:<20} {r['dataset']:<18} "
                f"{r.get(p_key, 0):>10.4f} {r.get(r_key, 0):>10.4f} {r.get(key, 0):>10.4f}"
            )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"benchmark_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
