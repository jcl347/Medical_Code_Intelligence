#!/usr/bin/env python3
"""
Evaluation script for trained Medical NER models.

Usage
-----
# Evaluate a fine-tuned model on NCBI Disease test set
python scripts/evaluate.py --model-path outputs/pubmedbert_ncbi_disease/best_model \\
    --dataset ncbi_disease

# With detailed error analysis
python scripts/evaluate.py --model-path outputs/pubmedbert_ncbi_disease/best_model \\
    --dataset ncbi_disease --error-analysis
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from configs.ner_config import DATASET_CONFIGS
from src.data.dataset_loader import load_ner_dataset, get_label_maps
from src.data.preprocessing import preprocess_dataset
from src.data.data_utils import create_data_collator
from src.evaluation.metrics import compute_ner_metrics, build_compute_metrics_fn
from src.evaluation.error_analysis import analyse_errors, print_error_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained NER model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument(
        "--dataset", type=str, default="ncbi_disease",
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--error-analysis", action="store_true", help="Run detailed error analysis")
    parser.add_argument("--output-file", type=str, default=None, help="Save results to JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    logger.info("Loading model from '%s'...", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    label_list = [id2label[i] for i in range(len(id2label))]
    label2id, _ = get_label_maps(label_list)

    # Load and preprocess dataset
    dataset, _ = load_ner_dataset(args.dataset)
    tokenized = preprocess_dataset(dataset, tokenizer, label2id, max_length=args.max_length)

    split = args.split
    if split not in tokenized:
        available = list(tokenized.keys())
        logger.warning("Split '%s' not found. Available: %s. Using '%s'.", split, available, available[-1])
        split = available[-1]

    eval_dataset = tokenized[split]
    data_collator = create_data_collator(tokenizer)

    # Run predictions
    logger.info("Running predictions on '%s' split (%d examples)...", split, len(eval_dataset))
    from transformers import Trainer, TrainingArguments

    eval_args = TrainingArguments(
        output_dir="/tmp/eval",
        per_device_eval_batch_size=args.batch_size,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(label_list),
    )

    results = trainer.evaluate(eval_dataset)
    logger.info("Results: %s", results)

    # Detailed evaluation with per-entity report
    predictions_output = trainer.predict(eval_dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    preds = np.argmax(logits, axis=-1)

    metrics = compute_ner_metrics(preds, labels, label_list)
    logger.info("\n%s", metrics["report"])

    # Error analysis
    if args.error_analysis:
        logger.info("Running error analysis...")
        raw_dataset = dataset[split]
        tokens_list = raw_dataset["tokens"]
        true_labels_list = raw_dataset["ner_labels"]

        # Get predicted labels for original tokens
        pred_labels_list = []
        for pred_seq, label_seq in zip(preds, labels):
            pred_labels = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                pred_labels.append(label_list[p] if p < len(label_list) else "O")
            pred_labels_list.append(pred_labels)

        # Trim to match lengths (tokenization may differ)
        trimmed_tokens = []
        trimmed_true = []
        trimmed_pred = []
        for tokens, true, pred in zip(tokens_list, true_labels_list, pred_labels_list):
            min_len = min(len(tokens), len(true), len(pred))
            trimmed_tokens.append(tokens[:min_len])
            trimmed_true.append(true[:min_len])
            trimmed_pred.append(pred[:min_len])

        analysis = analyse_errors(trimmed_tokens, trimmed_true, trimmed_pred)
        report = print_error_report(analysis)
        logger.info("\n%s", report)

    # Save results
    if args.output_file:
        output = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "split": split,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", args.output_file)


if __name__ == "__main__":
    main()
