#!/usr/bin/env python3
"""
Prediction script for Medical Coding NER.

Supports the full pipeline: shorthand expansion -> NER -> negation detection.

Usage
-----
# Interactive mode with full pipeline
python scripts/predict.py --model-path outputs/pubmedbert_ncbi_disease/best_model

# Predict a single text (with negation and shorthand)
python scripts/predict.py --model-path outputs/pubmedbert_ncbi_disease/best_model \
    --text "Pt denies cp or sob. Hx of dm2 and htn."

# Disable shorthand expansion
python scripts/predict.py --model-path outputs/pubmedbert_ncbi_disease/best_model \
    --text "Patient denies chest pain." --no-shorthand

# Predict on a file (one sentence per line)
python scripts/predict.py --model-path outputs/pubmedbert_ncbi_disease/best_model \
    --input-file data/test_sentences.txt --output-file results.json
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clinical.pipeline import MedicalCodingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Medical Coding NER predictions with negation detection",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--text", type=str, default=None, help="Single text to process")
    parser.add_argument("--input-file", type=str, default=None, help="File with one sentence per line")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--no-shorthand", action="store_true", help="Disable shorthand expansion")
    parser.add_argument("--no-negation", action="store_true", help="Disable negation detection")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = MedicalCodingPipeline(
        model_path=args.model_path,
        expand_shorthand=not args.no_shorthand,
        detect_negation=not args.no_negation,
        device=args.device,
    )

    if args.text:
        entities = pipeline.process(args.text)
        print(pipeline.format_output(args.text, entities))

    elif args.input_file:
        with open(args.input_file) as f:
            texts = [line.strip() for line in f if line.strip()]

        all_results = []
        for text in texts:
            entities = pipeline.process(text)
            all_results.append({
                "text": text,
                "entities": [e.to_dict() for e in entities],
            })
            print(pipeline.format_output(text, entities))
            print()

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info("Results saved to %s", args.output_file)

    else:
        print("Medical Coding NER - Interactive Mode")
        print("Features: shorthand expansion + NER + negation detection")
        print("Type a sentence and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue

            entities = pipeline.process(text)
            print(pipeline.format_output(text, entities))
            print()


if __name__ == "__main__":
    main()
