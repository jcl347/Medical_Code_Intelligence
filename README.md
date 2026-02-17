# Medical Code Intelligence

**State-of-the-art Medical Coding NER system** for extracting and classifying medical entities from clinical text, with built-in negation detection and physician shorthand expansion.

## Features

- **Transformer-based NER** — Fine-tune PubMedBERT, BioBERT, Bio_ClinicalBERT, SciBERT, or GatorTron on biomedical NER datasets
- **Public datasets** — Built-in loaders for NCBI Disease, BC5CDR, BC2GM, JNLPBA, LINNAEUS, and d4data/biomedical-ner-all (all via HuggingFace Hub)
- **Negation detection** — ConText/NegEx-style algorithm with 70+ trigger patterns, scope termination, pseudo-negation handling, and support for possibility/historical/family context
- **Physician shorthand expansion** — 300+ medical abbreviations with context-sensitive disambiguation and character offset tracking
- **Optional CRF layer** — Conditional Random Field for structured label decoding
- **Entity-level evaluation** — Strict entity-level precision/recall/F1 (seqeval or built-in fallback)
- **Error analysis** — Automated boundary error, type confusion, and false positive/negative analysis
- **Unified pipeline** — Single API chains shorthand expansion → NER → negation detection

## Project Structure

```
Medical_Code_Intelligence/
├── configs/
│   └── ner_config.py          # Model, dataset, and training configs
├── src/
│   ├── clinical/
│   │   ├── shorthand.py       # Physician abbreviation expansion (300+ terms)
│   │   ├── negation.py        # NegEx/ConText negation detection
│   │   └── pipeline.py        # Unified medical coding NER pipeline
│   ├── data/
│   │   ├── dataset_loader.py  # HuggingFace dataset loaders + normalisation
│   │   ├── preprocessing.py   # Subword tokenization & label alignment
│   │   └── data_utils.py      # Data collator, sliding window splitting
│   ├── models/
│   │   ├── ner_model.py       # Model builder (AutoModelForTokenClassification)
│   │   └── crf_model.py       # Optional CRF layer for structured prediction
│   ├── training/
│   │   ├── trainer.py         # HuggingFace Trainer with SOTA config
│   │   └── callbacks.py       # Early stopping with logging
│   ├── evaluation/
│   │   ├── metrics.py         # Entity-level F1 (seqeval + fallback)
│   │   └── error_analysis.py  # Detailed error categorisation
│   └── inference/
│       └── predictor.py       # Inference pipeline with batching
├── scripts/
│   ├── train.py               # Training CLI
│   ├── evaluate.py            # Evaluation CLI with error analysis
│   ├── predict.py             # Prediction CLI (interactive + batch)
│   └── benchmark.py           # Multi-model x multi-dataset benchmarking
├── tests/                     # 81 tests covering all modules
├── requirements.txt
└── setup.py
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train a Model

```bash
# Fine-tune PubMedBERT on NCBI Disease corpus
python scripts/train.py --model pubmedbert --dataset ncbi_disease

# Fine-tune BioBERT on BC5CDR (chemicals + diseases)
python scripts/train.py --model biobert --dataset bc5cdr --epochs 15 --lr 3e-5

# Train with CRF layer
python scripts/train.py --model pubmedbert --dataset ncbi_disease --use-crf
```

### Run Predictions (with negation + shorthand)

```bash
# Interactive mode
python scripts/predict.py --model-path outputs/pubmedbert_ncbi_disease/best_model

# Single text
python scripts/predict.py --model-path outputs/pubmedbert_ncbi_disease/best_model \
    --text "Pt denies cp or sob. Hx of dm2 and htn."
```

Example output:
```
Pt denies cp or sob. Hx of dm2 and htn.

  [chest pain](Disease, NEGATED, trigger="denies", from="cp", score=0.950)
  [shortness of breath](Disease, NEGATED, trigger="denies", from="sob", score=0.930)
  [type 2 diabetes mellitus](Disease, HISTORICAL, trigger="hx", from="dm2", score=0.970)
  [hypertension](Disease, HISTORICAL, trigger="hx", from="htn", score=0.960)
```

### Evaluate

```bash
python scripts/evaluate.py \
    --model-path outputs/pubmedbert_ncbi_disease/best_model \
    --dataset ncbi_disease --error-analysis
```

### Benchmark Multiple Models

```bash
python scripts/benchmark.py \
    --models pubmedbert biobert bio_clinicalbert \
    --datasets ncbi_disease bc5cdr jnlpba
```

## Python API

```python
from src.clinical.pipeline import MedicalCodingPipeline

# Full pipeline: shorthand expansion → NER → negation detection
pipeline = MedicalCodingPipeline(
    model_path="outputs/pubmedbert_ncbi_disease/best_model",
    expand_shorthand=True,
    detect_negation=True,
)

results = pipeline("Pt denies cp. Dx: dm2, htn.")
for entity in results:
    print(f"{entity.text} [{entity.label}] — {entity.negation}")
    # chest pain [Disease] — negated
    # type 2 diabetes mellitus [Disease] — affirmed
    # hypertension [Disease] — affirmed
```

### Negation Detection Only

```python
from src.clinical.negation import NegationDetector

detector = NegationDetector()
entities = [
    {"text": "fever", "label": "Symptom", "start": 15, "end": 20},
    {"text": "cough", "label": "Symptom", "start": 29, "end": 34},
]
annotated = detector.annotate_entities("Patient denies fever but has cough", entities)
# annotated[0]["negation"] == "negated"
# annotated[1]["negation"] == "affirmed"
```

### Shorthand Expansion Only

```python
from src.clinical.shorthand import ShorthandExpander

expander = ShorthandExpander()
text = expander.expand("pt c/o sob, htn well controlled on meds")
# "patient complaining of shortness of breath, hypertension well controlled on meds"
```

## Supported Models

| Key | Model | Description |
|-----|-------|-------------|
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | SOTA on BLURB benchmark |
| `biobert` | `dmis-lab/biobert-v1.1` | Pre-trained on PubMed abstracts |
| `bio_clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | Pre-trained on MIMIC-III clinical notes |
| `scibert` | `allenai/scibert_scivocab_uncased` | Pre-trained on scientific papers |
| `gatortron-base` | `UFNLP/gatortron-base` | Pre-trained on 90B words of clinical text |

## Supported Datasets

| Key | Source | Entity Types |
|-----|--------|-------------|
| `ncbi_disease` | NCBI Disease Corpus | Disease |
| `bc5cdr` | BioCreative V CDR | Chemical, Disease |
| `bc2gm` | BioCreative II GM | Gene |
| `jnlpba` | JNLPBA Shared Task | Protein, DNA, RNA, Cell_line, Cell_type |
| `biomedical_ner_all` | d4data combined | 10+ entity types |

## Training Best Practices

This system implements the following SOTA practices:

- **Subword label alignment** — BIO labels aligned to first subword token; continuation subwords receive ignore label (-100)
- **Dynamic padding** — Pad to longest-in-batch rather than max_length for efficiency
- **Learning rate warmup** — Linear warmup over 10% of training steps
- **Linear LR decay** — After warmup, decay to zero
- **Mixed precision (FP16)** — When GPU available, for 2x training speed
- **Gradient accumulation** — Configurable for effective larger batch sizes
- **Early stopping** — Patience-based on validation entity-level F1
- **Weight decay** — AdamW with 0.01 weight decay (no decay on bias/LayerNorm)
- **Gradient clipping** — Max norm 1.0

## Negation Detection

The negation module implements a ConText/NegEx-style algorithm with:

- **70+ pre-negation triggers** (forward-scoping): "no", "denies", "without", "negative for", "ruled out", etc.
- **20+ post-negation triggers** (backward-scoping): "not found", "absent", "negative", etc.
- **Pseudo-negation handling**: "no change", "gram negative" are correctly NOT treated as negation
- **Scope termination**: Conjunctions ("but", "however"), sentence boundaries, and section headers stop scope propagation
- **Contextual status**: Beyond negation, also detects possibility ("possible", "suspect"), historical ("history of", "prior"), and family context ("family history of")

## Tests

```bash
python -m pytest tests/ -v
```

81 tests covering shorthand expansion, negation detection, pipeline integration, tokenization alignment, label mapping, and sliding window splitting.
