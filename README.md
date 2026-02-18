# Medical Code Intelligence

**Data-driven Medical Coding NER system** for extracting and classifying medical entities from clinical text, with learned assertion detection, physician shorthand expansion, and ICD-10-CM entity linking.

## Features

- **Transformer-based NER** — Fine-tune PubMedBERT, BioBERT, Bio_ClinicalBERT, SciBERT, or GatorTron on biomedical NER datasets
- **Public datasets** — Built-in loaders for NCBI Disease, BC5CDR, BC2GM, JNLPBA, LINNAEUS, d4data/biomedical-ner-all, and knowledgator/biomed_NER (all via HuggingFace Hub)
- **Data-driven ICD-10-CM entity linking** — Loads the full 51K ICD-10-CM code set from [atta00/icd10-codes](https://huggingface.co/datasets/atta00/icd10-codes) (MIT licensed), builds a TF-IDF character n-gram index following [SciSpacy's](https://github.com/allenai/scispacy) EntityLinker architecture for fast entity-to-code matching
- **Transformer-based assertion detection** — Wraps [bvanaken/clinical-assertion-negation-bert](https://huggingface.co/bvanaken/clinical-assertion-negation-bert) (ClinicalBERT fine-tuned on i2b2) for learned PRESENT/ABSENT/POSSIBLE classification, as an alternative to rule-based NegEx/ConText
- **Rule-based negation detection** — ConText/NegEx-style algorithm with 70+ trigger patterns, scope termination, pseudo-negation handling, and support for possibility/historical/family context
- **Physician shorthand expansion** — 300+ medical abbreviations with context-sensitive disambiguation and character offset tracking
- **Span-to-BIO conversion** — Automatic conversion of character-offset annotations to BIO tag sequences for datasets that use span format
- **Optional CRF layer** — Conditional Random Field for structured label decoding
- **Entity-level evaluation** — Strict entity-level precision/recall/F1 (seqeval or built-in fallback)
- **Error analysis** — Automated boundary error, type confusion, and false positive/negative analysis
- **Unified pipeline** — Single API chains shorthand expansion → NER → negation/assertion → ICD coding

## Project Structure

```
Medical_Code_Intelligence/
├── configs/
│   └── ner_config.py          # Model, dataset, and training configs
├── src/
│   ├── clinical/
│   │   ├── shorthand.py       # Physician abbreviation expansion (300+ terms)
│   │   ├── negation.py        # NegEx/ConText rule-based negation detection
│   │   ├── assertion.py       # Transformer-based assertion classifier (bvanaken model)
│   │   ├── icd_codes.py       # Data-driven ICD-10-CM TF-IDF entity linker
│   │   ├── _icd_fallback.py   # Minimal offline fallback codes (~45 high-frequency)
│   │   └── pipeline.py        # Unified medical coding NER pipeline
│   ├── data/
│   │   ├── dataset_loader.py  # HuggingFace dataset loaders + span→BIO conversion
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
├── tests/                     # 185 tests covering all modules
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

# Train on span-annotated biomedical NER (24 ICD-relevant entity types)
python scripts/train.py --model pubmedbert --dataset biomed_ner

# Fine-tune BioBERT on BC5CDR (chemicals + diseases)
python scripts/train.py --model biobert --dataset bc5cdr --epochs 15 --lr 3e-5

# Train with CRF layer
python scripts/train.py --model pubmedbert --dataset ncbi_disease --use-crf
```

### Run Predictions (with negation + shorthand + ICD codes)

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
    --datasets ncbi_disease bc5cdr jnlpba biomed_ner
```

## Available Datasets & How to Use Them

### NER Training Datasets

These datasets provide token- or span-level entity annotations for training NER models.

| Key | Source | Entity Types | Size |
|-----|--------|-------------|------|
| `ncbi_disease` | NCBI Disease Corpus | Disease | 6.9K sentences |
| `bc5cdr` | BioCreative V CDR | Chemical, Disease | 1.5K abstracts |
| `bc2gm` | BioCreative II GM | Gene | 20K sentences |
| `jnlpba` | JNLPBA Shared Task | Protein, DNA, RNA, Cell_line, Cell_type | 22K sentences |
| `biomedical_ner_all` | d4data combined | 10+ entity types | 25K+ samples |
| `biomed_ner` | knowledgator/biomed_NER | 24 types (DISORDER, MEDICAL_PROCEDURE, CLINICAL_DRUG, ...) | Span-annotated |

**Train on a specific dataset:**

```bash
# Disease NER (most common starting point)
python scripts/train.py --model pubmedbert --dataset ncbi_disease

# Chemical + Disease NER
python scripts/train.py --model biobert --dataset bc5cdr

# Broad biomedical NER with 24 entity types (ICD-relevant)
python scripts/train.py --model pubmedbert --dataset biomed_ner

# Gene/protein NER
python scripts/train.py --model scibert --dataset jnlpba
```

### ICD-10-CM Code Datasets

These datasets provide ICD code mappings for entity linking rather than NER training.

| Key | Source | Description | Records |
|-----|--------|-------------|---------|
| `atta00/icd10-codes` | HuggingFace | Full ICD-10-CM hierarchy (used by `ICDCodeLookup`) | 51,438 codes |
| `icd10_terminology` | awacke1/ICD10-Clinical-Terminology | ICD-10-CM code/description pairs | 72,750 pairs |
| `icd10_code_description` | wangyichen25/ICD-10-CM_Code-Description_Pairs | Description→code instruction pairs | 1.4M pairs |

**Use ICD codes for entity linking:**

```python
from src.clinical.icd_codes import ICDCodeLookup

# Loads full 51K codes from atta00/icd10-codes (HuggingFace)
# Falls back to built-in ~45 high-frequency codes if download fails
lookup = ICDCodeLookup()

# Match a clinical entity to ICD-10-CM codes via TF-IDF similarity
matches = lookup.match_entity("congestive heart failure", top_k=3)
for m in matches:
    print(f"  {m.code}: {m.description} (score={m.score:.3f}, type={m.match_type})")
# Output:
#   I50.9: Heart failure, unspecified (score=0.782, type=tfidf)
#   I50.22: Chronic systolic heart failure (score=0.451, type=tfidf)
#   ...

# Direct code lookup
code = lookup.lookup_code("E11.9")
print(f"{code.code}: {code.description}")
# E11.9: Type 2 diabetes mellitus without complications

# Batch entity → code mapping
entities = [
    {"text": "hypertension", "label": "Disease"},
    {"text": "pneumonia", "label": "Disease"},
    {"text": "chest pain", "label": "Symptom"},
]
results = lookup.match_entities_batch(entities, top_k=3)
for r in results:
    print(f"{r['text']}: {[c['code'] for c in r['icd_codes']]}")
# hypertension: ['I10']
# pneumonia: ['J18.9']
# chest pain: ['R07.9']
```

### How the ICD Entity Linker Works

The `ICDCodeLookup` follows [SciSpacy's EntityLinker](https://github.com/allenai/scispacy) architecture:

1. **Data source**: Loads the full ICD-10-CM code set (51,438 codes with chapter/section/category hierarchy) from the public [atta00/icd10-codes](https://huggingface.co/datasets/atta00/icd10-codes) dataset (MIT licensed)
2. **TF-IDF index**: Builds character 3-gram + 4-gram TF-IDF vectors for all code descriptions using `sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char_wb")`
3. **Matching**: At query time, vectorises the entity text and computes cosine similarity against all code descriptions
4. **Ranking**: Returns top-k candidates above a minimum similarity threshold

This data-driven approach replaces the previous hardcoded dictionary and automatically covers the full ICD-10-CM taxonomy.

## Python API

### Full Pipeline: Shorthand → NER → Negation → ICD Coding

```python
from src.clinical.pipeline import MedicalCodingPipeline
from src.clinical.icd_codes import ICDCodeLookup

# Full pipeline: shorthand expansion → NER → negation detection
pipeline = MedicalCodingPipeline(
    model_path="outputs/pubmedbert_ncbi_disease/best_model",
    expand_shorthand=True,
    detect_negation=True,
    negation_strategy="rules",  # or "transformer" for learned assertion
)

results = pipeline("Pt denies cp. Dx: dm2, htn.")

# Map entities to ICD-10-CM codes
lookup = ICDCodeLookup()
for entity in results:
    codes = lookup.match_entity(entity.text)
    status = entity.negation.upper()
    icd = codes[0].code if codes else "N/A"
    print(f"{entity.text} [{entity.label}] — {status} — ICD: {icd}")
    # chest pain [Disease] — NEGATED — ICD: R07.9
    # type 2 diabetes mellitus [Disease] — AFFIRMED — ICD: E11.9
    # hypertension [Disease] — AFFIRMED — ICD: I10
```

### Negation Detection: Rules vs Transformer

Two negation/assertion strategies are available:

**Rule-based (ConText/NegEx) — fast, no GPU needed:**

```python
from src.clinical.negation import NegationDetector

detector = NegationDetector()
entities = [
    {"text": "fever", "label": "Symptom", "start": 15, "end": 20},
    {"text": "cough", "label": "Symptom", "start": 29, "end": 34},
]
annotated = detector.annotate_entities("Patient denies fever but has cough", entities)
# annotated[0]["negation"] == "negated"   (fever)
# annotated[1]["negation"] == "affirmed"  (cough)
```

**Transformer-based (bvanaken/clinical-assertion-negation-bert) — learned, more accurate:**

```python
from src.clinical.assertion import AssertionClassifier

classifier = AssertionClassifier()

# Single entity prediction
result = classifier.predict(
    text="Patient denies any chest pain or shortness of breath.",
    entity_text="chest pain",
    entity_start=19,
    entity_end=29,
)
# result == {'label': 'ABSENT', 'negation': 'negated', 'score': 0.97}

# Batch annotation
entities = [
    {"text": "chest pain", "start": 19, "end": 29},
    {"text": "shortness of breath", "start": 33, "end": 52},
]
annotated = classifier.annotate_entities(
    "Patient denies any chest pain or shortness of breath.",
    entities,
)
# Each entity gets: negation, assertion_label, assertion_score
```

**Use transformer negation in the pipeline:**

```python
from src.clinical.pipeline import MedicalCodingPipeline

pipeline = MedicalCodingPipeline(
    model_path="outputs/best_model",
    negation_strategy="transformer",  # uses bvanaken model
)
# The pipeline will automatically use the transformer classifier
# instead of rule-based NegEx for assertion detection
```

### Shorthand Expansion

```python
from src.clinical.shorthand import ShorthandExpander

expander = ShorthandExpander()
text = expander.expand("pt c/o sob, htn well controlled on meds")
# "patient complaining of shortness of breath, hypertension well controlled on meds"
```

### Span-to-BIO Conversion

```python
from src.data.dataset_loader import spans_to_bio

text = "Patient has congestive heart failure and diabetes."
entities = [
    {"start": 12, "end": 36, "class": "DISORDER"},
    {"start": 41, "end": 49, "class": "DISORDER"},
]
tokens, labels = spans_to_bio(text, entities)
# tokens: ['Patient', 'has', 'congestive', 'heart', 'failure', 'and', 'diabetes.']
# labels: ['O', 'O', 'B-DISORDER', 'I-DISORDER', 'I-DISORDER', 'O', 'B-DISORDER']
```

## Supported Models

| Key | Model | Description |
|-----|-------|-------------|
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | SOTA on BLURB benchmark |
| `biobert` | `dmis-lab/biobert-v1.1` | Pre-trained on PubMed abstracts |
| `bio_clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | Pre-trained on MIMIC-III clinical notes |
| `scibert` | `allenai/scibert_scivocab_uncased` | Pre-trained on scientific papers |
| `gatortron-base` | `UFNLP/gatortron-base` | Pre-trained on 90B words of clinical text |

## Architecture Decisions

### Why TF-IDF for Entity Linking (Following SciSpacy)?

SciSpacy's EntityLinker uses TF-IDF character n-grams rather than dense neural embeddings for linking entities to UMLS/ICD codes. This approach:

- **Captures morphological patterns** in medical terminology (e.g. "-itis", "-emia", "cardio-") that are critical for matching
- **Scales efficiently** to 50K+ codes without GPU
- **Is deterministic and interpretable** — you can inspect which n-grams drive a match
- **Outperforms word-level TF-IDF** for medical terms where character patterns are highly informative

### Why Separate Assertion Detection (Not Embedded in NER)?

Modern clinical NLP consensus (reflected in MedSpacy, cTAKES, and SciSpacy) favors a **pipeline approach**: extract entities first (NER), then classify their assertion status (negation/possibility/historicity) separately. This:

- Allows swapping negation strategies (rule-based for speed, transformer for accuracy) without retraining the NER model
- Matches the clinical workflow where entities are identified first, then contextualised
- Enables using the same NER model across different assertion needs

### Public Data Sources

| Component | Dataset | License | Records |
|-----------|---------|---------|---------|
| ICD-10-CM codes | [atta00/icd10-codes](https://huggingface.co/datasets/atta00/icd10-codes) | MIT | 51,438 |
| Assertion model | [bvanaken/clinical-assertion-negation-bert](https://huggingface.co/bvanaken/clinical-assertion-negation-bert) | Apache 2.0 | Fine-tuned on i2b2 |
| Biomedical NER | [knowledgator/biomed_NER](https://huggingface.co/datasets/knowledgator/biomed_NER) | Apache 2.0 | 24 entity types |
| Disease NER | [ncbi_disease](https://huggingface.co/datasets/ncbi_disease) | CC BY 4.0 | 6.9K sentences |
| Drug+Disease NER | [bc5cdr](https://huggingface.co/datasets/bigbio/bc5cdr) | Public domain | 1.5K abstracts |

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

## Tests

```bash
python -m pytest tests/ -v
```

185 tests covering ICD TF-IDF entity linking, assertion classification, shorthand expansion, negation detection, pipeline integration, span-to-BIO conversion, tokenization alignment, and clinical scenario end-to-end flows.
