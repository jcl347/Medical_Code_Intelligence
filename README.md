# Medical Code Intelligence

**ICD-10 NER system** for extracting diagnosis mentions from clinical text, resolving them to ICD-10-CM codes, with learned assertion detection and physician shorthand expansion.

## End-to-End ICD NER Pipeline

The core workflow: **Train** a diagnosis NER model, **Predict** entities from clinical text, **Resolve** to ICD-10-CM codes, **Evaluate** with entity-level F1.

```
Clinical Text ─→ Shorthand Expansion ─→ NER (DIAGNOSIS) ─→ Negation Detection ─→ ICD-10-CM Resolution
                                         │                                         │
  "Pt denies cp"    "Patient denies      "chest pain"         NEGATED              R07.9
                     chest pain"          [DIAGNOSIS]                               Chest pain,
                                                                                   unspecified
```

### Step 1: Install

```bash
pip install -r requirements.txt
```

### Step 2: Train on the ICD NER Dataset

The `icd_ner` dataset is a 5-source composite corpus with a unified `DIAGNOSIS` entity type:

1. **NCBI Disease** — 6.9K sentences from PubMed abstracts
2. **BC5CDR disease subset** — 1.5K abstracts (chemical entities filtered out)
3. **BioMed NER DISORDER/PHENOTYPE** — clinical case reports from `knowledgator/biomed_NER`
4. **ADE Corpus V2** — adverse drug effect spans from `ade_corpus_v2`
5. **Curated ICD examples** — 80+ hand-crafted clinical sentences targeting common FN/FP patterns

Garbage labels (broken BIO annotations from source corpora) are automatically cleaned at load time. The model learns only to detect diagnosable conditions that map to ICD codes.

```bash
# Train PubMedBERT on the ICD NER composite dataset
python scripts/train.py --model pubmedbert --dataset icd_ner

# Train Bio_ClinicalBERT (pre-trained on MIMIC-III clinical notes)
python scripts/train.py --model bio_clinicalbert --dataset icd_ner --lr 3e-5

# Train GatorTron (pre-trained on 90B words of clinical text)
python scripts/train.py --model gatortron-base --dataset icd_ner --lr 3e-5

# Train with CRF layer for structured label decoding
python scripts/train.py --model pubmedbert --dataset icd_ner --use-crf

# Custom hyperparameters
python scripts/train.py --model pubmedbert --dataset icd_ner \
    --epochs 15 --batch-size 32 --lr 3e-5 --patience 3 --scheduler cosine
```

The model is saved automatically to `outputs/pubmedbert_icd_ner/best_model/` (based on best validation F1).

### Step 3: Predict with ICD Code Resolution

```bash
# Single text with ICD code resolution
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --text "Pt denies cp or sob. Hx of dm2 and htn." \
    --icd-codes

# Interactive mode
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --icd-codes

# Batch prediction from file
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --input-file data/notes.txt --output-file results.json \
    --icd-codes --icd-top-k 5
```

Expected output:
```
Pt denies cp or sob. Hx of dm2 and htn.

  [chest pain](DIAGNOSIS, NEGATED, trigger="denies", from="cp", ICD=R07.9, score=0.950)
  [shortness of breath](DIAGNOSIS, NEGATED, trigger="denies", from="sob", ICD=R06.02, score=0.930)
  [type 2 diabetes mellitus](DIAGNOSIS, HISTORICAL, trigger="hx", from="dm2", ICD=E11.9, score=0.970)
  [hypertension](DIAGNOSIS, HISTORICAL, trigger="hx", from="htn", ICD=I10, score=0.960)
```

### Step 4: Evaluate

```bash
# Entity-level F1 on the ICD NER test set
python scripts/evaluate.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --dataset icd_ner

# With detailed error analysis (boundary errors, false positives/negatives)
python scripts/evaluate.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --dataset icd_ner --error-analysis
```

### Step 5: Benchmark Models

```bash
python scripts/benchmark.py \
    --models pubmedbert biobert bio_clinicalbert \
    --datasets icd_ner ncbi_disease bc5cdr
```

## Python API

### Full Pipeline: Shorthand → NER → Negation → ICD Coding

```python
from src.clinical.pipeline import MedicalCodingPipeline

# All-in-one pipeline with ICD resolution built in
pipeline = MedicalCodingPipeline(
    model_path="outputs/pubmedbert_icd_ner/best_model",
    expand_shorthand=True,
    detect_negation=True,
    negation_strategy="rules",      # or "transformer" for learned assertion
    resolve_icd_codes=True,         # enable ICD-10-CM resolution
    icd_top_k=3,                    # top-3 candidate codes per entity
)

results = pipeline("Pt denies cp. Dx: dm2, htn.")

for entity in results:
    status = entity.negation.upper()
    icd = entity.icd_codes[0]["code"] if entity.icd_codes else "N/A"
    print(f"{entity.text} [{entity.label}] — {status} — ICD: {icd}")
    # chest pain [DIAGNOSIS] — NEGATED — ICD: R07.9
    # type 2 diabetes mellitus [DIAGNOSIS] — AFFIRMED — ICD: E11.9
    # hypertension [DIAGNOSIS] — AFFIRMED — ICD: I10
```

### ICD-10-CM Code Resolution (Standalone)

```python
from src.clinical.icd_codes import ICDCodeLookup

# Loads full 51K codes from atta00/icd10-codes (HuggingFace, MIT licensed)
# Falls back to built-in ~45 high-frequency codes if download fails
lookup = ICDCodeLookup()

# Match entity text to ICD-10-CM codes via TF-IDF similarity
matches = lookup.match_entity("congestive heart failure", top_k=3)
for m in matches:
    print(f"  {m.code}: {m.description} (score={m.score:.3f})")
# I50.9: Heart failure, unspecified (score=0.782)
# I50.22: Chronic systolic heart failure (score=0.451)

# Direct code lookup
code = lookup.lookup_code("E11.9")
print(f"{code.code}: {code.description}")
# E11.9: Type 2 diabetes mellitus without complications

# Batch entity → code mapping
entities = [
    {"text": "hypertension", "label": "DIAGNOSIS"},
    {"text": "pneumonia", "label": "DIAGNOSIS"},
    {"text": "chest pain", "label": "DIAGNOSIS"},
]
results = lookup.match_entities_batch(entities, top_k=3)
for r in results:
    print(f"{r['text']}: {[c['code'] for c in r['icd_codes']]}")
# hypertension: ['I10']
# pneumonia: ['J18.9']
# chest pain: ['R07.9']
```

### Negation Detection

Two strategies are available:

**Rule-based (ConText/NegEx) — fast, no GPU needed:**

```python
from src.clinical.negation import NegationDetector

detector = NegationDetector()
entities = [
    {"text": "fever", "label": "DIAGNOSIS", "start": 15, "end": 20},
    {"text": "cough", "label": "DIAGNOSIS", "start": 29, "end": 34},
]
annotated = detector.annotate_entities("Patient denies fever but has cough", entities)
# annotated[0]["negation"] == "negated"   (fever)
# annotated[1]["negation"] == "affirmed"  (cough)
```

**Transformer-based (bvanaken/clinical-assertion-negation-bert) — learned:**

```python
from src.clinical.assertion import AssertionClassifier

classifier = AssertionClassifier()
result = classifier.predict(
    text="Patient denies any chest pain or shortness of breath.",
    entity_text="chest pain",
    entity_start=19,
    entity_end=29,
)
# result == {'label': 'ABSENT', 'negation': 'negated', 'score': 0.97}
```

### Shorthand Expansion

```python
from src.clinical.shorthand import ShorthandExpander

expander = ShorthandExpander()  # loads 104K abbreviations from Meta-Inventory

text = expander.expand("pt c/o sob, htn well controlled on meds")
# "patient complaining of shortness of breath, hypertension well controlled on meds"

# With offset tracking for NER alignment
expanded, offsets = expander.expand_with_offsets("dx: htn, dm2")
```

## Available Datasets

### NER Training Datasets

| Key | Source | Entity Types | Status |
|-----|--------|-------------|--------|
| **`icd_ner`** | **5-source composite (see below)** | **DIAGNOSIS** | **Recommended** |
| `ncbi_disease` | NCBI Disease Corpus | Disease | Available |
| `bc5cdr` | BioCreative V CDR | Chemical, Disease | Available |
| `bc2gm` | BioCreative II GM | Gene | Available |
| `jnlpba` | JNLPBA Shared Task | Protein, DNA, RNA, Cell_line, Cell_type | Available |
| `biomed_ner` | knowledgator/biomed_NER | 24 types (DISORDER, CLINICAL_DRUG, ...) | Available (span format) |

**`icd_ner`** is the recommended dataset for ICD coding. It merges five sources and normalizes all disease/disorder entities to a single `DIAGNOSIS` label:

| # | Source | What it contributes |
|---|--------|-------------------|
| 1 | NCBI Disease (6.9K sentences) | Broad disease mention coverage from PubMed |
| 2 | BC5CDR disease subset (1.5K abstracts) | Chemical-disease relation corpus, disease entities only |
| 3 | BioMed NER DISORDER/PHENOTYPE | Clinical case reports with disorder and phenotype spans |
| 4 | ADE Corpus V2 | Adverse drug effect spans (drug reactions as diagnoses) |
| 5 | Curated ICD examples (80+ sentences) | Hand-crafted examples targeting FN/FP patterns + negative examples |

Garbage labels (broken BIO tags on function words like "of", "and", "the") are automatically cleaned from all sources. The curated examples include negative training data (e.g. "blood pressure", "heart rate", "renal function") to reduce false positives on clinical measurements.

The model's job is to detect diagnosable conditions; the downstream `ICDCodeLookup` maps extracted text spans to specific ICD-10-CM codes.

### ICD-10-CM Code Lookup Datasets

| Key | Source | Records |
|-----|--------|---------|
| `atta00/icd10-codes` | Full ICD-10-CM hierarchy (used by `ICDCodeLookup`) | 51,438 codes |
| `icd10_terminology` | awacke1/ICD10-Clinical-Terminology | 72,750 pairs |
| `icd10_code_description` | wangyichen25/ICD-10-CM_Code-Description_Pairs | 1.4M pairs |

### Why No OASIS Dataset?

CMS publishes aggregate OASIS (Outcome and Assessment Information Set) statistics, but the underlying clinical text with span-level ICD annotations is not publicly released. The `icd_ner` composite dataset fills this gap using the best publicly available disease NER corpora.

> **Note:** `d4data/biomedical-ner-all` (`biomedical_ner_all`) has been removed from HuggingFace and is no longer available for training.

## Supported Models

| Key | Model | Best For |
|-----|-------|----------|
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | General biomedical NER (SOTA on BLURB) |
| `bio_clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | Clinical notes (pre-trained on MIMIC-III) |
| `biobert` | `dmis-lab/biobert-v1.1` | PubMed literature |
| `scibert` | `allenai/scibert_scivocab_uncased` | Scientific papers |
| `gatortron-base` | `UFNLP/gatortron-base` | Clinical text (90B words) |

For ICD NER on clinical notes, **`bio_clinicalbert`** or **`pubmedbert`** are recommended starting points.

## Project Structure

```
Medical_Code_Intelligence/
├── configs/
│   └── ner_config.py              # Model, dataset, and training configs
├── src/
│   ├── clinical/
│   │   ├── pipeline.py            # Unified pipeline: NER → negation → ICD
│   │   ├── icd_codes.py           # TF-IDF ICD-10-CM entity linker (51K codes)
│   │   ├── negation.py            # ConText/NegEx rule-based negation
│   │   ├── assertion.py           # Transformer assertion classifier
│   │   ├── shorthand.py           # Data-driven abbreviation expansion
│   │   ├── abbreviation_disambiguator.py  # MeDAL ELECTRA disambiguation
│   │   ├── _icd_fallback.py       # Offline fallback ICD codes
│   │   └── _shorthand_fallback.py # Built-in ~280 abbreviations
│   ├── data/
│   │   ├── icd_dataset.py         # ICD NER composite dataset loader
│   │   ├── dataset_loader.py      # HuggingFace dataset loaders + span→BIO
│   │   ├── preprocessing.py       # Subword tokenization & label alignment
│   │   └── data_utils.py          # Data collator, sliding window splitting
│   ├── models/
│   │   ├── ner_model.py           # AutoModelForTokenClassification builder
│   │   └── crf_model.py           # Optional CRF layer
│   ├── training/
│   │   ├── trainer.py             # HuggingFace Trainer setup
│   │   └── callbacks.py           # Early stopping
│   ├── evaluation/
│   │   ├── metrics.py             # Entity-level F1 (seqeval)
│   │   └── error_analysis.py      # Boundary/type/FP/FN analysis
│   └── inference/
│       ├── entity_utils.py        # Entity post-processing (stopword filter, merge)
│       └── predictor.py           # Inference with batching
├── scripts/
│   ├── train.py                   # Training CLI
│   ├── evaluate.py                # Evaluation CLI
│   ├── predict.py                 # Prediction CLI (interactive + batch)
│   └── benchmark.py               # Multi-model x multi-dataset benchmarking
├── tests/                         # Test suite
├── requirements.txt
└── setup.py
```

## How the ICD Entity Linker Works

The `ICDCodeLookup` follows [SciSpacy's EntityLinker](https://github.com/allenai/scispacy) architecture:

1. **Data source**: Loads 51,438 ICD-10-CM codes from [atta00/icd10-codes](https://huggingface.co/datasets/atta00/icd10-codes) (MIT licensed)
2. **TF-IDF index**: Builds character 3/4-gram TF-IDF vectors for all code descriptions
3. **Matching**: Cosine similarity between entity text and code descriptions
4. **Ranking**: Top-k candidates above a minimum similarity threshold

Character n-grams capture morphological patterns critical for medical terms (e.g. "-itis", "-emia", "cardio-"). No GPU required.

## Training Best Practices

- **Subword label alignment** — BIO labels aligned to first subword; continuations get ignore label (-100)
- **Dynamic padding** — Pad to longest-in-batch for efficiency
- **Learning rate warmup** — Linear warmup over 10% of training steps
- **Mixed precision (FP16)** — 2x training speed on GPU
- **Early stopping** — Patience-based on validation entity-level F1
- **Gradient clipping** — Max norm 1.0
- **Model saving** — Best model saved automatically based on validation F1

## Architecture Decisions

**Pipeline approach (NER → Assertion → ICD)**: Entities are extracted first, then negation is classified separately. This allows swapping negation strategies without retraining NER, matching the architecture of MedSpacy, cTAKES, and SciSpacy.

**TF-IDF for entity linking**: Character n-grams outperform word-level TF-IDF for medical terms. Scales to 50K+ codes without GPU. Deterministic and interpretable.

**Composite ICD NER dataset**: Merging five sources (NCBI Disease, BC5CDR, BioMed NER disorders, ADE Corpus adverse effects, and curated clinical examples) gives broad coverage of diagnosable conditions. Garbage labels from source corpora are cleaned automatically, and curated negative examples reduce false positives on clinical measurements. A single `DIAGNOSIS` label keeps the model focused on the ICD-relevant task.

## Public Data Sources

| Component | Dataset | License | Records |
|-----------|---------|---------|---------|
| ICD NER training | NCBI Disease (ncbi/ncbi_disease) | Public domain | 6.9K sentences |
| ICD NER training | BC5CDR (tner/bc5cdr) | CC BY 4.0 | 1.5K abstracts |
| ICD NER training | BioMed NER (knowledgator/biomed_NER) | Apache 2.0 | 500 case reports |
| ICD NER training | ADE Corpus V2 (ade_corpus_v2) | Public domain | Drug-effect spans |
| ICD NER training | Curated ICD examples | Project-internal | 80+ sentences |
| ICD-10-CM codes | [atta00/icd10-codes](https://huggingface.co/datasets/atta00/icd10-codes) | MIT | 51,438 |
| Assertion model | [bvanaken/clinical-assertion-negation-bert](https://huggingface.co/bvanaken/clinical-assertion-negation-bert) | Apache 2.0 | Fine-tuned on i2b2 |
| Abbreviations | [Meta-Inventory](https://zenodo.org/records/4567594) | CC-BY-4.0 | 104,057 |
| Disambiguation | [McGill-NLP/electra-medal](https://huggingface.co/McGill-NLP/electra-medal) | MIT | 14M abstracts |

## Tests

```bash
python -m pytest tests/ -v
```
