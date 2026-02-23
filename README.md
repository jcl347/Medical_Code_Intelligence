# Medical Code Intelligence

**ICD-10 NER system** for extracting diagnosis mentions from clinical text, resolving them to ICD-10-CM codes, estimating MS-DRG cost impact, with adversarial training, learned assertion detection, and physician shorthand expansion.

## End-to-End ICD NER Pipeline

The core workflow: **Train** a diagnosis NER model (with optional adversarial training), **Predict** entities from clinical text, **Resolve** to ICD-10-CM codes, **Estimate** MS-DRG cost impact, **Evaluate** with entity-level F1.

```
Clinical Text ─→ Shorthand Expansion ─→ NER (DIAGNOSIS) ─→ Negation Detection ─→ ICD-10-CM Resolution ─→ MS-DRG Cost
                                         │                                         │                      │
  "Pt denies cp"    "Patient denies      "chest pain"         NEGATED              R07.9                  DRG 292
                     chest pain"          [DIAGNOSIS]                               Chest pain,            $6,253
                                                                                   unspecified
```

### Step 1: Install

```bash
pip install -r requirements.txt
```

### Step 2: Train on the ICD NER Dataset

The `icd_ner` dataset is a 7-source composite corpus with a unified `DIAGNOSIS` entity type:

1. **NCBI Disease** — 6.9K sentences from PubMed abstracts
2. **BC5CDR disease subset** — 1.5K abstracts (chemical entities filtered out)
3. **BioMed NER DISORDER/PHENOTYPE** — clinical case reports from `knowledgator/biomed_NER`
4. **ADE Corpus V2** — adverse drug effect spans from `ade_corpus_v2`
5. **Curated ICD examples** — 80+ hand-crafted + ~80 template-generated clinical sentences targeting common NER failure patterns (abbreviations, multi-word boundaries, lab value confusion, negation contexts, rare diseases, high-frequency ICD codes)
6. **MedMentions** (optional) — up to 5K examples from 4,392 PubMed abstracts with 350K+ UMLS entity mentions, filtered for disease/disorder semantic types
7. **MACCROBAT** (optional) — up to 3K examples from 200 clinical case reports with DISEASE_DISORDER entities, providing clinical-note-style text that PubMed abstracts lack

Sources 6 and 7 download from HuggingFace on first use and fall back gracefully if unavailable. Garbage labels (broken BIO annotations from source corpora) are automatically cleaned at load time. The model learns only to detect diagnosable conditions that map to ICD codes.

```bash
# Train PubMedBERT on the ICD NER composite dataset
python scripts/train.py --model pubmedbert --dataset icd_ner

# Train Bio_ClinicalBERT (pre-trained on MIMIC-III clinical notes)
python scripts/train.py --model bio_clinicalbert --dataset icd_ner --lr 3e-5

# Train GatorTron (pre-trained on 90B words of clinical text)
python scripts/train.py --model gatortron-base --dataset icd_ner --lr 3e-5

# Train with CRF layer for structured label decoding
python scripts/train.py --model pubmedbert --dataset icd_ner --use-crf

# Train with adversarial training (FGM) for +0.5-1.5% F1 improvement
python scripts/train.py --model pubmedbert --dataset icd_ner --adversarial

# Train with PGD adversarial training (stronger but ~4x slower)
python scripts/train.py --model pubmedbert --dataset icd_ner --adversarial --adv-method pgd

# Custom adversarial epsilon
python scripts/train.py --model pubmedbert --dataset icd_ner --adversarial --adv-epsilon 0.5

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

### Full Pipeline: Shorthand → NER → Negation → ICD Coding → DRG Cost

```python
from src.clinical.pipeline import MedicalCodingPipeline

# All-in-one pipeline with ICD resolution and DRG cost estimation
pipeline = MedicalCodingPipeline(
    model_path="outputs/pubmedbert_icd_ner/best_model",
    expand_shorthand=True,
    detect_negation=True,
    negation_strategy="rules",      # or "transformer" for learned assertion
    resolve_icd_codes=True,         # enable ICD-10-CM resolution
    icd_top_k=3,                    # top-3 candidate codes per entity
    resolve_drg=True,               # enable MS-DRG cost estimation
    drg_base_rate=6752.61,          # FY 2026 national standardized amount
)

results = pipeline("Pt denies cp. Dx: dm2, htn.")

for entity in results:
    status = entity.negation.upper()
    icd = entity.icd_codes[0]["code"] if entity.icd_codes else "N/A"
    print(f"{entity.text} [{entity.label}] — {status} — ICD: {icd}")
    # chest pain [DIAGNOSIS] — NEGATED — ICD: R07.9
    # type 2 diabetes mellitus [DIAGNOSIS] — AFFIRMED — ICD: E11.9
    # hypertension [DIAGNOSIS] — AFFIRMED — ICD: I10

    # DRG info is attached to the primary affirmed entity
    if entity.drg_info:
        drg = entity.drg_info
        print(f"  DRG {drg['current']['drg_code']}: {drg['current']['drg_title']}")
        print(f"  Estimated payment: ${drg['current']['estimated_payment']:,.2f}")
        if drg['undercoding_risk']:
            print(f"  Revenue at risk: ${drg['revenue_at_risk']:,.2f}")
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

### MS-DRG Cost Estimation (Standalone)

```python
from src.clinical.drg_costs import DRGCostEstimator

# Maps ICD-10-CM codes to MS-DRGs and estimates financial impact
# Requires drgpy for grouper logic: pip install drgpy
# Falls back to built-in weights for 32 common DRGs without drgpy
estimator = DRGCostEstimator()

# Assign DRG and estimate cost
result = estimator.get_drg(["J18.9", "E11.9", "N17.9"])
if result:
    print(f"DRG {result.drg_code}: {result.drg_title}")
    print(f"Relative weight: {result.relative_weight}")
    print(f"Estimated payment: ${result.estimated_payment:,.2f}")
    print(f"Severity level: {result.severity_level}")

# Analyze CC/MCC impact — compare severity tiers
analysis = estimator.analyze_cost_impact(["J18.9", "E11.9"])
if analysis:
    print(f"Current DRG: {analysis.current_drg.drg_code}")
    print(f"Revenue at risk: ${analysis.revenue_at_risk:,.2f}")
    print(f"Undercoding risk: {analysis.undercoding_risk}")

    # Compare base vs CC vs MCC variants
    if analysis.mcc_drg:
        print(f"MCC variant: DRG {analysis.mcc_drg.drg_code} "
              f"(${analysis.mcc_drg.estimated_payment:,.2f})")

# Direct cost estimate by DRG code
cost = estimator.estimate_cost("291")  # Heart Failure & Shock W MCC
print(f"DRG 291 estimated payment: ${cost:,.2f}")
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
| **`icd_ner`** | **7-source composite (see below)** | **DIAGNOSIS** | **Recommended** |
| `ncbi_disease` | NCBI Disease Corpus | Disease | Available |
| `bc5cdr` | BioCreative V CDR | Chemical, Disease | Available |
| `bc2gm` | BioCreative II GM | Gene | Available |
| `jnlpba` | JNLPBA Shared Task | Protein, DNA, RNA, Cell_line, Cell_type | Available |
| `biomed_ner` | knowledgator/biomed_NER | 24 types (DISORDER, CLINICAL_DRUG, ...) | Available (span format) |

**`icd_ner`** is the recommended dataset for ICD coding. It merges seven sources and normalizes all disease/disorder entities to a single `DIAGNOSIS` label:

| # | Source | What it contributes |
|---|--------|-------------------|
| 1 | NCBI Disease (6.9K sentences) | Broad disease mention coverage from PubMed |
| 2 | BC5CDR disease subset (1.5K abstracts) | Chemical-disease relation corpus, disease entities only |
| 3 | BioMed NER DISORDER/PHENOTYPE | Clinical case reports with disorder and phenotype spans |
| 4 | ADE Corpus V2 | Adverse drug effect spans (drug reactions as diagnoses) |
| 5 | Curated ICD examples (80+ hand-crafted + ~80 template-generated) | Targets common NER failure patterns: abbreviations, multi-word boundaries, lab confusion, negation contexts, rare diseases, high-frequency ICD codes |
| 6 | MedMentions (optional, up to 5K) | 4,392 PubMed abstracts, disease/disorder UMLS semantic types (T047, T048, T019, T046, T191) |
| 7 | MACCROBAT (optional, up to 3K) | 200 clinical case reports with DISEASE_DISORDER entities — closes the PubMed-to-clinical domain gap |

Garbage labels (broken BIO tags on function words like "of", "and", "the") are automatically cleaned from all sources. The curated examples include negative training data (e.g. "blood pressure", "heart rate", "renal function") to reduce false positives on clinical measurements. Sources 6 and 7 are downloaded from HuggingFace on first use and skipped gracefully if unavailable.

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
│   │   ├── pipeline.py            # Unified pipeline: NER → negation → ICD → DRG
│   │   ├── icd_codes.py           # TF-IDF ICD-10-CM entity linker (51K codes)
│   │   ├── drg_costs.py           # MS-DRG assignment + cost impact analysis
│   │   ├── negation.py            # ConText/NegEx rule-based negation
│   │   ├── assertion.py           # Transformer assertion classifier
│   │   ├── shorthand.py           # Data-driven abbreviation expansion
│   │   ├── abbreviation_disambiguator.py  # MeDAL ELECTRA disambiguation
│   │   ├── _icd_fallback.py       # Offline fallback ICD codes
│   │   └── _shorthand_fallback.py # Built-in ~280 abbreviations
│   ├── data/
│   │   ├── icd_dataset.py         # ICD NER composite dataset loader (7 sources)
│   │   ├── dataset_loader.py      # HuggingFace dataset loaders + span→BIO
│   │   ├── preprocessing.py       # Subword tokenization & label alignment
│   │   └── data_utils.py          # Data collator, sliding window splitting
│   ├── models/
│   │   ├── ner_model.py           # AutoModelForTokenClassification builder
│   │   └── crf_model.py           # Optional CRF layer
│   ├── training/
│   │   ├── trainer.py             # HuggingFace Trainer setup
│   │   ├── adversarial.py         # FGM/PGD adversarial training (+0.5-1.5% F1)
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

## Command Reference

### `scripts/train.py` — Train a Biomedical NER Model

Trains a transformer-based NER model on any supported dataset. Handles dataset loading, tokenization, subword label alignment, training with early stopping, and saves the best model checkpoint automatically.

```bash
python scripts/train.py [OPTIONS]
```

**Model selection:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model <key>` | `pubmedbert` | Pre-trained model key. Choices: `pubmedbert`, `biobert`, `bio_clinicalbert`, `scibert`, `gatortron-base`. Each key maps to a HuggingFace model ID (see Supported Models table). |
| `--model-path <path>` | None | Override `--model` with any HuggingFace model ID or local path to a pre-trained model directory. Use this for custom models not in the built-in list. |
| `--use-crf` | off | Add a linear-chain CRF (Conditional Random Field) layer on top of the transformer. Enforces valid BIO transitions during decoding (e.g., I-DIAGNOSIS can only follow B-DIAGNOSIS). Adds ~5% training time. |

**Dataset selection:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset <key>` | `ncbi_disease` | Dataset to train on. Choices: `icd_ner`, `ncbi_disease`, `bc5cdr`, `bc2gm`, `jnlpba`, `biomed_ner`. Use `icd_ner` for the 7-source composite ICD coding dataset. |
| `--max-length <int>` | `512` | Maximum token sequence length after subword tokenization. Sequences longer than this are truncated. Reduce to 256 for faster training or if GPU memory is limited. |

**Training hyperparameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs <int>` | `20` | Maximum number of training epochs. Training may stop earlier if early stopping triggers. |
| `--batch-size <int>` | `16` | Per-device training batch size. Effective batch size = `batch-size * grad-accum * num_gpus`. Reduce to 8 for GatorTron or other large models if OOM. |
| `--lr <float>` | `5e-5` | Peak learning rate. Bio-domain models typically work well with 3e-5 to 5e-5. Lower values (2e-5) can help if training is unstable. |
| `--weight-decay <float>` | `0.01` | L2 weight decay regularization applied to all parameters except biases and layer norms. |
| `--warmup-ratio <float>` | `0.1` | Fraction of total training steps used for linear learning rate warmup. 0.1 means the first 10% of steps ramp from 0 to the peak LR. |
| `--grad-accum <int>` | `1` | Gradient accumulation steps. Simulates larger batch sizes without more GPU memory. Setting `--batch-size 8 --grad-accum 2` is equivalent to batch size 16. |
| `--patience <int>` | `5` | Early stopping patience. Training stops if validation entity-level F1 doesn't improve for this many evaluation rounds. |
| `--label-smoothing <float>` | `0.0` | Label smoothing factor (0.0-1.0). Redistributes a fraction of the target probability to non-target classes. Values of 0.05-0.1 can reduce overfitting. |
| `--scheduler <type>` | `linear` | Learning rate scheduler. `linear`: linear decay after warmup. `cosine`: cosine annealing. `constant_with_warmup`: constant LR after warmup period. |

**Adversarial training:**

| Flag | Default | Description |
|------|---------|-------------|
| `--adversarial` | off | Enable adversarial training on the word embedding layer. During each training step, the embeddings are perturbed in the direction that maximizes the loss, and the model also trains on these adversarial examples. Improves robustness and F1 by +0.5-1.5%. |
| `--adv-method <method>` | `fgm` | Adversarial method. `fgm` (Fast Gradient Method): single-step perturbation, ~2x training time. `pgd` (Projected Gradient Descent): multi-step perturbation with projection back to epsilon-ball, ~4x training time, slightly stronger. |
| `--adv-epsilon <float>` | auto | Perturbation magnitude (L2 norm). Controls how far the adversarial example can deviate from the original embedding. Default: 1.0 for FGM, 0.3 for PGD. Larger values = stronger perturbation but risk training instability. |

**Output and misc:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir <path>` | `outputs` | Root output directory. Model checkpoints save to `<output-dir>/<model>_<dataset>/best_model/`. |
| `--seed <int>` | `42` | Random seed for reproducibility. Controls dataset shuffling, weight initialization, and dropout. |
| `--fp16` | on | Enable mixed-precision (FP16) training. Roughly halves GPU memory usage and doubles throughput on compatible GPUs. Enabled by default when CUDA is available. |
| `--no-fp16` | off | Force disable mixed-precision training. Use if you encounter NaN losses or are training on CPU. |
| `--eval-steps <int>` | `200` | Evaluate on the validation set every N training steps. Also controls checkpoint saving frequency. Lower values give earlier stopping detection but add overhead. |
| `--wandb` | off | Enable Weights & Biases experiment tracking. Requires `wandb` package and login. Logs loss curves, learning rates, and evaluation metrics. |

**Examples:**

```bash
# Minimal: train PubMedBERT on ICD NER with all defaults
python scripts/train.py --model pubmedbert --dataset icd_ner

# Clinical model with tuned hyperparameters
python scripts/train.py --model bio_clinicalbert --dataset icd_ner \
    --lr 3e-5 --epochs 15 --patience 3 --scheduler cosine

# GatorTron with reduced batch size (345M params needs more memory)
python scripts/train.py --model gatortron-base --dataset icd_ner \
    --batch-size 8 --grad-accum 2 --lr 3e-5

# Adversarial training with CRF layer
python scripts/train.py --model pubmedbert --dataset icd_ner \
    --adversarial --adv-method fgm --use-crf

# PGD adversarial with custom epsilon and label smoothing
python scripts/train.py --model pubmedbert --dataset icd_ner \
    --adversarial --adv-method pgd --adv-epsilon 0.5 --label-smoothing 0.05
```

---

### `scripts/predict.py` — Run NER Predictions

Runs the full pipeline (shorthand expansion, NER, negation detection, ICD resolution) on clinical text. Supports three modes: single text, batch file, and interactive REPL.

```bash
python scripts/predict.py --model-path <path> [OPTIONS]
```

**Required:**

| Flag | Description |
|------|-------------|
| `--model-path <path>` | Path to a fine-tuned model directory (must contain `config.json` and model weights). This is the directory saved by `train.py`, e.g., `outputs/pubmedbert_icd_ner/best_model`. Not a model key — it's a local filesystem path or HuggingFace model ID with saved weights. |

**Input mode (pick one, or omit all for interactive):**

| Flag | Description |
|------|-------------|
| `--text "<string>"` | Process a single clinical text string and print results. Wrap in quotes if the text contains spaces or special characters. |
| `--input-file <path>` | Process a file with one clinical sentence per line. Each line is processed independently through the full pipeline. |
| *(neither)* | Starts interactive mode — a REPL where you type clinical text and see results. Type `quit` to exit. |

**Output:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output-file <path>` | None | Save batch results to a JSON file. Each entry contains the input text and a list of extracted entities with all annotations. Only used with `--input-file`. |

**Pipeline features:**

| Flag | Default | Description |
|------|---------|-------------|
| `--no-shorthand` | off | Disable physician shorthand expansion. By default, abbreviations like "cp", "sob", "htn" are expanded to their full forms before NER. Disable this if your input text is already in standard medical terminology. |
| `--no-negation` | off | Disable negation/assertion detection. By default, entities are annotated with assertion status (affirmed, negated, possible, historical, etc.). Disable to skip this step and treat all entities as affirmed. |
| `--icd-codes` | off | Enable ICD-10-CM code resolution. Each extracted entity is matched against 51K ICD-10-CM codes using TF-IDF character n-gram similarity. The top-k candidate codes are attached to each entity. |
| `--icd-top-k <int>` | `3` | Number of ICD-10-CM candidate codes to return per entity. Higher values give more options but may include lower-quality matches. |
| `--device <device>` | auto | Force a specific device (`cpu`, `cuda`, `cuda:0`, etc.). Auto-detects CUDA availability by default. |

**Examples:**

```bash
# Single text with full pipeline
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --text "Pt denies cp or sob. Hx of dm2 and htn."

# Single text with ICD code resolution
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --text "Pt denies cp or sob. Hx of dm2 and htn." \
    --icd-codes

# Batch processing with JSON output
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --input-file data/clinical_notes.txt \
    --output-file results.json \
    --icd-codes --icd-top-k 5

# Interactive mode (no --text, no --input-file)
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --icd-codes

# Skip shorthand expansion (input is already in standard terminology)
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --text "Patient has congestive heart failure and diabetes." \
    --no-shorthand

# NER only, no negation detection
python scripts/predict.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --text "No evidence of pneumonia." \
    --no-negation
```

**Output format:**

Each entity is displayed as `[entity text](label, STATUS, details)`:
```
Pt denies cp or sob. Hx of dm2 and htn.

  [chest pain](DIAGNOSIS, NEGATED, trigger="denies", from="cp", ICD=R07.9, score=0.950)
  [shortness of breath](DIAGNOSIS, NEGATED, trigger="denies", from="sob", ICD=R06.02, score=0.930)
  [type 2 diabetes mellitus](DIAGNOSIS, HISTORICAL, trigger="hx", from="dm2", ICD=E11.9, score=0.970)
  [hypertension](DIAGNOSIS, HISTORICAL, trigger="hx", from="htn", ICD=I10, score=0.960)
```

Fields: entity label, assertion status (AFFIRMED/NEGATED/POSSIBLE/HISTORICAL/HYPOTHETICAL/FAMILY), negation trigger word, abbreviation it was expanded from, best ICD-10-CM code, model confidence score.

---

### `scripts/evaluate.py` — Evaluate a Trained Model

Computes entity-level precision, recall, and F1 on a dataset split. Optionally runs detailed error analysis that categorizes mistakes into boundary errors, type errors, false positives, and false negatives.

```bash
python scripts/evaluate.py --model-path <path> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path <path>` | *(required)* | Path to a fine-tuned model directory (same as `predict.py`). |
| `--dataset <key>` | `ncbi_disease` | Dataset to evaluate on. The model's predictions are compared against the gold-standard labels from this dataset. Use the same dataset the model was trained on, or a different one for cross-dataset evaluation. |
| `--split <name>` | `test` | Dataset split to evaluate. Choices: `train`, `validation`, `test`. Falls back to the last available split if the requested one doesn't exist. |
| `--batch-size <int>` | `32` | Evaluation batch size. Can be larger than training batch size since no gradients are stored. Increase for faster evaluation if GPU memory allows. |
| `--max-length <int>` | `512` | Maximum sequence length for tokenization. Should match the value used during training. |
| `--error-analysis` | off | Run detailed error analysis after evaluation. Categorizes every prediction mistake: **boundary errors** (entity detected but wrong span), **type errors** (right span, wrong label), **false positives** (predicted entity that doesn't exist in gold), **false negatives** (gold entity that the model missed). Prints a report with example tokens for each error type. |
| `--output-file <path>` | None | Save evaluation metrics (precision, recall, F1) to a JSON file. |

**Examples:**

```bash
# Basic evaluation on test set
python scripts/evaluate.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --dataset icd_ner

# Evaluate on validation set with error analysis
python scripts/evaluate.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --dataset icd_ner --split validation --error-analysis

# Cross-dataset evaluation (train on icd_ner, evaluate on ncbi_disease)
python scripts/evaluate.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --dataset ncbi_disease

# Save metrics to file
python scripts/evaluate.py \
    --model-path outputs/pubmedbert_icd_ner/best_model \
    --dataset icd_ner --output-file eval_results.json
```

---

### `scripts/benchmark.py` — Compare Models Across Datasets

Trains and evaluates every combination of the specified models and datasets. Produces a comparison table and saves results to JSON. Useful for model selection and dataset ablation studies.

```bash
python scripts/benchmark.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--models <key> [key ...]` | `pubmedbert biobert bio_clinicalbert` | Space-separated list of model keys to benchmark. Each model is trained from scratch on each dataset. |
| `--datasets <key> [key ...]` | `ncbi_disease bc5cdr jnlpba` | Space-separated list of dataset keys. Total experiments = `len(models) * len(datasets)`. |
| `--epochs <int>` | `10` | Training epochs per experiment. Use fewer (3-5) for quick comparisons. |
| `--batch-size <int>` | `16` | Per-device batch size for all experiments. |
| `--lr <float>` | `5e-5` | Learning rate for all experiments. |
| `--eval-steps <int>` | `200` | Evaluation frequency (steps). |
| `--patience <int>` | `5` | Early stopping patience. |
| `--output-dir <path>` | `outputs/benchmark` | Directory for checkpoints and the results JSON file. |
| `--seed <int>` | `42` | Random seed (same for all experiments for fair comparison). |

**Examples:**

```bash
# Full benchmark: 3 models x 3 datasets = 9 experiments
python scripts/benchmark.py \
    --models pubmedbert biobert bio_clinicalbert \
    --datasets icd_ner ncbi_disease bc5cdr

# Quick comparison with fewer epochs
python scripts/benchmark.py \
    --models pubmedbert bio_clinicalbert \
    --datasets icd_ner \
    --epochs 3 --eval-steps 50

# Include GatorTron (will need reduced batch size manually if OOM)
python scripts/benchmark.py \
    --models pubmedbert gatortron-base \
    --datasets icd_ner ncbi_disease \
    --batch-size 8

# Single model ablation across all disease datasets
python scripts/benchmark.py \
    --models pubmedbert \
    --datasets icd_ner ncbi_disease bc5cdr
```

**Output:** Prints a summary table at the end and saves a timestamped JSON file:

```
================================================================================
BENCHMARK RESULTS
================================================================================
Model                Dataset            Precision    Recall        F1
--------------------------------------------------------------------------------
pubmedbert           icd_ner              0.8912    0.8745    0.8828
pubmedbert           ncbi_disease         0.8734    0.8521    0.8626
biobert              icd_ner              0.8856    0.8690    0.8772
...

Results saved to: outputs/benchmark/benchmark_20260223_143052.json
```

---

### `python -m pytest` — Run Tests

```bash
# Run the full test suite (270 tests, ~25 seconds on CPU)
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_negation.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run only tests matching a pattern
python -m pytest tests/ -v -k "test_curated"
```

All tests use mocked models and built-in fallback data — no GPU, network access, or downloaded models required.

## Adversarial Training

FGM (Fast Gradient Method) and PGD (Projected Gradient Descent) adversarial training on word embeddings, implemented in `src/training/adversarial.py`. Perturbs the embedding layer in the direction of the loss gradient, then trains on both clean and perturbed inputs. This regularizes the model against small input variations without any architecture changes.

| Method | Compute Overhead | Expected F1 Gain | CLI Flag |
|--------|-----------------|-------------------|----------|
| **FGM** (default) | ~2x training time | +0.5-1.5% | `--adversarial` |
| **PGD** (K=3) | ~4x training time | +0.5-1.5% | `--adversarial --adv-method pgd` |

Key parameters:
- `--adversarial` — Enable adversarial training (default: FGM)
- `--adv-method fgm|pgd` — Choose method (default: `fgm`)
- `--adv-epsilon <float>` — Perturbation magnitude (default: 1.0 for FGM, 0.3 for PGD)

```bash
# FGM adversarial training (recommended starting point)
python scripts/train.py --model pubmedbert --dataset icd_ner --adversarial

# PGD with custom epsilon
python scripts/train.py --model pubmedbert --dataset icd_ner \
    --adversarial --adv-method pgd --adv-epsilon 0.5
```

Based on: RanAT4BIE (2025), FreeLB (ICLR 2020), Miyato et al. (2017).

## MS-DRG Cost Estimation

The `DRGCostEstimator` (`src/clinical/drg_costs.py`) maps ICD-10-CM codes to Medicare Severity Diagnosis Related Groups (MS-DRGs) and estimates financial impact. This enables "revenue at risk" analysis — identifying cases where missed CC/MCC secondary diagnoses lead to lower-severity DRG assignments and reduced reimbursement.

**Key concepts:**

| Term | Description |
|------|-------------|
| **MS-DRG** | Groups inpatient stays into ~770 payment categories by diagnosis, severity, and procedures |
| **Relative Weight** | Multiplier reflecting resource intensity (1.0 = national average; 2.0 = twice average cost) |
| **CC/MCC** | Complication/Comorbidity (CC) and Major CC (MCC) — secondary diagnoses that increase severity tier and payment |
| **Revenue at Risk** | Payment gap between current DRG assignment and what it could be with proper CC/MCC capture |
| **Base Rate** | FY 2026 national standardized amount: $6,752.61 |

**Pipeline integration:** When `resolve_drg=True`, the pipeline:
1. Collects ICD codes from all affirmed (non-negated) entities
2. Groups them into an MS-DRG via drgpy
3. Finds related CC/MCC severity-tier variants
4. Calculates revenue at risk from undercoding
5. Attaches the cost analysis to the primary diagnosis entity

**Data sources:**
- `drgpy` library for ICD-10 to MS-DRG grouper logic (optional: `pip install drgpy`)
- CMS IPPS Table 5 relative weights (loadable from Excel via `table5_path` parameter)
- Built-in fallback: 32 common medical DRGs with FY 2026 weights for CI/testing

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
- **Adversarial training** — FGM/PGD perturbation on word embeddings for +0.5-1.5% F1 (see below)

## Architecture Decisions

**Pipeline approach (NER → Assertion → ICD → DRG)**: Entities are extracted first, then negation is classified, then mapped to ICD codes, and finally grouped into MS-DRGs for cost estimation. Components are independently swappable, matching the architecture of MedSpacy, cTAKES, and SciSpacy.

**TF-IDF for entity linking**: Character n-grams outperform word-level TF-IDF for medical terms. Scales to 50K+ codes without GPU. Deterministic and interpretable.

**Adversarial training (FGM/PGD)**: Perturbs word embeddings in the direction of the loss gradient during training, then trains on both clean and perturbed inputs. This regularizes the model against small input variations, improving entity-level F1 by +0.5-1.5% on biomedical NER benchmarks with no architecture changes.

**7-source composite ICD NER dataset**: Merging seven sources (NCBI Disease, BC5CDR, BioMed NER disorders, ADE Corpus adverse effects, curated clinical examples, MedMentions, and MACCROBAT) gives broad coverage of diagnosable conditions across PubMed abstracts and clinical notes. Template-generated examples target documented NER failure patterns (abbreviations, boundary errors, lab confusion). Garbage labels from source corpora are cleaned automatically, and curated negative examples reduce false positives on clinical measurements. A single `DIAGNOSIS` label keeps the model focused on the ICD-relevant task.

**MS-DRG cost scoping**: Maps extracted ICD codes to Medicare Severity Diagnosis Related Groups (~770 payment categories) and compares CC/MCC severity tiers to quantify revenue at risk from undercoding. Built-in fallback weights for 32 common DRGs ensure CI/testing works without external dependencies.

## Public Data Sources

| Component | Dataset | License | Records |
|-----------|---------|---------|---------|
| ICD NER training | NCBI Disease (ncbi/ncbi_disease) | Public domain | 6.9K sentences |
| ICD NER training | BC5CDR (tner/bc5cdr) | CC BY 4.0 | 1.5K abstracts |
| ICD NER training | BioMed NER (knowledgator/biomed_NER) | Apache 2.0 | 500 case reports |
| ICD NER training | ADE Corpus V2 (ade_corpus_v2) | Public domain | Drug-effect spans |
| ICD NER training | MedMentions (bigbio/medmentions) | CC0 1.0 | 4,392 abstracts |
| ICD NER training | MACCROBAT (singh-aditya/MACCROBAT_biomedical_ner) | CC BY 4.0 | 200 case reports |
| ICD NER training | Curated + template-generated examples | Project-internal | ~160 sentences |
| ICD-10-CM codes | [atta00/icd10-codes](https://huggingface.co/datasets/atta00/icd10-codes) | MIT | 51,438 |
| MS-DRG grouper | [drgpy](https://pypi.org/project/drgpy/) | Apache 2.0 | ~770 DRGs |
| DRG weights | CMS IPPS Table 5 (FY 2026) | Public domain | ~770 DRGs |
| Assertion model | [bvanaken/clinical-assertion-negation-bert](https://huggingface.co/bvanaken/clinical-assertion-negation-bert) | Apache 2.0 | Fine-tuned on i2b2 |
| Abbreviations | [Meta-Inventory](https://zenodo.org/records/4567594) | CC-BY-4.0 | 104,057 |
| Disambiguation | [McGill-NLP/electra-medal](https://huggingface.co/McGill-NLP/electra-medal) | MIT | 14M abstracts |

## Tests

```bash
python -m pytest tests/ -v
```
