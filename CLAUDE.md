# CLAUDE.md

## Project Overview

Medical Code Intelligence is a biomedical NER system that extracts diagnosis entities from clinical text and links them to ICD-10-CM codes. The pipeline chains four stages: physician shorthand expansion, transformer-based NER, negation/assertion detection, and ICD-10-CM code resolution.

**Python >=3.9 | PyTorch >=2.0 | HuggingFace Transformers >=4.36**

## Repository Structure

```
src/
  clinical/          # Core pipeline components
    pipeline.py        MedicalCodingPipeline — main user-facing API
    icd_codes.py       ICDCodeLookup — TF-IDF entity linker over 51K ICD-10-CM codes
    negation.py        NegationDetector — rule-based ConText/NegEx assertion (100+ triggers)
    assertion.py       AssertionClassifier — transformer assertion (clinical-assertion-negation-bert)
    shorthand.py       ShorthandExpander — abbreviation expansion with offset tracking
    abbreviation_disambiguator.py  AbbreviationDisambiguator — ELECTRA-MeDAL contextual sense selection
    _icd_fallback.py   Offline fallback: 45 common ICD codes for CI/testing
    _shorthand_fallback.py  Offline fallback: ~280 clinical abbreviations for CI/testing
  data/              # Dataset loading and preprocessing
    icd_dataset.py     load_icd_ner_dataset() — composite dataset from 5 sources
    dataset_loader.py  load_ner_dataset() — HuggingFace dataset loading + span-to-BIO conversion
    preprocessing.py   tokenize_and_align_labels() — subword label alignment
    data_utils.py      create_data_collator(), split_long_sentences()
  models/            # Model construction
    ner_model.py       build_ner_model() — wraps AutoModelForTokenClassification
    crf_model.py       CRF — optional linear-chain CRF layer with Viterbi decoding
  training/          # Training pipeline
    trainer.py         build_trainer(), build_training_args() — HuggingFace Trainer setup
    callbacks.py       EarlyStoppingWithLogging
  evaluation/        # Metrics and analysis
    metrics.py         compute_ner_metrics() — entity-level P/R/F1 via seqeval
    error_analysis.py  analyse_errors() — boundary/type/FP/FN categorization
  inference/         # Prediction
    predictor.py       NERPredictor — loads fine-tuned models, subword aggregation
    entity_utils.py    post_process_entities() — stopword filter, fragment merging

scripts/
  train.py           Main training CLI
  predict.py         Inference CLI (single, batch, interactive modes)
  evaluate.py        Evaluation CLI with optional error analysis
  benchmark.py       Multi-model x multi-dataset benchmarking

configs/
  ner_config.py      MODEL_CONFIGS, DATASET_CONFIGS, NERConfig dataclass

tests/               12 test files, ~2,900 lines total
```

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_negation.py -v

# Train a model
python scripts/train.py --model pubmedbert --dataset icd_ner --epochs 20 --lr 5e-5

# Train with CRF layer and mixed precision
python scripts/train.py --model bio_clinicalbert --dataset icd_ner --use-crf --fp16

# Run inference (single text)
python scripts/predict.py --model-path outputs/pubmedbert_icd_ner/best_model --text "Pt denies cp or sob."

# Run inference (batch)
python scripts/predict.py --model-path outputs/pubmedbert_icd_ner/best_model --input-file data.txt --output-file results.json

# Evaluate a trained model
python scripts/evaluate.py --model-path outputs/pubmedbert_icd_ner/best_model --dataset icd_ner --error-analysis

# Benchmark multiple models
python scripts/benchmark.py --models pubmedbert biobert bio_clinicalbert --datasets ncbi_disease bc5cdr
```

## Supported Models

| Key | HuggingFace ID | Params | Notes |
|-----|---------------|--------|-------|
| `pubmedbert` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | 110M | SOTA on BLURB benchmark |
| `biobert` | `dmis-lab/biobert-v1.1` | 110M | PubMed pre-trained |
| `bio_clinicalbert` | `emilyalsentzer/Bio_ClinicalBERT` | 110M | MIMIC-III clinical notes |
| `scibert` | `allenai/scibert_scivocab_uncased` | 110M | Scientific papers |
| `gatortron-base` | `UFNLP/gatortron-base` | 345M | 90B words clinical text |

Custom models can be passed via `--model-path <hf_id_or_local_path>`.

## Supported Datasets

| Key | Source | Task |
|-----|--------|------|
| `icd_ner` | Composite (NCBI + BC5CDR + BioMed NER + ADE + curated) | Unified DIAGNOSIS NER |
| `ncbi_disease` | NCBI Disease Corpus | Disease recognition |
| `bc5cdr` | BioCreative V CDR | Disease (chemical filtered out) |
| `bc2gm` | BioCreative II GM | Gene/protein mention |
| `jnlpba` | JNLPBA 2004 | Biomedical entity types |
| `biomed_ner` | knowledgator/biomed_NER | Clinical spans |

## Architecture Decisions

- **Pipeline approach** (NER -> Assertion -> ICD): Components are independently swappable. Matches clinical NLP patterns from MedSpacy, cTAKES, SciSpacy.
- **Single DIAGNOSIS label** for ICD NER: Collapses Disease/Disorder/Phenotype into one label to maximize training signal.
- **TF-IDF character n-grams** for ICD linking: Character 3/4-grams capture medical morphology (e.g., "cardio-", "-itis"). Scales to 50K+ codes without GPU.
- **Dual negation strategies**: Rule-based (fast, deterministic, no GPU) and transformer-based (learned, handles edge cases). Default is rule-based.
- **Offline fallbacks**: All external downloads (ICD codes, abbreviations) have built-in fallback data so tests and CI work without network access.
- **BIO label scheme**: Normalized across all datasets. Garbage labels from source corpora are cleaned automatically.

## Key Configuration (configs/ner_config.py)

Default training hyperparameters in `NERConfig`:
- Learning rate: `5e-5`, Warmup: `10%`, Weight decay: `0.01`
- Batch size: `16`, Max sequence length: `512`
- Early stopping patience: `5` (metric: entity-level F1)
- FP16 mixed precision: enabled by default
- Negation detection: enabled by default, scope window: 6 words

## Pipeline Flow

```
Clinical Text
  -> ShorthandExpander (abbreviation expansion with character offset tracking)
  -> NER Model (transformer token classification, BIO scheme)
  -> post_process_entities() (stopword filter, fragment merging)
  -> NegationDetector or AssertionClassifier (6 assertion statuses)
  -> ICDCodeLookup (TF-IDF matching against 51K codes)
  -> MedicalEntity list (text, label, negation, ICD codes, scores)
```

## Testing

Tests use mocked models and fallback data to avoid network calls and GPU requirements. All tests should run on CPU in CI.

```bash
# Full suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Key test files:
- `test_negation.py`, `test_assertion.py` — assertion detection (200+ assertions)
- `test_icd_ner_dataset.py` — composite dataset loading, garbage label cleaning
- `test_pipeline.py`, `test_icd_pipeline.py` — end-to-end pipeline integration
- `test_shorthand.py`, `test_disambiguation.py` — abbreviation handling
- `test_preprocessing.py` — tokenization and label alignment
- `test_entity_postprocessing.py` — entity filtering and merging

## Conventions

- Imports use `src.` prefix (e.g., `from src.clinical.negation import NegationDetector`)
- Trained models save to `outputs/{model}_{dataset}/best_model/`
- All datasets load via HuggingFace `datasets` library
- Docstrings follow NumPy style
- Config uses Python dataclasses (not YAML/JSON)
- pytest config is in `pyproject.toml` (testpaths, addopts)

## Common Pitfalls

- The `icd_ner` dataset is built at runtime by merging 5 HuggingFace datasets. First load downloads ~500MB. Subsequent loads use cache.
- GatorTron-base (345M params) needs ~2.5x more GPU memory than the 110M models. Reduce batch size or use gradient accumulation if OOM.
- The ICD code lookup downloads 51K codes from `atta00/icd10-codes` on first use. Falls back to 45 built-in codes if download fails.
- Shorthand expansion tries 3 data sources in order (Zenodo -> MEDIALpy -> built-in). Network failures are handled gracefully.
- The `--model-path` flag in predict.py/evaluate.py expects a directory containing a saved HuggingFace model (config.json + model weights), not a model key.
