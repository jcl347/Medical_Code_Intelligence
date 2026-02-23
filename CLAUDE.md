# CLAUDE.md

## Project Overview

Medical Code Intelligence is a biomedical NER system that extracts diagnosis entities from clinical text, links them to ICD-10-CM codes, and estimates financial impact via MS-DRG cost scoping. The pipeline chains six stages: physician shorthand expansion, adversarial-trained transformer NER, negation/assertion detection, ICD-10-CM code resolution, and MS-DRG cost estimation.

**Python >=3.9 | PyTorch >=2.0 | HuggingFace Transformers >=4.36**

## Repository Structure

```
src/
  clinical/          # Core pipeline components
    pipeline.py        MedicalCodingPipeline — main user-facing API
    icd_codes.py       ICDCodeLookup — TF-IDF entity linker over 51K ICD-10-CM codes
    drg_costs.py       DRGCostEstimator — MS-DRG assignment + cost impact analysis
    negation.py        NegationDetector — rule-based ConText/NegEx assertion (100+ triggers)
    assertion.py       AssertionClassifier — transformer assertion (clinical-assertion-negation-bert)
    shorthand.py       ShorthandExpander — abbreviation expansion with offset tracking
    abbreviation_disambiguator.py  AbbreviationDisambiguator — ELECTRA-MeDAL contextual sense selection
    _icd_fallback.py   Offline fallback: 45 common ICD codes for CI/testing
    _shorthand_fallback.py  Offline fallback: ~280 clinical abbreviations for CI/testing
  data/              # Dataset loading and preprocessing
    icd_dataset.py     load_icd_ner_dataset() — composite dataset from 7 sources
    dataset_loader.py  load_ner_dataset() — HuggingFace dataset loading + span-to-BIO conversion
    preprocessing.py   tokenize_and_align_labels() — subword label alignment
    data_utils.py      create_data_collator(), split_long_sentences()
  models/            # Model construction
    ner_model.py       build_ner_model() — wraps AutoModelForTokenClassification
    crf_model.py       CRF — optional linear-chain CRF layer with Viterbi decoding
  training/          # Training pipeline
    trainer.py         build_trainer(), build_training_args() — HuggingFace Trainer setup
    adversarial.py     AdversarialTrainer, FGM, PGD — adversarial training for +0.5-1.5% F1
    callbacks.py       EarlyStoppingWithLogging
  evaluation/        # Metrics and analysis
    metrics.py         compute_ner_metrics() — entity-level P/R/F1 via seqeval
    error_analysis.py  analyse_errors() — boundary/type/FP/FN categorization
  inference/         # Prediction
    predictor.py       NERPredictor — loads fine-tuned models, subword aggregation
    entity_utils.py    post_process_entities() — stopword filter, fragment merging

scripts/
  train.py           Main training CLI (supports --adversarial flag)
  predict.py         Inference CLI (single, batch, interactive modes)
  evaluate.py        Evaluation CLI with optional error analysis
  benchmark.py       Multi-model x multi-dataset benchmarking

configs/
  ner_config.py      MODEL_CONFIGS, DATASET_CONFIGS, NERConfig dataclass

notebooks/
  demo_all_components.ipynb  Interactive demo of all pipeline components (runs on CPU)

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

# Train with adversarial training (FGM) for improved F1
python scripts/train.py --model pubmedbert --dataset icd_ner --adversarial

# Train with PGD adversarial training (stronger but slower)
python scripts/train.py --model pubmedbert --dataset icd_ner --adversarial --adv-method pgd

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
| `icd_ner` | Composite (7 sources — see below) | Unified DIAGNOSIS NER |
| `ncbi_disease` | NCBI Disease Corpus | Disease recognition |
| `bc5cdr` | BioCreative V CDR | Disease (chemical filtered out) |
| `bc2gm` | BioCreative II GM | Gene/protein mention |
| `jnlpba` | JNLPBA 2004 | Biomedical entity types |
| `biomed_ner` | knowledgator/biomed_NER | Clinical spans |

### ICD NER Composite Dataset (7 Sources)

The `icd_ner` dataset merges seven sources with unified DIAGNOSIS labels:

1. **NCBI Disease** — 6.9K sentences from PubMed abstracts
2. **BC5CDR disease subset** — 1.5K abstracts (chemical entities filtered out)
3. **BioMed NER DISORDER/PHENOTYPE** — clinical case reports from knowledgator/biomed_NER
4. **ADE Corpus V2** — adverse drug effect spans from ade_corpus_v2
5. **Curated ICD examples** — 80+ hand-crafted + ~100 template-generated sentences targeting common NER failure patterns (abbreviations, multi-word boundaries, lab value confusion, negation contexts, rare diseases, high-frequency ICD codes)
6. **MedMentions** (optional) — up to 5K examples from 4,392 PubMed abstracts with 350K+ UMLS entity mentions, filtered for disease/disorder semantic types (T047, T048, T019, T046, T191)
7. **MACCROBAT** (optional) — up to 3K examples from 200 clinical case reports with DISEASE_DISORDER entities, providing clinical-note-style text that PubMed abstracts lack

Sources 6 and 7 download from HuggingFace on first use and fall back gracefully if unavailable.

## Adversarial Training

FGM (Fast Gradient Method) and PGD (Projected Gradient Descent) adversarial training on word embeddings, implemented in `src/training/adversarial.py`. Improves entity-level F1 by +0.5-1.5% on biomedical NER benchmarks.

**How it works**: Perturbs the embedding layer in the direction of the loss gradient, then trains on both clean and perturbed inputs. This regularizes the model against small input variations.

| Method | Compute Overhead | F1 Gain | Config |
|--------|-----------------|---------|--------|
| FGM | ~2x training time | +0.5-1.5% | `--adversarial` (default) |
| PGD (K=3) | ~4x training time | +0.5-1.5% | `--adversarial --adv-method pgd` |

Key parameters in `NERConfig`:
- `use_adversarial_training`: Enable/disable (default: False)
- `adv_method`: `"fgm"` or `"pgd"`
- `adv_epsilon`: Perturbation magnitude (default: 1.0 for FGM, 0.3 for PGD)
- `pgd_alpha`: PGD step size (default: 0.1)
- `pgd_steps`: PGD iterations (default: 3)

Based on: RanAT4BIE (2025), FreeLB (ICLR 2020), Miyato et al. (2017).

## MS-DRG Cost Scoping

The `DRGCostEstimator` in `src/clinical/drg_costs.py` maps ICD-10-CM codes to Medicare Severity Diagnosis Related Groups (MS-DRGs) and estimates financial impact.

**Key concepts**:
- **MS-DRG**: Groups inpatient stays into ~770 payment categories
- **Relative Weight**: Multiplier reflecting resource intensity (1.0 = national average)
- **CC/MCC Tiers**: Secondary diagnoses that increase severity and payment
- **Revenue at Risk**: Payment gap between current DRG and optimal CC/MCC capture

**Pipeline integration**: When `resolve_drg=True`, the pipeline collects ICD codes from all affirmed entities, runs DRG grouping, and attaches cost analysis to the primary diagnosis entity.

**Data sources**:
- `drgpy` library for ICD-10 to MS-DRG grouper logic (optional, `pip install drgpy`)
- CMS IPPS Table 5 relative weights (can load from Excel file via `table5_path`)
- Built-in fallback: 32 common medical DRGs with FY 2026 weights for CI/testing
- FY 2026 national standardized amount: $6,752.61

```python
from src.clinical.drg_costs import DRGCostEstimator

estimator = DRGCostEstimator()
# With drgpy installed:
result = estimator.get_drg(["J18.9", "E11.9", "N17.9"])
# Analyze CC/MCC impact:
analysis = estimator.analyze_cost_impact(["J18.9", "E11.9"])
print(f"Revenue at risk: ${analysis.revenue_at_risk:,.2f}")
```

## Architecture Decisions

- **Pipeline approach** (NER -> Assertion -> ICD -> DRG): Components are independently swappable. Matches clinical NLP patterns from MedSpacy, cTAKES, SciSpacy.
- **Single DIAGNOSIS label** for ICD NER: Collapses Disease/Disorder/Phenotype into one label to maximize training signal.
- **TF-IDF character n-grams** for ICD linking: Character 3/4-grams capture medical morphology (e.g., "cardio-", "-itis"). Scales to 50K+ codes without GPU.
- **Dual negation strategies**: Rule-based (fast, deterministic, no GPU) and transformer-based (learned, handles edge cases). Default is rule-based.
- **Adversarial training**: FGM/PGD perturbation on embeddings improves robustness and F1 with no architecture changes — just a training-time regularizer.
- **7-source composite dataset**: Combines public corpora, clinical case reports, and template-generated examples targeting documented NER failure patterns (abbreviations, boundary errors, lab value confusion).
- **MS-DRG cost scoping**: Maps extracted ICD codes to DRGs for financial impact estimation, with CC/MCC tier comparison to quantify revenue at risk.
- **Offline fallbacks**: All external downloads (ICD codes, abbreviations, DRG weights) have built-in fallback data so tests and CI work without network access.
- **BIO label scheme**: Normalized across all datasets. Garbage labels from source corpora are cleaned automatically.

## Key Configuration (configs/ner_config.py)

Default training hyperparameters in `NERConfig`:
- Learning rate: `5e-5`, Warmup: `10%`, Weight decay: `0.01`
- Batch size: `16`, Max sequence length: `512`
- Early stopping patience: `5` (metric: entity-level F1)
- FP16 mixed precision: enabled by default
- Adversarial training: disabled by default (`use_adversarial_training=False`)
- Negation detection: enabled by default, scope window: 6 words
- DRG cost scoping: disabled by default (`resolve_drg=False`)

## Pipeline Flow

```
Clinical Text
  -> ShorthandExpander (abbreviation expansion with character offset tracking)
  -> NER Model (transformer token classification, BIO scheme, optional adversarial training)
  -> post_process_entities() (stopword filter, fragment merging)
  -> NegationDetector or AssertionClassifier (6 assertion statuses)
  -> ICDCodeLookup (TF-IDF matching against 51K codes)
  -> DRGCostEstimator (ICD-10 -> MS-DRG -> cost estimate + CC/MCC analysis)
  -> MedicalEntity list (text, label, negation, ICD codes, DRG info, scores)
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

## Notebooks

The `notebooks/demo_all_components.ipynb` notebook demonstrates every pipeline component interactively. It runs entirely on CPU using built-in fallback data (no GPU, no network downloads).

### Keeping the Notebook Up to Date

When making changes to the repository, update the notebook to reflect those changes:

- **New pipeline component**: Add a new section (markdown header + code cells) demonstrating the component's constructor, key methods, and example output. Follow the existing pattern: import, instantiate with offline-friendly defaults, show 2-3 usage examples.
- **Changed API signature**: Update the relevant code cells to match the new parameters. Search the notebook for the class/function name.
- **New configuration option in `NERConfig`**: Add it to the config display cell (Section 1) and, if user-facing, to the custom config example.
- **New dataset source**: Mention it in Section 8 (Curated ICD Dataset) or add a new cell if it requires distinct loading logic.
- **New CLI flag**: Update the CLI reference cell (Section 12).
- **New test file**: Add it to the test reference cell (Section 13).
- **Removed or renamed module**: Remove or rename the corresponding notebook section and update all imports.

After editing the notebook, verify it runs cleanly:
```bash
cd notebooks && jupyter nbconvert --to notebook --execute demo_all_components.ipynb --output /dev/null
```

### Keeping the README Up to Date

When making changes to the repository, update `README.md` to reflect those changes:

- **New pipeline component**: Add it to the "Project Structure" tree, add a Python API usage example under the appropriate section, and update the pipeline flow diagram if it adds a new stage.
- **New CLI flag**: Add it to the relevant script's flag table in the Command Reference section.
- **New model or dataset**: Add a row to the Supported Models or Available Datasets table.
- **Changed default hyperparameters**: Update the Quick Commands examples and any affected API examples.
- **New data source for `icd_ner`**: Add a row to the ICD NER Composite Dataset table and increment the source count.

## Common Pitfalls

- The `icd_ner` dataset is built at runtime by merging up to 7 HuggingFace datasets. First load downloads ~500MB+. Subsequent loads use cache. Sources 6 (MedMentions) and 7 (MACCROBAT) are optional and skipped gracefully if unavailable.
- GatorTron-base (345M params) needs ~2.5x more GPU memory than the 110M models. Reduce batch size or use gradient accumulation if OOM.
- Adversarial training (`--adversarial`) roughly doubles training time (FGM) or quadruples it (PGD). The F1 gain is +0.5-1.5% on strong baselines.
- The ICD code lookup downloads 51K codes from `atta00/icd10-codes` on first use. Falls back to 45 built-in codes if download fails.
- MS-DRG grouping requires `drgpy` (`pip install drgpy`). Without it, cost estimation uses the fallback weight table only (no ICD-to-DRG grouper logic).
- Shorthand expansion tries 3 data sources in order (Zenodo -> MEDIALpy -> built-in). Network failures are handled gracefully.
- The `--model-path` flag in predict.py/evaluate.py expects a directory containing a saved HuggingFace model (config.json + model weights), not a model key.
