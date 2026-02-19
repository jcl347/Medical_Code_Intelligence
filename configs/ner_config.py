"""
Configuration for Medical/Biomedical NER experiments.

Defines dataset configs, model configs, and training hyperparameters
following state-of-the-art best practices for biomedical NER.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Supported pre-trained biomedical language models
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "pubmedbert": {
        "model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "description": "PubMedBERT - SOTA on BLURB biomedical NLP benchmark",
    },
    "biobert": {
        "model_name": "dmis-lab/biobert-v1.1",
        "description": "BioBERT v1.1 - pre-trained on PubMed abstracts",
    },
    "bio_clinicalbert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "Bio+Clinical BERT - pre-trained on MIMIC-III clinical notes",
    },
    "scibert": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "description": "SciBERT - pre-trained on Semantic Scholar papers",
    },
    "gatortron-base": {
        "model_name": "UFNLP/gatortron-base",
        "description": "GatorTron Base - pre-trained on 90B words of clinical text",
    },
}

# ---------------------------------------------------------------------------
# Supported public biomedical NER datasets (all on HuggingFace)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "ncbi_disease": {
        "hf_name": "ncbi/ncbi_disease",
        "revision": "refs/convert/parquet",
        "description": "NCBI Disease Corpus - disease name recognition",
        "entity_types": ["Disease"],
        "label_column": "ner_tags",
        "token_column": "tokens",
    },
    "bc5cdr": {
        "hf_name": "tner/bc5cdr",
        "revision": "refs/convert/parquet",
        "description": "BC5CDR - chemical and disease NER from PubMed articles",
        "entity_types": ["Chemical", "Disease"],
        "label_column": "tags",
        "token_column": "tokens",
    },
    "bc2gm": {
        "hf_name": "spyysalo/bc2gm_corpus",
        "description": "BC2GM - gene/protein mention recognition",
        "entity_types": ["Gene"],
        "label_column": "ner_tags",
        "token_column": "tokens",
    },
    "jnlpba": {
        "hf_name": "siddharthtumre/jnlpba-split",
        "description": "JNLPBA - biomedical entity recognition (proteins, DNA, RNA, cell lines, cell types)",
        "entity_types": ["Protein", "DNA", "RNA", "Cell_line", "Cell_type"],
        "label_column": "ner_tags",
        "token_column": "tokens",
    },
    "linnaeus": {
        "hf_name": "linnaeus",
        "description": "LINNAEUS - species name recognition",
        "entity_types": ["Species"],
        "label_column": "ner_tags",
        "token_column": "tokens",
    },
    "biomedical_ner_all": {
        "hf_name": "d4data/biomedical-ner-all",
        "description": "Combined biomedical NER (10+ entity types across multiple datasets)",
        "entity_types": [
            "Biomarker", "Body_part", "Cancer", "Chemical", "Diagnostic_procedure",
            "Disease", "Gene", "Medication", "Microorganism", "Sign_symptom",
        ],
        "label_column": "ner_tags",
        "token_column": "tokens",
    },
    # -------------------------------------------------------------------
    # ICD-specific and clinical coding datasets
    # -------------------------------------------------------------------
    "icd_ner": {
        "hf_name": "composite:ncbi_disease+bc5cdr+biomed_ner+ade_corpus+curated",
        "description": (
            "ICD-focused composite NER dataset. Combines five sources with "
            "a unified DIAGNOSIS entity type:\n"
            "  1. NCBI Disease (6.9K sentences, PubMed abstracts)\n"
            "  2. BC5CDR disease subset (1.5K abstracts)\n"
            "  3. BioMed NER DISORDER entities (clinical case reports)\n"
            "  4. ADE Corpus V2 adverse drug effects\n"
            "  5. Curated ICD clinical examples (80+ hand-crafted sentences)\n"
            "Designed for training models that feed into ICD-10-CM code "
            "resolution."
        ),
        "entity_types": ["DIAGNOSIS"],
        "format": "composite",
    },
    "biomed_ner": {
        "hf_name": "knowledgator/biomed_NER",
        "description": (
            "Biomedical NER with 24 entity types including DISORDER, "
            "MEDICAL_PROCEDURE, CLINICAL_DRUG, ANATOMICAL_STRUCTURE. "
            "Span-annotated (char offsets); converted to BIO at load time."
        ),
        "entity_types": [
            "DISORDER", "MEDICAL_PROCEDURE", "CLINICAL_DRUG",
            "ANATOMICAL_STRUCTURE", "PHENOTYPE", "CHEMICALS",
            "GENE_AND_GENE_PRODUCTS", "ORGANISM", "CELLS_AND_THEIR_COMPONENTS",
            "SIGNALING_MOLECULES", "BODY_SUBSTANCE", "FUNCTION", "ACTIVITY",
        ],
        "format": "span",  # uses char-offset annotations, not BIO columns
        "text_column": "text",
        "entities_column": "entities",
    },
    "icd10_terminology": {
        "hf_name": "awacke1/ICD10-Clinical-Terminology",
        "description": (
            "72,750 ICD-10-CM code/description pairs. Not a token-level NER "
            "dataset; used as a lookup table for entity→code mapping."
        ),
        "entity_types": ["ICD10_Code"],
        "code_column": "Code",
        "description_column": "Description",
        "format": "code_lookup",
    },
    "icd10_code_description": {
        "hf_name": "wangyichen25/ICD-10-CM_Code-Description_Pairs",
        "description": (
            "1.4M ICD-10-CM description→code pairs for training code "
            "prediction models. Instruction-formatted."
        ),
        "entity_types": ["ICD10_Code"],
        "format": "instruction",
        "input_column": "input",
        "output_column": "output",
    },
    # -------------------------------------------------------------------
    # Abbreviation disambiguation datasets
    # -------------------------------------------------------------------
    "medal": {
        "hf_name": "McGill-NLP/medal",
        "description": (
            "MeDAL: 14M PubMed abstracts with abbreviation annotations "
            "for pre-training abbreviation disambiguation models. "
            "Each example has an abbreviation in context with the "
            "correct expansion labeled."
        ),
        "format": "abbreviation_disambiguation",
        "text_column": "text",
        "location_column": "location",
        "label_column": "label",
    },
    "casi": {
        "hf_name": "mitclinicalml/clinical-ie",
        "description": (
            "CASI (Clinical Abbreviation Sense Inventory): 18,164 "
            "examples across 41 clinical acronyms with labeled senses "
            "from real clinical notes. Task #1 of the clinical-ie "
            "benchmark for abbreviation disambiguation evaluation."
        ),
        "format": "abbreviation_disambiguation",
        "task": "sense_disambiguation",
    },
}


@dataclass
class NERConfig:
    """Full configuration for a Medical NER experiment."""

    # --- Model ---
    model_key: str = "pubmedbert"
    model_name_or_path: Optional[str] = None  # override MODEL_CONFIGS

    # --- Dataset ---
    dataset_key: str = "ncbi_disease"
    max_seq_length: int = 512

    # --- Training hyperparameters (SOTA best practices) ---
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"

    # --- Mixed-precision & gradient accumulation ---
    fp16: bool = True
    gradient_accumulation_steps: int = 1

    # --- Early stopping ---
    early_stopping_patience: int = 5
    early_stopping_metric: str = "eval_f1"

    # --- CRF layer (optional, can boost NER performance) ---
    use_crf: bool = False

    # --- Label smoothing ---
    label_smoothing_factor: float = 0.0

    # --- Output ---
    output_dir: str = "outputs"
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 3

    # --- Reproducibility ---
    seed: int = 42

    # --- Clinical post-processing ---
    expand_shorthand: bool = True
    detect_negation: bool = True
    negation_scope_window: int = 6

    # --- Experiment tracking ---
    use_wandb: bool = False
    wandb_project: str = "medical-ner"
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.model_name_or_path is None:
            if self.model_key in MODEL_CONFIGS:
                self.model_name_or_path = MODEL_CONFIGS[self.model_key]["model_name"]
            else:
                raise ValueError(
                    f"Unknown model_key '{self.model_key}'. "
                    f"Choose from: {list(MODEL_CONFIGS.keys())} or set model_name_or_path directly."
                )
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_key}_{self.dataset_key}"
