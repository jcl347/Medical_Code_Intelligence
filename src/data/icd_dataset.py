"""
ICD-focused composite NER dataset.

Combines multiple disease-centric public datasets into a single training
corpus with a unified DIAGNOSIS entity type, specifically designed for
training NER models that feed into ICD-10-CM code resolution.

Datasets combined
-----------------
- **NCBI Disease Corpus**: 6,892 disease mentions from 793 PubMed abstracts.
  Disease entities → DIAGNOSIS.
- **BC5CDR** (BioCreative V CDR): Disease mentions from 1,500 PubMed
  abstracts. Chemical entities are filtered out; Disease → DIAGNOSIS.
- **BioMed NER** (knowledgator/biomed_NER): DISORDER entities from
  clinical case reports. Rich, diverse diagnosis vocabulary.
- **ADE Corpus V2**: Adverse drug effect spans from clinical text.
  Effect entities → DIAGNOSIS (e.g. "ototoxicity", "seizures").
- **Curated ICD Examples**: Hand-crafted + template-generated clinical
  sentences covering 80+ hand-crafted examples plus ~400 template-generated
  sentences targeting common NER failure patterns (abbreviations, multi-word
  boundaries, lab value confusion, negation contexts).
- **MedMentions** (optional, ibm/MedMentions-ZS): 29K pre-tokenized PubMed
  abstracts with UMLS entity mentions in BIO format. Disease entities
  (T038) are mapped to DIAGNOSIS.
- **MACCROBAT** (optional, singh-aditya/MACCROBAT_biomedical_ner): 200
  clinical case reports with DISEASE_DISORDER entities. Provides clinical-
  note-style text that PubMed abstracts lack.

All entity labels are normalized to a two-label BIO scheme:
    O, B-DIAGNOSIS, I-DIAGNOSIS

This simplification is intentional: the model's job is to detect *any*
diagnosable condition; the downstream ICD-10-CM code resolution step
(via ``ICDCodeLookup``) maps the extracted text span to a specific code.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

# Unified label scheme for ICD NER
ICD_NER_LABELS: List[str] = ["O", "B-DIAGNOSIS", "I-DIAGNOSIS"]

# ---------------------------------------------------------------------------
# Garbage label cleaning
# ---------------------------------------------------------------------------

# Lowercase tokens that should never be standalone DIAGNOSIS entities.
# These appear in the source corpora due to broken BIO alignment where
# function words inside multi-word disease mentions got isolated B-DIAGNOSIS
# tags (e.g. "of" 154x, "and" 145x, "the" 72x in the combined corpus).
_GARBAGE_STANDALONE_TOKENS = frozenset({
    # Function words / prepositions / conjunctions
    "a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "by",
    "with", "from", "at", "as", "is", "are", "was", "were", "be", "been",
    "not", "no", "nor", "but", "so", "if", "it", "its", "that", "this",
    "than", "then", "has", "had", "have", "do", "does", "did",
    # Punctuation / noise (tokens that look like words)
    "type", "due",
    # Single characters (except legit abbreviations handled separately)
})
ICD_NER_LABEL2ID = {label: i for i, label in enumerate(ICD_NER_LABELS)}

# Common feature schema so concatenation always succeeds
_UNIFIED_FEATURES = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(Value("int64")),
    "ner_labels": Sequence(Value("string")),
})


def _clean_garbage_labels(dataset: DatasetDict) -> DatasetDict:
    """
    Fix broken BIO annotations in the merged corpus.

    The source corpora (NCBI Disease, BC5CDR) contain isolated B-DIAGNOSIS
    tags on function words like "of", "and", "the" — leftovers from
    multi-word disease mentions whose surrounding B/I tokens were lost
    during format conversion.  These garbage labels hurt model precision.

    Rules:
    - A standalone B-DIAGNOSIS (not followed by I-DIAGNOSIS) on a
      lowercase token that appears in ``_GARBAGE_STANDALONE_TOKENS``
      is flipped to O.
    - Uppercase tokens (e.g. AS, AT, WAS) are preserved — they are
      legitimate disease abbreviations.
    - Single-character lowercase tokens (e.g. "a") are also flipped to O.
    """

    def _map(example):
        tokens = example["tokens"]
        labels = list(example["ner_labels"])
        n = len(labels)
        cleaned = 0

        for i in range(n):
            if labels[i] != "B-DIAGNOSIS":
                continue

            # Check if this B- has a following I- (part of a real entity)
            has_continuation = (i + 1 < n and labels[i + 1] == "I-DIAGNOSIS")
            if has_continuation:
                continue

            token = tokens[i]
            # Preserve uppercase tokens (legitimate abbreviations like AS, AT)
            if token.isupper() and len(token) >= 2:
                continue

            # Clean garbage: lowercase function words and single chars
            if token.lower() in _GARBAGE_STANDALONE_TOKENS or len(token) <= 1:
                labels[i] = "O"
                cleaned += 1

        return {
            "tokens": tokens,
            "ner_tags": [ICD_NER_LABEL2ID[lab] for lab in labels],
            "ner_labels": labels,
        }

    return dataset.map(_map, desc="Cleaning garbage labels")


def load_icd_ner_dataset(
    cache_dir: Optional[str] = None,
) -> Tuple[DatasetDict, List[str]]:
    """
    Load the composite ICD NER dataset.

    Merges eight sources with unified DIAGNOSIS labels:
    1. NCBI Disease Corpus (PubMed abstracts)
    2. BC5CDR disease subset (PubMed articles)
    3. BioMed NER DISORDER entities (clinical case reports)
    4. ADE Corpus V2 adverse effect entities
    5. Curated clinical ICD examples
    6. MedMentions (optional, disease semantic types from PubMed)
    7. MACCROBAT (optional, clinical case reports)
    8. Curated discharge summary examples (DRG-relevant diagnoses)

    Parameters
    ----------
    cache_dir : str, optional
        HuggingFace cache directory.

    Returns
    -------
    dataset : DatasetDict
        Columns: tokens (List[str]), ner_tags (List[int]), ner_labels (List[str]).
    label_list : list of str
        ``["O", "B-DIAGNOSIS", "I-DIAGNOSIS"]``
    """
    logger.info("Loading ICD NER composite dataset...")

    all_sources: List[Tuple[str, DatasetDict]] = []

    # --- Source 1: NCBI Disease → DIAGNOSIS ---
    logger.info("  [1/5] Loading NCBI Disease corpus...")
    ncbi = load_dataset(
        "ncbi/ncbi_disease",
        cache_dir=cache_dir,
        revision="refs/convert/parquet",
    )
    ncbi = _normalize_ncbi_to_diagnosis(ncbi)
    all_sources.append(("ncbi", ncbi))

    # --- Source 2: BC5CDR Disease → DIAGNOSIS (Chemical → O) ---
    logger.info("  [2/5] Loading BC5CDR corpus (disease entities only)...")
    bc5cdr = load_dataset(
        "tner/bc5cdr",
        cache_dir=cache_dir,
        revision="refs/convert/parquet",
    )
    bc5cdr = _normalize_bc5cdr_to_diagnosis(bc5cdr)
    all_sources.append(("bc5cdr", bc5cdr))

    # --- Source 3: BioMed NER DISORDER → DIAGNOSIS ---
    logger.info("  [3/5] Loading BioMed NER corpus (DISORDER entities)...")
    try:
        biomed = _load_biomed_ner_disorders(cache_dir=cache_dir)
        all_sources.append(("biomed_ner", biomed))
    except Exception as e:
        logger.warning("  Could not load BioMed NER: %s (skipping)", e)

    # --- Source 4: ADE Corpus V2 adverse effects → DIAGNOSIS ---
    logger.info("  [4/5] Loading ADE Corpus V2 (adverse effect entities)...")
    try:
        ade = _load_ade_effects(cache_dir=cache_dir)
        all_sources.append(("ade_corpus", ade))
    except Exception as e:
        logger.warning("  Could not load ADE Corpus V2: %s (skipping)", e)

    # --- Source 5: Curated ICD clinical examples (hand-crafted + template-generated) ---
    logger.info("  [5/8] Adding curated ICD clinical examples...")
    curated = _get_curated_icd_examples()
    all_sources.append(("curated_icd", curated))

    # --- Source 6: MedMentions (Disease/Disorder semantic types) ---
    logger.info("  [6/8] Loading MedMentions (disease semantic types)...")
    try:
        medmentions = _load_medmentions_diseases(cache_dir=cache_dir)
        all_sources.append(("medmentions", medmentions))
    except Exception as e:
        logger.warning("  Could not load MedMentions: %s (skipping)", e)

    # --- Source 7: MACCROBAT (clinical case DISEASE_DISORDER entities) ---
    logger.info("  [7/8] Loading MACCROBAT clinical case reports...")
    try:
        maccrobat = _load_maccrobat_diseases(cache_dir=cache_dir)
        all_sources.append(("maccrobat", maccrobat))
    except Exception as e:
        logger.warning("  Could not load MACCROBAT: %s (skipping)", e)

    # --- Source 8: Curated discharge summary examples (DRG-relevant) ---
    logger.info("  [8/8] Adding curated discharge summary examples...")
    discharge = _get_discharge_summary_examples()
    all_sources.append(("discharge_summaries", discharge))

    # --- Clean garbage labels from all sources ---
    logger.info("  Cleaning garbage labels from all sources...")
    cleaned_sources = []
    for name, ds in all_sources:
        ds = _clean_garbage_labels(ds)
        cleaned_sources.append((name, ds))

    # --- Cast to common schema and merge ---
    for name, ds in cleaned_sources:
        for split in list(ds.keys()):
            # Skip splits missing required columns (e.g. empty after filtering)
            if "ner_labels" not in ds[split].column_names:
                del ds[split]
                continue
            ds[split] = ds[split].cast(_UNIFIED_FEATURES)

    merged = {}
    for split in ["train", "validation", "test"]:
        parts = []
        for name, ds in cleaned_sources:
            if split in ds:
                parts.append(ds[split])
        if parts:
            merged[split] = concatenate_datasets(parts)

    dataset = DatasetDict(merged)

    for split in dataset:
        n_examples = len(dataset[split])
        n_entities = sum(
            1
            for ex in dataset[split]
            for lab in ex["ner_labels"]
            if lab.startswith("B-")
        )
        logger.info(
            "  %s: %d examples, %d diagnosis entities",
            split, n_examples, n_entities,
        )

    return dataset, ICD_NER_LABELS


# ---------------------------------------------------------------------------
# Per-dataset normalization
# ---------------------------------------------------------------------------

def _normalize_ncbi_to_diagnosis(dataset: DatasetDict) -> DatasetDict:
    """Map NCBI Disease integer tags to unified DIAGNOSIS labels."""
    # NCBI uses: 0 = O, 1 = B-Disease, 2 = I-Disease
    tag_map = {0: "O", 1: "B-DIAGNOSIS", 2: "I-DIAGNOSIS"}

    def _map(example):
        labels = [tag_map.get(t, "O") for t in example["ner_tags"]]
        return {
            "tokens": example["tokens"],
            "ner_tags": [ICD_NER_LABEL2ID[lab] for lab in labels],
            "ner_labels": labels,
        }

    cols_to_keep = {"tokens", "ner_tags", "ner_labels"}
    cols_to_remove = [
        c for c in dataset["train"].column_names if c not in cols_to_keep
    ]
    return dataset.map(_map, remove_columns=cols_to_remove)


def _normalize_bc5cdr_to_diagnosis(dataset: DatasetDict) -> DatasetDict:
    """Keep Disease → DIAGNOSIS, drop Chemical → O from BC5CDR."""
    # The parquet revision stores tags as plain ints (no ClassLabel metadata),
    # so we provide the canonical label names as a fallback.
    _BC5CDR_LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

    tag_feature = dataset["train"].features["tags"].feature
    if hasattr(tag_feature, "names"):
        label_names = tag_feature.names
    else:
        label_names = _BC5CDR_LABELS

    # Build mapping: Disease labels → DIAGNOSIS, everything else → O
    name_map = {}
    for name in label_names:
        if "Disease" in name:
            name_map[name] = name.replace("Disease", "DIAGNOSIS")
        else:
            name_map[name] = "O"

    def _map(example):
        original_labels = [label_names[t] for t in example["tags"]]
        diagnosis_labels = [name_map.get(lab, "O") for lab in original_labels]
        return {
            "tokens": example["tokens"],
            "ner_tags": [ICD_NER_LABEL2ID[lab] for lab in diagnosis_labels],
            "ner_labels": diagnosis_labels,
        }

    cols_to_keep = {"tokens", "ner_tags", "ner_labels"}
    cols_to_remove = [
        c for c in dataset["train"].column_names if c not in cols_to_keep
    ]
    return dataset.map(_map, remove_columns=cols_to_remove)


# ---------------------------------------------------------------------------
# Source 3: BioMed NER — DISORDER entities from clinical case reports
# ---------------------------------------------------------------------------

def _default_tokenize(text: str) -> List[Tuple[int, int]]:
    """Simple whitespace tokenizer returning (start, end) spans."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _spans_to_diagnosis_bio(
    text: str,
    entities: List[Dict],
    target_classes: frozenset,
) -> Tuple[List[str], List[str]]:
    """
    Convert character-offset span annotations to DIAGNOSIS BIO tags.

    Only entities whose class (uppercased) is in *target_classes* become
    B-DIAGNOSIS / I-DIAGNOSIS; everything else is O.
    """
    token_spans = _default_tokenize(text)
    tokens = [text[s:e] for s, e in token_spans]
    labels = ["O"] * len(tokens)

    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0))
    for ent in sorted_ents:
        ent_class = re.sub(r"[\s/]+", "_", ent.get("class", "").strip().upper())
        if ent_class not in target_classes:
            continue

        ent_start = ent.get("start", 0)
        ent_end = ent.get("end", 0)
        first = True
        for i, (ts, te) in enumerate(token_spans):
            if ts >= ent_start and te <= ent_end:
                labels[i] = "B-DIAGNOSIS" if first else "I-DIAGNOSIS"
                first = False
            elif ts < ent_end and te > ent_start:
                overlap = min(te, ent_end) - max(ts, ent_start)
                if overlap > (te - ts) / 2:
                    labels[i] = "B-DIAGNOSIS" if first else "I-DIAGNOSIS"
                    first = False

    return tokens, labels


# DISORDER and PHENOTYPE are both ICD-mappable diagnosis types
_BIOMED_DIAGNOSIS_CLASSES = frozenset({"DISORDER", "PHENOTYPE"})


def _load_biomed_ner_disorders(
    cache_dir: Optional[str] = None,
    max_examples: int = 5000,
) -> DatasetDict:
    """
    Load DISORDER + PHENOTYPE entities from knowledgator/biomed_NER.

    These are span-annotated clinical case reports with rich, diverse
    diagnosis vocabulary (encephalopathy, seizures, cardiomyopathy, etc.).
    Converts spans to DIAGNOSIS BIO tags.
    """
    raw = load_dataset("knowledgator/biomed_NER", cache_dir=cache_dir)

    all_tokens = []
    all_tags = []
    all_labels = []

    # Process train split (the main split)
    split_name = "train" if "train" in raw else list(raw.keys())[0]
    for i, example in enumerate(raw[split_name]):
        if i >= max_examples:
            break

        text = example.get("text", "")
        entities = example.get("entities", [])
        if not text or not entities:
            continue

        # Only keep examples that have at least one diagnosis entity
        has_diagnosis = any(
            re.sub(r"[\s/]+", "_", e.get("class", "").strip().upper())
            in _BIOMED_DIAGNOSIS_CLASSES
            for e in entities
        )
        if not has_diagnosis:
            continue

        tokens, labels = _spans_to_diagnosis_bio(
            text, entities, _BIOMED_DIAGNOSIS_CLASSES,
        )
        all_tokens.append(tokens)
        all_labels.append(labels)
        all_tags.append([ICD_NER_LABEL2ID[lab] for lab in labels])

    if not all_tokens:
        raise RuntimeError("No DISORDER entities found in BioMed NER")

    # Split 80/10/10
    n = len(all_tokens)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    def _make_ds(start, end):
        return Dataset.from_dict({
            "tokens": all_tokens[start:end],
            "ner_tags": all_tags[start:end],
            "ner_labels": all_labels[start:end],
        })

    result = DatasetDict({
        "train": _make_ds(0, n_train),
        "validation": _make_ds(n_train, n_train + n_val),
        "test": _make_ds(n_train + n_val, n),
    })

    n_entities = sum(lab.startswith("B-") for row in all_labels for lab in row)
    logger.info(
        "  BioMed NER: %d examples, %d DIAGNOSIS entities extracted",
        n, n_entities,
    )
    return result


# ---------------------------------------------------------------------------
# Source 4: ADE Corpus V2 — adverse drug effects as DIAGNOSIS
# ---------------------------------------------------------------------------

def _load_ade_effects(
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load adverse drug effect spans from ADE-Corpus-V2.

    Each example has a sentence with drug and adverse-effect char offsets.
    We convert the *effect* spans to DIAGNOSIS BIO tags (effects like
    "ototoxicity", "seizures", "hepatotoxicity" are ICD-mappable diagnoses).
    """
    raw = load_dataset(
        "ade_corpus_v2", "Ade_corpus_v2_drug_ade_relation",
        cache_dir=cache_dir,
    )

    all_tokens = []
    all_tags = []
    all_labels = []

    split_name = "train" if "train" in raw else list(raw.keys())[0]
    for example in raw[split_name]:
        text = example.get("text", "")
        effect = example.get("effect", "")
        indexes = example.get("indexes", {})

        if not text or not effect:
            continue

        # Build entity list from effect offsets
        effect_idx = indexes.get("effect", {})
        starts = effect_idx.get("start_char", [])
        ends = effect_idx.get("end_char", [])

        entities = [
            {"class": "DISORDER", "start": s, "end": e}
            for s, e in zip(starts, ends)
        ]
        if not entities:
            continue

        tokens, labels = _spans_to_diagnosis_bio(
            text, entities, frozenset({"DISORDER"}),
        )
        all_tokens.append(tokens)
        all_labels.append(labels)
        all_tags.append([ICD_NER_LABEL2ID[lab] for lab in labels])

    if not all_tokens:
        raise RuntimeError("No effect entities found in ADE Corpus V2")

    # Split 80/10/10
    n = len(all_tokens)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    def _make_ds(start, end):
        return Dataset.from_dict({
            "tokens": all_tokens[start:end],
            "ner_tags": all_tags[start:end],
            "ner_labels": all_labels[start:end],
        })

    result = DatasetDict({
        "train": _make_ds(0, n_train),
        "validation": _make_ds(n_train, n_train + n_val),
        "test": _make_ds(n_train + n_val, n),
    })

    n_entities = sum(lab.startswith("B-") for row in all_labels for lab in row)
    logger.info(
        "  ADE Corpus V2: %d examples, %d adverse-effect DIAGNOSIS entities",
        n, n_entities,
    )
    return result


# ---------------------------------------------------------------------------
# Source 5: Curated ICD clinical examples
# ---------------------------------------------------------------------------

# Hand-crafted clinical sentences with specific ICD-10-CM-mappable diagnoses.
# These target common FN patterns and fill vocabulary gaps from the public
# corpora (which are biased toward PubMed abstract language).
_CURATED_ICD_EXAMPLES: List[Dict] = [
    # Neurological — addresses FN: encephalopathy, seizures, coma, nystagmus
    {"tokens": ["Patient", "diagnosed", "with", "hepatic", "encephalopathy"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["History", "of", "recurrent", "seizures", "and", "epilepsy"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Presented", "in", "a", "coma", "secondary", "to", "diabetic", "ketoacidosis"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Examination", "revealed", "nystagmus", "and", "ataxia"],
     "labels": ["O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Wernicke", "encephalopathy", "was", "suspected"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["The", "patient", "has", "Parkinson", "disease"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["MRI", "showed", "multiple", "sclerosis", "lesions"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O"]},
    {"tokens": ["Diagnosed", "with", "amyotrophic", "lateral", "sclerosis"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Patient", "presents", "with", "migraine", "headaches"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Peripheral", "neuropathy", "secondary", "to", "diabetes"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS"]},

    # Cardiovascular — addresses FP fragments: heart, blood, pressure, diastolic
    {"tokens": ["Congestive", "heart", "failure", "with", "reduced", "ejection", "fraction"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]},
    {"tokens": ["Patient", "has", "essential", "hypertension", "and", "hyperlipidemia"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Acute", "myocardial", "infarction", "confirmed", "by", "troponin"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O"]},
    {"tokens": ["Atrial", "fibrillation", "with", "rapid", "ventricular", "response"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]},
    {"tokens": ["History", "of", "deep", "vein", "thrombosis", "and", "pulmonary", "embolism"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Stable", "angina", "pectoris", "managed", "with", "nitrates"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O"]},
    {"tokens": ["Aortic", "stenosis", "with", "mitral", "regurgitation"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Peripheral", "arterial", "disease", "of", "the", "lower", "extremities"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]},

    # Renal — addresses FP: renal, nephropathy
    {"tokens": ["Chronic", "kidney", "disease", "stage", "3"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Diabetic", "nephropathy", "with", "proteinuria"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Acute", "renal", "failure", "requiring", "dialysis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Nephrotic", "syndrome", "due", "to", "membranous", "nephropathy"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Urinary", "tract", "infection", "with", "pyelonephritis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},

    # Hepatic — addresses FN: liver
    {"tokens": ["Cirrhosis", "of", "the", "liver", "secondary", "to", "hepatitis", "C"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Nonalcoholic", "fatty", "liver", "disease", "with", "steatohepatitis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Acute", "liver", "failure", "due", "to", "acetaminophen", "overdose"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]},
    {"tokens": ["Hepatocellular", "carcinoma", "in", "setting", "of", "cirrhosis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "B-DIAGNOSIS"]},

    # Respiratory
    {"tokens": ["Chronic", "obstructive", "pulmonary", "disease", "with", "acute", "exacerbation"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Community", "acquired", "pneumonia", "in", "right", "lower", "lobe"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]},
    {"tokens": ["Pulmonary", "fibrosis", "with", "progressive", "dyspnea"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Acute", "respiratory", "distress", "syndrome"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Asthma", "exacerbation", "triggered", "by", "upper", "respiratory", "infection"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Pleural", "effusion", "secondary", "to", "lung", "cancer"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},

    # Endocrine / Metabolic
    {"tokens": ["Type", "2", "diabetes", "mellitus", "with", "peripheral", "neuropathy"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Hypothyroidism", "and", "adrenal", "insufficiency"],
     "labels": ["B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Diabetic", "ketoacidosis", "with", "hyperglycemia"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Syndrome", "of", "inappropriate", "antidiuretic", "hormone", "secretion"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Cushing", "syndrome", "due", "to", "pituitary", "adenoma"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Graves", "disease", "with", "thyrotoxicosis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},

    # Infectious
    {"tokens": ["Sepsis", "secondary", "to", "Staphylococcus", "aureus", "bacteremia"],
     "labels": ["B-DIAGNOSIS", "O", "O", "O", "O", "B-DIAGNOSIS"]},
    {"tokens": ["HIV", "infection", "with", "Pneumocystis", "jirovecii", "pneumonia"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Cellulitis", "of", "the", "left", "lower", "extremity"],
     "labels": ["B-DIAGNOSIS", "O", "O", "O", "O", "O"]},
    {"tokens": ["Clostridium", "difficile", "colitis", "after", "antibiotic", "therapy"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O"]},
    {"tokens": ["Active", "tuberculosis", "with", "cavitary", "lesions"],
     "labels": ["O", "B-DIAGNOSIS", "O", "O", "O"]},

    # Musculoskeletal / Rheumatologic
    {"tokens": ["Rheumatoid", "arthritis", "with", "joint", "erosions"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O"]},
    {"tokens": ["Systemic", "lupus", "erythematosus", "with", "lupus", "nephritis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Osteoporosis", "with", "pathologic", "fracture"],
     "labels": ["B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Gout", "with", "tophi", "and", "chronic", "gouty", "arthropathy"],
     "labels": ["B-DIAGNOSIS", "O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},

    # Oncology
    {"tokens": ["Non-small", "cell", "lung", "cancer", "stage", "IIIA"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Metastatic", "breast", "cancer", "with", "bone", "metastases"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Acute", "myeloid", "leukemia", "in", "remission"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Pancreatic", "adenocarcinoma", "with", "liver", "metastases"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Multiple", "myeloma", "with", "renal", "insufficiency"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},

    # Psychiatric / Behavioral
    {"tokens": ["Major", "depressive", "disorder", "with", "suicidal", "ideation"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Generalized", "anxiety", "disorder", "and", "panic", "attacks"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Bipolar", "disorder", "type", "I", "with", "manic", "episode"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Post-traumatic", "stress", "disorder", "with", "insomnia"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Schizophrenia", "with", "auditory", "hallucinations"],
     "labels": ["B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},

    # GI
    {"tokens": ["Crohn", "disease", "with", "small", "bowel", "obstruction"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Ulcerative", "colitis", "with", "toxic", "megacolon"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Gastroesophageal", "reflux", "disease", "with", "esophagitis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Acute", "pancreatitis", "secondary", "to", "gallstones"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Irritable", "bowel", "syndrome", "with", "diarrhea"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Peptic", "ulcer", "disease", "with", "upper", "GI", "bleeding"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},

    # Hematologic
    {"tokens": ["Iron", "deficiency", "anemia", "secondary", "to", "chronic", "blood", "loss"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Sickle", "cell", "disease", "with", "vaso-occlusive", "crisis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Disseminated", "intravascular", "coagulation"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Thrombocytopenia", "due", "to", "heparin-induced", "thrombocytopenia"],
     "labels": ["B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},

    # Dermatologic
    {"tokens": ["Psoriasis", "with", "psoriatic", "arthritis"],
     "labels": ["B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Stevens-Johnson", "syndrome", "secondary", "to", "allopurinol"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O"]},

    # Drug-related — addresses FN: cocaine, extended-release/lovastatin
    {"tokens": ["Cocaine", "abuse", "with", "cocaine-induced", "cardiomyopathy"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Opioid", "use", "disorder", "with", "withdrawal", "symptoms"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Alcohol", "dependence", "with", "alcoholic", "hepatitis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},

    # Misc clinical — common diagnoses
    {"tokens": ["Obesity", "with", "body", "mass", "index", "of", "42"],
     "labels": ["B-DIAGNOSIS", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Obstructive", "sleep", "apnea", "requiring", "CPAP"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Chronic", "pain", "syndrome", "with", "fibromyalgia"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Benign", "prostatic", "hyperplasia", "with", "urinary", "retention"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Glaucoma", "and", "age-related", "macular", "degeneration"],
     "labels": ["B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Anaphylaxis", "due", "to", "penicillin", "allergy"],
     "labels": ["B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},

    # Negative examples — teach model what is NOT a diagnosis
    {"tokens": ["Blood", "pressure", "was", "130", "/", "85", "mmHg"],
     "labels": ["O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Heart", "rate", "regular", "at", "72", "beats", "per", "minute"],
     "labels": ["O", "O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Renal", "function", "tests", "were", "within", "normal", "limits"],
     "labels": ["O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Liver", "enzymes", "are", "mildly", "elevated"],
     "labels": ["O", "O", "O", "O", "O"]},
    {"tokens": ["The", "nervous", "system", "examination", "was", "unremarkable"],
     "labels": ["O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Myocardial", "perfusion", "imaging", "showed", "no", "ischemia"],
     "labels": ["O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Adverse", "effects", "of", "the", "medication", "were", "monitored"],
     "labels": ["O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Patient", "denies", "chest", "pain", "and", "shortness", "of", "breath"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
]


def _generate_template_examples() -> List[Dict]:
    """
    Programmatically generate clinical NER training examples using templates.

    Targets documented NER failure patterns:
    - Abbreviation-heavy text (SOB, CP, HTN, DM2)
    - Multi-word entity boundary errors (acute vs chronic modifiers)
    - Lab values confused with diagnoses (negative examples)
    - Negation context entities
    - Rare disease mentions
    - High-frequency ICD codes (per HCUP/CMS billing data)

    Returns ~200 additional training examples beyond the 80+ hand-crafted ones.
    """
    examples: List[Dict] = []

    # --- Abbreviation contexts ---
    abbrev = [
        (["Pt", "c/o", "SOB", "and", "CP", "on", "exertion"],
         ["O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "O", "O"]),
        (["Hx", "of", "HTN", ",", "DM2", ",", "and", "CKD"],
         ["O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS"]),
        (["PMH", "significant", "for", "COPD", "and", "CHF"],
         ["O", "O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Dx", ":", "AFib", "with", "RVR"],
         ["O", "O", "B-DIAGNOSIS", "O", "O"]),
        (["R/O", "PE", "vs", "MI"],
         ["O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Pt", "with", "ESRD", "on", "HD"],
         ["O", "O", "B-DIAGNOSIS", "O", "O"]),
        (["Assessment", ":", "AMS", "likely", "secondary", "to", "UTI"],
         ["O", "O", "B-DIAGNOSIS", "O", "O", "O", "B-DIAGNOSIS"]),
        (["OSA", "on", "CPAP", ",", "GERD", "on", "PPI"],
         ["B-DIAGNOSIS", "O", "O", "O", "B-DIAGNOSIS", "O", "O"]),
        (["BPH", "with", "LUTS"],
         ["B-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Pt", "with", "h/o", "CVA", "and", "TIA"],
         ["O", "O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["DVT", "of", "left", "LE", ",", "s/p", "PE"],
         ["B-DIAGNOSIS", "O", "O", "O", "O", "O", "B-DIAGNOSIS"]),
        (["Known", "CAD", "s/p", "CABG", "x3"],
         ["O", "B-DIAGNOSIS", "O", "O", "O"]),
        (["RA", "on", "MTX", ",", "OA", "of", "bilateral", "knees"],
         ["B-DIAGNOSIS", "O", "O", "O", "B-DIAGNOSIS", "O", "O", "O"]),
        (["IBS-D", "and", "NAFLD"],
         ["B-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in abbrev)

    # --- Multi-word boundary patterns ---
    boundary = [
        (["Acute", "on", "chronic", "systolic", "heart", "failure"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Left", "sided", "hemiparesis", "due", "to", "right", "MCA", "stroke"],
         ["O", "O", "B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Stage", "IV", "chronic", "kidney", "disease"],
         ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Uncontrolled", "type", "2", "diabetes", "with", "neuropathy"],
         ["O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["New", "onset", "atrial", "fibrillation"],
         ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["End", "stage", "renal", "disease", "on", "hemodialysis"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]),
        (["Decompensated", "alcoholic", "cirrhosis", "with", "ascites"],
         ["O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Severe", "persistent", "asthma", "with", "acute", "exacerbation"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Advanced", "hepatocellular", "carcinoma"],
         ["O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Chronic", "low", "back", "pain", "with", "radiculopathy"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in boundary)

    # --- Lab value confusion (negative examples — NOT diagnoses) ---
    lab_negatives = [
        (["Hemoglobin", "7.2", "g/dL", ",", "hematocrit", "22%"],
         ["O", "O", "O", "O", "O", "O"]),
        (["Creatinine", "3.5", "mg/dL", ",", "BUN", "45"],
         ["O", "O", "O", "O", "O", "O"]),
        (["WBC", "15,000", ",", "bands", "12%"],
         ["O", "O", "O", "O", "O"]),
        (["Troponin", "I", "elevated", "at", "2.4", "ng/mL"],
         ["O", "O", "O", "O", "O", "O"]),
        (["A1c", "9.2%", ",", "fasting", "glucose", "210"],
         ["O", "O", "O", "O", "O", "O"]),
        (["BNP", "1200", "pg/mL"],
         ["O", "O", "O"]),
        (["Platelet", "count", "45,000"],
         ["O", "O", "O"]),
        (["INR", "3.8", ",", "PT", "42", "seconds"],
         ["O", "O", "O", "O", "O", "O"]),
        (["Oxygen", "saturation", "88%", "on", "room", "air"],
         ["O", "O", "O", "O", "O", "O"]),
        (["Temperature", "38.9", "C", ",", "heart", "rate", "110"],
         ["O", "O", "O", "O", "O", "O", "O"]),
        (["Sodium", "128", ",", "potassium", "5.8"],
         ["O", "O", "O", "O", "O"]),
        (["Lactic", "acid", "4.2", "mmol/L"],
         ["O", "O", "O", "O"]),
        (["CT", "head", "without", "acute", "intracranial", "findings"],
         ["O", "O", "O", "O", "O", "O"]),
        (["EKG", "showing", "sinus", "tachycardia"],
         ["O", "O", "O", "O"]),
        (["GFR", "estimated", "at", "25", "mL/min"],
         ["O", "O", "O", "O", "O"]),
        (["Procalcitonin", "0.8", "ng/mL"],
         ["O", "O", "O"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in lab_negatives)

    # --- Negation context (entities in negated/qualified settings) ---
    negation = [
        (["No", "evidence", "of", "pneumonia", "on", "chest", "X-ray"],
         ["O", "O", "O", "B-DIAGNOSIS", "O", "O", "O"]),
        (["Denies", "any", "chest", "pain", "or", "palpitations"],
         ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Rules", "out", "pulmonary", "embolism"],
         ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Patient", "without", "signs", "of", "heart", "failure"],
         ["O", "O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Negative", "for", "deep", "vein", "thrombosis"],
         ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["No", "recurrence", "of", "breast", "cancer"],
         ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Family", "history", "of", "colon", "cancer", "and", "diabetes"],
         ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Possible", "early", "Alzheimer", "disease"],
         ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in negation)

    # --- Rare and complex diagnoses ---
    rare = [
        (["Diagnosed", "with", "Takayasu", "arteritis"],
         ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Polyarteritis", "nodosa", "with", "renal", "involvement"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O"]),
        (["Hemophagocytic", "lymphohistiocytosis"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Primary", "biliary", "cholangitis"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["IgA", "nephropathy", "with", "hematuria"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Myasthenia", "gravis", "with", "thymoma"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Guillain-Barre", "syndrome", "post", "infection"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]),
        (["Pheochromocytoma", "presenting", "with", "hypertensive", "crisis"],
         ["B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Marfan", "syndrome", "with", "aortic", "root", "dilation"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Anti-NMDA", "receptor", "encephalitis"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Amyloidosis", "with", "cardiac", "involvement"],
         ["B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in rare)

    # --- High-frequency ICD codes (top billing diagnoses per HCUP/CMS) ---
    high_freq = [
        (["Severe", "sepsis", "with", "septic", "shock"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Acute", "decompensated", "heart", "failure"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Heart", "failure", "with", "preserved", "ejection", "fraction"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]),
        (["Aspiration", "pneumonia"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Hospital", "acquired", "pneumonia"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Type", "1", "diabetes", "with", "diabetic", "ketoacidosis"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Acute", "ischemic", "stroke", "of", "left", "MCA", "territory"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]),
        (["Hemorrhagic", "stroke", "with", "intraventricular", "hemorrhage"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Transient", "ischemic", "attack"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Acute", "kidney", "injury", "stage", "3"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]),
        (["Upper", "gastrointestinal", "hemorrhage"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]),
        (["Essential", "hypertension", "uncontrolled"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS", "O"]),
        (["Mixed", "hyperlipidemia"],
         ["B-DIAGNOSIS", "I-DIAGNOSIS"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in high_freq)

    # --- Procedure/medication contexts (negative examples) ---
    proc_negatives = [
        (["Status", "post", "total", "knee", "replacement"],
         ["O", "O", "O", "O", "O"]),
        (["Currently", "on", "metformin", "500mg", "twice", "daily"],
         ["O", "O", "O", "O", "O", "O"]),
        (["Started", "on", "lisinopril", "10mg", "for", "hypertension"],
         ["O", "O", "O", "O", "O", "B-DIAGNOSIS"]),
        (["Received", "2", "units", "packed", "red", "blood", "cells"],
         ["O", "O", "O", "O", "O", "O", "O"]),
        (["Physical", "therapy", "consult", "placed"],
         ["O", "O", "O", "O"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in proc_negatives)

    # --- Multi-diagnosis assessment sections ---
    multi_dx = [
        (["Assessment", ":", "1.", "Sepsis", "2.", "Acute", "kidney", "injury", "3.", "Anemia"],
         ["O", "O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Problems", ":", "Hypertension", ",", "Type", "2", "diabetes", ",", "Hyperlipidemia"],
         ["O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]),
        (["Active", "issues", ":", "pneumonia", ",", "COPD", "exacerbation", ",", "heart", "failure"],
         ["O", "O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]),
    ]
    examples.extend({"tokens": t, "labels": l} for t, l in multi_dx)

    return examples


def _get_curated_icd_examples() -> DatasetDict:
    """
    Build a DatasetDict from hand-crafted + template-generated clinical sentences.

    Combines 80+ hand-crafted examples with ~100 template-generated sentences
    targeting common NER failure patterns (abbreviations, multi-word boundaries,
    lab value confusion, negation contexts, rare diseases, high-frequency ICD codes).
    """
    # Combine hand-crafted and template-generated
    template_examples = _generate_template_examples()
    combined = _CURATED_ICD_EXAMPLES + template_examples

    all_tokens = [ex["tokens"] for ex in combined]
    all_labels = [ex["labels"] for ex in combined]
    all_tags = [
        [ICD_NER_LABEL2ID[lab] for lab in labs]
        for labs in all_labels
    ]

    # Duplicate 3x to increase weight in the composite dataset
    n_repeats = 3
    ds = Dataset.from_dict({
        "tokens": all_tokens * n_repeats,
        "ner_tags": all_tags * n_repeats,
        "ner_labels": all_labels * n_repeats,
    })

    n_hand = len(_CURATED_ICD_EXAMPLES)
    n_template = len(template_examples)
    n_entities = sum(lab.startswith("B-") for row in all_labels for lab in row)
    logger.info(
        "  Curated ICD: %d hand-crafted + %d template-generated = %d examples "
        "(%dx repeat), %d unique DIAGNOSIS entities",
        n_hand, n_template, len(combined), n_repeats, n_entities,
    )

    return DatasetDict({"train": ds})


# ---------------------------------------------------------------------------
# Source 6: MedMentions — Disease/Disorder semantic types from UMLS
# ---------------------------------------------------------------------------

# UMLS Semantic Type T038 is used for disease entities in ibm/MedMentions-ZS
# (zero-shot variant). Original MedMentions uses finer-grained types:
_MEDMENTIONS_DISEASE_TYPES = frozenset({
    "T047", "T048", "T019", "T046", "T191", "T020", "T190", "T049",
})
# In the ZS variant, these are grouped under T038.
_MEDMENTIONS_ZS_DISEASE_TAG = "T038"


def _load_medmentions_diseases(
    cache_dir: Optional[str] = None,
    max_examples: int = 5000,
) -> DatasetDict:
    """
    Load Disease/Disorder entities from MedMentions.

    Primary strategy: ``ibm/MedMentions-ZS`` — a parquet-based zero-shot
    variant with 29K pre-tokenized PubMed abstracts in BIO format. Disease
    entities are tagged as ``B-T038`` / ``I-T038`` (UMLS coarse type) and
    mapped to DIAGNOSIS. Loads via standard ``load_dataset()`` with no
    custom scripts.

    Fallback: ``bigbio/medmentions`` — downloads Parquet files from the
    ``refs/convert/parquet`` branch and processes character-offset entity
    annotations. Used only if the IBM dataset is unavailable.
    """
    try:
        return _load_medmentions_zs(cache_dir, max_examples)
    except Exception as e:
        logger.warning(
            "  ibm/MedMentions-ZS failed (%s), trying bigbio/medmentions parquet fallback...", e,
        )
        return _load_medmentions_parquet_fallback(cache_dir, max_examples)


def _load_medmentions_zs(
    cache_dir: Optional[str] = None,
    max_examples: int = 5000,
) -> DatasetDict:
    """
    Load disease entities from ibm/MedMentions-ZS (zero-shot variant).

    This dataset provides 29K pre-tokenized PubMed abstracts with BIO-format
    UMLS semantic type tags. Disease entities use the T038 coarse type.
    No custom loading scripts needed — standard parquet format.
    """
    raw = load_dataset("ibm/MedMentions-ZS", cache_dir=cache_dir)

    disease_b = f"B-{_MEDMENTIONS_ZS_DISEASE_TAG}"
    disease_i = f"I-{_MEDMENTIONS_ZS_DISEASE_TAG}"

    def _map_to_diagnosis(example):
        labels = []
        for tag in example["ner_tags"]:
            if tag == disease_b:
                labels.append("B-DIAGNOSIS")
            elif tag == disease_i:
                labels.append("I-DIAGNOSIS")
            else:
                labels.append("O")
        return {
            "tokens": example["tokens"],
            "ner_tags": [ICD_NER_LABEL2ID[lab] for lab in labels],
            "ner_labels": labels,
        }

    result = {}
    total_examples = 0
    total_entities = 0
    for split_name in ["train", "validation", "test"]:
        if split_name not in raw:
            continue
        split_ds = raw[split_name]
        # Cap per-split examples
        if len(split_ds) > max_examples:
            split_ds = split_ds.select(range(max_examples))
        # Filter to keep only examples with at least one disease entity
        split_ds = split_ds.filter(
            lambda ex: any(t == disease_b for t in ex["ner_tags"]),
        )
        # Skip empty splits (e.g. validation/test may have no disease entities)
        if len(split_ds) == 0:
            continue
        # Map tags to DIAGNOSIS
        cols_to_remove = [c for c in split_ds.column_names if c not in {"tokens", "ner_tags", "ner_labels"}]
        split_ds = split_ds.map(_map_to_diagnosis, remove_columns=cols_to_remove)
        result[split_name] = split_ds
        total_examples += len(split_ds)
        total_entities += sum(
            1 for ex in split_ds for lab in ex["ner_labels"] if lab.startswith("B-")
        )

    if not result:
        raise RuntimeError("No disease entities found in ibm/MedMentions-ZS")

    dataset = DatasetDict(result)
    logger.info(
        "  MedMentions-ZS: %d examples, %d DIAGNOSIS entities", total_examples, total_entities,
    )
    return dataset


def _load_medmentions_parquet_fallback(
    cache_dir: Optional[str] = None,
    max_examples: int = 5000,
) -> DatasetDict:
    """
    Fallback: load MedMentions from bigbio/medmentions parquet files.

    Downloads pre-converted Parquet files from the ``refs/convert/parquet``
    branch and converts character-offset entity annotations to BIO tags.
    Used when ibm/MedMentions-ZS is unavailable.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to load MedMentions. "
            "Install it with: pip install huggingface_hub"
        )

    import pyarrow.parquet as pq

    raw: Dict[str, list] = {}
    for split_name in ["train", "validation", "test"]:
        filename = f"medmentions_st21pv_source/{split_name}/0000.parquet"
        try:
            path = hf_hub_download(
                repo_id="bigbio/medmentions",
                filename=filename,
                repo_type="dataset",
                revision="refs/convert/parquet",
                cache_dir=cache_dir,
            )
            table = pq.read_table(path)
            rows = table.to_pydict()
            n_rows = len(rows[list(rows.keys())[0]])
            raw[split_name] = [{k: rows[k][i] for k in rows} for i in range(n_rows)]
            logger.info("  MedMentions %s: loaded %d documents from parquet", split_name, len(raw[split_name]))
        except Exception as e:
            logger.warning("  Could not load MedMentions %s parquet: %s", split_name, e)

    if not raw:
        raise RuntimeError("Could not load any MedMentions splits from parquet")

    all_tokens: List[List[str]] = []
    all_tags: List[List[int]] = []
    all_labels: List[List[str]] = []

    for split_name in ["train", "validation", "test"]:
        if split_name not in raw:
            continue
        count = 0
        for example in raw[split_name]:
            if count >= max_examples:
                break
            passages = example.get("passages", [])
            entities = example.get("entities", [])
            if not passages or not entities:
                continue
            text_parts = []
            for p in passages:
                t = p.get("text", "")
                if isinstance(t, list):
                    t = t[0] if t else ""
                text_parts.append(t)
            text = " ".join(text_parts)
            if not text.strip():
                continue
            disease_entities = []
            for ent in entities:
                sem_types = ent.get("semantic_type_id", [])
                is_disease = (
                    any(st in _MEDMENTIONS_DISEASE_TYPES for st in sem_types)
                    if isinstance(sem_types, list)
                    else sem_types in _MEDMENTIONS_DISEASE_TYPES
                )
                if is_disease:
                    for offset_pair in ent.get("offsets", []):
                        disease_entities.append({"class": "DISORDER", "start": offset_pair[0], "end": offset_pair[1]})
            if not disease_entities:
                continue
            tokens, labels = _spans_to_diagnosis_bio(text, disease_entities, frozenset({"DISORDER"}))
            if any(l.startswith("B-") for l in labels):
                all_tokens.append(tokens)
                all_labels.append(labels)
                all_tags.append([ICD_NER_LABEL2ID[lab] for lab in labels])
                count += 1

    if not all_tokens:
        raise RuntimeError("No disease entities found in MedMentions parquet fallback")

    n = len(all_tokens)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    def _make_ds(start, end):
        return Dataset.from_dict({"tokens": all_tokens[start:end], "ner_tags": all_tags[start:end], "ner_labels": all_labels[start:end]})

    result = DatasetDict({
        "train": _make_ds(0, n_train),
        "validation": _make_ds(n_train, n_train + n_val),
        "test": _make_ds(n_train + n_val, n),
    })
    n_entities = sum(lab.startswith("B-") for row in all_labels for lab in row)
    logger.info("  MedMentions (parquet fallback): %d examples, %d DIAGNOSIS entities", n, n_entities)
    return result


# ---------------------------------------------------------------------------
# Source 7: MACCROBAT — Disease entities from clinical case reports
# ---------------------------------------------------------------------------

_MACCROBAT_DISEASE_LABELS = frozenset({
    "DISEASE_DISORDER", "DISEASE", "DISORDER", "SIGN_SYMPTOM",
})


def _load_maccrobat_diseases(
    cache_dir: Optional[str] = None,
    max_examples: int = 3000,
) -> DatasetDict:
    """
    Load DISEASE_DISORDER entities from MACCROBAT clinical case reports.

    MACCROBAT provides 200 clinical case reports with clinical-note-style
    text, directly addressing the domain gap from PubMed abstracts.

    Loading strategy:
    The singh-aditya/MACCROBAT_biomedical_ner HuggingFace dataset uses a
    loading script that is no longer supported by the ``datasets`` library
    (>=4.x). We download the JSON data file directly from the repository
    and parse it ourselves, bypassing the deprecated loading script.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to load MACCROBAT. "
            "Install it with: pip install huggingface_hub"
        )

    import json

    json_path = hf_hub_download(
        repo_id="singh-aditya/MACCROBAT_biomedical_ner",
        filename="MACCROBAT2020-V2.json",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    with open(json_path) as f:
        data = json.load(f)

    label_names = data.get("all_ner_labels", [])
    documents = data.get("data", [])
    if not label_names or not documents:
        raise RuntimeError("MACCROBAT JSON has unexpected structure")

    all_tokens: List[List[str]] = []
    all_tags: List[List[int]] = []
    all_labels: List[List[str]] = []

    for i, doc in enumerate(documents):
        if i >= max_examples:
            break

        tokens = doc.get("tokens", [])
        ner_labels_raw = doc.get("ner_labels", [])
        if not tokens or not ner_labels_raw:
            continue

        # Map original labels to DIAGNOSIS
        mapped_labels = []
        for label in ner_labels_raw:
            label_upper = str(label).upper().replace("-", "_")

            prefix = ""
            entity_type = label_upper
            if label_upper.startswith("B_"):
                prefix = "B-"
                entity_type = label_upper[2:]
            elif label_upper.startswith("I_"):
                prefix = "I-"
                entity_type = label_upper[2:]

            if entity_type in _MACCROBAT_DISEASE_LABELS or any(
                w in entity_type for w in ["DISEASE", "DISORDER", "SIGN_SYMPTOM"]
            ):
                mapped_labels.append(f"{prefix}DIAGNOSIS" if prefix else "O")
            else:
                mapped_labels.append("O")

        # Fix orphaned I-DIAGNOSIS tags
        for j in range(len(mapped_labels)):
            if mapped_labels[j] == "I-DIAGNOSIS":
                if j == 0 or mapped_labels[j - 1] == "O":
                    mapped_labels[j] = "B-DIAGNOSIS"

        # Ensure tokens and labels are the same length
        min_len = min(len(tokens), len(mapped_labels))
        tokens = [str(t) for t in tokens[:min_len]]
        mapped_labels = mapped_labels[:min_len]

        if any(l.startswith("B-") for l in mapped_labels):
            all_tokens.append(tokens)
            all_labels.append(mapped_labels)
            all_tags.append([ICD_NER_LABEL2ID[lab] for lab in mapped_labels])

    if not all_tokens:
        raise RuntimeError("No disease entities found in MACCROBAT")

    n = len(all_tokens)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    def _make_ds(start, end):
        return Dataset.from_dict({
            "tokens": all_tokens[start:end],
            "ner_tags": all_tags[start:end],
            "ner_labels": all_labels[start:end],
        })

    result = DatasetDict({
        "train": _make_ds(0, n_train),
        "validation": _make_ds(n_train, n_train + n_val),
        "test": _make_ds(n_train + n_val, n),
    })

    n_entities = sum(lab.startswith("B-") for row in all_labels for lab in row)
    logger.info("  MACCROBAT: %d examples, %d DIAGNOSIS entities", n, n_entities)
    return result


# ---------------------------------------------------------------------------
# Source 8: Curated discharge summary examples — DRG-relevant diagnoses
# ---------------------------------------------------------------------------

_DISCHARGE_SUMMARY_EXAMPLES: List[Dict] = [
    # --- CC/MCC diagnoses that shift DRG severity tiers ---
    # These target high-value diagnoses frequently under-captured in NER
    {"tokens": ["Discharge", "diagnosis", ":", "acute", "respiratory", "failure", "with", "hypoxia"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Principal", "diagnosis", ":", "severe", "sepsis", "due", "to", "urinary", "tract", "infection"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Patient", "admitted", "for", "acute", "ST-elevation", "myocardial", "infarction"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Secondary", "diagnoses", ":", "acute", "kidney", "injury", ",", "hyperkalemia", ",", "metabolic", "acidosis"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Complicated", "by", "hospital-acquired", "pneumonia", "and", "Clostridium", "difficile", "infection"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Discharge", "summary", ":", "encephalopathy", "secondary", "to", "hepatic", "failure"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Primary", ":", "decompensated", "heart", "failure", "with", "cardiogenic", "shock"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Acute", "exacerbation", "of", "chronic", "obstructive", "pulmonary", "disease", "requiring", "intubation"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Diagnoses", "at", "discharge", ":", "diabetic", "ketoacidosis", ",", "type", "1", "diabetes", "mellitus"],
     "labels": ["O", "O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Final", "diagnosis", ":", "pulmonary", "embolism", "with", "right", "heart", "strain"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},

    # --- Inpatient co-morbidities (CC/MCC impact on DRG payment) ---
    {"tokens": ["Comorbidities", ":", "morbid", "obesity", ",", "obstructive", "sleep", "apnea", ",", "hypertension"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Additional", ":", "protein-calorie", "malnutrition", ",", "pressure", "ulcer", "stage", "3"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Other", ":", "chronic", "systolic", "heart", "failure", ",", "atrial", "fibrillation", ",", "CKD", "stage", "4"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "O", "O"]},
    {"tokens": ["Coagulopathy", "secondary", "to", "warfarin", "use", "with", "INR", "5.2"],
     "labels": ["B-DIAGNOSIS", "O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Altered", "mental", "status", "likely", "secondary", "to", "uremic", "encephalopathy"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Ventilator-associated", "pneumonia", "with", "acute", "respiratory", "distress", "syndrome"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Postoperative", "wound", "infection", "with", "dehiscence"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Catheter-associated", "urinary", "tract", "infection", "with", "urosepsis"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS"]},

    # --- Historical and family contexts in discharge summaries ---
    {"tokens": ["Past", "medical", "history", ":", "coronary", "artery", "disease", ",", "prior", "CABG", ",", "diabetes"],
     "labels": ["O", "O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O", "B-DIAGNOSIS"]},
    {"tokens": ["History", "significant", "for", "chronic", "hepatitis", "C", "and", "liver", "cirrhosis"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["Family", "history", "of", "premature", "coronary", "artery", "disease"],
     "labels": ["O", "O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},
    {"tokens": ["PMH", ":", "HTN", ",", "DM2", ",", "CKD", "3", ",", "HFrEF", ",", "AFib"],
     "labels": ["O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]},

    # --- Procedure-related diagnoses (affect surgical DRGs) ---
    {"tokens": ["Admitted", "for", "acute", "cholecystitis", ",", "underwent", "laparoscopic", "cholecystectomy"],
     "labels": ["O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O"]},
    {"tokens": ["Small", "bowel", "obstruction", "requiring", "surgical", "lysis", "of", "adhesions"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O", "O"]},
    {"tokens": ["Hip", "fracture", "status", "post", "open", "reduction", "internal", "fixation"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Spinal", "stenosis", "with", "neurogenic", "claudication", "treated", "with", "laminectomy"],
     "labels": ["B-DIAGNOSIS", "I-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "O", "O", "O"]},

    # --- Multi-diagnosis discharge summaries ---
    {"tokens": ["Discharge", "diagnoses", ":", "1.", "Pneumonia", "2.", "Acute", "kidney", "injury",
                 "3.", "Hypernatremia", "4.", "Delirium"],
     "labels": ["O", "O", "O", "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS",
                 "O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]},
    {"tokens": ["Problems", "addressed", ":", "congestive", "heart", "failure", "exacerbation", ",",
                 "acute", "on", "chronic", "kidney", "disease", ",", "anemia", "of", "chronic", "disease"],
     "labels": ["O", "O", "O", "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O",
                 "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "O",
                 "B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]},

    # --- Negative examples from discharge summaries ---
    {"tokens": ["Medications", "at", "discharge", ":", "metoprolol", "25mg", "BID", ",", "lisinopril", "10mg"],
     "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Follow", "up", "with", "PCP", "in", "1", "week", "and", "cardiology", "in", "2", "weeks"],
     "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Condition", "at", "discharge", ":", "stable", ",", "improved"],
     "labels": ["O", "O", "O", "O", "O", "O", "O"]},
    {"tokens": ["Vitals", "on", "discharge", ":", "BP", "128/76", ",", "HR", "72", ",", "O2", "sat", "96%"],
     "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]},
]


def _get_discharge_summary_examples() -> DatasetDict:
    """
    Build a DatasetDict from curated discharge summary examples.

    These target DRG-relevant diagnoses: principal diagnoses, CC/MCC
    comorbidities, hospital-acquired conditions, and the specific
    formatting patterns found in discharge summaries (numbered lists,
    abbreviations, historical contexts).

    This source fills the domain gap between PubMed abstracts (academic
    language) and real clinical discharge notes (terse, list-heavy,
    abbreviation-dense) — the primary input for DRG assignment.
    """
    all_tokens = [ex["tokens"] for ex in _DISCHARGE_SUMMARY_EXAMPLES]
    all_labels = [ex["labels"] for ex in _DISCHARGE_SUMMARY_EXAMPLES]
    all_tags = [
        [ICD_NER_LABEL2ID[lab] for lab in labs]
        for labs in all_labels
    ]

    # Duplicate 3x to increase weight (matches curated ICD examples strategy)
    n_repeats = 3
    ds = Dataset.from_dict({
        "tokens": all_tokens * n_repeats,
        "ner_tags": all_tags * n_repeats,
        "ner_labels": all_labels * n_repeats,
    })

    n_examples = len(_DISCHARGE_SUMMARY_EXAMPLES)
    n_entities = sum(lab.startswith("B-") for row in all_labels for lab in row)
    logger.info(
        "  Discharge summaries: %d examples (%dx repeat), %d unique DIAGNOSIS entities",
        n_examples, n_repeats, n_entities,
    )

    return DatasetDict({"train": ds})
