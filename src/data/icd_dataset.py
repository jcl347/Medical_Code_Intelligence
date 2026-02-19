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
- **Curated ICD Examples**: Hand-crafted clinical sentences covering
  the top 100+ ICD-10-CM diagnoses for targeted training signal.

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

    Merges five sources with unified DIAGNOSIS labels:
    1. NCBI Disease Corpus (PubMed abstracts)
    2. BC5CDR disease subset (PubMed articles)
    3. BioMed NER DISORDER entities (clinical case reports)
    4. ADE Corpus V2 adverse effect entities
    5. Curated clinical ICD examples

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

    # --- Source 5: Curated ICD clinical examples ---
    logger.info("  [5/5] Adding curated ICD clinical examples...")
    curated = _get_curated_icd_examples()
    all_sources.append(("curated_icd", curated))

    # --- Clean garbage labels from all sources ---
    logger.info("  Cleaning garbage labels from all sources...")
    cleaned_sources = []
    for name, ds in all_sources:
        ds = _clean_garbage_labels(ds)
        cleaned_sources.append((name, ds))

    # --- Cast to common schema and merge ---
    for name, ds in cleaned_sources:
        for split in list(ds.keys()):
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


def _get_curated_icd_examples() -> DatasetDict:
    """
    Build a DatasetDict from hand-crafted clinical sentences.

    These examples cover the most common ICD-10-CM diagnoses and provide
    targeted training signal for clinical vocabulary that public corpora
    under-represent (e.g. "encephalopathy", "seizures", multi-word
    diagnosis phrases like "congestive heart failure").
    """
    all_tokens = [ex["tokens"] for ex in _CURATED_ICD_EXAMPLES]
    all_labels = [ex["labels"] for ex in _CURATED_ICD_EXAMPLES]
    all_tags = [
        [ICD_NER_LABEL2ID[lab] for lab in labs]
        for labs in all_labels
    ]

    # Use all for training (these are curated, not evaluation data)
    # Duplicate 3x to increase weight in the composite dataset
    n_repeats = 3
    ds = Dataset.from_dict({
        "tokens": all_tokens * n_repeats,
        "ner_tags": all_tags * n_repeats,
        "ner_labels": all_labels * n_repeats,
    })

    n_entities = sum(lab.startswith("B-") for row in all_labels for lab in row)
    logger.info(
        "  Curated ICD: %d examples (%dx repeat), %d unique DIAGNOSIS entities",
        len(all_tokens), n_repeats, n_entities,
    )

    return DatasetDict({"train": ds})
