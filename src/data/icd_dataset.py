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

All entity labels are normalized to a two-label BIO scheme:
    O, B-DIAGNOSIS, I-DIAGNOSIS

This simplification is intentional: the model's job is to detect *any*
diagnosable condition; the downstream ICD-10-CM code resolution step
(via ``ICDCodeLookup``) maps the extracted text span to a specific code.
"""

import logging
from typing import List, Optional, Tuple

from datasets import DatasetDict, Features, Sequence, Value, concatenate_datasets, load_dataset

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

    Merges NCBI Disease + BC5CDR (disease subset) with unified DIAGNOSIS
    labels across train / validation / test splits.

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

    # --- NCBI Disease: Disease → DIAGNOSIS ---
    logger.info("  Loading NCBI Disease corpus...")
    ncbi = load_dataset(
        "ncbi/ncbi_disease",
        cache_dir=cache_dir,
        revision="refs/convert/parquet",
    )
    ncbi = _normalize_ncbi_to_diagnosis(ncbi)

    # --- BC5CDR: Disease → DIAGNOSIS, Chemical → O ---
    logger.info("  Loading BC5CDR corpus (disease entities only)...")
    bc5cdr = load_dataset(
        "tner/bc5cdr",
        cache_dir=cache_dir,
        revision="refs/convert/parquet",
    )
    bc5cdr = _normalize_bc5cdr_to_diagnosis(bc5cdr)

    # --- Clean garbage standalone B-DIAGNOSIS on function words ---
    logger.info("  Cleaning garbage labels from merged corpus...")
    ncbi = _clean_garbage_labels(ncbi)
    bc5cdr = _clean_garbage_labels(bc5cdr)

    # --- Cast to common schema (strips ClassLabel metadata) and merge ---
    for ds in (ncbi, bc5cdr):
        for split in list(ds.keys()):
            ds[split] = ds[split].cast(_UNIFIED_FEATURES)

    merged = {}
    for split in ["train", "validation", "test"]:
        parts = []
        if split in ncbi:
            parts.append(ncbi[split])
        if split in bc5cdr:
            parts.append(bc5cdr[split])
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
