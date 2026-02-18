"""
Dataset loaders for public biomedical NER datasets.

Supports loading from HuggingFace Hub with automatic label mapping
and consistent interface across different dataset formats.

Includes span-to-BIO conversion for datasets that use character-offset
annotations (e.g. knowledgator/biomed_NER).
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from configs.ner_config import DATASET_CONFIGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Span-to-BIO conversion utilities
# ---------------------------------------------------------------------------

def spans_to_bio(
    text: str,
    entities: List[Dict],
    tokenizer_fn=None,
) -> Tuple[List[str], List[str]]:
    """
    Convert character-offset span annotations to BIO-tagged token sequences.

    Parameters
    ----------
    text : str
        Raw text string.
    entities : list of dict
        Each dict has 'start' (int), 'end' (int), 'class' (str) keys.
    tokenizer_fn : callable, optional
        Custom tokenizer. Defaults to whitespace + punctuation splitting.

    Returns
    -------
    tokens : list of str
        Word-level tokens.
    labels : list of str
        BIO labels aligned with tokens.
    """
    if tokenizer_fn is None:
        tokenizer_fn = _default_tokenize

    token_spans = tokenizer_fn(text)
    tokens = [text[start:end] for start, end in token_spans]
    labels = ["O"] * len(tokens)

    # Sort entities by start position
    sorted_ents = sorted(entities, key=lambda e: e.get("start", 0))

    for ent in sorted_ents:
        ent_start = ent.get("start", 0)
        ent_end = ent.get("end", 0)
        # Normalise entity class name: spaces/special chars → underscore
        ent_class = re.sub(r"[\s/]+", "_", ent.get("class", "ENTITY").strip().upper())

        first_token = True
        for i, (tok_start, tok_end) in enumerate(token_spans):
            # Token overlaps with entity span
            if tok_start >= ent_start and tok_end <= ent_end:
                if first_token:
                    labels[i] = f"B-{ent_class}"
                    first_token = False
                else:
                    labels[i] = f"I-{ent_class}"
            elif tok_start < ent_end and tok_end > ent_start:
                # Partial overlap — include if majority of token is inside entity
                overlap = min(tok_end, ent_end) - max(tok_start, ent_start)
                if overlap > (tok_end - tok_start) / 2:
                    if first_token:
                        labels[i] = f"B-{ent_class}"
                        first_token = False
                    else:
                        labels[i] = f"I-{ent_class}"

    return tokens, labels


def _default_tokenize(text: str) -> List[Tuple[int, int]]:
    """Simple whitespace + punctuation tokenizer that returns (start, end) spans."""
    spans = []
    for match in re.finditer(r"\S+", text):
        spans.append((match.start(), match.end()))
    return spans


# ---------------------------------------------------------------------------
# Dataset-specific normalisers
# ---------------------------------------------------------------------------

def _normalise_ncbi_disease(dataset: DatasetDict) -> DatasetDict:
    """Normalise NCBI Disease corpus to standard (tokens, ner_tags) format."""
    label_map = {0: "O", 1: "B-Disease", 2: "I-Disease"}

    def _map_fn(example):
        return {
            "tokens": example["tokens"],
            "ner_tags": example["ner_tags"],
            "ner_labels": [label_map.get(t, "O") for t in example["ner_tags"]],
        }

    return dataset.map(_map_fn)


def _normalise_bc5cdr(dataset: DatasetDict) -> DatasetDict:
    """Normalise BC5CDR (tner format) to standard format."""
    label_names = dataset["train"].features["tags"].feature.names

    def _map_fn(example):
        return {
            "tokens": example["tokens"],
            "ner_tags": example["tags"],
            "ner_labels": [label_names[t] for t in example["tags"]],
        }

    return dataset.map(_map_fn)


def _normalise_jnlpba(dataset: DatasetDict) -> DatasetDict:
    """Normalise JNLPBA to standard format."""
    label_names = dataset["train"].features["ner_tags"].feature.names

    def _map_fn(example):
        return {
            "tokens": example["tokens"],
            "ner_tags": example["ner_tags"],
            "ner_labels": [label_names[t] for t in example["ner_tags"]],
        }

    return dataset.map(_map_fn)


def _normalise_biomed_ner(dataset: DatasetDict) -> DatasetDict:
    """
    Normalise knowledgator/biomed_NER: span annotations → BIO tags.

    The dataset has columns: text (str), entities (list of {class, start, end}).
    We convert to the standard tokens / ner_labels format.
    """
    def _map_fn(example):
        tokens, labels = spans_to_bio(example["text"], example["entities"])
        label2id = {lab: idx for idx, lab in enumerate(sorted(set(labels), key=lambda x: (x != "O", x)))}
        return {
            "tokens": tokens,
            "ner_tags": [label2id.get(l, 0) for l in labels],
            "ner_labels": labels,
        }

    return dataset.map(_map_fn)


def _normalise_generic(dataset: DatasetDict, dataset_key: str) -> DatasetDict:
    """Generic normaliser that reads config to find column names."""
    cfg = DATASET_CONFIGS[dataset_key]
    token_col = cfg["token_column"]
    label_col = cfg["label_column"]

    features = dataset["train"].features
    label_feature = features[label_col].feature
    if hasattr(label_feature, "names"):
        label_names = label_feature.names
    else:
        label_names = None

    def _map_fn(example):
        tags = example[label_col]
        if label_names is not None:
            labels = [label_names[t] for t in tags]
        else:
            labels = [str(t) for t in tags]
        return {
            "tokens": example[token_col],
            "ner_tags": tags,
            "ner_labels": labels,
        }

    return dataset.map(_map_fn)


# Registry of dataset-specific normalisers
_NORMALISERS = {
    "ncbi_disease": _normalise_ncbi_disease,
    "bc5cdr": _normalise_bc5cdr,
    "jnlpba": _normalise_jnlpba,
    "biomed_ner": _normalise_biomed_ner,
}


def load_ner_dataset(
    dataset_key: str,
    cache_dir: Optional[str] = None,
) -> Tuple[DatasetDict, List[str]]:
    """
    Load and normalise a biomedical NER dataset.

    Returns
    -------
    dataset : DatasetDict
        With splits 'train', 'validation' (or 'test') containing at minimum
        columns: tokens (List[str]), ner_tags (List[int]), ner_labels (List[str]).
    label_list : List[str]
        Ordered list of string labels (e.g. ['O', 'B-Disease', 'I-Disease']).
    """
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    # Composite datasets have their own loaders
    if DATASET_CONFIGS[dataset_key].get("format") == "composite":
        if dataset_key == "icd_ner":
            from src.data.icd_dataset import load_icd_ner_dataset
            return load_icd_ner_dataset(cache_dir=cache_dir)
        raise ValueError(f"No loader for composite dataset '{dataset_key}'.")

    cfg = DATASET_CONFIGS[dataset_key]

    # Non-NER datasets (code lookups, instruction data) can't be loaded as NER
    fmt = cfg.get("format")
    if fmt in ("code_lookup", "instruction"):
        raise ValueError(
            f"Dataset '{dataset_key}' (format={fmt}) is not a token-level NER "
            f"dataset. Use ICDCodeLookup for code-lookup datasets."
        )

    hf_name = cfg["hf_name"]
    revision = cfg.get("revision")
    logger.info("Loading dataset '%s' from HuggingFace Hub (%s)...", dataset_key, hf_name)
    dataset = load_dataset(hf_name, cache_dir=cache_dir, revision=revision)

    # Apply dataset-specific normalisation
    normaliser = _NORMALISERS.get(dataset_key)
    if normaliser is not None:
        dataset = normaliser(dataset)
    else:
        dataset = _normalise_generic(dataset, dataset_key)

    label_list = get_label_list(dataset)
    logger.info(
        "Dataset '%s' loaded. Labels (%d): %s",
        dataset_key, len(label_list), label_list,
    )
    return dataset, label_list


def get_label_list(dataset: DatasetDict) -> List[str]:
    """Extract sorted, deduplicated label list from the ner_labels column."""
    labels = set()
    for split in dataset:
        for example in dataset[split]:
            labels.update(example["ner_labels"])
    label_list = sorted(labels, key=lambda x: (x != "O", x))
    return label_list


def get_label_maps(label_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label-to-id and id-to-label mappings."""
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    return label2id, id2label
