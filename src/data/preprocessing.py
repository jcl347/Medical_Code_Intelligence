"""
Tokenization and label alignment for subword tokenizers.

Implements the standard approach for aligning BIO/IOB2 NER labels with
WordPiece / BPE subword tokens, following HuggingFace best practices.
"""

import logging
from typing import Dict, List, Optional

from datasets import DatasetDict
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Special label id used for subword tokens and special tokens ([CLS], [SEP], [PAD])
IGNORE_LABEL_ID = -100


def tokenize_and_align_labels(
    examples: Dict,
    tokenizer: PreTrainedTokenizerFast,
    label2id: Dict[str, int],
    max_length: int = 512,
    label_all_tokens: bool = False,
) -> Dict:
    """
    Tokenize sentences and align NER labels with subword tokens.

    Strategy (state-of-the-art):
    - First subword of each word gets the original label.
    - Subsequent subwords get IGNORE_LABEL_ID (default) or the same label
      if label_all_tokens=True (with B- converted to I- for continuation).
    - Special tokens ([CLS], [SEP], [PAD]) get IGNORE_LABEL_ID.

    Parameters
    ----------
    examples : dict
        Batch from HuggingFace dataset with 'tokens' and 'ner_labels'.
    tokenizer : PreTrainedTokenizerFast
        Subword tokenizer (must support word_ids()).
    label2id : dict
        Mapping from label string to integer id.
    max_length : int
        Max sequence length for truncation.
    label_all_tokens : bool
        If True, propagate labels to all subword pieces (with B->I conversion).

    Returns
    -------
    dict
        Tokenized inputs with 'labels' aligned to subword tokens.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,  # dynamic padding via data collator is more efficient
        max_length=max_length,
        is_split_into_words=True,
        return_offsets_mapping=False,
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special token
                label_ids.append(IGNORE_LABEL_ID)
            elif word_idx != previous_word_idx:
                # First subword of the word -> assign original label
                label_str = labels[word_idx] if word_idx < len(labels) else "O"
                label_ids.append(label2id.get(label_str, label2id.get("O", 0)))
            else:
                # Continuation subword
                if label_all_tokens:
                    label_str = labels[word_idx] if word_idx < len(labels) else "O"
                    # Convert B- prefix to I- for continuation subwords
                    if label_str.startswith("B-"):
                        label_str = "I-" + label_str[2:]
                    label_ids.append(label2id.get(label_str, label2id.get("O", 0)))
                else:
                    label_ids.append(IGNORE_LABEL_ID)

            previous_word_idx = word_idx

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerFast,
    label2id: Dict[str, int],
    max_length: int = 512,
    label_all_tokens: bool = False,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Apply tokenization and label alignment to all splits in a DatasetDict.

    Parameters
    ----------
    dataset : DatasetDict
        Raw dataset with 'tokens' and 'ner_labels' columns.
    tokenizer : PreTrainedTokenizerFast
        Subword tokenizer.
    label2id : dict
        Label to integer mapping.
    max_length : int
        Maximum sequence length.
    label_all_tokens : bool
        Whether to propagate labels to continuation subwords.
    num_proc : int
        Number of processes for parallel mapping.

    Returns
    -------
    DatasetDict
        Tokenized dataset ready for training.
    """
    logger.info("Preprocessing dataset (max_length=%d, label_all_tokens=%s)...", max_length, label_all_tokens)

    # Columns to remove after tokenization
    cols_to_remove = [
        c for c in dataset["train"].column_names
        if c not in ("input_ids", "attention_mask", "labels", "token_type_ids")
    ]

    tokenized = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id, max_length, label_all_tokens),
        batched=True,
        num_proc=num_proc,
        remove_columns=cols_to_remove,
        desc="Tokenizing and aligning labels",
    )

    logger.info("Preprocessing complete.")
    return tokenized
