"""
Data utilities: data collator and sentence splitting helpers.
"""

from typing import Dict, List, Optional

from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast


def create_data_collator(
    tokenizer: PreTrainedTokenizerFast,
    padding: str = "longest",
    label_pad_token_id: int = -100,
) -> DataCollatorForTokenClassification:
    """
    Create a data collator that dynamically pads batches for token classification.

    Dynamic padding (pad to longest in batch) is more efficient than padding
    to max_length for every example.
    """
    return DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=padding,
        label_pad_token_id=label_pad_token_id,
    )


def split_long_sentences(
    tokens: List[str],
    labels: List[str],
    max_tokens: int = 128,
    overlap: int = 16,
) -> List[Dict[str, List[str]]]:
    """
    Split long token sequences into overlapping windows.

    Useful for datasets with very long sentences that exceed model max_length
    even before subword tokenization.

    Parameters
    ----------
    tokens : list of str
        Word-level tokens.
    labels : list of str
        Corresponding BIO labels.
    max_tokens : int
        Maximum number of word-level tokens per window.
    overlap : int
        Number of overlapping tokens between consecutive windows.

    Returns
    -------
    list of dict
        Each dict has 'tokens' and 'ner_labels' keys.
    """
    if len(tokens) <= max_tokens:
        return [{"tokens": tokens, "ner_labels": labels}]

    step = max_tokens - overlap
    windows = []
    for start in range(0, len(tokens), step):
        end = min(start + max_tokens, len(tokens))
        windows.append({
            "tokens": tokens[start:end],
            "ner_labels": labels[start:end],
        })
        if end == len(tokens):
            break

    return windows
