"""
NER model builder.

Wraps HuggingFace AutoModelForTokenClassification with support for
multiple biomedical pre-trained models and optional CRF layer.
"""

import logging
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from configs.ner_config import MODEL_CONFIGS

logger = logging.getLogger(__name__)


def build_ner_model(
    model_name_or_path: str,
    label_list: List[str],
    use_crf: bool = False,
) -> tuple:
    """
    Build a token classification model and tokenizer for NER.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.
    label_list : list of str
        Ordered NER label strings (e.g. ['O', 'B-Disease', 'I-Disease']).
    use_crf : bool
        If True, wrap the model with a CRF layer for structured prediction.

    Returns
    -------
    model : PreTrainedModel
        Token classification model.
    tokenizer : PreTrainedTokenizerFast
        Associated tokenizer.
    """
    num_labels = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    logger.info("Loading model '%s' with %d labels...", model_name_or_path, num_labels)

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="ner",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        add_prefix_space=False,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )

    if use_crf:
        from src.models.crf_model import CRFTokenClassificationModel
        model = CRFTokenClassificationModel(model, num_labels)
        logger.info("CRF layer added on top of the transformer.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded. Parameters: %s trainable / %s total (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )

    return model, tokenizer
