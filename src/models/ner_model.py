"""
NER model builder.

Wraps HuggingFace AutoModelForTokenClassification with support for
multiple biomedical pre-trained models, optional CRF layer, and
LoRA/QLoRA parameter-efficient fine-tuning.

LoRA/QLoRA support:
- **LoRA**: Adds low-rank adapter layers to attention projections,
  training only ~0.5% of parameters while matching full fine-tuning.
- **QLoRA**: Loads the base model in 4-bit NF4 quantization + LoRA,
  reducing memory by ~87.5%. Useful for large models (e.g. GatorTron
  345M) on memory-constrained GPUs.

Requires: ``peft>=0.6.0`` for LoRA, plus ``bitsandbytes>=0.41.0``
for QLoRA quantization.
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
    use_lora: bool = False,
    use_qlora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
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
        Mutually exclusive with LoRA/QLoRA.
    use_lora : bool
        If True, apply LoRA adapters to attention layers (~0.5% trainable).
    use_qlora : bool
        If True, load in 4-bit NF4 quantization + LoRA adapters.
        Requires bitsandbytes and a CUDA GPU.
    lora_r : int
        LoRA rank. Higher = more capacity, more trainable params.
    lora_alpha : int
        LoRA scaling factor. Effective scaling is alpha/r.
    lora_dropout : float
        Dropout applied to LoRA layers.
    lora_target_modules : list of str, optional
        Attention modules to apply LoRA to.
        Default: ["query", "key", "value"] (works for BERT/MegatronBERT).

    Returns
    -------
    model : PreTrainedModel
        Token classification model (optionally with LoRA/QLoRA/CRF).
    tokenizer : PreTrainedTokenizerFast
        Associated tokenizer.
    """
    if use_crf and (use_lora or use_qlora):
        raise ValueError("CRF and LoRA/QLoRA are mutually exclusive.")

    if lora_target_modules is None:
        lora_target_modules = ["query", "key", "value"]

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

    # --- QLoRA: load in 4-bit quantization ---
    if use_qlora:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "QLoRA requires transformers>=4.36 with BitsAndBytesConfig. "
                "Run: pip install transformers>=4.36"
            )
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            raise ImportError(
                "QLoRA requires bitsandbytes. Run: pip install bitsandbytes>=0.41.0"
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading model in 4-bit NF4 quantization (QLoRA)...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config,
            quantization_config=bnb_config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=True,
        )

    # --- LoRA / QLoRA: apply adapters ---
    if use_lora or use_qlora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError(
                "LoRA/QLoRA requires peft. Run: pip install peft>=0.6.0"
            )

        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="all",
            target_modules=lora_target_modules,
            modules_to_save=["classifier"],  # Train NER head in full precision
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.info(
            "LoRA applied (r=%d, alpha=%d, targets=%s). "
            "Classifier head trained in full precision.",
            lora_r, lora_alpha, lora_target_modules,
        )

    # --- CRF layer ---
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
