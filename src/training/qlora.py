"""
QLoRA (Quantized Low-Rank Adaptation) for biomedical NER.

Enables memory-efficient fine-tuning of large transformer models (e.g.
GatorTron) by combining 4-bit NF4 quantization with low-rank adapters.

QLoRA reduces GPU memory by ~75% compared to full fine-tuning while
retaining 95–99% of performance on NER tasks.  This makes it practical
to fine-tune 345M+ parameter models on consumer GPUs (e.g. 16 GB VRAM).

How it works:
1. Load the base model in 4-bit NF4 quantization (bitsandbytes)
2. Freeze all base model weights
3. Attach small trainable LoRA adapters to attention layers
4. Train only the adapter parameters (~1–3% of total)

Required packages (install separately):
    pip install peft>=0.7.0 bitsandbytes>=0.41.0

Based on:
- Hu et al. (2022) — LoRA: Low-Rank Adaptation of Large Language Models
- Dettmers et al. (2023) — QLoRA: Efficient Finetuning of Quantized LLMs

Usage
-----
>>> from src.training.qlora import build_qlora_model
>>> model, tokenizer = build_qlora_model(
...     "UFNLP/gatortron-base",
...     label_list=["O", "B-DIAGNOSIS", "I-DIAGNOSIS"],
... )
>>> # Train with standard HuggingFace Trainer
"""

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def build_qlora_model(
    model_name_or_path: str,
    label_list: List[str],
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_4bit: bool = True,
    use_double_quant: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
) -> Tuple:
    """
    Build a QLoRA-wrapped token classification model.

    Loads the model in 4-bit quantization and applies LoRA adapters
    to the specified attention layers.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path (e.g. ``"UFNLP/gatortron-base"``).
    label_list : list of str
        Ordered NER label strings (e.g. ``["O", "B-DIAGNOSIS", "I-DIAGNOSIS"]``).
    lora_rank : int
        Rank of the LoRA decomposition. Higher = more capacity but more
        parameters.  16 is a good default for NER. Range: 8–64.
    lora_alpha : int
        LoRA scaling factor. Typically ``2 * lora_rank``.
    lora_dropout : float
        Dropout probability for LoRA layers. 0.05–0.1 recommended.
    target_modules : list of str, optional
        Model modules to apply LoRA to.  Defaults to
        ``["query", "value"]`` (attention query and value projections),
        which is standard for BERT-family models doing NER.
    use_4bit : bool
        Use 4-bit NF4 quantization. Set False for 8-bit or full precision.
    use_double_quant : bool
        Use double quantization for additional memory savings.
    bnb_4bit_compute_dtype : str
        Compute dtype for 4-bit ops. ``"bfloat16"`` for Ampere+ GPUs,
        ``"float16"`` for older GPUs.

    Returns
    -------
    model : PeftModel
        QLoRA-wrapped token classification model.
    tokenizer : PreTrainedTokenizerFast
        Associated tokenizer.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoConfig,
            AutoModelForTokenClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as e:
        raise ImportError(
            "QLoRA requires peft and bitsandbytes. Install with:\n"
            "  pip install peft>=0.7.0 bitsandbytes>=0.41.0"
        ) from e

    num_labels = len(label_list)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    if target_modules is None:
        target_modules = ["query", "value"]

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.bfloat16)

    logger.info(
        "Loading '%s' with QLoRA (4-bit=%s, rank=%d, alpha=%d, targets=%s)",
        model_name_or_path, use_4bit, lora_rank, lora_alpha, target_modules,
    )

    # --- 1. Quantization config ---
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )

    # --- 2. Load model in quantized form ---
    model_config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        finetuning_task="ner",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        config=model_config,
        quantization_config=bnb_config,
        ignore_mismatched_sizes=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        add_prefix_space=False,
    )

    # --- 3. Prepare for k-bit training ---
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # --- 4. Apply LoRA adapters ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
    )

    model = get_peft_model(model, lora_config)

    # Log parameter stats
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "QLoRA model loaded. Trainable: %s / %s total (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / total,
    )
    model.print_trainable_parameters()

    return model, tokenizer
