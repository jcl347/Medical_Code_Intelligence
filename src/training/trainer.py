"""
Training pipeline for biomedical NER.

Implements state-of-the-art training practices:
- Learning rate warmup + linear/cosine decay
- Mixed-precision (FP16) training
- Gradient accumulation for effective larger batch sizes
- Early stopping on validation F1
- Proper evaluation at regular intervals
- Reproducible seeding
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from configs.ner_config import NERConfig
from src.data.data_utils import create_data_collator
from src.evaluation.metrics import build_compute_metrics_fn

logger = logging.getLogger(__name__)


def build_training_args(config: NERConfig) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments from our NERConfig.

    Follows best practices:
    - eval and save aligned to same step interval
    - load_best_model_at_end for early stopping
    - metric_for_best_model set to entity-level F1
    - FP16 for speed (if GPU available)
    """
    output_dir = os.path.join(config.output_dir, config.experiment_name)

    return TrainingArguments(
        output_dir=output_dir,
        # Training schedule
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Optimiser
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        # Mixed precision
        fp16=config.fp16,
        # Label smoothing
        label_smoothing_factor=config.label_smoothing_factor,
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # Logging
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.experiment_name if config.use_wandb else None,
        # Reproducibility
        seed=config.seed,
        data_seed=config.seed,
        # Performance
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        # Misc
        remove_unused_columns=True,
        push_to_hub=False,
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    config: NERConfig,
    train_dataset,
    eval_dataset,
    label_list: List[str],
) -> Trainer:
    """
    Build a HuggingFace Trainer configured for NER.

    Parameters
    ----------
    model : PreTrainedModel
        Token classification model.
    tokenizer : PreTrainedTokenizerFast
        Tokenizer for data collation.
    config : NERConfig
        Experiment configuration.
    train_dataset : Dataset
        Tokenized training split.
    eval_dataset : Dataset
        Tokenized validation split.
    label_list : list of str
        Ordered label strings.

    Returns
    -------
    Trainer
        Configured trainer ready for .train().
    """
    training_args = build_training_args(config)
    data_collator = create_data_collator(tokenizer)
    compute_metrics = build_compute_metrics_fn(label_list)

    callbacks = []
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("Trainer built. Output dir: %s", training_args.output_dir)
    return trainer
