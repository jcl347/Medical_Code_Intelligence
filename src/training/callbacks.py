"""
Custom training callbacks.
"""

import logging

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EarlyStoppingWithLogging(TrainerCallback):
    """
    Early stopping callback that logs progress and best metric tracking.
    Extends the default HuggingFace EarlyStoppingCallback with better logging.
    """

    def __init__(self, patience: int = 5, metric: str = "eval_f1"):
        self.patience = patience
        self.metric = metric
        self.best_score = None
        self.wait = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        current_score = metrics.get(self.metric)
        if current_score is None:
            logger.warning("Metric '%s' not found in eval metrics.", self.metric)
            return

        if self.best_score is None or current_score > self.best_score:
            improvement = current_score - self.best_score if self.best_score else current_score
            self.best_score = current_score
            self.wait = 0
            logger.info(
                "New best %s: %.4f (+%.4f). Patience reset.",
                self.metric, current_score, improvement,
            )
        else:
            self.wait += 1
            logger.info(
                "%s: %.4f (best: %.4f). Patience: %d/%d",
                self.metric, current_score, self.best_score, self.wait, self.patience,
            )
            if self.wait >= self.patience:
                logger.info("Early stopping triggered after %d evaluations without improvement.", self.wait)
                control.should_training_stop = True
