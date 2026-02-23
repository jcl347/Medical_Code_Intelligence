"""
Adversarial training for biomedical NER.

Implements FGM (Fast Gradient Method) and PGD (Projected Gradient Descent)
adversarial perturbation on word embeddings to improve model robustness
and F1 scores.

Based on:
- Goodfellow et al. (2015) — adversarial perturbation of inputs
- Miyato et al. (2017) — adversarial training for text classification
- Zhu et al. (2020) — FreeLB for NLU (ICLR 2020)
- RanAT4BIE (2025) — adversarial training for biomedical IE

Reported gains: +0.5–1.5% entity-level F1 on biomedical NER benchmarks
(NCBI Disease, BC5CDR) when applied to PubMedBERT/BioBERT models.

Usage
-----
>>> from src.training.adversarial import AdversarialTrainer
>>> trainer = AdversarialTrainer(
...     model=model, args=training_args,
...     train_dataset=train_ds, eval_dataset=eval_ds,
...     adv_method="fgm", adv_epsilon=1.0,
... )
>>> trainer.train()
"""

import logging
from typing import Optional

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class FGM:
    """
    Fast Gradient Method for adversarial training.

    Computes a single-step perturbation on the embedding layer in the
    direction of the gradient, scaled to a fixed epsilon-ball.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose embeddings will be perturbed.
    epsilon : float
        Perturbation magnitude (L2 norm). Default 1.0 following
        standard practice for BERT-based models.
    emb_name : str
        Name fragment to match the embedding parameter. For BERT models
        this is typically ``"word_embeddings"``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 1.0,
        emb_name: str = "word_embeddings",
    ):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self._backup: dict = {}

    def attack(self):
        """Add adversarial perturbation to embedding parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self._backup[name] = param.data.clone()
                grad = param.grad
                if grad is None:
                    continue
                norm = torch.norm(grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = self.epsilon * grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        """Restore original embedding parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self._backup:
                    param.data = self._backup[name]
        self._backup = {}


class PGD:
    """
    Projected Gradient Descent for adversarial training.

    Multi-step variant of FGM: takes K gradient ascent steps within an
    epsilon-ball around the original embeddings.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose embeddings will be perturbed.
    epsilon : float
        Maximum perturbation magnitude (L2 norm). Default 0.3.
    alpha : float
        Step size for each PGD iteration. Default 0.1.
    num_steps : int
        Number of PGD iterations. Default 3. Research shows diminishing
        returns beyond K=3 for NLP tasks.
    emb_name : str
        Name fragment to match the embedding parameter.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 0.3,
        alpha: float = 0.1,
        num_steps: int = 3,
        emb_name: str = "word_embeddings",
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.emb_name = emb_name
        self._backup: dict = {}
        self._delta: dict = {}

    def save(self):
        """Save original embedding parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self._backup[name] = param.data.clone()
                self._delta[name] = torch.zeros_like(param.data)

    def attack_step(self):
        """Execute one PGD step: accumulate delta and project into epsilon-ball."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                grad = param.grad
                if grad is None:
                    continue
                norm = torch.norm(grad)
                if norm != 0 and not torch.isnan(norm):
                    self._delta[name] += self.alpha * grad / norm
                    # Project back into epsilon-ball
                    delta_norm = torch.norm(self._delta[name])
                    if delta_norm > self.epsilon:
                        self._delta[name] = (
                            self.epsilon * self._delta[name] / delta_norm
                        )
                    param.data = self._backup[name] + self._delta[name]

    def restore(self):
        """Restore original embedding parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self._backup:
                    param.data = self._backup[name]
        self._backup = {}
        self._delta = {}


class AdversarialTrainer(Trainer):
    """
    HuggingFace Trainer with adversarial training on word embeddings.

    Extends the standard ``training_step`` to:
    1. Compute loss and gradients on the clean input.
    2. Perturb the embedding layer in the gradient direction (FGM/PGD).
    3. Compute loss on the perturbed input and accumulate gradients.
    4. Restore original embeddings before the optimizer step.

    This regularizes the model to be robust to small input perturbations,
    consistently improving entity-level F1 by +0.5–1.5% on biomedical
    NER benchmarks.

    Parameters
    ----------
    adv_method : str
        Adversarial method: ``"fgm"`` (default, ~2x cost) or
        ``"pgd"`` (~(K+1)x cost).
    adv_epsilon : float
        Perturbation magnitude. Default 1.0 for FGM, 0.3 for PGD.
    pgd_alpha : float
        PGD step size. Only used when adv_method="pgd". Default 0.1.
    pgd_steps : int
        Number of PGD iterations. Only used when adv_method="pgd". Default 3.
    emb_name : str
        Name fragment to match the embedding parameter in the model.
    **kwargs
        All other arguments are passed to ``Trainer.__init__``.
    """

    def __init__(
        self,
        *args,
        adv_method: str = "fgm",
        adv_epsilon: Optional[float] = None,
        pgd_alpha: float = 0.1,
        pgd_steps: int = 3,
        emb_name: str = "word_embeddings",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.adv_method = adv_method.lower()

        if self.adv_method == "fgm":
            eps = adv_epsilon if adv_epsilon is not None else 1.0
            self._adversary = FGM(self.model, epsilon=eps, emb_name=emb_name)
            logger.info("Adversarial training: FGM (epsilon=%.2f)", eps)
        elif self.adv_method == "pgd":
            eps = adv_epsilon if adv_epsilon is not None else 0.3
            self._adversary = PGD(
                self.model,
                epsilon=eps,
                alpha=pgd_alpha,
                num_steps=pgd_steps,
                emb_name=emb_name,
            )
            logger.info(
                "Adversarial training: PGD (epsilon=%.2f, alpha=%.2f, steps=%d)",
                eps, pgd_alpha, pgd_steps,
            )
        else:
            raise ValueError(
                f"Unknown adv_method '{adv_method}'. Choose 'fgm' or 'pgd'."
            )

    def training_step(
        self, model: torch.nn.Module, inputs: dict, num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Training step with adversarial perturbation.

        For FGM: one extra forward+backward pass (~2x cost).
        For PGD: K extra forward+backward passes (~(K+1)x cost).
        """
        # Standard forward + backward
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.adv_method == "fgm":
            # Perturb embeddings using gradient direction
            self._adversary.attack()
            # Adversarial forward + backward (gradients accumulate)
            super().training_step(model, inputs, num_items_in_batch)
            # Restore clean embeddings
            self._adversary.restore()

        elif self.adv_method == "pgd":
            self._adversary.save()
            for _step in range(self._adversary.num_steps):
                self._adversary.attack_step()
                if _step < self._adversary.num_steps - 1:
                    # Zero gradients for intermediate steps
                    model.zero_grad()
                super().training_step(model, inputs, num_items_in_batch)
            self._adversary.restore()

        return loss
