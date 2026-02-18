"""
Contextual abbreviation disambiguation using MeDAL pre-trained models.

Uses ELECTRA pre-trained on the MeDAL task (Medical Abbreviation
Disambiguation for NLU, McGill-NLP) for context-aware expansion of
ambiguous medical abbreviations.

MeDAL pre-training objective: The model learns to detect whether
abbreviations in PubMed text have been replaced by their long forms.
This makes ELECTRA-MeDAL embeddings particularly good at distinguishing
correct vs incorrect abbreviation expansions in context.

Disambiguation approach:
1. Replace the abbreviation with each candidate sense in the text
2. Encode each variant using the ELECTRA discriminator
3. Pick the sense where the model assigns lowest "replaced" probability
   (i.e., the expansion that looks most natural in context)

The CASI (Clinical Abbreviation Sense Inventory) dataset from
mitclinicalml/clinical-ie can be used to evaluate disambiguation accuracy.

References:
  - MeDAL: Wen et al. (2020) "MeDAL: Medical Abbreviation Disambiguation
    Dataset for NLU Pretraining." ClinicalNLP @ EMNLP 2020.
  - CASI: Moon et al. (2014) "A sense inventory for clinical abbreviations
    and acronyms." JAMIA 21(2):299-307.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class AbbreviationDisambiguator:
    """
    Contextual abbreviation disambiguation using MeDAL ELECTRA.

    ELECTRA-MeDAL's discriminator was pre-trained specifically to detect
    replaced medical abbreviation tokens. When we substitute an abbreviation
    with each candidate expansion, the discriminator assigns a low "replaced"
    score to the contextually correct sense and a high score to incorrect ones.

    Supports evaluation against the CASI benchmark dataset (via HuggingFace
    mitclinicalml/clinical-ie Task #1: Clinical Sense Disambiguation).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: McGill-NLP/electra-medal.
    device : str, optional
        Torch device for inference. Auto-detected if None.
    max_length : int
        Maximum sequence length for tokenization.

    Usage
    -----
    >>> disambiguator = AbbreviationDisambiguator()
    >>> senses = ["shortness of breath", "side of bed"]
    >>> best = disambiguator.disambiguate(
    ...     "Patient presents with SOB and fatigue.",
    ...     abbreviation="SOB",
    ...     abbr_start=21, abbr_end=24,
    ...     senses=senses,
    ... )
    >>> best
    'shortness of breath'
    """

    MODEL_NAME = "McGill-NLP/electra-medal"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        self._model_name = model_name or self.MODEL_NAME
        self._device = device
        self._max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the ELECTRA model and tokenizer."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading MeDAL disambiguation model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info("MeDAL model loaded on %s.", self._device)

    def _get_sentence_embedding(self, text: str) -> "np.ndarray":
        """Get mean-pooled sentence embedding from the ELECTRA model."""
        import torch

        self._load_model()

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self._max_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Mean pool over non-padding tokens
        attention_mask = inputs["attention_mask"]
        hidden = outputs.last_hidden_state
        masked = hidden * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        return pooled.cpu().numpy().flatten()

    def disambiguate(
        self,
        text: str,
        abbreviation: str,
        abbr_start: int,
        abbr_end: int,
        senses: List[str],
    ) -> str:
        """
        Disambiguate an abbreviation in context.

        Replaces the abbreviation with each candidate sense, encodes each
        variant, and picks the sense whose embedding is most similar to
        the original text (measured by cosine similarity of mean-pooled
        ELECTRA hidden states).

        Parameters
        ----------
        text : str
            Full text containing the abbreviation.
        abbreviation : str
            The abbreviation text.
        abbr_start : int
            Character start position of the abbreviation.
        abbr_end : int
            Character end position of the abbreviation.
        senses : list of str
            Candidate expansions.

        Returns
        -------
        str
            The best matching sense.
        """
        if not senses:
            return abbreviation
        if len(senses) == 1:
            return senses[0]

        # Get embedding of original text (with abbreviation)
        original_emb = self._get_sentence_embedding(text)

        best_sense = senses[0]
        best_similarity = -1.0

        for sense in senses:
            # Replace abbreviation with this sense
            modified = text[:abbr_start] + sense + text[abbr_end:]

            # Get embedding of modified text
            modified_emb = self._get_sentence_embedding(modified)

            # Cosine similarity
            similarity = float(np.dot(original_emb, modified_emb) / (
                np.linalg.norm(original_emb) * np.linalg.norm(modified_emb) + 1e-8
            ))

            if similarity > best_similarity:
                best_similarity = similarity
                best_sense = sense

        return best_sense

    def disambiguate_from_context(
        self,
        context: str,
        abbreviation: str,
        senses: List[str],
    ) -> Optional[str]:
        """
        Disambiguate using preceding context (simplified API for ShorthandExpander).

        Parameters
        ----------
        context : str
            The context before the abbreviation (preceding words).
        abbreviation : str
            The abbreviation to disambiguate.
        senses : list of str
            Candidate expansions.

        Returns
        -------
        str or None
            The best matching sense, or None if disambiguation fails.
        """
        if not senses or len(senses) <= 1:
            return senses[0] if senses else None

        try:
            # Reconstruct a sentence for embedding
            synthetic_text = f"{context} {abbreviation} ."
            abbr_start = len(context) + 1
            abbr_end = abbr_start + len(abbreviation)
            return self.disambiguate(
                synthetic_text, abbreviation, abbr_start, abbr_end, senses,
            )
        except Exception as e:
            logger.warning("MeDAL disambiguation failed: %s", e)
            return None

    def evaluate_on_casi(
        self,
        max_examples: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate disambiguation accuracy on the CASI benchmark.

        Loads the CASI dataset from mitclinicalml/clinical-ie (Task #1)
        and tests abbreviation disambiguation accuracy.

        Parameters
        ----------
        max_examples : int, optional
            Limit evaluation to this many examples.

        Returns
        -------
        dict
            Evaluation metrics: accuracy, total, correct, per-abbreviation results.
        """
        from datasets import load_dataset

        logger.info("Loading CASI evaluation dataset from mitclinicalml/clinical-ie...")
        ds = load_dataset("mitclinicalml/clinical-ie", trust_remote_code=True)

        correct = 0
        total = 0
        per_abbr: Dict[str, Dict] = {}

        for split in ds:
            for example in ds[split]:
                # CASI format varies; adapt based on actual schema
                text = example.get("text", example.get("sentence", ""))
                abbr = example.get("acronym", example.get("abbreviation", ""))
                label = example.get("label", example.get("expansion", ""))
                senses = example.get("options", example.get("senses", []))

                if not text or not abbr or not senses:
                    continue

                if max_examples and total >= max_examples:
                    break

                # Find abbreviation position in text
                abbr_start = text.lower().find(abbr.lower())
                if abbr_start < 0:
                    continue
                abbr_end = abbr_start + len(abbr)

                predicted = self.disambiguate(text, abbr, abbr_start, abbr_end, senses)
                is_correct = predicted.lower() == label.lower()
                correct += int(is_correct)
                total += 1

                abbr_lower = abbr.lower()
                if abbr_lower not in per_abbr:
                    per_abbr[abbr_lower] = {"correct": 0, "total": 0}
                per_abbr[abbr_lower]["total"] += 1
                per_abbr[abbr_lower]["correct"] += int(is_correct)

        accuracy = correct / total if total > 0 else 0.0
        logger.info("CASI evaluation: %d/%d correct (%.1f%%)", correct, total, accuracy * 100)

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_abbreviation": per_abbr,
        }
