"""
Transformer-based clinical assertion detection.

Uses bvanaken/clinical-assertion-negation-bert (ClinicalBERT fine-tuned
on i2b2 assertion data) for learned negation/assertion classification.

This is a data-driven alternative to the rule-based NegEx/ConText approach.
The model classifies entity mentions in context as:
- PRESENT  (affirmed)
- ABSENT   (negated)
- POSSIBLE (uncertain)

Can be used standalone or stacked with the rule-based NegationDetector
for ensemble-style assertion detection.

Reference:
  van Aken et al. (2021) "Assertion Detection in Clinical Notes: Medical
  Language Models to the Rescue?" NAACL 2021 Clinical NLP Workshop.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Label mapping for the model
_LABEL_MAP = {0: "PRESENT", 1: "ABSENT", 2: "POSSIBLE"}
_LABEL_TO_NEGATION = {
    "PRESENT": "affirmed",
    "ABSENT": "negated",
    "POSSIBLE": "possible",
}


class AssertionClassifier:
    """
    Transformer-based assertion classifier for clinical entities.

    Wraps bvanaken/clinical-assertion-negation-bert. The model expects
    entity spans to be marked with [entity] tokens in the input text.

    Usage
    -----
    >>> classifier = AssertionClassifier()
    >>> result = classifier.predict(
    ...     "Patient denies any chest pain or shortness of breath.",
    ...     entity_text="chest pain",
    ...     entity_start=19,
    ...     entity_end=29,
    ... )
    >>> result
    {'label': 'ABSENT', 'negation': 'negated', 'score': 0.97}

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str, optional
        Device for inference. Auto-detected if None.
    """

    MODEL_NAME = "bvanaken/clinical-assertion-negation-bert"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self._model_name = model_name or self.MODEL_NAME
        self._device = device
        self._pipeline = None

    @property
    def pipeline(self):
        """Lazy-load the classification pipeline."""
        if self._pipeline is None:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                TextClassificationPipeline,
            )
            logger.info("Loading assertion model: %s", self._model_name)
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            device_arg = {}
            if self._device:
                device_arg["device"] = self._device
            self._pipeline = TextClassificationPipeline(
                model=model, tokenizer=tokenizer, **device_arg,
            )
        return self._pipeline

    @staticmethod
    def _mark_entity(text: str, entity_start: int, entity_end: int) -> str:
        """
        Insert [entity] markers around the target entity span.

        The model expects input like:
          "Patient denies any [entity] chest pain [entity]."
        """
        return (
            text[:entity_start]
            + "[entity] "
            + text[entity_start:entity_end]
            + " [entity]"
            + text[entity_end:]
        )

    def predict(
        self,
        text: str,
        entity_text: str,
        entity_start: int,
        entity_end: int,
    ) -> Dict:
        """
        Classify a single entity's assertion status.

        Parameters
        ----------
        text : str
            Full clinical text.
        entity_text : str
            The entity surface form.
        entity_start : int
            Character start offset.
        entity_end : int
            Character end offset.

        Returns
        -------
        dict
            Keys: label (PRESENT/ABSENT/POSSIBLE), negation (affirmed/negated/possible),
            score (float).
        """
        marked = self._mark_entity(text, entity_start, entity_end)
        result = self.pipeline(marked)[0]
        label = result["label"]
        return {
            "label": label,
            "negation": _LABEL_TO_NEGATION.get(label, "affirmed"),
            "score": result["score"],
        }

    def annotate_entities(
        self,
        text: str,
        entities: List[Dict],
    ) -> List[Dict]:
        """
        Annotate a list of entities with assertion status.

        Parameters
        ----------
        text : str
            Clinical text.
        entities : list of dict
            Each dict must have 'start', 'end', 'text' keys.

        Returns
        -------
        list of dict
            Same entities with added 'negation' and 'assertion_score' keys.
        """
        annotated = []
        for ent in entities:
            ent_start = ent.get("start", ent.get("start_char", 0))
            ent_end = ent.get("end", ent.get("end_char", 0))
            ent_text = ent.get("text", text[ent_start:ent_end])

            result = self.predict(text, ent_text, ent_start, ent_end)

            enriched = dict(ent)
            enriched["negation"] = result["negation"]
            enriched["assertion_label"] = result["label"]
            enriched["assertion_score"] = result["score"]
            annotated.append(enriched)

        return annotated
