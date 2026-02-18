"""
Unified Medical Coding NER pipeline.

Chains together:
1. Physician shorthand expansion (pre-processing)
2. Transformer-based NER (entity extraction)
3. Negation / context detection (post-processing)
4. Entity normalisation and medical code suggestion

This is the main user-facing API for extracting coded medical entities
from clinical text.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.clinical.negation import NegationDetector, NegationStatus
from src.clinical.shorthand import ShorthandExpander

logger = logging.getLogger(__name__)


@dataclass
class MedicalEntity:
    """A medical entity with full clinical context annotation."""
    text: str
    label: str
    start_char: int
    end_char: int
    score: float
    negation: str = "affirmed"          # affirmed / negated / possible / etc.
    negation_trigger: Optional[str] = None
    original_text: Optional[str] = None  # before shorthand expansion
    expanded_from: Optional[str] = None  # abbreviation that was expanded

    def to_dict(self) -> Dict:
        d = {
            "text": self.text,
            "label": self.label,
            "start": self.start_char,
            "end": self.end_char,
            "score": round(self.score, 4),
            "negation": self.negation,
        }
        if self.negation_trigger:
            d["negation_trigger"] = self.negation_trigger
        if self.expanded_from:
            d["expanded_from"] = self.expanded_from
            d["original_text"] = self.original_text
        return d

    @property
    def is_negated(self) -> bool:
        return self.negation == NegationStatus.NEGATED.value

    @property
    def is_affirmed(self) -> bool:
        return self.negation == NegationStatus.AFFIRMED.value


class MedicalCodingPipeline:
    """
    End-to-end pipeline for medical coding NER.

    Supports two negation strategies:
    - "rules" (default): ConText/NegEx rule-based detector (fast, no GPU)
    - "transformer": bvanaken/clinical-assertion-negation-bert (learned, GPU-optional)

    Usage
    -----
    >>> pipeline = MedicalCodingPipeline(model_path="outputs/best_model")
    >>> results = pipeline("Pt denies cp or sob. Hx of dm2 and htn.")
    >>> for entity in results:
    ...     print(f"{entity.text} [{entity.label}] - {entity.negation}")
    chest pain [Disease] - negated
    shortness of breath [Disease] - negated
    type 2 diabetes mellitus [Disease] - historical
    hypertension [Disease] - historical

    Parameters
    ----------
    model_path : str
        Path to a fine-tuned NER model (or HuggingFace model id).
    expand_shorthand : bool
        Whether to expand physician abbreviations before NER.
    detect_negation : bool
        Whether to run negation/context detection on extracted entities.
    negation_strategy : str
        "rules" for ConText/NegEx, "transformer" for learned assertion model.
    device : str, optional
        Torch device. Auto-detected if None.
    custom_abbreviations : dict, optional
        Additional abbreviation mappings to add.
    negation_scope_window : int
        Max word-distance for rule-based negation scope propagation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        expand_shorthand: bool = True,
        detect_negation: bool = True,
        negation_strategy: str = "rules",
        device: Optional[str] = None,
        custom_abbreviations: Optional[Dict[str, str]] = None,
        negation_scope_window: int = 6,
    ):
        self.expand_shorthand = expand_shorthand
        self.detect_negation = detect_negation
        self.negation_strategy = negation_strategy

        # Initialise sub-components
        if expand_shorthand:
            self.shorthand_expander = ShorthandExpander(
                custom_abbreviations=custom_abbreviations,
            )
        else:
            self.shorthand_expander = None

        if detect_negation:
            if negation_strategy == "transformer":
                from src.clinical.assertion import AssertionClassifier
                self.negation_detector = None
                self.assertion_classifier = AssertionClassifier(device=device)
            else:
                self.negation_detector = NegationDetector(
                    scope_window=negation_scope_window,
                )
                self.assertion_classifier = None
        else:
            self.negation_detector = None
            self.assertion_classifier = None

        # NER predictor (lazy-loaded if model_path given)
        self._predictor = None
        self._model_path = model_path
        self._device = device

    @property
    def predictor(self):
        """Lazy-load the NER model."""
        if self._predictor is None:
            if self._model_path is None:
                raise RuntimeError(
                    "No model_path provided. Either pass model_path to __init__ "
                    "or use process_with_entities() to supply pre-extracted entities."
                )
            from src.inference.predictor import NERPredictor
            self._predictor = NERPredictor(
                self._model_path, device=self._device,
            )
        return self._predictor

    def __call__(self, text: str) -> List[MedicalEntity]:
        """Run the full pipeline on a clinical text string."""
        return self.process(text)

    def process(self, text: str) -> List[MedicalEntity]:
        """
        Run the full medical coding NER pipeline.

        Steps:
        1. Expand physician shorthand (optional)
        2. Run NER model to extract entities
        3. Detect negation / clinical context (optional)
        4. Map expanded offsets back to original text

        Parameters
        ----------
        text : str
            Raw clinical text.

        Returns
        -------
        list of MedicalEntity
            Annotated entities with negation status.
        """
        original_text = text
        offset_map = []

        # Step 1: Shorthand expansion
        if self.shorthand_expander is not None:
            expanded_text, offset_map = self.shorthand_expander.expand_with_offsets(text)
        else:
            expanded_text = text

        # Step 2: NER
        raw_entities = self.predictor.predict(expanded_text)

        # Convert to dicts for negation detector
        entity_dicts = [
            {
                "text": e.text,
                "label": e.label,
                "start": e.start_char,
                "end": e.end_char,
                "score": e.score,
            }
            for e in raw_entities
        ]

        # Step 3: Negation detection
        if self.negation_detector is not None and entity_dicts:
            entity_dicts = self.negation_detector.annotate_entities(
                expanded_text, entity_dicts,
            )

        # Step 4: Build MedicalEntity objects with offset mapping
        results = []
        for ent in entity_dicts:
            # Check if this entity overlaps with an expanded abbreviation
            expanded_from = None
            original_span_text = None
            for om in offset_map:
                if (om["expanded_start"] <= ent["start"] < om["expanded_end"]
                        or om["expanded_start"] < ent["end"] <= om["expanded_end"]):
                    expanded_from = om["abbreviation"]
                    original_span_text = om["abbreviation"]
                    break

            results.append(MedicalEntity(
                text=ent["text"],
                label=ent["label"],
                start_char=ent["start"],
                end_char=ent["end"],
                score=ent["score"],
                negation=ent.get("negation", "affirmed"),
                negation_trigger=ent.get("negation_trigger"),
                original_text=original_span_text,
                expanded_from=expanded_from,
            ))

        return results

    def process_with_entities(
        self,
        text: str,
        entities: List[Dict],
    ) -> List[MedicalEntity]:
        """
        Run negation detection on pre-extracted entities (no NER model needed).

        Useful when entities are extracted by another system, or during
        evaluation when you want to test negation independently.

        Parameters
        ----------
        text : str
            Clinical text.
        entities : list of dict
            Pre-extracted entities with 'text', 'label', 'start', 'end' keys.

        Returns
        -------
        list of MedicalEntity
        """
        expanded_text = text
        offset_map = []

        if self.shorthand_expander is not None:
            expanded_text, offset_map = self.shorthand_expander.expand_with_offsets(text)

        # Apply negation detection (rule-based or transformer)
        if entities:
            if self.assertion_classifier is not None:
                entities = self.assertion_classifier.annotate_entities(
                    expanded_text, entities,
                )
            elif self.negation_detector is not None:
                entities = self.negation_detector.annotate_entities(
                    expanded_text, entities,
                )

        results = []
        for ent in entities:
            expanded_from = None
            for om in offset_map:
                if (om["expanded_start"] <= ent.get("start", 0) < om["expanded_end"]):
                    expanded_from = om["abbreviation"]
                    break

            results.append(MedicalEntity(
                text=ent.get("text", ""),
                label=ent.get("label", ""),
                start_char=ent.get("start", 0),
                end_char=ent.get("end", 0),
                score=ent.get("score", 1.0),
                negation=ent.get("negation", "affirmed"),
                negation_trigger=ent.get("negation_trigger"),
                expanded_from=expanded_from,
            ))

        return results

    def process_batch(self, texts: List[str]) -> List[List[MedicalEntity]]:
        """Run the pipeline on a batch of texts."""
        return [self.process(text) for text in texts]

    def format_output(self, text: str, entities: List[MedicalEntity]) -> str:
        """
        Format annotated entities for human-readable display.

        Example output:
            [chest pain](Disease, NEGATED, trigger="denies")
            [diabetes mellitus](Disease, AFFIRMED, from="dm")
        """
        if not entities:
            return text + "\n  (no entities found)"

        lines = [text, ""]
        for ent in entities:
            parts = [ent.label, ent.negation.upper()]
            if ent.negation_trigger:
                parts.append(f'trigger="{ent.negation_trigger}"')
            if ent.expanded_from:
                parts.append(f'from="{ent.expanded_from}"')
            parts.append(f"score={ent.score:.3f}")
            annotation = ", ".join(parts)
            lines.append(f"  [{ent.text}]({annotation})")

        return "\n".join(lines)
