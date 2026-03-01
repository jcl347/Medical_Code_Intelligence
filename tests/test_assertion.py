"""
Tests for transformer-based assertion/negation classifier.

Uses mocked transformer pipeline to avoid downloading the model in CI.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from src.clinical.assertion import AssertionClassifier, _LABEL_MAP, _LABEL_TO_NEGATION


# ---------------------------------------------------------------------------
# Entity marking
# ---------------------------------------------------------------------------

class TestEntityMarking:
    """Test the [entity] marker insertion logic."""

    def test_mark_entity_basic(self):
        text = "Patient denies chest pain or shortness of breath."
        marked = AssertionClassifier._mark_entity(text, 15, 25)
        assert "[entity] chest pain [entity]" in marked
        assert marked.startswith("Patient denies ")

    def test_mark_entity_at_start(self):
        text = "Fever noted on admission."
        marked = AssertionClassifier._mark_entity(text, 0, 5)
        assert marked.startswith("[entity] Fever [entity]")

    def test_mark_entity_at_end(self):
        text = "Patient has cough"
        marked = AssertionClassifier._mark_entity(text, 12, 17)
        assert marked.endswith("[entity] cough [entity]")

    def test_mark_entity_preserves_surrounding_text(self):
        text = "No evidence of pneumonia on imaging."
        marked = AssertionClassifier._mark_entity(text, 15, 24)
        assert "No evidence of" in marked
        assert "on imaging." in marked
        assert "[entity] pneumonia [entity]" in marked


# ---------------------------------------------------------------------------
# Mocked prediction
# ---------------------------------------------------------------------------

class TestPrediction:
    """Test predict() with mocked transformer pipeline."""

    @pytest.fixture
    def classifier(self):
        clf = AssertionClassifier.__new__(AssertionClassifier)
        clf._model_name = "mock-model"
        clf._device = None

        mock_pipeline = MagicMock()
        clf._pipeline = mock_pipeline
        return clf

    def test_predict_absent(self, classifier):
        classifier._pipeline.return_value = [{"label": "ABSENT", "score": 0.97}]

        result = classifier.predict(
            "Patient denies chest pain.",
            entity_text="chest pain",
            entity_start=15,
            entity_end=25,
        )
        assert result["label"] == "ABSENT"
        assert result["negation"] == "negated"
        assert result["score"] == 0.97

    def test_predict_present(self, classifier):
        classifier._pipeline.return_value = [{"label": "PRESENT", "score": 0.95}]

        result = classifier.predict(
            "Patient presents with fever.",
            entity_text="fever",
            entity_start=22,
            entity_end=27,
        )
        assert result["label"] == "PRESENT"
        assert result["negation"] == "affirmed"
        assert result["score"] == 0.95

    def test_predict_possible(self, classifier):
        classifier._pipeline.return_value = [{"label": "POSSIBLE", "score": 0.82}]

        result = classifier.predict(
            "Possible pneumonia on imaging.",
            entity_text="pneumonia",
            entity_start=9,
            entity_end=18,
        )
        assert result["label"] == "POSSIBLE"
        assert result["negation"] == "possible"
        assert result["score"] == 0.82

    def test_predict_passes_marked_text_to_pipeline(self, classifier):
        classifier._pipeline.return_value = [{"label": "PRESENT", "score": 0.9}]

        classifier.predict(
            "Patient has cough.",
            entity_text="cough",
            entity_start=12,
            entity_end=17,
        )

        call_args = classifier._pipeline.call_args[0][0]
        assert "[entity] cough [entity]" in call_args


# ---------------------------------------------------------------------------
# Batch annotation
# ---------------------------------------------------------------------------

class TestAnnotateEntities:
    """Test annotate_entities() with mocked pipeline."""

    @pytest.fixture
    def classifier(self):
        clf = AssertionClassifier.__new__(AssertionClassifier)
        clf._model_name = "mock-model"
        clf._device = None
        clf._pipeline = MagicMock()
        return clf

    def test_annotate_multiple_entities(self, classifier):
        responses = [
            [{"label": "ABSENT", "score": 0.95}],
            [{"label": "PRESENT", "score": 0.88}],
        ]
        classifier._pipeline.side_effect = responses

        text = "Denies fever. Reports cough."
        entities = [
            {"text": "fever", "label": "Symptom", "start": 7, "end": 12},
            {"text": "cough", "label": "Symptom", "start": 22, "end": 27},
        ]

        annotated = classifier.annotate_entities(text, entities)

        assert len(annotated) == 2
        assert annotated[0]["negation"] == "negated"
        assert annotated[0]["assertion_label"] == "ABSENT"
        assert annotated[0]["assertion_score"] == 0.95
        assert annotated[1]["negation"] == "affirmed"
        assert annotated[1]["assertion_label"] == "PRESENT"

    def test_annotate_preserves_original_fields(self, classifier):
        classifier._pipeline.return_value = [{"label": "PRESENT", "score": 0.9}]

        entities = [
            {"text": "pneumonia", "label": "Disease", "start": 10, "end": 19, "score": 0.95},
        ]
        annotated = classifier.annotate_entities("Patient has pneumonia.", entities)

        assert annotated[0]["text"] == "pneumonia"
        assert annotated[0]["label"] == "Disease"
        assert annotated[0]["score"] == 0.95

    def test_annotate_empty_list(self, classifier):
        annotated = classifier.annotate_entities("Some text.", [])
        assert annotated == []

    def test_annotate_uses_start_char_fallback(self, classifier):
        """Test that start_char/end_char keys are also supported."""
        classifier._pipeline.return_value = [{"label": "PRESENT", "score": 0.9}]

        entities = [
            {"text": "pain", "label": "Symptom", "start_char": 5, "end_char": 9},
        ]
        annotated = classifier.annotate_entities("Acute pain noted.", entities)
        assert annotated[0]["negation"] == "affirmed"


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

class TestLabelMapping:
    """Test the label map constants."""

    def test_label_map_has_three_classes(self):
        assert len(_LABEL_MAP) == 3
        assert set(_LABEL_MAP.values()) == {"PRESENT", "ABSENT", "POSSIBLE"}

    def test_negation_mapping_present(self):
        assert _LABEL_TO_NEGATION["PRESENT"] == "affirmed"

    def test_negation_mapping_absent(self):
        assert _LABEL_TO_NEGATION["ABSENT"] == "negated"

    def test_negation_mapping_possible(self):
        assert _LABEL_TO_NEGATION["POSSIBLE"] == "possible"


# ---------------------------------------------------------------------------
# Pipeline integration with transformer negation
# ---------------------------------------------------------------------------

class TestPipelineTransformerNegation:
    """Test MedicalCodingPipeline with negation_strategy='transformer'."""

    def test_pipeline_creates_assertion_classifier(self):
        with patch("src.clinical.assertion.AssertionClassifier") as MockCls:
            from src.clinical.pipeline import MedicalCodingPipeline
            pipeline = MedicalCodingPipeline(
                model_path=None,
                detect_negation=True,
                negation_strategy="transformer",
            )
            MockCls.assert_called_once()
            assert pipeline.assertion_classifier is not None
            # Rule-based detector is also created for HISTORICAL/FAMILY supplement
            assert pipeline.negation_detector is not None

    def test_pipeline_rules_creates_negation_detector(self):
        from src.clinical.pipeline import MedicalCodingPipeline
        pipeline = MedicalCodingPipeline(
            model_path=None,
            detect_negation=True,
            negation_strategy="rules",
        )
        assert pipeline.negation_detector is not None
        assert pipeline.assertion_classifier is None

    def test_pipeline_no_negation(self):
        from src.clinical.pipeline import MedicalCodingPipeline
        pipeline = MedicalCodingPipeline(
            model_path=None,
            detect_negation=False,
        )
        assert pipeline.negation_detector is None
        assert pipeline.assertion_classifier is None

    def test_process_with_entities_uses_assertion_classifier(self):
        with patch("src.clinical.assertion.AssertionClassifier") as MockCls:
            mock_clf = MagicMock()
            mock_clf.annotate_entities.return_value = [
                {"text": "fever", "label": "Symptom", "start": 7, "end": 12,
                 "negation": "negated", "assertion_label": "ABSENT",
                 "assertion_score": 0.95},
            ]
            MockCls.return_value = mock_clf

            from src.clinical.pipeline import MedicalCodingPipeline
            pipeline = MedicalCodingPipeline(
                model_path=None,
                expand_shorthand=False,
                detect_negation=True,
                negation_strategy="transformer",
            )

            text = "Denies fever."
            entities = [
                {"text": "fever", "label": "Symptom", "start": 7, "end": 12},
            ]
            results = pipeline.process_with_entities(text, entities)

            assert len(results) == 1
            assert results[0].negation == "negated"
            mock_clf.annotate_entities.assert_called_once()

    def test_default_strategy_is_transformer(self):
        """Verify the default negation strategy is 'transformer'."""
        with patch("src.clinical.assertion.AssertionClassifier") as MockCls:
            from src.clinical.pipeline import MedicalCodingPipeline
            pipeline = MedicalCodingPipeline(
                model_path=None,
                detect_negation=True,
            )
            MockCls.assert_called_once()
            assert pipeline.negation_strategy == "transformer"
            assert pipeline.assertion_classifier is not None
            assert pipeline.negation_detector is not None

    def test_hybrid_supplements_historical(self):
        """Transformer marks PRESENT, but rules detect HISTORICAL context."""
        with patch("src.clinical.assertion.AssertionClassifier") as MockCls:
            mock_clf = MagicMock()
            # Transformer says PRESENT (it can't detect HISTORICAL)
            mock_clf.annotate_entities.return_value = [
                {"text": "myocardial infarction", "label": "Disease",
                 "start": 11, "end": 31, "negation": "affirmed",
                 "assertion_label": "PRESENT", "assertion_score": 0.90},
            ]
            MockCls.return_value = mock_clf

            from src.clinical.pipeline import MedicalCodingPipeline
            pipeline = MedicalCodingPipeline(
                model_path=None,
                expand_shorthand=False,
                detect_negation=True,
                negation_strategy="transformer",
            )

            text = "History of myocardial infarction."
            entities = [
                {"text": "myocardial infarction", "label": "Disease",
                 "start": 11, "end": 31},
            ]
            results = pipeline.process_with_entities(text, entities)

            assert len(results) == 1
            # Rule-based supplement should override to historical
            assert results[0].negation == "historical"

    def test_hybrid_supplements_family(self):
        """Transformer marks PRESENT, but rules detect FAMILY context."""
        with patch("src.clinical.assertion.AssertionClassifier") as MockCls:
            mock_clf = MagicMock()
            mock_clf.annotate_entities.return_value = [
                {"text": "colon cancer", "label": "Disease",
                 "start": 18, "end": 30, "negation": "affirmed",
                 "assertion_label": "PRESENT", "assertion_score": 0.85},
            ]
            MockCls.return_value = mock_clf

            from src.clinical.pipeline import MedicalCodingPipeline
            pipeline = MedicalCodingPipeline(
                model_path=None,
                expand_shorthand=False,
                detect_negation=True,
                negation_strategy="transformer",
            )

            text = "Family history of colon cancer."
            entities = [
                {"text": "colon cancer", "label": "Disease",
                 "start": 18, "end": 30},
            ]
            results = pipeline.process_with_entities(text, entities)

            assert len(results) == 1
            assert results[0].negation == "family"

    def test_hybrid_preserves_transformer_negation(self):
        """When transformer says ABSENT, rules should not override it."""
        with patch("src.clinical.assertion.AssertionClassifier") as MockCls:
            mock_clf = MagicMock()
            mock_clf.annotate_entities.return_value = [
                {"text": "fever", "label": "Symptom",
                 "start": 15, "end": 20, "negation": "negated",
                 "assertion_label": "ABSENT", "assertion_score": 0.97},
            ]
            MockCls.return_value = mock_clf

            from src.clinical.pipeline import MedicalCodingPipeline
            pipeline = MedicalCodingPipeline(
                model_path=None,
                expand_shorthand=False,
                detect_negation=True,
                negation_strategy="transformer",
            )

            text = "Patient denies fever."
            entities = [
                {"text": "fever", "label": "Symptom",
                 "start": 15, "end": 20},
            ]
            results = pipeline.process_with_entities(text, entities)

            assert len(results) == 1
            assert results[0].negation == "negated"
