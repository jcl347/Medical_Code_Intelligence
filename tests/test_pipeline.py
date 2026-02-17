"""
Tests for the unified MedicalCodingPipeline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.clinical.pipeline import MedicalCodingPipeline, MedicalEntity
from src.clinical.negation import NegationStatus


class TestPipelineWithoutModel:
    """Test pipeline components that don't require a loaded NER model."""

    @pytest.fixture
    def pipeline(self):
        return MedicalCodingPipeline(
            model_path=None,
            expand_shorthand=True,
            detect_negation=True,
        )

    def test_process_with_entities_negation(self, pipeline):
        text = "Patient denies chest pain"
        entities = [
            {"text": "chest pain", "label": "Symptom", "start": 15, "end": 25, "score": 0.95},
        ]
        results = pipeline.process_with_entities(text, entities)
        assert len(results) == 1
        assert results[0].negation == "negated"
        assert isinstance(results[0], MedicalEntity)

    def test_process_with_entities_affirmed(self, pipeline):
        text = "Patient has diabetes"
        entities = [
            {"text": "diabetes", "label": "Disease", "start": 12, "end": 20, "score": 0.99},
        ]
        results = pipeline.process_with_entities(text, entities)
        assert results[0].negation == "affirmed"
        assert results[0].is_affirmed

    def test_process_with_entities_shorthand(self, pipeline):
        text = "pt with htn"
        entities = [
            {"text": "hypertension", "label": "Disease", "start": 13, "end": 25, "score": 0.9},
        ]
        results = pipeline.process_with_entities(text, entities)
        assert len(results) == 1

    def test_no_model_raises_on_process(self, pipeline):
        with pytest.raises(RuntimeError, match="No model_path"):
            pipeline.process("some text")

    def test_empty_entities(self, pipeline):
        results = pipeline.process_with_entities("some text", [])
        assert results == []


class TestMedicalEntity:
    """Test MedicalEntity data class."""

    def test_to_dict(self):
        entity = MedicalEntity(
            text="diabetes",
            label="Disease",
            start_char=10,
            end_char=18,
            score=0.95,
            negation="affirmed",
        )
        d = entity.to_dict()
        assert d["text"] == "diabetes"
        assert d["label"] == "Disease"
        assert d["negation"] == "affirmed"
        assert d["score"] == 0.95

    def test_to_dict_with_expansion(self):
        entity = MedicalEntity(
            text="hypertension",
            label="Disease",
            start_char=10,
            end_char=22,
            score=0.9,
            negation="affirmed",
            expanded_from="htn",
            original_text="htn",
        )
        d = entity.to_dict()
        assert d["expanded_from"] == "htn"
        assert d["original_text"] == "htn"

    def test_to_dict_with_negation_trigger(self):
        entity = MedicalEntity(
            text="fever",
            label="Symptom",
            start_char=3,
            end_char=8,
            score=0.97,
            negation="negated",
            negation_trigger="no",
        )
        d = entity.to_dict()
        assert d["negation"] == "negated"
        assert d["negation_trigger"] == "no"

    def test_is_negated(self):
        entity = MedicalEntity(
            text="fever", label="Symptom", start_char=0, end_char=5,
            score=0.9, negation="negated",
        )
        assert entity.is_negated
        assert not entity.is_affirmed

    def test_is_affirmed(self):
        entity = MedicalEntity(
            text="fever", label="Symptom", start_char=0, end_char=5,
            score=0.9, negation="affirmed",
        )
        assert entity.is_affirmed
        assert not entity.is_negated


class TestPipelineFormatOutput:
    """Test the format_output display method."""

    def test_format_with_entities(self):
        pipeline = MedicalCodingPipeline(
            model_path=None, expand_shorthand=False, detect_negation=False,
        )
        text = "Patient has diabetes"
        entities = [
            MedicalEntity(
                text="diabetes", label="Disease", start_char=12, end_char=20,
                score=0.95, negation="affirmed",
            ),
        ]
        output = pipeline.format_output(text, entities)
        assert "diabetes" in output
        assert "Disease" in output
        assert "AFFIRMED" in output

    def test_format_no_entities(self):
        pipeline = MedicalCodingPipeline(
            model_path=None, expand_shorthand=False, detect_negation=False,
        )
        output = pipeline.format_output("Normal exam", [])
        assert "no entities found" in output


class TestPipelineDisableFeatures:
    """Test that features can be individually disabled."""

    def test_no_shorthand(self):
        pipeline = MedicalCodingPipeline(
            model_path=None, expand_shorthand=False, detect_negation=True,
        )
        assert pipeline.shorthand_expander is None

    def test_no_negation(self):
        pipeline = MedicalCodingPipeline(
            model_path=None, expand_shorthand=True, detect_negation=False,
        )
        assert pipeline.negation_detector is None

    def test_disabled_negation_returns_affirmed(self):
        pipeline = MedicalCodingPipeline(
            model_path=None, expand_shorthand=False, detect_negation=False,
        )
        entities = [
            {"text": "fever", "label": "Symptom", "start": 3, "end": 8, "score": 0.9},
        ]
        results = pipeline.process_with_entities("no fever", entities)
        # Without negation detection, everything is affirmed
        assert results[0].negation == "affirmed"


class TestEndToEndClinicalScenarios:
    """Integration tests for realistic clinical scenarios using process_with_entities."""

    @pytest.fixture
    def pipeline(self):
        return MedicalCodingPipeline(
            model_path=None, expand_shorthand=True, detect_negation=True,
        )

    def test_mixed_negated_and_affirmed(self, pipeline):
        text = "Patient denies fever but complains of cough."
        entities = [
            {"text": "fever", "label": "Symptom", "start": 15, "end": 20, "score": 0.9},
            {"text": "cough", "label": "Symptom", "start": 38, "end": 43, "score": 0.9},
        ]
        results = pipeline.process_with_entities(text, entities)
        assert results[0].is_negated  # fever
        assert results[1].is_affirmed  # cough

    def test_historical_condition(self, pipeline):
        text = "History of myocardial infarction in 2019."
        entities = [
            {"text": "myocardial infarction", "label": "Disease", "start": 11, "end": 31, "score": 0.95},
        ]
        results = pipeline.process_with_entities(text, entities)
        assert results[0].negation == "historical"

    def test_family_history(self, pipeline):
        text = "Family history of colon cancer."
        entities = [
            {"text": "colon cancer", "label": "Disease", "start": 18, "end": 30, "score": 0.92},
        ]
        results = pipeline.process_with_entities(text, entities)
        assert results[0].negation == "family"
