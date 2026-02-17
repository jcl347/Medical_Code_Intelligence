"""
Tests for negation detection module.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.clinical.negation import NegationDetector, NegationStatus


@pytest.fixture
def detector():
    return NegationDetector(scope_window=6)


class TestPreNegation:
    """Test forward-scoping (pre-negation) triggers."""

    def test_no_keyword(self, detector):
        assert detector.is_negated("no fever", 3, 8)

    def test_denies(self, detector):
        assert detector.is_negated("patient denies chest pain", 15, 25)

    def test_denied(self, detector):
        assert detector.is_negated("denied shortness of breath", 7, 26)

    def test_without(self, detector):
        assert detector.is_negated("without headache", 8, 16)

    def test_no_evidence_of(self, detector):
        assert detector.is_negated("no evidence of pneumonia", 15, 24)

    def test_negative_for(self, detector):
        assert detector.is_negated("negative for influenza", 13, 22)

    def test_absence_of(self, detector):
        assert detector.is_negated("absence of edema", 11, 16)

    def test_no_history_of(self, detector):
        assert detector.is_negated("no history of seizures", 14, 22)

    def test_no_known(self, detector):
        assert detector.is_negated("no known allergies", 9, 18)

    def test_wo_abbreviation(self, detector):
        assert detector.is_negated("w/o nausea", 4, 10)

    def test_ruled_out(self, detector):
        assert detector.is_negated("ruled out pulmonary embolism", 10, 28)

    def test_not_keyword(self, detector):
        assert detector.is_negated("does not have diabetes", 14, 22)


class TestPostNegation:
    """Test backward-scoping (post-negation) triggers."""

    def test_not_found(self, detector):
        assert detector.is_negated("mass not found", 0, 4)

    def test_absent(self, detector):
        assert detector.is_negated("edema absent", 0, 5)

    def test_negative(self, detector):
        assert detector.is_negated("strep test negative", 0, 10)

    def test_was_negative(self, detector):
        assert detector.is_negated("flu test was negative", 0, 8)


class TestAffirmedEntities:
    """Test that affirmed entities are NOT marked as negated."""

    def test_simple_affirmed(self, detector):
        assert not detector.is_negated("patient has diabetes", 12, 20)

    def test_diagnosed_with(self, detector):
        assert not detector.is_negated("diagnosed with hypertension", 15, 27)

    def test_presents_with(self, detector):
        assert not detector.is_negated("presents with fever", 14, 19)

    def test_positive_for(self, detector):
        text = "the patient tested positive for COVID"
        # "COVID" starts at 31
        assert not detector.is_negated(text, 31, 36)


class TestPseudoNegation:
    """Test that pseudo-negation triggers are NOT treated as negation."""

    def test_no_change(self, detector):
        # "no change" is pseudo-negation, should NOT negate "tumor"
        text = "no change in the tumor"
        scopes = detector.detect(text)
        # The pseudo-negation should prevent "no change" from counting
        negated_scopes = [s for s in scopes if s.status == NegationStatus.NEGATED]
        # Even if there are negation scopes, "no change" context shouldn't
        # make "tumor" negated if pseudo-neg is handled right.
        # This is a nuanced test - "no" will trigger but "no change" should block it
        for s in negated_scopes:
            if "no change" in s.trigger_text:
                pytest.fail("'no change' should be treated as pseudo-negation")

    def test_gram_negative(self, detector):
        text = "gram negative bacteria found in culture"
        scopes = detector.detect(text)
        neg_scopes = [s for s in scopes if s.status == NegationStatus.NEGATED]
        for s in neg_scopes:
            if "gram negative" in s.trigger_text.lower() or "gram-negative" in s.trigger_text.lower():
                pytest.fail("'gram negative' should be treated as pseudo-negation")


class TestScopeTermination:
    """Test that negation scope is terminated by conjunctions and boundaries."""

    def test_but_terminates(self, detector):
        text = "no fever but has cough"
        # "fever" (3-8) should be negated
        assert detector.is_negated(text, 3, 8)
        # "cough" (16-21) should NOT be negated (after "but")
        assert not detector.is_negated(text, 16, 21)

    def test_however_terminates(self, detector):
        text = "denies chest pain however reports dyspnea"
        # "chest pain" should be negated
        assert detector.is_negated(text, 7, 17)
        # "dyspnea" should NOT be negated
        assert not detector.is_negated(text, 34, 41)

    def test_period_terminates(self, detector):
        text = "No fever. Patient has cough."
        # "fever" should be negated
        assert detector.is_negated(text, 3, 8)
        # "cough" should NOT be negated (different sentence)
        assert not detector.is_negated(text, 22, 27)


class TestAnnotateEntities:
    """Test entity annotation with negation status."""

    def test_annotate_mixed(self, detector):
        text = "Patient denies fever but has cough and sore throat."
        entities = [
            {"text": "fever", "label": "Symptom", "start": 15, "end": 20},
            {"text": "cough", "label": "Symptom", "start": 29, "end": 34},
            {"text": "sore throat", "label": "Symptom", "start": 39, "end": 50},
        ]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "negated"
        assert annotated[1]["negation"] == "affirmed"
        assert annotated[2]["negation"] == "affirmed"

    def test_annotate_adds_trigger(self, detector):
        text = "no history of diabetes"
        entities = [
            {"text": "diabetes", "label": "Disease", "start": 14, "end": 22},
        ]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "negated"
        assert "negation_trigger" in annotated[0]


class TestContextualAnnotation:
    """Test possibility, historical, and family context detection."""

    def test_possible(self, detector):
        text = "possible pneumonia"
        entities = [{"text": "pneumonia", "label": "Disease", "start": 9, "end": 18}]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "possible"

    def test_history_of(self, detector):
        text = "history of myocardial infarction"
        entities = [{"text": "myocardial infarction", "label": "Disease", "start": 11, "end": 31}]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "historical"

    def test_family_history(self, detector):
        text = "family history of breast cancer"
        entities = [{"text": "breast cancer", "label": "Disease", "start": 18, "end": 31}]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "family"

    def test_suspect(self, detector):
        text = "suspected appendicitis"
        entities = [{"text": "appendicitis", "label": "Disease", "start": 10, "end": 22}]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "possible"


class TestClinicalScenarios:
    """Test realistic clinical documentation patterns."""

    def test_review_of_systems_negatives(self, detector):
        text = "ROS: Denies fever, chills, or night sweats. Reports fatigue."
        entities = [
            {"text": "fever", "label": "Symptom", "start": 12, "end": 17},
            {"text": "chills", "label": "Symptom", "start": 19, "end": 25},
            {"text": "night sweats", "label": "Symptom", "start": 30, "end": 42},
            {"text": "fatigue", "label": "Symptom", "start": 52, "end": 59},
        ]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "negated"  # fever
        assert annotated[1]["negation"] == "negated"  # chills
        assert annotated[2]["negation"] == "negated"  # night sweats
        assert annotated[3]["negation"] == "affirmed"  # fatigue

    def test_assessment_plan(self, detector):
        text = "Assessment: No evidence of malignancy. Benign cyst noted."
        entities = [
            {"text": "malignancy", "label": "Disease", "start": 27, "end": 37},
            {"text": "cyst", "label": "Finding", "start": 46, "end": 50},
        ]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "negated"
        assert annotated[1]["negation"] == "affirmed"
