"""
Tests for ICD-specific pipeline integration.

Tests the full flow: shorthand expansion → NER (simulated) → negation → ICD coding.
Uses the fallback code set for deterministic CI behavior.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch
from src.clinical.icd_codes import ICDCodeLookup
from src.clinical.negation import NegationDetector
from src.clinical.shorthand import ShorthandExpander
from src.clinical.pipeline import MedicalCodingPipeline, MedicalEntity


@pytest.fixture
def lookup():
    """ICDCodeLookup with fallback codes (no HF download)."""
    with patch("src.clinical.icd_codes.ICDCodeLookup._load_codes") as mock_load:
        mock_load.return_value = None
        obj = ICDCodeLookup.__new__(ICDCodeLookup)
        obj.top_k = 5
        obj._ngram_range = (3, 4)
        obj._codes = {}
        obj._descriptions = []
        obj._code_keys = []
        obj._tfidf_matrix = None
        obj._vectoriser = None
    obj._load_builtin_fallback()
    obj._build_tfidf_index()
    return obj


@pytest.fixture
def detector():
    return NegationDetector()


@pytest.fixture
def expander():
    return ShorthandExpander()


# ---------------------------------------------------------------------------
# Shorthand → ICD integration
# ---------------------------------------------------------------------------

class TestShorthandToICD:
    """Test that physician shorthand expands to text that maps to ICD codes."""

    def test_htn_maps_to_i10(self, expander, lookup):
        expanded = expander.expand("pt with htn")
        assert "hypertension" in expanded
        matches = lookup.match_entity("hypertension")
        codes = [m.code for m in matches]
        assert "I10" in codes

    def test_dm2_maps_to_e11(self, expander, lookup):
        expanded = expander.expand("hx of dm2")
        assert "type 2 diabetes mellitus" in expanded
        matches = lookup.match_entity("type 2 diabetes mellitus")
        codes = [m.code for m in matches]
        assert "E11.9" in codes

    def test_chf_maps_to_i50(self, expander, lookup):
        expanded = expander.expand("dx: chf")
        assert "congestive heart failure" in expanded
        matches = lookup.match_entity("congestive heart failure")
        codes = [m.code for m in matches]
        assert "I50.9" in codes

    def test_copd_maps_to_j44(self, expander, lookup):
        expanded = expander.expand("copd exacerbation")
        assert "chronic obstructive pulmonary disease" in expanded
        matches = lookup.match_entity("chronic obstructive pulmonary disease")
        codes = [m.code for m in matches]
        assert any(c.startswith("J44") for c in codes)

    def test_uti_maps_to_n39(self, expander, lookup):
        expanded = expander.expand("treated for uti")
        assert "urinary tract infection" in expanded
        matches = lookup.match_entity("urinary tract infection")
        codes = [m.code for m in matches]
        assert "N39.0" in codes

    def test_sob_maps_to_r06(self, expander, lookup):
        expanded = expander.expand("c/o sob")
        assert "shortness of breath" in expanded
        matches = lookup.match_entity("shortness of breath")
        codes = [m.code for m in matches]
        assert "R06.02" in codes

    def test_afib_maps_to_i48(self, expander, lookup):
        expanded = expander.expand("new onset afib")
        assert "atrial fibrillation" in expanded
        matches = lookup.match_entity("atrial fibrillation")
        codes = [m.code for m in matches]
        assert "I48.91" in codes


# ---------------------------------------------------------------------------
# Negation + ICD integration
# ---------------------------------------------------------------------------

class TestNegationWithICD:
    """Test that negated entities are correctly flagged before ICD coding."""

    def test_negated_entity_still_gets_icd(self, detector, lookup):
        """Negated findings should still get ICD codes but be marked negated."""
        text = "Patient denies chest pain"
        entities = [
            {"text": "chest pain", "label": "Symptom", "start": 15, "end": 25},
        ]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "negated"

        # ICD lookup still works on negated entity text
        matches = lookup.match_entity(annotated[0]["text"])
        codes = [m.code for m in matches]
        assert "R07.9" in codes

    def test_affirmed_entity_gets_icd(self, detector, lookup):
        text = "Patient presents with shortness of breath"
        entities = [
            {"text": "shortness of breath", "label": "Symptom", "start": 22, "end": 41},
        ]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "affirmed"
        matches = lookup.match_entity(annotated[0]["text"])
        codes = [m.code for m in matches]
        assert "R06.02" in codes

    def test_historical_entity_gets_icd(self, detector, lookup):
        text = "History of myocardial infarction"
        entities = [
            {"text": "myocardial infarction", "label": "Disease", "start": 11, "end": 31},
        ]
        annotated = detector.annotate_entities(text, entities)
        assert annotated[0]["negation"] == "historical"
        matches = lookup.match_entity(annotated[0]["text"])
        # Should find some I21.x code via TF-IDF
        assert len(matches) >= 1


# ---------------------------------------------------------------------------
# Full pipeline → ICD integration
# ---------------------------------------------------------------------------

class TestPipelineToICD:
    """Test the MedicalCodingPipeline output feeds correctly into ICDCodeLookup."""

    def test_pipeline_entities_map_to_icd(self, lookup):
        pipeline = MedicalCodingPipeline(
            model_path=None, expand_shorthand=True, detect_negation=True,
            negation_strategy="rules",
        )
        # Simulate entities extracted by the NER model
        text = "Pt with htn and dm2, denies cp."
        entities = [
            {"text": "hypertension", "label": "Disease", "start": 8, "end": 20, "score": 0.95},
            {"text": "type 2 diabetes mellitus", "label": "Disease", "start": 25, "end": 49, "score": 0.92},
            {"text": "chest pain", "label": "Symptom", "start": 58, "end": 68, "score": 0.88},
        ]
        results = pipeline.process_with_entities(text, entities)

        # Map each entity to ICD codes
        for entity in results:
            matches = lookup.match_entity(entity.text)
            if entity.text == "hypertension":
                codes = [m.code for m in matches]
                assert "I10" in codes
                assert entity.is_affirmed
            elif entity.text == "type 2 diabetes mellitus":
                codes = [m.code for m in matches]
                assert "E11.9" in codes
                assert entity.is_affirmed

    def test_medical_entity_to_dict_with_icd(self, lookup):
        entity = MedicalEntity(
            text="hypertension", label="Disease",
            start_char=10, end_char=22, score=0.95,
            negation="affirmed",
        )
        matches = lookup.match_entity(entity.text)
        d = entity.to_dict()
        d["icd_codes"] = [m.to_dict() for m in matches]
        assert any(c["code"] == "I10" for c in d["icd_codes"])
        assert d["negation"] == "affirmed"


# ---------------------------------------------------------------------------
# Clinical scenario end-to-end
# ---------------------------------------------------------------------------

class TestClinicalScenarioICD:
    """Realistic clinical scenarios testing the full shorthand→negation→ICD flow."""

    def test_admission_note(self, expander, detector, lookup):
        raw = "72 yo M with htn, dm2, cad s/p cabg presents with sob and cp."
        expanded = expander.expand(raw)

        assert "hypertension" in expanded
        assert "type 2 diabetes mellitus" in expanded
        assert "coronary artery disease" in expanded
        assert "shortness of breath" in expanded
        assert "chest pain" in expanded

        conditions = [
            "hypertension", "type 2 diabetes mellitus",
            "shortness of breath", "chest pain",
        ]
        for condition in conditions:
            matches = lookup.match_entity(condition)
            assert len(matches) >= 1, f"No ICD match for: {condition}"

    def test_discharge_summary(self, expander, detector, lookup):
        raw = "Pt admitted with pna, r/o pe. Hx of chf and afib. No dvt on ultrasound."
        expanded = expander.expand(raw)

        assert "pneumonia" in expanded
        assert "congestive heart failure" in expanded
        assert "atrial fibrillation" in expanded

        matches = lookup.match_entity("pneumonia")
        codes = [m.code for m in matches]
        assert "J18.9" in codes

    def test_ros_with_negations(self, expander, detector, lookup):
        text = "Denies fever, chills, or nausea. Reports persistent cough and sob."

        entities = [
            {"text": "fever", "label": "Symptom", "start": 7, "end": 12},
            {"text": "chills", "label": "Symptom", "start": 14, "end": 20},
            {"text": "nausea", "label": "Symptom", "start": 25, "end": 31},
            {"text": "cough", "label": "Symptom", "start": 51, "end": 56},
        ]
        annotated = detector.annotate_entities(text, entities)

        assert annotated[0]["negation"] == "negated"   # fever
        assert annotated[1]["negation"] == "negated"   # chills
        assert annotated[2]["negation"] == "negated"   # nausea
        assert annotated[3]["negation"] == "affirmed"  # cough

        for ent in annotated:
            matches = lookup.match_entity(ent["text"])
            assert len(matches) >= 1, f"No ICD match for: {ent['text']}"
