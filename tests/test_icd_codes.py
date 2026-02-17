"""
Tests for ICD code lookup and entity-to-code mapping.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.clinical.icd_codes import ICDCodeLookup, ICDCode, ICDMatch


@pytest.fixture
def lookup():
    return ICDCodeLookup(load_from_hf=False)


# ---------------------------------------------------------------------------
# Code lookup
# ---------------------------------------------------------------------------

class TestCodeLookup:
    """Test direct code → description lookup."""

    def test_lookup_known_code(self, lookup):
        result = lookup.lookup_code("I10")
        assert result is not None
        assert isinstance(result, ICDCode)
        assert result.code == "I10"
        assert "hypertension" in result.description.lower()

    def test_lookup_diabetes(self, lookup):
        result = lookup.lookup_code("E11.9")
        assert result is not None
        assert "type 2 diabetes" in result.description.lower()

    def test_lookup_unknown_code(self, lookup):
        result = lookup.lookup_code("ZZZ.999")
        assert result is None

    def test_lookup_pneumonia(self, lookup):
        result = lookup.lookup_code("J18.9")
        assert result is not None
        assert "pneumonia" in result.description.lower()

    def test_lookup_ckd(self, lookup):
        result = lookup.lookup_code("N18.3")
        assert result is not None
        assert "chronic kidney disease" in result.description.lower()
        assert "stage 3" in result.description.lower()

    def test_lookup_chest_pain(self, lookup):
        result = lookup.lookup_code("R07.9")
        assert result is not None
        assert "chest pain" in result.description.lower()


class TestChapterDetection:
    """Test ICD-10-CM chapter detection from code prefix."""

    def test_chapter_i(self, lookup):
        result = lookup.lookup_code("I10")
        assert result.chapter == "Circulatory system"

    def test_chapter_e(self, lookup):
        result = lookup.lookup_code("E11.9")
        assert result.chapter == "Endocrine/metabolic"

    def test_chapter_j(self, lookup):
        result = lookup.lookup_code("J18.9")
        assert result.chapter == "Respiratory system"

    def test_chapter_r(self, lookup):
        result = lookup.lookup_code("R07.9")
        assert result.chapter == "Symptoms/signs"

    def test_chapter_c(self, lookup):
        result = lookup.lookup_code("C34.90")
        assert result.chapter == "Neoplasms"

    def test_chapter_z(self, lookup):
        result = lookup.lookup_code("Z23")
        assert result.chapter == "Factors influencing health status"


# ---------------------------------------------------------------------------
# Entity text → ICD code matching
# ---------------------------------------------------------------------------

class TestExactEntityMatching:
    """Test exact entity text → ICD code mapping for common conditions."""

    def test_hypertension(self, lookup):
        matches = lookup.match_entity("hypertension")
        assert len(matches) >= 1
        assert matches[0].code == "I10"
        assert matches[0].match_type == "exact"
        assert matches[0].score == 1.0

    def test_type_2_diabetes(self, lookup):
        matches = lookup.match_entity("type 2 diabetes mellitus")
        assert len(matches) >= 1
        assert matches[0].code == "E11.9"

    def test_congestive_heart_failure(self, lookup):
        matches = lookup.match_entity("congestive heart failure")
        assert len(matches) >= 1
        assert matches[0].code == "I50.9"

    def test_pneumonia(self, lookup):
        matches = lookup.match_entity("pneumonia")
        assert len(matches) >= 1
        assert matches[0].code == "J18.9"

    def test_chest_pain(self, lookup):
        matches = lookup.match_entity("chest pain")
        assert len(matches) >= 1
        assert matches[0].code == "R07.9"

    def test_atrial_fibrillation(self, lookup):
        matches = lookup.match_entity("atrial fibrillation")
        assert len(matches) >= 1
        assert matches[0].code == "I48.91"

    def test_copd(self, lookup):
        matches = lookup.match_entity("chronic obstructive pulmonary disease")
        assert len(matches) >= 1
        assert matches[0].code == "J44.9"

    def test_uti(self, lookup):
        matches = lookup.match_entity("urinary tract infection")
        assert len(matches) >= 1
        assert matches[0].code == "N39.0"

    def test_acute_kidney_injury(self, lookup):
        matches = lookup.match_entity("acute kidney injury")
        assert len(matches) >= 1
        assert matches[0].code == "N17.9"

    def test_depression(self, lookup):
        matches = lookup.match_entity("depression")
        assert len(matches) >= 1
        assert matches[0].code == "F32.9"

    def test_sepsis(self, lookup):
        matches = lookup.match_entity("sepsis")
        assert len(matches) >= 1
        assert matches[0].code == "A41.9"

    def test_stroke(self, lookup):
        matches = lookup.match_entity("stroke")
        assert len(matches) >= 1
        assert matches[0].code == "I63.9"

    def test_fever(self, lookup):
        matches = lookup.match_entity("fever")
        assert len(matches) >= 1
        assert matches[0].code == "R50.9"

    def test_nausea(self, lookup):
        matches = lookup.match_entity("nausea")
        assert len(matches) >= 1
        assert matches[0].code == "R11.0"

    def test_lung_cancer(self, lookup):
        matches = lookup.match_entity("lung cancer")
        assert len(matches) >= 1
        assert matches[0].code == "C34.90"


class TestFuzzyMatching:
    """Test token-overlap fuzzy matching for entities not in exact map."""

    def test_fuzzy_returns_candidates(self, lookup):
        # "acute pancreatitis" is not an exact match but should fuzzy-match
        matches = lookup.match_entity("acute pancreatitis")
        assert len(matches) >= 1
        # Should find K85.90 via token overlap
        codes = [m.code for m in matches]
        assert "K85.90" in codes

    def test_fuzzy_score_below_one(self, lookup):
        matches = lookup.match_entity("chronic systolic heart failure")
        # May find I50.22 via overlap
        for m in matches:
            if m.match_type == "token_overlap":
                assert m.score < 1.0
                assert m.score >= 0.3

    def test_no_match_for_garbage(self, lookup):
        matches = lookup.match_entity("xyzzy foobar baz")
        assert len(matches) == 0

    def test_top_k_limit(self, lookup):
        matches = lookup.match_entity("kidney disease", top_k=3)
        assert len(matches) <= 3

    def test_min_score_filter(self, lookup):
        matches = lookup.match_entity("heart", min_score=0.5)
        for m in matches:
            assert m.score >= 0.5


class TestBatchMatching:
    """Test batch entity → code matching."""

    def test_batch_multiple_entities(self, lookup):
        entities = [
            {"text": "hypertension", "label": "Disease"},
            {"text": "diabetes mellitus", "label": "Disease"},
            {"text": "chest pain", "label": "Symptom"},
        ]
        results = lookup.match_entities_batch(entities, top_k=3)
        assert len(results) == 3
        for r in results:
            assert "icd_codes" in r
            assert len(r["icd_codes"]) >= 1

    def test_batch_preserves_original_fields(self, lookup):
        entities = [
            {"text": "hypertension", "label": "Disease", "score": 0.95},
        ]
        results = lookup.match_entities_batch(entities)
        assert results[0]["text"] == "hypertension"
        assert results[0]["label"] == "Disease"
        assert results[0]["score"] == 0.95

    def test_batch_empty_input(self, lookup):
        results = lookup.match_entities_batch([])
        assert results == []


class TestICDDataClasses:
    """Test ICDCode and ICDMatch data classes."""

    def test_icd_code_to_dict(self):
        code = ICDCode(
            code="I10",
            description="Essential (primary) hypertension",
            is_billable=True,
            chapter="Circulatory system",
        )
        d = code.to_dict()
        assert d["code"] == "I10"
        assert d["description"] == "Essential (primary) hypertension"
        assert d["is_billable"] is True
        assert d["chapter"] == "Circulatory system"

    def test_icd_match_to_dict(self):
        match = ICDMatch(
            code="E11.9",
            description="Type 2 diabetes mellitus without complications",
            score=1.0,
            match_type="exact",
        )
        d = match.to_dict()
        assert d["code"] == "E11.9"
        assert d["score"] == 1.0
        assert d["match_type"] == "exact"


class TestCodeCoverage:
    """Test that the knowledge base has adequate coverage of common conditions."""

    def test_cardiovascular_codes_present(self, lookup):
        cv_codes = ["I10", "I48.91", "I50.9", "I25.10", "I21.3", "I63.9"]
        for code in cv_codes:
            assert lookup.lookup_code(code) is not None, f"Missing code: {code}"

    def test_respiratory_codes_present(self, lookup):
        resp_codes = ["J18.9", "J44.9", "J96.00", "J80"]
        for code in resp_codes:
            assert lookup.lookup_code(code) is not None, f"Missing code: {code}"

    def test_endocrine_codes_present(self, lookup):
        endo_codes = ["E10.9", "E11.9", "E78.5", "E86.0"]
        for code in endo_codes:
            assert lookup.lookup_code(code) is not None, f"Missing code: {code}"

    def test_renal_codes_present(self, lookup):
        renal_codes = ["N17.9", "N18.3", "N18.6", "N39.0"]
        for code in renal_codes:
            assert lookup.lookup_code(code) is not None, f"Missing code: {code}"

    def test_symptom_codes_present(self, lookup):
        symptom_codes = ["R07.9", "R06.02", "R50.9", "R11.0", "R55"]
        for code in symptom_codes:
            assert lookup.lookup_code(code) is not None, f"Missing code: {code}"

    def test_mental_health_codes_present(self, lookup):
        mh_codes = ["F32.9", "F41.9", "F10.20"]
        for code in mh_codes:
            assert lookup.lookup_code(code) is not None, f"Missing code: {code}"

    def test_total_code_count(self, lookup):
        """Knowledge base should have 150+ curated codes."""
        assert len(lookup.code_to_description) >= 150

    def test_total_entity_mapping_count(self, lookup):
        """Entity-to-code mapping should cover 100+ common conditions."""
        assert len(lookup.entity_to_codes) >= 100
