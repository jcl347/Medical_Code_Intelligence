"""
Tests for data-driven ICD-10-CM code lookup and TF-IDF entity linking.

These tests use the built-in fallback codes (no HuggingFace download needed)
to ensure deterministic behavior in CI environments.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch
from src.clinical.icd_codes import ICDCodeLookup, ICDCode, ICDMatch


@pytest.fixture
def lookup():
    """Create an ICDCodeLookup that always uses fallback codes."""
    with patch("src.clinical.icd_codes.ICDCodeLookup._load_codes") as mock_load:
        # Skip the HF loading; we'll load fallback manually after construction
        mock_load.return_value = None
        obj = ICDCodeLookup.__new__(ICDCodeLookup)
        obj.top_k = 5
        obj._ngram_range = (3, 4)
        obj._codes = {}
        obj._descriptions = []
        obj._code_keys = []
        obj._tfidf_matrix = None
        obj._vectoriser = None
    # Load fallback codes and build index
    obj._load_builtin_fallback()
    obj._build_tfidf_index()
    return obj


# ---------------------------------------------------------------------------
# Code lookup by code string
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

    def test_lookup_chest_pain(self, lookup):
        result = lookup.lookup_code("R07.9")
        assert result is not None
        assert "chest pain" in result.description.lower()

    def test_lookup_sepsis(self, lookup):
        result = lookup.lookup_code("A41.9")
        assert result is not None
        assert "sepsis" in result.description.lower()

    def test_lookup_heart_failure(self, lookup):
        result = lookup.lookup_code("I50.9")
        assert result is not None
        assert "heart failure" in result.description.lower()


# ---------------------------------------------------------------------------
# TF-IDF entity matching
# ---------------------------------------------------------------------------

class TestTFIDFMatching:
    """Test TF-IDF character n-gram entity→ICD matching."""

    def test_exact_description_match(self, lookup):
        """Exact description text should match with score 1.0."""
        result = lookup.lookup_code("R50.9")
        assert result is not None
        matches = lookup.match_entity(result.description)
        assert len(matches) >= 1
        assert matches[0].code == "R50.9"
        assert matches[0].score == 1.0
        assert matches[0].match_type == "exact"

    def test_hypertension_match(self, lookup):
        matches = lookup.match_entity("hypertension")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "I10" in codes

    def test_diabetes_match(self, lookup):
        matches = lookup.match_entity("type 2 diabetes mellitus")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "E11.9" in codes

    def test_pneumonia_match(self, lookup):
        matches = lookup.match_entity("pneumonia")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "J18.9" in codes

    def test_chest_pain_match(self, lookup):
        matches = lookup.match_entity("chest pain")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "R07.9" in codes

    def test_heart_failure_match(self, lookup):
        matches = lookup.match_entity("heart failure")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "I50.9" in codes

    def test_copd_match(self, lookup):
        matches = lookup.match_entity("chronic obstructive pulmonary disease")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert any(c.startswith("J44") for c in codes)

    def test_urinary_tract_infection_match(self, lookup):
        matches = lookup.match_entity("urinary tract infection")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "N39.0" in codes

    def test_depression_match(self, lookup):
        matches = lookup.match_entity("depression")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "F32.9" in codes

    def test_shortness_of_breath_match(self, lookup):
        matches = lookup.match_entity("shortness of breath")
        assert len(matches) >= 1
        codes = [m.code for m in matches]
        assert "R06.02" in codes

    def test_tfidf_score_between_0_and_1(self, lookup):
        """TF-IDF matches should have scores between 0 and 1."""
        matches = lookup.match_entity("kidney disease")
        for m in matches:
            assert 0 < m.score <= 1.0

    def test_tfidf_match_type(self, lookup):
        """Non-exact matches should be type 'tfidf'."""
        matches = lookup.match_entity("kidney problems")
        for m in matches:
            if m.score < 1.0:
                assert m.match_type == "tfidf"


class TestMatchFiltering:
    """Test top_k and min_score filtering."""

    def test_top_k_limit(self, lookup):
        matches = lookup.match_entity("disease", top_k=3)
        assert len(matches) <= 3

    def test_min_score_filter(self, lookup):
        matches = lookup.match_entity("heart", min_score=0.3)
        for m in matches:
            assert m.score >= 0.3

    def test_high_min_score_fewer_results(self, lookup):
        matches_low = lookup.match_entity("respiratory failure", min_score=0.1)
        matches_high = lookup.match_entity("respiratory failure", min_score=0.5)
        assert len(matches_high) <= len(matches_low)

    def test_no_match_for_garbage(self, lookup):
        matches = lookup.match_entity("xyzzy foobar baz")
        assert len(matches) == 0

    def test_empty_string_returns_empty(self, lookup):
        assert lookup.match_entity("") == []
        assert lookup.match_entity("   ") == []

    def test_none_like_empty(self, lookup):
        assert lookup.match_entity("") == []


# ---------------------------------------------------------------------------
# Batch matching
# ---------------------------------------------------------------------------

class TestBatchMatching:
    """Test batch entity → code matching."""

    def test_batch_multiple_entities(self, lookup):
        entities = [
            {"text": "hypertension", "label": "Disease"},
            {"text": "pneumonia", "label": "Disease"},
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

    def test_batch_icd_codes_have_required_keys(self, lookup):
        entities = [{"text": "sepsis", "label": "Disease"}]
        results = lookup.match_entities_batch(entities, top_k=2)
        for icd in results[0]["icd_codes"]:
            assert "code" in icd
            assert "description" in icd
            assert "score" in icd
            assert "match_type" in icd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class TestICDDataClasses:
    """Test ICDCode and ICDMatch data classes."""

    def test_icd_code_to_dict_basic(self):
        code = ICDCode(
            code="I10",
            description="Essential (primary) hypertension",
        )
        d = code.to_dict()
        assert d["code"] == "I10"
        assert d["description"] == "Essential (primary) hypertension"

    def test_icd_code_to_dict_with_hierarchy(self):
        code = ICDCode(
            code="I10",
            description="Essential (primary) hypertension",
            chapter="Diseases of the circulatory system",
            section="Hypertensive diseases",
            category="Essential (primary) hypertension",
        )
        d = code.to_dict()
        assert d["chapter"] == "Diseases of the circulatory system"
        assert d["section"] == "Hypertensive diseases"
        assert d["category"] == "Essential (primary) hypertension"

    def test_icd_code_to_dict_omits_none_fields(self):
        code = ICDCode(code="I10", description="Hypertension")
        d = code.to_dict()
        assert "chapter" not in d
        assert "section" not in d
        assert "category" not in d

    def test_icd_match_to_dict(self):
        match = ICDMatch(
            code="E11.9",
            description="Type 2 diabetes mellitus without complications",
            score=0.85,
            match_type="tfidf",
        )
        d = match.to_dict()
        assert d["code"] == "E11.9"
        assert d["score"] == 0.85
        assert d["match_type"] == "tfidf"

    def test_icd_match_score_rounded(self):
        match = ICDMatch(code="X", description="X", score=0.123456, match_type="tfidf")
        d = match.to_dict()
        assert d["score"] == 0.1235


# ---------------------------------------------------------------------------
# Code coverage (fallback set)
# ---------------------------------------------------------------------------

class TestFallbackCodeCoverage:
    """Test that the fallback code set covers essential conditions."""

    def test_cardiovascular_codes(self, lookup):
        for code in ["I10", "I48.91", "I50.9", "I25.10", "I21.3", "I63.9"]:
            assert lookup.lookup_code(code) is not None, f"Missing: {code}"

    def test_respiratory_codes(self, lookup):
        for code in ["J18.9", "J44.9", "J96.00", "J80"]:
            assert lookup.lookup_code(code) is not None, f"Missing: {code}"

    def test_endocrine_codes(self, lookup):
        for code in ["E10.9", "E11.9", "E78.5", "E86.0"]:
            assert lookup.lookup_code(code) is not None, f"Missing: {code}"

    def test_renal_codes(self, lookup):
        for code in ["N17.9", "N18.6", "N39.0"]:
            assert lookup.lookup_code(code) is not None, f"Missing: {code}"

    def test_symptom_codes(self, lookup):
        for code in ["R07.9", "R06.02", "R50.9", "R11.0", "R55"]:
            assert lookup.lookup_code(code) is not None, f"Missing: {code}"

    def test_mental_health_codes(self, lookup):
        for code in ["F32.9", "F41.9"]:
            assert lookup.lookup_code(code) is not None, f"Missing: {code}"

    def test_num_codes_property(self, lookup):
        """Fallback should have 30+ codes."""
        assert lookup.num_codes >= 30

    def test_tfidf_index_built(self, lookup):
        """TF-IDF matrix dimensions should match code count."""
        assert lookup._tfidf_matrix is not None
        assert lookup._tfidf_matrix.shape[0] == lookup.num_codes
