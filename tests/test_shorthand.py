"""
Tests for physician shorthand / abbreviation expansion.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.clinical.shorthand import ShorthandExpander


@pytest.fixture
def expander():
    return ShorthandExpander()


class TestBasicExpansion:
    """Test that common abbreviations expand correctly."""

    def test_common_diagnoses(self, expander):
        assert "hypertension" in expander.expand("pt with htn")
        assert "diabetes mellitus" in expander.expand("hx of dm")
        assert "congestive heart failure" in expander.expand("dx: chf")
        assert "coronary artery disease" in expander.expand("known cad")

    def test_type_specific_diabetes(self, expander):
        result = expander.expand("dm2 controlled on metformin")
        assert "type 2 diabetes mellitus" in result

    def test_symptoms(self, expander):
        assert "shortness of breath" in expander.expand("pt c/o sob")
        assert "chest pain" in expander.expand("denies cp")
        assert "nausea/vomiting" in expander.expand("reports n/v")

    def test_treatment_terms(self, expander):
        assert "antibiotics" in expander.expand("started on abx")
        assert "intravenous" in expander.expand("iv fluids")
        assert "as needed" in expander.expand("tylenol prn")

    def test_physical_exam(self, expander):
        assert "clear to auscultation bilaterally" in expander.expand("lungs ctab")
        assert "regular rate and rhythm" in expander.expand("heart rrr")
        assert "nontender nondistended" in expander.expand("abd ntnd")

    def test_anatomy(self, expander):
        assert "abdomen" in expander.expand("abd soft")
        assert "bilateral" in expander.expand("bilat le edema")
        assert "right upper quadrant" in expander.expand("ruq tenderness")

    def test_frequency_terms(self, expander):
        assert "twice daily" in expander.expand("metformin 500mg bid")
        assert "three times daily" in expander.expand("amoxicillin tid")
        assert "daily" in expander.expand("aspirin 81mg qd")
        assert "at bedtime" in expander.expand("melatonin qhs")

    def test_lab_terms(self, expander):
        assert "complete blood count" in expander.expand("check cbc")
        assert "basic metabolic panel" in expander.expand("bmp normal")
        assert "hemoglobin a1c" in expander.expand("hba1c 7.2")

    def test_imaging_terms(self, expander):
        assert "chest x-ray" in expander.expand("cxr unremarkable")
        assert "computed tomography" in expander.expand("ct abdomen")
        assert "magnetic resonance imaging" in expander.expand("mri brain")


class TestCaseInsensitivity:
    """Abbreviations should match regardless of case."""

    def test_uppercase(self, expander):
        assert "hypertension" in expander.expand("HTN")

    def test_mixed_case(self, expander):
        assert "diabetes mellitus" in expander.expand("Dm")

    def test_lowercase(self, expander):
        assert "coronary artery disease" in expander.expand("cad")


class TestWordBoundaries:
    """Abbreviations should only match as whole words."""

    def test_not_in_middle_of_word(self, expander):
        # "treatment" contains "tx" but should not be expanded
        result = expander.expand("treatment plan")
        assert result == "treatment plan"

    def test_not_partial_match(self, expander):
        # "caution" starts with "ca" but should not be expanded
        result = expander.expand("caution advised")
        assert "calcium" not in result


class TestOffsetTracking:
    """Test that character offsets are correctly tracked through expansion."""

    def test_offset_map_produced(self, expander):
        expanded, offset_map = expander.expand_with_offsets("pt with htn")
        assert len(offset_map) >= 2  # "pt" and "htn"

    def test_offset_positions(self, expander):
        expanded, offset_map = expander.expand_with_offsets("htn present")
        assert offset_map[0]["abbreviation"].lower() == "htn"
        assert offset_map[0]["expansion"] == "hypertension"

    def test_expanded_text_matches(self, expander):
        text = "dx: dm2"
        expanded, offset_map = expander.expand_with_offsets(text)
        for om in offset_map:
            actual = expanded[om["expanded_start"]:om["expanded_end"]]
            assert actual == om["expansion"]


class TestIdentifyWithoutExpand:
    """Test abbreviation identification without expanding."""

    def test_identify_returns_matches(self, expander):
        found = expander.identify_abbreviations("pt c/o sob and cp")
        abbrs = [f["abbreviation"].lower() for f in found]
        assert "sob" in abbrs
        assert "cp" in abbrs


class TestCustomAbbreviations:
    """Test adding custom abbreviations."""

    def test_custom_added(self):
        custom = {"myabbr": "my custom expansion"}
        exp = ShorthandExpander(custom_abbreviations=custom)
        assert "my custom expansion" in exp.expand("check myabbr")

    def test_custom_override(self):
        custom = {"htn": "HIGH BLOOD PRESSURE"}
        exp = ShorthandExpander(custom_abbreviations=custom)
        assert "HIGH BLOOD PRESSURE" in exp.expand("dx: htn")


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_empty_string(self, expander):
        assert expander.expand("") == ""

    def test_no_abbreviations(self, expander):
        text = "The patient was evaluated in the clinic today."
        assert expander.expand(text) == text

    def test_multiple_same_abbreviation(self, expander):
        result = expander.expand("htn and htn")
        assert result.count("hypertension") == 2

    def test_abbreviation_with_punctuation(self, expander):
        result = expander.expand("dx: htn, dm2, and cad.")
        assert "hypertension" in result
        assert "type 2 diabetes mellitus" in result
        assert "coronary artery disease" in result
