"""
Tests for data-driven physician shorthand / abbreviation expansion.

Tests use source="builtin" to ensure deterministic behavior in CI
(no Zenodo download required).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.clinical.shorthand import ShorthandExpander


@pytest.fixture
def expander():
    """Create expander with built-in abbreviations (no download)."""
    return ShorthandExpander(source="builtin")


# ---------------------------------------------------------------------------
# Basic expansion (backward-compatible with previous tests)
# ---------------------------------------------------------------------------

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
        result = expander.expand("treatment plan")
        assert result == "treatment plan"

    def test_not_partial_match(self, expander):
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
        exp = ShorthandExpander(source="builtin", custom_abbreviations=custom)
        assert "my custom expansion" in exp.expand("check myabbr")

    def test_custom_override(self):
        custom = {"htn": "HIGH BLOOD PRESSURE"}
        exp = ShorthandExpander(source="builtin", custom_abbreviations=custom)
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


# ---------------------------------------------------------------------------
# Data-driven source tests
# ---------------------------------------------------------------------------

class TestDataDrivenSources:
    """Test the data-driven loading mechanism."""

    def test_builtin_source_loads(self):
        exp = ShorthandExpander(source="builtin")
        assert exp.num_abbreviations >= 200
        assert "htn" in exp.abbreviations
        assert "dm2" in exp.abbreviations

    def test_auto_source_falls_back_gracefully(self):
        """In CI (no Zenodo access), auto should fall back to built-in."""
        exp = ShorthandExpander(source="auto")
        # Should still have at least the built-in abbreviations
        assert exp.num_abbreviations >= 200
        assert "hypertension" in exp.expand("htn")

    def test_disambiguation_preferred_is_default(self):
        exp = ShorthandExpander(source="builtin")
        assert exp._disambiguation == "preferred"

    def test_disambiguation_context_rules(self):
        exp = ShorthandExpander(source="builtin", disambiguation="context_rules")
        assert exp._disambiguation == "context_rules"
        # "pt" after lab context should expand to "prothrombin time"
        result = exp.expand("check pt")
        assert result == "check prothrombin time"

    def test_disambiguation_context_rules_patient(self):
        exp = ShorthandExpander(source="builtin", disambiguation="context_rules")
        # "pt" in general context should expand to "patient"
        result = exp.expand("the pt reports")
        assert "patient" in result

    def test_num_abbreviations_property(self):
        exp = ShorthandExpander(source="builtin")
        assert exp.num_abbreviations == len(exp.abbreviations)

    def test_get_senses_empty_for_builtin(self):
        """Built-in source has no sense inventory (not ambiguous)."""
        exp = ShorthandExpander(source="builtin")
        assert exp.get_senses("htn") == []
        assert exp.num_ambiguous == 0

    def test_identify_abbreviations_basic(self):
        exp = ShorthandExpander(source="builtin")
        found = exp.identify_abbreviations("pt with htn and dm2")
        abbrs = {f["abbreviation"].lower() for f in found}
        assert "htn" in abbrs
        assert "dm2" in abbrs

    def test_expand_in_place_false(self):
        exp = ShorthandExpander(source="builtin", expand_in_place=False)
        text = "pt with htn"
        assert exp.expand(text) == text  # no expansion


# ---------------------------------------------------------------------------
# Meta-Inventory integration (mocked)
# ---------------------------------------------------------------------------

class TestMetaInventoryParsing:
    """Test Meta-Inventory CSV parsing with mock data."""

    def test_parse_meta_inventory_csv(self, tmp_path):
        """Test parsing a CSV file with Meta-Inventory format."""
        from src.clinical.shorthand import _parse_meta_inventory

        csv_content = "SF,LF,PLF\nhtn,Hypertension,Hypertension\npt,Patient,Patient\npt,Prothrombin Time,Patient\nsob,Shortness of Breath,Shortness of Breath\n"
        csv_path = tmp_path / "test_meta.csv"
        csv_path.write_text(csv_content)

        abbreviations, sense_inventory = _parse_meta_inventory(str(csv_path))

        assert "htn" in abbreviations
        assert abbreviations["htn"].lower() == "hypertension"
        assert "sob" in abbreviations
        assert "pt" in abbreviations
        # "pt" is ambiguous — should have multiple senses
        assert "pt" in sense_inventory
        assert len(sense_inventory["pt"]) == 2

    def test_parse_filters_short_abbreviations(self, tmp_path):
        from src.clinical.shorthand import _parse_meta_inventory

        csv_content = "SF,LF,PLF\na,Alanine,Alanine\nhtn,Hypertension,Hypertension\n"
        csv_path = tmp_path / "test_short.csv"
        csv_path.write_text(csv_content)

        abbreviations, _ = _parse_meta_inventory(str(csv_path), min_length=2)
        assert "a" not in abbreviations
        assert "htn" in abbreviations

    def test_parse_filters_english_stopwords(self, tmp_path):
        from src.clinical.shorthand import _parse_meta_inventory

        csv_content = "SF,LF,PLF\nor,Operating Room,Operating Room\nhtn,Hypertension,Hypertension\n"
        csv_path = tmp_path / "test_stopwords.csv"
        csv_path.write_text(csv_content)

        abbreviations, _ = _parse_meta_inventory(str(csv_path))
        assert "or" not in abbreviations
        assert "htn" in abbreviations

    def test_builtin_overrides_meta_inventory(self, tmp_path):
        """Built-in abbreviations should override Meta-Inventory values."""
        from src.clinical.shorthand import _parse_meta_inventory

        # Create a Meta-Inventory with a different expansion for "htn"
        csv_content = "SF,LF,PLF\nhtn,High Blood Pressure,High Blood Pressure\n"
        csv_path = tmp_path / "test_override.csv"
        csv_path.write_text(csv_content)

        # Load with the CSV as source
        exp = ShorthandExpander(source=str(csv_path))
        # Built-in "htn" -> "hypertension" should override Meta-Inventory
        assert exp.abbreviations["htn"] == "hypertension"

    def test_meta_inventory_fills_gaps(self, tmp_path):
        """Meta-Inventory should provide abbreviations not in built-in."""
        csv_content = "SF,LF,PLF\nxyzmed,Experimental Medicine,Experimental Medicine\nhtn,Hypertension,Hypertension\n"
        csv_path = tmp_path / "test_gaps.csv"
        csv_path.write_text(csv_content)

        exp = ShorthandExpander(source=str(csv_path))
        # "xyzmed" only exists in Meta-Inventory
        assert "xyzmed" in exp.abbreviations
        assert exp.abbreviations["xyzmed"] == "Experimental Medicine"
