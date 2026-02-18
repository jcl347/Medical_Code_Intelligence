"""
Tests for MeDAL-based abbreviation disambiguation.

Uses mocked transformer model to avoid downloading in CI.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.clinical.abbreviation_disambiguator import AbbreviationDisambiguator


# ---------------------------------------------------------------------------
# Disambiguation logic
# ---------------------------------------------------------------------------

class TestDisambiguateLogic:
    """Test the disambiguation decision logic."""

    def test_single_sense_returns_it(self):
        disambiguator = AbbreviationDisambiguator.__new__(AbbreviationDisambiguator)
        disambiguator._model = None
        disambiguator._tokenizer = None
        disambiguator._device = None
        disambiguator._model_name = "mock"
        disambiguator._max_length = 256

        result = disambiguator.disambiguate(
            text="Patient with HTN.",
            abbreviation="HTN",
            abbr_start=13,
            abbr_end=16,
            senses=["hypertension"],
        )
        assert result == "hypertension"

    def test_empty_senses_returns_abbreviation(self):
        disambiguator = AbbreviationDisambiguator.__new__(AbbreviationDisambiguator)
        result = disambiguator.disambiguate(
            text="Patient with HTN.",
            abbreviation="HTN",
            abbr_start=13,
            abbr_end=16,
            senses=[],
        )
        assert result == "HTN"


class TestDisambiguateWithMockedModel:
    """Test disambiguation with a mocked ELECTRA model."""

    @pytest.fixture
    def disambiguator(self):
        d = AbbreviationDisambiguator.__new__(AbbreviationDisambiguator)
        d._model_name = "mock-model"
        d._device = "cpu"
        d._max_length = 256

        # Mock model that returns different embeddings for different texts
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        d._model = mock_model
        d._tokenizer = mock_tokenizer
        return d

    def test_picks_most_similar_sense(self, disambiguator):
        """The sense whose replacement is most similar to original should win."""
        call_count = [0]
        embeddings = [
            np.array([1.0, 0.0, 0.0]),   # original: "Patient with SOB"
            np.array([0.95, 0.05, 0.0]),  # "Patient with shortness of breath" (similar)
            np.array([0.1, 0.9, 0.0]),    # "Patient with side of bed" (different)
        ]

        def mock_get_embedding(text):
            idx = min(call_count[0], len(embeddings) - 1)
            call_count[0] += 1
            return embeddings[idx]

        disambiguator._get_sentence_embedding = mock_get_embedding

        result = disambiguator.disambiguate(
            text="Patient presents with SOB and fatigue.",
            abbreviation="SOB",
            abbr_start=21,
            abbr_end=24,
            senses=["shortness of breath", "side of bed"],
        )
        assert result == "shortness of breath"

    def test_disambiguate_from_context(self, disambiguator):
        """Test the simplified context-based API."""
        call_count = [0]
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.9, 0.1]),
            np.array([0.2, 0.8]),
        ]

        def mock_get_embedding(text):
            idx = min(call_count[0], len(embeddings) - 1)
            call_count[0] += 1
            return embeddings[idx]

        disambiguator._get_sentence_embedding = mock_get_embedding

        result = disambiguator.disambiguate_from_context(
            context="Patient denies",
            abbreviation="CP",
            senses=["chest pain", "cerebral palsy"],
        )
        assert result == "chest pain"

    def test_disambiguate_from_context_single_sense(self, disambiguator):
        result = disambiguator.disambiguate_from_context(
            context="Patient with",
            abbreviation="HTN",
            senses=["hypertension"],
        )
        assert result == "hypertension"

    def test_disambiguate_from_context_empty_senses(self, disambiguator):
        result = disambiguator.disambiguate_from_context(
            context="Patient with",
            abbreviation="XYZ",
            senses=[],
        )
        assert result is None


# ---------------------------------------------------------------------------
# Pipeline integration with transformer disambiguation
# ---------------------------------------------------------------------------

class TestPipelineDisambiguation:
    """Test ShorthandExpander integration with transformer disambiguation."""

    def test_transformer_disambiguation_setting(self):
        """Setting disambiguation='transformer' should store the setting."""
        from src.clinical.shorthand import ShorthandExpander
        exp = ShorthandExpander(source="builtin", disambiguation="transformer")
        assert exp._disambiguation == "transformer"

    def test_preferred_disambiguation_works(self):
        """Default preferred disambiguation should expand normally."""
        from src.clinical.shorthand import ShorthandExpander
        exp = ShorthandExpander(source="builtin", disambiguation="preferred")
        result = exp.expand("pt with htn")
        # With "preferred" strategy, uses the dictionary directly
        assert "hypertension" in result


# ---------------------------------------------------------------------------
# CASI evaluation config
# ---------------------------------------------------------------------------

class TestCASIDatasetConfig:
    """Test that CASI dataset config is properly defined."""

    def test_casi_config_exists(self):
        from configs.ner_config import DATASET_CONFIGS
        assert "casi" in DATASET_CONFIGS
        assert "mitclinicalml/clinical-ie" in DATASET_CONFIGS["casi"]["hf_name"]

    def test_medal_config_exists(self):
        from configs.ner_config import DATASET_CONFIGS
        assert "medal" in DATASET_CONFIGS
        assert "McGill-NLP/medal" in DATASET_CONFIGS["medal"]["hf_name"]

    def test_casi_format_is_abbreviation_disambiguation(self):
        from configs.ner_config import DATASET_CONFIGS
        assert DATASET_CONFIGS["casi"]["format"] == "abbreviation_disambiguation"

    def test_medal_format_is_abbreviation_disambiguation(self):
        from configs.ner_config import DATASET_CONFIGS
        assert DATASET_CONFIGS["medal"]["format"] == "abbreviation_disambiguation"
