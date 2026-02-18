"""
Tests for NER entity post-processing (stopword filtering + fragment merging).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.inference.entity_utils import NEREntity, post_process_entities


# ---------------------------------------------------------------------------
# Stopword / garbage filtering
# ---------------------------------------------------------------------------

class TestEntityFiltering:
    """Test that garbage entities are removed by post-processing."""

    def test_stopword_of_filtered(self):
        entities = [
            NEREntity(text="of", label="DIAGNOSIS", start_char=10, end_char=12, score=0.8),
        ]
        result = post_process_entities(entities, "evidence of disease")
        assert len(result) == 0

    def test_stopword_and_filtered(self):
        """'and' entity is filtered; adjacent same-label entities then merge."""
        entities = [
            NEREntity(text="diabetes", label="DIAGNOSIS", start_char=0, end_char=8, score=0.9),
            NEREntity(text="and", label="DIAGNOSIS", start_char=9, end_char=12, score=0.7),
            NEREntity(text="cancer", label="DIAGNOSIS", start_char=13, end_char=19, score=0.9),
        ]
        result = post_process_entities(entities, "diabetes and cancer")
        # After filtering "and", "diabetes" and "cancer" are 5 chars apart
        # with gap text "and" (a stopword) -> they merge
        assert len(result) == 1
        assert result[0].text == "diabetes and cancer"

    def test_stopword_the_filtered(self):
        entities = [
            NEREntity(text="the", label="DIAGNOSIS", start_char=0, end_char=3, score=0.6),
        ]
        result = post_process_entities(entities, "the patient")
        assert len(result) == 0

    def test_single_lowercase_char_filtered(self):
        entities = [
            NEREntity(text="a", label="DIAGNOSIS", start_char=0, end_char=1, score=0.5),
        ]
        result = post_process_entities(entities, "a disease")
        assert len(result) == 0

    def test_punctuation_entity_filtered(self):
        entities = [
            NEREntity(text=",", label="DIAGNOSIS", start_char=5, end_char=6, score=0.3),
        ]
        result = post_process_entities(entities, "hello, world")
        assert len(result) == 0

    def test_whitespace_entity_filtered(self):
        entities = [
            NEREntity(text="  ", label="DIAGNOSIS", start_char=5, end_char=7, score=0.3),
        ]
        result = post_process_entities(entities, "hello  world")
        assert len(result) == 0

    def test_legitimate_entity_preserved(self):
        entities = [
            NEREntity(text="diabetes", label="DIAGNOSIS", start_char=0, end_char=8, score=0.95),
        ]
        result = post_process_entities(entities, "diabetes mellitus")
        assert len(result) == 1
        assert result[0].text == "diabetes"

    def test_uppercase_abbreviation_preserved(self):
        """Uppercase disease abbreviations like 'AS' should NOT be filtered."""
        entities = [
            NEREntity(text="AS", label="DIAGNOSIS", start_char=0, end_char=2, score=0.85),
        ]
        result = post_process_entities(entities, "AS was diagnosed")
        assert len(result) == 1
        assert result[0].text == "AS"

    def test_empty_entities_returns_empty(self):
        result = post_process_entities([], "some text")
        assert result == []


# ---------------------------------------------------------------------------
# Fragment merging
# ---------------------------------------------------------------------------

class TestEntityMerging:
    """Test that adjacent entity fragments are merged."""

    def test_adjacent_fragments_merged(self):
        """Two adjacent same-label entities separated by a space should merge."""
        text = "congestive heart failure"
        entities = [
            NEREntity(text="congestive", label="DIAGNOSIS", start_char=0, end_char=10, score=0.9),
            NEREntity(text="heart failure", label="DIAGNOSIS", start_char=11, end_char=24, score=0.95),
        ]
        result = post_process_entities(entities, text)
        assert len(result) == 1
        assert result[0].text == "congestive heart failure"
        assert result[0].start_char == 0
        assert result[0].end_char == 24

    def test_fragments_with_function_word_gap_merged(self):
        """Fragments separated by 'of' should merge (e.g., 'loss of appetite')."""
        text = "loss of appetite"
        entities = [
            NEREntity(text="loss", label="DIAGNOSIS", start_char=0, end_char=4, score=0.85),
            NEREntity(text="appetite", label="DIAGNOSIS", start_char=8, end_char=16, score=0.9),
        ]
        result = post_process_entities(entities, text)
        assert len(result) == 1
        assert result[0].text == "loss of appetite"

    def test_different_labels_not_merged(self):
        """Entities with different labels should NOT be merged."""
        text = "diabetes and aspirin"
        entities = [
            NEREntity(text="diabetes", label="DIAGNOSIS", start_char=0, end_char=8, score=0.9),
            NEREntity(text="aspirin", label="CHEMICAL", start_char=13, end_char=20, score=0.85),
        ]
        result = post_process_entities(entities, text)
        assert len(result) == 2

    def test_far_apart_entities_not_merged(self):
        """Entities far apart should NOT be merged even with same label."""
        text = "diabetes is common. Cancer is also common."
        entities = [
            NEREntity(text="diabetes", label="DIAGNOSIS", start_char=0, end_char=8, score=0.9),
            NEREntity(text="Cancer", label="DIAGNOSIS", start_char=20, end_char=26, score=0.9),
        ]
        result = post_process_entities(entities, text)
        assert len(result) == 2

    def test_merge_takes_max_score(self):
        """Merged entity should have the maximum score of its components."""
        text = "type 2 diabetes"
        entities = [
            NEREntity(text="type", label="DIAGNOSIS", start_char=0, end_char=4, score=0.7),
            NEREntity(text="2 diabetes", label="DIAGNOSIS", start_char=5, end_char=15, score=0.95),
        ]
        result = post_process_entities(entities, text)
        assert len(result) == 1
        assert result[0].score == 0.95

    def test_three_fragments_merged(self):
        """Three adjacent fragments should all merge into one."""
        text = "acute renal failure"
        entities = [
            NEREntity(text="acute", label="DIAGNOSIS", start_char=0, end_char=5, score=0.8),
            NEREntity(text="renal", label="DIAGNOSIS", start_char=6, end_char=11, score=0.85),
            NEREntity(text="failure", label="DIAGNOSIS", start_char=12, end_char=19, score=0.9),
        ]
        result = post_process_entities(entities, text)
        assert len(result) == 1
        assert result[0].text == "acute renal failure"


# ---------------------------------------------------------------------------
# Combined filtering + merging
# ---------------------------------------------------------------------------

class TestFilterAndMergeCombined:
    """Test filtering and merging work together correctly."""

    def test_stopword_filtered_then_remaining_merged(self):
        """After filtering garbage, remaining fragments should still merge."""
        text = "diabetes and cancer of the lung"
        entities = [
            NEREntity(text="diabetes", label="DIAGNOSIS", start_char=0, end_char=8, score=0.9),
            NEREntity(text="and", label="DIAGNOSIS", start_char=9, end_char=12, score=0.5),
            NEREntity(text="cancer", label="DIAGNOSIS", start_char=13, end_char=19, score=0.9),
            NEREntity(text="of", label="DIAGNOSIS", start_char=20, end_char=22, score=0.4),
            NEREntity(text="the", label="DIAGNOSIS", start_char=23, end_char=26, score=0.3),
            NEREntity(text="lung", label="DIAGNOSIS", start_char=27, end_char=31, score=0.85),
        ]
        result = post_process_entities(entities, text)
        # "and", "of", "the" filtered out.
        # "diabetes" and "cancer" merge (gap=5, text="and ", a stopword).
        # "cancer" (end=19) to "lung" (start=27) gap=8 > merge_gap=5, so no merge.
        assert len(result) == 2
        assert result[0].text == "diabetes and cancer"
        assert result[1].text == "lung"
