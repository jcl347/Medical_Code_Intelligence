"""
Tests for ICD-specific dataset configs, span-to-BIO conversion,
and dataset loader behaviour with ICD-related datasets.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from configs.ner_config import DATASET_CONFIGS
from src.data.dataset_loader import (
    spans_to_bio,
    _default_tokenize,
    get_label_maps,
    load_ner_dataset,
)


# ---------------------------------------------------------------------------
# Span-to-BIO conversion
# ---------------------------------------------------------------------------

class TestSpansToBIO:
    """Test character-offset span annotations → BIO tag conversion."""

    def test_single_word_entity(self):
        text = "Patient has diabetes and hypertension."
        entities = [
            {"start": 12, "end": 20, "class": "DISORDER"},
        ]
        tokens, labels = spans_to_bio(text, entities)
        assert "diabetes" in tokens
        idx = tokens.index("diabetes")
        assert labels[idx] == "B-DISORDER"

    def test_multi_word_entity(self):
        text = "Diagnosed with congestive heart failure."
        entities = [
            {"start": 15, "end": 38, "class": "DISORDER"},
        ]
        tokens, labels = spans_to_bio(text, entities)
        assert "congestive" in tokens
        assert "heart" in tokens
        assert "failure." in tokens  # trailing punctuation attached
        idx_c = tokens.index("congestive")
        idx_h = tokens.index("heart")
        assert labels[idx_c] == "B-DISORDER"
        assert labels[idx_h] == "I-DISORDER"

    def test_multiple_entities(self):
        text = "Patient takes metformin for diabetes."
        entities = [
            {"start": 14, "end": 23, "class": "CLINICAL_DRUG"},
            {"start": 28, "end": 36, "class": "DISORDER"},
        ]
        tokens, labels = spans_to_bio(text, entities)
        idx_met = tokens.index("metformin")
        assert labels[idx_met] == "B-CLINICAL_DRUG"
        # "diabetes." includes punctuation
        diabetes_idx = [i for i, t in enumerate(tokens) if t.startswith("diabetes")][0]
        assert labels[diabetes_idx] == "B-DISORDER"

    def test_non_entity_tokens_are_O(self):
        text = "The patient is stable."
        entities = []
        tokens, labels = spans_to_bio(text, entities)
        assert all(l == "O" for l in labels)

    def test_adjacent_entities(self):
        text = "STEMI myocardial infarction"
        entities = [
            {"start": 0, "end": 5, "class": "DISORDER"},
            {"start": 6, "end": 27, "class": "DISORDER"},
        ]
        tokens, labels = spans_to_bio(text, entities)
        assert labels[0] == "B-DISORDER"  # STEMI
        assert labels[1] == "B-DISORDER"  # myocardial (new entity)
        assert labels[2] == "I-DISORDER"  # infarction

    def test_class_name_normalisation(self):
        """Entity class with spaces should be normalised to underscores."""
        text = "Had a chest x-ray done."
        entities = [
            {"start": 6, "end": 17, "class": "Medical Procedure"},
        ]
        tokens, labels = spans_to_bio(text, entities)
        # Find the labelled tokens
        bio_labels = [l for l in labels if l != "O"]
        for l in bio_labels:
            assert " " not in l
            assert "MEDICAL_PROCEDURE" in l

    def test_empty_text(self):
        tokens, labels = spans_to_bio("", [])
        assert tokens == []
        assert labels == []

    def test_entity_at_start(self):
        text = "Hypertension is present."
        entities = [{"start": 0, "end": 12, "class": "DISORDER"}]
        tokens, labels = spans_to_bio(text, entities)
        assert labels[0] == "B-DISORDER"

    def test_entity_at_end(self):
        text = "Patient has fever"
        entities = [{"start": 12, "end": 17, "class": "DISORDER"}]
        tokens, labels = spans_to_bio(text, entities)
        assert labels[-1] == "B-DISORDER"


class TestDefaultTokenizer:
    """Test the whitespace tokenizer used for span-to-BIO conversion."""

    def test_simple_sentence(self):
        spans = _default_tokenize("Hello world today")
        assert len(spans) == 3
        text = "Hello world today"
        assert text[spans[0][0]:spans[0][1]] == "Hello"
        assert text[spans[1][0]:spans[1][1]] == "world"
        assert text[spans[2][0]:spans[2][1]] == "today"

    def test_punctuation_attached(self):
        spans = _default_tokenize("Diagnosis: hypertension, diabetes.")
        text = "Diagnosis: hypertension, diabetes."
        tokens = [text[s:e] for s, e in spans]
        assert "Diagnosis:" in tokens
        assert "hypertension," in tokens
        assert "diabetes." in tokens

    def test_empty_string(self):
        spans = _default_tokenize("")
        assert spans == []

    def test_multiple_spaces(self):
        spans = _default_tokenize("word1    word2")
        assert len(spans) == 2


# ---------------------------------------------------------------------------
# ICD dataset config validation
# ---------------------------------------------------------------------------

class TestICDDatasetConfigs:
    """Validate that ICD-specific dataset configs are properly structured."""

    def test_biomed_ner_config_exists(self):
        assert "biomed_ner" in DATASET_CONFIGS

    def test_biomed_ner_has_span_format(self):
        cfg = DATASET_CONFIGS["biomed_ner"]
        assert cfg["format"] == "span"
        assert "text_column" in cfg
        assert "entities_column" in cfg

    def test_biomed_ner_entity_types(self):
        cfg = DATASET_CONFIGS["biomed_ner"]
        types = cfg["entity_types"]
        assert "DISORDER" in types
        assert "MEDICAL_PROCEDURE" in types
        assert "CLINICAL_DRUG" in types
        assert "ANATOMICAL_STRUCTURE" in types

    def test_icd10_terminology_config_exists(self):
        assert "icd10_terminology" in DATASET_CONFIGS

    def test_icd10_terminology_is_code_lookup(self):
        cfg = DATASET_CONFIGS["icd10_terminology"]
        assert cfg["format"] == "code_lookup"
        assert "code_column" in cfg
        assert "description_column" in cfg

    def test_icd10_code_description_config_exists(self):
        assert "icd10_code_description" in DATASET_CONFIGS

    def test_icd10_code_description_is_instruction(self):
        cfg = DATASET_CONFIGS["icd10_code_description"]
        assert cfg["format"] == "instruction"

    def test_all_icd_configs_have_hf_name(self):
        for key in ["biomed_ner", "icd10_terminology", "icd10_code_description"]:
            assert "hf_name" in DATASET_CONFIGS[key]

    def test_code_lookup_cannot_load_as_ner(self):
        """Code-lookup datasets should raise ValueError when loaded as NER."""
        with pytest.raises(ValueError, match="not a token-level NER"):
            load_ner_dataset("icd10_terminology")

    def test_instruction_cannot_load_as_ner(self):
        """Instruction datasets should raise ValueError when loaded as NER."""
        with pytest.raises(ValueError, match="not a token-level NER"):
            load_ner_dataset("icd10_code_description")


# ---------------------------------------------------------------------------
# BIO label map generation for ICD entity types
# ---------------------------------------------------------------------------

class TestICDBIOLabelGeneration:
    """Test BIO label generation for ICD-relevant entity types."""

    def test_bio_labels_from_icd_entities(self):
        """Simulate converting a clinical text with ICD-relevant entities."""
        text = "Patient with type 2 diabetes on metformin and chest x-ray ordered."
        entities = [
            {"start": 13, "end": 28, "class": "DISORDER"},
            {"start": 32, "end": 41, "class": "CLINICAL_DRUG"},
        ]
        tokens, labels = spans_to_bio(text, entities)

        # Collect unique labels
        unique_labels = sorted(set(labels), key=lambda x: (x != "O", x))
        assert "O" in unique_labels
        assert "B-DISORDER" in unique_labels
        assert "B-CLINICAL_DRUG" in unique_labels

        # Generate label maps
        label2id, id2label = get_label_maps(unique_labels)
        assert label2id["O"] == 0  # O should come first
        # Roundtrip check
        for label in unique_labels:
            assert id2label[label2id[label]] == label

    def test_icd_relevant_entity_coverage(self):
        """Ensure BIO labels are correctly produced for ICD entity types."""
        text = (
            "The echocardiogram showed reduced ejection fraction. "
            "Started on lisinopril for heart failure."
        )
        entities = [
            {"start": 4, "end": 18, "class": "MEDICAL_PROCEDURE"},
            {"start": 63, "end": 73, "class": "CLINICAL_DRUG"},
            {"start": 78, "end": 91, "class": "DISORDER"},
        ]
        tokens, labels = spans_to_bio(text, entities)

        # Verify each entity type appears at least once
        label_set = set(labels)
        assert any("MEDICAL_PROCEDURE" in l for l in label_set)
        assert any("CLINICAL_DRUG" in l for l in label_set)
        assert any("DISORDER" in l for l in label_set)

        # All B- labels should have correct prefix
        for l in labels:
            if l != "O":
                assert l.startswith("B-") or l.startswith("I-")


class TestSpanToBIOWithNegation:
    """Test span-to-BIO works correctly with negated entities (for ICD coding)."""

    def test_negated_entity_still_gets_bio_label(self):
        """Negation is a separate concern; BIO tagging should still label the entity."""
        text = "No evidence of pneumonia."
        entities = [{"start": 15, "end": 24, "class": "DISORDER"}]
        tokens, labels = spans_to_bio(text, entities)

        # "pneumonia." should still get a BIO label
        pneumonia_idx = [i for i, t in enumerate(tokens) if "pneumonia" in t]
        assert len(pneumonia_idx) >= 1
        assert labels[pneumonia_idx[0]] == "B-DISORDER"

    def test_bio_labels_agnostic_to_context(self):
        """BIO tagging is context-agnostic; negation is post-processing."""
        for text, ent_start, ent_end in [
            ("Patient has diabetes", 12, 20),
            ("Patient denies diabetes", 15, 23),
            ("No history of diabetes", 14, 22),
        ]:
            entities = [{"start": ent_start, "end": ent_end, "class": "DISORDER"}]
            tokens, labels = spans_to_bio(text, entities)
            diabetes_idx = [i for i, t in enumerate(tokens) if "diabetes" in t]
            assert len(diabetes_idx) >= 1
            assert labels[diabetes_idx[0]] == "B-DISORDER"
