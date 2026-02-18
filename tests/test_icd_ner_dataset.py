"""
Tests for the ICD NER composite dataset loader and label normalization.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.data.icd_dataset import (
    ICD_NER_LABELS,
    ICD_NER_LABEL2ID,
    _clean_garbage_labels,
    _normalize_ncbi_to_diagnosis,
    _normalize_bc5cdr_to_diagnosis,
)
from configs.ner_config import DATASET_CONFIGS


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestICDNERConfig:
    """Validate the icd_ner dataset config entry."""

    def test_icd_ner_config_exists(self):
        assert "icd_ner" in DATASET_CONFIGS

    def test_icd_ner_is_composite_format(self):
        cfg = DATASET_CONFIGS["icd_ner"]
        assert cfg["format"] == "composite"

    def test_icd_ner_entity_types(self):
        cfg = DATASET_CONFIGS["icd_ner"]
        assert cfg["entity_types"] == ["DIAGNOSIS"]

    def test_icd_ner_has_description(self):
        cfg = DATASET_CONFIGS["icd_ner"]
        assert "description" in cfg
        assert "DIAGNOSIS" in cfg["description"]


# ---------------------------------------------------------------------------
# Label scheme
# ---------------------------------------------------------------------------

class TestICDNERLabelScheme:
    """Validate the unified label scheme."""

    def test_label_list_contents(self):
        assert ICD_NER_LABELS == ["O", "B-DIAGNOSIS", "I-DIAGNOSIS"]

    def test_label2id_o_is_zero(self):
        assert ICD_NER_LABEL2ID["O"] == 0

    def test_label2id_roundtrip(self):
        id2label = {v: k for k, v in ICD_NER_LABEL2ID.items()}
        for label in ICD_NER_LABELS:
            assert id2label[ICD_NER_LABEL2ID[label]] == label

    def test_three_labels_only(self):
        assert len(ICD_NER_LABELS) == 3
        assert len(ICD_NER_LABEL2ID) == 3


# ---------------------------------------------------------------------------
# NCBI normalization
# ---------------------------------------------------------------------------

class TestNCBINormalization:
    """Test NCBI Disease → DIAGNOSIS normalization using mock data."""

    @pytest.fixture()
    def mock_ncbi_dataset(self):
        """Create a minimal NCBI-like dataset."""
        from datasets import Dataset, DatasetDict

        data = {
            "tokens": [
                ["Patient", "has", "diabetes"],
                ["No", "evidence", "of", "cancer"],
            ],
            "ner_tags": [
                [0, 0, 1],          # O O B-Disease
                [0, 0, 0, 1],       # O O O B-Disease
            ],
        }
        ds = Dataset.from_dict(data)
        return DatasetDict({"train": ds})

    def test_disease_becomes_diagnosis(self, mock_ncbi_dataset):
        result = _normalize_ncbi_to_diagnosis(mock_ncbi_dataset)
        labels = result["train"][0]["ner_labels"]
        assert labels == ["O", "O", "B-DIAGNOSIS"]

    def test_tags_are_integers(self, mock_ncbi_dataset):
        result = _normalize_ncbi_to_diagnosis(mock_ncbi_dataset)
        tags = result["train"][0]["ner_tags"]
        assert all(isinstance(t, int) for t in tags)
        assert tags[-1] == ICD_NER_LABEL2ID["B-DIAGNOSIS"]

    def test_multi_token_entity(self):
        """Test I-Disease → I-DIAGNOSIS for multi-word entities."""
        from datasets import Dataset, DatasetDict

        data = {
            "tokens": [["congestive", "heart", "failure"]],
            "ner_tags": [[1, 2, 2]],  # B-Disease I-Disease I-Disease
        }
        ds = DatasetDict({"train": Dataset.from_dict(data)})
        result = _normalize_ncbi_to_diagnosis(ds)
        labels = result["train"][0]["ner_labels"]
        assert labels == ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]

    def test_all_o_sentence(self):
        from datasets import Dataset, DatasetDict

        data = {
            "tokens": [["The", "patient", "is", "stable"]],
            "ner_tags": [[0, 0, 0, 0]],
        }
        ds = DatasetDict({"train": Dataset.from_dict(data)})
        result = _normalize_ncbi_to_diagnosis(ds)
        labels = result["train"][0]["ner_labels"]
        assert all(lab == "O" for lab in labels)


# ---------------------------------------------------------------------------
# BC5CDR normalization
# ---------------------------------------------------------------------------

class TestBC5CDRNormalization:
    """Test BC5CDR disease filtering and DIAGNOSIS normalization."""

    @pytest.fixture()
    def mock_bc5cdr_dataset(self):
        """Create a minimal BC5CDR-like dataset with Chemical + Disease tags."""
        from datasets import Dataset, DatasetDict, ClassLabel, Features, Sequence, Value

        label_names = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

        data = {
            "tokens": [
                ["Aspirin", "treats", "headache"],
                ["Metformin", "for", "type", "2", "diabetes"],
            ],
            "tags": [
                [1, 0, 3],          # B-Chemical O B-Disease
                [1, 0, 3, 4, 4],    # B-Chemical O B-Disease I-Disease I-Disease
            ],
        }

        features = Features({
            "tokens": Sequence(Value("string")),
            "tags": Sequence(ClassLabel(names=label_names)),
        })

        ds = Dataset.from_dict(data, features=features)
        return DatasetDict({"train": ds})

    def test_chemical_entities_become_o(self, mock_bc5cdr_dataset):
        result = _normalize_bc5cdr_to_diagnosis(mock_bc5cdr_dataset)
        labels = result["train"][0]["ner_labels"]
        # "Aspirin" was B-Chemical, should become O
        assert labels[0] == "O"

    def test_disease_entities_become_diagnosis(self, mock_bc5cdr_dataset):
        result = _normalize_bc5cdr_to_diagnosis(mock_bc5cdr_dataset)
        labels = result["train"][0]["ner_labels"]
        # "headache" was B-Disease, should become B-DIAGNOSIS
        assert labels[2] == "B-DIAGNOSIS"

    def test_multi_word_disease(self, mock_bc5cdr_dataset):
        result = _normalize_bc5cdr_to_diagnosis(mock_bc5cdr_dataset)
        labels = result["train"][1]["ner_labels"]
        # "Metformin" (B-Chemical) → O
        assert labels[0] == "O"
        # "type 2 diabetes" (B-Disease I-Disease I-Disease) → B/I-DIAGNOSIS
        assert labels[2] == "B-DIAGNOSIS"
        assert labels[3] == "I-DIAGNOSIS"
        assert labels[4] == "I-DIAGNOSIS"

    def test_only_diagnosis_labels_remain(self, mock_bc5cdr_dataset):
        result = _normalize_bc5cdr_to_diagnosis(mock_bc5cdr_dataset)
        for example in result["train"]:
            for label in example["ner_labels"]:
                assert label in ICD_NER_LABELS, f"Unexpected label: {label}"


# ---------------------------------------------------------------------------
# Garbage label cleaning
# ---------------------------------------------------------------------------

class TestGarbageLabelCleaning:
    """Test _clean_garbage_labels removes broken standalone B-DIAGNOSIS."""

    def _make_dataset(self, tokens_list, labels_list):
        """Helper to build a DatasetDict from tokens and string labels."""
        from datasets import Dataset, DatasetDict
        data = {
            "tokens": tokens_list,
            "ner_tags": [
                [ICD_NER_LABEL2ID[lab] for lab in labs]
                for labs in labels_list
            ],
            "ner_labels": labels_list,
        }
        return DatasetDict({"train": Dataset.from_dict(data)})

    def test_standalone_of_becomes_o(self):
        """Standalone 'of' with B-DIAGNOSIS should become O."""
        ds = self._make_dataset(
            [["evidence", "of", "disease"]],
            [["O", "B-DIAGNOSIS", "B-DIAGNOSIS"]],
        )
        result = _clean_garbage_labels(ds)
        labels = result["train"][0]["ner_labels"]
        assert labels[1] == "O"
        # "disease" is a real standalone entity — should be preserved
        assert labels[2] == "B-DIAGNOSIS"

    def test_standalone_and_becomes_o(self):
        ds = self._make_dataset(
            [["cancer", "and", "tumor"]],
            [["B-DIAGNOSIS", "B-DIAGNOSIS", "B-DIAGNOSIS"]],
        )
        result = _clean_garbage_labels(ds)
        labels = result["train"][0]["ner_labels"]
        assert labels[0] == "B-DIAGNOSIS"
        assert labels[1] == "O"  # "and" cleaned
        assert labels[2] == "B-DIAGNOSIS"

    def test_standalone_the_becomes_o(self):
        ds = self._make_dataset(
            [["the", "patient"]],
            [["B-DIAGNOSIS", "O"]],
        )
        result = _clean_garbage_labels(ds)
        assert result["train"][0]["ner_labels"][0] == "O"

    def test_uppercase_abbreviation_preserved(self):
        """Uppercase disease abbreviations (AS, AT, WAS) should NOT be cleaned."""
        ds = self._make_dataset(
            [["has", "AS", "and", "AT"]],
            [["O", "B-DIAGNOSIS", "O", "B-DIAGNOSIS"]],
        )
        result = _clean_garbage_labels(ds)
        labels = result["train"][0]["ner_labels"]
        assert labels[1] == "B-DIAGNOSIS"  # AS preserved
        assert labels[3] == "B-DIAGNOSIS"  # AT preserved

    def test_multi_word_entity_not_cleaned(self):
        """B-DIAGNOSIS followed by I-DIAGNOSIS should never be cleaned."""
        ds = self._make_dataset(
            [["of", "the", "heart"]],
            [["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]],
        )
        result = _clean_garbage_labels(ds)
        labels = result["train"][0]["ner_labels"]
        # "of" has continuation, so it's kept as part of a multi-word entity
        assert labels == ["B-DIAGNOSIS", "I-DIAGNOSIS", "I-DIAGNOSIS"]

    def test_single_lowercase_char_cleaned(self):
        """Single lowercase characters should be cleaned."""
        ds = self._make_dataset(
            [["x", "disease"]],
            [["B-DIAGNOSIS", "B-DIAGNOSIS"]],
        )
        result = _clean_garbage_labels(ds)
        labels = result["train"][0]["ner_labels"]
        assert labels[0] == "O"
        assert labels[1] == "B-DIAGNOSIS"

    def test_legitimate_standalone_disease_preserved(self):
        """Real disease names should not be cleaned."""
        ds = self._make_dataset(
            [["diabetes", "and", "cancer"]],
            [["B-DIAGNOSIS", "O", "B-DIAGNOSIS"]],
        )
        result = _clean_garbage_labels(ds)
        labels = result["train"][0]["ner_labels"]
        assert labels[0] == "B-DIAGNOSIS"
        assert labels[2] == "B-DIAGNOSIS"

    def test_tags_updated_consistently(self):
        """ner_tags should match ner_labels after cleaning."""
        ds = self._make_dataset(
            [["the", "disease"]],
            [["B-DIAGNOSIS", "B-DIAGNOSIS"]],
        )
        result = _clean_garbage_labels(ds)
        tags = result["train"][0]["ner_tags"]
        labels = result["train"][0]["ner_labels"]
        for tag, label in zip(tags, labels):
            assert tag == ICD_NER_LABEL2ID[label]


# ---------------------------------------------------------------------------
# Dataset loader dispatch
# ---------------------------------------------------------------------------

class TestDatasetLoaderDispatch:
    """Test that load_ner_dataset dispatches to icd_ner correctly."""

    def test_icd_ner_in_choices(self):
        """The icd_ner key should be recognized by the dataset loader."""
        from src.data.dataset_loader import load_ner_dataset
        # The config exists and format is composite
        assert "icd_ner" in DATASET_CONFIGS
        assert DATASET_CONFIGS["icd_ner"]["format"] == "composite"

    def test_unknown_composite_raises(self):
        """Unknown composite datasets should raise ValueError."""
        from src.data.dataset_loader import load_ner_dataset

        # Temporarily add a fake composite entry
        DATASET_CONFIGS["_test_fake_composite"] = {
            "hf_name": "fake",
            "format": "composite",
        }
        try:
            with pytest.raises(ValueError, match="No loader for composite"):
                load_ner_dataset("_test_fake_composite")
        finally:
            del DATASET_CONFIGS["_test_fake_composite"]


# ---------------------------------------------------------------------------
# Pipeline ICD resolution integration
# ---------------------------------------------------------------------------

class TestPipelineICDResolution:
    """Test that MedicalEntity ICD codes field works."""

    def test_medical_entity_icd_codes_default_none(self):
        from src.clinical.pipeline import MedicalEntity
        ent = MedicalEntity(
            text="diabetes", label="DIAGNOSIS",
            start_char=0, end_char=8, score=0.95,
        )
        assert ent.icd_codes is None

    def test_medical_entity_to_dict_with_icd(self):
        from src.clinical.pipeline import MedicalEntity
        ent = MedicalEntity(
            text="diabetes", label="DIAGNOSIS",
            start_char=0, end_char=8, score=0.95,
            icd_codes=[{"code": "E11.9", "description": "Type 2 diabetes", "score": 0.8}],
        )
        d = ent.to_dict()
        assert "icd_codes" in d
        assert d["icd_codes"][0]["code"] == "E11.9"

    def test_medical_entity_to_dict_without_icd(self):
        from src.clinical.pipeline import MedicalEntity
        ent = MedicalEntity(
            text="diabetes", label="DIAGNOSIS",
            start_char=0, end_char=8, score=0.95,
        )
        d = ent.to_dict()
        assert "icd_codes" not in d

    def test_pipeline_init_with_icd_resolution(self):
        """Pipeline should accept resolve_icd_codes parameter."""
        from src.clinical.pipeline import MedicalCodingPipeline
        pipeline = MedicalCodingPipeline(
            expand_shorthand=False,
            detect_negation=False,
            resolve_icd_codes=False,
        )
        assert pipeline.icd_lookup is None

    def test_pipeline_format_output_with_icd(self):
        """format_output should include ICD code when present."""
        from src.clinical.pipeline import MedicalCodingPipeline, MedicalEntity
        pipeline = MedicalCodingPipeline(
            expand_shorthand=False,
            detect_negation=False,
        )
        entities = [
            MedicalEntity(
                text="diabetes", label="DIAGNOSIS",
                start_char=12, end_char=20, score=0.95,
                icd_codes=[{"code": "E11.9", "description": "Type 2 DM", "score": 0.8}],
            ),
        ]
        output = pipeline.format_output("Patient has diabetes", entities)
        assert "ICD=E11.9" in output
