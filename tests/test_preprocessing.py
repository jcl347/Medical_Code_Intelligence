"""
Tests for data preprocessing and tokenization alignment.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class TestTokenizeAndAlignLabels:
    """Tests for subword tokenization label alignment."""

    def _get_mock_tokenizer(self):
        """Create a simple mock that simulates subword tokenization."""

        class MockTokenizerOutput:
            """Dict-like object that supports item assignment."""
            def __init__(self, input_ids, word_ids_map):
                self._data = {"input_ids": input_ids}
                self._word_ids_map = word_ids_map

            def __getitem__(self, key):
                return self._data.get(key)

            def __setitem__(self, key, val):
                self._data[key] = val

            def word_ids(self, batch_index=0):
                return self._word_ids_map[batch_index]

            def keys(self):
                return self._data.keys()

        class MockTokenizer:
            def __call__(self, tokens, **kwargs):
                batch_input_ids = []
                batch_word_ids_map = []
                for sent_tokens in tokens:
                    input_ids = [101]  # [CLS]
                    word_ids = [None]
                    for word_idx, token in enumerate(sent_tokens):
                        input_ids.append(hash(token) % 30000)
                        word_ids.append(word_idx)
                    input_ids.append(102)  # [SEP]
                    word_ids.append(None)
                    batch_input_ids.append(input_ids)
                    batch_word_ids_map.append(word_ids)
                return MockTokenizerOutput(batch_input_ids, batch_word_ids_map)

        return MockTokenizer()

    def test_basic_alignment(self):
        """Labels should align to first subword of each word."""
        from src.data.preprocessing import tokenize_and_align_labels, IGNORE_LABEL_ID

        tokenizer = self._get_mock_tokenizer()
        label2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}

        examples = {
            "tokens": [["The", "patient", "has", "diabetes"]],
            "ner_labels": [["O", "O", "O", "B-Disease"]],
        }

        result = tokenize_and_align_labels(examples, tokenizer, label2id)
        labels = result["labels"][0]

        # [CLS]=IGNORE, The=O, patient=O, has=O, diabetes=B-Disease, [SEP]=IGNORE
        assert labels[0] == IGNORE_LABEL_ID  # [CLS]
        assert labels[1] == 0  # The -> O
        assert labels[4] == 1  # diabetes -> B-Disease
        assert labels[-1] == IGNORE_LABEL_ID  # [SEP]

    def test_empty_sequence(self):
        """Empty token list should produce labels with only IGNORE for special tokens."""
        from src.data.preprocessing import tokenize_and_align_labels, IGNORE_LABEL_ID

        tokenizer = self._get_mock_tokenizer()
        label2id = {"O": 0}

        examples = {
            "tokens": [[]],
            "ner_labels": [[]],
        }

        result = tokenize_and_align_labels(examples, tokenizer, label2id)
        labels = result["labels"][0]
        # [CLS] and [SEP] only
        assert all(l == IGNORE_LABEL_ID for l in labels)


class TestGetLabelList:
    """Tests for label extraction from datasets."""

    def test_label_ordering(self):
        """O should always come first in label list."""
        from src.data.dataset_loader import get_label_list
        from datasets import Dataset, DatasetDict

        ds = DatasetDict({
            "train": Dataset.from_dict({
                "tokens": [["a", "b"], ["c"]],
                "ner_labels": [["B-Disease", "I-Disease"], ["O"]],
            })
        })

        label_list = get_label_list(ds)
        assert label_list[0] == "O"
        assert "B-Disease" in label_list
        assert "I-Disease" in label_list


class TestSplitLongSentences:
    """Tests for sliding window sentence splitting."""

    def test_short_sentence_unchanged(self):
        from src.data.data_utils import split_long_sentences

        tokens = ["The", "patient"]
        labels = ["O", "O"]
        windows = split_long_sentences(tokens, labels, max_tokens=10)
        assert len(windows) == 1
        assert windows[0]["tokens"] == tokens

    def test_long_sentence_split(self):
        from src.data.data_utils import split_long_sentences

        tokens = [f"word_{i}" for i in range(20)]
        labels = ["O"] * 20
        windows = split_long_sentences(tokens, labels, max_tokens=10, overlap=2)
        assert len(windows) >= 2
        # First window should have 10 tokens
        assert len(windows[0]["tokens"]) == 10
        # All original tokens should be covered
        all_tokens = []
        for w in windows:
            all_tokens.extend(w["tokens"])
        for t in tokens:
            assert t in all_tokens


class TestLabelMaps:
    """Tests for label mapping utilities."""

    def test_roundtrip(self):
        from src.data.dataset_loader import get_label_maps

        labels = ["O", "B-Disease", "I-Disease"]
        label2id, id2label = get_label_maps(labels)

        assert len(label2id) == 3
        assert len(id2label) == 3
        for label in labels:
            assert id2label[label2id[label]] == label
