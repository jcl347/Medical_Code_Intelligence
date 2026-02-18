"""
Inference pipeline for Medical NER.

Provides a clean API for running NER predictions on raw text,
with support for both standard softmax and CRF decoding.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    pipeline,
)

from src.inference.entity_utils import NEREntity, post_process_entities

logger = logging.getLogger(__name__)


class NERPredictor:
    """
    High-level NER prediction interface.

    Supports:
    - Loading from a fine-tuned model directory
    - Loading from HuggingFace model hub
    - Batch prediction
    - Aggregation of subword predictions into word-level entities
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        aggregation_strategy: str = "first",
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to fine-tuned model directory or HuggingFace model id.
        device : str, optional
            Device string ('cpu', 'cuda', 'cuda:0'). Auto-detected if None.
        aggregation_strategy : str
            How to aggregate subword predictions: 'first', 'average', 'max'.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.aggregation_strategy = aggregation_strategy

        # Resolve local paths so transformers doesn't treat them as HF repo IDs.
        # A valid HF repo ID has at most one "/" (namespace/repo); paths with more
        # separators or that exist on disk are always local.
        import os
        abs_path = os.path.abspath(model_path)
        if os.path.isdir(abs_path):
            model_path = abs_path
        elif os.sep in model_path or model_path.count("/") > 1:
            raise FileNotFoundError(
                f"Model directory not found: '{abs_path}'. "
                f"Train a model first or provide a valid HuggingFace model ID."
            )

        logger.info("Loading NER model from '%s' on device '%s'...", model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        logger.info("Model loaded. Labels: %s", list(self.id2label.values()))

        # Also build HuggingFace pipeline for convenience
        self._pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if "cuda" in device else -1,
            aggregation_strategy=aggregation_strategy,
        )

    def predict(self, text: str) -> List[NEREntity]:
        """
        Run NER on a single text string.

        Parameters
        ----------
        text : str
            Input text (e.g. a clinical note or PubMed abstract).

        Returns
        -------
        list of NEREntity
            Extracted entities with labels, positions, and confidence scores.
        """
        raw_results = self._pipeline(text)
        entities = []
        for r in raw_results:
            entity = NEREntity(
                text=r.get("word", ""),
                label=r.get("entity_group", r.get("entity", "")),
                start_char=r.get("start", 0),
                end_char=r.get("end", 0),
                score=r.get("score", 0.0),
            )
            entities.append(entity)
        return post_process_entities(entities, text)

    def predict_batch(self, texts: List[str]) -> List[List[NEREntity]]:
        """Run NER on a batch of texts."""
        all_results = self._pipeline(texts)
        batch_entities = []
        for text_str, raw_results in zip(texts, all_results):
            entities = []
            if isinstance(raw_results, dict):
                raw_results = [raw_results]
            for r in raw_results:
                entity = NEREntity(
                    text=r.get("word", ""),
                    label=r.get("entity_group", r.get("entity", "")),
                    start_char=r.get("start", 0),
                    end_char=r.get("end", 0),
                    score=r.get("score", 0.0),
                )
                entities.append(entity)
            batch_entities.append(post_process_entities(entities, text_str))
        return batch_entities

    @torch.no_grad()
    def predict_tokens(
        self, tokens: List[str],
    ) -> Tuple[List[str], List[float]]:
        """
        Run NER on pre-tokenized input (word-level tokens).

        Returns
        -------
        labels : list of str
            Predicted BIO labels for each token.
        scores : list of float
            Confidence score for each token prediction.
        """
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        outputs = self.model(**encoding)
        logits = outputs.logits[0]  # (seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        pred_scores = probs.max(dim=-1).values.cpu().numpy()

        # Align subword predictions back to word level
        word_ids = encoding.get("word_ids", None)
        if word_ids is None:
            # Fallback: use tokenizer's word_ids
            word_ids_list = self.tokenizer(
                tokens, is_split_into_words=True, return_tensors="pt",
            ).word_ids(0)
        else:
            word_ids_list = word_ids

        labels = []
        scores = []
        seen_words = set()

        for idx, word_idx in enumerate(word_ids_list):
            if word_idx is None or word_idx in seen_words:
                continue
            seen_words.add(word_idx)
            labels.append(self.id2label.get(int(pred_ids[idx]), "O"))
            scores.append(float(pred_scores[idx]))

        return labels, scores

    def format_entities(self, text: str, entities: List[NEREntity]) -> str:
        """Format entities for display with inline annotations."""
        if not entities:
            return text

        # Sort by start position
        sorted_ents = sorted(entities, key=lambda e: e.start_char)
        result = []
        last_end = 0

        for ent in sorted_ents:
            result.append(text[last_end:ent.start_char])
            result.append(f"[{ent.text}]({ent.label}:{ent.score:.2f})")
            last_end = ent.end_char

        result.append(text[last_end:])
        return "".join(result)
