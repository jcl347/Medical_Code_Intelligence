"""
Data-driven ICD-10-CM code lookup and entity linking.

Two matching strategies (following SciSpacy's architecture):
1. TF-IDF character n-gram vectorisation + cosine similarity (fast, no GPU)
2. Direct text matching against code descriptions

Data source: atta00/icd10-codes on HuggingFace (51,438 codes, MIT licensed)
with chapter/section/category hierarchy included.

Unlike the previous rule-based approach, this module loads the FULL ICD-10-CM
code set from a public dataset rather than relying on a hardcoded dictionary.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ICDCode:
    """A single ICD-10-CM code with hierarchy metadata."""
    code: str
    description: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    category: Optional[str] = None
    is_billable: bool = True

    def to_dict(self) -> Dict:
        d = {
            "code": self.code,
            "description": self.description,
        }
        if self.chapter:
            d["chapter"] = self.chapter
        if self.section:
            d["section"] = self.section
        if self.category:
            d["category"] = self.category
        return d


@dataclass
class ICDMatch:
    """A candidate ICD code match for a clinical entity."""
    code: str
    description: str
    score: float
    match_type: str  # "tfidf", "exact", "substring"

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "description": self.description,
            "score": round(self.score, 4),
            "match_type": self.match_type,
        }


class ICDCodeLookup:
    """
    Data-driven ICD-10-CM entity linker.

    Loads the full ICD-10-CM code set from HuggingFace (atta00/icd10-codes,
    51K codes) and builds a TF-IDF index for fast entity→code matching,
    following SciSpacy's proven architecture.

    Matching strategy (inspired by SciSpacy's EntityLinker):
    1. Build TF-IDF character 3-gram vectors for all code descriptions
    2. At query time, vectorise the entity text and find nearest neighbours
    3. Rank by cosine similarity

    Parameters
    ----------
    hf_dataset : str
        HuggingFace dataset to load codes from.
    cache_dir : str, optional
        Cache directory for downloaded data.
    ngram_range : tuple
        Character n-gram range for TF-IDF. Default (3, 4) balances
        specificity and recall for medical terms.
    top_k : int
        Default number of candidates to return.
    """

    def __init__(
        self,
        hf_dataset: str = "atta00/icd10-codes",
        cache_dir: Optional[str] = None,
        ngram_range: Tuple[int, int] = (3, 4),
        top_k: int = 5,
    ):
        self.top_k = top_k
        self._ngram_range = ngram_range
        self._codes: Dict[str, ICDCode] = {}
        self._descriptions: List[str] = []
        self._code_keys: List[str] = []
        self._tfidf_matrix = None
        self._vectoriser = None

        self._load_codes(hf_dataset, cache_dir)
        self._build_tfidf_index()

    def _load_codes(self, hf_dataset: str, cache_dir: Optional[str]) -> None:
        """Load ICD-10-CM codes from HuggingFace dataset."""
        try:
            from datasets import load_dataset
            logger.info("Loading ICD-10-CM codes from '%s'...", hf_dataset)
            ds = load_dataset(hf_dataset, split="train", cache_dir=cache_dir)
            for row in ds:
                code = row.get("code", row.get("Code", "")).strip()
                desc = row.get("description", row.get("Description", "")).strip()
                if not code or not desc:
                    continue
                self._codes[code] = ICDCode(
                    code=code,
                    description=desc,
                    chapter=row.get("chapter"),
                    section=row.get("section"),
                    category=row.get("category"),
                )
            logger.info("Loaded %d ICD-10-CM codes.", len(self._codes))
        except Exception as e:
            logger.warning(
                "Could not load from HuggingFace ('%s'): %s. "
                "Falling back to built-in high-frequency codes.",
                hf_dataset, e,
            )
            self._load_builtin_fallback()

    def _load_builtin_fallback(self) -> None:
        """Minimal fallback with high-frequency codes if HF download fails."""
        from src.clinical._icd_fallback import FALLBACK_CODES
        for code, desc in FALLBACK_CODES.items():
            self._codes[code] = ICDCode(code=code, description=desc)
        logger.info("Loaded %d fallback ICD codes.", len(self._codes))

    def _build_tfidf_index(self) -> None:
        """
        Build TF-IDF character n-gram index over all code descriptions.

        This follows SciSpacy's EntityLinker approach: character n-grams
        capture morphological patterns in medical terminology (e.g. "-itis",
        "-emia", "cardio-") that are critical for matching.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._code_keys = list(self._codes.keys())
        self._descriptions = [
            self._codes[k].description.lower() for k in self._code_keys
        ]

        self._vectoriser = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self._ngram_range,
            min_df=1,
            max_df=0.95,
            dtype=np.float32,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectoriser.fit_transform(self._descriptions)
        logger.info(
            "Built TF-IDF index: %d codes, %d features.",
            self._tfidf_matrix.shape[0], self._tfidf_matrix.shape[1],
        )

    def lookup_code(self, code: str) -> Optional[ICDCode]:
        """Look up a single ICD code by its code string."""
        return self._codes.get(code)

    def match_entity(
        self,
        entity_text: str,
        top_k: Optional[int] = None,
        min_score: float = 0.2,
    ) -> List[ICDMatch]:
        """
        Find candidate ICD-10-CM codes for an entity using TF-IDF similarity.

        Parameters
        ----------
        entity_text : str
            Clinical entity text (e.g. "congestive heart failure").
        top_k : int, optional
            Number of candidates to return. Defaults to self.top_k.
        min_score : float
            Minimum cosine similarity threshold.

        Returns
        -------
        list of ICDMatch
            Candidates ranked by similarity score.
        """
        if top_k is None:
            top_k = self.top_k

        if not entity_text or not entity_text.strip():
            return []

        normalised = entity_text.lower().strip()

        # Strategy 1: Exact description match
        for code_key, icd in self._codes.items():
            if icd.description.lower() == normalised:
                return [ICDMatch(
                    code=icd.code,
                    description=icd.description,
                    score=1.0,
                    match_type="exact",
                )]

        # Strategy 2: TF-IDF cosine similarity
        query_vec = self._vectoriser.transform([normalised])
        scores = (self._tfidf_matrix @ query_vec.T).toarray().ravel()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k * 2]  # over-fetch, then filter

        matches = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            code_key = self._code_keys[idx]
            icd = self._codes[code_key]
            matches.append(ICDMatch(
                code=icd.code,
                description=icd.description,
                score=score,
                match_type="tfidf",
            ))
            if len(matches) >= top_k:
                break

        return matches

    def match_entities_batch(
        self,
        entities: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Match a batch of entities to ICD codes.

        Parameters
        ----------
        entities : list of dict
            Each dict should have 'text' and optionally 'label'.

        Returns
        -------
        list of dict
            Same entities with added 'icd_codes' key.
        """
        results = []
        for ent in entities:
            text = ent.get("text", "")
            codes = self.match_entity(text, top_k=top_k)
            enriched = dict(ent)
            enriched["icd_codes"] = [c.to_dict() for c in codes]
            results.append(enriched)
        return results

    @property
    def num_codes(self) -> int:
        """Total number of ICD codes in the index."""
        return len(self._codes)
