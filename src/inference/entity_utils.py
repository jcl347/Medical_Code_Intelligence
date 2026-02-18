"""
Entity data structures and post-processing utilities.

Separated from predictor.py to avoid a hard dependency on torch/transformers
for code that only needs the NEREntity dataclass or post-processing logic.
"""

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class NEREntity:
    """A single recognized entity."""
    text: str
    label: str
    start_char: int
    end_char: int
    score: float

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start_char,
            "end": self.end_char,
            "score": round(self.score, 4),
        }


# ---------------------------------------------------------------------------
# Entity post-processing: filter garbage + merge fragments
# ---------------------------------------------------------------------------

# Tokens that should never be standalone entities (function words, articles, etc.)
_ENTITY_STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "by",
    "with", "from", "at", "as", "is", "are", "was", "were", "be", "been",
    "not", "no", "nor", "but", "so", "if", "it", "its", "that", "this",
    "than", "then", "has", "had", "have", "do", "does", "did", "will",
    "can", "may", "shall", "would", "could", "should", "type", "due",
})

# Regex for entities that are just punctuation or whitespace
_JUNK_ENTITY_RE = re.compile(r"^[\s\W]+$")


def post_process_entities(
    entities: List[NEREntity],
    text: str,
    merge_gap: int = 5,
) -> List[NEREntity]:
    """
    Clean up raw NER pipeline output.

    1. **Filter garbage**: remove entities that are stopwords, single
       lowercase characters, or pure punctuation/whitespace.
    2. **Merge adjacent fragments**: if two entities of the same label are
       separated by at most ``merge_gap`` characters of whitespace or
       function words, merge them into one span.

    Parameters
    ----------
    entities : list of NEREntity
        Raw entities from the HuggingFace NER pipeline.
    text : str
        Original input text (used for gap inspection during merging).
    merge_gap : int
        Maximum character distance between two entities to consider merging.
    """
    if not entities:
        return entities

    # --- Step 1: Filter garbage entities ---
    filtered = []
    for ent in entities:
        stripped = ent.text.strip()

        # Empty or pure junk (whitespace/punctuation only)
        if not stripped or _JUNK_ENTITY_RE.match(stripped):
            continue

        # Single lowercase character
        if len(stripped) <= 1 and stripped.islower():
            continue

        # Preserve uppercase abbreviations (AS, AT, WAS = disease abbreviations)
        if stripped.isupper() and len(stripped) >= 2:
            filtered.append(ent)
            continue

        # Standalone stopword (case-insensitive)
        if stripped.lower() in _ENTITY_STOPWORDS:
            continue

        filtered.append(ent)

    if not filtered:
        return filtered

    # --- Step 2: Merge adjacent same-label fragments ---
    filtered.sort(key=lambda e: e.start_char)
    merged = [filtered[0]]

    for ent in filtered[1:]:
        prev = merged[-1]

        # Same label and close enough?
        if ent.label == prev.label:
            gap = ent.start_char - prev.end_char
            if 0 <= gap <= merge_gap:
                gap_text = text[prev.end_char:ent.start_char]
                # Only merge if the gap is whitespace or a function word
                gap_words = gap_text.strip().lower().split()
                if all(w in _ENTITY_STOPWORDS or w == "" for w in gap_words) or gap_text.strip() == "":
                    # Merge: extend the previous entity to cover both spans
                    merged_text = text[prev.start_char:ent.end_char]
                    merged[-1] = NEREntity(
                        text=merged_text,
                        label=prev.label,
                        start_char=prev.start_char,
                        end_char=ent.end_char,
                        score=max(prev.score, ent.score),
                    )
                    continue

        merged.append(ent)

    return merged
