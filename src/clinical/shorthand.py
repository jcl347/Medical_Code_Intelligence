"""
Data-driven physician shorthand and medical abbreviation expansion.

Loads abbreviations from three public sources (tried in order):
1. Medical Abbreviation Meta-Inventory (104K abbreviations, CC-BY-4.0)
   - Source: Zenodo 10.5281/zenodo.4567594
   - Paper: Nature Scientific Data, 2021
2. MEDIALpy package (MIT, pip-installable)
   - Source: github.com/imantsm/medical_abbreviations
3. Built-in fallback (~280 hand-curated clinical abbreviations)
   - From Stedman's, JCAHO "Do Not Use" list, common EHR patterns

For ambiguous abbreviations (23% of Meta-Inventory have multiple senses),
two disambiguation strategies are supported:
- "preferred": Use the Preferred Long Form (PLF) from Meta-Inventory (fast)
- "transformer": Use MeDAL ELECTRA model for contextual disambiguation
- "context_rules": Use hand-crafted regex rules (legacy, fast)

Character offsets are preserved through expansion for NER alignment.
"""

import csv
import io
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meta-Inventory download configuration
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID = "4567594"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "medical_code_intelligence",
)

# Common English words that should NOT be expanded even if they appear
# as abbreviations in the Meta-Inventory (prevents false positives).
_ENGLISH_STOPWORDS = frozenset({
    "a", "an", "as", "at", "be", "by", "do", "go", "he", "i", "if", "in",
    "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us",
    "we", "am", "are", "was", "has", "had", "the", "and", "for", "but",
    "not", "you", "all", "can", "her", "him", "his", "how", "its", "may",
    "new", "now", "old", "see", "two", "way", "who", "did", "get", "got",
    "let", "say", "she", "too", "use", "man", "men", "end", "set", "run",
    "add", "big", "own", "off", "top", "yes", "red", "per", "nor",
})


def _download_meta_inventory(cache_dir: str) -> Optional[str]:
    """
    Download the Meta-Inventory CSV from Zenodo and cache locally.

    Returns the path to the cached CSV, or None if download fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cached_path = os.path.join(cache_dir, "meta_inventory.csv")

    if os.path.exists(cached_path):
        logger.info("Using cached Meta-Inventory: %s", cached_path)
        return cached_path

    try:
        logger.info("Querying Zenodo API for Meta-Inventory files...")
        req = urllib.request.Request(ZENODO_API_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            record = json.loads(resp.read().decode("utf-8"))

        # Find the CSV file in the record
        csv_url = None
        for f in record.get("files", []):
            key = f.get("key", "")
            if key.endswith(".csv"):
                csv_url = f.get("links", {}).get("self")
                break

        if csv_url is None:
            logger.warning("No CSV file found in Zenodo record %s.", ZENODO_RECORD_ID)
            return None

        logger.info("Downloading Meta-Inventory from %s ...", csv_url)
        urllib.request.urlretrieve(csv_url, cached_path)
        logger.info("Cached Meta-Inventory to %s", cached_path)
        return cached_path

    except Exception as e:
        logger.warning("Failed to download Meta-Inventory: %s", e)
        return None


def _parse_meta_inventory(csv_path: str, min_length: int = 2) -> Tuple[Dict, Dict]:
    """
    Parse the Meta-Inventory CSV into abbreviation dictionaries.

    Returns
    -------
    abbreviations : dict
        Mapping of abbreviation (lowercase) -> preferred long form.
    sense_inventory : dict
        Mapping of abbreviation (lowercase) -> list of all known senses.
    """
    abbreviations = {}
    sense_inventory = {}

    try:
        import pandas as pd
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

        # Auto-detect column names (handle different formats)
        sf_col = next((c for c in df.columns if c.upper() in ("SF", "SHORT_FORM", "ABBREVIATION")), None)
        lf_col = next((c for c in df.columns if c.upper() in ("LF", "LONG_FORM", "EXPANSION")), None)
        plf_col = next((c for c in df.columns if c.upper() in ("PLF", "PREFERRED_LONG_FORM")), None)

        if sf_col is None or lf_col is None:
            logger.warning("Meta-Inventory CSV missing expected columns (SF/LF). Found: %s", list(df.columns))
            return {}, {}

        for sf, group in df.groupby(sf_col):
            sf_lower = str(sf).strip().lower()

            # Filter: skip too-short or common English words
            if len(sf_lower) < min_length:
                continue
            if sf_lower in _ENGLISH_STOPWORDS:
                continue

            senses = [str(lf).strip() for lf in group[lf_col].unique() if str(lf).strip()]
            if not senses:
                continue

            sense_inventory[sf_lower] = senses

            # Use PLF (Preferred Long Form) if available, else first sense
            if plf_col and plf_col in df.columns:
                plf_values = [str(v).strip() for v in group[plf_col].unique() if str(v).strip()]
                abbreviations[sf_lower] = plf_values[0] if plf_values else senses[0]
            else:
                abbreviations[sf_lower] = senses[0]

        logger.info(
            "Parsed Meta-Inventory: %d abbreviations, %d ambiguous (multi-sense).",
            len(abbreviations),
            sum(1 for s in sense_inventory.values() if len(s) > 1),
        )

    except ImportError:
        logger.warning("pandas not available; falling back to csv module for Meta-Inventory.")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                _senses: Dict[str, List[str]] = {}
                _preferred: Dict[str, str] = {}
                for row in reader:
                    sf = (row.get("SF") or row.get("Short_Form") or row.get("abbreviation") or "").strip().lower()
                    lf = (row.get("LF") or row.get("Long_Form") or row.get("expansion") or "").strip()
                    plf = (row.get("PLF") or row.get("Preferred_Long_Form") or "").strip()
                    if not sf or not lf or len(sf) < min_length or sf in _ENGLISH_STOPWORDS:
                        continue
                    _senses.setdefault(sf, [])
                    if lf not in _senses[sf]:
                        _senses[sf].append(lf)
                    if plf and sf not in _preferred:
                        _preferred[sf] = plf

                sense_inventory = _senses
                for sf, senses in _senses.items():
                    abbreviations[sf] = _preferred.get(sf, senses[0])

            logger.info("Parsed Meta-Inventory (csv module): %d abbreviations.", len(abbreviations))
        except Exception as e:
            logger.warning("Failed to parse Meta-Inventory CSV: %s", e)

    except Exception as e:
        logger.warning("Failed to parse Meta-Inventory: %s", e)

    return abbreviations, sense_inventory


def _load_medialpy_abbreviations() -> Dict[str, str]:
    """
    Load abbreviations from the MEDIALpy package (pip install medialpy).

    Returns abbreviation -> expansion dict, or empty dict if not installed.
    """
    try:
        import medialpy
        abbreviations = {}
        # MEDIALpy stores abbreviations in alphabetical CSV files
        # Access through the package API
        for alpha_range in ["#-A", "B", "C", "D", "E", "F", "G", "H", "I",
                            "J-K", "L", "M", "N", "O", "P-Q", "R", "S",
                            "T", "U-Z"]:
            try:
                # medialpy.search returns matches
                pass  # The API is lookup-based, not enumerable
            except Exception:
                continue

        # MEDIALpy is lookup-only (not enumerable), so we use it as a
        # fallback resolver rather than a bulk dictionary source
        logger.info("MEDIALpy available for on-demand abbreviation lookup.")
        return {}  # Can't enumerate; used on-demand via resolve_with_medialpy()
    except ImportError:
        return {}


def _resolve_with_medialpy(abbreviation: str) -> Optional[str]:
    """Try to resolve an abbreviation using MEDIALpy."""
    try:
        import medialpy
        term = medialpy.find(abbreviation.upper())
        if term and hasattr(term, "meaning") and term.meaning:
            meanings = term.meaning
            if isinstance(meanings, list):
                return meanings[0].lower()
            return str(meanings).lower()
    except Exception:
        pass
    return None


class ShorthandExpander:
    """
    Data-driven physician shorthand expansion.

    Loads abbreviations from public sources in priority order:
    1. Meta-Inventory (104K abbreviations from Zenodo, CC-BY-4.0)
    2. MEDIALpy package (if pip-installed, MIT license)
    3. Built-in fallback (~280 hand-curated clinical abbreviations)

    Custom abbreviations and built-in clinical abbreviations always override
    the Meta-Inventory, since the built-in set is specifically curated for
    clinical note shorthand.

    For ambiguous abbreviations, supports three disambiguation strategies:
    - "preferred": Use Preferred Long Form from Meta-Inventory (default, fast)
    - "transformer": Use MeDAL ELECTRA model for contextual disambiguation
    - "context_rules": Use hand-crafted regex patterns (legacy)

    Parameters
    ----------
    source : str
        Abbreviation source: "auto" (try Meta-Inventory, then fallback),
        "meta_inventory", "builtin", or a path to a CSV file.
    cache_dir : str, optional
        Cache directory for downloaded data.
    disambiguation : str
        Strategy for ambiguous abbreviations: "preferred", "transformer",
        or "context_rules".
    custom_abbreviations : dict, optional
        Additional abbreviation -> expansion mappings (highest priority).
    min_abbreviation_length : int
        Minimum abbreviation length from Meta-Inventory (default 2).
    expand_in_place : bool
        If True, replace abbreviations in text. If False, only annotate.
    """

    def __init__(
        self,
        source: str = "auto",
        cache_dir: Optional[str] = None,
        disambiguation: str = "preferred",
        custom_abbreviations: Optional[Dict[str, str]] = None,
        min_abbreviation_length: int = 2,
        expand_in_place: bool = True,
    ):
        self.expand_in_place = expand_in_place
        self._disambiguation = disambiguation
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._min_length = min_abbreviation_length
        self._sense_inventory: Dict[str, List[str]] = {}
        self._disambiguator = None
        self._medialpy_available = False

        # Load abbreviations from the specified source
        self.abbreviations = self._load_abbreviations(source)
        self._source_loaded = source

        # Apply custom overrides (highest priority)
        if custom_abbreviations:
            self.abbreviations.update(
                {k.lower(): v for k, v in custom_abbreviations.items()}
            )

        # Load context rules for "context_rules" disambiguation
        from src.clinical._shorthand_fallback import FALLBACK_CONTEXT_RULES
        self.context_rules = list(FALLBACK_CONTEXT_RULES)

        # Check MEDIALpy availability
        try:
            import medialpy
            self._medialpy_available = True
        except ImportError:
            self._medialpy_available = False

        # Build regex pattern for matching
        self._build_pattern()
        logger.info(
            "ShorthandExpander: %d abbreviations loaded (source=%s, disambiguation=%s).",
            len(self.abbreviations), source, disambiguation,
        )

    def _load_abbreviations(self, source: str) -> Dict[str, str]:
        """Load abbreviations from the specified source."""
        from src.clinical._shorthand_fallback import FALLBACK_ABBREVIATIONS

        if source == "builtin":
            return dict(FALLBACK_ABBREVIATIONS)

        meta_inventory_abbrs = {}
        meta_sense_inventory = {}

        if source in ("auto", "meta_inventory"):
            csv_path = _download_meta_inventory(self._cache_dir)
            if csv_path:
                meta_inventory_abbrs, meta_sense_inventory = _parse_meta_inventory(
                    csv_path, min_length=self._min_length,
                )

        elif os.path.isfile(source):
            # Load from a local CSV file
            meta_inventory_abbrs, meta_sense_inventory = _parse_meta_inventory(
                source, min_length=self._min_length,
            )

        self._sense_inventory = meta_sense_inventory

        if meta_inventory_abbrs:
            # Start with Meta-Inventory, then override with built-in
            # (built-in abbreviations are clinical-note-specific and take priority)
            combined = dict(meta_inventory_abbrs)
            combined.update(FALLBACK_ABBREVIATIONS)
            return combined
        else:
            if source not in ("auto", "builtin"):
                logger.warning(
                    "Could not load from source '%s'. Using built-in fallback.", source,
                )
            return dict(FALLBACK_ABBREVIATIONS)

    def _build_pattern(self) -> None:
        """Build regex pattern for matching abbreviations as whole words."""
        escaped = [
            re.escape(abbr)
            for abbr in sorted(self.abbreviations, key=len, reverse=True)
        ]
        self._pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE,
        )

    def _resolve_expansion(self, abbr: str, context_before: str) -> str:
        """
        Resolve the best expansion for an abbreviation.

        Tries disambiguation strategies in order based on self._disambiguation.
        """
        abbr_lower = abbr.lower()

        # Strategy: context_rules
        if self._disambiguation == "context_rules":
            for rule_abbr, pattern, expansion in self.context_rules:
                if rule_abbr == abbr_lower:
                    if re.search(pattern, context_before, re.IGNORECASE):
                        return expansion

        # Strategy: transformer (lazy-loaded MeDAL model)
        if self._disambiguation == "transformer":
            senses = self._sense_inventory.get(abbr_lower, [])
            if len(senses) > 1 and self._disambiguator is not None:
                try:
                    result = self._disambiguator.disambiguate_from_context(
                        context_before, abbr, senses,
                    )
                    if result:
                        return result
                except Exception:
                    pass  # Fall through to dictionary lookup

        # Strategy: preferred (default) — just use the dictionary
        expansion = self.abbreviations.get(abbr_lower)
        if expansion:
            return expansion

        # Last resort: try MEDIALpy for unknown abbreviations
        if self._medialpy_available:
            medialpy_result = _resolve_with_medialpy(abbr)
            if medialpy_result:
                return medialpy_result

        return abbr

    @property
    def disambiguator(self):
        """Lazy-load the MeDAL disambiguation model."""
        if self._disambiguator is None and self._disambiguation == "transformer":
            from src.clinical.abbreviation_disambiguator import AbbreviationDisambiguator
            self._disambiguator = AbbreviationDisambiguator()
        return self._disambiguator

    def expand(self, text: str) -> str:
        """
        Expand abbreviations in text, returning the expanded string.

        Parameters
        ----------
        text : str
            Raw clinical text with physician shorthand.

        Returns
        -------
        str
            Text with abbreviations expanded.
        """
        if not self.expand_in_place:
            return text

        # Ensure disambiguator is loaded if needed
        if self._disambiguation == "transformer" and self._disambiguator is None:
            try:
                self._disambiguator = self.disambiguator
            except Exception:
                pass

        def _replace(match):
            abbr = match.group(0)
            context_before = text[:match.start()].split()
            context_word = context_before[-1] if context_before else ""
            return self._resolve_expansion(abbr, context_word)

        return self._pattern.sub(_replace, text)

    def expand_with_offsets(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Expand abbreviations and track character offset changes.

        Returns
        -------
        expanded_text : str
            Text with abbreviations expanded.
        offset_map : list of dict
            Each dict: {original_start, original_end, expanded_start, expanded_end,
                        abbreviation, expansion}
        """
        # Ensure disambiguator is loaded if needed
        if self._disambiguation == "transformer" and self._disambiguator is None:
            try:
                self._disambiguator = self.disambiguator
            except Exception:
                pass

        offset_map = []
        result_parts = []
        last_end = 0
        cumulative_shift = 0

        for match in self._pattern.finditer(text):
            abbr = match.group(0)

            # Resolve expansion
            context_before = text[:match.start()].split()
            context_word = context_before[-1] if context_before else ""
            expansion = self._resolve_expansion(abbr, context_word)

            result_parts.append(text[last_end:match.start()])
            expanded_start = match.start() + cumulative_shift
            result_parts.append(expansion)

            shift = len(expansion) - len(abbr)
            offset_map.append({
                "original_start": match.start(),
                "original_end": match.end(),
                "expanded_start": expanded_start,
                "expanded_end": expanded_start + len(expansion),
                "abbreviation": abbr,
                "expansion": expansion,
            })

            cumulative_shift += shift
            last_end = match.end()

        result_parts.append(text[last_end:])
        return "".join(result_parts), offset_map

    def identify_abbreviations(self, text: str) -> List[Dict]:
        """
        Identify abbreviations in text without expanding them.

        Useful for annotation or highlighting in a UI.
        """
        found = []
        for match in self._pattern.finditer(text):
            abbr = match.group(0)
            abbr_lower = abbr.lower()
            expansion = self.abbreviations.get(abbr_lower, "unknown")

            entry = {
                "abbreviation": abbr,
                "expansion": expansion,
                "start": match.start(),
                "end": match.end(),
            }

            # Add sense information if available
            senses = self._sense_inventory.get(abbr_lower, [])
            if len(senses) > 1:
                entry["ambiguous"] = True
                entry["senses"] = senses

            found.append(entry)
        return found

    @property
    def num_abbreviations(self) -> int:
        """Total number of abbreviations in the dictionary."""
        return len(self.abbreviations)

    @property
    def num_ambiguous(self) -> int:
        """Number of abbreviations with multiple known senses."""
        return sum(1 for s in self._sense_inventory.values() if len(s) > 1)

    def get_senses(self, abbreviation: str) -> List[str]:
        """Get all known senses for an abbreviation."""
        return self._sense_inventory.get(abbreviation.lower(), [])
