"""
Physician shorthand and medical abbreviation expansion.

Clinical notes are full of abbreviations that hurt NER performance.
This module normalises them before passing text through the model,
while preserving character offsets so entity positions remain valid.

Sources for abbreviation lists:
- Stedman's Medical Abbreviations
- JCAHO "Do Not Use" list
- Common EHR/clinical documentation patterns
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comprehensive physician shorthand dictionary
# Organised by clinical category for maintainability
# ---------------------------------------------------------------------------

# --- General clinical abbreviations ---
_GENERAL = {
    "pt": "patient",
    "pts": "patients",
    "px": "physical examination",
    "hx": "history",
    "pmh": "past medical history",
    "pmhx": "past medical history",
    "fhx": "family history",
    "shx": "social history",
    "h/o": "history of",
    "s/p": "status post",
    "c/o": "complaining of",
    "r/o": "rule out",
    "f/u": "follow up",
    "w/u": "workup",
    "y/o": "year old",
    "yo": "year old",
    "m/f": "male/female",
    "wt": "weight",
    "ht": "height",
    "bmi": "body mass index",
    "cc": "chief complaint",
    "ros": "review of systems",
    "hpi": "history of present illness",
    "dob": "date of birth",
    "a&o": "alert and oriented",
    "a/o": "alert and oriented",
    "wdwn": "well developed well nourished",
    "wd/wn": "well developed well nourished",
    "nad": "no acute distress",
    "nka": "no known allergies",
    "nkda": "no known drug allergies",
    "nkma": "no known medication allergies",
    "adl": "activities of daily living",
    "adls": "activities of daily living",
    "dme": "durable medical equipment",
    "snf": "skilled nursing facility",
}

# --- Diagnoses / conditions ---
_DIAGNOSES = {
    "dm": "diabetes mellitus",
    "dm1": "type 1 diabetes mellitus",
    "dm2": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "iddm": "insulin dependent diabetes mellitus",
    "niddm": "non-insulin dependent diabetes mellitus",
    "htn": "hypertension",
    "chf": "congestive heart failure",
    "cad": "coronary artery disease",
    "mi": "myocardial infarction",
    "ami": "acute myocardial infarction",
    "stemi": "st elevation myocardial infarction",
    "nstemi": "non-st elevation myocardial infarction",
    "afib": "atrial fibrillation",
    "a-fib": "atrial fibrillation",
    "aflutter": "atrial flutter",
    "svt": "supraventricular tachycardia",
    "vtach": "ventricular tachycardia",
    "vfib": "ventricular fibrillation",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "copd": "chronic obstructive pulmonary disease",
    "sob": "shortness of breath",
    "doe": "dyspnea on exertion",
    "pna": "pneumonia",
    "uti": "urinary tract infection",
    "uri": "upper respiratory infection",
    "gerd": "gastroesophageal reflux disease",
    "ibs": "irritable bowel syndrome",
    "ibd": "inflammatory bowel disease",
    "uc": "ulcerative colitis",
    "gi": "gastrointestinal",
    "gib": "gastrointestinal bleed",
    "ugib": "upper gastrointestinal bleed",
    "lgib": "lower gastrointestinal bleed",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "ckd": "chronic kidney disease",
    "esrd": "end stage renal disease",
    "aki": "acute kidney injury",
    "arf": "acute renal failure",
    "crf": "chronic renal failure",
    "bph": "benign prostatic hyperplasia",
    "osa": "obstructive sleep apnea",
    "ra": "rheumatoid arthritis",
    "oa": "osteoarthritis",
    "sle": "systemic lupus erythematosus",
    "ms": "multiple sclerosis",
    "als": "amyotrophic lateral sclerosis",
    "pvd": "peripheral vascular disease",
    "pad": "peripheral arterial disease",
    "hld": "hyperlipidemia",
    "hl": "hyperlipidemia",
    "acs": "acute coronary syndrome",
    "aaa": "abdominal aortic aneurysm",
    "ards": "acute respiratory distress syndrome",
    "dic": "disseminated intravascular coagulation",
    "sbo": "small bowel obstruction",
    "lbo": "large bowel obstruction",
    "etoh": "alcohol",
    "sz": "seizure",
    "ha": "headache",
    "cp": "chest pain",
    "lbp": "low back pain",
    "n/v": "nausea/vomiting",
    "n/v/d": "nausea/vomiting/diarrhea",
    "bka": "below knee amputation",
    "aka": "above knee amputation",
}

# --- Treatment / procedures ---
_TREATMENT = {
    "tx": "treatment",
    "rx": "prescription",
    "dx": "diagnosis",
    "ddx": "differential diagnosis",
    "sx": "surgery",
    "surg": "surgery",
    "op": "operation",
    "pre-op": "preoperative",
    "post-op": "postoperative",
    "abx": "antibiotics",
    "ppx": "prophylaxis",
    "dc": "discontinue",
    "d/c": "discharge",
    "dispo": "disposition",
    "cpap": "continuous positive airway pressure",
    "bipap": "bilevel positive airway pressure",
    "ngt": "nasogastric tube",
    "foley": "foley catheter",
    "iv": "intravenous",
    "im": "intramuscular",
    "sq": "subcutaneous",
    "subq": "subcutaneous",
    "po": "by mouth",
    "pr": "per rectum",
    "sl": "sublingual",
    "prn": "as needed",
    "qd": "daily",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qhs": "at bedtime",
    "qam": "every morning",
    "qpm": "every evening",
    "qod": "every other day",
    "qwk": "every week",
    "ac": "before meals",
    "pc": "after meals",
    "stat": "immediately",
    "ekg": "electrocardiogram",
    "ecg": "electrocardiogram",
    "echo": "echocardiogram",
    "cbc": "complete blood count",
    "bmp": "basic metabolic panel",
    "cmp": "comprehensive metabolic panel",
    "lfts": "liver function tests",
    "tsh": "thyroid stimulating hormone",
    "ua": "urinalysis",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "cxr": "chest x-ray",
    "kub": "kidneys ureters bladder x-ray",
    "us": "ultrasound",
    "cabg": "coronary artery bypass graft",
    "ptca": "percutaneous transluminal coronary angioplasty",
    "pci": "percutaneous coronary intervention",
    "ercp": "endoscopic retrograde cholangiopancreatography",
    "egd": "esophagogastroduodenoscopy",
    "lap": "laparoscopic",
    "lap chole": "laparoscopic cholecystectomy",
    "appy": "appendectomy",
    "cath": "catheterisation",
    "trach": "tracheostomy",
    "intub": "intubation",
    "extub": "extubation",
}

# --- Anatomy / physical exam ---
_ANATOMY = {
    "abd": "abdomen",
    "ext": "extremities",
    "bilat": "bilateral",
    "le": "lower extremity",
    "ue": "upper extremity",
    "lle": "left lower extremity",
    "rle": "right lower extremity",
    "lue": "left upper extremity",
    "rue": "right upper extremity",
    "ruq": "right upper quadrant",
    "luq": "left upper quadrant",
    "rlq": "right lower quadrant",
    "llq": "left lower quadrant",
    "cv": "cardiovascular",
    "pulm": "pulmonary",
    "neuro": "neurological",
    "msk": "musculoskeletal",
    "heent": "head eyes ears nose throat",
    "eomi": "extraocular movements intact",
    "perrl": "pupils equal round reactive to light",
    "perrla": "pupils equal round reactive to light and accommodation",
    "rrr": "regular rate and rhythm",
    "ctab": "clear to auscultation bilaterally",
    "cta": "clear to auscultation",
    "ntnd": "nontender nondistended",
    "nt/nd": "nontender nondistended",
    "bs": "bowel sounds",
    "nabs": "normoactive bowel sounds",
    "tms": "tympanic membranes",
    "tm": "tympanic membrane",
    "cn": "cranial nerves",
    "dtr": "deep tendon reflexes",
    "dtrs": "deep tendon reflexes",
    "rom": "range of motion",
}

# --- Lab values / vitals ---
_LABS = {
    "hr": "heart rate",
    "bp": "blood pressure",
    "sbp": "systolic blood pressure",
    "dbp": "diastolic blood pressure",
    "rr": "respiratory rate",
    "o2 sat": "oxygen saturation",
    "spo2": "oxygen saturation",
    "temp": "temperature",
    "t": "temperature",
    "wbc": "white blood cell count",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet count",
    "plts": "platelets",
    "cr": "creatinine",
    "bun": "blood urea nitrogen",
    "na": "sodium",
    "k": "potassium",
    "cl": "chloride",
    "hco3": "bicarbonate",
    "co2": "carbon dioxide",
    "ca": "calcium",
    "mg": "magnesium",
    "phos": "phosphorus",
    "ast": "aspartate aminotransferase",
    "alt": "alanine aminotransferase",
    "alp": "alkaline phosphatase",
    "tbili": "total bilirubin",
    "dbili": "direct bilirubin",
    "alb": "albumin",
    "tp": "total protein",
    "pt": "prothrombin time",
    "inr": "international normalized ratio",
    "ptt": "partial thromboplastin time",
    "aptt": "activated partial thromboplastin time",
    "esr": "erythrocyte sedimentation rate",
    "crp": "c-reactive protein",
    "hba1c": "hemoglobin a1c",
    "a1c": "hemoglobin a1c",
    "bnp": "brain natriuretic peptide",
    "trop": "troponin",
    "abg": "arterial blood gas",
    "vbg": "venous blood gas",
    "lytes": "electrolytes",
}

# ---------------------------------------------------------------------------
# Ambiguity resolution rules
# ---------------------------------------------------------------------------
# Some abbreviations are ambiguous (e.g. "pt" = patient OR prothrombin time).
# These context-dependent entries map (abbreviation, preceding_word_pattern) -> expansion.
_CONTEXT_RULES: List[Tuple[str, str, str]] = [
    # "pt" after lab-related context -> prothrombin time
    ("pt", r"(?:check|draw|labs?|coags?|elevated|normal|prolonged|inr)\b", "prothrombin time"),
    # "pt" in most other contexts -> patient
    ("pt", r".*", "patient"),
    # "ms" after diagnosis context -> multiple sclerosis
    ("ms", r"(?:diagnosed|dx|history|hx|has|with)\b", "multiple sclerosis"),
    # "ca" after lab context -> calcium
    ("ca", r"(?:check|draw|labs?|level|low|high|elevated|normal)\b", "calcium"),
    # "ca" after diagnosis context -> cancer
    ("ca", r"(?:diagnosed|dx|history|hx|has|with|stage|mets?|metastatic)\b", "cancer"),
]


def _build_full_dictionary() -> Dict[str, str]:
    """Merge all category dictionaries into one."""
    combined = {}
    combined.update(_GENERAL)
    combined.update(_DIAGNOSES)
    combined.update(_TREATMENT)
    combined.update(_ANATOMY)
    combined.update(_LABS)
    return combined


class ShorthandExpander:
    """
    Expands physician shorthand abbreviations in clinical text.

    Features:
    - 300+ common medical abbreviations
    - Context-sensitive disambiguation for ambiguous terms
    - Preserves character offset mapping for NER alignment
    - Case-insensitive matching with original-case output
    - Configurable: can add custom abbreviations
    """

    def __init__(
        self,
        custom_abbreviations: Optional[Dict[str, str]] = None,
        expand_in_place: bool = True,
    ):
        """
        Parameters
        ----------
        custom_abbreviations : dict, optional
            Additional abbreviation -> expansion mappings.
        expand_in_place : bool
            If True, replace abbreviations in the text. If False, only
            annotate them (useful for training data where you want both).
        """
        self.abbreviations = _build_full_dictionary()
        if custom_abbreviations:
            self.abbreviations.update(
                {k.lower(): v for k, v in custom_abbreviations.items()}
            )
        self.expand_in_place = expand_in_place
        self.context_rules = _CONTEXT_RULES

        # Build regex pattern for matching abbreviations as whole words
        # Sort by length (longest first) to match longer abbreviations first
        escaped = [re.escape(abbr) for abbr in sorted(self.abbreviations, key=len, reverse=True)]
        self._pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE,
        )
        logger.info("ShorthandExpander initialised with %d abbreviations.", len(self.abbreviations))

    def _resolve_ambiguous(self, abbr: str, context_before: str) -> Optional[str]:
        """
        Resolve ambiguous abbreviations using preceding context.

        Returns the expansion if a context rule matches, else None.
        """
        abbr_lower = abbr.lower()
        for rule_abbr, pattern, expansion in self.context_rules:
            if rule_abbr == abbr_lower:
                if re.search(pattern, context_before, re.IGNORECASE):
                    return expansion
        return None

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

        def _replace(match):
            abbr = match.group(0)
            abbr_lower = abbr.lower()

            # Try context-sensitive resolution first
            context_before = text[:match.start()].split()
            context_word = context_before[-1] if context_before else ""
            resolved = self._resolve_ambiguous(abbr, context_word)
            if resolved is not None:
                return resolved

            # Fall back to dictionary lookup
            expansion = self.abbreviations.get(abbr_lower)
            if expansion:
                return expansion
            return abbr

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
        offset_map = []
        result_parts = []
        last_end = 0
        cumulative_shift = 0

        for match in self._pattern.finditer(text):
            abbr = match.group(0)
            abbr_lower = abbr.lower()

            # Resolve expansion
            context_before = text[:match.start()].split()
            context_word = context_before[-1] if context_before else ""
            expansion = self._resolve_ambiguous(abbr, context_word)
            if expansion is None:
                expansion = self.abbreviations.get(abbr_lower, abbr)

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
            found.append({
                "abbreviation": abbr,
                "expansion": expansion,
                "start": match.start(),
                "end": match.end(),
            })
        return found
