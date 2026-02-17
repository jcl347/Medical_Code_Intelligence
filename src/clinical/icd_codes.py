"""
ICD code knowledge base and entity-to-code mapping.

Provides:
- ICD-10-CM code lookup by description text
- Entity text → candidate ICD code matching via fuzzy/token overlap
- Code hierarchy navigation (chapter → block → category)
- Integration with MedicalCodingPipeline for end-to-end code assignment

The lookup tables are curated from the CMS ICD-10-CM public code set.
For production use, load the full code set from HuggingFace or CMS files.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ICDCode:
    """A single ICD-10-CM or ICD-9-CM code."""
    code: str
    description: str
    is_billable: bool = True
    chapter: Optional[str] = None
    block: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "description": self.description,
            "is_billable": self.is_billable,
            "chapter": self.chapter,
            "block": self.block,
        }


@dataclass
class ICDMatch:
    """A candidate ICD code match for an entity."""
    code: str
    description: str
    score: float
    match_type: str  # "exact", "token_overlap", "prefix"

    def to_dict(self) -> Dict:
        return {
            "code": self.code,
            "description": self.description,
            "score": round(self.score, 4),
            "match_type": self.match_type,
        }


# ---------------------------------------------------------------------------
# Curated high-frequency ICD-10-CM codes for common clinical conditions
# Organised by ICD-10-CM chapter
# ---------------------------------------------------------------------------

# Chapter 1: Certain infectious and parasitic diseases (A00-B99)
_CHAPTER_A_B = {
    "A09": "Infectious gastroenteritis and colitis, unspecified",
    "A41.9": "Sepsis, unspecified organism",
    "A41.01": "Sepsis due to Methicillin susceptible Staphylococcus aureus",
    "A41.02": "Sepsis due to Methicillin resistant Staphylococcus aureus",
    "A49.02": "Methicillin resistant Staphylococcus aureus infection, unspecified site",
    "B18.2": "Chronic viral hepatitis C",
    "B20": "Human immunodeficiency virus [HIV] disease",
    "B34.9": "Viral infection, unspecified",
    "B37.0": "Candidal stomatitis",
    "B95.62": "Methicillin resistant Staphylococcus aureus as the cause of diseases",
    "B96.20": "Unspecified Escherichia coli as the cause of diseases classified elsewhere",
}

# Chapter 2: Neoplasms (C00-D49)
_CHAPTER_C_D49 = {
    "C18.9": "Malignant neoplasm of colon, unspecified",
    "C34.90": "Malignant neoplasm of unspecified part of unspecified bronchus or lung",
    "C50.919": "Malignant neoplasm of unspecified site of unspecified female breast",
    "C61": "Malignant neoplasm of prostate",
    "C64.9": "Malignant neoplasm of unspecified kidney, except renal pelvis",
    "C67.9": "Malignant neoplasm of bladder, unspecified",
    "C71.9": "Malignant neoplasm of brain, unspecified",
    "C78.7": "Secondary malignant neoplasm of liver and intrahepatic bile duct",
    "C79.51": "Secondary malignant neoplasm of bone",
    "C90.00": "Multiple myeloma not having achieved remission",
    "D64.9": "Anemia, unspecified",
}

# Chapter 4: Endocrine, nutritional and metabolic diseases (E00-E89)
_CHAPTER_E = {
    "E03.9": "Hypothyroidism, unspecified",
    "E05.90": "Thyrotoxicosis, unspecified without thyrotoxic crisis or storm",
    "E10.9": "Type 1 diabetes mellitus without complications",
    "E10.65": "Type 1 diabetes mellitus with hyperglycemia",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E11.65": "Type 2 diabetes mellitus with hyperglycemia",
    "E11.22": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
    "E11.40": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
    "E11.621": "Type 2 diabetes mellitus with foot ulcer",
    "E13.9": "Other specified diabetes mellitus without complications",
    "E46": "Unspecified protein-calorie malnutrition",
    "E55.9": "Vitamin D deficiency, unspecified",
    "E66.01": "Morbid (severe) obesity due to excess calories",
    "E78.5": "Hyperlipidemia, unspecified",
    "E78.00": "Pure hypercholesterolemia, unspecified",
    "E83.42": "Hypomagnesemia",
    "E86.0": "Dehydration",
    "E87.1": "Hypo-osmolality and hyponatremia",
    "E87.5": "Hyperkalemia",
    "E87.6": "Hypokalemia",
}

# Chapter 5: Mental, behavioral and neurodevelopmental disorders (F01-F99)
_CHAPTER_F = {
    "F03.90": "Unspecified dementia without behavioral disturbance",
    "F10.20": "Alcohol dependence, uncomplicated",
    "F10.239": "Alcohol dependence with withdrawal, unspecified",
    "F17.210": "Nicotine dependence, cigarettes, uncomplicated",
    "F20.9": "Schizophrenia, unspecified",
    "F31.9": "Bipolar disorder, unspecified",
    "F32.9": "Major depressive disorder, single episode, unspecified",
    "F33.0": "Major depressive disorder, recurrent, mild",
    "F41.1": "Generalized anxiety disorder",
    "F41.9": "Anxiety disorder, unspecified",
    "F43.10": "Post-traumatic stress disorder, unspecified",
}

# Chapter 9: Diseases of the circulatory system (I00-I99)
_CHAPTER_I = {
    "I10": "Essential (primary) hypertension",
    "I11.0": "Hypertensive heart disease with heart failure",
    "I11.9": "Hypertensive heart disease without heart failure",
    "I12.9": "Hypertensive chronic kidney disease with stage 1-4 or unspecified CKD",
    "I13.10": "Hypertensive heart and chronic kidney disease without heart failure",
    "I20.0": "Unstable angina",
    "I21.3": "ST elevation (STEMI) myocardial infarction of unspecified site",
    "I21.4": "Non-ST elevation (NSTEMI) myocardial infarction",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "I25.5": "Ischemic cardiomyopathy",
    "I26.99": "Other pulmonary embolism without acute cor pulmonale",
    "I42.9": "Cardiomyopathy, unspecified",
    "I48.91": "Unspecified atrial fibrillation",
    "I48.0": "Paroxysmal atrial fibrillation",
    "I48.1": "Persistent atrial fibrillation",
    "I48.2": "Chronic atrial fibrillation",
    "I50.9": "Heart failure, unspecified",
    "I50.20": "Unspecified systolic (congestive) heart failure",
    "I50.22": "Chronic systolic (congestive) heart failure",
    "I50.30": "Unspecified diastolic (congestive) heart failure",
    "I50.32": "Chronic diastolic (congestive) heart failure",
    "I63.9": "Cerebral infarction, unspecified",
    "I65.29": "Occlusion and stenosis of unspecified carotid artery",
    "I69.30": "Unspecified sequelae of cerebral infarction",
    "I70.0": "Atherosclerosis of aorta",
    "I73.9": "Peripheral vascular disease, unspecified",
    "I82.409": "Acute embolism and thrombosis of unspecified deep veins of unspecified lower extremity",
}

# Chapter 10: Diseases of the respiratory system (J00-J99)
_CHAPTER_J = {
    "J06.9": "Acute upper respiratory infection, unspecified",
    "J18.9": "Pneumonia, unspecified organism",
    "J18.1": "Lobar pneumonia, unspecified organism",
    "J20.9": "Acute bronchitis, unspecified",
    "J44.1": "Chronic obstructive pulmonary disease with (acute) exacerbation",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "J45.20": "Mild intermittent asthma, uncomplicated",
    "J45.50": "Severe persistent asthma, uncomplicated",
    "J80": "Acute respiratory distress syndrome",
    "J96.00": "Acute respiratory failure, unspecified whether with hypoxia or hypercapnia",
    "J96.01": "Acute respiratory failure with hypoxia",
    "J96.10": "Chronic respiratory failure, unspecified whether with hypoxia or hypercapnia",
}

# Chapter 11: Diseases of the digestive system (K00-K95)
_CHAPTER_K = {
    "K21.0": "Gastro-esophageal reflux disease with esophagitis",
    "K25.9": "Gastric ulcer, unspecified as acute or chronic, without hemorrhage or perforation",
    "K35.80": "Unspecified acute appendicitis",
    "K40.90": "Unilateral inguinal hernia, without obstruction or gangrene, not specified as recurrent",
    "K56.60": "Unspecified intestinal obstruction",
    "K57.30": "Diverticulosis of large intestine without perforation or abscess without bleeding",
    "K57.32": "Diverticulitis of large intestine without perforation or abscess without bleeding",
    "K59.00": "Constipation, unspecified",
    "K70.30": "Alcoholic cirrhosis of liver without ascites",
    "K74.60": "Unspecified cirrhosis of liver",
    "K76.0": "Fatty (change of) liver, not elsewhere classified",
    "K80.20": "Calculus of gallbladder without cholecystitis without obstruction",
    "K85.90": "Acute pancreatitis without necrosis or infection, unspecified",
    "K92.1": "Melena",
    "K92.2": "Gastrointestinal hemorrhage, unspecified",
}

# Chapter 14: Diseases of the genitourinary system (N00-N99)
_CHAPTER_N = {
    "N17.9": "Acute kidney failure, unspecified",
    "N18.1": "Chronic kidney disease, stage 1",
    "N18.2": "Chronic kidney disease, stage 2 (mild)",
    "N18.3": "Chronic kidney disease, stage 3 (moderate)",
    "N18.4": "Chronic kidney disease, stage 4 (severe)",
    "N18.5": "Chronic kidney disease, stage 5",
    "N18.6": "End stage renal disease",
    "N18.9": "Chronic kidney disease, unspecified",
    "N19": "Unspecified kidney failure",
    "N30.00": "Acute cystitis without hematuria",
    "N39.0": "Urinary tract infection, site not specified",
    "N40.0": "Benign prostatic hyperplasia without lower urinary tract symptoms",
}

# Chapter 18: Symptoms, signs and abnormal clinical/lab findings (R00-R99)
_CHAPTER_R = {
    "R00.0": "Tachycardia, unspecified",
    "R00.1": "Bradycardia, unspecified",
    "R04.0": "Epistaxis",
    "R05.9": "Cough, unspecified",
    "R06.00": "Dyspnea, unspecified",
    "R06.02": "Shortness of breath",
    "R07.9": "Chest pain, unspecified",
    "R09.02": "Hypoxemia",
    "R10.9": "Unspecified abdominal pain",
    "R11.0": "Nausea",
    "R11.10": "Vomiting, unspecified",
    "R11.2": "Nausea with vomiting, unspecified",
    "R13.10": "Dysphagia, unspecified",
    "R19.7": "Diarrhea, unspecified",
    "R20.0": "Anesthesia of skin",
    "R26.81": "Unsteadiness on feet",
    "R41.0": "Disorientation, unspecified",
    "R42": "Dizziness and giddiness",
    "R50.9": "Fever, unspecified",
    "R51.9": "Headache, unspecified",
    "R55": "Syncope and collapse",
    "R56.9": "Unspecified convulsions",
    "R60.0": "Localized edema",
    "R63.0": "Anorexia",
    "R63.4": "Abnormal weight loss",
    "R73.9": "Hyperglycemia, unspecified",
    "R79.89": "Other specified abnormal findings of blood chemistry",
    "R68.83": "Chills (without fever)",
    "R53.83": "Other fatigue",
    "R53.81": "Other malaise",
    "R53.1": "Weakness",
    "R41.82": "Altered mental status, unspecified",
    "R31.9": "Hematuria, unspecified",
    "R04.2": "Hemoptysis",
    "R33.9": "Retention of urine, unspecified",
    "R32": "Unspecified urinary incontinence",
}

# Chapter 19: Injury, poisoning (S00-T88)
_CHAPTER_S_T = {
    "S06.0X0A": "Concussion without loss of consciousness, initial encounter",
    "S72.001A": "Fracture of unspecified part of neck of right femur, initial encounter for closed fracture",
    "T78.40XA": "Allergy, unspecified, initial encounter",
    "T81.4XXA": "Infection following a procedure, initial encounter",
    "T84.54XA": "Periprosthetic osteolysis of internal prosthetic joint, initial encounter",
}

# Chapter 21: Factors influencing health status (Z00-Z99)
_CHAPTER_Z = {
    "Z23": "Encounter for immunization",
    "Z51.11": "Encounter for antineoplastic chemotherapy",
    "Z66": "Do not resuscitate",
    "Z79.4": "Long term (current) use of insulin",
    "Z79.82": "Long term (current) use of aspirin",
    "Z79.899": "Other long term (current) drug therapy",
    "Z85.3": "Personal history of malignant neoplasm of breast",
    "Z86.73": "Personal history of transient ischemic attack (TIA)",
    "Z87.891": "Personal history of nicotine dependence",
    "Z87.11": "Personal history of peptic ulcer disease",
    "Z91.19": "Patient's noncompliance with other medical treatment and regimen",
    "Z96.1": "Presence of intraocular lens",
    "Z99.2": "Dependence on renal dialysis",
}


def _build_icd10_lookup() -> Dict[str, str]:
    """Merge all ICD-10-CM chapter dictionaries into a single lookup."""
    combined = {}
    for chapter_dict in [
        _CHAPTER_A_B, _CHAPTER_C_D49, _CHAPTER_E, _CHAPTER_F,
        _CHAPTER_I, _CHAPTER_J, _CHAPTER_K, _CHAPTER_N,
        _CHAPTER_R, _CHAPTER_S_T, _CHAPTER_Z,
    ]:
        combined.update(chapter_dict)
    return combined


# ---------------------------------------------------------------------------
# Common diagnosis/condition text → ICD-10 mapping
# Maps normalized entity text to ICD codes for rapid lookup
# ---------------------------------------------------------------------------

_ENTITY_TO_ICD10 = {
    # Cardiovascular
    "hypertension": ["I10"],
    "essential hypertension": ["I10"],
    "high blood pressure": ["I10"],
    "atrial fibrillation": ["I48.91"],
    "paroxysmal atrial fibrillation": ["I48.0"],
    "persistent atrial fibrillation": ["I48.1"],
    "chronic atrial fibrillation": ["I48.2"],
    "congestive heart failure": ["I50.9"],
    "heart failure": ["I50.9"],
    "systolic heart failure": ["I50.20"],
    "diastolic heart failure": ["I50.30"],
    "coronary artery disease": ["I25.10"],
    "myocardial infarction": ["I21.3"],
    "acute myocardial infarction": ["I21.3"],
    "st elevation myocardial infarction": ["I21.3"],
    "non-st elevation myocardial infarction": ["I21.4"],
    "unstable angina": ["I20.0"],
    "pulmonary embolism": ["I26.99"],
    "deep vein thrombosis": ["I82.409"],
    "cerebrovascular accident": ["I63.9"],
    "stroke": ["I63.9"],
    "cerebral infarction": ["I63.9"],
    "peripheral vascular disease": ["I73.9"],
    "peripheral arterial disease": ["I73.9"],
    "cardiomyopathy": ["I42.9"],
    "aortic stenosis": ["I35.0"],

    # Respiratory
    "pneumonia": ["J18.9"],
    "chronic obstructive pulmonary disease": ["J44.9"],
    "copd exacerbation": ["J44.1"],
    "acute respiratory failure": ["J96.00"],
    "respiratory failure with hypoxia": ["J96.01"],
    "asthma": ["J45.20"],
    "acute respiratory distress syndrome": ["J80"],
    "upper respiratory infection": ["J06.9"],
    "acute bronchitis": ["J20.9"],

    # Endocrine / metabolic
    "diabetes mellitus": ["E11.9"],
    "type 1 diabetes mellitus": ["E10.9"],
    "type 2 diabetes mellitus": ["E11.9"],
    "type 2 diabetes": ["E11.9"],
    "insulin dependent diabetes mellitus": ["E10.9"],
    "non-insulin dependent diabetes mellitus": ["E11.9"],
    "diabetic neuropathy": ["E11.40"],
    "hypothyroidism": ["E03.9"],
    "hyperthyroidism": ["E05.90"],
    "hyperlipidemia": ["E78.5"],
    "hypercholesterolemia": ["E78.00"],
    "obesity": ["E66.01"],
    "dehydration": ["E86.0"],
    "hyponatremia": ["E87.1"],
    "hyperkalemia": ["E87.5"],
    "hypokalemia": ["E87.6"],
    "malnutrition": ["E46"],

    # Renal
    "acute kidney injury": ["N17.9"],
    "acute kidney failure": ["N17.9"],
    "acute renal failure": ["N17.9"],
    "chronic kidney disease": ["N18.9"],
    "end stage renal disease": ["N18.6"],
    "urinary tract infection": ["N39.0"],
    "benign prostatic hyperplasia": ["N40.0"],

    # Gastrointestinal
    "gastroesophageal reflux disease": ["K21.0"],
    "gastrointestinal hemorrhage": ["K92.2"],
    "gastrointestinal bleed": ["K92.2"],
    "upper gastrointestinal bleed": ["K92.2"],
    "lower gastrointestinal bleed": ["K92.2"],
    "cirrhosis": ["K74.60"],
    "alcoholic cirrhosis": ["K70.30"],
    "fatty liver": ["K76.0"],
    "pancreatitis": ["K85.90"],
    "appendicitis": ["K35.80"],
    "diverticulitis": ["K57.32"],
    "intestinal obstruction": ["K56.60"],
    "small bowel obstruction": ["K56.60"],
    "constipation": ["K59.00"],
    "cholecystitis": ["K80.20"],

    # Infectious
    "sepsis": ["A41.9"],
    "hepatitis c": ["B18.2"],

    # Mental health
    "depression": ["F32.9"],
    "major depressive disorder": ["F32.9"],
    "anxiety": ["F41.9"],
    "generalized anxiety disorder": ["F41.1"],
    "bipolar disorder": ["F31.9"],
    "schizophrenia": ["F20.9"],
    "dementia": ["F03.90"],
    "alcohol dependence": ["F10.20"],
    "alcohol withdrawal": ["F10.239"],
    "ptsd": ["F43.10"],
    "post-traumatic stress disorder": ["F43.10"],

    # Neoplasms
    "lung cancer": ["C34.90"],
    "breast cancer": ["C50.919"],
    "colon cancer": ["C18.9"],
    "prostate cancer": ["C61"],
    "bladder cancer": ["C67.9"],
    "brain cancer": ["C71.9"],
    "multiple myeloma": ["C90.00"],
    "anemia": ["D64.9"],

    # Symptoms / signs
    "chest pain": ["R07.9"],
    "shortness of breath": ["R06.02"],
    "dyspnea": ["R06.00"],
    "fever": ["R50.9"],
    "chills": ["R68.83"],
    "rigors": ["R68.83"],
    "headache": ["R51.9"],
    "nausea": ["R11.0"],
    "vomiting": ["R11.10"],
    "nausea and vomiting": ["R11.2"],
    "diarrhea": ["R19.7"],
    "cough": ["R05.9"],
    "syncope": ["R55"],
    "dizziness": ["R42"],
    "seizure": ["R56.9"],
    "convulsions": ["R56.9"],
    "abdominal pain": ["R10.9"],
    "edema": ["R60.0"],
    "weight loss": ["R63.4"],
    "tachycardia": ["R00.0"],
    "bradycardia": ["R00.1"],
    "hypoxemia": ["R09.02"],
    "dysphagia": ["R13.10"],
    "hyperglycemia": ["R73.9"],
    "fatigue": ["R53.83"],
    "malaise": ["R53.81"],
    "weakness": ["R53.1"],
    "altered mental status": ["R41.82"],
    "hematuria": ["R31.9"],
    "epistaxis": ["R04.0"],
    "hemoptysis": ["R04.2"],
    "anorexia": ["R63.0"],
    "constipation": ["K59.00"],
    "urinary retention": ["R33.9"],
    "urinary incontinence": ["R32"],

    # Autoimmune / musculoskeletal
    "rheumatoid arthritis": ["M06.9"],
    "osteoarthritis": ["M19.90"],
    "systemic lupus erythematosus": ["M32.9"],
    "gout": ["M10.9"],
    "low back pain": ["M54.5"],

    # Neurological
    "multiple sclerosis": ["M35.9"],
    "transient ischemic attack": ["G45.9"],
    "epilepsy": ["G40.909"],
    "obstructive sleep apnea": ["G47.33"],
}


def _tokenize_description(text: str) -> set:
    """Tokenize and normalize a description string for matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = set(text.split())
    # Remove very common stopwords
    stopwords = {"of", "the", "and", "or", "a", "an", "in", "to", "with", "without",
                 "for", "by", "on", "at", "as", "not", "is", "are", "was", "were",
                 "unspecified", "other", "due"}
    return tokens - stopwords


class ICDCodeLookup:
    """
    ICD-10-CM code lookup and entity-to-code matching.

    Supports:
    - Direct entity text → ICD code mapping (curated high-frequency codes)
    - Token-overlap fuzzy matching against 200+ code descriptions
    - Loading additional codes from HuggingFace datasets
    - Code hierarchy information
    """

    def __init__(self, load_from_hf: bool = False, hf_dataset: Optional[str] = None):
        """
        Parameters
        ----------
        load_from_hf : bool
            If True, load full ICD-10 code set from HuggingFace.
        hf_dataset : str, optional
            HuggingFace dataset path. Defaults to awacke1/ICD10-Clinical-Terminology.
        """
        self.code_to_description = _build_icd10_lookup()
        self.entity_to_codes = dict(_ENTITY_TO_ICD10)
        self._desc_tokens = {}  # code -> set of tokens

        # Pre-tokenize all descriptions for fuzzy matching
        for code, desc in self.code_to_description.items():
            self._desc_tokens[code] = _tokenize_description(desc)

        if load_from_hf:
            self._load_from_huggingface(hf_dataset or "awacke1/ICD10-Clinical-Terminology")

        logger.info(
            "ICDCodeLookup initialised with %d codes, %d entity mappings.",
            len(self.code_to_description), len(self.entity_to_codes),
        )

    def _load_from_huggingface(self, dataset_name: str) -> None:
        """Load additional codes from a HuggingFace dataset."""
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
            count = 0
            for row in ds:
                code = row.get("Code", row.get("code", ""))
                desc = row.get("Description", row.get("description", ""))
                if code and desc and code not in self.code_to_description:
                    self.code_to_description[code] = desc
                    self._desc_tokens[code] = _tokenize_description(desc)
                    count += 1
            logger.info("Loaded %d additional codes from '%s'.", count, dataset_name)
        except Exception as e:
            logger.warning("Failed to load HuggingFace dataset '%s': %s", dataset_name, e)

    def lookup_code(self, code: str) -> Optional[ICDCode]:
        """Look up a single ICD code by its code string."""
        desc = self.code_to_description.get(code)
        if desc is None:
            return None
        return ICDCode(
            code=code,
            description=desc,
            chapter=self._get_chapter(code),
        )

    def _get_chapter(self, code: str) -> Optional[str]:
        """Determine the ICD-10-CM chapter from the code prefix."""
        if not code:
            return None
        first = code[0].upper()
        chapters = {
            "A": "Infectious diseases", "B": "Infectious diseases",
            "C": "Neoplasms", "D": "Neoplasms/Blood diseases",
            "E": "Endocrine/metabolic", "F": "Mental/behavioral",
            "G": "Nervous system", "H": "Eye/Ear",
            "I": "Circulatory system", "J": "Respiratory system",
            "K": "Digestive system", "L": "Skin",
            "M": "Musculoskeletal", "N": "Genitourinary",
            "O": "Pregnancy", "P": "Perinatal",
            "Q": "Congenital", "R": "Symptoms/signs",
            "S": "Injury", "T": "Injury/poisoning",
            "V": "External causes", "W": "External causes",
            "X": "External causes", "Y": "External causes",
            "Z": "Factors influencing health status",
        }
        return chapters.get(first)

    def match_entity(
        self,
        entity_text: str,
        entity_label: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[ICDMatch]:
        """
        Find candidate ICD codes for an entity text.

        Uses a tiered matching strategy:
        1. Exact match against curated entity→code mapping
        2. Token-overlap scoring against all code descriptions

        Parameters
        ----------
        entity_text : str
            The recognized entity text (e.g. "type 2 diabetes").
        entity_label : str, optional
            NER label (e.g. "Disease") for filtering.
        top_k : int
            Maximum number of candidates to return.
        min_score : float
            Minimum token-overlap score to include.

        Returns
        -------
        list of ICDMatch
        """
        normalized = entity_text.lower().strip()
        matches = []

        # Tier 1: Direct entity mapping
        if normalized in self.entity_to_codes:
            for code in self.entity_to_codes[normalized]:
                desc = self.code_to_description.get(code, "")
                matches.append(ICDMatch(
                    code=code, description=desc, score=1.0, match_type="exact",
                ))
            return matches[:top_k]

        # Tier 2: Token-overlap fuzzy matching
        entity_tokens = _tokenize_description(entity_text)
        if not entity_tokens:
            return []

        scored = []
        for code, desc_tokens in self._desc_tokens.items():
            if not desc_tokens:
                continue
            overlap = entity_tokens & desc_tokens
            if not overlap:
                continue
            # Jaccard-like score weighted toward entity coverage
            entity_coverage = len(overlap) / len(entity_tokens)
            desc_coverage = len(overlap) / len(desc_tokens)
            score = 0.7 * entity_coverage + 0.3 * desc_coverage
            if score >= min_score:
                scored.append((code, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        for code, score in scored[:top_k]:
            matches.append(ICDMatch(
                code=code,
                description=self.code_to_description[code],
                score=score,
                match_type="token_overlap",
            ))

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
            label = ent.get("label", None)
            codes = self.match_entity(text, entity_label=label, top_k=top_k)
            enriched = dict(ent)
            enriched["icd_codes"] = [c.to_dict() for c in codes]
            results.append(enriched)
        return results
