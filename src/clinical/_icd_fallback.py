"""
Minimal ICD-10-CM fallback codes for offline/CI use.

Only used when the HuggingFace dataset (atta00/icd10-codes) cannot be
downloaded. In production, always prefer the full 51K dataset.
"""

FALLBACK_CODES = {
    # Circulatory
    "I10": "Essential (primary) hypertension",
    "I21.3": "ST elevation (STEMI) myocardial infarction of unspecified site",
    "I21.4": "Non-ST elevation (NSTEMI) myocardial infarction",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
    "I48.91": "Unspecified atrial fibrillation",
    "I50.9": "Heart failure, unspecified",
    "I63.9": "Cerebral infarction, unspecified",
    "I26.99": "Other pulmonary embolism without acute cor pulmonale",
    "I82.409": "Acute embolism and thrombosis of unspecified deep veins of lower extremity",
    # Respiratory
    "J18.9": "Pneumonia, unspecified organism",
    "J44.1": "Chronic obstructive pulmonary disease with (acute) exacerbation",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "J80": "Acute respiratory distress syndrome",
    "J96.00": "Acute respiratory failure, unspecified",
    # Endocrine
    "E10.9": "Type 1 diabetes mellitus without complications",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E78.5": "Hyperlipidemia, unspecified",
    "E86.0": "Dehydration",
    "E87.1": "Hypo-osmolality and hyponatremia",
    # Mental/behavioral
    "F32.9": "Major depressive disorder, single episode, unspecified",
    "F41.9": "Anxiety disorder, unspecified",
    # Renal
    "N17.9": "Acute kidney failure, unspecified",
    "N18.9": "Chronic kidney disease, unspecified",
    "N18.6": "End stage renal disease",
    "N39.0": "Urinary tract infection, site not specified",
    # GI
    "K21.0": "Gastro-esophageal reflux disease with esophagitis",
    "K85.90": "Acute pancreatitis without necrosis or infection, unspecified",
    "K92.2": "Gastrointestinal hemorrhage, unspecified",
    # Infectious
    "A41.9": "Sepsis, unspecified organism",
    # Neoplasms
    "C34.90": "Malignant neoplasm of unspecified part of unspecified bronchus or lung",
    "C50.919": "Malignant neoplasm of unspecified site of unspecified female breast",
    "D64.9": "Anemia, unspecified",
    # Symptoms
    "R06.02": "Shortness of breath",
    "R07.9": "Chest pain, unspecified",
    "R10.9": "Unspecified abdominal pain",
    "R11.0": "Nausea",
    "R19.7": "Diarrhea, unspecified",
    "R50.9": "Fever, unspecified",
    "R51.9": "Headache, unspecified",
    "R55": "Syncope and collapse",
    "R05.9": "Cough, unspecified",
    "R56.9": "Unspecified convulsions",
    "R68.83": "Chills (without fever)",
}
