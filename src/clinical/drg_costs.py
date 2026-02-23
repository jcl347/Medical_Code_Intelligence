"""
MS-DRG Cost Scoping Module.

Maps ICD-10-CM diagnosis codes to Medicare Severity Diagnosis Related Groups
(MS-DRGs) and estimates financial impact using CMS IPPS relative weights.

Key concepts:
- **MS-DRG**: Groups inpatient stays by diagnosis, severity, and procedures
  into ~770 payment categories.
- **Relative Weight**: A multiplier reflecting average resource intensity
  (weight 1.0 = national average; 2.0 = twice the average cost).
- **CC/MCC**: Complication/Comorbidity (CC) and Major CC (MCC) secondary
  diagnoses that increase the DRG severity tier and payment.
- **Revenue at risk**: The payment difference between the current DRG
  assignment and what it could be with proper CC/MCC capture.

Data sources:
- CMS IPPS Table 5 (DRG relative weights, published annually)
- drgpy library (ICD-10 to MS-DRG grouper, Apache 2.0)
- Built-in fallback data for CI/testing (32 common DRGs)

Usage
-----
>>> from src.clinical.drg_costs import DRGCostEstimator
>>> estimator = DRGCostEstimator()
>>> result = estimator.get_drg(["J18.9", "E11.9", "N17.9"])
>>> print(f"DRG {result.drg_code}: {result.drg_title}")
>>> print(f"Estimated payment: ${result.estimated_payment:,.2f}")
>>>
>>> # Analyze CC/MCC impact
>>> analysis = estimator.analyze_cost_impact(["J18.9", "E11.9"])
>>> print(f"Revenue at risk: ${analysis.revenue_at_risk:,.2f}")
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# FY 2026 CMS IPPS national standardized amount
FY2026_STANDARDIZED_AMOUNT = 6752.61


@dataclass
class DRGResult:
    """Result of MS-DRG assignment and cost estimation."""

    drg_code: str
    drg_title: str
    mdc: str                             # Major Diagnostic Category
    drg_type: str                        # "SURG" or "MED"
    relative_weight: float
    geometric_mean_los: float            # Geometric mean length of stay (days)
    arithmetic_mean_los: float           # Arithmetic mean length of stay (days)
    estimated_payment: float             # National unadjusted payment estimate
    severity_level: str                  # "base", "cc", "mcc"

    def to_dict(self) -> Dict:
        return {
            "drg_code": self.drg_code,
            "drg_title": self.drg_title,
            "mdc": self.mdc,
            "type": self.drg_type,
            "relative_weight": self.relative_weight,
            "geometric_mean_los": self.geometric_mean_los,
            "arithmetic_mean_los": self.arithmetic_mean_los,
            "estimated_payment": round(self.estimated_payment, 2),
            "severity_level": self.severity_level,
        }


@dataclass
class CostImpactAnalysis:
    """
    Financial impact analysis of DRG assignment.

    Compares the current DRG against its severity-tier variants
    (base / CC / MCC) to quantify revenue at risk from miscoding.
    """

    current_drg: DRGResult
    base_drg: Optional[DRGResult] = None
    cc_drg: Optional[DRGResult] = None
    mcc_drg: Optional[DRGResult] = None
    revenue_at_risk: float = 0.0
    undercoding_risk: bool = False

    def to_dict(self) -> Dict:
        result: Dict = {"current": self.current_drg.to_dict()}
        if self.base_drg:
            result["base_variant"] = self.base_drg.to_dict()
        if self.cc_drg:
            result["cc_variant"] = self.cc_drg.to_dict()
        if self.mcc_drg:
            result["mcc_variant"] = self.mcc_drg.to_dict()
        result["revenue_at_risk"] = round(self.revenue_at_risk, 2)
        result["undercoding_risk"] = self.undercoding_risk
        return result


class DRGCostEstimator:
    """
    Map ICD-10-CM codes to MS-DRGs and estimate financial impact.

    Uses drgpy for grouper logic and CMS IPPS Table 5 relative weights
    for payment estimation.  Falls back to built-in data when external
    sources are unavailable (following the project's offline-first pattern).

    Parameters
    ----------
    base_rate : float
        National standardized amount for payment estimation.
        Default is FY 2026 ($6,752.61).
    drg_version : str
        MS-DRG grouper version for drgpy. Default ``"v40"``.
    table5_path : str, optional
        Path to CMS IPPS Table 5 Excel file.  If not provided, uses
        built-in fallback weights for the 32 most common DRGs.
    """

    def __init__(
        self,
        base_rate: float = FY2026_STANDARDIZED_AMOUNT,
        drg_version: str = "v40",
        table5_path: Optional[str] = None,
    ):
        self.base_rate = base_rate
        self.drg_version = drg_version
        self._weights: Dict[str, Dict] = {}
        self._grouper = None

        self._load_weights(table5_path)
        self._init_grouper(drg_version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_drg(
        self,
        diagnosis_codes: List[str],
        procedure_codes: Optional[List[str]] = None,
        gender: str = "M",
        is_alive: bool = True,
    ) -> Optional[DRGResult]:
        """
        Assign MS-DRG and estimate cost for given ICD-10 codes.

        Parameters
        ----------
        diagnosis_codes : list of str
            ICD-10-CM codes. First code is the principal diagnosis.
        procedure_codes : list of str, optional
            ICD-10-PCS procedure codes.
        gender : str
            Patient gender (``"M"`` or ``"F"``).
        is_alive : bool
            Whether patient was alive at discharge.

        Returns
        -------
        DRGResult or None
        """
        procedure_codes = procedure_codes or []
        if not diagnosis_codes:
            return None

        drg_code = self._group(diagnosis_codes, procedure_codes, gender, is_alive)
        if drg_code is None:
            return None

        return self._build_result(drg_code)

    def estimate_cost(self, drg_code: str) -> float:
        """Return national unadjusted payment estimate for a DRG code."""
        weight = self._get_weight(drg_code)
        return self.base_rate * weight

    def analyze_cost_impact(
        self,
        diagnosis_codes: List[str],
        procedure_codes: Optional[List[str]] = None,
        gender: str = "M",
        is_alive: bool = True,
    ) -> Optional[CostImpactAnalysis]:
        """
        Compare current DRG against its CC/MCC severity-tier variants.

        Returns an analysis showing the current DRG, all available
        severity variants, and the revenue at risk from undercoding.
        """
        current = self.get_drg(diagnosis_codes, procedure_codes, gender, is_alive)
        if current is None:
            return None

        analysis = CostImpactAnalysis(current_drg=current)
        variants = self._find_drg_family(current.drg_code)

        for variant_code, severity in variants:
            result = self._build_result(variant_code)
            if result is None:
                continue
            if severity == "base":
                analysis.base_drg = result
            elif severity == "cc":
                analysis.cc_drg = result
            elif severity == "mcc":
                analysis.mcc_drg = result

        # Revenue at risk = gap between current and highest-severity variant
        highest = analysis.mcc_drg or analysis.cc_drg
        if highest and highest.estimated_payment > current.estimated_payment:
            analysis.revenue_at_risk = (
                highest.estimated_payment - current.estimated_payment
            )
            analysis.undercoding_risk = True

        return analysis

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_grouper(self, version: str):
        try:
            from drgpy.msdrg import DRGEngine
            self._grouper = DRGEngine(version=version)
            logger.info("Initialized drgpy DRG grouper %s", version)
        except ImportError:
            logger.warning(
                "drgpy not installed (pip install drgpy). "
                "DRG grouping unavailable; using weight-table lookup only."
            )
            self._grouper = None

    def _group(
        self, dx: List[str], pr: List[str], gender: str, is_alive: bool,
    ) -> Optional[str]:
        if self._grouper is None:
            return None
        try:
            result = self._grouper.get_drg(dx, pr, gender=gender, is_alive=is_alive)
            return str(result) if result else None
        except Exception as e:
            logger.debug("DRG grouping failed: %s", e)
            return None

    def _load_weights(self, table5_path: Optional[str] = None):
        """Load DRG relative weights from CMS Table 5 or fallback data."""
        if table5_path and os.path.exists(table5_path):
            try:
                import pandas as pd
                df = pd.read_excel(table5_path, dtype={"MS-DRG": str})
                for _, row in df.iterrows():
                    code = str(row.get("MS-DRG", "")).zfill(3)
                    self._weights[code] = {
                        "title": str(row.get("MS-DRG Title", "")),
                        "mdc": str(row.get("MDC", "")),
                        "type": str(row.get("Type", "")),
                        "weight": float(row.get("Relative Weight", 1.0)),
                        "geo_los": float(row.get("Geometric Mean LOS", 0)),
                        "arith_los": float(row.get("Arithmetic Mean LOS", 0)),
                    }
                logger.info("Loaded %d DRG weights from %s", len(self._weights), table5_path)
                return
            except Exception as e:
                logger.warning("Failed to load Table 5: %s", e)

        # Fall back to built-in weights
        self._weights = _get_fallback_weights()
        logger.info("Using built-in fallback DRG weights (%d DRGs)", len(self._weights))

    def _get_weight(self, drg_code: str) -> float:
        code = str(drg_code).zfill(3)
        entry = self._weights.get(code)
        return entry["weight"] if entry else 1.0

    def _build_result(self, drg_code: str) -> Optional[DRGResult]:
        code = str(drg_code).zfill(3)
        entry = self._weights.get(code)
        if entry is None:
            return None

        title = entry["title"]
        severity = "base"
        title_upper = title.upper()
        if "W MCC" in title_upper or "WITH MCC" in title_upper:
            severity = "mcc"
        elif "W CC" in title_upper or "WITH CC" in title_upper:
            severity = "cc"

        return DRGResult(
            drg_code=code,
            drg_title=title,
            mdc=entry["mdc"],
            drg_type=entry["type"],
            relative_weight=entry["weight"],
            geometric_mean_los=entry["geo_los"],
            arithmetic_mean_los=entry["arith_los"],
            estimated_payment=self.base_rate * entry["weight"],
            severity_level=severity,
        )

    def _find_drg_family(self, drg_code: str) -> List[Tuple[str, str]]:
        """Find related DRGs in the same clinical family (base/CC/MCC variants)."""
        code_int = int(drg_code)
        base_title = self._strip_severity(
            self._weights.get(str(code_int).zfill(3), {}).get("title", "")
        )
        results = []

        for offset in range(-3, 4):
            candidate = str(code_int + offset).zfill(3)
            if candidate == str(drg_code).zfill(3):
                continue
            entry = self._weights.get(candidate)
            if entry is None:
                continue
            candidate_base = self._strip_severity(entry["title"])
            if candidate_base == base_title and base_title:
                title_upper = entry["title"].upper()
                if "W MCC" in title_upper or "WITH MCC" in title_upper:
                    results.append((candidate, "mcc"))
                elif "W CC" in title_upper or "WITH CC" in title_upper:
                    results.append((candidate, "cc"))
                else:
                    results.append((candidate, "base"))

        return results

    @staticmethod
    def _strip_severity(title: str) -> str:
        """Remove CC/MCC suffixes to get the base clinical condition title."""
        return re.sub(
            r"\s*(W|WITH|W/O|WITHOUT)\s*(MCC|CC|CC/MCC).*", "",
            title, flags=re.IGNORECASE,
        ).strip()


# ---------------------------------------------------------------------------
# Fallback DRG weights for CI/testing (32 common medical DRGs)
# ---------------------------------------------------------------------------

def _get_fallback_weights() -> Dict[str, Dict]:
    """
    Built-in subset of DRG weights for offline use.

    Covers the most common medical DRGs by volume (per HCUP data):
    heart failure, septicemia, pneumonia, stroke, renal failure, diabetes,
    GI hemorrhage, cellulitis, UTI, and signs/symptoms.
    """
    data = [
        # (code, mdc, type, title, weight, geo_los, arith_los)
        ("064", "01", "SURG", "Intracranial Hemorrhage Or Cerebral Infarction W MCC", 1.9117, 5.8, 7.2),
        ("065", "01", "MED", "Intracranial Hemorrhage Or Cerebral Infarction W CC", 1.0619, 3.8, 4.7),
        ("066", "01", "MED", "Intracranial Hemorrhage Or Cerebral Infarction W/O CC/MCC", 0.7175, 2.6, 3.1),
        ("177", "04", "MED", "Respiratory Infections & Inflammations W MCC", 1.8729, 6.3, 7.8),
        ("178", "04", "MED", "Respiratory Infections & Inflammations W CC", 1.2384, 4.7, 5.6),
        ("179", "04", "MED", "Respiratory Infections & Inflammations W/O CC/MCC", 0.8463, 3.4, 4.0),
        ("193", "04", "MED", "Simple Pneumonia & Pleurisy W MCC", 1.2935, 4.5, 5.5),
        ("194", "04", "MED", "Simple Pneumonia & Pleurisy W CC", 0.8628, 3.4, 4.0),
        ("195", "04", "MED", "Simple Pneumonia & Pleurisy W/O CC/MCC", 0.6142, 2.5, 3.0),
        ("291", "05", "MED", "Heart Failure & Shock W MCC", 1.3968, 5.2, 6.3),
        ("292", "05", "MED", "Heart Failure & Shock W CC", 0.9259, 3.9, 4.6),
        ("293", "05", "MED", "Heart Failure & Shock W/O CC/MCC", 0.6521, 2.8, 3.3),
        ("378", "06", "MED", "G.I. Hemorrhage W MCC", 1.5726, 4.5, 5.6),
        ("379", "06", "MED", "G.I. Hemorrhage W CC", 0.9715, 3.1, 3.7),
        ("380", "06", "MED", "G.I. Hemorrhage W/O CC/MCC", 0.6423, 2.1, 2.6),
        ("469", "08", "SURG", "Major Hip And Knee Joint Replacement W MCC", 2.8156, 5.4, 6.8),
        ("470", "08", "SURG", "Major Hip And Knee Joint Replacement W/O MCC", 1.6897, 2.0, 2.4),
        ("602", "09", "MED", "Cellulitis W/O MCC", 0.7335, 3.4, 4.1),
        ("603", "09", "MED", "Cellulitis W MCC", 1.2146, 4.8, 5.9),
        ("638", "10", "MED", "Diabetes W MCC", 1.2856, 4.3, 5.2),
        ("639", "10", "MED", "Diabetes W CC", 0.7816, 3.0, 3.6),
        ("640", "10", "MED", "Diabetes W/O CC/MCC", 0.5495, 2.2, 2.6),
        ("683", "11", "MED", "Renal Failure W MCC", 1.4763, 4.8, 5.9),
        ("684", "11", "MED", "Renal Failure W CC", 0.8882, 3.3, 3.9),
        ("685", "11", "MED", "Renal Failure W/O CC/MCC", 0.5936, 2.3, 2.8),
        ("689", "11", "MED", "Kidney & Urinary Tract Infections W MCC", 1.1741, 4.3, 5.2),
        ("690", "11", "MED", "Kidney & Urinary Tract Infections W CC", 0.8271, 3.3, 3.9),
        ("691", "11", "MED", "Kidney & Urinary Tract Infections W/O CC/MCC", 0.5987, 2.6, 3.0),
        ("871", "18", "MED", "Septicemia Or Severe Sepsis W/O MV >96 Hours W MCC", 1.8192, 5.5, 6.7),
        ("872", "18", "MED", "Septicemia Or Severe Sepsis W/O MV >96 Hours W/O MCC", 1.0382, 3.7, 4.5),
        ("948", "00", "MED", "Signs & Symptoms W MCC", 1.0847, 3.7, 4.5),
        ("949", "00", "MED", "Signs & Symptoms W/O MCC", 0.6316, 2.4, 2.9),
    ]
    weights = {}
    for code, mdc, drg_type, title, weight, geo, arith in data:
        weights[code] = {
            "title": title,
            "mdc": mdc,
            "type": drg_type,
            "weight": weight,
            "geo_los": geo,
            "arith_los": arith,
        }
    return weights
