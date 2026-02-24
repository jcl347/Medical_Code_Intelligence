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
- **drgpy** library (Apache 2.0) — ICD-10 to MS-DRG grouper + complete
  DRG metadata (767 DRGs with titles, MDC, type). This is the primary
  data source for all DRG resolution.
- **NBER CMS Table 5 CSV** (auto-downloaded) — official FY 2026 relative
  weights, geometric and arithmetic mean LOS for ~770 DRGs. Downloaded
  from ``https://data.nber.org/drg/csv/drgweight2026FR.csv`` on first
  use and cached locally for subsequent runs.
- **CMS IPPS Table 5 Excel** (optional) — if a local Excel file is
  provided via ``table5_path``, it takes precedence over the NBER CSV.

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

    Uses drgpy for both ICD-to-DRG grouping and DRG metadata (all 767
    DRGs). Optionally loads CMS IPPS Table 5 for accurate per-DRG
    relative weights; without Table 5, DRG resolution still works but
    cost estimates use weight 1.0 (national average) for DRGs not in
    Table 5.

    Parameters
    ----------
    base_rate : float
        National standardized amount for payment estimation.
        Default is FY 2026 ($6,752.61).
    drg_version : str
        MS-DRG grouper version for drgpy. Default ``"v40"``.
    table5_path : str, optional
        Path to CMS IPPS Table 5 Excel file for accurate relative
        weights. If not provided, uses drgpy metadata with weight 1.0.
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

        self._init_grouper(drg_version)
        self._load_weights(table5_path)

    @property
    def num_drgs(self) -> int:
        """Number of DRGs available for resolution."""
        return len(self._weights)

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

        Requires drgpy for ICD-to-DRG grouping. Use ``analyze_drg_family``
        to analyse a known DRG code directly (no grouper needed).
        """
        current = self.get_drg(diagnosis_codes, procedure_codes, gender, is_alive)
        if current is None:
            return None

        return self._build_family_analysis(current)

    def analyze_drg_family(self, drg_code: str) -> Optional[CostImpactAnalysis]:
        """
        Analyse severity-tier variants for a known DRG code.

        Unlike ``analyze_cost_impact`` this does **not** require drgpy
        because no ICD-to-DRG grouping is performed — the DRG code is
        provided directly.

        Parameters
        ----------
        drg_code : str
            MS-DRG code (e.g. ``"292"``).

        Returns
        -------
        CostImpactAnalysis or None
            ``None`` if the DRG code is not in the weight table.
        """
        current = self._build_result(drg_code)
        if current is None:
            return None

        return self._build_family_analysis(current)

    def _build_family_analysis(self, current: DRGResult) -> CostImpactAnalysis:
        """Build a CostImpactAnalysis by comparing *current* against its family."""
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
                "DRG grouping and resolution unavailable."
            )
            self._grouper = None

    def _group(
        self, dx: List[str], pr: List[str], gender: str, is_alive: bool,
    ) -> Optional[str]:
        if self._grouper is None:
            return None

        try:
            # drgpy expects ICD codes without dots (e.g. "J189" not "J18.9")
            dx_clean = [c.replace(".", "") for c in dx]
            pr_clean = [c.replace(".", "") for c in pr]
            result = self._grouper.get_drg(
                dx_clean, pr_clean, gender=gender, is_alive=is_alive,
            )
            # DRG "000" means ungroupable
            if result and str(result) != "000":
                return str(result)
        except Exception as e:
            logger.debug("DRG grouping failed: %s", e)

        return None

    def _load_weights(self, table5_path: Optional[str] = None):
        """
        Load DRG data from drgpy + CMS relative weights.

        Strategy:
        1. Load all 767 DRGs from drgpy.drgmap (title, MDC, type)
        2. Overlay accurate relative weights from one of:
           a. Local CMS Table 5 Excel file (if ``table5_path`` provided)
           b. NBER-hosted CMS Table 5 CSV (auto-downloaded, cached)
           c. Default weight 1.0 if both fail
        """
        # Step 1: Build base table from drgpy's complete DRG map
        if self._grouper is not None:
            self._weights = _build_weights_from_drgpy(self._grouper)
            logger.info(
                "Loaded %d DRGs from drgpy (version %s)",
                len(self._weights), self.drg_version,
            )
        else:
            logger.warning("No DRG data available (drgpy not installed)")

        # Step 2a: Overlay from local CMS Table 5 Excel if provided
        if table5_path and os.path.exists(table5_path):
            n = _overlay_table5_excel(self._weights, table5_path)
            if n > 0:
                logger.info("Overlaid %d DRG weights from %s", n, table5_path)
                return

        # Step 2b: Download NBER CMS Table 5 CSV (with caching)
        n = _overlay_nber_weights(self._weights)
        if n > 0:
            logger.info(
                "Overlaid %d DRG weights from NBER CMS Table 5 (FY 2026)", n,
            )
        else:
            logger.warning(
                "Could not load CMS relative weights. "
                "Cost estimates will use weight=1.0 (national average)."
            )

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
        severity = _classify_severity(title)

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
        base_title = _strip_severity(
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
            candidate_base = _strip_severity(entry["title"])
            if candidate_base == base_title and base_title:
                severity = _classify_severity(entry["title"])
                results.append((candidate, severity))

        return results


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _classify_severity(title: str) -> str:
    """Classify a DRG title into severity level."""
    title_upper = title.upper()
    if "W MCC" in title_upper or "WITH MCC" in title_upper:
        return "mcc"
    if "W CC" in title_upper or "WITH CC" in title_upper:
        return "cc"
    return "base"


def _strip_severity(title: str) -> str:
    """Remove CC/MCC suffixes to get the base clinical condition title."""
    return re.sub(
        r"\s*(W|WITH|W/O|WITHOUT)\s*(MCC|CC|CC/MCC).*", "",
        title, flags=re.IGNORECASE,
    ).strip()


def _build_weights_from_drgpy(grouper) -> Dict[str, Dict]:
    """
    Build complete DRG weight table from drgpy's drgmap.

    drgpy's DRGEngine.drgmap contains all 767 MS-DRGs with:
    - drg: DRG code
    - desc: Full DRG title/description
    - mdc: Major Diagnostic Category
    - is_medical: True for medical DRGs
    - is_surgical: True for surgical DRGs

    Since drgpy does not include relative weights, we use weight=1.0
    as the default. Accurate weights can be overlaid from CMS Table 5.
    """
    weights: Dict[str, Dict] = {}

    for drg_code, info in grouper.drgmap.items():
        code = str(drg_code).zfill(3)
        desc = info.get("desc", f"MS-DRG {code}")
        mdc = info.get("mdc", "")
        is_surgical = info.get("is_surgical", False)
        drg_type = "SURG" if is_surgical else "MED"

        weights[code] = {
            "title": desc,
            "mdc": mdc,
            "type": drg_type,
            "weight": 1.0,   # Default; overlaid by Table 5 if available
            "geo_los": 0.0,
            "arith_los": 0.0,
        }

    return weights


# NBER-hosted CMS IPPS Table 5 CSV (FY 2026 Final Rule)
_NBER_TABLE5_URL = "https://data.nber.org/drg/csv/drgweight2026FR.csv"
_NBER_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "medical_code_intelligence")
_NBER_CACHE_FILE = os.path.join(_NBER_CACHE_DIR, "drgweight2026FR.csv")


def _overlay_nber_weights(weights: Dict[str, Dict]) -> int:
    """
    Download NBER CMS Table 5 CSV and overlay accurate relative weights.

    The NBER (National Bureau of Economic Research) hosts CMS IPPS Table 5
    as a clean CSV at ``https://data.nber.org/drg/csv/drgweight2026FR.csv``.
    Contains ~770 DRGs with FY 2026 relative weights, geometric mean LOS,
    and arithmetic mean LOS.

    The file is cached locally after first download to avoid repeated
    network calls. Returns the number of DRGs successfully overlaid.
    """
    csv_path = _download_nber_csv()
    if csv_path is None:
        return 0

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        # ms_drg column has NaN rows — drop them before int conversion
        df = df.dropna(subset=["ms_drg"])
        df["ms_drg"] = df["ms_drg"].astype(int)

        # weights, los_geo, los_mean are string type — convert to numeric
        for col in ["weights", "los_geo", "los_mean"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        overlaid = 0
        for _, row in df.iterrows():
            code = str(int(row["ms_drg"])).zfill(3)
            weight = float(row.get("weights", 0.0))
            geo_los = float(row.get("los_geo", 0.0))
            arith_los = float(row.get("los_mean", 0.0))

            if code in weights:
                if weight > 0:
                    weights[code]["weight"] = weight
                if geo_los > 0:
                    weights[code]["geo_los"] = geo_los
                if arith_los > 0:
                    weights[code]["arith_los"] = arith_los
                overlaid += 1
            else:
                # DRG exists in CMS Table 5 but not in drgpy — add it
                title = str(row.get("msdrg_title", f"MS-DRG {code}"))
                mdc = str(row.get("mdc", ""))
                drg_type = str(row.get("type", "MED")).upper()
                if drg_type not in ("SURG", "MED"):
                    drg_type = "MED"
                weights[code] = {
                    "title": title,
                    "mdc": mdc,
                    "type": drg_type,
                    "weight": weight if weight > 0 else 1.0,
                    "geo_los": geo_los,
                    "arith_los": arith_los,
                }
                overlaid += 1

        return overlaid

    except ImportError:
        logger.warning("pandas not installed — cannot parse NBER CMS Table 5 CSV")
        return 0
    except Exception as e:
        logger.warning("Failed to parse NBER CMS Table 5 CSV: %s", e)
        return 0


def _download_nber_csv() -> Optional[str]:
    """Download NBER CMS Table 5 CSV, caching locally. Returns path or None."""
    # Return cached file if it exists
    if os.path.exists(_NBER_CACHE_FILE):
        logger.debug("Using cached NBER CSV: %s", _NBER_CACHE_FILE)
        return _NBER_CACHE_FILE

    try:
        import urllib.request

        os.makedirs(_NBER_CACHE_DIR, exist_ok=True)
        logger.info("Downloading CMS Table 5 from NBER: %s", _NBER_TABLE5_URL)
        urllib.request.urlretrieve(_NBER_TABLE5_URL, _NBER_CACHE_FILE)
        logger.info("Cached NBER CSV to: %s", _NBER_CACHE_FILE)
        return _NBER_CACHE_FILE
    except Exception as e:
        logger.warning("Failed to download NBER CMS Table 5: %s", e)
        # Clean up partial download
        if os.path.exists(_NBER_CACHE_FILE):
            try:
                os.remove(_NBER_CACHE_FILE)
            except OSError:
                pass
        return None


def _overlay_table5_excel(weights: Dict[str, Dict], path: str) -> int:
    """
    Overlay relative weights from a local CMS IPPS Table 5 Excel file.

    The CMS distributes Table 5 as an Excel workbook. This function reads
    the first sheet and looks for columns containing DRG code, weight,
    and LOS data. Returns the number of DRGs overlaid.
    """
    try:
        import pandas as pd

        # CMS Table 5 Excel files vary in format across fiscal years.
        # Try common column name patterns.
        df = pd.read_excel(path, sheet_name=0)

        # Normalize column names to lowercase for matching
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Identify DRG code column
        drg_col = None
        for candidate in ["ms-drg", "ms_drg", "msdrg", "drg", "ms drg"]:
            if candidate in df.columns:
                drg_col = candidate
                break
        if drg_col is None:
            logger.warning("Could not find DRG code column in %s", path)
            return 0

        # Identify weight column
        weight_col = None
        for candidate in ["relative weight", "weights", "weight", "relative_weight"]:
            if candidate in df.columns:
                weight_col = candidate
                break

        # Identify LOS columns
        geo_col = None
        for candidate in ["geometric mean los", "geo_los", "los_geo", "gmlos"]:
            if candidate in df.columns:
                geo_col = candidate
                break

        arith_col = None
        for candidate in ["arithmetic mean los", "arith_los", "los_mean", "amlos"]:
            if candidate in df.columns:
                arith_col = candidate
                break

        df = df.dropna(subset=[drg_col])

        overlaid = 0
        for _, row in df.iterrows():
            try:
                code = str(int(float(row[drg_col]))).zfill(3)
            except (ValueError, TypeError):
                continue

            if code not in weights:
                continue

            if weight_col and pd.notna(row.get(weight_col)):
                w = float(row[weight_col])
                if w > 0:
                    weights[code]["weight"] = w

            if geo_col and pd.notna(row.get(geo_col)):
                g = float(row[geo_col])
                if g > 0:
                    weights[code]["geo_los"] = g

            if arith_col and pd.notna(row.get(arith_col)):
                a = float(row[arith_col])
                if a > 0:
                    weights[code]["arith_los"] = a

            overlaid += 1

        return overlaid

    except ImportError:
        logger.warning("pandas/openpyxl not installed — cannot parse Table 5 Excel")
        return 0
    except Exception as e:
        logger.warning("Failed to parse Table 5 Excel (%s): %s", path, e)
        return 0
