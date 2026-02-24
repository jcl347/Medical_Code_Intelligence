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
- CMS IPPS Table 5 (DRG relative weights, ~770 DRGs, published annually)
  Auto-downloaded from CMS.gov and cached locally on first use.
- drgpy library (ICD-10 to MS-DRG grouper, Apache 2.0)

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

import io
import logging
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# FY 2026 CMS IPPS national standardized amount
FY2026_STANDARDIZED_AMOUNT = 6752.61

# CMS IPPS Table 5 download URL (FY 2026 Final Rule)
# Published by CMS as public-domain data at:
#   https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps
# Override via CMS_TABLE5_URL environment variable or table5_url constructor param.
_DEFAULT_CMS_TABLE5_URL = os.environ.get(
    "CMS_TABLE5_URL",
    "https://www.cms.gov/files/zip/fy-2026-fr-table-5.zip",
)

# Local cache directory for downloaded CMS data
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "medical_code_intelligence",
)


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
    for payment estimation.  DRG weight data is loaded from CMS Table 5:

    1. Local Excel file (via ``table5_path``)
    2. Auto-download from CMS.gov (cached locally after first download)

    If neither source is available, cost estimation methods return ``None``.

    Parameters
    ----------
    base_rate : float
        National standardized amount for payment estimation.
        Default is FY 2026 ($6,752.61).
    drg_version : str
        MS-DRG grouper version for drgpy. Default ``"v40"``.
    table5_path : str, optional
        Path to a local CMS IPPS Table 5 Excel file. If provided and the
        file exists, weights are loaded from it directly.
    table5_url : str, optional
        URL to download CMS IPPS Table 5 zip file. Defaults to the FY 2026
        Final Rule Table 5 on CMS.gov. Override via ``CMS_TABLE5_URL``
        environment variable.
    cache_dir : str, optional
        Directory for caching downloaded CMS data. Defaults to
        ``~/.cache/medical_code_intelligence/``.
    """

    def __init__(
        self,
        base_rate: float = FY2026_STANDARDIZED_AMOUNT,
        drg_version: str = "v40",
        table5_path: Optional[str] = None,
        table5_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.base_rate = base_rate
        self.drg_version = drg_version
        self._table5_url = table5_url or _DEFAULT_CMS_TABLE5_URL
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
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
            ``None`` if grouping fails or no weight data is available.
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
                "ICD-to-DRG grouping is unavailable."
            )
            self._grouper = None

    def _group(
        self, dx: List[str], pr: List[str], gender: str, is_alive: bool,
    ) -> Optional[str]:
        """Group ICD codes into an MS-DRG using drgpy."""
        if self._grouper is None:
            logger.debug(
                "No DRG grouper available. Install drgpy: pip install drgpy"
            )
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
        Load DRG relative weights from CMS IPPS Table 5.

        Resolution order:
        1. Local Excel file (table5_path parameter)
        2. Auto-download from CMS.gov (cached after first download)
        """
        # 1. Try user-provided local Table 5 file
        if table5_path and os.path.exists(table5_path):
            if self._load_from_excel(table5_path):
                return

        # 2. Try cached CMS download
        cached_path = os.path.join(self._cache_dir, "cms_table5.xlsx")
        if os.path.exists(cached_path):
            if self._load_from_excel(cached_path):
                return

        # 3. Try downloading from CMS.gov
        downloaded = self._download_cms_table5()
        if downloaded and self._load_from_excel(downloaded):
            return

        # No weight data available
        logger.warning(
            "No DRG weight data available. Cost estimation will return None. "
            "To enable cost estimation, either:\n"
            "  1. Download CMS IPPS Table 5 from CMS.gov and pass via table5_path\n"
            "  2. Set CMS_TABLE5_URL env var to a valid CMS Table 5 zip URL\n"
            "  3. Ensure network access for auto-download from CMS.gov"
        )

    def _load_from_excel(self, path: str) -> bool:
        """Load DRG weights from a CMS Table 5 Excel file. Returns True on success."""
        try:
            import pandas as pd
            df = pd.read_excel(path, dtype={"MS-DRG": str})
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
            logger.info(
                "Loaded %d DRG weights from %s", len(self._weights), path,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load Table 5 from %s: %s", path, e)
            return False

    def _download_cms_table5(self) -> Optional[str]:
        """
        Download CMS IPPS Table 5 data and cache locally.

        Downloads the zip file from CMS.gov, extracts the Excel file,
        and saves it to the cache directory for future use.

        Returns
        -------
        str or None
            Path to the cached Excel file, or None if download failed.
        """
        url = self._table5_url
        cache_path = os.path.join(self._cache_dir, "cms_table5.xlsx")

        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            logger.info("Downloading CMS IPPS Table 5 from %s ...", url)

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MedicalCodeIntelligence/1.0"},
            )
            response = urllib.request.urlopen(req, timeout=60)
            data = response.read()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    xlsx_files = [
                        f for f in zf.namelist()
                        if f.lower().endswith(".xlsx")
                    ]
                    if not xlsx_files:
                        logger.warning("No .xlsx file found in CMS zip archive")
                        return None
                    with zf.open(xlsx_files[0]) as src:
                        with open(cache_path, "wb") as dst:
                            dst.write(src.read())
            else:
                with open(cache_path, "wb") as f:
                    f.write(data)

            logger.info("CMS Table 5 cached at %s", cache_path)
            return cache_path

        except Exception as e:
            logger.warning("Failed to download CMS Table 5: %s", e)
            return None

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
