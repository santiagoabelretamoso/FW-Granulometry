# -*- coding: utf-8 -*-
"""
FractalAnalysis.py

Fractal fragmentation analysis for Componentry-FW.

This module is intentionally independent from Componentry-FW's UI implementation.
It receives a ComponentryPreview-like host object and reads the already-loaded
sample databases through the host's public/current attributes:

    app.current_sample_name()
    app.current_db()
    app.sample_dbs

Two fragmentation methods are implemented:

NUMBER-BASED
    N(>r) ∝ r^(-D)
    log10[N(>r)] = m log10(r) + b
    D_N = -m

MASS-BASED
    M(r<R) / M_T ∝ R^v
    log10[M(r<R)/M_T] = v log10(R) + b
    D_M = 3 - v

For the number-based method, component mass is converted to particle number using
an explicit triaxial-ellipsoid particle model:

    A = (A/B) B
    C = (C/B) B
    V = (pi/6) A B C = (pi/6) (A/B)(C/B) B^3
    n = M / (rho V)

Lengths are handled in mm for geometry, volume is converted from mm^3 to cm^3,
and density is in g/cm^3.

The module stores all persistent model settings inside db.fractal_settings.  The
ComponentryDB serializer in the companion modified Componentry-FW.py persists
that dictionary in both standalone Componentry projects and FW full projects.
"""

from __future__ import annotations

import copy
import math
import os
import re
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


FRACTAL_SETTINGS_SCHEMA_VERSION = 1

METHOD_NUMBER = "Number-based (N)"
METHOD_MASS = "Mass-based (M)"
METHOD_COMPARE = "Compare both"

MASS_BASIS_NORMALIZED = "Normalized 100 g basis"
MASS_BASIS_ACTUAL = "Actual sample mass"

DENSITY_UNIFORM = "Uniform across fractions"
DENSITY_BY_FRACTION = "By fraction"
SHAPE_UNIFORM = "Uniform across fractions"
SHAPE_BY_FRACTION = "By fraction"

REPRESENTATIVE_CLASS_MIDPOINT = "Class midpoint"
REPRESENTATIVE_MANUAL = "Manual"

UNDEFINED_COMPONENT = "Undefined"

SHAPE_PRESETS: Dict[str, Tuple[float, float]] = {
    "Sphere": (1.0, 1.0),
    # Presets below are editable starting values only.  They are not scientific
    # constants and the calculation always uses the explicit A/B and C/B values.
    "Equant / triaxial": (1.20, 0.80),
    "Elongated / prolate": (2.00, 0.80),
    "Discoidal / oblate": (1.20, 0.30),
    "Laminar": (1.50, 0.10),
    "Custom": (1.0, 1.0),
}


# =============================================================================
# Pure mathematical helpers
# =============================================================================

def phi_to_mm(phi: float) -> float:
    """Convert Krumbein phi to physical diameter in millimetres."""
    return float(2.0 ** (-float(phi)))


def mm_to_phi(mm: float) -> float:
    mm = float(mm)
    if not np.isfinite(mm) or mm <= 0:
        raise ValueError("Particle size must be > 0 mm.")
    return float(-np.log2(mm))


def validate_particle_model(density_g_cm3: float, b_mm: float, alpha: float, gamma: float) -> None:
    density_g_cm3 = float(density_g_cm3)
    b_mm = float(b_mm)
    alpha = float(alpha)
    gamma = float(gamma)
    if not np.isfinite(density_g_cm3) or density_g_cm3 <= 0:
        raise ValueError("Density must be > 0 g/cm3.")
    if not np.isfinite(b_mm) or b_mm <= 0:
        raise ValueError("Representative B must be > 0 mm.")
    if not np.isfinite(alpha) or alpha < 1.0:
        raise ValueError("A/B must be >= 1.")
    if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
        raise ValueError("C/B must satisfy 0 < C/B <= 1.")


def ellipsoid_geometry(b_mm: float, alpha: float, gamma: float) -> Dict[str, float]:
    """
    Return triaxial-ellipsoid dimensions and volume.

    A >= B >= C is represented through alpha=A/B >= 1 and 0<gamma=C/B<=1.
    This function validates geometry only; density is not needed here.
    """
    b_mm = float(b_mm)
    alpha = float(alpha)
    gamma = float(gamma)
    if not np.isfinite(b_mm) or b_mm <= 0:
        raise ValueError("Representative B must be > 0 mm.")
    if not np.isfinite(alpha) or alpha < 1.0:
        raise ValueError("A/B must be >= 1.")
    if not np.isfinite(gamma) or gamma <= 0.0 or gamma > 1.0:
        raise ValueError("C/B must satisfy 0 < C/B <= 1.")

    a_mm = alpha * b_mm
    c_mm = gamma * b_mm
    volume_mm3 = (math.pi / 6.0) * a_mm * b_mm * c_mm
    volume_cm3 = volume_mm3 / 1000.0
    shape_volume_factor = alpha * gamma
    n_relative_sphere = 1.0 / shape_volume_factor
    return {
        "A_mm": float(a_mm),
        "B_mm": float(b_mm),
        "C_mm": float(c_mm),
        "volume_mm3": float(volume_mm3),
        "volume_cm3": float(volume_cm3),
        "shape_volume_factor": float(shape_volume_factor),
        "n_relative_sphere": float(n_relative_sphere),
    }


def estimate_particle_number(
    mass_g: float,
    density_g_cm3: float,
    b_mm: float,
    alpha: float,
    gamma: float,
) -> Dict[str, float]:
    """Convert component/bin mass to estimated particle number."""
    mass_g = float(mass_g)
    if not np.isfinite(mass_g) or mass_g < 0:
        raise ValueError("Mass must be >= 0 g.")
    validate_particle_model(density_g_cm3, b_mm, alpha, gamma)
    geom = ellipsoid_geometry(b_mm, alpha, gamma)
    particle_mass_g = float(density_g_cm3) * geom["volume_cm3"]
    if not np.isfinite(particle_mass_g) or particle_mass_g <= 0:
        raise ValueError("Calculated particle mass is invalid.")
    n = mass_g / particle_mass_g
    return {
        **geom,
        "density_g_cm3": float(density_g_cm3),
        "component_mass_g": mass_g,
        "particle_mass_g": float(particle_mass_g),
        "particle_number": float(n),
    }


def linear_regression(x: Sequence[float], y: Sequence[float], force_intercept_zero: bool = False) -> Dict[str, float]:
    """Float64 linear regression with optional intercept constrained to zero."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        raise ValueError("At least 2 valid points are required for regression.")

    if force_intercept_zero:
        denom = float(np.dot(x, x))
        if not np.isfinite(denom) or abs(denom) <= np.finfo(np.float64).eps:
            raise ValueError("Cannot fit a zero-intercept regression for this X range.")
        slope = float(np.dot(x, y) / denom)
        intercept = 0.0
    else:
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)

    predicted = slope * x + intercept
    residual = y - predicted
    sse = float(np.dot(residual, residual))
    centered = y - float(np.mean(y))
    sst = float(np.dot(centered, centered))
    r2 = float(1.0 - sse / sst) if sst > np.finfo(np.float64).eps else float("nan")
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n": int(x.size),
    }


def run_numeric_self_tests() -> Dict[str, bool]:
    """Minimum numerical checks required by the implementation specification."""
    out: Dict[str, bool] = {}

    b = 2.5
    sphere = ellipsoid_geometry(b, 1.0, 1.0)
    expected_sphere = math.pi / 6.0 * b ** 3
    out["sphere_volume"] = bool(np.isclose(sphere["volume_mm3"], expected_sphere, rtol=1e-12, atol=1e-12))

    lam = ellipsoid_geometry(b, 1.5, 0.1)
    out["laminar_volume_factor"] = bool(np.isclose(lam["volume_mm3"] / sphere["volume_mm3"], 0.15, rtol=1e-12))
    out["laminar_number_factor"] = bool(np.isclose(lam["n_relative_sphere"], 1.0 / 0.15, rtol=1e-12))

    n1 = estimate_particle_number(10.0, 1.0, b, 1.0, 1.0)["particle_number"]
    n2 = estimate_particle_number(10.0, 2.0, b, 1.0, 1.0)["particle_number"]
    out["density_inverse"] = bool(np.isclose(n2 / n1, 0.5, rtol=1e-12))

    n_big = estimate_particle_number(10.0, 1.0, 2.0 * b, 1.0, 1.0)["particle_number"]
    out["size_cube"] = bool(np.isclose(n_big / n1, 1.0 / 8.0, rtol=1e-12))

    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    y_number = -2.40 * x + 5.25
    reg_n = linear_regression(x, y_number, False)
    out["number_dimension"] = bool(np.isclose(-reg_n["slope"], 2.40, rtol=1e-12))

    y_mass = 0.70 * x - 0.3
    reg_m = linear_regression(x, y_mass, False)
    out["mass_dimension"] = bool(np.isclose(3.0 - reg_m["slope"], 2.30, rtol=1e-12))

    y_scaled = y_number + math.log10(37.0)
    reg_scaled = linear_regression(x, y_scaled, False)
    out["mass_scaling_slope"] = bool(np.isclose(reg_scaled["slope"], reg_n["slope"], rtol=1e-12))
    return out


# =============================================================================
# Persistent settings helpers
# =============================================================================

def _phi_key(phi: float) -> str:
    return format(float(phi), ".12g")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _json_float_or_none(value: Any) -> Optional[float]:
    v = _safe_float(value, float("nan"))
    return float(v) if np.isfinite(v) else None


def _default_component_settings() -> Dict[str, Any]:
    return {
        "density_mode": DENSITY_UNIFORM,
        "uniform_density": None,
        "shape_mode": SHAPE_UNIFORM,
        "uniform_shape": "Sphere",
        "uniform_alpha": 1.0,
        "uniform_gamma": 1.0,
        "by_phi": {},
    }


def default_fractal_settings() -> Dict[str, Any]:
    return {
        "schema_version": FRACTAL_SETTINGS_SCHEMA_VERSION,
        "mass_basis": MASS_BASIS_NORMALIZED,
        "actual_sample_mass_g": None,
        "representative_size_mode": REPRESENTATIVE_CLASS_MIDPOINT,
        "force_intercept_zero": False,
        "components": {},
        "fit_use": {"N": {}, "M": {}},
    }


def ensure_fractal_settings(db: Any) -> Dict[str, Any]:
    raw = getattr(db, "fractal_settings", None)
    if not isinstance(raw, dict):
        raw = {}
    defaults = default_fractal_settings()
    settings = raw
    for key, value in defaults.items():
        if key not in settings:
            settings[key] = copy.deepcopy(value)
    if settings.get("mass_basis") not in (MASS_BASIS_NORMALIZED, MASS_BASIS_ACTUAL):
        settings["mass_basis"] = MASS_BASIS_NORMALIZED
    if settings.get("representative_size_mode") not in (REPRESENTATIVE_CLASS_MIDPOINT, REPRESENTATIVE_MANUAL):
        settings["representative_size_mode"] = REPRESENTATIVE_CLASS_MIDPOINT
    settings["force_intercept_zero"] = bool(settings.get("force_intercept_zero", False))
    if not isinstance(settings.get("components"), dict):
        settings["components"] = {}
    if not isinstance(settings.get("fit_use"), dict):
        settings["fit_use"] = {"N": {}, "M": {}}
    settings["fit_use"].setdefault("N", {})
    settings["fit_use"].setdefault("M", {})
    settings["schema_version"] = FRACTAL_SETTINGS_SCHEMA_VERSION
    db.fractal_settings = settings
    return settings


def ensure_component_settings(db: Any, component: str, phis: Iterable[float]) -> Dict[str, Any]:
    settings = ensure_fractal_settings(db)
    comp_map = settings.setdefault("components", {})
    c = str(component)
    cs = comp_map.get(c)
    if not isinstance(cs, dict):
        cs = _default_component_settings()
        comp_map[c] = cs
    defaults = _default_component_settings()
    for k, v in defaults.items():
        if k not in cs:
            cs[k] = copy.deepcopy(v)
    if cs.get("density_mode") not in (DENSITY_UNIFORM, DENSITY_BY_FRACTION):
        cs["density_mode"] = DENSITY_UNIFORM
    if cs.get("shape_mode") not in (SHAPE_UNIFORM, SHAPE_BY_FRACTION):
        cs["shape_mode"] = SHAPE_UNIFORM
    if cs.get("uniform_shape") not in SHAPE_PRESETS:
        cs["uniform_shape"] = "Custom"
    if not isinstance(cs.get("by_phi"), dict):
        cs["by_phi"] = {}

    for phi in phis:
        pk = _phi_key(phi)
        row = cs["by_phi"].get(pk)
        if not isinstance(row, dict):
            row = {}
            cs["by_phi"][pk] = row
        row.setdefault("density", None)
        row.setdefault("shape", "Sphere")
        row.setdefault("alpha", 1.0)
        row.setdefault("gamma", 1.0)
        row.setdefault("manual_b_mm", None)
        if row.get("shape") not in SHAPE_PRESETS:
            row["shape"] = "Custom"
    return cs


def sample_phi_rows(db: Any) -> List[float]:
    """
    Return the physical sample interval, trimming zero padding outside the first
    and last non-zero whole-sample GSD bins, matching Componentry's histogram.
    """
    if db is None or not isinstance(getattr(db, "gsd", None), pd.DataFrame):
        return []
    gsd = db.gsd.copy()
    if "phi" not in gsd.columns or "wt" not in gsd.columns:
        return []
    gsd["phi"] = pd.to_numeric(gsd["phi"], errors="coerce")
    gsd["wt"] = pd.to_numeric(gsd["wt"], errors="coerce")
    gsd = gsd[np.isfinite(gsd["phi"]) & np.isfinite(gsd["wt"])].copy()
    if gsd.empty:
        return []
    gsd = gsd.groupby("phi", as_index=False)["wt"].sum().sort_values("phi").reset_index(drop=True)
    positive = np.flatnonzero(gsd["wt"].to_numpy(dtype=float) > 0.0)
    if positive.size == 0:
        return []
    gsd = gsd.iloc[int(positive[0]):int(positive[-1]) + 1]
    return [float(v) for v in gsd["phi"].to_numpy(dtype=float)]


def _gsd_wt_map(db: Any) -> Dict[float, float]:
    out: Dict[float, float] = {}
    if db is None or not isinstance(getattr(db, "gsd", None), pd.DataFrame):
        return out
    for _, row in db.gsd.iterrows():
        phi = _safe_float(row.get("phi"), float("nan"))
        wt = _safe_float(row.get("wt"), 0.0)
        if np.isfinite(phi):
            out[float(phi)] = out.get(float(phi), 0.0) + max(0.0, float(wt))
    return out


def component_mass_wt(db: Any, component: str, phi: float) -> float:
    """Mass contribution of a component to the complete sample, in wt%."""
    phi = float(phi)
    component = str(component)
    wt_map = _gsd_wt_map(db)
    bin_wt = max(0.0, float(wt_map.get(phi, 0.0)))

    if component == UNDEFINED_COMPONENT:
        known = 0.0
        total = getattr(db, "total", None)
        components = list(getattr(db, "components", []) or [])
        if isinstance(total, pd.DataFrame) and phi in total.index:
            for c in components:
                if c in total.columns:
                    v = _safe_float(total.loc[phi, c], 0.0)
                    if np.isfinite(v) and v > 0:
                        known += float(v)
        return max(0.0, bin_wt - known)

    total = getattr(db, "total", None)
    if not isinstance(total, pd.DataFrame) or component not in total.columns or phi not in total.index:
        return 0.0
    v = _safe_float(total.loc[phi, component], 0.0)
    return max(0.0, float(v)) if np.isfinite(v) else 0.0


def undefined_mass_exists(db: Any, phis: Optional[Iterable[float]] = None, tol: float = 1e-10) -> bool:
    phis = list(phis if phis is not None else sample_phi_rows(db))
    return any(component_mass_wt(db, UNDEFINED_COMPONENT, phi) > tol for phi in phis)


def observed_component_types(db: Any) -> List[str]:
    out: List[str] = []
    for c in list(getattr(db, "components", []) or []):
        t = str((getattr(db, "types", {}) or {}).get(c, "Accessory")).strip() or "Accessory"
        if t not in out:
            out.append(t)
    preferred = ["Juvenile", "Lithic", "Accessory"]
    ordered = [t for t in preferred if t in out]
    ordered.extend([t for t in out if t not in ordered])
    return ordered


def components_for_type(db: Any, component_type: str) -> List[str]:
    types = getattr(db, "types", {}) or {}
    return [c for c in list(getattr(db, "components", []) or []) if str(types.get(c, "Accessory")) == str(component_type)]


def population_key(kind: str, value: Optional[str] = None) -> str:
    if kind == "whole":
        return "whole"
    if kind == "type":
        return f"type:{value or ''}"
    if kind == "component":
        return f"component:{value or ''}"
    return f"{kind}:{value or ''}"


def population_label(kind: str, value: Optional[str] = None) -> str:
    if kind == "whole":
        return "Whole sample"
    if kind == "type":
        mapping = {
            "Juvenile": "All juveniles",
            "Lithic": "All lithics",
            "Accessory": "All accessories",
            "Crystal": "All crystals",
        }
        return mapping.get(str(value), f"All {str(value).lower()}")
    if kind == "component":
        return str(value or "Individual component")
    return str(value or kind)


def population_components(db: Any, kind: str, value: Optional[str] = None, include_undefined_for_whole: bool = True) -> List[str]:
    if kind == "whole":
        comps = list(getattr(db, "components", []) or [])
        if include_undefined_for_whole and undefined_mass_exists(db):
            comps.append(UNDEFINED_COMPONENT)
        return comps
    if kind == "type":
        return components_for_type(db, str(value))
    if kind == "component":
        return [str(value)] if value else []
    return []


def whole_componentry_coverage_percent(db: Any, phis: Optional[Iterable[float]] = None) -> float:
    phis = list(phis if phis is not None else sample_phi_rows(db))
    wt_map = _gsd_wt_map(db)
    total_mass = sum(max(0.0, wt_map.get(phi, 0.0)) for phi in phis)
    if total_mass <= 0:
        return float("nan")
    assigned = 0.0
    for phi in phis:
        known = 0.0
        for c in list(getattr(db, "components", []) or []):
            known += component_mass_wt(db, c, phi)
        assigned += min(max(0.0, wt_map.get(phi, 0.0)), known)
    return float(100.0 * assigned / total_mass)


def effective_component_params(db: Any, component: str, phi: float) -> Dict[str, Any]:
    phis = sample_phi_rows(db)
    cs = ensure_component_settings(db, component, phis)
    row = cs["by_phi"].setdefault(_phi_key(phi), {
        "density": None,
        "shape": "Sphere",
        "alpha": 1.0,
        "gamma": 1.0,
        "manual_b_mm": None,
    })

    if cs.get("density_mode") == DENSITY_BY_FRACTION:
        density = _json_float_or_none(row.get("density"))
    else:
        density = _json_float_or_none(cs.get("uniform_density"))

    if cs.get("shape_mode") == SHAPE_BY_FRACTION:
        shape = str(row.get("shape", "Sphere"))
        alpha = _safe_float(row.get("alpha"), 1.0)
        gamma = _safe_float(row.get("gamma"), 1.0)
    else:
        shape = str(cs.get("uniform_shape", "Sphere"))
        alpha = _safe_float(cs.get("uniform_alpha"), 1.0)
        gamma = _safe_float(cs.get("uniform_gamma"), 1.0)

    if shape == "Sphere":
        alpha = 1.0
        gamma = 1.0

    settings = ensure_fractal_settings(db)
    rep_mode = settings.get("representative_size_mode", REPRESENTATIVE_CLASS_MIDPOINT)
    manual_b = _json_float_or_none(row.get("manual_b_mm"))
    if rep_mode == REPRESENTATIVE_MANUAL:
        b_mm = manual_b
    else:
        # Componentry stores one phi coordinate for each histogram bin and renders
        # that coordinate as the bin center.  Therefore the class-midpoint model
        # uses the stored phi directly as phi_mid and B = 2^(-phi_mid).
        b_mm = phi_to_mm(phi)

    return {
        "density": density,
        "shape": shape,
        "alpha": float(alpha),
        "gamma": float(gamma),
        "b_mm": b_mm,
        "manual_b_mm": manual_b,
        "representative_size_mode": rep_mode,
    }


def _population_mass_wt_series(db: Any, kind: str, value: Optional[str], phis: Sequence[float]) -> Dict[float, float]:
    if kind == "whole":
        wt_map = _gsd_wt_map(db)
        return {float(phi): max(0.0, float(wt_map.get(float(phi), 0.0))) for phi in phis}
    if kind == "type":
        comps = components_for_type(db, str(value))
        return {float(phi): float(sum(component_mass_wt(db, c, phi) for c in comps)) for phi in phis}
    if kind == "component":
        c = str(value or "")
        return {float(phi): float(component_mass_wt(db, c, phi)) for phi in phis}
    return {float(phi): 0.0 for phi in phis}


def _fit_use_map(db: Any, method_code: str, pop_key: str) -> Dict[str, bool]:
    settings = ensure_fractal_settings(db)
    fit_use = settings.setdefault("fit_use", {}).setdefault(method_code, {})
    pmap = fit_use.get(pop_key)
    if not isinstance(pmap, dict):
        pmap = {}
        fit_use[pop_key] = pmap
    return pmap


def fit_use_for_phi(db: Any, method_code: str, pop_key: str, phi: float) -> bool:
    pmap = _fit_use_map(db, method_code, pop_key)
    return bool(pmap.get(_phi_key(phi), True))


def set_fit_use_for_phi(db: Any, method_code: str, pop_key: str, phi: float, use: bool) -> None:
    pmap = _fit_use_map(db, method_code, pop_key)
    pmap[_phi_key(phi)] = bool(use)


# =============================================================================
# Scientific calculations
# =============================================================================

def calculate_mass_method(
    db: Any,
    kind: str,
    value: Optional[str],
    force_intercept_zero: Optional[bool] = None,
) -> Dict[str, Any]:
    phis = sample_phi_rows(db)
    pop_key = population_key(kind, value)
    masses = _population_mass_wt_series(db, kind, value, phis)
    total_mass = float(sum(max(0.0, masses.get(phi, 0.0)) for phi in phis))
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if kind != "whole":
        cov = whole_componentry_coverage_percent(db, phis)
        if np.isfinite(cov) and cov < 99.999:
            warnings.append(f"Componentry coverage of the whole sample is {cov:.1f}%. Component/group masses use only available componentry.")

    if total_mass <= 0:
        return {
            "method": "M",
            "rows": [],
            "fit": None,
            "D": float("nan"),
            "v": float("nan"),
            "warnings": warnings + ["Selected population has no positive mass."],
            "population_key": pop_key,
            "population_mass_wt": total_mass,
        }

    # phi ascending = coarse -> fine because d=2^-phi decreases with increasing phi.
    # M(r<R) is therefore obtained by reverse cumulative sum (current + finer bins).
    mass_arr = np.array([max(0.0, masses.get(phi, 0.0)) for phi in phis], dtype=np.float64)
    cumulative_finer = np.cumsum(mass_arr[::-1], dtype=np.float64)[::-1]

    for i, phi in enumerate(phis):
        r_mm = phi_to_mm(phi)
        cum = float(cumulative_finer[i])
        frac = cum / total_mass if total_mass > 0 else float("nan")
        log_r = math.log10(r_mm) if r_mm > 0 else float("nan")
        log_f = math.log10(frac) if frac > 0 else float("nan")
        valid = bool(np.isfinite(log_r) and np.isfinite(log_f) and r_mm > 0 and frac > 0)
        use = valid and fit_use_for_phi(db, "M", pop_key, phi)
        rows.append({
            "use": bool(use),
            "valid": valid,
            "phi": float(phi),
            "R_mm": float(r_mm),
            "bin_mass": float(mass_arr[i]),
            "cumulative_mass_finer": cum,
            "mass_fraction": float(frac),
            "log10_R": float(log_r),
            "log10_mass_fraction": float(log_f),
        })

    if force_intercept_zero is None:
        force_intercept_zero = bool(ensure_fractal_settings(db).get("force_intercept_zero", False))

    used = [r for r in rows if r["use"] and r["valid"]]
    fit = None
    D = float("nan")
    v = float("nan")
    if len(used) >= 2:
        try:
            fit = linear_regression(
                [r["log10_R"] for r in used],
                [r["log10_mass_fraction"] for r in used],
                bool(force_intercept_zero),
            )
            v = float(fit["slope"])
            D = float(3.0 - v)
        except Exception as exc:
            warnings.append(str(exc))
    else:
        warnings.append("Fewer than 2 valid mass-based points are selected for regression.")

    return {
        "method": "M",
        "rows": rows,
        "fit": fit,
        "D": D,
        "v": v,
        "warnings": warnings,
        "population_key": pop_key,
        "population_mass_wt": total_mass,
    }


def calculate_number_method(
    db: Any,
    kind: str,
    value: Optional[str],
    force_intercept_zero: Optional[bool] = None,
) -> Dict[str, Any]:
    phis = sample_phi_rows(db)
    pop_key = population_key(kind, value)
    settings = ensure_fractal_settings(db)
    components = population_components(db, kind, value, include_undefined_for_whole=True)
    warnings: List[str] = []
    errors: List[str] = []
    detail_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []

    if not components:
        return {
            "method": "N",
            "rows": [],
            "detail_rows": [],
            "fit": None,
            "D": float("nan"),
            "warnings": ["Selected population contains no components."],
            "errors": [],
            "population_key": pop_key,
            "componentry_coverage": whole_componentry_coverage_percent(db, phis),
            "modeled_coverage": float("nan"),
        }

    mass_basis = settings.get("mass_basis", MASS_BASIS_NORMALIZED)
    if mass_basis == MASS_BASIS_ACTUAL:
        actual_mass = _json_float_or_none(settings.get("actual_sample_mass_g"))
        if actual_mass is None or actual_mass <= 0:
            errors.append("Actual sample mass must be > 0 g when 'Actual sample mass' is selected.")
            mass_scale = float("nan")
        else:
            mass_scale = float(actual_mass / 100.0)
    else:
        mass_scale = 1.0

    missing_density: List[str] = []
    invalid_geometry: List[str] = []
    missing_b: List[str] = []
    selected_mass_wt_total = 0.0
    modeled_mass_wt = 0.0
    n_by_phi: Dict[float, float] = {float(phi): 0.0 for phi in phis}
    detail_by_phi: Dict[float, List[Dict[str, Any]]] = {float(phi): [] for phi in phis}

    for phi in phis:
        for comp in components:
            mass_wt = component_mass_wt(db, comp, phi)
            if mass_wt <= 0:
                continue
            selected_mass_wt_total += mass_wt
            params = effective_component_params(db, comp, phi)
            density = params.get("density")
            b_mm = params.get("b_mm")
            alpha = _safe_float(params.get("alpha"), float("nan"))
            gamma = _safe_float(params.get("gamma"), float("nan"))
            shape = str(params.get("shape", "Custom"))
            component_mass_g = float(mass_wt * mass_scale) if np.isfinite(mass_scale) else float("nan")

            row = {
                "phi": float(phi),
                "threshold_r_mm": phi_to_mm(phi),
                "component": comp,
                "component_mass_wt": float(mass_wt),
                "component_mass_g": component_mass_g,
                "B_mm": b_mm if b_mm is not None else float("nan"),
                "density_g_cm3": density if density is not None else float("nan"),
                "shape": shape,
                "A_over_B": alpha,
                "C_over_B": gamma,
                "A_mm": float("nan"),
                "C_mm": float("nan"),
                "particle_volume_mm3": float("nan"),
                "particle_volume_cm3": float("nan"),
                "particle_mass_g": float("nan"),
                "particle_number_bin_component": float("nan"),
                "shape_volume_factor": float("nan"),
                "N_relative_to_sphere": float("nan"),
                "valid_model": False,
                "model_error": "",
            }

            # Actual-mass mode cannot produce particle counts until a positive
            # sample mass has been supplied.  Keep the row visible for inspection,
            # but do not cascade that missing scalar into misleading geometry errors.
            if not np.isfinite(mass_scale):
                row["model_error"] = "Actual sample mass undefined"
                detail_by_phi[float(phi)].append(row)
                continue

            if density is None or not np.isfinite(_safe_float(density, float("nan"))) or float(density) <= 0:
                msg = f"{comp} at phi {phi:g}"
                missing_density.append(msg)
                row["model_error"] = "Missing/invalid density"
                detail_by_phi[float(phi)].append(row)
                continue
            if b_mm is None or not np.isfinite(_safe_float(b_mm, float("nan"))) or float(b_mm) <= 0:
                msg = f"{comp} at phi {phi:g}"
                missing_b.append(msg)
                row["model_error"] = "Missing/invalid representative B"
                detail_by_phi[float(phi)].append(row)
                continue
            try:
                calc = estimate_particle_number(component_mass_g, float(density), float(b_mm), alpha, gamma)
                row.update({
                    "A_mm": calc["A_mm"],
                    "B_mm": calc["B_mm"],
                    "C_mm": calc["C_mm"],
                    "particle_volume_mm3": calc["volume_mm3"],
                    "particle_volume_cm3": calc["volume_cm3"],
                    "particle_mass_g": calc["particle_mass_g"],
                    "particle_number_bin_component": calc["particle_number"],
                    "shape_volume_factor": calc["shape_volume_factor"],
                    "N_relative_to_sphere": calc["n_relative_sphere"],
                    "valid_model": True,
                })
                n_by_phi[float(phi)] += float(calc["particle_number"])
                modeled_mass_wt += mass_wt
            except Exception as exc:
                invalid_geometry.append(f"{comp} at phi {phi:g}: {exc}")
                row["model_error"] = str(exc)
            detail_by_phi[float(phi)].append(row)

    componentry_coverage = whole_componentry_coverage_percent(db, phis)
    modeled_coverage = float(100.0 * modeled_mass_wt / selected_mass_wt_total) if selected_mass_wt_total > 0 else float("nan")

    if np.isfinite(componentry_coverage) and componentry_coverage < 99.999:
        warnings.append(f"Componentry coverage: {componentry_coverage:.1f}%.")
    if missing_density:
        unique = list(dict.fromkeys(missing_density))
        errors.append("Missing density for: " + ", ".join(unique[:20]) + (" ..." if len(unique) > 20 else ""))
    if missing_b:
        unique = list(dict.fromkeys(missing_b))
        errors.append("Missing representative B for: " + ", ".join(unique[:20]) + (" ..." if len(unique) > 20 else ""))
    if invalid_geometry:
        errors.append("Invalid particle geometry: " + " | ".join(invalid_geometry[:12]) + (" ..." if len(invalid_geometry) > 12 else ""))

    # Required model parameters must be complete before N(>r) and regression are
    # considered valid.  This avoids silently ignoring unmodelled positive mass.
    model_complete = (not errors) and np.isfinite(mass_scale)

    cumulative = 0.0
    for phi in phis:
        n_bin = float(n_by_phi.get(float(phi), 0.0))
        cumulative += n_bin
        r_mm = phi_to_mm(phi)
        log_r = math.log10(r_mm) if r_mm > 0 else float("nan")
        log_n = math.log10(cumulative) if cumulative > 0 else float("nan")
        valid = bool(model_complete and np.isfinite(log_r) and np.isfinite(log_n) and r_mm > 0 and cumulative > 0)
        use = valid and fit_use_for_phi(db, "N", pop_key, phi)
        aggregate = {
            "use": bool(use),
            "valid": valid,
            "phi": float(phi),
            "threshold_r_mm": float(r_mm),
            "particle_number_bin_total": n_bin,
            "N_gt_r": float(cumulative),
            "log10_r": float(log_r),
            "log10_N_gt_r": float(log_n),
        }
        aggregate_rows.append(aggregate)
        for row in detail_by_phi[float(phi)]:
            row.update({
                "use": bool(use),
                "particle_number_bin_total": n_bin,
                "N_gt_r": float(cumulative),
                "log10_r": float(log_r),
                "log10_N_gt_r": float(log_n),
            })
            detail_rows.append(row)

    if force_intercept_zero is None:
        force_intercept_zero = bool(settings.get("force_intercept_zero", False))

    fit = None
    D = float("nan")
    used = [r for r in aggregate_rows if r["use"] and r["valid"]]
    if model_complete:
        if len(used) >= 2:
            try:
                fit = linear_regression(
                    [r["log10_r"] for r in used],
                    [r["log10_N_gt_r"] for r in used],
                    bool(force_intercept_zero),
                )
                D = float(-fit["slope"])
            except Exception as exc:
                warnings.append(str(exc))
        else:
            warnings.append("Fewer than 2 valid number-based points are selected for regression.")

    return {
        "method": "N",
        "rows": aggregate_rows,
        "detail_rows": detail_rows,
        "fit": fit,
        "D": D,
        "warnings": warnings,
        "errors": errors,
        "population_key": pop_key,
        "componentry_coverage": componentry_coverage,
        "modeled_coverage": modeled_coverage,
        "mass_basis": mass_basis,
        "selected_mass_wt": float(selected_mass_wt_total),
    }


# =============================================================================
# Small UI helpers
# =============================================================================

def _read_only_item(text: str, user_data: Any = None) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    if user_data is not None:
        item.setData(Qt.UserRole, user_data)
    return item


def _number_text(value: Any, digits: int = 6) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e6 or av < 1e-4:
        return f"{v:.{digits}e}"
    return f"{v:.{digits}g}"


def _sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip().strip(".")
    return s or "fractal_analysis"


class ParticleModelWidget(QWidget):
    """Central editor for density, shape and representative B by component/phi."""

    def __init__(self, owner: "FractalAnalysisWindow"):
        super().__init__(owner)
        self.owner = owner
        self._loading = False
        self._shape_combos: Dict[int, QComboBox] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(7)

        top = QGridLayout()
        top.addWidget(QLabel("Component:"), 0, 0)
        self.cmb_component = QComboBox()
        self.cmb_component.currentIndexChanged.connect(self.refresh)
        top.addWidget(self.cmb_component, 0, 1)

        top.addWidget(QLabel("Density mode:"), 1, 0)
        self.cmb_density_mode = QComboBox()
        self.cmb_density_mode.addItems([DENSITY_UNIFORM, DENSITY_BY_FRACTION])
        self.cmb_density_mode.currentIndexChanged.connect(self._on_modes_changed)
        top.addWidget(self.cmb_density_mode, 1, 1)

        top.addWidget(QLabel("Uniform density:"), 1, 2)
        self.sp_density = QDoubleSpinBox()
        self.sp_density.setDecimals(5)
        self.sp_density.setRange(0.0, 30.0)
        self.sp_density.setSingleStep(0.05)
        self.sp_density.setSpecialValueText("Undefined")
        self.sp_density.setSuffix(" g/cm3")
        self.sp_density.valueChanged.connect(self._on_uniform_changed)
        top.addWidget(self.sp_density, 1, 3)

        top.addWidget(QLabel("Shape mode:"), 2, 0)
        self.cmb_shape_mode = QComboBox()
        self.cmb_shape_mode.addItems([SHAPE_UNIFORM, SHAPE_BY_FRACTION])
        self.cmb_shape_mode.currentIndexChanged.connect(self._on_modes_changed)
        top.addWidget(self.cmb_shape_mode, 2, 1)

        top.addWidget(QLabel("Uniform shape:"), 2, 2)
        self.cmb_shape = QComboBox()
        self.cmb_shape.addItems(list(SHAPE_PRESETS.keys()))
        self.cmb_shape.currentIndexChanged.connect(self._on_uniform_shape_changed)
        top.addWidget(self.cmb_shape, 2, 3)

        top.addWidget(QLabel("A/B:"), 3, 0)
        self.sp_alpha = QDoubleSpinBox()
        self.sp_alpha.setDecimals(5)
        self.sp_alpha.setRange(1.0, 100.0)
        self.sp_alpha.setSingleStep(0.05)
        self.sp_alpha.valueChanged.connect(self._on_uniform_changed)
        top.addWidget(self.sp_alpha, 3, 1)

        top.addWidget(QLabel("C/B:"), 3, 2)
        self.sp_gamma = QDoubleSpinBox()
        self.sp_gamma.setDecimals(5)
        self.sp_gamma.setRange(0.00001, 1.0)
        self.sp_gamma.setSingleStep(0.05)
        self.sp_gamma.valueChanged.connect(self._on_uniform_changed)
        top.addWidget(self.sp_gamma, 3, 3)
        top.setColumnStretch(1, 1)
        top.setColumnStretch(3, 1)
        root.addLayout(top)

        note = QLabel(
            "B is used as the representative particle dimension associated with the sieve size class. "
            "Preset shape names are editable starting values; calculations always use the displayed A/B and C/B."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555555;")
        root.addWidget(note)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "phi", "B mm", "Density g/cm3", "Shape", "A/B", "C/B",
            "Volume factor", "N/N sphere",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_table_item_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        root.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_apply_density_all = QPushButton("Apply density to all fractions")
        self.btn_apply_shape_all = QPushButton("Apply shape to all fractions")
        self.btn_copy_all = QPushButton("Copy parameters to all fractions")
        self.btn_copy_selected = QPushButton("Copy parameters to selected fractions")
        self.btn_copy_previous = QPushButton("Copy from previous fraction")
        self.btn_sphere = QPushButton("Reset shape to Sphere")
        self.btn_reset = QPushButton("Reset component parameters")
        for b in (
            self.btn_apply_density_all, self.btn_apply_shape_all, self.btn_copy_all,
            self.btn_copy_selected, self.btn_copy_previous, self.btn_sphere, self.btn_reset,
        ):
            buttons.addWidget(b)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.btn_apply_density_all.clicked.connect(self.apply_density_to_all)
        self.btn_apply_shape_all.clicked.connect(self.apply_shape_to_all)
        self.btn_copy_all.clicked.connect(self.copy_parameters_to_all)
        self.btn_copy_selected.clicked.connect(self.copy_parameters_to_selected)
        self.btn_copy_previous.clicked.connect(self.copy_from_previous)
        self.btn_sphere.clicked.connect(self.reset_shape_sphere)
        self.btn_reset.clicked.connect(self.reset_component_parameters)

    def _db(self) -> Any:
        return self.owner.current_db()

    def current_component(self) -> str:
        return str(self.cmb_component.currentText()).strip()

    def refresh_components(self, preferred: Optional[str] = None) -> None:
        db = self._db()
        current = preferred or self.current_component()
        self.cmb_component.blockSignals(True)
        self.cmb_component.clear()
        if db is not None:
            comps = list(getattr(db, "components", []) or [])
            if undefined_mass_exists(db):
                comps.append(UNDEFINED_COMPONENT)
            self.cmb_component.addItems(comps)
        self.cmb_component.blockSignals(False)
        if current:
            idx = self.cmb_component.findText(current)
            if idx >= 0:
                self.cmb_component.setCurrentIndex(idx)
        if self.cmb_component.count() and self.cmb_component.currentIndex() < 0:
            self.cmb_component.setCurrentIndex(0)
        self.refresh()

    def refresh(self, *args) -> None:
        if self._loading:
            return
        db = self._db()
        comp = self.current_component()
        self._loading = True
        try:
            if db is None or not comp:
                self.table.setRowCount(0)
                return
            phis = sample_phi_rows(db)
            cs = ensure_component_settings(db, comp, phis)
            self.cmb_density_mode.setCurrentText(cs.get("density_mode", DENSITY_UNIFORM))
            density = _json_float_or_none(cs.get("uniform_density"))
            self.sp_density.setValue(float(density) if density is not None else 0.0)
            self.cmb_shape_mode.setCurrentText(cs.get("shape_mode", SHAPE_UNIFORM))
            shape = str(cs.get("uniform_shape", "Sphere"))
            self.cmb_shape.setCurrentText(shape if shape in SHAPE_PRESETS else "Custom")
            self.sp_alpha.setValue(max(1.0, _safe_float(cs.get("uniform_alpha"), 1.0)))
            self.sp_gamma.setValue(min(1.0, max(0.00001, _safe_float(cs.get("uniform_gamma"), 1.0))))
            self._update_uniform_control_state()
            self._fill_table(db, comp, phis, cs)
        finally:
            self._loading = False

    def _update_uniform_control_state(self) -> None:
        density_uniform = self.cmb_density_mode.currentText() == DENSITY_UNIFORM
        shape_uniform = self.cmb_shape_mode.currentText() == SHAPE_UNIFORM
        self.sp_density.setEnabled(density_uniform)
        self.cmb_shape.setEnabled(shape_uniform)
        sphere = self.cmb_shape.currentText() == "Sphere"
        self.sp_alpha.setEnabled(shape_uniform and not sphere)
        self.sp_gamma.setEnabled(shape_uniform and not sphere)

    def _fill_table(self, db: Any, comp: str, phis: List[float], cs: Dict[str, Any]) -> None:
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(len(phis))
        self._shape_combos = {}
        rep_mode = ensure_fractal_settings(db).get("representative_size_mode", REPRESENTATIVE_CLASS_MIDPOINT)
        density_by = cs.get("density_mode") == DENSITY_BY_FRACTION
        shape_by = cs.get("shape_mode") == SHAPE_BY_FRACTION

        for row_idx, phi in enumerate(phis):
            pk = _phi_key(phi)
            row_cfg = cs["by_phi"].setdefault(pk, {
                "density": None, "shape": "Sphere", "alpha": 1.0, "gamma": 1.0, "manual_b_mm": None,
            })
            self.table.setItem(row_idx, 0, _read_only_item(f"{phi:g}", float(phi)))

            if rep_mode == REPRESENTATIVE_MANUAL:
                b = _json_float_or_none(row_cfg.get("manual_b_mm"))
                b_item = QTableWidgetItem("" if b is None else _number_text(b, 8))
            else:
                b_item = _read_only_item(_number_text(phi_to_mm(phi), 8))
            self.table.setItem(row_idx, 1, b_item)

            if density_by:
                rho = _json_float_or_none(row_cfg.get("density"))
                rho_item = QTableWidgetItem("" if rho is None else _number_text(rho, 8))
            else:
                rho_item = _read_only_item(_number_text(cs.get("uniform_density"), 8))
            self.table.setItem(row_idx, 2, rho_item)

            if shape_by:
                cmb = QComboBox()
                cmb.addItems(list(SHAPE_PRESETS.keys()))
                shape = str(row_cfg.get("shape", "Sphere"))
                cmb.setCurrentText(shape if shape in SHAPE_PRESETS else "Custom")
                cmb.currentTextChanged.connect(lambda text, r=row_idx: self._on_row_shape_changed(r, text))
                self.table.setCellWidget(row_idx, 3, cmb)
                self._shape_combos[row_idx] = cmb
                alpha = _safe_float(row_cfg.get("alpha"), 1.0)
                gamma = _safe_float(row_cfg.get("gamma"), 1.0)
                a_item = QTableWidgetItem(_number_text(alpha, 8))
                g_item = QTableWidgetItem(_number_text(gamma, 8))
                if shape == "Sphere":
                    a_item.setFlags(a_item.flags() & ~Qt.ItemIsEditable)
                    g_item.setFlags(g_item.flags() & ~Qt.ItemIsEditable)
            else:
                shape = str(cs.get("uniform_shape", "Sphere"))
                self.table.setItem(row_idx, 3, _read_only_item(shape))
                alpha = 1.0 if shape == "Sphere" else _safe_float(cs.get("uniform_alpha"), 1.0)
                gamma = 1.0 if shape == "Sphere" else _safe_float(cs.get("uniform_gamma"), 1.0)
                a_item = _read_only_item(_number_text(alpha, 8))
                g_item = _read_only_item(_number_text(gamma, 8))
            self.table.setItem(row_idx, 4, a_item)
            self.table.setItem(row_idx, 5, g_item)

            factor = alpha * gamma if np.isfinite(alpha) and np.isfinite(gamma) else float("nan")
            rel = 1.0 / factor if np.isfinite(factor) and factor > 0 else float("nan")
            self.table.setItem(row_idx, 6, _read_only_item(_number_text(factor, 8)))
            self.table.setItem(row_idx, 7, _read_only_item(_number_text(rel, 8)))

        self.table.blockSignals(False)

    def _on_modes_changed(self, *args) -> None:
        if self._loading:
            return
        db = self._db()
        comp = self.current_component()
        if db is None or not comp:
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cs["density_mode"] = self.cmb_density_mode.currentText()
        cs["shape_mode"] = self.cmb_shape_mode.currentText()
        self.owner.settings_changed(recalculate=True)
        self.refresh()

    def _on_uniform_shape_changed(self, shape: str) -> None:
        if self._loading:
            return
        db = self._db()
        comp = self.current_component()
        if db is None or not comp:
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        shape = str(shape)
        cs["uniform_shape"] = shape
        if shape in SHAPE_PRESETS and shape != "Custom":
            alpha, gamma = SHAPE_PRESETS[shape]
            cs["uniform_alpha"] = float(alpha)
            cs["uniform_gamma"] = float(gamma)
            self._loading = True
            try:
                self.sp_alpha.setValue(float(alpha))
                self.sp_gamma.setValue(float(gamma))
            finally:
                self._loading = False
        self._update_uniform_control_state()
        self.owner.settings_changed(recalculate=True)
        self.refresh()

    def _on_uniform_changed(self, *args) -> None:
        if self._loading:
            return
        db = self._db()
        comp = self.current_component()
        if db is None or not comp:
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cs["uniform_density"] = None if self.sp_density.value() <= 0 else float(self.sp_density.value())
        shape = self.cmb_shape.currentText()
        cs["uniform_shape"] = shape
        cs["uniform_alpha"] = 1.0 if shape == "Sphere" else float(self.sp_alpha.value())
        cs["uniform_gamma"] = 1.0 if shape == "Sphere" else float(self.sp_gamma.value())
        self.owner.settings_changed(recalculate=True)
        self.refresh()

    def _on_row_shape_changed(self, row_idx: int, shape: str) -> None:
        if self._loading:
            return
        db = self._db()
        comp = self.current_component()
        if db is None or not comp or row_idx < 0 or row_idx >= self.table.rowCount():
            return
        phi_item = self.table.item(row_idx, 0)
        if phi_item is None:
            return
        phi = _safe_float(phi_item.data(Qt.UserRole), float("nan"))
        if not np.isfinite(phi):
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cfg = cs["by_phi"].setdefault(_phi_key(phi), {})
        cfg["shape"] = str(shape)
        if shape in SHAPE_PRESETS and shape != "Custom":
            alpha, gamma = SHAPE_PRESETS[shape]
            cfg["alpha"] = float(alpha)
            cfg["gamma"] = float(gamma)
        self.owner.settings_changed(recalculate=True)
        self.refresh()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        row = item.row()
        col = item.column()
        if row < 0 or col not in (1, 2, 4, 5):
            return
        db = self._db()
        comp = self.current_component()
        if db is None or not comp:
            return
        phi_item = self.table.item(row, 0)
        phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else float("nan"), float("nan"))
        if not np.isfinite(phi):
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cfg = cs["by_phi"].setdefault(_phi_key(phi), {})
        val = _safe_float(item.text(), float("nan"))
        if col == 1 and ensure_fractal_settings(db).get("representative_size_mode") == REPRESENTATIVE_MANUAL:
            cfg["manual_b_mm"] = float(val) if np.isfinite(val) and val > 0 else None
        elif col == 2 and cs.get("density_mode") == DENSITY_BY_FRACTION:
            cfg["density"] = float(val) if np.isfinite(val) and val > 0 else None
        elif col == 4 and cs.get("shape_mode") == SHAPE_BY_FRACTION:
            cfg["alpha"] = float(val) if np.isfinite(val) else float("nan")
        elif col == 5 and cs.get("shape_mode") == SHAPE_BY_FRACTION:
            cfg["gamma"] = float(val) if np.isfinite(val) else float("nan")
        self.owner.settings_changed(recalculate=True)
        QTimer.singleShot(0, self.refresh)

    def _selected_source_row(self) -> Optional[int]:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if rows:
            return rows[0]
        return None

    def _row_config_snapshot(self, row: int) -> Optional[Dict[str, Any]]:
        db = self._db()
        comp = self.current_component()
        if db is None or not comp or row < 0 or row >= self.table.rowCount():
            return None
        phi_item = self.table.item(row, 0)
        phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else float("nan"), float("nan"))
        if not np.isfinite(phi):
            return None
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cfg = copy.deepcopy(cs["by_phi"].get(_phi_key(phi), {}))
        return cfg

    def apply_density_to_all(self) -> None:
        db = self._db(); comp = self.current_component()
        if db is None or not comp:
            return
        phis = sample_phi_rows(db)
        cs = ensure_component_settings(db, comp, phis)
        if cs.get("density_mode") == DENSITY_UNIFORM:
            density = _json_float_or_none(cs.get("uniform_density"))
        else:
            src = self._selected_source_row()
            snap = self._row_config_snapshot(src) if src is not None else None
            density = _json_float_or_none((snap or {}).get("density"))
        for phi in phis:
            cs["by_phi"][_phi_key(phi)]["density"] = density
        self.owner.settings_changed(recalculate=True); self.refresh()

    def apply_shape_to_all(self) -> None:
        db = self._db(); comp = self.current_component()
        if db is None or not comp:
            return
        phis = sample_phi_rows(db)
        cs = ensure_component_settings(db, comp, phis)
        if cs.get("shape_mode") == SHAPE_UNIFORM:
            shape = str(cs.get("uniform_shape", "Sphere"))
            alpha = 1.0 if shape == "Sphere" else _safe_float(cs.get("uniform_alpha"), 1.0)
            gamma = 1.0 if shape == "Sphere" else _safe_float(cs.get("uniform_gamma"), 1.0)
        else:
            src = self._selected_source_row()
            snap = self._row_config_snapshot(src) if src is not None else None
            if not snap:
                return
            shape = str(snap.get("shape", "Sphere"))
            alpha = _safe_float(snap.get("alpha"), 1.0)
            gamma = _safe_float(snap.get("gamma"), 1.0)
        for phi in phis:
            cfg = cs["by_phi"][_phi_key(phi)]
            cfg["shape"] = shape; cfg["alpha"] = float(alpha); cfg["gamma"] = float(gamma)
        self.owner.settings_changed(recalculate=True); self.refresh()

    def copy_parameters_to_all(self) -> None:
        src = self._selected_source_row()
        snap = self._row_config_snapshot(src) if src is not None else None
        db = self._db(); comp = self.current_component()
        if not snap or db is None or not comp:
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        for phi in sample_phi_rows(db):
            cs["by_phi"][_phi_key(phi)] = copy.deepcopy(snap)
        self.owner.settings_changed(recalculate=True); self.refresh()

    def copy_parameters_to_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        if len(rows) < 2:
            return
        snap = self._row_config_snapshot(rows[0])
        db = self._db(); comp = self.current_component()
        if not snap or db is None or not comp:
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        for row in rows[1:]:
            phi_item = self.table.item(row, 0)
            phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else float("nan"), float("nan"))
            if np.isfinite(phi):
                cs["by_phi"][_phi_key(phi)] = copy.deepcopy(snap)
        self.owner.settings_changed(recalculate=True); self.refresh()

    def copy_from_previous(self) -> None:
        src = self._selected_source_row()
        if src is None or src <= 0:
            return
        snap = self._row_config_snapshot(src - 1)
        db = self._db(); comp = self.current_component()
        if not snap or db is None or not comp:
            return
        phi_item = self.table.item(src, 0)
        phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else float("nan"), float("nan"))
        if not np.isfinite(phi):
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cs["by_phi"][_phi_key(phi)] = copy.deepcopy(snap)
        self.owner.settings_changed(recalculate=True); self.refresh()

    def reset_shape_sphere(self) -> None:
        db = self._db(); comp = self.current_component()
        if db is None or not comp:
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cs["uniform_shape"] = "Sphere"; cs["uniform_alpha"] = 1.0; cs["uniform_gamma"] = 1.0
        for cfg in cs["by_phi"].values():
            cfg["shape"] = "Sphere"; cfg["alpha"] = 1.0; cfg["gamma"] = 1.0
        self.owner.settings_changed(recalculate=True); self.refresh()

    def reset_component_parameters(self) -> None:
        db = self._db(); comp = self.current_component()
        if db is None or not comp:
            return
        ans = QMessageBox.question(self, "Reset particle model", f"Reset all fractal particle-model parameters for '{comp}'?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        settings = ensure_fractal_settings(db)
        settings.setdefault("components", {})[comp] = _default_component_settings()
        ensure_component_settings(db, comp, sample_phi_rows(db))
        self.owner.settings_changed(recalculate=True); self.refresh()


class FractalAnalysisWindow(QMainWindow):
    """Independent, resizable Fractal Analysis window."""

    def __init__(self, app: Any):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Fractal analysis")
        self.resize(1460, 900)
        self.setMinimumSize(1050, 700)
        self.setWindowFlags(self.windowFlags() | Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setFont(QFont("Arial", 10))

        self._loading = False
        self._recalculating = False
        self._last_sample_names: Tuple[str, ...] = tuple()
        self.number_result: Optional[Dict[str, Any]] = None
        self.mass_result: Optional[Dict[str, Any]] = None

        self._build_ui()
        self.refresh_from_componentry(initial=True)

        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(1200)
        self._watch_timer.timeout.connect(self._watch_componentry)
        self._watch_timer.start()

    # -------------------------------------------------------------------------
    # Host/sample access
    # -------------------------------------------------------------------------
    def sample_dbs(self) -> Dict[str, Any]:
        dbs = getattr(self.app, "sample_dbs", {})
        return dbs if isinstance(dbs, dict) else {}

    def current_sample_name(self) -> str:
        return str(self.cmb_sample.currentText()).strip()

    def current_db(self) -> Any:
        return self.sample_dbs().get(self.current_sample_name())

    def _notify_host_changed(self) -> None:
        try:
            cb = getattr(self.app, "_notify_host_componentry_changed", None)
            if callable(cb):
                cb()
        except Exception:
            pass

    def settings_changed(self, recalculate: bool = True) -> None:
        self._notify_host_changed()
        if recalculate:
            self.recalculate()

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        selectors = QGridLayout()
        selectors.addWidget(QLabel("Sample:"), 0, 0)
        self.cmb_sample = QComboBox()
        self.cmb_sample.setMinimumWidth(190)
        selectors.addWidget(self.cmb_sample, 0, 1)

        selectors.addWidget(QLabel("Population:"), 0, 2)
        self.cmb_population = QComboBox()
        self.cmb_population.setMinimumWidth(180)
        selectors.addWidget(self.cmb_population, 0, 3)

        self.lbl_component = QLabel("Component:")
        selectors.addWidget(self.lbl_component, 0, 4)
        self.cmb_component = QComboBox()
        self.cmb_component.setMinimumWidth(170)
        selectors.addWidget(self.cmb_component, 0, 5)

        selectors.addWidget(QLabel("Method:"), 0, 6)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems([METHOD_NUMBER, METHOD_MASS, METHOD_COMPARE])
        self.cmb_method.setMinimumWidth(160)
        selectors.addWidget(self.cmb_method, 0, 7)

        self.btn_refresh = QPushButton("Refresh from Componentry")
        selectors.addWidget(self.btn_refresh, 0, 8)
        self.btn_method_info = QPushButton("Method information")
        selectors.addWidget(self.btn_method_info, 0, 9)
        selectors.setColumnStretch(1, 1)
        selectors.setColumnStretch(3, 1)
        selectors.setColumnStretch(5, 1)
        root.addLayout(selectors)

        fit_box = QGroupBox("Regression point selection")
        fit_lay = QHBoxLayout(fit_box)
        fit_lay.addWidget(QLabel("Fit from phi:"))
        self.cmb_fit_from = QComboBox()
        fit_lay.addWidget(self.cmb_fit_from)
        fit_lay.addWidget(QLabel("Fit to phi:"))
        self.cmb_fit_to = QComboBox()
        fit_lay.addWidget(self.cmb_fit_to)
        self.btn_apply_range = QPushButton("Apply range")
        self.btn_select_all = QPushButton("Select all")
        self.btn_clear_all = QPushButton("Clear all")
        self.btn_invert = QPushButton("Invert selection")
        self.btn_recalculate = QPushButton("Recalculate")
        for b in (self.btn_apply_range, self.btn_select_all, self.btn_clear_all, self.btn_invert):
            fit_lay.addWidget(b)
        fit_lay.addStretch(1)
        fit_lay.addWidget(self.btn_recalculate)
        root.addWidget(fit_box)

        status_row = QHBoxLayout()
        self.lbl_coverage = QLabel("Componentry coverage: —")
        self.lbl_coverage.setStyleSheet("font-weight:600;")
        status_row.addWidget(self.lbl_coverage)
        status_row.addSpacing(18)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        status_row.addWidget(self.lbl_status, 1)
        root.addLayout(status_row)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        # Results tab
        results = QWidget()
        results_lay = QVBoxLayout(results)
        results_lay.setContentsMargins(4, 4, 4, 4)
        self.figure = Figure(figsize=(11.5, 6.6), dpi=95, facecolor="white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_lay.addWidget(self.canvas, 1)

        summary = QGridLayout()
        self.result_labels: Dict[str, QLabel] = {}
        keys = [
            ("DN", "D_N"), ("slopeN", "slope N"), ("intN", "intercept N"), ("r2N", "R² N"), ("nN", "n N"),
            ("rangeN", "fit range N"),
            ("DM", "D_M"), ("vM", "v"), ("intM", "intercept M"), ("r2M", "R² M"), ("nM", "n M"),
            ("rangeM", "fit range M"),
            ("delta", "Delta D"), ("absdelta", "|Delta D|"),
        ]
        for i, (key, label) in enumerate(keys):
            r = i // 6; c = (i % 6) * 2
            summary.addWidget(QLabel(label + ":"), r, c)
            val = QLabel("—")
            val.setStyleSheet("font-weight:600;")
            summary.addWidget(val, r, c + 1)
            self.result_labels[key] = val
        results_lay.addLayout(summary)
        self.tabs.addTab(results, "Results")

        # Number model parameters tab
        self.model_widget = ParticleModelWidget(self)
        self.tabs.addTab(self.model_widget, "Number model parameters")

        # Data tab with separate N and M tables
        data_tab = QWidget()
        data_lay = QVBoxLayout(data_tab)
        self.data_tabs = QTabWidget()
        data_lay.addWidget(self.data_tabs)
        self.tbl_number = QTableWidget()
        self.tbl_mass = QTableWidget()
        self.tbl_number.setAlternatingRowColors(True)
        self.tbl_mass.setAlternatingRowColors(True)
        self.tbl_number.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_mass.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_number.itemChanged.connect(lambda item: self._on_use_item_changed("N", item))
        self.tbl_mass.itemChanged.connect(lambda item: self._on_use_item_changed("M", item))
        self.data_tabs.addTab(self.tbl_number, "Number data")
        self.data_tabs.addTab(self.tbl_mass, "Mass data")
        self.tabs.addTab(data_tab, "Data")

        # Settings/export tab
        settings_tab = QWidget()
        settings_root = QVBoxLayout(settings_tab)
        settings_form = QFormLayout()
        self.cmb_mass_basis = QComboBox()
        self.cmb_mass_basis.addItems([MASS_BASIS_NORMALIZED, MASS_BASIS_ACTUAL])
        settings_form.addRow("Mass basis:", self.cmb_mass_basis)
        self.sp_actual_mass = QDoubleSpinBox()
        self.sp_actual_mass.setDecimals(6)
        self.sp_actual_mass.setRange(0.0, 1e12)
        self.sp_actual_mass.setSpecialValueText("Undefined")
        self.sp_actual_mass.setSuffix(" g")
        settings_form.addRow("Actual sample mass:", self.sp_actual_mass)
        self.cmb_rep_size = QComboBox()
        self.cmb_rep_size.addItems([REPRESENTATIVE_CLASS_MIDPOINT, REPRESENTATIVE_MANUAL])
        self.cmb_rep_size.setToolTip(
            "Class midpoint uses the stored Componentry bin phi as phi_mid, so B=2^(-phi_mid). "
            "Choose Manual when the stored phi should not be treated as the representative class midpoint."
        )
        settings_form.addRow("Representative size:", self.cmb_rep_size)
        self.chk_force_zero = QCheckBox("Force intercept = 0")
        self.chk_force_zero.setToolTip("Advanced option. Free intercept is the default.")
        settings_form.addRow("Regression:", self.chk_force_zero)
        settings_root.addLayout(settings_form)

        explanation = QLabel(
            "Number-based calculations require an explicit density for every positive-mass component included in the selected population. "
            "No scientific default density is assumed. Sphere is the default geometry (A/B=1, C/B=1)."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color:#4f4f4f;")
        settings_root.addWidget(explanation)

        export_box = QGroupBox("Export")
        export_lay = QHBoxLayout(export_box)
        self.btn_export_figure = QPushButton("Export current graph")
        self.btn_export_results = QPushButton("Export results/data")
        self.btn_self_tests = QPushButton("Run numeric self-tests")
        export_lay.addWidget(self.btn_export_figure)
        export_lay.addWidget(self.btn_export_results)
        export_lay.addWidget(self.btn_self_tests)
        export_lay.addStretch(1)
        settings_root.addWidget(export_box)
        settings_root.addStretch(1)
        self.tabs.addTab(settings_tab, "Settings")

        # Signals
        self.cmb_sample.currentIndexChanged.connect(self._on_sample_changed)
        self.cmb_population.currentIndexChanged.connect(self._on_population_changed)
        self.cmb_component.currentIndexChanged.connect(self._on_component_changed)
        self.cmb_method.currentIndexChanged.connect(self._on_method_changed)
        self.btn_refresh.clicked.connect(lambda: self.refresh_from_componentry(initial=False))
        self.btn_method_info.clicked.connect(self.show_method_information)
        self.btn_recalculate.clicked.connect(self.recalculate)
        self.btn_apply_range.clicked.connect(self.apply_fit_range)
        self.btn_select_all.clicked.connect(lambda: self.set_all_fit_points(True))
        self.btn_clear_all.clicked.connect(lambda: self.set_all_fit_points(False))
        self.btn_invert.clicked.connect(self.invert_fit_points)
        self.cmb_mass_basis.currentIndexChanged.connect(self._on_settings_controls_changed)
        self.sp_actual_mass.valueChanged.connect(self._on_settings_controls_changed)
        self.cmb_rep_size.currentIndexChanged.connect(self._on_settings_controls_changed)
        self.chk_force_zero.toggled.connect(self._on_settings_controls_changed)
        self.btn_export_figure.clicked.connect(self.export_figure)
        self.btn_export_results.clicked.connect(self.export_results)
        self.btn_self_tests.clicked.connect(self.show_self_tests)

    # -------------------------------------------------------------------------
    # Selection refresh
    # -------------------------------------------------------------------------
    def _watch_componentry(self) -> None:
        names = tuple(self.sample_dbs().keys())
        if names != self._last_sample_names:
            self.refresh_from_componentry(initial=False)

    def refresh_from_componentry(self, initial: bool = False) -> None:
        dbs = self.sample_dbs()
        current = self.current_sample_name()
        host_current = ""
        try:
            host_current = str(self.app.current_sample_name()).strip()
        except Exception:
            pass
        preferred = current or host_current
        if initial and host_current:
            preferred = host_current

        self._loading = True
        try:
            self.cmb_sample.clear()
            self.cmb_sample.addItems(list(dbs.keys()))
            if preferred:
                idx = self.cmb_sample.findText(preferred)
                if idx >= 0:
                    self.cmb_sample.setCurrentIndex(idx)
            if self.cmb_sample.count() and self.cmb_sample.currentIndex() < 0:
                self.cmb_sample.setCurrentIndex(0)
            self._last_sample_names = tuple(dbs.keys())
        finally:
            self._loading = False
        self._on_sample_changed()

    def _on_sample_changed(self, *args) -> None:
        if self._loading:
            return
        db = self.current_db()
        self._loading = True
        try:
            self.cmb_population.clear()
            if db is not None:
                self.cmb_population.addItem("Whole sample", ("whole", None))
                for t in observed_component_types(db):
                    self.cmb_population.addItem(population_label("type", t), ("type", t))
                self.cmb_population.addItem("Individual component", ("component", None))
            self._refresh_component_selector()
            self._load_sample_settings_controls()
            self._refresh_fit_phi_controls()
        finally:
            self._loading = False
        self.model_widget.refresh_components()
        self._update_method_ui()
        self.recalculate()

    def _population_spec(self) -> Tuple[str, Optional[str]]:
        data = self.cmb_population.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            kind, value = str(data[0]), data[1]
        else:
            kind, value = "whole", None
        if kind == "component":
            value = str(self.cmb_component.currentText()).strip() or None
        return kind, value

    def _refresh_component_selector(self) -> None:
        db = self.current_db()
        old = self.cmb_component.currentText().strip()
        self.cmb_component.blockSignals(True)
        self.cmb_component.clear()
        if db is not None:
            comps = list(getattr(db, "components", []) or [])
            if undefined_mass_exists(db):
                comps.append(UNDEFINED_COMPONENT)
            self.cmb_component.addItems(comps)
        if old:
            idx = self.cmb_component.findText(old)
            if idx >= 0:
                self.cmb_component.setCurrentIndex(idx)
        self.cmb_component.blockSignals(False)
        kind, _ = self._population_spec()
        visible = kind == "component"
        self.lbl_component.setVisible(visible)
        self.cmb_component.setVisible(visible)

    def _on_population_changed(self, *args) -> None:
        if self._loading:
            return
        self._refresh_component_selector()
        self._refresh_fit_phi_controls()
        self.recalculate()

    def _on_component_changed(self, *args) -> None:
        if self._loading:
            return
        self.model_widget.refresh_components(preferred=self.cmb_component.currentText())
        self.recalculate()

    def _on_method_changed(self, *args) -> None:
        if self._loading:
            return
        self._update_method_ui()
        self.recalculate()

    def _update_method_ui(self) -> None:
        method = self.cmb_method.currentText()
        number_active = method in (METHOD_NUMBER, METHOD_COMPARE)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.model_widget), number_active)
        self.data_tabs.setTabEnabled(0, number_active)
        self.data_tabs.setTabEnabled(1, method in (METHOD_MASS, METHOD_COMPARE))
        if method == METHOD_NUMBER:
            self.data_tabs.setCurrentIndex(0)
        elif method == METHOD_MASS:
            self.data_tabs.setCurrentIndex(1)

    def _load_sample_settings_controls(self) -> None:
        db = self.current_db()
        if db is None:
            return
        s = ensure_fractal_settings(db)
        self.cmb_mass_basis.setCurrentText(s.get("mass_basis", MASS_BASIS_NORMALIZED))
        actual = _json_float_or_none(s.get("actual_sample_mass_g"))
        self.sp_actual_mass.setValue(actual if actual is not None else 0.0)
        self.cmb_rep_size.setCurrentText(s.get("representative_size_mode", REPRESENTATIVE_CLASS_MIDPOINT))
        self.chk_force_zero.setChecked(bool(s.get("force_intercept_zero", False)))
        self.sp_actual_mass.setEnabled(self.cmb_mass_basis.currentText() == MASS_BASIS_ACTUAL)

    def _on_settings_controls_changed(self, *args) -> None:
        if self._loading:
            return
        db = self.current_db()
        if db is None:
            return
        s = ensure_fractal_settings(db)
        s["mass_basis"] = self.cmb_mass_basis.currentText()
        s["actual_sample_mass_g"] = None if self.sp_actual_mass.value() <= 0 else float(self.sp_actual_mass.value())
        s["representative_size_mode"] = self.cmb_rep_size.currentText()
        s["force_intercept_zero"] = bool(self.chk_force_zero.isChecked())
        self.sp_actual_mass.setEnabled(s["mass_basis"] == MASS_BASIS_ACTUAL)
        self.model_widget.refresh()
        self.settings_changed(recalculate=True)

    def _refresh_fit_phi_controls(self) -> None:
        db = self.current_db()
        phis = sample_phi_rows(db) if db is not None else []
        old_from = self.cmb_fit_from.currentText()
        old_to = self.cmb_fit_to.currentText()
        self.cmb_fit_from.clear(); self.cmb_fit_to.clear()
        for phi in phis:
            txt = f"{phi:g}"
            self.cmb_fit_from.addItem(txt, float(phi))
            self.cmb_fit_to.addItem(txt, float(phi))
        if phis:
            self.cmb_fit_from.setCurrentIndex(0)
            self.cmb_fit_to.setCurrentIndex(len(phis) - 1)
            for cmb, old in ((self.cmb_fit_from, old_from), (self.cmb_fit_to, old_to)):
                idx = cmb.findText(old)
                if idx >= 0:
                    cmb.setCurrentIndex(idx)

    # -------------------------------------------------------------------------
    # Fit selection controls
    # -------------------------------------------------------------------------
    def _active_method_codes(self) -> List[str]:
        m = self.cmb_method.currentText()
        if m == METHOD_NUMBER:
            return ["N"]
        if m == METHOD_MASS:
            return ["M"]
        return ["N", "M"]

    def apply_fit_range(self) -> None:
        db = self.current_db()
        if db is None or self.cmb_fit_from.count() == 0:
            return
        a = _safe_float(self.cmb_fit_from.currentData(), float("nan"))
        b = _safe_float(self.cmb_fit_to.currentData(), float("nan"))
        if not np.isfinite(a) or not np.isfinite(b):
            return
        lo, hi = min(a, b), max(a, b)
        kind, value = self._population_spec()
        pk = population_key(kind, value)
        for code in self._active_method_codes():
            for phi in sample_phi_rows(db):
                set_fit_use_for_phi(db, code, pk, phi, lo <= phi <= hi)
        self.settings_changed(recalculate=True)

    def set_all_fit_points(self, use: bool) -> None:
        db = self.current_db()
        if db is None:
            return
        kind, value = self._population_spec(); pk = population_key(kind, value)
        for code in self._active_method_codes():
            for phi in sample_phi_rows(db):
                set_fit_use_for_phi(db, code, pk, phi, bool(use))
        self.settings_changed(recalculate=True)

    def invert_fit_points(self) -> None:
        db = self.current_db()
        if db is None:
            return
        kind, value = self._population_spec(); pk = population_key(kind, value)
        for code in self._active_method_codes():
            for phi in sample_phi_rows(db):
                old = fit_use_for_phi(db, code, pk, phi)
                set_fit_use_for_phi(db, code, pk, phi, not old)
        self.settings_changed(recalculate=True)

    def _on_use_item_changed(self, code: str, item: QTableWidgetItem) -> None:
        if self._loading or self._recalculating or item.column() != 0:
            return
        phi = _safe_float(item.data(Qt.UserRole), float("nan"))
        if not np.isfinite(phi):
            return
        db = self.current_db()
        if db is None:
            return
        kind, value = self._population_spec(); pk = population_key(kind, value)
        set_fit_use_for_phi(db, code, pk, phi, item.checkState() == Qt.Checked)
        self.settings_changed(recalculate=True)

    # -------------------------------------------------------------------------
    # Recalculation and rendering
    # -------------------------------------------------------------------------
    def recalculate(self, *args) -> None:
        if self._loading or self._recalculating:
            return
        db = self.current_db()
        self._recalculating = True
        try:
            if db is None:
                self.number_result = None; self.mass_result = None
                self._draw_empty("No sample loaded")
                self._clear_tables_and_results()
                return

            kind, value = self._population_spec()
            method = self.cmb_method.currentText()
            force_zero = bool(ensure_fractal_settings(db).get("force_intercept_zero", False))
            self.number_result = calculate_number_method(db, kind, value, force_zero) if method in (METHOD_NUMBER, METHOD_COMPARE) else None
            self.mass_result = calculate_mass_method(db, kind, value, force_zero) if method in (METHOD_MASS, METHOD_COMPARE) else None
            self._fill_number_table(self.number_result)
            self._fill_mass_table(self.mass_result)
            self._update_status()
            self._update_result_labels()
            self._draw_results()
        except Exception:
            self.lbl_status.setText("Calculation error: " + traceback.format_exc().splitlines()[-1])
            self.lbl_status.setStyleSheet("color:#a00000;")
        finally:
            self._recalculating = False

    def _clear_tables_and_results(self) -> None:
        self.tbl_number.setRowCount(0); self.tbl_mass.setRowCount(0)
        for lbl in self.result_labels.values():
            lbl.setText("—")
        self.lbl_coverage.setText("Componentry coverage: —")
        self.lbl_status.setText("")

    def _draw_empty(self, text: str) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _update_status(self) -> None:
        db = self.current_db()
        cov = whole_componentry_coverage_percent(db) if db is not None else float("nan")
        if self.number_result is not None and np.isfinite(self.number_result.get("modeled_coverage", float("nan"))):
            self.lbl_coverage.setText(
                f"Componentry coverage: {cov:.1f}%   |   Number-model coverage: {self.number_result['modeled_coverage']:.1f}%"
            )
        elif np.isfinite(cov):
            self.lbl_coverage.setText(f"Componentry coverage: {cov:.1f}%")
        else:
            self.lbl_coverage.setText("Componentry coverage: —")

        messages: List[str] = []
        errors: List[str] = []
        for result in (self.number_result, self.mass_result):
            if not result:
                continue
            messages.extend([str(x) for x in result.get("warnings", []) if str(x)])
            errors.extend([str(x) for x in result.get("errors", []) if str(x)])
        if errors:
            self.lbl_status.setText(" | ".join(errors + messages))
            self.lbl_status.setStyleSheet("color:#a00000;")
        elif messages:
            self.lbl_status.setText(" | ".join(messages))
            self.lbl_status.setStyleSheet("color:#8a5a00;")
        else:
            self.lbl_status.setText("Calculation valid for the currently selected points.")
            self.lbl_status.setStyleSheet("color:#236523;")

    def _set_result(self, key: str, value: Any, digits: int = 3) -> None:
        lbl = self.result_labels[key]
        v = _safe_float(value, float("nan"))
        lbl.setText(f"{v:.{digits}f}" if np.isfinite(v) else "—")

    @staticmethod
    def _fit_range_text(result: Optional[Dict[str, Any]], size_key: str) -> str:
        if not result:
            return "—"
        used = [r for r in result.get("rows", []) if r.get("use") and r.get("valid")]
        if not used:
            return "—"
        phis = [float(r["phi"]) for r in used if np.isfinite(_safe_float(r.get("phi"), float("nan")))]
        sizes = [float(r[size_key]) for r in used if np.isfinite(_safe_float(r.get(size_key), float("nan")))]
        if not phis or not sizes:
            return "—"
        return f"φ {min(phis):g} to {max(phis):g}; {min(sizes):.6g}–{max(sizes):.6g} mm"

    def _update_result_labels(self) -> None:
        for lbl in self.result_labels.values():
            lbl.setText("—")
        nr = self.number_result; mr = self.mass_result
        if nr:
            self._set_result("DN", nr.get("D"))
            fit = nr.get("fit") or {}
            self._set_result("slopeN", fit.get("slope")); self._set_result("intN", fit.get("intercept"))
            self._set_result("r2N", fit.get("r2")); self._set_result("nN", fit.get("n"), 0)
            self.result_labels["rangeN"].setText(self._fit_range_text(nr, "threshold_r_mm"))
        if mr:
            self._set_result("DM", mr.get("D")); self._set_result("vM", mr.get("v"))
            fit = mr.get("fit") or {}
            self._set_result("intM", fit.get("intercept")); self._set_result("r2M", fit.get("r2")); self._set_result("nM", fit.get("n"), 0)
            self.result_labels["rangeM"].setText(self._fit_range_text(mr, "R_mm"))
        if nr and mr:
            dn = _safe_float(nr.get("D"), float("nan")); dm = _safe_float(mr.get("D"), float("nan"))
            if np.isfinite(dn) and np.isfinite(dm):
                self._set_result("delta", dn - dm); self._set_result("absdelta", abs(dn - dm))

    def _draw_results(self) -> None:
        self.figure.clear()
        method = self.cmb_method.currentText()
        if method == METHOD_COMPARE:
            ax_n = self.figure.add_subplot(121)
            ax_m = self.figure.add_subplot(122)
            self._draw_number(ax_n, self.number_result)
            self._draw_mass(ax_m, self.mass_result)
        elif method == METHOD_NUMBER:
            ax = self.figure.add_subplot(111)
            self._draw_number(ax, self.number_result)
        else:
            ax = self.figure.add_subplot(111)
            self._draw_mass(ax, self.mass_result)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_number(self, ax, result: Optional[Dict[str, Any]]) -> None:
        ax.set_facecolor("white")
        if not result or not result.get("rows"):
            ax.text(0.5, 0.5, "No number-based data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("log10(Sieve opening [mm])"); ax.set_ylabel("log10[N(>r)]")
            return
        rows = [r for r in result["rows"] if r.get("valid")]
        included = [r for r in rows if r.get("use")]
        excluded = [r for r in result["rows"] if (r.get("valid") and not r.get("use"))]
        if excluded:
            ax.scatter([r["log10_r"] for r in excluded], [r["log10_N_gt_r"] for r in excluded], s=34, facecolors="none", edgecolors="#8a8a8a", linewidths=1.0, label="Excluded", zorder=3)
        if included:
            ax.scatter([r["log10_r"] for r in included], [r["log10_N_gt_r"] for r in included], s=42, color="#2f5d8a", edgecolor="black", linewidth=0.45, label="Used", zorder=4)
        fit = result.get("fit")
        if fit and included:
            xs = np.array([r["log10_r"] for r in included], dtype=float)
            xx = np.linspace(float(np.min(xs)), float(np.max(xs)), 100)
            yy = fit["slope"] * xx + fit["intercept"]
            ax.plot(xx, yy, color="black", linewidth=1.4, label="Regression", zorder=2)
            txt = f"Number-based method\nD = {result['D']:.3f}\nslope = {fit['slope']:.3f}\nR² = {fit['r2']:.3f}\nn = {fit['n']}"
        else:
            txt = "Number-based method\nRegression unavailable"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#777777", alpha=0.92))
        ax.set_xlabel("log10(Sieve opening [mm])")
        ax.set_ylabel("log10[N(>r)]")
        ax.grid(True, alpha=0.18)
        if included or excluded:
            ax.legend(fontsize=8, frameon=True, loc="best")

    def _draw_mass(self, ax, result: Optional[Dict[str, Any]]) -> None:
        ax.set_facecolor("white")
        if not result or not result.get("rows"):
            ax.text(0.5, 0.5, "No mass-based data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel("log10(Particle diameter [mm])"); ax.set_ylabel("log10(Cumulative mass fraction)")
            return
        rows = [r for r in result["rows"] if r.get("valid")]
        included = [r for r in rows if r.get("use")]
        excluded = [r for r in result["rows"] if (r.get("valid") and not r.get("use"))]
        if excluded:
            ax.scatter([r["log10_R"] for r in excluded], [r["log10_mass_fraction"] for r in excluded], s=34, facecolors="none", edgecolors="#8a8a8a", linewidths=1.0, label="Excluded", zorder=3)
        if included:
            ax.scatter([r["log10_R"] for r in included], [r["log10_mass_fraction"] for r in included], s=42, color="#8a4f2f", edgecolor="black", linewidth=0.45, label="Used", zorder=4)
        fit = result.get("fit")
        if fit and included:
            xs = np.array([r["log10_R"] for r in included], dtype=float)
            xx = np.linspace(float(np.min(xs)), float(np.max(xs)), 100)
            yy = fit["slope"] * xx + fit["intercept"]
            ax.plot(xx, yy, color="black", linewidth=1.4, label="Regression", zorder=2)
            txt = f"Mass-based method\nv = {result['v']:.3f}\nD = {result['D']:.3f}\nR² = {fit['r2']:.3f}\nn = {fit['n']}"
        else:
            txt = "Mass-based method\nRegression unavailable"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#777777", alpha=0.92))
        ax.set_xlabel("log10(Particle diameter [mm])")
        ax.set_ylabel("log10(Cumulative mass fraction)")
        ax.grid(True, alpha=0.18)
        if included or excluded:
            ax.legend(fontsize=8, frameon=True, loc="best")

    # -------------------------------------------------------------------------
    # Data tables
    # -------------------------------------------------------------------------
    def _fill_number_table(self, result: Optional[Dict[str, Any]]) -> None:
        headers = [
            "Use", "phi", "threshold r mm", "component", "component mass wt%", "component mass g",
            "B mm", "density g/cm3", "shape", "A/B", "C/B", "A mm", "C mm",
            "particle volume mm3", "particle volume cm3", "particle mass g",
            "particle number in bin (component)", "total particle number in bin", "N(>r)",
            "log10(r)", "log10[N(>r)]", "shape volume factor", "N/N sphere", "model status",
        ]
        self.tbl_number.blockSignals(True)
        self.tbl_number.clear()
        self.tbl_number.setColumnCount(len(headers)); self.tbl_number.setHorizontalHeaderLabels(headers)
        rows = list((result or {}).get("detail_rows", []))
        # If there is no positive component mass yet, still expose aggregate phi points.
        if not rows and result:
            for a in result.get("rows", []):
                rows.append({**a, "component": "", "component_mass_wt": float("nan"), "component_mass_g": float("nan"), "B_mm": float("nan"), "density_g_cm3": float("nan"), "shape": "", "A_over_B": float("nan"), "C_over_B": float("nan"), "A_mm": float("nan"), "C_mm": float("nan"), "particle_volume_mm3": float("nan"), "particle_volume_cm3": float("nan"), "particle_mass_g": float("nan"), "particle_number_bin_component": float("nan"), "shape_volume_factor": float("nan"), "N_relative_to_sphere": float("nan"), "model_error": ""})
        self.tbl_number.setRowCount(len(rows))
        for i, r in enumerate(rows):
            use = QTableWidgetItem("")
            use.setFlags((use.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            use.setCheckState(Qt.Checked if r.get("use") else Qt.Unchecked)
            use.setData(Qt.UserRole, float(r.get("phi", float("nan"))))
            self.tbl_number.setItem(i, 0, use)
            vals = [
                f"{float(r.get('phi', float('nan'))):g}" if np.isfinite(_safe_float(r.get('phi'), float('nan'))) else "",
                _number_text(r.get("threshold_r_mm"), 8), str(r.get("component", "")),
                _number_text(r.get("component_mass_wt"), 8), _number_text(r.get("component_mass_g"), 8),
                _number_text(r.get("B_mm"), 8), _number_text(r.get("density_g_cm3"), 8), str(r.get("shape", "")),
                _number_text(r.get("A_over_B"), 8), _number_text(r.get("C_over_B"), 8),
                _number_text(r.get("A_mm"), 8), _number_text(r.get("C_mm"), 8),
                _number_text(r.get("particle_volume_mm3"), 8), _number_text(r.get("particle_volume_cm3"), 8),
                _number_text(r.get("particle_mass_g"), 8), _number_text(r.get("particle_number_bin_component"), 8),
                _number_text(r.get("particle_number_bin_total"), 8), _number_text(r.get("N_gt_r"), 8),
                _number_text(r.get("log10_r"), 8), _number_text(r.get("log10_N_gt_r"), 8),
                _number_text(r.get("shape_volume_factor"), 8), _number_text(r.get("N_relative_to_sphere"), 8),
                str(r.get("model_error", "")) or ("OK" if r.get("valid_model") else ""),
            ]
            for j, val in enumerate(vals, start=1):
                self.tbl_number.setItem(i, j, _read_only_item(val))
        self.tbl_number.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_number.horizontalHeader().setStretchLastSection(True)
        self.tbl_number.blockSignals(False)

    def _fill_mass_table(self, result: Optional[Dict[str, Any]]) -> None:
        headers = ["Use", "phi", "R mm", "bin mass", "cumulative mass finer", "M(r<R)/M_T", "log10(R)", "log10[M(r<R)/M_T]"]
        self.tbl_mass.blockSignals(True)
        self.tbl_mass.clear(); self.tbl_mass.setColumnCount(len(headers)); self.tbl_mass.setHorizontalHeaderLabels(headers)
        rows = list((result or {}).get("rows", []))
        self.tbl_mass.setRowCount(len(rows))
        for i, r in enumerate(rows):
            use = QTableWidgetItem("")
            use.setFlags((use.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
            use.setCheckState(Qt.Checked if r.get("use") else Qt.Unchecked)
            use.setData(Qt.UserRole, float(r["phi"]))
            self.tbl_mass.setItem(i, 0, use)
            vals = [f"{r['phi']:g}", _number_text(r["R_mm"], 8), _number_text(r["bin_mass"], 8), _number_text(r["cumulative_mass_finer"], 8), _number_text(r["mass_fraction"], 8), _number_text(r["log10_R"], 8), _number_text(r["log10_mass_fraction"], 8)]
            for j, val in enumerate(vals, start=1):
                self.tbl_mass.setItem(i, j, _read_only_item(val))
        self.tbl_mass.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_mass.horizontalHeader().setStretchLastSection(True)
        self.tbl_mass.blockSignals(False)

    # -------------------------------------------------------------------------
    # Method information and tests
    # -------------------------------------------------------------------------
    def show_method_information(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Fractal analysis — method information")
        dlg.resize(760, 650)
        lay = QVBoxLayout(dlg)
        txt = QTextEdit(); txt.setReadOnly(True)
        txt.setPlainText(
            "NUMBER-BASED METHOD\n\n"
            "N(>r) ∝ r^(-D)\n"
            "log10[N(>r)] = m log10(r) + b\n"
            "D_N = -m\n\n"
            "Mass in each component/fraction is converted to estimated particle number using:\n"
            "n = M / (rho V)\n\n"
            "TRIAXIAL ELLIPSOID PARTICLE MODEL\n\n"
            "A = (A/B) B\n"
            "C = (C/B) B\n"
            "V = pi/6 ABC = pi/6 (A/B)(C/B)B^3\n\n"
            "A, B and C are in mm. V is first calculated in mm3 and then converted to cm3 by V_cm3 = V_mm3/1000. "
            "Density rho is in g/cm3.  Geometry and density therefore affect only the number-based mass-to-number conversion.\n\n"
            "The fragmentation dimension D is a statistical property of the fragment population. A/B and C/B are geometric model parameters and are not fractal dimensions.\n\n"
            "MASS-BASED METHOD\n\n"
            "M(r<R)/M_T ∝ R^v\n"
            "log10[M(r<R)/M_T] = v log10(R) + b\n"
            "D_M = 3 - v\n\n"
            "Mass-based calculations use the selected population's mass distribution directly and do not require density, particle shape, volume or particle number.\n\n"
            "REGRESSION\n\n"
            "The default is a free intercept. The advanced 'Force intercept = 0' option is available for explicit use. "
            "Points can be included/excluded individually or by phi range; excluded points remain visible in the graph."
        )
        lay.addWidget(txt)
        bb = QDialogButtonBox(QDialogButtonBox.Close); bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec_()

    def show_self_tests(self) -> None:
        results = run_numeric_self_tests()
        body = "\n".join([f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in results.items()])
        if all(results.values()):
            QMessageBox.information(self, "Fractal numeric self-tests", body)
        else:
            QMessageBox.warning(self, "Fractal numeric self-tests", body)

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def export_figure(self) -> None:
        sample = _sanitize_filename(self.current_sample_name())
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Export fractal graph",
            f"{sample}_fractal.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg)",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            ext = ".png"; path += ext
        fmt_map = {".png": "png", ".svg": "svg", ".pdf": "pdf", ".tif": "tiff", ".tiff": "tiff", ".jpg": "jpeg", ".jpeg": "jpeg"}
        fmt = fmt_map.get(ext)
        if fmt is None:
            QMessageBox.warning(self, "Export graph", "Unsupported image format.")
            return
        dpi = 300 if fmt in ("png", "tiff", "jpeg") else 150
        try:
            self.figure.savefig(path, format=fmt, dpi=dpi, facecolor="white", bbox_inches="tight")
            QMessageBox.information(self, "Export graph", f"Graph exported:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export graph", str(exc))

    def _summary_export_rows(self) -> List[Dict[str, Any]]:
        kind, value = self._population_spec()
        sample = self.current_sample_name()
        rows: List[Dict[str, Any]] = []
        base = {
            "sample": sample,
            "population": population_label(kind, value),
            "population_key": population_key(kind, value),
            "force_intercept_zero": bool(self.chk_force_zero.isChecked()),
        }
        if self.number_result:
            fit = self.number_result.get("fit") or {}
            rows.append({**base, "method": "Number-based", "D": self.number_result.get("D"), "slope_or_v": fit.get("slope"), "intercept": fit.get("intercept"), "R2": fit.get("r2"), "n": fit.get("n"), "componentry_coverage_percent": self.number_result.get("componentry_coverage"), "modeled_coverage_percent": self.number_result.get("modeled_coverage")})
        if self.mass_result:
            fit = self.mass_result.get("fit") or {}
            rows.append({**base, "method": "Mass-based", "D": self.mass_result.get("D"), "slope_or_v": self.mass_result.get("v"), "intercept": fit.get("intercept"), "R2": fit.get("r2"), "n": fit.get("n")})
        if self.number_result and self.mass_result:
            dn = _safe_float(self.number_result.get("D"), float("nan")); dm = _safe_float(self.mass_result.get("D"), float("nan"))
            if np.isfinite(dn) and np.isfinite(dm):
                rows.append({**base, "method": "Compare", "D_N": dn, "D_M": dm, "Delta_D": dn - dm, "absolute_Delta_D": abs(dn - dm)})
        return rows

    def _number_export_df(self) -> pd.DataFrame:
        rows = list((self.number_result or {}).get("detail_rows", []))
        return pd.DataFrame(rows)

    def _mass_export_df(self) -> pd.DataFrame:
        return pd.DataFrame(list((self.mass_result or {}).get("rows", [])))

    def _model_export_df(self) -> pd.DataFrame:
        db = self.current_db()
        if db is None:
            return pd.DataFrame()
        phis = sample_phi_rows(db)
        comps = list(getattr(db, "components", []) or [])
        if undefined_mass_exists(db):
            comps.append(UNDEFINED_COMPONENT)
        out: List[Dict[str, Any]] = []
        for comp in comps:
            for phi in phis:
                p = effective_component_params(db, comp, phi)
                out.append({
                    "sample": self.current_sample_name(), "component": comp, "phi": phi,
                    "component_mass_wt": component_mass_wt(db, comp, phi),
                    "B_mm": p.get("b_mm"), "density_g_cm3": p.get("density"), "shape": p.get("shape"),
                    "A_over_B": p.get("alpha"), "C_over_B": p.get("gamma"),
                    "representative_size_mode": p.get("representative_size_mode"),
                })
        return pd.DataFrame(out)

    def export_results(self) -> None:
        sample = _sanitize_filename(self.current_sample_name())
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export fractal results",
            f"{sample}_fractal_results.xlsx",
            "Excel workbook (*.xlsx);;CSV (*.csv)",
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".xlsx", ".csv"):
            path += ".xlsx"; ext = ".xlsx"
        summary = pd.DataFrame(self._summary_export_rows())
        ndf = self._number_export_df()
        mdf = self._mass_export_df()
        model = self._model_export_df()
        try:
            if ext == ".xlsx":
                with pd.ExcelWriter(path, engine="openpyxl") as writer:
                    summary.to_excel(writer, sheet_name="Summary", index=False)
                    ndf.to_excel(writer, sheet_name="Number", index=False)
                    mdf.to_excel(writer, sheet_name="Mass", index=False)
                    model.to_excel(writer, sheet_name="Particle model", index=False)
            else:
                sections: List[pd.DataFrame] = []
                for name, df in (("summary", summary), ("number", ndf), ("mass", mdf), ("particle_model", model)):
                    if df is None or df.empty:
                        continue
                    d = df.copy(); d.insert(0, "section", name); sections.append(d)
                out = pd.concat(sections, axis=0, ignore_index=True, sort=False) if sections else pd.DataFrame()
                out.to_csv(path, index=False)
            QMessageBox.information(self, "Export results", f"Fractal results exported:\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export results", str(exc))

    def closeEvent(self, event) -> None:
        try:
            self._watch_timer.stop()
        except Exception:
            pass
        self._notify_host_changed()
        super().closeEvent(event)


__all__ = [
    "FractalAnalysisWindow",
    "phi_to_mm",
    "mm_to_phi",
    "ellipsoid_geometry",
    "estimate_particle_number",
    "linear_regression",
    "calculate_number_method",
    "calculate_mass_method",
    "run_numeric_self_tests",
    "ensure_fractal_settings",
]
