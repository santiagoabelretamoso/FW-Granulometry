# -*- coding: utf-8 -*-
"""
FractalAnalysis.py

Fractal fragmentation analysis for Componentry-FW (FW plugin).

Main reference
    Perugini D., Speziali A., Caricchi L., Kueppers U. (2011). Application of
    fractal fragmentation theory to natural pyroclastic deposits: insights into
    volcanic explosivity of the Valentano scoria cone (Italy).
    J. Volcanol. Geotherm. Res. 202, 200-210.

Supporting references
    Kueppers U., Perugini D., Dingwell D.B. (2006). "Explosive energy" during
        volcanic eruptions from fractal analysis of pyroclasts. EPSL 248, 800-807.
    Kueppers U., Scheu B., Spieler O., Dingwell D.B. (2006). Fragmentation
        efficiency of explosive volcanic eruptions. JVGR 153, 125-135.
    Perugini D., Kueppers U. (2012). Fractal analysis of experimentally generated
        pyroclasts: a tool for volcanic hazard assessment. Acta Geophys. 60, 682-698.
    Paredes-Marino J. et al. (2017). Syneruptive sequential fragmentation of
        pyroclasts from fractal modeling of grain size distributions of fall
        deposits: the Cretaio Tephra eruption (Ischia Island, Italy). JVGR 345, 161-171.
    Wiejaczka J., Giachetti T. (2022). Using eruption source parameters and
        high-resolution grain-size distributions of the 7.7 ka Cleetwood eruption
        of Mount Mazama. Front. Earth Sci. 10, 853021.

Size-class convention (FW / Componentry)
    Each stored phi is the sieve that retains the class.  The class holds the
    particles between the aperture of that sieve, 2^-phi mm (lower size limit),
    and the aperture of the next coarser sieve (upper size limit).  FW bins laser
    data the same way: phi in (X - step, X] is stored as X.

NUMBER-BASED METHOD (Perugini et al. 2011, Eqs. 2-3)
    N(>=r) ~ r^-D,   log10 N(>=r) = -D log10 r + b,   D = -slope
    r is the sieve aperture.  Mass is converted to particle number with
    n = M / (rho V), V = (pi/6) A B C (triaxial ellipsoid, sphere by default).
    B is the sieve aperture (Perugini et al. 2011), the geometric class midpoint
    (Wiejaczka & Giachetti 2022) or a manual value.  With uniform density and
    shape their absolute values do not change D.

MASS-BASED METHOD (Perugini & Kueppers 2012, Eqs. 4-9; Paredes-Marino et al. 2017)
    M(r<R)/M_T ~ R^v,   D = 3 - v
    R is a sieve aperture and M(r<R) the mass that passes it.  For each class:
    R = upper size limit of the class, M(r<R) = class mass + all finer classes.

FRAGMENTATION PROBABILITY (Perugini et al. 2011, Eq. 19)
    D = 3 ln(8p) / ln 8   ->   p = 2^D / 8, defined for 0 <= D <= 3.

FIT CHECK (Paredes-Marino et al. 2017)
    A straight line is accepted when R^2 > 0.9 and SE(D) <= 0.1.  A kink (two
    scaling regimes) can be fitted with two segments sharing the break point.

All persistent settings live in db.fractal_settings (JSON-serializable).  The
ComponentryDB serializer of Componentry-FW.py stores that dictionary in
standalone Componentry projects and in FW full projects.
"""

from __future__ import annotations

import copy
import math
import os
import re
import traceback
from html import escape as _esc
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterSciNotation, LogLocator, NullLocator

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


FRACTAL_SETTINGS_SCHEMA_VERSION = 2

METHOD_NUMBER = "By particle count"
METHOD_MASS = "By mass"
METHOD_COMPARE = "Both methods"

MASS_BASIS_NORMALIZED = "Normalized 100 g basis"
MASS_BASIS_ACTUAL = "Actual sample mass"

DENSITY_UNIFORM = "Uniform across fractions"
DENSITY_BY_FRACTION = "By fraction"
SHAPE_UNIFORM = "Uniform across fractions"
SHAPE_BY_FRACTION = "By fraction"

REPRESENTATIVE_SIEVE = "Sieve aperture (Perugini et al. 2011)"
REPRESENTATIVE_GEOMETRIC_MID = "Geometric class midpoint"
REPRESENTATIVE_MANUAL = "Manual"
REPRESENTATIVE_MODES = (REPRESENTATIVE_SIEVE, REPRESENTATIVE_GEOMETRIC_MID, REPRESENTATIVE_MANUAL)
# Version 1 called the stored phi a "class midpoint" but used B = 2^-phi, which is
# the sieve aperture.  Old projects therefore keep exactly the same numbers.
_LEGACY_REPRESENTATIVE_MODES = {"Class midpoint": REPRESENTATIVE_SIEVE}

UNDEFINED_COMPONENT = "Undefined"      # internal key, kept for saved projects
UNCLASSIFIED_LABEL = "Unclassified"
BULK_LABEL = "Whole sample (bulk)"

DEFAULT_BULK_DENSITY = 2.5             # g/cm3, sieving-only samples; it does not change D

PERUGINI_2011_PHI_RANGE = (-2.5, 2.0)  # sieves 5.66 to 0.25 mm (Perugini et al. 2011)

FIT_R2_MIN = 0.9                       # Paredes-Marino et al. (2017)
FIT_SE_MAX = 0.1                       # Paredes-Marino et al. (2017)
MIN_SEGMENT_POINTS = 3

VIEW_LOGLOG = "Fit (log scale)"
VIEW_LINEAR = "Linear scale"
VIEW_RESIDUALS = "Residuals"

# Visual style: one accent colour per method, large markers, light grid.
UI_ACCENT = {"N": "#2563EB", "M": "#EA580C"}
UI_METHOD_NAME = {"N": "Particle count method", "M": "Mass method"}
UI_METHOD_REF = {"N": "Perugini et al. (2011)", "M": "Paredes-Mariño et al. (2017)"}
UI_SEGMENT_COLORS = {"coarse": "#7C3AED", "fine": "#059669"}
UI_FIT_LINE = "#1F2937"
UI_MUTED = "#9CA3AF"
UI_TEXT = "#374151"
UI_GRID = "#EDEFF2"
UI_SPINE = "#D1D5DB"
FA_STYLESHEET = """
QWidget#faCentral { background: #F5F6F8; }
QWidget#faLeftBody { background: transparent; }
QScrollArea#faLeft { background: transparent; border: none; }
QFrame#faCard, QFrame#faChartCard { background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px; }
QLabel#faCaption { color: #6B7280; font-size: 12px; font-weight: 600; }
QLabel#faMethod { color: #4B5563; font-size: 12px; }
QLabel#faHint { color: #6B7280; font-size: 11px; }
QLabel#faSubtitle { color: #4B5563; font-size: 12px; }
QLabel#faNotice { background: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; color: #92400E; padding: 8px 10px; }
QLabel#faDetails { color: #374151; }
QToolButton#faSection { border: none; font-weight: 600; color: #374151; padding: 4px 0px; }
QPushButton#faPrimary { background: #2563EB; color: white; border: none; border-radius: 6px; padding: 7px 12px; font-weight: 600; }
QPushButton#faPrimary:hover { background: #1D4ED8; }
"""

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

NAN = float("nan")


# =============================================================================
# Mouse wheel: never changes control values (same rule as the rest of FW)
# =============================================================================

class _NoWheelValueFilter(QObject):
    """Spin boxes, combo boxes, sliders and tab bars ignore the wheel.

    The wheel scrolls the nearest scrollable container instead.  FW installs an
    identical filter; this copy covers Componentry when it runs on its own.
    """

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        try:
            if event.type() != QEvent.Wheel:
                return False
            if not isinstance(obj, (QAbstractSpinBox, QComboBox, QAbstractSlider, QTabBar)) or isinstance(obj, QScrollBar):
                return False
            parent = obj.parentWidget()
            while parent is not None and not isinstance(parent, QAbstractScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                delta = event.angleDelta()
                horizontal = abs(delta.x()) > abs(delta.y())
                bar = parent.horizontalScrollBar() if horizontal else parent.verticalScrollBar()
                if bar is not None and bar.isVisible():
                    QApplication.sendEvent(bar, event)
            return True
        except Exception:
            return False


def install_no_wheel_filter() -> None:
    app = QApplication.instance()
    if app is None or getattr(app, "_fw_no_wheel_values", None) is not None:
        return
    flt = _NoWheelValueFilter(app)
    app.installEventFilter(flt)
    app._fw_no_wheel_values = flt


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
    """Convert component/class mass to estimated particle number, n = M / (rho V)."""
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


def fragmentation_probability(D: Any) -> float:
    """Perugini et al. (2011), Eq. 19: D = 3 ln(8p)/ln 8, so p = 2^D / 8 (0 <= D <= 3)."""
    try:
        D = float(D)
    except Exception:
        return NAN
    if not np.isfinite(D) or D < 0.0 or D > 3.0:
        return NAN
    return float(2.0 ** D / 8.0)


def fragmentation_probability_se(D: Any, se_D: Any) -> float:
    """Propagated standard error of p: dp/dD = p ln 2."""
    p = fragmentation_probability(D)
    try:
        se = float(se_D)
    except Exception:
        return NAN
    if not np.isfinite(p) or not np.isfinite(se):
        return NAN
    return float(p * math.log(2.0) * se)


_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(dof: int) -> float:
    """Two-sided 95% Student t value (table up to 30 degrees of freedom, expansion beyond)."""
    dof = int(dof)
    if dof < 1:
        return NAN
    if dof in _T95:
        return float(_T95[dof])
    z = 1.959963985
    return float(z + (z ** 3 + z) / (4.0 * dof) + (5.0 * z ** 5 + 16.0 * z ** 3 + 3.0 * z) / (96.0 * dof ** 2))


def linear_regression(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Ordinary least squares with standard errors and 95% interval of the slope."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        raise ValueError("At least 2 valid points are required for regression.")
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    dx = x - xm
    sxx = float(np.dot(dx, dx))
    if not np.isfinite(sxx) or sxx <= np.finfo(np.float64).eps:
        raise ValueError("The selected points have the same size; the slope is undefined.")
    slope = float(np.dot(dx, y - ym) / sxx)
    intercept = float(ym - slope * xm)
    residual = y - (slope * x + intercept)
    sse = float(np.dot(residual, residual))
    centered = y - ym
    sst = float(np.dot(centered, centered))
    r2 = float(1.0 - sse / sst) if sst > np.finfo(np.float64).eps else NAN
    n = int(x.size)
    dof = n - 2
    if dof > 0:
        s2 = sse / dof
        se_slope = float(math.sqrt(max(0.0, s2 / sxx)))
        se_intercept = float(math.sqrt(max(0.0, s2 * (1.0 / n + xm * xm / sxx))))
        ci95 = float(t_critical_95(dof) * se_slope)
    else:
        se_slope = NAN
        se_intercept = NAN
        ci95 = NAN
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n": n,
        "se_slope": se_slope,
        "se_intercept": se_intercept,
        "ci95_slope": ci95,
        "sse": sse,
    }


def dimension_from_slope(code: str, slope: float) -> float:
    """Number-based: D = -slope.  Mass-based: D = 3 - v."""
    return float(-slope) if code == "N" else float(3.0 - slope)


def fit_check(fit: Optional[Dict[str, Any]]) -> str:
    """Acceptance criteria of Paredes-Marino et al. (2017): R^2 > 0.9 and SE <= 0.1."""
    if not fit:
        return "No fit"
    issues: List[str] = []
    n = int(fit.get("n", 0) or 0)
    r2 = _safe_float(fit.get("r2"))
    se = _safe_float(fit.get("se_slope"))
    if n < 3:
        issues.append("n < 3, no SE")
    if not np.isfinite(r2) or r2 <= FIT_R2_MIN:
        issues.append(f"R² ≤ {FIT_R2_MIN:g}")
    if np.isfinite(se) and se > FIT_SE_MAX:
        issues.append(f"SE > {FIT_SE_MAX:g}")
    return "OK" if not issues else "Check: " + ", ".join(issues)


def _segment_fit(code: str, pts: Sequence[Tuple[float, float, float]]) -> Dict[str, Any]:
    fit = linear_regression([p[1] for p in pts], [p[2] for p in pts])
    D = dimension_from_slope(code, fit["slope"])
    return {
        "fit": fit,
        "D": D,
        "D_se": fit["se_slope"],
        "D_ci95": fit["ci95_slope"],
        "p": fragmentation_probability(D),
        "check": fit_check(fit),
        "phi_min": float(min(p[0] for p in pts)),
        "phi_max": float(max(p[0] for p in pts)),
        "n": int(fit["n"]),
    }


def two_segment_fit(code: str, points: Sequence[Tuple[float, float, float]], break_phi: float) -> Dict[str, Any]:
    """
    Two power laws on each side of a kink (Paredes-Marino et al. 2017).

    points are (phi, log10 size, log10 cumulative).  The coarse segment holds the
    points with phi <= break, the fine segment those with phi >= break; the break
    point belongs to both.
    """
    pts = sorted(points, key=lambda p: p[0])
    bp = float(break_phi)
    coarse = [p for p in pts if p[0] <= bp + 1e-9]
    fine = [p for p in pts if p[0] >= bp - 1e-9]
    if len(coarse) < 2 or len(fine) < 2:
        raise ValueError("each segment needs at least 2 selected points")
    c = _segment_fit(code, coarse)
    f = _segment_fit(code, fine)
    return {"break_phi": bp, "coarse": c, "fine": f, "sse_total": float(c["fit"]["sse"] + f["fit"]["sse"])}


def suggest_break_phi(points: Sequence[Tuple[float, float, float]]) -> Optional[float]:
    """Break phi that minimises the total squared error of the two segments."""
    pts = sorted(points, key=lambda p: p[0])
    best: Optional[Tuple[float, float]] = None
    for k in range(len(pts)):
        coarse = pts[:k + 1]
        fine = pts[k:]
        if len(coarse) < MIN_SEGMENT_POINTS or len(fine) < MIN_SEGMENT_POINTS:
            continue
        try:
            sse = (
                linear_regression([p[1] for p in coarse], [p[2] for p in coarse])["sse"]
                + linear_regression([p[1] for p in fine], [p[2] for p in fine])["sse"]
            )
        except Exception:
            continue
        if best is None or sse < best[0] - 1e-15:
            best = (float(sse), float(pts[k][0]))
    return None if best is None else float(best[1])


# =============================================================================
# Persistent settings helpers
# =============================================================================

def _phi_key(phi: float) -> str:
    return format(float(phi), ".12g")


def _phi_round(phi: float) -> float:
    return round(float(phi), 6)


def _safe_float(value: Any, default: float = NAN) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    return float(v) if np.isfinite(v) else float(default)


def _json_float_or_none(value: Any) -> Optional[float]:
    v = _safe_float(value, NAN)
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
        "representative_size_mode": REPRESENTATIVE_SIEVE,
        "components": {},
        "fit_use": {"N": {}, "M": {}},
        "segments": {"N": {}, "M": {}},
    }


def ensure_fractal_settings(db: Any) -> Dict[str, Any]:
    raw = getattr(db, "fractal_settings", None)
    settings = raw if isinstance(raw, dict) else {}
    for key, value in default_fractal_settings().items():
        if key not in settings:
            settings[key] = copy.deepcopy(value)
    # Removed in version 2: a forced zero intercept makes D depend on density,
    # shape and sample mass (number-based) and on the size unit (mass-based).
    settings.pop("force_intercept_zero", None)
    if settings.get("mass_basis") not in (MASS_BASIS_NORMALIZED, MASS_BASIS_ACTUAL):
        settings["mass_basis"] = MASS_BASIS_NORMALIZED
    rep = settings.get("representative_size_mode")
    rep = _LEGACY_REPRESENTATIVE_MODES.get(rep, rep)
    if rep not in REPRESENTATIVE_MODES:
        rep = REPRESENTATIVE_SIEVE
    settings["representative_size_mode"] = rep
    if not isinstance(settings.get("components"), dict):
        settings["components"] = {}
    for name in ("fit_use", "segments"):
        if not isinstance(settings.get(name), dict):
            settings[name] = {"N": {}, "M": {}}
        for code in ("N", "M"):
            if not isinstance(settings[name].get(code), dict):
                settings[name][code] = {}
    settings.pop("position", None)
    settings.pop("porosity_percent", None)
    settings["schema_version"] = FRACTAL_SETTINGS_SCHEMA_VERSION
    try:
        db.fractal_settings = settings
    except Exception:
        pass
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
    and last non-zero whole-sample GSD classes, matching Componentry's histogram.
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


def gsd_class_phis(db: Any) -> List[float]:
    """All classes stored in the GSD (zero-mass ones included), coarse to fine."""
    if db is None or not isinstance(getattr(db, "gsd", None), pd.DataFrame) or "phi" not in db.gsd.columns:
        return []
    vals = pd.to_numeric(db.gsd["phi"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return sorted(set(float(v) for v in vals))


def class_bounds(db: Any, phis: Optional[Sequence[float]] = None) -> Dict[float, Dict[str, float]]:
    """
    Size limits of each class (mm).

    lower = 2^-phi, the aperture of the sieve that retains the class.
    upper = aperture of the next coarser class stored in the GSD; for the
            coarsest class the phi step of its finer neighbour is assumed.
    """
    if phis is None:
        phis = sample_phi_rows(db)
    all_phis = gsd_class_phis(db)
    out: Dict[float, Dict[str, float]] = {}
    for phi in phis:
        p = float(phi)
        coarser = [q for q in all_phis if q < p - 1e-9]
        if coarser:
            step = p - max(coarser)
        else:
            finer = [q for q in all_phis if q > p + 1e-9]
            step = (min(finer) - p) if finer else 1.0
        if not np.isfinite(step) or step <= 0:
            step = 1.0
        lower = phi_to_mm(p)
        upper = phi_to_mm(p - step)
        out[p] = {
            "lower_mm": float(lower),
            "upper_mm": float(upper),
            "mid_mm": float(math.sqrt(lower * upper)),
            "step_phi": float(step),
        }
    return out


def _gsd_wt_map(db: Any) -> Dict[float, float]:
    out: Dict[float, float] = {}
    if db is None or not isinstance(getattr(db, "gsd", None), pd.DataFrame):
        return out
    gsd = db.gsd
    if "phi" not in gsd.columns or "wt" not in gsd.columns:
        return out
    phis = pd.to_numeric(gsd["phi"], errors="coerce").to_numpy(dtype=float)
    wts = pd.to_numeric(gsd["wt"], errors="coerce").to_numpy(dtype=float)
    for phi, wt in zip(phis, wts):
        if np.isfinite(phi):
            w = float(wt) if np.isfinite(wt) else 0.0
            out[float(phi)] = out.get(float(phi), 0.0) + max(0.0, w)
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


def has_componentry(db: Any) -> bool:
    """True when the sample has components with positive mass (not sieving only)."""
    if db is None:
        return False
    comps = list(getattr(db, "components", []) or [])
    total = getattr(db, "total", None)
    if not comps or not isinstance(total, pd.DataFrame):
        return False
    cols = [c for c in comps if c in total.columns]
    if not cols:
        return False
    vals = total[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.nansum(np.clip(vals, 0.0, None)) > 0.0)


def classified_phi_set(db: Any) -> set:
    """Rounded phi values of the fractions with loaded componentry."""
    out = set()
    filled = getattr(db, "filled", None)
    if isinstance(filled, (set, frozenset, list, tuple)):
        for v in filled:
            f = _safe_float(v)
            if np.isfinite(f):
                out.add(_phi_round(f))
    if out:
        return out
    total = getattr(db, "total", None)
    comps = list(getattr(db, "components", []) or [])
    if isinstance(total, pd.DataFrame) and comps:
        cols = [c for c in comps if c in total.columns]
        if cols:
            sums = total[cols].apply(pd.to_numeric, errors="coerce").clip(lower=0.0).sum(axis=1, skipna=True)
            for phi, s in sums.items():
                f = _safe_float(phi)
                if np.isfinite(f) and float(s) > 0.0:
                    out.add(_phi_round(f))
    return out


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


def component_display_name(db: Any, component: str, componentry: Optional[bool] = None) -> str:
    if str(component) == UNDEFINED_COMPONENT:
        if componentry is None:
            componentry = has_componentry(db)
        return UNCLASSIFIED_LABEL if componentry else BULK_LABEL
    return str(component)


def model_component_keys(db: Any) -> List[str]:
    """Entries of the particle-model editor (internal keys)."""
    if db is None:
        return []
    if not has_componentry(db):
        return [UNDEFINED_COMPONENT]
    comps = list(getattr(db, "components", []) or [])
    if undefined_mass_exists(db):
        comps.append(UNDEFINED_COMPONENT)
    return comps


def population_components(
    db: Any,
    kind: str,
    value: Optional[str] = None,
    include_undefined_for_whole: bool = True,
    componentry: Optional[bool] = None,
) -> List[str]:
    if componentry is None:
        componentry = has_componentry(db)
    if kind == "whole":
        if not componentry:
            return [UNDEFINED_COMPONENT]
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
        return NAN
    assigned = 0.0
    for phi in phis:
        known = 0.0
        for c in list(getattr(db, "components", []) or []):
            known += component_mass_wt(db, c, phi)
        assigned += min(max(0.0, wt_map.get(phi, 0.0)), known)
    return float(100.0 * assigned / total_mass)


def representative_size_mm(rep_mode: str, bounds_row: Dict[str, float], manual_b: Optional[float]) -> Optional[float]:
    if rep_mode == REPRESENTATIVE_MANUAL:
        return manual_b
    if rep_mode == REPRESENTATIVE_GEOMETRIC_MID:
        return float(bounds_row["mid_mm"])
    return float(bounds_row["lower_mm"])


def effective_component_params(
    db: Any,
    component: str,
    phi: float,
    phis: Optional[Sequence[float]] = None,
    bounds: Optional[Dict[float, Dict[str, float]]] = None,
    componentry: Optional[bool] = None,
) -> Dict[str, Any]:
    if phis is None:
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

    if componentry is None:
        componentry = has_componentry(db)
    default_density = False
    if density is None and str(component) == UNDEFINED_COMPONENT and not componentry:
        density = DEFAULT_BULK_DENSITY
        default_density = True

    settings = ensure_fractal_settings(db)
    rep_mode = settings.get("representative_size_mode", REPRESENTATIVE_SIEVE)
    manual_b = _json_float_or_none(row.get("manual_b_mm"))
    if bounds is None or float(phi) not in bounds:
        bounds = class_bounds(db, [float(phi)])
    b_mm = representative_size_mm(rep_mode, bounds[float(phi)], manual_b)

    return {
        "density": density,
        "default_density": default_density,
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


def population_profile(
    db: Any,
    kind: str,
    value: Optional[str],
    phis: Sequence[float],
    componentry: Optional[bool] = None,
) -> Tuple[Dict[float, float], Dict[float, str], List[str]]:
    """
    Mass of the population by class and the analysed domain.

    status[phi] is "" for classes inside the domain.  Component and type
    populations use only classified fractions ("Not classified" otherwise).
    Classes outside the first..last class with positive population mass are
    "Outside population range", so empty tails do not flatten the fit.
    """
    masses = _population_mass_wt_series(db, kind, value, phis)
    status: Dict[float, str] = {float(phi): "" for phi in phis}
    notes: List[str] = []
    if kind != "whole":
        classified = classified_phi_set(db)
        for phi in phis:
            if _phi_round(phi) not in classified:
                status[float(phi)] = "Not classified"
    inside = [float(phi) for phi in phis if not status[float(phi)] and masses.get(float(phi), 0.0) > 0.0]
    if inside:
        lo, hi = min(inside), max(inside)
        for phi in phis:
            p = float(phi)
            if not status[p] and (p < lo - 1e-12 or p > hi + 1e-12):
                status[p] = "Outside population range"
        if kind != "whole":
            gaps = [float(phi) for phi in phis if lo < float(phi) < hi and status[float(phi)] == "Not classified"]
            if gaps:
                notes.append(
                    "Componentry is missing inside the analysed range (φ "
                    + ", ".join(f"{g:g}" for g in gaps)
                    + "); those fractions count as zero for this population."
                )
    else:
        for phi in phis:
            if not status[float(phi)]:
                status[float(phi)] = "No mass"
    return masses, status, notes


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


def segment_settings(db: Any, method_code: str, pop_key: str) -> Dict[str, Any]:
    settings = ensure_fractal_settings(db)
    seg = settings.setdefault("segments", {}).setdefault(method_code, {})
    cfg = seg.get(pop_key)
    if not isinstance(cfg, dict):
        cfg = {"enabled": False, "break_phi": None}
        seg[pop_key] = cfg
    cfg.setdefault("enabled", False)
    cfg.setdefault("break_phi", None)
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["break_phi"] = _json_float_or_none(cfg.get("break_phi"))
    return cfg


# =============================================================================
# Scientific calculations
# =============================================================================

def _empty_result(code: str, pop_key: str) -> Dict[str, Any]:
    return {
        "method": code,
        "rows": [],
        "detail_rows": [],
        "fit": None,
        "D": NAN,
        "D_se": NAN,
        "D_ci95": NAN,
        "v": NAN,
        "p": NAN,
        "p_se": NAN,
        "check": "No fit",
        "segments": None,
        "warnings": [],
        "errors": [],
        "info": [],
        "population_key": pop_key,
    }


def _finish_fit(
    result: Dict[str, Any],
    code: str,
    db: Any,
    pop_key: str,
    rows: List[Dict[str, Any]],
    xkey: str,
    ykey: str,
    warnings: List[str],
) -> None:
    result["rows"] = rows
    used = [r for r in rows if r.get("use") and r.get("valid")]
    fit = None
    D = NAN
    if len(used) >= 2:
        try:
            fit = linear_regression([r[xkey] for r in used], [r[ykey] for r in used])
            D = dimension_from_slope(code, fit["slope"])
        except Exception as exc:
            warnings.append(str(exc))
            fit = None
            D = NAN
    else:
        name = "number" if code == "N" else "mass"
        warnings.append(f"Fewer than 2 valid {name}-based points are selected for regression.")
    result["fit"] = fit
    result["D"] = D
    result["D_se"] = fit["se_slope"] if fit else NAN
    result["D_ci95"] = fit["ci95_slope"] if fit else NAN
    if code == "M":
        result["v"] = fit["slope"] if fit else NAN
    result["p"] = fragmentation_probability(D)
    result["p_se"] = fragmentation_probability_se(D, result["D_se"])
    result["check"] = fit_check(fit) if fit else "No fit"
    if fit and np.isfinite(D) and (D < 0.0 or D > 3.0):
        warnings.append(f"D = {D:.3f} is outside 0–3: the fragmentation probability p (Perugini et al. 2011, Eq. 19) is undefined.")

    result["segments"] = None
    cfg = segment_settings(db, code, pop_key)
    if cfg.get("enabled"):
        bp = cfg.get("break_phi")
        if bp is None:
            warnings.append("Two-segment fit: choose a break φ.")
        else:
            pts = [(float(r["phi"]), float(r[xkey]), float(r[ykey])) for r in used]
            try:
                result["segments"] = two_segment_fit(code, pts, float(bp))
            except Exception as exc:
                warnings.append(f"Two-segment fit: {exc}.")
    result["warnings"] = warnings


def calculate_mass_method(db: Any, kind: str = "whole", value: Optional[str] = None, *_unused: Any, **_unused_kw: Any) -> Dict[str, Any]:
    """
    Mass-based D (Perugini & Kueppers 2012; Paredes-Marino et al. 2017).

    For each class: R = upper size limit of the class (aperture of the next
    coarser sieve) and M(r<R) = mass of the class plus all finer classes, i.e.
    the mass that passes that sieve.  D = 3 - v.
    """
    phis = sample_phi_rows(db)
    pop_key = population_key(kind, value)
    result = _empty_result("M", pop_key)
    if db is None or not phis:
        result["warnings"] = ["The sample has no grain-size data."]
        return result
    componentry = has_componentry(db)
    bounds = class_bounds(db, phis)
    masses, status, notes = population_profile(db, kind, value, phis, componentry)
    warnings: List[str] = list(notes)

    if kind != "whole":
        cov = whole_componentry_coverage_percent(db, phis)
        if np.isfinite(cov) and cov < 99.999:
            warnings.append(f"Componentry covers {cov:.1f}% of the sample mass; this population uses only the classified fractions.")

    domain = [float(phi) for phi in phis if not status[float(phi)]]
    total_mass = float(sum(max(0.0, masses.get(phi, 0.0)) for phi in domain))
    result["population_mass_wt"] = total_mass
    if total_mass <= 0:
        result["warnings"] = warnings + ["Selected population has no positive mass."]
        return result

    cum_map: Dict[float, float] = {}
    acc = 0.0
    for phi in reversed(domain):
        acc += max(0.0, masses.get(phi, 0.0))
        cum_map[phi] = acc

    rows: List[Dict[str, Any]] = []
    for phi in phis:
        p = float(phi)
        b = bounds[p]
        st = status[p]
        row = {
            "use": False,
            "valid": False,
            "phi": p,
            "status": st or "OK",
            "lower_mm": b["lower_mm"],
            "R_mm": b["upper_mm"],
            "bin_mass": float(max(0.0, masses.get(p, 0.0))),
            "cumulative_mass_finer": NAN,
            "mass_fraction": NAN,
            "log10_R": NAN,
            "log10_mass_fraction": NAN,
        }
        if not st:
            cum = float(cum_map.get(p, 0.0))
            frac = cum / total_mass
            log_r = math.log10(b["upper_mm"]) if b["upper_mm"] > 0 else NAN
            log_f = math.log10(frac) if frac > 0 else NAN
            valid = bool(np.isfinite(log_r) and np.isfinite(log_f))
            row.update({
                "cumulative_mass_finer": cum,
                "mass_fraction": float(frac),
                "log10_R": float(log_r),
                "log10_mass_fraction": float(log_f),
                "valid": valid,
            })
            row["use"] = bool(valid and fit_use_for_phi(db, "M", pop_key, p))
        rows.append(row)

    _finish_fit(result, "M", db, pop_key, rows, "log10_R", "log10_mass_fraction", warnings)
    return result


def calculate_number_method(db: Any, kind: str = "whole", value: Optional[str] = None, *_unused: Any, **_unused_kw: Any) -> Dict[str, Any]:
    """
    Number-based D (Perugini et al. 2011, Eqs. 2-3).

    n = M / (rho V(B)) in each class; N(>=r) is the cumulative number from the
    coarsest class down to the class retained on the sieve of aperture r.
    D = -slope of log10 N(>=r) vs log10 r.
    """
    phis = sample_phi_rows(db)
    pop_key = population_key(kind, value)
    result = _empty_result("N", pop_key)
    result.update({
        "componentry": False,
        "componentry_coverage": NAN,
        "modeled_coverage": NAN,
        "missing_density": [],
        "mass_basis": MASS_BASIS_NORMALIZED,
        "selected_mass_wt": 0.0,
    })
    if db is None or not phis:
        result["warnings"] = ["The sample has no grain-size data."]
        return result

    settings = ensure_fractal_settings(db)
    componentry = has_componentry(db)
    bounds = class_bounds(db, phis)
    components = population_components(db, kind, value, True, componentry)
    masses, status, notes = population_profile(db, kind, value, phis, componentry)
    warnings: List[str] = list(notes)
    errors: List[str] = []
    info: List[str] = []
    result["componentry"] = componentry
    result["mass_basis"] = settings.get("mass_basis", MASS_BASIS_NORMALIZED)
    if componentry:
        result["componentry_coverage"] = whole_componentry_coverage_percent(db, phis)

    if not components:
        result["warnings"] = ["Selected population contains no components."]
        return result

    if result["mass_basis"] == MASS_BASIS_ACTUAL:
        actual_mass = _json_float_or_none(settings.get("actual_sample_mass_g"))
        if actual_mass is None or actual_mass <= 0:
            errors.append("Actual sample mass must be > 0 g when 'Actual sample mass' is selected.")
            mass_scale = NAN
        else:
            mass_scale = float(actual_mass / 100.0)
    else:
        mass_scale = 1.0

    if kind != "whole":
        cov = result["componentry_coverage"]
        if np.isfinite(cov) and cov < 99.999:
            warnings.append(f"Componentry covers {cov:.1f}% of the sample mass; this population uses only the classified fractions.")

    domain = {float(phi) for phi in phis if not status[float(phi)]}
    missing_density: List[str] = []
    invalid_geometry: List[str] = []
    missing_b: List[str] = []
    used_default_density = False
    selected_mass_wt_total = 0.0
    modeled_mass_wt = 0.0
    n_by_phi: Dict[float, float] = {float(phi): 0.0 for phi in phis}
    detail_by_phi: Dict[float, List[Dict[str, Any]]] = {float(phi): [] for phi in phis}

    for phi in phis:
        p = float(phi)
        if p not in domain:
            continue
        for comp in components:
            mass_wt = component_mass_wt(db, comp, p)
            if mass_wt <= 0:
                continue
            selected_mass_wt_total += mass_wt
            params = effective_component_params(db, comp, p, phis, bounds, componentry)
            density = params.get("density")
            if params.get("default_density"):
                used_default_density = True
            b_mm = params.get("b_mm")
            alpha = _safe_float(params.get("alpha"), NAN)
            gamma = _safe_float(params.get("gamma"), NAN)
            shape = str(params.get("shape", "Custom"))
            component_mass_g = float(mass_wt * mass_scale) if np.isfinite(mass_scale) else NAN
            display = component_display_name(db, comp, componentry)

            row = {
                "phi": p,
                "status": "OK",
                "threshold_r_mm": bounds[p]["lower_mm"],
                "upper_mm": bounds[p]["upper_mm"],
                "component": display,
                "component_key": comp,
                "component_mass_wt": float(mass_wt),
                "component_mass_g": component_mass_g,
                "B_mm": b_mm if b_mm is not None else NAN,
                "density_g_cm3": density if density is not None else NAN,
                "default_density": bool(params.get("default_density")),
                "shape": shape,
                "A_over_B": alpha,
                "C_over_B": gamma,
                "A_mm": NAN,
                "C_mm": NAN,
                "particle_volume_mm3": NAN,
                "particle_volume_cm3": NAN,
                "particle_mass_g": NAN,
                "particle_number_bin_component": NAN,
                "shape_volume_factor": NAN,
                "N_relative_to_sphere": NAN,
                "valid_model": False,
                "model_error": "",
            }

            # Actual-mass mode cannot produce particle counts until a positive
            # sample mass has been supplied.
            if not np.isfinite(mass_scale):
                row["model_error"] = "Actual sample mass undefined"
                detail_by_phi[p].append(row)
                continue

            if density is None or not np.isfinite(_safe_float(density, NAN)) or float(density) <= 0:
                missing_density.append(display)
                row["model_error"] = "Missing density"
                detail_by_phi[p].append(row)
                continue
            if b_mm is None or not np.isfinite(_safe_float(b_mm, NAN)) or float(b_mm) <= 0:
                missing_b.append(f"{display} at φ {p:g}")
                row["model_error"] = "Missing/invalid representative B"
                detail_by_phi[p].append(row)
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
                n_by_phi[p] += float(calc["particle_number"])
                modeled_mass_wt += mass_wt
            except Exception as exc:
                invalid_geometry.append(f"{display} at φ {p:g}: {exc}")
                row["model_error"] = str(exc)
            detail_by_phi[p].append(row)

    missing_unique = list(dict.fromkeys(missing_density))
    if missing_unique:
        errors.append(
            "Number-based D needs a density for: " + ", ".join(missing_unique[:20])
            + (" ..." if len(missing_unique) > 20 else "")
            + ". Enter it in 'Particle properties', or use the mass method (Paredes-Mariño et al. 2017)."
        )
    if missing_b:
        unique = list(dict.fromkeys(missing_b))
        errors.append("Missing representative B for: " + ", ".join(unique[:20]) + (" ..." if len(unique) > 20 else ""))
    if invalid_geometry:
        errors.append("Invalid particle geometry: " + " | ".join(invalid_geometry[:12]) + (" ..." if len(invalid_geometry) > 12 else ""))
    if used_default_density:
        info.append(
            f"Sieving only: uniform default density {DEFAULT_BULK_DENSITY:g} g/cm³. "
            "With uniform density and shape their values do not change D (Perugini et al. 2011)."
        )

    # Model parameters must be complete before N(>=r) is valid: N at a given r
    # includes every coarser class, so one unmodelled class corrupts the curve.
    model_complete = (not errors) and np.isfinite(mass_scale)

    aggregate_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    for phi in phis:
        p = float(phi)
        b = bounds[p]
        st = status[p]
        agg = {
            "use": False,
            "valid": False,
            "phi": p,
            "status": st or "OK",
            "threshold_r_mm": float(b["lower_mm"]),
            "upper_mm": float(b["upper_mm"]),
            "particle_number_bin_total": NAN,
            "N_gt_r": NAN,
            "log10_r": NAN,
            "log10_N_gt_r": NAN,
        }
        if not st:
            n_bin = float(n_by_phi.get(p, 0.0))
            cumulative += n_bin
            r_mm = float(b["lower_mm"])
            log_r = math.log10(r_mm) if r_mm > 0 else NAN
            log_n = math.log10(cumulative) if cumulative > 0 else NAN
            valid = bool(model_complete and np.isfinite(log_r) and np.isfinite(log_n))
            agg.update({
                "valid": valid,
                "particle_number_bin_total": n_bin,
                "N_gt_r": float(cumulative),
                "log10_r": float(log_r),
                "log10_N_gt_r": float(log_n),
            })
            agg["use"] = bool(valid and fit_use_for_phi(db, "N", pop_key, p))
        aggregate_rows.append(agg)
        for row in detail_by_phi.get(p, []):
            row.update({
                "use": agg["use"],
                "valid": agg["valid"],
                "particle_number_bin_total": agg["particle_number_bin_total"],
                "N_gt_r": agg["N_gt_r"],
                "log10_r": agg["log10_r"],
                "log10_N_gt_r": agg["log10_N_gt_r"],
            })
            detail_rows.append(row)

    result["detail_rows"] = detail_rows
    if model_complete:
        _finish_fit(result, "N", db, pop_key, aggregate_rows, "log10_r", "log10_N_gt_r", warnings)
    else:
        result["rows"] = aggregate_rows
        result["warnings"] = warnings
    result["errors"] = errors
    result["info"] = info
    result["missing_density"] = missing_unique
    result["selected_mass_wt"] = float(selected_mass_wt_total)
    result["modeled_coverage"] = float(100.0 * modeled_mass_wt / selected_mass_wt_total) if selected_mass_wt_total > 0 else NAN
    return result


# =============================================================================
# Numeric self-tests
# =============================================================================

class _SelfTestDB:
    """Minimal ComponentryDB stand-in for the self-tests."""

    def __init__(self, phis, wts, components=None, total=None, types=None, filled=None):
        idx = [float(p) for p in phis]
        self.gsd = pd.DataFrame({"phi": idx, "wt": [float(w) for w in wts]})
        self.components = list(components or [])
        self.types = dict(types or {})
        self.total = total if isinstance(total, pd.DataFrame) else pd.DataFrame(index=idx)
        self.filled = set(float(v) for v in (filled or []))
        self.fractal_settings = {}


def _power_law_class_masses(phis: Sequence[float], D: float) -> Tuple[List[float], List[float]]:
    """Class masses (wt%) of an exact power law M(r<R)/M_T = (R/R_max)^(3-D); the finest class is a pan."""
    phis = sorted(float(p) for p in phis)
    v = 3.0 - float(D)
    step0 = (phis[1] - phis[0]) if len(phis) > 1 else 1.0
    uppers = [phi_to_mm(phis[0] - step0)] + [phi_to_mm(p) for p in phis[:-1]]
    r_max = uppers[0]

    def F(r: float) -> float:
        return (r / r_max) ** v

    masses = []
    for i, p in enumerate(phis):
        below = F(phi_to_mm(p)) if i < len(phis) - 1 else 0.0
        masses.append(100.0 * (F(uppers[i]) - below))
    return phis, masses


def run_numeric_self_tests() -> Dict[str, bool]:
    """Numerical checks of the geometry, the regression and both fractal methods."""
    out: Dict[str, bool] = {}

    b = 2.5
    sphere = ellipsoid_geometry(b, 1.0, 1.0)
    out["sphere_volume"] = bool(np.isclose(sphere["volume_mm3"], math.pi / 6.0 * b ** 3, rtol=1e-12, atol=1e-12))
    lam = ellipsoid_geometry(b, 1.5, 0.1)
    out["laminar_volume_factor"] = bool(np.isclose(lam["volume_mm3"] / sphere["volume_mm3"], 0.15, rtol=1e-12))
    out["laminar_number_factor"] = bool(np.isclose(lam["n_relative_sphere"], 1.0 / 0.15, rtol=1e-12))
    n1 = estimate_particle_number(10.0, 1.0, b, 1.0, 1.0)["particle_number"]
    n2 = estimate_particle_number(10.0, 2.0, b, 1.0, 1.0)["particle_number"]
    out["density_inverse"] = bool(np.isclose(n2 / n1, 0.5, rtol=1e-12))
    n_big = estimate_particle_number(10.0, 1.0, 2.0 * b, 1.0, 1.0)["particle_number"]
    out["size_cube"] = bool(np.isclose(n_big / n1, 1.0 / 8.0, rtol=1e-12))

    # Textbook regression: slope 0.6, SE = sqrt(0.08).
    reg = linear_regression([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
    out["regression_slope"] = bool(np.isclose(reg["slope"], 0.6, rtol=1e-12))
    out["regression_standard_error"] = bool(np.isclose(reg["se_slope"], math.sqrt(0.08), rtol=1e-12))

    # Perugini et al. (2011), Eq. 19.
    out["probability_D2"] = bool(np.isclose(fragmentation_probability(2.0), 0.5))
    out["probability_D3"] = bool(np.isclose(fragmentation_probability(3.0), 1.0))
    out["probability_D0"] = bool(np.isclose(fragmentation_probability(0.0), 0.125))
    out["probability_outside_model"] = not np.isfinite(fragmentation_probability(3.2))

    # Mass-based D of an exact power law on unequal phi steps.
    phis, masses = _power_law_class_masses([-3, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 4], 2.4)
    res_m = calculate_mass_method(_SelfTestDB(phis, masses), "whole", None)
    out["mass_D_unequal_steps"] = bool(np.isfinite(res_m["D"]) and abs(res_m["D"] - 2.4) < 1e-9)

    # Number-based D does not depend on a uniform density or shape (Perugini et al. 2011).
    db1 = _SelfTestDB(phis, masses)
    ensure_component_settings(db1, UNDEFINED_COMPONENT, phis)["uniform_density"] = 1.0
    db2 = _SelfTestDB(phis, masses)
    cs2 = ensure_component_settings(db2, UNDEFINED_COMPONENT, phis)
    cs2.update({"uniform_density": 2.7, "uniform_shape": "Laminar", "uniform_alpha": 1.5, "uniform_gamma": 0.1})
    d1 = calculate_number_method(db1, "whole", None)["D"]
    d2 = calculate_number_method(db2, "whole", None)["D"]
    out["number_D_density_shape_invariant"] = bool(np.isfinite(d1) and np.isfinite(d2) and np.isclose(d1, d2, rtol=1e-10))
    d3 = calculate_number_method(_SelfTestDB(phis, masses), "whole", None)["D"]
    out["number_D_sieving_only_default_density"] = bool(np.isfinite(d3) and np.isclose(d3, d1, rtol=1e-10))

    # Two power laws meeting at phi = 0: coarse D = 1.5, fine D = 2.5.
    pts = []
    for ph in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        x = math.log10(phi_to_mm(ph))
        pts.append((ph, x, -1.5 * x if ph <= 0 else -2.5 * x))
    bp = suggest_break_phi(pts)
    seg = two_segment_fit("N", pts, 0.0)
    out["kink_detection"] = bool(bp is not None and abs(bp) < 1e-12)
    out["kink_dimensions"] = bool(np.isclose(seg["coarse"]["D"], 1.5) and np.isclose(seg["fine"]["D"], 2.5))

    # Component populations use only classified fractions; missing densities block N.
    cphis = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    total = pd.DataFrame(index=cphis, data={
        "J": [6.0, 12.0, 15.0, np.nan, np.nan, np.nan],
        "L": [4.0, 8.0, 15.0, np.nan, np.nan, np.nan],
    })
    dbc = _SelfTestDB(cphis, [10, 20, 30, 20, 15, 5], ["J", "L"], total, {"J": "Juvenile", "L": "Lithic"}, [-2.0, -1.0, 0.0])
    res_c = calculate_mass_method(dbc, "component", "J")
    out["component_unclassified_excluded"] = bool(sum(1 for r in res_c["rows"] if r["valid"]) == 3)
    res_n = calculate_number_method(dbc, "whole", None)
    out["componentry_missing_density_blocks_N"] = bool(res_n["errors"] and not np.isfinite(res_n["D"]) and "J" in res_n["missing_density"])
    return out


# =============================================================================
# Small UI helpers
# =============================================================================

def _read_only_item(text: str, user_data: Any = None) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    if user_data is not None:
        item.setData(Qt.UserRole, user_data)
    return item


def _use_item(phi: float, use: bool, valid: bool) -> QTableWidgetItem:
    item = QTableWidgetItem("")
    if valid:
        item.setFlags((item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
    else:
        item.setFlags(Qt.ItemIsSelectable)
    item.setCheckState(Qt.Checked if (use and valid) else Qt.Unchecked)
    item.setData(Qt.UserRole, float(phi))
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


def _fixed(value: Any, digits: int = 3) -> str:
    v = _safe_float(value, NAN)
    return f"{v:.{digits}f}" if np.isfinite(v) else "—"


def _sanitize_filename(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip().strip(".")
    return s or "fractal_analysis"


def _segment_summary(seg: Optional[Dict[str, Any]]) -> str:
    if not seg:
        return "—"
    c = seg["coarse"]
    f = seg["fine"]
    bp = seg["break_phi"]
    return (
        f"φ≤{bp:g}: D={_fixed(c['D'])}±{_fixed(c['D_se'])} (R² {_fixed(c['fit']['r2'])})   "
        f"φ≥{bp:g}: D={_fixed(f['D'])}±{_fixed(f['D_se'])} (R² {_fixed(f['fit']['r2'])})"
    )


def _fit_phi_range(result: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if not result:
        return NAN, NAN
    used = [float(r["phi"]) for r in result.get("rows", []) if r.get("use") and r.get("valid")]
    if not used:
        return NAN, NAN
    return min(used), max(used)


# =============================================================================
# Particle model editor
# =============================================================================

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

        self.lbl_mode = QLabel("")
        self.lbl_mode.setWordWrap(True)
        self.lbl_mode.setStyleSheet("font-weight:600; color:#2d4a63;")
        root.addWidget(self.lbl_mode)

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
        self.sp_density.setSpecialValueText("Not set")
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

        self.lbl_missing = QLabel("")
        self.lbl_missing.setWordWrap(True)
        self.lbl_missing.setStyleSheet("color:#92400E; font-weight:600;")
        root.addWidget(self.lbl_missing)

        note = QLabel(
            "Shape presets are starting values; the calculation uses the A/B and C/B shown. "
            "B is the particle size of each class (Advanced settings, below)."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#555555;")
        root.addWidget(note)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "φ", "B (mm)", "Density (g/cm³)", "Shape", "A/B", "C/B",
            "Volume factor", "N/N sphere",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.itemChanged.connect(self._on_table_item_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setColumnHidden(6, True)
        self.table.setColumnHidden(7, True)
        root.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_apply_density_all = QPushButton("Apply density to all fractions")
        self.btn_apply_shape_all = QPushButton("Apply shape to all fractions")
        self.btn_more = QToolButton()
        self.btn_more.setText("More actions")
        self.btn_more.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_more)
        menu.addAction("Copy parameters to all fractions", self.copy_parameters_to_all)
        menu.addAction("Copy parameters to selected fractions", self.copy_parameters_to_selected)
        menu.addAction("Copy from previous fraction", self.copy_from_previous)
        menu.addAction("Reset shape to sphere", self.reset_shape_sphere)
        menu.addAction("Reset component parameters", self.reset_component_parameters)
        self.btn_more.setMenu(menu)
        for b in (self.btn_apply_density_all, self.btn_apply_shape_all, self.btn_more):
            buttons.addWidget(b)
        buttons.addStretch(1)
        root.addLayout(buttons)

        self.btn_apply_density_all.clicked.connect(self.apply_density_to_all)
        self.btn_apply_shape_all.clicked.connect(self.apply_shape_to_all)

    def _db(self) -> Any:
        return self.owner.current_db()

    def current_component(self) -> str:
        data = self.cmb_component.currentData()
        if data:
            return str(data)
        return str(self.cmb_component.currentText()).strip()

    def refresh_components(self, preferred: Optional[str] = None) -> None:
        db = self._db()
        current = preferred or self.current_component()
        self.cmb_component.blockSignals(True)
        self.cmb_component.clear()
        componentry = has_componentry(db) if db is not None else False
        if db is not None:
            for comp in model_component_keys(db):
                self.cmb_component.addItem(component_display_name(db, comp, componentry), comp)
        self.cmb_component.blockSignals(False)
        if current:
            idx = self.cmb_component.findData(current)
            if idx < 0:
                idx = self.cmb_component.findText(current)
            if idx >= 0:
                self.cmb_component.setCurrentIndex(idx)
        if self.cmb_component.count() and self.cmb_component.currentIndex() < 0:
            self.cmb_component.setCurrentIndex(0)
        self.refresh()

    def _update_mode_texts(self, db: Any, comp: str) -> None:
        componentry = has_componentry(db) if db is not None else False
        bulk = (not componentry) and comp == UNDEFINED_COMPONENT
        if db is None:
            self.lbl_mode.setText("")
        elif not componentry:
            self.lbl_mode.setText(
                f"Sieving only: one density and one shape for all size classes (default {DEFAULT_BULK_DENSITY:g} g/cm³, sphere). "
                "With one value for all classes, D does not depend on it."
            )
        else:
            self.lbl_mode.setText(
                "Enter an approximate density and shape for each component. Without densities, the mass method is used. "
                f"'{UNCLASSIFIED_LABEL}' is the mass without componentry."
            )
        self.sp_density.setSpecialValueText(f"Default ({DEFAULT_BULK_DENSITY:g})" if bulk else "Not set")
        missing = self._missing_density_components(db) if (db is not None and componentry) else []
        self.lbl_missing.setText("Density missing for: " + ", ".join(missing) if missing else "")

    @staticmethod
    def _missing_density_components(db: Any) -> List[str]:
        phis = sample_phi_rows(db)
        out: List[str] = []
        for comp in model_component_keys(db):
            masses = {p: component_mass_wt(db, comp, p) for p in phis}
            if not any(m > 0 for m in masses.values()):
                continue
            cs = ensure_component_settings(db, comp, phis)
            if cs.get("density_mode") == DENSITY_BY_FRACTION:
                missing = any(
                    masses[p] > 0 and _json_float_or_none(cs["by_phi"].get(_phi_key(p), {}).get("density")) is None
                    for p in phis
                )
            else:
                missing = _json_float_or_none(cs.get("uniform_density")) is None
            if missing:
                out.append(component_display_name(db, comp, True))
        return out

    def refresh(self, *args) -> None:
        if self._loading:
            return
        db = self._db()
        comp = self.current_component()
        self._loading = True
        try:
            self._update_mode_texts(db, comp)
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
        rep_mode = ensure_fractal_settings(db).get("representative_size_mode", REPRESENTATIVE_SIEVE)
        bounds = class_bounds(db, phis)
        density_by = cs.get("density_mode") == DENSITY_BY_FRACTION
        shape_by = cs.get("shape_mode") == SHAPE_BY_FRACTION
        bulk = (comp == UNDEFINED_COMPONENT) and not has_componentry(db)
        self.table.setHorizontalHeaderItem(1, QTableWidgetItem("B (mm, " + ("manual" if rep_mode == REPRESENTATIVE_MANUAL else ("geometric midpoint" if rep_mode == REPRESENTATIVE_GEOMETRIC_MID else "sieve aperture")) + ")"))

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
                b_item = _read_only_item(_number_text(representative_size_mm(rep_mode, bounds[float(phi)], None), 8))
            self.table.setItem(row_idx, 1, b_item)

            if density_by:
                rho = _json_float_or_none(row_cfg.get("density"))
                if rho is None and bulk:
                    rho_item = QTableWidgetItem("")
                    rho_item.setToolTip(f"Empty: default {DEFAULT_BULK_DENSITY:g} g/cm³")
                else:
                    rho_item = QTableWidgetItem("" if rho is None else _number_text(rho, 8))
            else:
                rho = _json_float_or_none(cs.get("uniform_density"))
                if rho is None and bulk:
                    rho_item = _read_only_item(f"{DEFAULT_BULK_DENSITY:g} (default)")
                else:
                    rho_item = _read_only_item(_number_text(rho, 8))
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

            factor = alpha * gamma if np.isfinite(alpha) and np.isfinite(gamma) else NAN
            rel = 1.0 / factor if np.isfinite(factor) and factor > 0 else NAN
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

    def _on_uniform_shape_changed(self, *args) -> None:
        if self._loading:
            return
        db = self._db()
        comp = self.current_component()
        if db is None or not comp:
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        shape = str(self.cmb_shape.currentText())
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
        phi = _safe_float(phi_item.data(Qt.UserRole), NAN)
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
        QTimer.singleShot(0, self.refresh)

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
        phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else NAN, NAN)
        if not np.isfinite(phi):
            return
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        cfg = cs["by_phi"].setdefault(_phi_key(phi), {})
        val = _safe_float(item.text().replace(",", "."), NAN)
        if col == 1 and ensure_fractal_settings(db).get("representative_size_mode") == REPRESENTATIVE_MANUAL:
            cfg["manual_b_mm"] = float(val) if np.isfinite(val) and val > 0 else None
        elif col == 2 and cs.get("density_mode") == DENSITY_BY_FRACTION:
            cfg["density"] = float(val) if np.isfinite(val) and val > 0 else None
        elif col == 4 and cs.get("shape_mode") == SHAPE_BY_FRACTION:
            cfg["alpha"] = float(val) if np.isfinite(val) else NAN
        elif col == 5 and cs.get("shape_mode") == SHAPE_BY_FRACTION:
            cfg["gamma"] = float(val) if np.isfinite(val) else NAN
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
        phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else NAN, NAN)
        if not np.isfinite(phi):
            return None
        cs = ensure_component_settings(db, comp, sample_phi_rows(db))
        return copy.deepcopy(cs["by_phi"].get(_phi_key(phi), {}))

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
            phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else NAN, NAN)
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
        phi = _safe_float(phi_item.data(Qt.UserRole) if phi_item else NAN, NAN)
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
        name = component_display_name(db, comp)
        ans = QMessageBox.question(self, "Reset particle model", f"Reset all fractal particle-model parameters for '{name}'?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        settings = ensure_fractal_settings(db)
        settings.setdefault("components", {})[comp] = _default_component_settings()
        ensure_component_settings(db, comp, sample_phi_rows(db))
        self.owner.settings_changed(recalculate=True); self.refresh()


# =============================================================================
# Visual helpers: result cards, collapsible sections and chart style
# =============================================================================
_QUALITY_BADGE = {
    "good": ("Good fit", "#DCFCE7", "#166534"),
    "weak": ("Weak fit", "#FEF3C7", "#92400E"),
    "none": ("Not available", "#F3F4F6", "#6B7280"),
}

# Columns shown in the Data tab unless "Show all columns" is ticked.
_NUMBER_MAIN_COLUMNS = (0, 1, 2, 3, 5, 6, 9, 10, 18, 20)
_MASS_MAIN_COLUMNS = (0, 1, 2, 4, 5, 6)


def _fit_quality(res: Optional[Dict[str, Any]]) -> str:
    """'good', 'weak' or 'none', from the acceptance criteria of Paredes-Mariño et al. (2017)."""
    if not res or not res.get("fit") or not np.isfinite(_safe_float(res.get("D"), NAN)):
        return "none"
    return "good" if str(res.get("check", "")) == "OK" else "weak"


class _Collapsible(QWidget):
    """Section whose content is shown or hidden with its header button."""

    def __init__(self, title: str, parent: Optional[QWidget] = None, expanded: bool = False):
        super().__init__(parent)
        self.button = QToolButton(self)
        self.button.setObjectName("faSection")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body = QWidget(self)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 2, 2, 6)
        self.body_layout.setSpacing(6)
        self.body.setVisible(expanded)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.setAlignment(Qt.AlignTop)
        lay.addWidget(self.button)
        lay.addWidget(self.body)
        self.button.toggled.connect(self._toggle)

    def _toggle(self, on: bool) -> None:
        self.button.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)


class _ResultCard(QFrame):
    """Large display of one fractal dimension with a simple fit-quality badge."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("faCard")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 16)
        lay.setSpacing(2)
        self.lbl_caption = QLabel("Fractal dimension D")
        self.lbl_caption.setObjectName("faCaption")
        self.lbl_value = QLabel("—")
        self.lbl_method = QLabel("")
        self.lbl_method.setObjectName("faMethod")
        self.lbl_method.setWordWrap(True)
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 6, 0, 0)
        self.lbl_badge = QLabel("")
        badge_row.addWidget(self.lbl_badge)
        badge_row.addStretch(1)
        self.lbl_extra = QLabel("")
        self.lbl_extra.setObjectName("faMethod")
        self.lbl_extra.setWordWrap(True)
        self.lbl_extra.setVisible(False)
        lay.addWidget(self.lbl_caption)
        lay.addWidget(self.lbl_value)
        lay.addWidget(self.lbl_method)
        lay.addLayout(badge_row)
        lay.addWidget(self.lbl_extra)
        self.set_result("M", None)

    def set_result(self, code: str, res: Optional[Dict[str, Any]]) -> None:
        D = _safe_float((res or {}).get("D"), NAN)
        color = UI_ACCENT.get(code, UI_TEXT) if np.isfinite(D) else UI_MUTED
        self.lbl_value.setText(f"{D:.2f}" if np.isfinite(D) else "—")
        self.lbl_value.setStyleSheet(f"font-size:54px; font-weight:700; color:{color};")
        self.lbl_method.setText(UI_METHOD_NAME.get(code, ""))
        text, bg, fg = _QUALITY_BADGE[_fit_quality(res)]
        self.lbl_badge.setText(text)
        self.lbl_badge.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:10px; padding:3px 10px; font-weight:600;"
        )
        seg = (res or {}).get("segments")
        if seg:
            self.lbl_extra.setText(
                f"Two segments: coarse {_fixed(seg['coarse']['D'], 2)}  ·  fine {_fixed(seg['fine']['D'], 2)}"
            )
        self.lbl_extra.setVisible(bool(seg))


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(UI_SPINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors="#4B5563", labelsize=10, length=0, pad=6)
    ax.grid(True, which="major", color=UI_GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def _axes_message(ax, text: str) -> None:
    ax.cla()
    ax.set_facecolor("white")
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=13, color="#6B7280", wrap=True)


def _axis_labels(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11, color=UI_TEXT, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color=UI_TEXT, labelpad=8)


def _legend(ax, loc: str) -> None:
    handles, _labels = ax.get_legend_handles_labels()
    if not handles:
        return
    leg = ax.legend(loc=loc, frameon=False, fontsize=9.5, handletextpad=0.4, borderaxespad=0.6)
    for t in leg.get_texts():
        t.set_color(UI_TEXT)


def _mm_tick_label(v: float, _pos: Any = None) -> str:
    if not np.isfinite(v) or v <= 0:
        return ""
    return f"{v:g}" if v >= 1 else f"{v:.3g}"


def _percent_tick_label(v: float, _pos: Any = None) -> str:
    return f"{v:g}%" if np.isfinite(v) else ""


def _positive(values: Iterable[Any]) -> List[float]:
    return [float(v) for v in values if np.isfinite(_safe_float(v, NAN)) and float(v) > 0]


def _set_size_axis(ax, sizes: Iterable[Any], log: bool = True) -> None:
    """x axis in mm; on a log scale the ticks follow the sieve series (powers of 2)."""
    vals = _positive(sizes)
    if not log:
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.05 * ((hi - lo) if hi > lo else 1.0)
            ax.set_xlim(max(0.0, lo - pad), hi + pad)
        return
    ax.set_xscale("log")
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    pad = 10 ** 0.08
    ax.set_xlim(lo / pad, hi * pad)
    kmin = int(math.floor(math.log2(lo) + 1e-9))
    kmax = int(math.ceil(math.log2(hi) - 1e-9))
    ks = list(range(kmin, kmax + 1))
    step = 1 if len(ks) <= 12 else 2
    ax.xaxis.set_major_locator(FixedLocator([2.0 ** k for k in ks[::step]]))
    ax.xaxis.set_major_formatter(FuncFormatter(_mm_tick_label))
    ax.xaxis.set_minor_locator(NullLocator())


def _set_value_axis(ax, values: Iterable[Any], percent: bool, log: bool = True) -> None:
    vals = _positive(values)
    if not log:
        if percent:
            ax.yaxis.set_major_formatter(FuncFormatter(_percent_tick_label))
        return
    ax.set_yscale("log")
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    decades = math.log10(hi / lo) if hi > lo else 0.0
    if percent:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0) if decades <= 3.5 else (1.0,)))
        ax.yaxis.set_major_formatter(FuncFormatter(_percent_tick_label))
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())
    pad = 10 ** max(0.08, 0.05 * decades)
    ax.set_ylim(lo / pad, hi * pad)


def _draw_d_label(ax, code: str, res: Optional[Dict[str, Any]], corner: str) -> None:
    """Large 'D = x.xx' inside the chart, so that exported figures show it too."""
    D = _safe_float((res or {}).get("D"), NAN)
    if not np.isfinite(D):
        return
    x, ha = (0.03, "left") if corner == "left" else (0.97, "right")
    box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85)
    ax.annotate(f"D = {D:.2f}", xy=(x, 0.97), xycoords="axes fraction", ha=ha, va="top", fontsize=24,
                fontweight="bold", color=UI_ACCENT.get(code, UI_TEXT), bbox=box, zorder=7)
    sub = UI_METHOD_NAME.get(code, "")
    seg = (res or {}).get("segments")
    if seg:
        sub += f"\nCoarse {_fixed(seg['coarse']['D'], 2)}  ·  Fine {_fixed(seg['fine']['D'], 2)}"
    ax.annotate(sub, xy=(x, 0.97), xycoords="axes fraction", xytext=(0, -42), textcoords="offset points",
                ha=ha, va="top", fontsize=10.5, color="#6B7280", linespacing=1.5, bbox=box, zorder=7)


# =============================================================================
# Main window
# =============================================================================

class FractalAnalysisWindow(QMainWindow):
    """Independent, resizable Fractal Analysis window."""

    def __init__(self, app: Any):
        super().__init__(app)
        install_no_wheel_filter()
        self.app = app
        self.setWindowTitle("Fractal analysis")
        self.resize(1460, 920)
        self.setMinimumSize(1050, 700)
        self.setWindowFlags(self.windowFlags() | Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(FA_STYLESHEET)

        self._loading = False
        self._loading_segments = False
        self._recalculating = False
        self._last_sample_names: Tuple[str, ...] = tuple()
        self.number_result: Optional[Dict[str, Any]] = None
        self.mass_result: Optional[Dict[str, Any]] = None
        self.mass_is_fallback = False
        self._pick_targets: Dict[Any, Tuple[str, List[float]]] = {}
        self._pending_pick: Optional[Tuple[str, float]] = None

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
        central.setObjectName("faCentral")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # Header: what is analysed and how
        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(QLabel("Sample"))
        self.cmb_sample = QComboBox()
        self.cmb_sample.setMinimumWidth(190)
        header.addWidget(self.cmb_sample, 2)
        header.addSpacing(10)
        header.addWidget(QLabel("Population"))
        self.cmb_population = QComboBox()
        self.cmb_population.setMinimumWidth(160)
        header.addWidget(self.cmb_population, 2)
        self.lbl_component = QLabel("Component")
        header.addWidget(self.lbl_component)
        self.cmb_component = QComboBox()
        self.cmb_component.setMinimumWidth(150)
        header.addWidget(self.cmb_component, 2)
        header.addSpacing(10)
        header.addWidget(QLabel("Method"))
        self.cmb_method = QComboBox()
        self.cmb_method.addItems([METHOD_NUMBER, METHOD_MASS, METHOD_COMPARE])
        self.cmb_method.setToolTip(
            "By particle count: Perugini et al. (2011); needs the density of the particles.\n"
            "By mass: Paredes-Mariño et al. (2017); needs only the sieving data."
        )
        header.addWidget(self.cmb_method, 1)
        header.addStretch(1)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.setToolTip("Reload the samples from Componentry.")
        self.btn_method_info = QPushButton("About the method")
        header.addWidget(self.btn_refresh)
        header.addWidget(self.btn_method_info)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        # ---------------------------------------------------------------- Result
        results = QWidget()
        res_lay = QHBoxLayout(results)
        res_lay.setContentsMargins(0, 10, 0, 0)
        res_lay.setSpacing(12)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("faLeft")
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(345)
        left = QWidget()
        left.setObjectName("faLeftBody")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 10, 0)
        left_lay.setSpacing(10)

        self.cards: Dict[str, _ResultCard] = {"N": _ResultCard(), "M": _ResultCard()}
        for card in self.cards.values():
            left_lay.addWidget(card)
        hint = QLabel("Higher D means stronger fragmentation, with more fine particles.")
        hint.setObjectName("faHint")
        hint.setWordWrap(True)
        left_lay.addWidget(hint)

        self.lbl_notice = QLabel("")
        self.lbl_notice.setObjectName("faNotice")
        self.lbl_notice.setWordWrap(True)
        self.lbl_notice.setVisible(False)
        left_lay.addWidget(self.lbl_notice)
        self.btn_set_properties = QPushButton("Set particle densities")
        self.btn_set_properties.setObjectName("faPrimary")
        self.btn_set_properties.setVisible(False)
        left_lay.addWidget(self.btn_set_properties)

        # Fit options (hidden until opened)
        self.sec_fit = _Collapsible("Fit options")
        fl = self.sec_fit.body_layout
        range_lbl = QLabel("Size classes used in the fit")
        range_lbl.setObjectName("faHint")
        fl.addWidget(range_lbl)
        range_grid = QGridLayout()
        range_grid.setHorizontalSpacing(6)
        range_grid.addWidget(QLabel("From"), 0, 0)
        self.cmb_fit_from = QComboBox()
        range_grid.addWidget(self.cmb_fit_from, 0, 1)
        range_grid.addWidget(QLabel("To"), 1, 0)
        self.cmb_fit_to = QComboBox()
        range_grid.addWidget(self.cmb_fit_to, 1, 1)
        self.btn_apply_range = QPushButton("Apply")
        range_grid.addWidget(self.btn_apply_range, 0, 2, 2, 1)
        range_grid.setColumnStretch(1, 1)
        fl.addLayout(range_grid)
        quick = QHBoxLayout()
        self.btn_perugini_range = QPushButton("Standard range")
        self.btn_perugini_range.setToolTip(
            "φ −2.5 to 2 (sieves 5.66 to 0.25 mm), the range used by Perugini et al. (2011) "
            "to compare with Kueppers et al. (2006)."
        )
        self.btn_select_all = QPushButton("Use all points")
        quick.addWidget(self.btn_perugini_range)
        quick.addWidget(self.btn_select_all)
        fl.addLayout(quick)

        self.chk_seg: Dict[str, QCheckBox] = {}
        self.cmb_seg_break: Dict[str, QComboBox] = {}
        self.btn_seg_suggest: Dict[str, QPushButton] = {}
        self.seg_rows: Dict[str, QWidget] = {}
        for code in ("N", "M"):
            row = QWidget()
            rl = QGridLayout(row)
            rl.setContentsMargins(0, 6, 0, 0)
            rl.setHorizontalSpacing(6)
            chk = QCheckBox("Two segments (kink)")
            chk.setToolTip(
                "Fits one line on each side of a break size (Paredes-Mariño et al. 2017). "
                "The break point belongs to both segments."
            )
            cmb = QComboBox()
            btn = QPushButton("Suggest")
            btn.setToolTip(f"Finds the break that fits best (at least {MIN_SEGMENT_POINTS} points per segment).")
            rl.addWidget(chk, 0, 0, 1, 3)
            rl.addWidget(QLabel("Break at"), 1, 0)
            rl.addWidget(cmb, 1, 1)
            rl.addWidget(btn, 1, 2)
            rl.setColumnStretch(1, 1)
            fl.addWidget(row)
            self.chk_seg[code] = chk
            self.cmb_seg_break[code] = cmb
            self.btn_seg_suggest[code] = btn
            self.seg_rows[code] = row

        view_row = QHBoxLayout()
        view_row.setContentsMargins(0, 6, 0, 0)
        view_row.addWidget(QLabel("Chart"))
        self.cmb_view = QComboBox()
        self.cmb_view.addItems([VIEW_LOGLOG, VIEW_LINEAR, VIEW_RESIDUALS])
        self.cmb_view.setToolTip(
            "Fit (log scale): the straight line that gives D.\n"
            "Linear scale: the same data without logarithms (Perugini et al. 2011, Fig. 6).\n"
            "Residuals: a curved pattern means the data bend (a kink)."
        )
        view_row.addWidget(self.cmb_view, 1)
        fl.addLayout(view_row)
        left_lay.addWidget(self.sec_fit)

        # Details (hidden until opened)
        self.sec_details = _Collapsible("Details")
        self.lbl_details = QLabel("")
        self.lbl_details.setObjectName("faDetails")
        self.lbl_details.setWordWrap(True)
        self.lbl_details.setTextFormat(Qt.RichText)
        self.lbl_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sec_details.body_layout.addWidget(self.lbl_details)
        left_lay.addWidget(self.sec_details)
        left_lay.addStretch(1)

        export_row = QHBoxLayout()
        self.btn_export_figure = QPushButton("Save chart")
        self.btn_export_results = QPushButton("Export results")
        export_row.addWidget(self.btn_export_figure)
        export_row.addWidget(self.btn_export_results)
        left_lay.addLayout(export_row)
        left_scroll.setWidget(left)
        res_lay.addWidget(left_scroll)

        chart_card = QFrame()
        chart_card.setObjectName("faChartCard")
        cc = QVBoxLayout(chart_card)
        cc.setContentsMargins(10, 10, 10, 8)
        cc.setSpacing(4)
        self.figure = Figure(figsize=(10.0, 6.4), dpi=96, facecolor="white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cc.addWidget(self.canvas, 1)
        tip = QLabel("Click a point to include it in the fit or leave it out.")
        tip.setObjectName("faHint")
        cc.addWidget(tip)
        res_lay.addWidget(chart_card, 1)
        self.tabs.addTab(results, "Result")


        # --------------------------------------------- Particle properties (count)
        self.model_widget = ParticleModelWidget(self)
        adv = _Collapsible("Advanced settings")
        adv_form = QFormLayout()
        self.cmb_mass_basis = QComboBox()
        self.cmb_mass_basis.addItems([MASS_BASIS_NORMALIZED, MASS_BASIS_ACTUAL])
        self.cmb_mass_basis.setToolTip("Only changes the absolute particle numbers, not D.")
        adv_form.addRow("Mass basis:", self.cmb_mass_basis)
        self.sp_actual_mass = QDoubleSpinBox()
        self.sp_actual_mass.setDecimals(6)
        self.sp_actual_mass.setRange(0.0, 1e12)
        self.sp_actual_mass.setSpecialValueText("Not set")
        self.sp_actual_mass.setSuffix(" g")
        adv_form.addRow("Actual sample mass:", self.sp_actual_mass)
        self.cmb_rep_size = QComboBox()
        self.cmb_rep_size.addItems(list(REPRESENTATIVE_MODES))
        self.cmb_rep_size.setToolTip(
            "Particle size B used to turn the mass of a class into a number of particles.\n"
            "Sieve aperture: B = 2^-φ, the sieve that retains the class (Perugini et al. 2011; Kueppers et al. 2006).\n"
            "Geometric class midpoint: B = √(lower × upper limit) (Wiejaczka & Giachetti 2022).\n"
            "Manual: B typed for each class.\n"
            "With constant φ steps the choice does not change D."
        )
        adv_form.addRow("Representative size:", self.cmb_rep_size)
        adv.body_layout.addLayout(adv_form)
        self.model_widget.layout().addWidget(adv)

        # ------------------------------------------------------------------ Data
        data_tab = QWidget()
        data_lay = QVBoxLayout(data_tab)
        data_lay.setContentsMargins(0, 10, 0, 0)
        data_top = QHBoxLayout()
        data_hint = QLabel("The numbers behind the chart for this sample. Untick 'Use' to leave a size class out of the fit.")
        data_hint.setObjectName("faSubtitle")
        data_top.addWidget(data_hint)
        data_top.addStretch(1)
        self.chk_all_columns = QCheckBox("Show all columns")
        data_top.addWidget(self.chk_all_columns)
        data_lay.addLayout(data_top)
        self.data_tabs = QTabWidget()
        data_lay.addWidget(self.data_tabs, 1)
        self.tbl_number = QTableWidget()
        self.tbl_mass = QTableWidget()
        for tbl in (self.tbl_number, self.tbl_mass):
            tbl.setAlternatingRowColors(True)
            tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
            tbl.verticalHeader().setVisible(False)
        self.tbl_number.itemChanged.connect(lambda item: self._on_use_item_changed("N", item))
        self.tbl_mass.itemChanged.connect(lambda item: self._on_use_item_changed("M", item))
        self.data_tabs.addTab(self.tbl_number, "Particle count")
        self.data_tabs.addTab(self.tbl_mass, "Mass")
        self.tabs.addTab(data_tab, "Sample data")
        self.tabs.addTab(self.model_widget, "Particle properties")

        # Signals
        self.cmb_sample.currentIndexChanged.connect(self._on_sample_changed)
        self.cmb_population.currentIndexChanged.connect(self._on_population_changed)
        self.cmb_component.currentIndexChanged.connect(self._on_component_changed)
        self.cmb_method.currentIndexChanged.connect(self._on_method_changed)
        self.cmb_view.currentIndexChanged.connect(lambda *_: self._draw_results())
        self.btn_refresh.clicked.connect(lambda: self.refresh_from_componentry(initial=False))
        self.btn_method_info.clicked.connect(self.show_method_information)
        self.btn_apply_range.clicked.connect(self.apply_fit_range)
        self.btn_perugini_range.clicked.connect(self.apply_perugini_range)
        self.btn_select_all.clicked.connect(lambda: self.set_all_fit_points(True))
        self.btn_set_properties.clicked.connect(lambda: self.tabs.setCurrentWidget(self.model_widget))
        for code in ("N", "M"):
            self.chk_seg[code].toggled.connect(lambda _checked, c=code: self._on_segment_controls_changed(c))
            self.cmb_seg_break[code].currentIndexChanged.connect(lambda _idx, c=code: self._on_segment_controls_changed(c))
            self.btn_seg_suggest[code].clicked.connect(lambda _checked=False, c=code: self.suggest_segment_break(c))
        self.cmb_mass_basis.currentIndexChanged.connect(self._on_settings_controls_changed)
        self.sp_actual_mass.valueChanged.connect(self._on_settings_controls_changed)
        self.cmb_rep_size.currentIndexChanged.connect(self._on_settings_controls_changed)
        self.btn_export_figure.clicked.connect(self.export_figure)
        self.btn_export_results.clicked.connect(self.export_results)
        self.chk_all_columns.toggled.connect(lambda *_: self._apply_column_visibility())
        self.canvas.mpl_connect("pick_event", self._on_chart_pick)

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
        old_pop = self.cmb_population.currentData()
        self._loading = True
        try:
            self.cmb_population.clear()
            if db is not None:
                self.cmb_population.addItem("Whole sample", ("whole", None))
                if has_componentry(db):
                    for t in observed_component_types(db):
                        self.cmb_population.addItem(population_label("type", t), ("type", t))
                    self.cmb_population.addItem("Individual component", ("component", None))
                if old_pop is not None:
                    for i in range(self.cmb_population.count()):
                        if self.cmb_population.itemData(i) == old_pop:
                            self.cmb_population.setCurrentIndex(i)
                            break
            self._refresh_component_selector()
            self._load_sample_settings_controls()
            self._refresh_fit_phi_controls()
            self._load_segment_controls()
        finally:
            self._loading = False
        self.model_widget.refresh_components()
        self._update_method_ui(set_current=False)
        self.recalculate()

    def _population_spec(self) -> Tuple[str, Optional[str]]:
        data = self.cmb_population.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            kind, value = str(data[0]), data[1]
        else:
            kind, value = "whole", None
        if kind == "component":
            data_c = self.cmb_component.currentData()
            value = str(data_c) if data_c else (str(self.cmb_component.currentText()).strip() or None)
        return kind, value

    def _refresh_component_selector(self) -> None:
        db = self.current_db()
        old = self.cmb_component.currentData()
        self.cmb_component.blockSignals(True)
        self.cmb_component.clear()
        if db is not None and has_componentry(db):
            for c in list(getattr(db, "components", []) or []):
                self.cmb_component.addItem(str(c), str(c))
        if old:
            idx = self.cmb_component.findData(old)
            if idx >= 0:
                self.cmb_component.setCurrentIndex(idx)
        self.cmb_component.blockSignals(False)
        kind = "whole"
        data = self.cmb_population.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            kind = str(data[0])
        visible = kind == "component"
        self.lbl_component.setVisible(visible)
        self.cmb_component.setVisible(visible)

    def _on_population_changed(self, *args) -> None:
        if self._loading:
            return
        self._refresh_component_selector()
        self._refresh_fit_phi_controls()
        self._load_segment_controls()
        self.recalculate()

    def _on_component_changed(self, *args) -> None:
        if self._loading:
            return
        self.model_widget.refresh_components(preferred=self.cmb_component.currentData())
        self._load_segment_controls()
        self.recalculate()

    def _on_method_changed(self, *args) -> None:
        if self._loading:
            return
        self._update_method_ui(set_current=True)
        self.recalculate()

    def _update_method_ui(self, set_current: bool = False) -> None:
        method = self.cmb_method.currentText()
        number_active = method in (METHOD_NUMBER, METHOD_COMPARE)
        mass_active = method in (METHOD_MASS, METHOD_COMPARE) or self.mass_is_fallback
        self.tabs.setTabEnabled(self.tabs.indexOf(self.model_widget), number_active)
        self.data_tabs.setTabEnabled(0, number_active)
        self.data_tabs.setTabEnabled(1, mass_active)
        if set_current:
            if method == METHOD_NUMBER:
                self.data_tabs.setCurrentIndex(0)
            elif method == METHOD_MASS:
                self.data_tabs.setCurrentIndex(1)
        self._update_segment_enabled()

    def _load_sample_settings_controls(self) -> None:
        db = self.current_db()
        if db is None:
            return
        s = ensure_fractal_settings(db)
        self.cmb_mass_basis.setCurrentText(s.get("mass_basis", MASS_BASIS_NORMALIZED))
        actual = _json_float_or_none(s.get("actual_sample_mass_g"))
        self.sp_actual_mass.setValue(actual if actual is not None else 0.0)
        self.cmb_rep_size.setCurrentText(s.get("representative_size_mode", REPRESENTATIVE_SIEVE))
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
            txt = f"φ {phi:g}  ·  {phi_to_mm(phi):.3g} mm"
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
    # Two-segment (kink) controls
    # -------------------------------------------------------------------------
    def _load_segment_controls(self) -> None:
        db = self.current_db()
        phis = sample_phi_rows(db) if db is not None else []
        kind, value = self._population_spec()
        pk = population_key(kind, value)
        self._loading_segments = True
        try:
            for code in ("N", "M"):
                cmb = self.cmb_seg_break[code]
                cmb.clear()
                for phi in phis:
                    cmb.addItem(f"φ {phi:g}", float(phi))
                if db is None:
                    self.chk_seg[code].setChecked(False)
                    continue
                cfg = segment_settings(db, code, pk)
                self.chk_seg[code].setChecked(bool(cfg.get("enabled")))
                bp = cfg.get("break_phi")
                if bp is not None and phis:
                    idx = int(np.argmin([abs(float(p) - float(bp)) for p in phis]))
                    cmb.setCurrentIndex(idx)
                elif phis:
                    cmb.setCurrentIndex(len(phis) // 2)
        finally:
            self._loading_segments = False
        self._update_segment_enabled()

    def _selection_method_codes(self) -> List[str]:
        codes = self._active_method_codes()
        if self.mass_is_fallback and "M" not in codes:
            codes.append("M")
        return codes

    def _update_segment_enabled(self) -> None:
        codes = [c for c, _res in self._panels()]
        for code in ("N", "M"):
            self.seg_rows[code].setVisible(code in codes)
            if len(codes) > 1:
                self.chk_seg[code].setText("Two segments, " + ("particle count" if code == "N" else "mass"))
            else:
                self.chk_seg[code].setText("Two segments (kink)")

    def _on_segment_controls_changed(self, code: str) -> None:
        if self._loading or self._loading_segments:
            return
        db = self.current_db()
        if db is None:
            return
        kind, value = self._population_spec()
        cfg = segment_settings(db, code, population_key(kind, value))
        cfg["enabled"] = bool(self.chk_seg[code].isChecked())
        bp = _safe_float(self.cmb_seg_break[code].currentData(), NAN)
        cfg["break_phi"] = float(bp) if np.isfinite(bp) else None
        self.settings_changed(recalculate=True)

    def suggest_segment_break(self, code: str) -> None:
        db = self.current_db()
        if db is None:
            return
        kind, value = self._population_spec()
        result = self.number_result if code == "N" else self.mass_result
        if result is None:
            result = calculate_number_method(db, kind, value) if code == "N" else calculate_mass_method(db, kind, value)
        xkey, ykey = ("log10_r", "log10_N_gt_r") if code == "N" else ("log10_R", "log10_mass_fraction")
        pts = [(float(r["phi"]), float(r[xkey]), float(r[ykey])) for r in result.get("rows", []) if r.get("use") and r.get("valid")]
        bp = suggest_break_phi(pts)
        if bp is None:
            QMessageBox.information(
                self, "Two-segment fit",
                f"At least {2 * MIN_SEGMENT_POINTS - 1} selected valid points are needed to suggest a break.",
            )
            return
        cfg = segment_settings(db, code, population_key(kind, value))
        cfg["enabled"] = True
        cfg["break_phi"] = float(bp)
        self._load_segment_controls()
        self.settings_changed(recalculate=True)

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
        a = _safe_float(self.cmb_fit_from.currentData(), NAN)
        b = _safe_float(self.cmb_fit_to.currentData(), NAN)
        if not np.isfinite(a) or not np.isfinite(b):
            return
        lo, hi = min(a, b), max(a, b)
        kind, value = self._population_spec()
        pk = population_key(kind, value)
        for code in self._selection_method_codes():
            for phi in sample_phi_rows(db):
                set_fit_use_for_phi(db, code, pk, phi, lo <= phi <= hi)
        self.settings_changed(recalculate=True)

    def apply_perugini_range(self) -> None:
        db = self.current_db()
        if db is None:
            return
        lo, hi = PERUGINI_2011_PHI_RANGE
        phis = sample_phi_rows(db)
        inside = [p for p in phis if lo - 1e-9 <= p <= hi + 1e-9]
        if not inside:
            QMessageBox.information(self, "Perugini 2011 range", "This sample has no classes between φ −2.5 and 2 (5.66 to 0.25 mm).")
            return
        kind, value = self._population_spec()
        pk = population_key(kind, value)
        for code in self._selection_method_codes():
            for phi in phis:
                set_fit_use_for_phi(db, code, pk, phi, lo - 1e-9 <= phi <= hi + 1e-9)
        i_from = self.cmb_fit_from.findData(float(min(inside)))
        i_to = self.cmb_fit_to.findData(float(max(inside)))
        if i_from >= 0:
            self.cmb_fit_from.setCurrentIndex(i_from)
        if i_to >= 0:
            self.cmb_fit_to.setCurrentIndex(i_to)
        self.settings_changed(recalculate=True)

    def set_all_fit_points(self, use: bool) -> None:
        db = self.current_db()
        if db is None:
            return
        kind, value = self._population_spec(); pk = population_key(kind, value)
        for code in self._selection_method_codes():
            for phi in sample_phi_rows(db):
                set_fit_use_for_phi(db, code, pk, phi, bool(use))
        self.settings_changed(recalculate=True)

    def invert_fit_points(self) -> None:
        db = self.current_db()
        if db is None:
            return
        kind, value = self._population_spec(); pk = population_key(kind, value)
        for code in self._selection_method_codes():
            for phi in sample_phi_rows(db):
                old = fit_use_for_phi(db, code, pk, phi)
                set_fit_use_for_phi(db, code, pk, phi, not old)
        self.settings_changed(recalculate=True)

    def _on_use_item_changed(self, code: str, item: QTableWidgetItem) -> None:
        if self._loading or self._recalculating or item.column() != 0:
            return
        if not (item.flags() & Qt.ItemIsUserCheckable):
            return
        phi = _safe_float(item.data(Qt.UserRole), NAN)
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
                self.number_result = None; self.mass_result = None; self.mass_is_fallback = False
                self._draw_empty("No sample loaded")
                self._clear_tables_and_results()
                return

            kind, value = self._population_spec()
            method = self.cmb_method.currentText()
            self.number_result = calculate_number_method(db, kind, value) if method in (METHOD_NUMBER, METHOD_COMPARE) else None
            self.mass_is_fallback = bool(
                method == METHOD_NUMBER and self.number_result is not None and self.number_result.get("missing_density")
            )
            need_mass = method in (METHOD_MASS, METHOD_COMPARE) or self.mass_is_fallback
            self.mass_result = calculate_mass_method(db, kind, value) if need_mass else None
            self._fill_number_table(self.number_result)
            self._fill_mass_table(self.mass_result)
            self._update_status()
            self._update_result_labels()
            self._draw_results()
            self._update_method_ui(set_current=False)
        except Exception:
            self._set_notice("Calculation error: " + traceback.format_exc().splitlines()[-1])
        finally:
            self._recalculating = False

    def _set_notice(self, text: str) -> None:
        self.lbl_notice.setText(text)
        self.lbl_notice.setVisible(bool(text))

    def _clear_tables_and_results(self) -> None:
        self.tbl_number.setRowCount(0); self.tbl_mass.setRowCount(0)
        for code, card in self.cards.items():
            card.set_result(code, None)
            card.setVisible(code == "M")
        self.lbl_details.setText("")
        self._set_notice("No sample loaded." if self.current_db() is None else "")
        self.btn_set_properties.setVisible(False)

    def _draw_empty(self, text: str) -> None:
        self.figure.clear()
        self._pick_targets = {}
        ax = self.figure.add_subplot(111)
        _axes_message(ax, text)
        self.canvas.draw_idle()

    def _update_status(self) -> None:
        """One short notice for the user; the full messages go to Details."""
        db = self.current_db()
        nr, mr = self.number_result, self.mass_result
        text = ""
        need_properties = False
        if db is None:
            text = "No sample loaded."
        elif self.mass_is_fallback:
            text = "No particle densities: showing the mass method."
            need_properties = True
        elif nr is not None and nr.get("missing_density"):
            text = "No particle densities: the particle count method cannot be calculated."
            need_properties = True
        else:
            errors = [str(x) for r in (nr, mr) if r for x in r.get("errors", []) if str(x)]
            qualities = [_fit_quality(res) for _code, res in self._panels()]
            if errors:
                text = "Could not be calculated. See Details."
            elif any(q == "none" for q in qualities):
                text = "Too few points for a fit (at least 3)."
            elif any(q == "weak" for q in qualities):
                text = "The points do not form a straight line, so this D is uncertain."
        self._set_notice(text)
        self.btn_set_properties.setVisible(need_properties)
        self.lbl_details.setText(self._details_html())

    @staticmethod
    def _fit_range_text(result: Optional[Dict[str, Any]]) -> str:
        lo, hi = _fit_phi_range(result)
        if not np.isfinite(lo):
            return "—"
        return f"{phi_to_mm(hi):.3g}–{phi_to_mm(lo):.3g} mm (φ {lo:g} to {hi:g})"

    @staticmethod
    def _plain_problem(res: Dict[str, Any]) -> str:
        if res.get("missing_density"):
            return "Particle densities are not set."
        errors = [str(x) for x in res.get("errors", []) if str(x)]
        if errors:
            e = errors[0].lower()
            if "actual sample mass" in e:
                return "Enter the sample mass in Particle properties, Advanced settings."
            if "representative b" in e:
                return "Enter the particle size B of each class in Particle properties."
            if "geometry" in e:
                return "Check the particle shapes in Particle properties."
            return errors[0]
        if not res.get("fit"):
            return "Too few points for a fit."
        return ""

    def _details_html(self) -> str:
        if self.current_db() is None:
            return ""
        panels = [(c, r) for c, r in self._panels() if r]
        blocks: List[str] = []
        for code, res in panels:
            fit = res.get("fit") or {}
            rows: List[Tuple[str, str]] = []
            if fit and np.isfinite(_safe_float(res.get("D"), NAN)):
                rows.append(("D", f"{_fixed(res.get('D'), 2)} ± {_fixed(res.get('D_se'), 2)}"))
                rows.append(("R²", _fixed(fit.get("r2"), 2)))
                rows.append(("Points", str(fit.get("n", "—"))))
                rows.append(("Range", self._fit_range_text(res)))
                seg = res.get("segments")
                if seg:
                    rows.append(("Coarse part", f"{_fixed(seg['coarse']['D'], 2)} ± {_fixed(seg['coarse']['D_se'], 2)}"))
                    rows.append(("Fine part", f"{_fixed(seg['fine']['D'], 2)} ± {_fixed(seg['fine']['D_se'], 2)}"))
            html = ""
            if len(panels) > 1:
                html += f"<p style='margin:6px 0 2px 0'><b style='color:{UI_ACCENT[code]}'>{_esc(UI_METHOD_NAME[code])}</b></p>"
            if rows:
                html += "<table cellspacing='0' cellpadding='2'>" + "".join(
                    f"<tr><td style='color:#6B7280; padding-right:14px'>{_esc(k)}</td><td><b>{_esc(v)}</b></td></tr>"
                    for k, v in rows
                ) + "</table>"
            problem = self._plain_problem(res)
            if problem:
                html += f"<p style='color:#92400E; margin:4px 0'>{_esc(problem)}</p>"
            blocks.append(html)
        return "".join(blocks)

    def _update_result_labels(self) -> None:
        shown = dict(self._panels())
        for code, card in self.cards.items():
            card.setVisible(code in shown)
            if code in shown:
                card.set_result(code, shown[code])

    def _panels(self) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        method = self.cmb_method.currentText()
        if method == METHOD_COMPARE:
            return [("N", self.number_result), ("M", self.mass_result)]
        if method == METHOD_NUMBER:
            if self.mass_is_fallback:
                return [("M", self.mass_result)]
            return [("N", self.number_result)]
        return [("M", self.mass_result)]

    @staticmethod
    def _axis_spec(code: str) -> Dict[str, str]:
        if code == "N":
            return {
                "x": "log10_r", "y": "log10_N_gt_r", "size": "threshold_r_mm", "count": "N_gt_r",
                "xlabel": "Particle size (mm)", "ylabel": "Number of particles larger than this size",
                "title": "Particle count", "color": UI_ACCENT["N"],
            }
        return {
            "x": "log10_R", "y": "log10_mass_fraction", "size": "R_mm", "count": "mass_fraction",
            "xlabel": "Particle size (mm)", "ylabel": "Mass finer than this size (%)",
            "title": "Mass", "color": UI_ACCENT["M"],
        }

    def _draw_results(self) -> None:
        self.figure.clear()
        self._pick_targets = {}
        panels = self._panels()
        view = self.cmb_view.currentText()
        for i, (code, res) in enumerate(panels):
            ax = self.figure.add_subplot(1, len(panels), i + 1)
            if view == VIEW_LINEAR:
                self._draw_linear(ax, code, res)
            elif view == VIEW_RESIDUALS:
                self._draw_residuals(ax, code, res)
            else:
                self._draw_loglog(ax, code, res)
            if len(panels) > 1:
                ax.set_title(self._axis_spec(code)["title"], loc="left", fontsize=12, fontweight="bold",
                             color="#111827", pad=10)
        try:
            self.figure.tight_layout(pad=1.6)
        except Exception:
            pass
        self.canvas.draw_idle()

    @staticmethod
    def _split_points(res: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        valid = [r for r in res.get("rows", []) if r.get("valid")]
        return [r for r in valid if r.get("use")], [r for r in valid if not r.get("use")]

    @staticmethod
    def _no_data(ax, text: str) -> None:
        _axes_message(ax, text)

    def _plot_points(self, ax, code: str, used: List[Dict[str, Any]], excluded: List[Dict[str, Any]],
                     xkey: str, ykey: str, scale: float) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Large points; hollow grey points are left out of the fit. Both can be clicked."""
        xu = [float(r[xkey]) for r in used]
        yu = [float(r[ykey]) * scale for r in used]
        xe = [float(r[xkey]) for r in excluded]
        ye = [float(r[ykey]) * scale for r in excluded]
        if excluded:
            art = ax.scatter(xe, ye, s=85, facecolors="white", edgecolors=UI_MUTED, linewidths=1.8, zorder=3,
                             label="Not used in the fit", picker=True)
            art.set_pickradius(7)
            self._pick_targets[art] = (code, [float(r["phi"]) for r in excluded])
        if used:
            art = ax.scatter(xu, yu, s=115, color=UI_ACCENT[code], edgecolors="white", linewidths=1.5, zorder=4,
                             label="Used in the fit", picker=True)
            art.set_pickradius(7)
            self._pick_targets[art] = (code, [float(r["phi"]) for r in used])
        return xu, yu, xe, ye

    @staticmethod
    def _plot_segments(ax, seg: Dict[str, Any], used: List[Dict[str, Any]], spec: Dict[str, str], scale: float,
                       log: bool) -> None:
        bp = float(seg["break_phi"])
        for part in ("coarse", "fine"):
            sf = seg[part]["fit"]
            pts = [float(r[spec["size"]]) for r in used
                   if ((float(r["phi"]) <= bp + 1e-9) if part == "coarse" else (float(r["phi"]) >= bp - 1e-9))]
            if len(pts) >= 2:
                xx = np.geomspace(min(pts), max(pts), 60) if log else np.linspace(min(pts), max(pts), 120)
                ax.plot(xx, scale * (10.0 ** sf["intercept"]) * xx ** sf["slope"], color=UI_SEGMENT_COLORS[part],
                        linewidth=2.8, zorder=5, solid_capstyle="round", label=f"{part.capitalize()} segment")
        xb = next((float(r[spec["size"]]) for r in used if abs(float(r["phi"]) - bp) < 1e-9), None)
        if xb is not None:
            ax.axvline(xb, color=UI_MUTED, linestyle=":", linewidth=1.3, zorder=1)

    def _draw_loglog(self, ax, code: str, res: Optional[Dict[str, Any]]) -> None:
        spec = self._axis_spec(code)
        _style_axes(ax)
        if not res or not res.get("rows"):
            self._no_data(ax, "No data for this selection")
            return
        used, excluded = self._split_points(res)
        if not used and not excluded:
            self._no_data(ax, "No valid points for this selection")
            return
        scale = 100.0 if code == "M" else 1.0
        xu, yu, xe, ye = self._plot_points(ax, code, used, excluded, spec["size"], spec["count"], scale)
        fit = res.get("fit")
        seg = res.get("segments")
        if fit and len(xu) >= 2:
            xx = np.geomspace(min(xu), max(xu), 120)
            yy = scale * (10.0 ** fit["intercept"]) * xx ** fit["slope"]
            if seg:
                ax.plot(xx, yy, color=UI_MUTED, linewidth=1.4, linestyle="--", zorder=2, label="One line")
            else:
                ax.plot(xx, yy, color=UI_FIT_LINE, linewidth=2.6, alpha=0.9, zorder=2, solid_capstyle="round",
                        label="Fit line")
        if seg and used:
            self._plot_segments(ax, seg, used, spec, scale, log=True)
        _set_size_axis(ax, xu + xe, log=True)
        _set_value_axis(ax, yu + ye, percent=(code == "M"), log=True)
        _axis_labels(ax, spec["xlabel"], spec["ylabel"])
        _draw_d_label(ax, code, res, "right" if code == "N" else "left")
        _legend(ax, "lower left" if code == "N" else "lower right")

    def _draw_linear(self, ax, code: str, res: Optional[Dict[str, Any]]) -> None:
        spec = self._axis_spec(code)
        _style_axes(ax)
        if not res or not res.get("rows"):
            self._no_data(ax, "No data for this selection")
            return
        used, excluded = self._split_points(res)
        if not used and not excluded:
            self._no_data(ax, "No valid points for this selection")
            return
        scale = 100.0 if code == "M" else 1.0
        xu, yu, xe, ye = self._plot_points(ax, code, used, excluded, spec["size"], spec["count"], scale)
        fit = res.get("fit")
        if fit and len(xu) >= 2:
            xx = np.linspace(min(xu), max(xu), 240)
            ax.plot(xx, scale * (10.0 ** fit["intercept"]) * xx ** fit["slope"], color=UI_FIT_LINE, linewidth=2.6,
                    alpha=0.9, zorder=2, label="Fit curve")
        _set_size_axis(ax, xu + xe, log=False)
        _set_value_axis(ax, yu + ye, percent=(code == "M"), log=False)
        _axis_labels(ax, spec["xlabel"], spec["ylabel"])
        _draw_d_label(ax, code, res, "right" if code == "N" else "left")
        _legend(ax, "center right" if code == "N" else "lower right")

    def _draw_residuals(self, ax, code: str, res: Optional[Dict[str, Any]]) -> None:
        spec = self._axis_spec(code)
        _style_axes(ax)
        fit = (res or {}).get("fit")
        if not res or not fit:
            self._no_data(ax, "No fit for this selection")
            return
        used, _ = self._split_points(res)
        if not used:
            self._no_data(ax, "No points in the fit")
            return
        sizes = [float(r[spec["size"]]) for r in used]
        xs = np.array([float(r[spec["x"]]) for r in used], dtype=float)
        ys = np.array([float(r[spec["y"]]) for r in used], dtype=float)
        resid = ys - (fit["slope"] * xs + fit["intercept"])
        ax.axhline(0.0, color=UI_FIT_LINE, linewidth=1.6, alpha=0.85, zorder=2)
        ax.plot(sizes, resid, "-", color="#D1D5DB", linewidth=1.4, zorder=2)
        ax.scatter(sizes, resid, s=115, color=UI_ACCENT[code], edgecolors="white", linewidths=1.5, zorder=4,
                   label="One line")
        seg = res.get("segments")
        if seg:
            bp = float(seg["break_phi"])
            seg_res = []
            for r in used:
                sf = (seg["coarse"] if float(r["phi"]) <= bp + 1e-9 else seg["fine"])["fit"]
                seg_res.append(float(r[spec["y"]]) - (sf["slope"] * float(r[spec["x"]]) + sf["intercept"]))
            ax.scatter(sizes, seg_res, s=70, marker="s", facecolors="white", edgecolors=UI_SEGMENT_COLORS["coarse"],
                       linewidths=1.6, zorder=5, label="Two segments")
        _set_size_axis(ax, sizes, log=True)
        _axis_labels(ax, spec["xlabel"], "Distance from the line (log units)")
        ax.annotate("A curved pattern means the points bend (a kink).", xy=(0.02, 0.03), xycoords="axes fraction",
                    fontsize=9.5, color="#6B7280")
        _legend(ax, "upper right")

    def _on_chart_pick(self, event) -> None:
        target = self._pick_targets.get(event.artist)
        ind = getattr(event, "ind", None)
        if not target or ind is None or len(ind) == 0:
            return
        code, phis = target
        i = int(ind[0])
        if 0 <= i < len(phis):
            # Applied after the matplotlib event, because recalculating redraws the figure.
            self._pending_pick = (code, phis[i])
            QTimer.singleShot(0, self._apply_pending_pick)

    def _apply_pending_pick(self) -> None:
        pending, self._pending_pick = self._pending_pick, None
        db = self.current_db()
        if not pending or db is None:
            return
        code, phi = pending
        kind, value = self._population_spec()
        pk = population_key(kind, value)
        set_fit_use_for_phi(db, code, pk, phi, not fit_use_for_phi(db, code, pk, phi))
        self.settings_changed(recalculate=True)

    def _apply_column_visibility(self) -> None:
        show_all = self.chk_all_columns.isChecked()
        for tbl, main in ((self.tbl_number, _NUMBER_MAIN_COLUMNS), (self.tbl_mass, _MASS_MAIN_COLUMNS)):
            for j in range(tbl.columnCount()):
                tbl.setColumnHidden(j, (not show_all) and j not in main)

    # -------------------------------------------------------------------------
    # Data tables
    # -------------------------------------------------------------------------
    def _fill_number_table(self, result: Optional[Dict[str, Any]]) -> None:
        headers = [
            "Use", "φ", "Status", "Size r (mm)", "Class upper limit (mm)", "Component",
            "Mass (wt%)", "Mass (g)", "B (mm)", "Density (g/cm³)", "Shape", "A/B", "C/B", "A (mm)", "C (mm)",
            "Particle volume (mm³)", "Particle volume (cm³)", "Particle mass (g)",
            "Particles in class", "Particles in class (all components)", "Particles larger than r",
            "log10 r", "log10 N(≥r)", "Shape volume factor", "N/N sphere", "Model status",
        ]
        self.tbl_number.blockSignals(True)
        self.tbl_number.clear()
        self.tbl_number.setColumnCount(len(headers)); self.tbl_number.setHorizontalHeaderLabels(headers)
        rows: List[Dict[str, Any]] = []
        if result:
            details: Dict[float, List[Dict[str, Any]]] = {}
            for d in result.get("detail_rows", []):
                details.setdefault(float(d["phi"]), []).append(d)
            for a in result.get("rows", []):
                ds = details.get(float(a["phi"]))
                if ds:
                    for d in ds:
                        rows.append({**d, "status": a.get("status", "OK")})
                else:
                    rows.append({**a, "component": "", "model_error": ""})
        self.tbl_number.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.tbl_number.setItem(i, 0, _use_item(r.get("phi", NAN), bool(r.get("use")), bool(r.get("valid"))))
            density_txt = _number_text(r.get("density_g_cm3"), 8)
            if r.get("default_density"):
                density_txt += " (default)"
            vals = [
                f"{float(r.get('phi', NAN)):g}" if np.isfinite(_safe_float(r.get('phi'), NAN)) else "",
                str(r.get("status", "")),
                _number_text(r.get("threshold_r_mm"), 8), _number_text(r.get("upper_mm"), 8), str(r.get("component", "")),
                _number_text(r.get("component_mass_wt"), 8), _number_text(r.get("component_mass_g"), 8),
                _number_text(r.get("B_mm"), 8), density_txt, str(r.get("shape", "")),
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
        self._apply_column_visibility()
        self.tbl_number.blockSignals(False)

    def _fill_mass_table(self, result: Optional[Dict[str, Any]]) -> None:
        headers = [
            "Use", "φ", "Status", "Class lower limit (mm)", "Size R (mm)", "Class mass (wt%)",
            "Mass finer than R (wt%)", "M(r<R)/M_T", "log10 R", "log10 [M(r<R)/M_T]",
        ]
        self.tbl_mass.blockSignals(True)
        self.tbl_mass.clear(); self.tbl_mass.setColumnCount(len(headers)); self.tbl_mass.setHorizontalHeaderLabels(headers)
        rows = list((result or {}).get("rows", []))
        self.tbl_mass.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self.tbl_mass.setItem(i, 0, _use_item(r["phi"], bool(r.get("use")), bool(r.get("valid"))))
            vals = [
                f"{r['phi']:g}", str(r.get("status", "")), _number_text(r.get("lower_mm"), 8), _number_text(r.get("R_mm"), 8),
                _number_text(r.get("bin_mass"), 8), _number_text(r.get("cumulative_mass_finer"), 8),
                _number_text(r.get("mass_fraction"), 8), _number_text(r.get("log10_R"), 8),
                _number_text(r.get("log10_mass_fraction"), 8),
            ]
            for j, val in enumerate(vals, start=1):
                self.tbl_mass.setItem(i, j, _read_only_item(val))
        self.tbl_mass.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_mass.horizontalHeader().setStretchLastSection(True)
        self._apply_column_visibility()
        self.tbl_mass.blockSignals(False)

    # -------------------------------------------------------------------------
    # Method information and tests
    # -------------------------------------------------------------------------
    def show_method_information(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Fractal analysis — about the method")
        dlg.resize(820, 720)
        lay = QVBoxLayout(dlg)
        txt = QTextEdit(); txt.setReadOnly(True)
        txt.setPlainText(
            "FRACTAL FRAGMENTATION ANALYSIS\n"
            "Main reference: Perugini et al. (2011), J. Volcanol. Geotherm. Res. 202, 200–210.\n\n"
            "SIZE CLASSES\n"
            "Each stored φ is the sieve that retains the class. The class holds the particles between the aperture of that "
            "sieve, 2^-φ mm (lower limit), and the aperture of the next coarser sieve (upper limit). FW bins laser data the "
            "same way (φ in (X − step, X] is stored as X). If the finest class is a pan, exclude it from the fit.\n\n"
            "NUMBER-BASED METHOD (Perugini et al. 2011, Eqs. 2–3; Kueppers et al. 2006)\n"
            "N(≥r) ∝ r^-D        log10 N(≥r) = −D log10 r + b        D = −slope\n"
            "r is the sieve aperture. N(≥r) is the cumulative number of particles retained on that sieve and all coarser ones.\n"
            "Particles in each class: n = M / (ρ V), V = (π/6) A B C (triaxial ellipsoid; sphere by default).\n"
            "B: sieve aperture (Perugini et al. 2011, default), geometric class midpoint √(lower × upper) "
            "(Wiejaczka & Giachetti 2022) or manual. With constant φ steps this choice does not change D.\n"
            f"Sieving only (no componentry): one density and one shape for all fractions (default {DEFAULT_BULK_DENSITY:g} g/cm³, sphere). "
            "With uniform density and shape their values do not change D (Perugini et al. 2011).\n"
            "With componentry: an approximate density and shape is needed for every component of the analysed population "
            f"(and for the '{UNCLASSIFIED_LABEL}' mass of the whole sample). If a density is missing, the number-based D is not "
            "calculated and the mass-based D is shown.\n\n"
            "MASS-BASED METHOD (Perugini & Kueppers 2012, Eqs. 4–9; Paredes-Mariño et al. 2017)\n"
            "M(r<R)/M_T ∝ R^v        log10 [M(r<R)/M_T] = v log10 R + b        D = 3 − v\n"
            "R is a sieve aperture and M(r<R) the mass that passes it: for each class, R is the upper limit of the class and "
            "M(r<R) the mass of that class and all finer classes. No density or shape is needed. Absolute D values differ "
            "between both methods (Perugini & Kueppers 2012): use the same method for all samples.\n\n"
            "POPULATIONS\n"
            "Whole sample, all components of one type (for example all juveniles or all lithics) or one component. Component "
            "and type populations use only the classified fractions; the other fractions are excluded from the fit.\n\n"
            "UNCERTAINTY AND FIT CHECK\n"
            "SE is the standard error of the slope (= SE of D); the 95% interval uses Student's t with n − 2 degrees of "
            f"freedom. A straight line is accepted when R² > {FIT_R2_MIN:g} and SE ≤ {FIT_SE_MAX:g} (Paredes-Mariño et al. 2017). "
            "Perugini et al. (2011) report R² ≈ 0.99 and errors of D below 3%.\n\n"
            "LINEARITY AND KINKS\n"
            "A single straight line indicates a single fragmentation event. A kink indicates secondary processes (wind "
            "dispersal, transport, secondary fragmentation; Perugini et al. 2011) or two scaling regimes (Paredes-Mariño "
            "et al. 2017; Wiejaczka & Giachetti 2022). The residuals view shows curvature; the two-segment fit gives D on "
            "each side of the break (the break point belongs to both segments). 'Suggest break' minimises the total squared "
            "error with at least 3 points per segment.\n\n"
            "FRAGMENTATION PROBABILITY (Perugini et al. 2011, Eq. 19)\n"
            "D = 3 ln(8p) / ln 8    →    p = 2^D / 8\n"
            "Renormalization group model (Turcotte 1986): p is the probability that a cell fragments at each order; it is "
            "defined for 0 ≤ D ≤ 3 (p from 1/8 to 1) and is a proxy of the 'fragility' of the material.\n\n"
            "PERUGINI 2011 RANGE\n"
            "Classes retained on the 5.66 to 0.25 mm sieves (φ −2.5 to 2), the range used by Perugini et al. (2011) to "
            "compare with Kueppers et al. (2006).\n\n"
            "REGRESSION\n"
            "Ordinary least squares with free intercept. Points can be included or excluded individually or by φ range; "
            "excluded points remain visible in the graph as hollow grey circles. Click a point in the chart to include "
            "or exclude it."
        )
        lay.addWidget(txt)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        btn_tests = bb.addButton("Run numeric self-tests", QDialogButtonBox.ActionRole)
        btn_tests.clicked.connect(self.show_self_tests)
        bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec_()

    def show_self_tests(self) -> None:
        try:
            results = run_numeric_self_tests()
        except Exception:
            QMessageBox.warning(self, "Fractal numeric self-tests", traceback.format_exc())
            return
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
        db = self.current_db()
        rows: List[Dict[str, Any]] = []
        base = {
            "sample": self.current_sample_name(),
            "population": population_label(kind, value),
            "population_key": population_key(kind, value),
            "representative_size": ensure_fractal_settings(db).get("representative_size_mode") if db is not None else "",
            "componentry": bool(has_componentry(db)) if db is not None else False,
        }
        for name, res in (("Number-based", self.number_result), ("Mass-based", self.mass_result)):
            if not res:
                continue
            fit = res.get("fit") or {}
            lo, hi = _fit_phi_range(res)
            row = {
                **base,
                "method": name + (" (shown because the number-based D lacks densities)" if (name == "Mass-based" and self.mass_is_fallback) else ""),
                "D": res.get("D"),
                "SE_D": res.get("D_se"),
                "CI95_D": res.get("D_ci95"),
                "slope": fit.get("slope"),
                "intercept": fit.get("intercept"),
                "R2": fit.get("r2"),
                "n": fit.get("n"),
                "p": res.get("p"),
                "SE_p": res.get("p_se"),
                "fit_check": res.get("check"),
                "fit_phi_from": lo,
                "fit_phi_to": hi,
            }
            if name == "Number-based":
                row["componentry_coverage_percent"] = res.get("componentry_coverage")
                row["modeled_coverage_percent"] = res.get("modeled_coverage")
            seg = res.get("segments")
            if seg:
                for part in ("coarse", "fine"):
                    s = seg[part]
                    row.update({
                        f"D_{part}": s["D"], f"SE_D_{part}": s["D_se"], f"R2_{part}": s["fit"]["r2"],
                        f"n_{part}": s["n"], f"p_{part}": s["p"],
                    })
                row["break_phi"] = seg["break_phi"]
            rows.append(row)
        if self.number_result and self.mass_result:
            dn = _safe_float(self.number_result.get("D"), NAN); dm = _safe_float(self.mass_result.get("D"), NAN)
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
        bounds = class_bounds(db, phis)
        componentry = has_componentry(db)
        out: List[Dict[str, Any]] = []
        for comp in model_component_keys(db):
            for phi in phis:
                p = effective_component_params(db, comp, phi, phis, bounds, componentry)
                out.append({
                    "sample": self.current_sample_name(), "component": component_display_name(db, comp, componentry), "phi": phi,
                    "class_lower_mm": bounds[float(phi)]["lower_mm"], "class_upper_mm": bounds[float(phi)]["upper_mm"],
                    "component_mass_wt": component_mass_wt(db, comp, phi),
                    "B_mm": p.get("b_mm"), "density_g_cm3": p.get("density"), "default_density": p.get("default_density"),
                    "shape": p.get("shape"), "A_over_B": p.get("alpha"), "C_over_B": p.get("gamma"),
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

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
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
    "fragmentation_probability",
    "linear_regression",
    "two_segment_fit",
    "suggest_break_phi",
    "class_bounds",
    "calculate_number_method",
    "calculate_mass_method",
    "run_numeric_self_tests",
    "ensure_fractal_settings",
    "install_no_wheel_filter",
]
