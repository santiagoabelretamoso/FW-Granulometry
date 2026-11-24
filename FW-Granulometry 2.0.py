# region # === Bloque 0: Localization (EN-only) ===
"""
EN-only localization infrastructure.
- Does not touch program logic.
- Provides utility functions for UI texts and parameter labels.
- Integrated via `tr(...)`, `param_label(...)`, etc.
Place this block above Block 1.
"""

# Supported language (frozen)
LANGS = ("en",)
DEFAULT_LANG = "en"
_GLOBAL_LANG = DEFAULT_LANG  # fallback if qApp is not available yet


def _current_lang(lang=None):
    """Always return the active language (frozen to English)."""
    return "en"


def set_app_language(lang: str):
    """No-op: language is frozen to English (kept for compatibility)."""
    global _GLOBAL_LANG
    _GLOBAL_LANG = "en"
    try:
        qApp.lang = "en"  # type: ignore[name-defined]
    except Exception:
        pass


# === Canonical internal dataset keys + robust aliases ===
# We always use these INTERNAL KEYS everywhere in the code:
BASE_KEYS = ("Tamiz", "Laser", "Híbrido")

# Accepted aliases (with/without accents, ES/EN). All map to internal keys.
BASE_ALIASES = {
    # Tamiz
    "tamiz": "Tamiz",
    "sieve": "Tamiz",
    # Laser
    "laser": "Laser",
    "láser": "Laser",
    # Híbrido / Hybrid
    "hibrido": "Híbrido",
    "híbrido": "Híbrido",
    "hybrid": "Híbrido",
}


def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def base_key_canon(base_key: str) -> str:
    """
    Return canonical internal key ('Tamiz'|'Laser'|'Híbrido') from any alias.
    If it cannot normalize, return the original input.
    """
    if not isinstance(base_key, str):
        return base_key
    s = base_key.strip()
    if s in BASE_KEYS:
        return s
    s_low = _strip_accents(s).lower()
    return BASE_ALIASES.get(s_low, s)


# --- Symbols (fallback) ---
# NOTE: Block 2 defines SYMBOLS; here we only provide a fallback to avoid ordering dependencies.
def _sym(name: str) -> str:
    try:
        return SYMBOLS[name]  # type: ignore[name-defined]
    except Exception:
        _fallback = {
            "median":   "Mdφ",
            "mean":     "Mφ",
            "sigma":    "σφ",
            "skewness": "Skφ",
            "kurtosis": "Kφ",
        }
        return _fallback[name]


# --- Parameter labels (with symbol in parentheses) ---
def param_label(key: str, lang: str = None) -> str:
    """
    key ∈ {"median","mean","sigma","skewness","kurtosis"}
    Return EN label with symbol between parentheses.
    """
    s = _sym(key)
    mapping = {
        "median":   f"Median ({s})",
        "mean":     f"Mean ({s})",
        "sigma":    f"Sorting ({s})",
        "skewness": f"Skewness ({s})",
        "kurtosis": f"Kurtosis ({s})",
    }
    return mapping[key]


def param_option_labels(lang: str = None):
    """
    Return list of (key,label) pairs to populate X/Y combos:
      [("mean","Mean (Mφ)"), ("median","Median (Mdφ)"), ...]
    """
    keys = ["mean", "median", "sigma", "skewness", "kurtosis"]
    return [(k, param_label(k, "en")) for k in keys]


# --- Dataset display names (EN-only, using internal keys) ---
BASE_DISPLAY = {
    "Tamiz": "Sieve",
    "Laser": "Laser",
    "Híbrido": "Hybrid",
}


def base_name_display(base_key: str, lang: str = None) -> str:
    """Map any alias -> internal key -> EN display name."""
    key = base_key_canon(base_key)
    return BASE_DISPLAY.get(key, key)


def legend_group_base(lang: str = None) -> str:
    """Legend title."""
    return "Group (Base)"


# --- UI strings dictionary (EN-only) ---
I18N = {
    # App / Tabs
    "app_title": "FW-Granulometry 2.0",
    "tab_xy": "Granulometric parameters",
    "xy_plot_title": "Granulometric parameters plot",
    "tab_hist": "Histogram",

    # Menus
    "menu_file": "&File",
    "menu_config": "&Settings",
    "menu_db": "Database",
    "menu_help": "&Help",

    # File
    "file_load_sieve": "Load sieve Excel file",
    "file_load_laser": "Load laser Excel file",
    "file_export_image": "Export current image",
    "file_combine": "Combine sieve and laser",

    # Settings
    "cfg_method": "Computation method",
    "cfg_colors": "Group colors",
    "cfg_groups_all": "Visible groups (all tabs)",
    "cfg_ribbons": "Ribbons (XY)",
    "cfg_theme": "UI theme",

    # Database
    "db_view_sieve": "View sieve database",
    "db_view_laser": "View laser database",
    "db_view_hybrid": "View sieve and laser database",

    # Help
    "help_about": "About...",
    "about_title": "About",
    "about_text": (
        "Free and open-source software for granulometric analysis of pyroclastic deposits.\n\n"
        "More info: santiagoabelretamoso@gmail.com\n\n"
        "Folk and Ward Granulometry or\nFW-Granulometry - Python"
    ),

    # XY tab controls
    "btn_load_sieve": "Load sieve Excel file",
    "lbl_x": "X axis:",
    "lbl_y": "Y axis:",
    "btn_plot_xy": "Plot XY",
    "btn_choose_db": "Choose databases to plot",
    "chk_show_names": "Show names",
    "btn_filter_groups": "Groups",
    "btn_view_plot": "View plot",
    "btn_export_image": "Export image",
    "console_header": "Granulometric parameters",
    "btn_export_params": "Export granulometric parameters",
    "params_copy_table": "Copy table",
    "params_copied": "Copied",
    "params_filter_placeholder": "Filter by sample or group",
    "params_empty_title": "No granulometric parameters yet",
    "params_empty_text": "Load a sieve or laser Excel file to compute them. Each dataset (Sieve, Laser, Hybrid) gets its own table here.",

    # Histogram tab controls
    "hist_db": "Database:",
    "hist_sample": "Sample:",
    "hist_prev": "Previous",
    "hist_next": "Next",
    "hist_show_cumulative": "Cumulative curve",
    "hist_show_polygon": "Frequency polygon",
    "hist_show_um_axis": "μm axis",
    "hist_reverse_x": "Reverse X",
    "hist_show_params": "Granulometric parameters",
    "hist_show_modified_overlay": "Modified bins overlay",
    "hist_show_laser_curve": "Laser curve",
    "hist_show_mean_marker": "Mean",
    "hist_show_median_marker": "Median",
    "hist_show_sigma_marker": "Standard deviation",
    "hist_show_mode_marker": "Mode(s)",
    "hist_edit": "Edit histogram",
    "hist_export": "Export image(s)",
    "hist_export_current": "Export current histogram",
    "hist_export_batch": "Export multiple histograms",
    "hist_no_data": "No histogram data available for this sample.",
    "hist_load_prompt": "Load a dataset and choose a sample.",

    # Histogram edit dialog
    "hist_style_title": "Edit histogram",
    "hist_bar_fill": "Bar fill color",
    "hist_bar_edge": "Bar edge color",
    "hist_bar_width": "Bar width factor",
    "hist_bar_edge_width": "Bar edge width",
    "hist_axes_color": "Axes color",
    "hist_axes_width": "Axes line width",
    "hist_fig_width": "Figure width",
    "hist_fig_height": "Figure height",
    "hist_cum_color": "Cumulative curve color",
    "hist_cum_width": "Cumulative curve width",
    "hist_poly_color": "Frequency polygon color",
    "hist_poly_width": "Frequency polygon width",
    "hist_laser_color": "Hybrid laser-line color",
    "hist_laser_width": "Hybrid laser-line width",
    "hist_overlay_color": "Modified-bins overlay color",
    "hist_overlay_width": "Modified-bins overlay width",
    "hist_overlay_alpha": "Modified-bins overlay transparency",
    "hist_overlay_hatch": "Modified-bins hatch",
    "hist_x_axis_mode": "X axis mode",
    "hist_y_left_mode": "Left Y axis mode",
    "hist_y_right_mode": "Right Y axis mode",
    "hist_x_axis_phi": "Phi (φ)",
    "hist_x_axis_um": "Micrometers (μm)",
    "hist_y_left_wt": "%Wt",
    "hist_y_right_none": "None",
    "hist_y_right_cumulative": "Cumulative frequency",
    "hist_cum_layer": "Cumulative curve layer",
    "hist_cum_front": "Front",
    "hist_cum_back": "Back",
    "hist_um_axis_position": "μm axis position",
    "hist_um_axis_top": "Top",
    "hist_um_axis_bottom": "Bottom",
    "hist_x_label_phi_custom": "X label (phi mode)",
    "hist_x_label_um_custom": "X label (μm mode)",
    "hist_y_label_left_custom": "Left Y label",
    "hist_y_label_right_custom": "Right Y label",
    "hist_um_secondary_label_custom": "Secondary μm axis label",
    "hist_apply": "Apply",
    "hist_reset_defaults": "Reset defaults",
    "hist_font_missing": "STIX Two Math font was not found. Matplotlib will use a fallback font.",

    # Histogram export dialog
    "hist_export_title": "Export histogram(s)",
    "hist_export_select_db": "Databases:",
    "hist_export_select_samples": "Samples:",
    "hist_export_folder": "Output folder",
    "hist_export_choose_folder": "Choose folder",
    "hist_export_png": "PNG",
    "hist_export_svg": "SVG",
    "hist_export_pdf": "PDF",
    "hist_export_done": "Histogram export completed.",
    "hist_export_none": "No samples available for export.",
    "hist_export_need_folder": "Please choose an output folder.",

    # Generic dialogs
    "ok": "OK",
    "cancel": "Cancel",
    "close": "Close",
    "info": "Information",
    "error": "Error",
    "success": "Success",

    # Frequent texts
    "choose_theme_title": "UI theme",
    "theme_dark": "Dark (Fusion dark)",
    "theme_light": "Light (Fusion light)",
    "choose_method_title": "Choose computation method",
    "save_figure_title": "Save figure",
    "save_params_title": "Save parameters",

    # Match Sieve ↔ Laser
    "match_title": "Match Sieve and Laser samples",
    "match_sieve_samples": "Samples (Sieve)",
    "match_laser_samples": "Samples (Laser)",
    "match_pair_selection": "Pair selection",
    "match_col_origin": "Origin",
    "match_col_phi_star": "Extracted φ (φ*)",
    "match_need_11": "Please assign each sieve sample its matching laser sample.",
    "match_need_origin": "You must complete the 'Origin' field for every row.",
    "match_need_phi": "You must provide φ* (extracted phi) for every row.",

    # Hybrid method selector (M1/M2/M3)
    "hybrid_method_title": "Combine sieve and laser — choose method",
    "hybrid_m1_name": "METHOD 1: φ* + EDLT (current)",
    "hybrid_m2_name": "METHOD 2: SG (overlapped coarse)",
    "hybrid_m3_name": "METHOD 3 (coming soon)",
    "hybrid_m1_desc": "Use φ* and scale fines with k = EDLT/∑L(φ≥φ*); coarse sieve part remains intact.",
    "hybrid_m2_desc": "Fines as in M1 and transfer k·L(φ) to overlapping coarse (SG); renormalize without touching fixed coarse.",
    "hybrid_m3_desc": "Placeholder: algorithm to be defined.",
    "hybrid_continue": "Continue",
    "hybrid_select_required": "You must choose a method to continue.",
    "hybrid_not_implemented": "Method 3 is not implemented yet.",
    "load_both_required": "You must load Sieve and Laser data before combining.",
}


def tr(key: str, lang: str = None, **fmt) -> str:
    """
    Translate a key (EN-only). If missing, return the key itself.
    Supports formatting.
    """
    txt = I18N.get(key) or key
    try:
        return txt.format(**fmt) if fmt else txt
    except Exception:
        return txt

# endregion


#region BLOQUE 1 Imports
import sys
import os
import json
import sqlite3
import glob
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.path import Path as MplPath

from PyQt5.QtCore import (
    Qt, QObject, QEvent, QModelIndex, QAbstractTableModel, QSettings,
    QSize, QPoint, QRect
)
from PyQt5.QtGui import (
    QStandardItemModel, QStandardItem, QColor, QFont, QIcon
)
from PyQt5.QtWidgets import (
    QApplication, qApp, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QRadioButton, QButtonGroup,
    QTextEdit, QLineEdit, QTabWidget, QGroupBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QTableView, QHeaderView, QAbstractItemView,
    QListWidget, QListWidgetItem,
    QMessageBox, QColorDialog, QFileDialog, QDialogButtonBox,
    QSpinBox, QDoubleSpinBox, QStyledItemDelegate, QSizePolicy,
    QAction, QFrame, QToolBar, QDockWidget
)
#endregion





# region # === Bloque 2: Constants and dictionaries ===

# (SAFE DEBUG) Keep stylesheets as plain strings (no triple quotes).
DARK_STYLESHEET = (
    "QMainWindow { background-color: #232629; }\n"
    "QWidget { background-color: #232629; color: #edeef0; }\n"
    "QTabWidget::pane { border: 1px solid #444; background: #232629; }\n"
    "QTabBar::tab { background: #2b2e31; color: #edeef0; border-radius: 4px; padding: 6px; }\n"
    "QTabBar::tab:selected { background: #232629; color: #ffd60a; }\n"
    "QPushButton { background-color: #393e41; color: #edeef0; border: 1px solid #444; border-radius: 6px; padding: 5px 12px;}\n"
    "QPushButton:hover { background-color: #515457; }\n"
    "QLineEdit, QComboBox, QTextEdit { background-color: #2b2e31; color: #edeef0; border: 1px solid #444; }\n"
    "QGroupBox { border: 1px solid #444; margin-top: 10px; }\n"
    "QRadioButton, QCheckBox { color: #edeef0; }\n"
    "QMenuBar, QMenu, QMenu::item { background-color: #232629; color: #edeef0; }\n"
    "QMenu::item:selected { background-color: #515457; }\n"
    "QScrollBar { background: #232629; }\n"
    "QLabel { color: #edeef0; }\n"
)
LIGHT_STYLESHEET = (
    "QMainWindow, QWidget { background-color: #f4f6f8; color: #1f2937; }\n"
    "QTabWidget::pane { border: 1px solid #d9dee7; background: #ffffff; border-radius: 8px; }\n"
    "QTabBar::tab { background: #e9edf3; color: #334155; border: 1px solid #d9dee7; border-bottom: none; padding: 8px 14px; margin-right: 2px; border-top-left-radius: 8px; border-top-right-radius: 8px; min-width: 110px; }\n"
    "QTabBar::tab:selected { background: #ffffff; color: #0f172a; font-weight: 600; }\n"
    "QTabBar::tab:hover:!selected { background: #eef2f7; }\n"
    "QPushButton { background-color: #ffffff; color: #0f172a; border: 1px solid #cfd7e3; border-radius: 8px; padding: 6px 14px; min-height: 28px; }\n"
    "QPushButton:hover { background-color: #f8fafc; border-color: #b7c4d6; }\n"
    "QPushButton:pressed { background-color: #e9eef5; }\n"
    "QPushButton:disabled { background-color: #f2f4f7; color: #9aa4b2; border-color: #e2e8f0; }\n"
    "QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QTableView { background-color: #ffffff; color: #111827; border: 1px solid #cfd7e3; border-radius: 8px; padding: 4px 6px; selection-background-color: #dbeafe; selection-color: #111827; }\n"
    "QComboBox::drop-down { border: none; width: 24px; }\n"
    "QHeaderView::section { background-color: #eef2f7; color: #334155; border: 1px solid #d9dee7; padding: 6px; font-weight: 600; }\n"
    "QGroupBox { border: 1px solid #d9dee7; border-radius: 10px; margin-top: 14px; padding-top: 10px; background: #ffffff; }\n"
    "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #334155; }\n"
    "QMenuBar { background-color: #ffffff; color: #0f172a; border-bottom: 1px solid #d9dee7; }\n"
    "QMenuBar::item { background: transparent; padding: 6px 10px; }\n"
    "QMenuBar::item:selected { background: #eef2f7; border-radius: 6px; }\n"
    "QMenu { background-color: #ffffff; color: #0f172a; border: 1px solid #d9dee7; padding: 6px; }\n"
    "QMenu::item { padding: 7px 24px 7px 10px; border-radius: 6px; }\n"
    "QMenu::item:selected { background-color: #eef2f7; }\n"
    "QScrollArea { background: #f4f6f8; border: 1px solid #d9dee7; border-radius: 8px; }\n"
    "QCheckBox, QRadioButton, QLabel { color: #0f172a; }\n"
    "QToolTip { background-color: #111827; color: #ffffff; border: 1px solid #111827; padding: 4px; }\n"
)

# --- Candidate files for ICON and BANNER (branding) ---
ICON_CANDIDATES = [
    "icono.nuevo.png", "icon.png", "icono.png", "fw.png", "fw_icon.png",
    "app.png", "logo.png", "logo32.png", "logo64.png", "icon.ico"
]
# Additional patterns: search typical icons in the script folder
ICON_PATTERNS = ["*icon*.png", "*icono*.png", "*logo*.png", "*fw*.png", "*.ico"]

# Optional banner
BANNER_CANDIDATES = ["banner.png", "banner.jpg", "banner.jpeg", "logo.png", "fw.png"]

# --- Parameter symbols (language-independent). ---
SYMBOLS = {
    "mean": "Mφ",
    "median": "Mdφ",
    "sigma": "σφ",
    "skewness": "Skφ",
    "kurtosis": "Kφ"
}

# Methods available in the UI
METHODS = {
    "Folk & Ward (1957)": "folkward",
    "Inman (1952)": "inman",
    "Trask (1932)": "trask"
}

# Color palette for groups
GROUP_COLORS = [
    "#52b788",
    "#4895ef",
    "#e76f51",
    "#ffd60a",
    "#adb5bd",
    "#9b5de5",
    "#c1121f",
    "#1d3557",
    "#f15bb5",
    "#8d6e63",
    "#00b4d8",
    "#ff9f1c",
]

# If your project uses BASE_KEYS (defined in Block 0), keep these aliases:
BASES_VALIDAS = BASE_KEYS
BASES_XY_Y_AUTORES = BASE_KEYS

# --- Display options specific to Trask ---
TRASK_DISPLAY = "phi_equiv"  # "phi_equiv" or "raw"

# --- Histogram defaults ---
HIST_FONT_FAMILY = "STIX Two Math"

HIST_DEFAULT_STYLE = {
    # bars
    "bar_facecolor": "#8c8c8c",
    "bar_edgecolor": "#000000",
    "bar_width_factor": 0.92,
    "bar_edgewidth": 1.2,

    # axes / spines
    "axes_color": "#000000",
    "axes_linewidth": 2.0,

    # curves
    "cum_color": "#000000",
    "cum_linewidth": 1.8,

    "poly_color": "#000000",
    "poly_linewidth": 1.4,

    "laser_color": "#000000",
    "laser_linewidth": 1.1,

    # modified-bins overlay
    "overlay_color": "#000000",
    "overlay_linewidth": 1.0,
    "overlay_alpha": 0.20,
    "overlay_hatch": "////",

    # figure size (inches)
    "figure_width": 11.0,
    "figure_height": 7.0,

    # toggles
    "show_cumulative": True,
    "show_polygon": False,
    "show_um_axis": False,
    "reverse_x": False,
    "show_params_text": True,
    "show_modified_overlay": False,
    "show_laser_curve": False,
    "show_mean_marker": False,
    "show_median_marker": False,
    "show_sigma_marker": False,
    "show_mode_marker": False,

    # statistical markers style
    "stats_marker_color": "#808080",
    "stats_marker_alpha": 0.18,
    "stats_line_style": "--",
    "stats_linewidth": 1.0,

    # custom axis labels
    "x_label_phi": "",
    "x_label_um": "",
    "y_label_left": "",
    "y_label_right": "",
    "um_secondary_label": "",

    # placement / layer options
    "cumulative_layer": "front",   # "front" | "back"
    "um_axis_position": "top",     # "top" | "bottom"

    # axis modes
    "x_axis_mode": "phi",              # "phi" | "um"
    "y_left_mode": "wt",               # reserved for future expansion
    "y_right_mode": "cumulative",      # "cumulative" | "none"
}

# === Helpers for stable group key and UI label ===
def _norm_group_name(grp) -> str:
    try:
        s = str(grp).strip()
    except Exception:
        s = ""
    if not s or s.lower() in {"nan", "none", "null"}:
        return "No Group"
    return s


def _gkey(grp, base_key: str) -> str:
    # base_key must be canonical: 'Tamiz' | 'Laser' | 'Híbrido'
    return f"{_norm_group_name(grp)} ({base_key})"


def _glabel(grp, base_key: str) -> str:
    # UI label (base display name is EN-only now)
    try:
        disp = base_name_display(base_key)
    except Exception:
        disp = base_key
    return f"{_norm_group_name(grp)} ({disp})"

# (_axis_label_for_key helper lives in Block 10; not here.)
# endregion


# region # === Bloque 3: Calculation helper functions ===

def cumulative_distribution(phi, weights):
    """
    Return cumulative distribution in % for the weights.
    Robust to zero weights and NaN. If total<=0, returns NaN for the whole series.
    Uses np.nancumsum to ignore NaN during accumulation.
    """
    w = np.asarray(weights, dtype=float)
    total = np.nansum(w)
    if not np.isfinite(total) or total <= 0:
        return np.full(w.shape, np.nan, dtype=float)
    return np.nancumsum(w) * 100.0 / float(total)


def interp_percentile(phi, cum, target):
    """
    Robust linear interpolation on the cumulative curve (in %).
    Handles NaN and edges; uses np.interp with non-decreasing cum.
    """
    x = np.asarray(phi, dtype=float)
    y = np.asarray(cum, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0 or y.size == 0:
        return np.nan

    order = np.argsort(y)
    y = y[order]
    x = x[order]

    if target <= y[0]:
        return x[0]
    if target >= y[-1]:
        return x[-1]

    return float(np.interp(target, y, x))


def calculate_parameters(phi, weights, method="folkward"):
    """
    Parameter calculation with CONSISTENT keys:
      returns dict: {"median","sigma","skewness","kurtosis","mean"}

    Methods:
      - "folkward" (Folk & Ward, 1957) -> percentiles 5-16-25-50-75-84-95
      - "inman"    (Inman, 1952)       -> 16-50-84
      - "trask"    (Trask, 1932)       -> So = sqrt(d75/d25) in DIAMETER:
                                          So = 2**((phi25 - phi75)/2)
                                          'sigma' defaults to (phi75-phi25)/2 (phi-equivalent).
                                          If TRASK_DISPLAY == "raw", 'sigma' = So (dimensionless).
    """
    phi = np.asarray(phi, dtype=float)
    weights = np.asarray(weights, dtype=float)

    m = np.isfinite(phi) & np.isfinite(weights)
    phi = phi[m]
    weights = weights[m]
    if phi.size == 0 or np.nansum(weights) <= 0:
        return {
            "median": np.nan,
            "sigma": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "mean": np.nan
        }

    order = np.argsort(phi)
    phi = phi[order]
    weights = weights[order]

    def _percs(*plist):
        cum = cumulative_distribution(phi, weights)
        if not np.isfinite(cum).any():
            return {p: np.nan for p in plist}
        return {p: interp_percentile(phi, cum, p) for p in plist}

    # --- Folk & Ward (alias "blatt") ---
    if method in ("folkward", "blatt"):
        cum = cumulative_distribution(phi, weights)
        if not np.isfinite(cum).any():
            return {
                "median": np.nan,
                "sigma": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "mean": np.nan
            }

        phi5 = interp_percentile(phi, cum, 5)
        phi16 = interp_percentile(phi, cum, 16)
        phi25 = interp_percentile(phi, cum, 25)
        phi50 = interp_percentile(phi, cum, 50)
        phi75 = interp_percentile(phi, cum, 75)
        phi84 = interp_percentile(phi, cum, 84)
        phi95 = interp_percentile(phi, cum, 95)

        percs = [phi5, phi16, phi25, phi50, phi75, phi84, phi95]
        if not all(np.isfinite(percs)):
            return {
                "median": np.nan,
                "sigma": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "mean": np.nan
            }

        median = phi50
        d8416 = (phi84 - phi16)
        d955 = (phi95 - phi5)
        d7525 = (phi75 - phi25)

        sigma = (d8416 / 4.0 if d8416 != 0 else np.nan) + (d955 / 6.6 if d955 != 0 else np.nan)

        term1 = ((phi16 + phi84 - 2 * phi50) / (2 * d8416)) if d8416 != 0 else np.nan
        term2 = ((phi5 + phi95 - 2 * phi50) / (2 * d955)) if d955 != 0 else np.nan
        _terms = []
        if np.isfinite(term1):
            _terms.append(term1)
        if np.isfinite(term2):
            _terms.append(term2)
        skewness = float(np.sum(_terms)) if _terms else np.nan

        kurtosis = (d955 / (2.44 * d7525)) if d7525 != 0 else np.nan
        mean = (phi16 + phi50 + phi84) / 3.0

        return {
            "median": float(median),
            "sigma": float(sigma),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "mean": float(mean)
        }

    # --- Inman (1952) ---
    elif method == "inman":
        p = _percs(16, 50, 84)
        phi16, phi50, phi84 = p[16], p[50], p[84]
        if not all(np.isfinite([phi16, phi50, phi84])):
            return {
                "median": np.nan,
                "sigma": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "mean": np.nan
            }
        d = phi84 - phi16
        sigma = d / 2 if d != 0 else np.nan
        skewness = (phi16 + phi84 - 2 * phi50) / d if d != 0 else np.nan
        mean = (phi16 + phi84) / 2
        return {
            "median": float(phi50),
            "sigma": float(sigma),
            "skewness": float(skewness),
            "kurtosis": np.nan,
            "mean": float(mean)
        }

    # --- Trask (1932) ---
    elif method == "trask":
        p = _percs(25, 50, 75, 16, 84)
        phi25, phi50, phi75 = p[25], p[50], p[75]
        phi16, phi84 = p[16], p[84]
        if not all(np.isfinite([phi25, phi50, phi75])):
            return {
                "median": np.nan,
                "sigma": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "mean": np.nan
            }

        so = float(2.0 ** ((phi25 - phi75) / 2.0))

        trask_display = globals().get("TRASK_DISPLAY", "phi_equiv")
        if trask_display == "raw":
            sigma = so
        else:
            sigma = (phi75 - phi25) / 2.0

        if all(np.isfinite([phi16, phi84])):
            mean = (phi16 + phi50 + phi84) / 3.0
        else:
            mean = phi50

        return {
            "median": float(phi50),
            "sigma": float(sigma),
            "skewness": np.nan,
            "kurtosis": np.nan,
            "mean": float(mean)
        }

    else:
        raise ValueError("Unknown method")


# === Additional helper (laser um->phi and bin aggregation) ===
def agrupar_laser_por_phi(df_laser_raw: pd.DataFrame, step: float = 1.0):
    """
    Expects a DataFrame with columns: ['Sample', 'Diameter (μm)', '1 (%)', 'Group'].
    Converts μm -> φ and quantizes to user grid:
      - step=1.0  -> centers at integers (..., -2, -1, 0, 1, ...)
      - step=0.5  -> centers at multiples of 0.5
    Then sums % by (Sample, phi_center).
    Returns list of dicts: [{'Sample':..., 'phi':..., 'Wt':...}, ...]
    """
    required = ["Sample", "Diameter (μm)", "1 (%)"]
    for col in required:
        if col not in df_laser_raw.columns:
            raise ValueError("Missing column '{}' in laser file".format(col))

    df = df_laser_raw.copy()

    d_um = pd.to_numeric(
        df["Diameter (μm)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    wt = pd.to_numeric(
        df["1 (%)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    samp = df["Sample"].astype(str)

    m = np.isfinite(d_um) & (d_um > 0) & np.isfinite(wt)
    d_um = d_um[m]
    wt = wt[m]
    samp = samp[m]

    phi = -np.log2(d_um / 1000.0)
    phi_center = np.round(phi / step) * step

    tmp = pd.DataFrame({"Sample": samp.values, "phi": phi_center.values, "Wt": wt.values})
    agg = tmp.groupby(["Sample", "phi"], as_index=False, sort=True)["Wt"].sum()

    out_rows = [
        {"Sample": r["Sample"], "phi": float(np.round(r["phi"], 6)), "Wt": float(r["Wt"])}
        for _, r in agg.iterrows()
        if np.isfinite(r["Wt"]) and r["Wt"] > 0
    ]
    out_rows.sort(key=lambda r: (r["Sample"], r["phi"]))
    return out_rows

# endregion


# region # === Bloque 4: Generic plotting functions ===
def plot_ribbon(ax, x, y, color, alpha=0.17, width=0.07, smooth=15):
    # Robust fallback: if SciPy is not available, draw a smooth ellipse
    try:
        from scipy.spatial import ConvexHull
        from scipy.interpolate import splprep, splev
        points = np.vstack([x, y]).T
        if len(points) <= 2:
            return
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        centroid = np.mean(hull_points, axis=0)
        expanded = centroid + (hull_points - centroid) * (1 + width)
        expanded = np.vstack([expanded, expanded[0]])
        tck, _ = splprep([expanded[:, 0], expanded[:, 1]], s=0, per=True)
        u_fine = np.linspace(0, 1, len(expanded) * smooth)
        xf, yf = splev(u_fine, tck)
        ax.fill(xf, yf, color=color, alpha=alpha, linewidth=0, zorder=1)
    except Exception:
        if len(x) < 3:
            return
        xm, ym = np.mean(x), np.mean(y)
        rx, ry = np.std(x) * 1.8, np.std(y) * 1.8
        t = np.linspace(0, 2 * np.pi, 200)
        ax.fill(xm + rx * np.cos(t), ym + ry * np.sin(t), alpha=alpha, color=color, zorder=1, linewidth=0)


def get_script_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))
# endregion

# region # === Block 5: Dialogs, table model, and visualization helpers (clean version, intuitive matching) ===
"""
Includes:
  - PandasModel (Qt model for DataFrame, read-only)
  - Global 2-decimal delegate for tables (auto-installed)
  - PhiChoiceCombo (discrete phi* selector)
  - MatchDialog (more intuitive Sieve ↔ Laser assignment)

CHANGES IN THIS VERSION:
  - Matching is now easier and more dynamic:
      * choose ONE sieve sample
      * select ONE or MANY laser samples
      * add them to the assignment table at once
  - phi* is NO LONGER free with step 0.1
  - phi* is discrete and controlled from a combo box
  - valid phi* values are taken from:
      1) laser phi bins if available
      2) otherwise from laser_step (0.5 or 1.0)
  - exact duplicate pairs are blocked:
      (same sieve, same laser)
  - same sieve with multiple lasers is allowed
  - same laser with multiple sieves is allowed

Note:
  Auxiliary duplicated dialogs live in Block 15:
    - GroupSelectDialog
    - ColorDialog
    - DataBaseWindow
    - FixedSizeFigureCanvas
"""

from PyQt5.QtCore import Qt, QAbstractTableModel, QEvent, QObject, QModelIndex
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QTableView, QTableWidget, QTableWidgetItem, QDoubleSpinBox,
    QMessageBox, QStyledItemDelegate, QHeaderView, QAbstractItemView,
    QListWidget, QListWidgetItem, QComboBox, QFileDialog, QDialogButtonBox, QWidget
)
import pandas as pd
import numpy as np
import os


# ---------- 5.0: Qt model for DataFrames (read-only) ----------
class PandasModel(QAbstractTableModel):
    """
    Simple read-only model to display a pandas.DataFrame in a QTableView.
    - Supports column sorting.
    - Stores DataFrame in self._df so other dialogs can read it.
    """
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.reset_index(drop=True).copy()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            val = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(val) else str(val)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return ""
        else:
            return str(section + 1)

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def sort(self, column, order=Qt.AscendingOrder):
        if column < 0 or column >= len(self._df.columns):
            return
        self.layoutAboutToBeChanged.emit()
        colname = self._df.columns[column]
        try:
            key = pd.to_numeric(self._df[colname].astype(str).str.replace(",", "."), errors="coerce")
        except Exception:
            key = self._df[colname].astype(str)
        ascending = (order == Qt.AscendingOrder)
        self._df = (
            self._df
            .assign(_key=key)
            .sort_values("_key", ascending=ascending, kind="mergesort")
            .drop(columns="_key")
            .reset_index(drop=True)
        )
        self.layoutChanged.emit()


# ---------- 5.1: UI numeric formatting — global 2-decimal delegate ----------
class _TwoDecimalsDelegate(QStyledItemDelegate):
    """Display any numeric value with 2 decimals (visual only)."""
    def displayText(self, value, locale):
        try:
            v = float(str(value).replace(",", "."))
            if np.isfinite(v):
                return f"{v:.2f}"
        except Exception:
            return str(value)
        return f"{v:.2f}"


class _TableFormatter(QObject):
    """Installs the delegate on every QTableWidget/QTableView that gets shown."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._delegate = _TwoDecimalsDelegate()

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.Show and isinstance(obj, (QTableWidget, QTableView)):
                obj.setItemDelegate(self._delegate)
        except Exception:
            pass
        return super().eventFilter(obj, event)


# Auto-install the filter app-wide
try:
    _app = QApplication.instance()
    if _app is not None and not getattr(_app, "_fw_two_decimals_installed", False):
        _fmt = _TableFormatter(_app)
        _app.installEventFilter(_fmt)
        _app._fw_two_decimals_installed = True
        _app._fw_two_decimals_delegate = _fmt._delegate
        _app._fw_table_formatter = _fmt
except Exception:
    pass


# ---------- 5.2: Discrete phi* selector ----------
class PhiChoiceCombo(QComboBox):
    """
    Combo box with DISCRETE valid phi* values only.

    Compatibility helpers:
      - value()   -> float selected value or INVALID_VALUE
      - minimum() -> INVALID_VALUE sentinel
    """
    INVALID_VALUE = -9999.0

    def __init__(self, phi_choices, parent=None):
        super().__init__(parent)
        self._phi_choices = [float(v) for v in phi_choices]
        self.addItem("—", self.INVALID_VALUE)
        for v in self._phi_choices:
            self.addItem(self._fmt(v), float(v))
        self.setCurrentIndex(0)

    def _fmt(self, v):
        v = float(v)
        if np.isclose(v, round(v), atol=1e-9):
            return f"{int(round(v))}"
        return f"{v:.2f}".rstrip("0").rstrip(".")

    def value(self):
        try:
            return float(self.currentData())
        except Exception:
            return float(self.INVALID_VALUE)

    def minimum(self):
        return float(self.INVALID_VALUE)

    def set_value(self, v):
        try:
            vf = float(v)
        except Exception:
            self.setCurrentIndex(0)
            return
        idx = self.findData(vf)
        if idx >= 0:
            self.setCurrentIndex(idx)
        else:
            self.setCurrentIndex(0)


# ---------- 5.3: Dialog to match Sieve ↔ Laser samples ----------
class MatchDialog(QDialog):
    """
    More intuitive window to match Sieve and Laser samples and set φ*.

    Workflow:
      1) choose ONE sieve sample
      2) select ONE or MANY laser samples
      3) choose discrete phi*
      4) add rows to the assignment table
      5) optionally edit phi* row by row
      6) press OK
    """
    def __init__(self, df_tamiz, df_laser, parent=None):
        super().__init__(parent)

        self.df_tamiz = df_tamiz
        self.df_laser = df_laser
        self._parent_ref = parent

        title = tr("match_title")
        btn_done_txt = tr("ok")
        col_left = base_name_display("Tamiz")
        col_right = base_name_display("Laser")
        col_phi_star = tr("match_col_phi_star")

        self.setWindowTitle(title)
        self.resize(1120, 620)

        self.sieve_samples = self._collect_sieve_samples(df_tamiz)
        self.laser_samples = self._collect_laser_samples(df_laser)
        self.phi_choices = self._build_phi_choices(df_laser)

        root = QVBoxLayout(self)

        # ======= Top controls =======
        top = QHBoxLayout()

        left_box = QVBoxLayout()
        left_box.addWidget(QLabel(f"{base_name_display('Tamiz')}:"))
        self.cmb_tamiz = QComboBox()
        self.cmb_tamiz.addItems(self.sieve_samples)
        left_box.addWidget(self.cmb_tamiz)
        top.addLayout(left_box, 2)

        mid_box = QVBoxLayout()
        mid_box.addWidget(QLabel(f"{base_name_display('Laser')} (multi-select):"))
        self.lst_laser = QListWidget()
        self.lst_laser.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for s in self.laser_samples:
            self.lst_laser.addItem(QListWidgetItem(s))
        mid_box.addWidget(self.lst_laser)
        top.addLayout(mid_box, 3)

        right_box = QVBoxLayout()
        right_box.addWidget(QLabel("φ* (discrete valid values):"))
        self.cmb_phi = PhiChoiceCombo(self.phi_choices)
        right_box.addWidget(self.cmb_phi)

        self.btn_add = QPushButton("Add selected laser(s)")
        self.btn_add.clicked.connect(self.add_selected_pairs)
        right_box.addWidget(self.btn_add)

        self.btn_add_single = QPushButton("Add current laser only")
        self.btn_add_single.clicked.connect(self.add_current_laser_only)
        right_box.addWidget(self.btn_add_single)

        self.btn_delete = QPushButton("Delete row")
        self.btn_delete.clicked.connect(self.delete_current_row)
        right_box.addWidget(self.btn_delete)

        right_box.addStretch(1)
        top.addLayout(right_box, 2)

        root.addLayout(top)

        # ======= Assignment table =======
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([col_left, col_right, col_phi_star])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        try:
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        except Exception:
            pass
        root.addWidget(self.table)

        # ======= Hint =======
        self.lbl_hint = QLabel(
            "phi* is restricted to valid discrete values according to the laser binning "
            "(0.5 φ or 1.0 φ). Values such as 5.1, 5.2, 5.3 are intentionally forbidden."
        )
        self.lbl_hint.setWordWrap(True)
        root.addWidget(self.lbl_hint)

        # ======= Bottom buttons =======
        bottom = QHBoxLayout()
        bottom.addStretch()
        self.btn_ok = QPushButton(btn_done_txt)
        self.btn_ok.clicked.connect(self.accept)
        bottom.addWidget(self.btn_ok)
        root.addLayout(bottom)

    # ----- data preparation -----
    def _collect_sieve_samples(self, df_tamiz):
        try:
            return sorted(df_tamiz.iloc[:, 1].drop_duplicates().astype(str).tolist())
        except Exception:
            return []

    def _collect_laser_samples(self, df_laser):
        try:
            return sorted(df_laser["Sample"].drop_duplicates().astype(str).tolist())
        except Exception:
            return []

    def _infer_step_from_parent_or_data(self, df_laser):
        # 1) explicit parent setting
        try:
            step = getattr(self._parent_ref, "laser_step", None)
            if step is not None:
                step = float(step)
                if np.isclose(step, 0.5, atol=1e-9):
                    return 0.5
                if np.isclose(step, 1.0, atol=1e-9):
                    return 1.0
        except Exception:
            pass

        # 2) infer from existing laser phi bins
        try:
            if "phi" in df_laser.columns:
                vals = pd.to_numeric(df_laser["phi"], errors="coerce").dropna().astype(float).unique()
                vals = np.array(sorted(vals), dtype=float)
                if vals.size >= 2:
                    diffs = np.diff(vals)
                    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                    if diffs.size > 0:
                        step = float(np.min(diffs))
                        if np.isclose(step, 0.5, atol=1e-9):
                            return 0.5
                        if np.isclose(step, 1.0, atol=1e-9):
                            return 1.0
        except Exception:
            pass

        # fallback
        return 0.5

    def _build_phi_choices(self, df_laser):
        """
        Build the valid discrete phi* choices.

        Priority:
          1) actual laser phi bins if present
          2) generated range using inferred laser step
        """
        # 1) exact phi bins from laser data
        try:
            if "phi" in df_laser.columns:
                vals = pd.to_numeric(df_laser["phi"], errors="coerce").dropna().astype(float).unique()
                vals = np.array(sorted(vals), dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    return [float(np.round(v, 6)) for v in vals.tolist()]
        except Exception:
            pass

        # 2) fallback range based on step
        step = self._infer_step_from_parent_or_data(df_laser)
        vals = np.arange(-10.0, 10.0 + step * 0.5, step, dtype=float)
        return [float(np.round(v, 6)) for v in vals.tolist()]

    # ----- helpers -----
    def _selected_laser_texts(self):
        out = []
        for it in self.lst_laser.selectedItems():
            try:
                out.append(str(it.text()).strip())
            except Exception:
                pass
        return out

    def _current_phi_value(self):
        return self.cmb_phi.value()

    def _pair_exists_exact(self, t_name, l_name):
        for r in range(self.table.rowCount()):
            it_t = self.table.item(r, 0)
            it_l = self.table.item(r, 1)
            if it_t is None or it_l is None:
                continue
            if it_t.text().strip() == str(t_name).strip() and it_l.text().strip() == str(l_name).strip():
                return True
        return False

    def _make_phi_widget(self, initial_value=None):
        w = PhiChoiceCombo(self.phi_choices)
        if initial_value is not None:
            w.set_value(initial_value)
        return w

    def _insert_pair_row(self, t_name, l_name, phi_value=None):
        if self._pair_exists_exact(t_name, l_name):
            return False

        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(t_name)))
        self.table.setItem(row, 1, QTableWidgetItem(str(l_name)))
        self.table.setCellWidget(row, 2, self._make_phi_widget(phi_value))
        return True

    # ----- actions -----
    def add_selected_pairs(self):
        t_name = self.cmb_tamiz.currentText().strip()
        laser_list = self._selected_laser_texts()
        phi_value = self._current_phi_value()

        if not t_name or not laser_list:
            QMessageBox.warning(self, tr("error"), "Select one sieve sample and at least one laser sample.")
            return

        added = 0
        for l_name in laser_list:
            ok = self._insert_pair_row(t_name, l_name, None if phi_value == self.cmb_phi.minimum() else phi_value)
            if ok:
                added += 1

        if added == 0:
            QMessageBox.information(self, tr("info"), "All selected Sieve–Laser pairs were already added.")

    def add_current_laser_only(self):
        t_name = self.cmb_tamiz.currentText().strip()
        current_item = self.lst_laser.currentItem()
        phi_value = self._current_phi_value()

        if not t_name or current_item is None:
            QMessageBox.warning(self, tr("error"), "Select one sieve sample and one laser sample.")
            return

        l_name = str(current_item.text()).strip()
        ok = self._insert_pair_row(t_name, l_name, None if phi_value == self.cmb_phi.minimum() else phi_value)
        if not ok:
            QMessageBox.information(self, tr("info"), "That exact Sieve–Laser pair already exists.")

    def delete_current_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    # ----- validation / output -----
    def get_pairs(self):
        """
        Return a list of dicts:
          [{"tamiz":..., "laser":..., "phi":...}, ...]
        """
        out = []
        for r in range(self.table.rowCount()):
            it_t = self.table.item(r, 0)
            it_l = self.table.item(r, 1)
            phi_w = self.table.cellWidget(r, 2)

            if it_t is None or it_l is None or phi_w is None:
                continue

            out.append({
                "tamiz": it_t.text().strip(),
                "laser": it_l.text().strip(),
                "phi": float(phi_w.value()),
            })
        return out

    def accept(self):
        """
        Validation:
          - at least one row
          - each row complete
          - phi* must be selected in every row
          - exact duplicate pair is forbidden
        """
        n = self.table.rowCount()
        if n == 0:
            QMessageBox.warning(self, tr("error"), tr("match_need_11"))
            return

        used_pairs = set()

        for r in range(n):
            it_t = self.table.item(r, 0)
            it_l = self.table.item(r, 1)
            phi_w = self.table.cellWidget(r, 2)

            if it_t is None or it_l is None or phi_w is None:
                QMessageBox.warning(self, tr("error"), tr("match_need_11"))
                return

            t_name = it_t.text().strip()
            l_name = it_l.text().strip()

            try:
                phi_val = float(phi_w.value())
                if float(phi_val) == float(phi_w.minimum()):
                    QMessageBox.warning(self, tr("error"), tr("match_need_phi"))
                    return
            except Exception:
                QMessageBox.warning(self, tr("error"), tr("match_need_phi"))
                return

            pair = (t_name, l_name)
            if pair in used_pairs:
                QMessageBox.warning(self, tr("error"), "Exact duplicate Sieve–Laser pair detected.")
                return
            used_pairs.add(pair)

        super().accept()

# endregion


#region BLOQUE 6 Data model: ComponentryDB
class ComponentryDB:
    def __init__(self, gsd_df: pd.DataFrame):
        self.gsd = gsd_df.copy()
        self.components = []
        self.colors = {}
        self.types = {}   # component -> type

        phis = self.gsd["phi"].astype(float).values
        self.frac100 = pd.DataFrame(index=phis)  # per phi, sum=100 in filled fractions
        self.total = pd.DataFrame(index=phis)    # per phi, sum=bin wt%
        self.filled = set()                      # phis with loaded componentry

        # Guarda estado de análisis de imagen por phi
        # phi -> {
        #   mode, image_path, points, polygons, background_enabled, background_component, freehand
        # }
        self.image_sessions = {}

    def wt_map(self):
        return dict(zip(self.gsd["phi"].astype(float).values, self.gsd["wt"].astype(float).values))

    def has_data_for_component(self, name):
        if name not in self.frac100.columns:
            return False
        s1 = float(np.nansum(self.frac100[name].values))
        s2 = float(np.nansum(self.total[name].values))
        return (s1 > 0) or (s2 > 0)

    def _renormalize_all_filled(self):
        if not self.components:
            self.frac100 = pd.DataFrame(index=self.frac100.index)
            self.total = pd.DataFrame(index=self.total.index)
            self.filled = set()
            return

        wt_map = self.wt_map()
        for phi in list(self.filled):
            if phi not in wt_map:
                continue
            row = self.frac100.loc[phi, self.components].astype(float).values
            s = float(np.nansum(row))
            if s <= 0:
                self.frac100.loc[phi, self.components] = 0.0
                self.total.loc[phi, self.components] = 0.0
                self.filled.discard(phi)
                continue

            row100 = 100.0 * row / s
            self.frac100.loc[phi, self.components] = row100

            bin_wt = float(wt_map[phi])
            self.total.loc[phi, self.components] = row100 * (bin_wt / 100.0)

    def remove_component(self, name):
        if name not in self.components:
            return
        self.components = [c for c in self.components if c != name]
        self.colors.pop(name, None)
        self.types.pop(name, None)

        self.frac100 = self.frac100.drop(columns=[name], errors="ignore")
        self.total = self.total.drop(columns=[name], errors="ignore")

        # Limpiar sessions de imagen que referencien componente eliminado
        for phi, sess in list(self.image_sessions.items()):
            if not isinstance(sess, dict):
                continue
            if sess.get("background_component") == name:
                sess["background_component"] = None
                sess["background_enabled"] = False
            polys = sess.get("polygons", [])
            sess["polygons"] = [p for p in polys if p.get("component") != name]
            pts = sess.get("points", [])
            sess["points"] = [p for p in pts if (len(p) >= 3 and p[2] != name)]

        self._renormalize_all_filled()

    def set_components(self, new_names, colors_by_name, types_by_name):
        new_names = [str(n).strip() for n in new_names]
        if any((not n) for n in new_names):
            raise ValueError("Component names cannot be empty.")
        if len(set(new_names)) != len(new_names):
            raise ValueError("Component names must be unique.")

        old = list(self.components)

        # rename position-based when safe
        nmin = min(len(old), len(new_names))
        rename_map = {}
        for i in range(nmin):
            o = old[i]
            n = new_names[i]
            if o != n and (o not in new_names) and (n not in old):
                rename_map[o] = n

        if rename_map:
            self.frac100 = self.frac100.rename(columns=rename_map)
            self.total = self.total.rename(columns=rename_map)
            for o, n in rename_map.items():
                if o in self.colors:
                    self.colors[n] = self.colors.pop(o)
                if o in self.types:
                    self.types[n] = self.types.pop(o)

            # rename inside image sessions
            for phi, sess in self.image_sessions.items():
                if not isinstance(sess, dict):
                    continue
                bg = sess.get("background_component")
                if bg in rename_map:
                    sess["background_component"] = rename_map[bg]

                pts = sess.get("points", [])
                new_pts = []
                for p in pts:
                    if len(p) >= 3:
                        c = p[2]
                        c = rename_map.get(c, c)
                        new_pts.append([p[0], p[1], c])
                sess["points"] = new_pts

                polys = sess.get("polygons", [])
                new_polys = []
                for pg in polys:
                    c = rename_map.get(pg.get("component"), pg.get("component"))
                    new_polys.append({"component": c, "xy": pg.get("xy", [])})
                sess["polygons"] = new_polys

        # add missing columns
        for n in new_names:
            if n not in self.frac100.columns:
                self.frac100[n] = 0.0
            if n not in self.total.columns:
                self.total[n] = 0.0

        # drop extras
        extra = [c for c in self.frac100.columns if c not in new_names]
        if extra:
            self.frac100 = self.frac100.drop(columns=extra, errors="ignore")
            self.total = self.total.drop(columns=extra, errors="ignore")
            for c in extra:
                self.colors.pop(c, None)
                self.types.pop(c, None)

        # reorder
        if new_names:
            self.frac100 = self.frac100.loc[:, new_names]
            self.total = self.total.loc[:, new_names]
        else:
            self.frac100 = self.frac100.iloc[:, 0:0]
            self.total = self.total.iloc[:, 0:0]

        self.components = list(new_names)

        # ensure color + type
        for i, n in enumerate(self.components):
            self.colors[n] = colors_by_name.get(n) or self.colors.get(n) or PASTEL[i % len(PASTEL)]
            t = types_by_name.get(n) or self.types.get(n) or "Accessory"
            self.types[n] = t if t in TYPE_OPTIONS else "Accessory"

        # limpiar sessions por componentes no existentes
        valid_set = set(self.components)
        for phi, sess in list(self.image_sessions.items()):
            if not isinstance(sess, dict):
                continue

            bg = sess.get("background_component")
            if bg not in valid_set:
                sess["background_component"] = None
                sess["background_enabled"] = False

            pts = sess.get("points", [])
            sess["points"] = [p for p in pts if (len(p) >= 3 and p[2] in valid_set)]

            polys = sess.get("polygons", [])
            sess["polygons"] = [p for p in polys if p.get("component") in valid_set]

    def set_fraction_manual_raw(self, phi, raw_values_dict):
        """
        raw_values_dict: numbers by component.
        internally normalized to 100% in that fraction and scaled to bin wt.
        """
        if not self.components:
            raise ValueError("No components defined.")

        phi = float(phi)
        wt_map = self.wt_map()
        if phi not in wt_map:
            raise ValueError("Unknown phi bin.")
        bin_wt = float(wt_map[phi])

        vals = []
        for c in self.components:
            v = raw_values_dict.get(c, 0.0)
            v = float(_to_float(v))
            if not np.isfinite(v) or v < 0:
                v = 0.0
            vals.append(v)

        s = float(np.sum(vals))
        if s <= 0:
            raise ValueError("Sum of inputs is 0; cannot normalize.")

        frac100 = 100.0 * (np.array(vals, dtype=float) / s)
        total = frac100 * (bin_wt / 100.0)

        self.frac100.loc[phi, self.components] = frac100
        self.total.loc[phi, self.components] = total
        self.filled.add(phi)

    def set_fraction_from_percentages(self, phi, perc_values_dict):
        """
        perc_values_dict: % by component (not necessarily summing 100 exactly).
        """
        self.set_fraction_manual_raw(phi, perc_values_dict)

    def load_totals_matrix(self, total_df: pd.DataFrame, components_order=None, colors_by_name=None, types_by_name=None):
        """
        Direct load from total contribution matrix (wt%) per phi and component.
        total_df index=phi, columns=component, values=wt% total.
        """
        if total_df is None or total_df.empty:
            raise ValueError("total_df is empty.")

        if components_order is None:
            components_order = [str(c) for c in total_df.columns]
        components_order = [str(c).strip() for c in components_order if str(c).strip()]

        if not components_order:
            raise ValueError("No components in total_df.")

        colors_by_name = colors_by_name or {}
        types_by_name = types_by_name or {}

        self.set_components(components_order, colors_by_name, types_by_name)

        self.frac100.loc[:, self.components] = 0.0
        self.total.loc[:, self.components] = 0.0
        self.filled = set()

        mat = total_df.copy()
        mat.index = mat.index.astype(float)

        wt_map = self.wt_map()

        for phi in self.frac100.index.astype(float).values:
            if phi not in wt_map:
                continue
            bin_wt = float(wt_map[phi])

            row_vals = []
            for c in self.components:
                if c in mat.columns and phi in mat.index:
                    v = float(_to_float(mat.loc[phi, c]))
                    if not np.isfinite(v) or v < 0:
                        v = 0.0
                else:
                    v = 0.0
                row_vals.append(v)

            row_vals = np.array(row_vals, dtype=float)
            s = float(np.sum(row_vals))
            if s <= 0:
                self.frac100.loc[phi, self.components] = 0.0
                self.total.loc[phi, self.components] = 0.0
                continue

            scaled_total = row_vals * (bin_wt / s)
            frac100 = (scaled_total / bin_wt) * 100.0 if bin_wt > 0 else np.zeros_like(scaled_total)

            self.total.loc[phi, self.components] = scaled_total
            self.frac100.loc[phi, self.components] = frac100
            self.filled.add(float(phi))

    def totals_by_type(self):
        phis = self.total.index.astype(float).values
        out = pd.DataFrame(0.0, index=phis, columns=TYPE_OPTIONS)
        if not self.components:
            return out
        for c in self.components:
            t = self.types.get(c, "Accessory")
            if t not in TYPE_OPTIONS:
                t = "Accessory"
            if c in self.total.columns:
                out[t] = out[t].astype(float) + self.total[c].astype(float)
        return out

    def frac100_by_type(self):
        phis = self.frac100.index.astype(float).values
        out = pd.DataFrame(0.0, index=phis, columns=TYPE_OPTIONS)
        if not self.components:
            return out
        for c in self.components:
            t = self.types.get(c, "Accessory")
            if t not in TYPE_OPTIONS:
                t = "Accessory"
            if c in self.frac100.columns:
                out[t] = out[t].astype(float) + self.frac100[c].astype(float)
        return out

    def to_dict(self):
        return {
            "gsd": self.gsd.to_dict(orient="list"),
            "components": list(self.components),
            "colors": dict(self.colors),
            "types": dict(self.types),
            "frac100": df_to_payload(self.frac100),
            "total": df_to_payload(self.total),
            "filled": [float(v) for v in sorted(list(self.filled))],
            "image_sessions": self._serialize_image_sessions()
        }

    def _serialize_image_sessions(self):
        out = {}
        for k, sess in self.image_sessions.items():
            phi_key = str(float(k))
            if not isinstance(sess, dict):
                continue
            out[phi_key] = {
                "mode": str(sess.get("mode", "points")),
                "image_path": sess.get("image_path"),
                "points": sess.get("points", []),
                "polygons": sess.get("polygons", []),
                "background_enabled": bool(sess.get("background_enabled", False)),
                "background_component": sess.get("background_component"),
                "freehand": bool(sess.get("freehand", False)),
            }
        return out

    @classmethod
    def from_dict(cls, payload):
        gsd_payload = payload.get("gsd", {})
        gsd = pd.DataFrame(gsd_payload)
        if "phi" not in gsd.columns or "wt" not in gsd.columns:
            raise ValueError("Invalid sample payload: gsd missing phi/wt.")
        gsd["phi"] = gsd["phi"].astype(float)
        gsd["wt"] = gsd["wt"].astype(float)

        db = cls(gsd)

        db.components = [str(c) for c in payload.get("components", [])]
        db.colors = {str(k): str(v) for k, v in payload.get("colors", {}).items()}
        db.types = {str(k): str(v) for k, v in payload.get("types", {}).items()}

        frac_payload = payload.get("frac100", {})
        total_payload = payload.get("total", {})

        db.frac100 = payload_to_df(frac_payload)
        db.total = payload_to_df(total_payload)

        phis = db.gsd["phi"].astype(float).values
        db.frac100 = db.frac100.reindex(index=phis, columns=db.components, fill_value=0.0)
        db.total = db.total.reindex(index=phis, columns=db.components, fill_value=0.0)

        db.filled = set(float(v) for v in payload.get("filled", []))

        raw_sessions = payload.get("image_sessions", {})
        db.image_sessions = {}
        for k, sess in raw_sessions.items():
            try:
                phi = float(k)
            except Exception:
                continue
            if not isinstance(sess, dict):
                continue
            db.image_sessions[phi] = {
                "mode": str(sess.get("mode", "points")),
                "image_path": sess.get("image_path"),
                "points": sess.get("points", []),
                "polygons": sess.get("polygons", []),
                "background_enabled": bool(sess.get("background_enabled", False)),
                "background_component": sess.get("background_component"),
                "freehand": bool(sess.get("freehand", False)),
            }

        for c in db.components:
            if c not in db.colors:
                db.colors[c] = PASTEL[len(db.colors) % len(PASTEL)]
            if c not in db.types or db.types[c] not in TYPE_OPTIONS:
                db.types[c] = "Accessory"

        return db
#endregion





# region # === Block 6B: Granulometric parameters panel (modern table view) ===
"""
Modern, theme-aware "Granulometric parameters" panel shown in the XY tab.

What it shows:
  - One table per dataset (Sieve / Laser / Hybrid), chosen with the selector
    above the table. Only the datasets that are loaded are listed.
  - Real column headers with the parameter symbols (same labels as the XY axes).
  - Right-aligned numbers with a fixed number of decimals; an em dash when a
    value could not be computed.
  - The Group column shows the same color used for that group in the plots.
  - Rows start sorted by sample name in natural order (R2 < R10). Click a column
    header to sort by it; right click > "Clear sorting" shows the data order.
  - Type in the filter box to show only matching samples or groups.
  - "Copy table" / Ctrl+C copies the rows as tab-separated text, ready to paste
    into Excel (numbers use the decimal separator of the operating system).

Compatibility:
  - MainWindow.update_console() fills this panel (Block 10).
  - MainWindow.txt_console points to this panel and still accepts
    setPlainText / toPlainText / append / clear, so older code and plugins
    written for the former plain-text console keep working.
  - Exporting to Excel is unchanged: it reads the parameter DataFrames directly.
"""
import re as _fw_pp_re
from string import Template as _FwPPTemplate

from PyQt5.QtCore import Qt, QEvent, QSize, QRectF, QTimer, QLocale, QSortFilterProxyModel
from PyQt5.QtGui import (
    QColor, QFont, QFontInfo, QFontMetrics, QIcon, QPainter, QPixmap, QPen, QBrush, QKeySequence,
    QStandardItemModel, QStandardItem
)
from PyQt5.QtWidgets import (
    QApplication, QFrame, QWidget, QLabel, QPushButton, QLineEdit, QButtonGroup,
    QVBoxLayout, QHBoxLayout, QStackedWidget, QTableView, QHeaderView,
    QAbstractItemView, QStyledItemDelegate, QSizePolicy, QAction
)

_FW_PP_PARAM_KEYS = ("mean", "median", "sigma", "skewness", "kurtosis")
_FW_PP_ROLE_SORT = Qt.UserRole + 41
_FW_PP_ROLE_RAW = Qt.UserRole + 42
_FW_PP_DECIMALS = 3

_FW_PP_PARAM_HINTS = {
    "mean": "Mean grain size",
    "median": "Median grain size (φ50)",
    "sigma": "Sorting: spread of the distribution",
    "skewness": "Skewness: asymmetry of the distribution",
    "kurtosis": "Kurtosis: peakedness of the distribution",
}

_FW_PP_DATASET_HINTS = {
    "Tamiz": "Parameters computed from the sieve data",
    "Laser": "Parameters computed from the laser data",
    "Híbrido": "Parameters computed from the combined sieve + laser data",
}

# Colors of the panel for each UI theme (Settings > UI theme).
_FW_PP_PALETTES = {
    "light": {
        "card": "#ffffff", "border": "#dde3ea", "title": "#0f172a", "text": "#1e293b",
        "muted": "#64748b",
        "badge_bg": "#eef4ff", "badge_fg": "#1d4ed8", "badge_border": "#c9dafc",
        "seg_bg": "#f1f5f9", "seg_border": "#e2e8f0",
        "chip_fg": "#475569", "chip_hover": "#e2e8f0",
        "chip_on_bg": "#ffffff", "chip_on_fg": "#0f172a", "chip_on_border": "#cbd5e1",
        "input_bg": "#ffffff", "input_border": "#cfd7e3", "placeholder": "#94a3b8",
        "accent": "#2563eb", "accent_hover": "#1d4ed8", "accent_pressed": "#1e40af",
        "btn_bg": "#ffffff", "btn_fg": "#0f172a", "btn_border": "#cfd7e3",
        "btn_hover": "#f8fafc", "btn_pressed": "#eef2f7",
        "disabled_bg": "#f2f4f7", "disabled_fg": "#9aa4b2", "disabled_border": "#e2e8f0",
        "table_bg": "#ffffff", "table_alt": "#f8fafc", "row_line": "#edf1f5",
        "head_bg": "#f1f5f9", "head_fg": "#334155", "head_line": "#dde3ea",
        "sel_bg": "#dbeafe", "sel_fg": "#0f172a", "sel_inactive": "#e7effd",
        "scroll": "#cbd5e1", "scroll_hover": "#94a3b8",
        "na": "#a0aab8",
    },
    "dark": {
        "card": "#2b2e31", "border": "#3d4246", "title": "#f3f4f6", "text": "#e5e7eb",
        "muted": "#9aa3ad",
        "badge_bg": "#1f3550", "badge_fg": "#cfe1ff", "badge_border": "#2f4f78",
        "seg_bg": "#232629", "seg_border": "#3d4246",
        "chip_fg": "#c3c8ce", "chip_hover": "#34383c",
        "chip_on_bg": "#43484d", "chip_on_fg": "#ffffff", "chip_on_border": "#575d63",
        "input_bg": "#232629", "input_border": "#4a4f54", "placeholder": "#7d848c",
        "accent": "#3b82f6", "accent_hover": "#2f6fdb", "accent_pressed": "#2a5fbd",
        "btn_bg": "#393e41", "btn_fg": "#edeef0", "btn_border": "#4a4f54",
        "btn_hover": "#45494d", "btn_pressed": "#515457",
        "disabled_bg": "#2f3236", "disabled_fg": "#6b7178", "disabled_border": "#3a3e42",
        "table_bg": "#2b2e31", "table_alt": "#303438", "row_line": "#363a3e",
        "head_bg": "#33373b", "head_fg": "#d6dade", "head_line": "#43484d",
        "sel_bg": "#35537a", "sel_fg": "#ffffff", "sel_inactive": "#2f4560",
        "scroll": "#4a4f54", "scroll_hover": "#5d6369",
        "na": "#6b7178",
    },
}

_FW_PP_QSS = _FwPPTemplate(
    "QFrame#fwParamsCard { background-color: $card; border: 1px solid $border; border-radius: 10px; }\n"
    "QLabel { background: transparent; border: none; }\n"
    "QWidget#fwParamsToolbar, QStackedWidget#fwParamsStack, QWidget#fwParamsEmptyPage { background: transparent; border: none; }\n"
    "QLabel#fwParamsTitle { color: $title; font-size: $title_pt; font-weight: 600; }\n"
    "QLabel#fwParamsBadge { color: $badge_fg; background-color: $badge_bg; border: 1px solid $badge_border; border-radius: 6px; padding: 2px 8px; font-size: $badge_pt; font-weight: 600; }\n"
    "QLabel#fwParamsSummary { color: $muted; }\n"
    "QLabel#fwParamsEmptyTitle { color: $title; font-size: $empty_pt; font-weight: 600; }\n"
    "QLabel#fwParamsEmptyText { color: $muted; }\n"
    "QFrame#fwParamsSeg { background-color: $seg_bg; border: 1px solid $seg_border; border-radius: 8px; }\n"
    "QPushButton#fwParamsChip { background-color: transparent; color: $chip_fg; border: 1px solid transparent; border-radius: 6px; padding: 4px 16px; min-height: 18px; min-width: 0px; }\n"
    "QPushButton#fwParamsChip:hover { background-color: $chip_hover; }\n"
    "QPushButton#fwParamsChip:checked { background-color: $chip_on_bg; color: $chip_on_fg; border: 1px solid $chip_on_border; font-weight: 600; }\n"
    "QLineEdit#fwParamsFilter { background-color: $input_bg; color: $text; border: 1px solid $input_border; border-radius: 8px; padding: 4px 8px; min-height: 20px; selection-background-color: $sel_bg; selection-color: $sel_fg; }\n"
    "QLineEdit#fwParamsFilter:focus { border: 1px solid $accent; }\n"
    "QPushButton#fwParamsSecondary { background-color: $btn_bg; color: $btn_fg; border: 1px solid $btn_border; border-radius: 8px; padding: 5px 14px; min-height: 20px; }\n"
    "QPushButton#fwParamsSecondary:hover { background-color: $btn_hover; }\n"
    "QPushButton#fwParamsSecondary:pressed { background-color: $btn_pressed; }\n"
    "QPushButton#fwParamsPrimary { background-color: $accent; color: #ffffff; border: 1px solid $accent; border-radius: 8px; padding: 5px 16px; min-height: 20px; font-weight: 600; }\n"
    "QPushButton#fwParamsPrimary:hover { background-color: $accent_hover; border-color: $accent_hover; }\n"
    "QPushButton#fwParamsPrimary:pressed { background-color: $accent_pressed; border-color: $accent_pressed; }\n"
    "QPushButton#fwParamsPrimary:disabled, QPushButton#fwParamsSecondary:disabled { background-color: $disabled_bg; color: $disabled_fg; border: 1px solid $disabled_border; }\n"
    "QTableView#fwParamsTable { background-color: $table_bg; alternate-background-color: $table_alt; color: $text; border: 1px solid $border; border-radius: 8px; padding: 0px; gridline-color: $row_line; selection-background-color: $sel_bg; selection-color: $sel_fg; outline: 0; }\n"
    "QTableView#fwParamsTable::item { padding: 0px 12px; border: none; border-bottom: 1px solid $row_line; }\n"
    "QTableView#fwParamsTable::item:selected { background-color: $sel_bg; color: $sel_fg; }\n"
    "QTableView#fwParamsTable::item:selected:!active { background-color: $sel_inactive; color: $sel_fg; }\n"
    "QTableView#fwParamsTable QHeaderView { background-color: $head_bg; border: none; border-top-left-radius: 8px; border-top-right-radius: 8px; }\n"
    "QTableView#fwParamsTable QHeaderView::section { background-color: $head_bg; color: $head_fg; border: none; border-bottom: 1px solid $head_line; padding: 6px 15px; font-weight: 600; }\n"
    "QTableView#fwParamsTable QTableCornerButton::section { background-color: $head_bg; border: none; }\n"
    "QTableView#fwParamsTable QScrollBar:vertical { background: transparent; width: 10px; margin: 2px 1px 2px 1px; border: none; }\n"
    "QTableView#fwParamsTable QScrollBar::handle:vertical { background: $scroll; border-radius: 4px; min-height: 28px; }\n"
    "QTableView#fwParamsTable QScrollBar::handle:vertical:hover { background: $scroll_hover; }\n"
    "QTableView#fwParamsTable QScrollBar:horizontal { background: transparent; height: 10px; margin: 1px 2px 1px 2px; border: none; }\n"
    "QTableView#fwParamsTable QScrollBar::handle:horizontal { background: $scroll; border-radius: 4px; min-width: 28px; }\n"
    "QTableView#fwParamsTable QScrollBar::handle:horizontal:hover { background: $scroll_hover; }\n"
    "QTableView#fwParamsTable QScrollBar::add-line, QTableView#fwParamsTable QScrollBar::sub-line { width: 0px; height: 0px; border: none; background: none; }\n"
    "QTableView#fwParamsTable QScrollBar::add-page, QTableView#fwParamsTable QScrollBar::sub-page { background: none; }\n"
)


def _fw_method_display_name(code) -> str:
    """'folkward' -> 'Folk & Ward (1957)' (uses METHODS from Block 2)."""
    c = str(code or "").strip().lower()
    try:
        for label, key in METHODS.items():
            if str(key).strip().lower() == c:
                return str(label)
    except Exception:
        pass
    fallback = {
        "folkward": "Folk & Ward (1957)",
        "blatt": "Folk & Ward (1957)",
        "inman": "Inman (1952)",
        "trask": "Trask (1932)",
    }
    return fallback.get(c, str(code or ""))


def _fw_pp_is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _fw_pp_float(v):
    """float(v) or None when it is missing / not finite / not numeric."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _fw_pp_fmt(f, decimals=_FW_PP_DECIMALS, point=".", minus="\u2212"):
    """Fixed decimals, no '-0.000', typographic minus for display."""
    if f is None:
        return None
    if abs(f) < 0.5 * (10 ** (-decimals)):
        f = 0.0
    s = "{:.{d}f}".format(f, d=decimals)
    if s.startswith("-"):
        s = minus + s[1:]
    if point != ".":
        s = s.replace(".", point)
    return s


def _fw_pp_column_list(df, col):
    """Column values as a Python list, keeping each column's own type (ints stay ints)."""
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):      # duplicated column name: first one
            s = s.iloc[:, 0]
        return s.tolist()
    except Exception:
        return [None] * len(df)


def _fw_pp_color_name(color) -> str:
    """Any color accepted by the plots (hex, name, 'tab:blue', RGB tuple) -> '#rrggbb'."""
    try:
        from matplotlib.colors import to_hex
        return to_hex(color, keep_alpha=False)
    except Exception:
        s = str(color or "")
        return s if QColor.isValidColor(s) else "#94a3b8"


def _fw_pp_natural_key(text):
    """Natural sort key: 'R2' < 'R10'. Type-safe (str at even, int at odd positions)."""
    s = "" if text is None else str(text).strip().lower()
    out = []
    for i, part in enumerate(_fw_pp_re.split(r"(\d+)", s)):
        if i % 2:
            try:
                out.append(int(part))
            except Exception:
                out.append(0)
        else:
            out.append(part)
    return out


class _FwParamsProxy(QSortFilterProxyModel):
    """Filter by sample/group text and sort numbers numerically, names naturally."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._needle = ""

    def set_needle(self, text):
        self._needle = str(text or "").strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if not self._needle:
            return True
        m = self.sourceModel()
        if m is None:
            return True
        for col in (0, 1):
            try:
                val = m.data(m.index(source_row, col, source_parent), Qt.DisplayRole)
            except Exception:
                val = None
            if val is not None and self._needle in str(val).lower():
                return True
        return False

    def lessThan(self, left, right):
        try:
            m = self.sourceModel()
            a = m.data(left, _FW_PP_ROLE_SORT)
            b = m.data(right, _FW_PP_ROLE_SORT)
            a_num = isinstance(a, (int, float)) and not isinstance(a, bool)
            b_num = isinstance(b, (int, float)) and not isinstance(b, bool)
            if a_num or b_num:
                fa = float(a) if a_num else float("inf")   # missing values go last
                fb = float(b) if b_num else float("inf")
                if fa != fb:
                    return fa < fb
            else:
                ka = _fw_pp_natural_key(m.data(left, Qt.DisplayRole))
                kb = _fw_pp_natural_key(m.data(right, Qt.DisplayRole))
                if ka != kb:
                    return ka < kb
            # tie: order by sample name, then by original position
            sa = _fw_pp_natural_key(m.data(m.index(left.row(), 0), Qt.DisplayRole))
            sb = _fw_pp_natural_key(m.data(m.index(right.row(), 0), Qt.DisplayRole))
            if sa != sb:
                return sa < sb
            return left.row() < right.row()
        except Exception:
            return left.row() < right.row()


class _FwParamsTableView(QTableView):
    """Table that keeps its own item delegate and copies whole rows with Ctrl+C."""

    def __init__(self, on_copy=None, parent=None):
        super().__init__(parent)
        self._on_copy = on_copy
        self._own_delegate = QStyledItemDelegate(self)
        self.setItemDelegate(self._own_delegate)

    def showEvent(self, e):
        super().showEvent(e)
        # The global table formatter of Block 5 (when active) swaps delegates on Show.
        # This table already formats its numbers, so it keeps its own delegate.
        try:
            self.setItemDelegate(self._own_delegate)
        except Exception:
            pass

    def keyPressEvent(self, e):
        try:
            if e.matches(QKeySequence.Copy) and callable(self._on_copy):
                self._on_copy()
                e.accept()
                return
        except Exception:
            pass
        super().keyPressEvent(e)


class GranuloParamsPanel(QFrame):
    """
    "Granulometric parameters" card of the XY tab.

    Used by MainWindow:
        set_datasets(datasets, method_label, group_colors, legacy_text)
            datasets = [(base_key, DataFrame, group_column), ...]
        sync_theme()
    Kept for older code / plugins (former QTextEdit console):
        setPlainText, toPlainText, append, clear, setReadOnly, isReadOnly
    """

    _BASE_ORDER = ("Tamiz", "Laser", "Híbrido")
    # Below this height the selector and the filter move up next to the title (one header row),
    # so the table keeps some room on short screens; above _EXPAND_AT the two-row header returns.
    _COMPACT_BELOW = 200
    _EXPAND_AT = 250

    def __init__(self, owner=None, parent=None):
        super().__init__(parent)
        self._owner = owner
        self._pal = dict(_FW_PP_PALETTES["light"])
        self._applied_theme = None
        self._theme_pending = False
        self._datasets = {}
        self._order = []
        self._current = None
        self._known = set()
        self._method_label = ""
        self._group_colors = {}
        self._group_markers = {}
        self._group_of = None
        self._legacy_text = ""
        self._dot_cache = {}
        self._n_groups = 0
        self._flash_token = 0
        self._copy_token = 0
        self._compact = False
        self._compact_pending = False
        self._empty_title = tr("params_empty_title")
        self._empty_text = tr("params_empty_text")

        self.setObjectName("fwParamsCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        # ---- Row 1: title + method badge | actions
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(10)

        self.lbl_title = QLabel(tr("console_header"))
        self.lbl_title.setObjectName("fwParamsTitle")
        row1.addWidget(self.lbl_title, 0, Qt.AlignVCenter)

        self.lbl_method = QLabel("")
        self.lbl_method.setObjectName("fwParamsBadge")
        self.lbl_method.setToolTip("Computation method in use (Settings > Computation method)")
        self.lbl_method.setVisible(False)
        row1.addWidget(self.lbl_method, 0, Qt.AlignVCenter)

        row1.addStretch(1)

        self.btn_copy = QPushButton(tr("params_copy_table"))
        self.btn_copy.setObjectName("fwParamsSecondary")
        self.btn_copy.setCursor(Qt.PointingHandCursor)
        self.btn_copy.setToolTip("Copy the rows shown below to the clipboard, ready to paste into Excel")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(lambda *_: self.copy_visible_rows())
        row1.addWidget(self.btn_copy, 0, Qt.AlignVCenter)

        # Connected to MainWindow.export_params_to_excel in initUI_tabs (Block 7)
        self.btn_export = QPushButton(tr("btn_export_params"))
        self.btn_export.setObjectName("fwParamsPrimary")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setToolTip("Save every parameter table (and the data) to an Excel workbook")
        row1.addWidget(self.btn_export, 0, Qt.AlignVCenter)

        root.addLayout(row1)
        self._row1 = row1

        # ---- Row 2: dataset selector | summary + filter
        self.row2 = QWidget()
        self.row2.setObjectName("fwParamsToolbar")
        row2 = QHBoxLayout(self.row2)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(12)

        self.seg = QFrame()
        self.seg.setObjectName("fwParamsSeg")
        seg_lay = QHBoxLayout(self.seg)
        seg_lay.setContentsMargins(3, 3, 3, 3)
        seg_lay.setSpacing(2)
        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)
        self._chips = {}
        for bk in self._BASE_ORDER:
            b = QPushButton(base_name_display(bk))
            b.setObjectName("fwParamsChip")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(_FW_PP_DATASET_HINTS.get(bk, ""))
            b.setVisible(False)
            b.clicked.connect(lambda _checked=False, k=bk: self._select_dataset(k))
            self._chip_group.addButton(b)
            seg_lay.addWidget(b)
            self._chips[bk] = b
        row2.addWidget(self.seg, 0, Qt.AlignVCenter)

        row2.addStretch(1)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setObjectName("fwParamsSummary")
        row2.addWidget(self.lbl_summary, 0, Qt.AlignVCenter)

        self.txt_filter = QLineEdit()
        self.txt_filter.setObjectName("fwParamsFilter")
        self.txt_filter.setPlaceholderText(tr("params_filter_placeholder"))
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.setFixedWidth(240)
        self.txt_filter.textChanged.connect(lambda t: self._on_filter_changed(t))
        row2.addWidget(self.txt_filter, 0, Qt.AlignVCenter)

        root.addWidget(self.row2)
        self._row2_lay = row2

        # ---- Body: empty state / table
        self.stack = QStackedWidget()
        self.stack.setObjectName("fwParamsStack")

        self.page_empty = QWidget()
        self.page_empty.setObjectName("fwParamsEmptyPage")
        lay_e = QVBoxLayout(self.page_empty)
        lay_e.setContentsMargins(8, 8, 8, 8)
        lay_e.setSpacing(6)
        lay_e.addStretch(1)
        self.lbl_empty_title = QLabel(self._empty_title)
        self.lbl_empty_title.setObjectName("fwParamsEmptyTitle")
        self.lbl_empty_title.setAlignment(Qt.AlignCenter)
        lay_e.addWidget(self.lbl_empty_title)
        self.lbl_empty_text = QLabel(self._empty_text)
        self.lbl_empty_text.setObjectName("fwParamsEmptyText")
        self.lbl_empty_text.setAlignment(Qt.AlignCenter)
        self.lbl_empty_text.setWordWrap(True)
        lay_e.addWidget(self.lbl_empty_text)
        lay_e.addStretch(1)
        self.stack.addWidget(self.page_empty)

        self.model = QStandardItemModel(0, 2 + len(_FW_PP_PARAM_KEYS), self)
        self.proxy = _FwParamsProxy(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setDynamicSortFilter(True)

        self.table = _FwParamsTableView(on_copy=self.copy_selected_rows)
        self.table.setObjectName("fwParamsTable")
        self.table.setModel(self.proxy)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.setCornerButtonEnabled(False)
        self.table.setIconSize(QSize(10, 10))
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

        row_h = max(26, self.table.fontMetrics().height() + 10)
        vh = self.table.verticalHeader()
        vh.setVisible(False)
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(row_h)
        self.table.setMinimumHeight(row_h + max(30, self.table.fontMetrics().height() + 14))

        hh = self.table.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setSectionsClickable(True)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hh.setMinimumSectionSize(64)
        hh.setStretchLastSection(False)
        hh.setSortIndicator(0, Qt.AscendingOrder)   # start sorted by sample name (natural order)
        self.table.setSortingEnabled(True)

        self.table.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.act_copy_sel = QAction("Copy selected rows\tCtrl+C", self.table)
        self.act_copy_sel.triggered.connect(lambda *_: self.copy_selected_rows())
        self.table.addAction(self.act_copy_sel)
        self.act_copy_all = QAction("Copy table", self.table)
        self.act_copy_all.triggered.connect(lambda *_: self.copy_visible_rows())
        self.table.addAction(self.act_copy_all)
        self.act_sort_reset = QAction("Clear sorting (data order)", self.table)
        self.act_sort_reset.triggered.connect(lambda *_: self._reset_sort())
        self.table.addAction(self.act_sort_reset)

        self.stack.addWidget(self.table)
        root.addWidget(self.stack, 1)

        self.sync_theme(force=True)
        self._show_empty()

    # ------------------------------------------------------------------ data
    def set_datasets(self, datasets, method_label="", group_colors=None, legacy_text="",
                     group_markers=None, group_of=None):
        """datasets: list of (base_key, DataFrame, group_column).
        group_of(base_key, sample, data_label) -> group shown (custom groups); group_markers: {key: symbol}."""
        self._group_markers = dict(group_markers or {})
        self._group_of = group_of if callable(group_of) else None
        self._legacy_text = "" if legacy_text is None else str(legacy_text)
        self._method_label = str(method_label or "")
        self._group_colors = dict(group_colors or {})

        new, order = {}, []
        for item in (datasets or []):
            try:
                base_key, df, gcol = item
            except Exception:
                continue
            if df is None or getattr(df, "empty", True):
                continue
            try:
                bk = base_key_canon(base_key)
            except Exception:
                bk = base_key
            if bk in new:
                continue
            new[bk] = (df, gcol)
            order.append(bk)

        # A dataset that was just loaded/combined is shown right away;
        # otherwise the dataset the user was looking at stays selected.
        appeared = [k for k in order if k not in self._known]
        if self._known and len(appeared) == 1:
            self._current = appeared[0]
        elif self._current not in new:
            self._current = order[0] if order else None
        self._known = set(order)
        self._datasets = new
        self._order = order

        self._refresh_badge()
        self._refresh_chips()
        if not order:
            self._show_empty()
            return
        self._apply_toolbar_state()
        self.stack.setCurrentWidget(self.table)
        self._fill_table()

    def current_dataset(self):
        """Canonical key of the dataset shown ('Tamiz' | 'Laser' | 'Híbrido') or None."""
        return self._current

    def _refresh_badge(self):
        txt = self._method_label.strip()
        self.lbl_method.setText(txt)
        self.lbl_method.setVisible(bool(txt))

    def _refresh_chips(self):
        for bk, btn in self._chips.items():
            present = bk in self._datasets
            btn.setVisible(present)
            if present:
                n = int(len(self._datasets[bk][0]))
                txt = f"{base_name_display(bk)}  {n}"
                btn.setText(txt)
                btn.setMinimumWidth(self._chip_min_width(btn, txt))
                btn.setToolTip(f"{_FW_PP_DATASET_HINTS.get(bk, '')} ({n} {'sample' if n == 1 else 'samples'})")
        if self._current in self._chips and self._current in self._datasets:
            self._chips[self._current].setChecked(True)
        self._apply_toolbar_state()

    def _chip_min_width(self, btn, text):
        """Width that fits the text in bold (the selected chip is drawn bold) plus the padding."""
        try:
            f = QFont(btn.font())
            f.setWeight(QFont.Bold)
            fm = QFontMetrics(f)
            try:
                w = fm.horizontalAdvance(text)
            except AttributeError:
                w = fm.width(text)
            return int(w + 2 * 16 + 2 + 8)
        except Exception:
            return 0

    def _select_dataset(self, bk):
        try:
            if bk not in self._datasets:
                return
            if bk == self._current and self.model.rowCount() > 0:
                self._refresh_chips()
                return
            self._current = bk
            self._refresh_chips()
            self._fill_table()
        except Exception:
            traceback.print_exc()

    def _param_header(self, key):
        try:
            return param_label(key)
        except Exception:
            return str(key)

    def _sample_tooltip(self, extra, sample_txt, grp_txt, disp_base):
        """extra: {column: value} with the optional hybrid metadata of the row."""
        lines = [f"{sample_txt}  ·  {grp_txt} ({disp_base})"]
        for col, label in (("Tamiz Sample", "Sieve sample"),
                           ("Laser Sample", "Laser sample"),
                           ("Hybrid Method", "Hybrid method")):
            v = extra.get(col, None)
            if not _fw_pp_is_missing(v) and str(v).strip():
                lines.append(f"{label}: {v}")
        v = _fw_pp_float(extra.get("phi*", None))
        if v is not None:
            lines.append(f"φ* = {_fw_pp_fmt(v)}")
        return "\n".join(lines)

    def _dot_icon(self, color, marker="o"):
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        dpr = max(1.0, dpr)
        key = (repr(color), str(marker), dpr)
        ic = self._dot_cache.get(key)
        if ic is not None:
            return ic
        if marker and marker != "o":
            try:
                ic = fw_marker_icon(marker, color, 11, dpr)
                self._dot_cache[key] = ic
                return ic
            except Exception:
                pass
        size = 10
        pm = QPixmap(int(round(size * dpr)), int(round(size * dpr)))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            qc = QColor(_fw_pp_color_name(color))
            p.setPen(QPen(qc.darker(135), 1.0))
            p.setBrush(QBrush(qc))
            p.drawEllipse(QRectF(0.5, 0.5, size - 1.0, size - 1.0))
        finally:
            p.end()
        ic = QIcon(pm)
        self._dot_cache[key] = ic
        return ic

    def _fill_table(self):
        bk = self._current
        pack = self._datasets.get(bk)
        if pack is None:
            self._show_empty()
            return
        df, gcol = pack
        cols = list(df.columns)
        if "Hybrid Sample" in cols:
            scol = "Hybrid Sample"
        elif "Sample" in cols:
            scol = "Sample"
        else:
            scol = cols[0] if cols else None
        has_group = bool(gcol) and (gcol in cols)
        group_header = str(gcol) if has_group else "Group"
        keys = list(_FW_PP_PARAM_KEYS)
        headers = ["Sample", group_header] + [self._param_header(k) for k in keys]

        na_brush = QBrush(QColor(self._pal.get("na", "#a0aab8")))
        bold = QFont(self.table.font())
        bold.setWeight(QFont.DemiBold)
        disp_base = base_name_display(bk)

        self.table.setSortingEnabled(False)
        self.model.removeRows(0, self.model.rowCount())
        self.model.setColumnCount(len(headers))
        for c, text in enumerate(headers):
            hi = QStandardItem(text)
            if c < 2:
                hi.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            else:
                hi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tip = _FW_PP_PARAM_HINTS.get(keys[c - 2], "")
                if self._method_label:
                    tip = f"{tip}\nMethod: {self._method_label}"
                hi.setToolTip(tip)
            self.model.setHorizontalHeaderItem(c, hi)

        # Read column by column (row-wise iteration would turn integer names into floats)
        n_rows = len(df)
        blank = [None] * n_rows
        v_sample = _fw_pp_column_list(df, scol) if scol is not None else blank
        v_group = _fw_pp_column_list(df, gcol) if has_group else blank
        v_params = {k: (_fw_pp_column_list(df, k) if k in cols else blank) for k in keys}
        v_extra = {c: _fw_pp_column_list(df, c)
                   for c in ("Tamiz Sample", "Laser Sample", "Hybrid Method", "phi*") if c in cols}

        groups_seen = set()
        for i in range(n_rows):
            sample = v_sample[i]
            sample_txt = "" if _fw_pp_is_missing(sample) else str(sample)
            grp_raw = v_group[i]
            grp_eff = grp_raw
            if self._group_of is not None:
                try:
                    grp_eff = self._group_of(bk, sample, grp_raw)
                except Exception:
                    grp_eff = grp_raw
            grp_txt = _norm_group_name(grp_eff)
            groups_seen.add(grp_txt)
            gk_row = _gkey(grp_eff, bk)
            color = self._group_colors.get(gk_row, "#94a3b8")

            it_s = QStandardItem(sample_txt)
            it_s.setEditable(False)
            it_s.setFont(bold)
            it_s.setData(sample_txt, _FW_PP_ROLE_RAW)
            it_s.setToolTip(self._sample_tooltip({c: v[i] for c, v in v_extra.items()},
                                                 sample_txt, grp_txt, disp_base))

            it_g = QStandardItem(grp_txt)
            it_g.setEditable(False)
            it_g.setData(grp_txt, _FW_PP_ROLE_RAW)
            it_g.setIcon(self._dot_icon(color, self._group_markers.get(gk_row, "o")))
            if grp_txt != _norm_group_name(grp_raw):
                it_g.setToolTip(f"{grp_txt} ({disp_base})\nCustom group · data label: {_norm_group_name(grp_raw)}")
            else:
                it_g.setToolTip(f"{grp_txt} ({disp_base})")

            row_items = [it_s, it_g]
            for k in keys:
                f = _fw_pp_float(v_params[k][i])
                txt = _fw_pp_fmt(f)
                it = QStandardItem(txt if txt is not None else "\u2014")
                it.setEditable(False)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                it.setData(f, _FW_PP_ROLE_SORT)
                it.setData(f, _FW_PP_ROLE_RAW)
                if f is None:
                    it.setForeground(na_brush)
                    it.setToolTip("Not available for this sample")
                row_items.append(it)
            self.model.appendRow(row_items)

        self._n_groups = len(groups_seen)
        self.table.setSortingEnabled(True)   # re-applies the column the user sorted by
        self._apply_column_layout()
        self._update_summary()
        self.btn_copy.setEnabled(self.proxy.rowCount() > 0)

    def _apply_column_layout(self):
        t = self.table
        hh = t.horizontalHeader()
        n = self.model.columnCount()
        for c in range(n):
            hh.setSectionResizeMode(c, QHeaderView.Interactive if c < 2 else QHeaderView.Stretch)
        for c, (lo, hi) in ((0, (110, 320)), (1, (170, 380))):
            if c < n:
                t.resizeColumnToContents(c)
                t.setColumnWidth(c, max(lo, min(hi, t.columnWidth(c) + 20)))

    def _update_summary(self):
        total = self.model.rowCount()
        shown = self.proxy.rowCount()
        noun = "sample" if total == 1 else "samples"
        txt = f"{shown} of {total} {noun}" if shown != total else f"{total} {noun}"
        if self._n_groups:
            txt += f"  ·  {self._n_groups} {'group' if self._n_groups == 1 else 'groups'}"
        self.lbl_summary.setText(txt)

    def _on_filter_changed(self, text):
        try:
            self.proxy.set_needle(text)
            self._update_summary()
            self.btn_copy.setEnabled(self.proxy.rowCount() > 0)
        except Exception:
            traceback.print_exc()

    def _reset_sort(self):
        try:
            self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
            self.proxy.sort(-1)
        except Exception:
            traceback.print_exc()

    def _show_empty(self):
        self.table.setSortingEnabled(False)
        self.model.removeRows(0, self.model.rowCount())
        self.table.setSortingEnabled(True)
        self._n_groups = 0
        self.lbl_empty_title.setText(self._empty_title)
        self.lbl_empty_text.setText(self._empty_text)
        self.lbl_summary.setText("")
        self.btn_copy.setEnabled(False)
        self._apply_toolbar_state()
        self.stack.setCurrentWidget(self.page_empty)

    # ------------------------------------------------------------ layout
    def _apply_toolbar_state(self):
        has = bool(self._order)
        self.seg.setVisible(has)
        self.txt_filter.setVisible(has)
        self.lbl_summary.setVisible(has and not self._compact)
        self.row2.setVisible(has and not self._compact)

    def _set_compact(self, flag):
        """Compact = one header row (selector and filter next to the title)."""
        flag = bool(flag)
        if flag == self._compact:
            return
        r1, r2 = self._row1, self._row2_lay
        if flag:
            r2.removeWidget(self.seg)
            r2.removeWidget(self.txt_filter)
            # row 1: title, badge, [selector], stretch, [filter], copy, export
            r1.insertWidget(2, self.seg, 0, Qt.AlignVCenter)
            r1.insertWidget(4, self.txt_filter, 0, Qt.AlignVCenter)
            self.txt_filter.setFixedWidth(180)
        else:
            r1.removeWidget(self.seg)
            r1.removeWidget(self.txt_filter)
            # row 2: selector, stretch, summary, filter
            r2.insertWidget(0, self.seg, 0, Qt.AlignVCenter)
            r2.addWidget(self.txt_filter, 0, Qt.AlignVCenter)
            self.txt_filter.setFixedWidth(240)
        self._compact = flag
        self._apply_toolbar_state()

    def _auto_compact(self):
        self._compact_pending = False
        try:
            h = self.height()
            if not self._compact and h < self._COMPACT_BELOW:
                self._set_compact(True)
            elif self._compact and h >= self._EXPAND_AT:
                self._set_compact(False)
        except Exception:
            traceback.print_exc()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            if not self._compact_pending:
                self._compact_pending = True
                QTimer.singleShot(0, lambda: self._auto_compact())
        except Exception:
            pass

    # ------------------------------------------------------------------ copy
    def _decimal_point(self):
        try:
            p = str(QLocale.system().decimalPoint())
            return p[:1] if p else "."
        except Exception:
            return "."

    def _rows_as_tsv(self, proxy_rows):
        n = self.model.columnCount()
        point = self._decimal_point()
        header = []
        for c in range(n):
            h = self.model.headerData(c, Qt.Horizontal, Qt.DisplayRole)
            header.append("" if h is None else str(h))
        lines = ["\t".join(header)]
        for pr in proxy_rows:
            cells = []
            for c in range(n):
                sidx = self.proxy.mapToSource(self.proxy.index(pr, c))
                raw = self.model.data(sidx, _FW_PP_ROLE_RAW)
                if c >= 2:
                    f = _fw_pp_float(raw)
                    txt = _fw_pp_fmt(f, point=point, minus="-") if f is not None else ""
                else:
                    txt = "" if raw is None else str(raw)
                cells.append(txt.replace("\t", " ").replace("\r", " ").replace("\n", " "))
            lines.append("\t".join(cells))
        return "\n".join(lines) + "\n"

    def _to_clipboard(self, proxy_rows):
        if not proxy_rows:
            return
        QApplication.clipboard().setText(self._rows_as_tsv(proxy_rows))
        k = len(proxy_rows)
        self._flash(f"Copied {k} {'row' if k == 1 else 'rows'} to the clipboard")
        self._flash_copy_button()

    def _flash_copy_button(self, ms=1600):
        self._copy_token += 1
        token = self._copy_token
        self.btn_copy.setMinimumWidth(self.btn_copy.width())
        self.btn_copy.setText(tr("params_copied"))
        QTimer.singleShot(int(ms), lambda: self._end_copy_flash(token))

    def _end_copy_flash(self, token):
        try:
            if token != self._copy_token:
                return
            self.btn_copy.setText(tr("params_copy_table"))
            self.btn_copy.setMinimumWidth(0)
        except Exception:
            pass

    def copy_selected_rows(self):
        """Ctrl+C: copies the selected rows (or every visible row if none is selected)."""
        try:
            sm = self.table.selectionModel()
            rows = sorted({ix.row() for ix in sm.selectedRows()}) if sm is not None else []
            if not rows:
                rows = list(range(self.proxy.rowCount()))
            self._to_clipboard(rows)
        except Exception:
            traceback.print_exc()

    def copy_visible_rows(self):
        """'Copy table': copies every row currently shown (respects the filter)."""
        try:
            self._to_clipboard(list(range(self.proxy.rowCount())))
        except Exception:
            traceback.print_exc()

    def _flash(self, msg, ms=2600):
        self._flash_token += 1
        token = self._flash_token
        self.lbl_summary.setText(str(msg))
        QTimer.singleShot(int(ms), lambda: self._end_flash(token))

    def _end_flash(self, token):
        try:
            if token != self._flash_token:
                return
            if self._order:
                self._update_summary()
            else:
                self.lbl_summary.setText("")
        except Exception:
            pass

    # ------------------------------------------------------------------ theme
    def _theme_name(self):
        th = getattr(self._owner, "theme", None) if self._owner is not None else None
        if th is None:
            try:
                th = getattr(self.window(), "theme", None)
            except Exception:
                th = None
        th = str(th or "light").strip().lower()
        return "dark" if th == "dark" else "light"

    def _font_sizes(self):
        """Title/badge sizes relative to the application font (9 pt on Windows)."""
        try:
            base = float(QFontInfo(self.font()).pointSizeF())
        except Exception:
            base = 9.0
        if not (base > 0):
            base = 9.0
        return {
            "title_pt": f"{int(round(base + 2))}pt",
            "badge_pt": f"{max(7, int(round(base - 1)))}pt",
            "empty_pt": f"{int(round(base + 1))}pt",
        }

    def sync_theme(self, force=False):
        """Apply the light/dark colors that match MainWindow.theme (no-op if unchanged)."""
        self._theme_pending = False
        try:
            theme = self._theme_name()
            if not force and theme == self._applied_theme:
                return
            self._applied_theme = theme
            self._pal = dict(_FW_PP_PALETTES.get(theme, _FW_PP_PALETTES["light"]))
            self.setStyleSheet(_FW_PP_QSS.substitute(self._pal, **self._font_sizes()))
            if self._order and self._current in self._datasets:
                self._fill_table()
        except Exception:
            traceback.print_exc()

    def changeEvent(self, e):
        # Settings > UI theme replaces the application style sheet; follow it.
        try:
            if e.type() in (QEvent.StyleChange, QEvent.PaletteChange) and not self._theme_pending:
                self._theme_pending = True
                QTimer.singleShot(0, lambda: self.sync_theme())
        except Exception:
            pass
        super().changeEvent(e)

    # ------------------------------------------- legacy QTextEdit-like API
    def setPlainText(self, text):
        self._legacy_text = "" if text is None else str(text)
        if not self._order:
            msg = self._legacy_text.strip()
            self.lbl_empty_text.setText(msg if msg else self._empty_text)

    def toPlainText(self):
        return self._legacy_text

    def append(self, text):
        t = "" if text is None else str(text)
        self._legacy_text = (self._legacy_text + "\n" + t) if self._legacy_text else t
        last = t.strip().splitlines()[-1] if t.strip() else ""
        if last:
            if self._order:
                self._flash(last, ms=5000)
            else:
                self.lbl_empty_text.setText(last)

    def clear(self):
        self._legacy_text = ""

    def setReadOnly(self, flag=True):
        return None

    def isReadOnly(self):
        return True

# endregion



# region # === Block 6C: Draggable divider between the parameters table and the plot ===
"""
Divider of the "Granulometric parameters" tab: drag it up or down to give more
room to the table or to the plot. Double click on it restores the default sizes.
The position is remembered between sessions (QSettings).
"""
from PyQt5.QtWidgets import QSplitter, QSplitterHandle


class _FwSplitterHandle(QSplitterHandle):
    """Divider with a small visible grip."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._hover = False
        self._down = False
        self.setMouseTracking(True)
        self.setToolTip("Drag to resize the table and the plot (double click: default sizes)")
        self.setCursor(Qt.SplitVCursor if orientation == Qt.Vertical else Qt.SplitHCursor)

    def enterEvent(self, e):
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._down = True
        self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self._down = False
        self.update()
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        try:
            sp = self.splitter()
            if hasattr(sp, "reset_sizes"):
                sp.reset_sizes()
                e.accept()
                return
        except Exception:
            pass
        super().mouseDoubleClickEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            try:
                dark = str(getattr(self.window(), "theme", "light")).strip().lower() == "dark"
            except Exception:
                dark = False
            if dark:
                col = "#9aa3ad" if (self._hover or self._down) else "#5d6369"
            else:
                col = "#64748b" if (self._hover or self._down) else "#c3ccd8"
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(col)))
            r = self.rect()
            if self.orientation() == Qt.Vertical:
                w, h = 46.0, 4.0
            else:
                w, h = 4.0, 46.0
            x = r.x() + (r.width() - w) / 2.0
            y = r.y() + (r.height() - h) / 2.0
            p.drawRoundedRect(QRectF(x, y, w, h), 2.0, 2.0)
        finally:
            p.end()


class FwSplitter(QSplitter):
    """QSplitter with a visible grip, no collapsing, and remembered position."""

    def __init__(self, orientation, owner=None, settings_key=None, default_ratio=0.5, parent=None):
        super().__init__(orientation, parent)
        self._owner = owner
        self._settings_key = settings_key
        self._default_ratio = float(default_ratio)
        self.setObjectName("fwSplitter")
        self.setChildrenCollapsible(False)
        self.setHandleWidth(12)
        self.setOpaqueResize(True)
        self.splitterMoved.connect(lambda *_: self._save_state())

    def createHandle(self):
        return _FwSplitterHandle(self.orientation(), self)

    def _settings(self):
        return getattr(self._owner, "settings", None) if self._owner is not None else None

    def finish_setup(self):
        """Call after adding the widgets: restores the saved position or applies the default one."""
        if not self._restore_state():
            self.reset_sizes(save=False)

    def reset_sizes(self, save=True):
        r = min(0.9, max(0.1, self._default_ratio))
        total = self.height() if self.orientation() == Qt.Vertical else self.width()
        if self.isVisible() and total > 100:
            total -= self.handleWidth()
        else:
            total = 1000          # before the first show only the proportion matters
        a = int(round(total * r))
        self.setSizes([a, max(1, total - a)])
        if save:
            self._save_state()

    def _save_state(self):
        st = self._settings()
        if st is None or not self._settings_key:
            return
        try:
            st.setValue(self._settings_key, self.saveState())
        except Exception:
            pass

    def _restore_state(self):
        st = self._settings()
        if st is None or not self._settings_key:
            return False
        try:
            val = st.value(self._settings_key, None)
            if val is None:
                return False
            ok = bool(self.restoreState(val))
            if ok:
                sizes = self.sizes()
                if len(sizes) != self.count() or min(sizes or [0]) <= 0:
                    return False
            return ok
        except Exception:
            return False

# endregion



# region # === Block 6D: Groups (data labels or custom), colors, symbols, plot viewer and exact export ===
"""
Granulometric parameters tab:
  - "Groups" button: choose how samples are grouped (data labels of the files, or
    custom groups made by the user), the color and symbol of each group, which
    groups are shown in the plot and which ones get an envelope.
    The loaded files are never modified; custom groups are saved in the project.
  - "View plot" button: opens the plot in its own window at the exact size and
    quality of the exported image (square by default). Size, DPI, texts and legend
    set there are used for the export only; point size, colors and symbols are
    also applied to the plot of the main window.
  - "Export image": saves the plot with the size and DPI chosen in "View plot".
"""
import json as _fw_json
import math as _fw_math

from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPolygonF, QImage
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QRadioButton, QTableWidget, QTableWidgetItem, QComboBox,
    QToolButton, QColorDialog, QMessageBox, QScrollArea, QGroupBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QFileDialog, QGridLayout, QTabBar, QTabWidget, QFormLayout
)
from matplotlib.figure import Figure as _FwFigure
from matplotlib.backends.backend_agg import FigureCanvasAgg as _FwCanvasAgg

FW_MARKERS = [
    ("o", "Circle"), ("s", "Square"), ("^", "Triangle up"), ("v", "Triangle down"),
    ("D", "Diamond"), ("p", "Pentagon"), ("h", "Hexagon"), ("*", "Star"),
    ("P", "Plus"), ("X", "Cross"),
]
# Visual size correction so every symbol looks about as big as a circle of the same size.
_FW_MARKER_SCALE = {"o": 1.0, "s": 0.9, "^": 1.1, "v": 1.1, "D": 0.85, "p": 1.0,
                    "h": 1.0, "*": 1.5, "P": 1.05, "X": 1.05}

FW_LEGEND_LOCS = [
    ("best", "Best (automatic)"), ("upper right", "Upper right"), ("upper left", "Upper left"),
    ("lower right", "Lower right"), ("lower left", "Lower left"), ("center right", "Center right"),
    ("upper center", "Upper center"), ("lower center", "Lower center"),
    ("outside right", "Outside, right"), ("outside bottom", "Outside, below"),
]

FW_XY_STYLE_DEFAULTS = {
    # export only (View plot window)
    "width": 16.0, "height": 16.0, "unit": "cm", "dpi": 300,
    "show_title": True, "title_text": "Granulometric parameters plot", "title_size": 12.0,
    "label_size": 11.0, "tick_size": 10.0, "names_size": 8.0,
    "legend_show": True, "legend_loc": "best", "legend_size": 9.0, "legend_frame": True, "legend_cols": 1,
    "grid": True,
    # also applied to the plot of the main window
    "marker_size": 6.5, "edge_width": 0.45, "alpha": 0.95, "show_envelopes": True,
}
FW_XY_SHARED_KEYS = ("marker_size", "edge_width", "alpha", "show_envelopes")


# ------------------------------------------------------------------ helpers
def _fw_sample_id(v) -> str:
    """Stable text id of a sample (1.0 -> '1')."""
    if v is None:
        return ""
    try:
        if isinstance(v, float) and v == v and float(v).is_integer():
            return str(int(v))
    except Exception:
        pass
    return str(v).strip()


def _fw_marker_s(marker, size_pt) -> float:
    """matplotlib scatter 's' (pt^2) for a symbol of diameter size_pt."""
    try:
        d = float(size_pt) * _FW_MARKER_SCALE.get(marker, 1.0)
    except Exception:
        d = 6.5
    return max(1.0, d * d)


def _fw_marker_polygon(marker, cx, cy, r):
    def reg(n, rot, rad=r):
        return [QPointF(cx + rad * _fw_math.cos(rot + 2 * _fw_math.pi * k / n),
                        cy + rad * _fw_math.sin(rot + 2 * _fw_math.pi * k / n)) for k in range(n)]
    up = -_fw_math.pi / 2
    if marker == "s":
        q = r * 0.85
        return QPolygonF([QPointF(cx - q, cy - q), QPointF(cx + q, cy - q),
                          QPointF(cx + q, cy + q), QPointF(cx - q, cy + q)])
    if marker == "^":
        return QPolygonF(reg(3, up))
    if marker == "v":
        return QPolygonF(reg(3, -up))
    if marker == "D":
        return QPolygonF(reg(4, up))
    if marker == "p":
        return QPolygonF(reg(5, up))
    if marker == "h":
        return QPolygonF(reg(6, up))
    if marker == "*":
        pts = []
        for k in range(10):
            rad = r if k % 2 == 0 else r * 0.45
            a = up + _fw_math.pi * k / 5
            pts.append(QPointF(cx + rad * _fw_math.cos(a), cy + rad * _fw_math.sin(a)))
        return QPolygonF(pts)
    if marker in ("P", "X"):
        w = r * 0.38
        base = [(-w, -r), (w, -r), (w, -w), (r, -w), (r, w), (w, w), (w, r), (-w, r),
                (-w, w), (-r, w), (-r, -w), (-w, -w)]
        rot = _fw_math.pi / 4 if marker == "X" else 0.0
        c, s = _fw_math.cos(rot), _fw_math.sin(rot)
        return QPolygonF([QPointF(cx + x * c - y * s, cy + x * s + y * c) for x, y in base])
    return None   # circle


def fw_marker_icon(marker, color, size=12, dpr=1.0):
    """Icon with the group symbol drawn in the group color."""
    dpr = max(1.0, float(dpr or 1.0))
    pm = QPixmap(int(round(size * dpr)), int(round(size * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    try:
        p.setRenderHint(QPainter.Antialiasing, True)
        qc = QColor(_fw_pp_color_name(color))
        p.setPen(QPen(qc.darker(150), 1.0))
        p.setBrush(QBrush(qc))
        r = size / 2.0 - 0.8
        poly = _fw_marker_polygon(marker, size / 2.0, size / 2.0, r)
        if poly is None:
            p.drawEllipse(QRectF(0.8, 0.8, size - 1.6, size - 1.6))
        else:
            p.drawPolygon(poly)
    finally:
        p.end()
    return QIcon(pm)


def _fw_color_icon(color, w=26, h=16, dpr=1.0):
    dpr = max(1.0, float(dpr or 1.0))
    pm = QPixmap(int(round(w * dpr)), int(round(h * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    try:
        p.setRenderHint(QPainter.Antialiasing, True)
        qc = QColor(_fw_pp_color_name(color))
        p.setPen(QPen(qc.darker(150), 1.0))
        p.setBrush(QBrush(qc))
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), 4.0, 4.0)
    finally:
        p.end()
    return QIcon(pm)


def _fw_next_palette_color(used):
    used = {str(c).lower() for c in (used or [])}
    for c in GROUP_COLORS:
        if c.lower() not in used:
            return c
    return GROUP_COLORS[len(used) % len(GROUP_COLORS)]


def _fw_fig_auto_layout(fig):
    """Layout recomputed at every draw, so the plot always fits its canvas."""
    try:
        fig.set_layout_engine("tight")
        return
    except Exception:
        pass
    try:
        fig.set_tight_layout(True)
    except Exception:
        pass


def _fw_below_axes_transform(fig, ax):
    """Transform that places a legend just below the x axis labels (offset in points)."""
    from matplotlib.transforms import ScaledTranslation
    off_pt = 34.0
    try:
        rnd = fig.canvas.get_renderer()
        bb = ax.xaxis.get_tightbbox(rnd)
        abb = ax.get_window_extent(rnd)
        if bb is not None:
            off_pt = max(10.0, (abb.y0 - bb.y0) * 72.0 / float(fig.dpi) + 6.0)
    except Exception:
        pass
    return ax.transAxes + ScaledTranslation(0, -off_pt / 72.0, fig.dpi_scale_trans)


# ------------------------------------------------------------------ plot style
def _fw_xy_style(owner):
    st = getattr(owner, "xy_style", None)
    if not isinstance(st, dict):
        st = dict(FW_XY_STYLE_DEFAULTS)
        try:
            s = getattr(owner, "settings", None)
            raw = s.value("xy_style_json", None) if s is not None else None
            if raw:
                d = _fw_json.loads(str(raw))
                if isinstance(d, dict):
                    st.update({k: v for k, v in d.items() if k in FW_XY_STYLE_DEFAULTS})
        except Exception:
            pass
        owner.xy_style = st
    for k, v in FW_XY_STYLE_DEFAULTS.items():
        st.setdefault(k, v)
    return st


def _fw_save_xy_style(owner):
    try:
        s = getattr(owner, "settings", None)
        if s is not None:
            s.setValue("xy_style_json", _fw_json.dumps(_fw_xy_style(owner)))
    except Exception:
        pass


def _fw_xy_size_in(st):
    try:
        w, h = float(st.get("width", 16.0)), float(st.get("height", 16.0))
    except Exception:
        w, h = 16.0, 16.0
    f = 1.0 if str(st.get("unit", "cm")) == "in" else 1.0 / 2.54
    return max(0.5, w * f), max(0.5, h * f)


def _fw_xy_size_text(st):
    w_in, h_in = _fw_xy_size_in(st)
    dpi = int(st.get("dpi", 300))
    u = str(st.get("unit", "cm"))
    return (f"{float(st.get('width')):g} × {float(st.get('height')):g} {u} · {dpi} dpi · "
            f"{int(round(w_in * dpi))} × {int(round(h_in * dpi))} px")


def fw_draw_xy(owner, fig, style=None, export=True):
    """
    Draws the granulometric parameters plot on fig.
    export=True: every setting of the View plot window (exact export look).
    export=False: plot of the main window (shares points, colors and symbols).
    """
    st = dict(FW_XY_STYLE_DEFAULTS)
    if isinstance(style, dict):
        st.update(style)

    def num(key, main_default):
        if not export:
            return main_default
        try:
            return float(st.get(key))
        except Exception:
            return main_default

    fig.clear()
    try:
        fig.patch.set_facecolor("white")
    except Exception:
        pass
    _fw_fig_auto_layout(fig)
    ax = fig.add_subplot(111)

    xk = owner.cmb_x.currentText().strip() if hasattr(owner, "cmb_x") else "mean"
    yk = owner.cmb_y.currentText().strip() if hasattr(owner, "cmb_y") else "sigma"
    label_fs, tick_fs = num("label_size", None), num("tick_size", None)
    names_fs, legend_fs, title_fs = num("names_size", 8.0), num("legend_size", 8.0), num("title_size", None)

    kw_l = {"fontsize": label_fs} if label_fs else {}
    ax.set_xlabel(owner._axis_label_for_key(xk), **kw_l)
    ax.set_ylabel(owner._axis_label_for_key(yk), **kw_l)
    if tick_fs:
        ax.tick_params(labelsize=tick_fs)
    if (not export) or bool(st.get("grid", True)):
        ax.grid(True, alpha=0.25)
    if export:
        title = str(st.get("title_text", "") or "").strip()
        if bool(st.get("show_title", True)) and title:
            ax.set_title(title, **({"fontsize": title_fs} if title_fs else {}))
    else:
        ax.set_title(tr("xy_plot_title"))

    try:
        pts = list(owner._iter_points_for_plot(xk, yk, tab_name="xy"))
    except Exception:
        traceback.print_exc()
        pts = []
    if not pts:
        ax.text(0.5, 0.5, "No data to plot.", ha="center", va="center", transform=ax.transAxes)
        return ax

    by_group = {}
    for p in pts:
        by_group.setdefault(p["gkey"], []).append(p)
    order = list((getattr(owner, "group_labels", None) or {}).keys())
    keys = [k for k in order if k in by_group] + [k for k in by_group if k not in order]

    if bool(st.get("show_envelopes", True)):
        rib = getattr(owner, "selected_groups_ribbons", None)
        rib = set(rib) if isinstance(rib, (list, tuple, set)) else None
        for gk in keys:
            if rib is not None and gk not in rib:
                continue
            arr = by_group[gk]
            if len(arr) >= 3:
                try:
                    plot_ribbon(ax, np.array([a["x"] for a in arr], dtype=float),
                                np.array([a["y"] for a in arr], dtype=float),
                                arr[0]["color"], alpha=0.18, width=0.08, smooth=12)
                except Exception:
                    pass

    try:
        ms = float(st.get("marker_size", 6.5))
        ew = float(st.get("edge_width", 0.45))
        al = float(st.get("alpha", 0.95))
    except Exception:
        ms, ew, al = 6.5, 0.45, 0.95
    show_names = bool(owner.chk_show_names.isChecked()) if hasattr(owner, "chk_show_names") else False
    for gk in keys:
        arr = by_group[gk]
        m = arr[0].get("marker", "o") or "o"
        ax.scatter([a["x"] for a in arr], [a["y"] for a in arr], s=_fw_marker_s(m, ms), marker=m,
                   color=arr[0]["color"], edgecolors=("black" if ew > 0 else "none"),
                   linewidths=max(0.0, ew), alpha=al, label=arr[0]["glabel"], zorder=3)
        if show_names:
            for a in arr:
                ax.annotate(a["sample"], (a["x"], a["y"]), xytext=(4, 4), textcoords="offset points",
                            fontsize=names_fs, alpha=0.9)

    if (not export) or bool(st.get("legend_show", True)):
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            uniq.setdefault(l, h)
        if uniq:
            loc = str(st.get("legend_loc", "best")) if export else "best"
            try:
                cols = max(1, int(st.get("legend_cols", 1))) if export else 1
            except Exception:
                cols = 1
            kw = dict(fontsize=legend_fs, frameon=bool(st.get("legend_frame", True)) if export else True, ncol=cols)
            if loc == "outside right":
                kw.update(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            elif loc == "outside bottom":
                if cols <= 1:
                    kw["ncol"] = min(4, len(uniq))
                kw.update(loc="upper center", bbox_to_anchor=(0.5, 0.0),
                          bbox_transform=_fw_below_axes_transform(fig, ax), borderaxespad=0.0)
            else:
                kw["loc"] = loc if loc in dict(FW_LEGEND_LOCS) else "best"
            ax.legend(list(uniq.values()), list(uniq.keys()), **kw)
    return ax


_FW_EXPORT_FILTERS = ("PNG image (*.png);;TIFF image (*.tif *.tiff);;JPEG image (*.jpg *.jpeg);;"
                      "SVG vector (*.svg);;PDF document (*.pdf)")
_FW_EXPORT_EXT = {".png": "png", ".tif": "tiff", ".tiff": "tiff", ".jpg": "jpeg", ".jpeg": "jpeg",
                  ".svg": "svg", ".pdf": "pdf"}


def fw_export_xy(owner, parent=None):
    """Saves the plot with the exact size and DPI chosen in View plot."""
    parent = parent or owner
    st = _fw_xy_style(owner)
    w_in, h_in = _fw_xy_size_in(st)
    dpi = int(st.get("dpi", 300))
    start_dir = getattr(owner, "last_dir", None) or get_script_dir()
    file, sel = QFileDialog.getSaveFileName(
        parent, f"Export plot ({_fw_xy_size_text(st)})",
        os.path.join(start_dir, "granulometric_parameters_plot.png"), _FW_EXPORT_FILTERS)
    if not file:
        return None
    ext = os.path.splitext(file)[1].lower()
    if ext not in _FW_EXPORT_EXT:
        s = str(sel or "")
        ext = ".tif" if "TIFF" in s else ".jpg" if "JPEG" in s else ".svg" if "SVG" in s else ".pdf" if "PDF" in s else ".png"
        file += ext
    fmt = _FW_EXPORT_EXT[ext]
    try:
        fig = _FwFigure(figsize=(w_in, h_in), dpi=dpi)
        _FwCanvasAgg(fig)
        fw_draw_xy(owner, fig, st, export=True)
        kw = dict(dpi=dpi, facecolor="white", format=fmt)
        if fmt == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        elif fmt == "jpeg":
            kw["pil_kwargs"] = {"quality": 95}
        try:
            fig.savefig(file, **kw)
        except TypeError:
            kw.pop("pil_kwargs", None)
            fig.savefig(file, **kw)
    except Exception as e:
        QMessageBox.warning(parent, tr("error"), f"Could not export the plot:\n{e}")
        return None
    try:
        owner.last_dir = os.path.dirname(file)
        if getattr(owner, "settings", None) is not None:
            owner.settings.setValue("last_dir", owner.last_dir)
    except Exception:
        pass
    QMessageBox.information(parent, tr("info"), f"Image saved to:\n{file}\n\n{_fw_xy_size_text(st)}")
    return file


# ------------------------------------------------------------------ tabs (selected tab is bold)
class _FwTabBar(QTabBar):
    """The selected tab is drawn bold by the style sheet: reserve that width so the name is not cut."""

    def tabSizeHint(self, index):
        s = super().tabSizeHint(index)
        try:
            t = self.tabText(index)
            f = QFont(self.font())
            f.setWeight(QFont.Bold)
            extra = QFontMetrics(f).horizontalAdvance(t) - self.fontMetrics().horizontalAdvance(t)
            if extra > 0:
                s.setWidth(s.width() + extra + 8)
        except Exception:
            pass
        return s


class FwTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            self.setTabBar(_FwTabBar(self))
        except Exception:
            pass


# ------------------------------------------------------------------ Groups dialog
class FwGroupsDialog(QDialog):
    """Groups: data labels or custom groups, colors, symbols, visibility and envelopes."""

    C_SHOW, C_COLOR, C_SYMBOL, C_NAME, C_COUNT, C_ENV = range(6)

    def __init__(self, owner, show_visibility=True, parent=None):
        super().__init__(parent or owner)
        self.owner = owner
        self.show_visibility = bool(show_visibility)
        self.setWindowTitle("Groups")
        self.resize(1080, 660)
        try:
            self._dpr = float(self.devicePixelRatioF())
        except Exception:
            self._dpr = 1.0
        self._mode = "custom" if str(getattr(owner, "group_mode", "labels")) == "custom" else "labels"
        self._building = False

        cg = dict(getattr(owner, "custom_groups", None) or {})
        self._samples = []
        for bk, df, gcol, scol in owner._fw_group_bases():
            n = len(df)
            gv = _fw_pp_column_list(df, gcol) if gcol in df.columns else ["No Group"] * n
            sv = _fw_pp_column_list(df, scol) if scol in df.columns else [None] * n
            for s, g in zip(sv, gv):
                sid = _fw_sample_id(s)
                label = _norm_group_name(g)
                c = cg.get(f"{bk}|{sid}")
                self._samples.append({"base": bk, "sid": sid, "label": label,
                                      "custom": _norm_group_name(c) if c not in (None, "") else label})

        self._known = set((getattr(owner, "group_labels", None) or {}).keys())
        vis = list(getattr(owner, "selected_groups_xy", []) or [])
        self._vis = set(vis) if vis else None
        rib = getattr(owner, "selected_groups_ribbons", None)
        self._rib = set(rib) if isinstance(rib, (list, tuple, set)) else None
        self._style = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)
        t = QLabel("Groups")
        f = QFont(t.font())
        f.setPointSizeF(max(9.0, f.pointSizeF() + 3))
        f.setWeight(QFont.DemiBold)
        t.setFont(f)
        root.addWidget(t)
        cap = QLabel("Choose how samples are grouped and how each group looks in the plots. "
                     "Your files are never modified.")
        cap.setWordWrap(True)
        root.addWidget(cap)

        box = QGroupBox("Group samples by")
        bl = QVBoxLayout(box)
        self.rb_labels = QRadioButton("Data labels  (the Group column of the loaded files, default)")
        self.rb_custom = QRadioButton("Custom groups  (put each sample in the group you want)")
        (self.rb_custom if self._mode == "custom" else self.rb_labels).setChecked(True)
        self.rb_labels.toggled.connect(lambda on: self._set_mode("labels" if on else "custom"))
        bl.addWidget(self.rb_labels)
        bl.addWidget(self.rb_custom)
        root.addWidget(box)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        gl = QGroupBox("Groups")
        gv_l = QVBoxLayout(gl)
        self.tbl_g = QTableWidget(0, 6)
        self.tbl_g.setHorizontalHeaderLabels(["Show", "Color", "Symbol", "Group", "Samples", "Envelope"])
        self.tbl_g.verticalHeader().setVisible(False)
        self.tbl_g.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_g.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_g.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        hh = self.tbl_g.horizontalHeader()
        for c in range(6):
            hh.setSectionResizeMode(c, QHeaderView.Stretch if c == self.C_NAME else QHeaderView.ResizeToContents)
        self.tbl_g.verticalHeader().setDefaultSectionSize(max(30, self.fontMetrics().height() + 14))
        self.tbl_g.setColumnHidden(self.C_SHOW, not self.show_visibility)
        self.tbl_g.setColumnHidden(self.C_ENV, not self.show_visibility)
        self.tbl_g.itemChanged.connect(self._on_group_item)
        gv_l.addWidget(self.tbl_g, 1)
        self.lbl_g_hint = QLabel("")
        self.lbl_g_hint.setWordWrap(True)
        gv_l.addWidget(self.lbl_g_hint)
        rowb = QHBoxLayout()
        b1 = QPushButton("Different symbols")
        b1.setToolTip("Give each group a different symbol (useful for black and white figures)")
        b1.clicked.connect(lambda: self._auto_symbols(True))
        b2 = QPushButton("All circles")
        b2.clicked.connect(lambda: self._auto_symbols(False))
        b3 = QPushButton("New colors")
        b3.setToolTip("Give every group a different color from the palette")
        b3.clicked.connect(self._auto_colors)
        for b in (b1, b2, b3):
            rowb.addWidget(b)
        if self.show_visibility:
            b4 = QPushButton("Show all")
            b4.clicked.connect(self._show_all)
            rowb.addWidget(b4)
        rowb.addStretch(1)
        gv_l.addLayout(rowb)
        split.addWidget(gl)

        sl = QGroupBox("Samples")
        sv_l = QVBoxLayout(sl)
        self.txt_filter = QLineEdit()
        self.txt_filter.setPlaceholderText("Filter by sample, dataset or group")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.textChanged.connect(lambda *_: self._apply_filter())
        sv_l.addWidget(self.txt_filter)
        self.tbl_s = QTableWidget(0, 4)
        self.tbl_s.setHorizontalHeaderLabels(["Sample", "Dataset", "Data label", "Group"])
        self.tbl_s.verticalHeader().setVisible(False)
        self.tbl_s.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_s.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl_s.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        hs = self.tbl_s.horizontalHeader()
        for c in range(4):
            hs.setSectionResizeMode(c, QHeaderView.Stretch if c == 3 else QHeaderView.ResizeToContents)
        self.tbl_s.itemChanged.connect(self._on_sample_item)
        sv_l.addWidget(self.tbl_s, 1)
        mv = QHBoxLayout()
        self.lbl_move = QLabel("Move selected samples to:")
        mv.addWidget(self.lbl_move)
        self.cmb_target = QComboBox()
        self.cmb_target.setEditable(True)
        self.cmb_target.setMinimumWidth(170)
        try:
            self.cmb_target.lineEdit().setPlaceholderText("pick or type a new group")
        except Exception:
            pass
        mv.addWidget(self.cmb_target, 1)
        self.btn_move = QPushButton("Move")
        self.btn_move.clicked.connect(self._move_selected)
        mv.addWidget(self.btn_move)
        sv_l.addLayout(mv)
        rr = QHBoxLayout()
        self.btn_reset = QPushButton("Back to data labels")
        self.btn_reset.setToolTip("Every sample goes back to the group of its data label")
        self.btn_reset.clicked.connect(self._reset_custom)
        rr.addWidget(self.btn_reset)
        rr.addStretch(1)
        sv_l.addLayout(rr)
        self.lbl_s_hint = QLabel("")
        self.lbl_s_hint.setWordWrap(True)
        sv_l.addWidget(self.lbl_s_hint)
        split.addWidget(sl)
        split.setSizes([560, 480])
        root.addWidget(split, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        root.addWidget(bb)

        self._rebuild()

    # ---- model
    def _grp(self, s):
        return s["custom"] if self._mode == "custom" else s["label"]

    def _groups(self):
        """{key: [base, name, count]} in base order, then name order."""
        tmp = {}
        for s in self._samples:
            name = self._grp(s)
            k = _gkey(name, s["base"])
            if k not in tmp:
                tmp[k] = [s["base"], _norm_group_name(name), 0]
            tmp[k][2] += 1
        border = {"Tamiz": 0, "Laser": 1, "Híbrido": 2}
        keys = sorted(tmp, key=lambda k: (border.get(tmp[k][0], 9), tmp[k][1]))
        return {k: tmp[k] for k in keys}

    def _ensure_style(self, key):
        st = self._style.get(key)
        if st is not None:
            return st
        o = self.owner
        col = (getattr(o, "group_colors", None) or {}).get(key)
        if not col:
            col = _fw_next_palette_color([v["color"] for v in self._style.values()])
        known = key in self._known
        st = {
            "color": col,
            "marker": (getattr(o, "group_markers", None) or {}).get(key, "o"),
            "show": True if (self._vis is None or not known) else (key in self._vis),
            "env": True if (self._rib is None or not known) else (key in self._rib),
        }
        self._style[key] = st
        return st

    # ---- UI
    def _rebuild(self, samples=True):
        self._building = True
        try:
            groups = self._groups()
            custom = self._mode == "custom"
            self.tbl_g.setRowCount(len(groups))
            for r, (k, (bk, name, cnt)) in enumerate(groups.items()):
                st = self._ensure_style(k)
                it = QTableWidgetItem("")
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                it.setCheckState(Qt.Checked if st["show"] else Qt.Unchecked)
                it.setData(Qt.UserRole, k)
                self.tbl_g.setItem(r, self.C_SHOW, it)

                btn = QToolButton()
                btn.setIcon(_fw_color_icon(st["color"], dpr=self._dpr))
                btn.setIconSize(QSize(26, 16))
                btn.setAutoRaise(True)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip("Change the color of this group")
                btn.clicked.connect(lambda _=False, key=k: self._pick_color(key))
                self.tbl_g.setCellWidget(r, self.C_COLOR, btn)

                cmb = QComboBox()
                for m, mname in FW_MARKERS:
                    cmb.addItem(fw_marker_icon(m, st["color"], 14, self._dpr), mname, m)
                idx = max(0, cmb.findData(st["marker"]))
                cmb.setCurrentIndex(idx)
                cmb.currentIndexChanged.connect(lambda i, key=k, c=cmb: self._set_marker(key, c.itemData(i)))
                self.tbl_g.setCellWidget(r, self.C_SYMBOL, cmb)

                itn = QTableWidgetItem(name)
                fl = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                if custom:
                    fl |= Qt.ItemIsEditable
                itn.setFlags(fl)
                itn.setData(Qt.UserRole, k)
                itn.setToolTip(f"{name} ({base_name_display(bk)})" + ("\nDouble click to rename" if custom else ""))
                itn.setIcon(fw_marker_icon(st["marker"], st["color"], 12, self._dpr))
                self.tbl_g.setItem(r, self.C_NAME, itn)
                itn.setText(f"{name}  ·  {base_name_display(bk)}")
                itn.setData(Qt.UserRole + 1, name)

                itc = QTableWidgetItem(str(cnt))
                itc.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                itc.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_g.setItem(r, self.C_COUNT, itc)

                ite = QTableWidgetItem("")
                ite.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                ite.setCheckState(Qt.Checked if st["env"] else Qt.Unchecked)
                ite.setData(Qt.UserRole, k)
                ite.setToolTip("Draw a shaded envelope around this group (needs 3 or more samples)")
                self.tbl_g.setItem(r, self.C_ENV, ite)

            cur = self.cmb_target.currentText()
            names = sorted({g[1] for g in groups.values()}, key=lambda x: x.lower())
            self.cmb_target.clear()
            self.cmb_target.addItems(names)
            self.cmb_target.setEditText(cur if cur else "")

            if samples:
                self.tbl_s.setRowCount(len(self._samples))
                for r, s in enumerate(self._samples):
                    vals = [s["sid"], base_name_display(s["base"]), s["label"], self._grp(s)]
                    for c, v in enumerate(vals):
                        it = QTableWidgetItem(str(v))
                        fl = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                        if c == 3 and custom:
                            fl |= Qt.ItemIsEditable
                        it.setFlags(fl)
                        if c == 0:
                            it.setData(Qt.UserRole, r)
                        if c == 3:
                            k = _gkey(self._grp(s), s["base"])
                            stl = self._ensure_style(k)
                            it.setIcon(fw_marker_icon(stl["marker"], stl["color"], 12, self._dpr))
                            if custom and self._grp(s) != s["label"]:
                                f = QFont(it.font())
                                f.setBold(True)
                                it.setFont(f)
                        self.tbl_s.setItem(r, c, it)
            else:
                for r, s in enumerate(self._samples):
                    it = self.tbl_s.item(r, 3)
                    if it is not None:
                        stl = self._ensure_style(_gkey(self._grp(s), s["base"]))
                        it.setText(self._grp(s))
                        it.setIcon(fw_marker_icon(stl["marker"], stl["color"], 12, self._dpr))
            self._apply_filter()
            for w in (self.lbl_move, self.cmb_target, self.btn_move, self.btn_reset):
                w.setEnabled(custom)
            n_custom = sum(1 for s in self._samples if s["custom"] != s["label"])
            if custom:
                self.lbl_s_hint.setText(
                    "Select samples and move them to a group (pick one or type a new name), or double "
                    f"click the Group cell of a sample. Samples outside their data label: {n_custom}.")
                self.lbl_g_hint.setText("Double click a group name to rename it (same name = merge).")
            else:
                self.lbl_s_hint.setText("Groups come from the data labels. Choose \"Custom groups\" "
                                        "above to change the group of any sample.")
                self.lbl_g_hint.setText("Click the color to change it; pick a symbol from the list.")
        finally:
            self._building = False

    def _apply_filter(self):
        needle = self.txt_filter.text().strip().lower()
        for r in range(self.tbl_s.rowCount()):
            show = True
            if needle:
                show = any(needle in (self.tbl_s.item(r, c).text().lower() if self.tbl_s.item(r, c) else "")
                           for c in range(4))
            self.tbl_s.setRowHidden(r, not show)

    def _later(self, samples=True):
        QTimer.singleShot(0, lambda: self._rebuild(samples=samples))

    # ---- actions
    def _set_mode(self, mode):
        if self._building:
            return
        self._mode = mode
        self._later(True)

    def _pick_color(self, key):
        st = self._ensure_style(key)
        c = QColorDialog.getColor(QColor(_fw_pp_color_name(st["color"])), self, "Group color")
        if c.isValid():
            st["color"] = c.name()
            self._later(True)

    def _set_marker(self, key, marker):
        if self._building or not marker:
            return
        self._ensure_style(key)["marker"] = str(marker)
        self._later(True)

    def _auto_symbols(self, different):
        for i, k in enumerate(self._groups()):
            self._ensure_style(k)["marker"] = FW_MARKERS[i % len(FW_MARKERS)][0] if different else "o"
        self._later(True)

    def _auto_colors(self):
        for i, k in enumerate(self._groups()):
            self._ensure_style(k)["color"] = GROUP_COLORS[i % len(GROUP_COLORS)]
        self._later(True)

    def _show_all(self):
        for k in self._groups():
            self._ensure_style(k)["show"] = True
        self._later(False)

    def _on_group_item(self, it):
        if self._building or it is None:
            return
        k = it.data(Qt.UserRole)
        if not k:
            return
        col = it.column()
        if col in (self.C_SHOW, self.C_ENV):
            self._ensure_style(k)["show" if col == self.C_SHOW else "env"] = (it.checkState() == Qt.Checked)
            return
        if col == self.C_NAME and self._mode == "custom":
            old = it.data(Qt.UserRole + 1)
            new = it.text().split("  ·  ")[0].strip()
            groups = self._groups()
            if k not in groups:
                return
            bk = groups[k][0]
            if not new or new == old:
                self._later(False)
                return
            nk = _gkey(new, bk)
            if nk not in self._style:
                self._style[nk] = dict(self._ensure_style(k))
            for s in self._samples:
                if s["base"] == bk and _norm_group_name(self._grp(s)) == old:
                    s["custom"] = new
            self._later(True)

    def _on_sample_item(self, it):
        if self._building or it is None or it.column() != 3 or self._mode != "custom":
            return
        r = it.row()
        if 0 <= r < len(self._samples):
            s = self._samples[r]
            s["custom"] = it.text().strip() or s["label"]
            self._later(True)

    def _move_selected(self):
        target = self.cmb_target.currentText().strip()
        if not target:
            QMessageBox.information(self, "Groups", "Pick a group or type the name of a new group.")
            return
        rows = sorted({ix.row() for ix in self.tbl_s.selectionModel().selectedRows()})
        rows = [r for r in rows if not self.tbl_s.isRowHidden(r)]
        if not rows:
            QMessageBox.information(self, "Groups", "Select one or more samples in the list first.")
            return
        for r in rows:
            self._samples[r]["custom"] = target
        self._later(True)

    def _reset_custom(self):
        for s in self._samples:
            s["custom"] = s["label"]
        self._later(True)

    def _apply(self):
        groups = self._groups()
        for k in groups:
            self._ensure_style(k)
        if self.show_visibility and not any(self._style[k]["show"] for k in groups):
            QMessageBox.information(self, "Groups", "At least one group must be shown in the plot.")
            return False
        o = self.owner
        o.group_mode = self._mode
        o.custom_groups = {f"{s['base']}|{s['sid']}": s["custom"] for s in self._samples if s["custom"] != s["label"]}
        if not isinstance(getattr(o, "group_colors", None), dict):
            o.group_colors = {}
        if not isinstance(getattr(o, "group_markers", None), dict):
            o.group_markers = {}
        for k, st in self._style.items():
            o.group_colors[k] = st["color"]
            o.group_markers[k] = st["marker"]
        o._update_all_groups_and_colors()
        keys = list((getattr(o, "group_labels", None) or {}).keys())
        if self.show_visibility:
            o.selected_groups_xy = [k for k in keys if self._style.get(k, {}).get("show", True)]
            o.selected_groups_ribbons = [k for k in keys if self._style.get(k, {}).get("env", True)]
            self._vis = set(o.selected_groups_xy)
            self._rib = set(o.selected_groups_ribbons)
        o._fw_known_group_keys = list(keys)
        self._known = set(keys)
        o._fw_refresh_group_views()
        return True

    def _ok(self):
        if self._apply():
            self.accept()


# ------------------------------------------------------------------ View plot window
class FwXYViewDialog(QDialog):
    """The plot exactly as it will be exported, with its size, quality and style controls."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Granulometric parameters plot — View and export")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setModal(False)
        self._loading = False
        self._rows = {}
        self._row_keys = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._render)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(700)
        self._save_timer.timeout.connect(lambda: _fw_save_xy_style(self.owner))

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        body = QHBoxLayout()
        body.setSpacing(10)
        root.addLayout(body, 1)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("fwViewArea")
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setWidgetResizable(False)
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.lbl_img)
        body.addWidget(self.scroll, 1)

        side = QScrollArea()
        side.setWidgetResizable(True)
        side.setFixedWidth(372)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(4, 0, 8, 0)
        lay.setSpacing(10)
        side.setWidget(inner)
        body.addWidget(side)

        # 1) size and quality
        g1 = QGroupBox("Size and quality  (export)")
        f1 = QFormLayout(g1)
        self.sp_w = self._dspin(1.0, 100.0, 0.5, 1)
        self.sp_h = self._dspin(1.0, 100.0, 0.5, 1)
        self.cmb_unit = QComboBox()
        self.cmb_unit.addItem("cm", "cm")
        self.cmb_unit.addItem("in", "in")
        hw = QHBoxLayout()
        hw.addWidget(self.sp_w)
        hw.addWidget(QLabel("×"))
        hw.addWidget(self.sp_h)
        hw.addWidget(self.cmb_unit)
        f1.addRow("Width × height:", hw)
        hr = QHBoxLayout()
        for txt, ratio in (("Square", 1.0), ("4:3", 4 / 3), ("3:2", 1.5), ("16:9", 16 / 9)):
            b = QPushButton(txt)
            b.setToolTip("Keep the width and set the height for this shape")
            b.clicked.connect(lambda _=False, r=ratio: self._set_ratio(r))
            hr.addWidget(b)
        bswap = QPushButton("⇄")
        bswap.setToolTip("Swap width and height")
        bswap.clicked.connect(self._swap)
        hr.addWidget(bswap)
        f1.addRow("Shape:", hr)
        self.sp_dpi = QSpinBox()
        self.sp_dpi.setRange(72, 1200)
        self.sp_dpi.setSingleStep(50)
        self.sp_dpi.setSuffix(" dpi")
        self.sp_dpi.setToolTip("Quality (resolution) of PNG, TIFF and JPEG images. 300 dpi is the usual journal value.")
        f1.addRow("Quality:", self.sp_dpi)
        self.lbl_px = QLabel("")
        f1.addRow("Image:", self.lbl_px)
        lay.addWidget(g1)

        # 2) points and groups
        g2 = QGroupBox("Points and groups  (also in the main plot)")
        f2 = QFormLayout(g2)
        self.sp_ms = self._dspin(1.0, 40.0, 0.5, 1, " pt")
        f2.addRow("Point size:", self.sp_ms)
        self.sp_ew = self._dspin(0.0, 3.0, 0.05, 2, " pt")
        f2.addRow("Outline:", self.sp_ew)
        self.sp_al = self._dspin(0.1, 1.0, 0.05, 2)
        f2.addRow("Opacity:", self.sp_al)
        self.chk_env = QCheckBox("Envelopes")
        self.chk_names = QCheckBox("Sample names")
        he = QHBoxLayout()
        he.addWidget(self.chk_env)
        he.addWidget(self.chk_names)
        he.addStretch(1)
        f2.addRow(he)
        self.grp_box = QWidget()
        self.grp_grid = QGridLayout(self.grp_box)
        self.grp_grid.setContentsMargins(0, 0, 0, 0)
        self.grp_grid.setHorizontalSpacing(6)
        self.grp_grid.setVerticalSpacing(4)
        f2.addRow(self.grp_box)
        bg = QPushButton("Groups…")
        bg.setToolTip("Build the groups (data labels or custom), visibility and envelopes")
        bg.clicked.connect(lambda: _fw_open_groups_dialog(self.owner, True, parent=self))
        f2.addRow(bg)
        lay.addWidget(g2)

        # 3) text and legend
        g3 = QGroupBox("Text and legend  (export)")
        f3 = QFormLayout(g3)
        self.chk_title = QCheckBox("Title")
        self.txt_title = QLineEdit()
        ht = QHBoxLayout()
        ht.addWidget(self.chk_title)
        ht.addWidget(self.txt_title, 1)
        f3.addRow(ht)
        self.sp_title = self._dspin(5.0, 40.0, 0.5, 1, " pt")
        f3.addRow("Title size:", self.sp_title)
        self.sp_label = self._dspin(5.0, 40.0, 0.5, 1, " pt")
        f3.addRow("Axis titles:", self.sp_label)
        self.sp_tick = self._dspin(5.0, 40.0, 0.5, 1, " pt")
        f3.addRow("Axis numbers:", self.sp_tick)
        self.sp_names = self._dspin(4.0, 30.0, 0.5, 1, " pt")
        f3.addRow("Sample names:", self.sp_names)
        self.chk_leg = QCheckBox("Legend")
        self.chk_frame = QCheckBox("Frame")
        self.chk_grid = QCheckBox("Grid")
        hl = QHBoxLayout()
        for w_ in (self.chk_leg, self.chk_frame, self.chk_grid):
            hl.addWidget(w_)
        hl.addStretch(1)
        f3.addRow(hl)
        self.cmb_loc = QComboBox()
        for k, lbl in FW_LEGEND_LOCS:
            self.cmb_loc.addItem(lbl, k)
        f3.addRow("Legend position:", self.cmb_loc)
        self.sp_leg = self._dspin(4.0, 30.0, 0.5, 1, " pt")
        f3.addRow("Legend size:", self.sp_leg)
        self.sp_cols = QSpinBox()
        self.sp_cols.setRange(1, 8)
        f3.addRow("Legend columns:", self.sp_cols)
        lay.addWidget(g3)
        lay.addStretch(1)

        bottom = QHBoxLayout()
        self.lbl_info = QLabel("")
        bottom.addWidget(self.lbl_info, 1)
        self.cmb_zoom = QComboBox()
        self.cmb_zoom.addItem("Fit to window")
        self.cmb_zoom.addItem("Actual pixels (100%)")
        self.cmb_zoom.setToolTip("Fit: whole figure, same proportions. Actual pixels: the real resolution of the image")
        self.cmb_zoom.currentIndexChanged.connect(lambda *_: self.schedule_render())
        bottom.addWidget(self.cmb_zoom)
        b_reset = QPushButton("Reset style")
        b_reset.clicked.connect(self._reset_style)
        bottom.addWidget(b_reset)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.close)
        bottom.addWidget(b_close)
        self.btn_export = QPushButton("Export…")
        self.btn_export.setDefault(True)
        self.btn_export.setStyleSheet("QPushButton { background-color: #2563eb; color: #ffffff; border: 1px solid #2563eb; "
                                      "border-radius: 8px; padding: 6px 18px; font-weight: 600; }"
                                      "QPushButton:hover { background-color: #1d4ed8; }")
        self.btn_export.clicked.connect(lambda: fw_export_xy(self.owner, self))
        bottom.addWidget(self.btn_export)
        root.addLayout(bottom)

        # connections
        self.sp_w.valueChanged.connect(lambda v: self._set("width", float(v)))
        self.sp_h.valueChanged.connect(lambda v: self._set("height", float(v)))
        self.cmb_unit.currentIndexChanged.connect(lambda *_: self._unit_changed())
        self.sp_dpi.valueChanged.connect(lambda v: self._set("dpi", int(v)))
        self.sp_ms.valueChanged.connect(lambda v: self._set("marker_size", float(v)))
        self.sp_ew.valueChanged.connect(lambda v: self._set("edge_width", float(v)))
        self.sp_al.valueChanged.connect(lambda v: self._set("alpha", float(v)))
        self.chk_env.toggled.connect(lambda v: self._set("show_envelopes", bool(v)))
        self.chk_names.toggled.connect(self._names_toggled)
        self.chk_title.toggled.connect(lambda v: self._set("show_title", bool(v)))
        self.txt_title.textChanged.connect(lambda t: self._set("title_text", str(t)))
        self.sp_title.valueChanged.connect(lambda v: self._set("title_size", float(v)))
        self.sp_label.valueChanged.connect(lambda v: self._set("label_size", float(v)))
        self.sp_tick.valueChanged.connect(lambda v: self._set("tick_size", float(v)))
        self.sp_names.valueChanged.connect(lambda v: self._set("names_size", float(v)))
        self.chk_leg.toggled.connect(lambda v: self._set("legend_show", bool(v)))
        self.chk_frame.toggled.connect(lambda v: self._set("legend_frame", bool(v)))
        self.chk_grid.toggled.connect(lambda v: self._set("grid", bool(v)))
        self.cmb_loc.currentIndexChanged.connect(lambda i: self._set("legend_loc", str(self.cmb_loc.itemData(i))))
        self.sp_leg.valueChanged.connect(lambda v: self._set("legend_size", float(v)))
        self.sp_cols.valueChanged.connect(lambda v: self._set("legend_cols", int(v)))

        try:
            ag = QApplication.primaryScreen().availableGeometry()
            self.resize(int(min(1280, ag.width() * 0.86)), int(min(900, ag.height() * 0.88)))
        except Exception:
            self.resize(1200, 820)
        self._load_controls()
        self._apply_theme()

    def _dspin(self, lo, hi, step, dec, suffix=""):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(dec)
        if suffix:
            sp.setSuffix(suffix)
        sp.setKeyboardTracking(False)
        return sp

    def _apply_theme(self):
        dark = str(getattr(self.owner, "theme", "light")).lower() == "dark"
        bg = "#1b1d1f" if dark else "#e6eaf0"
        self.scroll.setStyleSheet(f"QScrollArea#fwViewArea {{ background: {bg}; border: none; border-radius: 8px; }}"
                                  f"QScrollArea#fwViewArea > QWidget > QWidget {{ background: {bg}; }}")

    def changeEvent(self, e):
        try:
            if e.type() == QEvent.StyleChange:
                QTimer.singleShot(0, self._apply_theme)
        except Exception:
            pass
        super().changeEvent(e)

    # ---- controls <-> style
    def _load_controls(self):
        st = _fw_xy_style(self.owner)
        self._loading = True
        try:
            self.cmb_unit.setCurrentIndex(1 if str(st["unit"]) == "in" else 0)
            self.sp_w.setValue(float(st["width"]))
            self.sp_h.setValue(float(st["height"]))
            self.sp_dpi.setValue(int(st["dpi"]))
            self.sp_ms.setValue(float(st["marker_size"]))
            self.sp_ew.setValue(float(st["edge_width"]))
            self.sp_al.setValue(float(st["alpha"]))
            self.chk_env.setChecked(bool(st["show_envelopes"]))
            self.chk_names.setChecked(bool(self.owner.chk_show_names.isChecked()) if hasattr(self.owner, "chk_show_names") else False)
            self.chk_title.setChecked(bool(st["show_title"]))
            self.txt_title.setText(str(st["title_text"]))
            self.sp_title.setValue(float(st["title_size"]))
            self.sp_label.setValue(float(st["label_size"]))
            self.sp_tick.setValue(float(st["tick_size"]))
            self.sp_names.setValue(float(st["names_size"]))
            self.chk_leg.setChecked(bool(st["legend_show"]))
            self.chk_frame.setChecked(bool(st["legend_frame"]))
            self.chk_grid.setChecked(bool(st["grid"]))
            self.cmb_loc.setCurrentIndex(max(0, self.cmb_loc.findData(str(st["legend_loc"]))))
            self.sp_leg.setValue(float(st["legend_size"]))
            self.sp_cols.setValue(int(st["legend_cols"]))
        finally:
            self._loading = False
        self._update_px()

    def _update_px(self):
        st = _fw_xy_style(self.owner)
        self.lbl_px.setText(_fw_xy_size_text(st).split(" · ")[-1])
        self.lbl_info.setText("Exported image: " + _fw_xy_size_text(st))

    def _set(self, key, value):
        if self._loading:
            return
        st = _fw_xy_style(self.owner)
        st[key] = value
        self._update_px()
        self._save_timer.start()
        if key in FW_XY_SHARED_KEYS:
            self.owner.plot_xy()           # main plot + this view
        else:
            self.schedule_render()

    def _unit_changed(self):
        if self._loading:
            return
        st = _fw_xy_style(self.owner)
        new = str(self.cmb_unit.currentData())
        if new == st.get("unit"):
            return
        f = 1 / 2.54 if new == "in" else 2.54
        st["unit"] = new
        st["width"] = round(float(st["width"]) * f, 2)
        st["height"] = round(float(st["height"]) * f, 2)
        self._load_controls()
        self._save_timer.start()
        self.schedule_render()

    def _set_ratio(self, ratio):
        st = _fw_xy_style(self.owner)
        st["height"] = round(float(st["width"]) / float(ratio), 2)
        self._load_controls()
        self._save_timer.start()
        self.schedule_render()

    def _swap(self):
        st = _fw_xy_style(self.owner)
        st["width"], st["height"] = st["height"], st["width"]
        self._load_controls()
        self._save_timer.start()
        self.schedule_render()

    def _names_toggled(self, on):
        if self._loading:
            return
        chk = getattr(self.owner, "chk_show_names", None)
        if chk is not None and chk.isChecked() != bool(on):
            chk.setChecked(bool(on))      # redraws the main plot and this view

    def _reset_style(self):
        keep = {k: _fw_xy_style(self.owner)[k] for k in ("width", "height", "unit", "dpi")}
        self.owner.xy_style = dict(FW_XY_STYLE_DEFAULTS)
        self.owner.xy_style.update(keep)
        _fw_save_xy_style(self.owner)
        self._load_controls()
        self.owner.plot_xy()

    # ---- groups (colors and symbols)
    def _plotted_keys(self):
        try:
            xk = self.owner.cmb_x.currentText().strip()
            yk = self.owner.cmb_y.currentText().strip()
            seen = []
            for p in self.owner._iter_points_for_plot(xk, yk, tab_name="xy"):
                if p["gkey"] not in seen:
                    seen.append(p["gkey"])
            order = list((getattr(self.owner, "group_labels", None) or {}).keys())
            return [k for k in order if k in seen] + [k for k in seen if k not in order]
        except Exception:
            return []

    def _refresh_groups(self):
        keys = self._plotted_keys()
        colors = getattr(self.owner, "group_colors", None) or {}
        markers = getattr(self.owner, "group_markers", None) or {}
        labels = getattr(self.owner, "group_labels", None) or {}
        try:
            dpr = float(self.devicePixelRatioF())
        except Exception:
            dpr = 1.0
        if keys != self._row_keys:
            while self.grp_grid.count():
                it = self.grp_grid.takeAt(0)
                w_ = it.widget()
                if w_ is not None:
                    w_.hide()
                    w_.deleteLater()
            self._rows = {}
            for r, k in enumerate(keys):
                btn = QToolButton()
                btn.setAutoRaise(True)
                btn.setIconSize(QSize(26, 16))
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip("Change the color of this group")
                btn.clicked.connect(lambda _=False, key=k: self._pick_color(key))
                cmb = QComboBox()
                for m, mname in FW_MARKERS:
                    cmb.addItem(mname, m)
                cmb.currentIndexChanged.connect(lambda i, key=k, c=cmb: self._set_marker(key, c.itemData(i)))
                lbl = QLabel(str(labels.get(k, k)))
                self.grp_grid.addWidget(btn, r, 0)
                self.grp_grid.addWidget(cmb, r, 1)
                self.grp_grid.addWidget(lbl, r, 2)
                self._rows[k] = (btn, cmb, lbl)
            self.grp_grid.setColumnStretch(2, 1)
            self._row_keys = list(keys)
        for k, (btn, cmb, lbl) in self._rows.items():
            col = colors.get(k, GROUP_COLORS[0])
            mk = markers.get(k, "o")
            btn.setIcon(_fw_color_icon(col, dpr=dpr))
            cmb.blockSignals(True)
            for i, (m, _n) in enumerate(FW_MARKERS):
                cmb.setItemIcon(i, fw_marker_icon(m, col, 14, dpr))
            cmb.setCurrentIndex(max(0, cmb.findData(mk)))
            cmb.blockSignals(False)
            lbl.setText(str(labels.get(k, k)))

    def _pick_color(self, key):
        colors = getattr(self.owner, "group_colors", None) or {}
        c = QColorDialog.getColor(QColor(_fw_pp_color_name(colors.get(key, GROUP_COLORS[0]))), self, "Group color")
        if c.isValid():
            self.owner.group_colors[key] = c.name()
            QTimer.singleShot(0, self.owner._fw_refresh_group_views)

    def _set_marker(self, key, marker):
        if not marker:
            return
        if not isinstance(getattr(self.owner, "group_markers", None), dict):
            self.owner.group_markers = {}
        self.owner.group_markers[key] = str(marker)
        QTimer.singleShot(0, self.owner._fw_refresh_group_views)

    # ---- rendering
    def refresh_from_owner(self):
        chk = getattr(self.owner, "chk_show_names", None)
        if chk is not None:
            self._loading = True
            try:
                self.chk_names.setChecked(bool(chk.isChecked()))
            finally:
                self._loading = False
        self._refresh_groups()
        self.schedule_render()

    def schedule_render(self):
        self._timer.start()

    def showEvent(self, e):
        super().showEvent(e)
        self.schedule_render()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.cmb_zoom.currentIndex() == 0:
            self.schedule_render()

    def _render(self):
        if not self.isVisible():
            return
        st = _fw_xy_style(self.owner)
        w_in, h_in = _fw_xy_size_in(st)
        dpi = float(st.get("dpi", 300))
        try:
            dpr = max(1.0, float(self.scroll.devicePixelRatioF()))
        except Exception:
            dpr = 1.0
        actual = self.cmb_zoom.currentIndex() == 1
        if actual:
            rdpi = dpi
            disp_w = w_in * dpi / dpr
        else:
            vw = max(120, self.scroll.viewport().width() - 28)
            vh = max(120, self.scroll.viewport().height() - 28)
            disp_w = min(float(vw), vh * w_in / h_in)
            rdpi = min(dpi, disp_w * dpr / w_in)
        rdpi = max(20.0, rdpi)
        try:
            fig = _FwFigure(figsize=(w_in, h_in), dpi=rdpi)
            canvas = _FwCanvasAgg(fig)
            fw_draw_xy(self.owner, fig, st, export=True)
            canvas.draw()
            wpx, hpx = canvas.get_width_height()
            buf = bytes(canvas.buffer_rgba())
            img = QImage(buf, int(wpx), int(hpx), QImage.Format_RGBA8888).copy()
        except Exception:
            traceback.print_exc()
            return
        pm = QPixmap.fromImage(img)
        if not actual:
            tw = max(1, int(round(disp_w * dpr)))
            if abs(pm.width() - tw) > 1:
                pm = pm.scaledToWidth(tw, Qt.SmoothTransformation)
        pm.setDevicePixelRatio(dpr)
        self.lbl_img.setPixmap(pm)
        self.lbl_img.resize(int(round(pm.width() / dpr)), int(round(pm.height() / dpr)))


# ------------------------------------------------------------------ MainWindow methods (bound at the end of the file)
def _fw_effective_group(self, base_key, sample, label):
    """Group of a sample: its data label, or its custom group (Groups dialog)."""
    if str(getattr(self, "group_mode", "labels")) == "custom":
        cg = getattr(self, "custom_groups", None)
        if cg:
            try:
                v = cg.get(f"{base_key_canon(base_key)}|{_fw_sample_id(sample)}")
            except Exception:
                v = None
            if v not in (None, ""):
                return v
    return label


def _fw_group_bases(self):
    """[(base_key, params DataFrame, group column, sample column)] of the loaded datasets."""
    out = []
    for bk, attr in (("Tamiz", "param_results"), ("Laser", "param_results_laser"), ("Híbrido", "param_results_hybrid")):
        df = getattr(self, attr, None)
        if df is None or getattr(df, "empty", True):
            continue
        cols = list(df.columns)
        gcol = "Group" if bk == "Laser" else getattr(self, "group_col", "Group")
        if gcol not in cols and "Group" in cols:
            gcol = "Group"
        if bk == "Híbrido":
            scol = next((c for c in ("Hybrid Sample", "Sample", "Tamiz Sample") if c in cols), cols[0])
        else:
            scol = "Sample" if "Sample" in cols else cols[0]
        out.append((bk, df, gcol, scol))
    return out


def _fw_ensure_group_colors(self):
    if not isinstance(getattr(self, "group_colors", None), dict):
        self.group_colors = {}
    keys = list((getattr(self, "group_labels", None) or {}).keys())
    used = [self.group_colors[k] for k in keys if k in self.group_colors]
    for k in keys:
        if k not in self.group_colors:
            c = _fw_next_palette_color(used)
            self.group_colors[k] = c
            used.append(c)


def _fw_refresh_group_views(self):
    for fn in ("update_console", "plot_xy", "plot_graphic"):
        try:
            f = getattr(self, fn, None)
            if callable(f):
                f()
        except Exception:
            traceback.print_exc()


def _fw_open_groups_dialog(self, show_visibility=True, parent=None):
    if not self._fw_group_bases():
        QMessageBox.information(parent or self, "Groups", "Load a dataset first.")
        return
    try:
        self._update_all_groups_and_colors()
    except Exception:
        pass
    dlg = FwGroupsDialog(self, show_visibility=show_visibility, parent=parent or self)
    dlg.exec_()


def _fw_open_xy_view(self):
    dlg = getattr(self, "_xy_view_dlg", None)
    if dlg is None:
        dlg = FwXYViewDialog(self)
        self._xy_view_dlg = dlg
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    dlg.refresh_from_owner()

# endregion



# region # === Block 7: MainWindow class – constructor, icon, and main menu (EN-only, no histogram) ===
from PyQt5.QtWidgets import QApplication

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- Language is fixed to English (EN-only build) ----
        self.lang = "en"
        try:
            qApp.lang = self.lang
        except Exception:
            pass

        # QSettings (restore preferences)
        try:
            from PyQt5.QtCore import QSettings
            self.settings = QSettings("LAAN", "FW-Granulometry")
        except Exception:
            self.settings = None

        self.setWindowTitle(tr("app_title"))

        # Initial size relative to screen (avoid huge windows)
        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                ag = screen.availableGeometry()
                w = max(1100, int(ag.width() * 0.88))
                h = max(720, int(ag.height() * 0.88))
                self.resize(w, h)
                self.setMinimumSize(1000, 680)
            else:
                self.resize(1280, 860)
                self.setMinimumSize(1000, 680)
        except Exception:
            self.resize(1280, 860)
            self.setMinimumSize(1000, 680)

        # Defaults
        self.theme = "light"
        self.current_method = "folkward"
        self.hybrid_method = None

        # Restore from QSettings if available
        if self.settings is not None:
            self.theme = self.settings.value("theme", self.theme)
            self.current_method = self.settings.value("method", self.current_method)
            self.hybrid_method = self.settings.value("hybrid_method", self.hybrid_method)
            self.last_dir = self.settings.value("last_dir", get_script_dir())
        else:
            self.last_dir = get_script_dir()

        # DataFrames / results
        self.df_data = None
        self.param_results = None
        self.group_col = "Group"

        self.df_laser = None
        self.param_results_laser = None

        self.df_hybrid = None
        self.param_results_hybrid = None

        # Groups and colors (stable keys: _gkey)
        self.groups = []
        self.group_labels = {}
        self.group_colors = {}
        self.selected_groups_xy = []
        self.selected_groups_ribbons = []

        # Initial theme
        try:
            qApp.setStyleSheet(LIGHT_STYLESHEET if self.theme == "light" else DARK_STYLESHEET)
        except Exception:
            pass

        # App/window icon (icon only, no banner)
        self._apply_icon()

        # Simple central container: tabs only (no banner image)
        self._init_central_no_banner()

        # Menu and tabs
        self.initMenu()
        self.initUI_tabs()

    # ---------- Icon ----------
    def _apply_icon(self):
        """
        Search icon in ICON_CANDIDATES (Block 2) and apply it to app and window.
        On Windows, .ico is recommended. If missing, use .png/.jpg.
        """
        try:
            from PyQt5.QtGui import QIcon
        except Exception:
            return

        icon_path = None
        base = get_script_dir()
        for name in globals().get("ICON_CANDIDATES", ["app.ico", "icon.ico", "logo.ico", "logo.png", "icon.png", "icono.nuevo.png"]):
            p = os.path.join(base, name)
            if os.path.exists(p):
                icon_path = p
                break
        if icon_path:
            ico = QIcon(icon_path)
            QApplication.setWindowIcon(ico)
            self.setWindowIcon(ico)

    # ---------- Center (no banner) ----------
    def _init_central_no_banner(self):
        """
        Creates the central QWidget and a vertical layout containing ONLY the QTabWidget.
        No flyer, no banner, no image.
        """
        self._central = QWidget(self)
        self._root = QVBoxLayout(self._central)
        self._root.setContentsMargins(6, 6, 6, 6)
        self._root.setSpacing(6)

        self.tabs = FwTabWidget(self._central)
        self._root.addWidget(self.tabs)

        self.setCentralWidget(self._central)

    # ---------- Main menu ----------
    def initMenu(self):
        menubar = self.menuBar()

        # --- File ---
        self.menu_file = menubar.addMenu(tr("menu_file"))

        self.act_load_tamiz = QAction(tr("file_load_sieve"), self)
        self.act_load_tamiz.triggered.connect(self.load_file)
        self.menu_file.addAction(self.act_load_tamiz)

        self.act_load_laser = QAction(tr("file_load_laser"), self)
        self.act_load_laser.triggered.connect(self.load_laser_file)
        self.menu_file.addAction(self.act_load_laser)

        self.act_save = QAction(tr("file_export_image"), self)
        self.act_save.triggered.connect(self.save_current_tab)
        self.menu_file.addAction(self.act_save)

        self.act_combine = QAction(tr("file_combine"), self)
        self.act_combine.triggered.connect(self.combinar_tamiz_laser)
        self.menu_file.addAction(self.act_combine)

        # --- Settings ---
        self.menu_cfg = menubar.addMenu(tr("menu_config"))
        self.act_cfg_method = QAction(tr("cfg_method"), self, triggered=self.choose_method)
        self.act_cfg_colors = QAction(tr("cfg_colors"), self, triggered=self.edit_colors)
        self.act_cfg_groups_all = QAction(tr("cfg_groups_all"), self, triggered=self.select_groups_all)
        self.act_cfg_ribbons = QAction(tr("cfg_ribbons"), self, triggered=self.select_ribbons)
        self.act_cfg_theme = QAction(tr("cfg_theme"), self, triggered=self.choose_theme)

        self.menu_cfg.addAction(self.act_cfg_method)
        self.menu_cfg.addAction(self.act_cfg_colors)
        self.menu_cfg.addAction(self.act_cfg_groups_all)
        self.menu_cfg.addAction(self.act_cfg_ribbons)
        self.menu_cfg.addAction(self.act_cfg_theme)

        # --- Database ---
        self.menu_db = menubar.addMenu(tr("menu_db"))

        self.act_viewdb_tamiz = QAction(tr("db_view_sieve"), self)
        self.act_viewdb_tamiz.setEnabled(False)
        self.act_viewdb_tamiz.triggered.connect(self.show_tamiz_db_window)
        self.menu_db.addAction(self.act_viewdb_tamiz)

        self.act_viewdb_laser = QAction(tr("db_view_laser"), self)
        self.act_viewdb_laser.setEnabled(False)
        self.act_viewdb_laser.triggered.connect(self.show_laser_db_window)
        self.menu_db.addAction(self.act_viewdb_laser)

        self.act_viewdb_hybrid = QAction(tr("db_view_hybrid"), self)
        self.act_viewdb_hybrid.setEnabled(False)
        self.act_viewdb_hybrid.triggered.connect(self.show_hybrid_db_window)
        self.menu_db.addAction(self.act_viewdb_hybrid)

        # --- Help ---
        self.menu_help = menubar.addMenu(tr("menu_help"))
        self.act_help_about = QAction(
            tr("help_about"), self,
            triggered=lambda: QMessageBox.information(self, tr("about_title"), tr("about_text"))
        )
        self.menu_help.addAction(self.act_help_about)

        self._update_db_menu_actions()

    def _update_db_menu_actions(self):
        has_tamiz = (self.df_data is not None) and (not getattr(self.df_data, "empty", True))
        has_laser = (self.df_laser is not None) and (not getattr(self.df_laser, "empty", True))
        has_hybrid = (self.df_hybrid is not None) and (not getattr(self.df_hybrid, "empty", True))
        try:
            self.act_viewdb_tamiz.setEnabled(has_tamiz)
            self.act_viewdb_laser.setEnabled(has_laser)
            self.act_viewdb_hybrid.setEnabled(has_hybrid)
        except Exception:
            pass

    # ---------- Groups/colors update ----------
    def _update_all_groups_and_colors(self):
        """
        Collect the groups of the loaded datasets (Sieve/Laser/Hybrid) using the
        grouping chosen in the Groups dialog (data labels or custom groups), and ensure:
          - self.group_labels / self.groups (legend and dialog labels, key order)
          - self.group_colors (a different palette color for every new group)
          - selections: groups the user hid stay hidden, new groups are shown
        """
        prev_known = getattr(self, "_fw_known_group_keys", None)
        self.group_labels = {}
        key_order, labels, counts = [], [], {}
        for base_key, df, gcol, scol in self._fw_group_bases():
            n = len(df)
            gvals = _fw_pp_column_list(df, gcol) if gcol in df.columns else ["No Group"] * n
            svals = _fw_pp_column_list(df, scol) if scol in df.columns else [None] * n
            names = set()
            for s, g in zip(svals, gvals):
                eff = str(self._fw_effective_group(base_key, s, g))
                names.add(eff)
                k = _gkey(eff, base_key)
                counts[k] = counts.get(k, 0) + 1
            for grp in sorted(names):
                k = _gkey(grp, base_key)
                if k not in self.group_labels:
                    lbl = _glabel(grp, base_key)
                    self.group_labels[k] = lbl
                    key_order.append(k)
                    labels.append(lbl)
        self.groups = labels
        self.group_counts = counts

        if not isinstance(getattr(self, "group_colors", None), dict):
            self.group_colors = {}
        if not isinstance(getattr(self, "group_markers", None), dict):
            self.group_markers = {}
        self._fw_ensure_group_colors()

        if prev_known is None:
            self.selected_groups_xy = list(key_order)
            self.selected_groups_ribbons = list(key_order)
        else:
            known = set(prev_known)
            valid = set(key_order)
            new_keys = [k for k in key_order if k not in known]

            def _merge(cur):
                if not isinstance(cur, (list, tuple, set)):
                    return list(key_order)
                kept = [k for k in cur if k in valid]
                return kept + [k for k in new_keys if k not in kept]

            self.selected_groups_xy = _merge(getattr(self, "selected_groups_xy", None))
            self.selected_groups_ribbons = _merge(getattr(self, "selected_groups_ribbons", None))
            if isinstance(getattr(self, "selected_groups_graphic", None), list) and self.selected_groups_graphic:
                self.selected_groups_graphic = _merge(self.selected_groups_graphic)
        self._fw_known_group_keys = list(key_order)

    # ---------- Tabs ----------
    def initUI_tabs(self):
        self.current_db_selection = {"tamiz": True, "laser": False, "hybrid": False}

        # === Tab XY ===
        self.tab_xy = QWidget()
        v_xy = QVBoxLayout(self.tab_xy)
        h_controls = QHBoxLayout()

        # ---- Full project controls (program-level, not Componentry-window-level)
        self.btn_open_full_project_xy = QPushButton("Open full project")
        self.btn_open_full_project_xy.clicked.connect(self.open_full_project_dialog)
        h_controls.addWidget(self.btn_open_full_project_xy)

        self.btn_save_full_project_xy = QPushButton("Save full project")
        self.btn_save_full_project_xy.clicked.connect(self.save_full_project)
        h_controls.addWidget(self.btn_save_full_project_xy)

        self.btn_save_full_project_as_xy = QPushButton("Save full project as...")
        self.btn_save_full_project_as_xy.clicked.connect(self.save_full_project_as)
        h_controls.addWidget(self.btn_save_full_project_as_xy)

        self.btn_load_tamiz = QPushButton(tr("btn_load_sieve"))
        self.btn_load_tamiz.clicked.connect(self.load_file)
        h_controls.addWidget(self.btn_load_tamiz)

        self.lbl_x = QLabel(tr("lbl_x"))
        h_controls.addWidget(self.lbl_x)

        self.cmb_x = QComboBox()
        h_controls.addWidget(self.cmb_x)

        self.lbl_y = QLabel(tr("lbl_y"))
        h_controls.addWidget(self.lbl_y)

        self.cmb_y = QComboBox()
        h_controls.addWidget(self.cmb_y)

        opts = ["mean", "median", "sigma", "skewness", "kurtosis"]
        self.cmb_x.addItems(opts)
        self.cmb_y.addItems(opts)
        self.cmb_x.setCurrentText("mean")
        self.cmb_y.setCurrentText("sigma")

        self.btn_plot_xy = QPushButton(tr("btn_plot_xy"))
        self.btn_plot_xy.clicked.connect(self.plot_xy)
        h_controls.addWidget(self.btn_plot_xy)

        self.btn_select_db = QPushButton(tr("btn_choose_db"))
        self.btn_select_db.clicked.connect(self.select_db_to_plot)
        h_controls.addWidget(self.btn_select_db)

        self.chk_show_names = QCheckBox(tr("chk_show_names"))
        self.chk_show_names.setChecked(False)
        self.chk_show_names.toggled.connect(self.plot_xy)
        h_controls.addWidget(self.chk_show_names)

        self.btn_filter_groups = QPushButton(tr("btn_filter_groups"))
        self.btn_filter_groups.setToolTip("Build the groups, and choose their colors, symbols and visibility")
        self.btn_filter_groups.clicked.connect(lambda *_: self.select_groups_xy())
        h_controls.addWidget(self.btn_filter_groups)

        h_controls.addStretch()

        self.btn_view_xy = QPushButton(tr("btn_view_plot"))
        self.btn_view_xy.setToolTip("Open the plot in its own window, exactly as it will be exported")
        self.btn_view_xy.clicked.connect(lambda *_: self.open_xy_view())
        h_controls.addWidget(self.btn_view_xy)

        self.btn_export_xy = QPushButton(tr("btn_export_image"))
        self.btn_export_xy.setToolTip("Save the plot with the size and quality set in View plot")
        self.btn_export_xy.clicked.connect(lambda *_: self.export_xy_plot())
        h_controls.addWidget(self.btn_export_xy)

        v_xy.addLayout(h_controls)

        # Granulometric parameters panel: modern table view (see Block 6B).
        self.params_panel = GranuloParamsPanel(owner=self)
        self.lbl_console = self.params_panel.lbl_title
        self.btn_export_params = self.params_panel.btn_export
        self.btn_export_params.clicked.connect(self.export_params_to_excel)
        # Backward compatibility: older code / plugins may still use txt_console
        self.txt_console = self.params_panel

        self.canvas_xy = FigureCanvas(plt.figure(figsize=(8, 4.5)))
        self.canvas_xy.setMinimumHeight(220)
        self.canvas_xy.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Table (top) and plot (bottom): drag the divider between them to resize both
        # (Block 6C). Double click on the divider restores the default sizes.
        self.xy_splitter = FwSplitter(Qt.Vertical, owner=self, settings_key="xy_splitter_state", default_ratio=0.45)
        self.xy_splitter.addWidget(self.params_panel)
        self.xy_splitter.addWidget(self.canvas_xy)
        self.xy_splitter.finish_setup()
        v_xy.addWidget(self.xy_splitter, 1)

        self.tabs.addTab(self.tab_xy, tr("tab_xy"))


    # --- Persist on close ---
    def closeEvent(self, e):
        try:
            if self.settings is not None:
                self.settings.setValue("theme", self.theme)
                self.settings.setValue("method", self.current_method)
                self.settings.setValue("hybrid_method", self.hybrid_method)
                self.settings.setValue("last_dir", getattr(self, "last_dir", get_script_dir()))
        finally:
            try:
                super().closeEvent(e)
            except Exception:
                pass
# endregion




# region # === Block 8: Plot tabs setup (legacy initUI wrapper; EN-only) ===
    def initUI(self):
        """
        Legacy entry point kept for backward compatibility.

        EN-only build:
          - No language switching
          - No logo/banner widgets here
          - Builds the same tab UI via _init_central_no_banner() + initUI_tabs()
        """
        # Ensure the "tabs-only" central widget exists
        if not hasattr(self, "tabs") or self.tabs is None:
            try:
                self._init_central_no_banner()
            except Exception:
                self.tabs = QTabWidget()
                self.setCentralWidget(self.tabs)

        # Ensure menu exists (guard against double init)
        if not hasattr(self, "menu_file"):
            self.initMenu()

        # Build tabs (guard against double init)
        if not hasattr(self, "tab_xy"):
            self.initUI_tabs()
# endregion

# region # === Block 9: File loading methods, menu actions and database views (EN-only, COMPLETE, hybrid-safe) ===

    # ========= SIEVE helpers: autodetection and normalization =========

    def _to_num(self, s):
        """
        Convert to numeric accepting comma or dot as decimal separator.
        Supports Series, arrays and DataFrames (column-by-column).
        """
        if isinstance(s, pd.DataFrame):
            return s.apply(
                lambda col: pd.to_numeric(
                    col.astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )
            )
        else:
            return pd.to_numeric(
                pd.Series(s).astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    def _is_multiple_of_half(self, arr, tol=1e-9):
        """True if all finite values are multiples of 0.5."""
        x = np.asarray(arr, dtype=float)
        m = np.isfinite(x)
        if not m.any():
            return False
        return np.all(np.isclose((x[m] * 2) - np.round(x[m] * 2), 0.0, atol=tol))

    def _make_unique(self, names):
        """
        Return unique names (append ' (2)', ' (3)', ... to duplicates).
        Keeps original order.
        """
        seen = {}
        out = []
        for n in names:
            n0 = str(n).strip()
            if n0 in seen:
                seen[n0] += 1
                out.append(f"{n0} ({seen[n0]})")
            else:
                seen[n0] = 1
                out.append(n0)
        return out

    def _is_numeric_like(self, v):
        """True if v can be interpreted as a number (comma or dot allowed)."""
        if pd.isna(v):
            return False
        s = str(v).strip().replace(",", ".")
        try:
            float(s)
            return True
        except Exception:
            return False

    def _fraction_nonnumeric(self, values):
        """Fraction of NON-numeric cells in a list/series."""
        vals = list(values)
        nn = sum(0 if self._is_numeric_like(v) else 1 for v in vals) if vals else 0
        return nn / float(len(vals) or 1)

    def _has_sample_repetition(self, series_like):
        """
        True if at least one Sample appears in ≥2 rows.
        (Avoid confusing the compact format with the wide format.)
        """
        s = pd.Series(series_like)
        vc = s.value_counts(dropna=False)
        return (vc >= 2).any()

    def _maybe_normalize_tamiz_to_100(self, df: pd.DataFrame, tol: float = 0.6):
        """
        If any sample (column 1) does not sum to ~100% in Weight (column 2),
        ask the user whether to normalize (scale to 100% per sample).

        Returns: (df_out, did_normalize: bool)
        """
        if df is None or df.empty:
            return df, False

        samp_col = df.columns[1]
        wt_col = df.columns[2]

        w = pd.to_numeric(df[wt_col], errors="coerce")
        sums = df.assign(**{wt_col: w}).groupby(samp_col, dropna=False)[wt_col].sum(min_count=1)

        bad_mask = ~np.isclose(sums.values, 100.0, atol=tol)
        bad = sums[bad_mask]
        if bad.empty:
            return df, False

        names = [str(s) for s in bad.index.tolist()]
        preview = ", ".join(names[:12]) + ("…" if len(names) > 12 else "")

        title = "Attention"
        text = (
            "YOUR SIEVE DATA ARE NOT NORMALIZED TO 100%.\n\n"
            f"Affected samples ({len(names)}): {preview}\n\n"
            "Do you want the software to normalize them to 100%?"
        )
        btn = QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if btn != QMessageBox.Yes:
            return df, False

        df2 = df.copy()
        df2[wt_col] = pd.to_numeric(df2[wt_col], errors="coerce")
        for s, total in sums.items():
            if not np.isfinite(total) or total <= 0:
                continue
            if not np.isclose(total, 100.0, atol=tol):
                mask = df2[samp_col].astype(str) == str(s)
                df2.loc[mask, wt_col] = df2.loc[mask, wt_col].astype(float) * (100.0 / float(total))
        return df2, True

    def _normalize_tamiz_formats(self, df0: pd.DataFrame):
        """
        Detect and normalize SIEVE data in two formats.

        Format 1 (row-wise, EXACTLY 4 columns):
          Col0: φ (numeric, multiples of 0.5)
          Col1: Sample (string or numeric)
          Col2: Weight (numeric)
          Col3: Group (string or numeric) — must be constant per Sample,
                and at least one Sample must repeat (≥2 rows)

        Format 2 (wide):
          df.columns           = headers → Col0=Phi, Col1..=sample names
          df.iloc[0, :]        = groups row (ideally mostly NON-numeric)
          df.iloc[1:, 0]       = φ (numeric, multiples of 0.5)
          df.iloc[1:, 1..n]    = Weight per sample (numeric)
          If the groups row is not mostly non-numeric, assume “No Group”
          and start data at df.iloc[0, :].

        Returns (df_norm, group_col_name) with columns:
          [phi_col, 'Sample', 'Weight', group_col_name]
        """
        if df0 is None or df0.shape[1] < 1:
            return None, None

        ncols = df0.shape[1]
        NO_GROUP = "No Group"

        # ---------- Try Format 1 (EXACTLY 4 columns) ----------
        if ncols == 4:
            df = df0.iloc[:, :4].copy()

            phi = self._to_num(df.iloc[:, 0])
            wt = self._to_num(df.iloc[:, 2])

            if self._is_multiple_of_half(phi) and wt.notna().sum() > 0:
                if self._has_sample_repetition(df.iloc[:, 1]):
                    df.iloc[:, 0] = phi
                    df.iloc[:, 2] = wt
                    df = df.dropna(subset=[df.columns[0], df.columns[2]]).copy()
                    df.sort_values([df.columns[1], df.columns[0]], inplace=True, kind="mergesort")
                    df.reset_index(drop=True, inplace=True)

                    group_col_name = str(df.columns[3])
                    samp_col_name = str(df.columns[1])

                    # Group consistency per sample
                    for sample, g in df.groupby(samp_col_name):
                        vals = g.iloc[:, 3].astype(str).str.strip()
                        non_empty = vals.replace({"": np.nan, "nan": np.nan}).dropna()
                        uniq = non_empty.unique()
                        if len(uniq) > 1:
                            return None, None
                        fill_val = uniq[0] if len(uniq) == 1 else NO_GROUP
                        df.loc[g.index, df.columns[3]] = fill_val

                    return df, group_col_name

        # ---------- Try Format 2 (wide) ----------
        if df0.shape[0] >= 1 and ncols >= 2:
            header_names = list(df0.columns)
            phi_col_name = str(header_names[0]) if str(header_names[0]).strip() else str(df0.columns[0])

            sample_cols = []
            sample_names_raw = []
            for j in range(1, ncols):
                sname = str(header_names[j]).strip()
                if sname == "" or sname.lower() == "nan":
                    continue
                sample_cols.append(j)
                sample_names_raw.append(sname)

            if not sample_cols:
                return None, None

            # Candidate groups row = first data row
            if df0.shape[0] >= 2:
                groups_row = list(df0.iloc[0, :])
                grp_vals_for_samples = [groups_row[j] for j in sample_cols]
                frac_nonn = self._fraction_nonnumeric(grp_vals_for_samples)
                groups_present = frac_nonn > 0.60
                data_start_row = 1 if groups_present else 0
            else:
                groups_present = False
                data_start_row = 0

            phi_series = self._to_num(df0.iloc[data_start_row:, 0])
            if not self._is_multiple_of_half(phi_series):
                return None, None

            sample_names = self._make_unique(sample_names_raw)

            wide = df0.iloc[data_start_row:, [0] + sample_cols].copy()
            wide.columns = [phi_col_name] + sample_names

            wide[phi_col_name] = self._to_num(wide[phi_col_name])
            wide[sample_names] = self._to_num(wide[sample_names])

            wide = wide.dropna(subset=[phi_col_name])

            long = wide.melt(
                id_vars=[phi_col_name],
                value_vars=sample_names,
                var_name="Sample",
                value_name="Weight"
            )
            long = long.dropna(subset=["Weight"]).copy()

            if groups_present:
                sample_groups_raw = [groups_row[j] for j in sample_cols]
                norm_groups = []
                for gv in sample_groups_raw:
                    gtxt = str(gv).strip()
                    if gtxt == "" or gtxt.lower() == "nan":
                        gtxt = NO_GROUP
                    norm_groups.append(gtxt)
                grp_map = dict(zip(sample_names, norm_groups))
                long["Group"] = long["Sample"].map(grp_map).fillna(NO_GROUP)
            else:
                long["Group"] = NO_GROUP

            long.sort_values(["Sample", phi_col_name], inplace=True, kind="mergesort")
            long.reset_index(drop=True, inplace=True)

            return long[[phi_col_name, "Sample", "Weight", "Group"]], "Group"

        return None, None

    # ------------------------ SIEVE: load with autodetection ------------------------
    def load_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            tr("file_load_sieve"),
            "",
            "Excel files (*.xls *.xlsx)"
        )
        if not file:
            return

        try:
            df0 = pd.read_excel(file, header=0)
        except Exception as e:
            QMessageBox.warning(self, tr("error"), f"Could not read file:\n{e}")
            return

        df_norm, group_col_name = self._normalize_tamiz_formats(df0)
        if df_norm is None:
            msg = (
                "The sieve Excel must match one of two formats:\n"
                " • Format 1 (4 columns): φ | Sample | Weight | Group\n"
                "   (φ multiples of 0.5; at least one Sample with ≥2 rows; Group constant per Sample)\n"
                " • Format 2 (wide): headers in df.columns, optional first row = groups\n"
                "   (mostly NON-numeric). If no valid groups row is found, 'No Group' is assumed\n"
                "   and data starts from the first row."
            )
            QMessageBox.warning(self, tr("info"), msg)
            return

        # Offer normalization to 100% per sample (sieve only)
        try:
            df_norm, _normalized = self._maybe_normalize_tamiz_to_100(df_norm, tol=0.6)
        except Exception:
            pass

        self.group_col = group_col_name
        self.df_data = df_norm

        # Compute parameters
        params = []
        for sample, grp in self.df_data.groupby(self.df_data.columns[1]):  # Sample
            phi = grp.iloc[:, 0].tolist()
            weights = grp.iloc[:, 2].tolist()
            gval = grp.iloc[0, 3]
            p = calculate_parameters(phi, weights, self.current_method)
            row = {"Sample": sample, self.group_col: gval, **p}
            params.append(row)

        self.param_results = pd.DataFrame(params)

        self._update_all_groups_and_colors()

        self.cmb_x.clear()
        self.cmb_y.clear()
        self.cmb_x.addItems(["mean", "median", "sigma", "skewness", "kurtosis"])
        self.cmb_y.addItems(["mean", "median", "sigma", "skewness", "kurtosis"])
        self.cmb_x.setCurrentText("mean")
        self.cmb_y.setCurrentText("sigma")

        self.update_console()
        self.plot_xy()
        self._update_db_menu_actions()

    # ------------------------------- LASER (two formats; autodetection) -------------------------------

    def _phi_from_um(self, d_um):
        """Convert diameter in μm to φ (d0=1 mm)."""
        x = pd.to_numeric(pd.Series(d_um).astype(str).str.replace(",", ".", regex=False), errors="coerce")
        return -np.log2(x.astype(float) / 1000.0)

    def _bin_laser_phi_right_closed(self, df_can: pd.DataFrame, step: float) -> pd.DataFrame:
        """
        RIGHT-CLOSED binning (X = (X-step, X]) for LASER.
        Expects columns: ['Sample', 'Diameter (μm)', '1 (%)'].
        Returns columns: ['Sample','phi','Wt'] where phi is the bin upper edge.
        """
        if df_can is None or df_can.empty:
            return pd.DataFrame(columns=["Sample", "phi", "Wt"])

        d_um = self._to_num(df_can["Diameter (μm)"])
        wt = self._to_num(df_can["1 (%)"])
        samp = df_can["Sample"].astype(str)

        mask = np.isfinite(d_um) & np.isfinite(wt) & (d_um > 0)
        d_um = d_um[mask]
        wt = wt[mask]
        samp = samp[mask]

        phi_cont = self._phi_from_um(d_um)
        step = float(step)
        lo = np.floor(phi_cont.min() / step) * step
        hi = np.ceil(phi_cont.max() / step) * step
        edges = np.round(np.arange(lo, hi + step + 1e-9, step), 6)

        cats = pd.cut(phi_cont, bins=edges, right=True, include_lowest=True)

        tmp = pd.DataFrame({"Sample": samp.values, "_bin": cats, "Wt": wt.values})
        gb = tmp.groupby(["Sample", "_bin"], as_index=False)["Wt"].sum()

        gb["phi"] = gb["_bin"].apply(lambda iv: float(np.round(iv.right, 6)))
        out = gb[["Sample", "phi", "Wt"]].sort_values(["Sample", "phi"]).reset_index(drop=True)
        return out

    def _map_laser_columns_by_name(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with EXACTLY these columns:
          ['Sample', 'Diameter (μm)', '1 (%)', 'Group']

        Detects columns by header name (English/Spanish tolerant),
        accents and symbols. Raises ValueError if any required column is missing.
        """
        import re
        import unicodedata

        def norm(s: str) -> str:
            s = unicodedata.normalize("NFKD", str(s))
            s = "".join(c for c in s if not unicodedata.combining(c))
            s = s.lower()
            s = s.replace("µ", "u").replace("μ", "u")
            s = re.sub(r"[^a-z0-9%]+", "", s)
            return s

        norm_map = {col: norm(col) for col in df_in.columns}

        def find_col(pred):
            for col, n in norm_map.items():
                if pred(n):
                    return col
            return None

        sample_col = find_col(lambda n: any(k in n for k in ("sample", "muestra", "nombre", "name", "id")))
        diam_col = find_col(lambda n: ("diamet" in n) or ("size" in n) or (n.startswith("d") and "um" in n))
        perc_col = find_col(
            lambda n: (n in ("1", "1%", "1percent")) or ("porcent" in n) or ("percent" in n) or ("wt" in n) or n.endswith("%")
        )
        group_col = find_col(lambda n: any(k in n for k in ("group", "grupo", "facies", "filtro")))

        missing = []
        if sample_col is None:
            missing.append("Sample")
        if diam_col is None:
            missing.append("Diameter (μm)")
        if perc_col is None:
            missing.append("1 (%) / wt (%) / %")
        if group_col is None:
            missing.append("Group")

        if missing:
            raise ValueError("missing headers: " + ", ".join(missing))

        df = df_in[[sample_col, diam_col, perc_col, group_col]].copy()
        df.columns = ["Sample", "Diameter (μm)", "1 (%)", "Group"]
        return df

    def _normalize_laser_inputs(self, df0: pd.DataFrame):
        """
        Try:
          A) Row-wise format detected by header names → _map_laser_columns_by_name
          B) Wide format: Col0 = Diameter(μm), Col1..n = samples, optional first row = groups/facies

        Returns: (df_canon, fmt_str) with columns ['Sample','Diameter (μm)','1 (%)','Group'].
        """
        try:
            df_a = self._map_laser_columns_by_name(df0)
            return df_a, "fmt1"
        except Exception:
            pass

        if df0 is None or df0.shape[1] < 2:
            return None, None

        ncols = df0.shape[1]
        header_names = list(df0.columns)

        diam_series_all = self._to_num(df0.iloc[:, 0])
        frac_num = np.isfinite(diam_series_all).mean()
        looks_like_diam = (frac_num > 0.60) and (not self._is_multiple_of_half(diam_series_all))
        if not looks_like_diam:
            return None, None

        sample_cols = []
        sample_names_raw = []
        for j in range(1, ncols):
            sname = str(header_names[j]).strip()
            if sname == "" or sname.lower() == "nan":
                continue
            sample_cols.append(j)
            sample_names_raw.append(sname)
        if not sample_cols:
            return None, None

        sample_names = self._make_unique(sample_names_raw)

        if df0.shape[0] >= 2:
            groups_row = list(df0.iloc[0, :])
            grp_vals = [groups_row[j] for j in sample_cols]
            groups_present = self._fraction_nonnumeric(grp_vals) > 0.60
            data_start_row = 1 if groups_present else 0
        else:
            groups_present = False
            data_start_row = 0

        wide = df0.iloc[data_start_row:, [0] + sample_cols].copy()
        wide.columns = ["Diameter (μm)"] + sample_names
        wide["Diameter (μm)"] = self._to_num(wide["Diameter (μm)"])
        wide[sample_names] = self._to_num(wide[sample_names])

        wide = wide.dropna(subset=["Diameter (μm)"])
        wide = wide[wide["Diameter (μm)"] > 0]

        long = wide.melt(
            id_vars=["Diameter (μm)"],
            value_vars=sample_names,
            var_name="Sample",
            value_name="1 (%)"
        ).dropna(subset=["1 (%)"])

        NO_GROUP = "No Group"
        if groups_present:
            sample_groups_raw = [groups_row[j] for j in sample_cols]
            norm_groups = []
            for gv in sample_groups_raw:
                gtxt = str(gv).strip()
                if gtxt == "" or gtxt.lower() == "nan":
                    gtxt = NO_GROUP
                norm_groups.append(gtxt)
            grp_map = dict(zip(sample_names, norm_groups))
            long["Group"] = long["Sample"].map(grp_map).fillna(NO_GROUP)
        else:
            long["Group"] = NO_GROUP

        long.sort_values(["Sample", "Diameter (μm)"], inplace=True, kind="mergesort")
        long.reset_index(drop=True, inplace=True)
        return long[["Sample", "Diameter (μm)", "1 (%)", "Group"]], "fmt2"

    def load_laser_file(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QPushButton

        file, _ = QFileDialog.getOpenFileName(
            self,
            tr("file_load_laser"),
            "",
            "Excel files (*.xls *.xlsx)"
        )
        if not file:
            return

        try:
            df = pd.read_excel(file)
        except Exception as e:
            QMessageBox.warning(self, tr("error"), f"Could not read file:\n{e}")
            return

        if df.shape[1] < 2:
            msg = (
                "Laser file must be either:\n"
                " • Row-wise with recognizable headers (Sample, Diameter μm, %, Group)\n"
                " • Wide: Col0=Diameter(μm), Col1..n=samples, optional first row = groups/facies."
            )
            QMessageBox.warning(self, tr("error"), msg)
            return

        # φ step
        dlg = QDialog(self)
        dlg.setWindowTitle("φ step for laser")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Process every 1.0 or 0.5 φ?"))
        rb1 = QRadioButton("1.0 φ")
        rb2 = QRadioButton("0.5 φ")
        rb1.setChecked(True)
        layout.addWidget(rb1)
        layout.addWidget(rb2)
        btn = QPushButton(tr("ok"))
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        if not dlg.exec_():
            return

        step = 1.0 if rb1.isChecked() else 0.5
        self.laser_step = step

        # Autodetect: row-wise headers or wide format
        df_laser_raw, fmt = self._normalize_laser_inputs(df)
        if df_laser_raw is None:
            QMessageBox.warning(self, tr("error"), "Could not recognize LASER format (criterion 1 or 2).")
            return

        # Validate Group not empty
        if df_laser_raw["Group"].isnull().any() or (df_laser_raw["Group"].astype(str).str.strip() == "").any():
            QMessageBox.warning(
                self,
                tr("error"),
                "At least one value is missing in the group/facies column (Group) of the laser file."
            )
            return

        # Correct binning: (X-Δ, X] with Δ=step
        try:
            df_bin = self._bin_laser_phi_right_closed(df_laser_raw, step)
        except Exception as e:
            QMessageBox.warning(self, tr("error"), f"An error occurred while binning by φ:\n{e}")
            return

        self.df_laser = df_bin.copy()

        # Map groups by sample (robust to duplicates)
        NO_GROUP = "No Group"
        group_map = (
            df_laser_raw[["Sample", "Group"]]
            .astype({"Sample": str})
            .dropna()
            .drop_duplicates(subset="Sample", keep="first")
            .set_index("Sample")["Group"]
            .to_dict()
        )
        self.df_laser["Group"] = self.df_laser["Sample"].map(group_map).fillna(NO_GROUP)

        self.df_laser.sort_values(["Sample", "phi"], inplace=True)
        self.df_laser.reset_index(drop=True, inplace=True)

        self._update_db_menu_actions()

        # Compute laser parameters
        params_l = []
        for sample, grp in self.df_laser.groupby("Sample"):
            phi_vals = grp["phi"].tolist()
            wt_vals = grp["Wt"].tolist()
            gval = grp["Group"].iloc[0]
            p = calculate_parameters(phi_vals, wt_vals, self.current_method)
            row = {"Sample": sample, "Group": gval, **p}
            params_l.append(row)
        self.param_results_laser = pd.DataFrame(params_l)

        self._update_all_groups_and_colors()
        self.update_console()

    # ----------------------- Database view windows (Format 2 view) -----------------------
    def _make_format2_view(self, df_long, *, phi_col, sample_col, weight_col, group_col="Group", group_row_label="Group"):
        """
        Convert a long-format DataFrame (phi, Sample, Weight, Group) into a Format-2 style view:
          - Column 0 = 'Phi'
          - Row 0 = group row (one cell per sample with its group/facies)
          - Columns 1..n = one per sample with weights by phi

        Hybrid-safe notes:
          - sample_col must be a UNIQUE hybrid identifier for hybrid data
          - aggregation uses sum instead of first to avoid silent data loss if duplicates slip in
        """
        s_sample = df_long[sample_col].astype(str)
        first_order = list(dict.fromkeys(s_sample.tolist()))

        wide = (
            df_long
            .pivot_table(index=phi_col, columns=s_sample, values=weight_col, aggfunc="sum")
            .sort_index()
        )
        wide = wide.reindex(columns=first_order)

        if group_col in df_long.columns:
            grp_map = (
                df_long
                .groupby(s_sample)[group_col]
                .apply(lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else "")
            )
        else:
            grp_map = pd.Series({s: "" for s in first_order})

        header_row = pd.DataFrame([grp_map], index=[group_row_label])

        out = pd.concat([header_row, wide])

        out.insert(0, "Phi", [group_row_label] + list(wide.index))
        return out.reset_index(drop=True)

    def show_tamiz_db_window(self):
        """Show SIEVE database in Format 2 view."""
        if self.df_data is None:
            return

        df_view = self._make_format2_view(
            self.df_data,
            phi_col=self.df_data.columns[0],
            sample_col=self.df_data.columns[1],
            weight_col=self.df_data.columns[2],
            group_col=self.group_col,
            group_row_label="Group"
        )
        title = "Sieve database (Format 2 view)"
        dlg = DataBaseWindow(title, df_view, self)
        dlg.exec_()

    def show_laser_db_window(self):
        """Show LASER database in Format 2 view."""
        if self.df_laser is None:
            return

        df_view = self._make_format2_view(
            self.df_laser,
            phi_col="phi",
            sample_col="Sample",
            weight_col="Wt",
            group_col="Group",
            group_row_label="Group"
        )
        title = "Laser database (Format 2 view)"
        dlg = DataBaseWindow(title, df_view, self)
        dlg.exec_()

    def show_hybrid_db_window(self):
        """
        Show HYBRID database in Format 2 view.

        IMPORTANT FIX:
        Use a UNIQUE hybrid identifier for the hybrid columns:
          1) "Hybrid Sample"
          2) "Sample"
          3) "Tamiz Sample" (old fallback only)

        This prevents different hybrids of the same sieve sample from collapsing
        into one visible column.
        """
        if self.df_hybrid is None or self.df_hybrid.empty:
            QMessageBox.information(self, tr("info"), "You haven't combined sieve and laser yet.")
            return

        gcol = self.group_col if (hasattr(self, "group_col") and self.group_col in self.df_hybrid.columns) else "Group"

        if "Hybrid Sample" in self.df_hybrid.columns:
            hybrid_sample_col = "Hybrid Sample"
        elif "Sample" in self.df_hybrid.columns:
            hybrid_sample_col = "Sample"
        elif "Tamiz Sample" in self.df_hybrid.columns:
            hybrid_sample_col = "Tamiz Sample"
        else:
            QMessageBox.warning(self, tr("error"), "Could not determine the hybrid sample identifier.")
            return

        weight_col = "wt%" if "wt%" in self.df_hybrid.columns else ("Wt" if "Wt" in self.df_hybrid.columns else None)
        if weight_col is None or "phi" not in self.df_hybrid.columns:
            QMessageBox.warning(self, tr("error"), "Hybrid database is missing required columns.")
            return

        df_view = self._make_format2_view(
            self.df_hybrid,
            phi_col="phi",
            sample_col=hybrid_sample_col,
            weight_col=weight_col,
            group_col=gcol if gcol in self.df_hybrid.columns else "Group",
            group_row_label="Group"
        )
        title = "Sieve + laser database (Format 2 view)"
        dlg = DataBaseWindow(title, df_view, self)
        dlg.exec_()

    # ========================= Menu actions required by initMenu =========================

    def _safe_group_keys(self):
        """
        Return stable group keys.
        """
        try:
            self._update_all_groups_and_colors()
        except Exception:
            pass

        keys = []
        gl = getattr(self, "group_labels", None)
        if isinstance(gl, dict) and len(gl) > 0:
            keys = list(gl.keys())
        else:
            groups = getattr(self, "groups", []) or []
            keys = [str(g) for g in groups]
            if not hasattr(self, "group_labels") or not isinstance(self.group_labels, dict):
                self.group_labels = {}
            for k in keys:
                self.group_labels.setdefault(k, k)

        if not hasattr(self, "group_colors") or not isinstance(self.group_colors, dict):
            self.group_colors = {}

        for i, k in enumerate(keys):
            if k not in self.group_colors:
                self.group_colors[k] = GROUP_COLORS[i % len(GROUP_COLORS)]

        return keys

    def _groups_multi_select_dialog(self, title, initial_keys):
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QListWidget, QListWidgetItem,
            QDialogButtonBox, QLabel, QAbstractItemView
        )
        from PyQt5.QtCore import Qt

        keys = self._safe_group_keys()
        if not keys:
            return []

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(460, 520)

        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select groups to display:"))

        lw = QListWidget()
        lw.setSelectionMode(QAbstractItemView.MultiSelection)

        init = set(initial_keys or [])
        for k in keys:
            label = self.group_labels.get(k, k)
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, k)
            lw.addItem(it)
            if k in init:
                it.setSelected(True)

        lay.addWidget(lw)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        if dlg.exec_() != QDialog.Accepted:
            return None

        out = []
        for i in range(lw.count()):
            it = lw.item(i)
            if it.isSelected():
                out.append(it.data(Qt.UserRole))
        return out

    def _recompute_sieve_params(self):
        if self.df_data is None or self.df_data.empty:
            self.param_results = None
            return

        params = []
        samp_col = self.df_data.columns[1]
        phi_col = self.df_data.columns[0]
        wt_col = self.df_data.columns[2]
        g_col = self.group_col if hasattr(self, "group_col") else self.df_data.columns[3]

        for sample, grp in self.df_data.groupby(samp_col):
            phi_vals = grp[phi_col].tolist()
            wt_vals = grp[wt_col].tolist()
            gval = grp[g_col].iloc[0] if g_col in grp.columns else "No Group"
            p = calculate_parameters(phi_vals, wt_vals, self.current_method)
            params.append({"Sample": sample, g_col: gval, **p})

        self.param_results = pd.DataFrame(params)

    def _recompute_laser_params(self):
        if self.df_laser is None or self.df_laser.empty:
            self.param_results_laser = None
            return

        rows = []
        for sample, grp in self.df_laser.groupby("Sample"):
            p = calculate_parameters(grp["phi"].tolist(), grp["Wt"].tolist(), self.current_method)
            gval = grp["Group"].iloc[0] if "Group" in grp.columns else "No Group"
            rows.append({"Sample": sample, "Group": gval, **p})
        self.param_results_laser = pd.DataFrame(rows)

    def _recompute_hybrid_params(self):
        if self.df_hybrid is None or self.df_hybrid.empty:
            self.param_results_hybrid = None
            return

        dfh = self.df_hybrid.copy()

        # IMPORTANT FIX: unique identity for hybrid rows
        if "Hybrid Sample" in dfh.columns:
            sample_col = "Hybrid Sample"
        elif "Sample" in dfh.columns:
            sample_col = "Sample"
        elif "Tamiz Sample" in dfh.columns:
            sample_col = "Tamiz Sample"
        else:
            self.param_results_hybrid = None
            return

        phi_col = "phi" if "phi" in dfh.columns else None
        wt_col = "wt%" if "wt%" in dfh.columns else ("Wt" if "Wt" in dfh.columns else None)
        gpref = self.group_col if hasattr(self, "group_col") else "Group"
        gcol = gpref if gpref in dfh.columns else ("Group" if "Group" in dfh.columns else None)

        if phi_col is None or wt_col is None:
            self.param_results_hybrid = None
            return

        if gcol is None:
            dfh["Group"] = "No Group"
            gcol = "Group"

        rows = []
        for sample, grp in dfh.groupby(sample_col):
            p = calculate_parameters(grp[phi_col].tolist(), grp[wt_col].tolist(), self.current_method)
            row = {"Sample": sample, gcol: grp[gcol].iloc[0], **p}

            if "Hybrid Sample" in grp.columns:
                row["Hybrid Sample"] = str(grp["Hybrid Sample"].iloc[0])
            if "Tamiz Sample" in grp.columns:
                row["Tamiz Sample"] = str(grp["Tamiz Sample"].iloc[0])
            if "Laser Sample" in grp.columns:
                row["Laser Sample"] = str(grp["Laser Sample"].iloc[0])
            if "Hybrid Method Code" in grp.columns:
                row["Hybrid Method Code"] = str(grp["Hybrid Method Code"].iloc[0])
            if "Hybrid Method" in grp.columns:
                row["Hybrid Method"] = str(grp["Hybrid Method"].iloc[0])
            if "phi*" in grp.columns:
                row["phi*"] = grp["phi*"].iloc[0]

            rows.append(row)

        self.param_results_hybrid = pd.DataFrame(rows)

    def _refresh_after_settings_change(self):
        try:
            self._update_all_groups_and_colors()
        except Exception:
            pass
        try:
            self.update_console()
        except Exception:
            pass

        for fn in ("plot_xy",):
            try:
                if hasattr(self, fn):
                    getattr(self, fn)()
            except Exception:
                pass

        try:
            self._update_db_menu_actions()
        except Exception:
            pass

    def choose_method(self):
        from PyQt5.QtWidgets import QInputDialog, QMessageBox

        labels = list(METHODS.keys())
        if not labels:
            QMessageBox.warning(self, tr("error"), "METHODS dictionary is empty.")
            return

        current_label = labels[0]
        for k, v in METHODS.items():
            if v == getattr(self, "current_method", "folkward"):
                current_label = k
                break
        idx = labels.index(current_label) if current_label in labels else 0

        chosen, ok = QInputDialog.getItem(
            self,
            tr("choose_method_title"),
            tr("cfg_method"),
            labels,
            idx,
            False
        )
        if not ok or not chosen:
            return

        self.current_method = METHODS[chosen]

        try:
            if self.settings is not None:
                self.settings.setValue("method", self.current_method)
        except Exception:
            pass

        try:
            self._recompute_sieve_params()
        except Exception:
            pass
        try:
            self._recompute_laser_params()
        except Exception:
            pass
        try:
            self._recompute_hybrid_params()
        except Exception:
            pass

        self._refresh_after_settings_change()

    def choose_theme(self):
        from PyQt5.QtWidgets import QInputDialog, QApplication

        opts = [tr("theme_light"), tr("theme_dark")]
        cur = 0 if getattr(self, "theme", "light") == "light" else 1

        chosen, ok = QInputDialog.getItem(
            self,
            tr("choose_theme_title"),
            tr("cfg_theme"),
            opts,
            cur,
            False
        )
        if not ok or not chosen:
            return

        if chosen == tr("theme_dark"):
            self.theme = "dark"
            QApplication.instance().setStyleSheet(DARK_STYLESHEET)
        else:
            self.theme = "light"
            QApplication.instance().setStyleSheet(LIGHT_STYLESHEET)

        try:
            if self.settings is not None:
                self.settings.setValue("theme", self.theme)
        except Exception:
            pass

        self._refresh_after_settings_change()

    def edit_colors(self):
        """
        Settings > Group colors.
        Uses ColorDialog if available (Block 15), otherwise fallback.
        """
        from PyQt5.QtWidgets import QColorDialog, QMessageBox, QInputDialog
        from PyQt5.QtGui import QColor

        keys = self._safe_group_keys()
        if not keys:
            QMessageBox.information(self, tr("info"), "No groups available.")
            return

        # Try custom dialog if exists
        try:
            if "ColorDialog" in globals():
                initial = {k: self.group_colors.get(k, GROUP_COLORS[i % len(GROUP_COLORS)]) for i, k in enumerate(keys)}
                dlg = ColorDialog(initial, parent=self)
                if dlg.exec_():
                    result = getattr(dlg, "selected_colors", None)
                    if isinstance(result, dict):
                        for k in keys:
                            if k in result:
                                self.group_colors[k] = result[k]
                        self._refresh_after_settings_change()
                        return
        except Exception:
            pass

        # Fallback
        labels = [self.group_labels.get(k, k) for k in keys]
        chosen_label, ok = QInputDialog.getItem(self, tr("cfg_colors"), "Group:", labels, 0, False)
        if not ok or not chosen_label:
            return

        idx = labels.index(chosen_label)
        k = keys[idx]
        current = self.group_colors.get(k, GROUP_COLORS[idx % len(GROUP_COLORS)])
        col = QColorDialog.getColor(QColor(current), self, f"Color: {chosen_label}")
        if col.isValid():
            self.group_colors[k] = col.name()
            self._refresh_after_settings_change()

    def select_groups_all(self):
        keys = self._safe_group_keys()
        if not keys:
            return

        current = getattr(self, "selected_groups_xy", keys)
        picked = self._groups_multi_select_dialog(tr("cfg_groups_all"), current)
        if picked is None:
            return

        self.selected_groups_xy = list(picked)
        self.selected_groups_ribbons = list(picked)

        self._refresh_after_settings_change()

    def select_groups_xy(self):
        keys = self._safe_group_keys()
        if not keys:
            return

        current = getattr(self, "selected_groups_xy", keys)
        picked = self._groups_multi_select_dialog(tr("btn_filter_groups"), current)
        if picked is None:
            return

        self.selected_groups_xy = list(picked)
        try:
            self.plot_xy()
        except Exception:
            pass

    def select_ribbons(self):
        keys = self._safe_group_keys()
        if not keys:
            return

        current = getattr(self, "selected_groups_ribbons", keys)
        picked = self._groups_multi_select_dialog(tr("cfg_ribbons"), current)
        if picked is None:
            return

        self.selected_groups_ribbons = list(picked)
        try:
            self.plot_xy()
        except Exception:
            pass

    def select_db_to_plot(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox

        has_t = self.df_data is not None and (not self.df_data.empty)
        has_l = self.df_laser is not None and (not self.df_laser.empty)
        has_h = self.df_hybrid is not None and (not self.df_hybrid.empty)

        dlg = QDialog(self)
        dlg.setWindowTitle(tr("btn_choose_db"))
        lay = QVBoxLayout(dlg)

        chk_t = QCheckBox(base_name_display("Tamiz"))
        chk_l = QCheckBox(base_name_display("Laser"))
        chk_h = QCheckBox(base_name_display("Híbrido"))

        chk_t.setEnabled(has_t)
        chk_l.setEnabled(has_l)
        chk_h.setEnabled(has_h)

        current = getattr(self, "current_db_selection", {"tamiz": True, "laser": False, "hybrid": False})
        chk_t.setChecked(bool(current.get("tamiz", has_t and not (has_l or has_h))))
        chk_l.setChecked(bool(current.get("laser", False)))
        chk_h.setChecked(bool(current.get("hybrid", False)))

        lay.addWidget(chk_t)
        lay.addWidget(chk_l)
        lay.addWidget(chk_h)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        if dlg.exec_() != QDialog.Accepted:
            return

        sel = {
            "tamiz": chk_t.isChecked(),
            "laser": chk_l.isChecked(),
            "hybrid": chk_h.isChecked(),
        }

        if not any(sel.values()):
            QMessageBox.warning(self, tr("error"), "Select at least one database.")
            return

        self.current_db_selection = sel
        try:
            self.plot_xy()
        except Exception:
            pass

# endregion



# region # === Block 10: Plotting, console and exports (EN-only, no histogram, hybrid-safe) ===

    # --------------------------- Generic helpers ---------------------------

    def _safe_filename(self, s: str) -> str:
        import re
        s = str(s).strip()
        s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s if s else "untitled"

    def _axis_label_for_key(self, key: str) -> str:
        try:
            return param_label(key)
        except Exception:
            fb = {
                "mean": "Mean (Mφ)",
                "median": "Median (Mdφ)",
                "sigma": "Sorting (σφ)",
                "skewness": "Skewness (Skφ)",
                "kurtosis": "Kurtosis (Kφ)",
            }
            return fb.get(key, str(key))

    def _selected_bases_xy(self):
        sel = getattr(self, "current_db_selection", {"tamiz": True, "laser": False, "hybrid": False})
        out = []
        if sel.get("tamiz", False):
            out.append("Tamiz")
        if sel.get("laser", False):
            out.append("Laser")
        if sel.get("hybrid", False):
            out.append("Híbrido")
        return out

    def _param_pack_by_base(self, base_key: str):
        """
        Returns tuple (df_params, group_col, sample_col) for base key.

        IMPORTANT FIX FOR HYBRID:
        For hybrid parameters, prefer the unique hybrid identity:
          1) "Hybrid Sample"
          2) "Sample"
          3) "Tamiz Sample" (old fallback only)
        """
        b = base_key_canon(base_key) if "base_key_canon" in globals() else base_key

        if b == "Tamiz":
            df = getattr(self, "param_results", None)
            gcol = getattr(self, "group_col", "Group")
            scol = "Sample"
            return df, gcol, scol

        if b == "Laser":
            df = getattr(self, "param_results_laser", None)
            gcol = "Group"
            scol = "Sample"
            return df, gcol, scol

        if b == "Híbrido":
            df = getattr(self, "param_results_hybrid", None)
            gcol = getattr(self, "group_col", "Group")

            if df is not None:
                if gcol not in df.columns and "Group" in df.columns:
                    gcol = "Group"

                if "Hybrid Sample" in df.columns:
                    scol = "Hybrid Sample"
                elif "Sample" in df.columns:
                    scol = "Sample"
                elif "Tamiz Sample" in df.columns:
                    scol = "Tamiz Sample"
                else:
                    scol = "Sample"
            else:
                scol = "Sample"

            return df, gcol, scol

        return None, "Group", "Sample"

    def _group_key(self, grp, base_key: str) -> str:
        try:
            return _gkey(grp, base_key)
        except Exception:
            return f"{str(grp).strip()} ({base_key})"

    def _group_label(self, grp, base_key: str) -> str:
        try:
            return _glabel(grp, base_key)
        except Exception:
            disp = base_name_display(base_key) if "base_name_display" in globals() else base_key
            return f"{str(grp).strip()} ({disp})"

    def _selected_group_keys_for_tab(self, tab_name: str):
        if tab_name == "xy":
            return set(getattr(self, "selected_groups_xy", []) or [])
        return set()

    def _iter_points_for_plot(self, x_key: str, y_key: str, tab_name: str = "xy"):
        """
        Unified generator of points across selected bases.
        Yields dict with keys: x,y,sample,group,base,gkey,glabel,color

        IMPORTANT FIX FOR HYBRID:
        Hybrid points are now taken from unique hybrid identities, so two hybrids
        with the same sieve sample but different laser/method/phi* do not collapse.
        """
        bases = self._selected_bases_xy()
        selected_groups = self._selected_group_keys_for_tab(tab_name)

        for base_key in bases:
            dfp, gcol, scol = self._param_pack_by_base(base_key)
            if dfp is None or getattr(dfp, "empty", True):
                continue

            if x_key not in dfp.columns or y_key not in dfp.columns:
                continue

            if scol not in dfp.columns:
                # robust fallback
                if base_key_canon(base_key) == "Híbrido":
                    if "Hybrid Sample" in dfp.columns:
                        scol = "Hybrid Sample"
                    elif "Sample" in dfp.columns:
                        scol = "Sample"
                    elif "Tamiz Sample" in dfp.columns:
                        scol = "Tamiz Sample"
                    else:
                        scol = dfp.columns[0]
                else:
                    scol = "Sample" if "Sample" in dfp.columns else dfp.columns[0]

            if gcol not in dfp.columns:
                gcol = "Group" if "Group" in dfp.columns else None

            n = len(dfp)
            xs = _fw_pp_column_list(dfp, x_key)
            ys = _fw_pp_column_list(dfp, y_key)
            ss = _fw_pp_column_list(dfp, scol)
            gs = _fw_pp_column_list(dfp, gcol) if gcol is not None else ["No Group"] * n
            markers = getattr(self, "group_markers", None) or {}
            colors = self.group_colors if isinstance(getattr(self, "group_colors", None), dict) else {}
            labels = getattr(self, "group_labels", None) or {}
            for i in range(n):
                try:
                    x = float(xs[i])
                    y = float(ys[i])
                except Exception:
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue

                grp = self._fw_effective_group(base_key, ss[i], gs[i])
                gk = self._group_key(grp, base_key)
                if selected_groups and (gk not in selected_groups):
                    continue
                gl = labels.get(gk, self._group_label(grp, base_key))
                color = colors.get(gk, GROUP_COLORS[0])
                sample_label = "" if ss[i] is None else str(ss[i])

                yield {
                    "x": x,
                    "y": y,
                    "sample": sample_label,
                    "group": str(grp),
                    "base": base_key,
                    "gkey": gk,
                    "glabel": gl,
                    "color": color,
                    "marker": markers.get(gk, "o"),
                }

    # --------------------------- Console ---------------------------

    def _format_params_table_text(self, title: str, df: pd.DataFrame, group_col: str):
        if df is None or df.empty:
            return f"{title}\n(no data)\n"

        # visible identity column
        if "Hybrid Sample" in df.columns:
            sample_col = "Hybrid Sample"
        elif "Sample" in df.columns:
            sample_col = "Sample"
        else:
            sample_col = df.columns[0]

        cols = [sample_col]
        if group_col in df.columns:
            cols.append(group_col)
        cols += [c for c in ["mean", "median", "sigma", "skewness", "kurtosis"] if c in df.columns]

        d = df.loc[:, cols].copy()

        # rename first column to "Sample" in text export for readability
        if sample_col != "Sample":
            d = d.rename(columns={sample_col: "Sample"})

        for c in ["mean", "median", "sigma", "skewness", "kurtosis"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")

        return f"{title}\n{d.to_string(index=False)}\n"

    def update_console(self):
        """
        Shows the parameter tables of the loaded datasets in the
        "Granulometric parameters" panel (modern table view, Block 6B).
        A plain-text copy is kept for compatibility (txt_console.toPlainText()).
        """
        if not hasattr(self, "txt_console"):
            return

        datasets = []
        chunks = []

        df = getattr(self, "param_results", None)
        if df is not None and not df.empty:
            gcol = getattr(self, "group_col", "Group")
            datasets.append(("Tamiz", df, gcol))
            chunks.append(self._format_params_table_text(f"[{base_name_display('Tamiz')}]", df, gcol))

        df = getattr(self, "param_results_laser", None)
        if df is not None and not df.empty:
            datasets.append(("Laser", df, "Group"))
            chunks.append(self._format_params_table_text(f"[{base_name_display('Laser')}]", df, "Group"))

        df = getattr(self, "param_results_hybrid", None)
        if df is not None and not df.empty:
            gcol = getattr(self, "group_col", "Group")
            if gcol not in df.columns and "Group" in df.columns:
                gcol = "Group"
            datasets.append(("Híbrido", df, gcol))
            chunks.append(self._format_params_table_text(f"[{base_name_display('Híbrido')}]", df, gcol))

        legacy_text = "\n".join(chunks) if chunks else "No parameters available. Load a dataset first."

        panel = getattr(self, "params_panel", None)
        if panel is not None and hasattr(panel, "set_datasets"):
            try:
                panel.set_datasets(
                    datasets,
                    method_label=_fw_method_display_name(getattr(self, "current_method", "")),
                    group_colors=getattr(self, "group_colors", None) or {},
                    group_markers=getattr(self, "group_markers", None) or {},
                    group_of=getattr(self, "_fw_effective_group", None),
                    legacy_text=legacy_text,
                )
                return
            except Exception:
                traceback.print_exc()

        try:
            self.txt_console.setPlainText(legacy_text)
        except Exception:
            pass

    # --------------------------- XY plot ---------------------------

    def plot_xy(self):
        """Main plot of the tab. Same drawing as View plot / export (Block 6D), sized to the window."""
        if not hasattr(self, "canvas_xy"):
            return
        try:
            fw_draw_xy(self, self.canvas_xy.figure, _fw_xy_style(self), export=False)
        except Exception:
            traceback.print_exc()
        self.canvas_xy.draw_idle()
        dlg = getattr(self, "_xy_view_dlg", None)
        if dlg is not None:
            try:
                if dlg.isVisible():
                    dlg.refresh_from_owner()
            except Exception:
                pass

    # --------------------------- Export helpers ---------------------------

    def export_canvas(self, canvas):
        if canvas is None or not hasattr(canvas, "figure"):
            QMessageBox.warning(self, tr("error"), "No figure available.")
            return

        start_dir = getattr(self, "last_dir", get_script_dir())
        file, _ = QFileDialog.getSaveFileName(
            self,
            tr("save_figure_title"),
            os.path.join(start_dir, "figure.png"),
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if not file:
            return

        try:
            canvas.figure.savefig(file, dpi=300, bbox_inches="tight")
            self.last_dir = os.path.dirname(file)
            QMessageBox.information(self, tr("success"), f"Saved:\n{file}")
        except Exception as e:
            QMessageBox.warning(self, tr("error"), f"Could not save figure:\n{e}")

    def save_current_tab(self):
        self.export_xy_plot()

    def export_params_to_excel(self):
        start_dir = getattr(self, "last_dir", get_script_dir())
        file, _ = QFileDialog.getSaveFileName(
            self,
            tr("save_params_title"),
            os.path.join(start_dir, "granulometric_parameters.xlsx"),
            "Excel (*.xlsx)"
        )
        if not file:
            return

        try:
            with pd.ExcelWriter(file, engine="openpyxl") as wr:
                meta = pd.DataFrame({
                    "field": ["export_time", "method", "theme"],
                    "value": [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        str(getattr(self, "current_method", "")),
                        str(getattr(self, "theme", "")),
                    ]
                })
                meta.to_excel(wr, index=False, sheet_name="Meta")

                if getattr(self, "param_results", None) is not None and not self.param_results.empty:
                    self.param_results.to_excel(wr, index=False, sheet_name="Sieve_Params")
                if getattr(self, "df_data", None) is not None and not self.df_data.empty:
                    self.df_data.to_excel(wr, index=False, sheet_name="Sieve_Data")

                if getattr(self, "param_results_laser", None) is not None and not self.param_results_laser.empty:
                    self.param_results_laser.to_excel(wr, index=False, sheet_name="Laser_Params")
                if getattr(self, "df_laser", None) is not None and not self.df_laser.empty:
                    self.df_laser.to_excel(wr, index=False, sheet_name="Laser_Data")

                if getattr(self, "param_results_hybrid", None) is not None and not self.param_results_hybrid.empty:
                    self.param_results_hybrid.to_excel(wr, index=False, sheet_name="Hybrid_Params")
                if getattr(self, "df_hybrid", None) is not None and not self.df_hybrid.empty:
                    self.df_hybrid.to_excel(wr, index=False, sheet_name="Hybrid_Data")

            self.last_dir = os.path.dirname(file)
            QMessageBox.information(self, tr("success"), f"Saved:\n{file}")
        except Exception as e:
            QMessageBox.warning(self, tr("error"), f"Could not export Excel:\n{e}")

# endregion





# region # === Block 11: Figure export, method selection, and theme (SAFE OVERRIDE, hybrid-safe) ===

# This block is kept for compatibility, aligned with the corrected hybrid logic:
# hybrid parameter recomputation now uses a UNIQUE hybrid identity
# ("Hybrid Sample" / "Sample") instead of collapsing everything by "Tamiz Sample".


def export_canvas(self, canvas):
    if canvas is None or not hasattr(canvas, "figure"):
        QMessageBox.warning(self, tr("error"), "No figure available.")
        return

    filters = "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"

    try:
        start_dir = None
        if getattr(self, "settings", None) is not None:
            start_dir = self.settings.value("last_dir", None)
        if not start_dir:
            start_dir = getattr(self, "last_dir", get_script_dir())
    except Exception:
        start_dir = get_script_dir()

    def _suggest_name():
        try:
            if canvas is getattr(self, "canvas_xy", None):
                return os.path.join(start_dir, "granulometric_parameters_plot.png")
        except Exception:
            pass
        return os.path.join(start_dir, "figure.png")

    file, sel = QFileDialog.getSaveFileName(
        self,
        tr("file_export_image"),
        _suggest_name(),
        filters
    )
    if not file:
        return

    if "." not in os.path.basename(file):
        if "SVG" in sel:
            file += ".svg"
        elif "PDF" in sel:
            file += ".pdf"
        else:
            file += ".png"

    try:
        canvas.figure.savefig(file, dpi=300, bbox_inches="tight")

        try:
            self.last_dir = os.path.dirname(file)
            if getattr(self, "settings", None) is not None:
                self.settings.setValue("last_dir", self.last_dir)
        except Exception:
            pass

        QMessageBox.information(self, tr("info"), "Image saved to:\n" + file)
    except Exception as e:
        QMessageBox.warning(self, tr("error"), f"Could not export figure:\n{e}")


def save_current_tab(self):
    self.export_xy_plot()


def choose_method(self):
    """
    Compatibility override aligned with Blocks 9, 10 and 12.
    Recomputes all parameter tables and refreshes plots safely.
    """
    dlg = QDialog(self)
    dlg.setWindowTitle(tr("cfg_method"))
    lay = QVBoxLayout(dlg)

    rb_fw = QRadioButton("Folk & Ward (1957)")
    rb_in = QRadioButton("Inman (1952)")
    rb_tr = QRadioButton("Trask (1932)")

    current = getattr(self, "current_method", "folkward")
    if current == "trask":
        rb_tr.setChecked(True)
    elif current == "inman":
        rb_in.setChecked(True)
    else:
        rb_fw.setChecked(True)

    lay.addWidget(QLabel("Method to compute grain-size parameters:"))
    lay.addWidget(rb_fw)
    lay.addWidget(rb_in)
    lay.addWidget(rb_tr)

    btn = QPushButton(tr("ok"))
    btn.clicked.connect(dlg.accept)
    lay.addWidget(btn)

    if not dlg.exec_():
        return

    if rb_tr.isChecked():
        new_method = "trask"
    elif rb_in.isChecked():
        new_method = "inman"
    else:
        new_method = "folkward"

    if new_method == getattr(self, "current_method", "folkward"):
        return

    self.current_method = new_method

    try:
        if getattr(self, "settings", None) is not None:
            self.settings.setValue("method", self.current_method)
    except Exception:
        pass

    self._recompute_all_params()


def _recompute_all_params(self):
    """
    Recompute parameters for Sieve / Laser / Hybrid using current method.

    IMPORTANT FIX:
    For Hybrid, use a UNIQUE hybrid identity:
      1) "Hybrid Sample"
      2) "Sample"
      3) "Tamiz Sample"   (fallback only for old files / old state)

    This prevents different hybrids of the same sieve sample from being merged.
    """

    # ---------------- SIEVE ----------------
    if getattr(self, "df_data", None) is not None and not getattr(self.df_data, "empty", True):
        phi_col = self.df_data.columns[0]
        samp_col = self.df_data.columns[1]
        wt_col = self.df_data.columns[2]
        grp_col = getattr(self, "group_col", "Group")

        out = []
        for s, g in self.df_data.groupby(samp_col):
            p = calculate_parameters(g[phi_col].tolist(), g[wt_col].tolist(), self.current_method)
            row = {"Sample": s, **p}
            if grp_col in g.columns:
                row[grp_col] = str(g[grp_col].iloc[0])
            out.append(row)

        self.param_results = pd.DataFrame(out)

    # ---------------- LASER ----------------
    if getattr(self, "df_laser", None) is not None and not getattr(self.df_laser, "empty", True):
        out = []
        for s, g in self.df_laser.groupby("Sample"):
            p = calculate_parameters(g["phi"].tolist(), g["Wt"].tolist(), self.current_method)
            row = {"Sample": s, "Group": str(g["Group"].iloc[0]) if "Group" in g.columns else "No Group", **p}
            out.append(row)

        self.param_results_laser = pd.DataFrame(out)

    # ---------------- HYBRID ----------------
    if getattr(self, "df_hybrid", None) is not None and not getattr(self.df_hybrid, "empty", True):
        dfh = self.df_hybrid.copy()

        # Unique identity priority:
        #   Hybrid Sample -> Sample -> Tamiz Sample (old fallback)
        if "Hybrid Sample" in dfh.columns:
            sample_col = "Hybrid Sample"
        elif "Sample" in dfh.columns:
            sample_col = "Sample"
        elif "Tamiz Sample" in dfh.columns:
            sample_col = "Tamiz Sample"
        else:
            sample_col = None

        wt_col = "wt%" if "wt%" in dfh.columns else ("Wt" if "Wt" in dfh.columns else None)
        phi_col = "phi" if "phi" in dfh.columns else None

        grp_pref = getattr(self, "group_col", "Group")
        if grp_pref in dfh.columns:
            grp_col = grp_pref
        elif "Group" in dfh.columns:
            grp_col = "Group"
        else:
            grp_col = None

        out = []

        if sample_col is not None and wt_col is not None and phi_col is not None:
            for s, g in dfh.groupby(sample_col):
                p = calculate_parameters(g[phi_col].tolist(), g[wt_col].tolist(), self.current_method)

                row = {"Sample": s, **p}

                # Preserve useful metadata for downstream blocks
                if "Hybrid Sample" in g.columns:
                    row["Hybrid Sample"] = str(g["Hybrid Sample"].iloc[0])
                else:
                    row["Hybrid Sample"] = str(s)

                if "Tamiz Sample" in g.columns:
                    row["Tamiz Sample"] = str(g["Tamiz Sample"].iloc[0])

                if "Laser Sample" in g.columns:
                    row["Laser Sample"] = str(g["Laser Sample"].iloc[0])

                if "Hybrid Method Code" in g.columns:
                    row["Hybrid Method Code"] = str(g["Hybrid Method Code"].iloc[0])

                if "Hybrid Method" in g.columns:
                    row["Hybrid Method"] = str(g["Hybrid Method"].iloc[0])

                if "phi*" in g.columns:
                    try:
                        row["phi*"] = float(g["phi*"].iloc[0])
                    except Exception:
                        row["phi*"] = g["phi*"].iloc[0]

                if grp_col is not None and grp_col in g.columns:
                    row[grp_col] = str(g[grp_col].iloc[0])
                elif "Group" not in row:
                    row["Group"] = "No Group"

                out.append(row)

        self.param_results_hybrid = pd.DataFrame(out)

    try:
        self._update_all_groups_and_colors()
    except Exception:
        pass

    try:
        self.update_console()
    except Exception:
        pass

    try:
        self.plot_xy()
    except Exception:
        pass

    try:
        self._update_db_menu_actions()
    except Exception:
        pass


def choose_theme(self):
    dlg = QDialog(self)
    dlg.setWindowTitle(tr("cfg_theme"))
    lay = QVBoxLayout(dlg)

    rb_light = QRadioButton("Light")
    rb_dark = QRadioButton("Dark")
    (rb_dark if getattr(self, "theme", "light") == "dark" else rb_light).setChecked(True)

    lay.addWidget(QLabel("Interface theme:"))
    lay.addWidget(rb_light)
    lay.addWidget(rb_dark)

    btn = QPushButton(tr("ok"))
    btn.clicked.connect(dlg.accept)
    lay.addWidget(btn)

    if not dlg.exec_():
        return

    self.theme = "dark" if rb_dark.isChecked() else "light"

    try:
        if getattr(self, "settings", None) is not None:
            self.settings.setValue("theme", self.theme)
    except Exception:
        pass

    try:
        qApp.setStyleSheet(DARK_STYLESHEET if self.theme == "dark" else LIGHT_STYLESHEET)
    except Exception:
        pass

    try:
        self.plot_xy()
    except Exception:
        pass

try:
    for _name in [
        "export_canvas",
        "save_current_tab",
        "choose_method",
        "_recompute_all_params",
        "choose_theme",
    ]:
        if _name in globals() and callable(globals()[_name]):
            setattr(MainWindow, _name, globals()[_name])
except Exception:
    pass

# endregion



# region # === Block 12 v5.1: Combine Sieve + Laser — symmetric multi-select UI + discrete phi* + no bottom hint ===
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QListWidget, QListWidgetItem, QMessageBox, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView
)
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np


# ---------- Utilities ----------
def _get_sieve_columns(self):
    phi_col = self.df_data.columns[0]
    samp_col = self.df_data.columns[1]
    wt_col = self.df_data.columns[2]
    return phi_col, samp_col, wt_col


def _sanitize_subtables(self, tamiz_sample, laser_sample):
    phi_col, sieve_samp_col, sieve_wt_col = _get_sieve_columns(self)
    sieve_sub = self.df_data[self.df_data[sieve_samp_col].astype(str) == str(tamiz_sample)].copy()
    laser_sub = self.df_laser[self.df_laser["Sample"].astype(str) == str(laser_sample)].copy()

    for c in (phi_col, sieve_wt_col):
        sieve_sub[c] = pd.to_numeric(sieve_sub[c], errors="coerce")
    for c in ("phi", "Wt"):
        laser_sub[c] = pd.to_numeric(laser_sub[c], errors="coerce")

    sieve_sub = sieve_sub[np.isfinite(sieve_sub[phi_col]) & np.isfinite(sieve_sub[sieve_wt_col])]
    laser_sub = laser_sub[np.isfinite(laser_sub["phi"]) & np.isfinite(laser_sub["Wt"])]
    return phi_col, sieve_sub, laser_sub, sieve_wt_col


def _attach_group_if_any(self, df_out, sieve_sub):
    try:
        if hasattr(self, "group_col") and self.group_col in self.df_data.columns:
            gval = sieve_sub[self.group_col].iloc[0] if not sieve_sub.empty else None
            df_out[self.group_col] = gval
    except Exception:
        pass
    return df_out


def _try_recompute_everything(self):
    for cand in ("_recompute_all_params", "recompute_all", "recalculate_all",
                 "compute_all", "update_all_plots", "refresh_all"):
        fn = getattr(self, cand, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass

    for cand in ("update_console", "plot_xy"):
        fn = getattr(self, cand, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _hybrid_method_label(method_code):
    method_code = str(method_code).lower().strip()
    mapping = {
        "m1": "M1",
        "m2": "M2",
        "m3": "M3",
    }
    return mapping.get(method_code, "M1")


def _hybrid_suffix_from_index(idx_zero_based):
    """
    0 -> A
    1 -> B
    ...
    25 -> Z
    26 -> AA
    27 -> AB
    ...
    """
    n = int(idx_zero_based)
    if n < 0:
        n = 0

    out = ""
    while True:
        n, rem = divmod(n, 26)
        out = chr(ord("A") + rem) + out
        if n == 0:
            break
        n -= 1
    return out


def _make_hybrid_sample_name(tamiz_sample, ordinal_idx_zero_based):
    """
    Visible short hybrid name:
      Tamiz-A, Tamiz-B, Tamiz-C, ...
    """
    t = str(tamiz_sample).strip()
    suffix = _hybrid_suffix_from_index(ordinal_idx_zero_based)
    return f"{t}-{suffix}"


def _make_hybrid_long_label(hybrid_sample_name, tamiz_sample, laser_sample, phi_cut, method_code="m1"):
    """
    Longer descriptive label stored as metadata only.
    Not used as the visible histogram title.
    """
    t = str(tamiz_sample).strip()
    l = str(laser_sample).strip()
    mlab = _hybrid_method_label(method_code)
    hs = str(hybrid_sample_name).strip()
    return f"{hs} | {t} ↔ {l} | {mlab} | φ*={float(phi_cut):.2f}"


def _hybrid_constant_fields(tamiz_sample, laser_sample, phi_cut, method_code, hybrid_sample_name):
    """
    Constant metadata columns inserted in every row of one hybrid.
    """
    hybrid_name = str(hybrid_sample_name).strip()
    long_label = _make_hybrid_long_label(hybrid_name, tamiz_sample, laser_sample, phi_cut, method_code)

    return {
        "Hybrid Sample": hybrid_name,
        "Sample": hybrid_name,
        "Hybrid Long Label": long_label,
        "Tamiz Sample": str(tamiz_sample),
        "Laser Sample": str(laser_sample),
        "Hybrid Method Code": str(method_code).lower().strip(),
        "Hybrid Method": _hybrid_method_label(method_code),
        "phi*": float(phi_cut),
    }


def _hybrid_collect_phi_choices(self):
    """
    Valid discrete phi* values.

    Priority:
      1) actual phi bins present in self.df_laser
      2) synthetic range based on self.laser_step (0.5 or 1.0)
    """
    try:
        if getattr(self, "df_laser", None) is not None and not self.df_laser.empty and "phi" in self.df_laser.columns:
            vals = pd.to_numeric(self.df_laser["phi"], errors="coerce").dropna().astype(float).unique()
            vals = np.array(sorted(vals), dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                return [float(np.round(v, 6)) for v in vals.tolist()]
    except Exception:
        pass

    try:
        step = float(getattr(self, "laser_step", 0.5))
    except Exception:
        step = 0.5

    if not (np.isclose(step, 0.5, atol=1e-9) or np.isclose(step, 1.0, atol=1e-9)):
        step = 0.5

    vals = np.arange(-10.0, 10.0 + step * 0.5, step, dtype=float)
    return [float(np.round(v, 6)) for v in vals.tolist()]


def _hybrid_phi_fmt(v):
    v = float(v)
    if np.isclose(v, round(v), atol=1e-9):
        return f"{int(round(v))}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


class _HybridPhiChoiceCombo(QComboBox):
    """
    Discrete phi* selector.
    No free decimals, no 0.1 stepping.
    """
    INVALID_VALUE = -9999.0

    def __init__(self, phi_choices, parent=None):
        super().__init__(parent)
        self._phi_choices = [float(v) for v in phi_choices]
        self.addItem("—", self.INVALID_VALUE)
        for v in self._phi_choices:
            self.addItem(_hybrid_phi_fmt(v), float(v))
        self.setCurrentIndex(0)

    def value(self):
        try:
            return float(self.currentData())
        except Exception:
            return float(self.INVALID_VALUE)

    def minimum(self):
        return float(self.INVALID_VALUE)

    def set_value(self, v):
        try:
            vf = float(v)
        except Exception:
            self.setCurrentIndex(0)
            return
        idx = self.findData(vf)
        if idx >= 0:
            self.setCurrentIndex(idx)
        else:
            self.setCurrentIndex(0)


class _MethodChoiceCombo(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem("METHOD 1 (original)", "m1")
        self.addItem("METHOD 2 (EDLT + SG)", "m2")
        self.addItem("METHOD 3 (placeholder)", "m3")

    def code(self):
        try:
            return str(self.currentData())
        except Exception:
            return "m1"

    def set_code(self, code):
        idx = self.findData(str(code).lower().strip())
        if idx >= 0:
            self.setCurrentIndex(idx)


# ---------- Combination methods ----------
def _build_hybrid_method1(self, tamiz_sample, laser_sample, phi_cut, method_code="m1", hybrid_sample_name=None):
    """
    Original:
      - coarse = sieve(<phi*)
      - fines  = laser(>=phi*) scaled to EDLT
    """
    phi_col, sieve_sub, laser_sub, sieve_wt_col = _sanitize_subtables(self, tamiz_sample, laser_sample)

    sieve_keep = sieve_sub[sieve_sub[phi_col] < float(phi_cut)].copy()
    laser_keep = laser_sub[laser_sub["phi"] >= float(phi_cut)].copy()

    s_coarse = float(sieve_keep[sieve_wt_col].sum()) if not sieve_keep.empty else 0.0
    l_fine = float(laser_keep["Wt"].sum()) if not laser_keep.empty else 0.0
    target_fine = max(0.0, 100.0 - s_coarse)
    scale = (target_fine / l_fine) if l_fine > 0 else 0.0

    if hybrid_sample_name is None:
        hybrid_sample_name = str(tamiz_sample).strip()

    const = _hybrid_constant_fields(tamiz_sample, laser_sample, phi_cut, method_code, hybrid_sample_name)

    rows = []
    if not sieve_keep.empty:
        for _, r in sieve_keep.iterrows():
            row = dict(const)
            row.update({
                "phi": float(r[phi_col]),
                "wt%": float(r[sieve_wt_col]),
            })
            rows.append(row)

    if not laser_keep.empty and scale > 0:
        for _, r in laser_keep.iterrows():
            row = dict(const)
            row.update({
                "phi": float(r["phi"]),
                "wt%": float(r["Wt"] * scale),
            })
            rows.append(row)

    if not rows:
        empty_cols = [
            "Hybrid Sample", "Sample", "Hybrid Long Label", "Tamiz Sample", "Laser Sample",
            "Hybrid Method Code", "Hybrid Method", "phi*", "phi", "wt%"
        ]
        return pd.DataFrame(columns=empty_cols), {
            "hybrid_sample": const["Hybrid Sample"],
            "hybrid_long_label": const["Hybrid Long Label"],
            "method": str(method_code).lower().strip(),
            "phi*": float(phi_cut),
            "phi_cut": float(phi_cut),
            "tamiz": str(tamiz_sample),
            "laser": str(laser_sample),
            "sieve_coarse_sum": 0.0,
            "laser_fine_sum": 0.0,
            "laser_fine_used": 0.0,
            "laser_fine_residue": 0.0,
            "scale_k": 0.0,
        }

    df_h = pd.DataFrame(rows)
    gb_cols = [
        "Hybrid Sample", "Sample", "Hybrid Long Label", "Tamiz Sample", "Laser Sample",
        "Hybrid Method Code", "Hybrid Method", "phi*", "phi"
    ]
    df_h = df_h.groupby(gb_cols, as_index=False)["wt%"].sum()
    df_h.sort_values(["Hybrid Sample", "phi"], inplace=True)

    tot = df_h.groupby("Hybrid Sample")["wt%"].transform("sum").replace(0, np.nan)
    df_h["wt%"] = np.where(np.isfinite(tot), df_h["wt%"] * (100.0 / tot), df_h["wt%"])

    df_h = _attach_group_if_any(self, df_h, sieve_sub)

    meta = {
        "hybrid_sample": const["Hybrid Sample"],
        "hybrid_long_label": const["Hybrid Long Label"],
        "method": str(method_code).lower().strip(),
        "phi*": float(phi_cut),
        "phi_cut": float(phi_cut),
        "tamiz": str(tamiz_sample),
        "laser": str(laser_sample),
        "sieve_coarse_sum": float(s_coarse),
        "laser_fine_sum": float(l_fine),
        "laser_fine_used": float(l_fine * scale),
        "laser_fine_residue": float(max(0.0, l_fine - l_fine * scale)),
        "scale_k": float(scale),
    }
    return df_h, meta


def _build_hybrid_method2(self, tamiz_sample, laser_sample, phi_cut, phi_round=2, method_code="m2", hybrid_sample_name=None):
    """
    EDLT + SG:
      - Fines (>=phi*): F_phi = k * L_phi, with k = EDLT/sum(L_fines)
      - SG (<phi* with exact matches): S'_phi = S_phi + k*L_phi
      - Coarse bins without laser (<phi* no match) remain fixed
      - Protected renormalization: only (SG+Fines) scaled by alpha=(100-Tfixed)/Mpre
    """
    phi_col, sieve_sub, laser_sub, sieve_wt_col = _sanitize_subtables(self, tamiz_sample, laser_sample)

    s_df = pd.DataFrame({
        "phi": np.round(sieve_sub[phi_col].astype(float).values, phi_round),
        "S": sieve_sub[sieve_wt_col].astype(float).values
    }).groupby("phi", as_index=False)["S"].sum()

    l_df = pd.DataFrame({
        "phi": np.round(laser_sub["phi"].astype(float).values, phi_round),
        "L": laser_sub["Wt"].astype(float).values
    }).groupby("phi", as_index=False)["L"].sum()

    EDLT = float(s_df.loc[s_df["phi"] >= float(phi_cut), "S"].sum())
    L_fines_sum = float(l_df.loc[l_df["phi"] >= float(phi_cut), "L"].sum())
    k = (EDLT / L_fines_sum) if L_fines_sum > 0 else 0.0

    fines = l_df[l_df["phi"] >= float(phi_cut)].copy()
    fines["wt%"] = k * fines["L"]

    s_coarse = s_df[s_df["phi"] < float(phi_cut)].copy()
    l_coarse = l_df[l_df["phi"] < float(phi_cut)].copy()

    ph_coinc = np.intersect1d(s_coarse["phi"].values, l_coarse["phi"].values)
    ph_fixed = np.setdiff1d(s_coarse["phi"].values, l_coarse["phi"].values)

    if ph_coinc.size:
        s_co = s_coarse[s_coarse["phi"].isin(ph_coinc)].set_index("phi")
        l_co = l_coarse[l_coarse["phi"].isin(ph_coinc)].set_index("phi")
        sg_adj = (s_co["S"] + k * l_co["L"]).reset_index()
        sg_adj.columns = ["phi", "wt%"]
    else:
        sg_adj = pd.DataFrame(columns=["phi", "wt%"])

    fixed = s_coarse[s_coarse["phi"].isin(ph_fixed)].copy()
    fixed = fixed.rename(columns={"S": "wt%"}).loc[:, ["phi", "wt%"]]

    Tfijo = float(fixed["wt%"].sum()) if not fixed.empty else 0.0
    Mpre = float(sg_adj["wt%"].sum()) + float(fines["wt%"].sum())
    Mtarget = max(0.0, 100.0 - Tfijo)
    alpha = (Mtarget / Mpre) if Mpre > 0 else 1.0

    sg_adj["wt%"] = sg_adj["wt%"] * alpha
    fines["wt%"] = fines["wt%"] * alpha

    if hybrid_sample_name is None:
        hybrid_sample_name = str(tamiz_sample).strip()

    const = _hybrid_constant_fields(tamiz_sample, laser_sample, phi_cut, method_code, hybrid_sample_name)

    rows = []
    for _, r in fixed.iterrows():
        row = dict(const)
        row.update({
            "phi": float(r["phi"]),
            "wt%": float(r["wt%"]),
        })
        rows.append(row)

    for _, r in sg_adj.iterrows():
        row = dict(const)
        row.update({
            "phi": float(r["phi"]),
            "wt%": float(r["wt%"]),
        })
        rows.append(row)

    for _, r in fines.iterrows():
        row = dict(const)
        row.update({
            "phi": float(r["phi"]),
            "wt%": float(r["wt%"]),
        })
        rows.append(row)

    if not rows:
        empty_cols = [
            "Hybrid Sample", "Sample", "Hybrid Long Label", "Tamiz Sample", "Laser Sample",
            "Hybrid Method Code", "Hybrid Method", "phi*", "phi", "wt%"
        ]
        return pd.DataFrame(columns=empty_cols), {
            "hybrid_sample": const["Hybrid Sample"],
            "hybrid_long_label": const["Hybrid Long Label"],
            "method": str(method_code).lower().strip(),
            "phi*": float(phi_cut),
            "phi_cut": float(phi_cut),
            "tamiz": str(tamiz_sample),
            "laser": str(laser_sample),
            "k": 0.0,
            "EDLT": float(EDLT),
            "alpha": 1.0,
            "coarse_fixed_sum": float(Tfijo),
            "pre_mass_sg_plus_fines": float(Mpre),
            "target_mass_sg_plus_fines": float(Mtarget),
        }

    df_h = pd.DataFrame(rows)
    gb_cols = [
        "Hybrid Sample", "Sample", "Hybrid Long Label", "Tamiz Sample", "Laser Sample",
        "Hybrid Method Code", "Hybrid Method", "phi*", "phi"
    ]
    df_h = df_h.groupby(gb_cols, as_index=False)["wt%"].sum()
    df_h.sort_values(["Hybrid Sample", "phi"], inplace=True)

    tot = df_h.groupby("Hybrid Sample")["wt%"].transform("sum").replace(0, np.nan)
    df_h["wt%"] = np.where(np.isfinite(tot), df_h["wt%"] * (100.0 / tot), df_h["wt%"])

    df_h = _attach_group_if_any(self, df_h, sieve_sub)

    meta = {
        "hybrid_sample": const["Hybrid Sample"],
        "hybrid_long_label": const["Hybrid Long Label"],
        "method": str(method_code).lower().strip(),
        "phi*": float(phi_cut),
        "phi_cut": float(phi_cut),
        "tamiz": str(tamiz_sample),
        "laser": str(laser_sample),
        "k": float(k),
        "EDLT": float(EDLT),
        "alpha": float(alpha),
        "coarse_fixed_sum": float(Tfijo),
        "pre_mass_sg_plus_fines": float(Mpre),
        "target_mass_sg_plus_fines": float(Mtarget),
    }
    return df_h, meta


def _build_hybrid_method3(self, tamiz_sample, laser_sample, phi_cut, hybrid_sample_name=None):
    """
    Placeholder: for now it uses Method 1 math,
    but preserves its own identity and metadata as M3.
    """
    return _build_hybrid_method1(
        self,
        tamiz_sample,
        laser_sample,
        phi_cut,
        method_code="m3",
        hybrid_sample_name=hybrid_sample_name
    )


def _build_hybrid_for_pair(self, tamiz_sample, laser_sample, phi_cut, method="m1", hybrid_sample_name=None):
    method = (method or "m1").lower()
    if method == "m1":
        return _build_hybrid_method1(self, tamiz_sample, laser_sample, phi_cut, method_code="m1", hybrid_sample_name=hybrid_sample_name)
    if method == "m2":
        return _build_hybrid_method2(self, tamiz_sample, laser_sample, phi_cut, method_code="m2", hybrid_sample_name=hybrid_sample_name)
    if method == "m3":
        return _build_hybrid_method3(self, tamiz_sample, laser_sample, phi_cut, hybrid_sample_name=hybrid_sample_name)
    return _build_hybrid_method1(self, tamiz_sample, laser_sample, phi_cut, method_code="m1", hybrid_sample_name=hybrid_sample_name)


# ---------- Dialog ----------
class _CombineDialog(QDialog):
    """
    Dynamic hybrid table.

    Workflow:
      1) select one or many sieve samples
      2) select one or many laser samples
      3) choose default method and default phi*
      4) add all combinations to the table
      5) optionally edit method and phi* row by row
      6) reorder rows if you want A/B/C/D in another order
      7) press OK
    """
    def __init__(self, sieve_samples, laser_samples, phi_choices, default_phi=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine sieve and laser")
        self.setModal(True)
        self.resize(1180, 670)
        self.pairs = []

        self.sieve_samples = [str(s) for s in sieve_samples]
        self.laser_samples = [str(s) for s in laser_samples]
        self.phi_choices = [float(v) for v in phi_choices]

        v = QVBoxLayout(self)

        # ---------- Top selection area ----------
        top = QHBoxLayout()

        # Sieve multi-select
        left = QVBoxLayout()
        left.addWidget(QLabel("Sieve sample(s):"))
        self.lst_tamiz = QListWidget()
        self.lst_tamiz.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for s in self.sieve_samples:
            self.lst_tamiz.addItem(QListWidgetItem(s))
        left.addWidget(self.lst_tamiz)
        top.addLayout(left, 3)

        # Laser multi-select
        mid = QVBoxLayout()
        mid.addWidget(QLabel("Laser sample(s):"))
        self.lst_laser = QListWidget()
        self.lst_laser.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for s in self.laser_samples:
            self.lst_laser.addItem(QListWidgetItem(s))
        mid.addWidget(self.lst_laser)
        top.addLayout(mid, 3)

        # Default controls
        right = QVBoxLayout()
        right.addWidget(QLabel("Default method:"))
        self.cb_method_default = _MethodChoiceCombo()
        right.addWidget(self.cb_method_default)

        right.addWidget(QLabel("Default φ*:"))
        self.cb_phi_default = _HybridPhiChoiceCombo(self.phi_choices)
        if default_phi is not None:
            self.cb_phi_default.set_value(default_phi)
        right.addWidget(self.cb_phi_default)

        self.btn_add_combinations = QPushButton("Add combinations")
        self.btn_add_combinations.clicked.connect(self._on_add_combinations)
        right.addWidget(self.btn_add_combinations)

        right.addStretch(1)
        top.addLayout(right, 2)

        v.addLayout(top)

        # ---------- Assignment table ----------
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Sieve", "Laser", "Method", "φ*", "Result name"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)

        try:
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        except Exception:
            pass

        v.addWidget(self.table)

        # ---------- Row actions ----------
        row_actions = QHBoxLayout()
        self.btn_up = QPushButton("Move up")
        self.btn_down = QPushButton("Move down")
        self.btn_delete = QPushButton("Delete row")
        self.btn_up.clicked.connect(self._move_current_row_up)
        self.btn_down.clicked.connect(self._move_current_row_down)
        self.btn_delete.clicked.connect(self._delete_current_row)

        row_actions.addWidget(self.btn_up)
        row_actions.addWidget(self.btn_down)
        row_actions.addWidget(self.btn_delete)
        row_actions.addStretch(1)
        v.addLayout(row_actions)

        # ---------- Bottom buttons ----------
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(self.btn_ok)
        bottom.addWidget(self.btn_cancel)
        v.addLayout(bottom)

    # ----- internal helpers -----
    def _selected_sieves(self):
        out = []
        for it in self.lst_tamiz.selectedItems():
            try:
                out.append(str(it.text()).strip())
            except Exception:
                pass
        return out

    def _selected_lasers(self):
        out = []
        for it in self.lst_laser.selectedItems():
            try:
                out.append(str(it.text()).strip())
            except Exception:
                pass
        return out

    def _current_default_phi(self):
        return self.cb_phi_default.value()

    def _current_default_method(self):
        return self.cb_method_default.code()

    def _config_exists_exact(self, tamiz_name, laser_name, method_code, phi_value):
        for r in range(self.table.rowCount()):
            it_t = self.table.item(r, 0)
            it_l = self.table.item(r, 1)
            w_m = self.table.cellWidget(r, 2)
            w_p = self.table.cellWidget(r, 3)
            if it_t is None or it_l is None or w_m is None or w_p is None:
                continue

            t0 = it_t.text().strip()
            l0 = it_l.text().strip()
            m0 = w_m.code() if hasattr(w_m, "code") else "m1"
            try:
                p0 = float(w_p.value())
            except Exception:
                p0 = _HybridPhiChoiceCombo.INVALID_VALUE

            same_phi = np.isclose(float(phi_value), float(p0), atol=1e-9)
            if t0 == str(tamiz_name).strip() and l0 == str(laser_name).strip() and m0 == str(method_code).strip() and same_phi:
                return True
        return False

    def _make_method_widget(self, method_code="m1"):
        w = _MethodChoiceCombo()
        w.set_code(method_code)
        return w

    def _make_phi_widget(self, phi_value=None):
        w = _HybridPhiChoiceCombo(self.phi_choices)
        if phi_value is not None:
            w.set_value(phi_value)
        return w

    def _insert_row(self, tamiz_name, laser_name, method_code=None, phi_value=None):
        if method_code is None:
            method_code = self._current_default_method()

        if phi_value is None:
            pv = self._current_default_phi()
            phi_value = None if np.isclose(float(pv), float(_HybridPhiChoiceCombo.INVALID_VALUE), atol=1e-9) else float(pv)

        check_phi = phi_value if phi_value is not None else _HybridPhiChoiceCombo.INVALID_VALUE
        if self._config_exists_exact(tamiz_name, laser_name, method_code, check_phi):
            return False

        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(str(tamiz_name)))
        self.table.setItem(row, 1, QTableWidgetItem(str(laser_name)))
        self.table.setCellWidget(row, 2, self._make_method_widget(method_code))
        self.table.setCellWidget(row, 3, self._make_phi_widget(phi_value))
        self.table.setItem(row, 4, QTableWidgetItem(""))

        self._refresh_result_names()
        return True

    def _refresh_result_names(self):
        """
        Recompute visible hybrid names based on CURRENT ROW ORDER,
        independently for each sieve sample.
        """
        per_tamiz_counter = {}
        for r in range(self.table.rowCount()):
            it_t = self.table.item(r, 0)
            it_res = self.table.item(r, 4)
            if it_t is None:
                continue
            tamiz_name = it_t.text().strip()
            idx = per_tamiz_counter.get(tamiz_name, 0)
            per_tamiz_counter[tamiz_name] = idx + 1
            hybrid_name = _make_hybrid_sample_name(tamiz_name, idx)

            if it_res is None:
                it_res = QTableWidgetItem(hybrid_name)
                self.table.setItem(r, 4, it_res)
            else:
                it_res.setText(hybrid_name)

    def _row_payload(self, row_idx):
        it_t = self.table.item(row_idx, 0)
        it_l = self.table.item(row_idx, 1)
        it_name = self.table.item(row_idx, 4)
        w_m = self.table.cellWidget(row_idx, 2)
        w_p = self.table.cellWidget(row_idx, 3)

        if it_t is None or it_l is None or it_name is None or w_m is None or w_p is None:
            return None

        try:
            method_code = w_m.code()
        except Exception:
            method_code = "m1"

        try:
            phi_val = float(w_p.value())
        except Exception:
            phi_val = float(_HybridPhiChoiceCombo.INVALID_VALUE)

        return {
            "tamiz": it_t.text().strip(),
            "laser": it_l.text().strip(),
            "method": method_code,
            "phi": phi_val,
            "hybrid_sample": it_name.text().strip(),
        }

    def _restore_row_from_payload(self, row_idx, payload):
        self.table.setItem(row_idx, 0, QTableWidgetItem(str(payload["tamiz"])))
        self.table.setItem(row_idx, 1, QTableWidgetItem(str(payload["laser"])))
        self.table.setCellWidget(row_idx, 2, self._make_method_widget(payload.get("method", "m1")))

        phi_val = payload.get("phi", _HybridPhiChoiceCombo.INVALID_VALUE)
        if np.isclose(float(phi_val), float(_HybridPhiChoiceCombo.INVALID_VALUE), atol=1e-9):
            self.table.setCellWidget(row_idx, 3, self._make_phi_widget(None))
        else:
            self.table.setCellWidget(row_idx, 3, self._make_phi_widget(phi_val))

        self.table.setItem(row_idx, 4, QTableWidgetItem(str(payload.get("hybrid_sample", ""))))

    # ----- actions -----
    def _on_add_combinations(self):
        tamices = self._selected_sieves()
        lasers = self._selected_lasers()

        if not tamices or not lasers:
            QMessageBox.warning(self, "Error", "Select at least one sieve sample and at least one laser sample.")
            return

        added = 0
        for tamiz_name in tamices:
            for laser_name in lasers:
                if self._insert_row(tamiz_name, laser_name):
                    added += 1

        if added == 0:
            QMessageBox.information(self, "Info", "All selected configurations already exist in the table.")

    def _delete_current_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._refresh_result_names()

    def _move_current_row_up(self):
        r = self.table.currentRow()
        if r <= 0:
            return

        p_cur = self._row_payload(r)
        p_prev = self._row_payload(r - 1)
        if p_cur is None or p_prev is None:
            return

        self._restore_row_from_payload(r - 1, p_cur)
        self._restore_row_from_payload(r, p_prev)
        self.table.setCurrentCell(r - 1, 0)
        self._refresh_result_names()

    def _move_current_row_down(self):
        r = self.table.currentRow()
        if r < 0 or r >= self.table.rowCount() - 1:
            return

        p_cur = self._row_payload(r)
        p_next = self._row_payload(r + 1)
        if p_cur is None or p_next is None:
            return

        self._restore_row_from_payload(r, p_next)
        self._restore_row_from_payload(r + 1, p_cur)
        self.table.setCurrentCell(r + 1, 0)
        self._refresh_result_names()

    def _on_accept(self):
        n = self.table.rowCount()
        if n == 0:
            QMessageBox.warning(self, "Error", "No hybrid configuration was added.")
            return

        exact_configs = set()
        out = []

        self._refresh_result_names()

        for r in range(n):
            payload = self._row_payload(r)
            if payload is None:
                QMessageBox.warning(self, "Error", "There is an incomplete row in the table.")
                return

            phi_val = float(payload["phi"])
            if np.isclose(phi_val, float(_HybridPhiChoiceCombo.INVALID_VALUE), atol=1e-9):
                QMessageBox.warning(self, "Error", "You must choose a valid φ* for every row.")
                return

            key = (
                payload["tamiz"],
                payload["laser"],
                payload["method"],
                round(phi_val, 9),
            )
            if key in exact_configs:
                QMessageBox.warning(self, "Error", "An exact duplicate configuration exists in the table.")
                return
            exact_configs.add(key)

            payload["hybrid_long_label"] = _make_hybrid_long_label(
                payload["hybrid_sample"],
                payload["tamiz"],
                payload["laser"],
                payload["phi"],
                payload["method"]
            )
            out.append(payload)

        self.pairs = out
        self.accept()


# ---------- Orchestrator ----------
def combinar_tamiz_laser(self):
    if getattr(self, "df_data", None) is None or self.df_data is None or self.df_data.empty:
        QMessageBox.warning(self, "Error", "Load the Sieve dataset first.")
        return
    if getattr(self, "df_laser", None) is None or self.df_laser is None or self.df_laser.empty:
        QMessageBox.warning(self, "Error", "Load the Laser dataset first.")
        return

    try:
        sieve_samp_col = self.df_data.columns[1]
        sieve_samples = sorted([str(s) for s in self.df_data[sieve_samp_col].dropna().astype(str).unique()])
    except Exception:
        sieve_samples = []

    try:
        laser_samples = sorted([str(s) for s in self.df_laser["Sample"].dropna().astype(str).unique()])
    except Exception:
        laser_samples = []

    if not sieve_samples or not laser_samples:
        QMessageBox.warning(self, "Error", "No sample names detected in Sieve/Laser.")
        return

    phi_choices = _hybrid_collect_phi_choices(self)
    default_phi = None

    dlg = _CombineDialog(
        sieve_samples=sieve_samples,
        laser_samples=laser_samples,
        phi_choices=phi_choices,
        default_phi=default_phi,
        parent=self
    )
    if not dlg.exec_():
        return

    pairs = dlg.pairs or []
    if not pairs:
        QMessageBox.information(self, "Info", "No Sieve–Laser configuration was added.")
        return

    all_rows = []
    meta = {}

    for p in pairs:
        t = p.get("tamiz")
        l = p.get("laser")
        phi = p.get("phi")
        method = p.get("method", "m1")
        hybrid_sample_name = p.get("hybrid_sample")

        try:
            df_h, m = _build_hybrid_for_pair(
                self,
                t,
                l,
                phi,
                method=method,
                hybrid_sample_name=hybrid_sample_name
            )

            if not df_h.empty:
                all_rows.append(df_h)

            hybrid_key = str(m.get("hybrid_sample", hybrid_sample_name))
            meta[hybrid_key] = m

        except Exception as e:
            print("Combine error:", t, l, e)

    if not all_rows:
        QMessageBox.warning(self, "Error", "Could not build a Hybrid dataset with the selected configurations.")
        return

    self.df_hybrid = pd.concat(all_rows, ignore_index=True)

    group_cols = [
        c for c in [
            "Hybrid Sample", "Sample", "Hybrid Long Label", "Tamiz Sample", "Laser Sample",
            "Hybrid Method Code", "Hybrid Method", "phi*", "phi"
        ]
        if c in self.df_hybrid.columns
    ]

    if group_cols:
        self.df_hybrid = (
            self.df_hybrid
            .groupby(group_cols, as_index=False)["wt%"]
            .sum()
            .sort_values(["Hybrid Sample", "phi"], kind="mergesort")
            .reset_index(drop=True)
        )

        if hasattr(self, "group_col") and self.group_col in all_rows[0].columns:
            grp_map = {}
            for df_part in all_rows:
                if self.group_col in df_part.columns and "Hybrid Sample" in df_part.columns:
                    sub = df_part[["Hybrid Sample", self.group_col]].drop_duplicates()
                    for _, rr in sub.iterrows():
                        grp_map[str(rr["Hybrid Sample"])] = rr[self.group_col]
            if grp_map:
                self.df_hybrid[self.group_col] = self.df_hybrid["Hybrid Sample"].map(grp_map)

    self._hybrid_meta = meta

    try:
        self.df_hybrid_plot = self.df_hybrid.rename(columns={"wt%": "Wt"}).copy()
    except Exception:
        self.df_hybrid_plot = self.df_hybrid.copy()

    try:
        self._update_db_menu_actions()
    except Exception:
        pass

    _try_recompute_everything(self)

    try:
        QMessageBox.information(self, "Info", "Hybrid generated successfully.")
    except Exception:
        pass


if "MainWindow" in globals():
    setattr(MainWindow, "combinar_tamiz_laser", combinar_tamiz_laser)

# endregion



# region # === Block 13: Export parameters to Excel (SAFE OVERRIDE) ===
def export_params_to_excel(self):
    """
    Export computed parameters (Sieve, Laser, and Hybrid if present) to an Excel workbook.
    Also creates a README sheet with brief notes.
    """
    has_tamiz = getattr(self, "param_results", None) is not None and not getattr(self.param_results, "empty", True)
    has_laser = getattr(self, "param_results_laser", None) is not None and not getattr(self.param_results_laser, "empty", True)
    has_hybrid = getattr(self, "param_results_hybrid", None) is not None and not getattr(self.param_results_hybrid, "empty", True)

    if not (has_tamiz or has_laser or has_hybrid):
        QMessageBox.information(self, tr("info"), "There are no parameters to export.")
        return

    try:
        start_dir = getattr(self, "last_dir", get_script_dir())
    except Exception:
        start_dir = ""

    suggested = os.path.join(start_dir, "Parameters.xlsx") if start_dir else "Parameters.xlsx"

    file, _ = QFileDialog.getSaveFileName(
        self,
        tr("btn_export_params"),
        suggested,
        "Excel files (*.xlsx)"
    )
    if not file:
        return
    if not file.lower().endswith(".xlsx"):
        file += ".xlsx"

    readme_rows = [
        "This workbook was exported from the application.",
        "Sheets may include: README, Meta, Sieve, Laser, Hybrid, and raw-data sheets if present.",
        "Values use units consistent with the selected method.",
    ]
    df_readme = pd.DataFrame({"Info": readme_rows})

    df_meta = pd.DataFrame({
        "Field": [
            "Export time",
            "Method",
            "Theme",
        ],
        "Value": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(getattr(self, "current_method", "")),
            str(getattr(self, "theme", "")),
        ]
    })

    def _write_excel(path):
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
                df_readme.to_excel(xw, index=False, sheet_name="README")
                df_meta.to_excel(xw, index=False, sheet_name="Meta")

                if has_tamiz:
                    self.param_results.to_excel(xw, index=False, sheet_name="Sieve")
                if has_laser:
                    self.param_results_laser.to_excel(xw, index=False, sheet_name="Laser")
                if has_hybrid:
                    self.param_results_hybrid.to_excel(xw, index=False, sheet_name="Hybrid")

                if getattr(self, "df_data", None) is not None and not getattr(self.df_data, "empty", True):
                    self.df_data.to_excel(xw, index=False, sheet_name="Sieve_Data")
                if getattr(self, "df_laser", None) is not None and not getattr(self.df_laser, "empty", True):
                    self.df_laser.to_excel(xw, index=False, sheet_name="Laser_Data")
                if getattr(self, "df_hybrid", None) is not None and not getattr(self.df_hybrid, "empty", True):
                    self.df_hybrid.to_excel(xw, index=False, sheet_name="Hybrid_Data")
            return True
        except Exception:
            pass

        with pd.ExcelWriter(path) as xw:
            df_readme.to_excel(xw, index=False, sheet_name="README")
            df_meta.to_excel(xw, index=False, sheet_name="Meta")

            if has_tamiz:
                self.param_results.to_excel(xw, index=False, sheet_name="Sieve")
            if has_laser:
                self.param_results_laser.to_excel(xw, index=False, sheet_name="Laser")
            if has_hybrid:
                self.param_results_hybrid.to_excel(xw, index=False, sheet_name="Hybrid")

            if getattr(self, "df_data", None) is not None and not getattr(self.df_data, "empty", True):
                self.df_data.to_excel(xw, index=False, sheet_name="Sieve_Data")
            if getattr(self, "df_laser", None) is not None and not getattr(self.df_laser, "empty", True):
                self.df_laser.to_excel(xw, index=False, sheet_name="Laser_Data")
            if getattr(self, "df_hybrid", None) is not None and not getattr(self.df_hybrid, "empty", True):
                self.df_hybrid.to_excel(xw, index=False, sheet_name="Hybrid_Data")
        return True

    try:
        _write_excel(file)
        try:
            self.last_dir = os.path.dirname(file)
            if getattr(self, "settings", None) is not None:
                self.settings.setValue("last_dir", self.last_dir)
        except Exception:
            pass

        QMessageBox.information(self, tr("info"), "Parameters exported to:\n" + file)
    except Exception as e:
        QMessageBox.warning(self, tr("error"), f"Could not export to Excel:\n{e}")


try:
    setattr(MainWindow, "export_params_to_excel", export_params_to_excel)
except Exception:
    pass

# endregion



# region # === Block 14 REMOVED ===
# All histogram / GMM / Weibull / modes-export code removed intentionally.
# endregion



# region # === Block 15: Helper dialogs and widgets (GroupSelect, ColorDialog, DataBaseWindow) ===
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QCheckBox, QGridLayout, QColorDialog,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
import os
import numpy as np


class GroupSelectDialog(QDialog):
    """Multi-select dialog for groups (checkbox list)."""
    def __init__(self, groups, preselected=None, title="Select groups", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._groups = list(groups)
        self._checks = {}

        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        cont = QWidget()
        v = QVBoxLayout(cont)

        for g in self._groups:
            cb = QCheckBox(g)
            cb.setChecked((preselected is None) or (g in preselected))
            v.addWidget(cb)
            self._checks[g] = cb

        v.addStretch(1)
        scroll.setWidget(cont)
        layout.addWidget(scroll)

        btns = QHBoxLayout()
        btn_ok = QPushButton(tr("ok"))
        btn_ok.clicked.connect(self.accept)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        layout.addLayout(btns)

    def get_selected(self):
        return [g for g, cb in self._checks.items() if cb.isChecked()]


class ColorDialog(QDialog):
    """
    Simple color editor per group label.

    Compatible with two calling styles:
      1) ColorDialog(initial_colors_dict, parent=self)
      2) ColorDialog(groups, current_colors_dict, parent=self)
    """
    def __init__(self, groups_or_colors, current_colors=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("cfg_colors"))

        if current_colors is None and isinstance(groups_or_colors, dict):
            groups = list(groups_or_colors.keys())
            self.selected_colors = dict(groups_or_colors)
        else:
            groups = list(groups_or_colors)
            self.selected_colors = dict(current_colors or {})

        layout = QVBoxLayout(self)
        self._rows = []

        grid = QGridLayout()
        grid.addWidget(QLabel("Group"), 0, 0)
        grid.addWidget(QLabel("Color"), 0, 1)

        for i, g in enumerate(groups, start=1):
            lbl = QLabel(str(g))
            btn = QPushButton(self.selected_colors.get(g, "#888888"))
            btn.clicked.connect(lambda _, key=g, b=btn: self._pick_color(key, b))
            grid.addWidget(lbl, i, 0)
            grid.addWidget(btn, i, 1)
            self._rows.append((g, btn))

        layout.addLayout(grid)

        btns = QHBoxLayout()
        btn_ok = QPushButton(tr("ok"))
        btn_cancel = QPushButton(tr("cancel"))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def _pick_color(self, key, btn):
        col = QColorDialog.getColor()
        if col.isValid():
            self.selected_colors[key] = col.name()
            btn.setText(col.name())


class DataBaseWindow(QDialog):
    """
    Show a DataFrame in a table with an 'Export to Excel' button.
    Display formatting: numeric cells are shown with 2 decimals.
    Excel export: numeric columns are rounded to 2 decimals.
    """
    def __init__(self, title, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 600)
        self._df = df

        layout = QVBoxLayout(self)
        self.table = QTableWidget(self)
        layout.addWidget(self.table)

        btns = QHBoxLayout()
        btn_export = QPushButton("Export to Excel")
        btn_export.clicked.connect(self._export_excel)
        btn_close = QPushButton(tr("ok"))
        btn_close.clicked.connect(self.accept)
        btns.addStretch(1)
        btns.addWidget(btn_export)
        btns.addWidget(btn_close)
        layout.addLayout(btns)

        self._populate()

    @staticmethod
    def _fmt_cell(val) -> str:
        """Return string with 2 decimals if numeric; otherwise return as-is."""
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass

        s = str(val).strip()
        try:
            v = float(s.replace(",", "."))
            return f"{v:.2f}"
        except Exception:
            return s

    def _populate(self):
        df = self._df
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for i in range(len(df)):
            for j, c in enumerate(df.columns):
                val = df.iloc[i, j]
                item = QTableWidgetItem(self._fmt_cell(val))
                try:
                    float(str(val).replace(",", "."))
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                except Exception:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                self.table.setItem(i, j, item)

        self.table.resizeColumnsToContents()

    def _export_excel(self):
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Export to Excel",
            "Data.xlsx",
            "Excel files (*.xlsx)"
        )
        if not file:
            return
        if not file.lower().endswith(".xlsx"):
            file += ".xlsx"

        df_out = self._df.copy()
        try:
            num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df_out[num_cols] = df_out[num_cols].round(2)
        except Exception:
            for col in df_out.columns:
                try:
                    v = pd.to_numeric(df_out[col], errors="coerce")
                    if v.notna().any():
                        df_out[col] = v.round(2)
                except Exception:
                    pass

        try:
            df_out.to_excel(file, index=False)
            QMessageBox.information(self, "OK", "Saved to:\n" + file)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not export:\n{e}")



# endregion


# region # === Block 16: Histogram canvas, style helpers and numeric utilities ===

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QDialog, QFormLayout, QColorDialog, QDoubleSpinBox,
    QDialogButtonBox, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QGroupBox, QAbstractItemView, QScrollArea, QLineEdit, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from matplotlib.figure import Figure
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# =========================================================
# Zoom + pan canvas for histogram
# =========================================================
class ZoomPanFigureCanvas(FigureCanvas):
    """
    Wheel = zoom
    Middle mouse button drag = pan by moving scrollbars

    Important:
    - zoom is VIEW zoom
    - content gets redrawn through on_zoom_callback so text/lines scale too
    """
    def __init__(self, figure):
        super().__init__(figure)
        self._zoom = 1.0
        self._base_w_in = float(figure.get_figwidth())
        self._base_h_in = float(figure.get_figheight())
        self._panning = False
        self._pan_start_global = None
        self._pan_start_h = 0
        self._pan_start_v = 0
        self.on_zoom_callback = None

        self.setStyleSheet("background-color: white;")
        try:
            self.figure.patch.set_facecolor("white")
        except Exception:
            pass

        self._apply_pixel_size()

    def _find_scroll_area(self):
        p = self.parent()
        while p is not None:
            if isinstance(p, QScrollArea):
                return p
            p = p.parent()
        return None

    def set_base_size_inches(self, w_in, h_in):
        self._base_w_in = max(2.0, float(w_in))
        self._base_h_in = max(2.0, float(h_in))
        self._apply_pixel_size()

    def reset_zoom(self):
        self._zoom = 1.0
        self._apply_pixel_size()
        if callable(self.on_zoom_callback):
            self.on_zoom_callback()

    def _apply_pixel_size(self):
        dpi = float(self.figure.get_dpi())
        w_px = max(300, int(round(self._base_w_in * dpi * self._zoom)))
        h_px = max(220, int(round(self._base_h_in * dpi * self._zoom)))

        try:
            self.figure.set_size_inches(w_px / dpi, h_px / dpi, forward=True)
        except Exception:
            pass

        self.setMinimumSize(w_px, h_px)
        self.resize(w_px, h_px)
        self.updateGeometry()
        self.update()

    def zoom_by(self, factor, event_pos=None):
        old_zoom = float(self._zoom)
        new_zoom = float(np.clip(old_zoom * factor, 0.35, 8.0))
        if abs(new_zoom - old_zoom) < 1e-12:
            return

        sa = self._find_scroll_area()
        old_w = max(1, self.width())
        old_h = max(1, self.height())

        old_content_x = None
        old_content_y = None
        old_viewport_x = None
        old_viewport_y = None

        if sa is not None and event_pos is not None:
            old_hbar = sa.horizontalScrollBar().value()
            old_vbar = sa.verticalScrollBar().value()
            old_content_x = int(event_pos.x())
            old_content_y = int(event_pos.y())
            old_viewport_x = old_content_x - old_hbar
            old_viewport_y = old_content_y - old_vbar

        self._zoom = new_zoom
        self._apply_pixel_size()

        if callable(self.on_zoom_callback):
            self.on_zoom_callback()

        if sa is not None and event_pos is not None:
            new_w = max(1, self.width())
            new_h = max(1, self.height())

            rel_x = old_content_x / float(old_w)
            rel_y = old_content_y / float(old_h)

            new_content_x = int(round(rel_x * new_w))
            new_content_y = int(round(rel_y * new_h))

            sa.horizontalScrollBar().setValue(max(0, new_content_x - old_viewport_x))
            sa.verticalScrollBar().setValue(max(0, new_content_y - old_viewport_y))

    def wheelEvent(self, event):
        try:
            delta = event.angleDelta().y()
        except Exception:
            delta = 0

        if delta == 0:
            super().wheelEvent(event)
            return

        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        self.zoom_by(factor, event.pos())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            sa = self._find_scroll_area()
            if sa is not None:
                self._panning = True
                self._pan_start_global = event.globalPos()
                self._pan_start_h = sa.horizontalScrollBar().value()
                self._pan_start_v = sa.verticalScrollBar().value()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            sa = self._find_scroll_area()
            if sa is not None and self._pan_start_global is not None:
                dx = event.globalX() - self._pan_start_global.x()
                dy = event.globalY() - self._pan_start_global.y()
                sa.horizontalScrollBar().setValue(self._pan_start_h - dx)
                sa.verticalScrollBar().setValue(self._pan_start_v - dy)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start_global = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


# =========================================================
# Global matplotlib font for histogram
# =========================================================
def _hist_apply_font_once(self=None):
    """
    Try to force STIX Two Math. If it is not installed, matplotlib will fallback silently.
    """
    wanted = globals().get("HIST_FONT_FAMILY", "STIX Two Math")
    try:
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["font.family"] = wanted
    except Exception:
        pass


# =========================================================
# Style coercion / persistence
# =========================================================
def _hist_to_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return bool(default)


def _hist_to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _hist_to_str(v, default=""):
    try:
        s = str(v)
        return s if s else str(default)
    except Exception:
        return str(default)


def _hist_ensure_style(self):
    local_defaults = {
        "bar_facecolor": "#8c8c8c",
        "bar_edgecolor": "#000000",
        "bar_width_factor": 0.92,
        "bar_edgewidth": 1.2,
        "axes_color": "#000000",
        "axes_linewidth": 2.0,
        "cum_color": "#000000",
        "cum_linewidth": 1.8,
        "poly_color": "#000000",
        "poly_linewidth": 1.4,
        "laser_color": "#000000",
        "laser_linewidth": 1.1,
        "overlay_color": "#000000",
        "overlay_linewidth": 1.0,
        "overlay_alpha": 0.20,
        "overlay_hatch": "////",
        "figure_width": 11.0,
        "figure_height": 7.0,
        "show_cumulative": True,
        "show_polygon": False,
        "show_um_axis": False,
        "reverse_x": False,
        "show_params_text": True,
        "show_modified_overlay": False,
        "show_laser_curve": False,
        "show_mean_marker": False,
        "show_median_marker": False,
        "show_sigma_marker": False,
        "show_mode_marker": False,
        "stats_marker_color": "#808080",
        "stats_marker_alpha": 0.18,
        "stats_line_style": "--",
        "stats_linewidth": 1.0,
        "x_label_phi": "",
        "x_label_um": "",
        "y_label_left": "",
        "y_label_right": "",
        "um_secondary_label": "",
        "cumulative_layer": "front",
        "um_axis_position": "top",
        "x_axis_mode": "phi",
        "y_left_mode": "wt",
        "y_right_mode": "cumulative",
    }

    defaults = dict(local_defaults)
    try:
        defaults.update(dict(globals().get("HIST_DEFAULT_STYLE", {})))
    except Exception:
        pass

    if not hasattr(self, "hist_style") or not isinstance(self.hist_style, dict):
        self.hist_style = dict(defaults)
    else:
        for k, v in defaults.items():
            self.hist_style.setdefault(k, v)

    try:
        if getattr(self, "settings", None) is not None:
            saved = self.settings.value("hist_style", None)
            if isinstance(saved, dict):
                for k, v in saved.items():
                    self.hist_style[k] = v
    except Exception:
        pass

    # normalize strings
    for key in [
        "bar_facecolor", "bar_edgecolor", "axes_color", "cum_color",
        "poly_color", "laser_color", "overlay_color", "overlay_hatch",
        "stats_marker_color", "stats_line_style",
        "x_label_phi", "x_label_um", "y_label_left", "y_label_right",
        "um_secondary_label", "x_axis_mode", "y_left_mode",
        "y_right_mode", "cumulative_layer", "um_axis_position"
    ]:
        self.hist_style[key] = _hist_to_str(self.hist_style.get(key), defaults.get(key, ""))

    # normalize floats
    for key in [
        "bar_width_factor", "bar_edgewidth", "axes_linewidth",
        "cum_linewidth", "poly_linewidth", "laser_linewidth",
        "overlay_linewidth", "overlay_alpha",
        "figure_width", "figure_height",
        "stats_marker_alpha", "stats_linewidth"
    ]:
        self.hist_style[key] = _hist_to_float(self.hist_style.get(key), defaults.get(key, 0.0))

    # normalize bools
    for key in [
        "show_cumulative", "show_polygon", "show_um_axis",
        "reverse_x", "show_params_text", "show_modified_overlay",
        "show_laser_curve", "show_mean_marker", "show_median_marker",
        "show_sigma_marker", "show_mode_marker"
    ]:
        self.hist_style[key] = _hist_to_bool(self.hist_style.get(key), defaults.get(key, False))

    # sanitize enums
    if self.hist_style["x_axis_mode"] not in {"phi", "um"}:
        self.hist_style["x_axis_mode"] = defaults["x_axis_mode"]
    if self.hist_style["y_left_mode"] not in {"wt"}:
        self.hist_style["y_left_mode"] = defaults["y_left_mode"]
    if self.hist_style["y_right_mode"] not in {"cumulative", "none"}:
        self.hist_style["y_right_mode"] = defaults["y_right_mode"]
    if self.hist_style["cumulative_layer"] not in {"front", "back"}:
        self.hist_style["cumulative_layer"] = defaults["cumulative_layer"]
    if self.hist_style["um_axis_position"] not in {"top", "bottom"}:
        self.hist_style["um_axis_position"] = defaults["um_axis_position"]


def _hist_save_style(self):
    try:
        if getattr(self, "settings", None) is not None:
            self.settings.setValue("hist_style", dict(self.hist_style))
    except Exception:
        pass


# =========================================================
# Numeric / conversion helpers
# =========================================================
def _hist_fmt_num(v, nd=2):
    try:
        v = float(v)
        if not np.isfinite(v):
            return ""
        s = f"{v:.{nd}f}"
        s = s.rstrip("0").rstrip(".")
        return s
    except Exception:
        return ""


def _phi_to_um(phi):
    try:
        return (2.0 ** (-float(phi))) * 1000.0
    except Exception:
        return np.nan


def _um_label_from_phi(phi):
    um = _phi_to_um(phi)
    if not np.isfinite(um):
        return ""
    if um >= 100:
        return f"{um:.0f}"
    if um >= 10:
        return f"{um:.1f}".rstrip("0").rstrip(".")
    return f"{um:.2f}".rstrip("0").rstrip(".")


def _hist_step_from_phi(phi_vals):
    vals = np.array(sorted(pd.unique(pd.Series(phi_vals).dropna().astype(float))), dtype=float)
    if vals.size < 2:
        return 1.0
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    step = float(np.median(diffs))
    return step if step > 0 else 1.0


def _hist_make_nice_ticks(vmin, vmax, step):
    """Ticks constrained to the actual phi range of the current sample.

    The previous implementation deliberately expanded the axis by one bin on
    each side (floor(vmin-step), ceil(vmax+step)).  That made every histogram
    show artificial empty tails.  Here the first and last ticks are always the
    real sample extrema; intermediate ticks follow the sample bin spacing and
    are thinned only when necessary to keep labels readable.
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([], dtype=float)

    vmin = float(vmin)
    vmax = float(vmax)
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if np.isclose(vmin, vmax):
        return np.array([vmin], dtype=float)

    step = float(step) if np.isfinite(step) and step > 0 else (vmax - vmin)
    n = int(np.floor((vmax - vmin) / step + 1e-9)) + 1
    ticks = vmin + np.arange(max(1, n), dtype=float) * step
    ticks = ticks[ticks <= vmax + max(1e-9, abs(step) * 1e-7)]

    if ticks.size == 0 or not np.isclose(ticks[0], vmin):
        ticks = np.insert(ticks, 0, vmin)
    if not np.isclose(ticks[-1], vmax):
        ticks = np.append(ticks, vmax)

    # Avoid an unreadable forest of labels for very fine laser spacing while
    # preserving the exact first/last sample limits.
    max_ticks = 21
    if ticks.size > max_ticks:
        idx = np.linspace(0, ticks.size - 1, max_ticks).round().astype(int)
        ticks = ticks[np.unique(idx)]
        ticks[0] = vmin
        ticks[-1] = vmax

    return np.round(ticks, 10)


def _hist_um_bar_geometry(phi_vals, width_factor):
    """
    Return x centers and widths for bars in μm space using phi bin width.
    """
    phi_vals = np.asarray(phi_vals, dtype=float)
    step = _hist_step_from_phi(phi_vals)
    centers = np.array([_phi_to_um(v) for v in phi_vals], dtype=float)

    widths = []
    for p in phi_vals:
        left_edge_um = _phi_to_um(p - step / 2.0)
        right_edge_um = _phi_to_um(p + step / 2.0)
        widths.append(abs(left_edge_um - right_edge_um) * float(width_factor))
    widths = np.asarray(widths, dtype=float)
    return centers, widths


def _hist_pick_sparse_positions(values, max_ticks=10):
    vals = np.asarray(values, dtype=float)
    if vals.size <= max_ticks:
        return vals
    idx = np.linspace(0, vals.size - 1, max_ticks).round().astype(int)
    idx = np.unique(idx)
    return vals[idx]


def _hist_style_axes(ax, ax2, style):
    c = style["axes_color"]
    lw = float(style["axes_linewidth"])

    try:
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_color(c)
            ax.spines[sp].set_linewidth(lw)

        ax.spines["top"].set_visible(False)

        if ax2.yaxis.get_visible():
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color(c)
            ax2.spines["right"].set_linewidth(lw)
        else:
            ax2.spines["right"].set_visible(False)

        ax2.spines["top"].set_visible(False)

        ax.tick_params(axis="both", which="both", width=max(1.0, lw * 0.9), length=7, labelsize=11, colors=c)
        ax2.tick_params(axis="y", which="both", width=max(1.0, lw * 0.9), length=7, labelsize=11, colors=c)
        ax.xaxis.label.set_color(c)
        ax.yaxis.label.set_color(c)
        ax2.yaxis.label.set_color(c)
        ax.title.set_color(c)
    except Exception:
        pass


def _hist_default_axis_labels():
    return {
        "x_phi": "PHI (Φ)",
        "x_um": "Particle size (μm)",
        "y_left": "%Wt",
        "y_right": "%Wt cumulative frequency",
        "um_secondary": "Particle size (μm)",
    }


def _hist_plot_x_from_phi_value(phi_value, x_mode):
    return _phi_to_um(phi_value) if str(x_mode).lower() == "um" else float(phi_value)


def _hist_modes_from_distribution(phi_vals, wt_vals, tol=1e-12, min_mode_fraction=0.10):
    """
    Return phi positions of all LOCAL modes.

    Rules:
    - A local mode is a bin whose wt is greater than the previous and next bin.
    - Edge bins can also be modes if they are greater than their only neighbor.
    - If several consecutive bins form a flat peak (plateau), the center phi is returned.
    - min_mode_fraction filters very small local peaks relative to the global maximum.
    """
    phi_vals = np.asarray(phi_vals, dtype=float)
    wt_vals = np.asarray(wt_vals, dtype=float)

    mask = np.isfinite(phi_vals) & np.isfinite(wt_vals)
    phi_vals = phi_vals[mask]
    wt_vals = wt_vals[mask]

    if phi_vals.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(phi_vals)
    phi_vals = phi_vals[order]
    wt_vals = wt_vals[order]

    n = wt_vals.size
    if n == 1:
        return phi_vals.copy()

    wmax = np.nanmax(wt_vals)
    if not np.isfinite(wmax):
        return np.array([], dtype=float)

    threshold = float(min_mode_fraction) * float(wmax)
    mode_positions = []

    i = 0
    while i < n:
        j = i
        while j + 1 < n and np.isclose(wt_vals[j + 1], wt_vals[i], atol=tol, rtol=0.0):
            j += 1

        left_val = wt_vals[i - 1] if i > 0 else -np.inf
        right_val = wt_vals[j + 1] if j + 1 < n else -np.inf
        plateau_val = wt_vals[i]

        is_local_mode = (plateau_val > left_val + tol) and (plateau_val > right_val + tol)
        is_high_enough = plateau_val >= threshold

        if is_local_mode and is_high_enough:
            center_idx = (i + j) // 2
            mode_positions.append(phi_vals[center_idx])

        i = j + 1

    return np.asarray(mode_positions, dtype=float)

# endregion


# region # === Block 17: Histogram data extraction, dialogs, callbacks and exports (hybrid-safe) ===

# =========================================================
# Data extraction
# =========================================================
def _hist_get_param_row(self, db_key, sample_name):
    """
    Return the parameter row associated with the selected sample.

    Hybrid-safe behavior:
    - for hybrid, search first by "Hybrid Sample"
    - then by "Sample"
    - and only as old fallback by "Tamiz Sample"
    """
    try:
        if db_key == "tamiz":
            dfp = getattr(self, "param_results", None)

        elif db_key == "laser":
            dfp = getattr(self, "param_results_laser", None)

        elif db_key == "hybrid":
            dfp = getattr(self, "param_results_hybrid", None)

        else:
            dfp = None

        if dfp is None or getattr(dfp, "empty", True):
            return None

        sample_name = str(sample_name)

        if db_key != "hybrid":
            if "Sample" not in dfp.columns:
                return None
            sub = dfp[dfp["Sample"].astype(str) == sample_name]
            if sub.empty:
                return None
            return sub.iloc[0]

        # ---- hybrid ----
        candidate_cols = []
        for c in ("Hybrid Sample", "Sample", "Tamiz Sample"):
            if c in dfp.columns:
                candidate_cols.append(c)

        for c in candidate_cols:
            sub = dfp[dfp[c].astype(str) == sample_name]
            if not sub.empty:
                return sub.iloc[0]

    except Exception:
        pass

    return None


def _hist_get_available_datasets(self):
    out = []
    if getattr(self, "df_data", None) is not None and not getattr(self.df_data, "empty", True):
        out.append(("tamiz", base_name_display("Tamiz")))
    if getattr(self, "df_laser", None) is not None and not getattr(self.df_laser, "empty", True):
        out.append(("laser", base_name_display("Laser")))
    if getattr(self, "df_hybrid", None) is not None and not getattr(self.df_hybrid, "empty", True):
        out.append(("hybrid", base_name_display("Híbrido")))
    return out


def _hist_get_sample_names(self, db_key):
    """
    Return visible sample names for histogram selectors.

    Hybrid-safe behavior:
    - prefer unique "Hybrid Sample"
    - then "Sample"
    - only fallback to "Tamiz Sample" for old data/state
    """
    try:
        if db_key == "tamiz":
            samp_col = self.df_data.columns[1]
            return sorted(self.df_data[samp_col].dropna().astype(str).unique().tolist())

        if db_key == "laser":
            return sorted(self.df_laser["Sample"].dropna().astype(str).unique().tolist())

        if db_key == "hybrid":
            if "Hybrid Sample" in self.df_hybrid.columns:
                return sorted(self.df_hybrid["Hybrid Sample"].dropna().astype(str).unique().tolist())

            if "Sample" in self.df_hybrid.columns:
                return sorted(self.df_hybrid["Sample"].dropna().astype(str).unique().tolist())

            if "Tamiz Sample" in self.df_hybrid.columns:
                return sorted(self.df_hybrid["Tamiz Sample"].dropna().astype(str).unique().tolist())

    except Exception:
        pass

    return []


def _hist_get_distribution(self, db_key, sample_name):
    """
    Return:
      df_main with columns ["phi","wt"]
      meta dict with labels + optional laser overlay

    Hybrid-safe behavior:
    - filters hybrid first by "Hybrid Sample", then "Sample", then old fallback
    - overlay metadata is retrieved with unique hybrid identity
    """
    sample_name = str(sample_name)

    if db_key == "tamiz":
        phi_col = self.df_data.columns[0]
        samp_col = self.df_data.columns[1]
        wt_col = self.df_data.columns[2]

        sub = self.df_data[self.df_data[samp_col].astype(str) == sample_name].copy()
        if sub.empty:
            return pd.DataFrame(columns=["phi", "wt"]), {}

        df_main = sub[[phi_col, wt_col]].rename(columns={phi_col: "phi", wt_col: "wt"}).copy()
        df_main["phi"] = pd.to_numeric(df_main["phi"], errors="coerce")
        df_main["wt"] = pd.to_numeric(df_main["wt"], errors="coerce")
        df_main = df_main[np.isfinite(df_main["phi"]) & np.isfinite(df_main["wt"])]
        df_main = df_main.groupby("phi", as_index=False)["wt"].sum().sort_values("phi")

        meta = {
            "db_label": base_name_display("Tamiz"),
            "sample_title": sample_name,
            "laser_overlay": None,
        }
        return df_main, meta

    if db_key == "laser":
        sub = self.df_laser[self.df_laser["Sample"].astype(str) == sample_name].copy()
        if sub.empty:
            return pd.DataFrame(columns=["phi", "wt"]), {}

        df_main = sub[["phi", "Wt"]].rename(columns={"Wt": "wt"}).copy()
        df_main["phi"] = pd.to_numeric(df_main["phi"], errors="coerce")
        df_main["wt"] = pd.to_numeric(df_main["wt"], errors="coerce")
        df_main = df_main[np.isfinite(df_main["phi"]) & np.isfinite(df_main["wt"])]
        df_main = df_main.groupby("phi", as_index=False)["wt"].sum().sort_values("phi")

        meta = {
            "db_label": base_name_display("Laser"),
            "sample_title": sample_name,
            "laser_overlay": None,
        }
        return df_main, meta

    if db_key == "hybrid":
        dfh = self.df_hybrid.copy()

        # unique hybrid identity priority
        if "Hybrid Sample" in dfh.columns:
            sub = dfh[dfh["Hybrid Sample"].astype(str) == sample_name].copy()
        elif "Sample" in dfh.columns:
            sub = dfh[dfh["Sample"].astype(str) == sample_name].copy()
        elif "Tamiz Sample" in dfh.columns:
            sub = dfh[dfh["Tamiz Sample"].astype(str) == sample_name].copy()
        else:
            sub = pd.DataFrame()

        if sub.empty:
            return pd.DataFrame(columns=["phi", "wt"]), {}

        wt_col = "wt%" if "wt%" in sub.columns else ("Wt" if "Wt" in sub.columns else None)
        if wt_col is None or "phi" not in sub.columns:
            return pd.DataFrame(columns=["phi", "wt"]), {}

        df_main = sub[["phi", wt_col]].rename(columns={wt_col: "wt"}).copy()
        df_main["phi"] = pd.to_numeric(df_main["phi"], errors="coerce")
        df_main["wt"] = pd.to_numeric(df_main["wt"], errors="coerce")
        df_main = df_main[np.isfinite(df_main["phi"]) & np.isfinite(df_main["wt"])]
        df_main = df_main.groupby("phi", as_index=False)["wt"].sum().sort_values("phi")

        laser_overlay = None
        try:
            hybrid_name = None
            if "Hybrid Sample" in sub.columns:
                hybrid_name = str(sub["Hybrid Sample"].iloc[0])
            elif "Sample" in sub.columns:
                hybrid_name = str(sub["Sample"].iloc[0])
            else:
                hybrid_name = sample_name

            laser_sample = None
            if "Laser Sample" in sub.columns:
                laser_sample = str(sub["Laser Sample"].iloc[0])

            if laser_sample and getattr(self, "df_laser", None) is not None and not self.df_laser.empty:
                lsub = self.df_laser[self.df_laser["Sample"].astype(str) == laser_sample].copy()
                if not lsub.empty:
                    ldf = lsub[["phi", "Wt"]].rename(columns={"Wt": "wt"}).copy()
                    ldf["phi"] = pd.to_numeric(ldf["phi"], errors="coerce")
                    ldf["wt"] = pd.to_numeric(ldf["wt"], errors="coerce")
                    ldf = ldf[np.isfinite(ldf["phi"]) & np.isfinite(ldf["wt"])]
                    ldf = ldf.groupby("phi", as_index=False)["wt"].sum().sort_values("phi")

                    meta_h = getattr(self, "_hybrid_meta", {}).get(hybrid_name, {})

                    phi_cut = meta_h.get("phi*", meta_h.get("phi_cut", None))
                    if phi_cut is None and "phi*" in sub.columns:
                        try:
                            phi_cut = float(sub["phi*"].iloc[0])
                        except Exception:
                            phi_cut = None

                    if phi_cut is not None:
                        ldf = ldf[ldf["phi"] >= float(phi_cut)].copy()

                    method = str(meta_h.get("method", "")).lower().strip()
                    if not method and "Hybrid Method Code" in sub.columns:
                        method = str(sub["Hybrid Method Code"].iloc[0]).lower().strip()

                    scale = np.nan

                    if method in ("m1", "m3"):
                        scale = float(meta_h.get("scale_k", np.nan))

                    elif method == "m2":
                        k = float(meta_h.get("k", np.nan))
                        alpha = float(meta_h.get("alpha", np.nan))
                        scale = k * alpha

                    if not np.isfinite(scale):
                        if phi_cut is not None:
                            hfine = float(df_main.loc[df_main["phi"] >= float(phi_cut), "wt"].sum())
                            lfine = float(ldf["wt"].sum())
                            scale = (hfine / lfine) if lfine > 0 else 1.0
                        else:
                            scale = 1.0

                    ldf["wt"] = ldf["wt"] * float(scale)

                    if not ldf.empty:
                        laser_overlay = {
                            "laser_sample": laser_sample,
                            "df": ldf.copy(),
                        }

        except Exception:
            laser_overlay = None

        meta = {
            "db_label": base_name_display("Híbrido"),
            "sample_title": sample_name,
            "laser_overlay": laser_overlay,
        }
        return df_main, meta

    return pd.DataFrame(columns=["phi", "wt"]), {}


def _hist_get_modified_overlay_df(self, hybrid_sample_name, df_hybrid_main):
    """
    For HYBRID samples only:
    return original SIEVE bins where the final hybrid differs from original sieve.
    The returned wt is the ORIGINAL sieve wt so the user can visualize what changed.

    Hybrid-safe behavior:
    - find the original sieve sample through the selected hybrid record
    - do NOT assume that hybrid_sample_name == tamiz sample name
    """
    try:
        if getattr(self, "df_data", None) is None or self.df_data.empty:
            return None
        if getattr(self, "df_hybrid", None) is None or self.df_hybrid.empty:
            return None

        hybrid_sample_name = str(hybrid_sample_name)
        dfh = self.df_hybrid.copy()

        if "Hybrid Sample" in dfh.columns:
            hsub = dfh[dfh["Hybrid Sample"].astype(str) == hybrid_sample_name].copy()
        elif "Sample" in dfh.columns:
            hsub = dfh[dfh["Sample"].astype(str) == hybrid_sample_name].copy()
        elif "Tamiz Sample" in dfh.columns:
            hsub = dfh[dfh["Tamiz Sample"].astype(str) == hybrid_sample_name].copy()
        else:
            hsub = pd.DataFrame()

        if hsub.empty:
            return None

        if "Tamiz Sample" in hsub.columns:
            tamiz_sample_name = str(hsub["Tamiz Sample"].iloc[0])
        else:
            # old fallback only
            tamiz_sample_name = hybrid_sample_name

        phi_col = self.df_data.columns[0]
        samp_col = self.df_data.columns[1]
        wt_col = self.df_data.columns[2]

        sub = self.df_data[self.df_data[samp_col].astype(str) == tamiz_sample_name].copy()
        if sub.empty:
            return None

        df_orig = sub[[phi_col, wt_col]].rename(columns={phi_col: "phi", wt_col: "wt"}).copy()
        df_orig["phi"] = pd.to_numeric(df_orig["phi"], errors="coerce")
        df_orig["wt"] = pd.to_numeric(df_orig["wt"], errors="coerce")
        df_orig = df_orig[np.isfinite(df_orig["phi"]) & np.isfinite(df_orig["wt"])]
        df_orig = df_orig.groupby("phi", as_index=False)["wt"].sum()

        df_fin = df_hybrid_main.copy()
        df_fin["phi"] = pd.to_numeric(df_fin["phi"], errors="coerce")
        df_fin["wt"] = pd.to_numeric(df_fin["wt"], errors="coerce")
        df_fin = df_fin.groupby("phi", as_index=False)["wt"].sum()

        merged = df_orig.merge(df_fin, on="phi", how="outer", suffixes=("_orig", "_final")).fillna(0.0)
        merged["delta"] = (merged["wt_orig"] - merged["wt_final"]).abs()

        out = merged[merged["delta"] > 1e-9][["phi", "wt_orig"]].copy()
        if out.empty:
            return None

        out = out.rename(columns={"wt_orig": "wt"}).sort_values("phi").reset_index(drop=True)
        return out

    except Exception:
        return None


# =========================================================
# Color-swatch helpers
# =========================================================
def _hist_color_button_style(color_hex):
    return (
        "QPushButton {"
        f"background-color: {color_hex};"
        "border: 1px solid #444;"
        "border-radius: 4px;"
        "min-height: 24px;"
        "min-width: 80px;"
        "}"
    )


def _hist_set_color_button(btn, color_hex):
    btn.setText("")
    btn.setToolTip(str(color_hex))
    btn.setStyleSheet(_hist_color_button_style(color_hex))


# =========================================================
# Histogram dialogs
# =========================================================
class HistogramStyleDialog(QDialog):
    def __init__(self, style_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("hist_style_title"))
        self.resize(700, 520)
        self.style = dict(style_dict)

        defaults = _hist_default_axis_labels()

        outer = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        # ---------- Column 1: bars / figure ----------
        gb_bars = QGroupBox("Bars and figure")
        form_bars = QFormLayout(gb_bars)

        self.btn_bar_fill = QPushButton()
        _hist_set_color_button(self.btn_bar_fill, self.style["bar_facecolor"])
        self.btn_bar_fill.clicked.connect(lambda: self._pick("bar_facecolor", self.btn_bar_fill))
        form_bars.addRow(tr("hist_bar_fill"), self.btn_bar_fill)

        self.btn_bar_edge = QPushButton()
        _hist_set_color_button(self.btn_bar_edge, self.style["bar_edgecolor"])
        self.btn_bar_edge.clicked.connect(lambda: self._pick("bar_edgecolor", self.btn_bar_edge))
        form_bars.addRow(tr("hist_bar_edge"), self.btn_bar_edge)

        self.sp_bar_width = QDoubleSpinBox()
        self.sp_bar_width.setRange(0.05, 1.50)
        self.sp_bar_width.setDecimals(2)
        self.sp_bar_width.setSingleStep(0.05)
        self.sp_bar_width.setValue(float(self.style["bar_width_factor"]))
        form_bars.addRow(tr("hist_bar_width"), self.sp_bar_width)

        self.sp_bar_edge_width = QDoubleSpinBox()
        self.sp_bar_edge_width.setRange(0.0, 6.0)
        self.sp_bar_edge_width.setDecimals(2)
        self.sp_bar_edge_width.setSingleStep(0.1)
        self.sp_bar_edge_width.setValue(float(self.style["bar_edgewidth"]))
        form_bars.addRow(tr("hist_bar_edge_width"), self.sp_bar_edge_width)

        self.sp_fig_w = QDoubleSpinBox()
        self.sp_fig_w.setRange(4.0, 40.0)
        self.sp_fig_w.setDecimals(2)
        self.sp_fig_w.setSingleStep(0.25)
        self.sp_fig_w.setValue(float(self.style["figure_width"]))
        form_bars.addRow(tr("hist_fig_width"), self.sp_fig_w)

        self.sp_fig_h = QDoubleSpinBox()
        self.sp_fig_h.setRange(3.0, 40.0)
        self.sp_fig_h.setDecimals(2)
        self.sp_fig_h.setSingleStep(0.25)
        self.sp_fig_h.setValue(float(self.style["figure_height"]))
        form_bars.addRow(tr("hist_fig_height"), self.sp_fig_h)

        # ---------- Column 2: axes / curves ----------
        gb_axes = QGroupBox("Axes and curves")
        form_axes = QFormLayout(gb_axes)

        self.btn_axes = QPushButton()
        _hist_set_color_button(self.btn_axes, self.style["axes_color"])
        self.btn_axes.clicked.connect(lambda: self._pick("axes_color", self.btn_axes))
        form_axes.addRow(tr("hist_axes_color"), self.btn_axes)

        self.sp_axes_width = QDoubleSpinBox()
        self.sp_axes_width.setRange(0.5, 8.0)
        self.sp_axes_width.setDecimals(2)
        self.sp_axes_width.setSingleStep(0.1)
        self.sp_axes_width.setValue(float(self.style["axes_linewidth"]))
        form_axes.addRow(tr("hist_axes_width"), self.sp_axes_width)

        self.btn_cum = QPushButton()
        _hist_set_color_button(self.btn_cum, self.style["cum_color"])
        self.btn_cum.clicked.connect(lambda: self._pick("cum_color", self.btn_cum))
        form_axes.addRow(tr("hist_cum_color"), self.btn_cum)

        self.sp_cum_width = QDoubleSpinBox()
        self.sp_cum_width.setRange(0.1, 8.0)
        self.sp_cum_width.setDecimals(2)
        self.sp_cum_width.setSingleStep(0.1)
        self.sp_cum_width.setValue(float(self.style["cum_linewidth"]))
        form_axes.addRow(tr("hist_cum_width"), self.sp_cum_width)

        self.btn_poly = QPushButton()
        _hist_set_color_button(self.btn_poly, self.style["poly_color"])
        self.btn_poly.clicked.connect(lambda: self._pick("poly_color", self.btn_poly))
        form_axes.addRow(tr("hist_poly_color"), self.btn_poly)

        self.sp_poly_width = QDoubleSpinBox()
        self.sp_poly_width.setRange(0.1, 8.0)
        self.sp_poly_width.setDecimals(2)
        self.sp_poly_width.setSingleStep(0.1)
        self.sp_poly_width.setValue(float(self.style["poly_linewidth"]))
        form_axes.addRow(tr("hist_poly_width"), self.sp_poly_width)

        self.btn_laser = QPushButton()
        _hist_set_color_button(self.btn_laser, self.style["laser_color"])
        self.btn_laser.clicked.connect(lambda: self._pick("laser_color", self.btn_laser))
        form_axes.addRow(tr("hist_laser_color"), self.btn_laser)

        self.sp_laser_width = QDoubleSpinBox()
        self.sp_laser_width.setRange(0.1, 8.0)
        self.sp_laser_width.setDecimals(2)
        self.sp_laser_width.setSingleStep(0.1)
        self.sp_laser_width.setValue(float(self.style["laser_linewidth"]))
        form_axes.addRow(tr("hist_laser_width"), self.sp_laser_width)

        self.cmb_x_mode = QComboBox()
        self.cmb_x_mode.addItem(tr("hist_x_axis_phi"), "phi")
        self.cmb_x_mode.addItem(tr("hist_x_axis_um"), "um")
        idx = self.cmb_x_mode.findData(self.style.get("x_axis_mode", "phi"))
        if idx >= 0:
            self.cmb_x_mode.setCurrentIndex(idx)
        form_axes.addRow(tr("hist_x_axis_mode"), self.cmb_x_mode)

        self.cmb_y_right = QComboBox()
        self.cmb_y_right.addItem(tr("hist_y_right_cumulative"), "cumulative")
        self.cmb_y_right.addItem(tr("hist_y_right_none"), "none")
        idx = self.cmb_y_right.findData(self.style.get("y_right_mode", "cumulative"))
        if idx >= 0:
            self.cmb_y_right.setCurrentIndex(idx)
        form_axes.addRow(tr("hist_y_right_mode"), self.cmb_y_right)

        self.cmb_cum_layer = QComboBox()
        self.cmb_cum_layer.addItem(tr("hist_cum_front"), "front")
        self.cmb_cum_layer.addItem(tr("hist_cum_back"), "back")
        idx = self.cmb_cum_layer.findData(self.style.get("cumulative_layer", "front"))
        if idx >= 0:
            self.cmb_cum_layer.setCurrentIndex(idx)
        form_axes.addRow(tr("hist_cum_layer"), self.cmb_cum_layer)

        self.cmb_um_axis_position = QComboBox()
        self.cmb_um_axis_position.addItem(tr("hist_um_axis_top"), "top")
        self.cmb_um_axis_position.addItem(tr("hist_um_axis_bottom"), "bottom")
        idx = self.cmb_um_axis_position.findData(self.style.get("um_axis_position", "top"))
        if idx >= 0:
            self.cmb_um_axis_position.setCurrentIndex(idx)
        form_axes.addRow(tr("hist_um_axis_position"), self.cmb_um_axis_position)

        # ---------- Column 3: overlay / labels / stats ----------
        gb_overlay = QGroupBox("Overlay, labels and markers")
        form_overlay = QFormLayout(gb_overlay)

        self.btn_overlay = QPushButton()
        _hist_set_color_button(self.btn_overlay, self.style["overlay_color"])
        self.btn_overlay.clicked.connect(lambda: self._pick("overlay_color", self.btn_overlay))
        form_overlay.addRow(tr("hist_overlay_color"), self.btn_overlay)

        self.sp_overlay_width = QDoubleSpinBox()
        self.sp_overlay_width.setRange(0.1, 8.0)
        self.sp_overlay_width.setDecimals(2)
        self.sp_overlay_width.setSingleStep(0.1)
        self.sp_overlay_width.setValue(float(self.style["overlay_linewidth"]))
        form_overlay.addRow(tr("hist_overlay_width"), self.sp_overlay_width)

        self.sp_overlay_alpha = QDoubleSpinBox()
        self.sp_overlay_alpha.setRange(0.0, 1.0)
        self.sp_overlay_alpha.setDecimals(2)
        self.sp_overlay_alpha.setSingleStep(0.05)
        self.sp_overlay_alpha.setValue(float(self.style["overlay_alpha"]))
        form_overlay.addRow(tr("hist_overlay_alpha"), self.sp_overlay_alpha)

        self.cmb_overlay_hatch = QComboBox()
        hatch_opts = ["////", "\\\\\\\\", "xx", "++", "..", "--", "||"]
        self.cmb_overlay_hatch.addItems(hatch_opts)
        if self.style["overlay_hatch"] in hatch_opts:
            self.cmb_overlay_hatch.setCurrentText(self.style["overlay_hatch"])
        form_overlay.addRow(tr("hist_overlay_hatch"), self.cmb_overlay_hatch)

        self.btn_stats = QPushButton()
        _hist_set_color_button(self.btn_stats, self.style["stats_marker_color"])
        self.btn_stats.clicked.connect(lambda: self._pick("stats_marker_color", self.btn_stats))
        form_overlay.addRow("Stat markers color", self.btn_stats)

        self.sp_stats_alpha = QDoubleSpinBox()
        self.sp_stats_alpha.setRange(0.00, 1.00)
        self.sp_stats_alpha.setDecimals(2)
        self.sp_stats_alpha.setSingleStep(0.05)
        self.sp_stats_alpha.setValue(float(self.style["stats_marker_alpha"]))
        form_overlay.addRow("Stat markers transparency", self.sp_stats_alpha)

        self.sp_stats_linewidth = QDoubleSpinBox()
        self.sp_stats_linewidth.setRange(0.1, 6.0)
        self.sp_stats_linewidth.setDecimals(2)
        self.sp_stats_linewidth.setSingleStep(0.1)
        self.sp_stats_linewidth.setValue(float(self.style["stats_linewidth"]))
        form_overlay.addRow("Stat markers line width", self.sp_stats_linewidth)

        self.ed_x_phi = QLineEdit(_hist_to_str(self.style.get("x_label_phi", ""), ""))
        self.ed_x_phi.setPlaceholderText(defaults["x_phi"])
        form_overlay.addRow(tr("hist_x_label_phi_custom"), self.ed_x_phi)

        self.ed_x_um = QLineEdit(_hist_to_str(self.style.get("x_label_um", ""), ""))
        self.ed_x_um.setPlaceholderText(defaults["x_um"])
        form_overlay.addRow(tr("hist_x_label_um_custom"), self.ed_x_um)

        self.ed_y_left = QLineEdit(_hist_to_str(self.style.get("y_label_left", ""), ""))
        self.ed_y_left.setPlaceholderText(defaults["y_left"])
        form_overlay.addRow(tr("hist_y_label_left_custom"), self.ed_y_left)

        self.ed_y_right = QLineEdit(_hist_to_str(self.style.get("y_label_right", ""), ""))
        self.ed_y_right.setPlaceholderText(defaults["y_right"])
        form_overlay.addRow(tr("hist_y_label_right_custom"), self.ed_y_right)

        self.ed_um_secondary = QLineEdit(_hist_to_str(self.style.get("um_secondary_label", ""), ""))
        self.ed_um_secondary.setPlaceholderText(defaults["um_secondary"])
        form_overlay.addRow(tr("hist_um_secondary_label_custom"), self.ed_um_secondary)

        grid.addWidget(gb_bars, 0, 0)
        grid.addWidget(gb_axes, 0, 1)
        grid.addWidget(gb_overlay, 0, 2)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        btns_row = QHBoxLayout()
        self.btn_defaults = QPushButton(tr("hist_reset_defaults"))
        self.btn_defaults.clicked.connect(self._reset_defaults)
        btns_row.addWidget(self.btn_defaults)
        btns_row.addStretch(1)
        outer.addLayout(btns_row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    def _pick(self, key, btn):
        c = QColorDialog.getColor(QColor(self.style.get(key, "#000000")), self)
        if c.isValid():
            self.style[key] = c.name()
            _hist_set_color_button(btn, c.name())

    def _reset_defaults(self):
        defaults = dict(globals().get("HIST_DEFAULT_STYLE", {}))
        defaults.setdefault("show_median_marker", False)
        axis_defaults = _hist_default_axis_labels()

        self.style.update(defaults)

        _hist_set_color_button(self.btn_bar_fill, self.style["bar_facecolor"])
        _hist_set_color_button(self.btn_bar_edge, self.style["bar_edgecolor"])
        _hist_set_color_button(self.btn_axes, self.style["axes_color"])
        _hist_set_color_button(self.btn_cum, self.style["cum_color"])
        _hist_set_color_button(self.btn_poly, self.style["poly_color"])
        _hist_set_color_button(self.btn_laser, self.style["laser_color"])
        _hist_set_color_button(self.btn_overlay, self.style["overlay_color"])
        _hist_set_color_button(self.btn_stats, self.style["stats_marker_color"])

        self.sp_bar_width.setValue(float(self.style["bar_width_factor"]))
        self.sp_bar_edge_width.setValue(float(self.style["bar_edgewidth"]))
        self.sp_axes_width.setValue(float(self.style["axes_linewidth"]))
        self.sp_cum_width.setValue(float(self.style["cum_linewidth"]))
        self.sp_poly_width.setValue(float(self.style["poly_linewidth"]))
        self.sp_laser_width.setValue(float(self.style["laser_linewidth"]))
        self.sp_overlay_width.setValue(float(self.style["overlay_linewidth"]))
        self.sp_overlay_alpha.setValue(float(self.style["overlay_alpha"]))
        self.sp_fig_w.setValue(float(self.style["figure_width"]))
        self.sp_fig_h.setValue(float(self.style["figure_height"]))
        self.sp_stats_alpha.setValue(float(self.style["stats_marker_alpha"]))
        self.sp_stats_linewidth.setValue(float(self.style["stats_linewidth"]))

        idx = self.cmb_overlay_hatch.findText(self.style["overlay_hatch"])
        if idx >= 0:
            self.cmb_overlay_hatch.setCurrentIndex(idx)

        idx = self.cmb_x_mode.findData(self.style.get("x_axis_mode", "phi"))
        if idx >= 0:
            self.cmb_x_mode.setCurrentIndex(idx)

        idx = self.cmb_y_right.findData(self.style.get("y_right_mode", "cumulative"))
        if idx >= 0:
            self.cmb_y_right.setCurrentIndex(idx)

        idx = self.cmb_cum_layer.findData(self.style.get("cumulative_layer", "front"))
        if idx >= 0:
            self.cmb_cum_layer.setCurrentIndex(idx)

        idx = self.cmb_um_axis_position.findData(self.style.get("um_axis_position", "top"))
        if idx >= 0:
            self.cmb_um_axis_position.setCurrentIndex(idx)

        self.ed_x_phi.setText(_hist_to_str(self.style.get("x_label_phi", ""), ""))
        self.ed_x_phi.setPlaceholderText(axis_defaults["x_phi"])

        self.ed_x_um.setText(_hist_to_str(self.style.get("x_label_um", ""), ""))
        self.ed_x_um.setPlaceholderText(axis_defaults["x_um"])

        self.ed_y_left.setText(_hist_to_str(self.style.get("y_label_left", ""), ""))
        self.ed_y_left.setPlaceholderText(axis_defaults["y_left"])

        self.ed_y_right.setText(_hist_to_str(self.style.get("y_label_right", ""), ""))
        self.ed_y_right.setPlaceholderText(axis_defaults["y_right"])

        self.ed_um_secondary.setText(_hist_to_str(self.style.get("um_secondary_label", ""), ""))
        self.ed_um_secondary.setPlaceholderText(axis_defaults["um_secondary"])

    def _accept(self):
        self.style["bar_width_factor"] = float(self.sp_bar_width.value())
        self.style["bar_edgewidth"] = float(self.sp_bar_edge_width.value())
        self.style["axes_linewidth"] = float(self.sp_axes_width.value())
        self.style["cum_linewidth"] = float(self.sp_cum_width.value())
        self.style["poly_linewidth"] = float(self.sp_poly_width.value())
        self.style["laser_linewidth"] = float(self.sp_laser_width.value())
        self.style["overlay_linewidth"] = float(self.sp_overlay_width.value())
        self.style["overlay_alpha"] = float(self.sp_overlay_alpha.value())
        self.style["overlay_hatch"] = str(self.cmb_overlay_hatch.currentText())
        self.style["figure_width"] = float(self.sp_fig_w.value())
        self.style["figure_height"] = float(self.sp_fig_h.value())
        self.style["stats_marker_alpha"] = float(self.sp_stats_alpha.value())
        self.style["stats_linewidth"] = float(self.sp_stats_linewidth.value())

        self.style["x_axis_mode"] = self.cmb_x_mode.currentData()
        self.style["y_right_mode"] = self.cmb_y_right.currentData()
        self.style["cumulative_layer"] = self.cmb_cum_layer.currentData()
        self.style["um_axis_position"] = self.cmb_um_axis_position.currentData()

        self.style["x_label_phi"] = self.ed_x_phi.text().strip()
        self.style["x_label_um"] = self.ed_x_um.text().strip()
        self.style["y_label_left"] = self.ed_y_left.text().strip()
        self.style["y_label_right"] = self.ed_y_right.text().strip()
        self.style["um_secondary_label"] = self.ed_um_secondary.text().strip()

        self.accept()


class HistogramExportDialog(QDialog):
    def __init__(self, datasets, get_samples_fn, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("hist_export_title"))
        self.resize(700, 520)

        self.datasets = datasets
        self.get_samples_fn = get_samples_fn

        lay = QVBoxLayout(self)

        grp_db = QGroupBox(tr("hist_export_select_db"))
        vdb = QVBoxLayout(grp_db)
        self.db_list = QListWidget()
        self.db_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for key, label in datasets:
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, key)
            it.setSelected(True)
            self.db_list.addItem(it)
        self.db_list.itemSelectionChanged.connect(self._refresh_samples)
        vdb.addWidget(self.db_list)
        lay.addWidget(grp_db)

        grp_samples = QGroupBox(tr("hist_export_select_samples"))
        vs = QVBoxLayout(grp_samples)
        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.MultiSelection)
        vs.addWidget(self.sample_list)
        lay.addWidget(grp_samples)

        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel(tr("hist_export_folder")))
        self.lbl_folder = QLabel("")
        self.lbl_folder.setWordWrap(True)
        folder_row.addWidget(self.lbl_folder, 1)
        self.btn_folder = QPushButton(tr("hist_export_choose_folder"))
        self.btn_folder.clicked.connect(self._choose_folder)
        folder_row.addWidget(self.btn_folder)
        lay.addLayout(folder_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self.cmb_fmt = QComboBox()
        self.cmb_fmt.addItems([tr("hist_export_png"), tr("hist_export_svg"), tr("hist_export_pdf")])
        fmt_row.addWidget(self.cmb_fmt)
        fmt_row.addStretch(1)
        lay.addLayout(fmt_row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._validate_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.output_folder = ""
        self._refresh_samples()

    def _selected_db_keys(self):
        out = []
        for i in range(self.db_list.count()):
            it = self.db_list.item(i)
            if it.isSelected():
                out.append(it.data(Qt.UserRole))
        return out

    def _refresh_samples(self):
        old_selected = {
            self.sample_list.item(i).text()
            for i in range(self.sample_list.count())
            if self.sample_list.item(i).isSelected()
        }
        self.sample_list.clear()

        items = []
        for db_key in self._selected_db_keys():
            for s in self.get_samples_fn(db_key):
                items.append((db_key, s))

        visible = []
        for db_key, s in items:
            label = f"[{db_key}] {s}"
            visible.append((label, db_key, s))

        for label, db_key, s in visible:
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, (db_key, s))
            if not old_selected or label in old_selected:
                it.setSelected(True)
            self.sample_list.addItem(it)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr("hist_export_choose_folder"), "")
        if folder:
            self.output_folder = folder
            self.lbl_folder.setText(folder)

    def _validate_accept(self):
        if not self.output_folder:
            QMessageBox.warning(self, tr("error"), tr("hist_export_need_folder"))
            return
        nsel = 0
        for i in range(self.sample_list.count()):
            if self.sample_list.item(i).isSelected():
                nsel += 1
        if nsel == 0:
            QMessageBox.warning(self, tr("error"), tr("hist_export_none"))
            return
        self.accept()

    def get_payload(self):
        fmt_map = {
            tr("hist_export_png"): "png",
            tr("hist_export_svg"): "svg",
            tr("hist_export_pdf"): "pdf",
        }
        fmt = fmt_map.get(self.cmb_fmt.currentText(), "png")
        sel = []
        for i in range(self.sample_list.count()):
            it = self.sample_list.item(i)
            if it.isSelected():
                db_key, s = it.data(Qt.UserRole)
                sel.append((db_key, s))
        return {
            "folder": self.output_folder,
            "format": fmt,
            "samples": sel,
        }


# =========================================================
# Histogram UI callbacks
# =========================================================
def _hist_update_sample_combo(self, preferred_sample=None):
    if not hasattr(self, "cmb_hist_sample"):
        return

    db_key = self.cmb_hist_db.currentData() if self.cmb_hist_db.count() > 0 else None
    names = _hist_get_sample_names(self, db_key)

    old_sample = preferred_sample
    if old_sample is None and self.cmb_hist_sample.count() > 0:
        old_sample = self.cmb_hist_sample.currentText()

    self.cmb_hist_sample.blockSignals(True)
    self.cmb_hist_sample.clear()
    for name in names:
        self.cmb_hist_sample.addItem(name)
    self.cmb_hist_sample.blockSignals(False)

    if names:
        if old_sample in names:
            self.cmb_hist_sample.setCurrentText(old_sample)
        else:
            self.cmb_hist_sample.setCurrentIndex(0)

    if hasattr(self, "btn_hist_prev"):
        self.btn_hist_prev.setEnabled(len(names) > 1)
    if hasattr(self, "btn_hist_next"):
        self.btn_hist_next.setEnabled(len(names) > 1)


def refresh_histogram_selectors(self, preferred_db=None, preferred_sample=None):
    if not hasattr(self, "cmb_hist_db"):
        return

    _hist_ensure_style(self)
    _hist_apply_font_once(self)

    datasets = _hist_get_available_datasets(self)

    old_db = preferred_db
    if old_db is None and self.cmb_hist_db.count() > 0:
        old_db = self.cmb_hist_db.currentData()

    self.cmb_hist_db.blockSignals(True)
    self.cmb_hist_db.clear()
    for key, label in datasets:
        self.cmb_hist_db.addItem(label, key)
    self.cmb_hist_db.blockSignals(False)

    if datasets:
        keys = [k for k, _ in datasets]
        if old_db in keys:
            self.cmb_hist_db.setCurrentIndex(keys.index(old_db))
        else:
            self.cmb_hist_db.setCurrentIndex(0)

    _hist_update_sample_combo(self, preferred_sample=preferred_sample)
    plot_histogram(self)


def _on_hist_db_changed(self):
    _hist_update_sample_combo(self)
    plot_histogram(self)


def _hist_move_sample(self, direction):
    if not hasattr(self, "cmb_hist_sample"):
        return
    n = self.cmb_hist_sample.count()
    if n <= 1:
        return
    i = self.cmb_hist_sample.currentIndex()
    if i < 0:
        self.cmb_hist_sample.setCurrentIndex(0)
        return
    new_i = (i + int(direction)) % n
    self.cmb_hist_sample.setCurrentIndex(new_i)
    plot_histogram(self)


def _hist_toggle_from_checkbox(self):
    _hist_ensure_style(self)
    self.hist_style["show_cumulative"] = bool(self.chk_hist_cum.isChecked())
    self.hist_style["show_polygon"] = bool(self.chk_hist_poly.isChecked())
    self.hist_style["show_um_axis"] = bool(self.chk_hist_um.isChecked())
    self.hist_style["reverse_x"] = bool(self.chk_hist_reverse.isChecked())
    self.hist_style["show_params_text"] = bool(self.chk_hist_params.isChecked())
    self.hist_style["show_modified_overlay"] = bool(self.chk_hist_overlay.isChecked())
    self.hist_style["show_laser_curve"] = bool(self.chk_hist_laser.isChecked())
    self.hist_style["show_mean_marker"] = bool(self.chk_hist_mean.isChecked())
    self.hist_style["show_median_marker"] = bool(self.chk_hist_median.isChecked())
    self.hist_style["show_sigma_marker"] = bool(self.chk_hist_sigma.isChecked())
    self.hist_style["show_mode_marker"] = bool(self.chk_hist_mode.isChecked())
    _hist_save_style(self)
    plot_histogram(self)


def _hist_sync_checkboxes_from_style(self):
    _hist_ensure_style(self)

    widgets = [
        ("chk_hist_cum", "show_cumulative"),
        ("chk_hist_poly", "show_polygon"),
        ("chk_hist_um", "show_um_axis"),
        ("chk_hist_reverse", "reverse_x"),
        ("chk_hist_params", "show_params_text"),
        ("chk_hist_overlay", "show_modified_overlay"),
        ("chk_hist_laser", "show_laser_curve"),
        ("chk_hist_mean", "show_mean_marker"),
        ("chk_hist_median", "show_median_marker"),
        ("chk_hist_sigma", "show_sigma_marker"),
        ("chk_hist_mode", "show_mode_marker"),
    ]

    for attr, key in widgets:
        if hasattr(self, attr):
            w = getattr(self, attr)
            w.blockSignals(True)
            w.setChecked(bool(self.hist_style.get(key, False)))
            w.blockSignals(False)


def edit_histogram_style(self):
    _hist_ensure_style(self)
    dlg = HistogramStyleDialog(self.hist_style, self)
    if dlg.exec_():
        self.hist_style.update(dlg.style)
        _hist_save_style(self)

        try:
            self.canvas_hist.set_base_size_inches(
                float(self.hist_style["figure_width"]),
                float(self.hist_style["figure_height"])
            )
        except Exception:
            pass

        _hist_sync_checkboxes_from_style(self)
        plot_histogram(self)


# =========================================================
# Export
# =========================================================
def _export_hist_current(self):
    if not hasattr(self, "canvas_hist"):
        return
    self.export_canvas(self.canvas_hist)


def _export_hist_batch(self):
    datasets = _hist_get_available_datasets(self)
    if not datasets:
        QMessageBox.information(self, tr("info"), tr("hist_export_none"))
        return

    dlg = HistogramExportDialog(datasets, lambda db_key: _hist_get_sample_names(self, db_key), self)
    if not dlg.exec_():
        return

    payload = dlg.get_payload()
    folder = payload["folder"]
    fmt = payload["format"]
    items = payload["samples"]

    if not items:
        QMessageBox.information(self, tr("info"), tr("hist_export_none"))
        return

    current_db = self.cmb_hist_db.currentData() if self.cmb_hist_db.count() > 0 else None
    current_sample = self.cmb_hist_sample.currentText() if self.cmb_hist_sample.count() > 0 else None

    for db_key, sample_name in items:
        refresh_histogram_selectors(self, preferred_db=db_key, preferred_sample=sample_name)
        plot_histogram(self)

        safe_sample = self._safe_filename(str(sample_name)) if hasattr(self, "_safe_filename") else str(sample_name).replace("/", "_")
        base_label = dict(_hist_get_available_datasets(self)).get(db_key, db_key)
        safe_base = self._safe_filename(str(base_label)) if hasattr(self, "_safe_filename") else str(base_label).replace("/", "_")

        out = os.path.join(folder, f"{safe_base}__{safe_sample}.{fmt}")
        try:
            self.canvas_hist.figure.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        except Exception as e:
            QMessageBox.warning(self, tr("error"), f"Could not export:\n{out}\n\n{e}")

    refresh_histogram_selectors(self, preferred_db=current_db, preferred_sample=current_sample)
    plot_histogram(self)

    QMessageBox.information(self, tr("info"), tr("hist_export_done"))


def export_histograms(self):
    dlg = QDialog(self)
    dlg.setWindowTitle(tr("hist_export_title"))
    lay = QVBoxLayout(dlg)

    btn_current = QPushButton(tr("hist_export_current"))
    btn_batch = QPushButton(tr("hist_export_batch"))
    btn_cancel = QPushButton(tr("cancel"))

    btn_current.clicked.connect(lambda: dlg.done(1))
    btn_batch.clicked.connect(lambda: dlg.done(2))
    btn_cancel.clicked.connect(dlg.reject)

    lay.addWidget(btn_current)
    lay.addWidget(btn_batch)
    lay.addWidget(btn_cancel)

    rc = dlg.exec_()
    if rc == 1:
        _export_hist_current(self)
    elif rc == 2:
        _export_hist_batch(self)

# endregion


# region # === Block 18: Histogram plotting, UI patching and entry point ===

# =========================================================
# Plot
# =========================================================
def plot_histogram(self):
    if not hasattr(self, "canvas_hist"):
        return

    _hist_ensure_style(self)

    try:
        self.canvas_hist.set_base_size_inches(
            float(self.hist_style["figure_width"]),
            float(self.hist_style["figure_height"])
        )
    except Exception:
        pass

    zoom = float(getattr(self.canvas_hist, "_zoom", 1.0))

    title_fs = 24.0 * zoom
    label_fs = 16.0 * zoom
    right_label_fs = 14.0 * zoom
    tick_fs = 11.0 * zoom
    annot_fs = 12.0 * zoom

    cum_lw = float(self.hist_style["cum_linewidth"]) * zoom
    poly_lw = float(self.hist_style["poly_linewidth"]) * zoom
    laser_lw = float(self.hist_style["laser_linewidth"]) * zoom
    bar_edge_lw = float(self.hist_style["bar_edgewidth"]) * zoom
    axes_lw = float(self.hist_style["axes_linewidth"]) * zoom
    overlay_lw = float(self.hist_style["overlay_linewidth"]) * zoom
    stats_lw = float(self.hist_style.get("stats_linewidth", 1.0)) * zoom
    stats_alpha = float(self.hist_style.get("stats_marker_alpha", 0.18))
    stats_color = self.hist_style.get("stats_marker_color", "#808080")
    stats_ls = self.hist_style.get("stats_line_style", "--")

    fig = self.canvas_hist.figure
    fig.clear()
    fig.patch.set_facecolor("white")

    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    ax.set_facecolor("none")
    ax2.set_facecolor("none")
    try:
        ax.patch.set_alpha(0.0)
        ax2.patch.set_alpha(0.0)
    except Exception:
        pass

    if str(self.hist_style.get("cumulative_layer", "front")).lower() == "back":
        ax2.set_zorder(0)
        ax.set_zorder(1)
    else:
        ax.set_zorder(0)
        ax2.set_zorder(1)

    db_key = self.cmb_hist_db.currentData() if hasattr(self, "cmb_hist_db") and self.cmb_hist_db.count() > 0 else None
    sample_name = self.cmb_hist_sample.currentText().strip() if hasattr(self, "cmb_hist_sample") and self.cmb_hist_sample.count() > 0 else ""

    if not db_key or not sample_name:
        ax.text(0.5, 0.5, tr("hist_load_prompt"), ha="center", va="center", transform=ax.transAxes, fontsize=14 * zoom)
        ax.set_axis_off()
        ax2.set_axis_off()
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.90)
        self.canvas_hist.draw()
        self.canvas_hist.update()
        return

    df_main, meta = _hist_get_distribution(self, db_key, sample_name)
    if df_main is None or df_main.empty:
        ax.text(0.5, 0.5, tr("hist_no_data"), ha="center", va="center", transform=ax.transAxes, fontsize=14 * zoom)
        ax.set_axis_off()
        ax2.set_axis_off()
        fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.90)
        self.canvas_hist.draw()
        self.canvas_hist.update()
        return

    df_main = df_main.sort_values("phi").reset_index(drop=True)

    # Plot only the interval bracketed by the first and last NON-ZERO fraction.
    # Zero-weight bins OUTSIDE that interval are database padding and must never
    # be drawn. Zero-weight bins INSIDE the occupied interval are retained so the
    # histogram preserves real gaps in the sample distribution.
    _phi_all = pd.to_numeric(df_main["phi"], errors="coerce").to_numpy(dtype=float)
    _wt_all = pd.to_numeric(df_main["wt"], errors="coerce").to_numpy(dtype=float)
    _finite = np.isfinite(_phi_all) & np.isfinite(_wt_all)
    _positive_idx = np.flatnonzero(_finite & (_wt_all > 0.0))
    if _positive_idx.size:
        _first = int(_positive_idx[0])
        _last = int(_positive_idx[-1])
        df_main = df_main.iloc[_first:_last + 1].copy().reset_index(drop=True)
    else:
        df_main = df_main.loc[_finite].copy().reset_index(drop=True)

    phi = df_main["phi"].astype(float).values
    wt = df_main["wt"].astype(float).values
    phi_extent = phi[np.isfinite(phi)]
    if phi_extent.size == 0:
        phi_extent = phi

    axis_defaults = _hist_default_axis_labels()
    x_mode = str(self.hist_style.get("x_axis_mode", "phi")).lower()

    if x_mode == "um":
        x_vals, bar_widths = _hist_um_bar_geometry(phi, self.hist_style["bar_width_factor"])
        x_extent = np.array([_phi_to_um(v) for v in phi_extent], dtype=float)
        tick_pos = _hist_pick_sparse_positions(x_extent, max_ticks=10)
        tick_labels = [_hist_fmt_num(v, 1) for v in tick_pos]
        x_label = self.hist_style.get("x_label_um", "").strip() or axis_defaults["x_um"]
    else:
        step = _hist_step_from_phi(phi_extent)
        x_vals = phi.copy()
        bar_widths = np.full_like(x_vals, max(0.01, float(self.hist_style["bar_width_factor"]) * step), dtype=float)
        tick_pos = _hist_make_nice_ticks(np.nanmin(phi_extent), np.nanmax(phi_extent), step)
        # Never let the tick generator escape the occupied sample range.
        lo_phi = float(np.nanmin(phi_extent))
        hi_phi = float(np.nanmax(phi_extent))
        tick_pos = np.asarray([v for v in tick_pos if lo_phi - 1e-9 <= float(v) <= hi_phi + 1e-9], dtype=float)
        if tick_pos.size == 0 or not np.isclose(tick_pos[0], lo_phi):
            tick_pos = np.insert(tick_pos, 0, lo_phi)
        if not np.isclose(tick_pos[-1], hi_phi):
            tick_pos = np.append(tick_pos, hi_phi)
        tick_labels = [_hist_fmt_num(v, 2) for v in tick_pos]
        x_label = self.hist_style.get("x_label_phi", "").strip() or axis_defaults["x_phi"]

    y_left_label = self.hist_style.get("y_label_left", "").strip() or axis_defaults["y_left"]
    y_right_label = self.hist_style.get("y_label_right", "").strip() or axis_defaults["y_right"]
    um_secondary_label = self.hist_style.get("um_secondary_label", "").strip() or axis_defaults["um_secondary"]

    cum = cumulative_distribution(phi, wt)

    ax.bar(
        x_vals,
        wt,
        width=bar_widths,
        align="center",
        color=self.hist_style["bar_facecolor"],
        edgecolor=self.hist_style["bar_edgecolor"],
        linewidth=bar_edge_lw,
        zorder=2
    )

    if db_key == "hybrid" and bool(self.hist_style.get("show_modified_overlay", False)):
        df_overlay = _hist_get_modified_overlay_df(self, sample_name, df_main)
        if df_overlay is not None and not df_overlay.empty:
            phi_ov = df_overlay["phi"].astype(float).values
            wt_ov = df_overlay["wt"].astype(float).values

            if x_mode == "um":
                x_ov, widths_ov = _hist_um_bar_geometry(phi_ov, self.hist_style["bar_width_factor"])
            else:
                step_ov = _hist_step_from_phi(phi_ov)
                x_ov = phi_ov.copy()
                widths_ov = np.full_like(x_ov, max(0.01, float(self.hist_style["bar_width_factor"]) * step_ov), dtype=float)

            ax.bar(
                x_ov,
                wt_ov,
                width=widths_ov,
                align="center",
                facecolor=self.hist_style["overlay_color"],
                edgecolor=self.hist_style["overlay_color"],
                linewidth=overlay_lw,
                alpha=float(self.hist_style["overlay_alpha"]),
                hatch=self.hist_style["overlay_hatch"],
                fill=True,
                zorder=3
            )

    if bool(self.hist_style.get("show_polygon", False)):
        ax.plot(
            x_vals,
            wt,
            color=self.hist_style["poly_color"],
            linewidth=poly_lw,
            marker="o",
            markersize=4.5 * zoom,
            zorder=4
        )

    y_right_mode = str(self.hist_style.get("y_right_mode", "cumulative")).lower()
    show_cum = bool(self.hist_style.get("show_cumulative", True)) and (y_right_mode == "cumulative")
    if show_cum:
        ax2.plot(
            x_vals,
            cum,
            color=self.hist_style["cum_color"],
            linewidth=cum_lw,
            marker="s",
            markersize=5.5 * zoom,
            markerfacecolor="white",
            markeredgecolor=self.hist_style["cum_color"],
            markeredgewidth=max(0.8, cum_lw * 0.7),
            zorder=1 if str(self.hist_style.get("cumulative_layer", "front")).lower() == "back" else 8
        )
        ax2.set_ylim(0, 102)
        ax2.set_yticks(np.arange(0, 101, 20))
        ax2.set_ylabel(y_right_label, fontsize=right_label_fs, fontweight="bold")
        ax2.yaxis.set_visible(True)
    else:
        ax2.set_yticks([])
        ax2.set_ylabel("")
        ax2.yaxis.set_visible(False)
        try:
            ax2.spines["right"].set_visible(False)
        except Exception:
            pass

    laser_overlay = meta.get("laser_overlay", None)
    if bool(self.hist_style.get("show_laser_curve", False)) and laser_overlay is not None and isinstance(laser_overlay, dict):
        ldf = laser_overlay.get("df", pd.DataFrame()).copy()
        if not ldf.empty:
            phi_l = ldf["phi"].astype(float).values
            wt_l = ldf["wt"].astype(float).values
            if x_mode == "um":
                x_l = np.array([_phi_to_um(v) for v in phi_l], dtype=float)
            else:
                x_l = phi_l
            ax.plot(
                x_l,
                wt_l,
                color=self.hist_style["laser_color"],
                linewidth=laser_lw,
                marker="x",
                markersize=6.0 * zoom,
                zorder=6
            )

    # statistical markers
    prow = _hist_get_param_row(self, db_key, sample_name)
    mean_phi = np.nan
    median_phi = np.nan
    sigma_val = np.nan

    if prow is not None:
        try:
            mean_phi = float(prow["mean"]) if "mean" in prow.index and pd.notna(prow["mean"]) else np.nan
        except Exception:
            mean_phi = np.nan
        try:
            median_phi = float(prow["median"]) if "median" in prow.index and pd.notna(prow["median"]) else np.nan
        except Exception:
            median_phi = np.nan
        try:
            sigma_val = float(prow["sigma"]) if "sigma" in prow.index and pd.notna(prow["sigma"]) else np.nan
        except Exception:
            sigma_val = np.nan

    if bool(self.hist_style.get("show_mean_marker", False)) and np.isfinite(mean_phi):
        x_mean = _hist_plot_x_from_phi_value(mean_phi, x_mode)
        ax.axvline(
            x=x_mean,
            color=stats_color,
            linewidth=stats_lw,
            linestyle=stats_ls,
            alpha=min(1.0, max(0.0, stats_alpha + 0.15)),
            zorder=7
        )

    if bool(self.hist_style.get("show_median_marker", False)) and np.isfinite(median_phi):
        x_median = _hist_plot_x_from_phi_value(median_phi, x_mode)
        ax.axvline(
            x=x_median,
            color=stats_color,
            linewidth=stats_lw,
            linestyle="-.",
            alpha=min(1.0, max(0.0, stats_alpha + 0.15)),
            zorder=7
        )

    if bool(self.hist_style.get("show_sigma_marker", False)) and np.isfinite(mean_phi) and np.isfinite(sigma_val) and sigma_val > 0:
        phi_left = mean_phi - sigma_val
        phi_right = mean_phi + sigma_val
        x_left = _hist_plot_x_from_phi_value(phi_left, x_mode)
        x_right = _hist_plot_x_from_phi_value(phi_right, x_mode)
        xa, xb = (min(x_left, x_right), max(x_left, x_right))
        ax.axvspan(
            xa,
            xb,
            color=stats_color,
            alpha=stats_alpha,
            zorder=1.5
        )
        ax.axvline(
            x=x_left,
            color=stats_color,
            linewidth=stats_lw,
            linestyle=stats_ls,
            alpha=min(1.0, max(0.0, stats_alpha + 0.10)),
            zorder=7
        )
        ax.axvline(
            x=x_right,
            color=stats_color,
            linewidth=stats_lw,
            linestyle=stats_ls,
            alpha=min(1.0, max(0.0, stats_alpha + 0.10)),
            zorder=7
        )

    if bool(self.hist_style.get("show_mode_marker", False)):
        mode_phi_values = _hist_modes_from_distribution(phi, wt, min_mode_fraction=0.10)
        for p_mode in mode_phi_values:
            x_mode_val = _hist_plot_x_from_phi_value(p_mode, x_mode)
            ax.axvline(
                x=x_mode_val,
                color=stats_color,
                linewidth=stats_lw,
                linestyle=":",
                alpha=min(1.0, max(0.0, stats_alpha + 0.20)),
                zorder=7
            )

    if bool(self.hist_style.get("show_params_text", True)):
        ann_lines = []
        if prow is not None:
            if "sigma" in prow.index and pd.notna(prow["sigma"]):
                ann_lines.append(f"σ: {_hist_fmt_num(prow['sigma'])}")
            if "kurtosis" in prow.index and pd.notna(prow["kurtosis"]):
                ann_lines.append(f"K: {_hist_fmt_num(prow['kurtosis'])}")
            if "median" in prow.index and pd.notna(prow["median"]):
                ann_lines.append(f"Md: {_hist_fmt_num(prow['median'])}")
            if "mean" in prow.index and pd.notna(prow["mean"]):
                ann_lines.append(f"M: {_hist_fmt_num(prow['mean'])}")
            if "skewness" in prow.index and pd.notna(prow["skewness"]):
                ann_lines.append(f"Sk: {_hist_fmt_num(prow['skewness'])}")

        if ann_lines:
            ax.text(
                0.10, 0.88,
                "\n".join(ann_lines),
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=annot_fs,
                color="#666666"
            )

    ax.set_title(sample_name, fontsize=title_fs, fontweight="bold", pad=14 * zoom)
    ax.set_xlabel(x_label, fontsize=label_fs, fontweight="bold")
    ax.set_ylabel(y_left_label, fontsize=label_fs, fontweight="bold")

    ymax = float(np.nanmax(wt)) if len(wt) else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.18)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)

    # The visible limits must contain the COMPLETE first and last bars.
    # The extreme phi values are bar CENTRES; using them directly as x-limits
    # cuts half of the first/last bars.  Use the real outside bar edges instead.
    _bar_left_edges = np.asarray(x_vals, dtype=float) - (np.asarray(bar_widths, dtype=float) / 2.0)
    _bar_right_edges = np.asarray(x_vals, dtype=float) + (np.asarray(bar_widths, dtype=float) / 2.0)
    _valid_edges = np.isfinite(_bar_left_edges) & np.isfinite(_bar_right_edges)
    if np.any(_valid_edges):
        xmin = float(np.nanmin(_bar_left_edges[_valid_edges]))
        xmax = float(np.nanmax(_bar_right_edges[_valid_edges]))
        _span = max(abs(xmax - xmin), 1e-9)
        _max_bar_width = float(np.nanmax(np.abs(np.asarray(bar_widths, dtype=float)))) if len(bar_widths) else 0.0
        # Small breathing room outside the bar outline; it does NOT introduce
        # extra phi bins or ticks, it only prevents the outline from touching
        # the axes frame.
        _edge_pad = max(_span * 0.008, _max_bar_width * 0.10, 1e-9)
        ax.set_xlim(xmin - _edge_pad, xmax + _edge_pad)
    else:
        # Defensive fallback for malformed/empty geometry.
        xmin = float(np.nanmin(phi_extent))
        xmax = float(np.nanmax(phi_extent))
        if np.isfinite(xmin) and np.isfinite(xmax) and np.isclose(xmin, xmax):
            pad = max(0.5, abs(float(_hist_step_from_phi(phi))) * 0.5)
            xmin -= pad
            xmax += pad
        ax.set_xlim(xmin, xmax)

    if bool(self.hist_style.get("reverse_x", False)):
        ax.invert_xaxis()
        try:
            ax2.set_xlim(ax.get_xlim())
        except Exception:
            pass

    secax = None
    if bool(self.hist_style.get("show_um_axis", False)) and x_mode == "phi":
        um_pos = str(self.hist_style.get("um_axis_position", "top")).lower()
        secax = ax.secondary_xaxis("top" if um_pos == "top" else "bottom")
        # Micrometre labels are wider than phi labels. Use an independent sparse
        # tick set so they never pile on top of one another while preserving both
        # real sample extremes.
        um_tick_pos = _hist_pick_sparse_positions(np.asarray(tick_pos, dtype=float), max_ticks=9)
        secax.set_xticks(um_tick_pos)
        secax.set_xticklabels([_um_label_from_phi(v) for v in um_tick_pos], fontsize=tick_fs)
        secax.set_xlabel(um_secondary_label, fontsize=13.0 * zoom, fontweight="bold")

        try:
            if um_pos == "bottom":
                secax.xaxis.set_ticks_position("bottom")
                secax.xaxis.set_label_position("bottom")
                secax.spines["bottom"].set_position(("outward", 34))
            else:
                secax.xaxis.set_ticks_position("top")
                secax.xaxis.set_label_position("top")

            side = "bottom" if um_pos == "bottom" else "top"
            secax.spines[side].set_color(self.hist_style["axes_color"])
            secax.spines[side].set_linewidth(axes_lw)
            secax.tick_params(
                axis="x",
                colors=self.hist_style["axes_color"],
                width=max(1.0, axes_lw * 0.9),
                length=6 * zoom,
                labelsize=tick_fs
            )
            secax.xaxis.label.set_color(self.hist_style["axes_color"])
        except Exception:
            pass

    ax.grid(False)
    ax2.grid(False)

    style_axes = dict(self.hist_style)
    style_axes["axes_linewidth"] = axes_lw
    _hist_style_axes(ax, ax2, style_axes)

    try:
        ax.tick_params(axis="both", which="both", labelsize=tick_fs, length=7 * zoom)
        ax2.tick_params(axis="y", which="both", labelsize=tick_fs, length=7 * zoom)
    except Exception:
        pass

    bottom_margin = 0.16
    top_margin = 0.84
    if secax is not None:
        um_pos = str(self.hist_style.get("um_axis_position", "top")).lower()
        if um_pos == "bottom":
            bottom_margin = 0.25
        else:
            top_margin = 0.78

    fig.subplots_adjust(
        left=0.12,
        right=0.88 if show_cum else 0.92,
        bottom=bottom_margin,
        top=top_margin
    )

    self.canvas_hist.draw()
    self.canvas_hist.update()


# =========================================================
# Patch initUI_tabs
# =========================================================
_old_initUI_tabs = MainWindow.initUI_tabs

def _patched_initUI_tabs(self):
    _old_initUI_tabs(self)

    if hasattr(self, "tab_hist"):
        return

    _hist_ensure_style(self)
    _hist_apply_font_once(self)

    self.tab_hist = QWidget()
    v_hist = QVBoxLayout(self.tab_hist)
    v_hist.setContentsMargins(8, 8, 8, 8)
    v_hist.setSpacing(8)

    # Row 1
    h1 = QHBoxLayout()
    h1.addWidget(QLabel(tr("hist_db")))
    self.cmb_hist_db = QComboBox()
    self.cmb_hist_db.currentIndexChanged.connect(lambda *_: _on_hist_db_changed(self))
    h1.addWidget(self.cmb_hist_db)

    h1.addSpacing(12)
    h1.addWidget(QLabel(tr("hist_sample")))

    self.btn_hist_prev = QPushButton("◀")
    self.btn_hist_prev.clicked.connect(lambda: _hist_move_sample(self, -1))
    h1.addWidget(self.btn_hist_prev)

    self.cmb_hist_sample = QComboBox()
    self.cmb_hist_sample.currentIndexChanged.connect(lambda *_: plot_histogram(self))
    h1.addWidget(self.cmb_hist_sample)

    self.btn_hist_next = QPushButton("▶")
    self.btn_hist_next.clicked.connect(lambda: _hist_move_sample(self, +1))
    h1.addWidget(self.btn_hist_next)

    h1.addStretch(1)
    v_hist.addLayout(h1)

    # Row 2
    h2 = QHBoxLayout()

    self.chk_hist_cum = QCheckBox(tr("hist_show_cumulative"))
    self.chk_hist_cum.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_cum)

    self.chk_hist_poly = QCheckBox(tr("hist_show_polygon"))
    self.chk_hist_poly.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_poly)

    self.chk_hist_um = QCheckBox(tr("hist_show_um_axis"))
    self.chk_hist_um.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_um)

    self.chk_hist_reverse = QCheckBox(tr("hist_reverse_x"))
    self.chk_hist_reverse.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_reverse)

    self.chk_hist_params = QCheckBox(tr("hist_show_params"))
    self.chk_hist_params.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_params)

    self.chk_hist_overlay = QCheckBox(tr("hist_show_modified_overlay"))
    self.chk_hist_overlay.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_overlay)

    self.chk_hist_laser = QCheckBox(tr("hist_show_laser_curve"))
    self.chk_hist_laser.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_laser)

    self.chk_hist_mean = QCheckBox(tr("hist_show_mean_marker"))
    self.chk_hist_mean.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_mean)

    self.chk_hist_median = QCheckBox(tr("hist_show_median_marker"))
    self.chk_hist_median.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_median)

    self.chk_hist_sigma = QCheckBox(tr("hist_show_sigma_marker"))
    self.chk_hist_sigma.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_sigma)

    self.chk_hist_mode = QCheckBox(tr("hist_show_mode_marker"))
    self.chk_hist_mode.toggled.connect(lambda *_: _hist_toggle_from_checkbox(self))
    h2.addWidget(self.chk_hist_mode)

    h2.addStretch(1)

    self.btn_hist_edit = QPushButton(tr("hist_edit"))
    self.btn_hist_edit.clicked.connect(lambda: edit_histogram_style(self))
    h2.addWidget(self.btn_hist_edit)

    self.btn_export_hist = QPushButton(tr("hist_export"))
    self.btn_export_hist.clicked.connect(lambda: export_histograms(self))
    h2.addWidget(self.btn_export_hist)

    v_hist.addLayout(h2)

    # Scroll area + centered zoomable canvas
    self.hist_scroll = QScrollArea()
    self.hist_scroll.setWidgetResizable(False)
    self.hist_scroll.setAlignment(Qt.AlignCenter)
    self.hist_scroll.setStyleSheet("QScrollArea { background: white; border: 1px solid #bfbfbf; }")

    fig = Figure(
        figsize=(float(self.hist_style["figure_width"]), float(self.hist_style["figure_height"])),
        dpi=100
    )
    fig.patch.set_facecolor("white")

    self.canvas_hist = ZoomPanFigureCanvas(fig)
    self.canvas_hist.set_base_size_inches(
        float(self.hist_style["figure_width"]),
        float(self.hist_style["figure_height"])
    )
    self.canvas_hist.on_zoom_callback = lambda: plot_histogram(self)

    self.hist_scroll.setWidget(self.canvas_hist)
    v_hist.addWidget(self.hist_scroll, 1)

    _hist_sync_checkboxes_from_style(self)

    try:
        self.tabs.insertTab(1, self.tab_hist, tr("tab_hist"))
    except Exception:
        self.tabs.addTab(self.tab_hist, tr("tab_hist"))

    refresh_histogram_selectors(self)

MainWindow.initUI_tabs = _patched_initUI_tabs


# =========================================================
# Patch load / refresh methods
# =========================================================
_old_load_file = MainWindow.load_file
def _patched_load_file(self):
    _old_load_file(self)
    try:
        refresh_histogram_selectors(self, preferred_db="tamiz")
    except Exception:
        pass
MainWindow.load_file = _patched_load_file

_old_load_laser_file = MainWindow.load_laser_file
def _patched_load_laser_file(self):
    _old_load_laser_file(self)
    try:
        refresh_histogram_selectors(self, preferred_db="laser")
    except Exception:
        pass
MainWindow.load_laser_file = _patched_load_laser_file

_old_combinar_tamiz_laser = MainWindow.combinar_tamiz_laser
def _patched_combinar_tamiz_laser(self):
    _old_combinar_tamiz_laser(self)
    try:
        refresh_histogram_selectors(self, preferred_db="hybrid")
    except Exception:
        pass
MainWindow.combinar_tamiz_laser = _patched_combinar_tamiz_laser

if hasattr(MainWindow, "_refresh_after_settings_change"):
    _old_refresh_after_settings_change = MainWindow._refresh_after_settings_change
    def _patched_refresh_after_settings_change(self):
        _old_refresh_after_settings_change(self)
        try:
            _hist_sync_checkboxes_from_style(self)
            plot_histogram(self)
        except Exception:
            pass
    MainWindow._refresh_after_settings_change = _patched_refresh_after_settings_change

if hasattr(MainWindow, "_recompute_all_params"):
    _old_recompute_all_params = MainWindow._recompute_all_params
    def _patched_recompute_all_params(self):
        _old_recompute_all_params(self)
        try:
            _hist_sync_checkboxes_from_style(self)
            plot_histogram(self)
        except Exception:
            pass
    MainWindow._recompute_all_params = _patched_recompute_all_params

if hasattr(MainWindow, "choose_theme"):
    _old_choose_theme = MainWindow.choose_theme
    def _patched_choose_theme(self):
        _old_choose_theme(self)
        try:
            plot_histogram(self)
        except Exception:
            pass
    MainWindow.choose_theme = _patched_choose_theme


# =========================================================
# Patch save_current_tab
# =========================================================
def _patched_save_current_tab(self):
    current = self.tabs.currentWidget()

    if current is getattr(self, "tab_xy", None):
        self.export_xy_plot()
    elif current is getattr(self, "tab_hist", None):
        self.export_canvas(self.canvas_hist)

MainWindow.save_current_tab = _patched_save_current_tab


# =========================================================
# Bind methods to class
# =========================================================
MainWindow.refresh_histogram_selectors = refresh_histogram_selectors
MainWindow.plot_histogram = plot_histogram
MainWindow.edit_histogram_style = edit_histogram_style
MainWindow.export_histograms = export_histograms




# =========================================================
# FW + Componentry integration block
# =========================================================
def _fw_json_default(obj):
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    try:
        import pandas as _pd
        if isinstance(obj, (_pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _fw_df_to_payload(df):
    if df is None:
        return None
    try:
        if getattr(df, "empty", False) and len(getattr(df, "columns", [])) == 0:
            return {"columns": [], "data": [], "index": []}
        out = df.copy()
        return {
            "columns": [str(c) for c in out.columns.tolist()],
            "index": [str(i) for i in out.index.tolist()],
            "data": out.where(pd.notnull(out), None).values.tolist(),
        }
    except Exception:
        return None


def _fw_df_from_payload(payload):
    if payload is None:
        return None
    try:
        return pd.DataFrame(payload.get("data", []), columns=payload.get("columns", []))
    except Exception:
        return None


def _fw_componentry_key_id(db_key, sample_name):
    return json.dumps([str(db_key), str(sample_name)], ensure_ascii=False)


def _fw_componentry_key_from_id(key_id):
    try:
        a = json.loads(key_id)
        if isinstance(a, list) and len(a) == 2:
            return (str(a[0]), str(a[1]))
    except Exception:
        pass
    if "::" in str(key_id):
        a, b = str(key_id).split("::", 1)
        return (a, b)
    return ("unknown", str(key_id))


def _fw_componentry_base_label(db_key):
    if db_key == "tamiz":
        return base_name_display("Tamiz")
    if db_key == "laser":
        return base_name_display("Laser")
    if db_key == "hybrid":
        return base_name_display("Híbrido")
    return str(db_key)


def _fw_componentry_display_label(self, key):
    db_key, sample_name = key
    return f"{_fw_componentry_base_label(db_key)} — {sample_name}"


def _fw_get_componentry_module(self):
    if hasattr(self, "_componentry_module") and self._componentry_module is not None:
        return self._componentry_module

    import importlib.util as _importlib_util
    candidates = []
    search_dirs = []
    try:
        search_dirs.append(get_script_dir())
    except Exception:
        pass
    try:
        search_dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    search_dirs.append(os.getcwd())
    search_dirs.append("/mnt/data")

    preferred_names = [
        "Componentry-FW.py",
        "componentry_fw.py",
        "componentry-integrated.py",
        "componentry_integrated.py",
    ]

    for d in search_dirs:
        try:
            for name in preferred_names:
                candidates.append(os.path.join(d, name))
        except Exception:
            pass

    for d in search_dirs:
        try:
            if not d or (not os.path.isdir(d)):
                continue
            for name in sorted(os.listdir(d)):
                low = name.lower()
                if low.endswith('.py') and ('componentry' in low):
                    p = os.path.join(d, name)
                    if p not in candidates:
                        candidates.append(p)
        except Exception:
            pass

    path = None
    for p in candidates:
        if p and os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(
            "Componentry-FW was not found in the same folder as FW-Granulometry 2.0.\n"
            "Keep both .py files in the same folder. Supported names include Componentry-FW.py and componentry_integrated.py."
        )

    spec = _importlib_util.spec_from_file_location("fw_componentry_integrated", path)
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    self._componentry_module = mod
    return mod


def _fw_normalize_gsd_df(df):
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["phi", "wt"])
    out = df[["phi", "wt"]].copy()
    out["phi"] = pd.to_numeric(out["phi"], errors="coerce")
    out["wt"] = pd.to_numeric(out["wt"], errors="coerce")
    out = out[np.isfinite(out["phi"]) & np.isfinite(out["wt"])]
    out = out.groupby("phi", as_index=False)["wt"].sum().sort_values("phi").reset_index(drop=True)
    total = float(out["wt"].sum()) if not out.empty else 0.0
    if total > 0 and not np.isclose(total, 100.0, atol=0.2):
        out["wt"] = out["wt"] * (100.0 / total)
    return out


def _fw_componentry_db_has_results(db):
    try:
        if bool(getattr(db, "components", [])):
            return True
        if bool(getattr(db, "filled", set())):
            return True
        total = getattr(db, "total", None)
        if total is not None and not getattr(total, "empty", True):
            return float(np.nansum(total.values)) > 0
    except Exception:
        pass
    return False


def _fw_gsd_equivalent(a, b):
    try:
        aa = _fw_normalize_gsd_df(a).reset_index(drop=True)
        bb = _fw_normalize_gsd_df(b).reset_index(drop=True)
        if len(aa) != len(bb):
            return False
        return np.allclose(aa["phi"].values, bb["phi"].values, equal_nan=True) and np.allclose(aa["wt"].values, bb["wt"].values, atol=1e-6, equal_nan=True)
    except Exception:
        return False



def _fw_componentry_after_change(self, view_state=None):
    if isinstance(view_state, dict):
        self.componentry_view_state = {
            "plot_style": dict(view_state.get("plot_style", {}) or {}),
            "palette_state": dict(view_state.get("palette_state", {}) or {}),
        }
    try:
        self._update_db_menu_actions()
    except Exception:
        pass


def _fw_componentry_rows_from_db(db, table_kind="total"):
    try:
        if table_kind == "frac100":
            df = db.frac100
            cols = list(db.components)
        elif table_kind == "type":
            df = db.totals_by_type()
            cols = list(getattr(_fw_get_componentry_module.__self__, "TYPE_OPTIONS", []))
        else:
            df = db.total
            cols = list(db.components)
    except Exception:
        df = pd.DataFrame()
        cols = []
    return df, cols


def show_componentry_db_window(self):
    try:
        if not getattr(self, "componentry_dbs", None):
            QMessageBox.information(self, "Componentry database", "No componentry data has been created yet.")
            return

        comp_mod = _fw_get_componentry_module(self)
        dlg = QDialog(self)
        dlg.setWindowTitle("Componentry database")
        dlg.resize(1100, 700)
        root = QVBoxLayout(dlg)

        top = QHBoxLayout()
        top.addWidget(QLabel("Sample:"))
        cmb = QComboBox()
        key_by_label = {}
        for key in self.componentry_dbs.keys():
            label = _fw_componentry_display_label(self, key)
            key_by_label[label] = key
            cmb.addItem(label)
        top.addWidget(cmb, 1)
        root.addLayout(top)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        table_total = QTableWidget()
        table_frac = QTableWidget()
        table_type = QTableWidget()
        tabs.addTab(table_total, "Total per fraction (wt%)")
        tabs.addTab(table_frac, "Vol. 100%")
        tabs.addTab(table_type, "By component type (wt%)")

        def fill_table(table, df, cols, phis):
            header = ["phi"] + list(cols)
            table.setColumnCount(len(header))
            table.setHorizontalHeaderLabels(header)
            table.setRowCount(len(phis))
            for i, phi in enumerate(phis):
                it_phi = QTableWidgetItem(f"{float(phi):g}")
                it_phi.setFlags(it_phi.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, 0, it_phi)
                for j, c in enumerate(cols, start=1):
                    try:
                        v = float(df.loc[float(phi), c])
                    except Exception:
                        v = 0.0
                    it = QTableWidgetItem(f"{v:.4f}")
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    table.setItem(i, j, it)
            try:
                table.resizeColumnsToContents()
            except Exception:
                pass

        def refresh_tables():
            key = key_by_label.get(cmb.currentText())
            db = self.componentry_dbs.get(key)
            if db is None:
                return
            try:
                phis = db.gsd[db.gsd["wt"].astype(float) > 0]["phi"].astype(float).values.tolist()
            except Exception:
                phis = []
            fill_table(table_total, db.total, db.components, phis)
            fill_table(table_frac, db.frac100, db.components, phis)
            fill_table(table_type, db.totals_by_type(), comp_mod.TYPE_OPTIONS, phis)

        def export_current():
            key = key_by_label.get(cmb.currentText())
            if key is None:
                return
            label = cmb.currentText()
            path, _ = QFileDialog.getSaveFileName(dlg, "Export componentry", "", "Excel files (*.xlsx)")
            if not path:
                return
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            try:
                db = self.componentry_dbs[key]
                df_export = comp_mod.build_export_dataframe({label: db}, [label])
                with pd.ExcelWriter(path, engine="openpyxl") as w:
                    df_export.to_excel(w, index=False, header=False, sheet_name="componentry")
                QMessageBox.information(dlg, "Componentry database", f"Exported:\n{path}")
            except Exception as e:
                QMessageBox.warning(dlg, "Componentry database", str(e))

        def export_all():
            path, _ = QFileDialog.getSaveFileName(dlg, "Export componentry", "", "Excel files (*.xlsx)")
            if not path:
                return
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            try:
                sample_map = {_fw_componentry_display_label(self, k): db for k, db in self.componentry_dbs.items()}
                names = list(sample_map.keys())
                df_export = comp_mod.build_export_dataframe(sample_map, names)
                with pd.ExcelWriter(path, engine="openpyxl") as w:
                    df_export.to_excel(w, index=False, header=False, sheet_name="componentry")
                QMessageBox.information(dlg, "Componentry database", f"Exported:\n{path}")
            except Exception as e:
                QMessageBox.warning(dlg, "Componentry database", str(e))

        cmb.currentIndexChanged.connect(lambda *_: refresh_tables())
        refresh_tables()

        hb = QHBoxLayout()
        btn_exp_current = QPushButton("Export current")
        btn_exp_all = QPushButton("Export all")
        btn_close = QPushButton("Close")
        btn_exp_current.clicked.connect(export_current)
        btn_exp_all.clicked.connect(export_all)
        btn_close.clicked.connect(dlg.accept)
        hb.addWidget(btn_exp_current)
        hb.addWidget(btn_exp_all)
        hb.addStretch(1)
        hb.addWidget(btn_close)
        root.addLayout(hb)
        dlg.exec_()
    except Exception as e:
        QMessageBox.warning(self, "Componentry database", str(e))


def _fw_collect_project_dict(self):
    comp_payload = {}
    if getattr(self, "componentry_dbs", None):
        for key, db in self.componentry_dbs.items():
            db_key, sample = key
            comp_payload[_fw_componentry_key_id(db_key, sample)] = {
                "db_key": db_key,
                "sample": sample,
                "display": _fw_componentry_display_label(self, key),
                "db": db.to_dict(),
            }

    return {
        "app": "FW-Granulometry 2.0 + Componentry-FW",
        "schema_version": 2,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "fw": {
            "df_data": _fw_df_to_payload(getattr(self, "df_data", None)),
            "param_results": _fw_df_to_payload(getattr(self, "param_results", None)),
            "df_laser": _fw_df_to_payload(getattr(self, "df_laser", None)),
            "param_results_laser": _fw_df_to_payload(getattr(self, "param_results_laser", None)),
            "df_hybrid": _fw_df_to_payload(getattr(self, "df_hybrid", None)),
            "param_results_hybrid": _fw_df_to_payload(getattr(self, "param_results_hybrid", None)),
            "group_col": getattr(self, "group_col", "Group"),
            "current_method": getattr(self, "current_method", "folkward"),
            "hybrid_method": getattr(self, "hybrid_method", None),
            "laser_step": getattr(self, "laser_step", None),
            "hybrid_meta": getattr(self, "_hybrid_meta", {}),
            "group_colors": getattr(self, "group_colors", {}),
            "selected_groups_xy": getattr(self, "selected_groups_xy", []),
            "selected_groups_ribbons": getattr(self, "selected_groups_ribbons", []),
            "hist_style": getattr(self, "hist_style", None),
            "hist_current_db": self.cmb_hist_db.currentData() if hasattr(self, "cmb_hist_db") and self.cmb_hist_db.count() else None,
            "hist_current_sample": self.cmb_hist_sample.currentText() if hasattr(self, "cmb_hist_sample") and self.cmb_hist_sample.count() else None,
        },
        "componentry": {
            "samples": comp_payload,
            "sample_order": list(comp_payload.keys()),
            "view_state": dict(getattr(self, "componentry_view_state", {}) or {}),
        }
    }


def _fw_apply_project_dict(self, data):
    fw_data = data.get("fw", {}) or {}
    self.df_data = _fw_df_from_payload(fw_data.get("df_data"))
    self.param_results = _fw_df_from_payload(fw_data.get("param_results"))
    self.df_laser = _fw_df_from_payload(fw_data.get("df_laser"))
    self.param_results_laser = _fw_df_from_payload(fw_data.get("param_results_laser"))
    self.df_hybrid = _fw_df_from_payload(fw_data.get("df_hybrid"))
    self.param_results_hybrid = _fw_df_from_payload(fw_data.get("param_results_hybrid"))

    self.group_col = fw_data.get("group_col", "Group")
    self.current_method = fw_data.get("current_method", "folkward")
    self.hybrid_method = fw_data.get("hybrid_method", None)
    self.laser_step = fw_data.get("laser_step", None)
    self._hybrid_meta = fw_data.get("hybrid_meta", {}) or {}

    comp_mod = _fw_get_componentry_module(self)
    self.componentry_dbs = {}
    componentry_payload = data.get("componentry", {}) or {}
    loaded_view_state = componentry_payload.get("view_state", {}) or {}
    self.componentry_view_state = {
        "plot_style": dict(loaded_view_state.get("plot_style", {}) or {}),
        "palette_state": dict(loaded_view_state.get("palette_state", {}) or {}),
    }
    comp_data = componentry_payload.get("samples", {}) or {}
    for key_id, rec in comp_data.items():
        db_key = rec.get("db_key")
        sample = rec.get("sample")
        if db_key is None or sample is None:
            db_key, sample = _fw_componentry_key_from_id(key_id)
        try:
            self.componentry_dbs[(str(db_key), str(sample))] = comp_mod.ComponentryDB.from_dict(rec.get("db", {}))
        except Exception:
            pass

    try:
        self._update_all_groups_and_colors()
    except Exception:
        pass

    # Restore group colors/selections after regenerating keys.
    for attr, default in [
        ("group_colors", {}),
        ("selected_groups_xy", []),
        ("selected_groups_ribbons", []),
    ]:
        val = fw_data.get(attr, default)
        try:
            setattr(self, attr, val)
        except Exception:
            pass

    if fw_data.get("hist_style"):
        self.hist_style = dict(fw_data.get("hist_style") or {})

    try:
        self._update_db_menu_actions()
    except Exception:
        pass
    try:
        self.update_console()
    except Exception:
        pass
    try:
        refresh_histogram_selectors(self, preferred_db=fw_data.get("hist_current_db"), preferred_sample=fw_data.get("hist_current_sample"))
        _hist_sync_checkboxes_from_style(self)
        plot_histogram(self)
    except Exception:
        pass
    for meth in ("plot_xy",):
        try:
            getattr(self, meth)()
        except Exception:
            pass


def _fw_save_project_to_path(self, path):
    data = _fw_collect_project_dict(self)
    payload = json.dumps(data, ensure_ascii=False, default=_fw_json_default)
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        cur.execute("INSERT OR REPLACE INTO kv(k, v) VALUES(?, ?)", ("project_json", payload))
        con.commit()
    finally:
        con.close()
    self.project_path = path
    try:
        view = getattr(self, "componentry_view", None)
        if view is not None and hasattr(view, "_set_project_label"):
            view._set_project_label()
    except Exception:
        pass


def save_full_project(self):
    if not getattr(self, "project_path", None):
        return save_full_project_as(self)
    try:
        _fw_save_project_to_path(self, self.project_path)
        QMessageBox.information(self, "Project saved", f"Project saved successfully:\n{self.project_path}")
    except Exception as e:
        QMessageBox.warning(self, "Save project error", str(e))


def save_full_project_as(self):
    path, _ = QFileDialog.getSaveFileName(self, "Save full project as", "", "FW project (*.fwgproj);;Componentry project (*.componentryproj);;All files (*)")
    if not path:
        return
    if not (path.lower().endswith(".fwgproj") or path.lower().endswith(".componentryproj")):
        path += ".fwgproj"
    try:
        _fw_save_project_to_path(self, path)
        QMessageBox.information(self, "Project saved", f"Project saved successfully:\n{path}")
    except Exception as e:
        QMessageBox.warning(self, "Save project error", str(e))


def open_full_project_dialog(self):
    path, _ = QFileDialog.getOpenFileName(self, "Open full project", "", "FW/Componentry project (*.fwgproj *.componentryproj);;All files (*)")
    if not path:
        return
    try:
        con = sqlite3.connect(path)
        try:
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            cur.execute("SELECT v FROM kv WHERE k='project_json'")
            row = cur.fetchone()
        finally:
            con.close()
        if not row:
            raise ValueError("Invalid project file: project_json not found.")
        data = json.loads(row[0])
        _fw_apply_project_dict(self, data)
        self.project_path = path
        try:
            view = getattr(self, "componentry_view", None)
            if view is not None and hasattr(view, "_set_project_label"):
                view._set_project_label()
        except Exception:
            pass
        QMessageBox.information(self, "Project opened", f"Project loaded successfully:\n{path}")
    except Exception as e:
        QMessageBox.warning(self, "Open project error", str(e))



def _fw_execute_plugin_callback(self, callback):
    try:
        callback()
    except Exception:
        tb = traceback.format_exc()
        if not hasattr(self, "plugin_manager") or self.plugin_manager is None:
            QMessageBox.warning(self, "Plugin error", tb)
            return
        self.plugin_manager.plugin_errors["callback"] = tb
        QMessageBox.warning(self, "Plugin error", "The plugin function produced an error. Use Complementos > View plugin errors for details.")


def add_plugin_menu_action(self, plugin_name, action_name, callback, tooltip=None):
    if not callable(callback):
        raise ValueError("callback must be callable.")
    if not hasattr(self, "plugin_menu") or self.plugin_menu is None:
        return None
    plugin_name = str(plugin_name).strip() or "Plugin"
    action_name = str(action_name).strip() or "Plugin action"
    act = QAction(f"{plugin_name}: {action_name}", self)
    if tooltip:
        act.setToolTip(str(tooltip))
        act.setStatusTip(str(tooltip))
    act.triggered.connect(lambda _checked=False, cb=callback: self._fw_execute_plugin_callback(cb))
    self.plugin_menu.addAction(act)
    self.plugin_ui_actions.append({"plugin_name": plugin_name, "action": act})
    return act


def add_plugin_toolbar_button(self, plugin_name, text, callback, tooltip=None):
    if not callable(callback):
        raise ValueError("callback must be callable.")
    if not hasattr(self, "plugin_toolbar") or self.plugin_toolbar is None:
        return None
    plugin_name = str(plugin_name).strip() or "Plugin"
    text = str(text).strip() or "Plugin"
    act = QAction(text, self)
    if tooltip:
        act.setToolTip(str(tooltip))
        act.setStatusTip(str(tooltip))
    act.triggered.connect(lambda _checked=False, cb=callback: self._fw_execute_plugin_callback(cb))
    self.plugin_toolbar.addAction(act)
    self.plugin_ui_toolbar_actions.append({"plugin_name": plugin_name, "action": act})
    return act


def add_plugin_dock_widget(self, plugin_name, title, widget, area="right"):
    plugin_name = str(plugin_name).strip() or "Plugin"
    title = str(title).strip() or plugin_name
    dock = QDockWidget(title, self)
    dock.setObjectName(f"fw_plugin_dock_{plugin_name}_{title}".replace(" ", "_"))
    dock.setWidget(widget)
    area_map = {
        "left": Qt.LeftDockWidgetArea,
        "right": Qt.RightDockWidgetArea,
        "top": Qt.TopDockWidgetArea,
        "bottom": Qt.BottomDockWidgetArea,
    }
    qt_area = area_map.get(str(area).strip().lower(), Qt.RightDockWidgetArea)
    self.addDockWidget(qt_area, dock)
    self.plugin_dock_widgets.append({"plugin_name": plugin_name, "dock": dock})
    return dock


def add_plugin_window(self, plugin_name, window, show=True):
    plugin_name = str(plugin_name).strip() or "Plugin"
    self.plugin_windows.append({"plugin_name": plugin_name, "window": window})
    if show and hasattr(window, "show"):
        window.show()
    return window


def clear_plugin_ui(self):
    for rec in list(getattr(self, "plugin_ui_actions", [])):
        act = rec.get("action")
        try:
            if act is not None and hasattr(self, "plugin_menu"):
                self.plugin_menu.removeAction(act)
                act.deleteLater()
        except Exception:
            pass
    self.plugin_ui_actions = []

    for rec in list(getattr(self, "plugin_ui_toolbar_actions", [])):
        act = rec.get("action")
        try:
            if act is not None and hasattr(self, "plugin_toolbar"):
                self.plugin_toolbar.removeAction(act)
                act.deleteLater()
        except Exception:
            pass
    self.plugin_ui_toolbar_actions = []

    for rec in list(getattr(self, "plugin_dock_widgets", [])):
        dock = rec.get("dock")
        try:
            if dock is not None:
                self.removeDockWidget(dock)
                dock.close()
                dock.deleteLater()
        except Exception:
            pass
    self.plugin_dock_widgets = []

    for rec in list(getattr(self, "plugin_windows", [])):
        win = rec.get("window")
        try:
            if win is not None:
                win.close()
                win.deleteLater()
        except Exception:
            pass
    self.plugin_windows = []


def remove_plugin_ui(self, plugin_name):
    plugin_name = str(plugin_name).strip()
    if not plugin_name:
        return
    keep = []
    for rec in getattr(self, "plugin_ui_actions", []):
        if rec.get("plugin_name") == plugin_name:
            try:
                self.plugin_menu.removeAction(rec.get("action"))
                rec.get("action").deleteLater()
            except Exception:
                pass
        else:
            keep.append(rec)
    self.plugin_ui_actions = keep

    keep = []
    for rec in getattr(self, "plugin_ui_toolbar_actions", []):
        if rec.get("plugin_name") == plugin_name:
            try:
                self.plugin_toolbar.removeAction(rec.get("action"))
                rec.get("action").deleteLater()
            except Exception:
                pass
        else:
            keep.append(rec)
    self.plugin_ui_toolbar_actions = keep

    keep = []
    for rec in getattr(self, "plugin_dock_widgets", []):
        if rec.get("plugin_name") == plugin_name:
            try:
                self.removeDockWidget(rec.get("dock"))
                rec.get("dock").close()
                rec.get("dock").deleteLater()
            except Exception:
                pass
        else:
            keep.append(rec)
    self.plugin_dock_widgets = keep

    keep = []
    for rec in getattr(self, "plugin_windows", []):
        if rec.get("plugin_name") == plugin_name:
            try:
                rec.get("window").close()
                rec.get("window").deleteLater()
            except Exception:
                pass
        else:
            keep.append(rec)
    self.plugin_windows = keep


def get_plugin_context(self):
    db_key = self.cmb_hist_db.currentData() if hasattr(self, "cmb_hist_db") and self.cmb_hist_db.count() else None
    sample_name = self.cmb_hist_sample.currentText() if hasattr(self, "cmb_hist_sample") and self.cmb_hist_sample.count() else ""
    comp_key = (str(db_key), str(sample_name)) if db_key and sample_name else None
    return {
        "app": self,
        "fw_main_window": self,
        "hist_db_key": db_key,
        "sample_name": sample_name,
        "componentry_key": comp_key,
        "componentry_db": getattr(self, "componentry_dbs", {}).get(comp_key) if comp_key else None,
        "componentry_dbs": getattr(self, "componentry_dbs", {}),
        "project_path": getattr(self, "project_path", None),
    }


def install_plugin_dialog(self):
    try:
        if not hasattr(self, "plugin_manager") or self.plugin_manager is None:
            _fw_setup_componentry_plugins(self)
        installed_path = self.plugin_manager.install_plugin_dialog(parent=self)
        if installed_path is None:
            return
        loaded, errors = self.plugin_manager.reload_plugins()
        if errors:
            QMessageBox.warning(self, "Complementos", f"Plugin installed in:\n{installed_path}\n\nLoaded {len(loaded)} plugin(s), with {len(errors)} error(s).")
        else:
            QMessageBox.information(self, "Complementos", f"Plugin installed and loaded:\n{installed_path}")
    except Exception as e:
        QMessageBox.warning(self, "Plugin install error", str(e))


def reload_plugins_dialog(self):
    try:
        loaded, errors = self.plugin_manager.reload_plugins()
        if errors:
            QMessageBox.warning(self, "Complementos", f"Loaded {len(loaded)} plugin(s), with {len(errors)} error(s).")
        else:
            QMessageBox.information(self, "Complementos", f"Plugins loaded: {len(loaded)}")
    except Exception as e:
        QMessageBox.warning(self, "Plugin reload error", str(e))


def open_plugin_folder(self):
    try:
        self.plugin_manager.open_plugin_folder()
    except Exception as e:
        QMessageBox.warning(self, "Complementos", str(e))


def show_plugin_errors_dialog(self):
    dlg = QDialog(self)
    dlg.setWindowTitle("Plugin errors")
    dlg.resize(900, 620)
    lay = QVBoxLayout(dlg)
    txt = QTextEdit()
    txt.setReadOnly(True)
    if getattr(self.plugin_manager, "plugin_errors", None):
        body = []
        for key, value in self.plugin_manager.plugin_errors.items():
            body.append(f"===== {key} =====\n{value}")
        txt.setPlainText("\n\n".join(body))
    else:
        txt.setPlainText("No plugin errors registered.")
    lay.addWidget(txt, 1)
    hb = QHBoxLayout()
    hb.addStretch(1)
    btn_close = QPushButton("Close")
    btn_close.clicked.connect(dlg.accept)
    hb.addWidget(btn_close)
    lay.addLayout(hb)
    dlg.exec_()


def _fw_setup_componentry_plugins(self):
    """
    Disabled by design.

    Componentry plugins must be managed and displayed only by the
    ComponentryPreview window. FW-Granulometry keeps the Componentry data
    integration, but it does not create a main-window plugin menu/toolbar and
    it does not load plugins into the FW main window.
    """
    self.plugin_manager = None
    self.plugin_menu = None
    self.plugin_toolbar = None
    self.plugin_ui_actions = []
    self.plugin_ui_toolbar_actions = []
    self.plugin_dock_widgets = []
    self.plugin_windows = []
    return None

def _fw_add_project_and_componentry_menus(self):
    # Persistent state for full project and componentry.
    if not hasattr(self, "project_path"):
        self.project_path = None
    if not hasattr(self, "componentry_dbs"):
        self.componentry_dbs = {}
    if not hasattr(self, "_componentry_module"):
        self._componentry_module = None
    if not hasattr(self, "componentry_view_state"):
        self.componentry_view_state = {
            "plot_style": {},
            "palette_state": {},
        }

    try:
        self.menu_file.insertSeparator(self.act_load_tamiz)
        self.act_open_project = QAction("Open full project", self)
        self.act_open_project.triggered.connect(self.open_full_project_dialog)
        self.menu_file.insertAction(self.act_load_tamiz, self.act_open_project)

        self.act_save_project = QAction("Save full project", self)
        self.act_save_project.triggered.connect(self.save_full_project)
        self.menu_file.insertAction(self.act_load_tamiz, self.act_save_project)

        self.act_save_project_as = QAction("Save full project as...", self)
        self.act_save_project_as.triggered.connect(self.save_full_project_as)
        self.menu_file.insertAction(self.act_load_tamiz, self.act_save_project_as)
    except Exception:
        pass

    try:
        self.act_viewdb_componentry = QAction("View componentry database", self)
        self.act_viewdb_componentry.setEnabled(bool(self.componentry_dbs))
        self.act_viewdb_componentry.triggered.connect(self.show_componentry_db_window)
        self.menu_db.addAction(self.act_viewdb_componentry)
    except Exception:
        pass


def _fw_update_db_menu_actions_patched(self):
    try:
        _fw_old_update_db_menu_actions(self)
    except Exception:
        pass
    try:
        if hasattr(self, "act_viewdb_componentry"):
            self.act_viewdb_componentry.setEnabled(bool(getattr(self, "componentry_dbs", {})))
    except Exception:
        pass


def _fw_post_init(self):
    try:
        _fw_add_project_and_componentry_menus(self)
    except Exception:
        pass

    # IMPORTANT:
    # Componentry plugins belong only to the ComponentryPreview window.
    # Do NOT initialize the Componentry plugin manager in the FW-Granulometry
    # main window; otherwise plugins such as GDEINPUT appear in the wrong UI.
    self.plugin_manager = None
    self.plugin_menu = None
    self.plugin_toolbar = None
    self.plugin_ui_actions = []
    self.plugin_ui_toolbar_actions = []
    self.plugin_dock_widgets = []
    self.plugin_windows = []

    try:
        self._update_db_menu_actions()
    except Exception:
        pass


# Bind integration methods.
MainWindow._fw_get_componentry_module = _fw_get_componentry_module
MainWindow._fw_componentry_after_change = _fw_componentry_after_change
MainWindow.show_componentry_db_window = show_componentry_db_window
MainWindow.save_full_project = save_full_project
MainWindow.save_full_project_as = save_full_project_as
MainWindow.open_full_project_dialog = open_full_project_dialog

MainWindow._fw_execute_plugin_callback = _fw_execute_plugin_callback
MainWindow.add_plugin_menu_action = add_plugin_menu_action
MainWindow.add_plugin_toolbar_button = add_plugin_toolbar_button
MainWindow.add_plugin_dock_widget = add_plugin_dock_widget
MainWindow.add_plugin_window = add_plugin_window
MainWindow.clear_plugin_ui = clear_plugin_ui
MainWindow.remove_plugin_ui = remove_plugin_ui
MainWindow.get_plugin_context = get_plugin_context
MainWindow.install_plugin_dialog = install_plugin_dialog
MainWindow.reload_plugins_dialog = reload_plugins_dialog
MainWindow.open_plugin_folder = open_plugin_folder
MainWindow.show_plugin_errors_dialog = show_plugin_errors_dialog

# Patch MainWindow.__init__ after all original UI patches are in place.
_fw_old_mainwindow_init = MainWindow.__init__
def _fw_integrated_mainwindow_init(self, *args, **kwargs):
    _fw_old_mainwindow_init(self, *args, **kwargs)
    _fw_post_init(self)
MainWindow.__init__ = _fw_integrated_mainwindow_init

# Patch DB action updater to include componentry.
_fw_old_update_db_menu_actions = MainWindow._update_db_menu_actions
MainWindow._update_db_menu_actions = _fw_update_db_menu_actions_patched


# =========================================================
# Embedded Graphic vector-tab support
# Graphic is the only classification-diagram tab in the main interface.
# =========================================================

# -*- coding: utf-8 -*-
"""
Graphic.py

Vector/graphic classification tab for FW-Granulometry.

Graphic contains the vector classification diagrams used by the application.
Field-specific line, fill and label editing is handled here; vector geometries
are loaded from GraphicVectors.py.
"""


from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Tuple, Any
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba

# Avoid terminal flooding when a requested font is not installed.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFontDatabase
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QColorDialog, QMessageBox,
    QGroupBox, QScrollArea, QSizePolicy
)


# =========================================================
# Vector diagram definitions
# Coordinate system: X = mean (phi), Y = sorting/sigma (phi)
# =========================================================

# Vector geometry lives in an external script.
# Local canonical filename: GraphicVectors.py
# Download/versioned fallback accepted: GraphicVectors_Pardo_Walker_v15.py
# The loader validates that BOTH GK and Pardo are present, so an older GK-only
# GraphicVectors.py cannot silently disable the Pardo selector.
import importlib.util as _fw_importlib_util

_VECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
_VECTOR_CANDIDATES = ["GraphicVectors.py", "GraphicVectors_Pardo_Walker_v15.py"]
_vector_mod = None
_vector_errors = []

for _candidate in _VECTOR_CANDIDATES:
    _candidate_path = os.path.join(_VECTOR_DIR, _candidate)
    if not os.path.isfile(_candidate_path):
        continue
    try:
        _spec_name = "_fw_graphic_vectors_" + _candidate.replace(".", "_").replace("-", "_")
        _vector_spec = _fw_importlib_util.spec_from_file_location(_spec_name, _candidate_path)
        if _vector_spec is None or _vector_spec.loader is None:
            raise ImportError("Could not create module spec")
        _candidate_mod = _fw_importlib_util.module_from_spec(_vector_spec)
        sys.modules[_vector_spec.name] = _candidate_mod
        _vector_spec.loader.exec_module(_candidate_mod)
        _diagrams = getattr(_candidate_mod, "VECTOR_DIAGRAMS", {})
        if not isinstance(_diagrams, dict) or "gk" not in _diagrams or "pardo" not in _diagrams:
            raise ImportError("vector file does not contain both supported classification diagrams")
        _vector_mod = _candidate_mod
        _VECTOR_MODULE_PATH = _candidate_path
        break
    except Exception as _exc:
        _vector_errors.append(f"{_candidate}: {_exc}")

if _vector_mod is None:
    raise FileNotFoundError(
        "A valid vector file was not found next to the program.\n"
        "Expected GraphicVectors.py (canonical) or GraphicVectors_Pardo_Walker_v15.py (download name), "
        "containing both supported classification diagrams.\n" + "\n".join(_vector_errors)
    )

ZoneDef = _vector_mod.ZoneDef
VECTOR_DIAGRAMS = _vector_mod.VECTOR_DIAGRAMS
DEFAULT_DIAGRAM_KEY = _vector_mod.DEFAULT_DIAGRAM_KEY

LINESTYLE_OPTIONS = {
    "Auto": None,
    "Solid": "-",
    "Dashed": "--",
    "Dotted": ":",
    "Dash-dot": "-.",
}

DEFAULT_AXIS_LIMITS = (-3.5, 5.5, 0.0, 4.2)


class BoundedGraphicCanvas(FigureCanvas):
    """
    Stable Matplotlib canvas for the Graphic tab.

    v9 design:
    - wheel zoom changes only the visual canvas/figure size in the QScrollArea;
    - it does not rebuild the plot and does not change scientific coordinates;
    - middle-button drag pans the QScrollArea viewport;
    - interactive border/corner resize was removed because it was unstable.
    """

    def __init__(self, width_in=8.8, height_in=5.4, dpi=100):
        fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
        super().__init__(fig)
        self._dpi = int(dpi)
        self._base_w_px = int(round(width_in * dpi))
        self._base_h_px = int(round(height_in * dpi))
        self._zoom = 1.0
        self._min_w_px = 500
        self._min_h_px = 340
        self._max_w_px = 2200
        self._max_h_px = 1600
        self._scroll_area = None
        self._panning = False
        self._pan_start_global = None
        self._pan_start_h = 0
        self._pan_start_v = 0
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: white;")
        self._apply_sizes(redraw=False)

    def set_scroll_area(self, scroll_area):
        self._scroll_area = scroll_area

    def set_resize_callback(self, callback):
        # Kept for compatibility with older v8 binding. No interactive resize in v9.
        self._resize_callback = callback

    def set_edit_layout(self, enabled: bool):
        # Disabled in v9. Numeric width/height controls remain available.
        self.unsetCursor()

    def set_base_size_px(self, width_px, height_px, redraw=True):
        self._base_w_px = int(max(self._min_w_px, min(self._max_w_px, round(float(width_px)))))
        self._base_h_px = int(max(self._min_h_px, min(self._max_h_px, round(float(height_px)))))
        self._apply_sizes(redraw=redraw)

    def base_size_px(self):
        return int(self._base_w_px), int(self._base_h_px)

    def set_zoom(self, zoom, redraw=True):
        self._zoom = float(max(0.35, min(3.5, float(zoom))))
        self._apply_sizes(redraw=redraw)

    def zoom(self):
        return float(self._zoom)

    def reset_zoom(self):
        self.set_zoom(1.0, redraw=True)

    def _current_size_px(self):
        w = int(round(self._base_w_px * self._zoom))
        h = int(round(self._base_h_px * self._zoom))
        return max(1, w), max(1, h)

    def _apply_sizes(self, redraw=True):
        w, h = self._current_size_px()
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)
        self.setFixedSize(w, h)
        try:
            # Figure size follows the visible canvas. Text, markers and lines remain
            # point-based, so they do not inflate like raster pixels.
            self.figure.set_size_inches(float(w) / self._dpi, float(h) / self._dpi, forward=False)
            self.figure.set_dpi(self._dpi)
        except Exception:
            pass
        if redraw:
            try:
                self.draw_idle()
            except Exception:
                pass

    def wheelEvent(self, event):
        try:
            delta = event.angleDelta().y()
        except Exception:
            delta = 0
        if delta == 0:
            return super().wheelEvent(event)

        old_zoom = float(self._zoom)
        factor = 1.12 if delta > 0 else (1.0 / 1.12)
        new_zoom = max(0.35, min(3.5, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-9:
            event.accept()
            return

        scroll = self._scroll_area
        try:
            cursor_x = float(event.pos().x())
            cursor_y = float(event.pos().y())
        except Exception:
            cursor_x = float(self.width()) * 0.5
            cursor_y = float(self.height()) * 0.5

        old_h_value = 0
        old_v_value = 0
        if scroll is not None:
            try:
                old_h_value = int(scroll.horizontalScrollBar().value())
                old_v_value = int(scroll.verticalScrollBar().value())
            except Exception:
                pass

        content_x = old_h_value + cursor_x
        content_y = old_v_value + cursor_y
        scale = new_zoom / old_zoom

        self.set_zoom(new_zoom, redraw=True)

        if scroll is not None:
            try:
                hs = scroll.horizontalScrollBar()
                vs = scroll.verticalScrollBar()
                hs.setValue(int(round(content_x * scale - cursor_x)))
                vs.setValue(int(round(content_y * scale - cursor_y)))
            except Exception:
                pass

        event.accept()

    def mousePressEvent(self, event):
        try:
            button = event.button()
        except Exception:
            button = None

        if button == Qt.MiddleButton:
            self._panning = True
            self._pan_start_global = event.globalPos()
            if self._scroll_area is not None:
                self._pan_start_h = self._scroll_area.horizontalScrollBar().value()
                self._pan_start_v = self._scroll_area.verticalScrollBar().value()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning and self._scroll_area is not None and self._pan_start_global is not None:
            delta = event.globalPos() - self._pan_start_global
            self._scroll_area.horizontalScrollBar().setValue(self._pan_start_h - delta.x())
            self._scroll_area.verticalScrollBar().setValue(self._pan_start_v - delta.y())
            event.accept()
            return
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            button = event.button()
        except Exception:
            button = None
        if button == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_start_global = None
            self.unsetCursor()
            event.accept()
            return
        return super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if not self._panning:
            self.unsetCursor()
        return super().leaveEvent(event)


# =========================================================
# Style helpers
# =========================================================

CUSTOM_FIELD_VIEW_NAME = "Custom selection"


def _diagram_meta(diagram_key: str = None) -> Dict[str, Any]:
    key = str(diagram_key or DEFAULT_DIAGRAM_KEY)
    return VECTOR_DIAGRAMS.get(key, VECTOR_DIAGRAMS[DEFAULT_DIAGRAM_KEY])


def _diagram_keys() -> List[str]:
    return list(VECTOR_DIAGRAMS.keys())


def _diagram_display_name(diagram_key: str) -> str:
    meta = _diagram_meta(diagram_key)
    return str(meta.get("title", diagram_key))


def _diagram_view_options(diagram_key: str) -> List[str]:
    meta = _diagram_meta(diagram_key)
    base = list(meta.get("view_options", []))
    if CUSTOM_FIELD_VIEW_NAME not in base:
        base.insert(1 if "All" in base else 0, CUSTOM_FIELD_VIEW_NAME)
    return base


def _all_zone_defs() -> List[ZoneDef]:
    out: List[ZoneDef] = []
    seen = set()
    for meta in VECTOR_DIAGRAMS.values():
        for zone in meta.get("zones", []):
            if zone.name not in seen:
                seen.add(zone.name)
                out.append(zone)
    return out


def _zone_defaults(diagram_key: str = None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    zones = _all_zone_defs() if diagram_key is None else list(_diagram_meta(diagram_key).get("zones", []))
    for zone in zones:
        out[zone.name] = {
            "edgecolor": zone.edgecolor,
            "facecolor": zone.facecolor,
            "alpha": float(zone.alpha),
            "linewidth": float(zone.linewidth),
            "linestyle_name": "Auto",
            "label_size": int(zone.label_size),
            "label_font": _safe_font_name(zone.label_font),
            "label_x": float(zone.label_pos[0]),
            "label_y": float(zone.label_pos[1]),
        }
    return out


def _field_style(zone_styles: Dict[str, Dict[str, Any]], zone: ZoneDef) -> Dict[str, Any]:
    defaults = _zone_defaults().get(zone.name, {})
    style = dict(defaults)
    style.update(zone_styles.get(zone.name, {}) or {})
    return style


def _style_linestyle(style: Dict[str, Any], zone: ZoneDef):
    name = str(style.get("linestyle_name", "Auto"))
    override = LINESTYLE_OPTIONS.get(name, None)
    return zone.linestyle if override is None else override


def _display_label(name: str) -> str:
    replacements = {
        "Fine-depleted flow": "Fine-depleted\nflow",
        "Surge Dunes": "Surge\nDunes",
        "Pyroclastic surges": "Pyroclastic\nsurges",
        "Pyroclastic flow": "Pyroclastic\nflow",
        "Zone I": "I",
        "Zone II": "II",
        "Zone III": "III",
    }
    return replacements.get(name, name)


def _font_exists(font_name: str) -> bool:
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
        return True
    except Exception:
        return False


def _safe_font_name(font_name: str) -> str:
    name = str(font_name or "").strip()
    if name and _font_exists(name):
        return name
    return "DejaVu Sans"


def _math_font_candidates() -> List[str]:
    preferred = [
        "DejaVu Sans",
        "DejaVu Serif",
        "DejaVu Sans Mono",
        "STIXGeneral",
        "STIXNonUnicode",
        "STIXSizeOneSym",
        "STIXSizeTwoSym",
        "STIXSizeThreeSym",
        "STIXSizeFourSym",
        "DejaVu Math TeX Gyre",
        "Computer Modern Roman",
        "CMU Serif",
        "Cambria Math",
        "Arial",
        "Times New Roman",
    ]
    out = []
    for f in preferred:
        if f not in out and _font_exists(f):
            out.append(f)
    try:
        installed = set(QFontDatabase().families())
        math_like = sorted(
            n for n in installed
            if any(t in n.lower() for t in ["math", "stix", "cmu", "dejavu", "times", "cambria", "arial"])
        )
        for n in math_like:
            if n not in out and _font_exists(n):
                out.append(n)
    except Exception:
        pass
    if not out:
        out = ["DejaVu Sans"]
    return out


def _chaikin_closed(points, iterations=3):
    pts = np.array(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts
    for _ in range(max(0, int(iterations))):
        out = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            out.extend([0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1])
        pts = np.array(out, dtype=float)
    return pts


def _chaikin_open(points, iterations=3):
    pts = np.array(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts
    for _ in range(max(0, int(iterations))):
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            out.extend([0.75 * p0 + 0.25 * p1, 0.25 * p0 + 0.75 * p1])
        out.append(pts[-1])
        pts = np.array(out, dtype=float)
    return pts


def draw_vector_diagram(
    ax,
    diagram_key: str,
    view: str,
    zone_styles: Dict[str, Dict[str, Any]],
    show_labels: bool,
    selected_fields: List[str] = None,
):
    meta = _diagram_meta(diagram_key)
    library_map = meta.get("library", {}) or {}
    base_view = "All" if str(view) == CUSTOM_FIELD_VIEW_NAME else str(view)
    library = library_map.get(base_view, library_map.get("All", {}))

    zones = list(library.get("zones", []))
    lines = list(library.get("extra_lines", []))
    smooth_iterations = int(meta.get("smooth_iterations", 3))
    label_overrides = dict(meta.get("label_overrides", {}) or {})
    label_annotations = dict(meta.get("label_annotations", {}) or {})

    if selected_fields:
        selected = set(selected_fields)
        zones = [z for z in zones if z.name in selected]
        lines = [ln for ln in lines if str(ln.get("zone_name", "")) in selected or not str(ln.get("zone_name", ""))]

    for zone in zones:
        style = _field_style(zone_styles, zone)
        smooth_vertices = _chaikin_closed(zone.vertices, iterations=smooth_iterations)
        fill_alpha = float(style.get("alpha", zone.alpha))

        # GK keeps the original appearance exactly as it was before the
        # Pardo/Walker visibility work: alpha applies to the whole patch,
        # including its outline. Pardo and Walker use opaque outlines plus
        # translucent gray fills so their fields remain clearly readable.
        if str(diagram_key).lower() == "gk":
            poly = Polygon(
                smooth_vertices,
                closed=True,
                facecolor=style.get("facecolor", zone.facecolor),
                edgecolor=style.get("edgecolor", zone.edgecolor),
                alpha=fill_alpha,
                linewidth=float(style.get("linewidth", zone.linewidth)),
                linestyle=_style_linestyle(style, zone),
                hatch=zone.hatch,
                zorder=zone.zorder,
                joinstyle="round",
                capstyle="round",
                antialiased=True,
            )
        else:
            face_rgba = to_rgba(style.get("facecolor", zone.facecolor), fill_alpha)
            edge_rgba = to_rgba(style.get("edgecolor", zone.edgecolor), 1.0)
            poly = Polygon(
                smooth_vertices,
                closed=True,
                facecolor=face_rgba,
                edgecolor=edge_rgba,
                alpha=None,
                linewidth=float(style.get("linewidth", zone.linewidth)),
                linestyle=_style_linestyle(style, zone),
                hatch=zone.hatch,
                zorder=zone.zorder,
                joinstyle="round",
                capstyle="round",
                antialiased=True,
            )
        ax.add_patch(poly)
        if show_labels:
            label_text = str(label_overrides.get(zone.name, _display_label(zone.name)))
            label_x = float(style.get("label_x", zone.label_pos[0]))
            label_y = float(style.get("label_y", zone.label_pos[1]))
            ann = label_annotations.get(zone.name)
            if isinstance(ann, dict) and ann.get("target") is not None:
                ax.annotate(
                    label_text,
                    xy=tuple(ann.get("target")),
                    xytext=(label_x, label_y),
                    textcoords="data",
                    fontsize=int(style.get("label_size", zone.label_size)),
                    fontfamily=_safe_font_name(str(style.get("label_font", zone.label_font))),
                    ha=str(ann.get("ha", "center")),
                    va=str(ann.get("va", "center")),
                    color="#333333",
                    arrowprops={
                        "arrowstyle": str(ann.get("arrowstyle", "->")),
                        "color": str(ann.get("arrow_color", "#222222")),
                        "linewidth": float(ann.get("arrow_linewidth", 0.9)),
                        "shrinkA": 4,
                        "shrinkB": 2,
                    },
                    zorder=max(12, zone.zorder + 7),
                    annotation_clip=False,
                )
            else:
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    fontsize=int(style.get("label_size", zone.label_size)),
                    fontfamily=_safe_font_name(str(style.get("label_font", zone.label_font))),
                    ha="center",
                    va="center",
                    color="#444444",
                    zorder=max(10, zone.zorder + 5),
                )

    for line in lines:
        line_iterations = int(line.get("smooth_iterations", smooth_iterations))
        xy = _chaikin_open(line["xy"], iterations=line_iterations)
        zone_name = line.get("zone_name", "")
        style = dict(zone_styles.get(zone_name, {}) or {})
        line_style_name = str(style.get("linestyle_name", "Auto"))
        line_style_override = LINESTYLE_OPTIONS.get(line_style_name, None)
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=style.get("edgecolor", line.get("color", "#111111")),
            linewidth=float(style.get("linewidth", line.get("linewidth", 1.0))),
            linestyle=line.get("linestyle", "-") if line_style_override is None else line_style_override,
            zorder=line.get("zorder", 4),
            solid_joinstyle="round",
            solid_capstyle="round",
            dash_joinstyle="round",
            dash_capstyle="round",
            antialiased=True,
        )

    if show_labels:
        for rec in list(meta.get("fixed_texts", []) or []):
            try:
                x, y = rec.get("xy", (0.0, 0.0))
                ax.text(
                    float(x),
                    float(y),
                    str(rec.get("text", "")),
                    fontsize=float(rec.get("fontsize", 10)),
                    fontfamily=_safe_font_name(str(rec.get("fontfamily", "DejaVu Sans"))),
                    fontweight=str(rec.get("fontweight", "normal")),
                    ha=str(rec.get("ha", "center")),
                    va=str(rec.get("va", "center")),
                    color=str(rec.get("color", "#222222")),
                    zorder=int(rec.get("zorder", 20)),
                    clip_on=False,
                )
            except Exception:
                pass


# =========================================================
# MainWindow integration / monkey patch
# =========================================================

def install_graphic_support(MainWindow):
    if getattr(MainWindow, "_graphic_support_installed", False):
        return

    old_initUI_tabs = MainWindow.initUI_tabs
    old_save_current_tab = getattr(MainWindow, "save_current_tab", None)
    old_refresh_after_settings_change = getattr(MainWindow, "_refresh_after_settings_change", None)
    old_selected_group_keys_for_tab = getattr(MainWindow, "_selected_group_keys_for_tab", None)
    old_update_all_groups_and_colors = getattr(MainWindow, "_update_all_groups_and_colors", None)

    def _graphic_ensure_state(self):
        if not hasattr(self, "graphic_style") or not isinstance(self.graphic_style, dict):
            self.graphic_style = {
                "diagram_key": DEFAULT_DIAGRAM_KEY,
                "view": "All",
                "show_labels": True,
                "show_sample_names": False,
            }
        self.graphic_style.setdefault("diagram_key", DEFAULT_DIAGRAM_KEY)
        self.graphic_style.setdefault("view", "All")
        self.graphic_style.setdefault("show_labels", True)
        self.graphic_style.setdefault("show_sample_names", False)
        self.graphic_style.setdefault("layout_width_px", 920)
        self.graphic_style.setdefault("layout_height_px", 600)
        self.graphic_style.setdefault("layout_zoom", 1.0)
        self.graphic_style.setdefault("edit_layout", False)

        if not hasattr(self, "graphic_visible_fields") or not isinstance(self.graphic_visible_fields, dict):
            self.graphic_visible_fields = {}
        for dkey, meta in VECTOR_DIAGRAMS.items():
            opts = list(meta.get("field_options", []))
            picked = self.graphic_visible_fields.get(dkey, list(opts))
            picked = [x for x in picked if x in opts]
            if not picked:
                picked = list(opts)
            self.graphic_visible_fields[dkey] = picked

        _graphic_style_version = "vector_style_v16_ternary"
        defaults = _zone_defaults()
        if (not hasattr(self, "graphic_zone_styles") or not isinstance(self.graphic_zone_styles, dict)
                or self.graphic_style.get("_graphic_style_version") != _graphic_style_version):
            self.graphic_zone_styles = _zone_defaults()
            self.graphic_style["_graphic_style_version"] = _graphic_style_version
        else:
            for k, v in defaults.items():
                self.graphic_zone_styles.setdefault(k, dict(v))
                for kk, vv in v.items():
                    self.graphic_zone_styles[k].setdefault(kk, vv)

        if not hasattr(self, "selected_groups_graphic") or self.selected_groups_graphic is None:
            try:
                self.selected_groups_graphic = list(self._safe_group_keys())
            except Exception:
                self.selected_groups_graphic = []

    def _graphic_current_diagram_key(self) -> str:
        try:
            key = str(self.cmb_graphic_diagram.currentData() or self.cmb_graphic_diagram.currentText()).strip().lower()
        except Exception:
            key = str(self.graphic_style.get("diagram_key", DEFAULT_DIAGRAM_KEY)).strip().lower()
        if key not in VECTOR_DIAGRAMS:
            key = DEFAULT_DIAGRAM_KEY
        return key

    def _graphic_current_diagram_meta(self) -> Dict[str, Any]:
        return _diagram_meta(self.graphic_current_diagram_key())

    def _graphic_current_field_options(self) -> List[str]:
        return list(self.graphic_current_diagram_meta().get("field_options", []))

    def _graphic_visible_fields_summary(self, diagram_key: str = None) -> str:
        dkey = str(diagram_key or self.graphic_current_diagram_key())
        meta = _diagram_meta(dkey)
        opts = list(meta.get("field_options", []))
        picked = list(self.graphic_visible_fields.get(dkey, opts))
        if len(picked) == len(opts):
            return "All fields visible"
        if not picked:
            return "No fields selected"
        return f"Visible: {', '.join(picked)}"

    def _graphic_set_diagram(self, diagram_key: str):
        key = str(diagram_key or "").strip().lower()
        if key not in VECTOR_DIAGRAMS:
            return
        idx = self.cmb_graphic_diagram.findData(key)
        if idx >= 0:
            self.cmb_graphic_diagram.setCurrentIndex(idx)
        else:
            self.graphic_style["diagram_key"] = key
            self.graphic_on_diagram_changed()

    def _graphic_selector_changed(self, *args):
        try:
            key = str(self.cmb_graphic_selector.currentData() or self.cmb_graphic_selector.currentText()).strip().lower()
        except Exception:
            key = DEFAULT_DIAGRAM_KEY
        self.graphic_set_diagram(key)

    def _graphic_sync_diagram_controls(self, preserve_plot=True):
        self._graphic_ensure_state()
        meta = self.graphic_current_diagram_meta()
        diagram_key = self.graphic_current_diagram_key()

        view_options = _diagram_view_options(diagram_key)
        current_view = str(self.graphic_style.get("view", "All"))
        if current_view not in view_options:
            current_view = "All"
            self.graphic_style["view"] = current_view

        old = self.cmb_graphic_view.blockSignals(True)
        try:
            self.cmb_graphic_view.clear()
            self.cmb_graphic_view.addItems(view_options)
            self.cmb_graphic_view.setCurrentText(current_view)
        finally:
            self.cmb_graphic_view.blockSignals(old)

        field_options = list(meta.get("field_options", []))
        current_field = None
        try:
            current_field = str(self.cmb_graphic_edit_field.currentText())
        except Exception:
            current_field = None
        if current_field not in field_options:
            current_field = field_options[0] if field_options else ""
        old = self.cmb_graphic_edit_field.blockSignals(True)
        try:
            self.cmb_graphic_edit_field.clear()
            self.cmb_graphic_edit_field.addItems(field_options)
            if current_field:
                self.cmb_graphic_edit_field.setCurrentText(current_field)
        finally:
            self.cmb_graphic_edit_field.blockSignals(old)

        if hasattr(self, "lbl_graphic_visible_fields"):
            self.lbl_graphic_visible_fields.setText(self.graphic_visible_fields_summary(diagram_key))
        if hasattr(self, "cmb_graphic_selector"):
            old_sel = self.cmb_graphic_selector.blockSignals(True)
            try:
                idx = self.cmb_graphic_selector.findData(diagram_key)
                if idx >= 0:
                    self.cmb_graphic_selector.setCurrentIndex(idx)
            finally:
                self.cmb_graphic_selector.blockSignals(old_sel)

        self.graphic_load_selected_field_controls(redraw=preserve_plot)

    def _graphic_initUI_tabs(self):
        old_initUI_tabs(self)
        if hasattr(self, "tab_graphic"):
            return

        self._graphic_ensure_state()

        self.tab_graphic = QWidget()
        root = QHBoxLayout(self.tab_graphic)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        left_panel = QWidget()
        left_panel.setMinimumWidth(390)
        left_panel.setMaximumWidth(480)
        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        box_selector = QGroupBox("Graphic selector")
        grid_selector = QGridLayout(box_selector)
        grid_selector.setContentsMargins(8, 8, 8, 8)
        grid_selector.setHorizontalSpacing(6)
        grid_selector.setVerticalSpacing(6)

        self.cmb_graphic_diagram = QComboBox()
        for dkey in _diagram_keys():
            self.cmb_graphic_diagram.addItem(_diagram_display_name(dkey), dkey)
        initial_key = str(self.graphic_style.get("diagram_key", DEFAULT_DIAGRAM_KEY))
        idx = self.cmb_graphic_diagram.findData(initial_key)
        if idx < 0:
            idx = 0
        self.cmb_graphic_diagram.setCurrentIndex(idx)
        self.cmb_graphic_diagram.currentIndexChanged.connect(self.graphic_on_diagram_changed)

        self.cmb_graphic_selector = QComboBox()
        for dkey in _diagram_keys():
            self.cmb_graphic_selector.addItem(_diagram_display_name(dkey), dkey)
        idx2 = self.cmb_graphic_selector.findData(initial_key)
        if idx2 < 0:
            idx2 = 0
        self.cmb_graphic_selector.setCurrentIndex(idx2)
        self.cmb_graphic_selector.currentIndexChanged.connect(self.graphic_selector_changed)
        self.cmb_graphic_selector.setToolTip("Choose the diagram shown in Graphic")

        grid_selector.addWidget(QLabel("Graphic:"), 0, 0)
        grid_selector.addWidget(self.cmb_graphic_selector, 0, 1)
        left_layout.addWidget(box_selector)

        box_data = QGroupBox("Data and points")
        grid_data = QGridLayout(box_data)
        grid_data.setContentsMargins(8, 8, 8, 8)
        grid_data.setHorizontalSpacing(6)
        grid_data.setVerticalSpacing(6)

        self.btn_graphic_choose_db = QPushButton("Choose databases")
        self.btn_graphic_choose_db.clicked.connect(self.graphic_choose_databases)
        grid_data.addWidget(self.btn_graphic_choose_db, 0, 0)

        self.btn_graphic_filter_groups = QPushButton("Filter groups")
        self.btn_graphic_filter_groups.clicked.connect(self.select_groups_graphic)
        grid_data.addWidget(self.btn_graphic_filter_groups, 0, 1)

        self.btn_graphic_point_colors = QPushButton("Point/group colors")
        self.btn_graphic_point_colors.clicked.connect(self.graphic_edit_point_colors)
        grid_data.addWidget(self.btn_graphic_point_colors, 1, 0)

        self.chk_graphic_sample_names = QCheckBox("Show sample names")
        self.chk_graphic_sample_names.setChecked(bool(self.graphic_style.get("show_sample_names", False)))
        self.chk_graphic_sample_names.toggled.connect(self.plot_graphic)
        grid_data.addWidget(self.chk_graphic_sample_names, 1, 1)
        left_layout.addWidget(box_data)

        box_display = QGroupBox("Graphic display")
        grid_display = QGridLayout(box_display)
        grid_display.setContentsMargins(8, 8, 8, 8)
        grid_display.setHorizontalSpacing(6)
        grid_display.setVerticalSpacing(6)

        self.cmb_graphic_view = QComboBox()
        self.cmb_graphic_view.currentIndexChanged.connect(self.plot_graphic)

        self.btn_graphic_visible_fields = QPushButton("Visible fields...")
        self.btn_graphic_visible_fields.clicked.connect(self.graphic_select_visible_fields)

        self.btn_graphic_all_fields = QPushButton("All fields")
        self.btn_graphic_all_fields.clicked.connect(self.graphic_reset_visible_fields)

        self.lbl_graphic_visible_fields = QLabel("")
        self.lbl_graphic_visible_fields.setWordWrap(True)
        self.lbl_graphic_visible_fields.setStyleSheet("color:#555555;")

        self.cmb_graphic_edit_field = QComboBox()
        self.cmb_graphic_edit_field.currentIndexChanged.connect(self.graphic_load_selected_field_controls)

        self.chk_graphic_labels = QCheckBox("Show labels")
        self.chk_graphic_labels.setChecked(bool(self.graphic_style.get("show_labels", True)))
        self.chk_graphic_labels.toggled.connect(self.plot_graphic)

        grid_display.addWidget(QLabel("View:"), 0, 0)
        grid_display.addWidget(self.cmb_graphic_view, 0, 1)
        grid_display.addWidget(self.btn_graphic_visible_fields, 1, 0)
        grid_display.addWidget(self.btn_graphic_all_fields, 1, 1)
        grid_display.addWidget(self.lbl_graphic_visible_fields, 2, 0, 1, 2)
        grid_display.addWidget(QLabel("Edit field:"), 3, 0)
        grid_display.addWidget(self.cmb_graphic_edit_field, 3, 1)
        grid_display.addWidget(self.chk_graphic_labels, 4, 0, 1, 2)
        left_layout.addWidget(box_display)

        box_field = QGroupBox("Selected field style")
        grid_field = QGridLayout(box_field)
        grid_field.setContentsMargins(8, 8, 8, 8)
        grid_field.setHorizontalSpacing(6)
        grid_field.setVerticalSpacing(6)

        self.cmb_graphic_linestyle = QComboBox()
        self.cmb_graphic_linestyle.addItems(list(LINESTYLE_OPTIONS.keys()))
        self.cmb_graphic_linestyle.currentIndexChanged.connect(self.graphic_apply_selected_field_controls)

        self.spn_graphic_linewidth = QDoubleSpinBox()
        self.spn_graphic_linewidth.setRange(0.40, 5.00)
        self.spn_graphic_linewidth.setSingleStep(0.10)
        self.spn_graphic_linewidth.setDecimals(2)
        self.spn_graphic_linewidth.valueChanged.connect(self.graphic_apply_selected_field_controls)

        self.btn_graphic_line_color = QPushButton("Line color")
        self.btn_graphic_line_color.clicked.connect(lambda: self.graphic_pick_color("edgecolor"))

        self.btn_graphic_fill_color = QPushButton("Fill color")
        self.btn_graphic_fill_color.clicked.connect(lambda: self.graphic_pick_color("facecolor"))

        self.spn_graphic_alpha = QDoubleSpinBox()
        self.spn_graphic_alpha.setRange(0.0, 1.0)
        self.spn_graphic_alpha.setSingleStep(0.1)
        self.spn_graphic_alpha.setDecimals(2)
        self.spn_graphic_alpha.valueChanged.connect(self.graphic_apply_selected_field_controls)

        grid_field.addWidget(QLabel("Line style:"), 0, 0)
        grid_field.addWidget(self.cmb_graphic_linestyle, 0, 1)
        grid_field.addWidget(QLabel("Line width:"), 1, 0)
        grid_field.addWidget(self.spn_graphic_linewidth, 1, 1)
        grid_field.addWidget(QLabel("Fill alpha:"), 2, 0)
        grid_field.addWidget(self.spn_graphic_alpha, 2, 1)
        grid_field.addWidget(self.btn_graphic_line_color, 3, 0)
        grid_field.addWidget(self.btn_graphic_fill_color, 3, 1)
        left_layout.addWidget(box_field)

        box_label = QGroupBox("Selected field label")
        grid_label = QGridLayout(box_label)
        grid_label.setContentsMargins(8, 8, 8, 8)
        grid_label.setHorizontalSpacing(6)
        grid_label.setVerticalSpacing(6)

        self.spn_graphic_labelsize = QSpinBox()
        self.spn_graphic_labelsize.setRange(6, 36)
        self.spn_graphic_labelsize.valueChanged.connect(self.graphic_apply_selected_field_controls)

        self.cmb_graphic_label_font = QComboBox()
        self.cmb_graphic_label_font.addItems(_math_font_candidates())
        self.cmb_graphic_label_font.currentIndexChanged.connect(self.graphic_apply_selected_field_controls)

        self.spn_graphic_label_x = QDoubleSpinBox()
        self.spn_graphic_label_x.setRange(-20.0, 20.0)
        self.spn_graphic_label_x.setSingleStep(0.10)
        self.spn_graphic_label_x.setDecimals(2)
        self.spn_graphic_label_x.valueChanged.connect(self.graphic_apply_selected_field_controls)

        self.spn_graphic_label_y = QDoubleSpinBox()
        self.spn_graphic_label_y.setRange(-20.0, 20.0)
        self.spn_graphic_label_y.setSingleStep(0.10)
        self.spn_graphic_label_y.setDecimals(2)
        self.spn_graphic_label_y.valueChanged.connect(self.graphic_apply_selected_field_controls)

        grid_label.addWidget(QLabel("Label size:"), 0, 0)
        grid_label.addWidget(self.spn_graphic_labelsize, 0, 1)
        grid_label.addWidget(QLabel("Label font:"), 1, 0)
        grid_label.addWidget(self.cmb_graphic_label_font, 1, 1)
        grid_label.addWidget(QLabel("Label X:"), 2, 0)
        grid_label.addWidget(self.spn_graphic_label_x, 2, 1)
        grid_label.addWidget(QLabel("Label Y:"), 3, 0)
        grid_label.addWidget(self.spn_graphic_label_y, 3, 1)
        left_layout.addWidget(box_label)

        box_layout = QGroupBox("Graphic layout")
        grid_layout_controls = QGridLayout(box_layout)
        grid_layout_controls.setContentsMargins(8, 8, 8, 8)
        grid_layout_controls.setHorizontalSpacing(6)
        grid_layout_controls.setVerticalSpacing(6)

        self.spn_graphic_canvas_width = QSpinBox()
        self.spn_graphic_canvas_width.setRange(500, 2200)
        self.spn_graphic_canvas_width.setSingleStep(20)
        self.spn_graphic_canvas_width.setValue(int(self.graphic_style.get("layout_width_px", 920)))
        self.spn_graphic_canvas_width.valueChanged.connect(self.graphic_apply_layout_controls)

        self.spn_graphic_canvas_height = QSpinBox()
        self.spn_graphic_canvas_height.setRange(340, 1600)
        self.spn_graphic_canvas_height.setSingleStep(20)
        self.spn_graphic_canvas_height.setValue(int(self.graphic_style.get("layout_height_px", 600)))
        self.spn_graphic_canvas_height.valueChanged.connect(self.graphic_apply_layout_controls)

        self.btn_graphic_reset_layout = QPushButton("Reset layout")
        self.btn_graphic_reset_layout.clicked.connect(self.graphic_reset_layout)

        self.btn_graphic_reset_zoom = QPushButton("Reset zoom")
        self.btn_graphic_reset_zoom.clicked.connect(self.graphic_reset_zoom)

        grid_layout_controls.addWidget(QLabel("Width:"), 0, 0)
        grid_layout_controls.addWidget(self.spn_graphic_canvas_width, 0, 1)
        grid_layout_controls.addWidget(QLabel("Height:"), 1, 0)
        grid_layout_controls.addWidget(self.spn_graphic_canvas_height, 1, 1)
        grid_layout_controls.addWidget(self.btn_graphic_reset_layout, 2, 0)
        grid_layout_controls.addWidget(self.btn_graphic_reset_zoom, 2, 1)
        left_layout.addWidget(box_layout)

        self.btn_export_graphic = QPushButton("Export image")
        self.btn_export_graphic.clicked.connect(lambda: self.export_canvas(self.canvas_graphic))
        left_layout.addWidget(self.btn_export_graphic)
        left_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(430)
        left_scroll.setMaximumWidth(520)
        root.addWidget(left_scroll, 0)

        self.canvas_graphic = BoundedGraphicCanvas(width_in=9.2, height_in=6.0, dpi=100)
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(False)
        plot_scroll.setAlignment(Qt.AlignCenter)
        plot_scroll.setWidget(self.canvas_graphic)
        plot_scroll.setStyleSheet("QScrollArea { background: #f4f6f8; border: 1px solid #d9dee7; }")
        self.graphic_plot_scroll = plot_scroll
        self.canvas_graphic.set_scroll_area(plot_scroll)
        root.addWidget(plot_scroll, 1)

        self.tabs.addTab(self.tab_graphic, "Graphic")
        self.graphic_sync_diagram_controls(preserve_plot=False)
        self.graphic_apply_canvas_layout(redraw=False)
        self.plot_graphic()

    def _graphic_canvas_resize_callback(self, width_px, height_px):
        self._graphic_ensure_state()
        self.graphic_style["layout_width_px"] = int(width_px)
        self.graphic_style["layout_height_px"] = int(height_px)
        try:
            widgets = [self.spn_graphic_canvas_width, self.spn_graphic_canvas_height]
            old_blocks = [w.blockSignals(True) for w in widgets]
            self.spn_graphic_canvas_width.setValue(int(width_px))
            self.spn_graphic_canvas_height.setValue(int(height_px))
            for w, old in zip(widgets, old_blocks):
                w.blockSignals(old)
        except Exception:
            pass

    def _graphic_apply_layout_controls(self, *args):
        if getattr(self, "_graphic_loading_layout_controls", False):
            return
        self._graphic_ensure_state()
        try:
            w = int(self.spn_graphic_canvas_width.value())
            h = int(self.spn_graphic_canvas_height.value())
        except Exception:
            w = int(self.graphic_style.get("layout_width_px", 920))
            h = int(self.graphic_style.get("layout_height_px", 600))
        self.graphic_style["layout_width_px"] = int(w)
        self.graphic_style["layout_height_px"] = int(h)
        if hasattr(self, "canvas_graphic"):
            self.canvas_graphic.set_base_size_px(w, h, redraw=False)
        self.plot_graphic()

    def _graphic_apply_canvas_layout(self, redraw=True):
        self._graphic_ensure_state()
        if not hasattr(self, "canvas_graphic"):
            return
        try:
            w = int(self.graphic_style.get("layout_width_px", 920))
            h = int(self.graphic_style.get("layout_height_px", 600))
            z = float(self.graphic_style.get("layout_zoom", 1.0))
            self.canvas_graphic.set_base_size_px(w, h, redraw=False)
            self.canvas_graphic.set_zoom(z, redraw=False)
            if redraw:
                self.plot_graphic()
        except Exception:
            pass

    def _graphic_reset_layout(self):
        self._graphic_ensure_state()
        self.graphic_style["layout_width_px"] = 920
        self.graphic_style["layout_height_px"] = 600
        self._graphic_loading_layout_controls = True
        try:
            self.spn_graphic_canvas_width.setValue(920)
            self.spn_graphic_canvas_height.setValue(600)
        finally:
            self._graphic_loading_layout_controls = False
        self.graphic_apply_canvas_layout(redraw=True)

    def _graphic_reset_zoom(self):
        self._graphic_ensure_state()
        self.graphic_style["layout_zoom"] = 1.0
        if hasattr(self, "canvas_graphic"):
            self.canvas_graphic.reset_zoom()
        try:
            if hasattr(self, "graphic_plot_scroll"):
                self.graphic_plot_scroll.horizontalScrollBar().setValue(0)
                self.graphic_plot_scroll.verticalScrollBar().setValue(0)
        except Exception:
            pass

    def _graphic_edit_layout_toggled(self, checked):
        self._graphic_ensure_state()
        self.graphic_style["edit_layout"] = False
        if hasattr(self, "canvas_graphic"):
            self.canvas_graphic.set_edit_layout(False)

    def _graphic_fields_multi_select_dialog(self, title, field_names, initial_fields):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox, QLabel, QAbstractItemView
        from PyQt5.QtCore import Qt

        field_names = list(field_names or [])
        if not field_names:
            return []

        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(460, 520)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select fields to display:"))

        lw = QListWidget()
        lw.setSelectionMode(QAbstractItemView.MultiSelection)
        init = set(initial_fields or [])
        for name in field_names:
            it = QListWidgetItem(name)
            it.setData(Qt.UserRole, name)
            lw.addItem(it)
            if name in init:
                it.setSelected(True)
        lay.addWidget(lw)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)

        if dlg.exec_() != QDialog.Accepted:
            return None
        out = []
        for i in range(lw.count()):
            it = lw.item(i)
            if it.isSelected():
                out.append(it.data(Qt.UserRole))
        return out

    def _graphic_on_diagram_changed(self, *args):
        self._graphic_ensure_state()
        self.graphic_style["diagram_key"] = self.graphic_current_diagram_key()
        self.graphic_style["view"] = "All"
        self.graphic_sync_diagram_controls(preserve_plot=True)

    def _graphic_select_visible_fields(self):
        self._graphic_ensure_state()
        dkey = self.graphic_current_diagram_key()
        opts = self.graphic_current_field_options()
        current = list(self.graphic_visible_fields.get(dkey, opts))
        picked = self.graphic_fields_multi_select_dialog(f"{_diagram_display_name(dkey)} — Visible fields", opts, current)
        if picked is None:
            return
        if not picked:
            QMessageBox.information(self, "Graphic", "At least one field must remain visible.")
            return
        self.graphic_visible_fields[dkey] = list(picked)
        if len(picked) == len(opts):
            self.graphic_style["view"] = "All"
            self.cmb_graphic_view.setCurrentText("All")
        else:
            self.graphic_style["view"] = CUSTOM_FIELD_VIEW_NAME
            self.cmb_graphic_view.setCurrentText(CUSTOM_FIELD_VIEW_NAME)
        if hasattr(self, "lbl_graphic_visible_fields"):
            self.lbl_graphic_visible_fields.setText(self.graphic_visible_fields_summary(dkey))
        self.plot_graphic()

    def _graphic_reset_visible_fields(self):
        self._graphic_ensure_state()
        dkey = self.graphic_current_diagram_key()
        opts = self.graphic_current_field_options()
        self.graphic_visible_fields[dkey] = list(opts)
        self.graphic_style["view"] = "All"
        self.cmb_graphic_view.setCurrentText("All")
        if hasattr(self, "lbl_graphic_visible_fields"):
            self.lbl_graphic_visible_fields.setText(self.graphic_visible_fields_summary(dkey))
        self.plot_graphic()

    def _graphic_current_field(self) -> str:
        try:
            field = str(self.cmb_graphic_edit_field.currentText())
            if field:
                return field
        except Exception:
            pass
        opts = self.graphic_current_field_options()
        return opts[0] if opts else ""

    def _graphic_load_selected_field_controls(self, *args, redraw=True):
        self._graphic_ensure_state()
        field = self.graphic_current_field()
        defaults = _zone_defaults(self.graphic_current_diagram_key())
        st = self.graphic_zone_styles.get(field, defaults.get(field, {}))

        self._graphic_loading_controls = True
        try:
            widgets = [
                self.cmb_graphic_linestyle,
                self.spn_graphic_linewidth,
                self.spn_graphic_alpha,
                self.spn_graphic_labelsize,
                self.cmb_graphic_label_font,
                self.spn_graphic_label_x,
                self.spn_graphic_label_y,
            ]
            old_blocks = [w.blockSignals(True) for w in widgets]

            self.cmb_graphic_linestyle.setCurrentText(str(st.get("linestyle_name", "Auto")))
            self.spn_graphic_linewidth.setValue(float(st.get("linewidth", 1.1)))
            self.spn_graphic_alpha.setValue(float(st.get("alpha", 0.1)))
            self.spn_graphic_labelsize.setValue(int(st.get("label_size", 11)))

            font_name = _safe_font_name(str(st.get("label_font", "DejaVu Sans")))
            if self.cmb_graphic_label_font.findText(font_name) < 0:
                self.cmb_graphic_label_font.insertItem(0, font_name)
            self.cmb_graphic_label_font.setCurrentText(font_name)

            self.spn_graphic_label_x.setValue(float(st.get("label_x", 0.0)))
            self.spn_graphic_label_y.setValue(float(st.get("label_y", 0.0)))

            for w, old in zip(widgets, old_blocks):
                w.blockSignals(old)
        finally:
            self._graphic_loading_controls = False

        if redraw:
            self.plot_graphic()

    def _graphic_apply_selected_field_controls(self, *args):
        if getattr(self, "_graphic_loading_controls", False):
            return
        self._graphic_ensure_state()
        field = self.graphic_current_field()
        self.graphic_zone_styles.setdefault(field, {})
        st = self.graphic_zone_styles[field]
        try:
            st["linestyle_name"] = str(self.cmb_graphic_linestyle.currentText())
            st["linewidth"] = float(self.spn_graphic_linewidth.value())
            st["alpha"] = float(self.spn_graphic_alpha.value())
            st["label_size"] = int(self.spn_graphic_labelsize.value())
            st["label_font"] = str(self.cmb_graphic_label_font.currentText())
            st["label_x"] = float(self.spn_graphic_label_x.value())
            st["label_y"] = float(self.spn_graphic_label_y.value())
        except Exception:
            pass
        self.plot_graphic()

    def _graphic_pick_color(self, key: str):
        self._graphic_ensure_state()
        field = self.graphic_current_field()
        self.graphic_zone_styles.setdefault(field, {})
        default_color = "#111111" if key == "edgecolor" else "#d9d9d9"
        current = self.graphic_zone_styles[field].get(key, default_color)
        color = QColorDialog.getColor(QColor(str(current)), self, f"{field} — {'Line color' if key == 'edgecolor' else 'Fill color'}")
        if color.isValid():
            self.graphic_zone_styles[field][key] = color.name()
            self.plot_graphic()

    def _graphic_choose_databases(self):
        try:
            self.select_db_to_plot()
        except Exception as e:
            QMessageBox.warning(self, "Graphic", str(e))
        try:
            self.plot_graphic()
        except Exception:
            pass

    def _graphic_edit_point_colors(self):
        try:
            self.edit_colors()
        except Exception as e:
            QMessageBox.warning(self, "Graphic", str(e))
        try:
            self.plot_graphic()
        except Exception:
            pass

    def _select_groups_graphic(self):
        try:
            keys = self._safe_group_keys()
        except Exception:
            keys = []
        if not keys:
            QMessageBox.information(self, "Graphic", "No groups available.")
            return
        current = getattr(self, "selected_groups_graphic", keys)
        picked = self._groups_multi_select_dialog("Graphic — Filter groups", current)
        if picked is None:
            return
        self.selected_groups_graphic = list(picked)
        self.plot_graphic()

    def _graphic_selected_group_keys_for_tab(self, tab_name: str):
        if tab_name == "graphic":
            return set(getattr(self, "selected_groups_graphic", []) or [])
        if old_selected_group_keys_for_tab is not None:
            return old_selected_group_keys_for_tab(self, tab_name)
        return set()

    def _graphic_update_all_groups_and_colors(self):
        if old_update_all_groups_and_colors is not None:
            old_update_all_groups_and_colors(self)
        try:
            keys = list(getattr(self, "group_labels", {}).keys())
            if not hasattr(self, "selected_groups_graphic") or self.selected_groups_graphic is None:
                self.selected_groups_graphic = list(keys)
            elif not self.selected_groups_graphic and keys:
                self.selected_groups_graphic = list(keys)
            else:
                valid = set(keys)
                self.selected_groups_graphic = [k for k in self.selected_groups_graphic if k in valid]
        except Exception:
            pass

    def _graphic_refresh_after_settings_change(self):
        if old_refresh_after_settings_change is not None:
            old_refresh_after_settings_change(self)
        try:
            self.plot_graphic()
        except Exception:
            pass

    def _iter_graphic_ternary_points(self):
        """
        Yield sample points for the blocks/bombs-lapilli-ash ternary diagram.

        Grain-size classes are evaluated in phi units:
          blocks and bombs: phi < -6        (> 64 mm)
          lapilli:          -6 <= phi <= -1 (64-2 mm)
          ash:              phi > -1         (< 2 mm)

        The three class sums are normalized to 100 before barycentric projection.
        """
        bases = self._selected_bases_xy()
        selected_groups = self._selected_group_keys_for_tab("graphic")
        h = 0.8660254037844386

        for base_key in bases:
            b = base_key_canon(base_key) if "base_key_canon" in globals() else base_key

            if b == "Tamiz":
                df = getattr(self, "df_data", None)
                if df is None or getattr(df, "empty", True) or len(df.columns) < 3:
                    continue
                phi_col = df.columns[0]
                sample_col = df.columns[1]
                weight_col = df.columns[2]
                group_col = getattr(self, "group_col", "Group")
                if group_col not in df.columns:
                    group_col = "Group" if "Group" in df.columns else None

            elif b == "Laser":
                df = getattr(self, "df_laser", None)
                if df is None or getattr(df, "empty", True):
                    continue
                phi_col = "phi" if "phi" in df.columns else None
                sample_col = "Sample" if "Sample" in df.columns else None
                weight_col = "wt%" if "wt%" in df.columns else ("Wt" if "Wt" in df.columns else None)
                group_col = "Group" if "Group" in df.columns else None
                if phi_col is None or sample_col is None or weight_col is None:
                    continue

            elif b == "Híbrido":
                df = getattr(self, "df_hybrid", None)
                if df is None or getattr(df, "empty", True):
                    continue
                phi_col = "phi" if "phi" in df.columns else None
                if "Hybrid Sample" in df.columns:
                    sample_col = "Hybrid Sample"
                elif "Sample" in df.columns:
                    sample_col = "Sample"
                elif "Tamiz Sample" in df.columns:
                    sample_col = "Tamiz Sample"
                else:
                    sample_col = None
                weight_col = "wt%" if "wt%" in df.columns else ("Wt" if "Wt" in df.columns else None)
                group_col = getattr(self, "group_col", "Group")
                if group_col not in df.columns:
                    group_col = "Group" if "Group" in df.columns else None
                if phi_col is None or sample_col is None or weight_col is None:
                    continue
            else:
                continue

            work = df.copy()
            work[phi_col] = pd.to_numeric(work[phi_col], errors="coerce")
            work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce")
            work = work[np.isfinite(work[phi_col]) & np.isfinite(work[weight_col])]
            if work.empty:
                continue

            for sample_value, grp_df in work.groupby(sample_col, sort=False):
                phi = grp_df[phi_col].astype(float)
                wt = grp_df[weight_col].astype(float)

                blocks = float(wt[phi < -6.0].sum())
                lapilli = float(wt[(phi >= -6.0) & (phi <= -1.0)].sum())
                ash = float(wt[phi > -1.0].sum())
                total = blocks + lapilli + ash
                if not np.isfinite(total) or total <= 0.0:
                    continue

                blocks_pct = 100.0 * blocks / total
                lapilli_pct = 100.0 * lapilli / total
                ash_pct = 100.0 * ash / total

                # Barycentric projection using vertices:
                # blocks/bombs=(0.5,h), lapilli=(0,0), ash=(1,0).
                x = (0.5 * blocks_pct + ash_pct) / 100.0
                y = (h * blocks_pct) / 100.0

                if group_col is not None and group_col in grp_df.columns and not grp_df[group_col].dropna().empty:
                    grp = grp_df[group_col].dropna().iloc[0]
                else:
                    grp = "No Group"

                try:
                    grp = self._fw_effective_group(base_key, sample_value, grp)
                except Exception:
                    pass
                gk = self._group_key(grp, base_key)
                if selected_groups and gk not in selected_groups:
                    continue
                gl = self.group_labels.get(gk, self._group_label(grp, base_key)) if hasattr(self, "group_labels") else self._group_label(grp, base_key)
                color = self.group_colors.get(gk, GROUP_COLORS[0]) if hasattr(self, "group_colors") and isinstance(self.group_colors, dict) else GROUP_COLORS[0]

                yield {
                    "x": float(x),
                    "y": float(y),
                    "sample": str(sample_value),
                    "group": str(grp),
                    "base": base_key,
                    "gkey": gk,
                    "glabel": gl,
                    "color": color,
                    "marker": (getattr(self, "group_markers", None) or {}).get(gk, "o"),
                    "blocks_pct": blocks_pct,
                    "lapilli_pct": lapilli_pct,
                    "ash_pct": ash_pct,
                }

    def _plot_graphic(self, *args):
        if not hasattr(self, "canvas_graphic"):
            return

        self._graphic_ensure_state()

        diagram_key = self.graphic_current_diagram_key() if hasattr(self, "cmb_graphic_diagram") else str(self.graphic_style.get("diagram_key", DEFAULT_DIAGRAM_KEY))
        meta = _diagram_meta(diagram_key)
        view_options = _diagram_view_options(diagram_key)

        self.graphic_style["diagram_key"] = diagram_key
        self.graphic_style["view"] = str(self.cmb_graphic_view.currentText()) if hasattr(self, "cmb_graphic_view") else str(self.graphic_style.get("view", "All"))
        if self.graphic_style["view"] not in view_options:
            self.graphic_style["view"] = "All"
        self.graphic_style["show_labels"] = bool(self.chk_graphic_labels.isChecked())
        self.graphic_style["show_sample_names"] = bool(self.chk_graphic_sample_names.isChecked())

        try:
            self.graphic_style["layout_zoom"] = float(self.canvas_graphic.zoom())
        except Exception:
            pass

        fig = self.canvas_graphic.figure
        fig.clear()
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        view = str(self.graphic_style.get("view", "All"))
        selected_fields = None
        if view == CUSTOM_FIELD_VIEW_NAME:
            selected_fields = list(self.graphic_visible_fields.get(diagram_key, meta.get("field_options", [])))

        draw_vector_diagram(
            ax,
            diagram_key=diagram_key,
            view=view,
            zone_styles=self.graphic_zone_styles,
            show_labels=bool(self.graphic_style.get("show_labels", True)),
            selected_fields=selected_fields,
        )

        coordinate_mode = str(meta.get("coordinate_mode", "xy")).strip().lower()
        pts = []
        try:
            if coordinate_mode == "ternary_grain_size":
                pts = list(self._iter_graphic_ternary_points())
            else:
                pts = list(self._iter_points_for_plot("mean", "sigma", tab_name="graphic"))
        except Exception:
            pts = []

        xmin, xmax, ymin, ymax = meta.get("axis_limits", DEFAULT_AXIS_LIMITS)

        if pts:
            # XY classification diagrams may expand to include outlying data.
            # Ternary geometry is normalized and must keep its fixed frame.
            if coordinate_mode != "ternary_grain_size":
                xs_all = np.array([p["x"] for p in pts], dtype=float)
                ys_all = np.array([p["y"] for p in pts], dtype=float)
                if xs_all.size and ys_all.size and np.isfinite(xs_all).any() and np.isfinite(ys_all).any():
                    xmin = min(xmin, float(np.nanmin(xs_all)) - 0.2)
                    xmax = max(xmax, float(np.nanmax(xs_all)) + 0.2)
                    ymin = max(0.0, min(ymin, float(np.nanmin(ys_all)) - 0.15))
                    ymax = max(ymax, float(np.nanmax(ys_all)) + 0.2)

            by_group = {}
            for p in pts:
                by_group.setdefault(p["gkey"], []).append(p)

            for gk, arr in by_group.items():
                xs = [a["x"] for a in arr]
                ys = [a["y"] for a in arr]
                color = arr[0]["color"]
                label = arr[0]["glabel"]
                mk = arr[0].get("marker", "o") or "o"
                ax.scatter(xs, ys, s=_fw_marker_s(mk, 6.5), marker=mk, color=color, edgecolor="black", linewidth=0.45, alpha=0.95, label=label, zorder=25)
                if bool(self.graphic_style.get("show_sample_names", False)):
                    for a in arr:
                        ax.annotate(a["sample"], (a["x"], a["y"]), xytext=(4, 4), textcoords="offset points", fontsize=8, alpha=0.9, zorder=30)

            handles, labels = ax.get_legend_handles_labels()
            uniq = {}
            for h, l in zip(handles, labels):
                if l not in uniq:
                    uniq[l] = h
            if uniq:
                ax.legend(list(uniq.values()), list(uniq.keys()), fontsize=8, frameon=True, loc="best")
        else:
            if bool(meta.get("show_no_data_message", True)):
                ax.text(0.5, 0.03, "No data points loaded yet.", ha="center", va="bottom", transform=ax.transAxes, fontsize=9, color="#666666")

        if bool(meta.get("hide_axes", False)):
            ax.set_axis_off()
        else:
            try:
                ax.set_xlabel(self._axis_label_for_key("mean"))
                ax.set_ylabel(self._axis_label_for_key("sigma"))
            except Exception:
                ax.set_xlabel("Mean (Mφ)")
                ax.set_ylabel("Sorting (σφ)")
            ax.grid(True, alpha=0.18)

        if bool(meta.get("title_on_plot", True)):
            ax.set_title(str(meta.get("title", diagram_key)))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if bool(meta.get("equal_aspect", False)):
            ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        self.canvas_graphic.draw_idle()

    def _graphic_save_current_tab(self):
        current = self.tabs.currentWidget()
        if current is getattr(self, "tab_graphic", None):
            return self.export_canvas(self.canvas_graphic)
        if old_save_current_tab is not None:
            return old_save_current_tab(self)
        return None

    # Bind Graphic methods.
    MainWindow._graphic_ensure_state = _graphic_ensure_state
    MainWindow.graphic_canvas_resize_callback = _graphic_canvas_resize_callback
    MainWindow.graphic_apply_layout_controls = _graphic_apply_layout_controls
    MainWindow.graphic_apply_canvas_layout = _graphic_apply_canvas_layout
    MainWindow.graphic_reset_layout = _graphic_reset_layout
    MainWindow.graphic_reset_zoom = _graphic_reset_zoom
    MainWindow.graphic_edit_layout_toggled = _graphic_edit_layout_toggled
    MainWindow.graphic_current_diagram_key = _graphic_current_diagram_key
    MainWindow.graphic_current_diagram_meta = _graphic_current_diagram_meta
    MainWindow.graphic_set_diagram = _graphic_set_diagram
    MainWindow.graphic_selector_changed = _graphic_selector_changed
    MainWindow.graphic_current_field_options = _graphic_current_field_options
    MainWindow.graphic_visible_fields_summary = _graphic_visible_fields_summary
    MainWindow.graphic_sync_diagram_controls = _graphic_sync_diagram_controls
    MainWindow.graphic_fields_multi_select_dialog = _graphic_fields_multi_select_dialog
    MainWindow.graphic_on_diagram_changed = _graphic_on_diagram_changed
    MainWindow.graphic_select_visible_fields = _graphic_select_visible_fields
    MainWindow.graphic_reset_visible_fields = _graphic_reset_visible_fields
    MainWindow.graphic_current_field = _graphic_current_field
    MainWindow.graphic_load_selected_field_controls = _graphic_load_selected_field_controls
    MainWindow.graphic_apply_selected_field_controls = _graphic_apply_selected_field_controls
    MainWindow.graphic_pick_color = _graphic_pick_color
    MainWindow.graphic_choose_databases = _graphic_choose_databases
    MainWindow.graphic_edit_point_colors = _graphic_edit_point_colors
    MainWindow.select_groups_graphic = _select_groups_graphic
    MainWindow._iter_graphic_ternary_points = _iter_graphic_ternary_points
    MainWindow.plot_graphic = _plot_graphic

    MainWindow.initUI_tabs = _graphic_initUI_tabs
    MainWindow.save_current_tab = _graphic_save_current_tab
    MainWindow._refresh_after_settings_change = _graphic_refresh_after_settings_change
    MainWindow._selected_group_keys_for_tab = _graphic_selected_group_keys_for_tab
    MainWindow._update_all_groups_and_colors = _graphic_update_all_groups_and_colors
    MainWindow._graphic_support_installed = True

    # Optional: include Graphic state in full-project save/load if the FW integration
    # functions exist in the main module.
    try:
        main_mod = sys.modules.get(MainWindow.__module__)
        if main_mod is not None and not getattr(main_mod, "_graphic_project_patch_installed", False):
            old_collect = getattr(main_mod, "_fw_collect_project_dict", None)
            old_apply = getattr(main_mod, "_fw_apply_project_dict", None)

            if old_collect is not None:
                def _collect_with_graphic(self):
                    data = old_collect(self)
                    fw = data.setdefault("fw", {})
                    try:
                        self._graphic_ensure_state()
                    except Exception:
                        pass
                    fw["graphic_style"] = deepcopy(getattr(self, "graphic_style", {}))
                    fw["graphic_zone_styles"] = deepcopy(getattr(self, "graphic_zone_styles", {}))
                    fw["graphic_visible_fields"] = deepcopy(getattr(self, "graphic_visible_fields", {}))
                    fw["selected_groups_graphic"] = list(getattr(self, "selected_groups_graphic", []))
                    return data
                setattr(main_mod, "_fw_collect_project_dict", _collect_with_graphic)

            if old_apply is not None:
                def _apply_with_graphic(self, data):
                    old_apply(self, data)
                    fw = (data.get("fw", {}) or {}) if isinstance(data, dict) else {}
                    try:
                        self._graphic_ensure_state()
                        if isinstance(fw.get("graphic_style"), dict):
                            self.graphic_style.update(fw.get("graphic_style") or {})
                        if isinstance(fw.get("graphic_zone_styles"), dict):
                            defaults = _zone_defaults()
                            restored = fw.get("graphic_zone_styles") or {}
                            self.graphic_zone_styles = defaults
                            for k, v in restored.items():
                                if isinstance(v, dict):
                                    self.graphic_zone_styles.setdefault(k, {})
                                    self.graphic_zone_styles[k].update(v)
                        if isinstance(fw.get("graphic_visible_fields"), dict):
                            self.graphic_visible_fields = deepcopy(fw.get("graphic_visible_fields") or {})
                        if isinstance(fw.get("selected_groups_graphic"), list):
                            self.selected_groups_graphic = list(fw.get("selected_groups_graphic") or [])
                        if hasattr(self, "cmb_graphic_diagram"):
                            idx = self.cmb_graphic_diagram.findData(str(self.graphic_style.get("diagram_key", DEFAULT_DIAGRAM_KEY)))
                            if idx >= 0:
                                self.cmb_graphic_diagram.setCurrentIndex(idx)
                        if hasattr(self, "graphic_sync_diagram_controls"):
                            self.graphic_sync_diagram_controls(preserve_plot=False)
                        if hasattr(self, "cmb_graphic_view"):
                            self.cmb_graphic_view.setCurrentText(str(self.graphic_style.get("view", "All")))
                        if hasattr(self, "chk_graphic_labels"):
                            self.chk_graphic_labels.setChecked(bool(self.graphic_style.get("show_labels", True)))
                        if hasattr(self, "chk_graphic_sample_names"):
                            self.chk_graphic_sample_names.setChecked(bool(self.graphic_style.get("show_sample_names", False)))
                        if hasattr(self, "spn_graphic_canvas_width") and hasattr(self, "spn_graphic_canvas_height"):
                            try:
                                self._graphic_loading_layout_controls = True
                                self.spn_graphic_canvas_width.setValue(int(self.graphic_style.get("layout_width_px", 920)))
                                self.spn_graphic_canvas_height.setValue(int(self.graphic_style.get("layout_height_px", 600)))
                            finally:
                                self._graphic_loading_layout_controls = False
                        if hasattr(self, "chk_graphic_edit_layout"):
                            self.chk_graphic_edit_layout.setChecked(bool(self.graphic_style.get("edit_layout", False)))
                        if hasattr(self, "graphic_apply_canvas_layout"):
                            self.graphic_apply_canvas_layout(redraw=False)
                        if hasattr(self, "graphic_load_selected_field_controls"):
                            self.graphic_load_selected_field_controls()
                        if hasattr(self, "plot_graphic"):
                            self.plot_graphic()
                    except Exception:
                        pass
                setattr(main_mod, "_fw_apply_project_dict", _apply_with_graphic)

            setattr(main_mod, "_graphic_project_patch_installed", True)
    except Exception:
        pass



# =========================================================
# EDITABLE SIEVE/LASER DATABASE PATCH
# =========================================================
class DataBaseWindow(QDialog):
    """
    Database viewer/editor used by the Sieve and Laser database windows.

    Editable cells:
      - group/facies row
      - granulometric values

    Locked structure:
      - Phi column
      - sample names (column headers)
      - row/column count
    """
    def __init__(self, title, df: pd.DataFrame, parent=None, editable=False,
                 database_type=None, apply_callback=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 650)
        self._df = pd.DataFrame(df).copy(deep=True)
        self._saved_df = self._df.copy(deep=True)
        self._editable_enabled = bool(editable and callable(apply_callback))
        self._database_type = database_type
        self._apply_callback = apply_callback
        self._editing = False

        layout = QVBoxLayout(self)
        self.table = QTableWidget(self)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        layout.addWidget(self.table)

        self.lbl_edit_info = QLabel(
            "Edit mode is disabled. Phi classes and sample names are protected."
        )
        layout.addWidget(self.lbl_edit_info)

        btns = QHBoxLayout()
        self.btn_edit = QPushButton("Edit data")
        self.btn_save_changes = QPushButton("Save changes")
        self.btn_cancel_changes = QPushButton("Cancel changes")
        self.btn_export = QPushButton("Export to Excel")
        self.btn_close = QPushButton("Close")

        self.btn_edit.clicked.connect(self._begin_edit)
        self.btn_save_changes.clicked.connect(self._save_changes)
        self.btn_cancel_changes.clicked.connect(self._cancel_changes)
        self.btn_export.clicked.connect(self._export_excel)
        self.btn_close.clicked.connect(self._close_requested)

        self.btn_edit.setVisible(self._editable_enabled)
        self.btn_save_changes.setVisible(self._editable_enabled)
        self.btn_cancel_changes.setVisible(self._editable_enabled)

        btns.addWidget(self.btn_edit)
        btns.addWidget(self.btn_save_changes)
        btns.addWidget(self.btn_cancel_changes)
        btns.addStretch(1)
        btns.addWidget(self.btn_export)
        btns.addWidget(self.btn_close)
        layout.addLayout(btns)

        self._populate(self._df)
        self._set_edit_mode(False)

    @staticmethod
    def _item_from_value(value):
        item = QTableWidgetItem()
        try:
            if pd.isna(value):
                item.setText("")
                return item
        except Exception:
            pass
        try:
            numeric = float(str(value).strip().replace(",", "."))
            if np.isfinite(numeric):
                item.setData(Qt.EditRole, float(numeric))
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                return item
        except Exception:
            pass
        item.setText(str(value))
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        return item

    def _populate(self, df):
        self.table.blockSignals(True)
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for row in range(len(df)):
            for col in range(len(df.columns)):
                value = df.iloc[row, col]
                if row == 0:
                    item = QTableWidgetItem("" if pd.isna(value) else str(value))
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                else:
                    item = self._item_from_value(value)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self._apply_item_flags()

    def _apply_item_flags(self):
        for row in range(self.table.rowCount()):
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    item = QTableWidgetItem("")
                    self.table.setItem(row, col, item)
                flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                # Column 0 contains the protected Group/Phi labels and phi values.
                # Every sample column is editable only while edit mode is active.
                if self._editing and self._editable_enabled and col > 0:
                    flags |= Qt.ItemIsEditable
                item.setFlags(flags)

    def _set_edit_mode(self, enabled):
        self._editing = bool(enabled and self._editable_enabled)
        self._apply_item_flags()
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed |
            QAbstractItemView.SelectedClicked
            if self._editing else QAbstractItemView.NoEditTriggers
        )
        self.btn_edit.setEnabled(self._editable_enabled and not self._editing)
        self.btn_save_changes.setEnabled(self._editing)
        self.btn_cancel_changes.setEnabled(self._editing)
        self.btn_export.setEnabled(not self._editing)
        if self._editing:
            self.lbl_edit_info.setText(
                "Edit mode: modify group/facies or granulometric values. "
                "Phi classes and sample names are protected."
            )
        else:
            self.lbl_edit_info.setText(
                "Edit mode is disabled. Phi classes and sample names are protected."
            )

    def _begin_edit(self):
        self._saved_df = self._df.copy(deep=True)
        self._set_edit_mode(True)

    def _table_to_dataframe(self):
        columns = [str(self.table.horizontalHeaderItem(c).text()) for c in range(self.table.columnCount())]
        rows = []
        for row in range(self.table.rowCount()):
            values = []
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item is None:
                    values.append("")
                    continue
                value = item.data(Qt.EditRole)
                if value is None:
                    value = item.text()
                values.append(value)
            rows.append(values)
        return pd.DataFrame(rows, columns=columns)

    def _save_changes(self):
        if not self._editing:
            return
        edited = self._table_to_dataframe()
        try:
            result = self._apply_callback(edited)
        except Exception as exc:
            QMessageBox.warning(self, "Database edit error", str(exc))
            return
        if result is False or result is None:
            return
        if isinstance(result, pd.DataFrame):
            self._df = result.copy(deep=True)
        else:
            self._df = edited.copy(deep=True)
        self._saved_df = self._df.copy(deep=True)
        self._populate(self._df)
        self._set_edit_mode(False)

    def _cancel_changes(self):
        if not self._editing:
            return
        self._df = self._saved_df.copy(deep=True)
        self._populate(self._df)
        self._set_edit_mode(False)

    def _close_requested(self):
        if self._editing:
            reply = QMessageBox.question(
                self,
                "Discard changes?",
                "There are unapplied edits. Discard them and close the window?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self.accept()

    def reject(self):
        if self._editing:
            reply = QMessageBox.question(
                self,
                "Discard changes?",
                "There are unapplied edits. Discard them and close the window?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        super().reject()

    def _export_excel(self):
        file, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "Data.xlsx", "Excel files (*.xlsx)"
        )
        if not file:
            return
        if not file.lower().endswith(".xlsx"):
            file += ".xlsx"
        df_out = self._df.copy()
        for col in df_out.columns:
            if col == df_out.columns[0]:
                continue
            converted = pd.to_numeric(
                df_out[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            # Preserve the group row and convert numeric data rows where possible.
            if len(converted) > 1 and converted.iloc[1:].notna().any():
                df_out.loc[df_out.index[1:], col] = converted.iloc[1:].round(6).values
        try:
            df_out.to_excel(file, index=False)
            QMessageBox.information(self, "Export complete", "Saved to:\n" + file)
        except Exception as exc:
            QMessageBox.warning(self, "Export error", f"Could not export:\n{exc}")


def _fw_edit_parse_number(value, sample_name, phi_value):
    text = "" if value is None else str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return 0.0
    try:
        number = float(text.replace(",", "."))
    except Exception:
        raise ValueError(
            f"Invalid value for sample '{sample_name}' at phi {phi_value}: {text!r}."
        )
    if not np.isfinite(number):
        raise ValueError(
            f"Non-finite value for sample '{sample_name}' at phi {phi_value}."
        )
    if number < 0:
        raise ValueError(
            f"Negative value for sample '{sample_name}' at phi {phi_value}."
        )
    return float(number)


def _fw_edit_format2_to_long(self, view_df, db_key):
    view = pd.DataFrame(view_df).copy()
    if view.empty or len(view.columns) < 2 or len(view) < 2:
        raise ValueError("The database table is empty or incomplete.")

    db_key = str(db_key).strip().lower()
    if db_key == "tamiz":
        source = getattr(self, "df_data", None)
        if source is None or source.empty:
            raise ValueError("No Sieve database is loaded.")
        phi_col, sample_col, weight_col = source.columns[:3]
        group_col = getattr(self, "group_col", "Group")
    elif db_key == "laser":
        source = getattr(self, "df_laser", None)
        if source is None or source.empty:
            raise ValueError("No Laser database is loaded.")
        phi_col, sample_col, weight_col, group_col = "phi", "Sample", "Wt", "Group"
    else:
        raise ValueError(f"Unsupported editable database: {db_key}")

    sample_names = [str(c).strip() for c in view.columns[1:]]
    if any(not name for name in sample_names):
        raise ValueError("Sample names cannot be empty.")
    if len(set(sample_names)) != len(sample_names):
        raise ValueError("Duplicate sample names were detected in the table.")

    expected_samples = list(dict.fromkeys(source[sample_col].astype(str).tolist()))
    if sample_names != expected_samples:
        raise ValueError("Sample names or their order changed. These fields are protected.")

    raw_phis = view.iloc[1:, 0].tolist()
    phi_values = []
    for raw_phi in raw_phis:
        try:
            phi = float(str(raw_phi).strip().replace(",", "."))
        except Exception:
            raise ValueError(f"Invalid protected phi value: {raw_phi!r}.")
        if not np.isfinite(phi):
            raise ValueError(f"Invalid protected phi value: {raw_phi!r}.")
        phi_values.append(float(phi))

    expected_phis = sorted(pd.to_numeric(source[phi_col], errors="coerce").dropna().astype(float).unique().tolist())
    if len(phi_values) != len(expected_phis) or not np.allclose(phi_values, expected_phis, atol=1e-9, rtol=0.0):
        raise ValueError("Phi classes changed. The Phi column is protected.")

    groups = {}
    for col_index, sample_name in enumerate(sample_names, start=1):
        raw_group = view.iloc[0, col_index]
        group = "" if raw_group is None else str(raw_group).strip()
        if not group or group.lower() in {"nan", "none", "null"}:
            group = "No Group"
        groups[sample_name] = group

    rows = []
    totals = {}
    for col_index, sample_name in enumerate(sample_names, start=1):
        total = 0.0
        for row_offset, phi in enumerate(phi_values, start=1):
            weight = _fw_edit_parse_number(view.iloc[row_offset, col_index], sample_name, phi)
            total += weight
            rows.append({
                phi_col: float(phi),
                sample_col: sample_name,
                weight_col: float(weight),
                group_col: groups[sample_name],
            })
        totals[sample_name] = total

    zero_samples = [name for name, total in totals.items() if total <= 0]
    if zero_samples:
        raise ValueError(
            "Every sample must have a total greater than zero. Invalid sample(s): "
            + ", ".join(zero_samples)
        )

    out = pd.DataFrame(rows, columns=[phi_col, sample_col, weight_col, group_col])
    out.sort_values([sample_col, phi_col], kind="mergesort", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _fw_edit_normalize_long(df, sample_col, weight_col):
    out = df.copy()
    totals = out.groupby(sample_col)[weight_col].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Cannot normalize a sample whose sum is zero.")
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce") * (100.0 / totals)
    return out


def _fw_edit_changed_samples(old_df, new_df, phi_col, sample_col, weight_col, group_col):
    names = sorted(set(old_df[sample_col].astype(str)) | set(new_df[sample_col].astype(str)))
    changed = []
    for name in names:
        old = old_df[old_df[sample_col].astype(str) == name].copy().sort_values(phi_col)
        new = new_df[new_df[sample_col].astype(str) == name].copy().sort_values(phi_col)
        if len(old) != len(new):
            changed.append(name)
            continue
        old_phi = pd.to_numeric(old[phi_col], errors="coerce").to_numpy(dtype=float)
        new_phi = pd.to_numeric(new[phi_col], errors="coerce").to_numpy(dtype=float)
        old_wt = pd.to_numeric(old[weight_col], errors="coerce").to_numpy(dtype=float)
        new_wt = pd.to_numeric(new[weight_col], errors="coerce").to_numpy(dtype=float)
        old_group = str(old[group_col].iloc[0]) if group_col in old.columns and not old.empty else "No Group"
        new_group = str(new[group_col].iloc[0]) if group_col in new.columns and not new.empty else "No Group"
        if (
            not np.allclose(old_phi, new_phi, atol=1e-9, rtol=0.0, equal_nan=True)
            or not np.allclose(old_wt, new_wt, atol=1e-10, rtol=0.0, equal_nan=True)
            or old_group != new_group
        ):
            changed.append(name)
    return changed


def _fw_edit_hybrid_identity_rows(df_hybrid):
    if df_hybrid is None or df_hybrid.empty:
        return []
    for col in ("Hybrid Sample", "Sample", "Tamiz Sample"):
        if col in df_hybrid.columns:
            return list(dict.fromkeys(df_hybrid[col].astype(str).tolist()))
    return []


def _fw_edit_rebuild_affected_hybrids(self, source_db_key, changed_samples):
    df_hybrid = getattr(self, "df_hybrid", None)
    if df_hybrid is None or df_hybrid.empty:
        return []

    changed_set = set(str(v) for v in changed_samples)
    old_meta = dict(getattr(self, "_hybrid_meta", {}) or {})
    names = _fw_edit_hybrid_identity_rows(df_hybrid)
    if not names:
        return []

    identity_col = "Hybrid Sample" if "Hybrid Sample" in df_hybrid.columns else (
        "Sample" if "Sample" in df_hybrid.columns else "Tamiz Sample"
    )
    rebuilt_parts = []
    new_meta = {}
    affected_names = []

    for hybrid_name in names:
        old_part = df_hybrid[df_hybrid[identity_col].astype(str) == str(hybrid_name)].copy()
        meta = dict(old_meta.get(str(hybrid_name), {}) or {})

        tamiz_name = str(meta.get("tamiz", "")).strip()
        laser_name = str(meta.get("laser", "")).strip()
        method = str(meta.get("method", "")).strip().lower()
        phi_cut = meta.get("phi*", meta.get("phi_cut", None))

        if not tamiz_name and "Tamiz Sample" in old_part.columns and not old_part.empty:
            tamiz_name = str(old_part["Tamiz Sample"].iloc[0]).strip()
        if not laser_name and "Laser Sample" in old_part.columns and not old_part.empty:
            laser_name = str(old_part["Laser Sample"].iloc[0]).strip()
        if not method and "Hybrid Method Code" in old_part.columns and not old_part.empty:
            method = str(old_part["Hybrid Method Code"].iloc[0]).strip().lower()
        if phi_cut is None and "phi*" in old_part.columns and not old_part.empty:
            phi_cut = float(old_part["phi*"].iloc[0])
        if method not in {"m1", "m2", "m3"}:
            method = "m1"

        affected = (
            (source_db_key == "tamiz" and tamiz_name in changed_set)
            or (source_db_key == "laser" and laser_name in changed_set)
        )

        if affected:
            if not tamiz_name or not laser_name or phi_cut is None:
                raise ValueError(
                    f"Hybrid '{hybrid_name}' cannot be rebuilt because its source metadata are incomplete."
                )
            new_part, updated_meta = _build_hybrid_for_pair(
                self,
                tamiz_name,
                laser_name,
                float(phi_cut),
                method=method,
                hybrid_sample_name=str(hybrid_name),
            )
            if new_part is None or new_part.empty:
                raise ValueError(f"Hybrid '{hybrid_name}' became empty after the database correction.")
            rebuilt_parts.append(new_part)
            new_meta[str(hybrid_name)] = dict(updated_meta)
            affected_names.append(str(hybrid_name))
        else:
            rebuilt_parts.append(old_part)
            if meta:
                new_meta[str(hybrid_name)] = meta

    self.df_hybrid = pd.concat(rebuilt_parts, ignore_index=True) if rebuilt_parts else None
    self._hybrid_meta = new_meta
    if self.df_hybrid is not None and not self.df_hybrid.empty:
        try:
            self.df_hybrid_plot = self.df_hybrid.rename(columns={"wt%": "Wt"}).copy()
        except Exception:
            self.df_hybrid_plot = self.df_hybrid.copy()
    return affected_names


def _fw_edit_sync_componentry(self, db_key, sample_names=None):
    databases = getattr(self, "componentry_dbs", {}) or {}
    allowed = None if sample_names is None else set(str(v) for v in sample_names)
    summaries = []
    for key, db in list(databases.items()):
        try:
            stored_db_key, sample_name = str(key[0]), str(key[1])
        except Exception:
            continue
        if stored_db_key != str(db_key):
            continue
        if allowed is not None and sample_name not in allowed:
            continue
        gsd, _meta = _hist_get_distribution(self, stored_db_key, sample_name)
        gsd = _fw_normalize_gsd_df(gsd, normalize=False)
        if gsd.empty:
            continue
        updater = getattr(db, "update_gsd_preserving_componentry", None)
        if not callable(updater):
            raise RuntimeError(
                "The loaded Componentry-FW script does not support granulometry synchronization. "
                "Use the modified Componentry-FW script supplied with this FW version."
            )
        summary = updater(gsd)
        summaries.append((key, summary))
    return summaries


def _fw_edit_refresh_all_views(self, preferred_db=None, preferred_sample=None, selection_snapshot=None):
    old_colors = dict(getattr(self, "group_colors", {}) or {})
    try:
        self._recompute_all_params()
    except Exception:
        self._recompute_sieve_params()
        self._recompute_laser_params()
        self._recompute_hybrid_params()

    try:
        self.group_colors.update(old_colors)
    except Exception:
        pass

    if isinstance(selection_snapshot, dict):
        valid = set(getattr(self, "group_labels", {}).keys())
        for attr, previous in selection_snapshot.items():
            if hasattr(self, attr):
                kept = [key for key in previous if key in valid]
                setattr(self, attr, kept if kept else list(valid))

    try:
        self.update_console()
    except Exception:
        pass
    try:
        self._update_db_menu_actions()
    except Exception:
        pass
    try:
        refresh_histogram_selectors(
            self,
            preferred_db=preferred_db,
            preferred_sample=preferred_sample,
        )
    except Exception:
        pass
    for method_name in ("plot_xy", "plot_graphic"):
        try:
            method = getattr(self, method_name, None)
            if callable(method):
                method()
        except Exception:
            pass


def _fw_set_project_dirty(self, dirty=True):
    self.project_dirty = bool(dirty)
    title = tr("app_title")
    path = getattr(self, "project_path", None)
    if path:
        title += " — " + os.path.basename(str(path))
    if self.project_dirty:
        title += " *"
    try:
        self.setWindowTitle(title)
    except Exception:
        pass


def _fw_edit_apply_database_view(self, db_key, edited_view):
    db_key = str(db_key).strip().lower()
    preferred_db = self.cmb_hist_db.currentData() if hasattr(self, "cmb_hist_db") and self.cmb_hist_db.count() else db_key
    preferred_sample = self.cmb_hist_sample.currentText() if hasattr(self, "cmb_hist_sample") and self.cmb_hist_sample.count() else None
    selection_snapshot = {
        attr: list(getattr(self, attr, []) or [])
        for attr in (
            "selected_groups_xy", "selected_groups_ribbons", "selected_groups_graphic",
        )
    }

    if db_key == "tamiz":
        old_base = self.df_data.copy(deep=True)
        phi_col, sample_col, weight_col = old_base.columns[:3]
        group_col = getattr(self, "group_col", "Group")
    elif db_key == "laser":
        old_base = self.df_laser.copy(deep=True)
        phi_col, sample_col, weight_col, group_col = "phi", "Sample", "Wt", "Group"
    else:
        raise ValueError(f"Unsupported editable database: {db_key}")

    new_base = _fw_edit_format2_to_long(self, edited_view, db_key)
    changed_samples = _fw_edit_changed_samples(
        old_base, new_base, phi_col, sample_col, weight_col, group_col
    )
    if not changed_samples:
        QMessageBox.information(self, "Database", "No changes were detected.")
        return self._make_format2_view(
            old_base,
            phi_col=phi_col,
            sample_col=sample_col,
            weight_col=weight_col,
            group_col=group_col,
            group_row_label="Group",
        )

    reply = QMessageBox.question(
        self,
        "Normalize edited data",
        "Do you want to normalize every edited sample so that its granulometric values sum to 100%?\n\n"
        "Yes: normalize each sample to 100%.\n"
        "No: keep the entered values exactly as written.\n"
        "Cancel: return to the table without applying changes.",
        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        QMessageBox.Yes,
    )
    if reply == QMessageBox.Cancel:
        return False
    if reply == QMessageBox.Yes:
        new_base = _fw_edit_normalize_long(new_base, sample_col, weight_col)
        changed_samples = _fw_edit_changed_samples(
            old_base, new_base, phi_col, sample_col, weight_col, group_col
        )

    old_hybrid = None if getattr(self, "df_hybrid", None) is None else self.df_hybrid.copy(deep=True)
    old_hybrid_meta = dict(getattr(self, "_hybrid_meta", {}) or {})

    try:
        if db_key == "tamiz":
            self.df_data = new_base
        else:
            self.df_laser = new_base

        affected_hybrids = _fw_edit_rebuild_affected_hybrids(self, db_key, changed_samples)
        _fw_edit_sync_componentry(self, db_key, changed_samples)
        if affected_hybrids:
            _fw_edit_sync_componentry(self, "hybrid", affected_hybrids)

        _fw_edit_refresh_all_views(
            self,
            preferred_db=preferred_db,
            preferred_sample=preferred_sample,
            selection_snapshot=selection_snapshot,
        )
        _fw_edit_refresh_componentry_tab(self)
        _fw_set_project_dirty(self, True)

    except Exception:
        if db_key == "tamiz":
            self.df_data = old_base
        else:
            self.df_laser = old_base
        self.df_hybrid = old_hybrid
        self._hybrid_meta = old_hybrid_meta
        try:
            _fw_edit_sync_componentry(self, db_key, changed_samples)
            if old_hybrid is not None:
                _fw_edit_sync_componentry(self, "hybrid", None)
        except Exception:
            pass
        _fw_edit_refresh_all_views(
            self,
            preferred_db=preferred_db,
            preferred_sample=preferred_sample,
            selection_snapshot=selection_snapshot,
        )
        raise

    current = self.df_data if db_key == "tamiz" else self.df_laser
    return self._make_format2_view(
        current,
        phi_col=phi_col,
        sample_col=sample_col,
        weight_col=weight_col,
        group_col=group_col,
        group_row_label="Group",
    )


def _fw_edit_show_tamiz_db_window(self):
    if self.df_data is None or self.df_data.empty:
        return
    phi_col, sample_col, weight_col = self.df_data.columns[:3]
    group_col = getattr(self, "group_col", "Group")
    view = self._make_format2_view(
        self.df_data,
        phi_col=phi_col,
        sample_col=sample_col,
        weight_col=weight_col,
        group_col=group_col,
        group_row_label="Group",
    )
    dialog = DataBaseWindow(
        "Sieve database (editable Format 2 view)",
        view,
        self,
        editable=True,
        database_type="tamiz",
        apply_callback=lambda edited: _fw_edit_apply_database_view(self, "tamiz", edited),
    )
    dialog.exec_()


def _fw_edit_show_laser_db_window(self):
    if self.df_laser is None or self.df_laser.empty:
        return
    view = self._make_format2_view(
        self.df_laser,
        phi_col="phi",
        sample_col="Sample",
        weight_col="Wt",
        group_col="Group",
        group_row_label="Group",
    )
    dialog = DataBaseWindow(
        "Laser database (editable Format 2 view)",
        view,
        self,
        editable=True,
        database_type="laser",
        apply_callback=lambda edited: _fw_edit_apply_database_view(self, "laser", edited),
    )
    dialog.exec_()


def _fw_normalize_gsd_df(df, normalize=False):
    """Clean a phi/wt table; normalization is explicit rather than automatic."""
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["phi", "wt"])
    out = pd.DataFrame(df).loc[:, ["phi", "wt"]].copy()
    out["phi"] = pd.to_numeric(out["phi"], errors="coerce")
    out["wt"] = pd.to_numeric(out["wt"], errors="coerce")
    out = out[np.isfinite(out["phi"]) & np.isfinite(out["wt"])]
    out = out[out["wt"] >= 0]
    out = out.groupby("phi", as_index=False)["wt"].sum().sort_values("phi").reset_index(drop=True)
    total = float(out["wt"].sum()) if not out.empty else 0.0
    if normalize and total > 0 and not np.isclose(total, 100.0, atol=0.2):
        out["wt"] = out["wt"] * (100.0 / total)
    return out



# Bind editable-database behavior to the main window.
MainWindow.show_tamiz_db_window = _fw_edit_show_tamiz_db_window
MainWindow.show_laser_db_window = _fw_edit_show_laser_db_window
MainWindow._set_project_dirty = _fw_set_project_dirty

# Ensure the edited-database menu actions call the patched methods. Existing Qt
# signal connections resolve the bound method at connection time, so reconnect
# them after the normal integrated initialization has built the menus.
_fw_edit_old_mainwindow_init = MainWindow.__init__
def _fw_edit_mainwindow_init(self, *args, **kwargs):
    _fw_edit_old_mainwindow_init(self, *args, **kwargs)
    self.project_dirty = False
    try:
        self.act_viewdb_tamiz.triggered.disconnect()
    except Exception:
        pass
    try:
        self.act_viewdb_tamiz.triggered.connect(self.show_tamiz_db_window)
    except Exception:
        pass
    try:
        self.act_viewdb_laser.triggered.disconnect()
    except Exception:
        pass
    try:
        self.act_viewdb_laser.triggered.connect(self.show_laser_db_window)
    except Exception:
        pass
    _fw_set_project_dirty(self, False)
MainWindow.__init__ = _fw_edit_mainwindow_init

# Clear the dirty marker only after a project was actually written.
_fw_edit_old_save_project_to_path = _fw_save_project_to_path
def _fw_save_project_to_path(self, path):
    _fw_edit_old_save_project_to_path(self, path)
    _fw_set_project_dirty(self, False)

# Keep the title/dirty state consistent after opening a project.
_fw_edit_old_open_full_project_dialog = MainWindow.open_full_project_dialog
def _fw_edit_open_full_project_dialog(self):
    before = getattr(self, "project_path", None)
    _fw_edit_old_open_full_project_dialog(self)
    after = getattr(self, "project_path", None)
    if after and after != before:
        _fw_set_project_dirty(self, False)
MainWindow.open_full_project_dialog = _fw_edit_open_full_project_dialog



# =========================================================
# COMPONENTRY AS A FIRST-CLASS MAIN TAB (v18)
# =========================================================
# Componentry is no longer launched from Histogram.  Histogram and Componentry
# keep independent database/sample selectors.  ComponentryDB objects are still
# created lazily, only when a Componentry sample is actually activated.


def _fw_get_componentry_module_v18(self):
    cached = getattr(self, "_componentry_module", None)
    if cached is not None and int(getattr(cached, "FW_EMBEDDED_TAB_API", 0) or 0) >= 1:
        return cached

    import importlib.util as _importlib_util
    search_dirs = []
    for getter in (
        lambda: get_script_dir(),
        lambda: os.path.dirname(os.path.abspath(__file__)),
        lambda: os.getcwd(),
        lambda: "/mnt/data",
    ):
        try:
            d = getter()
            if d and d not in search_dirs:
                search_dirs.append(d)
        except Exception:
            pass

    preferred_names = [
        "Componentry-FW.py",
        "Componentry-FW_WYSIWYG_v24.py",
        "componentry_fw.py",
        "componentry-integrated.py",
        "componentry_integrated.py",
    ]

    candidates = []
    for d in search_dirs:
        for name in preferred_names:
            p = os.path.join(d, name)
            if p not in candidates:
                candidates.append(p)
    for d in search_dirs:
        try:
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                low = name.lower()
                if low.endswith(".py") and "componentry" in low:
                    p = os.path.join(d, name)
                    if p not in candidates:
                        candidates.append(p)
        except Exception:
            pass

    rejected = []
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            module_name = "fw_componentry_embedded_" + str(abs(hash(os.path.abspath(path))))
            spec = _importlib_util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                rejected.append(f"{path}: invalid Python module spec")
                continue
            mod = _importlib_util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            api = int(getattr(mod, "FW_EMBEDDED_TAB_API", 0) or 0)
            if api < 3 or not hasattr(mod, "ComponentryPreview") or not hasattr(mod, "ComponentryDB"):
                rejected.append(f"{path}: does not provide the embedded-tab API")
                continue
            self._componentry_module = mod
            return mod
        except Exception as exc:
            rejected.append(f"{path}: {exc}")

    details = "\n".join(rejected[:8])
    if details:
        details = "\n\nChecked files:\n" + details
    raise FileNotFoundError(
        "The Componentry script with embedded-tab support was not found.\n"
        "Keep Componentry-FW.py (v24 or newer) in the same folder as FW-Granulometry 2.0."
        + details
    )


# Replace both the global lookup used by legacy helper functions and the bound
# method used from MainWindow.
_fw_get_componentry_module = _fw_get_componentry_module_v18
MainWindow._fw_get_componentry_module = _fw_get_componentry_module_v18


def _fw_componentry_valid_source_keys(self):
    valid = set()
    for db_key, _label in _hist_get_available_datasets(self):
        for sample_name in _hist_get_sample_names(self, db_key):
            valid.add((str(db_key), str(sample_name)))
    return valid


def _fw_componentry_existing_sample_map(self):
    valid = _fw_componentry_valid_source_keys(self)
    out = {}
    for key, db in (getattr(self, "componentry_dbs", {}) or {}).items():
        try:
            norm_key = (str(key[0]), str(key[1]))
        except Exception:
            continue
        if norm_key in valid:
            out[_fw_componentry_display_label(self, norm_key)] = db
    return out


def _fw_componentry_update_sample_selector(self, preferred_sample=None):
    combo_db = getattr(self, "cmb_componentry_db", None)
    combo_sample = getattr(self, "cmb_componentry_sample", None)
    if combo_db is None or combo_sample is None:
        return

    db_key = combo_db.currentData() if combo_db.count() else None
    names = _hist_get_sample_names(self, db_key) if db_key else []
    old_sample = preferred_sample
    if old_sample is None and combo_sample.count():
        old_sample = combo_sample.currentText()

    combo_sample.blockSignals(True)
    combo_sample.clear()
    for name in names:
        combo_sample.addItem(str(name))
    if names:
        if old_sample in names:
            combo_sample.setCurrentText(str(old_sample))
        else:
            combo_sample.setCurrentIndex(0)
    combo_sample.blockSignals(False)

    if hasattr(self, "btn_componentry_prev"):
        self.btn_componentry_prev.setEnabled(len(names) > 1)
    if hasattr(self, "btn_componentry_next"):
        self.btn_componentry_next.setEnabled(len(names) > 1)


def _fw_refresh_componentry_selectors(self, preferred_db=None, preferred_sample=None, activate=False):
    combo_db = getattr(self, "cmb_componentry_db", None)
    combo_sample = getattr(self, "cmb_componentry_sample", None)
    if combo_db is None or combo_sample is None:
        return

    datasets = _hist_get_available_datasets(self)
    old_db = preferred_db
    if old_db is None and combo_db.count():
        old_db = combo_db.currentData()

    combo_db.blockSignals(True)
    combo_db.clear()
    for key, label in datasets:
        combo_db.addItem(str(label), str(key))
    if datasets:
        keys = [str(k) for k, _ in datasets]
        if str(old_db) in keys:
            combo_db.setCurrentIndex(keys.index(str(old_db)))
        else:
            combo_db.setCurrentIndex(0)
    combo_db.blockSignals(False)

    _fw_componentry_update_sample_selector(self, preferred_sample=preferred_sample)

    has_data = bool(datasets and combo_sample.count())
    for attr in ("cmb_componentry_db", "cmb_componentry_sample", "btn_componentry_prev", "btn_componentry_next"):
        w = getattr(self, attr, None)
        if w is not None:
            w.setEnabled(has_data if attr not in ("btn_componentry_prev", "btn_componentry_next") else (has_data and combo_sample.count() > 1))

    if activate and has_data:
        _fw_componentry_activate_selection(self)
    elif not has_data:
        view = getattr(self, "componentry_view", None)
        if view is not None:
            try:
                view.set_external_samples({}, current_sample=None, source_context={})
            except Exception:
                pass


def _fw_componentry_activate_selection(self):
    view = getattr(self, "componentry_view", None)
    combo_db = getattr(self, "cmb_componentry_db", None)
    combo_sample = getattr(self, "cmb_componentry_sample", None)
    if view is None or combo_db is None or combo_sample is None:
        return
    if combo_db.count() <= 0 or combo_sample.count() <= 0:
        try:
            view.set_external_samples({}, current_sample=None, source_context={})
        except Exception:
            pass
        return

    db_key = combo_db.currentData()
    sample_name = combo_sample.currentText().strip()
    if not db_key or not sample_name:
        return

    df_main, meta = _hist_get_distribution(self, db_key, sample_name)
    gsd = _fw_normalize_gsd_df(df_main, normalize=False)
    if gsd.empty or float(gsd["wt"].sum()) <= 0:
        QMessageBox.warning(self, "Componentry", "The selected sample has no usable phi/wt data.")
        return

    comp_mod = _fw_get_componentry_module_v18(self)
    if not hasattr(self, "componentry_dbs") or not isinstance(self.componentry_dbs, dict):
        self.componentry_dbs = {}

    key = (str(db_key), str(sample_name))
    if key not in self.componentry_dbs:
        self.componentry_dbs[key] = comp_mod.ComponentryDB(gsd)
    else:
        old_db = self.componentry_dbs[key]
        if not _fw_gsd_equivalent(getattr(old_db, "gsd", None), gsd):
            updater = getattr(old_db, "update_gsd_preserving_componentry", None)
            if callable(updater):
                updater(gsd)
            elif _fw_componentry_db_has_results(old_db):
                reply = QMessageBox.question(
                    self,
                    "Componentry",
                    "The granulometry changed and this Componentry version cannot synchronize it.\n\n"
                    "Reset the Componentry entry with the current granulometry?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
                self.componentry_dbs[key] = comp_mod.ComponentryDB(gsd)
            else:
                self.componentry_dbs[key] = comp_mod.ComponentryDB(gsd)

    sample_map = _fw_componentry_existing_sample_map(self)
    current_display = _fw_componentry_display_label(self, key)
    context = {
        "db_key": str(db_key),
        "sample_name": str(sample_name),
        "base_label": str(meta.get("db_label") or _fw_componentry_base_label(db_key)),
    }

    # The Componentry objects in sample_map are the exact same instances stored
    # by FW.  No copy is introduced by the tab integration.
    view.set_external_samples(sample_map, current_sample=current_display, source_context=context)
    try:
        self._update_db_menu_actions()
    except Exception:
        pass


def _fw_componentry_on_db_changed(self, *_args):
    _fw_componentry_update_sample_selector(self)
    _fw_componentry_activate_selection(self)


def _fw_componentry_on_sample_changed(self, *_args):
    _fw_componentry_activate_selection(self)


def _fw_componentry_move_sample(self, direction):
    combo = getattr(self, "cmb_componentry_sample", None)
    if combo is None or combo.count() <= 1:
        return
    new_index = max(0, min(combo.count() - 1, combo.currentIndex() + int(direction)))
    if new_index != combo.currentIndex():
        combo.setCurrentIndex(new_index)


def _fw_refresh_componentry_plugin_strip(self):
    """Mirror installed Componentry toolbar actions beside the host Sample selector."""
    layout = getattr(self, "componentry_plugin_strip_layout", None)
    if layout is None:
        return

    old_buttons = list(getattr(self, "_componentry_plugin_buttons", []) or [])
    for btn in old_buttons:
        try:
            layout.removeWidget(btn)
            btn.deleteLater()
        except Exception:
            pass
    self._componentry_plugin_buttons = []

    view = getattr(self, "componentry_view", None)
    if view is None:
        return

    records = list(getattr(view, "plugin_ui_toolbar_actions", []) or [])
    for rec in records:
        act = rec.get("action") if isinstance(rec, dict) else None
        if act is None:
            continue
        try:
            text_value = str(act.text()).strip()
        except Exception:
            text_value = ""
        if not text_value:
            continue

        btn = QPushButton(text_value)
        btn.setMinimumHeight(28)
        btn.setStyleSheet(
            "QPushButton { background-color: rgba(120,120,120,38); "
            "border: 1px solid rgba(100,100,100,70); border-radius: 5px; "
            "padding: 4px 10px; } "
            "QPushButton:hover { background-color: rgba(120,120,120,58); }"
        )
        try:
            tooltip = act.toolTip() or act.statusTip()
            if tooltip:
                btn.setToolTip(str(tooltip))
        except Exception:
            pass
        try:
            btn.setEnabled(bool(act.isEnabled()))
        except Exception:
            pass
        btn.clicked.connect(lambda _checked=False, a=act: a.trigger())
        layout.addWidget(btn)
        self._componentry_plugin_buttons.append(btn)


def _fw_componentry_tab_changed(self, _index=None):
    tabs = getattr(self, "tabs", None)
    tab = getattr(self, "tab_componentry", None)
    if tabs is None or tab is None or tabs.currentWidget() is not tab:
        return
    _fw_refresh_componentry_selectors(self, activate=True)


def _fw_build_componentry_tab(self):
    if hasattr(self, "tab_componentry") and getattr(self, "tab_componentry", None) is not None:
        return
    if not hasattr(self, "tabs") or self.tabs is None:
        return

    comp_mod = _fw_get_componentry_module_v18(self)

    self.tab_componentry = QWidget()
    outer = QVBoxLayout(self.tab_componentry)
    outer.setContentsMargins(6, 6, 6, 6)
    outer.setSpacing(6)

    selector_row = QHBoxLayout()
    selector_row.setContentsMargins(0, 0, 0, 0)
    selector_row.setSpacing(6)

    selector_row.addWidget(QLabel("Database:"))
    self.cmb_componentry_db = QComboBox()
    self.cmb_componentry_db.setMinimumWidth(150)
    selector_row.addWidget(self.cmb_componentry_db)

    selector_row.addSpacing(12)
    selector_row.addWidget(QLabel("Sample:"))
    self.btn_componentry_prev = QPushButton("◀")
    self.btn_componentry_prev.setFixedWidth(28)
    selector_row.addWidget(self.btn_componentry_prev)
    self.cmb_componentry_sample = QComboBox()
    self.cmb_componentry_sample.setMinimumWidth(220)
    selector_row.addWidget(self.cmb_componentry_sample, 1)
    self.btn_componentry_next = QPushButton("▶")
    self.btn_componentry_next.setFixedWidth(28)
    selector_row.addWidget(self.btn_componentry_next)

    selector_row.addSpacing(12)
    self.btn_componentry_plugins = QPushButton("Complementos")
    self.btn_componentry_plugins.setMinimumHeight(28)
    selector_row.addWidget(self.btn_componentry_plugins)

    self.componentry_plugin_strip_widget = QWidget()
    self.componentry_plugin_strip_layout = QHBoxLayout(self.componentry_plugin_strip_widget)
    self.componentry_plugin_strip_layout.setContentsMargins(0, 0, 0, 0)
    self.componentry_plugin_strip_layout.setSpacing(5)
    self._componentry_plugin_buttons = []
    selector_row.addWidget(self.componentry_plugin_strip_widget)

    selector_row.addStretch(1)
    outer.addLayout(selector_row)

    self.componentry_view = comp_mod.ComponentryPreview(
        host_app=self,
        integrated_mode=True,
        source_context={},
        embedded_mode=True,
    )

    # Componentry owns the actual QMenu/QAction objects; FW only places their
    # visible controls in this compact top row. Installed toolbar actions appear
    # only when a plugin has actually registered one.
    try:
        self.btn_componentry_plugins.setMenu(self.componentry_view.plugin_menu)
    except Exception:
        pass
    try:
        self.componentry_view.set_embedded_plugin_ui_callback(
            lambda: _fw_refresh_componentry_plugin_strip(self)
        )
    except Exception:
        _fw_refresh_componentry_plugin_strip(self)

    try:
        self.componentry_view.set_view_state(getattr(self, "componentry_view_state", {}) or {})
    except Exception:
        pass
    outer.addWidget(self.componentry_view, 1)

    # Componentry is deliberately added last: Granulometric parameters, Histogram, Graphic,
    # Componentry.
    self.tabs.addTab(self.tab_componentry, "Componentry")

    self.cmb_componentry_db.currentIndexChanged.connect(self._fw_componentry_on_db_changed)
    self.cmb_componentry_sample.currentIndexChanged.connect(self._fw_componentry_on_sample_changed)
    self.btn_componentry_prev.clicked.connect(lambda: self._fw_componentry_move_sample(-1))
    self.btn_componentry_next.clicked.connect(lambda: self._fw_componentry_move_sample(+1))
    self.tabs.currentChanged.connect(self._fw_componentry_tab_changed)

    _fw_refresh_componentry_selectors(self, activate=False)


def _fw_open_componentry_tab(self):
    tab = getattr(self, "tab_componentry", None)
    tabs = getattr(self, "tabs", None)
    if tab is None or tabs is None:
        return
    idx = tabs.indexOf(tab)
    if idx >= 0:
        tabs.setCurrentIndex(idx)
        _fw_componentry_tab_changed(self, idx)


# Bind tab-specific methods.  The compatibility name remains available so an
# old menu/plugin action opens the Componentry tab instead of spawning a window.
MainWindow._fw_refresh_componentry_selectors = _fw_refresh_componentry_selectors
MainWindow._fw_componentry_activate_selection = _fw_componentry_activate_selection
MainWindow._fw_componentry_on_db_changed = _fw_componentry_on_db_changed
MainWindow._fw_componentry_on_sample_changed = _fw_componentry_on_sample_changed
MainWindow._fw_componentry_move_sample = _fw_componentry_move_sample
MainWindow._fw_refresh_componentry_plugin_strip = _fw_refresh_componentry_plugin_strip
MainWindow._fw_componentry_tab_changed = _fw_componentry_tab_changed
MainWindow.open_componentry_for_current_histogram = _fw_open_componentry_tab


# Keep the Componentry selector catalog refreshed whenever the main granulometry
# selectors are rebuilt, without creating ComponentryDB objects in the background.
_fw_v18_old_refresh_histogram_selectors = refresh_histogram_selectors
def refresh_histogram_selectors(self, preferred_db=None, preferred_sample=None):
    _fw_v18_old_refresh_histogram_selectors(self, preferred_db=preferred_db, preferred_sample=preferred_sample)
    try:
        _fw_refresh_componentry_selectors(self, activate=False)
    except Exception:
        pass
MainWindow.refresh_histogram_selectors = refresh_histogram_selectors


# Persist the live embedded Componentry view state before the existing full
# project serializer runs. Graphic's project wrapper is installed afterwards and
# therefore composes on top of this wrapper.
_fw_v18_old_collect_project_dict = _fw_collect_project_dict
def _fw_collect_project_dict(self):
    view = getattr(self, "componentry_view", None)
    if view is not None:
        try:
            state = view.get_view_state()
            if isinstance(state, dict):
                self.componentry_view_state = {
                    "plot_style": dict(state.get("plot_style", {}) or {}),
                    "palette_state": dict(state.get("palette_state", {}) or {}),
                }
        except Exception:
            pass
    return _fw_v18_old_collect_project_dict(self)


# After loading a project, rebuild the independent Componentry selectors and, if
# that tab is visible, bind the embedded view to the newly restored DB objects.
_fw_v18_old_apply_project_dict = _fw_apply_project_dict
def _fw_apply_project_dict(self, data):
    _fw_v18_old_apply_project_dict(self, data)
    try:
        active = bool(
            hasattr(self, "tabs") and hasattr(self, "tab_componentry")
            and self.tabs.currentWidget() is self.tab_componentry
        )
        _fw_refresh_componentry_selectors(self, activate=active)
    except Exception:
        pass


# Database editing used to refresh a list of floating Componentry windows.  In
# tab mode there is one persistent view, so refresh only that view and preserve
# its current selection.
def _fw_edit_refresh_componentry_tab(self):
    view = getattr(self, "componentry_view", None)
    if view is None:
        return
    try:
        active_db = self.cmb_componentry_db.currentData() if self.cmb_componentry_db.count() else None
        active_sample = self.cmb_componentry_sample.currentText().strip() if self.cmb_componentry_sample.count() else None
        _fw_refresh_componentry_selectors(
            self,
            preferred_db=active_db,
            preferred_sample=active_sample,
            activate=(getattr(self, "tabs", None) is not None and self.tabs.currentWidget() is getattr(self, "tab_componentry", None)),
        )
    except Exception:
        pass


try:
    install_graphic_support(MainWindow)
except Exception:
    traceback.print_exc()


# Build Componentry only after every earlier MainWindow/UI wrapper (including
# Graphic) has completed.  This guarantees the stable final tab order.
_fw_v18_old_mainwindow_init = MainWindow.__init__
def _fw_v18_mainwindow_init(self, *args, **kwargs):
    _fw_v18_old_mainwindow_init(self, *args, **kwargs)
    try:
        _fw_build_componentry_tab(self)
    except Exception as exc:
        # Keep the rest of FW usable, but make the integration failure explicit.
        try:
            QMessageBox.warning(self, "Componentry", f"Could not create the Componentry tab:\n{exc}")
        except Exception:
            pass
MainWindow.__init__ = _fw_v18_mainwindow_init

# =========================================================
# Groups, symbols, View plot and exact export (Block 6D) + project persistence
# =========================================================
MainWindow._fw_effective_group = _fw_effective_group
MainWindow._fw_group_bases = _fw_group_bases
MainWindow._fw_ensure_group_colors = _fw_ensure_group_colors
MainWindow._fw_refresh_group_views = _fw_refresh_group_views
MainWindow.open_xy_view = _fw_open_xy_view
MainWindow.export_xy_plot = lambda self: fw_export_xy(self, self)
MainWindow.select_groups_xy = lambda self: _fw_open_groups_dialog(self, True)
MainWindow.edit_colors = lambda self: _fw_open_groups_dialog(self, False)

_fw_v19_old_collect_project_dict = _fw_collect_project_dict
def _fw_collect_project_dict(self):
    data = _fw_v19_old_collect_project_dict(self)
    fw = data.setdefault("fw", {})
    fw["group_mode"] = str(getattr(self, "group_mode", "labels") or "labels")
    fw["custom_groups"] = dict(getattr(self, "custom_groups", None) or {})
    fw["group_markers"] = dict(getattr(self, "group_markers", None) or {})
    fw["xy_style"] = dict(_fw_xy_style(self))
    if hasattr(self, "chk_show_names"):
        fw["xy_show_names"] = bool(self.chk_show_names.isChecked())
    return data


_fw_v19_old_apply_project_dict = _fw_apply_project_dict
def _fw_apply_project_dict(self, data):
    fw = (data.get("fw", {}) or {}) if isinstance(data, dict) else {}
    self.group_mode = "custom" if fw.get("group_mode") == "custom" else "labels"
    cg = fw.get("custom_groups")
    self.custom_groups = {str(k): str(v) for k, v in cg.items()} if isinstance(cg, dict) else {}
    gm = fw.get("group_markers")
    self.group_markers = {str(k): str(v) for k, v in gm.items()} if isinstance(gm, dict) else {}
    if isinstance(fw.get("xy_style"), dict):
        st = dict(FW_XY_STYLE_DEFAULTS)
        st.update({k: v for k, v in fw["xy_style"].items() if k in FW_XY_STYLE_DEFAULTS})
        self.xy_style = st
    if "xy_show_names" in fw and hasattr(self, "chk_show_names"):
        self.chk_show_names.blockSignals(True)
        self.chk_show_names.setChecked(bool(fw.get("xy_show_names")))
        self.chk_show_names.blockSignals(False)
    self._fw_known_group_keys = None
    _fw_v19_old_apply_project_dict(self, data)
    try:
        self._fw_ensure_group_colors()
        self._fw_known_group_keys = list((getattr(self, "group_labels", None) or {}).keys())
        dlg = getattr(self, "_xy_view_dlg", None)
        if dlg is not None:
            dlg._load_controls()
        self._fw_refresh_group_views()
    except Exception:
        traceback.print_exc()


# =========================================================
# Mouse wheel never changes values (whole program, plugins included)
# =========================================================
# Spin boxes, combo boxes, sliders and tab bars ignore the wheel; the wheel scrolls
# the panel/list that contains them instead (or zooms, in the plots that zoom).
# Installed once on the QApplication, so it covers every window of FW, Componentry,
# Graphic, GdeInput and FractalAnalysis that runs inside this program.
from PyQt5.QtCore import QObject as _FwQObject
from PyQt5.QtWidgets import (
    QAbstractSpinBox as _FwSpin, QComboBox as _FwCombo, QAbstractSlider as _FwSlider,
    QScrollBar as _FwScrollBar, QTabBar as _FwTabBarW, QAbstractScrollArea as _FwScrollArea
)


class _FwNoWheelValues(_FwQObject):
    def eventFilter(self, obj, event):
        try:
            if event.type() != QEvent.Wheel:
                return False
            if not isinstance(obj, (_FwSpin, _FwCombo, _FwSlider, _FwTabBarW)) or isinstance(obj, _FwScrollBar):
                return False
            # pass the wheel to the nearest scrollable container
            p = obj.parentWidget()
            while p is not None and not isinstance(p, _FwScrollArea):
                p = p.parentWidget()
            if p is not None:
                d = event.angleDelta()
                horizontal = abs(d.x()) > abs(d.y())
                bar = p.horizontalScrollBar() if horizontal else p.verticalScrollBar()
                if bar is not None and bar.isVisible():
                    QApplication.sendEvent(bar, event)
            return True        # the widget never receives it: its value does not change
        except Exception:
            return False


def _fw_install_no_wheel_values():
    app = QApplication.instance()
    if app is None or getattr(app, "_fw_no_wheel_values", None) is not None:
        return
    flt = _FwNoWheelValues(app)
    app.installEventFilter(flt)
    app._fw_no_wheel_values = flt


_fw_v19_old_mainwindow_init = MainWindow.__init__
def _fw_v19_mainwindow_init(self, *args, **kwargs):
    try:
        _fw_install_no_wheel_values()
    except Exception:
        pass
    _fw_v19_old_mainwindow_init(self, *args, **kwargs)
MainWindow.__init__ = _fw_v19_mainwindow_init


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

# endregion