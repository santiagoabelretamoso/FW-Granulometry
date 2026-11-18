FW_EMBEDDED_TAB_API = 3

# PACKAGE_BUILD_MARKER: COMPONENTRY_WYSIWYG_PREVIEW_V19
#region BLOQUE 1 Imports
import sys
import os
import json
import sqlite3
import importlib.util
import traceback
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

from PyQt5.QtCore import Qt, QTimer, QUrl, QEvent
from PyQt5.QtGui import (
    QStandardItemModel, QStandardItem, QColor, QFont, QDesktopServices,
    QIcon, QPixmap, QPainter, QPen
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QColorDialog, QGroupBox, QLineEdit, QTabWidget,
    QAbstractItemView, QListWidget, QListWidgetItem,
    QFileDialog, QSizePolicy, QSpinBox, QDoubleSpinBox,
    QAction, QToolBar, QDockWidget, QTextEdit, QDialogButtonBox,
    QScrollArea, QHeaderView, QSplitter
)
#endregion


class ReorderableComponentTable(QTableWidget):
    """QTableWidget with manual, non-destructive whole-row reordering.

    Important design choice: Qt's built-in InternalMove is deliberately NOT used.
    With QTableWidget it can remove/move individual QTableWidgetItems after the
    custom callback has already rebuilt the row, which can make cells appear to
    vanish.  Reordering is handled only from mouse press/move/release events and
    the host callback receives source/target row indices.  The table model itself
    is never modified by Qt drag-and-drop.

    Dragging starts only from column 0 (Component).  This avoids interfering with
    the color cell and the Palette/Type comboboxes.  The arrow buttons remain the
    accessible/non-drag alternative.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reorder_callback = None

        # Disable Qt's model-level drag/drop completely.  This is the key safety
        # measure: no implicit MoveAction can delete source items after our code
        # has already reordered the table.
        self.setDragEnabled(False)
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self.setDropIndicatorShown(False)
        self.setDragDropOverwriteMode(False)
        self.setDragDropMode(QAbstractItemView.NoDragDrop)

        self._manual_drag_source_row = -1
        self._manual_drag_target_row = -1
        self._manual_drag_start_pos = None
        self._manual_drag_active = False

    def _reset_manual_drag(self):
        self._manual_drag_source_row = -1
        self._manual_drag_target_row = -1
        self._manual_drag_start_pos = None
        self._manual_drag_active = False
        try:
            self.viewport().unsetCursor()
        except Exception:
            pass

    def _row_from_y_for_drop(self, y, source_row):
        """Translate mouse Y position to final row index after removing source."""
        n = self.rowCount()
        if n <= 0:
            return -1

        row = self.rowAt(int(y))
        if row < 0:
            # Above first row -> top; below last row -> bottom.
            first = self.visualItemRect(self.item(0, 0)) if self.item(0, 0) else None
            if first is not None and y < first.top():
                insertion = 0
            else:
                insertion = n
        else:
            item = self.item(row, 0)
            rect = self.visualItemRect(item) if item is not None else None
            insertion = row
            if rect is not None and y > rect.center().y():
                insertion += 1

        # The source row will be removed before insertion, so positions below it
        # shift upward by one.
        if insertion > source_row:
            insertion -= 1

        return max(0, min(int(insertion), n - 1))

    def mousePressEvent(self, event):
        # Let QTableWidget perform normal selection/edit handling first.
        super().mousePressEvent(event)

        self._reset_manual_drag()
        if event.button() != Qt.LeftButton:
            return

        index = self.indexAt(event.pos())
        if not index.isValid() or index.column() != 0:
            return

        # Never initiate a drag while an in-place text editor is active.
        if self.state() == QAbstractItemView.EditingState:
            return

        self._manual_drag_source_row = int(index.row())
        self._manual_drag_target_row = int(index.row())
        self._manual_drag_start_pos = event.pos()

    def mouseMoveEvent(self, event):
        if (
            self._manual_drag_source_row < 0
            or self._manual_drag_start_pos is None
            or not (event.buttons() & Qt.LeftButton)
        ):
            super().mouseMoveEvent(event)
            return

        if not self._manual_drag_active:
            distance = (event.pos() - self._manual_drag_start_pos).manhattanLength()
            if distance < QApplication.startDragDistance():
                super().mouseMoveEvent(event)
                return
            self._manual_drag_active = True
            try:
                self.viewport().setCursor(Qt.ClosedHandCursor)
            except Exception:
                pass

        target = self._row_from_y_for_drop(event.pos().y(), self._manual_drag_source_row)
        if target >= 0:
            self._manual_drag_target_row = target
            # Visual feedback only.  The source row is stored separately, so
            # changing current selection here cannot change what gets moved.
            try:
                self.setCurrentCell(target, 0)
                self.selectRow(target)
            except Exception:
                pass

        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._manual_drag_active:
            source = int(self._manual_drag_source_row)
            target = self._row_from_y_for_drop(event.pos().y(), source)
            callback = self.reorder_callback
            self._reset_manual_drag()

            if target >= 0 and target != source and callable(callback):
                callback(source, target)
            else:
                try:
                    self.setCurrentCell(max(0, source), 0)
                    self.selectRow(max(0, source))
                except Exception:
                    pass

            event.accept()
            return

        self._reset_manual_drag()
        super().mouseReleaseEvent(event)


#region BLOQUE 2 Matplotlib style (modern sans-serif, Arial-preferred)
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["axes.linewidth"] = 1.0
#endregion


#region BLOQUE 3 No default fallback data
# The program now starts with no sample loaded.
# No default/example GSD is defined here.
#endregion


#region BLOQUE 4 Constants
PASTEL = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
    "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
    "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
    "#a6cee3", "#b2df8a", "#cab2d6", "#fdbf6f",
]

# Palette catalogue used by the Componentry editor.  Discrete palettes retain
# the supplied swatches; continuous palettes are sampled from Matplotlib.
COMPONENT_PALETTES = {
    # Pastel / discrete palettes inspired by the examples shown in the UI brief.
    "Pastel — Set3": {
        "category": "Pastel / discrete", "kind": "discrete", "colors": PASTEL,
    },
    "Pastel — Dreamy hues": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#d7d9d9", "#c8bfd8", "#d2a2db", "#bd83d8", "#9c6bd0"],
    },
    "Pastel — Sunny sand dunes": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#e6cfad", "#f1d3bd", "#f8ece4", "#eeece6", "#d6d1bf"],
    },
    "Pastel — Blushing marshmallow": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#dfa6a1", "#f3b3af", "#f8d4d2", "#fae7e7", "#fbf8f8"],
    },
    "Pastel — Sunny lemonade": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#bdc99a", "#d9dfa1", "#fff0aa", "#f1cd73", "#cb955b"],
    },
    "Pastel — Whimsical rainbow": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#a8ddf2", "#a8c9ed", "#abb9ee", "#b2a9ef", "#ba99ec", "#c186e9", "#c976e8", "#cf63e7"],
    },
    "Pastel — Meadow sunflower": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#d1df91", "#f2efcb", "#c5d9b8", "#ed9874"],
    },
    "Pastel — Golden hour": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#f5a553", "#f4b250", "#f5be50", "#f5c94f", "#f5d44e", "#f6df4d", "#f7e84d"],
    },
    "Pastel — Lavender rose": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#9e87ee", "#b79df0", "#efb4dc", "#ed9abb", "#eb7ea1"],
    },
    "Pastel — Soft coastal": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#ff9b92", "#f6efd4", "#a9e5da", "#9bcbd4", "#b9c7db", "#d8c3e6"],
    },
    "Pastel — Mint and coral": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#b8e0d2", "#d6eadf", "#eac4d5", "#f7d1cd", "#f4acb7", "#9d8189"],
    },
    "Pastel — Earth and cream": {
        "category": "Pastel / discrete", "kind": "discrete",
        "colors": ["#d9d7cf", "#f7f3ea", "#e8d4bd", "#c9b79c", "#a98d74", "#8d6e63"],
    },

    # Curated qualitative palettes for componentry.  These palettes are
    # intentionally discrete and live together under Family = Others so the
    # user can browse alternatives designed for categorical components.
    "Okabe–Ito": {
        "category": "Others", "kind": "discrete",
        "colors": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
    },
    "Tol — Bright": {
        "category": "Others", "kind": "discrete",
        "colors": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
    },
    "Tol — Muted": {
        "category": "Others", "kind": "discrete",
        "colors": ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"],
    },
    "ColorBrewer — Set2": {
        "category": "Others", "kind": "discrete",
        "colors": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    },
    "Tableau — Classic 10": {
        "category": "Others", "kind": "discrete",
        "colors": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"],
    },
    "Tol — Vibrant": {
        "category": "Others", "kind": "discrete",
        "colors": ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"],
    },
    "Tol — Light": {
        "category": "Others", "kind": "discrete",
        "colors": ["#77AADD", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00", "#EEDD88", "#EE8866", "#FFAABB", "#DDDDDD"],
    },
    "ColorBrewer — Set1": {
        "category": "Others", "kind": "discrete",
        "colors": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"],
    },
    "ColorBrewer — Dark2": {
        "category": "Others", "kind": "discrete",
        "colors": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"],
    },
    "ColorBrewer — Accent": {
        "category": "Others", "kind": "discrete",
        "colors": ["#7FC97F", "#BEAED4", "#FDC086", "#FFFF99", "#386CB0", "#F0027F", "#BF5B17", "#666666"],
    },
    "ColorBrewer — Paired": {
        "category": "Others", "kind": "discrete",
        "colors": ["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"],
    },
    "Plotly — Bold": {
        "category": "Others", "kind": "discrete",
        "colors": ["#7F3C8D", "#11A579", "#3969AC", "#F2B701", "#E73F74", "#80BA5A", "#E68310", "#008695", "#CF1C90", "#F97B72", "#4B4B8F", "#A5AA99"],
    },
    "Plotly — Vivid": {
        "category": "Others", "kind": "discrete",
        "colors": ["#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C", "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#CC3A8E", "#A5AA99"],
    },
    "Plotly — Safe": {
        "category": "Others", "kind": "discrete",
        "colors": ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", "#44AA99", "#999933", "#882255", "#661100", "#6699CC", "#888888"],
    },
    "Distinct — Mineral": {
        "category": "Others", "kind": "discrete",
        "colors": ["#2F5597", "#D97706", "#2A9D8F", "#C43D3D", "#7A5195", "#8A9A33", "#2B8CBE", "#B56576", "#6B4F2A", "#C9A227"],
    },

    # Categorical palettes with high component-to-component separation.
    "Categorical — Tableau 10": {
        "category": "Categorical", "kind": "continuous", "mpl": "tab10",
    },
    "Categorical — Tableau 20": {
        "category": "Categorical", "kind": "continuous", "mpl": "tab20",
    },
    "Categorical — Set1": {
        "category": "Categorical", "kind": "continuous", "mpl": "Set1",
    },
    "Categorical — Set2": {
        "category": "Categorical", "kind": "continuous", "mpl": "Set2",
    },
    "Categorical — Paired": {
        "category": "Categorical", "kind": "continuous", "mpl": "Paired",
    },
    "Categorical — Accent": {
        "category": "Categorical", "kind": "continuous", "mpl": "Accent",
    },
    "Categorical — Dark2": {
        "category": "Categorical", "kind": "continuous", "mpl": "Dark2",
    },

    # Sequential scientific palettes.
    "Sequential — Viridis": {"category": "Sequential", "kind": "continuous", "mpl": "viridis"},
    "Sequential — Plasma": {"category": "Sequential", "kind": "continuous", "mpl": "plasma"},
    "Sequential — Inferno": {"category": "Sequential", "kind": "continuous", "mpl": "inferno"},
    "Sequential — Magma": {"category": "Sequential", "kind": "continuous", "mpl": "magma"},
    "Sequential — Cividis": {"category": "Sequential", "kind": "continuous", "mpl": "cividis"},
    "Sequential — Turbo": {"category": "Sequential", "kind": "continuous", "mpl": "turbo"},
    "Sequential — Blues": {"category": "Sequential", "kind": "continuous", "mpl": "Blues"},
    "Sequential — Greens": {"category": "Sequential", "kind": "continuous", "mpl": "Greens"},
    "Sequential — Oranges": {"category": "Sequential", "kind": "continuous", "mpl": "Oranges"},
    "Sequential — Purples": {"category": "Sequential", "kind": "continuous", "mpl": "Purples"},
    "Sequential — YlGnBu": {"category": "Sequential", "kind": "continuous", "mpl": "YlGnBu"},
    "Sequential — Cubehelix": {"category": "Sequential", "kind": "continuous", "mpl": "cubehelix"},

    # Diverging palettes.
    "Diverging — Coolwarm": {"category": "Diverging", "kind": "continuous", "mpl": "coolwarm"},
    "Diverging — Spectral": {"category": "Diverging", "kind": "continuous", "mpl": "Spectral"},
    "Diverging — RdYlBu": {"category": "Diverging", "kind": "continuous", "mpl": "RdYlBu"},
    "Diverging — RdBu": {"category": "Diverging", "kind": "continuous", "mpl": "RdBu"},
    "Diverging — BrBG": {"category": "Diverging", "kind": "continuous", "mpl": "BrBG"},
    "Diverging — PiYG": {"category": "Diverging", "kind": "continuous", "mpl": "PiYG"},
    "Diverging — PuOr": {"category": "Diverging", "kind": "continuous", "mpl": "PuOr"},
}

DEFAULT_PALETTE_NAME = "Pastel — Set3"
DEFAULT_PALETTE_CATEGORY = COMPONENT_PALETTES[DEFAULT_PALETTE_NAME]["category"]

TYPE_OPTIONS = ["Juvenile", "Lithic", "Accessory"]
TYPE_COLORS = {
    "Juvenile": "#8dd3c7",
    "Lithic":   "#fdb462",
    "Accessory":"#bebada",
}

PROJECT_SCHEMA_VERSION = 2
UNDEFINED_COMPONENT = "Undefined"

COMPONENTRY_HIST_DEFAULT_STYLE = {
    # bars
    "bar_width_factor": 0.85,
    "bar_alpha": 1.0,
    "unclassified_facecolor": "#8c8c8c",
    "undefined_facecolor": "#cfcfcf",
    "bar_outer_edgecolor": "#000000",
    "bar_outer_edgewidth": 1.2,
    "segment_edgecolor": "#000000",
    "segment_edgewidth": 0.0,

    # axes and ticks
    "axes_color": "#000000",
    "axes_linewidth": 1.6,
    "tick_color": "#000000",
    "tick_width": 1.2,
    "tick_length": 7.0,

    # cumulative curve
    "show_cumulative": True,
    "cum_color": "#000000",
    "cum_linewidth": 1.8,
    "cum_linestyle": "-",
    "cum_marker": "s",
    "cum_markersize": 5.5,
    "cum_marker_facecolor": "#ffffff",
    "cum_marker_edgecolor": "#000000",
    "cum_marker_edgewidth": 1.2,
    "cumulative_layer": "front",

    # frequency polygon
    "show_polygon": False,
    "poly_color": "#000000",
    "poly_linewidth": 1.4,
    "poly_linestyle": "-",
    "poly_marker": "o",
    "poly_markersize": 4.5,

    # axis modes and labels
    "show_um_axis": True,
    "reverse_x": False,
    "x_axis_mode": "phi",
    "y_right_mode": "cumulative",
    "um_axis_position": "top",
    "x_label_phi": "",
    "x_label_um": "",
    "y_label_left": "",
    "y_label_right": "",
    "um_secondary_label": "",

    # grid
    "grid_visible": False,
    "grid_color": "#b0b0b0",
    "grid_linewidth": 0.8,
    "grid_linestyle": "--",
    "grid_alpha": 0.35,

    # Figure and fonts. Width/height are the single source of truth for both
    # the on-screen preview and exported files.  The preview uses a lower DPI,
    # but keeps the exact same physical figure size, aspect ratio and layout.
    "figure_width": 11.5,
    "figure_height": 6.4,
    "title_fontsize": 19.0,
    "axis_label_fontsize": 13.0,
    "right_axis_label_fontsize": 11.0,
    "tick_fontsize": 9.5,
    "legend_fontsize": 8.5,
    "title_fontweight": "bold",
    "axis_label_fontweight": "bold",

    # legend
    "legend_frame": False,
    "legend_framecolor": "#000000",
    "legend_framewidth": 1.0,
    "legend_alpha": 1.0,

    # Folk & Ward markers and text
    "fw_median_color": "#333333",
    "fw_median_linewidth": 1.0,
    "fw_outer_color": "#777777",
    "fw_outer_linewidth": 0.8,
    "fw_line_alpha": 0.8,
    "fw_band_color": "#000000",
    "fw_band_alpha": 0.06,
    "fw_text_fontsize": 9.5,
    "fw_box_facecolor": "#ffffff",
    "fw_box_edgecolor": "#888888",
    "fw_box_linewidth": 1.0,
    "fw_box_alpha": 0.85,
}

LINESTYLE_OPTIONS = {
    "Solid": "-",
    "Dashed": "--",
    "Dotted": ":",
    "Dash-dot": "-.",
}

MARKER_OPTIONS = {
    "None": "None",
    "Square": "s",
    "Circle": "o",
    "Triangle": "^",
    "Diamond": "D",
    "Cross": "x",
    "Plus": "+",
}

# =========================================================
# Plugin system constants
# =========================================================
PLUGIN_API_VERSION = 1
PLUGIN_FOLDER_NAME = "componentry_plugins"
PLUGIN_MANIFEST_NAME = "plugin.json"
PLUGIN_ENTRY_DEFAULT = "main.py"
PLUGIN_CLASS_DEFAULT = "Plugin"

PLUGIN_MENU_TITLE = "Complementos"
PLUGIN_TOOLBAR_TITLE = "Complementos"
#endregion


#region BLOQUE 5 Helpers
def _to_float(x):
    try:
        return float(str(x).strip().replace(",", "."))
    except Exception:
        return np.nan


def _safe_float(x, default=0.0):
    v = _to_float(x)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def phi_to_um(phi):
    return 1000.0 * (2.0 ** (-float(phi)))



def _normalize_hex_color(value, fallback="#808080"):
    q = QColor(str(value))
    if q.isValid():
        return q.name()
    qf = QColor(str(fallback))
    return qf.name() if qf.isValid() else "#808080"


def _palette_categories():
    out = []
    for spec in COMPONENT_PALETTES.values():
        cat = str(spec.get("category", "Other"))
        if cat not in out:
            out.append(cat)
    return out


def _palette_names_for_category(category):
    return [
        name for name, spec in COMPONENT_PALETTES.items()
        if str(spec.get("category", "Other")) == str(category)
    ]


def _palette_base_swatches(palette_name, reverse=False, continuous_count=12):
    spec = COMPONENT_PALETTES.get(palette_name) or COMPONENT_PALETTES[DEFAULT_PALETTE_NAME]
    if spec.get("kind") == "discrete":
        colors = [_normalize_hex_color(c) for c in spec.get("colors", [])]
    else:
        cmap_name = str(spec.get("mpl", "viridis"))
        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = matplotlib.colormaps.get_cmap("viridis")
        n = max(2, int(continuous_count))
        positions = np.linspace(0.05, 0.95, n)
        colors = [matplotlib.colors.to_hex(cmap(float(p)), keep_alpha=False) for p in positions]
    if not colors:
        colors = list(PASTEL)
    if reverse:
        colors = list(reversed(colors))
    return colors


def component_palette_colors(palette_name, count, reverse=False):
    count = max(1, int(count))
    spec = COMPONENT_PALETTES.get(palette_name) or COMPONENT_PALETTES[DEFAULT_PALETTE_NAME]
    if spec.get("kind") == "continuous":
        cmap_name = str(spec.get("mpl", "viridis"))
        try:
            cmap = matplotlib.colormaps.get_cmap(cmap_name)
        except Exception:
            cmap = matplotlib.colormaps.get_cmap("viridis")
        positions = np.linspace(0.05, 0.95, count)
        colors = [matplotlib.colors.to_hex(cmap(float(p)), keep_alpha=False) for p in positions]
    else:
        base = [_normalize_hex_color(c) for c in spec.get("colors", [])]
        if not base:
            base = list(PASTEL)
        if count <= len(base):
            # Evenly distribute selections when fewer components than swatches.
            idx = np.linspace(0, len(base) - 1, count).round().astype(int)
            colors = [base[int(i)] for i in idx]
        else:
            # Preserve all supplied colors, then interpolate only when more colors
            # are needed than the palette explicitly contains.
            try:
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "componentry_dynamic_palette", base, N=count
                )
                positions = np.linspace(0.0, 1.0, count)
                colors = [matplotlib.colors.to_hex(cmap(float(p)), keep_alpha=False) for p in positions]
            except Exception:
                colors = [base[i % len(base)] for i in range(count)]
    if reverse:
        colors = list(reversed(colors))
    return colors


def default_component_color(index):
    colors = component_palette_colors(DEFAULT_PALETTE_NAME, max(1, int(index) + 1), False)
    return colors[int(index) % len(colors)]


def _color_icon(color_hex, width=34, height=16):
    pix = QPixmap(int(width), int(height))
    pix.fill(QColor(_normalize_hex_color(color_hex)))
    painter = QPainter(pix)
    painter.setPen(QPen(QColor("#666666"), 1))
    painter.drawRect(0, 0, pix.width() - 1, pix.height() - 1)
    painter.end()
    return QIcon(pix)


def _palette_preview_pixmap(colors, width=320, height=28):
    colors = list(colors or ["#808080"])
    pix = QPixmap(int(width), int(height))
    pix.fill(QColor("#ffffff"))
    painter = QPainter(pix)
    n = max(1, len(colors))
    for i, color in enumerate(colors):
        x0 = int(round(i * width / n))
        x1 = int(round((i + 1) * width / n))
        painter.fillRect(x0, 0, max(1, x1 - x0), height, QColor(_normalize_hex_color(color)))
    painter.setPen(QPen(QColor("#777777"), 1))
    painter.drawRect(0, 0, pix.width() - 1, pix.height() - 1)
    painter.end()
    return pix


def _style_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def normalize_componentry_hist_style(style=None):
    out = dict(COMPONENTRY_HIST_DEFAULT_STYLE)
    if isinstance(style, dict):
        out.update(style)

    color_keys = [
        "unclassified_facecolor", "undefined_facecolor",
        "bar_outer_edgecolor", "segment_edgecolor", "axes_color", "tick_color",
        "cum_color", "cum_marker_facecolor", "cum_marker_edgecolor",
        "poly_color", "grid_color", "legend_framecolor",
        "fw_median_color", "fw_outer_color", "fw_band_color",
        "fw_box_facecolor", "fw_box_edgecolor",
    ]
    for key in color_keys:
        out[key] = _normalize_hex_color(out.get(key), COMPONENTRY_HIST_DEFAULT_STYLE[key])

    float_ranges = {
        "bar_width_factor": (0.05, 1.50),
        "bar_alpha": (0.0, 1.0),
        "bar_outer_edgewidth": (0.0, 10.0),
        "segment_edgewidth": (0.0, 10.0),
        "axes_linewidth": (0.1, 10.0),
        "tick_width": (0.1, 10.0),
        "tick_length": (0.0, 30.0),
        "cum_linewidth": (0.0, 10.0),
        "cum_markersize": (0.0, 30.0),
        "cum_marker_edgewidth": (0.0, 10.0),
        "poly_linewidth": (0.0, 10.0),
        "poly_markersize": (0.0, 30.0),
        "grid_linewidth": (0.0, 10.0),
        "grid_alpha": (0.0, 1.0),
        "figure_width": (4.0, 40.0),
        "figure_height": (3.0, 40.0),
        "title_fontsize": (6.0, 60.0),
        "axis_label_fontsize": (6.0, 50.0),
        "right_axis_label_fontsize": (6.0, 50.0),
        "tick_fontsize": (5.0, 40.0),
        "legend_fontsize": (5.0, 40.0),
        "legend_framewidth": (0.0, 10.0),
        "legend_alpha": (0.0, 1.0),
        "fw_median_linewidth": (0.0, 10.0),
        "fw_outer_linewidth": (0.0, 10.0),
        "fw_line_alpha": (0.0, 1.0),
        "fw_band_alpha": (0.0, 1.0),
        "fw_text_fontsize": (5.0, 40.0),
        "fw_box_linewidth": (0.0, 10.0),
        "fw_box_alpha": (0.0, 1.0),
    }
    for key, (lo, hi) in float_ranges.items():
        try:
            value = float(out.get(key, COMPONENTRY_HIST_DEFAULT_STYLE[key]))
        except Exception:
            value = float(COMPONENTRY_HIST_DEFAULT_STYLE[key])
        out[key] = min(hi, max(lo, value))

    bool_keys = [
        "show_cumulative", "show_polygon", "show_um_axis", "reverse_x",
        "grid_visible", "legend_frame",
    ]
    for key in bool_keys:
        out[key] = _style_bool(out.get(key), COMPONENTRY_HIST_DEFAULT_STYLE[key])

    enum_defaults = {
        "cum_linestyle": set(LINESTYLE_OPTIONS.values()),
        "poly_linestyle": set(LINESTYLE_OPTIONS.values()),
        "grid_linestyle": set(LINESTYLE_OPTIONS.values()),
        "cum_marker": set(MARKER_OPTIONS.values()),
        "poly_marker": set(MARKER_OPTIONS.values()),
        "cumulative_layer": {"front", "back"},
        "x_axis_mode": {"phi", "um"},
        "y_right_mode": {"cumulative", "none"},
        "um_axis_position": {"top", "bottom"},
        "title_fontweight": {"normal", "bold"},
        "axis_label_fontweight": {"normal", "bold"},
    }
    for key, valid in enum_defaults.items():
        value = str(out.get(key, COMPONENTRY_HIST_DEFAULT_STYLE[key]))
        if value not in valid:
            value = str(COMPONENTRY_HIST_DEFAULT_STYLE[key])
        out[key] = value

    for key in ["x_label_phi", "x_label_um", "y_label_left", "y_label_right", "um_secondary_label"]:
        out[key] = str(out.get(key, "") or "")

    return out


def _componentry_tick_positions(phi_values):
    vals = np.asarray(phi_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=float)
    uniq = np.unique(np.round(vals, 8))
    if uniq.size == 1:
        return uniq
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 1e-9]
    step = float(np.nanmin(diffs)) if diffs.size else 1.0
    start = np.floor(float(np.nanmin(uniq)) / step) * step
    stop = np.ceil(float(np.nanmax(uniq)) / step) * step
    n = int(round((stop - start) / step)) + 1
    if n <= 0 or n > 80:
        return uniq
    return np.round(start + np.arange(n) * step, 8)


def _componentry_um_bar_geometry(phi_values, width_factor):
    phi_values = np.asarray(phi_values, dtype=float)
    uniq = np.unique(phi_values[np.isfinite(phi_values)])
    if uniq.size > 1:
        diffs = np.diff(np.sort(uniq))
        diffs = diffs[diffs > 1e-9]
        step = float(np.nanmin(diffs)) if diffs.size else 1.0
    else:
        step = 1.0
    centers = np.array([phi_to_um(v) for v in phi_values], dtype=float)
    widths = []
    for p in phi_values:
        a = phi_to_um(float(p) - step / 2.0)
        b = phi_to_um(float(p) + step / 2.0)
        widths.append(abs(a - b) * float(width_factor))
    return centers, np.asarray(widths, dtype=float)


def _sparse_values(values, max_count=12):
    values = np.asarray(values, dtype=float)
    if values.size <= max_count:
        return values
    idx = np.linspace(0, values.size - 1, max_count).round().astype(int)
    return values[np.unique(idx)]


def excel_col_letter(col_idx_zero_based: int) -> str:
    n = int(col_idx_zero_based) + 1
    letters = []
    while n > 0:
        n, r = divmod(n - 1, 26)
        letters.append(chr(65 + r))
    return "".join(reversed(letters))


def unique_name(name: str, used: set, default_prefix="Sample") -> str:
    base = str(name).strip()
    if (not base) or (base.lower() == "nan"):
        base = default_prefix
    out = base
    k = 2
    while out in used:
        out = f"{base} ({k})"
        k += 1
    used.add(out)
    return out


def ordered_union(seq_lists):
    out = []
    seen = set()
    for seq in seq_lists:
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
    return out


def guess_type_from_component(component_name: str) -> str:
    s = str(component_name).strip().lower()

    juvenile_keys = ["pum", "scor", "juven", "glass", "vitric", "jv"]
    lithic_keys = ["lit", "lith", "accidental", "country rock", "lx"]

    if any(k in s for k in lithic_keys):
        return "Lithic"
    if any(k in s for k in juvenile_keys):
        return "Juvenile"
    return "Accessory"


def build_gsd_df_from_phi_wt(phi_vals, wt_vals):
    phi = np.array([_to_float(v) for v in phi_vals], dtype=float)
    wt = np.array([_to_float(v) for v in wt_vals], dtype=float)

    m = np.isfinite(phi) & np.isfinite(wt)
    phi, wt = phi[m], wt[m]
    if phi.size == 0 or wt.size == 0:
        raise ValueError("Empty distribution.")
    if (wt < 0).any():
        raise ValueError("Negative wt% is invalid.")

    df = pd.DataFrame({"phi": phi, "wt": wt})
    df = df.groupby("phi", as_index=False)["wt"].sum().sort_values("phi").reset_index(drop=True)

    total = float(df["wt"].sum())
    if total <= 0:
        raise ValueError("Total wt% <= 0.")

    if not np.isclose(total, 100.0, atol=0.2):
        df["wt"] = df["wt"] * (100.0 / total)

    return df


# ---------- Grain-size statistics helpers (Folk & Ward) ----------
def _normalize_hist_series(phi_vals, wt_vals):
    phi = np.array([_to_float(v) for v in phi_vals], dtype=float)
    wt = np.array([_to_float(v) for v in wt_vals], dtype=float)

    m = np.isfinite(phi) & np.isfinite(wt)
    phi = phi[m]
    wt = wt[m]

    if phi.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    wt = np.clip(wt, 0.0, None)

    df = pd.DataFrame({"phi": phi, "wt": wt})
    df = df.groupby("phi", as_index=False)["wt"].sum().sort_values("phi").reset_index(drop=True)
    return df["phi"].astype(float).values, df["wt"].astype(float).values


def histogram_percentiles_phi(phi_vals, wt_vals, percentiles=(5, 16, 25, 50, 75, 84, 95)):
    """
    Percentiles by linear interpolation over cumulative wt% in phi space.
    Returns dict {percentile: phi_value} with np.nan when not computable.
    """
    phi, wt = _normalize_hist_series(phi_vals, wt_vals)

    out = {float(p): np.nan for p in percentiles}
    if phi.size == 0:
        return out

    total = float(np.nansum(wt))
    if total <= 0:
        return out

    cum = 100.0 * np.cumsum(wt) / total

    # Anchor at 0% to stabilize interpolation near the first bin
    x_cum = np.r_[0.0, cum]
    y_phi = np.r_[phi[0], phi]

    for p in percentiles:
        pp = float(p)
        if pp < 0 or pp > 100:
            out[pp] = np.nan
            continue
        out[pp] = float(np.interp(pp, x_cum, y_phi))

    return out


def folk_ward_graphic_params(phi_vals, wt_vals):
    """
    Folk & Ward (1957) graphic parameters in phi units.

      Mz  = (phi16 + phi50 + phi84) / 3
      sigma_i = (phi84 - phi16)/4 + (phi95 - phi5)/6.6
      SK_i = ((phi84 + phi16 - 2*phi50)/(2*(phi84-phi16))) + ((phi95 + phi5 - 2*phi50)/(2*(phi95-phi5)))
      Kg = (phi95 - phi5)/(2.44*(phi75 - phi25))
    """
    P = histogram_percentiles_phi(phi_vals, wt_vals, percentiles=(5, 16, 25, 50, 75, 84, 95))

    phi5 = P[5.0]
    phi16 = P[16.0]
    phi25 = P[25.0]
    phi50 = P[50.0]
    phi75 = P[75.0]
    phi84 = P[84.0]
    phi95 = P[95.0]

    def _safe_div(num, den):
        den = float(den)
        if (not np.isfinite(den)) or abs(den) < 1e-12:
            return np.nan
        return float(num) / den

    Mz = np.nan
    if np.isfinite(phi16) and np.isfinite(phi50) and np.isfinite(phi84):
        Mz = (phi16 + phi50 + phi84) / 3.0

    sigma_i = np.nan
    if all(np.isfinite(v) for v in [phi84, phi16, phi95, phi5]):
        sigma_i = ((phi84 - phi16) / 4.0) + ((phi95 - phi5) / 6.6)

    SK_i = np.nan
    if all(np.isfinite(v) for v in [phi84, phi16, phi50, phi95, phi5]):
        term1 = _safe_div((phi84 + phi16 - 2.0 * phi50), 2.0 * (phi84 - phi16))
        term2 = _safe_div((phi95 + phi5 - 2.0 * phi50), 2.0 * (phi95 - phi5))
        if np.isfinite(term1) and np.isfinite(term2):
            SK_i = term1 + term2

    Kg = np.nan
    if all(np.isfinite(v) for v in [phi95, phi5, phi75, phi25]):
        Kg = _safe_div((phi95 - phi5), 2.44 * (phi75 - phi25))

    return {
        "phi5": phi5, "phi16": phi16, "phi25": phi25, "phi50": phi50,
        "phi75": phi75, "phi84": phi84, "phi95": phi95,
        "Mz": Mz,
        "sigma_i": sigma_i,
        "SK_i": SK_i,
        "Kg": Kg,
    }


def _row_is_sum_row(component_value: str, sample_values: list) -> bool:
    s = str(component_value).strip().lower()
    if s in {"sum", "suma", "total", "totals"}:
        return True

    vals = np.array([_safe_float(v, default=np.nan) for v in sample_values], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False

    # Si muchos valores están ~100, probablemente es fila sumatoria por muestra.
    close_to_100 = np.isclose(vals, 100.0, atol=0.5)
    if np.count_nonzero(close_to_100) >= max(1, int(0.6 * vals.size)):
        return True

    return False


def parse_components_excel(path: str):
    """
    Formato esperado:
      - Col A: nombre fracción textual (opcional)
      - Col B: phi
      - Col C: component
      - Col D..: muestras
      - Puede incluir fila de suma final (100) -> se ignora automáticamente.

    Esta función está endurecida para NO confundir un Excel granulométrico
    simple con una matriz de componentes.
    """
    raw = pd.read_excel(path, header=None, sheet_name=0)
    if raw.shape[1] < 4:
        raise ValueError("The workbook must have at least 4 columns (A..D).")

    first_phi = _to_float(raw.iloc[0, 1]) if raw.shape[0] > 0 else np.nan
    has_header = not np.isfinite(first_phi)

    sample_cols = list(range(3, raw.shape[1]))
    if not sample_cols:
        raise ValueError("No sample columns found (need columns D..).")

    if has_header:
        sample_names_raw = []
        for c in sample_cols:
            v = raw.iloc[0, c]
            name = str(v).strip() if pd.notna(v) else ""
            if (not name) or (name.lower() == "nan"):
                name = f"Sample {excel_col_letter(c)}"
            sample_names_raw.append(name)
        data = raw.iloc[1:, :].copy()
    else:
        sample_names_raw = [f"Sample {excel_col_letter(c)}" for c in sample_cols]
        data = raw.copy()

    data = data.reset_index(drop=True)

    # Eliminar filas de sumatoria explícita
    keep_mask = []
    for i in range(len(data)):
        comp_val = data.iloc[i, 2] if data.shape[1] > 2 else ""
        sample_vals = [data.iloc[i, c] for c in sample_cols]
        keep_mask.append(not _row_is_sum_row(comp_val, sample_vals))
    data = data.loc[keep_mask].copy().reset_index(drop=True)

    if data.empty:
        raise ValueError("No usable rows found after removing sum rows.")

    # Validación adicional:
    # en una matriz de componentes real, la columna C debe ser mayormente textual.
    comp_candidate = data.iloc[:, 2]
    comp_as_float = comp_candidate.map(_to_float)
    numeric_ratio = float(np.mean(np.isfinite(comp_as_float))) if len(comp_candidate) > 0 else 0.0
    if numeric_ratio > 0.5:
        raise ValueError(
            "Column C looks numeric in most rows; this does not look like a component matrix."
        )

    data_phi = data.iloc[:, 1].map(_to_float)
    data_comp = data.iloc[:, 2].astype(str).str.strip()

    valid = np.isfinite(data_phi) & data_comp.ne("") & (data_comp.str.lower() != "nan")
    data = data.loc[valid].copy()
    data_phi = data_phi.loc[valid].astype(float)
    data_comp = data_comp.loc[valid].astype(str)

    if data.empty:
        raise ValueError("No valid rows found (need numeric phi in column B and component in column C).")

    components_order = []
    for c in data_comp.tolist():
        if c not in components_order:
            components_order.append(c)

    all_phis = sorted({float(v) for v in data_phi.values.tolist()})

    out = {}
    used_names = set()

    for idx, c in enumerate(sample_cols):
        sname = unique_name(sample_names_raw[idx], used_names, default_prefix=f"Sample {excel_col_letter(c)}")

        vals = data.iloc[:, c].map(_to_float).fillna(0.0).astype(float)
        vals = vals.clip(lower=0.0)

        tmp = pd.DataFrame({
            "phi": data_phi.values.astype(float),
            "component": data_comp.values.astype(str),
            "value": vals.values.astype(float)
        })

        grp = tmp.groupby(["phi", "component"], as_index=False)["value"].sum()

        mat = (
            grp.pivot(index="phi", columns="component", values="value")
            .reindex(index=all_phis, columns=components_order, fill_value=0.0)
            .astype(float)
        )

        total_sum = float(np.nansum(mat.values))
        if total_sum <= 0:
            continue

        if not np.isclose(total_sum, 100.0, atol=0.2):
            mat = mat * (100.0 / total_sum)

        gsd = pd.DataFrame({
            "phi": np.array(all_phis, dtype=float),
            "wt": mat.sum(axis=1).values.astype(float)
        })
        gsd = gsd.groupby("phi", as_index=False)["wt"].sum().sort_values("phi").reset_index(drop=True)

        out[sname] = {
            "gsd": gsd,
            "total_matrix": mat.copy(),
            "components_order": list(components_order)
        }

    if not out:
        raise ValueError("No non-empty sample columns were found in the workbook.")

    return out


def parse_gsd_excel(path: str):
    """
    Formato esperado para granulometría simple:
      - Columna A: contiene etiquetas o valores phi.
      - Filas donde A es numérico => datos granulométricos.
      - Filas donde A NO es numérico => etiquetas / metadata / texto auxiliar.
      - Fila 1 (índice 0) de cada columna B.. = nombre de muestra.
      - Columnas B.. = una muestra por columna.
      - Los phi pueden tener paso 1, 0.5 o cualquier otro paso numérico.

    Reglas:
      - El nombre de muestra se toma SOLO desde la primera fila de cada columna.
      - Las demás filas no-phi se ignoran.
      - Solo se usan filas con phi numérico en columna A.
    """
    raw = pd.read_excel(path, header=None, sheet_name=0)

    if raw.shape[1] < 2:
        raise ValueError("The workbook must have at least 2 columns (phi + one sample).")

    sample_cols = list(range(1, raw.shape[1]))
    used_names = set()
    out = {}

    # Detectar filas con phi numérico en columna A
    phi_series = raw.iloc[:, 0].map(_to_float)
    phi_mask = phi_series.map(np.isfinite)

    if not phi_mask.any():
        raise ValueError("Column A does not contain numeric phi values.")

    for c in sample_cols:
        # SOLO la primera fila de la columna es el nombre de la muestra
        v = raw.iloc[0, c]
        base_name = str(v).strip() if pd.notna(v) else ""
        if (not base_name) or (base_name.lower() == "nan"):
            base_name = f"Sample {excel_col_letter(c)}"

        sname = unique_name(base_name, used_names, default_prefix=f"Sample {excel_col_letter(c)}")

        # Usar únicamente las filas donde A tiene phi numérico
        phi = phi_series[phi_mask].astype(float).to_numpy()
        wt_series = raw.iloc[:, c].map(_to_float)
        wt = wt_series[phi_mask].to_numpy(dtype=float)

        # Celdas no numéricas dentro de filas phi => 0
        wt[~np.isfinite(wt)] = 0.0
        wt = np.clip(wt, 0.0, None)

        if phi.size == 0:
            continue
        if float(np.nansum(wt)) <= 0:
            continue

        gsd = build_gsd_df_from_phi_wt(phi, wt)

        out[sname] = {
            "gsd": gsd
        }

    if not out:
        raise ValueError("No valid sample columns were found in the workbook.")

    return out


def _shoelace_area(poly_xy):
    pts = np.array(poly_xy, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segments_intersect(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and
                min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    if abs(o1) < 1e-12 and on_segment(p1, p2, p3):
        return True
    if abs(o2) < 1e-12 and on_segment(p1, p2, p4):
        return True
    if abs(o3) < 1e-12 and on_segment(p3, p4, p1):
        return True
    if abs(o4) < 1e-12 and on_segment(p3, p4, p2):
        return True

    return False


def _poly_edges(poly):
    n = len(poly)
    for i in range(n):
        yield poly[i], poly[(i + 1) % n]


def polygons_overlap(poly_a, poly_b):
    if len(poly_a) < 3 or len(poly_b) < 3:
        return False

    for e1a, e1b in _poly_edges(poly_a):
        for e2a, e2b in _poly_edges(poly_b):
            if _segments_intersect(e1a, e1b, e2a, e2b):
                return True

    path_a = MplPath(np.array(poly_a, dtype=float))
    path_b = MplPath(np.array(poly_b, dtype=float))

    a_inside_b = np.any(path_b.contains_points(np.array(poly_a, dtype=float)))
    b_inside_a = np.any(path_a.contains_points(np.array(poly_b, dtype=float)))

    return bool(a_inside_b or b_inside_a)


def df_to_payload(df: pd.DataFrame):
    return {
        "index": [float(v) for v in df.index.tolist()],
        "columns": [str(c) for c in df.columns.tolist()],
        "data": [[float(_safe_float(x, default=0.0)) for x in row] for row in df.values.tolist()]
    }


def payload_to_df(payload):
    idx = payload.get("index", [])
    cols = payload.get("columns", [])
    data = payload.get("data", [])
    if len(cols) == 0:
        return pd.DataFrame(index=np.array(idx, dtype=float))
    return pd.DataFrame(data=np.array(data, dtype=float), index=np.array(idx, dtype=float), columns=cols)


def build_export_dataframe(sample_dbs: dict, sample_names: list):
    """
    Construye tabla de exportación en formato:
    A: fraction label
    B: phi
    C: component
    D..: muestras

    Incluye componente 'Undefined' forzado.
    Si una fracción no tiene componentes definidos en una muestra:
      Undefined = bin_wt (100% de la fracción), resto 0.
    """
    if not sample_names:
        raise ValueError("No samples selected for export.")

    phi_union = sorted(set(
        float(phi)
        for s in sample_names
        for phi in sample_dbs[s].gsd["phi"].astype(float).values.tolist()
    ))

    comp_union_lists = []
    for s in sample_names:
        db = sample_dbs[s]
        comp_union_lists.append(list(db.components))
        comp_union_lists.append(list(db.total.columns))
    components = ordered_union(comp_union_lists)
    components = [c for c in components if str(c).strip() and c != UNDEFINED_COMPONENT]
    components_with_undef = components + [UNDEFINED_COMPONENT]

    rows = []
    header = ["fraction", "phi", "component"] + sample_names
    rows.append(header)

    for phi in phi_union:
        frac_label = f"{phi:g} φ"

        for comp in components_with_undef:
            row_vals = []
            for s in sample_names:
                db = sample_dbs[s]
                wt_map = db.wt_map()
                bin_wt = float(wt_map.get(phi, 0.0))

                if comp != UNDEFINED_COMPONENT:
                    if (comp in db.total.columns) and (phi in db.total.index):
                        v = float(_safe_float(db.total.loc[phi, comp], default=0.0))
                    else:
                        v = 0.0
                else:
                    if bin_wt <= 0:
                        v = 0.0
                    else:
                        if phi in db.total.index:
                            row_sum = float(np.nansum([
                                _safe_float(db.total.loc[phi, c], default=0.0)
                                for c in db.total.columns
                            ]))
                        else:
                            row_sum = 0.0

                        if float(phi) not in db.filled:
                            v = bin_wt
                        else:
                            v = max(0.0, bin_wt - row_sum)

                row_vals.append(float(v))

            rows.append([frac_label, phi, comp] + row_vals)

    sum_vals = []
    for s in sample_names:
        db = sample_dbs[s]
        total = float(np.nansum(db.gsd["wt"].astype(float).values))
        if not np.isclose(total, 100.0, atol=0.2) and total > 0:
            total = 100.0
        sum_vals.append(total)
    rows.append(["SUM", "", "SUM"] + sum_vals)

    return pd.DataFrame(rows)


def sqlite_save_project(path: str, project_dict: dict):
    payload = json.dumps(project_dict, ensure_ascii=False)

    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS project_data (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                json_text TEXT NOT NULL
            )
        """)
        cur.execute("DELETE FROM project_data")
        cur.execute("INSERT INTO project_data (id, json_text) VALUES (1, ?)", (payload,))
        conn.commit()
    finally:
        conn.close()


def sqlite_load_project(path: str):
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT json_text
            FROM project_data
            WHERE id = 1
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            raise ValueError("Invalid project file: no project_data payload found.")
        return json.loads(row[0])
    finally:
        conn.close()
#endregion

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

        # Persistent Fractal Analysis state.  The external FractalAnalysis.py
        # module owns the schema and calculations; ComponentryDB only stores the
        # JSON-serializable per-sample settings so standalone and FW projects keep
        # densities, shapes, A/B, C/B, manual B values and fit selections.
        self.fractal_settings = {}

        # Estado transitorio de análisis de imagen por phi.
        #
        # IMPORTANTE:
        # - NO se serializa en proyecto.
        # - Puede venir en formato legacy (una sola imagen por phi) o
        #   en formato nuevo (múltiples imágenes por phi).
        #
        # Legacy:
        #   image_sessions[phi] = {
        #       mode, image_path, crop_rect,
        #       points, polygons,
        #       background_enabled, background_component, freehand,
        #       grid_enabled, grid_rows, grid_cols, grid_color,
        #       vertex_auto_enabled, vertex_current_index, vertex_grid_locked,
        #       vertex_assignments,
        #       zoom_visible, zoom_follow, zoom_level
        #   }
        #
        # Nuevo multi-image:
        #   image_sessions[phi] = {
        #       "mode": "points" | "areas" | "vertices",
        #       "current_index": 0,
        #       "images": [
        #           {
        #               "label": "Image 1",
        #               "image_path": ...,
        #               "crop_rect": ...,
        #               "points": ...,
        #               "polygons": ...,
        #               "background_enabled": ...,
        #               "background_component": ...,
        #               "freehand": ...,
        #               "grid_enabled": ...,
        #               "grid_rows": ...,
        #               "grid_cols": ...,
        #               "grid_color": ...,
        #               "vertex_auto_enabled": ...,
        #               "vertex_current_index": ...,
        #               "vertex_grid_locked": ...,
        #               "vertex_assignments": ...,
        #               "zoom_visible": ...,
        #               "zoom_follow": ...,
        #               "zoom_level": ...
        #           },
        #           ...
        #       ]
        #   }
        self.image_sessions = {}

    # =========================================================
    # BASIC HELPERS
    # =========================================================
    def wt_map(self):
        return dict(zip(self.gsd["phi"].astype(float).values, self.gsd["wt"].astype(float).values))

    def has_data_for_component(self, name):
        if name not in self.frac100.columns:
            return False
        s1 = float(np.nansum(self.frac100[name].values))
        s2 = float(np.nansum(self.total[name].values))
        return (s1 > 0) or (s2 > 0)

    # =========================================================
    # IMAGE SESSION HELPERS (runtime only, not serialized)
    # =========================================================
    def _normalize_mode(self, mode):
        m = str(mode).strip().lower()
        if m in {"points", "areas", "vertices"}:
            return m
        return "points"

    def _clean_crop_rect(self, crop_rect):
        if isinstance(crop_rect, (list, tuple)) and len(crop_rect) == 4:
            return [int(round(_safe_float(v, default=0.0))) for v in crop_rect]
        return None

    def _clean_points(self, points):
        out = []
        for p in points or []:
            if not isinstance(p, (list, tuple)) or len(p) < 3:
                continue
            x = _safe_float(p[0], default=np.nan)
            y = _safe_float(p[1], default=np.nan)
            c = str(p[2]).strip()
            if np.isfinite(x) and np.isfinite(y) and c:
                out.append([float(x), float(y), c])
        return out

    def _clean_polygons(self, polygons):
        out = []
        for pg in polygons or []:
            if not isinstance(pg, dict):
                continue
            comp = str(pg.get("component", "")).strip()
            if not comp:
                continue

            xy_clean = []
            for pt in pg.get("xy", []):
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                x = _safe_float(pt[0], default=np.nan)
                y = _safe_float(pt[1], default=np.nan)
                if np.isfinite(x) and np.isfinite(y):
                    xy_clean.append([float(x), float(y)])

            if len(xy_clean) >= 3:
                out.append({"component": comp, "xy": xy_clean})

        return out

    def _clean_vertex_assignments(self, raw_va):
        va_clean = {}

        if isinstance(raw_va, dict):
            for kk, vv in raw_va.items():
                key = str(kk).strip()
                comp = str(vv).strip()
                if key and comp:
                    va_clean[key] = comp

        elif isinstance(raw_va, list):
            for item in raw_va:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key", "")).strip()
                comp = str(item.get("component", "")).strip()
                if key and comp:
                    va_clean[key] = comp

        return va_clean

    def _new_empty_image_state(self, label="Image 1"):
        return {
            "label": str(label),
            "image_path": None,
            "crop_rect": None,
            "points": [],
            "polygons": [],
            "background_enabled": False,
            "background_component": None,
            "freehand": False,
            "grid_enabled": False,
            "grid_rows": 8,
            "grid_cols": 8,
            "grid_color": "#ffffff",
            "vertex_auto_enabled": False,
            "vertex_current_index": 0,
            "vertex_grid_locked": False,
            "vertex_assignments": {},
            "zoom_visible": False,
            "zoom_follow": True,
            "zoom_level": 5,
        }

    def _normalize_single_image_state(self, img_payload, default_label="Image 1"):
        base = self._new_empty_image_state(label=default_label)

        if not isinstance(img_payload, dict):
            return base

        base["label"] = str(img_payload.get("label", default_label)).strip() or str(default_label)
        base["image_path"] = img_payload.get("image_path", None)
        base["crop_rect"] = self._clean_crop_rect(img_payload.get("crop_rect", None))
        base["points"] = self._clean_points(img_payload.get("points", []))
        base["polygons"] = self._clean_polygons(img_payload.get("polygons", []))
        base["background_enabled"] = bool(img_payload.get("background_enabled", False))
        base["background_component"] = img_payload.get("background_component", None)
        base["freehand"] = bool(img_payload.get("freehand", False))
        base["grid_enabled"] = bool(img_payload.get("grid_enabled", False))
        base["grid_rows"] = max(2, int(_safe_float(img_payload.get("grid_rows", 8), default=8)))
        base["grid_cols"] = max(2, int(_safe_float(img_payload.get("grid_cols", 8), default=8)))
        base["grid_color"] = str(img_payload.get("grid_color", "#ffffff"))
        base["vertex_auto_enabled"] = bool(img_payload.get("vertex_auto_enabled", False))
        base["vertex_current_index"] = max(0, int(_safe_float(img_payload.get("vertex_current_index", 0), default=0)))
        base["vertex_grid_locked"] = bool(img_payload.get("vertex_grid_locked", False))
        base["vertex_assignments"] = self._clean_vertex_assignments(img_payload.get("vertex_assignments", {}))
        base["zoom_visible"] = bool(img_payload.get("zoom_visible", False))
        base["zoom_follow"] = bool(img_payload.get("zoom_follow", True))
        base["zoom_level"] = max(2, min(20, int(_safe_float(img_payload.get("zoom_level", 5), default=5))))

        return base

    def _image_state_has_content(self, img_state):
        if not isinstance(img_state, dict):
            return False
        if img_state.get("image_path"):
            return True
        if img_state.get("crop_rect") is not None:
            return True
        if img_state.get("points"):
            return True
        if img_state.get("polygons"):
            return True
        if img_state.get("vertex_assignments"):
            return True
        return False

    def _normalize_runtime_session(self, sess):
        """
        Devuelve siempre una sesión multi-image normalizada.
        Acepta formato legacy o formato nuevo.
        """
        mode = "points"
        current_index = 0
        images = []

        if not isinstance(sess, dict):
            return {"mode": "points", "current_index": 0, "images": []}

        # Formato nuevo
        if isinstance(sess.get("images", None), list):
            mode = self._normalize_mode(sess.get("mode", "points"))
            raw_images = sess.get("images", [])
            for i, img_payload in enumerate(raw_images, start=1):
                img = self._normalize_single_image_state(img_payload, default_label=f"Image {i}")
                img["label"] = f"Image {i}"
                images.append(img)

            current_index = int(_safe_float(sess.get("current_index", 0), default=0))
            if images:
                current_index = max(0, min(current_index, len(images) - 1))
            else:
                current_index = 0

            return {
                "mode": mode,
                "current_index": current_index,
                "images": images
            }

        # Formato legacy (una sola imagen)
        mode = self._normalize_mode(sess.get("mode", "points"))
        img = self._normalize_single_image_state(sess, default_label="Image 1")

        if self._image_state_has_content(img):
            images = [img]
            current_index = 0
        else:
            images = []
            current_index = 0

        return {
            "mode": mode,
            "current_index": current_index,
            "images": images
        }

    def get_image_session(self, phi, create=False):
        """
        Devuelve la sesión multi-image normalizada para una phi.
        Si create=True y no existe, crea una sesión vacía.
        """
        phi = float(phi)

        if phi not in self.image_sessions:
            if not create:
                return None
            self.image_sessions[phi] = {
                "mode": "points",
                "current_index": 0,
                "images": []
            }
            return self.image_sessions[phi]

        sess = self._normalize_runtime_session(self.image_sessions.get(phi))
        self.image_sessions[phi] = sess
        return sess

    def set_image_session(self, phi, session_payload):
        """
        Guarda una sesión runtime ya normalizada o normalizable.
        """
        phi = float(phi)
        self.image_sessions[phi] = self._normalize_runtime_session(session_payload)

    def clear_image_session(self, phi):
        phi = float(phi)
        self.image_sessions.pop(phi, None)

    def clear_all_image_sessions(self):
        self.image_sessions = {}

    # =========================================================
    # INTERNAL RENORMALIZATION
    # =========================================================
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

    # =========================================================
    # COMPONENT MANAGEMENT
    # =========================================================
    def remove_component(self, name):
        if name not in self.components:
            return

        self.components = [c for c in self.components if c != name]
        self.colors.pop(name, None)
        self.types.pop(name, None)

        self.frac100 = self.frac100.drop(columns=[name], errors="ignore")
        self.total = self.total.drop(columns=[name], errors="ignore")

        # limpiar sesiones runtime (legacy o multi-image)
        for phi in list(self.image_sessions.keys()):
            sess = self.get_image_session(phi, create=False)
            if not isinstance(sess, dict):
                continue

            for img in sess.get("images", []):
                if img.get("background_component") == name:
                    img["background_component"] = None
                    img["background_enabled"] = False

                polys = img.get("polygons", [])
                img["polygons"] = [p for p in polys if p.get("component") != name]

                pts = img.get("points", [])
                img["points"] = [p for p in pts if (len(p) >= 3 and p[2] != name)]

                raw_va = img.get("vertex_assignments", {})
                if isinstance(raw_va, dict):
                    img["vertex_assignments"] = {
                        str(k): str(v) for k, v in raw_va.items()
                        if str(v) != name
                    }
                else:
                    img["vertex_assignments"] = {}

            self.image_sessions[float(phi)] = sess

        self._renormalize_all_filled()

    def rename_component(self, old_name, new_name):
        """Rename one component without changing its values, color, type or image assignments."""
        old_name = str(old_name).strip()
        new_name = str(new_name).strip()
        if not old_name or not new_name:
            raise ValueError("Component names cannot be empty.")
        if old_name == new_name:
            return False
        if old_name not in self.components:
            return False
        if new_name in self.components:
            raise ValueError(f"Component '{new_name}' already exists.")

        idx = self.components.index(old_name)
        self.components[idx] = new_name
        self.frac100 = self.frac100.rename(columns={old_name: new_name})
        self.total = self.total.rename(columns={old_name: new_name})

        if old_name in self.colors:
            self.colors[new_name] = self.colors.pop(old_name)
        if old_name in self.types:
            self.types[new_name] = self.types.pop(old_name)

        # Fractal particle-model settings are keyed by component name and must
        # follow an explicit rename just like colors/types/data columns.
        try:
            fs = self.fractal_settings if isinstance(self.fractal_settings, dict) else {}
            comp_settings = fs.get("components", {}) if isinstance(fs.get("components", {}), dict) else {}
            if old_name in comp_settings and new_name not in comp_settings:
                comp_settings[new_name] = comp_settings.pop(old_name)
            fs["components"] = comp_settings
            self.fractal_settings = fs
        except Exception:
            pass

        # Runtime image-analysis assignments must follow the renamed component.
        for phi in list(self.image_sessions.keys()):
            sess = self.get_image_session(phi, create=False)
            if not isinstance(sess, dict):
                continue
            for img in sess.get("images", []):
                if img.get("background_component") == old_name:
                    img["background_component"] = new_name

                pts = img.get("points", [])
                img["points"] = [
                    [p[0], p[1], new_name if (len(p) >= 3 and p[2] == old_name) else p[2]]
                    for p in pts if len(p) >= 3
                ]

                polys = img.get("polygons", [])
                for pg in polys:
                    if pg.get("component") == old_name:
                        pg["component"] = new_name

                raw_va = img.get("vertex_assignments", {})
                if isinstance(raw_va, dict):
                    img["vertex_assignments"] = {
                        str(k): (new_name if str(v) == old_name else str(v))
                        for k, v in raw_va.items()
                    }
            self.image_sessions[float(phi)] = sess
        return True

    def set_component_type(self, name, component_type):
        """Change only the component type; component color and data are untouched."""
        name = str(name).strip()
        component_type = str(component_type).strip()
        if component_type not in TYPE_OPTIONS:
            raise ValueError(f"Invalid component type: {component_type}")
        if name not in self.components:
            return False
        self.types[name] = component_type
        return True

    def set_components(self, new_names, colors_by_name, types_by_name):
        new_names = [str(n).strip() for n in new_names]

        if any((not n) for n in new_names):
            raise ValueError("Component names cannot be empty.")
        if len(set(new_names)) != len(new_names):
            raise ValueError("Component names must be unique.")

        old = list(self.components)

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

            try:
                fs = self.fractal_settings if isinstance(self.fractal_settings, dict) else {}
                comp_settings = fs.get("components", {}) if isinstance(fs.get("components", {}), dict) else {}
                for o, n in rename_map.items():
                    if o in comp_settings and n not in comp_settings:
                        comp_settings[n] = comp_settings.pop(o)
                fs["components"] = comp_settings
                self.fractal_settings = fs
            except Exception:
                pass

            # renombrar componentes en sesiones runtime
            for phi in list(self.image_sessions.keys()):
                sess = self.get_image_session(phi, create=False)
                if not isinstance(sess, dict):
                    continue

                for img in sess.get("images", []):
                    bg = img.get("background_component")
                    if bg in rename_map:
                        img["background_component"] = rename_map[bg]

                    pts = img.get("points", [])
                    new_pts = []
                    for p in pts:
                        if len(p) >= 3:
                            c = rename_map.get(p[2], p[2])
                            new_pts.append([p[0], p[1], c])
                    img["points"] = new_pts

                    polys = img.get("polygons", [])
                    new_polys = []
                    for pg in polys:
                        c = rename_map.get(pg.get("component"), pg.get("component"))
                        new_polys.append({"component": c, "xy": pg.get("xy", [])})
                    img["polygons"] = new_polys

                    raw_va = img.get("vertex_assignments", {})
                    if isinstance(raw_va, dict):
                        new_va = {}
                        for k, v in raw_va.items():
                            new_va[str(k)] = rename_map.get(v, v)
                        img["vertex_assignments"] = new_va
                    else:
                        img["vertex_assignments"] = {}

                self.image_sessions[float(phi)] = sess

        for n in new_names:
            if n not in self.frac100.columns:
                self.frac100[n] = 0.0
            if n not in self.total.columns:
                self.total[n] = 0.0

        extra = [c for c in self.frac100.columns if c not in new_names]
        if extra:
            self.frac100 = self.frac100.drop(columns=extra, errors="ignore")
            self.total = self.total.drop(columns=extra, errors="ignore")
            for c in extra:
                self.colors.pop(c, None)
                self.types.pop(c, None)
            try:
                fs = self.fractal_settings if isinstance(self.fractal_settings, dict) else {}
                comp_settings = fs.get("components", {}) if isinstance(fs.get("components", {}), dict) else {}
                for c in extra:
                    comp_settings.pop(c, None)
                fs["components"] = comp_settings
                self.fractal_settings = fs
            except Exception:
                pass

        if new_names:
            self.frac100 = self.frac100.loc[:, new_names]
            self.total = self.total.loc[:, new_names]
        else:
            self.frac100 = self.frac100.iloc[:, 0:0]
            self.total = self.total.iloc[:, 0:0]

        self.components = list(new_names)

        for i, n in enumerate(self.components):
            self.colors[n] = colors_by_name.get(n) or self.colors.get(n) or default_component_color(i)
            t = types_by_name.get(n) or self.types.get(n) or "Accessory"
            self.types[n] = t if t in TYPE_OPTIONS else "Accessory"

        valid_set = set(self.components)

        # depurar sesiones runtime contra componentes válidos
        for phi in list(self.image_sessions.keys()):
            sess = self.get_image_session(phi, create=False)
            if not isinstance(sess, dict):
                continue

            for img in sess.get("images", []):
                bg = img.get("background_component")
                if bg not in valid_set:
                    img["background_component"] = None
                    img["background_enabled"] = False

                pts = img.get("points", [])
                img["points"] = [p for p in pts if (len(p) >= 3 and p[2] in valid_set)]

                polys = img.get("polygons", [])
                img["polygons"] = [p for p in polys if p.get("component") in valid_set]

                raw_va = img.get("vertex_assignments", {})
                if isinstance(raw_va, dict):
                    img["vertex_assignments"] = {
                        str(k): str(v) for k, v in raw_va.items()
                        if str(v) in valid_set
                    }
                else:
                    img["vertex_assignments"] = {}

            self.image_sessions[float(phi)] = sess

    # =========================================================
    # FRACTION DATA LOAD
    # =========================================================
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

    # =========================================================
    # AGGREGATIONS
    # =========================================================
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

    # =========================================================
    # SERIALIZATION
    # =========================================================
    def to_dict(self):
        """
        IMPORTANTE:
        image_sessions NO se serializa.
        Las imágenes deben recargarse manualmente al reabrir el proyecto.
        """
        return {
            "gsd": self.gsd.to_dict(orient="list"),
            "components": list(self.components),
            "colors": dict(self.colors),
            "types": dict(self.types),
            "frac100": df_to_payload(self.frac100),
            "total": df_to_payload(self.total),
            "filled": [float(v) for v in sorted(list(self.filled))],
            "fractal_settings": self.fractal_settings if isinstance(self.fractal_settings, dict) else {}
        }

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

        # Optional field: old projects have no fractal_settings and remain fully
        # compatible.  FractalAnalysis.py supplies defaults lazily when opened.
        fs = payload.get("fractal_settings", {})
        db.fractal_settings = fs if isinstance(fs, dict) else {}

        # runtime only: al reabrir proyecto no restauramos imágenes
        db.image_sessions = {}

        for c in db.components:
            if c not in db.colors:
                db.colors[c] = default_component_color(len(db.colors))
            if c not in db.types or db.types[c] not in TYPE_OPTIONS:
                db.types[c] = "Accessory"

        return db
#endregion


#region BLOQUE 7a Image analysis dialog
from PyQt5.QtCore import QTimer


def _dedupe_polygon_points(poly_xy, tol=1e-9):
    out = []
    for pt in poly_xy:
        if len(pt) < 2:
            continue
        x = float(pt[0])
        y = float(pt[1])
        if not out:
            out.append([x, y])
            continue
        if np.hypot(out[-1][0] - x, out[-1][1] - y) > tol:
            out.append([x, y])

    if len(out) >= 2 and np.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol:
        out.pop()

    return out


def _clip_polygon_to_rect(poly_xy, x_min, y_min, x_max, y_max):
    """
    Sutherland-Hodgman polygon clipping against axis-aligned rectangle.
    Rectangle limits are inclusive in logical sense, but clipping uses
    closed boundaries [x_min, x_max] x [y_min, y_max].
    """
    pts = _dedupe_polygon_points(poly_xy)
    if len(pts) < 3:
        return []

    def clip_edge(points, inside_fn, intersect_fn):
        if not points:
            return []

        output = []
        s = points[-1]
        s_in = inside_fn(s)

        for e in points:
            e_in = inside_fn(e)

            if e_in:
                if not s_in:
                    output.append(intersect_fn(s, e))
                output.append([float(e[0]), float(e[1])])
            elif s_in:
                output.append(intersect_fn(s, e))

            s = e
            s_in = e_in

        return _dedupe_polygon_points(output)

    def intersect_vertical(p1, p2, x_const):
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dx = x2 - x1
        if abs(dx) < 1e-12:
            return [float(x_const), float(y1)]
        t = (float(x_const) - x1) / dx
        return [float(x_const), float(y1 + t * (y2 - y1))]

    def intersect_horizontal(p1, p2, y_const):
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        dy = y2 - y1
        if abs(dy) < 1e-12:
            return [float(x1), float(y_const)]
        t = (float(y_const) - y1) / dy
        return [float(x1 + t * (x2 - x1)), float(y_const)]

    pts = clip_edge(
        pts,
        inside_fn=lambda p: float(p[0]) >= float(x_min),
        intersect_fn=lambda s, e: intersect_vertical(s, e, x_min)
    )
    pts = clip_edge(
        pts,
        inside_fn=lambda p: float(p[0]) <= float(x_max),
        intersect_fn=lambda s, e: intersect_vertical(s, e, x_max)
    )
    pts = clip_edge(
        pts,
        inside_fn=lambda p: float(p[1]) >= float(y_min),
        intersect_fn=lambda s, e: intersect_horizontal(s, e, y_min)
    )
    pts = clip_edge(
        pts,
        inside_fn=lambda p: float(p[1]) <= float(y_max),
        intersect_fn=lambda s, e: intersect_horizontal(s, e, y_max)
    )

    pts = _dedupe_polygon_points(pts)
    if len(pts) < 3:
        return []

    if _shoelace_area(pts) <= 1e-9:
        return []

    return pts


class ImageAnalysisDialog(QDialog):
    """
    Modes:
      1) Point count
      2) Area polygons
      3) Vertex mode

    Multi-image behavior for a single phi:
      - one global mode per phi
      - multiple images can be loaded at once
      - each image has its own:
          crop, points, polygons, background, grid,
          vertex assignments, zoom flags
      - final save combines all images:
          * points   -> sums counts from all images
          * vertices -> sums classified vertices from all images
          * areas    -> sums raw areas from all images and normalizes once
    """
    def __init__(self, db: ComponentryDB, sample_name: str, phi: float, parent=None):
        super().__init__(parent)
        self.db = db
        self.sample_name = str(sample_name)
        self.phi = float(phi)
        self.setModal(True)

        sub = self.db.gsd[self.db.gsd["phi"].astype(float) == self.phi]
        self.bin_wt = float(sub["wt"].iloc[0]) if not sub.empty else np.nan

        self.setWindowTitle(f"Image analysis — {self.sample_name} — {self.phi:g}φ")
        self.resize(1540, 900)

        # ---------------------------------------------------------
        # Global dialog/session state (per phi)
        # ---------------------------------------------------------
        self._selected_component = None
        self._preferred_mode = "points"

        # session runtime for this phi
        self._session = None
        self._current_image_idx = 0

        # transient interaction flags (not per-image persisted)
        self._crop_mode = False
        self._crop_dragging = False
        self._crop_start = None
        self._crop_end = None
        self._pending_crop_rect = None

        self.current_poly = []   # current click-polygon vertices (current image only)
        self._dragging = False
        self._drag_points = []
        self.vertex_history = []

        self._last_main_xy = None
        self._zoom_dragging = False
        self._zoom_drag_last_xy = None
        self._zoom_shadow_visible = False
        self._zoom_shadow_timer = QTimer(self)
        self._zoom_shadow_timer.setSingleShot(True)
        self._zoom_shadow_timer.timeout.connect(self._hide_zoom_shadow)

        self.vertex_shortcut_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9"] + list("QWERTYUIOPASDFGHJKLZXCVBNM")
        self.vertex_buttons = []

        # Tamaño configurable de puntos para Point count
        self.point_marker_size_main = 80
        self.point_marker_size_zoom = 140

        # ---------------------------------------------------------
        # Active image state (loaded from current image payload)
        # ---------------------------------------------------------
        self._img = None
        self._img_original = None
        self._img_path = None
        self.crop_rect = None

        self.grid_enabled = False
        self.grid_rows = 8
        self.grid_cols = 8
        self.grid_color = "#ffff00"

        self.points = []                 # [x, y, component]
        self.polygons = []               # list of {"component": str, "xy": [[x,y],...]}
        self.background_enabled = False
        self.background_component = None
        self.freehand_enabled = False

        self.vertex_assignments = {}     # "r:c" -> component
        self.vertex_auto_enabled = False
        self.vertex_current_index = 0
        self.vertex_grid_locked = False

        self.zoom_visible = False
        self.zoom_follow = True
        self.zoom_level = 5
        self.zoom_center = None

        # Load runtime session for this phi, if any
        self._load_session_if_any()

        # ---------------------------------------------------------
        # UI
        # ---------------------------------------------------------
        root = QHBoxLayout(self)

        # =========================
        # LEFT PANEL
        # =========================
        left = QWidget()
        left_v = QVBoxLayout(left)

        # ---- top row: images + crop + status
        top = QHBoxLayout()

        self.btn_load_img = QPushButton("Load image(s)…")
        self.btn_load_img.clicked.connect(self.load_images)
        top.addWidget(self.btn_load_img)

        self.btn_prev_img = QPushButton("◀")
        self.btn_prev_img.setFixedWidth(32)
        self.btn_prev_img.clicked.connect(self.prev_image)
        top.addWidget(self.btn_prev_img)

        self.lbl_image_nav = QLabel("No images loaded")
        self.lbl_image_nav.setMinimumWidth(150)
        top.addWidget(self.lbl_image_nav)

        self.btn_next_img = QPushButton("▶")
        self.btn_next_img.setFixedWidth(32)
        self.btn_next_img.clicked.connect(self.next_image)
        top.addWidget(self.btn_next_img)

        self.btn_remove_img = QPushButton("Remove image")
        self.btn_remove_img.clicked.connect(self.remove_current_image)
        top.addWidget(self.btn_remove_img)

        self.btn_crop_mode = QPushButton("Select crop")
        self.btn_crop_mode.setCheckable(True)
        self.btn_crop_mode.toggled.connect(self._on_crop_mode_toggled)
        top.addWidget(self.btn_crop_mode)

        self.btn_apply_crop = QPushButton("Apply crop")
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        top.addWidget(self.btn_apply_crop)

        self.btn_reset_crop = QPushButton("Reset crop")
        self.btn_reset_crop.clicked.connect(self.reset_crop)
        top.addWidget(self.btn_reset_crop)

        self.lbl_status = QLabel("No image loaded.")
        top.addWidget(self.lbl_status, 1)

        left_v.addLayout(top)

        # ---- mode row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Point count", "Area polygons", "Vertex mode"])
        if self._preferred_mode == "areas":
            self.cmb_mode.setCurrentIndex(1)
        elif self._preferred_mode == "vertices":
            self.cmb_mode.setCurrentIndex(2)
        else:
            self.cmb_mode.setCurrentIndex(0)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.cmb_mode)

        self.chk_freehand = QCheckBox("Freehand drag polygon")
        self.chk_freehand.setChecked(bool(self.freehand_enabled))
        self.chk_freehand.toggled.connect(self._on_freehand_toggled)
        mode_row.addWidget(self.chk_freehand)

        mode_row.addStretch(1)
        left_v.addLayout(mode_row)

        # ---- grid row
        grid_row = QHBoxLayout()

        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(bool(self.grid_enabled))
        self.chk_grid.toggled.connect(self._on_grid_toggled)
        grid_row.addWidget(self.chk_grid)

        grid_row.addWidget(QLabel("Grid vertices X:"))
        self.spn_grid_cols = QSpinBox()
        self.spn_grid_cols.setRange(2, 500)
        self.spn_grid_cols.setValue(max(2, int(self.grid_cols)))
        self.spn_grid_cols.valueChanged.connect(self._on_grid_value_changed)
        grid_row.addWidget(self.spn_grid_cols)

        grid_row.addWidget(QLabel("Grid vertices Y:"))
        self.spn_grid_rows = QSpinBox()
        self.spn_grid_rows.setRange(2, 500)
        self.spn_grid_rows.setValue(max(2, int(self.grid_rows)))
        self.spn_grid_rows.valueChanged.connect(self._on_grid_value_changed)
        grid_row.addWidget(self.spn_grid_rows)

        self.btn_grid_color = QPushButton("Grid color")
        self.btn_grid_color.clicked.connect(self.pick_grid_color)
        grid_row.addWidget(self.btn_grid_color)

        self.lbl_grid_color = QLabel("      ")
        self.lbl_grid_color.setFixedWidth(30)
        self.lbl_grid_color.setStyleSheet(f"background:{self.grid_color}; border:1px solid #333;")
        grid_row.addWidget(self.lbl_grid_color)

        self.btn_fix_vertex_grid = QPushButton("Fix grid")
        self.btn_fix_vertex_grid.clicked.connect(self.fix_vertex_grid)
        grid_row.addWidget(self.btn_fix_vertex_grid)

        grid_row.addStretch(1)
        grid_row.addWidget(QLabel("Total vertices:"))
        self.lbl_vertex_total = QLabel("0")
        self.lbl_vertex_total.setMinimumWidth(60)
        self.lbl_vertex_total.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid_row.addWidget(self.lbl_vertex_total)

        left_v.addLayout(grid_row)

        # ---- zoom row
        zoom_toggle_row = QHBoxLayout()

        self.chk_show_zoom = QCheckBox("Show zoom panel")
        self.chk_show_zoom.setChecked(bool(self.zoom_visible))
        self.chk_show_zoom.toggled.connect(self._on_zoom_visibility_toggled)
        zoom_toggle_row.addWidget(self.chk_show_zoom)

        self.chk_zoom_follow = QCheckBox("Follow")
        self.chk_zoom_follow.setChecked(bool(self.zoom_follow))
        self.chk_zoom_follow.toggled.connect(self._on_zoom_follow_toggled)
        zoom_toggle_row.addWidget(self.chk_zoom_follow)

        zoom_toggle_row.addWidget(QLabel("Zoom:"))
        self.spn_zoom = QSpinBox()
        self.spn_zoom.setRange(2, 20)
        self.spn_zoom.setValue(int(self.zoom_level))
        self.spn_zoom.valueChanged.connect(self._on_zoom_level_changed)
        zoom_toggle_row.addWidget(self.spn_zoom)

        zoom_toggle_row.addStretch(1)
        left_v.addLayout(zoom_toggle_row)

        # ---- main image + zoom panel
        image_row = QHBoxLayout()

        self.canvas = FigureCanvas(plt.figure(figsize=(8.8, 6.1)))
        image_row.addWidget(self.canvas, 1)

        self.gb_zoom = QGroupBox("Zoom")
        self.gb_zoom.setMinimumWidth(320)
        self.gb_zoom.setMaximumWidth(420)
        zoom_v = QVBoxLayout(self.gb_zoom)

        self.zoom_canvas = FigureCanvas(plt.figure(figsize=(3.2, 3.2)))
        zoom_v.addWidget(self.zoom_canvas, 1)

        image_row.addWidget(self.gb_zoom, 0)
        left_v.addLayout(image_row, 1)

        root.addWidget(left, 3)

        # =========================
        # RIGHT PANEL
        # =========================
        right = QWidget()
        right_v = QVBoxLayout(right)

        info = QLabel(f"Fraction: {self.phi:g}φ | bin wt = {self.bin_wt:.2f}%")
        right_v.addWidget(info)
        right_v.addWidget(QLabel("Components (click to select):"))

        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.SingleSelection)
        right_v.addWidget(self.lst, 2)

        for c in self.db.components:
            it = QListWidgetItem(c)
            col = self.db.colors.get(c, "#cccccc")
            it.setBackground(QColor(col))
            it.setForeground(QColor("#000000"))
            self.lst.addItem(it)

        self.lst.currentTextChanged.connect(self._on_component_selected)

        if self.db.components:
            if self._selected_component in self.db.components:
                idx = self.db.components.index(self._selected_component)
                self.lst.setCurrentRow(idx)
            else:
                self.lst.setCurrentRow(0)

        bg_row = QHBoxLayout()

        self.chk_bg = QCheckBox("Use one background component")
        self.chk_bg.setChecked(bool(self.background_enabled))
        self.chk_bg.toggled.connect(self._on_bg_toggled)
        bg_row.addWidget(self.chk_bg)

        self.cmb_bg = QComboBox()
        self.cmb_bg.addItems(self.db.components)
        if self.background_component in self.db.components:
            self.cmb_bg.setCurrentText(self.background_component)
        elif self.db.components:
            self.cmb_bg.setCurrentIndex(0)
            self.background_component = self.db.components[0]
        self.cmb_bg.currentTextChanged.connect(self._on_bg_component_changed)
        bg_row.addWidget(self.cmb_bg)

        right_v.addLayout(bg_row)

        self.lbl_table_title = QLabel("Counts:")
        right_v.addWidget(self.lbl_table_title)

        self.tbl_counts = QTableWidget()
        self.tbl_counts.setColumnCount(2)
        self.tbl_counts.setHorizontalHeaderLabels(["Component", "Value"])
        self.tbl_counts.setEditTriggers(QAbstractItemView.NoEditTriggers)
        right_v.addWidget(self.tbl_counts, 1)

        # ---- vertex mode box
        self.gb_vertex = QGroupBox("Vertex mode")
        gv = QVBoxLayout(self.gb_vertex)

        self.lbl_vertex_grid_state = QLabel("")
        gv.addWidget(self.lbl_vertex_grid_state)

        auto_row = QHBoxLayout()
        self.chk_vertex_auto = QCheckBox("Automatic stepping")
        self.chk_vertex_auto.setChecked(bool(self.vertex_auto_enabled))
        self.chk_vertex_auto.toggled.connect(self._on_vertex_auto_toggled)
        auto_row.addWidget(self.chk_vertex_auto)
        auto_row.addStretch(1)
        gv.addLayout(auto_row)

        self.lbl_vertex_info = QLabel("No vertex selected.")
        gv.addWidget(self.lbl_vertex_info)

        self.vertex_map_layout = QGridLayout()
        gv.addLayout(self.vertex_map_layout)

        nav_row = QHBoxLayout()

        self.btn_vertex_prev = QPushButton("Previous")
        self.btn_vertex_prev.clicked.connect(self.vertex_prev)
        nav_row.addWidget(self.btn_vertex_prev)

        self.btn_vertex_skip = QPushButton("Skip")
        self.btn_vertex_skip.clicked.connect(self.vertex_skip)
        nav_row.addWidget(self.btn_vertex_skip)

        self.btn_vertex_clear = QPushButton("Clear current")
        self.btn_vertex_clear.clicked.connect(self.vertex_clear_current)
        nav_row.addWidget(self.btn_vertex_clear)

        self.btn_vertex_first_empty = QPushButton("First empty")
        self.btn_vertex_first_empty.clicked.connect(self.vertex_go_first_unclassified)
        nav_row.addWidget(self.btn_vertex_first_empty)

        gv.addLayout(nav_row)
        right_v.addWidget(self.gb_vertex)

        hb1 = QHBoxLayout()

        self.btn_close_poly = QPushButton("Close polygon")
        self.btn_close_poly.clicked.connect(self.close_current_polygon)
        hb1.addWidget(self.btn_close_poly)

        self.btn_undo = QPushButton("Undo last")
        self.btn_undo.clicked.connect(self.undo_last)
        hb1.addWidget(self.btn_undo)

        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.clicked.connect(self.clear_all)
        hb1.addWidget(self.btn_clear)

        right_v.addLayout(hb1)

        hb2 = QHBoxLayout()

        self.btn_finish = QPushButton("Finish and save")
        self.btn_finish.setMinimumHeight(44)
        self.btn_finish.setStyleSheet("font-weight: bold;")
        self.btn_finish.clicked.connect(self.finish_and_save)
        hb2.addWidget(self.btn_finish)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        hb2.addWidget(self.btn_cancel)

        right_v.addLayout(hb2)

        root.addWidget(right, 2)

        # ---- matplotlib event hooks: main canvas
        self.fig = self.canvas.figure
        self.ax = self.fig.add_subplot(111)
        self._cid_press = self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._cid_release = self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._cid_move = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        # ---- matplotlib event hooks: zoom canvas
        self.zoom_fig = self.zoom_canvas.figure
        self.zoom_ax = self.zoom_fig.add_subplot(111)
        self._zcid_press = self.zoom_canvas.mpl_connect("button_press_event", self._on_zoom_mouse_press)
        self._zcid_release = self.zoom_canvas.mpl_connect("button_release_event", self._on_zoom_mouse_release)
        self._zcid_move = self.zoom_canvas.mpl_connect("motion_notify_event", self._on_zoom_mouse_move)
        self._zcid_scroll = self.zoom_canvas.mpl_connect("scroll_event", self._on_zoom_scroll)

        self._rebuild_vertex_buttons()
        self._refresh_image_nav_ui()
        self._refresh_vertex_count_label()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_crop_buttons()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._redraw()
        self._redraw_zoom()

    # =========================================================
    # session / multi-image helpers
    # =========================================================
    def _current_mode_key(self):
        txt = self.cmb_mode.currentText().strip().lower() if hasattr(self, "cmb_mode") else "point count"
        if "vertex" in txt:
            return "vertices"
        if "area" in txt:
            return "areas"
        return "points"

    def _mode_key_from_index(self, idx: int):
        if int(idx) == 1:
            return "areas"
        if int(idx) == 2:
            return "vertices"
        return "points"

    def _combo_index_from_mode_key(self, mode: str):
        m = str(mode).strip().lower()
        if m == "areas":
            return 1
        if m == "vertices":
            return 2
        return 0

    def _images(self):
        if not isinstance(self._session, dict):
            return []
        imgs = self._session.get("images", [])
        return imgs if isinstance(imgs, list) else []

    def _image_count(self):
        return len(self._images())

    def _has_any_image(self):
        return self._image_count() > 0

    def _current_image_payload(self):
        imgs = self._images()
        if not imgs:
            return None
        idx = max(0, min(int(self._current_image_idx), len(imgs) - 1))
        self._current_image_idx = idx
        return imgs[idx]

    def _renumber_image_labels(self):
        for i, img in enumerate(self._images(), start=1):
            img["label"] = f"Image {i}"

    def _current_image_label(self):
        if not self._has_any_image():
            return "No images"
        return f"Image {self._current_image_idx + 1} / {self._image_count()}"

    def _reset_active_image_state(self):
        self._img = None
        self._img_original = None
        self._img_path = None
        self.crop_rect = None

        self.points = []
        self.polygons = []
        self.background_enabled = False
        self.background_component = None
        self.freehand_enabled = False

        self.grid_enabled = False
        self.grid_rows = 8
        self.grid_cols = 8
        self.grid_color = "#ffff00"

        self.vertex_assignments = {}
        self.vertex_auto_enabled = False
        self.vertex_current_index = 0
        self.vertex_grid_locked = False

        self.zoom_visible = False
        self.zoom_follow = True
        self.zoom_level = 5
        self.zoom_center = None

        self.current_poly = []
        self._drag_points = []
        self._dragging = False
        self.vertex_history = []

        self._crop_dragging = False
        self._crop_start = None
        self._crop_end = None
        self._pending_crop_rect = None
        self._last_main_xy = None

    def _load_pixels_from_path_and_crop(self, image_path, crop_rect):
        self._img = None
        self._img_original = None
        self._img_path = None
        self.crop_rect = None

        if not image_path or (not isinstance(image_path, str)):
            return

        if not os.path.exists(image_path):
            return

        try:
            img = mpimg.imread(image_path)
        except Exception:
            return

        self._img_original = np.array(img, copy=True)
        self._img = np.array(img, copy=True)
        self._img_path = image_path

        raw_crop = crop_rect
        if isinstance(raw_crop, (list, tuple)) and len(raw_crop) == 4:
            h0, w0 = self._img_original.shape[0], self._img_original.shape[1]
            x0 = max(0, min(int(round(_safe_float(raw_crop[0], default=0.0))), w0 - 1))
            y0 = max(0, min(int(round(_safe_float(raw_crop[1], default=0.0))), h0 - 1))
            x1 = max(x0 + 1, min(int(round(_safe_float(raw_crop[2], default=w0))), w0))
            y1 = max(y0 + 1, min(int(round(_safe_float(raw_crop[3], default=h0))), h0))

            if (x1 - x0) >= 2 and (y1 - y0) >= 2:
                self.crop_rect = [x0, y0, x1, y1]
                self._img = np.array(self._img_original[y0:y1, x0:x1], copy=True)
            else:
                self.crop_rect = None

    def _load_active_image_from_payload(self, payload):
        self._reset_active_image_state()

        if not isinstance(payload, dict):
            return

        self._load_pixels_from_path_and_crop(
            payload.get("image_path", None),
            payload.get("crop_rect", None)
        )

        self.points = []
        for p in payload.get("points", []):
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                x = _safe_float(p[0], default=np.nan)
                y = _safe_float(p[1], default=np.nan)
                c = str(p[2]).strip()
                if np.isfinite(x) and np.isfinite(y) and c:
                    self.points.append([float(x), float(y), c])

        self.polygons = []
        for pg in payload.get("polygons", []):
            if not isinstance(pg, dict):
                continue
            c = str(pg.get("component", "")).strip()
            if not c:
                continue
            xy = pg.get("xy", [])
            clean_xy = []
            for pt in xy:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    x = _safe_float(pt[0], default=np.nan)
                    y = _safe_float(pt[1], default=np.nan)
                    if np.isfinite(x) and np.isfinite(y):
                        clean_xy.append([float(x), float(y)])
            clean_xy = _dedupe_polygon_points(clean_xy)
            if len(clean_xy) >= 3 and _shoelace_area(clean_xy) > 1e-9:
                self.polygons.append({"component": c, "xy": clean_xy})

        self.background_enabled = bool(payload.get("background_enabled", False))
        self.background_component = payload.get("background_component", None)
        self.freehand_enabled = bool(payload.get("freehand", False))

        self.grid_enabled = bool(payload.get("grid_enabled", False))
        self.grid_rows = max(2, int(_safe_float(payload.get("grid_rows", 8), default=8)))
        self.grid_cols = max(2, int(_safe_float(payload.get("grid_cols", 8), default=8)))
        self.grid_color = str(payload.get("grid_color", "#ffff00"))

        self.vertex_auto_enabled = bool(payload.get("vertex_auto_enabled", False))
        self.vertex_current_index = max(0, int(_safe_float(payload.get("vertex_current_index", 0), default=0)))
        self.vertex_grid_locked = bool(payload.get("vertex_grid_locked", False))

        raw_va = payload.get("vertex_assignments", {})
        self.vertex_assignments = {}
        if isinstance(raw_va, dict):
            for kk, vv in raw_va.items():
                k = str(kk).strip()
                v = str(vv).strip()
                if k and v:
                    self.vertex_assignments[k] = v
        elif isinstance(raw_va, list):
            for item in raw_va:
                if isinstance(item, dict):
                    k = str(item.get("key", "")).strip()
                    v = str(item.get("component", "")).strip()
                    if k and v:
                        self.vertex_assignments[k] = v

        self.zoom_visible = bool(payload.get("zoom_visible", False))
        self.zoom_follow = bool(payload.get("zoom_follow", True))
        self.zoom_level = max(2, min(20, int(_safe_float(payload.get("zoom_level", 5), default=5))))
        self.zoom_center = None

        self.current_poly = []
        self._drag_points = []
        self._dragging = False
        self.vertex_history = []
        self._crop_dragging = False
        self._crop_start = None
        self._crop_end = None
        self._pending_crop_rect = None
        self._last_main_xy = None

    def _store_active_image_to_payload(self, payload):
        if not isinstance(payload, dict):
            return

        crop_payload = None
        if isinstance(self.crop_rect, (list, tuple)) and len(self.crop_rect) == 4:
            crop_payload = [int(round(_safe_float(v, default=0.0))) for v in self.crop_rect]

        payload["image_path"] = self._img_path
        payload["crop_rect"] = crop_payload
        payload["points"] = [[float(p[0]), float(p[1]), str(p[2])] for p in self.points if len(p) >= 3]
        payload["polygons"] = [
            {
                "component": str(pg.get("component")),
                "xy": [[float(pt[0]), float(pt[1])] for pt in pg.get("xy", []) if len(pt) >= 2]
            }
            for pg in self.polygons
            if isinstance(pg, dict)
        ]
        payload["background_enabled"] = bool(self.background_enabled)
        payload["background_component"] = self.background_component
        payload["freehand"] = bool(self.freehand_enabled)

        payload["grid_enabled"] = bool(self.grid_enabled)
        payload["grid_rows"] = int(self.grid_rows)
        payload["grid_cols"] = int(self.grid_cols)
        payload["grid_color"] = str(self.grid_color)

        payload["vertex_auto_enabled"] = bool(self.vertex_auto_enabled)
        payload["vertex_current_index"] = int(self.vertex_current_index)
        payload["vertex_grid_locked"] = bool(self.vertex_grid_locked)
        payload["vertex_assignments"] = {str(k): str(v) for k, v in self.vertex_assignments.items()}

        payload["zoom_visible"] = bool(self.zoom_visible)
        payload["zoom_follow"] = bool(self.zoom_follow)
        payload["zoom_level"] = int(self.zoom_level)

    def _save_current_image_if_possible(self):
        payload = self._current_image_payload()
        if payload is None:
            return
        self._store_active_image_to_payload(payload)

    def _activate_image(self, idx: int, store_current=True):
        imgs = self._images()

        if store_current and imgs:
            self._save_current_image_if_possible()

        if not imgs:
            self._current_image_idx = 0
            self._reset_active_image_state()
            return

        idx = max(0, min(int(idx), len(imgs) - 1))
        self._current_image_idx = idx
        self._load_active_image_from_payload(imgs[idx])

    def _load_session_if_any(self):
        sess = self.db.get_image_session(float(self.phi), create=True)
        if not isinstance(sess, dict):
            sess = {"mode": "points", "current_index": 0, "images": []}

        self._session = sess
        self._preferred_mode = str(sess.get("mode", "points")).strip().lower()
        if self._preferred_mode not in {"points", "areas", "vertices"}:
            self._preferred_mode = "points"

        self._current_image_idx = max(0, int(_safe_float(sess.get("current_index", 0), default=0)))

        if self._images():
            self._activate_image(self._current_image_idx, store_current=False)
        else:
            self._reset_active_image_state()

    def _save_session(self):
        if self._session is None:
            self._session = {"mode": "points", "current_index": 0, "images": []}

        if self._images():
            self._save_current_image_if_possible()

        self._session["mode"] = self._current_mode_key()
        self._session["current_index"] = int(self._current_image_idx)
        self.db.set_image_session(float(self.phi), self._session)

    def _refresh_image_nav_ui(self):
        has_images = self._has_any_image()

        if hasattr(self, "lbl_image_nav"):
            self.lbl_image_nav.setText(self._current_image_label())

        if hasattr(self, "btn_prev_img"):
            self.btn_prev_img.setEnabled(has_images and self._current_image_idx > 0)

        if hasattr(self, "btn_next_img"):
            self.btn_next_img.setEnabled(has_images and self._current_image_idx < (self._image_count() - 1))

        if hasattr(self, "btn_remove_img"):
            self.btn_remove_img.setEnabled(has_images)

    def _apply_active_image_to_widgets(self):
        if hasattr(self, "chk_freehand"):
            self.chk_freehand.blockSignals(True)
            self.chk_freehand.setChecked(bool(self.freehand_enabled))
            self.chk_freehand.blockSignals(False)

        if hasattr(self, "chk_bg"):
            self.chk_bg.blockSignals(True)
            self.chk_bg.setChecked(bool(self.background_enabled))
            self.chk_bg.blockSignals(False)

        if hasattr(self, "cmb_bg"):
            self.cmb_bg.blockSignals(True)
            if self.background_component in self.db.components:
                self.cmb_bg.setCurrentText(self.background_component)
            elif self.db.components:
                self.cmb_bg.setCurrentIndex(0)
                self.background_component = self.db.components[0]
            self.cmb_bg.blockSignals(False)

        if hasattr(self, "chk_grid"):
            self.chk_grid.blockSignals(True)
            self.chk_grid.setChecked(bool(self.grid_enabled))
            self.chk_grid.blockSignals(False)

        if hasattr(self, "spn_grid_cols"):
            self.spn_grid_cols.blockSignals(True)
            self.spn_grid_cols.setValue(max(2, int(self.grid_cols)))
            self.spn_grid_cols.blockSignals(False)

        if hasattr(self, "spn_grid_rows"):
            self.spn_grid_rows.blockSignals(True)
            self.spn_grid_rows.setValue(max(2, int(self.grid_rows)))
            self.spn_grid_rows.blockSignals(False)

        if hasattr(self, "lbl_grid_color"):
            self.lbl_grid_color.setStyleSheet(f"background:{self.grid_color}; border:1px solid #333;")

        if hasattr(self, "chk_vertex_auto"):
            self.chk_vertex_auto.blockSignals(True)
            self.chk_vertex_auto.setChecked(bool(self.vertex_auto_enabled))
            self.chk_vertex_auto.blockSignals(False)

        if hasattr(self, "chk_show_zoom"):
            self.chk_show_zoom.blockSignals(True)
            self.chk_show_zoom.setChecked(bool(self.zoom_visible))
            self.chk_show_zoom.blockSignals(False)

        if hasattr(self, "chk_zoom_follow"):
            self.chk_zoom_follow.blockSignals(True)
            self.chk_zoom_follow.setChecked(bool(self.zoom_follow))
            self.chk_zoom_follow.blockSignals(False)

        if hasattr(self, "spn_zoom"):
            self.spn_zoom.blockSignals(True)
            self.spn_zoom.setValue(int(self.zoom_level))
            self.spn_zoom.blockSignals(False)

        self._refresh_image_nav_ui()

    # =========================================================
    # image loading / navigation
    # =========================================================
    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if not paths:
            return

        if self._session is None:
            self._session = {"mode": "points", "current_index": 0, "images": []}

        loaded = []
        failed = []

        for path in paths:
            try:
                _ = mpimg.imread(path)
            except Exception as e:
                failed.append(f"{os.path.basename(path)}: {e}")
                continue

            loaded.append({
                "label": "",
                "image_path": path,
                "crop_rect": None,
                "points": [],
                "polygons": [],
                "background_enabled": False,
                "background_component": None,
                "freehand": False,
                "grid_enabled": False,
                "grid_rows": 8,
                "grid_cols": 8,
                "grid_color": "#ffff00",
                "vertex_auto_enabled": False,
                "vertex_current_index": 0,
                "vertex_grid_locked": False,
                "vertex_assignments": {},
                "zoom_visible": False,
                "zoom_follow": True,
                "zoom_level": 5,
            })

        if not loaded:
            QMessageBox.warning(
                self,
                "Load images",
                "No selected image could be loaded.\n\n" + "\n".join(failed[:10])
            )
            return

        # guarda la imagen actual antes de agregar más
        self._save_current_image_if_possible()

        start_idx = len(self._images())
        self._images().extend(loaded)
        self._renumber_image_labels()

        self._activate_image(start_idx, store_current=False)
        self._apply_active_image_to_widgets()
        self._refresh_crop_buttons()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_vertex_count_label()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

        if failed:
            QMessageBox.warning(
                self,
                "Load images",
                f"Loaded {len(loaded)} image(s), but some files failed:\n\n" + "\n".join(failed[:10])
            )


    def remove_current_image(self):
        """Remove the active source image and its associated points/polygons for this fraction."""
        if not self._has_any_image():
            return

        idx = int(self._current_image_idx)
        reply = QMessageBox.question(
            self,
            "Remove image",
            "Remove the current image and its associated classifications for this fraction?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        imgs = self._images()
        if 0 <= idx < len(imgs):
            imgs.pop(idx)

        self._renumber_image_labels()

        if imgs:
            new_idx = min(idx, len(imgs) - 1)
            self._activate_image(new_idx, store_current=False)
        else:
            self._current_image_idx = 0
            self._reset_active_image_state()

        self._apply_active_image_to_widgets()
        self._refresh_crop_buttons()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_vertex_count_label()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def prev_image(self):
        if not self._has_any_image():
            return
        if self._current_image_idx <= 0:
            return

        self._activate_image(self._current_image_idx - 1, store_current=True)
        self._apply_active_image_to_widgets()
        self._refresh_crop_buttons()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_vertex_count_label()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def next_image(self):
        if not self._has_any_image():
            return
        if self._current_image_idx >= (self._image_count() - 1):
            return

        self._activate_image(self._current_image_idx + 1, store_current=True)
        self._apply_active_image_to_widgets()
        self._refresh_crop_buttons()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_vertex_count_label()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()
#endregion

#region BLOQUE 7b Image analysis dialog
    # =========================================================
    # general helpers / mode / ui state
    # =========================================================
    def _image_has_analysis_content(self, img_payload):
        if not isinstance(img_payload, dict):
            return False

        if img_payload.get("points"):
            return True

        if img_payload.get("polygons"):
            return True

        if bool(img_payload.get("background_enabled", False)):
            bg = str(img_payload.get("background_component", "")).strip()
            if bg:
                return True

        raw_va = img_payload.get("vertex_assignments", {})
        if isinstance(raw_va, dict) and raw_va:
            return True
        if isinstance(raw_va, list) and len(raw_va) > 0:
            return True

        return False

    def _session_has_any_analysis_content(self):
        for img in self._images():
            if self._image_has_analysis_content(img):
                return True
        return False

    def _clear_all_analysis_data_in_session(self):
        self._save_current_image_if_possible()

        for img in self._images():
            img["points"] = []
            img["polygons"] = []
            img["background_enabled"] = False
            img["background_component"] = None
            img["freehand"] = False
            img["vertex_auto_enabled"] = False
            img["vertex_current_index"] = 0
            img["vertex_grid_locked"] = False
            img["vertex_assignments"] = {}

        if self._has_any_image():
            self._activate_image(self._current_image_idx, store_current=False)
        else:
            self._reset_active_image_state()

        self.current_poly = []
        self._drag_points = []
        self._dragging = False
        self.vertex_history = []

    def _valid_vertex_count(self):
        rows = max(0, int(self.grid_rows) - 2)
        cols = max(0, int(self.grid_cols) - 2)
        return int(rows * cols)

    def _refresh_vertex_count_label(self):
        self.lbl_vertex_total.setText(str(self._valid_vertex_count()))

    def _hide_zoom_shadow(self):
        self._zoom_shadow_visible = False
        self._redraw()

    def _trigger_zoom_shadow(self, ms=450):
        self._zoom_shadow_visible = True
        self._zoom_shadow_timer.start(int(ms))
        self._redraw()

    def _refresh_zoom_ui(self):
        show_zoom = bool(self.zoom_visible)

        self.chk_show_zoom.blockSignals(True)
        self.chk_show_zoom.setChecked(show_zoom)
        self.chk_show_zoom.blockSignals(False)

        self.gb_zoom.setVisible(show_zoom)
        self.spn_zoom.setEnabled(show_zoom)

        forced_follow = (
            self._current_mode_key() == "vertices"
            and self.vertex_auto_enabled
            and self.vertex_grid_locked
        )

        follow_value = True if forced_follow else bool(self.zoom_follow)

        self.chk_zoom_follow.blockSignals(True)
        self.chk_zoom_follow.setChecked(follow_value)
        self.chk_zoom_follow.blockSignals(False)
        self.chk_zoom_follow.setEnabled(show_zoom and (not forced_follow))

    def _on_zoom_visibility_toggled(self, checked):
        self.zoom_visible = bool(checked)
        if not self.zoom_visible:
            self._zoom_shadow_visible = False
        self._refresh_zoom_ui()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def _on_zoom_follow_toggled(self, checked):
        self.zoom_follow = bool(checked)
        self._zoom_shadow_visible = False
        self._refresh_zoom_ui()
        self._save_session()
        self._redraw_zoom()

    def _on_zoom_level_changed(self, value):
        self.zoom_level = max(2, int(value))
        if self.zoom_visible:
            self._trigger_zoom_shadow(ms=450)
        self._save_session()
        self._redraw_zoom()

    def _effective_zoom_follow(self):
        return bool(
            self._current_mode_key() == "vertices"
            and self.vertex_auto_enabled
            and self.vertex_grid_locked
        ) or bool(self.zoom_follow)

    def _clamp_zoom_center(self, cx, cy, half_side):
        if self._img is None:
            return 0.0, 0.0

        h, w = self._img.shape[0], self._img.shape[1]
        half_side = max(1.0, float(half_side))

        cx = float(cx)
        cy = float(cy)

        cx = max(half_side, min(cx, (w - 1) - half_side))
        cy = max(half_side, min(cy, (h - 1) - half_side))

        if w <= 2 * half_side:
            cx = 0.5 * (w - 1)
        if h <= 2 * half_side:
            cy = 0.5 * (h - 1)

        return cx, cy

    def _current_zoom_center(self):
        if self._img is None:
            return None

        mode = self._current_mode_key()

        if mode == "vertices" and self.vertex_auto_enabled and self.vertex_grid_locked:
            k = self._current_vertex_key()
            xy = self._vertex_xy_from_key(k)
            if xy is not None:
                self.zoom_center = xy
                return xy

        if self._effective_zoom_follow() and self._last_main_xy is not None:
            self.zoom_center = self._last_main_xy

        if self.zoom_center is None:
            if mode == "vertices" and self.vertex_grid_locked:
                k = self._current_vertex_key()
                xy = self._vertex_xy_from_key(k)
                if xy is not None:
                    self.zoom_center = xy
            if self.zoom_center is None:
                h, w = self._img.shape[0], self._img.shape[1]
                self.zoom_center = (0.5 * (w - 1), 0.5 * (h - 1))

        return self.zoom_center

    def _current_zoom_rect(self):
        if self._img is None:
            return None

        h, w = self._img.shape[0], self._img.shape[1]
        level = max(2, int(self.zoom_level))
        mode = self._current_mode_key()

        if self.grid_enabled and self.grid_cols >= 2 and self.grid_rows >= 2:
            xs, ys = self._grid_positions()
            dx = float(np.median(np.diff(xs))) if len(xs) > 1 else max(10.0, w / 8.0)
            dy = float(np.median(np.diff(ys))) if len(ys) > 1 else max(10.0, h / 8.0)
            cell = max(dx, dy)
        else:
            cell = max(10.0, min(w, h) / 8.0)

        if mode == "vertices":
            side = max(20.0, (25.0 * cell) / float(level))
        else:
            side = max(20.0, (min(w, h) * 0.40) * (5.0 / float(level)))

        half = 0.5 * side
        center = self._current_zoom_center()
        if center is None:
            return None

        cx, cy = self._clamp_zoom_center(center[0], center[1], half)

        x0 = max(0.0, cx - half)
        y0 = max(0.0, cy - half)
        x1 = min(float(w - 1), cx + half)
        y1 = min(float(h - 1), cy + half)

        return x0, y0, x1, y1

    def _update_status_text(self):
        if self._img is None:
            if self._has_any_image():
                self.lbl_status.setText("Current image could not be loaded from disk.")
            else:
                self.lbl_status.setText("No image loaded.")
            return

        if self._crop_mode:
            if self._crop_dragging:
                self.lbl_status.setText("Crop mode: release mouse to define the rectangle.")
                return
            if self._pending_crop_rect is not None:
                self.lbl_status.setText("Crop selected. Press 'Apply crop' to keep the selected rectangle.")
                return
            self.lbl_status.setText("Crop mode: drag a rectangle on the image.")
            return

        mode = self._current_mode_key()

        if mode == "vertices":
            if not self.vertex_grid_locked:
                self.lbl_status.setText("Vertex mode: adjust the grid, then press 'Fix grid'.")
                return
            if self.vertex_auto_enabled:
                self.lbl_status.setText("Vertex mode: automatic stepping active. Zoom follows the current vertex.")
            else:
                if self._selected_component:
                    self.lbl_status.setText(
                        f"Vertex mode: click an internal grid vertex to assign '{self._selected_component}'."
                    )
                else:
                    self.lbl_status.setText("Vertex mode: select a component, then click an internal grid vertex.")
            return

        if not self._selected_component:
            self.lbl_status.setText("Image loaded. Select a component first.")
            return

        if mode == "points":
            self.lbl_status.setText(f"Selected: {self._selected_component} | Click image to add points.")
        else:
            tip = "drag to draw polygon" if self.freehand_enabled else "click vertices, then 'Close polygon'"
            self.lbl_status.setText(f"Selected: {self._selected_component} | Area mode ({tip}).")

    def _on_component_selected(self, name: str):
        name = str(name).strip()
        self._selected_component = name if name in self.db.components else None
        self._update_status_text()

    def _on_mode_changed(self):
        new_mode = self._current_mode_key()
        old_mode = str(self._session.get("mode", "points")).strip().lower() if isinstance(self._session, dict) else "points"

        if old_mode not in {"points", "areas", "vertices"}:
            old_mode = "points"

        if new_mode != old_mode and self._session_has_any_analysis_content():
            ans = QMessageBox.question(
                self,
                "Change analysis mode",
                "Changing the mode for this φ will clear image-based analysis data from all loaded images "
                "for this fraction.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if ans != QMessageBox.Yes:
                self.cmb_mode.blockSignals(True)
                self.cmb_mode.setCurrentIndex(self._combo_index_from_mode_key(old_mode))
                self.cmb_mode.blockSignals(False)
                return

            self._clear_all_analysis_data_in_session()

        if new_mode == "vertices":
            self.grid_enabled = True
            self.chk_grid.blockSignals(True)
            self.chk_grid.setChecked(True)
            self.chk_grid.blockSignals(False)

        if isinstance(self._session, dict):
            self._session["mode"] = new_mode

        self._refresh_vertex_count_label()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_crop_buttons()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def _on_freehand_toggled(self, checked):
        self.freehand_enabled = bool(checked)
        self._update_status_text()
        self._save_session()

    def _on_bg_toggled(self, checked):
        self.background_enabled = bool(checked)
        self._refresh_value_table()
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def _on_bg_component_changed(self, name):
        n = str(name).strip()
        self.background_component = n if n in self.db.components else None
        self._refresh_value_table()
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def _on_grid_toggled(self, checked):
        self.grid_enabled = bool(checked)
        if self._current_mode_key() == "vertices":
            self.grid_enabled = True
            self.chk_grid.blockSignals(True)
            self.chk_grid.setChecked(True)
            self.chk_grid.blockSignals(False)
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def _clear_vertex_assignments(self):
        self.vertex_assignments = {}
        self.vertex_current_index = 0
        self.vertex_history = []
        self.vertex_auto_enabled = False

    def _on_grid_value_changed(self, _value):
        new_cols = max(2, int(self.spn_grid_cols.value()))
        new_rows = max(2, int(self.spn_grid_rows.value()))

        if new_cols == self.grid_cols and new_rows == self.grid_rows:
            self._refresh_vertex_count_label()
            return

        self.grid_cols = new_cols
        self.grid_rows = new_rows
        self._refresh_vertex_count_label()

        if self.vertex_assignments:
            self._clear_vertex_assignments()

        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def pick_grid_color(self):
        col = QColorDialog.getColor(QColor(self.grid_color), self, "Choose grid color")
        if not col.isValid():
            return
        self.grid_color = col.name()
        self.lbl_grid_color.setStyleSheet(f"background:{self.grid_color}; border:1px solid #333;")
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def fix_vertex_grid(self):
        if self._current_mode_key() != "vertices":
            return
        if self._img is None:
            QMessageBox.information(self, "Vertex mode", "Load an image first.")
            return
        if self._valid_vertex_count() <= 0:
            QMessageBox.warning(
                self,
                "Vertex mode",
                "The selected grid has no valid internal vertices.\nIncrease Grid vertices X and/or Y."
            )
            return

        self.vertex_grid_locked = True
        self.vertex_current_index = self._first_unclassified_index()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_vertex_ui()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def _refresh_mode_visibility(self):
        mode = self._current_mode_key()
        area_mode = (mode == "areas")
        vertex_mode = (mode == "vertices")

        self.chk_freehand.setEnabled(area_mode and (not self._crop_mode))
        self.chk_bg.setEnabled(area_mode)
        self.cmb_bg.setEnabled(area_mode and self.chk_bg.isChecked())
        self.btn_close_poly.setEnabled(area_mode and (not self.freehand_enabled) and (not self._crop_mode))

        if mode == "points":
            self.lbl_table_title.setText("Counts:")
        elif mode == "areas":
            self.lbl_table_title.setText("Area (%):")
        else:
            self.lbl_table_title.setText("Vertex counts:")

        self.gb_vertex.setVisible(vertex_mode)

        if vertex_mode:
            self.chk_grid.blockSignals(True)
            self.chk_grid.setChecked(True)
            self.chk_grid.blockSignals(False)
            self.grid_enabled = True
            self.chk_grid.setEnabled(False)
            self.btn_fix_vertex_grid.setEnabled(not self.vertex_grid_locked)
            self.btn_fix_vertex_grid.setText("Grid fixed" if self.vertex_grid_locked else "Fix grid")
        else:
            self.chk_grid.setEnabled(True)
            self.btn_fix_vertex_grid.setEnabled(False)
            self.btn_fix_vertex_grid.setText("Fix grid")

        allow_grid_edit = (not vertex_mode) or (not self.vertex_grid_locked)
        self.spn_grid_cols.setEnabled(allow_grid_edit)
        self.spn_grid_rows.setEnabled(allow_grid_edit)

    def _refresh_crop_buttons(self):
        has_image = self._img is not None
        has_pending = self._pending_crop_rect is not None
        has_crop = self.crop_rect is not None

        self.btn_crop_mode.setEnabled(has_image)
        self.btn_apply_crop.setEnabled(has_image and has_pending)
        self.btn_reset_crop.setEnabled(has_image and has_crop)

        if not has_image:
            self.btn_crop_mode.blockSignals(True)
            self.btn_crop_mode.setChecked(False)
            self.btn_crop_mode.blockSignals(False)

    # =========================================================
    # crop helpers
    # =========================================================
    def _normalize_crop_rect(self, p0, p1):
        if self._img is None or p0 is None or p1 is None:
            return None

        h, w = self._img.shape[0], self._img.shape[1]

        x0 = int(np.floor(min(float(p0[0]), float(p1[0]))))
        y0 = int(np.floor(min(float(p0[1]), float(p1[1]))))
        x1 = int(np.ceil(max(float(p0[0]), float(p1[0]))))
        y1 = int(np.ceil(max(float(p0[1]), float(p1[1]))))

        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))

        if (x1 - x0) < 2 or (y1 - y0) < 2:
            return None

        return [x0, y0, x1, y1]

    def _shift_points(self, dx, dy):
        out = []
        for p in self.points:
            if len(p) < 3:
                continue
            out.append([float(p[0]) + dx, float(p[1]) + dy, p[2]])
        self.points = out

    def _shift_polygons(self, dx, dy):
        out = []
        for pg in self.polygons:
            xy = pg.get("xy", [])
            new_xy = [[float(pt[0]) + dx, float(pt[1]) + dy] for pt in xy if len(pt) >= 2]
            new_xy = _dedupe_polygon_points(new_xy)
            if len(new_xy) >= 3 and _shoelace_area(new_xy) > 1e-9:
                out.append({"component": pg.get("component"), "xy": new_xy})
        self.polygons = out

    def _clip_points_to_current_rect(self, rect):
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        out = []
        for p in self.points:
            if len(p) < 3:
                continue
            x = float(p[0])
            y = float(p[1])
            if (x0 <= x < x1) and (y0 <= y < y1):
                out.append([x - x0, y - y0, p[2]])
        self.points = out

    def _clip_polygons_to_current_rect(self, rect):
        if rect is None:
            return
        x0, y0, x1, y1 = rect
        xmax = float(x1 - 1)
        ymax = float(y1 - 1)

        out = []
        for pg in self.polygons:
            xy = pg.get("xy", [])
            clipped = _clip_polygon_to_rect(xy, float(x0), float(y0), xmax, ymax)
            if len(clipped) < 3:
                continue
            shifted = [[float(pt[0]) - x0, float(pt[1]) - y0] for pt in clipped]
            shifted = _dedupe_polygon_points(shifted)
            if len(shifted) >= 3 and _shoelace_area(shifted) > 1e-9:
                out.append({"component": pg.get("component"), "xy": shifted})

        self.polygons = out

    def _on_crop_mode_toggled(self, checked):
        if checked and self._img is None:
            QMessageBox.information(self, "Crop", "Load an image first.")
            self.btn_crop_mode.blockSignals(True)
            self.btn_crop_mode.setChecked(False)
            self.btn_crop_mode.blockSignals(False)
            return

        self._crop_mode = bool(checked)
        self._crop_dragging = False
        self._crop_start = None
        self._crop_end = None
        self._pending_crop_rect = None

        self.btn_crop_mode.setText("Cancel crop" if self._crop_mode else "Select crop")

        self._refresh_crop_buttons()
        self._refresh_mode_visibility()
        self._update_status_text()
        self._redraw()
        self._redraw_zoom()

    def apply_crop(self):
        if self._img is None:
            QMessageBox.warning(self, "Crop", "Load an image first.")
            return

        rect = self._pending_crop_rect
        if rect is None:
            QMessageBox.information(self, "Crop", "Select a crop rectangle first.")
            return

        x0, y0, x1, y1 = rect
        if (x1 - x0) < 2 or (y1 - y0) < 2:
            QMessageBox.warning(self, "Crop", "Crop rectangle is too small.")
            return

        self._clip_points_to_current_rect(rect)
        self._clip_polygons_to_current_rect(rect)

        if self.vertex_assignments:
            self._clear_vertex_assignments()

        self.vertex_grid_locked = False
        self.current_poly = []
        self._drag_points = []
        self._dragging = False

        self._img = np.array(self._img[y0:y1, x0:x1], copy=True)

        if self.crop_rect is None:
            base_x0, base_y0 = 0, 0
        else:
            base_x0, base_y0 = int(self.crop_rect[0]), int(self.crop_rect[1])

        self.crop_rect = [
            int(base_x0 + x0),
            int(base_y0 + y0),
            int(base_x0 + x1),
            int(base_y0 + y1),
        ]

        self._pending_crop_rect = None
        self._crop_start = None
        self._crop_end = None
        self._crop_dragging = False
        self._last_main_xy = None
        self.zoom_center = None

        if self.btn_crop_mode.isChecked():
            self.btn_crop_mode.blockSignals(True)
            self.btn_crop_mode.setChecked(False)
            self.btn_crop_mode.blockSignals(False)
        self._crop_mode = False
        self.btn_crop_mode.setText("Select crop")

        self._refresh_crop_buttons()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_vertex_count_label()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def reset_crop(self):
        if self._img_original is None:
            return
        if self.crop_rect is None:
            return

        x0, y0 = int(self.crop_rect[0]), int(self.crop_rect[1])

        self._shift_points(dx=x0, dy=y0)
        self._shift_polygons(dx=x0, dy=y0)

        self._img = np.array(self._img_original, copy=True)
        self.crop_rect = None
        self._pending_crop_rect = None
        self._crop_start = None
        self._crop_end = None
        self._crop_dragging = False
        self.current_poly = []
        self._drag_points = []
        self._dragging = False

        self._clear_vertex_assignments()
        self.vertex_grid_locked = False
        self._last_main_xy = None
        self.zoom_center = None

        if self.btn_crop_mode.isChecked():
            self.btn_crop_mode.blockSignals(True)
            self.btn_crop_mode.setChecked(False)
            self.btn_crop_mode.blockSignals(False)
        self._crop_mode = False
        self.btn_crop_mode.setText("Select crop")

        self._refresh_crop_buttons()
        self._refresh_mode_visibility()
        self._refresh_zoom_ui()
        self._refresh_vertex_count_label()
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()
#endregion

#region BLOQUE 7c Image analysis dialog
    # =========================================================
    # vertex helpers
    # =========================================================
    def _vertex_order_keys(self):
        rows = max(0, int(self.grid_rows))
        cols = max(0, int(self.grid_cols))
        if rows < 3 or cols < 3:
            return []
        return [f"{r}:{c}" for r in range(1, rows - 1) for c in range(1, cols - 1)]

    def _grid_positions(self):
        if self._img is None:
            return np.array([], dtype=float), np.array([], dtype=float)

        h, w = self._img.shape[0], self._img.shape[1]
        cols = max(2, int(self.grid_cols))
        rows = max(2, int(self.grid_rows))

        xs = np.linspace(0.0, float(w - 1), cols)
        ys = np.linspace(0.0, float(h - 1), rows)
        return xs, ys

    def _vertex_xy_from_key(self, key):
        try:
            s = str(key).strip()
            r_str, c_str = s.split(":")
            r = int(r_str)
            c = int(c_str)
        except Exception:
            return None

        xs, ys = self._grid_positions()
        if xs.size == 0 or ys.size == 0:
            return None
        if r < 0 or r >= len(ys) or c < 0 or c >= len(xs):
            return None

        return float(xs[c]), float(ys[r])

    def _nearest_vertex_key(self, xy):
        if self._img is None or xy is None:
            return None

        x, y = float(xy[0]), float(xy[1])
        xs, ys = self._grid_positions()
        if xs.size == 0 or ys.size == 0:
            return None

        c = int(np.argmin(np.abs(xs - x)))
        r = int(np.argmin(np.abs(ys - y)))

        if r <= 0 or r >= (len(ys) - 1) or c <= 0 or c >= (len(xs) - 1):
            return None

        return f"{r}:{c}"

    def _current_vertex_key(self):
        keys = self._vertex_order_keys()
        if not keys:
            return None
        idx = max(0, min(int(self.vertex_current_index), len(keys) - 1))
        self.vertex_current_index = idx
        return keys[idx]

    def _set_current_vertex_key(self, key):
        keys = self._vertex_order_keys()
        if key in keys:
            self.vertex_current_index = keys.index(key)
        else:
            self.vertex_current_index = 0

    def _first_unclassified_index(self):
        keys = self._vertex_order_keys()
        for i, k in enumerate(keys):
            if k not in self.vertex_assignments:
                return i
        return 0 if keys else 0

    def _vertex_counts_current_image(self):
        counts = {c: 0 for c in self.db.components}
        for comp in self.vertex_assignments.values():
            if comp in counts:
                counts[comp] += 1
        return counts

    def _rebuild_vertex_buttons(self):
        while self.vertex_map_layout.count():
            item = self.vertex_map_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        self.vertex_buttons = []

        ncols = 2
        for i, comp in enumerate(self.db.components):
            shortcut = self.vertex_shortcut_keys[i] if i < len(self.vertex_shortcut_keys) else ""
            label = f"{shortcut} — {comp}" if shortcut else comp
            btn = QPushButton(label)
            btn.setStyleSheet(f"background:{self.db.colors.get(comp, '#cccccc')};")
            btn.clicked.connect(lambda _, c=comp: self._assign_current_vertex(c))

            row = i // ncols
            col = i % ncols
            self.vertex_map_layout.addWidget(btn, row, col)
            self.vertex_buttons.append((shortcut, comp, btn))

    def _refresh_vertex_ui(self):
        vertex_mode = (self._current_mode_key() == "vertices")
        self._refresh_vertex_count_label()
        self.gb_vertex.setVisible(vertex_mode)

        auto_allowed = vertex_mode and self.vertex_grid_locked
        auto_on = bool(self.vertex_auto_enabled) and auto_allowed

        self.chk_vertex_auto.blockSignals(True)
        self.chk_vertex_auto.setChecked(bool(self.vertex_auto_enabled))
        self.chk_vertex_auto.setEnabled(auto_allowed)
        self.chk_vertex_auto.blockSignals(False)

        for shortcut, comp, btn in self.vertex_buttons:
            btn.setEnabled(vertex_mode and auto_on)

        self.btn_vertex_prev.setEnabled(vertex_mode and auto_on)
        self.btn_vertex_skip.setEnabled(vertex_mode and auto_on)
        self.btn_vertex_clear.setEnabled(vertex_mode and self.vertex_grid_locked)
        self.btn_vertex_first_empty.setEnabled(vertex_mode and self.vertex_grid_locked)

        if not vertex_mode:
            return

        if not self.vertex_grid_locked:
            self.lbl_vertex_grid_state.setText(
                "Grid is editable. Choose Grid vertices X/Y, then press 'Fix grid' when you are ready."
            )
            self.lbl_vertex_info.setText(f"{self._valid_vertex_count()} valid internal vertices")
            return

        self.lbl_vertex_grid_state.setText("Grid is fixed for this image. Border vertices are excluded.")

        keys = self._vertex_order_keys()
        total = len(keys)

        if total <= 0:
            self.lbl_vertex_info.setText("No valid internal vertices.")
            return

        self.vertex_current_index = max(0, min(int(self.vertex_current_index), total - 1))
        key = self._current_vertex_key()
        comp = self.vertex_assignments.get(key, None)

        try:
            rr, cc = [int(v) for v in key.split(":")]
        except Exception:
            rr, cc = 0, 0

        assigned_n = len(self.vertex_assignments)
        comp_txt = comp if comp else "Unclassified"
        self.lbl_vertex_info.setText(
            f"Current vertex: {self.vertex_current_index + 1}/{total} | row={rr + 1}, col={cc + 1} | "
            f"Assigned={comp_txt} | Classified={assigned_n}/{total}"
        )

    def _assign_current_vertex(self, component_name):
        if not self.vertex_grid_locked:
            QMessageBox.information(self, "Vertex mode", "Fix the grid first.")
            return
        key = self._current_vertex_key()
        if key is None:
            return
        self._assign_vertex_key(key, component_name, advance=bool(self.vertex_auto_enabled))

    def _assign_vertex_key(self, key, component_name, advance=True):
        if not self.vertex_grid_locked:
            QMessageBox.information(self, "Vertex mode", "Fix the grid first.")
            return

        if component_name not in self.db.components:
            return

        keys = self._vertex_order_keys()
        if key not in keys:
            return

        idx_before = keys.index(key)
        old_comp = self.vertex_assignments.get(key, None)

        self.vertex_assignments[key] = component_name

        idx_after = idx_before
        if advance and idx_before < len(keys) - 1:
            idx_after = idx_before + 1

        self.vertex_history.append({
            "key": key,
            "old_comp": old_comp,
            "new_comp": component_name,
            "index_before": idx_before,
            "index_after": idx_after,
        })

        self.vertex_current_index = idx_after
        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def _on_vertex_auto_toggled(self, checked):
        if checked and not self.vertex_grid_locked:
            self.chk_vertex_auto.blockSignals(True)
            self.chk_vertex_auto.setChecked(False)
            self.chk_vertex_auto.blockSignals(False)
            self.vertex_auto_enabled = False
            QMessageBox.information(self, "Vertex mode", "Fix the grid first.")
            return

        self.vertex_auto_enabled = bool(checked)
        if self.vertex_auto_enabled:
            self.vertex_current_index = self._first_unclassified_index()

        self._refresh_zoom_ui()
        self._refresh_vertex_ui()
        self._update_status_text()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def vertex_prev(self):
        keys = self._vertex_order_keys()
        if not keys:
            return
        self.vertex_current_index = max(0, int(self.vertex_current_index) - 1)
        self._refresh_vertex_ui()
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def vertex_skip(self):
        keys = self._vertex_order_keys()
        if not keys:
            return
        self.vertex_current_index = min(len(keys) - 1, int(self.vertex_current_index) + 1)
        self._refresh_vertex_ui()
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def vertex_clear_current(self):
        if not self.vertex_grid_locked:
            return

        key = self._current_vertex_key()
        if key is None:
            return

        old_comp = self.vertex_assignments.get(key, None)
        if key in self.vertex_assignments:
            del self.vertex_assignments[key]

        self.vertex_history.append({
            "key": key,
            "old_comp": old_comp,
            "new_comp": None,
            "index_before": int(self.vertex_current_index),
            "index_after": int(self.vertex_current_index),
        })

        self._refresh_vertex_ui()
        self._refresh_value_table()
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    def vertex_go_first_unclassified(self):
        self.vertex_current_index = self._first_unclassified_index()
        self._refresh_vertex_ui()
        self._redraw()
        self._redraw_zoom()
        self._save_session()

    # =========================================================
    # zoom panel interactions
    # =========================================================
    def _on_zoom_mouse_press(self, event):
        if not self.zoom_visible:
            return
        if self._effective_zoom_follow():
            return
        if event.inaxes != self.zoom_ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._zoom_dragging = True
        self._zoom_drag_last_xy = (float(event.xdata), float(event.ydata))
        self._zoom_shadow_visible = True
        self._zoom_shadow_timer.stop()
        self._redraw()

    def _on_zoom_mouse_move(self, event):
        if not self._zoom_dragging:
            return
        if event.inaxes != self.zoom_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        x_prev, y_prev = self._zoom_drag_last_xy

        dx = x - x_prev
        dy = y - y_prev

        if self.zoom_center is None:
            self.zoom_center = self._current_zoom_center()

        if self.zoom_center is None:
            return

        rect = self._current_zoom_rect()
        if rect is None:
            return

        x0, y0, x1, y1 = rect
        half = 0.5 * max(x1 - x0, y1 - y0)

        cx = float(self.zoom_center[0]) - dx
        cy = float(self.zoom_center[1]) - dy
        cx, cy = self._clamp_zoom_center(cx, cy, half)

        self.zoom_center = (cx, cy)
        self._zoom_drag_last_xy = (x, y)

        self._zoom_shadow_visible = True
        self._redraw()
        self._redraw_zoom()

    def _on_zoom_mouse_release(self, event):
        if not self._zoom_dragging:
            return
        self._zoom_dragging = False
        self._zoom_drag_last_xy = None
        self._zoom_shadow_visible = False
        self._redraw()

    def _on_zoom_scroll(self, event):
        if not self.zoom_visible:
            return
        if event.inaxes != self.zoom_ax:
            return

        if event.button == "up":
            self.zoom_level = min(20, int(self.zoom_level) + 1)
        elif event.button == "down":
            self.zoom_level = max(2, int(self.zoom_level) - 1)
        else:
            return

        self.spn_zoom.blockSignals(True)
        self.spn_zoom.setValue(int(self.zoom_level))
        self.spn_zoom.blockSignals(False)

        if event.xdata is not None and event.ydata is not None and not self._effective_zoom_follow():
            self.zoom_center = (float(event.xdata), float(event.ydata))

        self._trigger_zoom_shadow(ms=450)
        self._save_session()
        self._redraw_zoom()

    # =========================================================
    # drawing logic
    # =========================================================
    def _event_xy(self, event):
        if event.inaxes != self.ax:
            return None
        if self._img is None:
            return None
        if event.xdata is None or event.ydata is None:
            return None
        return float(event.xdata), float(event.ydata)

    def _on_mouse_press(self, event):
        xy = self._event_xy(event)
        if xy is None:
            return
        if event.button != 1:
            return

        if self._effective_zoom_follow():
            self._last_main_xy = xy
            self.zoom_center = xy
            self._redraw_zoom()

        if self._crop_mode:
            self._crop_dragging = True
            self._crop_start = list(xy)
            self._crop_end = list(xy)
            self._pending_crop_rect = None
            self._refresh_crop_buttons()
            self._update_status_text()
            self._redraw()
            return

        mode = self._current_mode_key()

        if mode == "vertices":
            if not self.vertex_grid_locked:
                QMessageBox.information(self, "Vertex mode", "Adjust the grid first, then press 'Fix grid'.")
                return

            key = self._nearest_vertex_key(xy)
            if key is None:
                return

            if self.vertex_auto_enabled:
                self._set_current_vertex_key(key)
                self._refresh_vertex_ui()
                self._redraw()
                self._redraw_zoom()
                self._save_session()
                return

            if not self._selected_component:
                QMessageBox.information(self, "Vertex mode", "Select a component first.")
                return

            self._assign_vertex_key(key, self._selected_component, advance=False)
            return

        if not self._selected_component:
            QMessageBox.information(self, "Image analysis", "Select a component first.")
            return

        if mode == "points":
            x, y = xy
            self.points.append([x, y, self._selected_component])
            self._refresh_value_table()
            self._save_session()
            self._redraw()
            self._redraw_zoom()
            return

        if self.freehand_enabled:
            self._dragging = True
            self._drag_points = [list(xy)]
            return

        self.current_poly.append(list(xy))
        if getattr(event, "dblclick", False):
            self.close_current_polygon()
        else:
            self._redraw()
            self._redraw_zoom()

    def _on_mouse_move(self, event):
        xy = self._event_xy(event)

        if xy is not None and self._effective_zoom_follow():
            self._last_main_xy = xy
            self.zoom_center = xy
            self._redraw_zoom()

        if self._crop_mode and self._crop_dragging:
            if xy is None:
                return
            self._crop_end = list(xy)
            self._redraw()
            return

        if not self._dragging:
            return
        if xy is None:
            return

        self._drag_points.append(list(xy))
        self._redraw(preview_drag=True)
        self._redraw_zoom()

    def _on_mouse_release(self, event):
        xy = self._event_xy(event)

        if self._crop_mode and self._crop_dragging:
            self._crop_dragging = False
            if xy is not None:
                self._crop_end = list(xy)
            rect = self._normalize_crop_rect(self._crop_start, self._crop_end)
            self._pending_crop_rect = rect
            self._refresh_crop_buttons()
            self._update_status_text()
            self._redraw()
            return

        if not self._dragging:
            return

        self._dragging = False

        if len(self._drag_points) < 3:
            self._drag_points = []
            self._redraw()
            self._redraw_zoom()
            return

        self._try_add_polygon(self._drag_points, self._selected_component)
        self._drag_points = []
        self._refresh_value_table()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def close_current_polygon(self):
        if self._current_mode_key() != "areas":
            return
        if len(self.current_poly) < 3:
            QMessageBox.information(self, "Area polygons", "Need at least 3 vertices to close a polygon.")
            return
        self._try_add_polygon(self.current_poly, self._selected_component)
        self.current_poly = []
        self._refresh_value_table()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def _try_add_polygon(self, poly_xy, component_name):
        if not component_name:
            QMessageBox.information(self, "Area polygons", "Select a component first.")
            return

        cleaned = []
        for pt in poly_xy:
            if not cleaned:
                cleaned.append(pt)
            else:
                if np.hypot(cleaned[-1][0] - pt[0], cleaned[-1][1] - pt[1]) > 1e-6:
                    cleaned.append(pt)

        cleaned = _dedupe_polygon_points(cleaned)

        if len(cleaned) < 3:
            QMessageBox.information(self, "Area polygons", "Polygon has less than 3 valid points.")
            return

        if _shoelace_area(cleaned) <= 1e-9:
            QMessageBox.information(self, "Area polygons", "Polygon area is zero.")
            return

        for old in self.polygons:
            if polygons_overlap(cleaned, old["xy"]):
                QMessageBox.warning(self, "Area polygons", "Overlapping polygons are not allowed.")
                return

        self.polygons.append({"component": str(component_name), "xy": [list(p) for p in cleaned]})

    # =========================================================
    # per-image + aggregated stats
    # =========================================================
    def _counts_current_image(self):
        counts = {c: 0 for c in self.db.components}
        for p in self.points:
            if len(p) >= 3 and p[2] in counts:
                counts[p[2]] += 1
        return counts

    def _areas_raw_current_image(self):
        out = {c: 0.0 for c in self.db.components}
        if self._img is None:
            return out, 0.0, 0.0

        h, w = self._img.shape[0], self._img.shape[1]
        total_img_area = float(max(1.0, h * w))

        poly_sum = 0.0
        for pg in self.polygons:
            comp = pg.get("component")
            xy = pg.get("xy", [])
            a = max(0.0, _shoelace_area(xy))
            if comp in out:
                out[comp] += a
            poly_sum += a

        if self.background_enabled and self.background_component in out:
            bg = max(0.0, total_img_area - poly_sum)
            out[self.background_component] += bg

        return out, poly_sum, total_img_area

    def _areas_percent_current_image(self):
        out = {c: 0.0 for c in self.db.components}
        if self._img is None:
            return out

        raw, poly_sum, total_img_area = self._areas_raw_current_image()

        if self.background_enabled and self.background_component in out:
            denom = total_img_area
        else:
            denom = max(1e-12, poly_sum)

        for c in out:
            out[c] = 100.0 * raw.get(c, 0.0) / denom if denom > 0 else 0.0

        return out

    def _aggregate_point_counts_all_images(self):
        out = {c: 0 for c in self.db.components}

        self._save_current_image_if_possible()

        for img in self._images():
            for p in img.get("points", []):
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    comp = str(p[2]).strip()
                    if comp in out:
                        out[comp] += 1
        return out

    def _aggregate_vertex_counts_all_images(self):
        out = {c: 0 for c in self.db.components}

        self._save_current_image_if_possible()

        for img in self._images():
            raw_va = img.get("vertex_assignments", {})
            if isinstance(raw_va, dict):
                vals = raw_va.values()
            else:
                vals = []

            for comp in vals:
                comp = str(comp).strip()
                if comp in out:
                    out[comp] += 1

        return out

    def _aggregate_area_percentages_all_images(self):
        out = {c: 0.0 for c in self.db.components}
        total_denom = 0.0
        has_content = False

        self._save_current_image_if_possible()

        for img in self._images():
            image_path = img.get("image_path", None)
            if not image_path or (not isinstance(image_path, str)) or (not os.path.exists(image_path)):
                continue

            try:
                arr = mpimg.imread(image_path)
            except Exception:
                continue

            img_original = np.array(arr, copy=True)
            crop_rect = img.get("crop_rect", None)
            img_view = np.array(img_original, copy=True)

            if isinstance(crop_rect, (list, tuple)) and len(crop_rect) == 4:
                h0, w0 = img_original.shape[0], img_original.shape[1]
                x0 = max(0, min(int(round(_safe_float(crop_rect[0], default=0.0))), w0 - 1))
                y0 = max(0, min(int(round(_safe_float(crop_rect[1], default=0.0))), h0 - 1))
                x1 = max(x0 + 1, min(int(round(_safe_float(crop_rect[2], default=w0))), w0))
                y1 = max(y0 + 1, min(int(round(_safe_float(crop_rect[3], default=h0))), h0))
                if (x1 - x0) >= 2 and (y1 - y0) >= 2:
                    img_view = np.array(img_original[y0:y1, x0:x1], copy=True)

            h, w = img_view.shape[0], img_view.shape[1]
            total_img_area = float(max(1.0, h * w))

            raw = {c: 0.0 for c in self.db.components}
            poly_sum = 0.0

            for pg in img.get("polygons", []):
                if not isinstance(pg, dict):
                    continue
                comp = str(pg.get("component", "")).strip()
                if comp not in raw:
                    continue

                xy = []
                for pt in pg.get("xy", []):
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        x = _safe_float(pt[0], default=np.nan)
                        y = _safe_float(pt[1], default=np.nan)
                        if np.isfinite(x) and np.isfinite(y):
                            xy.append([float(x), float(y)])

                xy = _dedupe_polygon_points(xy)
                if len(xy) < 3:
                    continue

                a = max(0.0, _shoelace_area(xy))
                raw[comp] += a
                poly_sum += a

            bg_enabled = bool(img.get("background_enabled", False))
            bg_comp = img.get("background_component", None)
            if bg_enabled and bg_comp in raw:
                bg = max(0.0, total_img_area - poly_sum)
                raw[bg_comp] += bg
                denom = total_img_area
            else:
                denom = max(0.0, poly_sum)

            if denom > 0:
                has_content = True
                total_denom += denom
                for c in out:
                    out[c] += raw.get(c, 0.0)

        if (not has_content) or total_denom <= 0:
            return {c: 0.0 for c in self.db.components}

        for c in out:
            out[c] = 100.0 * out[c] / total_denom

        return out

    def _refresh_value_table(self):
        mode = self._current_mode_key()
        self.tbl_counts.setRowCount(len(self.db.components))

        if mode == "points":
            vals = self._aggregate_point_counts_all_images()
        elif mode == "areas":
            vals = self._aggregate_area_percentages_all_images()
        else:
            vals = self._aggregate_vertex_counts_all_images()

        for i, c in enumerate(self.db.components):
            it_c = QTableWidgetItem(c)
            it_c.setFlags(it_c.flags() & ~Qt.ItemIsEditable)
            it_c.setBackground(QColor(self.db.colors.get(c, "#cccccc")))
            self.tbl_counts.setItem(i, 0, it_c)

            if mode == "areas":
                txt = f"{float(vals.get(c, 0.0)):.3f}"
            else:
                txt = str(int(vals.get(c, 0)))

            it_v = QTableWidgetItem(txt)
            it_v.setFlags(it_v.flags() & ~Qt.ItemIsEditable)
            self.tbl_counts.setItem(i, 1, it_v)

        try:
            self.tbl_counts.resizeColumnsToContents()
        except Exception:
            pass

    # =========================================================
    # edit actions
    # =========================================================
    def undo_last(self):
        if self._crop_mode:
            self._pending_crop_rect = None
            self._crop_start = None
            self._crop_end = None
            self._crop_dragging = False
            self._refresh_crop_buttons()
            self._update_status_text()
            self._redraw()
            return

        mode = self._current_mode_key()

        if mode == "vertices":
            if not self.vertex_history:
                return

            step = self.vertex_history.pop()
            key = step.get("key")
            old_comp = step.get("old_comp", None)
            idx_before = int(step.get("index_before", 0))

            if old_comp is None:
                self.vertex_assignments.pop(key, None)
            else:
                self.vertex_assignments[key] = old_comp

            self.vertex_current_index = max(0, idx_before)
            self._refresh_vertex_ui()
            self._refresh_value_table()
            self._redraw()
            self._redraw_zoom()
            self._save_session()
            return

        if mode == "points":
            if self.points:
                self.points.pop()
        else:
            if self.freehand_enabled:
                if self.polygons:
                    self.polygons.pop()
            else:
                if self.current_poly:
                    self.current_poly.pop()
                elif self.polygons:
                    self.polygons.pop()

        self._refresh_value_table()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    def clear_all(self):
        if self._crop_mode:
            self._pending_crop_rect = None
            self._crop_start = None
            self._crop_end = None
            self._crop_dragging = False
            self._refresh_crop_buttons()
            self._update_status_text()
            self._redraw()
            return

        mode = self._current_mode_key()

        if mode == "points":
            self.points = []
        elif mode == "areas":
            self.polygons = []
            self.current_poly = []
            self._drag_points = []
            self._dragging = False
            self.background_enabled = False
        else:
            self._clear_vertex_assignments()
            self._refresh_vertex_ui()

        self._refresh_value_table()
        self._save_session()
        self._redraw()
        self._redraw_zoom()

    # =========================================================
    # draw helpers
    # =========================================================
    def _draw_grid_on_axes(self, ax, alpha=0.90, linewidth=0.9):
        if (self._img is None) or (not self.grid_enabled):
            return

        xs, ys = self._grid_positions()
        if xs.size == 0 or ys.size == 0:
            return

        col = self.grid_color if self.grid_color else "#ffff00"

        for x in xs:
            ax.plot([x, x], [ys[0], ys[-1]], color=col, linewidth=linewidth, alpha=alpha, zorder=2)

        for y in ys:
            ax.plot([xs[0], xs[-1]], [y, y], color=col, linewidth=linewidth, alpha=alpha, zorder=2)

    def _draw_zoom_grid_on_axes(self, ax):
        if self._current_mode_key() == "vertices":
            self._draw_grid_on_axes(ax, alpha=0.20, linewidth=0.35)
        else:
            self._draw_grid_on_axes(ax, alpha=0.55, linewidth=0.65)

    def _draw_zoom_center_cross_on_axes(self, ax):
        rect = self._current_zoom_rect()
        if rect is None:
            return

        x0, y0, x1, y1 = rect
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)

        ax.axvline(cx, color="#000000", linewidth=0.4, alpha=0.9, zorder=20)
        ax.axhline(cy, color="#000000", linewidth=0.4, alpha=0.9, zorder=20)

    def _draw_points_on_axes(self, ax, marker_size=24):
        by_comp = {}
        for x, y, c in self.points:
            by_comp.setdefault(c, []).append((x, y))

        for c, pts in by_comp.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(
                xs, ys, s=marker_size,
                c=self.db.colors.get(c, "#ff0000"),
                edgecolors="black", linewidths=0.4, zorder=5
            )

    def _draw_polygons_on_axes(self, ax, include_preview=True):
        for pg in self.polygons:
            c = pg.get("component")
            xy = pg.get("xy", [])
            if len(xy) < 3:
                continue
            arr = np.array(xy, dtype=float)
            color = self.db.colors.get(c, "#888888")
            ax.fill(arr[:, 0], arr[:, 1], color=color, alpha=0.35, linewidth=0, zorder=4)
            ax.plot(arr[:, 0], arr[:, 1], color="black", linewidth=0.8, zorder=5)
            ax.plot([arr[-1, 0], arr[0, 0]], [arr[-1, 1], arr[0, 1]], color="black", linewidth=0.8, zorder=5)

        if include_preview and len(self.current_poly) > 0:
            arr = np.array(self.current_poly, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="#222222", linewidth=1.0, zorder=6)
            ax.scatter(arr[:, 0], arr[:, 1], s=10, c="#222222", zorder=6)

        if include_preview and len(self._drag_points) > 1 and self._dragging:
            arr = np.array(self._drag_points, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], color="#222222", linewidth=1.0, zorder=6)

    def _draw_vertex_assignments_on_axes(self, ax, zoom_mode=False):
        for key, comp in self.vertex_assignments.items():
            xy = self._vertex_xy_from_key(key)
            if xy is None:
                continue
            x, y = xy
            color = self.db.colors.get(comp, "#ff0000")
            ax.scatter([x], [y], s=50 if zoom_mode else 55, c=color, edgecolors="black", linewidths=0.7, zorder=6)

        if self.vertex_grid_locked:
            k = self._current_vertex_key()
            xy = self._vertex_xy_from_key(k)
            if xy is not None:
                x, y = xy
                ax.scatter([x], [y], s=110 if zoom_mode else 120, marker="x", c="#000000", linewidths=1.8, zorder=8)
                ax.scatter([x], [y], s=150 if zoom_mode else 170, marker="x", c="#ffffff", linewidths=0.8, zorder=7)

    def _redraw(self, preview_drag=False):
        self.ax.clear()

        if self._img is not None:
            self.ax.imshow(self._img, interpolation="nearest")
            h, w = self._img.shape[0], self._img.shape[1]
            self.ax.set_xlim(0, w - 1)
            self.ax.set_ylim(h - 1, 0)

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if self._img is not None and self.grid_enabled:
            self._draw_grid_on_axes(self.ax, alpha=0.90, linewidth=0.9)

        mode = self._current_mode_key()
        if mode == "points":
            self._draw_points_on_axes(self.ax, marker_size=self.point_marker_size_main)
        elif mode == "areas":
            self._draw_polygons_on_axes(self.ax, include_preview=True)
            if preview_drag and len(self._drag_points) > 1:
                arr = np.array(self._drag_points, dtype=float)
                self.ax.plot(arr[:, 0], arr[:, 1], color="#222222", linewidth=1.0, zorder=6)
        else:
            self._draw_vertex_assignments_on_axes(self.ax, zoom_mode=False)

        if self._zoom_shadow_visible and self.zoom_visible:
            rect = self._current_zoom_rect()
            if rect is not None:
                x0, y0, x1, y1 = rect
                patch = Rectangle(
                    (x0, y0),
                    float(x1 - x0),
                    float(y1 - y0),
                    facecolor="#000000",
                    edgecolor="#000000",
                    alpha=0.15,
                    linewidth=0.8,
                    zorder=9
                )
                self.ax.add_patch(patch)

        crop_preview = None
        if self._crop_dragging and self._crop_start is not None and self._crop_end is not None:
            crop_preview = self._normalize_crop_rect(self._crop_start, self._crop_end)
        elif self._pending_crop_rect is not None:
            crop_preview = self._pending_crop_rect

        if crop_preview is not None:
            x0, y0, x1, y1 = crop_preview
            rect = Rectangle(
                (x0, y0),
                float(x1 - x0),
                float(y1 - y0),
                fill=False,
                edgecolor="#ffd400",
                linewidth=1.6,
                linestyle="--",
                zorder=10
            )
            self.ax.add_patch(rect)

        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        self.canvas.draw()

    def _redraw_zoom(self):
        self.zoom_ax.clear()

        if (not self.zoom_visible) or (self._img is None):
            self.zoom_ax.set_xticks([])
            self.zoom_ax.set_yticks([])
            self.zoom_canvas.draw()
            return

        self.zoom_ax.imshow(self._img, interpolation="nearest")
        h, w = self._img.shape[0], self._img.shape[1]

        rect = self._current_zoom_rect()
        if rect is None:
            self.zoom_ax.set_xlim(0, w - 1)
            self.zoom_ax.set_ylim(h - 1, 0)
        else:
            x0, y0, x1, y1 = rect
            self.zoom_ax.set_xlim(x0, x1)
            self.zoom_ax.set_ylim(y1, y0)

        self.zoom_ax.set_xticks([])
        self.zoom_ax.set_yticks([])

        if self._img is not None and self.grid_enabled:
            self._draw_zoom_grid_on_axes(self.zoom_ax)

        mode = self._current_mode_key()
        if mode == "points":
            self._draw_points_on_axes(self.zoom_ax, marker_size=self.point_marker_size_zoom)
        elif mode == "areas":
            self._draw_polygons_on_axes(self.zoom_ax, include_preview=True)
            self._draw_zoom_center_cross_on_axes(self.zoom_ax)
        else:
            self._draw_vertex_assignments_on_axes(self.zoom_ax, zoom_mode=True)

        self.zoom_fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        self.zoom_canvas.draw()

    # =========================================================
    # keyboard
    # =========================================================
    def keyPressEvent(self, event):
        mode = self._current_mode_key()

        if mode == "vertices":
            if self.vertex_auto_enabled and self.vertex_grid_locked:
                text = event.text().strip().upper()
                if text:
                    for shortcut, comp, _btn in self.vertex_buttons:
                        if shortcut.upper() == text:
                            self._assign_current_vertex(comp)
                            return

                if event.key() == Qt.Key_Left:
                    self.vertex_prev()
                    return
                if event.key() in (Qt.Key_Right, Qt.Key_Space):
                    self.vertex_skip()
                    return
                if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
                    self.vertex_clear_current()
                    return

            return super().keyPressEvent(event)

        return super().keyPressEvent(event)

    # =========================================================
    # finish
    # =========================================================
    def finish_and_save(self):
        if not self._has_any_image():
            QMessageBox.warning(self, "Image analysis", "Load at least one image first.")
            return

        self._save_session()
        mode = self._current_mode_key()

        if mode == "points":
            counts = self._aggregate_point_counts_all_images()
            total_points = int(sum(counts.values()))
            if total_points <= 0:
                QMessageBox.warning(self, "Image analysis", "No points counted across loaded images.")
                return
            raw = counts

        elif mode == "areas":
            perc = self._aggregate_area_percentages_all_images()
            if float(np.sum(list(perc.values()))) <= 0:
                QMessageBox.warning(
                    self,
                    "Image analysis",
                    "Computed areas are zero across loaded images.\n"
                    "Check polygons, background assignment, and image availability."
                )
                return
            raw = perc

        else:
            counts = self._aggregate_vertex_counts_all_images()
            total_vertices = int(sum(counts.values()))
            if total_vertices <= 0:
                QMessageBox.warning(
                    self,
                    "Vertex mode",
                    "No classified internal vertices across loaded images."
                )
                return
            raw = counts

        try:
            self.db.set_fraction_from_percentages(self.phi, raw)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        QMessageBox.information(
            self,
            "Saved",
            f"Saved image-based componentry for {self.phi:g} φ using {self._image_count()} image(s)."
        )
        self.accept()
#endregion

#region BLOQUE 8 Load components dialog
class LoadComponentsDialog(QDialog):
    def __init__(self, db: ComponentryDB, sample_name: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.sample_name = str(sample_name)
        self.setModal(True)
        self.setWindowTitle("Load fraction components")
        self.resize(780, 560)

        self._selected_phi = None

        root = QVBoxLayout(self)

        gb1 = QGroupBox("1) Choose fraction (phi bin)")
        f1 = QFormLayout(gb1)

        self.cmb_phi = QComboBox()
        dfp = self.db.gsd[self.db.gsd["wt"].astype(float) > 0].sort_values("phi")
        for _, r in dfp.iterrows():
            phi = float(r["phi"])
            wt = float(r["wt"])
            self.cmb_phi.addItem(f"{phi:g} φ (bin wt={wt:.2f}%)", userData=phi)

        self.lbl_bininfo = QLabel("")
        f1.addRow("Fraction:", self.cmb_phi)
        f1.addRow("Info:", self.lbl_bininfo)
        root.addWidget(gb1)

        gb2 = QGroupBox("2) Choose method")
        f2 = QFormLayout(gb2)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["Manual entry", "Image analysis (multi-image)"])
        self.cmb_method.setCurrentIndex(0)

        self.lbl_method_help = QLabel(
            "Image analysis supports multiple images for the same φ fraction.\n"
            "All loaded images within that fraction must use the same analysis mode."
        )
        self.lbl_method_help.setWordWrap(True)

        self.btn_start = QPushButton("Continue")
        self.btn_start.clicked.connect(self._on_continue)

        f2.addRow("Method:", self.cmb_method)
        f2.addRow("", self.lbl_method_help)
        f2.addRow("", self.btn_start)
        root.addWidget(gb2)

        self.gb3 = QGroupBox("3) Manual input")
        self.gb3.setEnabled(False)
        v3 = QVBoxLayout(self.gb3)

        self.lbl_title = QLabel("")
        v3.addWidget(self.lbl_title)

        self.form_manual = QFormLayout()
        self.edits = {}
        v3.addLayout(self.form_manual)

        self.btn_save = QPushButton("Save fraction componentry")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_manual)
        v3.addWidget(self.btn_save)

        root.addWidget(self.gb3)

        hb = QHBoxLayout()
        hb.addStretch(1)
        btn_cancel = QPushButton("Close")
        btn_cancel.clicked.connect(self.reject)
        hb.addWidget(btn_cancel)
        root.addLayout(hb)

        self.cmb_phi.currentIndexChanged.connect(self._update_bininfo)
        self._update_bininfo()

    def _update_bininfo(self):
        phi = float(self.cmb_phi.currentData())
        sub = self.db.gsd[self.db.gsd["phi"].astype(float) == phi]
        wt = float(sub["wt"].iloc[0]) if not sub.empty else np.nan
        self.lbl_bininfo.setText(f"Selected: {phi:g} φ | bin wt = {wt:.2f}%")
        self._selected_phi = phi

    def _on_continue(self):
        phi = float(self._selected_phi)
        method = self.cmb_method.currentText().strip().lower()

        if "manual" in method:
            self._enable_manual_section(phi)
            return

        dlg = ImageAnalysisDialog(self.db, self.sample_name, phi, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.accept()

    def _enable_manual_section(self, phi):
        sub = self.db.gsd[self.db.gsd["phi"].astype(float) == float(phi)]
        wt = float(sub["wt"].iloc[0]) if not sub.empty else np.nan

        while self.form_manual.rowCount() > 0:
            self.form_manual.removeRow(0)

        self.edits = {}
        for c in self.db.components:
            le = QLineEdit()
            le.setPlaceholderText("Enter any number (will be normalized)")
            self.edits[c] = le
            self.form_manual.addRow(f"{c}:", le)

        self.gb3.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.lbl_title.setText(f"Fraction {phi:g}φ | sample {self.sample_name} | bin wt={wt:.2f}%")

        if float(phi) in self.db.filled:
            for c in self.db.components:
                try:
                    v = float(self.db.frac100.loc[float(phi), c])
                    self.edits[c].setText(f"{v:.4f}")
                except Exception:
                    pass

    def _save_manual(self):
        phi = float(self._selected_phi)
        raw = {c: self.edits[c].text().strip() for c in self.db.components}

        try:
            self.db.set_fraction_manual_raw(phi, raw)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        self.accept()
#endregion

#region BLOQUE 9 View DB dialog
class ViewDBDialog(QDialog):
    def __init__(self, db: ComponentryDB, sample_name: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.sample_name = sample_name
        self.setModal(True)
        self.setWindowTitle(f"Componentry database — {sample_name}")
        self.resize(1020, 620)

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        w_total = QWidget()
        l_total = QVBoxLayout(w_total)
        t_total = QTableWidget()
        l_total.addWidget(t_total)
        tabs.addTab(w_total, "Total per fraction (wt%)")

        w_vol = QWidget()
        l_vol = QVBoxLayout(w_vol)
        t_vol = QTableWidget()
        l_vol.addWidget(t_vol)
        tabs.addTab(w_vol, "Vol. 100%")

        w_type = QWidget()
        l_type = QVBoxLayout(w_type)
        t_type = QTableWidget()
        l_type.addWidget(t_type)
        tabs.addTab(w_type, "By component type (wt%)")

        phis = db.gsd[db.gsd["wt"].astype(float) > 0]["phi"].astype(float).values.tolist()
        self._fill_table(t_total, db.total, db.components, phis, fmt="{:.4f}")
        self._fill_table(t_vol, db.frac100, db.components, phis, fmt="{:.4f}")

        df_type = db.totals_by_type()
        self._fill_table(t_type, df_type, TYPE_OPTIONS, phis, fmt="{:.4f}")

        tabs.setCurrentIndex(0)

        hb = QHBoxLayout()
        hb.addStretch(1)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        hb.addWidget(btn)
        root.addLayout(hb)

    def _fill_table(self, table: QTableWidget, df: pd.DataFrame, cols, phis, fmt="{:.4f}"):
        header = ["phi"] + list(cols)
        table.setColumnCount(len(header))
        table.setHorizontalHeaderLabels(header)
        table.setRowCount(len(phis))

        for i, phi in enumerate(phis):
            it_phi = QTableWidgetItem(f"{phi:g}")
            it_phi.setFlags(it_phi.flags() & ~Qt.ItemIsEditable)
            table.setItem(i, 0, it_phi)

            for j, c in enumerate(cols, start=1):
                try:
                    v = float(df.loc[phi, c])
                except Exception:
                    v = 0.0
                it = QTableWidgetItem(fmt.format(v))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, j, it)

        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)

        try:
            table.resizeColumnsToContents()
        except Exception:
            pass
#endregion


#region BLOQUE 9b Plugin system
class PluginLoadError(Exception):
    pass


class PluginManager:
    """
    Manager for external Componentry plugins.

    Supported plugin formats:

    1) Single-file plugin:
       my_plugin.py

       The file can expose one of these entry points:
         - create_plugin(app) -> plugin_object
         - register_plugin(app) -> None

    2) Folder plugin:
       my_plugin_folder/
           plugin.json
           main.py
           other_files.py

       plugin.json example:
       {
         "name": "My Plugin",
         "version": "1.0",
         "api_version": 1,
         "entry_point": "main.py",
         "class_name": "Plugin"
       }

       The entry module can expose:
         - create_plugin(app) -> plugin_object
         - class Plugin:
               def __init__(self, app): ...
               def setup(self): ...
               def teardown(self): ...   # optional
         - register_plugin(app) -> None

    Plugin objects may add menus, toolbar buttons, dock widgets, dialogs,
    or modify the current project through the public methods exposed by
    ComponentryPreview.
    """

    def __init__(self, app):
        self.app = app
        self.plugin_dir = self._default_plugin_dir()
        self.loaded_plugins = {}
        self.loaded_modules = {}
        self.plugin_errors = {}
        self.ensure_plugin_dir()

    # =========================================================
    # Paths
    # =========================================================
    def _default_plugin_dir(self):
        base = Path.home() / ".componentry_analysis"
        return base / PLUGIN_FOLDER_NAME

    def ensure_plugin_dir(self):
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

    def open_plugin_folder(self):
        self.ensure_plugin_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.plugin_dir)))

    # =========================================================
    # Installation
    # =========================================================
    def install_plugin_dialog(self, parent=None):
        """
        Installs either:
          - a .py plugin file
          - a plugin folder selected by the user

        If a plugin with the same file/folder name already exists in the
        plugin folder, the user is asked whether to replace it.
        """
        parent = parent or self.app

        path, _ = QFileDialog.getOpenFileName(
            parent,
            "Install plugin file",
            "",
            "Python plugin (*.py);;All files (*)"
        )

        if path:
            return self.install_plugin_path(path)

        folder = QFileDialog.getExistingDirectory(
            parent,
            "Install plugin folder",
            ""
        )

        if folder:
            return self.install_plugin_path(folder)

        return None

    def install_plugin_path(self, source_path):
        """
        Installs a plugin into the persistent plugin directory.

        Corrected behavior:
          - If GdeInput.py already exists and the user installs GdeInput.py again,
            the program asks whether to replace it.
          - It no longer creates GdeInput_2.py, GdeInput_3.py, etc.
          - Same logic for plugin folders.
        """
        self.ensure_plugin_dir()

        src = Path(source_path)
        if not src.exists():
            raise PluginLoadError(f"Plugin path does not exist:\n{source_path}")

        if src.is_file():
            if src.suffix.lower() != ".py":
                raise PluginLoadError("Only .py files can be installed as single-file plugins.")

            dest = self.plugin_dir / src.name

            # If the user selected the already installed file itself, do nothing.
            try:
                if dest.exists() and src.resolve() == dest.resolve():
                    return dest
            except Exception:
                pass

            if dest.exists():
                ans = QMessageBox.question(
                    self.app,
                    "Reemplazar complemento",
                    (
                        f"Ya existe un complemento instalado con este nombre:\n\n"
                        f"{dest.name}\n\n"
                        "¿Querés reemplazarlo por el archivo seleccionado?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if ans != QMessageBox.Yes:
                    return None

                try:
                    dest.unlink()
                except Exception as e:
                    raise PluginLoadError(
                        f"No pude eliminar el complemento anterior:\n{dest}\n\n{e}"
                    )

            shutil.copy2(str(src), str(dest))
            return dest

        if src.is_dir():
            dest = self.plugin_dir / src.name

            # If the user selected the already installed folder itself, do nothing.
            try:
                if dest.exists() and src.resolve() == dest.resolve():
                    return dest
            except Exception:
                pass

            if dest.exists():
                ans = QMessageBox.question(
                    self.app,
                    "Reemplazar complemento",
                    (
                        f"Ya existe una carpeta de complemento con este nombre:\n\n"
                        f"{dest.name}\n\n"
                        "¿Querés reemplazarla por la carpeta seleccionada?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if ans != QMessageBox.Yes:
                    return None

                try:
                    shutil.rmtree(str(dest))
                except Exception as e:
                    raise PluginLoadError(
                        f"No pude eliminar la carpeta anterior:\n{dest}\n\n{e}"
                    )

            shutil.copytree(str(src), str(dest))
            return dest

        raise PluginLoadError("Unsupported plugin path.")

    def _non_overwriting_path(self, dest: Path):
        """
        Legacy helper. Kept for compatibility, but no longer used by
        install_plugin_path(), because plugin installation now replaces
        same-name plugins only after user confirmation.
        """
        if not dest.exists():
            return dest

        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent

        k = 2
        while True:
            candidate = parent / f"{stem}_{k}{suffix}"
            if not candidate.exists():
                return candidate
            k += 1

    # =========================================================
    # Discovery
    # =========================================================
    def discover_plugins(self):
        self.ensure_plugin_dir()

        plugins = []

        for p in sorted(self.plugin_dir.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file() and p.suffix.lower() == ".py":
                plugins.append(p)
            elif p.is_dir():
                manifest = p / PLUGIN_MANIFEST_NAME
                default_entry = p / PLUGIN_ENTRY_DEFAULT
                if manifest.exists() or default_entry.exists():
                    plugins.append(p)

        return plugins

    # =========================================================
    # Loading / unloading
    # =========================================================
    def reload_plugins(self):
        self.unload_plugins()
        return self.load_all_plugins()

    def load_all_plugins(self):
        self.ensure_plugin_dir()

        loaded = []
        errors = []

        for plugin_path in self.discover_plugins():
            try:
                plugin_id = self.load_plugin(plugin_path)
                loaded.append(plugin_id)
            except Exception as e:
                tb = traceback.format_exc()
                self.plugin_errors[str(plugin_path)] = tb
                errors.append((plugin_path, e, tb))

        return loaded, errors

    def unload_plugins(self):
        """
        Calls plugin.teardown() when available and asks the main window
        to remove all registered plugin UI.
        """
        for plugin_id, plugin_obj in list(self.loaded_plugins.items()):
            try:
                teardown = getattr(plugin_obj, "teardown", None)
                if callable(teardown):
                    teardown()
            except Exception:
                self.plugin_errors[plugin_id] = traceback.format_exc()

        if hasattr(self.app, "clear_plugin_ui"):
            self.app.clear_plugin_ui()

        self.loaded_plugins = {}
        self.loaded_modules = {}

    def load_plugin(self, plugin_path):
        plugin_path = Path(plugin_path)

        if plugin_path.is_file():
            return self._load_single_file_plugin(plugin_path)

        if plugin_path.is_dir():
            return self._load_folder_plugin(plugin_path)

        raise PluginLoadError(f"Invalid plugin path:\n{plugin_path}")

    def _load_single_file_plugin(self, path: Path):
        plugin_id = self._plugin_id_from_path(path)
        module = self._import_module_from_file(path, plugin_id)

        plugin_obj = self._create_or_register_plugin(
            module=module,
            plugin_id=plugin_id,
            manifest={
                "name": path.stem,
                "version": "",
                "api_version": PLUGIN_API_VERSION,
                "entry_point": path.name,
                "class_name": PLUGIN_CLASS_DEFAULT,
            }
        )

        self.loaded_modules[plugin_id] = module
        self.loaded_plugins[plugin_id] = plugin_obj
        return plugin_id

    def _load_folder_plugin(self, folder: Path):
        manifest = self._read_manifest(folder)

        api_version = int(manifest.get("api_version", PLUGIN_API_VERSION))
        if api_version != PLUGIN_API_VERSION:
            raise PluginLoadError(
                f"Plugin API version mismatch in {folder.name}.\n"
                f"Plugin API: {api_version}\n"
                f"Program API: {PLUGIN_API_VERSION}"
            )

        entry_name = str(manifest.get("entry_point", PLUGIN_ENTRY_DEFAULT)).strip() or PLUGIN_ENTRY_DEFAULT
        entry_path = folder / entry_name

        if not entry_path.exists():
            raise PluginLoadError(f"Plugin entry point not found:\n{entry_path}")

        if entry_path.suffix.lower() != ".py":
            raise PluginLoadError("Plugin entry point must be a .py file.")

        plugin_id = self._plugin_id_from_path(folder)
        module = self._import_module_from_file(entry_path, plugin_id, package_root=folder)

        plugin_obj = self._create_or_register_plugin(
            module=module,
            plugin_id=plugin_id,
            manifest=manifest
        )

        self.loaded_modules[plugin_id] = module
        self.loaded_plugins[plugin_id] = plugin_obj
        return plugin_id

    # =========================================================
    # Manifest / module import
    # =========================================================
    def _read_manifest(self, folder: Path):
        manifest_path = folder / PLUGIN_MANIFEST_NAME

        if not manifest_path.exists():
            return {
                "name": folder.name,
                "version": "",
                "api_version": PLUGIN_API_VERSION,
                "entry_point": PLUGIN_ENTRY_DEFAULT,
                "class_name": PLUGIN_CLASS_DEFAULT,
            }

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise PluginLoadError(f"Could not read plugin manifest:\n{manifest_path}\n\n{e}")

        if not isinstance(data, dict):
            raise PluginLoadError(f"Invalid plugin manifest:\n{manifest_path}")

        if not str(data.get("name", "")).strip():
            data["name"] = folder.name

        if "api_version" not in data:
            data["api_version"] = PLUGIN_API_VERSION

        if "entry_point" not in data:
            data["entry_point"] = PLUGIN_ENTRY_DEFAULT

        if "class_name" not in data:
            data["class_name"] = PLUGIN_CLASS_DEFAULT

        return data

    def _plugin_id_from_path(self, path: Path):
        raw = str(path.resolve())
        safe = "".join(ch if ch.isalnum() else "_" for ch in raw)
        return f"componentry_plugin_{safe}"

    def _import_module_from_file(self, file_path: Path, plugin_id: str, package_root: Path = None):
        file_path = Path(file_path)

        if package_root is not None:
            package_root = Path(package_root)
            package_root_str = str(package_root.resolve())
            if package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)

        module_name = f"{plugin_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Could not create import spec for plugin:\n{file_path}")

        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise PluginLoadError(
                f"Plugin import failed:\n{file_path}\n\n{e}"
            )

        return module

    # =========================================================
    # Plugin entry points
    # =========================================================
    def _create_or_register_plugin(self, module, plugin_id: str, manifest: dict):
        plugin_name = str(manifest.get("name", plugin_id)).strip() or plugin_id

        # Preferred entry point for complex plugins.
        create_plugin = getattr(module, "create_plugin", None)
        if callable(create_plugin):
            plugin_obj = create_plugin(self.app)
            if plugin_obj is None:
                raise PluginLoadError(f"create_plugin(app) returned None in plugin '{plugin_name}'.")

            self._attach_plugin_metadata(plugin_obj, plugin_id, manifest)
            self._setup_plugin_object(plugin_obj)
            return plugin_obj

        # Manifest-defined class.
        class_name = str(manifest.get("class_name", PLUGIN_CLASS_DEFAULT)).strip() or PLUGIN_CLASS_DEFAULT
        plugin_class = getattr(module, class_name, None)

        if plugin_class is not None:
            try:
                plugin_obj = plugin_class(self.app)
            except TypeError:
                plugin_obj = plugin_class()

            self._attach_plugin_metadata(plugin_obj, plugin_id, manifest)
            self._setup_plugin_object(plugin_obj)
            return plugin_obj

        # Minimal legacy entry point.
        register_plugin = getattr(module, "register_plugin", None)
        if callable(register_plugin):
            register_plugin(self.app)

            wrapper = LoadedFunctionPlugin(
                app=self.app,
                plugin_id=plugin_id,
                name=plugin_name,
                manifest=manifest
            )
            return wrapper

        raise PluginLoadError(
            f"Plugin '{plugin_name}' has no valid entry point.\n\n"
            "Expected one of:\n"
            "  - create_plugin(app)\n"
            f"  - class {class_name}\n"
            "  - register_plugin(app)"
        )

    def _attach_plugin_metadata(self, plugin_obj, plugin_id: str, manifest: dict):
        try:
            plugin_obj.plugin_id = plugin_id
        except Exception:
            pass

        try:
            plugin_obj.plugin_manifest = dict(manifest)
        except Exception:
            pass

        if not hasattr(plugin_obj, "name"):
            try:
                plugin_obj.name = str(manifest.get("name", plugin_id))
            except Exception:
                pass

        if not hasattr(plugin_obj, "version"):
            try:
                plugin_obj.version = str(manifest.get("version", ""))
            except Exception:
                pass

    def _setup_plugin_object(self, plugin_obj):
        setup = getattr(plugin_obj, "setup", None)
        if callable(setup):
            setup()


class LoadedFunctionPlugin:
    """
    Wrapper for legacy plugins using register_plugin(app).
    This gives the PluginManager a stable object to store.
    """
    def __init__(self, app, plugin_id, name, manifest):
        self.app = app
        self.plugin_id = plugin_id
        self.name = name
        self.plugin_manifest = dict(manifest)

    def teardown(self):
        pass
#endregion

#region BLOQUE 10 Main window


class ComponentryHistogramStyleDialog(QDialog):
    """Independent style editor for the Componentry histogram."""

    def __init__(self, style_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit componentry graph")
        self.resize(940, 700)
        self.style = normalize_componentry_hist_style(style_dict)
        self._color_buttons = {}

        outer = QVBoxLayout(self)
        tabs = QTabWidget(self)
        outer.addWidget(tabs, 1)

        tabs.addTab(self._build_bars_tab(), "Bars and figure")
        tabs.addTab(self._build_axes_tab(), "Axes and curves")
        tabs.addTab(self._build_text_fw_tab(), "Text, legend and Folk & Ward")

        row = QHBoxLayout()
        btn_defaults = QPushButton("Reset defaults")
        btn_defaults.clicked.connect(self._reset_defaults)
        row.addWidget(btn_defaults)
        row.addStretch(1)
        outer.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    def _scroll_form(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(content)
        return scroll, layout

    def _color_button(self, key):
        btn = QPushButton()
        btn.setMinimumWidth(120)
        self._color_buttons[key] = btn
        self._set_color_button(key)
        btn.clicked.connect(lambda _=False, k=key: self._pick_color(k))
        return btn

    def _set_color_button(self, key):
        btn = self._color_buttons.get(key)
        if btn is None:
            return
        color = _normalize_hex_color(self.style.get(key), COMPONENTRY_HIST_DEFAULT_STYLE.get(key, "#808080"))
        btn.setText(color.upper())
        btn.setIcon(_color_icon(color, 44, 18))
        btn.setStyleSheet("text-align:left; padding:4px;")

    def _pick_color(self, key):
        current = QColor(_normalize_hex_color(self.style.get(key)))
        color = QColorDialog.getColor(current, self, "Choose color")
        if color.isValid():
            self.style[key] = color.name()
            self._set_color_button(key)

    @staticmethod
    def _spin(value, minimum, maximum, step=0.1, decimals=2):
        sp = QDoubleSpinBox()
        sp.setRange(float(minimum), float(maximum))
        sp.setDecimals(int(decimals))
        sp.setSingleStep(float(step))
        sp.setValue(float(value))
        return sp

    @staticmethod
    def _enum_combo(mapping, current):
        cmb = QComboBox()
        for label, data in mapping.items():
            cmb.addItem(label, data)
        idx = cmb.findData(current)
        if idx >= 0:
            cmb.setCurrentIndex(idx)
        return cmb

    def _build_bars_tab(self):
        scroll, lay = self._scroll_form()

        gb = QGroupBox("Bars")
        form = QFormLayout(gb)
        self.sp_bar_width = self._spin(self.style["bar_width_factor"], 0.05, 1.50, 0.05)
        form.addRow("Bar width factor", self.sp_bar_width)
        self.sp_bar_alpha = self._spin(self.style["bar_alpha"], 0.0, 1.0, 0.05)
        form.addRow("Bar transparency", self.sp_bar_alpha)
        form.addRow("Unclassified bar color", self._color_button("unclassified_facecolor"))
        form.addRow("Undefined color", self._color_button("undefined_facecolor"))
        form.addRow("Outer border color", self._color_button("bar_outer_edgecolor"))
        self.sp_outer_lw = self._spin(self.style["bar_outer_edgewidth"], 0.0, 10.0, 0.1)
        form.addRow("Outer border width", self.sp_outer_lw)
        form.addRow("Internal segment border color", self._color_button("segment_edgecolor"))
        self.sp_segment_lw = self._spin(self.style["segment_edgewidth"], 0.0, 10.0, 0.1)
        form.addRow("Internal segment border width", self.sp_segment_lw)
        lay.addWidget(gb)

        gb_fig = QGroupBox("Figure")
        form_fig = QFormLayout(gb_fig)
        self.sp_fig_w = self._spin(self.style["figure_width"], 4.0, 40.0, 0.25)
        self.sp_fig_h = self._spin(self.style["figure_height"], 3.0, 40.0, 0.25)
        form_fig.addRow("Export width (inches)", self.sp_fig_w)
        form_fig.addRow("Export height (inches)", self.sp_fig_h)
        lay.addWidget(gb_fig)
        lay.addStretch(1)
        return scroll

    def _build_axes_tab(self):
        scroll, lay = self._scroll_form()

        gb_axes = QGroupBox("Axes and ticks")
        form = QFormLayout(gb_axes)
        form.addRow("Axes color", self._color_button("axes_color"))
        self.sp_axes_lw = self._spin(self.style["axes_linewidth"], 0.1, 10.0, 0.1)
        form.addRow("Axes line width", self.sp_axes_lw)
        form.addRow("Tick color", self._color_button("tick_color"))
        self.sp_tick_w = self._spin(self.style["tick_width"], 0.1, 10.0, 0.1)
        self.sp_tick_len = self._spin(self.style["tick_length"], 0.0, 30.0, 0.5)
        form.addRow("Tick line width", self.sp_tick_w)
        form.addRow("Tick length", self.sp_tick_len)
        self.cmb_x_mode = self._enum_combo({"Phi (φ)": "phi", "Micrometers (μm)": "um"}, self.style["x_axis_mode"])
        form.addRow("X axis mode", self.cmb_x_mode)
        self.cmb_y_right = self._enum_combo({"Cumulative frequency": "cumulative", "None": "none"}, self.style["y_right_mode"])
        form.addRow("Right Y axis", self.cmb_y_right)
        self.chk_um = QCheckBox("Show secondary μm axis in phi mode")
        self.chk_um.setChecked(bool(self.style["show_um_axis"]))
        form.addRow(self.chk_um)
        self.cmb_um_pos = self._enum_combo({"Top": "top", "Bottom": "bottom"}, self.style["um_axis_position"])
        form.addRow("Secondary μm axis position", self.cmb_um_pos)
        self.chk_reverse = QCheckBox("Reverse X axis")
        self.chk_reverse.setChecked(bool(self.style["reverse_x"]))
        form.addRow(self.chk_reverse)
        lay.addWidget(gb_axes)

        gb_cum = QGroupBox("Cumulative curve")
        form_cum = QFormLayout(gb_cum)
        self.chk_cum = QCheckBox("Show cumulative curve")
        self.chk_cum.setChecked(bool(self.style["show_cumulative"]))
        form_cum.addRow(self.chk_cum)
        form_cum.addRow("Line color", self._color_button("cum_color"))
        self.sp_cum_lw = self._spin(self.style["cum_linewidth"], 0.0, 10.0, 0.1)
        form_cum.addRow("Line width", self.sp_cum_lw)
        self.cmb_cum_ls = self._enum_combo(LINESTYLE_OPTIONS, self.style["cum_linestyle"])
        form_cum.addRow("Line style", self.cmb_cum_ls)
        self.cmb_cum_marker = self._enum_combo(MARKER_OPTIONS, self.style["cum_marker"])
        form_cum.addRow("Marker", self.cmb_cum_marker)
        self.sp_cum_ms = self._spin(self.style["cum_markersize"], 0.0, 30.0, 0.5)
        form_cum.addRow("Marker size", self.sp_cum_ms)
        form_cum.addRow("Marker fill color", self._color_button("cum_marker_facecolor"))
        form_cum.addRow("Marker edge color", self._color_button("cum_marker_edgecolor"))
        self.sp_cum_mew = self._spin(self.style["cum_marker_edgewidth"], 0.0, 10.0, 0.1)
        form_cum.addRow("Marker edge width", self.sp_cum_mew)
        self.cmb_cum_layer = self._enum_combo({"Front": "front", "Back": "back"}, self.style["cumulative_layer"])
        form_cum.addRow("Layer", self.cmb_cum_layer)
        lay.addWidget(gb_cum)

        gb_poly = QGroupBox("Frequency polygon")
        form_poly = QFormLayout(gb_poly)
        self.chk_poly = QCheckBox("Show frequency polygon")
        self.chk_poly.setChecked(bool(self.style["show_polygon"]))
        form_poly.addRow(self.chk_poly)
        form_poly.addRow("Line color", self._color_button("poly_color"))
        self.sp_poly_lw = self._spin(self.style["poly_linewidth"], 0.0, 10.0, 0.1)
        form_poly.addRow("Line width", self.sp_poly_lw)
        self.cmb_poly_ls = self._enum_combo(LINESTYLE_OPTIONS, self.style["poly_linestyle"])
        form_poly.addRow("Line style", self.cmb_poly_ls)
        self.cmb_poly_marker = self._enum_combo(MARKER_OPTIONS, self.style["poly_marker"])
        form_poly.addRow("Marker", self.cmb_poly_marker)
        self.sp_poly_ms = self._spin(self.style["poly_markersize"], 0.0, 30.0, 0.5)
        form_poly.addRow("Marker size", self.sp_poly_ms)
        lay.addWidget(gb_poly)

        gb_grid = QGroupBox("Grid")
        form_grid = QFormLayout(gb_grid)
        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(bool(self.style["grid_visible"]))
        form_grid.addRow(self.chk_grid)
        form_grid.addRow("Grid color", self._color_button("grid_color"))
        self.sp_grid_lw = self._spin(self.style["grid_linewidth"], 0.0, 10.0, 0.1)
        form_grid.addRow("Grid line width", self.sp_grid_lw)
        self.cmb_grid_ls = self._enum_combo(LINESTYLE_OPTIONS, self.style["grid_linestyle"])
        form_grid.addRow("Grid line style", self.cmb_grid_ls)
        self.sp_grid_alpha = self._spin(self.style["grid_alpha"], 0.0, 1.0, 0.05)
        form_grid.addRow("Grid transparency", self.sp_grid_alpha)
        lay.addWidget(gb_grid)
        lay.addStretch(1)
        return scroll

    def _build_text_fw_tab(self):
        scroll, lay = self._scroll_form()

        gb_fonts = QGroupBox("Fonts")
        form = QFormLayout(gb_fonts)
        self.sp_title_fs = self._spin(self.style["title_fontsize"], 6.0, 60.0, 1.0, 1)
        self.sp_label_fs = self._spin(self.style["axis_label_fontsize"], 6.0, 50.0, 1.0, 1)
        self.sp_right_fs = self._spin(self.style["right_axis_label_fontsize"], 6.0, 50.0, 1.0, 1)
        self.sp_tick_fs = self._spin(self.style["tick_fontsize"], 5.0, 40.0, 1.0, 1)
        self.sp_legend_fs = self._spin(self.style["legend_fontsize"], 5.0, 40.0, 1.0, 1)
        form.addRow("Title size", self.sp_title_fs)
        form.addRow("Axis-label size", self.sp_label_fs)
        form.addRow("Right-axis label size", self.sp_right_fs)
        form.addRow("Tick-label size", self.sp_tick_fs)
        form.addRow("Legend size", self.sp_legend_fs)
        self.cmb_title_weight = self._enum_combo({"Normal": "normal", "Bold": "bold"}, self.style["title_fontweight"])
        self.cmb_label_weight = self._enum_combo({"Normal": "normal", "Bold": "bold"}, self.style["axis_label_fontweight"])
        form.addRow("Title weight", self.cmb_title_weight)
        form.addRow("Axis-label weight", self.cmb_label_weight)
        lay.addWidget(gb_fonts)

        gb_labels = QGroupBox("Custom labels (leave empty for defaults)")
        form_labels = QFormLayout(gb_labels)
        self.ed_x_phi = QLineEdit(self.style["x_label_phi"])
        self.ed_x_phi.setPlaceholderText("PHI (Φ)")
        self.ed_x_um = QLineEdit(self.style["x_label_um"])
        self.ed_x_um.setPlaceholderText("Particle size (μm)")
        self.ed_y_left = QLineEdit(self.style["y_label_left"])
        self.ed_y_left.setPlaceholderText("%Wt / Vol. %")
        self.ed_y_right = QLineEdit(self.style["y_label_right"])
        self.ed_y_right.setPlaceholderText("%Wt cumulative frequency")
        self.ed_um_secondary = QLineEdit(self.style["um_secondary_label"])
        self.ed_um_secondary.setPlaceholderText("Particle size (μm)")
        form_labels.addRow("X label in phi mode", self.ed_x_phi)
        form_labels.addRow("X label in μm mode", self.ed_x_um)
        form_labels.addRow("Left Y label", self.ed_y_left)
        form_labels.addRow("Right Y label", self.ed_y_right)
        form_labels.addRow("Secondary μm label", self.ed_um_secondary)
        lay.addWidget(gb_labels)

        gb_leg = QGroupBox("Legend")
        form_leg = QFormLayout(gb_leg)
        self.chk_legend_frame = QCheckBox("Show legend frame")
        self.chk_legend_frame.setChecked(bool(self.style["legend_frame"]))
        form_leg.addRow(self.chk_legend_frame)
        form_leg.addRow("Frame color", self._color_button("legend_framecolor"))
        self.sp_legend_lw = self._spin(self.style["legend_framewidth"], 0.0, 10.0, 0.1)
        self.sp_legend_alpha = self._spin(self.style["legend_alpha"], 0.0, 1.0, 0.05)
        form_leg.addRow("Frame width", self.sp_legend_lw)
        form_leg.addRow("Frame transparency", self.sp_legend_alpha)
        lay.addWidget(gb_leg)

        gb_fw = QGroupBox("Folk & Ward markers and annotation")
        form_fw = QFormLayout(gb_fw)
        form_fw.addRow("Median line color", self._color_button("fw_median_color"))
        self.sp_fw_median_lw = self._spin(self.style["fw_median_linewidth"], 0.0, 10.0, 0.1)
        form_fw.addRow("Median line width", self.sp_fw_median_lw)
        form_fw.addRow("φ16 / φ84 line color", self._color_button("fw_outer_color"))
        self.sp_fw_outer_lw = self._spin(self.style["fw_outer_linewidth"], 0.0, 10.0, 0.1)
        form_fw.addRow("φ16 / φ84 line width", self.sp_fw_outer_lw)
        self.sp_fw_line_alpha = self._spin(self.style["fw_line_alpha"], 0.0, 1.0, 0.05)
        form_fw.addRow("Marker-line transparency", self.sp_fw_line_alpha)
        form_fw.addRow("φ16–φ84 band color", self._color_button("fw_band_color"))
        self.sp_fw_band_alpha = self._spin(self.style["fw_band_alpha"], 0.0, 1.0, 0.01)
        form_fw.addRow("Band transparency", self.sp_fw_band_alpha)
        self.sp_fw_text_fs = self._spin(self.style["fw_text_fontsize"], 5.0, 40.0, 0.5, 1)
        form_fw.addRow("Annotation text size", self.sp_fw_text_fs)
        form_fw.addRow("Annotation box fill", self._color_button("fw_box_facecolor"))
        form_fw.addRow("Annotation box border", self._color_button("fw_box_edgecolor"))
        self.sp_fw_box_lw = self._spin(self.style["fw_box_linewidth"], 0.0, 10.0, 0.1)
        self.sp_fw_box_alpha = self._spin(self.style["fw_box_alpha"], 0.0, 1.0, 0.05)
        form_fw.addRow("Annotation box border width", self.sp_fw_box_lw)
        form_fw.addRow("Annotation box transparency", self.sp_fw_box_alpha)
        lay.addWidget(gb_fw)
        lay.addStretch(1)
        return scroll

    def _collect_widget_values(self):
        self.style.update({
            "bar_width_factor": self.sp_bar_width.value(),
            "bar_alpha": self.sp_bar_alpha.value(),
            "bar_outer_edgewidth": self.sp_outer_lw.value(),
            "segment_edgewidth": self.sp_segment_lw.value(),
            "figure_width": self.sp_fig_w.value(),
            "figure_height": self.sp_fig_h.value(),
            "axes_linewidth": self.sp_axes_lw.value(),
            "tick_width": self.sp_tick_w.value(),
            "tick_length": self.sp_tick_len.value(),
            "x_axis_mode": self.cmb_x_mode.currentData(),
            "y_right_mode": self.cmb_y_right.currentData(),
            "show_um_axis": self.chk_um.isChecked(),
            "um_axis_position": self.cmb_um_pos.currentData(),
            "reverse_x": self.chk_reverse.isChecked(),
            "show_cumulative": self.chk_cum.isChecked(),
            "cum_linewidth": self.sp_cum_lw.value(),
            "cum_linestyle": self.cmb_cum_ls.currentData(),
            "cum_marker": self.cmb_cum_marker.currentData(),
            "cum_markersize": self.sp_cum_ms.value(),
            "cum_marker_edgewidth": self.sp_cum_mew.value(),
            "cumulative_layer": self.cmb_cum_layer.currentData(),
            "show_polygon": self.chk_poly.isChecked(),
            "poly_linewidth": self.sp_poly_lw.value(),
            "poly_linestyle": self.cmb_poly_ls.currentData(),
            "poly_marker": self.cmb_poly_marker.currentData(),
            "poly_markersize": self.sp_poly_ms.value(),
            "grid_visible": self.chk_grid.isChecked(),
            "grid_linewidth": self.sp_grid_lw.value(),
            "grid_linestyle": self.cmb_grid_ls.currentData(),
            "grid_alpha": self.sp_grid_alpha.value(),
            "title_fontsize": self.sp_title_fs.value(),
            "axis_label_fontsize": self.sp_label_fs.value(),
            "right_axis_label_fontsize": self.sp_right_fs.value(),
            "tick_fontsize": self.sp_tick_fs.value(),
            "legend_fontsize": self.sp_legend_fs.value(),
            "title_fontweight": self.cmb_title_weight.currentData(),
            "axis_label_fontweight": self.cmb_label_weight.currentData(),
            "x_label_phi": self.ed_x_phi.text().strip(),
            "x_label_um": self.ed_x_um.text().strip(),
            "y_label_left": self.ed_y_left.text().strip(),
            "y_label_right": self.ed_y_right.text().strip(),
            "um_secondary_label": self.ed_um_secondary.text().strip(),
            "legend_frame": self.chk_legend_frame.isChecked(),
            "legend_framewidth": self.sp_legend_lw.value(),
            "legend_alpha": self.sp_legend_alpha.value(),
            "fw_median_linewidth": self.sp_fw_median_lw.value(),
            "fw_outer_linewidth": self.sp_fw_outer_lw.value(),
            "fw_line_alpha": self.sp_fw_line_alpha.value(),
            "fw_band_alpha": self.sp_fw_band_alpha.value(),
            "fw_text_fontsize": self.sp_fw_text_fs.value(),
            "fw_box_linewidth": self.sp_fw_box_lw.value(),
            "fw_box_alpha": self.sp_fw_box_alpha.value(),
        })
        self.style = normalize_componentry_hist_style(self.style)

    def _reset_defaults(self):
        self.style = normalize_componentry_hist_style(COMPONENTRY_HIST_DEFAULT_STYLE)

        for key in self._color_buttons:
            self._set_color_button(key)

        self.sp_bar_width.setValue(self.style["bar_width_factor"])
        self.sp_bar_alpha.setValue(self.style["bar_alpha"])
        self.sp_outer_lw.setValue(self.style["bar_outer_edgewidth"])
        self.sp_segment_lw.setValue(self.style["segment_edgewidth"])
        self.sp_fig_w.setValue(self.style["figure_width"])
        self.sp_fig_h.setValue(self.style["figure_height"])
        self.sp_axes_lw.setValue(self.style["axes_linewidth"])
        self.sp_tick_w.setValue(self.style["tick_width"])
        self.sp_tick_len.setValue(self.style["tick_length"])
        self.sp_cum_lw.setValue(self.style["cum_linewidth"])
        self.sp_cum_ms.setValue(self.style["cum_markersize"])
        self.sp_cum_mew.setValue(self.style["cum_marker_edgewidth"])
        self.sp_poly_lw.setValue(self.style["poly_linewidth"])
        self.sp_poly_ms.setValue(self.style["poly_markersize"])
        self.sp_grid_lw.setValue(self.style["grid_linewidth"])
        self.sp_grid_alpha.setValue(self.style["grid_alpha"])
        self.sp_title_fs.setValue(self.style["title_fontsize"])
        self.sp_label_fs.setValue(self.style["axis_label_fontsize"])
        self.sp_right_fs.setValue(self.style["right_axis_label_fontsize"])
        self.sp_tick_fs.setValue(self.style["tick_fontsize"])
        self.sp_legend_fs.setValue(self.style["legend_fontsize"])
        self.sp_legend_lw.setValue(self.style["legend_framewidth"])
        self.sp_legend_alpha.setValue(self.style["legend_alpha"])
        self.sp_fw_median_lw.setValue(self.style["fw_median_linewidth"])
        self.sp_fw_outer_lw.setValue(self.style["fw_outer_linewidth"])
        self.sp_fw_line_alpha.setValue(self.style["fw_line_alpha"])
        self.sp_fw_band_alpha.setValue(self.style["fw_band_alpha"])
        self.sp_fw_text_fs.setValue(self.style["fw_text_fontsize"])
        self.sp_fw_box_lw.setValue(self.style["fw_box_linewidth"])
        self.sp_fw_box_alpha.setValue(self.style["fw_box_alpha"])

        self.chk_um.setChecked(self.style["show_um_axis"])
        self.chk_reverse.setChecked(self.style["reverse_x"])
        self.chk_cum.setChecked(self.style["show_cumulative"])
        self.chk_poly.setChecked(self.style["show_polygon"])
        self.chk_grid.setChecked(self.style["grid_visible"])
        self.chk_legend_frame.setChecked(self.style["legend_frame"])

        for combo, key in [
            (self.cmb_x_mode, "x_axis_mode"),
            (self.cmb_y_right, "y_right_mode"),
            (self.cmb_um_pos, "um_axis_position"),
            (self.cmb_cum_ls, "cum_linestyle"),
            (self.cmb_cum_marker, "cum_marker"),
            (self.cmb_cum_layer, "cumulative_layer"),
            (self.cmb_poly_ls, "poly_linestyle"),
            (self.cmb_poly_marker, "poly_marker"),
            (self.cmb_grid_ls, "grid_linestyle"),
            (self.cmb_title_weight, "title_fontweight"),
            (self.cmb_label_weight, "axis_label_fontweight"),
        ]:
            idx = combo.findData(self.style[key])
            if idx >= 0:
                combo.setCurrentIndex(idx)

        self.ed_x_phi.setText(self.style["x_label_phi"])
        self.ed_x_um.setText(self.style["x_label_um"])
        self.ed_y_left.setText(self.style["y_label_left"])
        self.ed_y_right.setText(self.style["y_label_right"])
        self.ed_um_secondary.setText(self.style["um_secondary_label"])

    def _accept(self):
        self._collect_widget_values()
        self.accept()


class ComponentryPreview(QMainWindow):
    def __init__(self, host_app=None, integrated_mode=False, source_context=None, embedded_mode=False):
        super().__init__()
        self.host_app = host_app
        self.integrated_mode = bool(integrated_mode)
        self.embedded_mode = bool(embedded_mode)
        self.source_context = source_context or {}
        if self.embedded_mode:
            # Keep QMainWindow semantics for menus/toolbars/docks used by plugins,
            # but make the instance a child widget suitable for a QTabWidget.
            self.setWindowFlags(Qt.Widget)
        self.setWindowTitle("Componentry-FW")
        self.resize(1240, 820)
        if self.embedded_mode:
            self.setMinimumSize(0, 0)
        else:
            self.setMinimumSize(980, 680)
        self.setFont(QFont("Arial", 10))

        # Estado proyecto
        self.project_path = None

        # sample_name -> ComponentryDB
        self.sample_dbs = {}
        # Start empty, like a normal program before an input file is loaded.

        # Componentes globales y estado visual independiente de Componentry.
        self.palette_state = {
            "category": DEFAULT_PALETTE_CATEGORY,
            "name": DEFAULT_PALETTE_NAME,
            "reversed": False,
            "auto_apply": False,
        }
        self.componentry_hist_style = normalize_componentry_hist_style()

        self.global_components = []
        self.global_colors = {}
        self.global_types = {}
        self._sync_global_from_samples()

        # Sistema de complementos
        self.plugin_manager = PluginManager(self)
        self.plugin_menu = None
        self.plugin_menu_separator = None
        self.plugin_toolbar = None
        self.plugin_ui_actions = []
        self.plugin_ui_toolbar_actions = []
        self.plugin_dock_widgets = []
        self.plugin_windows = []

        # Native Fractal Analysis module/window state.  This is intentionally not
        # part of the plugin manager: Fractal analysis is a first-class Componentry
        # tool and remains accessible even when no external plugins are installed.
        self._fractal_analysis_module = None
        self._fractal_analysis_windows = []

        # In embedded FW mode, the host tab can mirror the plugin controls in
        # its own top selector row.  The callback is attached by FW after this
        # object has finished constructing and loading installed plugins.
        self._embedded_plugin_ui_callback = None
        self._build_menu_bar()
        self._build_plugin_toolbar()

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        # User-resizable separator between the plot area and the component panel.
        # Non-opaque resizing avoids continuously redrawing/compressing the
        # Matplotlib canvas while the separator is being dragged.
        self.main_splitter = QSplitter(Qt.Horizontal, central)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(7)
        self.main_splitter.setOpaqueResize(False)
        root.addWidget(self.main_splitter, 1)

        # Rendering guard: palette/table updates can emit several signals in a
        # single event-loop pass.  Only one complete redraw is allowed at once.
        self._plot_in_progress = False
        self._plot_pending = False

        # =========================
        # LEFT
        # =========================
        left = QWidget()
        left.setMinimumWidth(620)
        left.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(7)

        # ---- Project row
        # In standalone Componentry this row is visible. In FW-integrated mode it is
        # hidden because full project open/save belongs to the FW main window.
        self.project_row_widget = QWidget()
        row_project = QHBoxLayout(self.project_row_widget)
        row_project.setContentsMargins(0, 0, 0, 0)
        row_project.setSpacing(6)

        self.btn_open_project = QPushButton("Open project")
        self.btn_open_project.clicked.connect(self.open_project_dialog)
        row_project.addWidget(self.btn_open_project)

        self.btn_save_project = QPushButton("Save project")
        self.btn_save_project.clicked.connect(self.save_project)
        row_project.addWidget(self.btn_save_project)

        self.btn_save_project_as = QPushButton("Save project as...")
        self.btn_save_project_as.clicked.connect(self.save_project_as)
        row_project.addWidget(self.btn_save_project_as)

        self.lbl_project = QLabel("Project: (unsaved)")
        row_project.addWidget(self.lbl_project, 1)
        left_v.addWidget(self.project_row_widget)

        # ---- Main controls rows 1-2.  A grid avoids forcing the whole
        # window to become wider than the screen when all controls are visible.
        controls_grid = QGridLayout()
        controls_grid.setContentsMargins(0, 0, 0, 0)
        controls_grid.setHorizontalSpacing(6)
        controls_grid.setVerticalSpacing(4)

        self.btn_prev_sample = QPushButton("◀")
        self.btn_prev_sample.setFixedWidth(28)
        self.btn_prev_sample.clicked.connect(self.prev_sample)
        controls_grid.addWidget(self.btn_prev_sample, 0, 0)

        self.lbl_sample = QLabel("Sample:")
        controls_grid.addWidget(self.lbl_sample, 0, 1)
        self.cmb_sample = QComboBox()
        self.cmb_sample.setMinimumWidth(150)
        self.cmb_sample.currentIndexChanged.connect(self.on_sample_changed)
        controls_grid.addWidget(self.cmb_sample, 0, 2)

        self.btn_next_sample = QPushButton("▶")
        self.btn_next_sample.setFixedWidth(28)
        self.btn_next_sample.clicked.connect(self.next_sample)
        controls_grid.addWidget(self.btn_next_sample, 0, 3)

        self.btn_load_table = QPushButton("Import component table")
        self.btn_load_table.clicked.connect(self.open_load_components_table)
        controls_grid.addWidget(self.btn_load_table, 0, 4)

        self.btn_export_excel = QPushButton("Export Excel")
        self.btn_export_excel.clicked.connect(self.open_export_excel)
        controls_grid.addWidget(self.btn_export_excel, 0, 5)

        self.btn_export_images = QPushButton("Export images")
        self.btn_export_images.clicked.connect(self.open_export_images)
        controls_grid.addWidget(self.btn_export_images, 0, 6)

        self.chk_100 = QCheckBox("Graph bars at 100%")
        self.chk_100.toggled.connect(self.plot_histogram)
        controls_grid.addWidget(self.chk_100, 1, 0, 1, 2)

        self.chk_type = QCheckBox("Graph Juvenile/Lithic/Accessory")
        self.chk_type.toggled.connect(self.plot_histogram)
        controls_grid.addWidget(self.chk_type, 1, 2, 1, 3)

        self.btn_view_db = QPushButton("View componentry database")
        self.btn_view_db.clicked.connect(self.open_view_db)
        controls_grid.addWidget(self.btn_view_db, 1, 5, 1, 2)

        controls_grid.setColumnStretch(2, 1)

        # ---- Main controls row 2 (FW + filter)
        row_controls2 = QHBoxLayout()

        self.chk_show_fw = QCheckBox("Show Folk & Ward")
        self.chk_show_fw.toggled.connect(self.plot_histogram)
        row_controls2.addWidget(self.chk_show_fw)

        self.chk_mark_fw = QCheckBox("Mark φ percentiles")
        self.chk_mark_fw.toggled.connect(self.plot_histogram)
        row_controls2.addWidget(self.chk_mark_fw)

        self.chk_hist_filter = QCheckBox("Histogram filter")
        self.chk_hist_filter.toggled.connect(self._on_hist_filter_toggled)
        row_controls2.addWidget(self.chk_hist_filter)

        self.cmb_hist_filter_mode = QComboBox()
        self.cmb_hist_filter_mode.addItems(["By component", "By type"])
        self.cmb_hist_filter_mode.currentIndexChanged.connect(self._on_hist_filter_mode_changed)
        row_controls2.addWidget(self.cmb_hist_filter_mode)

        self.cmb_hist_filter_value = QComboBox()
        self.cmb_hist_filter_value.currentIndexChanged.connect(self.plot_histogram)
        row_controls2.addWidget(self.cmb_hist_filter_value)

        self.chk_hist_include_undef = QCheckBox("Include Undefined")
        self.chk_hist_include_undef.toggled.connect(self.plot_histogram)
        row_controls2.addWidget(self.chk_hist_include_undef)

        self.btn_edit_componentry_graph = QPushButton("Edit componentry graph")
        self.btn_edit_componentry_graph.clicked.connect(self.open_componentry_histogram_style_dialog)
        row_controls2.addWidget(self.btn_edit_componentry_graph)

        self.btn_fractal_analysis = QPushButton("Fractal analysis")
        self.btn_fractal_analysis.setToolTip("Open number-based and mass-based fractal fragmentation analysis for the loaded Componentry samples.")
        self.btn_fractal_analysis.clicked.connect(self.open_fractal_analysis)
        row_controls2.addWidget(self.btn_fractal_analysis)

        row_controls2.addStretch(1)

        # In embedded FW mode the host already owns sample navigation. Repack
        # the remaining Componentry controls into two genuinely compact rows
        # instead of leaving the hidden navigation grid occupying a third row.
        if self.embedded_mode:
            for w in (self.btn_export_excel, self.btn_export_images, self.chk_100,
                      self.chk_type, self.btn_view_db):
                controls_grid.removeWidget(w)
            for w in (self.chk_show_fw, self.chk_mark_fw, self.chk_hist_filter,
                      self.cmb_hist_filter_mode, self.cmb_hist_filter_value,
                      self.chk_hist_include_undef, self.btn_edit_componentry_graph,
                      self.btn_fractal_analysis):
                row_controls2.removeWidget(w)

            compact_row1 = QHBoxLayout()
            compact_row1.setContentsMargins(0, 0, 0, 0)
            compact_row1.setSpacing(8)
            compact_row1.addWidget(self.chk_100)
            compact_row1.addWidget(self.chk_type)
            compact_row1.addWidget(self.chk_show_fw)
            compact_row1.addWidget(self.chk_mark_fw)
            compact_row1.addStretch(1)
            compact_row1.addWidget(self.btn_fractal_analysis)
            compact_row1.addWidget(self.btn_export_excel)
            compact_row1.addWidget(self.btn_export_images)
            compact_row1.addWidget(self.btn_view_db)
            left_v.addLayout(compact_row1)

            compact_row2 = QHBoxLayout()
            compact_row2.setContentsMargins(0, 0, 0, 0)
            compact_row2.setSpacing(8)
            compact_row2.addWidget(self.chk_hist_filter)
            compact_row2.addWidget(self.cmb_hist_filter_mode)
            compact_row2.addWidget(self.cmb_hist_filter_value)
            compact_row2.addWidget(self.chk_hist_include_undef)
            compact_row2.addWidget(self.btn_edit_componentry_graph)
            compact_row2.addStretch(1)
            left_v.addLayout(compact_row2)
        else:
            left_v.addLayout(controls_grid)
            left_v.addLayout(row_controls2)

        # ---- Compact sample summary overlay
        # This is deliberately NOT inserted into left_v. It floats over the
        # preview viewport and therefore consumes exactly zero vertical layout
        # space.
        self.summary_box = QWidget()
        self.summary_box.setObjectName("componentry_summary_overlay")
        self.summary_box.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.summary_box.setStyleSheet(
            "#componentry_summary_overlay { background: rgba(255,255,255,238); "
            "border: 1px solid #b9bec4; border-radius: 5px; }"
        )
        sg = QHBoxLayout(self.summary_box)
        sg.setContentsMargins(7, 3, 7, 3)
        sg.setSpacing(7)
        self.sum_labels = {}
        fields = [
            ("sample", "Sample"),
            ("fractions", "n"),
            ("sumwt", "Wt"),
            ("phirange", "φ"),
            ("meanphi", "Mφ"),
        ]
        for k, txt in fields:
            key = QLabel(txt + ":")
            key.setStyleSheet("color:#5a5a5a; font-size:9px;")
            val = QLabel("—")
            val.setStyleSheet("font-weight:600; color:#202020; font-size:9px;")
            val.setWordWrap(False)
            if k == "sample":
                val.setMaximumWidth(125)
            sg.addWidget(key)
            sg.addWidget(val)
            self.sum_labels[k] = val

        # ---- Plot / export-layout preview (WYSIWYG)
        # The Matplotlib figure keeps the same physical dimensions used by
        # export.  Only DPI changes between preview and exported raster files.
        # A scroll area lets the page retain its geometry instead of being
        # distorted by the Qt layout when the available window becomes small.
        _initial_style = normalize_componentry_hist_style(self.componentry_hist_style)
        self._componentry_preview_dpi = 90.0
        self.canvas = FigureCanvas(Figure(
            figsize=(_initial_style["figure_width"], _initial_style["figure_height"]),
            dpi=self._componentry_preview_dpi,
            facecolor="white",
        ))
        self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.canvas.setStyleSheet(
            "background: white; border: 1px solid #b8bec6;"
        )

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setWidgetResizable(False)
        self.plot_scroll.setAlignment(Qt.AlignCenter)
        self.plot_scroll.setFrameShape(QScrollArea.StyledPanel)
        self.plot_scroll.setStyleSheet(
            "QScrollArea { background: #edf0f2; border: 1px solid #c8cdd2; }"
            "QScrollArea > QWidget > QWidget { background: #edf0f2; }"
        )
        self.plot_scroll.setWidget(self.canvas)
        left_v.addWidget(self.plot_scroll, 1)

        # Overlay summary: top-right of the viewport, outside the layout flow.
        self.summary_box.setParent(self.plot_scroll.viewport())
        self.summary_box.show()
        self.summary_box.raise_()
        self.plot_scroll.viewport().installEventFilter(self)
        QTimer.singleShot(0, self._position_summary_overlay)

        self._sync_histogram_preview_geometry(_initial_style)

        self.main_splitter.addWidget(left)

        # =========================
        # RIGHT
        # =========================
        right = QWidget()
        right.setMinimumWidth(300)
        right.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(0)

        gb = QGroupBox("Components (current sample)")
        gb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panel = QVBoxLayout(gb)
        panel.setContentsMargins(8, 10, 8, 8)
        panel.setSpacing(6)

        palette_box = QGroupBox("Component color palette")
        palette_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        palette_grid = QGridLayout(palette_box)
        palette_grid.setContentsMargins(7, 9, 7, 7)
        palette_grid.setHorizontalSpacing(5)
        palette_grid.setVerticalSpacing(4)
        palette_grid.addWidget(QLabel("Family:"), 0, 0)
        self.cmb_palette_category = QComboBox()
        self.cmb_palette_category.setMinimumWidth(105)
        self.cmb_palette_category.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_palette_category.setMinimumContentsLength(10)
        palette_grid.addWidget(self.cmb_palette_category, 0, 1)
        self.chk_palette_reverse = QCheckBox("Reverse")
        palette_grid.addWidget(self.chk_palette_reverse, 0, 2)

        palette_grid.addWidget(QLabel("Palette:"), 1, 0)
        palette_nav = QHBoxLayout()
        palette_nav.setContentsMargins(0, 0, 0, 0)
        palette_nav.setSpacing(4)
        self.btn_palette_prev = QPushButton("<")
        self.btn_palette_prev.setFixedWidth(28)
        self.btn_palette_prev.setToolTip("Previous palette in this family")
        self.cmb_palette = QComboBox()
        self.cmb_palette.setMinimumWidth(105)
        self.cmb_palette.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_palette.setMinimumContentsLength(12)
        self.btn_palette_next = QPushButton(">")
        self.btn_palette_next.setFixedWidth(28)
        self.btn_palette_next.setToolTip("Next palette in this family")
        palette_nav.addWidget(self.btn_palette_prev)
        palette_nav.addWidget(self.cmb_palette, 1)
        palette_nav.addWidget(self.btn_palette_next)
        palette_grid.addLayout(palette_nav, 1, 1, 1, 2)

        self.lbl_palette_preview = QLabel()
        self.lbl_palette_preview.setFixedHeight(22)
        self.lbl_palette_preview.setScaledContents(True)
        palette_grid.addWidget(self.lbl_palette_preview, 2, 0, 1, 3)

        self.chk_palette_auto_apply = QCheckBox("Apply on change")
        self.chk_palette_auto_apply.setToolTip(
            "When enabled, changing Family, Palette, Reverse, or using < > immediately updates the graph."
        )
        self.btn_apply_palette = QPushButton("Apply palette")
        self.btn_restore_palette = QPushButton("Default")
        self.btn_apply_palette.setMinimumHeight(28)
        self.btn_restore_palette.setMinimumHeight(28)
        palette_grid.addWidget(self.chk_palette_auto_apply, 3, 0)
        palette_grid.addWidget(self.btn_apply_palette, 3, 1)
        palette_grid.addWidget(self.btn_restore_palette, 3, 2)
        palette_grid.setColumnStretch(1, 1)
        panel.addWidget(palette_box, 0)

        self.lbl_component_order_hint = QLabel("Order: top row = top of stack")
        self.lbl_component_order_hint.setStyleSheet("color:#5f6368; font-size:9pt;")
        self.lbl_component_order_hint.setToolTip(
            "The first component is drawn at the top of each stacked bar; the last component is drawn at the bottom."
        )
        panel.addWidget(self.lbl_component_order_hint)

        self.tbl = ReorderableComponentTable()
        self.tbl.reorder_callback = self._move_component_row
        self.tbl.setColumnCount(4)
        self.tbl.setHorizontalHeaderLabels(["Component", "Color", "Palette", "Type"])
        self.tbl.setRowCount(0)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tbl.setMinimumHeight(360)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setToolTip(
            "Drag the component name (first column) up or down to change its stacking order. Use ↑/↓ as an alternative."
        )
        self.tbl.cellClicked.connect(self._on_tbl_cell_clicked)
        self.tbl.itemChanged.connect(self._on_component_name_item_changed)
        try:
            header = self.tbl.horizontalHeader()
            header.setStretchLastSection(False)
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Fixed)
            header.setSectionResizeMode(2, QHeaderView.Fixed)
            header.setSectionResizeMode(3, QHeaderView.Fixed)
            header.resizeSection(1, 40)
            header.resizeSection(2, 92)
            header.resizeSection(3, 72)
            self.tbl.verticalHeader().setDefaultSectionSize(30)
        except Exception:
            pass
        panel.addWidget(self.tbl, 1)

        hb = QHBoxLayout()
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(6)
        self.btn_add = QPushButton("Add component")
        self.btn_remove = QPushButton("Remove component")
        self.btn_move_component_up = QPushButton("↑")
        self.btn_move_component_down = QPushButton("↓")
        self.btn_move_component_up.setFixedWidth(34)
        self.btn_move_component_down.setFixedWidth(34)
        self.btn_move_component_up.setToolTip("Move selected component up")
        self.btn_move_component_down.setToolTip("Move selected component down")
        self.btn_add.clicked.connect(self.add_component_row)
        self.btn_remove.clicked.connect(self.remove_component_row)
        self.btn_move_component_up.clicked.connect(lambda: self._move_selected_component(-1))
        self.btn_move_component_down.clicked.connect(lambda: self._move_selected_component(1))
        hb.addWidget(self.btn_add)
        hb.addWidget(self.btn_remove)
        hb.addStretch(1)
        hb.addWidget(self.btn_move_component_up)
        hb.addWidget(self.btn_move_component_down)
        panel.addLayout(hb)

        self.btn_apply = QPushButton("Apply other changes")
        self.btn_apply.setMinimumHeight(34)
        self.btn_apply.clicked.connect(self.apply_changes)
        panel.addWidget(self.btn_apply)

        self.cmb_palette_category.currentIndexChanged.connect(self._on_palette_category_changed)
        self.cmb_palette.currentIndexChanged.connect(self._on_palette_selection_changed)
        self.chk_palette_reverse.toggled.connect(self._on_palette_selection_changed)
        self.chk_palette_auto_apply.toggled.connect(self._on_palette_auto_apply_toggled)
        self.btn_palette_prev.clicked.connect(lambda: self._step_palette(-1))
        self.btn_palette_next.clicked.connect(lambda: self._step_palette(1))
        self.btn_apply_palette.clicked.connect(self.apply_selected_palette)
        self.btn_restore_palette.clicked.connect(self.restore_default_palette)

        self.btn_load_fraction = QPushButton("Load fraction components")
        self.btn_load_fraction.setEnabled(False)
        self.btn_load_fraction.setMinimumHeight(44)
        self.btn_load_fraction.setStyleSheet("font-weight: bold;")
        self.btn_load_fraction.clicked.connect(self.open_load_components)
        panel.addWidget(self.btn_load_fraction)

        right_v.addWidget(gb, 1)

        self.main_splitter.addWidget(right)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)

        # Start with a useful plot/panel proportion, but leave the separator
        # completely under user control afterwards.
        QTimer.singleShot(0, self._set_default_splitter_sizes)

        # init UI state
        self._populate_palette_controls()
        self._reload_sample_combo(select_first=True)
        self.refresh_component_table_from_global()
        self._refresh_palette_color_widgets()
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
        self.update_summary()
        self.plot_histogram()
        self._apply_integrated_mode_ui()

        # Cargar complementos instalados al final de la inicialización,
        # cuando la interfaz principal ya existe.
        self._load_plugins_on_startup()


    def _set_default_splitter_sizes(self):
        splitter = getattr(self, "main_splitter", None)
        if splitter is None:
            return
        total = int(splitter.width())
        if total <= 0:
            total = max(940, int(self.width()) - 16)
        right_width = min(430, max(340, int(round(total * 0.27))))
        left_width = max(620, total - right_width - splitter.handleWidth())
        splitter.setSizes([left_width, right_width])

    def get_view_state(self):
        return {
            "plot_style": dict(normalize_componentry_hist_style(self.componentry_hist_style)),
            "palette_state": dict(self.palette_state),
        }

    def set_view_state(self, state):
        state = state or {}
        self.componentry_hist_style = normalize_componentry_hist_style(state.get("plot_style", {}))

        pstate = state.get("palette_state", {}) or {}
        name = str(pstate.get("name", DEFAULT_PALETTE_NAME))
        if name not in COMPONENT_PALETTES:
            name = DEFAULT_PALETTE_NAME
        category = COMPONENT_PALETTES[name]["category"]
        self.palette_state = {
            "category": category,
            "name": name,
            "reversed": bool(pstate.get("reversed", False)),
            "auto_apply": bool(pstate.get("auto_apply", False)),
        }

        if hasattr(self, "cmb_palette_category"):
            self._populate_palette_controls()
            self._refresh_palette_color_widgets()
        if hasattr(self, "plot_histogram"):
            self.plot_histogram()

    def closeEvent(self, event):
        try:
            self._notify_host_componentry_changed()
        except Exception:
            pass
        try:
            for win in list(getattr(self, "_fractal_analysis_windows", [])):
                try:
                    win.close()
                except Exception:
                    pass
            self._fractal_analysis_windows = []
        except Exception:
            pass
        super().closeEvent(event)

    # =========================================================
    # FW integration helpers
    # =========================================================
    def _apply_integrated_mode_ui(self):
        if not getattr(self, "integrated_mode", False):
            return

        try:
            self.setWindowTitle("Componentry-FW")
        except Exception:
            pass

        # In integrated mode, granulometry and full-project persistence belong
        # to FW-Granulometry. Componentry remains only the component editor.
        for attr in ("btn_load_table",):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(False)
                w.setEnabled(False)

        # Hide Componentry-local project controls when Componentry is opened from FW.
        row = getattr(self, "project_row_widget", None)
        if row is not None:
            row.setVisible(False)
            row.setEnabled(False)
        for attr in ("btn_open_project", "btn_save_project", "btn_save_project_as", "lbl_project"):
            w = getattr(self, attr, None)
            if w is not None:
                w.setVisible(False)
                w.setEnabled(False)

        # In FW tab mode the host owns database/sample navigation.  Keep the
        # internal combo alive for all existing Componentry logic, but hide only
        # its duplicate navigation controls from the embedded UI.
        if getattr(self, "embedded_mode", False):
            for attr in ("btn_prev_sample", "lbl_sample", "cmb_sample", "btn_next_sample"):
                w = getattr(self, attr, None)
                if w is not None:
                    w.setVisible(False)
                    w.setEnabled(False)

            # FW owns the visible top row in embedded mode.  Keep the QMenu,
            # QToolBar and their QAction objects alive for plugin compatibility,
            # but hide Componentry's own menu/toolbar so they consume no vertical
            # space. FW mirrors them beside its Sample selector.
            try:
                self.menuBar().setVisible(False)
            except Exception:
                pass
            try:
                if self.plugin_toolbar is not None:
                    self.plugin_toolbar.setVisible(False)
            except Exception:
                pass

        self._set_project_label()

    def set_external_samples(self, sample_dbs: dict, current_sample: str = None, source_context: dict = None):
        """
        Replace the visible Componentry sample map with externally owned FW samples.
        The ComponentryDB objects are shared references, so edits in this window are
        immediately part of the FW project state.
        """
        self.sample_dbs = dict(sample_dbs or {})
        self.source_context = source_context or getattr(self, "source_context", {}) or {}
        # Samples are authoritative: only derive the global view here. Pushing
        # globals back would overwrite per-sample component types on every
        # refresh (same rule as when a project is opened).
        self._sync_global_from_samples()
        self._reload_sample_combo(select_first=True)
        self.refresh_component_table_from_global()
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
        self.update_summary()

        if current_sample:
            idx = self.cmb_sample.findText(str(current_sample))
            if idx >= 0:
                self.cmb_sample.setCurrentIndex(idx)

        base_label = str(self.source_context.get("base_label", "")).strip()
        sample_label = str(self.source_context.get("sample_name", "")).strip()
        if base_label and sample_label:
            self.setWindowTitle(f"Componentry-FW — {base_label} — {sample_label}")
        else:
            self.setWindowTitle("Componentry-FW")

        self._apply_integrated_mode_ui()
        self.plot_histogram()

    def _host_project_path(self):
        host = getattr(self, "host_app", None)
        if host is not None:
            return getattr(host, "project_path", None)
        return None

    # =========================================================
    # PLUGIN UI / EXTENSION API
    # =========================================================
    def _build_menu_bar(self):
        menubar = self.menuBar()

        self.plugin_menu = menubar.addMenu(PLUGIN_MENU_TITLE)

        act_install = QAction("Instalar complemento...", self)
        act_install.triggered.connect(self.install_plugin_dialog)
        self.plugin_menu.addAction(act_install)

        act_reload = QAction("Recargar complementos", self)
        act_reload.triggered.connect(self.reload_plugins_dialog)
        self.plugin_menu.addAction(act_reload)

        act_folder = QAction("Abrir carpeta de complementos", self)
        act_folder.triggered.connect(self.open_plugin_folder)
        self.plugin_menu.addAction(act_folder)

        act_errors = QAction("Ver errores de complementos", self)
        act_errors.triggered.connect(self.show_plugin_errors_dialog)
        self.plugin_menu.addAction(act_errors)

        self.plugin_menu_separator = self.plugin_menu.addSeparator()

    def _build_plugin_toolbar(self):
        self.plugin_toolbar = QToolBar(PLUGIN_TOOLBAR_TITLE, self)
        self.plugin_toolbar.setObjectName("plugin_toolbar")
        self.plugin_toolbar.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, self.plugin_toolbar)

    def set_embedded_plugin_ui_callback(self, callback):
        """Register the FW host callback used to mirror plugin controls."""
        self._embedded_plugin_ui_callback = callback if callable(callback) else None
        self._notify_embedded_plugin_ui_changed()

    def _notify_embedded_plugin_ui_changed(self):
        callback = getattr(self, "_embedded_plugin_ui_callback", None)
        if callable(callback):
            try:
                callback()
            except Exception:
                pass

    def _load_plugins_on_startup(self):
        try:
            _loaded, errors = self.plugin_manager.load_all_plugins()
            if errors:
                self.plugin_manager.plugin_errors["startup"] = "\n\n".join(
                    [f"{path}:\n{tb}" for path, _err, tb in errors]
                )
        except Exception:
            self.plugin_manager.plugin_errors["startup"] = traceback.format_exc()

    def install_plugin_dialog(self):
        try:
            installed_path = self.plugin_manager.install_plugin_dialog(parent=self)
            if installed_path is None:
                return

            loaded, errors = self.plugin_manager.reload_plugins()

            if errors:
                QMessageBox.warning(
                    self,
                    "Complementos",
                    f"Complemento instalado en:\n{installed_path}\n\n"
                    f"Se cargaron {len(loaded)} complemento(s), pero hubo {len(errors)} error(es).\n"
                    "Usá 'Ver errores de complementos' para revisar el detalle."
                )
            else:
                QMessageBox.information(
                    self,
                    "Complementos",
                    f"Complemento instalado y cargado correctamente:\n{installed_path}"
                )
        except Exception as e:
            QMessageBox.warning(self, "Error al instalar complemento", str(e))

    def reload_plugins_dialog(self):
        try:
            loaded, errors = self.plugin_manager.reload_plugins()

            if errors:
                QMessageBox.warning(
                    self,
                    "Complementos",
                    f"Se cargaron {len(loaded)} complemento(s), pero hubo {len(errors)} error(es).\n"
                    "Usá 'Ver errores de complementos' para revisar el detalle."
                )
            else:
                QMessageBox.information(
                    self,
                    "Complementos",
                    f"Complementos cargados correctamente: {len(loaded)}"
                )
        except Exception as e:
            QMessageBox.warning(self, "Error al recargar complementos", str(e))

    def open_plugin_folder(self):
        try:
            self.plugin_manager.open_plugin_folder()
        except Exception as e:
            QMessageBox.warning(self, "Complementos", str(e))

    def show_plugin_errors_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Errores de complementos")
        dlg.resize(900, 620)

        lay = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)

        if self.plugin_manager.plugin_errors:
            body = []
            for key, value in self.plugin_manager.plugin_errors.items():
                body.append(f"===== {key} =====\n{value}")
            txt.setPlainText("\n\n".join(body))
        else:
            txt.setPlainText("No hay errores registrados de complementos.")

        lay.addWidget(txt, 1)

        hb = QHBoxLayout()
        hb.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        hb.addWidget(btn_close)
        lay.addLayout(hb)

        dlg.exec_()

    def _execute_plugin_callback(self, callback):
        try:
            callback()
        except Exception:
            tb = traceback.format_exc()
            self.plugin_manager.plugin_errors["callback"] = tb
            QMessageBox.warning(
                self,
                "Error en complemento",
                "La función del complemento produjo un error.\n\n"
                "Usá 'Complementos > Ver errores de complementos' para revisar el detalle."
            )

    def add_plugin_menu_action(self, plugin_name, action_name, callback, tooltip=None):
        """
        API pública para complementos.
        Agrega una acción al menú superior 'Complementos'.
        callback debe ser callable y no recibir argumentos.
        """
        if not callable(callback):
            raise ValueError("callback must be callable.")

        plugin_name = str(plugin_name).strip() or "Plugin"
        action_name = str(action_name).strip() or "Plugin action"

        act = QAction(f"{plugin_name}: {action_name}", self)
        if tooltip:
            act.setToolTip(str(tooltip))
            act.setStatusTip(str(tooltip))
        act.triggered.connect(lambda _checked=False, cb=callback: self._execute_plugin_callback(cb))

        self.plugin_menu.addAction(act)
        self.plugin_ui_actions.append({
            "plugin_name": plugin_name,
            "action": act,
        })
        return act

    def add_plugin_toolbar_button(self, plugin_name, text, callback, tooltip=None):
        """
        API pública para complementos.
        Agrega un botón a la barra de herramientas de complementos.
        callback debe ser callable y no recibir argumentos.
        """
        if not callable(callback):
            raise ValueError("callback must be callable.")

        plugin_name = str(plugin_name).strip() or "Plugin"
        text = str(text).strip() or "Plugin"

        act = QAction(text, self)
        if tooltip:
            act.setToolTip(str(tooltip))
            act.setStatusTip(str(tooltip))
        act.triggered.connect(lambda _checked=False, cb=callback: self._execute_plugin_callback(cb))

        self.plugin_toolbar.addAction(act)
        self.plugin_ui_toolbar_actions.append({
            "plugin_name": plugin_name,
            "action": act,
        })
        self._notify_embedded_plugin_ui_changed()
        return act

    def add_plugin_dock_widget(self, plugin_name, title, widget, area="right"):
        """
        API pública para complementos.
        Agrega un QDockWidget con un widget provisto por el complemento.
        area: 'left', 'right', 'top' o 'bottom'.
        """
        plugin_name = str(plugin_name).strip() or "Plugin"
        title = str(title).strip() or plugin_name

        dock = QDockWidget(title, self)
        dock.setObjectName(f"plugin_dock_{plugin_name}_{title}".replace(" ", "_"))
        dock.setWidget(widget)

        area_key = str(area).strip().lower()
        area_map = {
            "left": Qt.LeftDockWidgetArea,
            "right": Qt.RightDockWidgetArea,
            "top": Qt.TopDockWidgetArea,
            "bottom": Qt.BottomDockWidgetArea,
        }
        qt_area = area_map.get(area_key, Qt.RightDockWidgetArea)

        self.addDockWidget(qt_area, dock)
        self.plugin_dock_widgets.append({
            "plugin_name": plugin_name,
            "dock": dock,
        })
        return dock

    def add_plugin_window(self, plugin_name, window, show=True):
        """
        API pública para complementos.
        Guarda una referencia a una ventana creada por el complemento para evitar
        que Python la destruya por garbage collection.
        """
        plugin_name = str(plugin_name).strip() or "Plugin"
        self.plugin_windows.append({
            "plugin_name": plugin_name,
            "window": window,
        })
        if show and hasattr(window, "show"):
            window.show()
        return window

    def clear_plugin_ui(self):
        """Elimina acciones, botones, docks y ventanas registrados por complementos."""
        for rec in list(self.plugin_ui_actions):
            act = rec.get("action")
            try:
                if act is not None:
                    self.plugin_menu.removeAction(act)
                    act.deleteLater()
            except Exception:
                pass
        self.plugin_ui_actions = []

        for rec in list(self.plugin_ui_toolbar_actions):
            act = rec.get("action")
            try:
                if act is not None:
                    self.plugin_toolbar.removeAction(act)
                    act.deleteLater()
            except Exception:
                pass
        self.plugin_ui_toolbar_actions = []

        for rec in list(self.plugin_dock_widgets):
            dock = rec.get("dock")
            try:
                if dock is not None:
                    self.removeDockWidget(dock)
                    dock.close()
                    dock.deleteLater()
            except Exception:
                pass
        self.plugin_dock_widgets = []

        for rec in list(self.plugin_windows):
            win = rec.get("window")
            try:
                if win is not None:
                    win.close()
                    win.deleteLater()
            except Exception:
                pass
        self.plugin_windows = []
        self._notify_embedded_plugin_ui_changed()

    def remove_plugin_ui(self, plugin_name):
        """Elimina solo la interfaz registrada por un complemento específico."""
        plugin_name = str(plugin_name).strip()
        if not plugin_name:
            return

        keep_actions = []
        for rec in self.plugin_ui_actions:
            if rec.get("plugin_name") == plugin_name:
                act = rec.get("action")
                try:
                    self.plugin_menu.removeAction(act)
                    act.deleteLater()
                except Exception:
                    pass
            else:
                keep_actions.append(rec)
        self.plugin_ui_actions = keep_actions

        keep_toolbar = []
        for rec in self.plugin_ui_toolbar_actions:
            if rec.get("plugin_name") == plugin_name:
                act = rec.get("action")
                try:
                    self.plugin_toolbar.removeAction(act)
                    act.deleteLater()
                except Exception:
                    pass
            else:
                keep_toolbar.append(rec)
        self.plugin_ui_toolbar_actions = keep_toolbar

        keep_docks = []
        for rec in self.plugin_dock_widgets:
            if rec.get("plugin_name") == plugin_name:
                dock = rec.get("dock")
                try:
                    self.removeDockWidget(dock)
                    dock.close()
                    dock.deleteLater()
                except Exception:
                    pass
            else:
                keep_docks.append(rec)
        self.plugin_dock_widgets = keep_docks

        keep_windows = []
        for rec in self.plugin_windows:
            if rec.get("plugin_name") == plugin_name:
                win = rec.get("window")
                try:
                    win.close()
                    win.deleteLater()
                except Exception:
                    pass
            else:
                keep_windows.append(rec)
        self.plugin_windows = keep_windows
        self._notify_embedded_plugin_ui_changed()

    def get_plugin_context(self):
        """
        API pública para complementos.
        Devuelve referencias útiles al estado actual del programa.
        """
        return {
            "app": self,
            "sample_name": self.current_sample_name(),
            "db": self.current_db(),
            "sample_dbs": self.sample_dbs,
            "global_components": self.global_components,
            "global_colors": self.global_colors,
            "global_types": self.global_types,
            "plot_style": self.componentry_hist_style,
            "palette_state": self.palette_state,
            "project_path": self.project_path,
        }

    # =========================================================
    # INIT / GLOBAL COMPONENTS
    # =========================================================
    def _refresh_app_state(self):
        """Enable/disable controls according to whether a real sample is loaded."""
        has_samples = bool(self.sample_dbs)
        current_db = self.current_db() if has_samples else None
        has_current = current_db is not None
        has_components = bool(current_db.components) if current_db is not None else False

        if hasattr(self, "cmb_sample"):
            self.cmb_sample.setEnabled(has_samples)

        # These controls should behave as inactive until an Excel/project is loaded.
        controls_requiring_sample = [
            "btn_export_excel", "btn_export_images", "btn_view_db",
            "btn_save_project", "btn_save_project_as",
            "chk_100", "chk_type", "chk_show_fw", "chk_mark_fw",
            "chk_hist_filter", "cmb_hist_filter_mode",
            "cmb_hist_filter_value", "chk_hist_include_undef",
            "tbl", "btn_add", "btn_remove", "btn_move_component_up",
            "btn_move_component_down", "btn_apply",
            "btn_edit_componentry_graph", "btn_fractal_analysis",
            "cmb_palette_category", "cmb_palette",
            "chk_palette_reverse", "btn_apply_palette", "btn_restore_palette",
        ]
        for attr in controls_requiring_sample:
            w = getattr(self, attr, None)
            if w is not None:
                w.setEnabled(has_samples)

        if hasattr(self, "btn_prev_sample"):
            self.btn_prev_sample.setEnabled(has_samples and self.cmb_sample.count() > 1 and self.cmb_sample.currentIndex() > 0)
        if hasattr(self, "btn_next_sample"):
            self.btn_next_sample.setEnabled(has_samples and self.cmb_sample.count() > 1 and self.cmb_sample.currentIndex() < self.cmb_sample.count() - 1)

        if hasattr(self, "btn_load_fraction"):
            self.btn_load_fraction.setEnabled(has_current and has_components)

        # Keep the filter widgets internally consistent when samples exist.
        if has_samples and hasattr(self, "chk_hist_filter"):
            self._refresh_hist_filter_ui_state()

    def _sync_global_from_samples(self):
        ordered = []
        colors = {}
        types = {}
        for _, db in self.sample_dbs.items():
            for c in db.components:
                if c not in ordered:
                    ordered.append(c)
                if c not in colors:
                    idx = max(0, len(ordered) - 1)
                    colors[c] = db.colors.get(c, self._palette_color_for_index(idx, max(1, len(ordered))))
                if c not in types:
                    t = db.types.get(c, "Accessory")
                    types[c] = t if t in TYPE_OPTIONS else "Accessory"

        palette_colors = self._current_palette_colors(max(1, len(ordered)))
        self.global_components = ordered
        self.global_colors = {
            c: colors.get(c, palette_colors[i % len(palette_colors)])
            for i, c in enumerate(ordered)
        }
        self.global_types = {c: types.get(c, "Accessory") for c in ordered}

    def _push_global_to_all_samples(self):
        for _, db in self.sample_dbs.items():
            db.set_components(self.global_components, self.global_colors, self.global_types)

    # =========================================================
    # PROJECT (SQLite .componentryproj)
    # =========================================================
    def _set_project_label(self):
        if getattr(self, "integrated_mode", False):
            host_path = self._host_project_path() if hasattr(self, "_host_project_path") else None
            if host_path:
                self.lbl_project.setText(f"Project: {host_path}")
            else:
                self.lbl_project.setText("Project: (unsaved)")
            return
        if self.project_path:
            self.lbl_project.setText(f"Project: {self.project_path}")
        else:
            self.lbl_project.setText("Project: (unsaved)")

    def _db_to_dict(self, db: ComponentryDB):
        return db.to_dict()

    def _dict_to_db(self, payload: dict):
        # New format (df_to_payload)
        if isinstance(payload.get("gsd", None), dict):
            return ComponentryDB.from_dict(payload)

        # Old compatibility fallback (orient="split")
        import io
        gsd = pd.read_json(io.StringIO(payload["gsd"]), orient="split")
        db = ComponentryDB(gsd)

        comps = [str(c) for c in payload.get("components", [])]
        colors = payload.get("colors", {}) or {}
        types = payload.get("types", {}) or {}
        db.set_components(comps, colors, types)

        frac100 = pd.read_json(io.StringIO(payload["frac100"]), orient="split")
        total = pd.read_json(io.StringIO(payload["total"]), orient="split")

        phis = db.gsd["phi"].astype(float).values
        frac100.index = frac100.index.astype(float)
        total.index = total.index.astype(float)

        db.frac100 = frac100.reindex(index=phis, columns=db.components, fill_value=0.0).astype(float)
        db.total = total.reindex(index=phis, columns=db.components, fill_value=0.0).astype(float)
        db.filled = set([float(v) for v in payload.get("filled", [])])

        return db

    def _collect_project_dict(self):
        return {
            "version": PROJECT_SCHEMA_VERSION,
            "sample_order": list(self.sample_dbs.keys()),
            "samples": {k: self._db_to_dict(v) for k, v in self.sample_dbs.items()},
            "global_components": list(self.global_components),
            "global_colors": dict(self.global_colors),
            "global_types": dict(self.global_types),
            "plot_style": dict(self.componentry_hist_style),
            "palette_state": dict(self.palette_state),
            "ui": {
                "current_sample": self.current_sample_name(),
                "bars100": bool(self.chk_100.isChecked()),
                "by_type": bool(self.chk_type.isChecked()),
                "show_fw": bool(self.chk_show_fw.isChecked()),
                "mark_fw": bool(self.chk_mark_fw.isChecked()),
                "hist_filter_enabled": bool(self.chk_hist_filter.isChecked()),
                "hist_filter_mode": self.cmb_hist_filter_mode.currentText(),
                "hist_filter_value": self.cmb_hist_filter_value.currentText(),
                "hist_filter_include_undef": bool(self.chk_hist_include_undef.isChecked()),
                "width": int(self.width()),
                "height": int(self.height()),
            }
        }

    def _apply_project_dict(self, data: dict):
        sample_order = data.get("sample_order", [])
        samples_payload = data.get("samples", {})
        new_map = {}

        for sname in sample_order:
            if sname in samples_payload:
                new_map[sname] = self._dict_to_db(samples_payload[sname])

        if not new_map:
            for sname, payload in samples_payload.items():
                new_map[sname] = self._dict_to_db(payload)

        if not new_map:
            raise ValueError("Project file has no valid samples.")

        self.sample_dbs = new_map

        self.componentry_hist_style = normalize_componentry_hist_style(data.get("plot_style", {}))
        pstate = data.get("palette_state", {}) or {}
        pname = str(pstate.get("name", DEFAULT_PALETTE_NAME))
        if pname not in COMPONENT_PALETTES:
            pname = DEFAULT_PALETTE_NAME
        pcat = COMPONENT_PALETTES[pname]["category"]
        self.palette_state = {
            "category": pcat,
            "name": pname,
            "reversed": bool(pstate.get("reversed", False)),
            "auto_apply": bool(pstate.get("auto_apply", False)),
        }
        self._populate_palette_controls()

        g_comps = data.get("global_components", [])
        g_colors = data.get("global_colors", {}) or {}
        g_types = data.get("global_types", {}) or {}

        # Samples are authoritative.  This is required for projects where a
        # component name/type was intentionally changed in only one sample.
        # Old projects that contain globals but no per-sample component definitions
        # still receive the legacy global definition once for compatibility.
        if g_comps and all(not db.components for db in self.sample_dbs.values()):
            self.global_components = [str(c).strip() for c in g_comps if str(c).strip()]
            palette_colors = self._current_palette_colors(max(1, len(self.global_components)))
            self.global_colors = {
                c: g_colors.get(c, palette_colors[i % len(palette_colors)])
                for i, c in enumerate(self.global_components)
            }
            self.global_types = {
                c: (g_types.get(c, "Accessory") if g_types.get(c, "Accessory") in TYPE_OPTIONS else "Accessory")
                for c in self.global_components
            }
            self._push_global_to_all_samples()
        else:
            self._sync_global_from_samples()

        ui = data.get("ui", {}) or {}

        self.chk_100.setChecked(bool(ui.get("bars100", False)))
        self.chk_type.setChecked(bool(ui.get("by_type", False)))
        self.chk_show_fw.setChecked(bool(ui.get("show_fw", False)))
        self.chk_mark_fw.setChecked(bool(ui.get("mark_fw", False)))

        self._reload_sample_combo(select_first=True)

        target = str(ui.get("current_sample", "")).strip()
        if target:
            idx = self.cmb_sample.findText(target)
            if idx >= 0:
                self.cmb_sample.setCurrentIndex(idx)

        # restore histogram filter after globals are loaded
        self.chk_hist_filter.setChecked(bool(ui.get("hist_filter_enabled", False)))

        mode_txt = str(ui.get("hist_filter_mode", "")).strip()
        if mode_txt:
            idx_mode = self.cmb_hist_filter_mode.findText(mode_txt)
            if idx_mode >= 0:
                self.cmb_hist_filter_mode.setCurrentIndex(idx_mode)

        self._refresh_hist_filter_value_options()

        val_txt = str(ui.get("hist_filter_value", "")).strip()
        if val_txt:
            idx_val = self.cmb_hist_filter_value.findText(val_txt)
            if idx_val >= 0:
                self.cmb_hist_filter_value.setCurrentIndex(idx_val)

        self.chk_hist_include_undef.setChecked(bool(ui.get("hist_filter_include_undef", False)))
        self._refresh_hist_filter_ui_state()

        try:
            w_saved = int(ui.get("width", self.width()))
            h_saved = int(ui.get("height", self.height()))
        except Exception:
            w_saved, h_saved = self.width(), self.height()
        self._safe_resize_from_saved(w_saved, h_saved)

        self.refresh_component_table_from_global()
        self.update_summary()
        self.plot_histogram()

    def _safe_resize_from_saved(self, w_saved: int, h_saved: int):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(max(980, min(int(w_saved), 1360)),
                        max(680, min(int(h_saved), 980)))
            return

        geo = screen.availableGeometry()
        max_w = max(980, min(1360, geo.width() - 60))
        max_h = max(680, geo.height() - 60)

        w = max(980, min(int(w_saved), int(max_w)))
        h = max(680, min(int(h_saved), int(max_h)))
        self.resize(w, h)

    def open_project_dialog(self):
        if getattr(self, "integrated_mode", False) and getattr(self, "host_app", None) is not None:
            return self.host_app.open_full_project_dialog()

        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "Componentry Project (*.componentryproj);;All files (*)"
        )
        if not path:
            return

        try:
            con = sqlite3.connect(path)
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            cur.execute("SELECT v FROM kv WHERE k='project_json'")
            row = cur.fetchone()
            con.close()

            if not row:
                raise ValueError("Invalid project file: project_json not found.")

            data = json.loads(row[0])
            self._apply_project_dict(data)
            self.project_path = path
            self._set_project_label()

            QMessageBox.information(
                self,
                "Project opened",
                "Project loaded successfully.\n\n"
                "Image-based componentry was restored, but source images must be loaded again manually."
            )

        except Exception as e:
            QMessageBox.warning(self, "Open project error", str(e))

    def save_project(self):
        if getattr(self, "integrated_mode", False) and getattr(self, "host_app", None) is not None:
            self._save_session_refresh_host() if hasattr(self, "_save_session_refresh_host") else None
            return self.host_app.save_full_project()

        if not self.project_path:
            self.save_project_as()
            return
        self._save_project_to_path(self.project_path)

    def save_project_as(self):
        if getattr(self, "integrated_mode", False) and getattr(self, "host_app", None) is not None:
            return self.host_app.save_full_project_as()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save project as", "", "Componentry Project (*.componentryproj)"
        )
        if not path:
            return
        if not path.lower().endswith(".componentryproj"):
            path += ".componentryproj"

        self._save_project_to_path(path)
        self.project_path = path
        self._set_project_label()

    def _save_project_to_path(self, path: str):
        try:
            data = self._collect_project_dict()
            payload = json.dumps(data, ensure_ascii=False)

            con = sqlite3.connect(path)
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
            cur.execute("INSERT OR REPLACE INTO kv(k, v) VALUES(?, ?)", ("project_json", payload))
            con.commit()
            con.close()

            QMessageBox.information(
                self,
                "Project saved",
                "Project saved successfully.\n\n"
                "Note: loaded analysis images are not stored in the project file."
            )
        except Exception as e:
            QMessageBox.warning(self, "Save project error", str(e))

    # =========================================================
    # SAMPLE NAVIGATION / ACCESS
    # =========================================================
    def _reload_sample_combo(self, select_first=False):
        current = self.cmb_sample.currentText().strip() if self.cmb_sample.count() else ""

        self.cmb_sample.blockSignals(True)
        self.cmb_sample.clear()

        if not self.sample_dbs:
            self.cmb_sample.addItem("No sample loaded")
            self.cmb_sample.setCurrentIndex(0)
            self.cmb_sample.blockSignals(False)
            self._refresh_app_state()
            return

        for name in self.sample_dbs.keys():
            self.cmb_sample.addItem(name)

        self.cmb_sample.blockSignals(False)

        if select_first:
            self.cmb_sample.setCurrentIndex(0)
        else:
            idx = self.cmb_sample.findText(current)
            self.cmb_sample.setCurrentIndex(idx if idx >= 0 else 0)

        self._refresh_app_state()
        self.on_sample_changed()

    def current_sample_name(self):
        if not self.sample_dbs:
            return ""
        return self.cmb_sample.currentText().strip()

    def current_db(self):
        return self.sample_dbs.get(self.current_sample_name(), None)

    def prev_sample(self):
        n = self.cmb_sample.count()
        if n <= 0:
            return
        i = self.cmb_sample.currentIndex()
        if i > 0:
            self.cmb_sample.setCurrentIndex(i - 1)

    def next_sample(self):
        n = self.cmb_sample.count()
        if n <= 0:
            return
        i = self.cmb_sample.currentIndex()
        if i < n - 1:
            self.cmb_sample.setCurrentIndex(i + 1)

    def on_sample_changed(self):
        db = self.current_db()
        self.btn_load_fraction.setEnabled((db is not None) and bool(db.components if db is not None else []))
        self._refresh_app_state()
        if hasattr(self, "tbl"):
            self.refresh_component_table_from_global()
        self.update_summary()
        self.plot_histogram()

    # =========================================================
    # LOAD TABLE (EXCEL)
    # =========================================================
    def _load_component_matrix_workbook(self, parsed: dict):
        new_map = {}
        union_components = []

        for _, payload in parsed.items():
            for c in payload["components_order"]:
                if c not in union_components:
                    union_components.append(c)

        if not union_components:
            raise ValueError("No components found in workbook.")

        palette_colors = self._current_palette_colors(max(1, len(union_components)))
        g_colors = {c: palette_colors[i % len(palette_colors)] for i, c in enumerate(union_components)}
        g_types = {c: guess_type_from_component(c) for c in union_components}

        for sname, payload in parsed.items():
            gsd = payload["gsd"]
            mat = payload["total_matrix"]

            db = ComponentryDB(gsd)
            db.load_totals_matrix(
                total_df=mat,
                components_order=union_components,
                colors_by_name=g_colors,
                types_by_name=g_types
            )
            new_map[sname] = db

        if not new_map:
            raise ValueError("No valid sample columns were imported.")

        self.sample_dbs = new_map
        self.global_components = list(union_components)
        self.global_colors = dict(g_colors)
        self.global_types = dict(g_types)

        self.project_path = None
        self._set_project_label()

        self.chk_hist_filter.setChecked(False)

        self._reload_sample_combo(select_first=True)
        self.refresh_component_table_from_global()
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
        self.update_summary()
        self.plot_histogram()

    def _load_gsd_workbook(self, parsed: dict):
        """
        Carga un Excel granulométrico simple:
          - sample_dbs con las GSD
          - sin componentes todavía
        """
        new_map = {}

        for sname, payload in parsed.items():
            gsd = payload.get("gsd", None)
            if gsd is None or gsd.empty:
                continue

            db = ComponentryDB(gsd)
            new_map[sname] = db

        if not new_map:
            raise ValueError("No valid granulometry samples were found.")

        self.sample_dbs = new_map

        # Este formato no trae componentes
        self.global_components = []
        self.global_colors = {}
        self.global_types = {}

        self.project_path = None
        self._set_project_label()

        self.chk_hist_filter.setChecked(False)

        self._reload_sample_combo(select_first=True)
        self.refresh_component_table_from_global()
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
        self.update_summary()
        self.plot_histogram()

    def open_load_components_table(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import component table",
            "",
            "Excel files (*.xlsx *.xls *.xlsm)"
        )
        if not path:
            return

        err_gsd = None
        err_components = None

        # Intento 1: formato granulométrico / GSD
        try:
            parsed = parse_gsd_excel(path)
            self._load_gsd_workbook(parsed)
            QMessageBox.information(
                self,
                "Import completed",
                f"Loaded {len(self.sample_dbs)} granulometry sample(s) from Excel."
            )
            return
        except Exception as e:
            err_gsd = str(e)

        # Intento 2: formato de tabla de componentes
        try:
            parsed = parse_components_excel(path)
            self._load_component_matrix_workbook(parsed)
            QMessageBox.information(
                self,
                "Import completed",
                f"Loaded {len(self.sample_dbs)} sample(s) from component table."
            )
            return
        except Exception as e:
            err_components = str(e)

        QMessageBox.warning(
            self,
            "Import error",
            "The workbook could not be interpreted with either supported format.\n\n"
            "1) Granulometry / GSD format:\n"
            "   row 1 = sample name, A = phi/labels, B.. = samples\n"
            "   rows with numeric A are used as phi rows\n\n"
            "2) Component matrix format:\n"
            "   A = fraction, B = phi, C = component, D.. = samples\n\n"
            f"GSD parser error:\n{err_gsd}\n\n"
            f"Component parser error:\n{err_components}"
        )

    # =========================================================
    # EXPORT EXCEL
    # =========================================================
    def open_export_excel(self):
        if not self.sample_dbs:
            QMessageBox.warning(self, "Export Excel", "No samples loaded.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Export Excel")
        dlg.resize(460, 170)

        lay = QVBoxLayout(dlg)
        form = QFormLayout()

        cmb_scope = QComboBox()
        cmb_scope.addItems(["Current sample", "All samples"])
        cmb_scope.setCurrentIndex(0)
        form.addRow("Export scope:", cmb_scope)
        lay.addLayout(form)

        hb = QHBoxLayout()
        hb.addStretch(1)
        btn_ok = QPushButton("Continue")
        btn_cancel = QPushButton("Cancel")
        hb.addWidget(btn_ok)
        hb.addWidget(btn_cancel)
        lay.addLayout(hb)

        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec_() != QDialog.Accepted:
            return

        scope = cmb_scope.currentText().strip()
        samples = [self.current_sample_name()] if scope == "Current sample" else list(self.sample_dbs.keys())

        path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel files (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        try:
            df_export = self._build_export_excel_dataframe(samples)
            with pd.ExcelWriter(path, engine="openpyxl") as w:
                df_export.to_excel(w, index=False, header=False, sheet_name="componentry")
            QMessageBox.information(self, "Export Excel", f"Excel exported:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Excel error", str(e))

    def _build_export_excel_dataframe(self, sample_names):
        """
        Formato compatible con parse_components_excel:
        - Col A: etiqueta libre
        - Col B: phi
        - Col C: component
        - Col D..: samples
        Incluye componente "No definido" para cubrir fracciones sin carga de componentes.
        """
        components = list(self.global_components)
        undef_name = "No definido"
        if undef_name not in components:
            components.append(undef_name)

        phis_set = set()
        for s in sample_names:
            db = self.sample_dbs[s]
            for _, r in db.gsd.iterrows():
                phi = float(r["phi"])
                wt = float(r["wt"])
                if wt > 0:
                    phis_set.add(phi)

        phis = sorted(list(phis_set))
        rows = []

        header = ["Fraction", "phi", "component"] + sample_names
        rows.append(header)

        for phi in phis:
            for comp in components:
                row = ["", float(phi), comp]

                for s in sample_names:
                    db = self.sample_dbs[s]
                    wt_map = db.wt_map()
                    bin_wt = float(wt_map.get(float(phi), 0.0))

                    if comp != undef_name:
                        v = 0.0
                        if (comp in db.total.columns) and (float(phi) in db.total.index):
                            vv = _to_float(db.total.loc[float(phi), comp])
                            if np.isfinite(vv) and vv > 0:
                                v = float(vv)
                        row.append(v)
                    else:
                        if float(phi) not in db.filled:
                            row.append(bin_wt)
                        else:
                            s_known = 0.0
                            for c in self.global_components:
                                if (c in db.total.columns) and (float(phi) in db.total.index):
                                    vv = _to_float(db.total.loc[float(phi), c])
                                    if np.isfinite(vv) and vv > 0:
                                        s_known += float(vv)
                            residual = max(0.0, bin_wt - s_known)
                            row.append(residual)

                rows.append(row)

        sum_row = ["SUM", "", ""] + [100.0 for _ in sample_names]
        rows.append(sum_row)

        return pd.DataFrame(rows)

    # =========================================================
    # EXPORT IMAGES
    # =========================================================
    def open_export_images(self):
        if not self.sample_dbs:
            QMessageBox.warning(self, "Export images", "No samples loaded.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Export images")
        dlg.resize(560, 220)
        lay = QVBoxLayout(dlg)

        form = QFormLayout()

        cmb_scope = QComboBox()
        cmb_scope.addItems(["Current sample", "All samples"])
        cmb_scope.setCurrentIndex(0)
        form.addRow("Export scope:", cmb_scope)

        cmb_format = QComboBox()
        cmb_format.addItems(["JPEG", "PDF", "TIFF", "SVG", "PNG"])
        cmb_format.setCurrentText("PNG")
        form.addRow("Format:", cmb_format)

        w_folder = QWidget()
        hf = QHBoxLayout(w_folder)
        hf.setContentsMargins(0, 0, 0, 0)
        le_folder = QLineEdit()
        le_folder.setPlaceholderText("Select destination folder")
        btn_browse = QPushButton("Browse...")
        hf.addWidget(le_folder, 1)
        hf.addWidget(btn_browse)
        form.addRow("Folder:", w_folder)

        lay.addLayout(form)

        hb = QHBoxLayout()
        hb.addStretch(1)
        btn_ok = QPushButton("Export")
        btn_cancel = QPushButton("Cancel")
        hb.addWidget(btn_ok)
        hb.addWidget(btn_cancel)
        lay.addLayout(hb)

        def _browse_folder():
            d = QFileDialog.getExistingDirectory(self, "Select destination folder", "")
            if d:
                le_folder.setText(d)

        btn_browse.clicked.connect(_browse_folder)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)

        if dlg.exec_() != QDialog.Accepted:
            return

        scope = cmb_scope.currentText().strip()
        fmt = cmb_format.currentText().strip().upper()
        folder = le_folder.text().strip()

        if not folder:
            QMessageBox.warning(self, "Export images", "Please select a destination folder.")
            return

        try:
            self._export_images(scope, fmt, folder)
        except Exception as e:
            QMessageBox.warning(self, "Export images error", str(e))

    def _sanitize_filename(self, name: str):
        import re
        s = str(name)
        s = re.sub(r'[\\/:*?"<>|]+', "_", s)
        s = s.strip().strip(".")
        return s or "sample"

    def _export_images(self, scope: str, fmt: str, folder: str):
        format_map = {
            "JPEG": ("jpg", "jpeg"),
            "PDF": ("pdf", "pdf"),
            "TIFF": ("tiff", "tiff"),
            "SVG": ("svg", "svg"),
            "PNG": ("png", "png"),
        }
        if fmt not in format_map:
            raise ValueError("Unsupported format.")

        ext, mpl_fmt = format_map[fmt]

        if scope == "Current sample":
            samples = [self.current_sample_name()]
        else:
            samples = list(self.sample_dbs.keys())

        if not samples:
            raise ValueError("No samples available for export.")

        bars_100 = bool(self.chk_100.isChecked())
        by_type = bool(self.chk_type.isChecked())
        show_fw = bool(self.chk_show_fw.isChecked())
        mark_fw = bool(self.chk_mark_fw.isChecked())

        ok = 0
        errors = []

        for sname in samples:
            db = self.sample_dbs.get(sname)
            if db is None:
                errors.append(f"{sname}: sample not found.")
                continue

            out_name = f"{self._sanitize_filename(sname)}.{ext}"
            out_path = os.path.join(folder, out_name)

            fig = None
            try:
                style = normalize_componentry_hist_style(self.componentry_hist_style)
                export_dpi = 300 if ext in ("jpg", "png", "tiff") else 150
                fig = Figure(
                    figsize=(style["figure_width"], style["figure_height"]),
                    dpi=export_dpi,
                    facecolor="white",
                )
                ax = fig.add_subplot(111)

                self._draw_histogram(
                    ax=ax,
                    db=db,
                    sample_title=sname,
                    bars_100=bars_100,
                    by_type=by_type,
                    show_fw=show_fw,
                    mark_fw=mark_fw,
                )

                fig.savefig(out_path, format=mpl_fmt, dpi=export_dpi, bbox_inches=None, facecolor="white")
                fig.clear()
                ok += 1

            except Exception as e:
                if fig is not None:
                    try:
                        fig.clear()
                    except Exception:
                        pass
                errors.append(f"{sname}: {e}")

        if errors:
            msg = (
                f"Exported {ok}/{len(samples)} image(s).\n\n"
                "Errors:\n" + "\n".join(errors[:10])
            )
            QMessageBox.warning(self, "Export images completed with warnings", msg)
        else:
            QMessageBox.information(self, "Export images", f"Exported {ok} image(s) successfully.")

    # =========================================================
    # LOAD FRACTION / VIEW DB
    # =========================================================
    def open_load_components(self):
        db = self.current_db()
        sname = self.current_sample_name()
        if db is None:
            return

        if not db.components:
            QMessageBox.warning(self, "Error", "Define components first, then Apply changes.")
            return

        dlg = LoadComponentsDialog(db, sname, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._notify_host_componentry_changed()
            self.plot_histogram()

    def open_view_db(self):
        db = self.current_db()
        sname = self.current_sample_name()
        if db is None:
            return
        dlg = ViewDBDialog(db, sname, parent=self)
        dlg.exec_()

    # =========================================================
    # FRACTAL ANALYSIS (native external module)
    # =========================================================
    def _load_fractal_analysis_module(self):
        cached = getattr(self, "_fractal_analysis_module", None)
        if cached is not None and hasattr(cached, "FractalAnalysisWindow"):
            return cached

        search_dirs = []
        for getter in (
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

        checked = []
        for d in search_dirs:
            path = os.path.join(d, "FractalAnalysis.py")
            checked.append(path)
            if not os.path.isfile(path):
                continue
            try:
                module_name = "componentry_fractal_analysis_" + str(abs(hash(os.path.abspath(path))))
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError("Could not create module spec.")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                if not hasattr(module, "FractalAnalysisWindow"):
                    raise ImportError("FractalAnalysisWindow was not found in FractalAnalysis.py.")
                self._fractal_analysis_module = module
                return module
            except Exception:
                tb = traceback.format_exc()
                raise ImportError(f"Could not load FractalAnalysis.py from:\n{path}\n\n{tb}")

        raise FileNotFoundError(
            "FractalAnalysis.py was not found. Keep it in the same folder as Componentry-FW.py.\n\n"
            + "Checked:\n" + "\n".join(checked)
        )

    def _prune_fractal_windows(self):
        live = []
        for win in list(getattr(self, "_fractal_analysis_windows", [])):
            try:
                if win is not None and win.isVisible():
                    live.append(win)
            except RuntimeError:
                pass
            except Exception:
                pass
        self._fractal_analysis_windows = live

    def open_fractal_analysis(self):
        if not self.sample_dbs or self.current_db() is None:
            QMessageBox.information(self, "Fractal analysis", "Load at least one sample before opening Fractal analysis.")
            return None

        self._prune_fractal_windows()
        if self._fractal_analysis_windows:
            # Keep one live native analysis window per Componentry view.  Repeated
            # clicks activate it; after closing it can be opened again normally.
            win = self._fractal_analysis_windows[0]
            try:
                win.show()
                win.raise_()
                win.activateWindow()
                if hasattr(win, "refresh_from_componentry"):
                    win.refresh_from_componentry(initial=False)
                return win
            except Exception:
                self._fractal_analysis_windows = []

        try:
            module = self._load_fractal_analysis_module()
            win = module.FractalAnalysisWindow(self)
            self._fractal_analysis_windows.append(win)

            def _forget_window(*_args, _win=win):
                try:
                    self._fractal_analysis_windows = [w for w in self._fractal_analysis_windows if w is not _win]
                except Exception:
                    self._fractal_analysis_windows = []

            win.destroyed.connect(_forget_window)
            win.show()
            win.raise_()
            win.activateWindow()
            return win
        except Exception as exc:
            QMessageBox.warning(self, "Fractal analysis", str(exc))
            return None


    def _notify_host_componentry_changed(self):
        host = getattr(self, "host_app", None)
        if host is not None and hasattr(host, "_fw_componentry_after_change"):
            try:
                host._fw_componentry_after_change(self.get_view_state())
            except TypeError:
                try:
                    host._fw_componentry_after_change()
                except Exception:
                    pass
            except Exception:
                pass

    # =========================================================
    # COMPONENT TABLE (GLOBAL) / PALETTES
    # =========================================================
    def _current_palette_name(self):
        name = str(self.palette_state.get("name", DEFAULT_PALETTE_NAME))
        return name if name in COMPONENT_PALETTES else DEFAULT_PALETTE_NAME

    def _current_palette_colors(self, count):
        return component_palette_colors(
            self._current_palette_name(),
            max(1, int(count)),
            bool(self.palette_state.get("reversed", False)),
        )

    def _current_palette_swatches(self):
        name = self._current_palette_name()
        reverse = bool(self.palette_state.get("reversed", False))
        spec = COMPONENT_PALETTES.get(name, COMPONENT_PALETTES[DEFAULT_PALETTE_NAME])
        if spec.get("kind") == "discrete":
            base_count = max(1, len(spec.get("colors", []) or []))
            row_count = self.tbl.rowCount() if hasattr(self, "tbl") else 0
            return component_palette_colors(name, max(base_count, row_count), reverse)
        return _palette_base_swatches(name, reverse, continuous_count=12)

    def _palette_color_for_index(self, index, count=None):
        n = max(int(index) + 1, int(count or 1))
        colors = self._current_palette_colors(n)
        return colors[int(index) % len(colors)]

    def _populate_palette_controls(self):
        if not hasattr(self, "cmb_palette_category"):
            return

        desired_name = str(self.palette_state.get("name", DEFAULT_PALETTE_NAME))
        if desired_name not in COMPONENT_PALETTES:
            desired_name = DEFAULT_PALETTE_NAME
        desired_category = str(self.palette_state.get("category", COMPONENT_PALETTES[desired_name]["category"]))
        if desired_category not in _palette_categories():
            desired_category = COMPONENT_PALETTES[desired_name]["category"]

        self.cmb_palette_category.blockSignals(True)
        self.cmb_palette_category.clear()
        self.cmb_palette_category.addItems(_palette_categories())
        idx_cat = self.cmb_palette_category.findText(desired_category)
        self.cmb_palette_category.setCurrentIndex(idx_cat if idx_cat >= 0 else 0)
        self.cmb_palette_category.blockSignals(False)

        self._repopulate_palette_names(desired_name)
        self.chk_palette_reverse.blockSignals(True)
        self.chk_palette_reverse.setChecked(bool(self.palette_state.get("reversed", False)))
        self.chk_palette_reverse.blockSignals(False)
        self.chk_palette_auto_apply.blockSignals(True)
        self.chk_palette_auto_apply.setChecked(bool(self.palette_state.get("auto_apply", False)))
        self.chk_palette_auto_apply.blockSignals(False)
        self._update_palette_state_from_controls()
        self._update_palette_preview()

    def _repopulate_palette_names(self, preferred_name=None):
        category = self.cmb_palette_category.currentText().strip() or DEFAULT_PALETTE_CATEGORY
        names = _palette_names_for_category(category)
        if not names:
            names = [DEFAULT_PALETTE_NAME]

        self.cmb_palette.blockSignals(True)
        self.cmb_palette.clear()
        self.cmb_palette.addItems(names)
        target = preferred_name if preferred_name in names else names[0]
        idx = self.cmb_palette.findText(target)
        self.cmb_palette.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_palette.blockSignals(False)

    def _update_palette_state_from_controls(self):
        if not hasattr(self, "cmb_palette"):
            return
        name = self.cmb_palette.currentText().strip()
        if name not in COMPONENT_PALETTES:
            name = DEFAULT_PALETTE_NAME
        auto_apply = False
        if hasattr(self, "chk_palette_auto_apply"):
            auto_apply = bool(self.chk_palette_auto_apply.isChecked())
        self.palette_state = {
            "category": COMPONENT_PALETTES[name]["category"],
            "name": name,
            "reversed": bool(self.chk_palette_reverse.isChecked()),
            "auto_apply": auto_apply,
        }

    def _palette_auto_apply_enabled(self):
        return bool(
            hasattr(self, "chk_palette_auto_apply")
            and self.chk_palette_auto_apply.isChecked()
        )

    def _step_palette(self, step):
        if not hasattr(self, "cmb_palette"):
            return
        count = self.cmb_palette.count()
        if count <= 0:
            return
        current = self.cmb_palette.currentIndex()
        if current < 0:
            current = 0
        self.cmb_palette.setCurrentIndex((current + int(step)) % count)

    def _on_palette_category_changed(self):
        self._repopulate_palette_names()
        self._update_palette_state_from_controls()
        self._update_palette_preview()
        if self._palette_auto_apply_enabled():
            self.apply_selected_palette()
        else:
            self._refresh_palette_color_widgets()

    def _on_palette_selection_changed(self):
        self._update_palette_state_from_controls()
        self._update_palette_preview()
        if self._palette_auto_apply_enabled():
            self.apply_selected_palette()
        else:
            self._refresh_palette_color_widgets()

    def _on_palette_auto_apply_toggled(self, checked):
        self._update_palette_state_from_controls()
        if bool(checked):
            self.apply_selected_palette()

    def _update_palette_preview(self):
        if not hasattr(self, "lbl_palette_preview"):
            return
        colors = self._current_palette_swatches()
        width = max(240, self.lbl_palette_preview.width() or 320)
        self.lbl_palette_preview.setPixmap(_palette_preview_pixmap(colors, width=width, height=28))
        self.lbl_palette_preview.setToolTip(
            f"{self._current_palette_name()}\n" + ", ".join(colors)
        )

    def apply_selected_palette(self):
        n = self.tbl.rowCount()
        if n <= 0:
            return
        self._update_palette_state_from_controls()
        colors = self._current_palette_colors(n)
        for row in range(n):
            self._set_color_cell(row, colors[row % len(colors)])
        self._refresh_palette_color_widgets()
        self.apply_changes()

    def restore_default_palette(self):
        auto_apply = bool(
            self.chk_palette_auto_apply.isChecked()
            if hasattr(self, "chk_palette_auto_apply")
            else self.palette_state.get("auto_apply", False)
        )
        self.palette_state = {
            "category": DEFAULT_PALETTE_CATEGORY,
            "name": DEFAULT_PALETTE_NAME,
            "reversed": False,
            "auto_apply": auto_apply,
        }
        self._populate_palette_controls()
        self.apply_selected_palette()

    def _set_color_cell(self, row, hex_color):
        color = _normalize_hex_color(hex_color, self._palette_color_for_index(row, max(1, self.tbl.rowCount())))
        it = self.tbl.item(row, 1)
        if it is None:
            it = QTableWidgetItem("")
            self.tbl.setItem(row, 1, it)
        it.setFlags(it.flags() & ~Qt.ItemIsEditable)
        it.setText("")
        it.setBackground(QColor(color))
        it.setData(Qt.UserRole, color)
        it.setToolTip(color.upper())

    def _get_color_cell(self, row):
        it = self.tbl.item(row, 1)
        fallback = self._palette_color_for_index(row, max(1, self.tbl.rowCount()))
        if it is None:
            return fallback
        v = it.data(Qt.UserRole)
        if isinstance(v, str) and QColor(v).isValid():
            return QColor(v).name()
        try:
            c = it.background().color()
            return c.name() if c.isValid() else fallback
        except Exception:
            return fallback

    def _ensure_palette_widget(self, row, current_color=None):
        current_color = _normalize_hex_color(current_color or self._get_color_cell(row))
        cmb = self.tbl.cellWidget(row, 2)
        if not isinstance(cmb, QComboBox):
            cmb = QComboBox()
            cmb.currentIndexChanged.connect(lambda _idx, w=cmb: self._on_row_palette_combo_changed(w))
            self.tbl.setCellWidget(row, 2, cmb)

        cmb.blockSignals(True)
        cmb.clear()
        swatches = self._current_palette_swatches()
        match_index = -1
        for i, color in enumerate(swatches):
            c = _normalize_hex_color(color)
            cmb.addItem(_color_icon(c), f"{i + 1}: {c.upper()}", c)
            if c.lower() == current_color.lower():
                match_index = i
        if match_index < 0:
            cmb.addItem(_color_icon(current_color), f"Custom: {current_color.upper()}", current_color)
            match_index = cmb.count() - 1
        cmb.setCurrentIndex(match_index)
        cmb.blockSignals(False)
        return cmb

    def _on_row_palette_combo_changed(self, combo):
        row = -1
        for r in range(self.tbl.rowCount()):
            if self.tbl.cellWidget(r, 2) is combo:
                row = r
                break
        if row < 0:
            return
        color = combo.currentData()
        if isinstance(color, str) and QColor(color).isValid():
            self._set_color_cell(row, color)

    def _refresh_palette_color_widgets(self):
        if not hasattr(self, "tbl"):
            return
        for row in range(self.tbl.rowCount()):
            self._ensure_palette_widget(row, self._get_color_cell(row))

    def _ensure_type_widget(self, row, initial="Accessory"):
        w = self.tbl.cellWidget(row, 3)
        desired = initial if initial in TYPE_OPTIONS else "Accessory"
        if not isinstance(w, QComboBox):
            cmb = QComboBox()
            cmb.addItems(TYPE_OPTIONS)
            cmb.setCurrentText(desired)
            cmb.currentTextChanged.connect(
                lambda _text, combo=cmb: self._on_component_type_combo_changed(combo)
            )
            self.tbl.setCellWidget(row, 3, cmb)
            return cmb

        # Reordering/reloading may retain an existing combobox at this visual row.
        # Synchronize it without firing the scope-selection dialog.
        w.blockSignals(True)
        w.setCurrentText(desired)
        w.blockSignals(False)
        return w

    def _get_type_value(self, row):
        w = self.tbl.cellWidget(row, 3)
        if isinstance(w, QComboBox):
            t = str(w.currentText()).strip()
            return t if t in TYPE_OPTIONS else "Accessory"
        return "Accessory"

    def _get_selected_row(self):
        r = self.tbl.currentRow()
        if r >= 0:
            return r
        idxs = self.tbl.selectionModel().selectedRows()
        if idxs:
            return int(idxs[0].row())
        return -1

    def _on_tbl_cell_clicked(self, row, col):
        try:
            self.tbl.setCurrentCell(row, col)
        except Exception:
            pass
        if col == 1:
            self.pick_color_for_row(row)

    def _restore_component_name_item(self, item, old_name):
        if item is None:
            return
        item.blockSignals(True)
        item.setText(str(old_name))
        item.blockSignals(False)

    def _finish_scoped_component_metadata_change(self):
        self._sync_global_from_samples()
        self.refresh_component_table_from_global()
        self.update_summary()
        self.plot_histogram()
        self._notify_host_componentry_changed()

    def _on_component_name_item_changed(self, item):
        """Rename a component immediately and ask for current/all-sample scope.

        Only the component identifier is changed. Existing numerical data, type and
        color remain attached to that component.
        """
        if getattr(self, "_component_scope_change_in_progress", False):
            return
        if item is None or item.column() != 0:
            return

        db = self.current_db()
        if db is None:
            return

        old_name = item.data(Qt.UserRole)
        old_name = str(old_name).strip() if old_name not in (None, "") else ""
        new_name = str(item.text()).strip()

        # New rows do not yet have an original component identity; Apply changes
        # remains responsible for adding them.
        if not old_name:
            return
        if new_name == old_name:
            return
        if not new_name:
            QMessageBox.warning(self, "Error", "Component names cannot be empty.")
            self._restore_component_name_item(item, old_name)
            return

        rename_map = {old_name: new_name}
        scope = self._ask_component_change_scope(rename_map, [])
        if scope == "cancel":
            self._restore_component_name_item(item, old_name)
            return

        targets = list(self.sample_dbs.values()) if scope == "all" else [db]

        # Validate every affected sample first so an all-sample rename is atomic.
        try:
            for target_db in targets:
                if old_name not in target_db.components:
                    continue
                if new_name in target_db.components and new_name != old_name:
                    raise ValueError(
                        f"Cannot rename '{old_name}' to '{new_name}' because "
                        f"'{new_name}' already exists in an affected sample."
                    )

            self._component_scope_change_in_progress = True
            for target_db in targets:
                if old_name in target_db.components:
                    target_db.rename_component(old_name, new_name)
        except Exception as e:
            QMessageBox.warning(self, "Rename component", str(e))
            self._restore_component_name_item(item, old_name)
            return
        finally:
            self._component_scope_change_in_progress = False

        self._finish_scoped_component_metadata_change()

    def _on_component_type_combo_changed(self, combo):
        """Change Juvenile/Lithic/Accessory immediately with current/all scope.

        The component name, numerical data and color are not modified.
        """
        if getattr(self, "_component_scope_change_in_progress", False):
            return
        if not isinstance(combo, QComboBox):
            return

        row = -1
        for r in range(self.tbl.rowCount()):
            if self.tbl.cellWidget(r, 3) is combo:
                row = r
                break
        if row < 0:
            return

        db = self.current_db()
        if db is None:
            return
        item = self.tbl.item(row, 0)
        if item is None:
            return

        component_name = str(item.text()).strip()
        if component_name not in db.components:
            return

        new_type = str(combo.currentText()).strip()
        old_type = db.types.get(component_name, "Accessory")
        if new_type not in TYPE_OPTIONS or new_type == old_type:
            return

        scope = self._ask_component_change_scope(
            {}, [(component_name, component_name, new_type)]
        )
        if scope == "cancel":
            combo.blockSignals(True)
            combo.setCurrentText(old_type)
            combo.blockSignals(False)
            return

        targets = list(self.sample_dbs.values()) if scope == "all" else [db]
        try:
            self._component_scope_change_in_progress = True
            for target_db in targets:
                # Apply to the same named component wherever it exists. Samples
                # without that component are left untouched.
                if component_name in target_db.components:
                    target_db.set_component_type(component_name, new_type)
        except Exception as e:
            combo.blockSignals(True)
            combo.setCurrentText(old_type)
            combo.blockSignals(False)
            QMessageBox.warning(self, "Component type", str(e))
            return
        finally:
            self._component_scope_change_in_progress = False

        self._finish_scoped_component_metadata_change()

    def pick_color_for_row(self, row):
        current = self._get_color_cell(row)
        col = QColorDialog.getColor(QColor(current), self, "Choose component color")
        if not col.isValid():
            return
        self._set_color_cell(row, col.name())
        self._ensure_palette_widget(row, col.name())

    def refresh_component_table_from_global(self):
        """Refresh the editor from the current sample.

        The historical method name is retained for plugin/backward compatibility.
        Global component dictionaries are now an aggregate/index; the editable
        definition is the ComponentryDB belonging to the selected sample.
        """
        db = self.current_db()
        if db is not None:
            components = list(db.components)
            colors_by_name = dict(db.colors)
            types_by_name = dict(db.types)
        else:
            components = list(self.global_components)
            colors_by_name = dict(self.global_colors)
            types_by_name = dict(self.global_types)

        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(components))

        palette_colors = self._current_palette_colors(max(1, len(components)))
        for i, comp in enumerate(components):
            it_name = QTableWidgetItem(comp)
            # Keep row identity even if its visible name is edited or rows move.
            it_name.setData(Qt.UserRole, comp)
            self.tbl.setItem(i, 0, it_name)

            color = colors_by_name.get(comp, palette_colors[i % len(palette_colors)])
            self._set_color_cell(i, color)
            self._ensure_palette_widget(i, color)
            self._ensure_type_widget(i, types_by_name.get(comp, "Accessory"))

        self.tbl.blockSignals(False)

        try:
            self.tbl.resizeRowsToContents()
        except Exception:
            pass

        self.btn_load_fraction.setEnabled((db is not None) and bool(db.components if db is not None else []))
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
        self._refresh_app_state()
        self._update_palette_preview()

    def _component_table_rows(self):
        """Return the complete editable row state in its current visual order."""
        rows = []
        for row in range(self.tbl.rowCount()):
            it_name = self.tbl.item(row, 0)
            name = it_name.text() if it_name is not None else ""
            original_name = it_name.data(Qt.UserRole) if it_name is not None else None
            original_name = str(original_name).strip() if original_name not in (None, "") else None
            rows.append({
                "name": name,
                "original_name": original_name,
                "color": self._get_color_cell(row),
                "type": self._get_type_value(row),
            })
        return rows

    def _rebuild_component_table_rows(self, rows, selected_row=None):
        """Rebuild rows after a move while preserving name/color/type together."""
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(rows))
        for row, payload in enumerate(rows):
            it_name = QTableWidgetItem(str(payload.get("name", "")))
            original_name = payload.get("original_name", None)
            if original_name not in (None, ""):
                it_name.setData(Qt.UserRole, str(original_name))
            self.tbl.setItem(row, 0, it_name)
            color = _normalize_hex_color(
                payload.get("color"),
                self._palette_color_for_index(row, max(1, len(rows))),
            )
            self._set_color_cell(row, color)
            self._ensure_palette_widget(row, color)
            self._ensure_type_widget(row, payload.get("type", "Accessory"))
        self.tbl.blockSignals(False)

        try:
            self.tbl.resizeRowsToContents()
        except Exception:
            pass

        if selected_row is not None and self.tbl.rowCount() > 0:
            selected_row = max(0, min(int(selected_row), self.tbl.rowCount() - 1))
            self.tbl.setCurrentCell(selected_row, 0)
            self.tbl.selectRow(selected_row)

        self._refresh_palette_color_widgets()

    def _component_definitions_uniform(self):
        """True when every sample has the same ordered component names and types."""
        if not self.sample_dbs:
            return True
        sigs = []
        for db in self.sample_dbs.values():
            names = tuple(db.components)
            types = tuple((c, db.types.get(c, "Accessory")) for c in db.components)
            sigs.append((names, types))
        return all(sig == sigs[0] for sig in sigs[1:])

    def _commit_component_order_if_safe(self):
        """Apply a pure reorder immediately without converting it into a rename."""
        db = self.current_db()
        if db is None:
            return False

        rows = self._component_table_rows()
        names = [str(row["name"]).strip() for row in rows]
        if (
            len(names) != len(db.components)
            or any(not name for name in names)
            or len(set(names)) != len(names)
            or set(names) != set(db.components)
        ):
            return False

        uniform_before = self._component_definitions_uniform()
        targets = list(self.sample_dbs.values()) if uniform_before else [db]
        for target_db in targets:
            if set(target_db.components) != set(names):
                continue
            target_db.set_components(names, dict(target_db.colors), dict(target_db.types))

        self._sync_global_from_samples()
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
        self.plot_histogram()
        self._notify_host_componentry_changed()
        return True

    def _move_component_row(self, source_row, target_row):
        """Move one complete component row with rollback on any unexpected error.

        Reordering never removes a component from the data model.  We first take
        a complete snapshot of every editable row, reorder that in memory, and
        only then rebuild the table.  If anything fails, the original snapshot is
        restored immediately.
        """
        n = self.tbl.rowCount()
        try:
            source_row = int(source_row)
            target_row = int(target_row)
        except Exception:
            return

        if n <= 1 or source_row < 0 or source_row >= n:
            return
        target_row = max(0, min(target_row, n - 1))
        if target_row == source_row:
            return

        original_rows = self._component_table_rows()
        if len(original_rows) != n:
            return

        try:
            reordered_rows = [dict(row) for row in original_rows]
            payload = reordered_rows.pop(source_row)
            reordered_rows.insert(target_row, payload)

            # Hard invariant: a move must preserve the exact number of rows.
            if len(reordered_rows) != len(original_rows):
                raise RuntimeError("Component row count changed during reorder.")

            self._rebuild_component_table_rows(reordered_rows, selected_row=target_row)

            # Verify that every component-name cell survived the rebuild.  Empty
            # names are allowed while editing, but the sequence itself must match
            # the in-memory reordered snapshot exactly.
            rebuilt_rows = self._component_table_rows()
            expected_names = [str(row.get("name", "")) for row in reordered_rows]
            rebuilt_names = [str(row.get("name", "")) for row in rebuilt_rows]
            if len(rebuilt_rows) != len(reordered_rows) or rebuilt_names != expected_names:
                raise RuntimeError("Component table rebuild did not preserve all rows.")

            self._commit_component_order_if_safe()

        except Exception:
            # Fail closed: restore exactly what existed before the move.  This
            # prevents a UI gesture from ever costing component definitions.
            try:
                self._rebuild_component_table_rows(original_rows, selected_row=source_row)
            except Exception:
                pass
            QMessageBox.warning(
                self,
                "Component order",
                "The component could not be moved safely. The original order was restored."
            )

    def _move_selected_component(self, direction):
        row = self._get_selected_row()
        if row < 0:
            return
        target = row + int(direction)
        if target < 0 or target >= self.tbl.rowCount():
            return
        self._move_component_row(row, target)

    # ---------- buttons
    def add_component_row(self):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)

        it_name = QTableWidgetItem("")
        it_name.setData(Qt.UserRole, None)
        self.tbl.setItem(r, 0, it_name)

        color = self._palette_color_for_index(r, r + 1)
        self._set_color_cell(r, color)
        self._ensure_palette_widget(r, color)
        self._ensure_type_widget(r, "Accessory")

        try:
            self.tbl.setCurrentCell(r, 0)
            self.tbl.editItem(it_name)
        except Exception:
            pass

    def remove_component_row(self):
        r = self._get_selected_row()
        if r < 0:
            return
        self.tbl.removeRow(r)
        self._refresh_palette_color_widgets()

    def _rename_map_for_rows(self, db, rows):
        out = {}
        valid_old = set(db.components)
        for row in rows:
            old = row.get("original_name")
            new = str(row.get("name", "")).strip()
            if old in valid_old and old != new:
                out[str(old)] = new
        return out

    def _validate_rename_map_for_db(self, db, rename_map):
        sources = {old for old in rename_map if old in db.components}
        for old, new in rename_map.items():
            if old not in db.components or old == new:
                continue
            if new in db.components and new not in sources:
                raise ValueError(
                    f"Cannot rename '{old}' to '{new}' in this sample because '{new}' already exists."
                )

    def _apply_rename_map_to_db(self, db, rename_map):
        """Apply possibly multiple/swap renames safely, preserving data/color/type."""
        self._validate_rename_map_for_db(db, rename_map)
        applicable = [(old, new) for old, new in rename_map.items() if old in db.components and old != new]
        if not applicable:
            return

        used = set(db.components) | set(rename_map.values())
        temp_by_old = {}
        for idx, (old, _new) in enumerate(applicable):
            temp = f"__componentry_tmp_rename_{idx}__"
            k = 2
            while temp in used:
                temp = f"__componentry_tmp_rename_{idx}_{k}__"
                k += 1
            used.add(temp)
            temp_by_old[old] = temp
            db.rename_component(old, temp)

        for old, new in applicable:
            db.rename_component(temp_by_old[old], new)

    def _apply_rows_to_db(self, db, rows, preserve_unedited_colors=False,
                          preserve_unedited_types=False, explicit_color_old=None,
                          explicit_type_old=None):
        """Apply the current editor rows to one ComponentryDB without losing component data."""
        explicit_color_old = set(explicit_color_old or set())
        explicit_type_old = set(explicit_type_old or set())
        final_names = [str(row.get("name", "")).strip() for row in rows]
        rename_map = self._rename_map_for_rows(db, rows)

        pre_colors = dict(db.colors)
        pre_types = dict(db.types)
        self._apply_rename_map_to_db(db, rename_map)

        # Components deleted from the table are removed explicitly so the existing
        # renormalization behavior of remove_component() is preserved.
        for comp in list(db.components):
            if comp not in final_names:
                db.remove_component(comp)

        colors = {}
        types = {}
        for row in rows:
            new = str(row.get("name", "")).strip()
            old = row.get("original_name")
            old = str(old).strip() if old not in (None, "") else None

            if preserve_unedited_colors and old and old not in explicit_color_old:
                colors[new] = pre_colors.get(old, pre_colors.get(new, row.get("color")))
            else:
                colors[new] = row.get("color")

            if preserve_unedited_types and old and old not in explicit_type_old:
                t = pre_types.get(old, pre_types.get(new, row.get("type", "Accessory")))
                types[new] = t if t in TYPE_OPTIONS else "Accessory"
            else:
                t = row.get("type", "Accessory")
                types[new] = t if t in TYPE_OPTIONS else "Accessory"

        db.set_components(final_names, colors, types)

    def _ask_component_change_scope(self, rename_map, type_changes):
        sample_name = self.current_sample_name() or "current sample"
        lines = []
        for old, new in rename_map.items():
            lines.append(f"Name: {old}  →  {new}")
        for old, new_name, new_type in type_changes:
            label = new_name if new_name else old
            lines.append(f"Type: {label}  →  {new_type}")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Apply component change")
        box.setText("A component name or type was changed.")
        box.setInformativeText(
            "Apply this change only to the current sample or to all samples?\n\n"
            f"Current sample: {sample_name}"
            + ("\n\n" + "\n".join(lines) if lines else "")
            + "\n\nExisting component colors will be preserved unless you explicitly edited a color."
        )
        btn_current = box.addButton("Current sample only", QMessageBox.AcceptRole)
        btn_all = box.addButton("All samples", QMessageBox.ActionRole)
        btn_cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(btn_current)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_all:
            return "all"
        if clicked is btn_current:
            return "current"
        return "cancel"

    def apply_changes(self):
        if not self.sample_dbs:
            QMessageBox.warning(self, "Error", "Load an Excel file or open a project first.")
            return

        db = self.current_db()
        if db is None:
            return

        rows = self._component_table_rows()
        if not rows:
            QMessageBox.warning(self, "Error", "Add at least one component.")
            return

        names = [str(row.get("name", "")).strip() for row in rows]
        if any(not name for name in names):
            QMessageBox.warning(self, "Error", "Component names cannot be empty.")
            return
        if len(set(names)) != len(names):
            QMessageBox.warning(self, "Error", "Component names must be unique.")
            return

        uniform_before = self._component_definitions_uniform()
        rename_map = self._rename_map_for_rows(db, rows)

        type_changes = []
        explicit_type_old = set()
        explicit_color_old = set()
        for row in rows:
            old = row.get("original_name")
            if old not in db.components:
                continue
            new = str(row.get("name", "")).strip()
            new_type = row.get("type", "Accessory")
            old_type = db.types.get(old, "Accessory")
            if new_type != old_type:
                type_changes.append((old, new, new_type))
                explicit_type_old.add(old)

            old_color = _normalize_hex_color(db.colors.get(old, "#888888"))
            new_color = _normalize_hex_color(row.get("color", old_color))
            if new_color.lower() != old_color.lower():
                explicit_color_old.add(old)

        scope = None
        if rename_map or type_changes:
            scope = self._ask_component_change_scope(rename_map, type_changes)
            if scope == "cancel":
                return

        # Confirm destructive removals in the sample(s) that would actually lose data.
        original_in_rows = {row.get("original_name") for row in rows if row.get("original_name")}
        removed_current = [c for c in db.components if c not in original_in_rows]
        removal_is_global = (scope == "all") or (scope is None and uniform_before)
        removed_with_data = []
        for c in removed_current:
            if removal_is_global:
                if any(target_db.has_data_for_component(c) for target_db in self.sample_dbs.values()):
                    removed_with_data.append(c)
            elif db.has_data_for_component(c):
                removed_with_data.append(c)

        if removed_with_data:
            where_txt = "one or more samples" if removal_is_global else "the current sample"
            msg = (
                f"You are removing component(s) that already have data in {where_txt}:\n\n"
                + "\n".join([f" - {c}" for c in removed_with_data])
                + "\n\nRemoving them will renormalize remaining components in affected fractions.\n\nContinue?"
            )
            ans = QMessageBox.question(
                self, "Confirm removal", msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if ans != QMessageBox.Yes:
                return


        try:
            if scope == "all":
                # Validate every rename before changing anything, preventing partial updates.
                for target_db in self.sample_dbs.values():
                    self._validate_rename_map_for_db(target_db, rename_map)

                if uniform_before:
                    # Normal legacy/global case: synchronize the full definition, while
                    # retaining each sample's existing colors/types unless the user
                    # explicitly changed those fields.
                    for target_db in self.sample_dbs.values():
                        self._apply_rows_to_db(
                            target_db, rows,
                            preserve_unedited_colors=True,
                            preserve_unedited_types=True,
                            explicit_color_old=explicit_color_old,
                            explicit_type_old=explicit_type_old,
                        )
                else:
                    # Samples already differ.  Propagate only the requested metadata
                    # corrections, never overwrite unrelated local components/colors.
                    for target_db in self.sample_dbs.values():
                        self._apply_rename_map_to_db(target_db, rename_map)
                        for old, new, new_type in type_changes:
                            target = new if new in target_db.components else old
                            if target in target_db.components:
                                target_db.set_component_type(target, new_type)
                        for row in rows:
                            old = row.get("original_name")
                            if old not in explicit_color_old:
                                continue
                            new = str(row.get("name", "")).strip()
                            target = new if new in target_db.components else old
                            if target in target_db.components:
                                target_db.colors[target] = _normalize_hex_color(row.get("color"))

                    # Add/remove/order edits made at the same time remain local when
                    # samples were already intentionally different.
                    self._apply_rows_to_db(db, rows)

            elif scope == "current":
                self._apply_rows_to_db(db, rows)

            else:
                # No rename/type change.  Preserve the historical all-sample behavior
                # while definitions are uniform.  Once samples intentionally differ,
                # unrelated edits stay local so they cannot erase those differences.
                if uniform_before:
                    for target_db in self.sample_dbs.values():
                        self._apply_rows_to_db(target_db, rows)
                else:
                    self._apply_rows_to_db(db, rows)

        except Exception as e:
            QMessageBox.warning(self, "Apply changes error", str(e))
            return

        self._sync_global_from_samples()
        self.refresh_component_table_from_global()
        self.update_summary()
        self.plot_histogram()
        self._notify_host_componentry_changed()

    def _sync_histogram_preview_geometry(self, style=None):
        """Scale the WYSIWYG preview to the available viewport.

        Figure width/height remain the export dimensions in inches. Only preview
        DPI changes, so the entire page is uniformly zoomed without changing any
        relative size, margin, font, legend position or bar geometry.
        """
        if not hasattr(self, "canvas") or self.canvas is None:
            return
        style = normalize_componentry_hist_style(style or self.componentry_hist_style)
        width_in = max(1.0, float(style["figure_width"]))
        height_in = max(1.0, float(style["figure_height"]))

        fallback_dpi = float(getattr(self, "_componentry_preview_dpi", 90.0) or 90.0)
        dpi = fallback_dpi
        scroll = getattr(self, "plot_scroll", None)
        if scroll is not None:
            try:
                vp = scroll.viewport()
                avail_w = max(1, int(vp.width()) - 24)
                avail_h = max(1, int(vp.height()) - 24)
                if avail_w > 200 and avail_h > 180:
                    dpi = min(avail_w / width_in, avail_h / height_in)
            except Exception:
                dpi = fallback_dpi

        # Fit to the available area when possible; retain scrolling only for very
        # small windows. A generous upper cap avoids needless redraw cost.
        dpi = max(55.0, min(170.0, float(dpi)))
        self._componentry_preview_dpi = dpi

        fig = self.canvas.figure
        fig.set_dpi(dpi)
        fig.set_size_inches(width_in, height_in, forward=False)

        width_px = max(320, int(round(width_in * dpi)))
        height_px = max(240, int(round(height_in * dpi)))
        self.canvas.setFixedSize(width_px, height_px)

    def open_componentry_histogram_style_dialog(self):
        dlg = ComponentryHistogramStyleDialog(self.componentry_hist_style, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.componentry_hist_style = normalize_componentry_hist_style(dlg.style)
        self.plot_histogram()
        self._notify_host_componentry_changed()

    # =========================================================
    # HISTOGRAM FILTER
    # =========================================================
    def _on_hist_filter_toggled(self, checked):
        self._refresh_hist_filter_ui_state()
        self.plot_histogram()

    def _on_hist_filter_mode_changed(self):
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
        self.plot_histogram()

    def _refresh_hist_filter_value_options(self):
        if not hasattr(self, "cmb_hist_filter_mode") or not hasattr(self, "cmb_hist_filter_value"):
            return

        prev = self.cmb_hist_filter_value.currentText().strip()

        self.cmb_hist_filter_value.blockSignals(True)
        self.cmb_hist_filter_value.clear()

        mode_txt = self.cmb_hist_filter_mode.currentText().strip().lower()

        if "type" in mode_txt:
            for t in TYPE_OPTIONS:
                self.cmb_hist_filter_value.addItem(t)
        else:
            db = self.current_db()
            components = list(db.components) if db is not None else list(self.global_components)
            for c in components:
                self.cmb_hist_filter_value.addItem(c)

        if prev:
            idx = self.cmb_hist_filter_value.findText(prev)
            if idx >= 0:
                self.cmb_hist_filter_value.setCurrentIndex(idx)

        self.cmb_hist_filter_value.blockSignals(False)

    def _refresh_hist_filter_ui_state(self):
        has_samples = bool(getattr(self, "sample_dbs", {}))
        enabled = has_samples and bool(self.chk_hist_filter.isChecked()) if hasattr(self, "chk_hist_filter") else False

        self.cmb_hist_filter_mode.setEnabled(enabled)
        self.cmb_hist_filter_value.setEnabled(enabled)
        self.chk_hist_include_undef.setEnabled(enabled)

        # si el filtro está activo, el modo "by type" general no aplica visualmente
        self.chk_type.setEnabled(has_samples and (not enabled))

    def _current_hist_filter_spec(self):
        if not hasattr(self, "chk_hist_filter"):
            return {"enabled": False, "mode": None, "value": None, "include_undefined": False}

        enabled = bool(self.chk_hist_filter.isChecked())
        if not enabled:
            return {"enabled": False, "mode": None, "value": None, "include_undefined": False}

        mode_txt = self.cmb_hist_filter_mode.currentText().strip().lower()
        mode_key = "type" if "type" in mode_txt else "component"
        value = self.cmb_hist_filter_value.currentText().strip()
        include_undef = bool(self.chk_hist_include_undef.isChecked())

        if not value:
            return {"enabled": False, "mode": None, "value": None, "include_undefined": include_undef}

        return {
            "enabled": True,
            "mode": mode_key,
            "value": value,
            "include_undefined": include_undef,
        }

    def _filtered_hist_data_for_db(self, db: ComponentryDB):
        """
        Retorna:
          x, y_sel_abs, y_undef_abs, y_total_abs, color_sel, label_sel, filter_desc, include_undef
        """
        spec = self._current_hist_filter_spec()
        if not spec["enabled"]:
            return None

        dfp = db.gsd[db.gsd["wt"].astype(float) > 0].copy().sort_values("phi")
        if dfp.empty:
            return None

        x = dfp["phi"].astype(float).values
        y_sel_abs = np.zeros_like(x, dtype=float)
        y_undef_abs = np.zeros_like(x, dtype=float)

        if spec["mode"] == "component":
            comp = str(spec["value"])
            for i, phi in enumerate(x):
                if (comp in db.total.columns) and (float(phi) in db.total.index):
                    vv = _to_float(db.total.loc[float(phi), comp])
                    if np.isfinite(vv) and vv > 0:
                        y_sel_abs[i] = float(vv)
            color_sel = db.colors.get(comp, "#808080")
            label_sel = comp
            filter_desc = f"Component: {comp}"

        else:
            tsel = str(spec["value"])
            for i, phi in enumerate(x):
                s = 0.0
                for c in db.components:
                    t = db.types.get(c, "Accessory")
                    if t != tsel:
                        continue
                    if (c in db.total.columns) and (float(phi) in db.total.index):
                        vv = _to_float(db.total.loc[float(phi), c])
                        if np.isfinite(vv) and vv > 0:
                            s += float(vv)
                y_sel_abs[i] = s
            color_sel = TYPE_COLORS.get(tsel, "#808080")
            label_sel = tsel
            filter_desc = f"Type: {tsel}"

        if bool(spec["include_undefined"]):
            wt_map = db.wt_map()
            for i, phi in enumerate(x):
                bin_wt = float(wt_map.get(float(phi), 0.0))
                if float(phi) not in db.filled:
                    y_undef_abs[i] = max(0.0, bin_wt)
                else:
                    known = 0.0
                    if float(phi) in db.total.index:
                        for c in db.total.columns:
                            vv = _to_float(db.total.loc[float(phi), c])
                            if np.isfinite(vv) and vv > 0:
                                known += float(vv)
                    y_undef_abs[i] = max(0.0, bin_wt - known)

        y_total_abs = y_sel_abs + y_undef_abs

        return {
            "x": x,
            "y_sel_abs": y_sel_abs,
            "y_undef_abs": y_undef_abs,
            "y_total_abs": y_total_abs,
            "color_sel": color_sel,
            "label_sel": label_sel,
            "filter_desc": filter_desc,
            "include_undef": bool(spec["include_undefined"]),
        }

    # =========================================================
    # SUMMARY
    # =========================================================
    def _position_summary_overlay(self):
        box = getattr(self, "summary_box", None)
        scroll = getattr(self, "plot_scroll", None)
        if box is None or scroll is None:
            return
        vp = scroll.viewport()
        try:
            box.adjustSize()
            hint = box.sizeHint()
            w = min(max(260, hint.width()), max(260, vp.width() - 20))
            h = min(30, max(24, hint.height()))
            box.resize(w, h)
            box.move(max(8, vp.width() - w - 10), 8)
            box.raise_()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        try:
            if (hasattr(self, "plot_scroll") and obj is self.plot_scroll.viewport()
                    and event.type() in (QEvent.Resize, QEvent.Show)):
                QTimer.singleShot(0, self._sync_histogram_preview_geometry)
                QTimer.singleShot(0, self._position_summary_overlay)
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def update_summary(self):
        db = self.current_db()
        if db is None:
            for k in self.sum_labels:
                self.sum_labels[k].setText("—")
            return

        gsd = db.gsd.copy()
        dfp = gsd[gsd["wt"].astype(float) > 0].sort_values("phi").reset_index(drop=True)

        total = float(dfp["wt"].sum()) if not dfp.empty else 0.0
        phi_min = float(dfp["phi"].min()) if not dfp.empty else np.nan
        phi_max = float(dfp["phi"].max()) if not dfp.empty else np.nan

        w = dfp["wt"].astype(float).values
        x = dfp["phi"].astype(float).values
        mean_phi = float(np.sum(x * w) / np.sum(w)) if np.sum(w) > 0 else np.nan

        self.sum_labels["sample"].setText(self.current_sample_name())
        self.sum_labels["fractions"].setText(str(len(dfp)))
        self.sum_labels["sumwt"].setText(f"{total:.2f}%")
        self.sum_labels["phirange"].setText(f"[{phi_min:.2f}, {phi_max:.2f}]")
        self.sum_labels["meanphi"].setText(f"{mean_phi:.2f}")
        QTimer.singleShot(0, self._position_summary_overlay)

    # =========================================================
    # FOLK & WARD (plot helpers)
    # =========================================================
    def _plot_x_from_phi(self, phi_value, x_mode):
        return phi_to_um(phi_value) if str(x_mode).lower() == "um" else float(phi_value)

    def _draw_fw_markers(self, ax, fw, x_mode="phi"):
        if not fw:
            return
        style = normalize_componentry_hist_style(self.componentry_hist_style)
        x16 = self._plot_x_from_phi(fw["phi16"], x_mode)
        x50 = self._plot_x_from_phi(fw["phi50"], x_mode)
        x84 = self._plot_x_from_phi(fw["phi84"], x_mode)
        try:
            ax.axvspan(
                min(x16, x84), max(x16, x84),
                color=style["fw_band_color"], alpha=style["fw_band_alpha"], zorder=1
            )
        except Exception:
            pass
        ax.axvline(
            x50, color=style["fw_median_color"], linestyle="--",
            linewidth=style["fw_median_linewidth"], alpha=style["fw_line_alpha"], zorder=7
        )
        ax.axvline(
            x16, color=style["fw_outer_color"], linestyle=":",
            linewidth=style["fw_outer_linewidth"], alpha=style["fw_line_alpha"], zorder=7
        )
        ax.axvline(
            x84, color=style["fw_outer_color"], linestyle=":",
            linewidth=style["fw_outer_linewidth"], alpha=style["fw_line_alpha"], zorder=7
        )

    def _draw_fw_text(self, ax, fw, x_vals, y_vals):
        if not fw:
            return
        style = normalize_componentry_hist_style(self.componentry_hist_style)
        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & (y >= 0)

        if np.count_nonzero(m) == 0:
            x_anchor = 0.02
            ha = "left"
        else:
            x = x[m]
            y = y[m]
            ysum = float(np.sum(y))
            if ysum <= 0:
                x_anchor = 0.02
                ha = "left"
            else:
                com = float(np.sum(x * y) / ysum)
                mid = 0.5 * (float(np.min(x)) + float(np.max(x)))
                if com >= mid:
                    x_anchor = 0.02
                    ha = "left"
                else:
                    x_anchor = 0.98
                    ha = "right"

        txt = (
            rf"$M_z$ = {fw['Mz']:.3f}" "\n"
            rf"$\sigma_i$ = {fw['sigma_i']:.3f}" "\n"
            rf"$SK_i$ = {fw['SK_i']:.3f}" "\n"
            rf"$K_g$ = {fw['Kg']:.3f}"
        )

        ax.text(
            x_anchor, 0.98, txt,
            transform=ax.transAxes,
            ha=ha, va="top",
            fontsize=style["fw_text_fontsize"],
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=style["fw_box_facecolor"],
                edgecolor=style["fw_box_edgecolor"],
                linewidth=style["fw_box_linewidth"],
                alpha=style["fw_box_alpha"],
            ),
            zorder=12
        )

    # =========================================================
    # PLOTTING
    # =========================================================
    def _draw_histogram(self, ax, db: ComponentryDB, sample_title: str, bars_100: bool, by_type: bool,
                        show_fw: bool = False, mark_fw: bool = False):
        style = normalize_componentry_hist_style(self.componentry_hist_style)
        self.componentry_hist_style = dict(style)
        fig = ax.figure
        fig.patch.set_facecolor("white")
        ax.set_facecolor("none")

        dfp = db.gsd.copy()
        dfp["phi"] = pd.to_numeric(dfp["phi"], errors="coerce")
        dfp["wt"] = pd.to_numeric(dfp["wt"], errors="coerce")
        dfp = dfp[np.isfinite(dfp["phi"]) & np.isfinite(dfp["wt"])].copy().sort_values("phi").reset_index(drop=True)

        # Keep zero bins only when they lie BETWEEN the first and last non-zero
        # fractions. Database padding outside the real sample interval is not
        # part of the histogram and must not be rendered.
        _positive_idx = np.flatnonzero(dfp["wt"].to_numpy(dtype=float) > 0.0)
        if _positive_idx.size:
            dfp = dfp.iloc[int(_positive_idx[0]):int(_positive_idx[-1]) + 1].copy().reset_index(drop=True)
        else:
            dfp = dfp.iloc[0:0].copy()

        if dfp.empty:
            ax.set_title(
                sample_title,
                fontfamily="Arial",
                fontsize=style["title_fontsize"],
                fontweight=style["title_fontweight"],
            )
            ax.text(
                0.5, 0.5, "No granulometric data",
                transform=ax.transAxes, ha="center", va="center",
                fontfamily="Arial", fontsize=style["axis_label_fontsize"]
            )
            fig.subplots_adjust(left=0.10, right=0.95, bottom=0.18, top=0.90)
            return

        phi_values = dfp["phi"].astype(float).values
        y_gsd = dfp["wt"].astype(float).values
        x_mode = str(style["x_axis_mode"]).lower()

        if x_mode == "um":
            x_plot, bar_widths = _componentry_um_bar_geometry(phi_values, style["bar_width_factor"])
            tick_pos = _sparse_values(np.sort(np.unique(x_plot)), max_count=12)
            tick_labels = [f"{v:.0f}" if abs(v) >= 10 else f"{v:.2g}" for v in tick_pos]
            x_label = style["x_label_um"].strip() or "Particle size (μm)"
        else:
            ticks_phi = _componentry_tick_positions(phi_values)
            tick_pos = ticks_phi if ticks_phi.size else np.unique(phi_values)
            tick_labels = [f"{v:g}" for v in tick_pos]
            uniq = np.unique(phi_values)
            spacing = float(np.min(np.diff(uniq))) if uniq.size > 1 else 1.0
            x_plot = phi_values.copy()
            bar_widths = np.full_like(x_plot, max(0.01, style["bar_width_factor"] * spacing), dtype=float)
            x_label = style["x_label_phi"].strip() or "PHI (Φ)"

        coord_by_phi = {float(p): float(x) for p, x in zip(phi_values, x_plot)}
        width_by_phi = {float(p): float(w) for p, w in zip(phi_values, bar_widths)}

        if style["cumulative_layer"] == "back":
            ax2 = ax.twinx()
            ax2.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_alpha(0.0)
        else:
            ax2 = ax.twinx()
            ax.set_zorder(0)
            ax2.set_zorder(1)
            ax2.patch.set_alpha(0.0)
        ax2.set_facecolor("none")
        ax2.xaxis.set_visible(False)
        ax2.tick_params(axis="x", which="both", bottom=False, top=False,
                        labelbottom=False, labeltop=False)

        legend_handles = []
        legend_labels = []
        fw_x = np.array(phi_values, dtype=float)
        fw_y = y_gsd.copy()
        polygon_y = y_gsd.copy()
        full_title = sample_title

        segment_edge = style["segment_edgecolor"] if style["segment_edgewidth"] > 0 else "none"
        segment_lw = style["segment_edgewidth"]
        outer_edge = style["bar_outer_edgecolor"]
        outer_lw = style["bar_outer_edgewidth"]
        alpha = style["bar_alpha"]

        def draw_segment(phi, height, bottom, color):
            if height <= 0:
                return
            ax.bar(
                coord_by_phi[float(phi)], height,
                width=width_by_phi[float(phi)], bottom=bottom,
                color=color, edgecolor=segment_edge, linewidth=segment_lw,
                alpha=alpha, align="center", zorder=2,
            )

        def draw_outline(phi, height):
            if height <= 0 or outer_lw <= 0:
                return
            ax.bar(
                coord_by_phi[float(phi)], height,
                width=width_by_phi[float(phi)], fill=False,
                edgecolor=outer_edge, linewidth=outer_lw,
                align="center", zorder=4,
            )

        filter_spec = self._current_hist_filter_spec()
        filter_active = bool(filter_spec.get("enabled", False))

        if filter_active:
            fdat = self._filtered_hist_data_for_db(db)
            if fdat is None:
                ax.text(0.5, 0.5, "No filtered histogram data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=style["axis_label_fontsize"])
                ax2.set_axis_off()
                return

            x_phi = np.array(fdat["x"], dtype=float)
            y_sel_abs = np.array(fdat["y_sel_abs"], dtype=float)
            y_undef_abs = np.array(fdat["y_undef_abs"], dtype=float)
            y_total_abs = np.array(fdat["y_total_abs"], dtype=float)
            fw_x = x_phi.copy()
            fw_y = y_total_abs.copy()

            if bars_100:
                total = float(np.sum(y_total_abs))
                scale = 100.0 / total if total > 0 else 0.0
                y_sel_plot = y_sel_abs * scale
                y_undef_plot = y_undef_abs * scale
                y_total_plot = y_total_abs * scale
                default_y_label = "Vol. %"
                base_top = 100.0
            else:
                y_sel_plot = y_sel_abs
                y_undef_plot = y_undef_abs
                y_total_plot = y_total_abs
                default_y_label = "%Wt"
                base_top = float(np.nanmax(y_total_plot)) if y_total_plot.size else 0.0

            polygon_y = y_total_plot.copy()
            for phi, h1, h2 in zip(x_phi, y_sel_plot, y_undef_plot):
                draw_segment(phi, float(h1), 0.0, fdat["color_sel"])
                draw_segment(phi, float(h2), float(h1), style["undefined_facecolor"])
                draw_outline(phi, float(h1 + h2))

            from matplotlib.patches import Patch
            legend_handles.append(Patch(facecolor=fdat["color_sel"], edgecolor="none", alpha=alpha))
            legend_labels.append(fdat["label_sel"])
            if np.any(y_undef_plot > 0):
                legend_handles.append(Patch(facecolor=style["undefined_facecolor"], edgecolor="none", alpha=alpha))
                legend_labels.append("Undefined")

            full_title = sample_title + " | " + fdat["filter_desc"]
            if fdat["include_undef"]:
                full_title += " + Undefined"

        else:
            if by_type:
                df_total_type = db.totals_by_type()
                df_100_type = db.frac100_by_type()
                legend_labels_base = TYPE_OPTIONS
                legend_colors = {t: TYPE_COLORS[t] for t in TYPE_OPTIONS}
            else:
                df_total_type = None
                df_100_type = None
                legend_labels_base = list(db.components)
                legend_colors = dict(db.colors)

            polygon_y = np.where(y_gsd > 0.0, 100.0, 0.0) if bars_100 else y_gsd.copy()

            for phi, wt in zip(phi_values, y_gsd):
                target_height = (100.0 if float(wt) > 0.0 else 0.0) if bars_100 else float(wt)
                if phi in db.filled:
                    bottom = 0.0
                    if by_type:
                        source = df_100_type if bars_100 else df_total_type
                        for t in TYPE_OPTIONS:
                            h = float(source.loc[phi, t]) if source is not None and phi in source.index else 0.0
                            draw_segment(phi, h, bottom, TYPE_COLORS[t])
                            bottom += max(0.0, h)
                    else:
                        source = db.frac100 if bars_100 else db.total
                        # Component order follows the editor table literally:
                        # lower rows are lower in the stacked bar; upper rows are upper.
                        for comp in reversed(db.components):
                            h = float(source.loc[phi, comp]) if comp in source.columns and phi in source.index else 0.0
                            draw_segment(phi, h, bottom, db.colors.get(comp, "#888888"))
                            bottom += max(0.0, h)
                    draw_outline(phi, target_height)
                else:
                    ax.bar(
                        coord_by_phi[float(phi)], target_height,
                        width=width_by_phi[float(phi)],
                        color=style["unclassified_facecolor"],
                        edgecolor=outer_edge if outer_lw > 0 else "none",
                        linewidth=outer_lw,
                        alpha=alpha, align="center", zorder=2,
                    )

            default_y_label = "Vol. %" if bars_100 else "%Wt"
            base_top = 100.0 if bars_100 else (float(np.nanmax(y_gsd)) if y_gsd.size else 0.0)

            from matplotlib.patches import Patch
            for label in legend_labels_base:
                color = legend_colors.get(label)
                if color:
                    legend_handles.append(Patch(facecolor=color, edgecolor="none", alpha=alpha))
                    legend_labels.append(label)

        # Frequency polygon follows the displayed bar tops.
        if style["show_polygon"]:
            marker = None if style["poly_marker"] == "None" else style["poly_marker"]
            ax.plot(
                x_plot, polygon_y,
                color=style["poly_color"], linewidth=style["poly_linewidth"],
                linestyle=style["poly_linestyle"], marker=marker,
                markersize=style["poly_markersize"], zorder=6,
            )

        # Cumulative curve uses the effective absolute histogram.
        y_right_mode = style["y_right_mode"]
        show_cum = style["show_cumulative"] and y_right_mode == "cumulative"
        if show_cum:
            total = float(np.nansum(np.clip(fw_y, 0.0, None)))
            cumulative = 100.0 * np.cumsum(np.clip(fw_y, 0.0, None)) / total if total > 0 else np.full_like(fw_y, np.nan)
            cum_x = np.array([self._plot_x_from_phi(p, x_mode) for p in fw_x], dtype=float)
            marker = None if style["cum_marker"] == "None" else style["cum_marker"]
            ax2.plot(
                cum_x, cumulative,
                color=style["cum_color"], linewidth=style["cum_linewidth"],
                linestyle=style["cum_linestyle"], marker=marker,
                markersize=style["cum_markersize"],
                markerfacecolor=style["cum_marker_facecolor"],
                markeredgecolor=style["cum_marker_edgecolor"],
                markeredgewidth=style["cum_marker_edgewidth"],
                zorder=1 if style["cumulative_layer"] == "back" else 8,
            )
            ax2.set_ylim(0, 102)
            ax2.set_yticks(np.arange(0, 101, 20))
            right_label = style["y_label_right"].strip() or "%Wt cumulative frequency"
            ax2.set_ylabel(
                right_label,
                fontsize=style["right_axis_label_fontsize"],
                fontweight=style["axis_label_fontweight"],
                fontfamily="Arial",
            )
            ax2.yaxis.set_visible(True)
        else:
            ax2.set_yticks([])
            ax2.set_ylabel("")
            ax2.yaxis.set_visible(False)
            ax2.spines["right"].set_visible(False)

        top_pad = 1.18 if (show_fw or mark_fw) else 1.08
        if bars_100 and not filter_active:
            ax.set_ylim(0, max(100.0 * top_pad, 1.0))
        else:
            ymax = max(float(base_top), 1e-6)
            ax.set_ylim(0, ymax * top_pad)

        fw = folk_ward_graphic_params(fw_x, fw_y) if (show_fw or mark_fw) else None
        valid_fw = fw is not None and np.isfinite(fw.get("Mz", np.nan))
        if mark_fw and valid_fw:
            self._draw_fw_markers(ax, fw, x_mode=x_mode)
        if show_fw and valid_fw:
            self._draw_fw_text(ax, fw, fw_x, fw_y)
        elif show_fw and not valid_fw:
            ax.text(
                0.02, 0.98, "Folk & Ward: insufficient data",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=style["fw_text_fontsize"],
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor=style["fw_box_facecolor"],
                    edgecolor=style["fw_box_edgecolor"],
                    linewidth=style["fw_box_linewidth"],
                    alpha=style["fw_box_alpha"],
                ), zorder=12,
            )

        left_label = style["y_label_left"].strip() or default_y_label
        ax.set_xlabel(
            x_label, fontfamily="Arial", fontsize=style["axis_label_fontsize"],
            fontweight=style["axis_label_fontweight"],
        )
        ax.set_ylabel(
            left_label, fontfamily="Arial", fontsize=style["axis_label_fontsize"],
            fontweight=style["axis_label_fontweight"],
        )
        ax.set_title(
            full_title, fontfamily="Arial", fontsize=style["title_fontsize"],
            fontweight=style["title_fontweight"], pad=14,
        )

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)

        # Use the real external edges of the first and last bars for both
        # phi and micrometre modes.  Limiting the axis to the tick centres
        # clips half of the first and last bars (and their outlines).
        bar_left_edges = x_plot - (bar_widths / 2.0)
        bar_right_edges = x_plot + (bar_widths / 2.0)
        valid_edges = np.isfinite(bar_left_edges) & np.isfinite(bar_right_edges)
        if np.any(valid_edges):
            xmin = float(np.nanmin(bar_left_edges[valid_edges]))
            xmax = float(np.nanmax(bar_right_edges[valid_edges]))
            if np.isfinite(xmin) and np.isfinite(xmax) and xmin != xmax:
                span = abs(xmax - xmin)
                max_bar_width = float(np.nanmax(np.abs(bar_widths))) if bar_widths.size else 0.0
                # Leave a small visible gap outside the COMPLETE extreme bars.
                # This changes only the plotting viewport, never the data/ticks.
                edge_padding = max(span * 0.008, max_bar_width * 0.10, 1e-9)
                ax.set_xlim(xmin - edge_padding, xmax + edge_padding)

        if style["reverse_x"]:
            ax.invert_xaxis()
            ax2.set_xlim(ax.get_xlim())

        secax = None
        if style["show_um_axis"] and x_mode == "phi":
            position = style["um_axis_position"]
            secax = ax.secondary_xaxis("top" if position == "top" else "bottom")
            # Micrometre labels are substantially wider than phi values. Keep a
            # separate sparse set (including both extremes) so labels never overlap.
            um_tick_pos = _sparse_values(np.asarray(tick_pos, dtype=float), max_count=9)
            secax.set_xticks(um_tick_pos)
            secax.set_xticklabels([
                f"{phi_to_um(v):.0f}" if phi_to_um(v) >= 10 else f"{phi_to_um(v):.2g}"
                for v in um_tick_pos
            ])
            secax.set_xlabel(
                style["um_secondary_label"].strip() or "Particle size (μm)",
                fontsize=style["right_axis_label_fontsize"],
                fontweight=style["axis_label_fontweight"], fontfamily="Arial",
            )
            if position == "bottom":
                secax.xaxis.set_ticks_position("bottom")
                secax.xaxis.set_label_position("bottom")
                secax.spines["bottom"].set_position(("outward", 36))

        # Axes, ticks and grid.
        for side in ["left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(style["axes_color"])
            ax.spines[side].set_linewidth(style["axes_linewidth"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_cum:
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color(style["axes_color"])
            ax2.spines["right"].set_linewidth(style["axes_linewidth"])
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        ax.tick_params(
            axis="both", which="both", colors=style["tick_color"],
            width=style["tick_width"], length=style["tick_length"],
            labelsize=style["tick_fontsize"],
        )
        if show_cum:
            ax2.tick_params(
                axis="y", which="both", colors=style["tick_color"],
                width=style["tick_width"], length=style["tick_length"],
                labelsize=style["tick_fontsize"], pad=6,
            )
        for lab in ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels():
            lab.set_fontfamily("Arial")

        if secax is not None:
            side = "bottom" if style["um_axis_position"] == "bottom" else "top"
            secax.spines[side].set_color(style["axes_color"])
            secax.spines[side].set_linewidth(style["axes_linewidth"])
            secax.tick_params(
                axis="x", colors=style["tick_color"],
                width=style["tick_width"], length=style["tick_length"],
                labelsize=style["tick_fontsize"],
            )
            for lab in secax.get_xticklabels():
                lab.set_fontfamily("Arial")

        if style["grid_visible"]:
            ax.grid(
                True,
                color=style["grid_color"], linewidth=style["grid_linewidth"],
                linestyle=style["grid_linestyle"], alpha=style["grid_alpha"],
                axis="both", zorder=0,
            )
        else:
            ax.grid(False)
        ax2.grid(False)

        # Reserve a true right-side legend gutter inside the exported figure.
        # Reserve three independent horizontal zones: plot, cumulative-axis
        # labels, and legend. This prevents any collision regardless of label length.
        try:
            fig.legends.clear()
        except Exception:
            pass

        if show_cum:
            # Keep the cumulative-axis title close to its own tick labels while
            # still leaving a clean gap.  The legend occupies a separate gutter
            # farther to the right, so the ylabel does not need to be pushed
            # across the page.
            ax2.yaxis.set_label_coords(1.105, 0.50)
            ax2.yaxis.label.set_horizontalalignment("center")
            ax2.yaxis.label.set_verticalalignment("center")

        bottom_margin = 0.12
        top_margin = 0.91
        if secax is not None:
            if style["um_axis_position"] == "bottom":
                bottom_margin = 0.22
                top_margin = 0.91
            else:
                top_margin = 0.84

        if legend_handles:
            # Leave a dedicated gutter for cumulative ticks/title and another,
            # wider gutter for the legend.  This keeps every element inside the
            # physical figure without clipping long component names.
            right_margin = 0.68 if show_cum else 0.73
        else:
            right_margin = 0.90 if show_cum else 0.97

        fig.subplots_adjust(
            left=0.095, right=right_margin,
            bottom=bottom_margin, top=top_margin,
        )

        if legend_handles:
            # A dedicated axes is used instead of fig.legend so the legend can
            # never migrate back over the cumulative axis.  The gutter is wide
            # enough for the longest normal component labels and stays entirely
            # inside the exported page.
            legend_left = 0.785 if show_cum else 0.755
            legend_right = 0.985
            legend_width = legend_right - legend_left
            leg_ax = fig.add_axes([legend_left, bottom_margin + 0.02,
                                   legend_width, max(0.20, top_margin - bottom_margin - 0.04)])
            leg_ax.set_axis_off()
            legend_fs = min(float(style["legend_fontsize"]), 8.0)
            leg = leg_ax.legend(
                legend_handles, legend_labels,
                frameon=bool(style["legend_frame"]),
                fontsize=legend_fs,
                loc="center left",
                bbox_to_anchor=(0.0, 0.5),
                borderaxespad=0.0,
                handlelength=1.10,
                handletextpad=0.42,
                labelspacing=0.26,
            )
            if leg is not None and style["legend_frame"]:
                frame = leg.get_frame()
                frame.set_edgecolor(style["legend_framecolor"])
                frame.set_linewidth(style["legend_framewidth"])
                frame.set_alpha(style["legend_alpha"])

    def plot_histogram(self):
        # Palette application, table refreshes and style-dialog acceptance may
        # request a redraw almost simultaneously.  Coalesce those requests so a
        # partially updated figure is never shown.
        if getattr(self, "_plot_in_progress", False):
            self._plot_pending = True
            return

        self._plot_in_progress = True
        self._plot_pending = False
        try:
            db = self.current_db()
            style = normalize_componentry_hist_style(self.componentry_hist_style)
            self.componentry_hist_style = dict(style)
            self._sync_histogram_preview_geometry(style)

            # Reuse the Figure already owned by the FigureCanvas.  Replacing the
            # Figure object changes FigureCanvas.sizeHint(), which was the cause
            # of the preview becoming smaller after accepting the style dialog.
            # Clearing this same Figure removes every twinx/secondary_xaxis from
            # the previous render, preventing the temporary duplicated graph.
            fig = self.canvas.figure
            fig.clear()
            fig.set_facecolor("white")
            ax = fig.add_subplot(111)

            if db is None:
                ax.set_axis_off()
                ax.text(
                    0.5, 0.54, "No sample loaded",
                    ha="center", va="center",
                    fontsize=14, fontfamily="Arial", fontweight="bold",
                    transform=ax.transAxes
                )
                ax.text(
                    0.5, 0.46, ("Choose a database and sample above to begin." if getattr(self, "embedded_mode", False) else "Load an Excel file or open a project to begin."),
                    ha="center", va="center",
                    fontsize=10.5, fontfamily="Arial", color="#555555",
                    transform=ax.transAxes
                )
                fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.94)
            else:
                self.update_summary()
                self._draw_histogram(
                    ax=ax,
                    db=db,
                    sample_title=self.current_sample_name(),
                    bars_100=bool(self.chk_100.isChecked()),
                    by_type=bool(self.chk_type.isChecked()),
                    show_fw=bool(self.chk_show_fw.isChecked()),
                    mark_fw=bool(self.chk_mark_fw.isChecked()),
                )

            # Synchronous draw: the old frame is replaced atomically instead of
            # remaining visible while a queued draw builds the new axes.
            self.canvas.draw()
            self.canvas.repaint()
            self._position_summary_overlay()

        finally:
            self._plot_in_progress = False
            if self._plot_pending:
                self._plot_pending = False
                QTimer.singleShot(0, self.plot_histogram)
#endregion

# =========================================================
# FW DATABASE EDIT SYNCHRONIZATION PATCH
# =========================================================
def _componentry_update_gsd_preserving_componentry(self, new_gsd, phi_tolerance=1e-9):
    """
    Replace the granulometric distribution while preserving the component
    percentages already determined for every unchanged phi class.

    frac100 is preserved by phi. total is recalculated from the new bin weight:
        total(component, phi) = frac100(component, phi) * wt(phi) / 100

    New phi classes start empty. Removed phi classes and their transient image
    sessions are discarded. Component definitions, colors and types are kept.
    """
    if new_gsd is None:
        raise ValueError("new_gsd is required.")

    clean = pd.DataFrame(new_gsd).copy()
    if "phi" not in clean.columns or "wt" not in clean.columns:
        raise ValueError("The updated granulometry must contain phi and wt columns.")

    clean = clean.loc[:, ["phi", "wt"]].copy()
    clean["phi"] = pd.to_numeric(clean["phi"], errors="coerce")
    clean["wt"] = pd.to_numeric(clean["wt"], errors="coerce")
    clean = clean[np.isfinite(clean["phi"]) & np.isfinite(clean["wt"])]
    if clean.empty:
        raise ValueError("The updated granulometry has no valid phi/wt values.")
    if (clean["wt"] < 0).any():
        raise ValueError("Granulometric weights cannot be negative.")

    clean = (
        clean.groupby("phi", as_index=False)["wt"]
        .sum()
        .sort_values("phi", kind="mergesort")
        .reset_index(drop=True)
    )
    if float(clean["wt"].sum()) <= 0:
        raise ValueError("The updated granulometry must have a total weight greater than zero.")

    old_gsd = getattr(self, "gsd", pd.DataFrame(columns=["phi", "wt"])).copy()
    old_phis = []
    if old_gsd is not None and not old_gsd.empty and "phi" in old_gsd.columns:
        old_phis = [float(v) for v in pd.to_numeric(old_gsd["phi"], errors="coerce").dropna().tolist()]

    components = list(getattr(self, "components", []) or [])
    old_frac = getattr(self, "frac100", pd.DataFrame()).copy()
    old_total = getattr(self, "total", pd.DataFrame()).copy()
    old_filled = set(float(v) for v in (getattr(self, "filled", set()) or set()))
    old_sessions = dict(getattr(self, "image_sessions", {}) or {})

    try:
        old_frac.index = old_frac.index.astype(float)
    except Exception:
        pass
    try:
        old_total.index = old_total.index.astype(float)
    except Exception:
        pass

    new_phis = [float(v) for v in clean["phi"].tolist()]
    new_frac = pd.DataFrame(0.0, index=new_phis, columns=components)
    new_total = pd.DataFrame(0.0, index=new_phis, columns=components)
    new_filled = set()
    new_sessions = {}
    matched_old = set()

    def _matching_old_phi(phi_value):
        candidates = []
        for old_phi in old_phis:
            if np.isclose(float(phi_value), float(old_phi), atol=float(phi_tolerance), rtol=0.0):
                candidates.append(float(old_phi))
        return candidates[0] if candidates else None

    wt_map = dict(zip(new_phis, [float(v) for v in clean["wt"].tolist()]))

    for new_phi in new_phis:
        old_phi = _matching_old_phi(new_phi)
        if old_phi is None:
            continue
        matched_old.add(float(old_phi))

        if components and old_phi in old_frac.index:
            vals = []
            for comp in components:
                try:
                    value = float(_to_float(old_frac.loc[old_phi, comp])) if comp in old_frac.columns else 0.0
                except Exception:
                    value = 0.0
                if not np.isfinite(value) or value < 0:
                    value = 0.0
                vals.append(value)

            row_sum = float(np.sum(vals))
            was_filled = any(
                np.isclose(float(old_phi), float(v), atol=float(phi_tolerance), rtol=0.0)
                for v in old_filled
            )
            if row_sum > 0:
                vals = 100.0 * np.asarray(vals, dtype=float) / row_sum
                new_frac.loc[new_phi, components] = vals
                if was_filled:
                    new_total.loc[new_phi, components] = vals * (float(wt_map[new_phi]) / 100.0)
                    new_filled.add(float(new_phi))

        for session_phi, session_state in old_sessions.items():
            try:
                if np.isclose(float(session_phi), float(old_phi), atol=float(phi_tolerance), rtol=0.0):
                    new_sessions[float(new_phi)] = session_state
                    break
            except Exception:
                continue

    self.gsd = clean
    self.frac100 = new_frac
    self.total = new_total
    self.filled = new_filled
    self.image_sessions = new_sessions

    removed = [p for p in old_phis if p not in matched_old]
    added = [p for p in new_phis if _matching_old_phi(p) is None]
    return {
        "preserved_filled": len(new_filled),
        "added_phi": added,
        "removed_phi": removed,
        "old_total": float(pd.to_numeric(old_gsd.get("wt", pd.Series(dtype=float)), errors="coerce").sum()) if old_gsd is not None else 0.0,
        "new_total": float(clean["wt"].sum()),
    }


def _componentry_refresh_after_external_gsd_change(self, sample_dbs=None, current_sample=None, source_context=None):
    """Refresh an open integrated Componentry window after FW edits its GSD."""
    previous = str(current_sample or self.current_sample_name() or "").strip()
    if sample_dbs is not None:
        self.sample_dbs = dict(sample_dbs or {})
    if isinstance(source_context, dict):
        self.source_context = dict(source_context)

    try:
        # Keep each sample's component names/types intact after an external GSD edit.
        self._sync_global_from_samples()
    except Exception:
        pass

    self._reload_sample_combo(select_first=False)
    if previous:
        idx = self.cmb_sample.findText(previous)
        if idx >= 0:
            self.cmb_sample.setCurrentIndex(idx)

    try:
        self.refresh_component_table_from_global()
    except Exception:
        pass
    try:
        self._refresh_hist_filter_value_options()
        self._refresh_hist_filter_ui_state()
    except Exception:
        pass
    try:
        self.update_summary()
    except Exception:
        pass
    try:
        self.plot_histogram()
    except Exception:
        pass


ComponentryDB.update_gsd_preserving_componentry = _componentry_update_gsd_preserving_componentry
ComponentryPreview.refresh_after_external_gsd_change = _componentry_refresh_after_external_gsd_change

