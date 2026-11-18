# PACKAGE_BUILD_MARKER: GRAPHIC_EXTERNAL_V16
# -*- coding: utf-8 -*-
"""
Graphic.py

External Graphic vector engine for FW-Granulometry. Filename expected by FW: Graphic.py.

This file contains the vector definitions and Graphic tab implementation for:
- GK
- Walker
- Pardo

FW should import this file and call:
    install_graphic_support(MainWindow)

Do not embed this content inside FW.
"""

GRAPHIC_ENGINE_VERSION = "v15-Graphic.py-external-vector-library"
# =========================================================

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Tuple, Any
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Polygon

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFontDatabase
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QColorDialog,
    QMessageBox, QGroupBox, QScrollArea, QSizePolicy, QTabWidget
)


@dataclass
class ZoneDef:
    name: str
    vertices: List[Tuple[float, float]]
    label_pos: Tuple[float, float]
    edgecolor: str = "#111111"
    facecolor: str = "#d9d9d9"
    alpha: float = 0.10
    linewidth: float = 1.10
    linestyle: Any = "-"
    hatch: str = ""
    zorder: int = 1
    label_size: int = 11
    label_font: str = "DejaVu Sans"


@dataclass
class DiagramDef:
    key: str
    title: str
    axis_limits: Tuple[float, float, float, float]
    zones: List[ZoneDef]
    views: Dict[str, List[str]]


LINESTYLE_OPTIONS = {
    "Auto": None,
    "Solid": "-",
    "Dashed": "--",
    "Dotted": ":",
    "Dash-dot": "-.",
}


# =========================================================
# Vector libraries
# Coordinate system: X = mean/median phi, Y = sorting sigma phi.
# These are editable vector reconstructions. They are meant to replace the
# previous raster backgrounds and can later be refined with digitized vertices.
# =========================================================

GK_ZONES = [
    ZoneDef(
        name="Flow",
        vertices=[
            (-3.10, 3.15), (-2.55, 3.85), (-1.75, 4.45), (-0.90, 4.90),
            (-0.15, 5.05), (0.65, 4.82), (1.25, 4.55), (1.75, 4.20),
            (2.40, 3.85), (3.05, 3.35), (3.45, 2.85), (3.65, 2.35),
            (3.42, 2.05), (2.85, 1.70), (2.10, 1.38), (1.20, 1.12),
            (0.25, 0.98), (-0.85, 1.04), (-1.75, 1.35), (-2.45, 1.82),
            (-2.95, 2.35),
        ],
        label_pos=(-0.25, 4.25),
        facecolor="#d8d8d8",
        alpha=0.10,
        linewidth=1.10,
        linestyle=(0, (4, 3, 1, 3)),
        zorder=1,
    ),
    ZoneDef(
        name="Fall",
        vertices=[
            (-4.00, 2.55), (-3.10, 2.85), (-2.15, 3.22), (-1.25, 3.55),
            (-0.25, 3.62), (0.65, 3.55), (1.45, 3.35), (2.40, 3.28),
            (3.25, 3.15), (4.10, 2.95), (5.05, 2.95), (5.75, 2.85),
            (5.98, 2.60), (5.90, 2.10), (5.95, 1.45), (5.92, 0.95),
            (5.78, 0.55), (5.28, 0.42), (4.62, 0.42), (4.10, 0.32),
            (3.65, 0.42), (3.35, 0.78), (2.85, 0.95), (2.05, 0.72),
            (1.20, 0.55), (0.30, 0.48), (-0.35, 0.48), (-1.00, 0.65),
            (-1.85, 0.65), (-2.80, 0.52), (-3.55, 0.50), (-4.00, 0.50),
        ],
        label_pos=(4.55, 1.70),
        facecolor="#f0f0f0",
        alpha=0.08,
        linewidth=1.05,
        linestyle="-",
        zorder=0,
    ),
    ZoneDef(
        name="Surge Dunes",
        vertices=[
            (0.75, 1.80), (1.05, 2.18), (1.52, 2.12), (2.00, 1.92),
            (2.42, 1.70), (2.82, 1.48), (3.02, 1.18), (2.82, 0.92),
            (2.38, 0.78), (1.82, 0.72), (1.28, 0.78), (0.92, 1.05),
            (0.72, 1.42),
        ],
        label_pos=(1.70, 1.35),
        facecolor="#efefef",
        alpha=0.10,
        linewidth=1.20,
        linestyle=(0, (8, 4)),
        zorder=3,
    ),
]

WALKER_ZONES = [
    ZoneDef(
        name="Fall deposit",
        vertices=[
            (-4.00, 2.45), (-3.25, 2.80), (-2.35, 3.15), (-1.45, 3.45),
            (-0.45, 3.60), (0.55, 3.55), (1.50, 3.35), (2.45, 3.10),
            (3.25, 2.90), (4.10, 2.75), (5.05, 2.65), (5.90, 2.52),
            (6.00, 0.55), (5.25, 0.50), (4.45, 0.48), (3.65, 0.58),
            (2.85, 0.78), (2.05, 0.80), (1.25, 0.65), (0.35, 0.55),
            (-0.65, 0.50), (-1.55, 0.58), (-2.50, 0.62), (-3.35, 0.55),
            (-4.00, 0.45),
        ],
        label_pos=(4.50, 1.55),
        facecolor="#efefef",
        alpha=0.08,
        linewidth=1.05,
        linestyle="-",
        zorder=0,
    ),
    ZoneDef(
        name="Fine-depleted flow",
        vertices=[
            (-3.65, 2.05), (-3.25, 2.85), (-2.45, 3.55), (-1.45, 4.15),
            (-0.35, 4.35), (0.85, 4.12), (1.85, 3.65), (2.60, 3.05),
            (3.05, 2.40), (2.88, 1.82), (2.10, 1.30), (1.05, 1.05),
            (-0.05, 1.05), (-1.05, 1.28), (-2.05, 1.52), (-2.95, 1.75),
        ],
        label_pos=(-2.10, 3.00),
        facecolor="#dddddd",
        alpha=0.10,
        linewidth=1.10,
        linestyle=(0, (6, 4, 1, 4)),
        zorder=2,
    ),
    ZoneDef(
        name="Pyroclastic surge",
        vertices=[
            (-0.30, 1.65), (0.30, 2.10), (1.15, 2.40), (2.05, 2.35),
            (2.95, 2.05), (3.50, 1.55), (3.35, 1.10), (2.55, 0.90),
            (1.60, 0.88), (0.70, 1.05), (0.00, 1.30),
        ],
        label_pos=(1.75, 1.72),
        facecolor="#e8e8e8",
        alpha=0.10,
        linewidth=1.15,
        linestyle=(0, (8, 4)),
        zorder=3,
    ),
    ZoneDef(
        name="Pyroclastic flow",
        vertices=[
            (-3.40, 3.25), (-2.70, 4.10), (-1.65, 4.80), (-0.45, 5.10),
            (0.80, 4.95), (1.75, 4.55), (2.55, 4.05), (3.15, 3.35),
            (3.25, 2.65), (2.70, 2.20), (1.80, 2.02), (0.70, 2.02),
            (-0.35, 2.18), (-1.35, 2.42), (-2.35, 2.78),
        ],
        label_pos=(0.05, 4.45),
        facecolor="#d4d4d4",
        alpha=0.09,
        linewidth=1.10,
        linestyle=":",
        zorder=1,
    ),
]

PARDO_ZONES = [
    ZoneDef(
        name="Fallout",
        vertices=[
            (-4.00, 2.60), (-3.25, 3.00), (-2.25, 3.45), (-1.20, 3.70),
            (-0.10, 3.65), (0.95, 3.45), (1.95, 3.25), (2.95, 3.05),
            (4.05, 2.85), (5.20, 2.70), (6.00, 2.55), (6.00, 0.42),
            (5.10, 0.35), (4.20, 0.40), (3.25, 0.55), (2.30, 0.65),
            (1.25, 0.60), (0.25, 0.52), (-0.75, 0.52), (-1.75, 0.60),
            (-2.75, 0.55), (-3.55, 0.45), (-4.00, 0.42),
        ],
        label_pos=(4.65, 1.58),
        facecolor="#f0f0f0",
        alpha=0.08,
        linewidth=1.05,
        linestyle="-",
        zorder=0,
    ),
    ZoneDef(
        name="PF",
        vertices=[
            (-3.40, 3.05), (-2.70, 3.95), (-1.75, 4.55), (-0.65, 4.80),
            (0.35, 4.60), (1.15, 4.25), (1.85, 3.70), (2.35, 3.10),
            (2.45, 2.45), (2.05, 1.95), (1.25, 1.65), (0.20, 1.55),
            (-0.85, 1.70), (-1.85, 1.95), (-2.75, 2.35),
        ],
        label_pos=(-0.45, 4.05),
        facecolor="#d6d6d6",
        alpha=0.10,
        linewidth=1.10,
        linestyle=(0, (5, 3)),
        zorder=2,
    ),
    ZoneDef(
        name="PS",
        vertices=[
            (0.30, 1.45), (0.85, 2.00), (1.60, 2.25), (2.45, 2.18),
            (3.25, 1.85), (3.65, 1.35), (3.40, 0.95), (2.55, 0.78),
            (1.60, 0.82), (0.75, 1.02),
        ],
        label_pos=(2.10, 1.50),
        facecolor="#e5e5e5",
        alpha=0.10,
        linewidth=1.15,
        linestyle=":",
        zorder=3,
    ),
    ZoneDef(
        name="Zone 1",
        vertices=[
            (-3.60, 0.55), (-2.75, 0.70), (-1.75, 0.80), (-0.70, 0.85),
            (0.35, 0.80), (0.85, 0.62), (0.35, 0.45), (-0.85, 0.40),
            (-2.10, 0.42), (-3.20, 0.45),
        ],
        label_pos=(-1.70, 0.62),
        facecolor="#eeeeee",
        alpha=0.08,
        linewidth=1.00,
        linestyle=(0, (6, 4)),
        zorder=4,
    ),
    ZoneDef(
        name="Zone 2",
        vertices=[
            (-1.05, 0.90), (-0.25, 1.22), (0.80, 1.42), (1.60, 1.30),
            (1.85, 1.00), (1.35, 0.75), (0.35, 0.65), (-0.65, 0.70),
        ],
        label_pos=(0.45, 1.05),
        facecolor="#eeeeee",
        alpha=0.08,
        linewidth=1.00,
        linestyle=(0, (6, 4)),
        zorder=5,
    ),
    ZoneDef(
        name="Zone 3",
        vertices=[
            (1.20, 0.92), (2.05, 1.22), (3.05, 1.18), (3.80, 0.95),
            (4.15, 0.62), (3.45, 0.42), (2.35, 0.40), (1.45, 0.55),
        ],
        label_pos=(2.75, 0.78),
        facecolor="#eeeeee",
        alpha=0.08,
        linewidth=1.00,
        linestyle=(0, (6, 4)),
        zorder=5,
    ),
]


def _views_from_zones(zones: List[ZoneDef]) -> Dict[str, List[str]]:
    return {"All": [z.name for z in zones], **{z.name: [z.name] for z in zones}}


DIAGRAMS: Dict[str, DiagramDef] = {
    "gk": DiagramDef(
        key="gk",
        title="Gençalioğlu-Kuşcu et al. 2007",
        axis_limits=(-4.0, 6.0, 0.0, 6.0),
        zones=GK_ZONES,
        views=_views_from_zones(GK_ZONES),
    ),
    "walker": DiagramDef(
        key="walker",
        title="Walker (1971 and 1983)",
        axis_limits=(-4.0, 6.0, 0.0, 6.0),
        zones=WALKER_ZONES,
        views=_views_from_zones(WALKER_ZONES),
    ),
    "pardo": DiagramDef(
        key="pardo",
        title="Pardo et al. 2009",
        axis_limits=(-4.0, 6.0, 0.0, 6.0),
        zones=PARDO_ZONES,
        views=_views_from_zones(PARDO_ZONES),
    ),
}


# =========================================================
# Font handling
# =========================================================

_AVAILABLE_FONT_CACHE = None


def _available_font_names() -> List[str]:
    global _AVAILABLE_FONT_CACHE
    if _AVAILABLE_FONT_CACHE is not None:
        return list(_AVAILABLE_FONT_CACHE)
    names = set()
    try:
        names.update(QFontDatabase().families())
    except Exception:
        pass
    try:
        for f in font_manager.fontManager.ttflist:
            if getattr(f, "name", None):
                names.add(str(f.name))
    except Exception:
        pass
    preferred = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono", "Arial", "Times New Roman", "Calibri", "Cambria"]
    out = []
    for p in preferred:
        if p in names and p not in out:
            out.append(p)
    for n in sorted(names):
        if n not in out:
            out.append(n)
    if "DejaVu Sans" not in out:
        out.insert(0, "DejaVu Sans")
    _AVAILABLE_FONT_CACHE = out
    return list(out)


def _safe_font_name(name: str) -> str:
    names = set(_available_font_names())
    if name in names:
        return name
    return "DejaVu Sans"


# =========================================================
# Canvas with stable zoom/pan
# =========================================================

class BoundedGraphicCanvas(FigureCanvas):
    def __init__(self, width_in=8.8, height_in=5.4, dpi=100):
        fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
        super().__init__(fig)
        self._dpi = int(dpi)
        self._base_w_px = int(round(width_in * dpi))
        self._base_h_px = int(round(height_in * dpi))
        self._zoom = 1.0
        self._min_w_px = 500
        self._min_h_px = 340
        self._max_w_px = 2600
        self._max_h_px = 1900
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
        return int(round(self._base_w_px * self._zoom)), int(round(self._base_h_px * self._zoom))

    def _apply_sizes(self, redraw=True):
        w, h = self._current_size_px()
        w = max(1, w)
        h = max(1, h)
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)
        self.setFixedSize(w, h)
        try:
            self.figure.set_size_inches(float(w) / self._dpi, float(h) / self._dpi, forward=False)
            self.figure.set_dpi(self._dpi)
        except Exception:
            pass
        if redraw:
            self.draw_idle()

    def wheelEvent(self, event):
        try:
            delta = event.angleDelta().y()
        except Exception:
            delta = 0
        if delta == 0:
            return super().wheelEvent(event)

        old_zoom = float(self._zoom)
        factor = 1.12 if delta > 0 else 1.0 / 1.12
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
            if event.button() == Qt.MiddleButton:
                self._panning = True
                self._pan_start_global = event.globalPos()
                if self._scroll_area is not None:
                    self._pan_start_h = self._scroll_area.horizontalScrollBar().value()
                    self._pan_start_v = self._scroll_area.verticalScrollBar().value()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        except Exception:
            pass
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            if self._panning and self._scroll_area is not None and self._pan_start_global is not None:
                delta = event.globalPos() - self._pan_start_global
                self._scroll_area.horizontalScrollBar().setValue(self._pan_start_h - delta.x())
                self._scroll_area.verticalScrollBar().setValue(self._pan_start_v - delta.y())
                event.accept()
                return
        except Exception:
            pass
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.MiddleButton and self._panning:
                self._panning = False
                self._pan_start_global = None
                self.unsetCursor()
                event.accept()
                return
        except Exception:
            pass
        super().mouseReleaseEvent(event)


# =========================================================
# Drawing helpers
# =========================================================

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


def _display_label(name: str) -> str:
    replacements = {
        "Fine-depleted flow": "Fine-depleted\nflow",
        "Pyroclastic surge": "Pyroclastic\nsurge",
        "Pyroclastic flow": "Pyroclastic\nflow",
        "Surge Dunes": "Surge\nDunes",
        "Fall deposit": "Fall\ndeposit",
    }
    return replacements.get(name, name)


def _zone_defaults(diagram: DiagramDef) -> Dict[str, Dict[str, Any]]:
    out = {}
    for zone in diagram.zones:
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


def _field_style(diagram: DiagramDef, zone_styles: Dict[str, Dict[str, Any]], zone: ZoneDef) -> Dict[str, Any]:
    style = dict(_zone_defaults(diagram).get(zone.name, {}))
    style.update(zone_styles.get(zone.name, {}) or {})
    style["label_font"] = _safe_font_name(str(style.get("label_font", "DejaVu Sans")))
    return style


def _style_linestyle(style: Dict[str, Any], zone: ZoneDef):
    name = str(style.get("linestyle_name", "Auto"))
    override = LINESTYLE_OPTIONS.get(name, None)
    return zone.linestyle if override is None else override


def draw_vector_diagram(ax, diagram: DiagramDef, view: str, zone_styles: Dict[str, Dict[str, Any]], show_labels: bool):
    zone_names = diagram.views.get(view, diagram.views.get("All", []))
    for zone in diagram.zones:
        if zone.name not in zone_names:
            continue
        style = _field_style(diagram, zone_styles, zone)
        poly = Polygon(
            _chaikin_closed(zone.vertices, iterations=3),
            closed=True,
            facecolor=style.get("facecolor", zone.facecolor),
            edgecolor=style.get("edgecolor", zone.edgecolor),
            alpha=float(style.get("alpha", zone.alpha)),
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
            ax.text(
                float(style.get("label_x", zone.label_pos[0])),
                float(style.get("label_y", zone.label_pos[1])),
                _display_label(zone.name),
                fontsize=int(style.get("label_size", zone.label_size)),
                fontfamily=_safe_font_name(str(style.get("label_font", "DejaVu Sans"))),
                ha="center",
                va="center",
                color="#444444",
                zorder=max(10, zone.zorder + 5),
            )


class GraphicFamilyTab(QWidget):
    def __init__(self, main_window, family_key: str, diagram: DiagramDef):
        super().__init__(main_window)
        self.main = main_window
        self.family_key = family_key
        self.diagram = diagram
        self._loading_controls = False
        self._loading_layout_controls = False
        self._build_ui()
        self.load_controls_from_state()
        self.apply_canvas_layout(redraw=False)
        self.plot()

    def state(self) -> Dict[str, Any]:
        self.main._graphic_ensure_state()
        return self.main.graphic_family_states[self.family_key]

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.panel = QWidget()
        self.panel.setMinimumWidth(360)
        self.panel.setMaximumWidth(420)
        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        panel_layout.setSpacing(8)

        lbl_version = QLabel(f"Graphic engine: v15 — {self.diagram.title}")
        lbl_version.setStyleSheet("font-weight:bold; color:#444444;")
        panel_layout.addWidget(lbl_version)

        box_data = QGroupBox("Data and points")
        g_data = QGridLayout(box_data)
        self.btn_choose_db = QPushButton("Choose databases")
        self.btn_choose_db.clicked.connect(self.choose_databases)
        self.btn_filter_groups = QPushButton("Filter groups")
        self.btn_filter_groups.clicked.connect(self.select_groups)
        self.btn_point_colors = QPushButton("Point/group colors")
        self.btn_point_colors.clicked.connect(self.edit_point_colors)
        self.chk_sample_names = QCheckBox("Show sample names")
        self.chk_sample_names.toggled.connect(self.plot)
        g_data.addWidget(self.btn_choose_db, 0, 0)
        g_data.addWidget(self.btn_filter_groups, 0, 1)
        g_data.addWidget(self.btn_point_colors, 1, 0)
        g_data.addWidget(self.chk_sample_names, 1, 1)
        panel_layout.addWidget(box_data)

        box_display = QGroupBox("Graphic display")
        g_display = QGridLayout(box_display)
        self.cmb_view = QComboBox()
        self.cmb_view.addItems(list(self.diagram.views.keys()))
        self.cmb_view.currentIndexChanged.connect(self.plot)
        self.cmb_edit_field = QComboBox()
        self.cmb_edit_field.addItems([z.name for z in self.diagram.zones])
        self.cmb_edit_field.currentIndexChanged.connect(self.load_selected_field_controls)
        self.chk_labels = QCheckBox("Show labels")
        self.chk_labels.toggled.connect(self.plot)
        g_display.addWidget(QLabel("View:"), 0, 0)
        g_display.addWidget(self.cmb_view, 0, 1)
        g_display.addWidget(QLabel("Edit field:"), 1, 0)
        g_display.addWidget(self.cmb_edit_field, 1, 1)
        g_display.addWidget(self.chk_labels, 2, 0, 1, 2)
        panel_layout.addWidget(box_display)

        box_style = QGroupBox("Selected field style")
        g_style = QGridLayout(box_style)
        self.cmb_linestyle = QComboBox()
        self.cmb_linestyle.addItems(list(LINESTYLE_OPTIONS.keys()))
        self.cmb_linestyle.currentIndexChanged.connect(self.apply_selected_field_controls)
        self.spn_linewidth = QDoubleSpinBox()
        self.spn_linewidth.setRange(0.40, 5.00)
        self.spn_linewidth.setSingleStep(0.10)
        self.spn_linewidth.setDecimals(2)
        self.spn_linewidth.valueChanged.connect(self.apply_selected_field_controls)
        self.spn_alpha = QDoubleSpinBox()
        self.spn_alpha.setRange(0.0, 1.0)
        self.spn_alpha.setSingleStep(0.1)
        self.spn_alpha.setDecimals(1)
        self.spn_alpha.valueChanged.connect(self.apply_selected_field_controls)
        self.btn_line_color = QPushButton("Line color")
        self.btn_line_color.clicked.connect(lambda: self.pick_color("edgecolor"))
        self.btn_fill_color = QPushButton("Fill color")
        self.btn_fill_color.clicked.connect(lambda: self.pick_color("facecolor"))
        g_style.addWidget(QLabel("Line style:"), 0, 0)
        g_style.addWidget(self.cmb_linestyle, 0, 1)
        g_style.addWidget(QLabel("Line width:"), 1, 0)
        g_style.addWidget(self.spn_linewidth, 1, 1)
        g_style.addWidget(QLabel("Fill alpha:"), 2, 0)
        g_style.addWidget(self.spn_alpha, 2, 1)
        g_style.addWidget(self.btn_line_color, 3, 0)
        g_style.addWidget(self.btn_fill_color, 3, 1)
        panel_layout.addWidget(box_style)

        box_label = QGroupBox("Selected field label")
        g_label = QGridLayout(box_label)
        self.spn_labelsize = QSpinBox()
        self.spn_labelsize.setRange(6, 36)
        self.spn_labelsize.valueChanged.connect(self.apply_selected_field_controls)
        self.cmb_label_font = QComboBox()
        self.cmb_label_font.addItems(_available_font_names())
        self.cmb_label_font.currentIndexChanged.connect(self.apply_selected_field_controls)
        self.spn_label_x = QDoubleSpinBox()
        self.spn_label_x.setRange(-20.0, 20.0)
        self.spn_label_x.setSingleStep(0.10)
        self.spn_label_x.setDecimals(2)
        self.spn_label_x.valueChanged.connect(self.apply_selected_field_controls)
        self.spn_label_y = QDoubleSpinBox()
        self.spn_label_y.setRange(-20.0, 20.0)
        self.spn_label_y.setSingleStep(0.10)
        self.spn_label_y.setDecimals(2)
        self.spn_label_y.valueChanged.connect(self.apply_selected_field_controls)
        g_label.addWidget(QLabel("Label size:"), 0, 0)
        g_label.addWidget(self.spn_labelsize, 0, 1)
        g_label.addWidget(QLabel("Label font:"), 1, 0)
        g_label.addWidget(self.cmb_label_font, 1, 1)
        g_label.addWidget(QLabel("Label X:"), 2, 0)
        g_label.addWidget(self.spn_label_x, 2, 1)
        g_label.addWidget(QLabel("Label Y:"), 3, 0)
        g_label.addWidget(self.spn_label_y, 3, 1)
        panel_layout.addWidget(box_label)

        box_layout = QGroupBox("Layout and zoom")
        g_layout = QGridLayout(box_layout)
        self.spn_canvas_width = QSpinBox()
        self.spn_canvas_width.setRange(500, 2600)
        self.spn_canvas_width.setSingleStep(20)
        self.spn_canvas_width.valueChanged.connect(self.apply_layout_controls)
        self.spn_canvas_height = QSpinBox()
        self.spn_canvas_height.setRange(340, 1900)
        self.spn_canvas_height.setSingleStep(20)
        self.spn_canvas_height.valueChanged.connect(self.apply_layout_controls)
        self.btn_reset_layout = QPushButton("Reset layout")
        self.btn_reset_layout.clicked.connect(self.reset_layout)
        self.btn_reset_zoom = QPushButton("Reset zoom")
        self.btn_reset_zoom.clicked.connect(self.reset_zoom)
        g_layout.addWidget(QLabel("Width:"), 0, 0)
        g_layout.addWidget(self.spn_canvas_width, 0, 1)
        g_layout.addWidget(QLabel("Height:"), 1, 0)
        g_layout.addWidget(self.spn_canvas_height, 1, 1)
        g_layout.addWidget(self.btn_reset_layout, 2, 0)
        g_layout.addWidget(self.btn_reset_zoom, 2, 1)
        panel_layout.addWidget(box_layout)

        self.btn_export = QPushButton("Export image")
        self.btn_export.clicked.connect(lambda: self.main.export_canvas(self.canvas))
        panel_layout.addWidget(self.btn_export)
        panel_layout.addStretch(1)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("QScrollArea { background: #f4f6f8; border: 1px solid #d6dde8; border-radius: 8px; }")
        self.canvas = BoundedGraphicCanvas(width_in=9.2, height_in=6.0, dpi=100)
        self.canvas.set_scroll_area(self.scroll)
        self.scroll.setWidget(self.canvas)

        root.addWidget(self.panel, 0)
        root.addWidget(self.scroll, 1)

    def load_controls_from_state(self):
        st = self.state()
        self._loading_controls = True
        self._loading_layout_controls = True
        try:
            self.cmb_view.setCurrentText(str(st.get("view", "All")) if str(st.get("view", "All")) in self.diagram.views else "All")
            self.chk_labels.setChecked(bool(st.get("show_labels", True)))
            self.chk_sample_names.setChecked(bool(st.get("show_sample_names", False)))
            self.spn_canvas_width.setValue(int(st.get("layout_width_px", 920)))
            self.spn_canvas_height.setValue(int(st.get("layout_height_px", 600)))
        finally:
            self._loading_layout_controls = False
            self._loading_controls = False
        self.load_selected_field_controls()

    def current_field(self) -> str:
        try:
            return str(self.cmb_edit_field.currentText())
        except Exception:
            return self.diagram.zones[0].name

    def load_selected_field_controls(self, *args):
        st = self.state()
        zone_styles = st.setdefault("zone_styles", _zone_defaults(self.diagram))
        field = self.current_field()
        fstyle = zone_styles.get(field, _zone_defaults(self.diagram).get(field, {}))
        self._loading_controls = True
        try:
            widgets = [self.cmb_linestyle, self.spn_linewidth, self.spn_alpha, self.spn_labelsize, self.cmb_label_font, self.spn_label_x, self.spn_label_y]
            old_blocks = [w.blockSignals(True) for w in widgets]
            self.cmb_linestyle.setCurrentText(str(fstyle.get("linestyle_name", "Auto")))
            self.spn_linewidth.setValue(float(fstyle.get("linewidth", 1.1)))
            self.spn_alpha.setValue(float(fstyle.get("alpha", 0.1)))
            self.spn_labelsize.setValue(int(fstyle.get("label_size", 11)))
            font_name = _safe_font_name(str(fstyle.get("label_font", "DejaVu Sans")))
            if self.cmb_label_font.findText(font_name) < 0:
                self.cmb_label_font.insertItem(0, font_name)
            self.cmb_label_font.setCurrentText(font_name)
            self.spn_label_x.setValue(float(fstyle.get("label_x", 0.0)))
            self.spn_label_y.setValue(float(fstyle.get("label_y", 0.0)))
            for w, old in zip(widgets, old_blocks):
                w.blockSignals(old)
        finally:
            self._loading_controls = False
        self.plot()

    def apply_selected_field_controls(self, *args):
        if self._loading_controls:
            return
        st = self.state()
        zone_styles = st.setdefault("zone_styles", _zone_defaults(self.diagram))
        field = self.current_field()
        zone_styles.setdefault(field, {})
        fstyle = zone_styles[field]
        fstyle["linestyle_name"] = str(self.cmb_linestyle.currentText())
        fstyle["linewidth"] = float(self.spn_linewidth.value())
        fstyle["alpha"] = float(self.spn_alpha.value())
        fstyle["label_size"] = int(self.spn_labelsize.value())
        fstyle["label_font"] = _safe_font_name(str(self.cmb_label_font.currentText()))
        fstyle["label_x"] = float(self.spn_label_x.value())
        fstyle["label_y"] = float(self.spn_label_y.value())
        self.plot()

    def pick_color(self, key: str):
        st = self.state()
        zone_styles = st.setdefault("zone_styles", _zone_defaults(self.diagram))
        field = self.current_field()
        zone_styles.setdefault(field, {})
        current = zone_styles[field].get(key, "#111111" if key == "edgecolor" else "#d9d9d9")
        color = QColorDialog.getColor(QColor(str(current)), self, f"{self.diagram.title} — {field}")
        if color.isValid():
            zone_styles[field][key] = color.name()
            self.plot()

    def apply_layout_controls(self, *args):
        if self._loading_layout_controls:
            return
        st = self.state()
        st["layout_width_px"] = int(self.spn_canvas_width.value())
        st["layout_height_px"] = int(self.spn_canvas_height.value())
        self.apply_canvas_layout(redraw=True)

    def apply_canvas_layout(self, redraw=True):
        st = self.state()
        w = int(st.get("layout_width_px", 920))
        h = int(st.get("layout_height_px", 600))
        z = float(st.get("layout_zoom", 1.0))
        self.canvas.set_base_size_px(w, h, redraw=False)
        self.canvas.set_zoom(z, redraw=redraw)

    def reset_layout(self):
        st = self.state()
        st["layout_width_px"] = 920
        st["layout_height_px"] = 600
        self._loading_layout_controls = True
        try:
            self.spn_canvas_width.setValue(920)
            self.spn_canvas_height.setValue(600)
        finally:
            self._loading_layout_controls = False
        self.apply_canvas_layout(redraw=True)

    def reset_zoom(self):
        st = self.state()
        st["layout_zoom"] = 1.0
        self.canvas.reset_zoom()

    def choose_databases(self):
        try:
            self.main.select_db_to_plot()
        except Exception as e:
            QMessageBox.warning(self, "Graphic", str(e))
        self.main.plot_all_graphic_tabs()

    def edit_point_colors(self):
        try:
            self.main.edit_colors()
        except Exception as e:
            QMessageBox.warning(self, "Graphic", str(e))
        self.main.plot_all_graphic_tabs()

    def select_groups(self):
        try:
            keys = self.main._safe_group_keys()
        except Exception:
            keys = []
        if not keys:
            QMessageBox.information(self, "Graphic", "No groups available.")
            return
        st = self.state()
        current = st.get("selected_groups", keys)
        picked = self.main._groups_multi_select_dialog(f"Graphic — {self.diagram.title} — Filter groups", current)
        if picked is None:
            return
        st["selected_groups"] = list(picked)
        self.plot()

    def selected_groups(self):
        return list(self.state().get("selected_groups", []) or [])

    def plot(self, *args):
        st = self.state()
        st["view"] = str(self.cmb_view.currentText()) if str(self.cmb_view.currentText()) in self.diagram.views else "All"
        st["show_labels"] = bool(self.chk_labels.isChecked())
        st["show_sample_names"] = bool(self.chk_sample_names.isChecked())
        try:
            st["layout_zoom"] = float(self.canvas.zoom())
        except Exception:
            pass

        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        view = str(st.get("view", "All")) if str(st.get("view", "All")) in self.diagram.views else "All"
        draw_vector_diagram(ax, self.diagram, view, st.get("zone_styles", {}), bool(st.get("show_labels", True)))

        pts = []
        try:
            pts = list(self.main._iter_points_for_plot("mean", "sigma", tab_name=f"graphic_{self.family_key}"))
        except Exception:
            pts = []

        xmin, xmax, ymin, ymax = self.diagram.axis_limits
        if pts:
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
                ax.scatter(xs, ys, s=42, color=color, edgecolor="black", linewidth=0.45, alpha=0.95, label=label, zorder=25)
                if bool(st.get("show_sample_names", False)):
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
            ax.text(0.5, 0.03, "No data points loaded yet.", ha="center", va="bottom", transform=ax.transAxes, fontsize=9, color="#666666")

        try:
            ax.set_xlabel(self.main._axis_label_for_key("mean"))
            ax.set_ylabel(self.main._axis_label_for_key("sigma"))
        except Exception:
            ax.set_xlabel("Mean (Mφ)")
            ax.set_ylabel("Sorting (σφ)")

        ax.set_title(f"Graphic — {self.diagram.title} — {view}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.18)
        fig.tight_layout()
        self.canvas.draw_idle()

    def update_after_project_load(self):
        self.load_controls_from_state()
        self.apply_canvas_layout(redraw=False)
        self.plot()


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
        if not hasattr(self, "graphic_family_states") or not isinstance(self.graphic_family_states, dict):
            self.graphic_family_states = {}
        for key, diagram in DIAGRAMS.items():
            st = self.graphic_family_states.setdefault(key, {})
            st.setdefault("view", "All")
            if str(st.get("view", "All")) not in diagram.views:
                st["view"] = "All"
            st.setdefault("show_labels", True)
            st.setdefault("show_sample_names", False)
            st.setdefault("layout_width_px", 920)
            st.setdefault("layout_height_px", 600)
            st.setdefault("layout_zoom", 1.0)
            defaults = _zone_defaults(diagram)
            restored = st.get("zone_styles", {}) if isinstance(st.get("zone_styles", {}), dict) else {}
            merged = defaults
            for name, fs in restored.items():
                if name in defaults and isinstance(fs, dict):
                    merged[name].update(fs)
                    merged[name]["label_font"] = _safe_font_name(str(merged[name].get("label_font", "DejaVu Sans")))
            st["zone_styles"] = merged
            try:
                if not st.get("selected_groups"):
                    st["selected_groups"] = list(self._safe_group_keys())
            except Exception:
                st.setdefault("selected_groups", [])

    def _graphic_initUI_tabs(self):
        old_initUI_tabs(self)
        if hasattr(self, "tab_graphic"):
            return
        self._graphic_ensure_state()
        self.tab_graphic = QWidget()
        root = QVBoxLayout(self.tab_graphic)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.graphic_tabs_widget = QTabWidget()
        self.graphic_family_tabs = {}
        for key, label in [("gk", "GK"), ("walker", "Walker"), ("pardo", "Pardo")]:
            tab = GraphicFamilyTab(self, key, DIAGRAMS[key])
            self.graphic_family_tabs[key] = tab
            self.graphic_tabs_widget.addTab(tab, label)
        root.addWidget(self.graphic_tabs_widget)

        self.tabs.addTab(self.tab_graphic, "Graphic v15")

    def _plot_all_graphic_tabs(self):
        for tab in getattr(self, "graphic_family_tabs", {}).values():
            try:
                tab.plot()
            except Exception:
                pass

    def _graphic_selected_group_keys_for_tab(self, tab_name: str):
        if str(tab_name).startswith("graphic_"):
            key = str(tab_name).split("graphic_", 1)[1]
            try:
                self._graphic_ensure_state()
                return set(self.graphic_family_states.get(key, {}).get("selected_groups", []) or [])
            except Exception:
                return set()
        if old_selected_group_keys_for_tab is not None:
            return old_selected_group_keys_for_tab(self, tab_name)
        return set()

    def _graphic_update_all_groups_and_colors(self):
        if old_update_all_groups_and_colors is not None:
            old_update_all_groups_and_colors(self)
        try:
            keys = list(getattr(self, "group_labels", {}).keys())
            self._graphic_ensure_state()
            valid = set(keys)
            for st in self.graphic_family_states.values():
                selected = list(st.get("selected_groups", []) or [])
                selected = [k for k in selected if k in valid]
                if not selected and keys:
                    selected = list(keys)
                st["selected_groups"] = selected
        except Exception:
            pass

    def _graphic_refresh_after_settings_change(self):
        if old_refresh_after_settings_change is not None:
            old_refresh_after_settings_change(self)
        self.plot_all_graphic_tabs()

    def _graphic_save_current_tab(self):
        current = self.tabs.currentWidget()
        if current is getattr(self, "tab_graphic", None):
            try:
                sub = self.graphic_tabs_widget.currentWidget()
                return self.export_canvas(sub.canvas)
            except Exception:
                return None
        if old_save_current_tab is not None:
            return old_save_current_tab(self)
        return None

    MainWindow._graphic_ensure_state = _graphic_ensure_state
    MainWindow.plot_all_graphic_tabs = _plot_all_graphic_tabs
    MainWindow.initUI_tabs = _graphic_initUI_tabs
    MainWindow.save_current_tab = _graphic_save_current_tab
    MainWindow._refresh_after_settings_change = _graphic_refresh_after_settings_change
    MainWindow._selected_group_keys_for_tab = _graphic_selected_group_keys_for_tab
    MainWindow._update_all_groups_and_colors = _graphic_update_all_groups_and_colors
    MainWindow._graphic_support_installed = True

    # Full project save/load support.
    try:
        main_mod = sys.modules.get(MainWindow.__module__)
        if main_mod is not None and not getattr(main_mod, "_graphic_project_patch_installed_v13", False):
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
                    fw["graphic_engine_version"] = GRAPHIC_ENGINE_VERSION
                    fw["graphic_family_states"] = deepcopy(getattr(self, "graphic_family_states", {}))
                    return data
                setattr(main_mod, "_fw_collect_project_dict", _collect_with_graphic)

            if old_apply is not None:
                def _apply_with_graphic(self, data):
                    old_apply(self, data)
                    fw = (data.get("fw", {}) or {}) if isinstance(data, dict) else {}
                    try:
                        if isinstance(fw.get("graphic_family_states"), dict):
                            self.graphic_family_states = deepcopy(fw.get("graphic_family_states") or {})
                        self._graphic_ensure_state()
                        for tab in getattr(self, "graphic_family_tabs", {}).values():
                            tab.update_after_project_load()
                    except Exception:
                        pass
                setattr(main_mod, "_fw_apply_project_dict", _apply_with_graphic)

            setattr(main_mod, "_graphic_project_patch_installed_v13", True)
    except Exception:
        pass
