# -*- coding: utf-8 -*-
"""
GraphicVectors.py

Vector reconstructions used by the Graphic tab of FW-Granulometry.

This module contains geometry only. It does not create PyQt widgets, canvases,
menus, dialogs, or application state. Graphic/UI code should import the vector
libraries from here.

Current contents:
- Gençalioğlu-Kuşcu et al. (2007) reconstruction.
- Walker (1971 and 1983) reconstruction.
- Pardo et al. (2009) reconstruction.
- Pyroclastic grain-size ternary classification (blocks/bombs-lapilli-ash).

The Pardo reconstruction keeps the same editable/vector aesthetic used for GK:
soft fills, editable line style, editable label position, and no raster image
loaded at runtime.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


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


# =========================================================
# Gençalioğlu-Kuşcu et al. (2007)
# Coordinate system: X = mean phi, Y = sorting/sigma phi.
# =========================================================

GK_FLOW_ZONE = ZoneDef(
    name="Flow",
    vertices=[
        (-3.1, 2.55), (-2.8, 3.00), (-2.25, 3.55), (-1.65, 3.95),
        (-0.85, 4.15), (0.05, 4.05), (0.85, 3.90), (1.55, 3.55),
        (2.15, 3.35), (2.75, 2.90), (3.30, 2.60), (3.65, 2.35),
        (3.95, 2.02), (4.05, 1.72), (3.92, 1.50), (3.40, 1.10),
        (2.75, 0.85), (1.75, 0.60), (0.95, 0.42), (0.25, 0.38),
        (-0.55, 0.45), (-1.25, 0.62), (-2.00, 1.05), (-2.70, 1.60),
        (-3.08, 2.00),
    ],
    label_pos=(-0.9, 3.45),
    facecolor="#d8d8d8",
    alpha=0.10,
    linewidth=1.10,
    linestyle="-",
    zorder=1,
)

GK_FALL_ZONE = ZoneDef(
    name="Fall",
    vertices=[
        (-3.5, 2.10), (-2.75, 2.62), (-1.90, 2.95), (-0.55, 2.88),
        (0.90, 2.70), (2.10, 2.86), (3.25, 2.62), (4.30, 2.38),
        (5.45, 2.28), (5.50, 0.52), (4.95, 0.42), (4.15, 0.45),
        (3.62, 0.40), (2.85, 0.58), (1.85, 0.42), (0.70, 0.38),
        (-0.60, 0.42), (-1.55, 0.55), (-2.40, 0.52), (-3.15, 0.60),
        (-3.50, 0.70),
    ],
    label_pos=(4.55, 1.20),
    facecolor="#f0f0f0",
    alpha=0.10,
    linewidth=1.00,
    linestyle=(0, (10, 7)),
    zorder=0,
)

GK_SURGE_ZONE = ZoneDef(
    name="Surge",
    vertices=[
        (0.95, 1.55), (1.15, 1.50), (1.45, 1.40), (1.85, 1.28),
        (2.22, 1.18), (2.60, 1.03), (3.05, 0.96), (3.28, 1.05),
        (3.62, 1.28), (3.88, 1.52), (3.98, 1.78), (3.82, 2.02),
        (3.45, 2.15), (3.10, 2.22), (2.72, 2.28), (2.30, 2.18),
        (1.90, 2.05), (1.52, 1.88), (1.20, 1.70),
    ],
    label_pos=(3.25, 1.95),
    facecolor="#e5e5e5",
    alpha=0.10,
    linewidth=1.00,
    linestyle=(0, (6, 5)),
    zorder=2,
)

GK_SURGE_DUNES_ZONE = ZoneDef(
    name="Surge Dunes",
    vertices=[
        (1.22, 1.60), (1.45, 1.66), (1.82, 1.60), (2.15, 1.48),
        (2.45, 1.34), (2.70, 1.18), (2.95, 0.95), (3.06, 0.72),
        (2.92, 0.56), (2.62, 0.42), (2.15, 0.35), (1.85, 0.28),
        (1.58, 0.28), (1.35, 0.42), (1.08, 0.70), (0.95, 1.02),
        (1.00, 1.30),
    ],
    label_pos=(1.55, 1.35),
    facecolor="#efefef",
    alpha=0.10,
    linewidth=1.10,
    linestyle=":",
    zorder=3,
)

GK_FINE_DEPLETED_FLOW_ZONE = ZoneDef(
    name="Fine-depleted flow",
    vertices=[
        (-3.45, 2.25), (-3.10, 2.80), (-2.20, 3.10), (-1.35, 3.02),
        (-0.45, 2.60), (0.45, 1.90), (1.25, 1.35), (2.00, 0.82),
        (2.55, 0.58), (2.85, 0.78), (2.95, 1.22), (2.78, 1.75),
        (2.30, 2.10), (1.55, 2.55), (0.55, 2.92), (-0.45, 3.20),
        (-1.20, 3.18), (-2.05, 3.00), (-2.85, 2.72),
    ],
    label_pos=(-2.95, 2.60),
    facecolor="#ededed",
    alpha=0.10,
    linewidth=1.10,
    linestyle=(0, (6, 5, 1, 5)),
    zorder=1,
)

GK_ZONES = [
    GK_FLOW_ZONE,
    GK_FALL_ZONE,
    GK_SURGE_ZONE,
    GK_SURGE_DUNES_ZONE,
    GK_FINE_DEPLETED_FLOW_ZONE,
]

GK_VECTOR_LIBRARY: Dict[str, Dict[str, Any]] = {
    "All": {
        "zones": [GK_FLOW_ZONE, GK_FALL_ZONE, GK_SURGE_ZONE, GK_SURGE_DUNES_ZONE],
        "extra_lines": [
            {
                "xy": [
                    (-3.5, 2.05), (-2.90, 2.38), (-2.20, 2.70), (-1.40, 2.92),
                    (-0.40, 2.84), (0.70, 2.66), (1.55, 2.58), (2.35, 2.75),
                    (3.15, 2.68), (4.10, 2.45), (5.50, 2.28),
                ],
                "zone_name": "Fall",
                "linestyle": (0, (10, 7)),
                "linewidth": 1.00,
                "color": "#111111",
                "zorder": 4,
            },
            {
                "xy": [
                    (-3.5, 0.70), (-2.70, 0.55), (-1.90, 0.56), (-1.05, 0.66),
                    (-0.20, 0.55), (0.65, 0.55), (1.65, 0.60), (2.35, 0.62),
                    (2.80, 0.80), (3.15, 1.02), (3.70, 1.05),
                ],
                "zone_name": "Fall",
                "linestyle": (0, (10, 7)),
                "linewidth": 1.00,
                "color": "#111111",
                "zorder": 4,
            },
        ],
    },
    "Flow": {"zones": [GK_FLOW_ZONE], "extra_lines": []},
    "Fall": {"zones": [GK_FALL_ZONE], "extra_lines": []},
    "Surge": {"zones": [GK_SURGE_ZONE], "extra_lines": []},
    "Surge Dunes": {"zones": [GK_SURGE_DUNES_ZONE], "extra_lines": []},
    "Fine-depleted flow": {"zones": [GK_FINE_DEPLETED_FLOW_ZONE], "extra_lines": []},
}


# =========================================================
# Pardo et al. (2009)
# Coordinate system: X = mean phi, Y = sorting/sigma phi.
# The geometry below is an editable vector reconstruction based on the
# provided Pardo reference images, using the same aesthetic approach as GK.
# =========================================================

PARDO_FALLOUT_ZONE = ZoneDef(
    name="Fallout",
    vertices=[
        (-4.15, 1.55), (-3.90, 1.95), (-3.30, 2.18), (-2.45, 2.38),
        (-1.45, 2.52), (-0.20, 2.55), (1.05, 2.35), (2.05, 2.10),
        (2.85, 1.92), (3.50, 1.85), (4.05, 1.82), (4.05, 1.45),
        (3.70, 0.95), (3.15, 0.42), (2.10, 0.12), (0.80, 0.10),
        (-0.55, 0.18), (-1.85, 0.32), (-2.85, 0.58), (-3.55, 0.95),
        (-4.05, 1.28),
    ],
    label_pos=(-3.10, 1.55),
    edgecolor="#595959",
    facecolor="#eeeeee",
    alpha=0.07,
    linewidth=2.00,
    linestyle=(0, (12, 7)),
    zorder=1,
)

PARDO_PYROCLASTIC_FLOW_ZONE = ZoneDef(
    name="Pyroclastic flow",
    vertices=[
        (-2.25, 3.42), (-1.95, 3.82), (-1.25, 4.05), (-0.30, 4.28),
        (0.75, 4.55), (1.35, 4.75), (2.10, 4.62), (2.70, 4.48),
        (3.10, 4.28), (3.45, 4.02), (3.80, 3.75), (4.10, 3.55),
        (4.58, 3.18), (5.00, 2.82), (5.42, 2.38), (5.78, 1.98),
        (5.95, 1.58), (5.78, 1.28), (5.35, 1.02), (4.85, 0.92),
        (4.15, 0.92), (3.20, 1.10), (2.25, 1.32), (1.35, 1.62),
        (0.40, 2.00), (-0.55, 2.32), (-1.30, 2.62), (-1.92, 2.95),
    ],
    label_pos=(1.55, 4.28),
    edgecolor="#4c4c4c",
    facecolor="#dcdcdc",
    alpha=0.11,
    linewidth=2.10,
    linestyle="-",
    zorder=5,
)

PARDO_PYROCLASTIC_SURGES_ZONE = ZoneDef(
    name="Pyroclastic surges",
    vertices=[
        (-2.30, 2.15), (-1.95, 2.48), (-1.10, 2.68), (-0.15, 2.82),
        (0.18, 3.18), (0.55, 3.78), (1.22, 3.72), (1.95, 3.42),
        (2.75, 3.20), (3.55, 3.08), (4.30, 2.98), (4.95, 2.86),
        (5.42, 2.48), (5.38, 1.92), (5.10, 1.28), (4.78, 0.82),
        (4.18, 0.65), (3.10, 0.60), (1.85, 0.58), (0.55, 0.62),
        (-0.55, 0.72), (-1.28, 0.98), (-1.78, 1.35), (-2.08, 1.78),
    ],
    label_pos=(2.80, 3.05),
    edgecolor="#555555",
    facecolor="#e7e7e7",
    alpha=0.08,
    linewidth=2.00,
    linestyle=":",
    zorder=4,
)

PARDO_ZONE_I = ZoneDef(
    name="Zone I",
    vertices=[
        (-3.65, 3.08), (-2.95, 3.18), (-2.05, 3.12), (-1.20, 2.98),
        (-0.48, 2.68), (0.02, 2.18), (0.18, 1.52), (0.15, 0.88),
        (-0.12, 0.50), (-0.92, 0.42), (-1.92, 0.45), (-2.82, 0.58),
        (-3.35, 0.88), (-3.58, 1.62),
    ],
    label_pos=(-3.18, 2.52),
    edgecolor="#8c8c8c",
    facecolor="#eeeeee",
    alpha=0.06,
    linewidth=1.50,
    linestyle=":",
    zorder=2,
)

PARDO_ZONE_II = ZoneDef(
    name="Zone II",
    vertices=[
        (0.15, 1.55), (0.82, 1.95), (1.65, 2.15), (2.55, 2.15),
        (3.35, 2.10), (4.05, 1.92), (4.55, 1.58), (4.75, 1.10),
        (4.60, 0.62), (4.10, 0.28), (3.28, 0.12), (2.30, 0.10),
        (1.30, 0.20), (0.55, 0.60), (0.18, 1.05),
    ],
    label_pos=(2.05, 1.95),
    edgecolor="#969696",
    facecolor="#eeeeee",
    alpha=0.06,
    linewidth=1.50,
    linestyle=":",
    zorder=2,
)

PARDO_ZONE_III = ZoneDef(
    name="Zone III",
    vertices=[
        (4.08, 1.62), (4.15, 2.18), (4.35, 2.85), (4.62, 3.22),
        (4.92, 3.18), (5.10, 2.78), (5.12, 2.18), (5.02, 1.55),
        (4.85, 1.08), (4.52, 0.98), (4.20, 1.18),
    ],
    label_pos=(4.72, 2.62),
    edgecolor="#a8a8a8",
    facecolor="#f4f4f4",
    alpha=0.05,
    linewidth=1.45,
    linestyle=":",
    zorder=2,
)

PARDO_ZONES = [
    PARDO_FALLOUT_ZONE,
    PARDO_PYROCLASTIC_FLOW_ZONE,
    PARDO_PYROCLASTIC_SURGES_ZONE,
    PARDO_ZONE_I,
    PARDO_ZONE_II,
    PARDO_ZONE_III,
]

PARDO_VECTOR_LIBRARY: Dict[str, Dict[str, Any]] = {
    "All": {
        "zones": PARDO_ZONES,
        "extra_lines": [],
    },
    "Fallout": {"zones": [PARDO_FALLOUT_ZONE], "extra_lines": []},
    "Pyroclastic flow": {"zones": [PARDO_PYROCLASTIC_FLOW_ZONE], "extra_lines": []},
    "Pyroclastic surges": {"zones": [PARDO_PYROCLASTIC_SURGES_ZONE], "extra_lines": []},
    "Zone I": {"zones": [PARDO_ZONE_I], "extra_lines": []},
    "Zone II": {"zones": [PARDO_ZONE_II], "extra_lines": []},
    "Zone III": {"zones": [PARDO_ZONE_III], "extra_lines": []},
}


WALKER_FALL_DEPOSIT_ZONE = ZoneDef(
    name="Fall deposit",
    vertices=[
        (-4.10, 1.55), (-3.55, 1.95), (-2.55, 2.28), (-1.45, 2.50),
        (-0.25, 2.55), (0.85, 2.38), (1.85, 2.15), (2.45, 1.92),
        (2.80, 1.55), (2.72, 0.92), (2.18, 0.38), (1.25, 0.12),
        (0.10, 0.10), (-1.20, 0.25), (-2.35, 0.48), (-3.25, 0.78),
        (-3.92, 1.15),
    ],
    label_pos=(-3.05, 1.55),
    edgecolor="#585858",
    facecolor="#eeeeee",
    alpha=0.07,
    linewidth=2.05,
    linestyle=(0, (14, 8)),
    zorder=1,
)

WALKER_FINE_DEPLETED_FLOW_ZONE = ZoneDef(
    name="Fine-depleted flow",
    vertices=[
        (-3.75, 3.08), (-3.15, 3.48), (-2.10, 3.68), (-1.05, 3.88),
        (-0.10, 3.95), (0.65, 3.62), (1.08, 3.20), (1.18, 2.72),
        (0.82, 2.28), (0.02, 2.08), (-0.95, 2.10), (-1.88, 2.18),
        (-2.62, 2.42), (-3.20, 2.72), (-3.65, 2.90),
    ],
    label_pos=(-2.95, 2.78),
    edgecolor="#6e6e6e",
    facecolor="#e8e8e8",
    alpha=0.08,
    linewidth=1.95,
    linestyle=(0, (6, 5, 1.5, 5)),
    zorder=3,
)

WALKER_PYROCLASTIC_SURGE_ZONE = ZoneDef(
    name="Pyroclastic surge",
    vertices=[
        (-2.20, 2.22), (-1.70, 2.52), (-0.90, 2.72), (-0.05, 2.92),
        (0.62, 3.12), (1.05, 3.82), (1.65, 3.62), (2.22, 3.38),
        (2.95, 3.18), (3.78, 3.05), (4.62, 2.90), (5.15, 2.58),
        (5.48, 2.15), (5.30, 1.68), (4.95, 1.08), (4.72, 0.72),
        (4.12, 0.55), (3.15, 0.48), (1.85, 0.48), (0.55, 0.58),
        (-0.65, 0.68), (-1.38, 0.98), (-1.92, 1.45), (-2.20, 1.92),
    ],
    label_pos=(3.85, 0.82),
    edgecolor="#505050",
    facecolor="#e2e2e2",
    alpha=0.09,
    linewidth=2.05,
    linestyle=":",
    zorder=4,
)

WALKER_PYROCLASTIC_FLOW_ZONE = ZoneDef(
    name="Pyroclastic flow",
    vertices=[
        (-2.28, 3.42), (-1.72, 3.92), (-0.82, 4.20), (0.18, 4.48),
        (0.98, 4.72), (1.62, 4.58), (2.08, 4.42), (2.45, 4.18),
        (2.75, 3.92), (3.10, 3.72), (3.78, 3.48), (4.38, 3.02),
        (4.92, 2.48), (5.42, 1.95), (5.85, 1.45), (5.82, 1.18),
        (5.40, 0.98), (4.78, 0.92), (3.95, 1.00), (3.05, 1.20),
        (2.18, 1.32), (1.32, 1.62), (0.45, 1.95), (-0.45, 2.28),
        (-1.25, 2.60), (-1.88, 2.92), (-2.18, 3.18),
    ],
    label_pos=(1.28, 4.18),
    edgecolor="#4a4a4a",
    facecolor="#d9d9d9",
    alpha=0.11,
    linewidth=2.15,
    linestyle="-",
    zorder=5,
)

WALKER_ZONES = [
    WALKER_FALL_DEPOSIT_ZONE,
    WALKER_FINE_DEPLETED_FLOW_ZONE,
    WALKER_PYROCLASTIC_SURGE_ZONE,
    WALKER_PYROCLASTIC_FLOW_ZONE,
]

WALKER_VECTOR_LIBRARY: Dict[str, Dict[str, Any]] = {
    "All": {"zones": WALKER_ZONES, "extra_lines": []},
    "Fall deposit": {"zones": [WALKER_FALL_DEPOSIT_ZONE], "extra_lines": []},
    "Fine-depleted flow": {"zones": [WALKER_FINE_DEPLETED_FLOW_ZONE], "extra_lines": []},
    "Pyroclastic surge": {"zones": [WALKER_PYROCLASTIC_SURGE_ZONE], "extra_lines": []},
    "Pyroclastic flow": {"zones": [WALKER_PYROCLASTIC_FLOW_ZONE], "extra_lines": []},
}


# =========================================================
# Pyroclastic grain-size ternary classification
# Vertices:
#   top        = blocks and bombs (> 64 mm)
#   lower left = lapilli (64-2 mm)
#   lower right= ash (< 2 mm)
#
# Cartesian coordinates are barycentric projections of the three end members.
# This is a true vector reconstruction: no raster background is used.
# =========================================================

TERNARY_H = 0.8660254037844386
TERNARY_TOP = (0.50, TERNARY_H)
TERNARY_LAPILLI = (0.00, 0.00)
TERNARY_ASH = (1.00, 0.00)

# Constant blocks+bombs boundaries.
TERNARY_B75_L = (0.375, 0.75 * TERNARY_H)
TERNARY_B75_R = (0.625, 0.75 * TERNARY_H)
TERNARY_B25_L = (0.125, 0.25 * TERNARY_H)
TERNARY_B25_R = (0.875, 0.25 * TERNARY_H)

# Base intersections at 75/25 lapilli/ash and 25/75 lapilli/ash.
TERNARY_BASE_25 = (0.25, 0.00)
TERNARY_BASE_75 = (0.75, 0.00)

TERNARY_PYROCLASTIC_BRECCIA_ZONE = ZoneDef(
    name="Pyroclastic breccia / agglomerate",
    vertices=[TERNARY_TOP, TERNARY_B75_R, TERNARY_B75_L],
    label_pos=(0.78, 0.765),
    edgecolor="#3f3f3f",
    facecolor="#f3f3f3",
    alpha=0.10,
    linewidth=1.10,
    linestyle="-",
    zorder=3,
    label_size=11,
)

TERNARY_TUFF_BRECCIA_ZONE = ZoneDef(
    name="Tuff breccia",
    vertices=[TERNARY_B75_L, TERNARY_B75_R, TERNARY_B25_R, TERNARY_B25_L],
    label_pos=(0.50, 0.435),
    edgecolor="#4b4b4b",
    facecolor="#e3e3e3",
    alpha=0.11,
    linewidth=1.05,
    linestyle="-",
    zorder=2,
    label_size=12,
)

TERNARY_LAPILLISTONE_ZONE = ZoneDef(
    name="Lapillistone",
    vertices=[TERNARY_LAPILLI, TERNARY_BASE_25, TERNARY_B25_L],
    label_pos=(-0.18, 0.145),
    edgecolor="#555555",
    facecolor="#f1f1f1",
    alpha=0.08,
    linewidth=1.00,
    linestyle="-",
    zorder=2,
    label_size=11,
)

TERNARY_LAPILLI_TUFF_ZONE = ZoneDef(
    name="Lapilli tuff",
    vertices=[TERNARY_B25_L, TERNARY_B25_R, TERNARY_BASE_75, TERNARY_BASE_25],
    label_pos=(0.50, 0.115),
    edgecolor="#555555",
    facecolor="#ececec",
    alpha=0.10,
    linewidth=1.00,
    linestyle="-",
    zorder=2,
    label_size=12,
)

TERNARY_TUFF_ASH_TUFF_ZONE = ZoneDef(
    name="Tuff or ash tuff",
    vertices=[TERNARY_BASE_75, TERNARY_ASH, TERNARY_B25_R],
    label_pos=(1.18, 0.145),
    edgecolor="#555555",
    facecolor="#f1f1f1",
    alpha=0.08,
    linewidth=1.00,
    linestyle="-",
    zorder=2,
    label_size=11,
)

TERNARY_ZONES = [
    TERNARY_PYROCLASTIC_BRECCIA_ZONE,
    TERNARY_TUFF_BRECCIA_ZONE,
    TERNARY_LAPILLISTONE_ZONE,
    TERNARY_LAPILLI_TUFF_ZONE,
    TERNARY_TUFF_ASH_TUFF_ZONE,
]

TERNARY_BOUNDARY_LINES = [
    {
        "xy": [TERNARY_LAPILLI, TERNARY_TOP, TERNARY_ASH, TERNARY_LAPILLI],
        "linewidth": 1.25,
        "linestyle": "-",
        "color": "#222222",
        "zorder": 8,
        "smooth_iterations": 0,
    },
    {
        "xy": [TERNARY_B75_L, TERNARY_B75_R],
        "linewidth": 0.90,
        "linestyle": "-",
        "color": "#666666",
        "zorder": 8,
        "smooth_iterations": 0,
    },
    {
        "xy": [TERNARY_B25_L, TERNARY_B25_R],
        "linewidth": 0.90,
        "linestyle": "-",
        "color": "#666666",
        "zorder": 8,
        "smooth_iterations": 0,
    },
    {
        "xy": [TERNARY_B25_L, TERNARY_BASE_25],
        "linewidth": 0.90,
        "linestyle": (0, (6, 5)),
        "color": "#666666",
        "zorder": 8,
        "smooth_iterations": 0,
    },
    {
        "xy": [TERNARY_B25_R, TERNARY_BASE_75],
        "linewidth": 0.90,
        "linestyle": (0, (6, 5)),
        "color": "#666666",
        "zorder": 8,
        "smooth_iterations": 0,
    },
]

TERNARY_VECTOR_LIBRARY: Dict[str, Dict[str, Any]] = {
    "All": {"zones": TERNARY_ZONES, "extra_lines": TERNARY_BOUNDARY_LINES},
    "Pyroclastic breccia / agglomerate": {"zones": [TERNARY_PYROCLASTIC_BRECCIA_ZONE], "extra_lines": []},
    "Tuff breccia": {"zones": [TERNARY_TUFF_BRECCIA_ZONE], "extra_lines": []},
    "Lapillistone": {"zones": [TERNARY_LAPILLISTONE_ZONE], "extra_lines": []},
    "Lapilli tuff": {"zones": [TERNARY_LAPILLI_TUFF_ZONE], "extra_lines": []},
    "Tuff or ash tuff": {"zones": [TERNARY_TUFF_ASH_TUFF_ZONE], "extra_lines": []},
}

# Text that belongs to the diagram geometry rather than to one editable field.
TERNARY_FIXED_TEXTS = [
    {"xy": (0.50, 0.995), "text": "blocks and bombs\n(> 64 mm)", "fontsize": 12, "ha": "center", "va": "center"},
    {"xy": (-0.035, -0.075), "text": "lapilli\n(64–2 mm)", "fontsize": 12, "ha": "center", "va": "center"},
    {"xy": (1.035, -0.075), "text": "ash\n(< 2 mm)", "fontsize": 12, "ha": "center", "va": "center"},

    {"xy": (0.348, 0.664), "text": "75", "fontsize": 9, "ha": "right", "va": "center"},
    {"xy": (0.652, 0.664), "text": "75", "fontsize": 9, "ha": "left", "va": "center"},
    {"xy": (0.104, 0.228), "text": "25", "fontsize": 9, "ha": "right", "va": "center"},
    {"xy": (0.896, 0.228), "text": "25", "fontsize": 9, "ha": "left", "va": "center"},
    {"xy": (0.25, -0.022), "text": "25", "fontsize": 9, "ha": "center", "va": "top"},
    {"xy": (0.75, -0.022), "text": "75", "fontsize": 9, "ha": "center", "va": "top"},
]

# Overrides keep the field names concise in the editor while reproducing the
# wording and arrow placement of the reference classification diagram.
TERNARY_LABEL_OVERRIDES = {
    "Pyroclastic breccia / agglomerate": "pyroclastic breccia (blocks)\nagglomerate (bombs)",
    "Tuff breccia": "tuff breccia",
    "Lapillistone": "lapillistone",
    "Lapilli tuff": "lapilli tuff",
    "Tuff or ash tuff": "tuff or ash tuff",
}

TERNARY_LABEL_ANNOTATIONS = {
    "Pyroclastic breccia / agglomerate": {
        "target": (0.51, 0.705),
        "ha": "left",
        "va": "center",
        "arrowstyle": "->",
        "arrow_color": "#222222",
        "arrow_linewidth": 0.95,
    },
    "Lapillistone": {
        "target": (0.145, 0.105),
        "ha": "right",
        "va": "center",
        "arrowstyle": "->",
        "arrow_color": "#222222",
        "arrow_linewidth": 0.95,
    },
    "Tuff or ash tuff": {
        "target": (0.855, 0.105),
        "ha": "left",
        "va": "center",
        "arrowstyle": "->",
        "arrow_color": "#222222",
        "arrow_linewidth": 0.95,
    },
}

VECTOR_DIAGRAMS: Dict[str, Dict[str, Any]] = {
    "gk": {
        "key": "gk",
        "title": "Gençalioğlu-Kuşcu et al. (2007)",
        "axis_limits": (-3.5, 5.5, 0.0, 4.2),
        "zones": GK_ZONES,
        "library": GK_VECTOR_LIBRARY,
        "field_options": ["Flow", "Fall", "Surge", "Surge Dunes", "Fine-depleted flow"],
        "view_options": list(GK_VECTOR_LIBRARY.keys()),
    },
    "walker": {
        "key": "walker",
        "title": "Walker (1971 and 1983)",
        "axis_limits": (-4.2, 6.0, 0.0, 5.0),
        "zones": WALKER_ZONES,
        "library": WALKER_VECTOR_LIBRARY,
        "field_options": ["Fall deposit", "Fine-depleted flow", "Pyroclastic surge", "Pyroclastic flow"],
        "view_options": list(WALKER_VECTOR_LIBRARY.keys()),
    },
    "pardo": {
        "key": "pardo",
        "title": "Pardo et al. (2009)",
        "axis_limits": (-4.2, 6.0, 0.0, 5.0),
        "zones": PARDO_ZONES,
        "library": PARDO_VECTOR_LIBRARY,
        "field_options": ["Fallout", "Pyroclastic flow", "Pyroclastic surges", "Zone I", "Zone II", "Zone III"],
        "view_options": list(PARDO_VECTOR_LIBRARY.keys()),
    },
    "pyroclastic_grain_size": {
        "key": "pyroclastic_grain_size",
        "title": "Pyroclastic grain-size triangle",
        "axis_limits": (-0.34, 1.42, -0.16, 1.07),
        "zones": TERNARY_ZONES,
        "library": TERNARY_VECTOR_LIBRARY,
        "field_options": [
            "Pyroclastic breccia / agglomerate",
            "Tuff breccia",
            "Lapillistone",
            "Lapilli tuff",
            "Tuff or ash tuff",
        ],
        "view_options": list(TERNARY_VECTOR_LIBRARY.keys()),
        "coordinate_mode": "ternary_grain_size",
        "smooth_iterations": 0,
        "hide_axes": True,
        "equal_aspect": True,
        "title_on_plot": False,
        "show_no_data_message": False,
        "fixed_texts": TERNARY_FIXED_TEXTS,
        "label_overrides": TERNARY_LABEL_OVERRIDES,
        "label_annotations": TERNARY_LABEL_ANNOTATIONS,
    },
}

DEFAULT_DIAGRAM_KEY = "gk"

# Backward-compatible names for the older GK-only integration.
GK_VECTOR_LIBRARY = GK_VECTOR_LIBRARY
GRAPHIC_VIEW_OPTIONS = list(GK_VECTOR_LIBRARY.keys())
GRAPHIC_FIELD_OPTIONS = ["Flow", "Fall", "Surge", "Surge Dunes", "Fine-depleted flow"]

__all__ = [
    "ZoneDef",
    "GK_ZONES",
    "PARDO_ZONES",
    "WALKER_ZONES",
    "TERNARY_ZONES",
    "GK_VECTOR_LIBRARY",
    "PARDO_VECTOR_LIBRARY",
    "WALKER_VECTOR_LIBRARY",
    "TERNARY_VECTOR_LIBRARY",
    "VECTOR_DIAGRAMS",
    "DEFAULT_DIAGRAM_KEY",
    "GRAPHIC_VIEW_OPTIONS",
    "GRAPHIC_FIELD_OPTIONS",
]
