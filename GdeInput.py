# INSTALL_NAME_REQUIRED: GdeInput.py
# Do not keep multiple GdeInput*.py files in the Componentry plugin directory.
# -*- coding: utf-8 -*-
"""
GdeInput.py
Complemento externo para Componentry Analysis.

Función:
    Agrega una ventana para generar archivos .dat desde datos cargados en
    Componentry y desde parámetros editados por el usuario.

Instalación:
    Instalar SIEMPRE con el nombre canónico GdeInput.py.
    No conservar GdeInput_V2.py, GdeInput_V3.py, etc. en la carpeta de complementos.

Requiere que el programa principal exponga:
    app.current_db()
    app.current_sample_name()
    app.add_plugin_menu_action(...)
    app.add_plugin_toolbar_button(...)
    app.add_plugin_window(...)
"""

import os
import re
import json
import sys
import platform
import subprocess
import unicodedata
import shlex
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QUrl, QProcess, QSettings, QObject, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QDesktopServices, QPixmap, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QComboBox, QLineEdit,
    QFileDialog, QMessageBox, QGroupBox, QTabWidget,
    QPlainTextEdit, QTableWidget, QTableWidgetItem,
    QListWidget, QAbstractItemView, QScrollArea, QSizePolicy, QSplitter
)


PLUGIN_NAME = "GDEINPUT"
PLUGIN_VERSION = "8.19-WSL-VISIBLE-RUN-DIR"
PACKAGE_BUILD_MARKER = "GDEINPUT_V8_19_20260915"

PROJECT_FILE_FORMAT = "GDEINPUT_PROJECT"
PROJECT_SCHEMA_VERSION = 1


# =========================================================
# PYFLOW runner: portado del launcher batch que ya funciona
# =========================================================
DEFAULT_WSL_DISTRO = "Ubuntu-22.04"
DEFAULT_PYFLOW_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "PYFLOW-v_2.5")
DEFAULT_PYFLOW_EXECUTABLE = "pyflow_2.5"
INACTIVITY_TIMEOUT_MS = 120000
MAX_CAPTURED_OUTPUT_CHARS = 500000

GENERIC_ENTER_PATTERNS = [
    r"\bOK\s+ENTER\b",
    r"press\s+enter",
    r"hit\s+enter",
    r"waiting\s+for\s+enter",
]

LOW_ACCURACY_STOP_PATTERNS = [
    r"very\s+low\s+accuracy\s+level\s+of\s+the\s+t\s*-\s*student\s+test",
    r'type\s+"?stop"?\s+if\s+you\s+want\s+to\s+stop\s+the\s+calculations',
    r"any\s+other\s+input\s+will\s+continue\s+the\s+calculations",
]

FATAL_PATTERNS = [
    r"fatal error!",
    r"pyflow is going to be stopped",
    r"not able to find a solution for the reference level",
]


def safe_decode(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")


def detect_binary_type(path: str) -> str:
    if not path or not os.path.isfile(path):
        return "missing"
    try:
        with open(path, "rb") as f:
            head = f.read(4)
        if head == b"\x7fELF":
            return "linux_elf"
        if head.startswith(b"MZ"):
            return "windows_pe"
        return "unknown"
    except Exception:
        return "unknown"


def win_to_wsl_path(win_path: str) -> str:
    p = os.path.abspath(win_path).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if not m:
        return p
    drive = m.group(1).lower()
    rest = m.group(2)
    return f"/mnt/{drive}/{rest}"


def bash_quote(s: str) -> str:
    return shlex.quote(s)


def list_wsl_distros() -> List[str]:
    try:
        out = subprocess.check_output(["wsl.exe", "-l", "-q"], stderr=subprocess.STDOUT)
        txt = safe_decode(out).replace("\x00", "")
        distros = []
        for line in txt.splitlines():
            name = line.strip().lstrip("*").strip()
            if name:
                distros.append(name)
        return distros
    except Exception:
        return []


def has_wsl() -> bool:
    try:
        subprocess.check_output(["wsl.exe", "-l", "-q"], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False


def path_latest_mtime(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    try:
        latest = os.path.getmtime(path)
    except Exception:
        latest = 0.0
    if os.path.isfile(path):
        return latest
    for root, dirs, files in os.walk(path):
        for name in dirs:
            p = os.path.join(root, name)
            try:
                latest = max(latest, os.path.getmtime(p))
            except Exception:
                pass
        for name in files:
            p = os.path.join(root, name)
            try:
                latest = max(latest, os.path.getmtime(p))
            except Exception:
                pass
    return latest

ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=\s*(.*?)\s*$",
    re.IGNORECASE
)

LOGICAL_OPTIONS = [".T.", ".F."]


class NoWheelComboBox(QComboBox):
    """QComboBox que nunca cambia de valor con la rueda del mouse.

    El evento se ignora para que pueda propagarse al QScrollArea padre.
    La selección se cambia únicamente haciendo clic y eligiendo una opción.
    """
    def wheelEvent(self, event):
        event.ignore()


MODEL_OPTIONS = ["TWOCOMPONENTS"]
INPUT_WEIGHT_OPTIONS = ["WT", "MASS"]

RHO_LAWS = [
    "CUSTOM", "POLLENA", "AVERNO2", "AMS", "POMPEI",
    "SIAL_XX", "FEM_XX", "LITHIC", "MERCATO", "ASTRONI"
]

DRAG_LAWS = [
    "SPHERE", "HAIDLEV", "SWAMOJ", "GANSER", "CHIEN", "TRANCONG",
    "DELLINO", "HOLZSOMM", "DIOGMELE", "DIOG2017", "DIOG2018"
]

FIELD_LABELS = {
    "MODEL": "Modelo de cálculo",
    "DEPRATES": "Calcular tasas de depositación",
    "DISTR1": "Usar distribución granulométrica del componente 1",
    "DISTR2": "Usar distribución granulométrica del componente 2",
    "INPUT_WEIGHT": "Tipo de datos de WEIGHT",

    "PROBT": "Nivel de significancia del test t-Student",
    "DOTESTCHI(1)": "Ejecutar test χ² para componente 1",
    "DOTESTCHI(2)": "Ejecutar test χ² para componente 2",
    "SIGLEVCHI(1)": "Nivel de significancia χ² del componente 1",
    "SIGLEVCHI(2)": "Nivel de significancia χ² del componente 2",
    "C0": "Concentración volumétrica de partículas en el nivel de referencia",
    "MU": "Viscosidad dinámica del gas",
    "DENGAS": "Densidad del gas",

    "ZLAM": "ZLAM — espesor de la capa laminada modelada",
    "ZLAMS": "ZLAMS — espesor de lámina de referencia, espesor de una lámina que conforma mi nivel / altura z0",
    "KS": "Rugosidad del sustrato",

    "RHOLAW(1)": "Ley de densidad del componente 1",
    "RHO_CUSTOM(1)": "Tipo de densidad CUSTOM del componente 1",
    "RHOS(1,0)": "Densidad de partículas del componente 1 (clase mediana)",
    "CDLAW(1)": "Ley de arrastre del componente 1",
    "SPHERICITY(1,0)": "Esfericidad 3D del componente 1",
    "COREY(1,0)": "Corey Shape Factor del componente 1",
    "VOLEQSPHD(1,0)": "Diámetro de esfera de volumen equivalente del componente 1",
    "CIRCEQARD(1,0)": "Diámetro de círculo de igual área proyectada del componente 1",
    "ISOMETRIC(1)": "Tratar el componente 1 como partícula isométrica en GANSER",
    "CIRCULARITY(1,0)": "Circularidad de Tran-Cong del componente 1 (Pc/P)",
    "FLATRATIO(1,0)": "Flatness ratio de Tran-Cong del componente 1",
    "SHAPEFACT(1,0)": "Shape factor Ψ de Dellino/Dioguardi del componente 1",
    "LONGSPHER(1,0)": "Esfericidad longitudinal del componente 1",
    "CROSSSPHER(1,0)": "Esfericidad transversal del componente 1",
    "FRACTAL(1)": "Usar dimensión fractal en DIOG2017 para componente 1",
    "FRACTDIM(1,0)": "Dimensión fractal 3D del componente 1",
    "DAVGEQSPH(1)": "DAVGEQSPH(1) — φ equivalente 3D del componente 1 (clase mediana)",
    "PHIMIN(1)": "Phi mínimo del componente 1 (clase más gruesa)",
    "PHIMAX(1)": "Phi máximo del componente 1 (clase más fina)",
    "DPHI(1)": "Intervalo entre clases phi del componente 1",
    "NCLASS(1)": "Número calculado de clases del componente 1",

    "RHOLAW(2)": "Ley de densidad del componente 2",
    "RHO_CUSTOM(2)": "Tipo de densidad CUSTOM del componente 2",
    "RHOS(2,0)": "Densidad de partículas del componente 2 (clase mediana)",
    "CDLAW(2)": "Ley de arrastre del componente 2",
    "SPHERICITY(2,0)": "Esfericidad 3D del componente 2",
    "COREY(2,0)": "Corey Shape Factor del componente 2",
    "VOLEQSPHD(2,0)": "Diámetro de esfera de volumen equivalente del componente 2",
    "CIRCEQARD(2,0)": "Diámetro de círculo de igual área proyectada del componente 2",
    "ISOMETRIC(2)": "Tratar el componente 2 como partícula isométrica en GANSER",
    "CIRCULARITY(2,0)": "Circularidad de Tran-Cong del componente 2 (Pc/P)",
    "FLATRATIO(2,0)": "Flatness ratio de Tran-Cong del componente 2",
    "SHAPEFACT(2,0)": "Shape factor Ψ de Dellino/Dioguardi del componente 2",
    "LONGSPHER(2,0)": "Esfericidad longitudinal del componente 2",
    "CROSSSPHER(2,0)": "Esfericidad transversal del componente 2",
    "FRACTAL(2)": "Usar dimensión fractal en DIOG2017 para componente 2",
    "FRACTDIM(2,0)": "Dimensión fractal 3D del componente 2",
    "DAVGEQSPH(2)": "DAVGEQSPH(2) — φ equivalente 3D del componente 2 (clase mediana)",
    "PHIMIN(2)": "Phi mínimo del componente 2 (clase más gruesa)",
    "PHIMAX(2)": "Phi máximo del componente 2 (clase más fina)",
    "DPHI(2)": "Intervalo entre clases phi del componente 2",
    "NCLASS(2)": "Número calculado de clases del componente 2",
}

# Unidades que PYFLOW 2.5 espera en input.dat. Son deliberadamente explícitas:
# el error RHOS=1.4 en lugar de 1400 surgió porque la interfaz anterior no
# mostraba la unidad exigida por PYFLOW.
FIELD_UNITS = {
    "PROBT": "-",
    "SIGLEVCHI(1)": "-",
    "SIGLEVCHI(2)": "-",
    "C0": "-",
    "MU": "Pa s",
    "DENGAS": "kg/m³",
    "ZLAM": "m",
    "ZLAMS": "m",
    "KS": "m",
    "RHOS(1,0)": "kg/m³",
    "RHOS(2,0)": "kg/m³",
    "SPHERICITY(1,0)": "-", "SPHERICITY(2,0)": "-",
    "COREY(1,0)": "-", "COREY(2,0)": "-",
    "VOLEQSPHD(1,0)": "m", "VOLEQSPHD(2,0)": "m",
    "CIRCEQARD(1,0)": "m", "CIRCEQARD(2,0)": "m",
    "CIRCULARITY(1,0)": "-", "CIRCULARITY(2,0)": "-",
    "FLATRATIO(1,0)": "-", "FLATRATIO(2,0)": "-",
    "SHAPEFACT(1,0)": "-", "SHAPEFACT(2,0)": "-",
    "LONGSPHER(1,0)": "-", "LONGSPHER(2,0)": "-",
    "CROSSSPHER(1,0)": "-", "CROSSSPHER(2,0)": "-",
    "FRACTDIM(1,0)": "-", "FRACTDIM(2,0)": "-",
    "DAVGEQSPH(1)": "φ",
    "DAVGEQSPH(2)": "φ",
    "PHIMIN(1)": "φ",
    "PHIMAX(1)": "φ",
    "DPHI(1)": "φ",
    "PHIMIN(2)": "φ",
    "PHIMAX(2)": "φ",
    "DPHI(2)": "φ",
}

FIELD_HELP = {
    "INPUT_WEIGHT": "Con WT, PYFLOW renormaliza internamente la distribución granulométrica de cada componente para que su suma sea 1 (100%) en el análisis TWOCOMPONENTS. No hay un keyword adicional para pedir esa normalización.",
    "RHOS(1,0)": "PYFLOW exige kg/m³. Ejemplo: 1.4 g/cm³ debe escribirse como 1400 kg/m³.",
    "RHOS(2,0)": "PYFLOW exige kg/m³. Ejemplo: 1.8 g/cm³ debe escribirse como 1800 kg/m³.",
    "ZLAM": "Espesor total de la capa laminada correspondiente a la oleada/corriente que se modela. Si esa capa laminada mide 4 m, ingrese 4.0. No es el espesor de la muestra ni el de una laminita. PYFLOW exige metros.",
    "ZLAMS": "Espesor de lámina/subcapa de referencia asociado a la altura basal z0 del perfil. Es un parámetro DISTINTO de ZLAM: ZLAM es el espesor total de la capa laminada modelada (por ejemplo 4.0 m); ZLAMS representa la escala basal/de referencia dentro de ella. Si se omite, PYFLOW 2.5 usa inicialmente 0.01 m (1 cm).",
    "KS": "PYFLOW exige metros. Ejemplo: 1 cm debe escribirse como 0.01 m.",
    "CDLAW(1)": "Ley de arrastre del componente 1. Se configura de forma independiente.",
    "CDLAW(2)": "Ley de arrastre del componente 2. Se configura de forma independiente.",
    "SPHERICITY(1,0)": "Esfericidad 3D. Se usa en HAIDLEV, CHIEN, HOLZSOMM y DIOG2017 con FRACTAL=.F.; GANSER la usa si ISOMETRIC=.T.",
    "SPHERICITY(2,0)": "Esfericidad 3D. Se usa en HAIDLEV, CHIEN, HOLZSOMM y DIOG2017 con FRACTAL=.F.; GANSER la usa si ISOMETRIC=.T.",
    "COREY(1,0)": "Corey Shape Factor para SWAMOJ: Ds/sqrt(Dl*Dm).",
    "COREY(2,0)": "Corey Shape Factor para SWAMOJ: Ds/sqrt(Dl*Dm).",
    "CIRCULARITY(1,0)": "Circularidad que PYFLOW pasa a TRANCONG. Convención Pc/P, no P/Pc de Dellino.",
    "CIRCULARITY(2,0)": "Circularidad que PYFLOW pasa a TRANCONG. Convención Pc/P, no P/Pc de Dellino.",
    "ISOMETRIC(1)": "GANSER: .T. usa SPHERICITY; .F. usa VOLEQSPHD y CIRCEQARD.",
    "ISOMETRIC(2)": "GANSER: .T. usa SPHERICITY; .F. usa VOLEQSPHD y CIRCEQARD.",
    "FRACTAL(1)": "DIOG2017: .F. usa SPHERICITY; .T. usa FRACTDIM.",
    "FRACTAL(2)": "DIOG2017: .F. usa SPHERICITY; .T. usa FRACTDIM.",
    "DAVGEQSPH(1)": "Pegá aquí el DAVGEQSPH que entrega el programa 3D para la clase mediana. Es φ equivalente: φ=-log2(d_eq en mm). NO es el φ50 del tamiz.",
    "DAVGEQSPH(2)": "Pegá aquí el DAVGEQSPH que entrega el programa 3D para la clase mediana. Es φ equivalente: φ=-log2(d_eq en mm). NO es el φ50 del tamiz.",
}

DEFAULTS = {
    "MODEL": "TWOCOMPONENTS",
    "DEPRATES": ".F.",
    "DISTR1": ".T.",
    "DISTR2": ".T.",
    "INPUT_WEIGHT": "WT",

    "PROBT": "0.05",
    "DOTESTCHI(1)": ".T.",
    "DOTESTCHI(2)": ".T.",
    "SIGLEVCHI(1)": "0.05",
    "SIGLEVCHI(2)": "0.05",
    "C0": "0.75",
    "MU": "2.d-5",
    "DENGAS": "0.38",

    "ZLAM": "",
    "ZLAMS": "",
    "KS": "0.01",

    "RHOLAW(1)": "CUSTOM",
    "RHOLAW(2)": "CUSTOM",
    "RHO_CUSTOM(1)": "CONSTANT",
    "RHO_CUSTOM(2)": "CONSTANT",
    "RHOS(1,0)": "",
    "RHOS(2,0)": "",
    # Valor por defecto de GDEINPUT.
    "CDLAW(1)": "DIOGMELE",
    "CDLAW(2)": "DIOGMELE",
    "SPHERICITY(1,0)": "", "SPHERICITY(2,0)": "",
    "COREY(1,0)": "", "COREY(2,0)": "",
    "VOLEQSPHD(1,0)": "", "VOLEQSPHD(2,0)": "",
    "CIRCEQARD(1,0)": "", "CIRCEQARD(2,0)": "",
    "ISOMETRIC(1)": ".F.", "ISOMETRIC(2)": ".F.",
    "CIRCULARITY(1,0)": "", "CIRCULARITY(2,0)": "",
    "FLATRATIO(1,0)": "", "FLATRATIO(2,0)": "",
    "SHAPEFACT(1,0)": "", "SHAPEFACT(2,0)": "",
    "LONGSPHER(1,0)": "", "LONGSPHER(2,0)": "",
    "CROSSSPHER(1,0)": "", "CROSSSPHER(2,0)": "",
    "FRACTAL(1)": ".F.", "FRACTAL(2)": ".F.",
    "FRACTDIM(1,0)": "", "FRACTDIM(2,0)": "",
    "DAVGEQSPH(1)": "",
    "DAVGEQSPH(2)": "",

    "PHIMIN(1)": "-3.5",
    "PHIMAX(1)": "4.5",
    "DPHI(1)": "0.5",
    "NCLASS(1)": "17",
    "PHIMIN(2)": "-3.5",
    "PHIMAX(2)": "4.5",
    "DPHI(2)": "0.5",
    "NCLASS(2)": "17",
}

TEXT_KEYWORDS = {
    "MODEL", "INPUT_WEIGHT",
    "RHOLAW(1)", "RHOLAW(2)",
    "RHO_CUSTOM(1)", "RHO_CUSTOM(2)",
    "CDLAW(1)", "CDLAW(2)",
}

LOGICAL_KEYWORDS = {
    "DEPRATES", "DISTR1", "DISTR2", "DOTESTCHI(1)", "DOTESTCHI(2)",
    "ISOMETRIC(1)", "ISOMETRIC(2)", "FRACTAL(1)", "FRACTAL(2)"
}

NUMERIC_KEYWORDS = {
    "PROBT", "SIGLEVCHI(1)", "SIGLEVCHI(2)",
    "C0", "MU", "DENGAS",
    "ZLAM", "ZLAMS", "KS",
    "RHOS(1,0)", "RHOS(2,0)",
    "SPHERICITY(1,0)", "SPHERICITY(2,0)",
    "COREY(1,0)", "COREY(2,0)",
    "VOLEQSPHD(1,0)", "VOLEQSPHD(2,0)",
    "CIRCEQARD(1,0)", "CIRCEQARD(2,0)",
    "CIRCULARITY(1,0)", "CIRCULARITY(2,0)",
    "FLATRATIO(1,0)", "FLATRATIO(2,0)",
    "SHAPEFACT(1,0)", "SHAPEFACT(2,0)",
    "LONGSPHER(1,0)", "LONGSPHER(2,0)",
    "CROSSSPHER(1,0)", "CROSSSPHER(2,0)",
    "FRACTDIM(1,0)", "FRACTDIM(2,0)",
    "DAVGEQSPH(1)", "DAVGEQSPH(2)",
    "PHIMIN(1)", "PHIMAX(1)", "DPHI(1)", "NCLASS(1)",
    "PHIMIN(2)", "PHIMAX(2)", "DPHI(2)", "NCLASS(2)",
}

# ZLAMS es opcional. DAVGEQSPH se ingresa manualmente cuando existe medición 3D; RHOS y los descriptores se validan de forma
# condicional de acuerdo con RHOLAW y CDLAW.
REQUIRED_BASE_KEYWORDS = [
    "MODEL", "DEPRATES", "DISTR1", "DISTR2", "INPUT_WEIGHT",
    "PROBT", "DOTESTCHI(1)", "DOTESTCHI(2)", "SIGLEVCHI(1)", "SIGLEVCHI(2)",
    "C0", "MU", "DENGAS", "ZLAM", "KS",
    "RHOLAW(1)", "CDLAW(1)", "PHIMIN(1)", "PHIMAX(1)", "DPHI(1)",
    "RHOLAW(2)", "CDLAW(2)", "PHIMIN(2)", "PHIMAX(2)", "DPHI(2)",
]

SECTIONS = [
    (
        "Depósito y flujo — datos que carga el usuario",
        ["ZLAM", "KS"]
    ),
    (
        "Componente 1 — datos que carga el usuario",
        [
            "RHOLAW(1)", "RHOS(1,0)",
            "CDLAW(1)",
            "SPHERICITY(1,0)", "COREY(1,0)",
            "VOLEQSPHD(1,0)", "CIRCEQARD(1,0)", "ISOMETRIC(1)",
            "CIRCULARITY(1,0)", "FLATRATIO(1,0)",
            "SHAPEFACT(1,0)", "LONGSPHER(1,0)", "CROSSSPHER(1,0)",
            "FRACTAL(1)", "FRACTDIM(1,0)",
            "DAVGEQSPH(1)",
            "PHIMIN(1)", "PHIMAX(1)", "DPHI(1)", "NCLASS(1)",
        ]
    ),
    (
        "Componente 2 — datos que carga el usuario",
        [
            "RHOLAW(2)", "RHOS(2,0)",
            "CDLAW(2)",
            "SPHERICITY(2,0)", "COREY(2,0)",
            "VOLEQSPHD(2,0)", "CIRCEQARD(2,0)", "ISOMETRIC(2)",
            "CIRCULARITY(2,0)", "FLATRATIO(2,0)",
            "SHAPEFACT(2,0)", "LONGSPHER(2,0)", "CROSSSPHER(2,0)",
            "FRACTAL(2)", "FRACTDIM(2,0)",
            "DAVGEQSPH(2)",
            "PHIMIN(2)", "PHIMAX(2)", "DPHI(2)", "NCLASS(2)",
        ]
    ),
]

SHAPEFACTOR_DRAG_LAWS = {"DELLINO", "DIOGMELE", "DIOG2018"}
SHAPE_DESCRIPTOR_BASES = {
    "SPHERICITY", "COREY", "VOLEQSPHD", "CIRCEQARD", "CIRCULARITY",
    "FLATRATIO", "SHAPEFACT", "LONGSPHER", "CROSSSPHER", "FRACTDIM"
}

def drag_fields_for_law(cdlaw, idx, isometric=False, fractal=False):
    cdlaw = str(cdlaw).upper()
    if cdlaw == "SPHERE": return []
    if cdlaw == "HAIDLEV": return [f"SPHERICITY({idx},0)"]
    if cdlaw == "SWAMOJ": return [f"COREY({idx},0)"]
    if cdlaw == "GANSER":
        return [f"ISOMETRIC({idx})", f"SPHERICITY({idx},0)"] if isometric else [f"ISOMETRIC({idx})", f"VOLEQSPHD({idx},0)", f"CIRCEQARD({idx},0)"]
    if cdlaw == "CHIEN": return [f"SPHERICITY({idx},0)"]
    if cdlaw == "TRANCONG": return [f"CIRCULARITY({idx},0)", f"FLATRATIO({idx},0)"]
    if cdlaw in SHAPEFACTOR_DRAG_LAWS: return [f"SHAPEFACT({idx},0)"]
    if cdlaw == "HOLZSOMM": return [f"SPHERICITY({idx},0)", f"LONGSPHER({idx},0)", f"CROSSSPHER({idx},0)"]
    if cdlaw == "DIOG2017":
        return [f"FRACTAL({idx})", f"FRACTDIM({idx},0)"] if fractal else [f"FRACTAL({idx})", f"SPHERICITY({idx},0)"]
    return []


class PyflowRunner(QObject):
    output_text = pyqtSignal(str)
    finished = pyqtSignal(int, str)
    prompt_needed = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        self.proc = QProcess()
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_ready_read_stdout)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error_occurred)

        self._prompt_lock = False
        self._wsl_script_path: Optional[str] = None
        self._current_output_dir: Optional[str] = None
        self._run_started_ts: float = 0.0
        self._had_fatal_error: bool = False
        self._fatal_messages: List[str] = []
        self._last_results_dir: Optional[str] = None
        self._captured_output: str = ""
        self._recent_text: str = ""
        self._current_distro: str = ""
        self._pid_file_path: Optional[str] = None
        self._after_stop_mode = False
        self._timed_out = False
        self._fatal_kill_scheduled = False

        self._force_kill_timer = QTimer(self)
        self._force_kill_timer.setSingleShot(True)
        self._force_kill_timer.timeout.connect(self._force_kill_after_stop_timeout)

        self._inactivity_timer = QTimer(self)
        self._inactivity_timer.setSingleShot(True)
        self._inactivity_timer.timeout.connect(self._on_inactivity_timeout)

    @property
    def current_output_dir(self) -> Optional[str]:
        return self._current_output_dir

    @property
    def had_fatal_error(self) -> bool:
        return self._had_fatal_error

    @property
    def fatal_messages(self) -> List[str]:
        return list(self._fatal_messages)

    @property
    def last_results_dir(self) -> Optional[str]:
        return self._last_results_dir

    @property
    def captured_output(self) -> str:
        return self._captured_output

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    def is_running(self) -> bool:
        return self.proc.state() != QProcess.NotRunning

    def _cleanup_wsl_script(self):
        if self._wsl_script_path and os.path.isfile(self._wsl_script_path):
            try:
                os.remove(self._wsl_script_path)
            except Exception:
                pass

        self._wsl_script_path = None

    def _cleanup_previous_results(self, output_dir: str) -> Tuple[bool, str]:
        results_dir = os.path.join(output_dir, "results")

        try:
            if os.path.isdir(results_dir):
                shutil.rmtree(results_dir)

            return True, ""

        except Exception as e:
            return False, f"No pude borrar la carpeta results anterior:\n{e}"

    def _prepare_input(self, input_file: str, output_dir: str) -> Tuple[bool, str, str]:
        target = os.path.join(output_dir, "input.dat")

        try:
            shutil.copy2(input_file, target)
            return True, target, ""

        except Exception as e:
            return False, "", f"No pude preparar input.dat:\n{e}"

    def _pick_wsl_distro(self) -> str:
        distros = list_wsl_distros()

        if not distros:
            return ""

        if DEFAULT_WSL_DISTRO in distros:
            return DEFAULT_WSL_DISTRO

        return distros[0]

    def _kill_wsl_child_by_pidfile(self):
        if not self._pid_file_path or not os.path.isfile(self._pid_file_path):
            return

        pid = ""

        try:
            with open(self._pid_file_path, "r", encoding="utf-8", errors="replace") as f:
                pid = f.read().strip()
        except Exception:
            pid = ""

        if not pid or not pid.isdigit():
            return

        args = []

        if self._current_distro:
            args += ["--distribution", self._current_distro]

        args += [
            "--",
            "bash",
            "-lc",
            f"kill -TERM {pid} 2>/dev/null || true; sleep 1; kill -KILL {pid} 2>/dev/null || true",
        ]

        try:
            subprocess.Popen(["wsl.exe"] + args)
        except Exception:
            pass

    def _create_wsl_runner_script(
        self,
        exe_path: str,
        input_dat_path: str,
        output_dir: str,
    ) -> Tuple[str, List[str]]:
        wsl_exec = win_to_wsl_path(exe_path)
        wsl_input = win_to_wsl_path(input_dat_path)
        wsl_output = win_to_wsl_path(output_dir)
        exec_name = os.path.basename(exe_path)

        fd, script_path = tempfile.mkstemp(
            prefix="__pyflow_wsl__",
            suffix=".sh",
            dir=output_dir,
            text=True,
        )
        os.close(fd)

        pid_file_win = os.path.join(output_dir, ".pyflow_child.pid")
        self._pid_file_path = pid_file_win
        pid_file_wsl = win_to_wsl_path(pid_file_win)

        if os.path.isfile(pid_file_win):
            try:
                os.remove(pid_file_win)
            except Exception:
                pass

        script_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "export LC_ALL=C.UTF-8",
            "export LANG=C.UTF-8",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            f"EXE_SRC={bash_quote(wsl_exec)}",
            f"IN_SRC={bash_quote(wsl_input)}",
            f"OUT_DIR={bash_quote(wsl_output)}",
            f"EXE_NAME={bash_quote(exec_name)}",
            f"PID_FILE={bash_quote(pid_file_wsl)}",
            'STAGE_DIR="/tmp/pyflow_launcher_${PPID}_$$"',
            'STDIN_FIFO="$STAGE_DIR/.stdin_fifo"',
            'CHILD_PID=""',
            'FEED_PID=""',
            'if [ ! -f "$EXE_SRC" ]; then',
            '  echo "[WSL][ERROR] No existe el ejecutable dentro de WSL: $EXE_SRC" >&2',
            '  exit 11',
            'fi',
            'if [ ! -f "$IN_SRC" ]; then',
            '  echo "[WSL][ERROR] No existe input.dat dentro de WSL: $IN_SRC" >&2',
            '  exit 12',
            'fi',
            'cleanup_and_exit() {',
            '  local rc="${1:-0}"',
            '  if [ -n "${FEED_PID:-}" ]; then',
            '    kill "${FEED_PID}" 2>/dev/null || true',
            '  fi',
            '  if [ -n "${CHILD_PID:-}" ]; then',
            '    kill -TERM "${CHILD_PID}" 2>/dev/null || true',
            '    sleep 1',
            '    kill -KILL "${CHILD_PID}" 2>/dev/null || true',
            '  fi',
            '  rm -f -- "$PID_FILE" 2>/dev/null || true',
            '  rm -f -- "$STDIN_FIFO" 2>/dev/null || true',
            '  if [ -n "${OUT_DIR:-}" ]; then',
            '    mkdir -p -- "$OUT_DIR"',
            '    shopt -s dotglob nullglob',
            '    for item in "$STAGE_DIR"/*; do',
            '      name="$(basename -- "$item")"',
            '      if [ "$name" = "$EXE_NAME" ]; then',
            '        continue',
            '      fi',
            '      if [ "$name" = ".stdin_fifo" ]; then',
            '        continue',
            '      fi',
            '      cp -a -- "$item" "$OUT_DIR/" 2>/dev/null || true',
            '    done',
            '  fi',
            '  rm -rf -- "$STAGE_DIR" 2>/dev/null || true',
            '  exit "$rc"',
            '}',
            'on_term() {',
            '  if [ -n "${CHILD_PID:-}" ]; then',
            '    kill -TERM "${CHILD_PID}" 2>/dev/null || true',
            '  fi',
            '  if [ -n "${FEED_PID:-}" ]; then',
            '    kill "${FEED_PID}" 2>/dev/null || true',
            '  fi',
            '}',
            "trap 'on_term' INT TERM HUP",
            "trap 'rc=$?; trap - EXIT; cleanup_and_exit \"$rc\"' EXIT",
            'rm -rf -- "$STAGE_DIR"',
            'mkdir -p -- "$STAGE_DIR"',
            'cp -- "$EXE_SRC" "$STAGE_DIR/$EXE_NAME"',
            'cp -- "$IN_SRC" "$STAGE_DIR/input.dat"',
            'chmod +x -- "$STAGE_DIR/$EXE_NAME"',
            'mkfifo "$STDIN_FIFO"',
            'cd "$STAGE_DIR"',
            '( cat || true ) > "$STDIN_FIFO" &',
            'FEED_PID=$!',
            'set +e',
            '"./$EXE_NAME" < "$STDIN_FIFO" &',
            'CHILD_PID=$!',
            'echo "$CHILD_PID" > "$PID_FILE"',
            'wait "$CHILD_PID"',
            'RC=$?',
            'set -e',
            'wait "$FEED_PID" 2>/dev/null || true',
            'exit "$RC"',
        ]

        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(script_lines) + "\n")

        self._wsl_script_path = script_path
        distro = self._pick_wsl_distro()
        self._current_distro = distro

        args: List[str] = []

        if distro:
            args += ["--distribution", distro]

        args += ["--", "bash", win_to_wsl_path(script_path)]

        return "wsl.exe", args

    def _find_results_dir_for_current_run(self) -> Optional[str]:
        if not self._current_output_dir:
            return None

        base_dir = self._current_output_dir
        res = os.path.join(base_dir, "results")

        if not os.path.isdir(res):
            return None

        threshold = self._run_started_ts - 1.0

        if path_latest_mtime(res) < threshold:
            return None

        slope_dirs = []

        try:
            for name in os.listdir(res):
                p = os.path.join(res, name)

                if os.path.isdir(p) and name.lower().startswith("slope_"):
                    slope_dirs.append(p)

        except Exception:
            slope_dirs = []

        valid_slope_dirs = [p for p in slope_dirs if path_latest_mtime(p) >= threshold]

        if valid_slope_dirs:
            valid_slope_dirs.sort(key=lambda p: path_latest_mtime(p), reverse=True)
            return valid_slope_dirs[0]

        return res

    def _drain_process_output(self):
        try:
            out = bytes(self.proc.readAllStandardOutput())

            if out:
                self._handle_text(safe_decode(out))

        except Exception:
            pass

    def _reset_inactivity_timer(self):
        if self.is_running():
            self._inactivity_timer.start(INACTIVITY_TIMEOUT_MS)

    def _detect_prompt_kind(self) -> Optional[str]:
        txt = self._recent_text.lower()

        low_hits = 0

        for pat in LOW_ACCURACY_STOP_PATTERNS:
            if re.search(pat, txt, flags=re.IGNORECASE):
                low_hits += 1

        if low_hits >= 2:
            return "low_accuracy_stop"

        for pat in GENERIC_ENTER_PATTERNS:
            if re.search(pat, txt, flags=re.IGNORECASE):
                return "enter"

        return None

    def _emit_prompt_if_needed(self):
        if self._prompt_lock:
            return

        kind = self._detect_prompt_kind()

        if kind is not None:
            self._prompt_lock = True
            self.prompt_needed.emit(kind, self._recent_text[-2500:].strip())

    def _schedule_fatal_kill_once(self):
        if self._fatal_kill_scheduled:
            return

        self._fatal_kill_scheduled = True

        self.output_text.emit(
            "\n[FATAL] Corrida inválida: PYFLOW no pudo resolver el caso o entró en error repetitivo. "
            "Se mata esta corrida y se libera el slot.\n"
        )

        QTimer.singleShot(100, self.hard_kill)

    def _handle_text(self, txt: str):
        if not txt:
            return

        self.output_text.emit(txt)

        self._captured_output += txt

        if len(self._captured_output) > MAX_CAPTURED_OUTPUT_CHARS:
            self._captured_output = self._captured_output[-MAX_CAPTURED_OUTPUT_CHARS:]

        self._recent_text = (self._recent_text + txt)[-12000:]
        self._reset_inactivity_timer()

        if self._after_stop_mode:
            self._force_kill_timer.start(8000)

        low = txt.lower()

        for pat in FATAL_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                self._had_fatal_error = True

                cleaned = txt.strip()

                if cleaned and cleaned not in self._fatal_messages:
                    self._fatal_messages.append(cleaned)

                self._schedule_fatal_kill_once()
                break

        self._emit_prompt_if_needed()

    def start(self, exe_path: str, input_file: str, work_dir: str) -> Tuple[bool, str]:
        if self.is_running():
            return False, "Ya hay un proceso corriendo."

        if not exe_path or not os.path.isfile(exe_path):
            return False, "El ejecutable no existe."

        if not input_file or not os.path.isfile(input_file):
            return False, "El archivo de input no existe."

        if not work_dir:
            return False, "La carpeta temporal de trabajo no existe."

        exe_path = os.path.abspath(exe_path)
        input_file = os.path.abspath(input_file)
        output_dir = os.path.abspath(work_dir)
        os.makedirs(output_dir, exist_ok=True)

        self._current_output_dir = output_dir
        self._run_started_ts = 0.0
        self._had_fatal_error = False
        self._fatal_messages = []
        self._last_results_dir = None
        self._cleanup_wsl_script()
        self._captured_output = ""
        self._recent_text = ""
        self._prompt_lock = False
        self._after_stop_mode = False
        self._force_kill_timer.stop()
        self._inactivity_timer.stop()
        self._current_distro = ""
        self._pid_file_path = None
        self._timed_out = False
        self._fatal_kill_scheduled = False

        ok, err = self._cleanup_previous_results(output_dir)

        if not ok:
            return False, err

        ok, input_dat_path, err = self._prepare_input(input_file, output_dir)

        if not ok:
            return False, err

        btype = detect_binary_type(exe_path)

        if btype == "missing":
            return False, "No encontré el ejecutable."

        if btype != "linux_elf":
            return False, (
                "Este launcher está hecho únicamente para el ejecutable Linux ELF de PYFLOW.\n\n"
                f"Tipo detectado: {btype}\n\n"
                "Ese archivo no corresponde al formato esperado para correrlo vía WSL."
            )

        if not has_wsl():
            return False, (
                "Detecté el ejecutable Linux ELF de PYFLOW, pero WSL no está disponible.\n"
                "Instalá/configurá WSL antes de ejecutarlo."
            )

        distro = self._pick_wsl_distro()

        if not distro:
            return False, (
                "WSL está instalado, pero no encontré ninguna distro disponible.\n"
                "Verificá `wsl -l -q`."
            )

        self._run_started_ts = datetime.now().timestamp()
        program, wsl_args = self._create_wsl_runner_script(exe_path, input_dat_path, output_dir)

        self.proc.setWorkingDirectory(output_dir)
        self.proc.start(program, wsl_args)

        if not self.proc.waitForStarted(5000):
            self._cleanup_wsl_script()
            return False, f"No pude iniciar WSL:\n{self.proc.errorString()}"

        self.output_text.emit(f"[INFO] Corrida iniciada: {os.path.basename(input_file)}\n")
        self._reset_inactivity_timer()

        return True, ""

    def stop(self):
        if not self.is_running():
            return

        self.output_text.emit("[INFO] Deteniendo...\n")
        self._kill_wsl_child_by_pidfile()
        self.proc.terminate()
        QTimer.singleShot(2500, self._kill_if_needed)

    def hard_kill(self):
        if not self.is_running():
            return

        self.output_text.emit("[INFO] Forzando cierre del proceso...\n")
        self._kill_wsl_child_by_pidfile()

        try:
            self.proc.kill()

        except Exception:
            pass

    def shutdown_for_app_close(self):
        if not self.is_running():
            return

        self.send_stop()
        self._kill_wsl_child_by_pidfile()
        self.proc.terminate()
        self.proc.waitForFinished(3000)

        if self.is_running():
            self._kill_wsl_child_by_pidfile()
            self.proc.kill()
            self.proc.waitForFinished(2000)

    def send_text(self, text: str):
        if not self.is_running():
            return

        try:
            self.proc.write(text.encode("utf-8", errors="ignore"))
            self.proc.waitForBytesWritten(1000)

        except Exception:
            pass

        self._prompt_lock = False
        self._recent_text = ""
        self._reset_inactivity_timer()

    def send_enter(self):
        self.send_text("\n")

    def send_stop(self):
        self._after_stop_mode = True
        self._force_kill_timer.start(8000)
        self.send_text("stop\n")

    def _kill_if_needed(self):
        if self.is_running():
            self.output_text.emit("[INFO] El proceso no respondió; aplicando kill().\n")
            self._kill_wsl_child_by_pidfile()
            self.proc.kill()

    def _force_kill_after_stop_timeout(self):
        if not self.is_running():
            return

        self.output_text.emit("[INFO] Se envió STOP pero el proceso siguió colgado. Se fuerza cierre.\n")
        self.hard_kill()

    def _on_inactivity_timeout(self):
        if not self.is_running():
            return

        self._timed_out = True
        self._had_fatal_error = True

        msg = (
            f"Timeout por inactividad: no hubo salida durante {INACTIVITY_TIMEOUT_MS // 1000} s. "
            "La corrida se considera colgada o inválida."
        )

        if msg not in self._fatal_messages:
            self._fatal_messages.append(msg)

        self.output_text.emit(f"\n[ERROR] {msg}\n")
        self.hard_kill()

    def _on_ready_read_stdout(self):
        data = bytes(self.proc.readAllStandardOutput())
        self._handle_text(safe_decode(data))

    def _on_error_occurred(self, _err):
        self._drain_process_output()
        self.output_text.emit(f"\n[ERROR] QProcess: {self.proc.errorString()}\n")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
        self._drain_process_output()
        self._prompt_lock = False
        self._cleanup_wsl_script()
        self._after_stop_mode = False
        self._force_kill_timer.stop()
        self._inactivity_timer.stop()
        self._last_results_dir = self._find_results_dir_for_current_run()

        status_str = "NormalExit" if exit_status == QProcess.NormalExit else "CrashExit"
        self.finished.emit(exit_code, status_str)


# =========================================================
# Plugin entry point
# =========================================================
def create_plugin(app):
    return GdeInputPlugin(app)


class GdeInputPlugin:
    name = PLUGIN_NAME
    version = PLUGIN_VERSION

    def __init__(self, app):
        self.app = app
        self.windows = []
        self.enabled = False

    def _is_componentry_host(self):
        """
        GDEINPUT is a Componentry plugin. It must register itself only when
        the host application is the ComponentryPreview window.

        FW-Granulometry also exposes compatibility methods named
        add_plugin_menu_action/add_plugin_toolbar_button, but that does not
        make it the correct host for this plugin.
        """
        try:
            if self.app.__class__.__name__ == "ComponentryPreview":
                return True
        except Exception:
            pass

        try:
            get_context = getattr(self.app, "get_plugin_context", None)
            if callable(get_context):
                ctx = get_context()
                if isinstance(ctx, dict):
                    return (
                        "db" in ctx
                        and "sample_dbs" in ctx
                        and "global_components" in ctx
                        and "fw_main_window" not in ctx
                    )
        except Exception:
            pass

        return False

    def setup(self):
        if not self._is_componentry_host():
            # Wrong host: do not add menu actions, toolbar buttons, dialogs,
            # or dock widgets to FW-Granulometry's main window.
            return

        self.enabled = True

        self.app.add_plugin_menu_action(
            plugin_name=self.name,
            action_name="Generar input .dat",
            callback=self.open_generator,
            tooltip="Genera archivos .dat desde la muestra actual."
        )

        self.app.add_plugin_toolbar_button(
            plugin_name=self.name,
            text="Generar input .dat",
            callback=self.open_generator,
            tooltip="Genera archivos .dat desde la muestra actual."
        )

    def teardown(self):
        for win in list(self.windows):
            try:
                win.close()
                win.deleteLater()
            except Exception:
                pass
        self.windows = []

    def open_generator(self):
        if not self.enabled or not self._is_componentry_host():
            return

        dlg = GdeInputDialog(self.app, self)
        self.windows.append(dlg)
        self.app.add_plugin_window(self.name, dlg, show=True)


# =========================================================
# Dialog
# =========================================================
class GdeInputDialog(QDialog):
    def __init__(self, app, plugin):
        super().__init__(app)

        self.app = app
        self.plugin = plugin

        self.setWindowTitle("GDEINPUT V8.19 — PYFLOW 2.5 / ejecución WSL robusta")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 850)
        self.setSizeGripEnabled(True)

        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        self.setFont(QFont("Arial", 10))

        self.param_widgets = {}
        self.param_labels = {}
        self.weight_widgets = {1: {}, 2: {}}
        self.weight_tables = {}
        self.weight_phi = {1: [], 2: []}
        self.last_preview_text = ""
        self.sample_metadata = {}
        self.sample_metadata_path = ""
        self.project_state_path = ""
        self._changing_sample_from_gdeinput = False

        # Estado de trabajo independiente para cada muestra de Componentry.
        # Esto impide que ZLAM, KS, componentes, descriptores o WEIGHT de una
        # muestra queden visibles al pasar a otra.
        self._active_sample_name = ""
        self._sample_work_states = {}
        self._suspend_sample_state_tracking = False

        # Densidad estándar por nombre de componente. Es un fallback global de
        # GDEINPUT: usuario en Parámetros > TXT de la muestra > estándar del
        # componente > vacío. El origen efectivo de RHOS se conserva por muestra.
        self.component_density_defaults = {}
        self._density_sources = {
            1: {"source": "", "component": ""},
            2: {"source": "", "component": ""},
        }

        # Shape factor estándar por nombre de componente. Usa exactamente la
        # misma jerarquía que RHOS: usuario > TXT > estándar > vacío.
        self.component_shape_defaults = {}
        self._shape_sources = {
            1: {"source": "", "component": ""},
            2: {"source": "", "component": ""},
        }
        self._updating_component_density_table = False

        # Ejecución integrada de PYFLOW. La ruta elegida se recuerda entre
        # sesiones y, si no existe, se busca automáticamente en carpetas cercanas.
        self.pyflow_exe_path = ""
        self.pyflow_process = None
        self.pyflow_runner = None
        self.pyflow_run_dir = ""
        self.pyflow_last_outputs = []
        self.pyflow_execution_mode = ""
        self.pyflow_wsl_script = ""
        self._wsl_probe_cache = None
        self._pyflow_bad_executables = set()
        self._pyflow_settings = QSettings("GDEINPUT", "GDEINPUT")

        self._fit_window_to_screen()
        self._build_ui()
        self._connect_host_sample_signal()
        self.refresh_from_app()
        self._set_default_output_path()
        self._restore_pyflow_executable_preference()
        self._find_pyflow_executable(ask_user=False)
        self.update_classes()
        self.update_preview()

    def _fit_window_to_screen(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 850)
            return

        geo = screen.availableGeometry()

        w = min(1280, max(1100, int(geo.width() * 0.94)))
        h = min(850, max(720, int(geo.height() * 0.90)))

        self.resize(w, h)

        x = geo.x() + max(0, int((geo.width() - w) / 2))
        y = geo.y() + max(0, int((geo.height() - h) / 2))

        self.move(x, y)

    # =========================================================
    # UI
    # =========================================================
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header_row = QHBoxLayout()
        title = QLabel("GDEINPUT V8.19 — generador y ejecución de PYFLOW")
        title.setStyleSheet("font-weight:bold; font-size:14px;")
        header_row.addWidget(title)

        self.btn_toggle_sample_panel = QPushButton("Ocultar muestras")
        self.btn_toggle_sample_panel.setToolTip(
            "Oculta o recupera el panel superior sin perder el tamaño que tenía."
        )
        self.btn_toggle_sample_panel.clicked.connect(self._toggle_sample_panel)
        header_row.addWidget(self.btn_toggle_sample_panel)

        header_row.addStretch(1)

        btn_open_project = QPushButton("Abrir proyecto")
        btn_open_project.setToolTip(
            "Restaura una sesión de GDEINPUT. La muestra y los componentes guardados deben existir en Componentry."
        )
        btn_open_project.clicked.connect(self.open_project_file)
        header_row.addWidget(btn_open_project)

        btn_save_project = QPushButton("Guardar proyecto")
        btn_save_project.setToolTip(
            "Guarda muestra, componentes, parámetros, pesos y datos por muestra en un archivo .gdeinput."
        )
        btn_save_project.clicked.connect(self.save_project_file)
        header_row.addWidget(btn_save_project)

        btn_open = QPushButton("Abrir .dat")
        btn_open.clicked.connect(self.open_dat_file)
        header_row.addWidget(btn_open)

        btn_save = QPushButton("Guardar .dat")
        btn_save.setStyleSheet("font-weight:bold;")
        btn_save.clicked.connect(self.save_dat_file)
        header_row.addWidget(btn_save)

        outer.addLayout(header_row)

        units_note = QLabel(
            "Modo: DOS COMPONENTES SELECCIONADOS. MU, DENGAS, MODEL, PROBT, C0, DISTR1, DISTR2, "
            "DEPRATES, INPUT_WEIGHT y los tests χ² son constantes y se escriben automáticamente. "
            "Los WEIGHT se toman del histograma wt% de cada componente respecto de la muestra completa; NO se normalizan juntos a 100%."
        )
        units_note.setWordWrap(True)
        units_note.setStyleSheet("background:#fff4cc; border:1px solid #d6b656; padding:6px;")
        outer.addWidget(units_note)

        sample_box = QGroupBox("Muestra y componentes")
        sample_layout = QVBoxLayout(sample_box)
        sample_layout.setContentsMargins(8, 8, 8, 8)
        sample_layout.setSpacing(6)

        sample_select_row = QHBoxLayout()
        sample_select_row.addWidget(QLabel("Muestra:"))
        self.cmb_sample = NoWheelComboBox()
        self.cmb_sample.setMinimumWidth(240)
        self.cmb_sample.setToolTip(
            "Selecciona la muestra de Componentry que usará GDEINPUT. "
            "La rueda del mouse no cambia esta selección."
        )
        self.cmb_sample.currentTextChanged.connect(self._on_gde_sample_changed)
        sample_select_row.addWidget(self.cmb_sample)
        self.lbl_sample = QLabel("Muestra actual: —")
        self.lbl_sample.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sample_select_row.addWidget(self.lbl_sample)
        sample_select_row.addStretch(1)
        sample_layout.addLayout(sample_select_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Modo de selección:"))
        self.cmb_component_mode = NoWheelComboBox()
        self.cmb_component_mode.addItem("Dos componentes")
        self.cmb_component_mode.setEnabled(False)
        mode_row.addWidget(self.cmb_component_mode)
        mode_row.addStretch(1)
        sample_layout.addLayout(mode_row)

        group_grid = QGridLayout()
        group_grid.setHorizontalSpacing(8)
        group_grid.setVerticalSpacing(6)

        available_box = QGroupBox("Componentes disponibles")
        available_layout = QVBoxLayout(available_box)

        self.tbl_available = QTableWidget()
        self.tbl_available.setColumnCount(3)
        self.tbl_available.setHorizontalHeaderLabels([
            "Componente",
            "Densidad estándar [kg/m³]",
            "Shape factor estándar [-]",
        ])
        self.tbl_available.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_available.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tbl_available.setAlternatingRowColors(True)
        self.tbl_available.setToolTip(
            "Seleccioná una fila para asignarla como componente 1 o 2. "
            "Densidad y shape factor estándar son fallbacks por componente."
        )
        self.tbl_available.setMinimumHeight(105)
        self.tbl_available.setMaximumHeight(165)
        self.tbl_available.horizontalHeader().setStretchLastSection(True)
        self.tbl_available.setColumnWidth(0, 180)
        self.tbl_available.setColumnWidth(1, 190)
        self.tbl_available.cellChanged.connect(self._on_component_density_cell_changed)
        available_layout.addWidget(self.tbl_available)

        density_priority_note = QLabel(
            "Prioridad RHOS y SHAPEFACT: Parámetros (usuario) > TXT de la muestra > "
            "valor estándar del componente > vacío."
        )
        density_priority_note.setWordWrap(True)
        density_priority_note.setStyleSheet("color:#555;")
        available_layout.addWidget(density_priority_note)

        btn_to_1 = QPushButton("Agregar selección al componente 1 →")
        btn_to_1.clicked.connect(lambda: self._add_selected_to_group(1))
        available_layout.addWidget(btn_to_1)

        btn_to_2 = QPushButton("Agregar selección al componente 2 →")
        btn_to_2.clicked.connect(lambda: self._add_selected_to_group(2))
        available_layout.addWidget(btn_to_2)

        group_grid.addWidget(available_box, 0, 0)

        group1_box = self._build_component_group_box(1)
        group2_box = self._build_component_group_box(2)
        group_grid.addWidget(group1_box, 0, 1)
        group_grid.addWidget(group2_box, 0, 2)

        sample_layout.addLayout(group_grid)

        sample_btns = QHBoxLayout()
        btn_refresh = QPushButton("Actualizar desde muestra")
        btn_refresh.clicked.connect(self.refresh_from_app)
        sample_btns.addWidget(btn_refresh)

        btn_autofill = QPushButton("Autocargar φ y pesos desde componentry")
        btn_autofill.clicked.connect(self.autofill_from_componentry)
        sample_btns.addWidget(btn_autofill)

        sample_btns.addStretch(1)
        sample_layout.addLayout(sample_btns)

        # Panel superior redimensionable de forma continua.  Se envuelve en un
        # QScrollArea para que el sizeHint interno de los controles no obligue al
        # QSplitter a saltar entre "grande" y "chico".
        self.sample_box = sample_box
        self.sample_panel_scroll = QScrollArea()
        self.sample_panel_scroll.setWidgetResizable(True)
        self.sample_panel_scroll.setFrameShape(QScrollArea.NoFrame)
        self.sample_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sample_panel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.sample_panel_scroll.setMinimumHeight(90)
        self.sample_panel_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sample_panel_scroll.setWidget(sample_box)

        self.main_vertical_splitter = QSplitter(Qt.Vertical)
        self.main_vertical_splitter.setChildrenCollapsible(False)
        self.main_vertical_splitter.setOpaqueResize(True)
        self.main_vertical_splitter.setHandleWidth(10)
        self.main_vertical_splitter.setStyleSheet(
            "QSplitter::handle:vertical { background: #c8c8c8; margin: 2px 0px; }"
        )
        self.main_vertical_splitter.addWidget(self.sample_panel_scroll)

        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(220)
        self.tabs.addTab(self._build_parameters_tab(), "Parámetros")
        self.tabs.addTab(self._build_sample_data_tab(), "Datos por muestra")
        self.tabs.addTab(self._build_weights_tab(), "Pesos granulométricos")
        self.tabs.addTab(self._build_preview_tab(), "Vista previa")
        self.pyflow_tab = self._build_pyflow_tab()
        self.tabs.addTab(self.pyflow_tab, "PYFLOW")
        self.main_vertical_splitter.addWidget(self.tabs)
        self.main_vertical_splitter.setCollapsible(0, False)
        self.main_vertical_splitter.setCollapsible(1, False)
        self.main_vertical_splitter.setStretchFactor(0, 0)
        self.main_vertical_splitter.setStretchFactor(1, 1)
        self.main_vertical_splitter.setSizes([300, 650])
        self._sample_panel_last_height = 300
        self._sample_panel_hidden = False
        self.main_vertical_splitter.splitterMoved.connect(self._remember_sample_panel_height)
        outer.addWidget(self.main_vertical_splitter, 1)

        output_box = QGroupBox("Salida")
        output_layout = QGridLayout(output_box)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.setHorizontalSpacing(6)
        output_layout.setVerticalSpacing(6)

        output_layout.addWidget(QLabel("Archivo .dat:"), 0, 0)
        self.output_edit = QLineEdit()
        output_layout.addWidget(self.output_edit, 0, 1)

        btn_choose = QPushButton("Elegir...")
        btn_choose.clicked.connect(self.choose_output_file)
        output_layout.addWidget(btn_choose, 0, 2)

        btn_folder = QPushButton("Abrir carpeta")
        btn_folder.clicked.connect(self.open_output_folder)
        output_layout.addWidget(btn_folder, 0, 3)

        outer.addWidget(output_box)

        bottom = QHBoxLayout()
        bottom.addStretch(1)

        btn_update = QPushButton("Actualizar clases")
        btn_update.clicked.connect(self.update_classes)
        bottom.addWidget(btn_update)

        btn_preview = QPushButton("Actualizar vista previa")
        btn_preview.clicked.connect(self.update_preview)
        bottom.addWidget(btn_preview)

        btn_save2 = QPushButton("Guardar .dat")
        btn_save2.setStyleSheet("font-weight:bold;")
        btn_save2.clicked.connect(self.save_dat_file)
        bottom.addWidget(btn_save2)

        self.btn_run_pyflow = QPushButton("Correr PYFLOW")
        self.btn_run_pyflow.setStyleSheet("font-weight:bold; padding:6px 16px;")
        self.btn_run_pyflow.setToolTip(
            "Valida el input actual, genera input.dat, ejecuta pyflow.exe y muestra los resultados."
        )
        self.btn_run_pyflow.clicked.connect(self.run_pyflow)
        bottom.addWidget(self.btn_run_pyflow)

        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.close)
        bottom.addWidget(btn_close)

        outer.addLayout(bottom)

    def _remember_sample_panel_height(self, _pos=0, _index=0):
        """Recuerda la altura elegida por el usuario mientras el panel está visible."""
        if getattr(self, "_sample_panel_hidden", False):
            return
        splitter = getattr(self, "main_vertical_splitter", None)
        if splitter is None:
            return
        sizes = splitter.sizes()
        if sizes and sizes[0] >= 90:
            self._sample_panel_last_height = int(sizes[0])

    def _toggle_sample_panel(self):
        """Oculta/restaura el panel superior de forma reversible."""
        panel = getattr(self, "sample_panel_scroll", None)
        splitter = getattr(self, "main_vertical_splitter", None)
        button = getattr(self, "btn_toggle_sample_panel", None)
        if panel is None or splitter is None:
            return

        if not getattr(self, "_sample_panel_hidden", False):
            sizes = splitter.sizes()
            if sizes and sizes[0] >= 90:
                self._sample_panel_last_height = int(sizes[0])
            panel.hide()
            self._sample_panel_hidden = True
            if button is not None:
                button.setText("Mostrar muestras")
        else:
            panel.show()
            self._sample_panel_hidden = False
            total = max(1, sum(splitter.sizes()))
            wanted = int(getattr(self, "_sample_panel_last_height", 300))
            wanted = max(90, min(wanted, max(90, total - 220)))
            splitter.setSizes([wanted, max(220, total - wanted)])
            if button is not None:
                button.setText("Ocultar muestras")

    def _build_component_group_box(self, idx):
        box = QGroupBox(f"Componente {idx}")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.SingleSelection)
        lst.setMinimumHeight(70)
        lst.setMaximumHeight(95)
        setattr(self, f"lst_group{idx}", lst)
        layout.addWidget(lst)

        btn_remove = QPushButton(f"← Quitar selección del componente {idx}")
        btn_remove.clicked.connect(lambda: self._remove_selected_from_group(idx))
        layout.addWidget(btn_remove)

        return box

    def _build_parameters_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        for title, keys in SECTIONS:
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignRight)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setContentsMargins(8, 8, 8, 8)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(5)

            for key in keys:
                self._add_param_widget(form, key)

            root.addWidget(box)

        for idx in (1, 2):
            self._update_drag_law_fields(idx)
            self._update_density_fields(idx)

        root.addStretch(1)
        return scroll

    def _build_sample_data_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        info = QLabel(
            "Carga un TXT con datos por muestra. Se acepta directamente el formato por bloques: "
            "una línea con el nombre de la muestra (por ejemplo R20) y debajo líneas KEY: valor, "
            "como ZLAM: 0.45 y KS: 0.01. También se mantiene compatibilidad con tablas que tengan "
            "una columna SAMPLE o MUESTRA. Podés agregar más keywords de GDEINPUT en el futuro."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        btn_row = QHBoxLayout()
        btn_load = QPushButton("Cargar TXT de datos por muestra")
        btn_load.clicked.connect(self.load_sample_metadata_txt)
        btn_row.addWidget(btn_load)

        btn_template = QPushButton("Guardar plantilla TXT")
        btn_template.clicked.connect(self.save_sample_metadata_template)
        btn_row.addWidget(btn_template)

        btn_apply = QPushButton("Aplicar a muestra seleccionada")
        btn_apply.clicked.connect(self.apply_metadata_to_selected_sample)
        btn_row.addWidget(btn_apply)

        btn_clear = QPushButton("Quitar datos cargados")
        btn_clear.clicked.connect(self.clear_sample_metadata)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.lbl_metadata_file = QLabel("TXT cargado: —")
        self.lbl_metadata_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_metadata_file)

        self.lbl_metadata_status = QLabel(
            "Al seleccionar una muestra, sus valores del TXT se aplicarán automáticamente a los campos de GDEINPUT."
        )
        self.lbl_metadata_status.setWordWrap(True)
        root.addWidget(self.lbl_metadata_status)

        self.tbl_sample_metadata = QTableWidget()
        self.tbl_sample_metadata.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_sample_metadata.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_sample_metadata.setAlternatingRowColors(True)
        root.addWidget(self.tbl_sample_metadata, 1)

        return w

    def _build_weights_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_autofill = QPushButton("Autocargar fracciones desde el histograma")
        btn_autofill.clicked.connect(self.autofill_from_componentry)
        btn_row.addWidget(btn_autofill)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.lbl_weight_status = QLabel(
            "Pesos: se copian como wt% de la muestra completa. Los dos componentes seleccionados no tienen que sumar 100%."
        )
        self.lbl_weight_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_weight_status)

        tables_grid = QGridLayout()
        tables_grid.setHorizontalSpacing(8)

        for idx in (1, 2):
            box = QGroupBox(f"Fracciones del componente {idx} — automáticas")
            lay = QVBoxLayout(box)

            table = QTableWidget()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(["Clase", "φ", "KEYWORD", "wt% muestra completa"])
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            table.setMinimumHeight(260)

            lay.addWidget(table, 1)
            self.weight_tables[idx] = table
            tables_grid.addWidget(box, 0, idx - 1)

        root.addLayout(tables_grid, 1)
        return w

    def _build_preview_tab(self):
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setLineWrapMode(QPlainTextEdit.NoWrap)
        root.addWidget(self.preview, 1)

        return w

    def _build_pyflow_tab(self):
        # La pestaña PYFLOW puede crecer mucho cuando aparecen varios gráficos.
        # Se usa UN scroll vertical exterior para que Qt no comprima log, archivos
        # y figuras intentando hacer que todo entre en la altura de la ventana.
        page_scroll = QScrollArea()
        page_scroll.setWidgetResizable(True)
        page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        w = QWidget()
        w.setMinimumWidth(980)
        root = QVBoxLayout(w)
        root.setContentsMargins(8, 8, 8, 16)
        root.setSpacing(10)

        exe_box = QGroupBox("Ejecutable de PYFLOW")
        exe_layout = QGridLayout(exe_box)
        exe_layout.addWidget(QLabel("PYFLOW.exe:"), 0, 0)
        self.pyflow_exe_edit = QLineEdit()
        self.pyflow_exe_edit.setReadOnly(True)
        self.pyflow_exe_edit.setPlaceholderText("GDEINPUT buscará PYFLOW automáticamente")
        exe_layout.addWidget(self.pyflow_exe_edit, 0, 1)
        btn_find = QPushButton("Buscar...")
        btn_find.clicked.connect(self.choose_pyflow_executable)
        exe_layout.addWidget(btn_find, 0, 2)
        self.lbl_pyflow_runner = QLabel("Modo de ejecución: —")
        self.lbl_pyflow_runner.setTextInteractionFlags(Qt.TextSelectableByMouse)
        exe_layout.addWidget(self.lbl_pyflow_runner, 1, 0, 1, 3)
        root.addWidget(exe_box)

        self.lbl_pyflow_status = QLabel(
            "Listo. GDEINPUT generará input.dat, ejecutará PYFLOW y mostrará aquí sus resultados."
        )
        self.lbl_pyflow_status.setWordWrap(True)
        self.lbl_pyflow_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.lbl_pyflow_status)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Resultados de PYFLOW:"))
        self.pyflow_run_dir_edit = QLineEdit()
        self.pyflow_run_dir_edit.setReadOnly(True)
        run_row.addWidget(self.pyflow_run_dir_edit, 1)
        btn_open_run = QPushButton("Abrir resultados")
        btn_open_run.clicked.connect(self.open_pyflow_run_folder)
        run_row.addWidget(btn_open_run)
        root.addLayout(run_row)

        split = QGridLayout()
        split.setHorizontalSpacing(10)

        log_box = QGroupBox("Log de PYFLOW")
        log_box.setMinimumHeight(300)
        log_layout = QVBoxLayout(log_box)
        self.pyflow_log = QPlainTextEdit()
        self.pyflow_log.setReadOnly(True)
        self.pyflow_log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.pyflow_log.setMinimumHeight(250)
        log_layout.addWidget(self.pyflow_log)
        split.addWidget(log_box, 0, 0)

        files_box = QGroupBox("Archivos generados — doble clic para abrir")
        files_box.setMinimumHeight(300)
        files_layout = QVBoxLayout(files_box)
        self.lst_pyflow_outputs = QListWidget()
        self.lst_pyflow_outputs.setMinimumHeight(250)
        self.lst_pyflow_outputs.itemDoubleClicked.connect(self._open_pyflow_output_item)
        files_layout.addWidget(self.lst_pyflow_outputs)
        split.addWidget(files_box, 0, 1)
        split.setColumnStretch(0, 2)
        split.setColumnStretch(1, 1)
        root.addLayout(split)

        # Los gráficos se agregan directamente a la página. No hay un segundo
        # scroll interno que los achique: cada PNG conserva un tamaño legible y
        # el usuario recorre toda la pestaña con el scroll exterior.
        graph_box = QGroupBox("Gráficos generados")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(8, 8, 8, 8)
        graph_layout.setSpacing(8)

        self.pyflow_graph_container = QWidget()
        self.pyflow_graph_layout = QVBoxLayout(self.pyflow_graph_container)
        self.pyflow_graph_layout.setContentsMargins(6, 6, 6, 6)
        self.pyflow_graph_layout.setSpacing(16)
        graph_layout.addWidget(self.pyflow_graph_container)
        root.addWidget(graph_box)

        root.addStretch(1)
        page_scroll.setWidget(w)
        return page_scroll

    def _add_param_widget(self, form, key):
        unit = FIELD_UNITS.get(key, "")
        unit_txt = f" [{unit}]" if unit else ""
        label = QLabel(f"{FIELD_LABELS.get(key, key)}{unit_txt} ({key}) =")
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        help_txt = FIELD_HELP.get(key, "")
        if help_txt:
            label.setToolTip(help_txt)

        if key in LOGICAL_KEYWORDS:
            widget = self._make_combo(LOGICAL_OPTIONS, DEFAULTS.get(key, ".FALSE."), editable=False)
        elif key.startswith("RHOLAW"):
            widget = self._make_combo(RHO_LAWS, DEFAULTS.get(key, "CUSTOM"), editable=False)
        elif key.startswith("RHO_CUSTOM"):
            widget = self._make_combo(["CONSTANT", "VARIABLE"], DEFAULTS.get(key, "CONSTANT"), editable=False)
        elif key.startswith("CDLAW"):
            widget = self._make_combo(DRAG_LAWS, DEFAULTS.get(key, "DIOGMELE"), editable=False)
        else:
            widget = QLineEdit()
            widget.setText(DEFAULTS.get(key, ""))
            if key.startswith(("PHIMIN", "PHIMAX", "DPHI", "NCLASS")):
                widget.setReadOnly(True)
                widget.setToolTip("Valor automático tomado/calculado desde el histograma del componente seleccionado.")
            widget.textChanged.connect(self.update_preview)

        if isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(self.update_preview)
            if widget.isEditable() and widget.lineEdit() is not None:
                widget.lineEdit().textChanged.connect(self.update_preview)

        self.param_widgets[key.upper()] = widget
        self.param_labels[key.upper()] = label
        form.addRow(label, widget)

        m_rhos = re.match(r"^RHOS\((1|2),0\)$", key.upper())
        if m_rhos and isinstance(widget, QLineEdit):
            idx_rhos = int(m_rhos.group(1))
            widget.textEdited.connect(lambda _text, i=idx_rhos: self._on_density_user_edited(i))
            widget.editingFinished.connect(lambda i=idx_rhos: self._on_density_editing_finished(i))

        m_shape = re.match(r"^SHAPEFACT\((1|2),0\)$", key.upper())
        if m_shape and isinstance(widget, QLineEdit):
            idx_shape = int(m_shape.group(1))
            widget.textEdited.connect(lambda _text, i=idx_shape: self._on_shape_user_edited(i))
            widget.editingFinished.connect(lambda i=idx_shape: self._on_shape_editing_finished(i))

        m = re.match(r"^(CDLAW|ISOMETRIC|FRACTAL)\((1|2)\)$", key.upper())
        if m and isinstance(widget, QComboBox):
            idx = int(m.group(2))
            widget.currentTextChanged.connect(lambda _text, i=idx: self._update_drag_law_fields(i))

        m_rho = re.match(r"^RHOLAW\((1|2)\)$", key.upper())
        if m_rho and isinstance(widget, QComboBox):
            idx = int(m_rho.group(1))
            widget.currentTextChanged.connect(lambda _text, i=idx: self._update_density_fields(i))

    def _update_density_fields(self, idx):
        """En este modo (DEPRATES=.F.), RHOS(i,0) se pide solo para RHOLAW='CUSTOM'."""
        rholaw = strip_quotes(self._widget_value(f"RHOLAW({idx})")).upper()
        key = f"RHOS({idx},0)"
        widget = self.param_widgets.get(key)
        label = self.param_labels.get(key)
        show = rholaw == "CUSTOM"
        if widget is not None:
            widget.setVisible(show)
            widget.setEnabled(show)
        if label is not None:
            label.setVisible(show)
            label.setEnabled(show)
        self.update_preview()

    def _drag_fields_for_component(self, idx):
        cdlaw = strip_quotes(self._widget_value(f"CDLAW({idx})")).upper()
        isometric = normalize_bool(self._widget_value(f"ISOMETRIC({idx})")) == ".T."
        fractal = normalize_bool(self._widget_value(f"FRACTAL({idx})")) == ".T."
        return drag_fields_for_law(cdlaw, idx, isometric=isometric, fractal=fractal)

    def _update_drag_law_fields(self, idx):
        required = set(self._drag_fields_for_component(idx))
        all_keys = [
            f"SPHERICITY({idx},0)", f"COREY({idx},0)",
            f"VOLEQSPHD({idx},0)", f"CIRCEQARD({idx},0)", f"ISOMETRIC({idx})",
            f"CIRCULARITY({idx},0)", f"FLATRATIO({idx},0)",
            f"SHAPEFACT({idx},0)", f"LONGSPHER({idx},0)", f"CROSSSPHER({idx},0)",
            f"FRACTAL({idx})", f"FRACTDIM({idx},0)",
        ]
        cdlaw = strip_quotes(self._widget_value(f"CDLAW({idx})")).upper()
        for key in all_keys:
            widget = self.param_widgets.get(key)
            label = self.param_labels.get(key)
            show = key in required
            # The mode switches themselves must remain visible when their law is selected.
            if key == f"ISOMETRIC({idx})": show = cdlaw == "GANSER"
            if key == f"FRACTAL({idx})": show = cdlaw == "DIOG2017"
            if widget is not None:
                widget.setVisible(show)
                widget.setEnabled(show)
            if label is not None:
                label.setVisible(show)
                label.setEnabled(show)
        self.update_preview()

    def _make_combo(self, options, default, editable=False):
        cb = NoWheelComboBox()
        cb.addItems([str(v) for v in options])
        cb.setEditable(bool(editable))
        set_combo_text(cb, str(default))
        return cb

    # =========================================================
    # Component group handling
    # =========================================================
    def _snapshot_current_sample_state(self):
        """Captura solamente el estado editable asociado a la muestra activa."""
        return {
            "parameters": {
                str(key): self._widget_value(key)
                for key in sorted(self.param_widgets.keys())
            },
            "components": {
                "1": self._group_components(1),
                "2": self._group_components(2),
            },
            "weights": {
                "1": dict(self._collect_weight_values().get(1, {})),
                "2": dict(self._collect_weight_values().get(2, {})),
            },
            "density_sources": {
                "1": dict(self._density_sources.get(1, {})),
                "2": dict(self._density_sources.get(2, {})),
            },
            "shape_sources": {
                "1": dict(self._shape_sources.get(1, {})),
                "2": dict(self._shape_sources.get(2, {})),
            },
        }

    def _save_active_sample_state(self):
        """Guarda el formulario actual bajo el nombre de la muestra que lo originó."""
        if self._suspend_sample_state_tracking:
            return
        name = str(self._active_sample_name).strip()
        if not name:
            return
        self._sample_work_states[name] = self._snapshot_current_sample_state()

    def _clear_weight_tables_for_sample_switch(self):
        """Vacía por completo tablas y cache de WEIGHT antes de entrar a otra muestra."""
        for idx in (1, 2):
            table = self.weight_tables.get(idx)
            if table is not None:
                table.blockSignals(True)
                try:
                    table.setRowCount(0)
                finally:
                    table.blockSignals(False)
            self.weight_phi[idx] = []
            self.weight_widgets[idx] = {}

    def _reset_form_for_new_sample(self):
        """Inicializa una muestra que aún no tiene estado propio en esta sesión."""
        self._density_sources = {
            1: {"source": "", "component": ""},
            2: {"source": "", "component": ""},
        }
        self._shape_sources = {
            1: {"source": "", "component": ""},
            2: {"source": "", "component": ""},
        }
        # Primero eliminar selecciones y pesos de la muestra anterior.
        self._set_group_components(1, [])
        self._set_group_components(2, [])
        self._clear_weight_tables_for_sample_switch()

        # Cada control vuelve a su valor inicial de GDEINPUT. Así, un dato no
        # especificado para la nueva muestra jamás hereda el de la anterior.
        for key in sorted(self.param_widgets.keys()):
            self._set_widget_value(key, DEFAULTS.get(key, ""))

        for idx in (1, 2):
            self._update_density_fields(idx)
            self._update_drag_law_fields(idx)

        # Reconstruir únicamente la grilla φ inicial; los pesos quedan vacíos.
        self.update_classes()
        self.clear_weights()

    def _restore_sample_work_state(self, state, available_components):
        """Restaura el estado que pertenece exclusivamente a una muestra."""
        if not isinstance(state, dict):
            self._reset_form_for_new_sample()
            return

        self._clear_weight_tables_for_sample_switch()

        parameters = state.get("parameters", {})
        if isinstance(parameters, dict):
            for key in sorted(self.param_widgets.keys()):
                value = parameters.get(key, DEFAULTS.get(key, ""))
                self._set_widget_value(key, "" if value is None else str(value))
        else:
            for key in sorted(self.param_widgets.keys()):
                self._set_widget_value(key, DEFAULTS.get(key, ""))

        self._density_sources = self._normalized_density_sources(state.get("density_sources", {}))
        self._shape_sources = self._normalized_shape_sources(state.get("shape_sources", {}))

        components = state.get("components", {})
        if not isinstance(components, dict):
            components = {}
        allowed = set(str(c) for c in available_components)
        group1 = [str(c) for c in components.get("1", []) if str(c) in allowed]
        group2 = [str(c) for c in components.get("2", []) if str(c) in allowed]
        self._set_group_components(1, group1)
        self._set_group_components(2, group2)

        for idx in (1, 2):
            self._update_density_fields(idx)
            self._update_drag_law_fields(idx)

        self.update_classes()
        self.clear_weights()

        weights = state.get("weights", {})
        if isinstance(weights, dict):
            for idx in (1, 2):
                values = weights.get(str(idx), {})
                if not isinstance(values, dict):
                    continue
                table = self.weight_tables.get(idx)
                if table is None:
                    continue
                table.blockSignals(True)
                try:
                    for key, value in values.items():
                        m = re.match(r"^WEIGHT\(\s*(\d+)\s*,\s*(\d+)\s*\)$", str(key), re.IGNORECASE)
                        if not m or int(m.group(1)) != idx:
                            continue
                        self._set_weight_value(idx, int(m.group(2)), "" if value is None else str(value))
                finally:
                    table.blockSignals(False)

    def refresh_from_app(self):
        sname = ""
        try:
            sname = str(self.app.current_sample_name()).strip()
        except Exception:
            sname = ""

        old_name = str(self._active_sample_name).strip()
        changed_sample = sname != old_name

        # Antes de tocar el formulario, guardar el estado bajo la muestra vieja.
        if changed_sample and old_name:
            self._save_active_sample_state()

        self._refresh_sample_selector(sname)
        self.lbl_sample.setText(f"Muestra actual: {sname or '—'}")

        components = []
        try:
            db = self.app.current_db()
        except Exception:
            db = None

        if db is not None:
            if hasattr(db, "components") and db.components:
                components = [str(c) for c in db.components]
            elif hasattr(db, "total") and isinstance(db.total, pd.DataFrame):
                components = [str(c) for c in db.total.columns]

        self._refresh_component_density_table(components)

        if changed_sample:
            self._active_sample_name = sname
            own_state = self._sample_work_states.get(sname) if sname else None
            if own_state is None:
                self._reset_form_for_new_sample()
            else:
                self._restore_sample_work_state(own_state, components)
        else:
            # Refresco de la misma muestra: conservar únicamente selecciones que
            # sigan existiendo en su base actual.
            current_group_1 = self._group_components(1)
            current_group_2 = self._group_components(2)
            self._set_group_components(1, [c for c in current_group_1 if c in components])
            self._set_group_components(2, [c for c in current_group_2 if c in components])
            if not self._active_sample_name and sname:
                self._active_sample_name = sname

        # Los datos externos de esa muestra tienen prioridad sobre el estado
        # manual únicamente para las keywords presentes en el TXT.
        self._apply_metadata_for_sample(sname)
        self.update_preview()

    def _available_sample_names(self):
        sample_dbs = getattr(self.app, "sample_dbs", {})
        if isinstance(sample_dbs, dict):
            return [str(name) for name in sample_dbs.keys()]
        return []

    def _refresh_sample_selector(self, current_name=""):
        names = self._available_sample_names()
        self.cmb_sample.blockSignals(True)
        self.cmb_sample.clear()
        self.cmb_sample.addItems(names)
        if current_name:
            idx = self.cmb_sample.findText(str(current_name))
            if idx >= 0:
                self.cmb_sample.setCurrentIndex(idx)
        self.cmb_sample.blockSignals(False)
        self.cmb_sample.setEnabled(bool(names))

    def _connect_host_sample_signal(self):
        """Mantiene GDEINPUT sincronizado cuando la muestra cambia en Componentry."""
        try:
            host_combo = getattr(self.app, "cmb_sample", None)
            if isinstance(host_combo, QComboBox):
                host_combo.currentTextChanged.connect(self._on_host_sample_changed)
        except Exception:
            pass

    def _on_host_sample_changed(self, sample_name):
        if self._changing_sample_from_gdeinput:
            return
        sample_name = str(sample_name).strip()
        if not sample_name:
            return
        self.refresh_from_app()
        self._set_default_output_path()

    def _on_gde_sample_changed(self, sample_name):
        sample_name = str(sample_name).strip()
        if not sample_name or self._changing_sample_from_gdeinput:
            return

        self._changing_sample_from_gdeinput = True
        try:
            host_combo = getattr(self.app, "cmb_sample", None)
            if isinstance(host_combo, QComboBox):
                idx = host_combo.findText(sample_name)
                if idx >= 0 and host_combo.currentIndex() != idx:
                    host_combo.setCurrentIndex(idx)
            self.refresh_from_app()
            self._set_default_output_path()
        finally:
            self._changing_sample_from_gdeinput = False

    def _metadata_column_to_key(self, column):
        raw = str(column).strip()
        up = raw.upper().replace(" ", "_")
        aliases = {
            "ESPESOR": "ZLAM",
            "THICKNESS": "ZLAM",
            "ESPESOR_M": "ZLAM",
            "RUGOSIDAD": "KS",
            "ROUGHNESS": "KS",
            "K_S": "KS",
        }
        if up in aliases:
            return aliases[up]
        return raw.upper()

    def _normalize_metadata_sample_name(self, sample_name):
        """Normaliza un identificador de muestra para comparaciones TXT ↔ Componentry."""
        text = str(sample_name).replace("\ufeff", "").strip()
        text = os.path.basename(text)
        root, ext = os.path.splitext(text)
        if ext.lower() in {".xlsx", ".xls", ".csv", ".txt", ".dat"}:
            text = root
        text = re.sub(r"\s+", "", text)
        return text.casefold()

    def _metadata_sample_candidates(self, sample_name):
        """Devuelve identificadores equivalentes para una muestra.

        FW/Componentry puede exponer nombres como ``Sieve — R16`` mientras que
        el archivo de parámetros usa simplemente ``R16``.  Ambos representan
        la misma muestra y deben cruzarse automáticamente.
        """
        raw = str(sample_name).replace("\ufeff", "").strip()
        if not raw:
            return []

        candidates = []

        def add(value):
            value = str(value).strip()
            if not value:
                return
            norm = self._normalize_metadata_sample_name(value)
            if norm and norm not in candidates:
                candidates.append(norm)

        add(raw)

        # Prefijos que FW usa para distinguir la base de procedencia.
        known_prefixes = {
            "sieve", "tamiz", "laser", "láser",
            "hybrid", "hibrido", "híbrido",
        }

        # Incorporar también el base_label que Componentry recibe desde FW.
        try:
            ctx = getattr(self.app, "source_context", {}) or {}
            base_label = str(ctx.get("base_label", "")).strip().casefold()
            if base_label:
                known_prefixes.add(base_label)
        except Exception:
            pass

        # Solo se elimina el texto anterior al guion cuando es un prefijo de
        # base reconocido. Así no se rompen nombres reales que contengan guiones.
        for sep in (" — ", " – ", " - ", "—", "–"):
            if sep not in raw:
                continue
            left, right = raw.split(sep, 1)
            if left.strip().casefold() in known_prefixes:
                add(right)

        return candidates

    def _sample_metadata_match(self, sample_name):
        """Retorna (nombre_en_TXT, registro) para la muestra solicitada."""
        targets = set(self._metadata_sample_candidates(sample_name))
        if not targets:
            return None, None

        for name, record in self.sample_metadata.items():
            source_candidates = set(self._metadata_sample_candidates(name))
            if targets.intersection(source_candidates):
                return name, record
        return None, None

    def _sample_metadata_record(self, sample_name):
        _matched_name, record = self._sample_metadata_match(sample_name)
        return record

    def _apply_metadata_for_sample(self, sample_name):
        sample_name = str(sample_name).strip()

        # Incluso sin TXT, RHOS y SHAPEFACT pueden caer al valor estándar del componente.
        if not sample_name or not self.sample_metadata:
            for idx in (1, 2):
                comp = self._group_components(idx)
                selected = comp[0] if comp else ""
                self._resolve_density_for_component(idx, selected, sample_name=sample_name)
                self._resolve_shape_for_component(idx, selected, sample_name=sample_name)
            return []

        matched_name, record = self._sample_metadata_match(sample_name)
        if record is None:
            for idx in (1, 2):
                comp = self._group_components(idx)
                selected = comp[0] if comp else ""
                self._resolve_density_for_component(idx, selected, sample_name=sample_name)
                self._resolve_shape_for_component(idx, selected, sample_name=sample_name)
            if hasattr(self, "lbl_metadata_status"):
                available = ", ".join(self.sample_metadata.keys())
                self.lbl_metadata_status.setText(
                    f"La muestra '{sample_name}' no coincide con ninguna muestra del TXT. "
                    f"Muestras del TXT: {available}. Se usan los valores estándar de componente cuando existan."
                )
            return []

        applied = []
        cleared = []
        ignored = []

        # RHOS y SHAPEFACT se resuelven aparte para respetar estrictamente:
        # usuario > TXT > estándar del componente > vacío.
        for column, value in record.items():
            key = self._metadata_column_to_key(column)
            if re.fullmatch(r"(?:RHOS|SHAPEFACT)\((1|2),0\)", key, re.IGNORECASE):
                continue
            if key not in self.param_widgets:
                ignored.append(str(column))
                continue

            text = "" if value is None else str(value).strip()
            if text.lower() == "nan":
                text = ""

            widget = self.param_widgets.get(key)
            if text:
                self._set_widget_value(key, text)
                applied.append(key)
            elif isinstance(widget, QLineEdit):
                self._set_widget_value(key, "")
                cleared.append(key)

        for idx in (1, 2):
            comp = self._group_components(idx)
            selected = comp[0] if comp else ""
            rho_source = self._resolve_density_for_component(
                idx, selected, sample_name=sample_name
            )
            shape_source = self._resolve_shape_for_component(
                idx, selected, sample_name=sample_name
            )
            if rho_source == "txt":
                applied.append(f"RHOS({idx},0)")
            if shape_source == "txt":
                applied.append(f"SHAPEFACT({idx},0)")
            self._update_density_fields(idx)
            self._update_drag_law_fields(idx)

        self.update_preview()

        if hasattr(self, "lbl_metadata_status"):
            if str(matched_name).strip() == sample_name:
                match_txt = f"Muestra '{sample_name}'"
            else:
                match_txt = f"Componentry '{sample_name}' ↔ TXT '{matched_name}'"
            msg = f"{match_txt}: {len(applied)} valor(es) cargado(s) automáticamente"
            if applied:
                msg += " (" + ", ".join(applied) + ")"
            if cleared:
                msg += ". Campos vacíos limpiados: " + ", ".join(cleared)
            if ignored:
                msg += ". Keywords sin campo equivalente: " + ", ".join(ignored)
            self.lbl_metadata_status.setText(msg + ".")

        if sample_name and sample_name == str(self._active_sample_name).strip():
            self._sample_work_states[sample_name] = self._snapshot_current_sample_state()

        return applied

    def apply_metadata_to_selected_sample(self):
        """Aplica manualmente el TXT a la muestra visible en GDEINPUT."""
        if not self.sample_metadata:
            QMessageBox.warning(self, "Datos por muestra", "Primero cargá un TXT de datos por muestra.")
            return

        sample_name = self.cmb_sample.currentText().strip()
        if not sample_name:
            try:
                sample_name = str(self.app.current_sample_name()).strip()
            except Exception:
                sample_name = ""

        if not sample_name:
            QMessageBox.warning(self, "Datos por muestra", "No hay una muestra seleccionada.")
            return

        applied = self._apply_metadata_for_sample(sample_name)
        if applied:
            QMessageBox.information(
                self, "Datos por muestra",
                f"Datos aplicados a '{sample_name}': " + ", ".join(applied)
            )
        else:
            QMessageBox.warning(
                self, "Datos por muestra",
                f"No se aplicó ningún valor a '{sample_name}'. Revisá el mensaje de estado de esta pestaña."
            )

    def _read_metadata_dataframe(self, path):
        """Lee datos por muestra en formato tabla o en formato por bloques.

        Formato por bloques aceptado (recomendado):

            R20
            ZLAM: 0.45
            KS: 0.01

            R06
            ZLAM: 1.4
            KS: 0.01

        Cualquier línea no vacía sin ':' ni '=' inicia una nueva muestra.
        Las líneas KEY: valor o KEY = valor se asignan a esa muestra. Esto
        permite agregar nuevas keywords más adelante sin cambiar el parser.

        También se conserva compatibilidad con TXT/TSV/CSV tabulares cuya
        cabecera tenga SAMPLE, MUESTRA, SAMPLE_NAME, NOMBRE o NAME.
        """
        with open(path, "r", encoding="utf-8-sig") as f:
            raw_lines = f.readlines()

        useful_lines = [
            line.rstrip("\r\n")
            for line in raw_lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not useful_lines:
            raise ValueError("El TXT está vacío.")

        sample_aliases = {"SAMPLE", "MUESTRA", "SAMPLE_NAME", "NOMBRE", "NAME"}
        first = useful_lines[0].strip()

        # 1) Formato tabular anterior: SAMPLE<TAB>ZLAM<TAB>KS ...
        # Solo se interpreta como tabla si la primera fila parece realmente
        # una cabecera de muestras. Así, 'R20' se reconoce como muestra.
        for sep in ("\t", ";", ","):
            if sep not in first:
                continue
            header_tokens = [
                token.strip().upper().replace(" ", "_")
                for token in first.split(sep)
            ]
            if any(token in sample_aliases for token in header_tokens):
                df = pd.read_csv(
                    path, sep=sep, dtype=str, comment="#", encoding="utf-8-sig"
                )
                df.columns = [str(c).strip() for c in df.columns]
                return df

        # 2) Formato por bloques: nombre de muestra + KEY: valor.
        records = {}
        sample_order = []
        key_order = []
        current_sample = None

        for line_number, raw in enumerate(raw_lines, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            match = re.match(r"^([^:=]+?)\s*[:=]\s*(.*?)\s*$", line)
            if match:
                if current_sample is None:
                    raise ValueError(
                        f"Línea {line_number}: encontré un parámetro antes del nombre de la muestra: {line}"
                    )
                key = match.group(1).strip()
                value = match.group(2).strip()
                if not key:
                    raise ValueError(f"Línea {line_number}: keyword vacía.")
                records[current_sample][key] = value
                if key not in key_order:
                    key_order.append(key)
                continue

            # Una línea sin ':' ni '=' es el nombre de una nueva muestra.
            current_sample = line
            if current_sample not in records:
                records[current_sample] = {}
                sample_order.append(current_sample)

        if not records:
            raise ValueError("No se encontraron muestras válidas en el TXT.")

        rows = []
        for sample in sample_order:
            row = {"SAMPLE": sample}
            for key in key_order:
                row[key] = records[sample].get(key, "")
            rows.append(row)

        return pd.DataFrame(rows, columns=["SAMPLE"] + key_order, dtype=str)

    def load_sample_metadata_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Cargar datos por muestra",
            self.sample_metadata_path or "",
            "Text files (*.txt *.tsv *.csv);;All files (*)"
        )
        if not path:
            return

        try:
            df = self._read_metadata_dataframe(path)
            sample_col = None
            sample_aliases = {"SAMPLE", "MUESTRA", "SAMPLE_NAME", "NOMBRE", "NAME"}
            for col in df.columns:
                if str(col).strip().upper().replace(" ", "_") in sample_aliases:
                    sample_col = col
                    break
            if sample_col is None:
                raise ValueError(
                    "Falta una columna SAMPLE o MUESTRA que identifique cada fila."
                )

            metadata = {}
            for _, row in df.iterrows():
                sample = str(row.get(sample_col, "")).strip()
                if not sample or sample.lower() == "nan":
                    continue
                metadata[sample] = {
                    col: row.get(col, "")
                    for col in df.columns
                    if col != sample_col
                }

            if not metadata:
                raise ValueError("No se encontraron muestras válidas en el TXT.")

            self.sample_metadata = metadata
            self.sample_metadata_path = path
            self.lbl_metadata_file.setText(f"TXT cargado: {path}")
            self._show_metadata_dataframe(df)

            # Releer la muestra directamente desde Componentry al cargar el TXT.
            # Esto evita depender de que el combo interno de GDEINPUT estuviera
            # actualizado antes de abrir/cargar el archivo de metadatos.
            self.refresh_from_app()
            try:
                sample_name = str(self.app.current_sample_name()).strip()
            except Exception:
                sample_name = ""
            if not sample_name:
                sample_name = self.cmb_sample.currentText().strip()

            matched_name, _record = self._sample_metadata_match(sample_name) if sample_name else (None, None)
            applied = self._apply_metadata_for_sample(sample_name) if sample_name else []
            self.update_preview()

            if sample_name and applied:
                if matched_name and str(matched_name).strip() != sample_name:
                    shown = f"{sample_name} ↔ {matched_name}"
                else:
                    shown = sample_name
                QMessageBox.information(
                    self, "Datos por muestra",
                    f"TXT cargado. Muestra '{shown}': " + ", ".join(applied) + " aplicado(s)."
                )
            elif sample_name:
                QMessageBox.warning(
                    self, "Datos por muestra",
                    f"El TXT se cargó, pero no pude aplicar valores a la muestra actual '{sample_name}'. "
                    "Revisá que el nombre de la muestra coincida con el del TXT."
                )
            else:
                QMessageBox.warning(
                    self, "Datos por muestra",
                    "El TXT se cargó, pero Componentry no tiene una muestra actual seleccionada."
                )
        except UnicodeDecodeError:
            QMessageBox.warning(
                self, "Datos por muestra",
                "No pude leer el TXT como UTF-8. Guardalo como UTF-8 y volvé a cargarlo."
            )
        except Exception as e:
            QMessageBox.warning(self, "Datos por muestra", str(e))

    def _show_metadata_dataframe(self, df):
        self.tbl_sample_metadata.clear()
        self.tbl_sample_metadata.setRowCount(len(df.index))
        self.tbl_sample_metadata.setColumnCount(len(df.columns))
        self.tbl_sample_metadata.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(len(df.index)):
            for c, col in enumerate(df.columns):
                value = df.iloc[r, c]
                text = "" if pd.isna(value) else str(value)
                self.tbl_sample_metadata.setItem(r, c, QTableWidgetItem(text))
        self.tbl_sample_metadata.resizeColumnsToContents()

    def clear_sample_metadata(self):
        self.sample_metadata = {}
        self.sample_metadata_path = ""
        if hasattr(self, "tbl_sample_metadata"):
            self.tbl_sample_metadata.clear()
            self.tbl_sample_metadata.setRowCount(0)
            self.tbl_sample_metadata.setColumnCount(0)
        if hasattr(self, "lbl_metadata_file"):
            self.lbl_metadata_file.setText("TXT cargado: —")
        if hasattr(self, "lbl_metadata_status"):
            self.lbl_metadata_status.setText("No hay datos por muestra cargados.")

    def save_sample_metadata_template(self):
        suggested = os.path.join(os.getcwd(), "gdeinput_datos_muestras.txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar plantilla TXT", suggested, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"

        names = self._available_sample_names() or ["MUESTRA_1"]
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for i, name in enumerate(names):
                    if i:
                        f.write("\n")
                    f.write(f"{name}\n\n")
                    f.write("ZLAM: \n")
                    f.write("KS: \n")
            QMessageBox.information(self, "Datos por muestra", f"Plantilla guardada:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Datos por muestra", str(e))

    def _normalized_density_sources(self, raw):
        out = {
            1: {"source": "", "component": ""},
            2: {"source": "", "component": ""},
        }
        if not isinstance(raw, dict):
            return out
        for idx in (1, 2):
            item = raw.get(str(idx), raw.get(idx, {}))
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip().lower()
            if source not in {"", "user", "txt", "default"}:
                source = ""
            out[idx] = {
                "source": source,
                "component": str(item.get("component", "") or "").strip(),
            }
        return out

    def _refresh_component_density_table(self, components):
        if not hasattr(self, "tbl_available"):
            return
        self._updating_component_density_table = True
        self.tbl_available.blockSignals(True)
        try:
            selected_name = ""
            rows = self.tbl_available.selectionModel().selectedRows() if self.tbl_available.selectionModel() else []
            if rows:
                item = self.tbl_available.item(rows[0].row(), 0)
                selected_name = item.text().strip() if item else ""

            self.tbl_available.setRowCount(len(components))
            selected_row = -1
            for row, comp in enumerate(components):
                comp = str(comp).strip()
                item_comp = QTableWidgetItem(comp)
                item_comp.setFlags(item_comp.flags() & ~Qt.ItemIsEditable)
                self.tbl_available.setItem(row, 0, item_comp)

                density = str(self.component_density_defaults.get(comp, "") or "").strip()
                item_rho = QTableWidgetItem(density)
                item_rho.setToolTip(
                    "Densidad estándar en kg/m³. Se usa solo si RHOS no fue elegido "
                    "manualmente y el TXT de la muestra tampoco aporta RHOS."
                )
                self.tbl_available.setItem(row, 1, item_rho)

                shape = str(self.component_shape_defaults.get(comp, "") or "").strip()
                item_shape = QTableWidgetItem(shape)
                item_shape.setToolTip(
                    "Shape factor estándar del componente (0 < Ψ ≤ 1). Se usa solo si "
                    "SHAPEFACT no fue elegido manualmente y el TXT tampoco lo aporta."
                )
                self.tbl_available.setItem(row, 2, item_shape)

                if comp == selected_name:
                    selected_row = row

            if selected_row >= 0:
                self.tbl_available.selectRow(selected_row)
        finally:
            self.tbl_available.blockSignals(False)
            self._updating_component_density_table = False

    def _component_default_density(self, component):
        component = str(component or "").strip()
        if not component:
            return ""
        text = str(self.component_density_defaults.get(component, "") or "").strip()
        if not text:
            return ""
        try:
            if parse_fortran_float(text) <= 100:
                return ""
        except Exception:
            return ""
        return text

    def _on_component_density_cell_changed(self, row, column):
        if self._updating_component_density_table or column not in (1, 2):
            return
        comp_item = self.tbl_available.item(row, 0)
        if comp_item is None:
            return
        comp = comp_item.text().strip()
        if not comp:
            return

        if column == 1:
            rho_item = self.tbl_available.item(row, 1)
            text = rho_item.text().strip() if rho_item is not None else ""
            self.component_density_defaults[comp] = text
            if rho_item is not None:
                if not text:
                    rho_item.setToolTip("Sin densidad estándar definida.")
                else:
                    try:
                        rho = parse_fortran_float(text)
                        if rho <= 100:
                            raise ValueError
                        rho_item.setToolTip("Densidad estándar válida en kg/m³.")
                    except Exception:
                        rho_item.setToolTip(
                            "Valor no válido: la densidad estándar debe ser numérica y > 100 kg/m³."
                        )
            resolver = self._resolve_density_for_component
        else:
            shape_item = self.tbl_available.item(row, 2)
            text = shape_item.text().strip() if shape_item is not None else ""
            self.component_shape_defaults[comp] = text
            if shape_item is not None:
                if not text:
                    shape_item.setToolTip("Sin shape factor estándar definido.")
                else:
                    try:
                        psi = parse_fortran_float(text)
                        if not (0.0 < psi <= 1.0):
                            raise ValueError
                        shape_item.setToolTip("Shape factor estándar válido (0 < Ψ ≤ 1).")
                    except Exception:
                        shape_item.setToolTip(
                            "Valor no válido: el shape factor estándar debe cumplir 0 < Ψ ≤ 1."
                        )
            resolver = self._resolve_shape_for_component

        # Si este componente está seleccionado, recalcular el parámetro respetando
        # la prioridad completa. Un valor manual o del TXT no será reemplazado.
        for idx in (1, 2):
            current = self._group_components(idx)
            if current and current[0] == comp:
                resolver(idx, comp)
        self.update_preview()

    def _metadata_density_value(self, sample_name, idx):
        if not sample_name or not self.sample_metadata:
            return ""
        _matched, record = self._sample_metadata_match(sample_name)
        if not isinstance(record, dict):
            return ""
        target = f"RHOS({idx},0)"
        for column, value in record.items():
            if self._metadata_column_to_key(column).upper() != target:
                continue
            text = "" if value is None else str(value).strip()
            if text.lower() == "nan":
                text = ""
            return text
        return ""

    def _set_density_programmatic(self, idx, value, source, component):
        key = f"RHOS({idx},0)"
        self._set_widget_value(key, "" if value is None else str(value))
        self._density_sources[idx] = {
            "source": str(source or "").strip().lower(),
            "component": str(component or "").strip(),
        }
        self._update_density_origin_tooltip(idx)

    def _update_density_origin_tooltip(self, idx):
        key = f"RHOS({idx},0)"
        widget = self.param_widgets.get(key)
        if widget is None:
            return
        state = self._density_sources.get(idx, {})
        source = str(state.get("source", ""))
        component = str(state.get("component", ""))
        source_label = {
            "user": "usuario (Parámetros)",
            "txt": "TXT de la muestra",
            "default": "densidad estándar del componente",
            "": "sin dato",
        }.get(source, source)
        base = FIELD_HELP.get(key, "")
        extra = f"Origen actual: {source_label}"
        if component:
            extra += f" — {component}"
        widget.setToolTip((base + "\n" + extra).strip())

    def _on_density_user_edited(self, idx):
        comp = self._group_components(idx)
        self._density_sources[idx] = {
            "source": "user",
            "component": comp[0] if comp else "",
        }
        self._update_density_origin_tooltip(idx)

    def _on_density_editing_finished(self, idx):
        # Borrar manualmente RHOS significa renunciar al override del usuario y
        # volver a la cadena TXT > estándar > vacío.
        key = f"RHOS({idx},0)"
        if self._widget_value(key).strip():
            return
        state = self._density_sources.get(idx, {})
        if state.get("source") == "user":
            self._density_sources[idx] = {"source": "", "component": ""}
            comp = self._group_components(idx)
            self._resolve_density_for_component(idx, comp[0] if comp else "")

    def _resolve_density_for_component(self, idx, component, sample_name=None):
        component = str(component or "").strip()
        if sample_name is None:
            sample_name = str(self._active_sample_name or "").strip()

        state = self._density_sources.get(idx, {"source": "", "component": ""})
        source = str(state.get("source", "")).strip().lower()
        source_component = str(state.get("component", "") or "").strip()

        # 1) El valor escrito por el usuario siempre gana mientras corresponda
        # al componente actualmente seleccionado. Si fue escrito antes de elegir
        # componente, se vincula al primero que se seleccione.
        if source == "user" and (not source_component or source_component == component):
            if component and not source_component:
                self._density_sources[idx]["component"] = component
            self._update_density_origin_tooltip(idx)
            return "user"

        # Al cambiar de componente se descarta un override manual perteneciente
        # al componente anterior.
        if source == "user" and source_component and source_component != component:
            self._density_sources[idx] = {"source": "", "component": ""}

        # 2) TXT de la muestra. Un RHOS no vacío tiene prioridad sobre el estándar.
        txt_value = self._metadata_density_value(sample_name, idx)
        if txt_value:
            self._set_density_programmatic(idx, txt_value, "txt", component)
            return "txt"

        # 3) Densidad estándar del nombre de componente.
        default_value = self._component_default_density(component)
        if default_value:
            self._set_density_programmatic(idx, default_value, "default", component)
            return "default"

        # 4) Sin ninguna fuente: RHOS queda vacío y validate_all bloqueará la
        # corrida cuando RHOLAW=CUSTOM.
        self._set_density_programmatic(idx, "", "", component)
        return ""

    def _normalized_shape_sources(self, raw):
        out = {
            1: {"source": "", "component": ""},
            2: {"source": "", "component": ""},
        }
        if not isinstance(raw, dict):
            return out
        for idx in (1, 2):
            item = raw.get(str(idx), raw.get(idx, {}))
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip().lower()
            if source not in {"", "user", "txt", "default"}:
                source = ""
            out[idx] = {
                "source": source,
                "component": str(item.get("component", "") or "").strip(),
            }
        return out

    def _component_default_shape(self, component):
        component = str(component or "").strip()
        if not component:
            return ""
        text = str(self.component_shape_defaults.get(component, "") or "").strip()
        if not text:
            return ""
        try:
            value = parse_fortran_float(text)
            if not (0.0 < value <= 1.0):
                return ""
        except Exception:
            return ""
        return text

    def _metadata_shape_value(self, sample_name, idx):
        if not sample_name or not self.sample_metadata:
            return ""
        _matched, record = self._sample_metadata_match(sample_name)
        if not isinstance(record, dict):
            return ""
        target = f"SHAPEFACT({idx},0)"
        for column, value in record.items():
            if self._metadata_column_to_key(column).upper() != target:
                continue
            text = "" if value is None else str(value).strip()
            if text.lower() == "nan":
                text = ""
            return text
        return ""

    def _set_shape_programmatic(self, idx, value, source, component):
        key = f"SHAPEFACT({idx},0)"
        self._set_widget_value(key, "" if value is None else str(value))
        self._shape_sources[idx] = {
            "source": str(source or "").strip().lower(),
            "component": str(component or "").strip(),
        }
        self._update_shape_origin_tooltip(idx)

    def _update_shape_origin_tooltip(self, idx):
        key = f"SHAPEFACT({idx},0)"
        widget = self.param_widgets.get(key)
        if widget is None:
            return
        state = self._shape_sources.get(idx, {})
        source = str(state.get("source", ""))
        component = str(state.get("component", ""))
        source_label = {
            "user": "usuario (Parámetros)",
            "txt": "TXT de la muestra",
            "default": "shape factor estándar del componente",
            "": "sin dato",
        }.get(source, source)
        base = FIELD_HELP.get(key, "")
        extra = f"Origen actual: {source_label}"
        if component:
            extra += f" — {component}"
        widget.setToolTip((base + "\n" + extra).strip())

    def _on_shape_user_edited(self, idx):
        comp = self._group_components(idx)
        self._shape_sources[idx] = {
            "source": "user",
            "component": comp[0] if comp else "",
        }
        self._update_shape_origin_tooltip(idx)

    def _on_shape_editing_finished(self, idx):
        # Borrar manualmente SHAPEFACT elimina el override del usuario y vuelve
        # a la cadena TXT > estándar > vacío.
        key = f"SHAPEFACT({idx},0)"
        if self._widget_value(key).strip():
            return
        state = self._shape_sources.get(idx, {})
        if state.get("source") == "user":
            self._shape_sources[idx] = {"source": "", "component": ""}
            comp = self._group_components(idx)
            self._resolve_shape_for_component(idx, comp[0] if comp else "")

    def _resolve_shape_for_component(self, idx, component, sample_name=None):
        component = str(component or "").strip()
        if sample_name is None:
            sample_name = str(self._active_sample_name or "").strip()

        state = self._shape_sources.get(idx, {"source": "", "component": ""})
        source = str(state.get("source", "")).strip().lower()
        source_component = str(state.get("component", "") or "").strip()

        # 1) El valor manual siempre gana mientras pertenezca al componente actual.
        if source == "user" and (not source_component or source_component == component):
            if component and not source_component:
                self._shape_sources[idx]["component"] = component
            self._update_shape_origin_tooltip(idx)
            return "user"

        if source == "user" and source_component and source_component != component:
            self._shape_sources[idx] = {"source": "", "component": ""}

        # 2) TXT de la muestra.
        txt_value = self._metadata_shape_value(sample_name, idx)
        if txt_value:
            self._set_shape_programmatic(idx, txt_value, "txt", component)
            return "txt"

        # 3) Shape factor estándar del componente.
        default_value = self._component_default_shape(component)
        if default_value:
            self._set_shape_programmatic(idx, default_value, "default", component)
            return "default"

        # 4) Sin dato: queda vacío. Si la ley elegida lo requiere, validate_all
        # impedirá correr/guardar.
        self._set_shape_programmatic(idx, "", "", component)
        return ""

    def _selected_available_components(self):
        if not hasattr(self, "tbl_available"):
            return []
        rows = self.tbl_available.selectionModel().selectedRows() if self.tbl_available.selectionModel() else []
        if not rows:
            return []
        item = self.tbl_available.item(rows[0].row(), 0)
        text = item.text().strip() if item is not None else ""
        return [text] if text else []

    def _group_components(self, idx):
        lst = getattr(self, f"lst_group{idx}")
        return [lst.item(i).text().strip() for i in range(lst.count()) if lst.item(i).text().strip()]

    def _set_group_components(self, idx, components):
        lst = getattr(self, f"lst_group{idx}")
        lst.clear()
        selected_comp = ""
        for comp in components:
            comp = str(comp).strip()
            if comp:
                lst.addItem(comp)
                selected_comp = comp
                break
        self._resolve_density_for_component(idx, selected_comp)
        self._resolve_shape_for_component(idx, selected_comp)

    def _add_selected_to_group(self, idx):
        selected = self._selected_available_components()
        if not selected:
            return

        comp = selected[0]
        other_idx = 2 if idx == 1 else 1
        other = self._group_components(other_idx)
        if other and other[0] == comp:
            QMessageBox.warning(self, "GDEINPUT", "El mismo componente no puede ser componente 1 y componente 2.")
            return

        self._set_group_components(idx, [comp])
        self.tbl_available.clearSelection()
        self.tbl_available.setCurrentCell(-1, -1)

        if len(self._group_components(1)) == 1 and len(self._group_components(2)) == 1:
            self.autofill_from_componentry()
        else:
            self.update_preview()

    def _remove_selected_from_group(self, idx):
        lst = getattr(self, f"lst_group{idx}")
        selected = set(it.text().strip() for it in lst.selectedItems())
        if not selected:
            return
        remaining = [c for c in self._group_components(idx) if c not in selected]
        self._set_group_components(idx, remaining)
        self.update_preview()

    # =========================================================
    # Values
    # =========================================================
    def _widget_value(self, key):
        key = key.upper()
        widget = self.param_widgets.get(key)
        if widget is None:
            return ""

        if isinstance(widget, QComboBox):
            return widget.currentText().strip()

        if isinstance(widget, QLineEdit):
            return widget.text().strip()

        return ""

    def _set_widget_value(self, key, value):
        key = key.upper()
        widget = self.param_widgets.get(key)
        if widget is None:
            return

        value = strip_quotes(str(value).strip())
        if key in LOGICAL_KEYWORDS:
            value = normalize_bool(value)

        if isinstance(widget, QComboBox):
            set_combo_text(widget, value)
        elif isinstance(widget, QLineEdit):
            widget.setText(value)

    def _assignment_value(self, key):
        key = key.upper()
        value = self._widget_value(key)

        if key in TEXT_KEYWORDS:
            return quote_fortran(value)
        if key in LOGICAL_KEYWORDS:
            return normalize_bool(value)
        if key in NUMERIC_KEYWORDS:
            # PYFLOW/Fortran necesita punto decimal. La GUI acepta tanto coma
            # como punto, pero el .dat SIEMPRE se exporta con punto.
            if not str(value).strip():
                return ""
            return format_number(parse_fortran_float(value))
        return value.strip()

    def _field_comment(self, key):
        unit = FIELD_UNITS.get(key, "")
        unit_txt = f" [{unit}]" if unit else ""
        return f"{FIELD_LABELS.get(key, key)}{unit_txt}"

    def _field_assignment_lines(self, key, optional=False):
        value = self._assignment_value(key)
        if optional and not str(value).strip():
            return []
        return [f"{key} = {value}    ! {self._field_comment(key)}"]

    # =========================================================
    # Phi classes and weights
    # =========================================================
    def update_classes(self):
        old_weights = self._collect_weight_values()
        errors = []

        for idx in (1, 2):
            try:
                phimin = parse_fortran_float(self._widget_value(f"PHIMIN({idx})"))
                phimax = parse_fortran_float(self._widget_value(f"PHIMAX({idx})"))
                dphi = parse_fortran_float(self._widget_value(f"DPHI({idx})"))
                nclass, phis = calculate_nclass_and_phi(phimin, phimax, dphi)
                self._set_widget_value(f"NCLASS({idx})", str(nclass))
                self._build_weight_table(idx, phis, old_weights.get(idx, {}))
            except Exception as e:
                errors.append(f"Componente {idx}: {e}")
                self._build_weight_table(idx, [], old_weights.get(idx, {}))

        if errors:
            self.lbl_weight_status.setText("Pesos: error — " + " | ".join(errors))
        else:
            n1 = len(self.weight_phi.get(1, []))
            n2 = len(self.weight_phi.get(2, []))
            self.lbl_weight_status.setText(f"Pesos: componente 1 N={n1}; componente 2 N={n2}.")

        self.update_preview()

    def _build_weight_table(self, idx, phis, previous_values=None):
        previous_values = previous_values or {}
        table = self.weight_tables[idx]
        self.weight_phi[idx] = list(phis)
        self.weight_widgets[idx] = {}

        table.blockSignals(True)
        table.setRowCount(len(phis))

        for row, phi in enumerate(phis):
            j = row + 1
            key = f"WEIGHT({idx},{j})"

            item_class = QTableWidgetItem(str(j))
            item_class.setFlags(item_class.flags() & ~Qt.ItemIsEditable)
            item_class.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 0, item_class)

            item_phi = QTableWidgetItem(format_number(phi))
            item_phi.setFlags(item_phi.flags() & ~Qt.ItemIsEditable)
            item_phi.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 1, item_phi)

            item_key = QTableWidgetItem(key)
            item_key.setFlags(item_key.flags() & ~Qt.ItemIsEditable)
            item_key.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            table.setItem(row, 2, item_key)

            value = previous_values.get(key, "")
            item_val = QTableWidgetItem(str(value))
            item_val.setFlags(item_val.flags() & ~Qt.ItemIsEditable)
            item_val.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 3, item_val)
            self.weight_widgets[idx][key] = item_val

        try:
            table.resizeColumnsToContents()
        except Exception:
            pass

        table.blockSignals(False)

    def _on_weight_cell_changed(self, row, col):
        if col == 3:
            self.update_preview()

    def _collect_weight_values(self):
        out = {1: {}, 2: {}}
        for idx in (1, 2):
            table = self.weight_tables.get(idx)
            if table is None:
                continue
            for row in range(table.rowCount()):
                key_item = table.item(row, 2)
                val_item = table.item(row, 3)
                if key_item is None:
                    continue
                key = key_item.text().strip().upper()
                val = val_item.text().strip() if val_item is not None else ""
                out[idx][key] = val
        return out

    def _set_weight_value(self, idx, j, value):
        table = self.weight_tables[idx]
        row = int(j) - 1
        if row < 0 or row >= table.rowCount():
            return
        item = table.item(row, 3)
        if item is None:
            item = QTableWidgetItem("")
            table.setItem(row, 3, item)
        item.setText(str(value))

    def clear_weights(self):
        for idx in (1, 2):
            table = self.weight_tables[idx]
            table.blockSignals(True)
            for row in range(table.rowCount()):
                item = table.item(row, 3)
                if item is not None:
                    item.setText("")
            table.blockSignals(False)
        self.update_preview()

    def normalize_weights_to_100(self):
        QMessageBox.information(
            self,
            "GDEINPUT",
            "En el modo Dos componentes los wt% se conservan respecto de la muestra completa. No se normalizan a 100%."
        )

    # =========================================================
    # Autocomplete from current Componentry database
    # =========================================================
    def _get_current_db(self):
        try:
            return self.app.current_db()
        except Exception:
            return None

    def autofill_from_componentry(self):
        db = self._get_current_db()
        if db is None:
            QMessageBox.warning(self, "GDEINPUT", "No hay muestra cargada.")
            return

        group1 = self._group_components(1)
        group2 = self._group_components(2)
        if len(group1) != 1 or len(group2) != 1:
            QMessageBox.warning(
                self,
                "GDEINPUT",
                "Seleccioná exactamente un componente como componente 1 y exactamente uno como componente 2."
            )
            return
        if group1[0] == group2[0]:
            QMessageBox.warning(self, "GDEINPUT", "Los dos componentes seleccionados deben ser diferentes.")
            return

        if not hasattr(db, "total") or not isinstance(db.total, pd.DataFrame) or db.total.empty:
            QMessageBox.warning(self, "GDEINPUT", "La muestra no tiene la matriz wt% por componente (db.total).")
            return

        try:
            total_df = db.total.copy()
            total_df.index = total_df.index.astype(float)
            all_phi = sorted(set(float(v) for v in total_df.index if np.isfinite(float(v))))
            if len(all_phi) < 1:
                raise ValueError("No se encontraron clases phi en db.total.")

            dphi = infer_dphi(all_phi)

            for idx, comp in ((1, group1[0]), (2, group2[0])):
                if comp not in total_df.columns:
                    raise ValueError(f"El componente '{comp}' no existe en db.total.")

                series = pd.to_numeric(total_df[comp], errors="coerce")
                active_phi = [
                    float(phi) for phi, value in series.items()
                    if np.isfinite(float(value)) and float(value) > 0.0
                ]
                if not active_phi:
                    raise ValueError(f"El componente '{comp}' no tiene fracciones wt% > 0 en esta muestra.")

                phimin = min(active_phi)
                phimax = max(active_phi)
                def phi_text(value):
                    out = f"{float(value):.10f}".rstrip("0")
                    if out.endswith("."):
                        out += "0"
                    return out

                self._set_widget_value(f"PHIMIN({idx})", phi_text(phimin))
                self._set_widget_value(f"PHIMAX({idx})", phi_text(phimax))
                self._set_widget_value(f"DPHI({idx})", phi_text(dphi))

            self.update_classes()
            self._fill_weights_from_db(db, 1, [group1[0]])
            self._fill_weights_from_db(db, 2, [group2[0]])
            self.update_preview()
        except Exception as e:
            QMessageBox.warning(self, "Autocargar desde componentry", str(e))

    def _infer_phi_values_from_db(self, db):
        vals = []

        if hasattr(db, "gsd") and isinstance(db.gsd, pd.DataFrame) and "phi" in db.gsd.columns:
            df = db.gsd.copy()
            if "wt" in df.columns:
                df = df[df["wt"].astype(float) > 0]
            vals = [float(v) for v in df["phi"].values if np.isfinite(float(v))]

        if not vals and hasattr(db, "total") and isinstance(db.total, pd.DataFrame):
            vals = [float(v) for v in db.total.index.tolist() if np.isfinite(float(v))]

        return sorted(set(vals))

    def _fill_weights_from_db(self, db, idx, components):
        if not hasattr(db, "total") or not isinstance(db.total, pd.DataFrame) or db.total.empty:
            raise ValueError("La muestra no tiene matriz db.total cargada.")

        total_df = db.total.copy()
        total_df.index = total_df.index.astype(float)

        missing = [c for c in components if c not in total_df.columns]
        if missing:
            raise ValueError("No existen en db.total: " + ", ".join(missing))

        table = self.weight_tables[idx]
        table.blockSignals(True)

        for row, phi in enumerate(self.weight_phi[idx]):
            value = 0.0
            for comp in components:
                value += value_at_phi(total_df[comp], phi)

            item = table.item(row, 3)
            if item is None:
                item = QTableWidgetItem("")
                table.setItem(row, 3, item)
            item.setText(f"{value:.6f}")

        table.blockSignals(False)

    # =========================================================
    # GDEINPUT project save / load
    # =========================================================
    def _current_project_sample_name(self):
        """Nombre exacto de la muestra activa tal como la expone Componentry."""
        name = ""
        try:
            name = str(self.app.current_sample_name()).strip()
        except Exception:
            name = ""
        if not name and hasattr(self, "cmb_sample"):
            name = self.cmb_sample.currentText().strip()
        return name

    def _db_for_sample_name(self, sample_name):
        sample_dbs = getattr(self.app, "sample_dbs", {})
        if isinstance(sample_dbs, dict):
            return sample_dbs.get(str(sample_name))
        return None

    def _project_sample_signature(self, sample_name):
        """Información mínima para comprobar que el proyecto sigue ligado a la misma muestra."""
        db = self._db_for_sample_name(sample_name)
        if db is None:
            return {"components": [], "phi_classes": []}

        components = []
        if hasattr(db, "components") and db.components:
            components = [str(c) for c in db.components]
        elif hasattr(db, "total") and isinstance(db.total, pd.DataFrame):
            components = [str(c) for c in db.total.columns]

        phi_classes = []
        if hasattr(db, "total") and isinstance(db.total, pd.DataFrame) and not db.total.empty:
            for value in db.total.index.tolist():
                try:
                    f = float(value)
                    if np.isfinite(f):
                        phi_classes.append(f)
                except Exception:
                    pass
            phi_classes = sorted(set(phi_classes))
        elif hasattr(db, "gsd") and isinstance(db.gsd, pd.DataFrame) and "phi" in db.gsd.columns:
            for value in db.gsd["phi"].tolist():
                try:
                    f = float(value)
                    if np.isfinite(f):
                        phi_classes.append(f)
                except Exception:
                    pass
            phi_classes = sorted(set(phi_classes))

        return {
            "components": components,
            "phi_classes": phi_classes,
        }

    def _sample_metadata_dataframe_from_dict(self, metadata):
        metadata = metadata if isinstance(metadata, dict) else {}
        if not metadata:
            return pd.DataFrame()

        keys = []
        for record in metadata.values():
            if not isinstance(record, dict):
                continue
            for key in record.keys():
                key = str(key)
                if key not in keys:
                    keys.append(key)

        rows = []
        for sample, record in metadata.items():
            record = record if isinstance(record, dict) else {}
            row = {"SAMPLE": str(sample)}
            for key in keys:
                value = record.get(key, "")
                row[key] = "" if value is None else str(value)
            rows.append(row)
        return pd.DataFrame(rows, columns=["SAMPLE"] + keys, dtype=str)

    def _project_state_dict(self):
        sample_name = self._current_project_sample_name()
        if not sample_name:
            raise ValueError("No hay una muestra activa en Componentry.")

        weights = self._collect_weight_values()
        parameters = {
            str(key): self._widget_value(key)
            for key in sorted(self.param_widgets.keys())
        }

        # El contenido del TXT se incrusta en el proyecto. De este modo el
        # proyecto puede restaurarse aunque el TXT original haya sido movido.
        embedded_metadata = {}
        for sample, record in self.sample_metadata.items():
            embedded_metadata[str(sample)] = {
                str(key): "" if value is None else str(value)
                for key, value in (record.items() if isinstance(record, dict) else [])
            }

        return {
            "format": PROJECT_FILE_FORMAT,
            "schema_version": PROJECT_SCHEMA_VERSION,
            "gdeinput_version": PLUGIN_VERSION,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "sample": {
                "name": sample_name,
                "signature": self._project_sample_signature(sample_name),
            },
            "components": {
                "1": self._group_components(1),
                "2": self._group_components(2),
            },
            "parameters": parameters,
            "weights": {
                "1": dict(weights.get(1, {})),
                "2": dict(weights.get(2, {})),
            },
            "sample_metadata": embedded_metadata,
            "sample_metadata_source": self.sample_metadata_path,
            "component_density_defaults": {
                str(k): "" if v is None else str(v)
                for k, v in self.component_density_defaults.items()
            },
            "component_shape_defaults": {
                str(k): "" if v is None else str(v)
                for k, v in self.component_shape_defaults.items()
            },
            "density_sources": {
                "1": dict(self._density_sources.get(1, {})),
                "2": dict(self._density_sources.get(2, {})),
            },
            "shape_sources": {
                "1": dict(self._shape_sources.get(1, {})),
                "2": dict(self._shape_sources.get(2, {})),
            },
            "output_path": self.output_edit.text().strip(),
            "pyflow_exe_path": self.pyflow_exe_path,
            "active_tab": int(self.tabs.currentIndex()) if hasattr(self, "tabs") else 0,
        }

    def save_project_file(self):
        try:
            state = self._project_state_dict()
        except Exception as e:
            QMessageBox.warning(self, "Guardar proyecto", str(e))
            return

        sample_name = state.get("sample", {}).get("name", "proyecto")
        suggested_name = sanitize_filename(sample_name) or "gdeinput_proyecto"
        if not suggested_name.lower().endswith(".gdeinput"):
            suggested_name += ".gdeinput"

        start = self.project_state_path.strip() if self.project_state_path else ""
        if not start:
            folder = ""
            try:
                host_project = getattr(self.app, "project_path", None)
                if host_project:
                    folder = os.path.dirname(str(host_project))
            except Exception:
                folder = ""
            if not folder:
                folder = os.getcwd()
            start = os.path.join(folder, suggested_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar proyecto GDEINPUT",
            start,
            "GDEINPUT project (*.gdeinput);;JSON (*.json);;All files (*)"
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".gdeinput"

        try:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.write("\n")
            self.project_state_path = path
        except Exception as e:
            QMessageBox.critical(self, "Guardar proyecto", f"No pude guardar el proyecto:\n{e}")
            return

        QMessageBox.information(
            self,
            "Guardar proyecto",
            "Proyecto GDEINPUT guardado.\n\n"
            f"Muestra: {sample_name}\n"
            f"Archivo: {path}"
        )

    def _validate_project_against_componentry(self, state):
        if not isinstance(state, dict):
            raise ValueError("El archivo de proyecto no contiene un objeto válido.")
        if state.get("format") != PROJECT_FILE_FORMAT:
            raise ValueError("El archivo seleccionado no es un proyecto GDEINPUT válido.")

        schema = state.get("schema_version")
        if schema != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"Versión de proyecto no compatible: {schema}. "
                f"Esta versión de GDEINPUT usa el esquema {PROJECT_SCHEMA_VERSION}."
            )

        sample_info = state.get("sample", {})
        sample_name = str(sample_info.get("name", "")).strip()
        if not sample_name:
            raise ValueError("El proyecto no contiene el nombre de la muestra.")

        available_samples = self._available_sample_names()
        if sample_name not in available_samples:
            available_txt = ", ".join(available_samples) if available_samples else "ninguna"
            raise ValueError(
                "La muestra guardada en el proyecto no existe en el Componentry actual.\n\n"
                f"Muestra requerida: {sample_name}\n"
                f"Muestras disponibles: {available_txt}\n\n"
                "Cargá primero el proyecto/base de Componentry que contiene esa misma muestra."
            )

        db = self._db_for_sample_name(sample_name)
        if db is None:
            raise ValueError(f"No pude acceder a la base de datos de la muestra '{sample_name}'.")

        current_signature = self._project_sample_signature(sample_name)
        current_components = set(current_signature.get("components", []))
        saved_components = state.get("components", {}) if isinstance(state.get("components"), dict) else {}

        for idx in (1, 2):
            group = saved_components.get(str(idx), [])
            if group is None:
                group = []
            if not isinstance(group, list):
                raise ValueError(f"La selección del componente {idx} está dañada en el proyecto.")
            for comp in group:
                comp = str(comp)
                if comp not in current_components:
                    raise ValueError(
                        f"El proyecto requiere el componente '{comp}' como componente {idx}, "
                        f"pero ese componente no existe en la muestra '{sample_name}'."
                    )

        saved_signature = sample_info.get("signature", {})
        if isinstance(saved_signature, dict):
            saved_phi = saved_signature.get("phi_classes", [])
            current_phi = current_signature.get("phi_classes", [])
            if saved_phi and current_phi:
                try:
                    saved_arr = np.asarray([float(v) for v in saved_phi], dtype=float)
                    current_arr = np.asarray([float(v) for v in current_phi], dtype=float)
                    same_phi = (
                        saved_arr.shape == current_arr.shape
                        and np.allclose(saved_arr, current_arr, rtol=0.0, atol=1e-9)
                    )
                except Exception:
                    same_phi = False
                if not same_phi:
                    raise ValueError(
                        "La muestra existe y tiene el mismo nombre, pero sus clases φ actuales "
                        "no coinciden con las que tenía cuando se guardó el proyecto. "
                        "No se restaurará el proyecto para evitar asociar pesos a otra granulometría."
                    )

        return sample_name

    def _activate_project_sample(self, sample_name):
        """Selecciona exactamente la muestra requerida en Componentry."""
        self._changing_sample_from_gdeinput = True
        try:
            host_combo = getattr(self.app, "cmb_sample", None)
            if isinstance(host_combo, QComboBox):
                idx = host_combo.findText(sample_name)
                if idx < 0:
                    raise ValueError(
                        f"Componentry contiene '{sample_name}' en sus datos, pero no aparece en su selector de muestras."
                    )
                if host_combo.currentIndex() != idx:
                    host_combo.setCurrentIndex(idx)
        finally:
            self._changing_sample_from_gdeinput = False

        # La aplicación host procesa el cambio de muestra mediante su propio
        # QComboBox. Después se reconstruyen los controles de GDEINPUT.
        self.refresh_from_app()
        current = self._current_project_sample_name()
        if current != sample_name:
            raise ValueError(
                f"No se pudo activar la muestra requerida. Proyecto: '{sample_name}'; activa: '{current or '—'}'."
            )

    def _restore_project_state(self, state, path=""):
        sample_name = self._validate_project_against_componentry(state)

        # Restaurar primero los datos por muestra incrustados. Si refresh_from_app
        # los aplica durante el cambio de muestra, luego los parámetros exactos
        # guardados en el proyecto vuelven a imponerse debajo.
        metadata = state.get("sample_metadata", {})
        clean_metadata = {}
        if isinstance(metadata, dict):
            for sample, record in metadata.items():
                if not isinstance(record, dict):
                    continue
                clean_metadata[str(sample)] = {
                    str(key): "" if value is None else str(value)
                    for key, value in record.items()
                }
        self.sample_metadata = clean_metadata
        self.sample_metadata_path = str(state.get("sample_metadata_source", "") or "")

        defaults = state.get("component_density_defaults", {})
        if isinstance(defaults, dict):
            self.component_density_defaults = {
                str(k): "" if v is None else str(v)
                for k, v in defaults.items()
            }

        shape_defaults = state.get("component_shape_defaults", {})
        if isinstance(shape_defaults, dict):
            self.component_shape_defaults = {
                str(k): "" if v is None else str(v)
                for k, v in shape_defaults.items()
            }

        self._activate_project_sample(sample_name)

        components = state.get("components", {}) if isinstance(state.get("components"), dict) else {}
        self._set_group_components(1, components.get("1", []))
        self._set_group_components(2, components.get("2", []))

        parameters = state.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("La sección de parámetros del proyecto está dañada.")
        for key, value in parameters.items():
            key_up = str(key).upper()
            if key_up in self.param_widgets:
                self._set_widget_value(key_up, "" if value is None else str(value))

        saved_density_sources = state.get("density_sources")
        if isinstance(saved_density_sources, dict):
            self._density_sources = self._normalized_density_sources(saved_density_sources)
        else:
            # Compatibilidad con proyectos anteriores a V8.17: un RHOS guardado
            # se considera decisión del usuario para no perderlo al abrirlo.
            self._density_sources = {1: {"source": "", "component": ""}, 2: {"source": "", "component": ""}}
            for idx in (1, 2):
                rho = self._widget_value(f"RHOS({idx},0)").strip()
                comp = self._group_components(idx)
                if rho:
                    self._density_sources[idx] = {"source": "user", "component": comp[0] if comp else ""}

        saved_shape_sources = state.get("shape_sources")
        if isinstance(saved_shape_sources, dict):
            self._shape_sources = self._normalized_shape_sources(saved_shape_sources)
        else:
            # Compatibilidad con proyectos anteriores a V8.18. Si había un
            # SHAPEFACT guardado, se conserva como decisión del usuario.
            self._shape_sources = {1: {"source": "", "component": ""}, 2: {"source": "", "component": ""}}
            for idx in (1, 2):
                shape = self._widget_value(f"SHAPEFACT({idx},0)").strip()
                comp = self._group_components(idx)
                if shape:
                    self._shape_sources[idx] = {"source": "user", "component": comp[0] if comp else ""}

        for idx in (1, 2):
            comp = self._group_components(idx)
            selected = comp[0] if comp else ""
            self._resolve_density_for_component(idx, selected, sample_name=sample_name)
            self._resolve_shape_for_component(idx, selected, sample_name=sample_name)
            self._update_density_fields(idx)
            self._update_drag_law_fields(idx)
        self.update_classes()

        weights = state.get("weights", {})
        if isinstance(weights, dict):
            for idx in (1, 2):
                values = weights.get(str(idx), {})
                if not isinstance(values, dict):
                    continue
                table = self.weight_tables.get(idx)
                if table is None:
                    continue
                table.blockSignals(True)
                try:
                    for key, value in values.items():
                        m = re.match(r"^WEIGHT\(\s*(\d+)\s*,\s*(\d+)\s*\)$", str(key), re.IGNORECASE)
                        if not m or int(m.group(1)) != idx:
                            continue
                        self._set_weight_value(idx, int(m.group(2)), "" if value is None else str(value))
                finally:
                    table.blockSignals(False)

        # Restaurar visualización de los datos por muestra incrustados.
        if hasattr(self, "tbl_sample_metadata"):
            df = self._sample_metadata_dataframe_from_dict(self.sample_metadata)
            if df.empty:
                self.tbl_sample_metadata.clear()
                self.tbl_sample_metadata.setRowCount(0)
                self.tbl_sample_metadata.setColumnCount(0)
            else:
                self._show_metadata_dataframe(df)
        if hasattr(self, "lbl_metadata_file"):
            if self.sample_metadata:
                source = self.sample_metadata_path or "datos incrustados en el proyecto"
                self.lbl_metadata_file.setText(f"Datos restaurados desde proyecto. Origen: {source}")
            else:
                self.lbl_metadata_file.setText("TXT cargado: —")
        if hasattr(self, "lbl_metadata_status"):
            if self.sample_metadata:
                self.lbl_metadata_status.setText(
                    f"Proyecto restaurado: datos por muestra disponibles para {len(self.sample_metadata)} muestra(s)."
                )
            else:
                self.lbl_metadata_status.setText("Proyecto restaurado sin datos por muestra incrustados.")

        output_path = str(state.get("output_path", "") or "").strip()
        if output_path:
            self.output_edit.setText(output_path)
        else:
            self._set_default_output_path()

        pyflow_path = str(state.get("pyflow_exe_path", "") or "").strip()
        if pyflow_path and os.path.isfile(pyflow_path):
            self._set_pyflow_executable(pyflow_path)

        try:
            tab_index = int(state.get("active_tab", 0))
            if 0 <= tab_index < self.tabs.count():
                self.tabs.setCurrentIndex(tab_index)
        except Exception:
            pass

        self.project_state_path = path or self.project_state_path
        self._active_sample_name = sample_name
        self._sample_work_states[sample_name] = self._snapshot_current_sample_state()
        self.update_preview()

    def open_project_file(self):
        start = self.project_state_path.strip() if self.project_state_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir proyecto GDEINPUT",
            start,
            "GDEINPUT project (*.gdeinput *.json);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                state = json.load(f)
            sample_name = self._validate_project_against_componentry(state)
        except Exception as e:
            QMessageBox.warning(self, "Abrir proyecto", str(e))
            return

        try:
            self._restore_project_state(state, path)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Abrir proyecto",
                "El proyecto pasó la validación inicial pero no pudo restaurarse completamente:\n"
                + str(e)
            )
            return

        group1 = self._group_components(1)
        group2 = self._group_components(2)
        QMessageBox.information(
            self,
            "Abrir proyecto",
            "Proyecto GDEINPUT restaurado.\n\n"
            f"Muestra: {sample_name}\n"
            f"Componente 1: {group1[0] if group1 else '—'}\n"
            f"Componente 2: {group2[0] if group2 else '—'}"
        )

    # =========================================================
    # PYFLOW integrated runner
    # =========================================================
    def _set_pyflow_executable(self, path, remember=True):
        path = os.path.abspath(str(path).strip()) if path else ""
        self.pyflow_exe_path = path if path and os.path.isfile(path) else ""
        if hasattr(self, "pyflow_exe_edit"):
            self.pyflow_exe_edit.setText(self.pyflow_exe_path)

        self.pyflow_execution_mode = ""
        if self.pyflow_exe_path:
            btype = detect_binary_type(self.pyflow_exe_path)
            if btype == "linux_elf":
                self.pyflow_execution_mode = "wsl"
                if hasattr(self, "lbl_pyflow_runner"):
                    self.lbl_pyflow_runner.setText(
                        "Modo de ejecución: WSL — mismo runner del launcher batch"
                    )
            elif hasattr(self, "lbl_pyflow_runner"):
                self.lbl_pyflow_runner.setText(
                    f"Formato detectado: {btype} — este runner requiere el binario Linux ELF"
                )
        elif hasattr(self, "lbl_pyflow_runner"):
            self.lbl_pyflow_runner.setText("Modo de ejecución: —")

        if remember and self.pyflow_exe_path:
            try:
                self._pyflow_settings.setValue("pyflow_exe_path", self.pyflow_exe_path)
                self._pyflow_settings.sync()
            except Exception:
                pass
        return self.pyflow_exe_path

    def _clear_pyflow_executable_preference(self):
        try:
            self._pyflow_settings.remove("pyflow_exe_path")
            self._pyflow_settings.sync()
        except Exception:
            pass

    def _restore_pyflow_executable_preference(self):
        """Recupera el ejecutable elegido en una sesión anterior, si todavía existe."""
        try:
            saved = str(self._pyflow_settings.value("pyflow_exe_path", "") or "").strip()
        except Exception:
            saved = ""
        if saved and os.path.isfile(saved):
            self._set_pyflow_executable(saved, remember=False)
            return saved
        return ""

    def _windows_host_architecture(self):
        arch = (
            os.environ.get("PROCESSOR_ARCHITEW6432")
            or os.environ.get("PROCESSOR_ARCHITECTURE")
            or platform.machine()
            or ""
        ).strip().lower()
        if arch in {"amd64", "x86_64", "x64"}:
            return "x64"
        if arch in {"x86", "i386", "i686"}:
            return "x86"
        if arch in {"arm64", "aarch64"}:
            return "arm64"
        return arch or "unknown"

    def _wsl_available(self, force=False):
        """Comprueba que WSL y una distribución Linux funcional estén disponibles."""
        if os.name != "nt":
            return False
        if self._wsl_probe_cache is not None and not force:
            return bool(self._wsl_probe_cache)
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(
                ["wsl.exe", "sh", "-lc", "printf GDEINPUT_WSL_OK"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8,
                creationflags=flags,
            )
            self._wsl_probe_cache = proc.returncode == 0 and "GDEINPUT_WSL_OK" in (proc.stdout or "")
        except Exception:
            self._wsl_probe_cache = False
        return bool(self._wsl_probe_cache)

    def _windows_path_to_wsl(self, path):
        """Convierte una ruta de Windows a la ruta visible dentro de WSL."""
        path = os.path.abspath(str(path))
        if os.name != "nt":
            return path.replace("\\", "/")
        if not self._wsl_available():
            raise RuntimeError(
                "PYFLOW es un binario Linux/ELF y requiere WSL, pero GDEINPUT no pudo iniciar WSL. "
                "Instalá/activá WSL y una distribución Linux, o usá una compilación PE nativa de Windows."
            )
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.run(
            ["wsl.exe", "wslpath", "-a", "-u", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            creationflags=flags,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "WSL no pudo convertir esta ruta de Windows:\n"
                + path
                + "\n\n"
                + (proc.stderr or proc.stdout or "Error desconocido").strip()
            )
        out = (proc.stdout or "").strip()
        if not out:
            raise RuntimeError("WSL devolvió una ruta vacía para: " + path)
        return out

    def _inspect_pyflow_executable(self, path):
        """Determina si PYFLOW debe ejecutarse nativamente o mediante WSL."""
        result = {
            "path": os.path.abspath(str(path)) if path else "",
            "format": "unknown",
            "machine": "unknown",
            "compatible": None,
            "runner": "",
            "detail": "",
        }
        if not path or not os.path.isfile(path):
            result["compatible"] = False
            result["detail"] = "El archivo no existe."
            return result
        try:
            with open(path, "rb") as f:
                hdr = f.read(64)
                if hdr.startswith(b"\x7fELF"):
                    elf_class = "64-bit" if len(hdr) > 4 and hdr[4] == 2 else "32-bit"
                    endian = "little" if len(hdr) > 5 and hdr[5] == 1 else "big"
                    machine_code = int.from_bytes(hdr[18:20], endian, signed=False) if len(hdr) >= 20 else None
                    machine_names = {3: "x86", 62: "x86-64", 183: "ARM64"}
                    machine = machine_names.get(machine_code, f"ELF e_machine={machine_code}")
                    if os.name == "nt":
                        wsl_ok = self._wsl_available()
                        result.update(
                            format="ELF",
                            machine=f"{machine} {elf_class}",
                            runner="wsl" if wsl_ok else "",
                            compatible=wsl_ok,
                            detail=(
                                f"Binario Linux ELF {elf_class} ({machine}). Se ejecutará mediante WSL."
                                if wsl_ok else
                                f"Binario Linux ELF {elf_class} ({machine}). WSL no está disponible o no tiene una distribución funcional."
                            ),
                        )
                    else:
                        result.update(
                            format="ELF", machine=f"{machine} {elf_class}",
                            runner="native", compatible=True,
                            detail=f"Binario Linux ELF {elf_class} ({machine})."
                        )
                    return result

                if hdr[:2] != b"MZ":
                    result.update(
                        format="unknown", compatible=False,
                        detail="No tiene un encabezado ejecutable PE de Windows ni ELF de Linux."
                    )
                    return result
                if len(hdr) < 64:
                    raise ValueError("encabezado DOS incompleto")
                pe_offset = int.from_bytes(hdr[0x3C:0x40], "little")
                f.seek(pe_offset)
                sig = f.read(4)
                if sig != b"PE\x00\x00":
                    result.update(
                        format="MZ", compatible=False,
                        detail="Tiene encabezado MZ, pero no es un ejecutable PE moderno de Windows."
                    )
                    return result
                machine_code = int.from_bytes(f.read(2), "little")
        except Exception as exc:
            result["compatible"] = False
            result["detail"] = f"No se pudo inspeccionar el ejecutable: {exc}"
            return result

        names = {0x014C: "x86", 0x8664: "x64", 0xAA64: "arm64"}
        machine = names.get(machine_code, f"0x{machine_code:04X}")
        if os.name != "nt":
            result.update(
                format="PE", machine=machine, compatible=False,
                detail="Es un ejecutable PE de Windows y este sistema no es Windows."
            )
            return result
        host = self._windows_host_architecture()
        if host == "x64":
            compatible = machine in {"x86", "x64"}
        elif host == "x86":
            compatible = machine == "x86"
        elif host == "arm64":
            compatible = machine in {"x86", "x64", "arm64"}
        else:
            compatible = True
        result.update(format="PE", machine=machine, compatible=compatible, runner="native" if compatible else "")
        if compatible:
            result["detail"] = f"PE de Windows ({machine}); sistema detectado: {host}."
        else:
            result["detail"] = (
                f"El ejecutable es {machine}, pero Windows fue detectado como {host}. "
                "Esa combinación no se puede iniciar directamente."
            )
        return result

    def _prepare_pyflow_wsl_launcher(self, exe, input_path, run_dir):
        """Prepara el launcher WSL para la distribución ELF de PYFLOW 2.5."""
        exe_wsl = self._windows_path_to_wsl(exe)
        input_wsl = self._windows_path_to_wsl(input_path)
        out_wsl = self._windows_path_to_wsl(run_dir)
        script_path = os.path.join(run_dir, "__gdeinput_pyflow_wsl__.sh")
        q = shlex.quote
        lines = [
            "#!/usr/bin/env bash",
            "set -u",
            "export LC_ALL=C.UTF-8",
            "export LANG=C.UTF-8",
            f"EXE_SRC={q(exe_wsl)}",
            f"IN_SRC={q(input_wsl)}",
            f"OUT_DIR={q(out_wsl)}",
            "EXE_NAME=pyflow_2.5.exe",
            'STAGE_DIR="$(mktemp -d /tmp/gdeinput_pyflow_XXXXXX)"',
            'echo "[GDEINPUT/WSL] EXE_SRC=$EXE_SRC"',
            'echo "[GDEINPUT/WSL] IN_SRC=$IN_SRC"',
            'echo "[GDEINPUT/WSL] OUT_DIR=$OUT_DIR"',
            'echo "[GDEINPUT/WSL] STAGE_DIR=$STAGE_DIR"',
            'if [ ! -f "$EXE_SRC" ]; then echo "[GDEINPUT/WSL][ERROR] No existe el binario PYFLOW: $EXE_SRC" >&2; exit 11; fi',
            'if [ ! -f "$IN_SRC" ]; then echo "[GDEINPUT/WSL][ERROR] No existe input.dat: $IN_SRC" >&2; exit 12; fi',
            'mkdir -p -- "$OUT_DIR"',
            'cp -- "$EXE_SRC" "$STAGE_DIR/$EXE_NAME"',
            'cp -- "$IN_SRC" "$STAGE_DIR/input.dat"',
            'chmod +x -- "$STAGE_DIR/$EXE_NAME"',
            'cd "$STAGE_DIR"',
            'if command -v ldd >/dev/null 2>&1; then',
            '  MISSING_LIBS="$(ldd "./$EXE_NAME" 2>/dev/null | grep "not found" || true)"',
            '  if [ -n "$MISSING_LIBS" ]; then echo "[GDEINPUT/WSL][ERROR] Faltan bibliotecas Linux para ejecutar PYFLOW:" >&2; echo "$MISSING_LIBS" >&2; exit 14; fi',
            'fi',
            'echo "[GDEINPUT/WSL] Ejecutando PYFLOW dentro de WSL..."',
            "set +e",
            "printf 'continue\ncontinue\ncontinue\ncontinue\n' | \"./$EXE_NAME\"",
            "RC=$?",
            "set -e",
            'echo "[GDEINPUT/WSL] Código de salida de PYFLOW: $RC"',
            "shopt -s dotglob nullglob",
            'for item in "$STAGE_DIR"/*; do',
            '  name="$(basename -- "$item")"',
            '  if [ "$name" = "$EXE_NAME" ]; then continue; fi',
            '  rm -rf -- "$OUT_DIR/$name"',
            '  cp -a -- "$item" "$OUT_DIR/"',
            "done",
            "cd /",
            'rm -rf -- "$STAGE_DIR"',
            'exit "$RC"',
        ]
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        self.pyflow_wsl_script = script_path
        return script_path, self._windows_path_to_wsl(script_path)

    def _automatic_pyflow_search_roots(self):
        roots = []

        def add_dir(value):
            if not value:
                return
            try:
                value = os.path.abspath(str(value))
            except Exception:
                return
            if os.path.isdir(value) and value not in roots:
                roots.append(value)

        try:
            add_dir(os.path.dirname(os.path.abspath(__file__)))
        except Exception:
            pass
        try:
            if sys.argv and sys.argv[0]:
                add_dir(os.path.dirname(os.path.abspath(sys.argv[0])))
        except Exception:
            pass
        try:
            add_dir(os.path.dirname(os.path.abspath(sys.executable)))
        except Exception:
            pass
        add_dir(os.getcwd())

        # Priorizar ubicaciones concretas. No recorrer toda la carpeta HOME antes
        # de Downloads/OneDrive: eso puede agotar el límite de búsqueda y evitar
        # que se inspeccione justamente la carpeta donde está PYFLOW.
        try:
            home = os.path.expanduser("~")
            add_dir(os.path.join(home, "Downloads"))
            add_dir(os.path.join(home, "Desktop"))
            add_dir(os.path.join(home, "Documents"))
        except Exception:
            home = ""
        for env_name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            try:
                add_dir(os.environ.get(env_name, ""))
            except Exception:
                pass

        output_path = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if output_path:
            add_dir(os.path.dirname(os.path.abspath(output_path)))
        if self.project_state_path:
            add_dir(os.path.dirname(os.path.abspath(self.project_state_path)))

        # Si el proyecto/salida está, por ejemplo, en Escritorio\BLAST y PYFLOW
        # está en Escritorio\PYFLOW-v_2.5, el padre común también debe buscarse.
        for original in list(roots):
            parent = os.path.dirname(original)
            if parent and parent != original:
                add_dir(parent)
            grand = os.path.dirname(parent) if parent else ""
            # No recorrer la raíz de un disco completo (C:\) automáticamente.
            if grand and grand != parent and os.path.dirname(grand) != grand:
                add_dir(grand)
        return roots

    def _preferred_pyflow_executable_path(self):
        """Primera ruta: exactamente la que usa el launcher batch que ya funciona."""
        return os.path.join(DEFAULT_PYFLOW_FOLDER, DEFAULT_PYFLOW_EXECUTABLE)

    def _automatic_pyflow_candidates(self):
        """Busca únicamente binarios Linux ELF de PYFLOW, con o sin extensión .exe."""
        found = []
        seen_files = set()
        max_dirs_per_root = 3500
        max_depth = 6
        preferred_names = {
            "pyflow_2.5": 0,
            "pyflow_2.5.exe": 1,
            "pyflow": 2,
            "pyflow.exe": 3,
        }

        for root in self._automatic_pyflow_search_roots():
            root = os.path.abspath(root)
            # El límite se reinicia por raíz. De lo contrario una raíz grande
            # (por ejemplo HOME) puede impedir revisar Downloads u OneDrive.
            seen_dirs_root = set()
            for current, dirs, files in os.walk(root):
                norm_current = os.path.normcase(current)
                if norm_current in seen_dirs_root:
                    dirs[:] = []
                    continue
                seen_dirs_root.add(norm_current)
                if len(seen_dirs_root) > max_dirs_per_root:
                    dirs[:] = []
                    break
                try:
                    rel = os.path.relpath(current, root)
                    depth = 0 if rel == "." else rel.count(os.sep) + 1
                except Exception:
                    depth = max_depth
                if depth >= max_depth:
                    dirs[:] = []
                else:
                    skip = {
                        ".git", "__pycache__", "node_modules", "$recycle.bin",
                        "windows", "program files", "program files (x86)",
                        "appdata", "system volume information"
                    }
                    dirs[:] = [d for d in dirs if d.casefold() not in skip]

                for name in files:
                    low = name.casefold()
                    if not low.startswith("pyflow"):
                        continue
                    full = os.path.abspath(os.path.join(current, name))
                    norm = os.path.normcase(full)
                    if norm in seen_files:
                        continue
                    seen_files.add(norm)
                    if detect_binary_type(full) != "linux_elf":
                        continue
                    score = preferred_names.get(low, 4)
                    found.append((score, depth, len(full), full))

        found.sort(key=lambda item: (item[0], item[1], item[2], item[3].casefold()))
        return [item[-1] for item in found]

    def _find_pyflow_executable(self, ask_user=False):
        # 1) Probar primero rutas directas basadas en el launcher batch que ya
        # funciona. Esto evita búsquedas recursivas innecesarias y no abre diálogos.
        preferred = self._preferred_pyflow_executable_path()
        direct_candidates = [
            preferred,
            preferred + ".exe",
            os.path.join(DEFAULT_PYFLOW_FOLDER, "src", DEFAULT_PYFLOW_EXECUTABLE),
            os.path.join(DEFAULT_PYFLOW_FOLDER, "src", DEFAULT_PYFLOW_EXECUTABLE + ".exe"),
            os.path.join(DEFAULT_PYFLOW_FOLDER, "PYFLOW-v_2.5", "src", DEFAULT_PYFLOW_EXECUTABLE),
            os.path.join(DEFAULT_PYFLOW_FOLDER, "PYFLOW-v_2.5", "src", DEFAULT_PYFLOW_EXECUTABLE + ".exe"),
        ]
        for candidate in direct_candidates:
            if os.path.isfile(candidate) and detect_binary_type(candidate) == "linux_elf":
                return self._set_pyflow_executable(candidate)

        # 2) La ruta ya mostrada/seleccionada, pero solo si es el mismo tipo ELF.
        current = str(self.pyflow_exe_path or "").strip()
        if current and os.path.isfile(current) and detect_binary_type(current) == "linux_elf":
            return self._set_pyflow_executable(current)

        # 3) Última ruta guardada.
        saved = self._restore_pyflow_executable_preference()
        if saved and os.path.isfile(saved) and detect_binary_type(saved) == "linux_elf":
            return self._set_pyflow_executable(saved)

        # 4) Buscar copias ELF cercanas. Nunca elegir automáticamente un PE de Windows.
        for candidate in self._automatic_pyflow_candidates():
            if os.path.isfile(candidate) and detect_binary_type(candidate) == "linux_elf":
                return self._set_pyflow_executable(candidate)

        if ask_user:
            return self.choose_pyflow_executable()
        return ""

    def choose_pyflow_executable(self):
        start_dir = ""
        preferred = self._preferred_pyflow_executable_path()
        current = str(self.pyflow_exe_path or "").strip()
        if current and os.path.isfile(current):
            start_dir = os.path.dirname(current)
        elif os.path.isfile(preferred):
            start_dir = os.path.dirname(preferred)
        elif os.path.isdir(DEFAULT_PYFLOW_FOLDER):
            start_dir = DEFAULT_PYFLOW_FOLDER
        else:
            output_path = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
            if output_path:
                start_dir = os.path.dirname(os.path.abspath(output_path))

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Buscar binario PYFLOW Linux/ELF",
            start_dir,
            "All files (*.*)"
        )
        if not path:
            return ""
        path = os.path.abspath(path)
        if detect_binary_type(path) != "linux_elf":
            QMessageBox.critical(
                self,
                "PYFLOW",
                "El launcher que funciona con tu PYFLOW utiliza el binario Linux ELF mediante WSL.\n\n"
                f"El archivo seleccionado es: {detect_binary_type(path)}\n\n"
                "Elegí el mismo binario que usás en corre-varios-input-VARIOS-NUCLEOS."
            )
            return ""
        self._set_pyflow_executable(path)
        return self.pyflow_exe_path

    def _pyflow_base_output_folder(self):
        output_path = self.output_edit.text().strip() if hasattr(self, "output_edit") else ""
        if output_path:
            folder = os.path.dirname(os.path.abspath(output_path))
            if folder:
                return folder
        if self.project_state_path:
            folder = os.path.dirname(os.path.abspath(self.project_state_path))
            if folder:
                return folder
        try:
            host_project = getattr(self.app, "project_path", None)
            if host_project:
                folder = os.path.dirname(os.path.abspath(str(host_project)))
                if folder:
                    return folder
        except Exception:
            pass
        return os.getcwd()

    def _pyflow_safe_sample_token(self, sample_name):
        """Devuelve un identificador ASCII sin espacios para usar como carpeta de trabajo.

        PYFLOW 2.5 construye comandos ``mkdir`` concatenando el directorio de
        trabajo sin comillas. Por eso una carpeta que contenga espacios, guiones
        Unicode u otros caracteres especiales puede romper la creación de sus
        subdirectorios. Para ``Sieve — R12`` devolvemos, por ejemplo, ``R12``.
        """
        raw = str(sample_name or "").strip()

        # Quitar el prefijo de procedencia que agrega Componentry/FW.
        known_prefixes = {
            "sieve", "tamiz", "laser", "láser", "hybrid", "hibrido", "híbrido"
        }
        for sep in (" — ", " – ", " - ", "—", "–"):
            if sep in raw:
                left, right = raw.split(sep, 1)
                if left.strip().casefold() in known_prefixes and right.strip():
                    raw = right.strip()
                    break

        # Transliteration y limpieza estricta: únicamente ASCII alfanumérico,
        # punto, guion y guion bajo. Nunca espacios.
        raw = unicodedata.normalize("NFKD", raw)
        raw = raw.encode("ascii", "ignore").decode("ascii")
        raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
        return raw[:80] or "sample"

    def _pyflow_path_has_unsafe_chars(self, path):
        """Indica si la ruta puede romper los comandos SYSTEM de PYFLOW 2.5."""
        path = os.path.abspath(str(path))
        try:
            path.encode("ascii")
        except UnicodeEncodeError:
            return True
        # El código Fortran de PYFLOW concatena ``mkdir `` + ruta sin comillas.
        # Espacios y metacaracteres del shell deben evitarse en el CWD.
        return bool(re.search(r"[\s&()\[\]{}^=;!`,+~]", path))

    def _wsl_directory_visible(self, folder):
        """Comprueba que una carpeta Windows exista también desde la distro WSL usada por PYFLOW."""
        if os.name != "nt":
            return True

        folder = os.path.abspath(str(folder))
        if not os.path.isdir(folder):
            return False

        distros = list_wsl_distros()
        if not distros:
            return False
        distro = DEFAULT_WSL_DISTRO if DEFAULT_WSL_DISTRO in distros else distros[0]
        wsl_folder = win_to_wsl_path(folder)

        args = ["wsl.exe", "--distribution", distro, "--", "bash", "-lc",
                f"test -d {bash_quote(wsl_folder)}"]
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": 8,
            "check": False,
        }
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            completed = subprocess.run(args, **kwargs)
            return completed.returncode == 0
        except Exception:
            return False

    def _pyflow_run_root_candidates(self):
        """Raíces candidatas, en orden de preferencia, para una corrida visible desde WSL."""
        roots = []

        def add(path):
            if not path:
                return
            path = os.path.abspath(str(path))
            norm = os.path.normcase(path)
            if all(os.path.normcase(existing) != norm for existing in roots):
                roots.append(path)

        # 1) Carpeta del .dat elegido por el usuario.
        add(self._pyflow_base_output_folder())

        # 2) Carpeta del TXT de datos por muestra. En el flujo habitual suele ser
        #    la carpeta de trabajo del proyecto (por ejemplo BLAST).
        if getattr(self, "sample_metadata_path", ""):
            add(os.path.dirname(os.path.abspath(self.sample_metadata_path)))

        # 3) Carpeta del proyecto GDEINPUT.
        if getattr(self, "project_state_path", ""):
            add(os.path.dirname(os.path.abspath(self.project_state_path)))

        # 4) Proyecto del host Componentry, si existe.
        try:
            host_project = getattr(self.app, "project_path", None)
            if host_project:
                add(os.path.dirname(os.path.abspath(str(host_project))))
        except Exception:
            pass

        # 5) Fallback estable dentro del perfil de Windows. Evita AppData/Temp,
        #    que en algunas instalaciones de WSL no son utilizables como CWD.
        if os.name == "nt":
            user_profile = os.environ.get("USERPROFILE", "").strip()
            if user_profile:
                add(os.path.join(user_profile, "GDEINPUT_PYFLOW"))

        return roots

    def _prepare_pyflow_run_directory(self):
        """Crea un CWD seguro, ASCII y comprobado explícitamente desde WSL.

        PYFLOW necesita ``input.dat`` en su directorio de trabajo y además crea
        subdirectorios mediante comandos SYSTEM. GDEINPUT solo acepta una raíz
        después de verificar que la misma distro WSL puede verla. Así se evita
        lanzar ``wsl.exe`` apuntando a una carpeta Windows inaccesible.
        """
        sample_name = ""
        try:
            sample_name = str(self.app.current_sample_name()).strip()
        except Exception:
            sample_name = str(self._active_sample_name).strip()

        sample_safe = self._pyflow_safe_sample_token(sample_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rejected = []

        for root in self._pyflow_run_root_candidates():
            if not root:
                continue
            root = os.path.abspath(root)

            if self._pyflow_path_has_unsafe_chars(root):
                rejected.append(f"{root} (ruta no segura para PYFLOW)")
                continue

            try:
                os.makedirs(root, exist_ok=True)
            except Exception as exc:
                rejected.append(f"{root} (Windows: {exc})")
                continue

            if os.name == "nt" and not self._wsl_directory_visible(root):
                rejected.append(f"{root} (WSL no puede acceder)")
                continue

            run_root = os.path.join(root, "PYFLOW_results", sample_safe)
            try:
                os.makedirs(run_root, exist_ok=True)
            except Exception as exc:
                rejected.append(f"{run_root} (Windows: {exc})")
                continue

            candidate = os.path.join(run_root, stamp)
            suffix = 1
            while os.path.exists(candidate):
                suffix += 1
                candidate = os.path.join(run_root, f"{stamp}_{suffix}")

            try:
                os.makedirs(candidate, exist_ok=False)
            except Exception as exc:
                rejected.append(f"{candidate} (Windows: {exc})")
                continue

            if self._pyflow_path_has_unsafe_chars(candidate):
                shutil.rmtree(candidate, ignore_errors=True)
                rejected.append(f"{candidate} (ruta final no segura)")
                continue

            # Comprobación final: no alcanza con que la raíz sea visible; la
            # carpeta exacta que QProcess usará como CWD también debe existir en WSL.
            if os.name == "nt" and not self._wsl_directory_visible(candidate):
                shutil.rmtree(candidate, ignore_errors=True)
                rejected.append(f"{candidate} (carpeta final invisible para WSL)")
                continue

            return candidate

        detail = "\n".join(f"- {item}" for item in rejected[-6:])
        raise RuntimeError(
            "No encontré una carpeta de corrida que Windows y WSL puedan usar a la vez.\n\n"
            "GDEINPUT comprobó las rutas antes de iniciar PYFLOW, por lo que la corrida no fue lanzada. "
            "Elegí el .dat o guardá el proyecto en una carpeta simple accesible desde WSL.\n"
            + ("\nRutas descartadas:\n" + detail if detail else "")
        )

    def _open_folder_windows_reliable(self, folder):
        """Abre una carpeta con el explorador; usa API nativa en Windows."""
        folder = os.path.abspath(str(folder))
        if not os.path.isdir(folder):
            return False, "La carpeta no existe."
        errors = []
        if os.name == "nt":
            try:
                os.startfile(folder)  # type: ignore[attr-defined]
                return True, ""
            except Exception as exc:
                errors.append(str(exc))
            try:
                subprocess.Popen(["explorer.exe", folder], close_fds=True)
                return True, ""
            except Exception as exc:
                errors.append(str(exc))
        try:
            ok = QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
            if ok:
                return True, ""
            errors.append("QDesktopServices no pudo abrir la ruta.")
        except Exception as exc:
            errors.append(str(exc))
        return False, "; ".join(errors)

    def _confirm_sanity_warnings_for_run(self):
        warnings = self._sanity_warnings()
        if not warnings:
            return True
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("GDEINPUT — revisar antes de correr PYFLOW")
        msg.setText("El input es sintácticamente válido, pero hay valores que conviene revisar:")
        msg.setDetailedText("\n".join(warnings))
        msg.setInformativeText("¿Correr PYFLOW de todos modos?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        return msg.exec_() == QMessageBox.Yes

    def _append_pyflow_log(self, text):
        if not text or not hasattr(self, "pyflow_log"):
            return
        cursor = self.pyflow_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(str(text))
        self.pyflow_log.setTextCursor(cursor)
        self.pyflow_log.ensureCursorVisible()

    def _clear_pyflow_graphs(self):
        if not hasattr(self, "pyflow_graph_layout"):
            return
        while self.pyflow_graph_layout.count():
            item = self.pyflow_graph_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _read_pyflow_stdout(self):
        proc = self.pyflow_process
        if proc is None:
            return
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._append_pyflow_log(data)

    def _read_pyflow_stderr(self):
        proc = self.pyflow_process
        if proc is None:
            return
        data = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self._append_pyflow_log(data)

    def _read_pyflow_log_file(self, append=True):
        """Lee el ``log.txt`` que PYFLOW escribe directamente a disco.

        PYFLOW no envía toda la información diagnóstica a stdout/stderr. Su
        fuente abre ``log.txt`` antes de leer ``input.dat``; por eso este archivo
        es la fuente principal para explicar una corrida fallida.
        """
        run_dir = str(self.pyflow_run_dir or "").strip()
        if not run_dir:
            return ""
        path = os.path.join(run_dir, "log.txt")
        if not os.path.isfile(path):
            return ""

        data = None
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                with open(path, "r", encoding=encoding, errors="strict") as f:
                    data = f.read()
                break
            except Exception:
                data = None
        if data is None:
            try:
                with open(path, "rb") as f:
                    data = f.read().decode("utf-8", errors="replace")
            except Exception:
                return ""

        data = str(data).replace("\r\n", "\n").replace("\r", "\n")
        if append and data.strip():
            self._append_pyflow_log(
                "\n==================== log.txt de PYFLOW ====================\n"
                + data.rstrip()
                + "\n===========================================================\n"
            )
        return data

    def _pyflow_log_has_failure(self, log_text):
        """Detecta errores que PYFLOW puede reportar aun devolviendo exit code 0."""
        low = str(log_text or "").casefold()
        fatal_tokens = (
            "fatal error",
            "****input error****",
            "pyflow_2.5 is going to be stopped",
            "pyflow 2.5 is going to be stopped",
        )
        return any(token in low for token in fatal_tokens)

    def _pyflow_log_diagnosis(self, log_text, exit_code=None):
        """Extrae un diagnóstico corto y útil del log producido por PYFLOW."""
        text = str(log_text or "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        interesting = []
        triggers = (
            "fatal error", "input error", "error!", "error ", "wrong ",
            "missing", "cannot", "stopped", "warning"
        )
        for line in lines:
            low = line.casefold()
            if any(token in low for token in triggers):
                if line not in interesting:
                    interesting.append(line)

        if interesting:
            return "\n".join(interesting[-8:])

        code = int(exit_code) if exit_code is not None else None
        if code == -1073740940:
            return (
                "Windows informó 0xC0000374 (STATUS_HEAP_CORRUPTION): "
                "PYFLOW se cayó dentro del proceso. GDEINPUT ejecutó la corrida "
                "en una ruta ASCII sin espacios y validó el máximo de clases; "
                "si vuelve a ocurrir, el log.txt y el input.dat de esta corrida "
                "permiten localizar el punto exacto."
            )
        return ""

    def _pyflow_user_failure_summary(self, log_text, runner=None, exit_code=None, exit_status=None):
        """Mensaje breve para el usuario; el detalle numérico queda solo en el log."""
        combined = str(log_text or "")
        if runner is not None:
            try:
                combined += "\n" + str(runner.captured_output or "")
            except Exception:
                pass
            try:
                combined += "\n" + "\n".join(str(x) for x in runner.fatal_messages)
            except Exception:
                pass
        low = combined.casefold()

        if runner is not None and getattr(runner, "timed_out", False):
            return "PYFLOW dejó de responder y la corrida fue detenida por timeout."

        if "not able to find a solution for the reference level" in low:
            return (
                "PYFLOW no encontró una solución numérica para el nivel de referencia z₀ "
                "con esta combinación de parámetros."
            )

        if "****input error****" in low or "unknown command" in low:
            return "PYFLOW encontró un error en el input.dat generado."

        missing = []
        for line in combined.splitlines():
            clean = line.strip()
            if "fatal error" in clean.casefold() and "missing" in clean.casefold():
                if clean not in missing:
                    missing.append(clean)
        if missing:
            # Como máximo una línea concreta; nunca mostrar el volcado del solver.
            return missing[0]

        if "fatal error" in low or "pyflow is going to be stopped" in low:
            return "PYFLOW detuvo la corrida porque las condiciones ingresadas no produjeron una solución válida."

        if exit_status and str(exit_status) != "NormalExit":
            return "PYFLOW se cerró inesperadamente durante la corrida."

        try:
            code = int(exit_code)
        except Exception:
            code = 0
        if code != 0:
            return f"PYFLOW terminó con un error interno (código {code})."

        return "PYFLOW terminó sin generar resultados válidos para esta combinación de parámetros."

    def _generate_pyflow_profile_plots(self):
        """Genera PNG con la misma convención gráfica del launcher batch probado.

        Convención del script de varios inputs:
        - Height / Current height en X.
        - Variable del perfil en Y.
        - Todas las curvas en negro; se distinguen por estilo de línea.
        """
        run_dir = str(self.pyflow_run_dir or "").strip()
        if not run_dir or not os.path.isdir(run_dir):
            return []
        results_dir = os.path.join(run_dir, "results")
        if not os.path.isdir(results_dir):
            return []

        specs = {
            "concentration_profile.dat": ("Particle volumetric concentration", "Particle volumetric concentration (-)"),
            "density_profile.dat": ("Density", "Density (kg/m³)"),
            "dynamic_pressure_profile.dat": ("Dynamic Pressure", "Dynamic Pressure (Pa)"),
            "velocity_profile.dat": ("Velocity", "Velocity (m/s)"),
            "temperature_profile.dat": ("Temperature", "Temperature (K)"),
        }
        plot_dir = os.path.join(run_dir, "GDEINPUT_plots")
        made = []
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except Exception as exc:
            self._append_pyflow_log(f"\n[GDEINPUT] No se pudieron cargar las rutinas de gráficos: {exc}\n")
            return []

        for filename, (title, ylabel) in specs.items():
            src = os.path.join(results_dir, filename)
            if not os.path.isfile(src):
                continue
            try:
                with open(src, "rb") as fh:
                    if sum(1 for _ in fh) <= 1:
                        continue
            except Exception:
                continue
            try:
                data = np.loadtxt(src, skiprows=1)
            except Exception:
                continue
            data = np.asarray(data, dtype=float)
            if data.size == 0:
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 2:
                continue

            # Columna 0 = height. Las siguientes columnas son 50th, 84th y 16th.
            x = data[:, 0]
            finite_x = np.isfinite(x)
            if not np.any(finite_x):
                continue
            data = data[finite_x]
            x = data[:, 0]

            os.makedirs(plot_dir, exist_ok=True)
            fig = Figure(figsize=(9.2, 6.2), dpi=120, facecolor="white")
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            # Igual que el launcher batch: negro + estilos de línea.
            # Orden visual: 16th, 50th, 84th.
            series = []
            if data.shape[1] > 3:
                series.append((data[:, 3], "16th percentile", 1.2, (0, (2, 2))))
            if data.shape[1] > 1:
                series.append((data[:, 1], "50th percentile", 1.5, "-"))
            if data.shape[1] > 2:
                series.append((data[:, 2], "84th percentile", 1.5, (0, (9, 7))))

            for y, label, linewidth, linestyle in series:
                mask = np.isfinite(x) & np.isfinite(y)
                if np.any(mask):
                    ax.plot(
                        x[mask], y[mask],
                        color="black",
                        linewidth=linewidth,
                        linestyle=linestyle,
                        label=label,
                    )

            if not ax.lines:
                continue

            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Current height (m)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(False)
            ax.set_facecolor("white")
            ax.tick_params(axis="both", labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if len(ax.lines) > 1:
                ax.legend(loc="best", frameon=False, fontsize=10)

            finite_height = x[np.isfinite(x)]
            if finite_height.size >= 2:
                xmin = float(np.min(finite_height))
                xmax = float(np.max(finite_height))
                if xmin != xmax:
                    ax.set_xlim(xmin, xmax)
            ax.margins(x=0.02, y=0.05)
            fig.tight_layout()

            out = os.path.join(plot_dir, os.path.splitext(filename)[0] + ".png")
            fig.savefig(out)
            made.append(out)

        if made:
            self._append_pyflow_log(
                f"\n[GDEINPUT] Se generaron {len(made)} gráfico(s) con la convención del launcher batch.\n"
            )
        return made

    def _collect_pyflow_outputs(self):
        run_dir = self.pyflow_run_dir
        out = []
        if run_dir and os.path.isdir(run_dir):
            for root, _dirs, files in os.walk(run_dir):
                for name in files:
                    full = os.path.join(root, name)
                    if os.path.abspath(full) == os.path.abspath(os.path.join(run_dir, "input.dat")):
                        continue
                    if os.path.basename(full).startswith("__gdeinput_pyflow_wsl__"):
                        continue
                    out.append(full)
        out.sort(key=lambda p: os.path.relpath(p, run_dir).casefold() if run_dir else p.casefold())
        self.pyflow_last_outputs = out
        return out

    def _refresh_pyflow_results_ui(self):
        outputs = self._collect_pyflow_outputs()
        self.lst_pyflow_outputs.clear()
        self._clear_pyflow_graphs()

        run_dir = self.pyflow_run_dir
        raster_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
        external_graph_exts = {".pdf", ".svg", ".eps", ".ps"}
        raster_files = []
        external_graphs = []

        for full in outputs:
            rel = os.path.relpath(full, run_dir) if run_dir else full
            self.lst_pyflow_outputs.addItem(rel)
            ext = os.path.splitext(full)[1].lower()
            if ext in raster_exts:
                raster_files.append(full)
            elif ext in external_graph_exts:
                external_graphs.append(full)

        for full in raster_files:
            title = QLabel(os.path.basename(full))
            title.setStyleSheet("font-weight:bold;")
            title.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.pyflow_graph_layout.addWidget(title)

            pix = QPixmap(full)
            if pix.isNull():
                err = QLabel(f"No se pudo previsualizar: {full}")
                err.setTextInteractionFlags(Qt.TextSelectableByMouse)
                self.pyflow_graph_layout.addWidget(err)
                continue
            max_width = 1100
            if pix.width() > max_width:
                pix = pix.scaledToWidth(max_width, Qt.SmoothTransformation)
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            lbl.setToolTip(full)
            self.pyflow_graph_layout.addWidget(lbl)

        if not raster_files:
            if external_graphs:
                info = QLabel(
                    "PYFLOW generó gráficos vectoriales/PDF. Aparecen en la lista de archivos; "
                    "hacé doble clic para abrirlos con el visor del sistema."
                )
            else:
                info = QLabel("No se detectaron gráficos rasterizados en esta corrida.")
            info.setWordWrap(True)
            self.pyflow_graph_layout.addWidget(info)

        self.pyflow_graph_layout.addStretch(1)
        return outputs, raster_files, external_graphs

    def _open_pyflow_output_item(self, item):
        if item is None or not self.pyflow_run_dir:
            return
        full = os.path.abspath(os.path.join(self.pyflow_run_dir, item.text()))
        if os.path.isfile(full):
            QDesktopServices.openUrl(QUrl.fromLocalFile(full))

    def open_pyflow_run_folder(self):
        folder = str(self.pyflow_run_dir).strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "PYFLOW", "Todavía no hay resultados de PYFLOW para abrir.")
            return
        ok, error = self._open_folder_windows_reliable(folder)
        if not ok:
            QMessageBox.critical(
                self,
                "PYFLOW",
                "No se pudo abrir la carpeta de resultados.\n\n" + (error or folder)
            )

    def run_pyflow(self):
        """Ejecuta PYFLOW con el MISMO PyflowRunner del launcher batch probado."""
        if self.pyflow_runner is not None and self.pyflow_runner.is_running():
            QMessageBox.information(self, "PYFLOW", "PYFLOW ya está ejecutándose.")
            return

        try:
            self.validate_all()
            if not self._confirm_sanity_warnings_for_run():
                return
            dat_text = self.build_dat_text()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Correr PYFLOW",
                "No se puede ejecutar PYFLOW porque el input actual no es válido:\n\n" + str(e)
            )
            return

        exe = self._find_pyflow_executable(ask_user=False)
        if not exe:
            QMessageBox.warning(
                self,
                "Correr PYFLOW",
                "No encontré automáticamente el binario Linux ELF de PYFLOW.\n\n"
                "Usá el botón 'Buscar...' de la pestaña PYFLOW una sola vez para seleccionarlo. "
                "Después GDEINPUT recordará esa ruta y 'Correr PYFLOW' no volverá a abrir un selector."
            )
            return

        if detect_binary_type(exe) != "linux_elf":
            QMessageBox.critical(
                self,
                "PYFLOW",
                "GDEINPUT V8.19 usa el runner WSL del launcher batch. "
                "Seleccioná el mismo binario Linux ELF de PYFLOW que funciona allí."
            )
            return

        try:
            run_dir = self._prepare_pyflow_run_directory()
            # El runner probado copia el input recibido a work_dir/input.dat. Por eso
            # la fuente se guarda con otro nombre para no provocar SameFileError.
            input_source = os.path.join(run_dir, "__gdeinput_source.dat")
            with open(input_source, "w", encoding="utf-8", newline="\n") as f:
                f.write(dat_text)
        except Exception as e:
            QMessageBox.critical(self, "Correr PYFLOW", f"No pude preparar la corrida:\n{e}")
            return

        self.pyflow_run_dir = run_dir
        self.pyflow_run_dir_edit.setText(run_dir)
        self.pyflow_log.clear()
        self.lst_pyflow_outputs.clear()
        self._clear_pyflow_graphs()
        self.tabs.setCurrentWidget(self.pyflow_tab)
        self.lbl_pyflow_status.setText("PYFLOW se está ejecutando mediante WSL...")
        self.btn_run_pyflow.setEnabled(False)

        self._append_pyflow_log(
            "GDEINPUT V8.19 — runner del launcher batch probado\n"
            f"PYFLOW ELF: {exe}\n"
            f"Resultados: {run_dir}\n"
            "------------------------------------------------------------\n"
        )

        runner = PyflowRunner()
        self.pyflow_runner = runner
        self.pyflow_process = runner.proc  # compatibilidad con código viejo de la UI
        self.pyflow_execution_mode = "wsl"
        if hasattr(self, "lbl_pyflow_runner"):
            self.lbl_pyflow_runner.setText("Modo de ejecución: WSL — runner batch probado")

        runner.output_text.connect(self._append_pyflow_log)
        runner.prompt_needed.connect(
            lambda prompt_kind, trigger_text, r=runner: self._on_gde_pyflow_prompt(r, prompt_kind, trigger_text)
        )
        runner.finished.connect(
            lambda exit_code, exit_status, r=runner: self._on_gde_pyflow_finished(r, exit_code, exit_status)
        )

        ok, err = runner.start(
            exe_path=exe,
            input_file=input_source,
            work_dir=run_dir,
        )
        if not ok:
            self.btn_run_pyflow.setEnabled(True)
            self.pyflow_runner = None
            self.pyflow_process = None
            self.lbl_pyflow_status.setText("No se pudo iniciar PYFLOW.")
            QMessageBox.critical(self, "PYFLOW", err)
            return

    def _on_gde_pyflow_prompt(self, runner, prompt_kind, trigger_text):
        """Replica las opciones automáticas por defecto del launcher batch."""
        if runner is not self.pyflow_runner:
            return
        if prompt_kind == "low_accuracy_stop":
            self._append_pyflow_log(
                "\n[INFO] Prompt de baja precisión detectado. Se envía STOP automáticamente, "
                "igual que en el launcher batch.\n"
            )
            runner.send_stop()
            return

        self._append_pyflow_log(
            "\n[INFO] Prompt ENTER detectado. Se envía ENTER automáticamente, "
            "igual que en el launcher batch.\n"
        )
        runner.send_enter()

    def _on_gde_pyflow_finished(self, runner, exit_code, exit_status):
        if runner is not self.pyflow_runner:
            return

        self.btn_run_pyflow.setEnabled(True)
        self.pyflow_process = None

        self._append_pyflow_log(
            "\n------------------------------------------------------------\n"
            f"PYFLOW finalizó. exit_code={int(exit_code)}; estado={exit_status}\n"
        )

        # El runner ya devolvió log.txt/results/grainsize_analysis desde /tmp a Windows.
        file_log = self._read_pyflow_log_file(append=True)
        success = (
            exit_status == "NormalExit"
            and int(exit_code) == 0
            and not runner.had_fatal_error
            and not runner.timed_out
            and not self._pyflow_log_has_failure(file_log)
            and bool(runner.last_results_dir)
        )

        self._generate_pyflow_profile_plots()
        outputs, raster_files, external_graphs = self._refresh_pyflow_results_ui()
        graph_count = len(raster_files) + len(external_graphs)

        if success:
            status = (
                f"PYFLOW terminó correctamente. {len(outputs)} archivo(s) de salida"
                + (f" y {graph_count} gráfico(s)." if graph_count else ".")
            )
            self.lbl_pyflow_status.setText(status)
            QMessageBox.information(self, "PYFLOW", status)
        else:
            summary = self._pyflow_user_failure_summary(
                file_log, runner=runner, exit_code=exit_code, exit_status=exit_status
            )
            status = summary + " Revisá el log de la pestaña PYFLOW si necesitás el detalle."
            self.lbl_pyflow_status.setText(status)
            QMessageBox.critical(self, "PYFLOW", status)

        try:
            runner.deleteLater()
        except Exception:
            pass
        self.pyflow_runner = None

    def closeEvent(self, event):
        runner = getattr(self, "pyflow_runner", None)
        if runner is not None and runner.is_running():
            try:
                runner.shutdown_for_app_close()
            except Exception:
                pass
        super().closeEvent(event)

    # =========================================================
    # Load / save .dat
    # =========================================================
    def open_dat_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir .dat",
            "",
            "DAT files (*.dat);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()
        except Exception as e:
            QMessageBox.warning(self, "Abrir .dat", str(e))
            return

        try:
            assignments = parse_assignments(text)
            self._load_assignments(assignments)
            self.output_edit.setText(path)
            self.update_preview()
        except Exception as e:
            QMessageBox.warning(self, "Abrir .dat", str(e))

    def _load_assignments(self, assignments):
        for key, value in assignments.items():
            key_up = key.upper()
            if key_up in self.param_widgets:
                self._set_widget_value(key_up, value)

        self.update_classes()

        for key, value in assignments.items():
            key_up = key.upper()
            m = re.match(r"^WEIGHT\((\d+)\s*,\s*(\d+)\)$", key_up, re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1))
            j = int(m.group(2))
            if idx in (1, 2):
                self._set_weight_value(idx, j, value)

    def choose_output_file(self):
        current = self.output_edit.text().strip()
        start_dir = os.path.dirname(current) if current else ""
        suggested = current if current else os.path.join(start_dir, "input.dat")

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar .dat",
            suggested,
            "DAT files (*.dat);;All files (*)"
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".dat"
        self.output_edit.setText(path)

    def open_output_folder(self):
        path = self.output_edit.text().strip()
        folder = os.path.dirname(path) if path else ""
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Carpeta", "La carpeta de salida no existe.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    def _sanity_warnings(self):
        warnings = []
        for idx in (1, 2):
            cdlaw = strip_quotes(self._widget_value(f"CDLAW({idx})")).upper()
            if cdlaw in SHAPEFACTOR_DRAG_LAWS:
                txt = self._widget_value(f"SHAPEFACT({idx},0)").strip()
                if txt:
                    try:
                        psi = parse_fortran_float(txt)
                        if 0.0 < psi < 0.15:
                            warnings.append(
                                f"SHAPEFACT({idx},0)={psi:g} es extremadamente bajo; revisá el decimal."
                            )
                    except Exception:
                        pass
        return warnings

    def save_dat_file(self):
        try:
            self.validate_all()
            warnings = self._sanity_warnings()
            if warnings:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("GDEINPUT — revisar antes de guardar")
                msg.setText("El input es sintácticamente válido, pero hay valores que conviene revisar:")
                msg.setDetailedText("\n".join(warnings))
                msg.setInformativeText("¿Guardar de todos modos?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.No)
                if msg.exec_() != QMessageBox.Yes:
                    return
            text = self.build_dat_text()
        except Exception as e:
            QMessageBox.warning(self, "GDEINPUT", f"No se puede guardar:\n{e}")
            return

        path = self.output_edit.text().strip()
        if not path:
            self.choose_output_file()
            path = self.output_edit.text().strip()
            if not path:
                return

        if not os.path.splitext(path)[1]:
            path += ".dat"
            self.output_edit.setText(path)

        folder = os.path.dirname(path)
        if folder:
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Salida", f"No pude crear la carpeta:\n{e}")
                return

        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        except Exception as e:
            QMessageBox.critical(self, "Salida", f"No pude guardar el archivo:\n{e}")
            return

        QMessageBox.information(self, "GDEINPUT", f"Archivo guardado:\n{path}")

    def _set_default_output_path(self):
        sname = ""
        try:
            sname = str(self.app.current_sample_name()).strip()
        except Exception:
            sname = ""

        base = sanitize_filename(sname) if sname else "input"
        folder = ""
        try:
            project_path = getattr(self.app, "project_path", None)
            if project_path:
                folder = os.path.dirname(project_path)
        except Exception:
            folder = ""

        if not folder:
            folder = os.getcwd()

        self.output_edit.setText(os.path.join(folder, f"{base}.dat"))

    # =========================================================
    # Validation and output
    # =========================================================
    def validate_all(self):
        errors = []
        self._clear_param_styles()
        self._clear_weight_styles()

        group1 = self._group_components(1)
        group2 = self._group_components(2)
        if len(group1) != 1:
            errors.append("Seleccioná exactamente un componente para Component 1.")
        if len(group2) != 1:
            errors.append("Seleccioná exactamente un componente para Component 2.")
        if group1 and group2 and group1[0] == group2[0]:
            errors.append("Component 1 y Component 2 deben ser diferentes.")

        for key in ("ZLAM", "KS"):
            txt = self._widget_value(key).strip()
            if not txt:
                errors.append(f"Falta {key}.")
                self._mark_param_error(key)
                continue
            try:
                value = parse_fortran_float(txt)
                if value <= 0:
                    raise ValueError
            except Exception:
                errors.append(f"{key} debe ser un número > 0 en metros.")
                self._mark_param_error(key)

        for idx in (1, 2):
            rholaw_key = f"RHOLAW({idx})"
            rholaw = strip_quotes(self._widget_value(rholaw_key)).upper()
            if not rholaw:
                errors.append(f"Falta {rholaw_key}.")
                self._mark_param_error(rholaw_key)

            if rholaw == "CUSTOM":
                rho_key = f"RHOS({idx},0)"
                try:
                    rho = parse_fortran_float(self._widget_value(rho_key))
                    if rho <= 100:
                        raise ValueError
                except Exception:
                    errors.append(f"{rho_key} debe contener la densidad en kg/m³ (valor > 100).")
                    self._mark_param_error(rho_key)

            cdlaw_key = f"CDLAW({idx})"
            cdlaw = strip_quotes(self._widget_value(cdlaw_key)).upper()
            if cdlaw not in DRAG_LAWS:
                errors.append(f"{cdlaw_key}='{cdlaw}' no es una ley reconocida.")
                self._mark_param_error(cdlaw_key)
            else:
                for key in self._drag_fields_for_component(idx):
                    txt = self._widget_value(key).strip()
                    if key in LOGICAL_KEYWORDS:
                        if normalize_bool(txt) not in LOGICAL_OPTIONS:
                            errors.append(f"{key} debe ser .T. o .F.")
                            self._mark_param_error(key)
                        continue
                    if not txt:
                        errors.append(f"Falta {key} para {cdlaw_key}='{cdlaw}'.")
                        self._mark_param_error(key)
                        continue
                    try:
                        v = parse_fortran_float(txt)
                        base = key.split("(", 1)[0]
                        if base in {"SPHERICITY", "COREY", "CIRCULARITY", "SHAPEFACT", "LONGSPHER", "CROSSSPHER"} and not (0 < v <= 1):
                            raise ValueError
                        if base in {"VOLEQSPHD", "CIRCEQARD", "FLATRATIO"} and v <= 0:
                            raise ValueError
                        if base == "FRACTDIM" and not (2 <= v <= 3):
                            raise ValueError
                    except Exception:
                        errors.append(f"Valor inválido en {key}.")
                        self._mark_param_error(key)

            # DAVGEQSPH es opcional para ambos componentes; si se ingresa, debe estar en φ.
            davg_key = f"DAVGEQSPH({idx})"
            davg_txt = self._widget_value(davg_key).strip()
            if davg_txt:
                try:
                    parse_fortran_float(davg_txt)
                except Exception:
                    errors.append(f"{davg_key} debe ser numérico en unidades φ.")
                    self._mark_param_error(davg_key)

            try:
                phimin = parse_fortran_float(self._widget_value(f"PHIMIN({idx})"))
                phimax = parse_fortran_float(self._widget_value(f"PHIMAX({idx})"))
                dphi = parse_fortran_float(self._widget_value(f"DPHI({idx})"))
                n_calc, _ = calculate_nclass_and_phi(phimin, phimax, dphi)
                # PYFLOW v2.5 declara WEIGHT(6,30) y varios arreglos equivalentes
                # con 30 clases máximas. Escribir WEIGHT(i,31) o superior puede
                # corromper memoria en compilaciones sin bounds checking.
                if n_calc > 30:
                    raise ValueError(
                        f"NCLASS={n_calc}, pero PYFLOW 2.5 admite como máximo 30 clases por componente"
                    )
                n_gui = int(round(parse_fortran_float(self._widget_value(f"NCLASS({idx})"))))
                if n_calc != n_gui:
                    raise ValueError(f"NCLASS debería ser {n_calc}")
            except Exception as e:
                errors.append(f"La grilla φ automática del componente {idx} no es válida: {e}. Volvé a autocargar desde el histograma.")

            table = self.weight_tables.get(idx)
            if table is None or table.rowCount() == 0:
                errors.append(f"No hay WEIGHT automáticos para el componente {idx}. Volvé a autocargar desde el histograma.")
                continue
            if table.rowCount() > 30:
                errors.append(
                    f"El componente {idx} tiene {table.rowCount()} clases, pero PYFLOW 2.5 "
                    "solo admite 30 WEIGHT por componente."
                )
                continue
            for row in range(table.rowCount()):
                item = table.item(row, 3)
                txt = item.text().strip() if item is not None else ""
                try:
                    value = parse_fortran_float(txt)
                    if value < 0:
                        raise ValueError
                except Exception:
                    errors.append(f"WEIGHT({idx},{row+1}) no contiene un wt% válido.")
                    self._mark_weight_error(idx, row)

        if errors:
            raise ValueError("\n".join(errors[:60]))

    def _clear_param_styles(self):
        for widget in self.param_widgets.values():
            widget.setStyleSheet("")

    def _mark_param_error(self, key):
        widget = self.param_widgets.get(key.upper())
        if widget is not None:
            widget.setStyleSheet("background:#ffd6d6;")

    def _clear_weight_styles(self):
        for idx in (1, 2):
            table = self.weight_tables[idx]
            for row in range(table.rowCount()):
                item = table.item(row, 3)
                if item is not None:
                    item.setBackground(Qt.white)

    def _mark_weight_error(self, idx, row):
        table = self.weight_tables[idx]
        item = table.item(row, 3)
        if item is not None:
            item.setBackground(Qt.red)

    def _weight_label(self, idx, j):
        return f"Peso componente {idx}, clase {j}"

    def build_dat_text(self):
        """Genera el modo de dos componentes con el esqueleto exacto del archivo de referencia."""
        def num(key):
            raw = self._widget_value(key).strip().replace(",", ".")
            parse_fortran_float(raw)
            return raw

        def txt(key):
            return quote_fortran(self._widget_value(key))

        def padded(statement, comment, column):
            return statement + (" " * max(1, column - len(statement))) + comment

        def append_drag_descriptor(lines, idx):
            """Escribe únicamente los parámetros requeridos por CDLAW del componente."""
            cdlaw = strip_quotes(self._widget_value(f"CDLAW({idx})")).upper()
            for key in self._drag_fields_for_component(idx):
                if key in LOGICAL_KEYWORDS:
                    value = normalize_bool(self._widget_value(key))
                else:
                    value = num(key)

                # Conserva exactamente los comentarios existentes del caso DIOGMELE.
                if key == f"SHAPEFACT({idx},0)" and cdlaw == "DIOGMELE":
                    if idx == 1:
                        lines.append(padded(
                            f"SHAPEFACT(1,0) = {value}",
                            "! factor de forma de mi componente en su medianay",
                            47,
                        ))
                    else:
                        lines.append(f"SHAPEFACT(2,0) = {value}")
                else:
                    lines.append(f"{key} = {value}")

        lines = []
        lines.append("! Run control data")
        lines.append("MU=2.d-5                      ! viscosidad dinámica de la fase gaseosa, SE USA 2 *10^5")
        lines.append("DENGAS=0.38                   ! Densidad del gas, consultar si puedo modificarlo")
        lines.append("MODEL = 'TWOCOMPONENTS'")
        lines.append("PROBT = 0.05                  ! nivel de significancia del test t de Student")
        lines.append(padded(f"ZLAM = {num('ZLAM')}", "! espesor de la capa laminada (sacando capa a y c)", 30))
        lines.append("C0 = 0.75                     ! concentración volumétrica de particulas en el nivel de referencia zo, es el CO del calculo de C(z)")
        lines.append(padded(f"KS = {num('KS')}", "! coeficiente de rugosidad", 30))
        lines.append("")
        lines.append("DISTR1 = .T.                   ! en fortrand T es true (activado) y F false (desactivado). ")
        lines.append("DISTR2 = .T.                   ! y esto es para calcular la distribución granulométrica de los componentes para mediana y sorting")
        lines.append("")
        lines.append("DEPRATES = .F.")
        lines.append("")
        lines.append("INPUT_WEIGHT='WT'              !cálculo de tasas y tiempos de depositación, hay que incluir todos los componentes con sus datos")
        lines.append("")
        lines.append("")
        lines.append("! ============================================================")
        lines.append("! Component 1 section")
        lines.append("! ")
        lines.append("! ============================================================")
        lines.append("")
        lines.append(padded(f"RHOLAW(1) = {txt('RHOLAW(1)')}", "! pongo mi densidad, no elijo una ley que la calcula sola.", 47))
        if strip_quotes(self._widget_value("RHOLAW(1)")).upper() == "CUSTOM":
            lines.append(padded(f"RHOS(1,0) = {num('RHOS(1,0)')}", "! densidad de componente 1", 47))
        lines.append("")
        lines.append(padded(f"CDLAW(1) = {txt('CDLAW(1)')}", "! ley de factor de forma", 47))
        append_drag_descriptor(lines, 1)
        lines.append("")
        lines.append("DOTESTCHI(1)=.T.                               ! test de chi cuadrado")
        lines.append("SIGLEVCHI(1)=0.05")
        lines.append("")
        if self._widget_value("DAVGEQSPH(1)").strip():
            lines.append(padded(f"DAVGEQSPH(1)={num('DAVGEQSPH(1)')}", "! CONSULTAR diámetro medio de la esfera de volumen equivalente, ver Mele et al. 2015", 47))
            lines.append("")
        lines.append(f"PHIMIN(1)={num('PHIMIN(1)')}")
        lines.append(f"PHIMAX(1)={num('PHIMAX(1)')}")
        lines.append(f"DPHI(1)={num('DPHI(1)')}")
        lines.append("")

        table = self.weight_tables[1]
        for row in range(table.rowCount()):
            item = table.item(row, 3)
            value = parse_fortran_float(item.text().strip())
            lines.append(f"WEIGHT(1,{row+1})={value:.4f}")

        lines.append("")
        lines.append("")
        lines.append("! ============================================================")
        lines.append("! Component 2 section")
        lines.append("! ")
        lines.append("! ============================================================")
        lines.append("")
        lines.append(f"RHOLAW(2) = {txt('RHOLAW(2)')}")
        if strip_quotes(self._widget_value("RHOLAW(2)")).upper() == "CUSTOM":
            lines.append(f"RHOS(2,0) = {num('RHOS(2,0)')}")
        lines.append("")
        lines.append(f"CDLAW(2) = {txt('CDLAW(2)')}")
        append_drag_descriptor(lines, 2)
        lines.append("")
        lines.append("DOTESTCHI(2)=.T.")
        lines.append("SIGLEVCHI(2)=0.05")
        lines.append("")
        if self._widget_value("DAVGEQSPH(2)").strip():
            lines.append(f"DAVGEQSPH(2)={num('DAVGEQSPH(2)')}")
            lines.append("")
        lines.append(f"PHIMIN(2)={num('PHIMIN(2)')}")
        lines.append(f"PHIMAX(2)={num('PHIMAX(2)')}")
        lines.append(f"DPHI(2)={num('DPHI(2)')}")
        lines.append("")

        table = self.weight_tables[2]
        for row in range(table.rowCount()):
            item = table.item(row, 3)
            value = parse_fortran_float(item.text().strip())
            lines.append(f"WEIGHT(2,{row+1})={value:.4f}")

        return "\n".join(lines).rstrip() + "\n"

    def update_preview(self, *args):
        try:
            text = self.build_dat_text()
            self.last_preview_text = text
            if hasattr(self, "preview"):
                self.preview.setPlainText(text)
        except Exception as e:
            self.last_preview_text = ""
            if hasattr(self, "preview"):
                self.preview.setPlainText(f"ERROR: {e}")


# =========================================================
# Utility functions
# =========================================================
def remove_inline_comment(line):
    text = str(line)
    cut = len(text)
    for symbol in ("!", "#"):
        pos = text.find(symbol)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut].strip()


def parse_assignments(text):
    out = {}
    for raw in str(text).splitlines():
        line = remove_inline_comment(raw)
        if not line:
            continue
        m = ASSIGN_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip().upper()
        value = m.group(2).strip()
        out[key] = value
    return out


def strip_quotes(text):
    t = str(text).strip()
    if len(t) >= 2:
        if (t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"'):
            return t[1:-1]
    return t


def quote_fortran(text):
    t = strip_quotes(str(text).strip())
    if not t:
        return "''"
    return "'" + t.upper() + "'"


def normalize_bool(text):
    t = str(text).strip().upper()
    if t in {".T.", "T", "TRUE", ".TRUE."}:
        return ".T."
    if t in {".F.", "F", "FALSE", ".FALSE."}:
        return ".F."
    return t


def parse_fortran_float(text):
    s = str(text).strip()
    if not s:
        raise ValueError("campo vacío")
    s = s.replace(",", ".")
    s = re.sub(r"([0-9.])([dD])([+-]?\d+)", r"\1e\3", s)
    return float(s)


def format_number(value):
    try:
        v = float(value)
    except Exception:
        return str(value)
    if abs(v) < 1e-12:
        v = 0.0
    if float(v).is_integer():
        return str(int(round(v)))
    return f"{v:.10g}".rstrip("0").rstrip(".")


def calculate_nclass_and_phi(phimin, phimax, dphi):
    phimin = float(phimin)
    phimax = float(phimax)
    dphi = float(dphi)

    if phimax < phimin:
        raise ValueError("PHIMAX debe ser mayor o igual que PHIMIN")
    if dphi <= 0:
        raise ValueError("DPHI debe ser mayor que 0")

    exact = (phimax - phimin) / dphi
    rounded = round(exact)
    if abs(exact - rounded) > 1e-7:
        raise ValueError("la grilla phi no cierra con PHIMIN, PHIMAX y DPHI")

    nclass = int(rounded) + 1
    if nclass <= 0:
        raise ValueError("NCLASS inválido")

    phis = [round(phimin + i * dphi, 10) for i in range(nclass)]
    return nclass, phis


def infer_dphi(phi_values):
    vals = sorted(set(float(v) for v in phi_values if np.isfinite(float(v))))
    if len(vals) < 2:
        return 1.0

    diffs = []
    for a, b in zip(vals[:-1], vals[1:]):
        d = b - a
        if d > 1e-9:
            diffs.append(round(d, 8))

    if not diffs:
        return 1.0

    counts = {}
    for d in diffs:
        counts[d] = counts.get(d, 0) + 1

    return float(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def value_at_phi(series, phi):
    phi = float(phi)
    try:
        if phi in series.index:
            v = series.loc[phi]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            return float(v)
    except Exception:
        pass

    for idx, val in series.items():
        try:
            if abs(float(idx) - phi) < 1e-8:
                return float(val)
        except Exception:
            continue
    return 0.0


def set_combo_text(combo, text):
    text = str(text).strip()
    idx = combo.findText(text, Qt.MatchFixedString)
    if idx >= 0:
        combo.setCurrentIndex(idx)
        return
    if combo.isEditable():
        combo.setCurrentText(text)
        return
    if combo.count() > 0:
        combo.setCurrentIndex(0)


def sanitize_filename(name):
    s = str(name).strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = s.strip().strip(".")
    return s or "input"