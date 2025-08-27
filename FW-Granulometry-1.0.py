# region # === Bloque 0: Localización (i18n) ===
"""
Infraestructura de localización ES/EN.
- No toca la lógica del programa.
- Provee funciones utilitarias para traducir textos de UI y rótulos de parámetros.
- Se integra con el resto usando `tr(...)`, `param_label(...)`, etc.
Colocar este bloque arriba del Bloque 1.
"""

# Idiomas soportados
LANGS = ("es", "en")
DEFAULT_LANG = "es"
_GLOBAL_LANG = DEFAULT_LANG  # respaldo si aún no existe qApp

def _current_lang(lang=None):
    """Devuelve el idioma activo. Prioriza argumento explícito, luego qApp.lang, luego respaldo global."""
    if lang in LANGS:
        return lang
    try:
        # qApp se define en el Bloque 15 (o en main)
        from PyQt5.QtWidgets import QApplication  # noqa: F401
        return getattr(qApp, "lang", _GLOBAL_LANG)  # type: ignore[name-defined]
    except Exception:
        return _GLOBAL_LANG

def set_app_language(lang: str):
    """Setter global de idioma (usado por el menú Idioma)."""
    global _GLOBAL_LANG
    if lang not in LANGS:
        return
    try:
        qApp.lang = lang  # type: ignore[name-defined]
    except Exception:
        _GLOBAL_LANG = lang

# --- Símbolos (respaldo) ---
# NOTA: El Bloque 2 define SYMBOLS; aquí solo proveemos un fallback para evitar dependencias de orden.
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

# --- Etiquetas de parámetros (con símbolo entre paréntesis) ---
def param_label(key: str, lang: str = None) -> str:
    """
    key ∈ {"median","mean","sigma","skewness","kurtosis"}
    Devuelve el rótulo bilingüe con símbolo entre paréntesis.
    """
    L = _current_lang(lang)
    s = _sym(key)
    if L == "en":
        mapping = {
            "median":   f"Median ({s})",
            "mean":     f"Mean ({s})",
            "sigma":    f"Sorting ({s})",
            "skewness": f"Skewness ({s})",
            "kurtosis": f"Kurtosis ({s})",
        }
    else:  # "es"
        mapping = {
            "median":   f"Mediana ({s})",
            "mean":     f"Media ({s})",
            "sigma":    f"Selección ({s})",
            "skewness": f"Asimetría ({s})",
            "kurtosis": f"Kurtosis ({s})",
        }
    return mapping[key]

def param_option_labels(lang: str = None):
    """
    Devuelve lista de pares (key,label) para poblar combos X/Y respetando idioma:
      [("mean","Mean (Mφ)"), ("median","Median (Mdφ)"), ...]
    """
    L = _current_lang(lang)
    keys = ["mean", "median", "sigma", "skewness", "kurtosis"]
    return [(k, param_label(k, L)) for k in keys]

# --- Nombres de bases (solo para DISPLAY; las claves internas siguen siendo Tamiz/Laser/Híbrido) ---
BASE_DISPLAY = {
    "es": {"Tamiz": "Tamiz", "Laser": "Láser", "Híbrido": "Híbrido"},
    "en": {"Tamiz": "Sieve", "Laser": "Laser", "Híbrido": "Hybrid"},
}

def base_name_display(base_key: str, lang: str = None) -> str:
    """Mapea nombre interno de base a display bilingüe."""
    L = _current_lang(lang)
    return BASE_DISPLAY.get(L, {}).get(base_key, base_key)

def legend_group_base(lang: str = None) -> str:
    """Etiqueta 'Group (Base)' / 'Grupo (Base)'."""
    return "Group (Base)" if _current_lang(lang) == "en" else "Grupo (Base)"

def size_um_label(lang: str = None) -> str:
    """Etiqueta del eje secundario inferior en el histograma."""
    return "Size (µm)" if _current_lang(lang) == "en" else "Tamaño (µm)"

# --- Diccionario de textos generales de UI ---
# Agregados clave para:
#   - Diálogo de emparejado Tamiz↔Láser: columnas, validaciones y mensajes.
#   - Histograma híbrido: toggle de superposición/residuo y tamiz original.
#   - Selector de método (M1/M2/M3).
#   - NUEVO: Selector de método de modos (GMM/Weibull) + labels genéricos (n/k) + exportación unificada.
I18N = {
    "es": {
        # App / Tabs
        "app_title": "FW Granulometry 1.0 - LAAN",
        "tab_xy": "Gráfico XY",
        "tab_walker": "Walker (1971 y 1983)",
        "tab_gk": "Gençalioğlu-Kuşcu et al. 2007",
        "tab_hist": "Histograma",
        # Menús
        "menu_file": "&Archivo",
        "menu_config": "&Configuración",
        "menu_db": "Base de datos",
        "menu_help": "&Ayuda",
        # Archivo
        "file_load_sieve": "Cargar archivo Excel de tamizado",
        "file_load_laser": "Cargar archivo Excel de láser",
        "file_export_image": "Exportar imagen actual",
        "file_combine": "Combinar tamiz + láser",
        # Configuración
        "cfg_method": "Método de cálculo",
        "cfg_colors": "Colores de grupos",
        "cfg_groups_all": "Grupos visibles (todas las pestañas)",
        "cfg_ribbons": "Sombras (XY)",
        "cfg_theme": "Estilo de interfaz",
        "cfg_language": "Idioma / Language",
        "cfg_lang_es": "Español",
        "cfg_lang_en": "English",
        # DB
        "db_view_sieve": "Ver base de datos de tamiz",
        "db_view_laser": "Ver base de datos de láser",
        "db_view_hybrid": "Ver base de datos tamiz + láser",
        # Ayuda
        "help_about": "Acerca de...",
        "about_title": "Acerca de",
        "about_text": (
            "Software libre y gratuito de análisis granulométrico de depósitos piroclásticos.\n\n"
            "Más información: santiagoabelretamoso@gmail.com\n\n"
            "Folk and Ward Granulometry or\nFW-Granulometry - Python"
        ),
        # XY tab controls
        "btn_load_sieve": "Cargar archivo Excel de tamizado",
        "lbl_x": "Eje X:",
        "lbl_y": "Eje Y:",
        "btn_plot_xy": "Graficar XY",
        "btn_choose_db": "Elegir bases a graficar",
        "chk_show_names": "Mostrar nombres",
        "btn_filter_groups": "Filtrar grupos",
        "btn_export_image": "Exportar imagen",
        "console_header": "Cálculo de parámetros granulométricos",
        "btn_export_params": "Exportar parámetros granulométricos",
        # Walker/GK
        "btn_group_walker": "Seleccionar grupos",
        "btn_group_gk": "Seleccionar grupos",
        # Histograma
        "lbl_hist_width": "Ancho de barra (%):",
        "btn_hist_colors": "Colores de barras",
        "btn_hist_labels": "Editar histograma",
        "chk_hist_bars": "Histograma (barras)",
        "chk_hist_poly": "Polígono de frecuencia",
        "chk_hist_cum": "Curva acumulativa",
        "btn_export_hist": "Exportar gráfico",
        "chk_gmm": "Detectar modos (IA)",
        "lbl_gmm_kmax": "k máx:",
        "btn_export_gmm": "Exportar modos (CSV)",  # alias hacia exportación unificada
        "lbl_hist_base": "Base:",
        "lbl_hist_sample": "Muestra:",
        "btn_plot_hist": "Graficar histograma",
        # Valores por defecto de histograma
        "hist_title_def": "Histograma",
        "hist_xlabel_def": "φ",
        "hist_ylabel_def": "wt (%)",
        "hist_ylabel2_def": "Frecuencia Acumulativa",
        # Diálogos genéricos
        "ok": "Aceptar",
        "cancel": "Cancelar",
        "close": "Cerrar",
        "info": "Información",
        "error": "Error",
        "success": "Éxito",
        # Otros textos frecuentes
        "choose_theme_title": "Estilo de interfaz",
        "theme_dark": "Oscuro (Fusion dark)",
        "theme_light": "Claro (Fusion light)",
        "choose_method_title": "Elegir método de cálculo",
        "language_title": "Idioma / Language",
        "save_figure_title": "Guardar figura",
        "save_params_title": "Guardar parámetros",

        # === Emparejar Tamiz ↔ Láser ===
        "match_title": "Emparejar muestras Tamiz y Láser",
        "match_sieve_samples": "Muestras (Tamiz)",
        "match_laser_samples": "Muestras (Láser)",
        "match_pair_selection": "Emparejar selección",
        "match_col_origin": "Origen",
        "match_col_phi_star": "φ extraído (φ*)",
        "match_need_11": "Asigne a cada muestra tamizada su correspondiente de láser.",
        "match_need_origin": "Debes completar el campo 'Origen' en cada fila.",
        "match_need_phi": "Debes indicar el valor de φ* (phi extraído) en cada fila.",

        # === Histograma híbrido (superposición/residuo) ===
        "hist_overlay_toggle": "Mostrar superposición y residuo",
        "hist_residual_label": "Residuo del láser: {x}%",

        # === NUEVO: Selector de método (M1/M2/M3) ===
        "hybrid_method_title": "Unir tamiz + láser — elegir método",
        "hybrid_m1_name": "MÉTODO 1: φ* + EDLT (actual)",
        "hybrid_m2_name": "MÉTODO 2: SG (grueso solapado)",
        "hybrid_m3_name": "MÉTODO 3 (próximamente)",
        "hybrid_m1_desc": "Usa φ* y ajusta los finos con k = EDLT/∑L(φ≥φ*); el grueso del tamiz queda intacto.",
        "hybrid_m2_desc": "Finos como M1 y transfiere k·L(φ) al grueso coincidente (SG); renormaliza sin tocar el grueso fijo.",
        "hybrid_m3_desc": "Placeholder: definiremos el algoritmo.",
        "hybrid_continue": "Continuar",
        "hybrid_select_required": "Debes elegir un método para continuar.",
        "hybrid_not_implemented": "El Método 3 aún no está implementado.",
        "load_both_required": "Debes cargar datos de Tamiz y de Láser antes de combinar.",

        # === NUEVO: Histograma — tamiz original detrás del híbrido ===
        "hist_show_original_sieve": "Mostrar tamiz original (sombreado)",
        "hist_show_original_sieve_tip": "Superponer el histograma original del tamiz (solo en híbridos).",

        # === NUEVO: Histograma — Método de detección de modos (GMM/Weibull) ===
        "lbl_hist_method": "Método:",
        "hist_method_gmm": "GMM",
        "hist_method_weibull": "Weibull",
        "lbl_modes_n": "n subpoblaciones:",
        "lbl_modes_k": "k (componentes):",
        "btn_export_modes": "Exportar modos (CSV)",  # clave unificada (GMM o Weibull)
    },
    "en": {
        # App / Tabs
        "app_title": "FW Granulometry 1.0 - LAAN",
        "tab_xy": "XY Plot",
        "tab_walker": "Walker (1971 and 1983)",
        "tab_gk": "Gençalioğlu-Kuşcu et al. 2007",
        "tab_hist": "Histogram",
        # Menús
        "menu_file": "&File",
        "menu_config": "&Settings",
        "menu_db": "Database",
        "menu_help": "&Help",
        # Archivo
        "file_load_sieve": "Load sieve Excel file",
        "file_load_laser": "Load laser Excel file",
        "file_export_image": "Export current image",
        "file_combine": "Combine sieve + laser",
        # Configuración
        "cfg_method": "Computation method",
        "cfg_colors": "Group colors",
        "cfg_groups_all": "Visible groups (all tabs)",
        "cfg_ribbons": "Ribbons (XY)",
        "cfg_theme": "UI theme",
        "cfg_language": "Idioma / Language",
        "cfg_lang_es": "Español",
        "cfg_lang_en": "English",
        # DB
        "db_view_sieve": "View sieve database",
        "db_view_laser": "View laser database",
        "db_view_hybrid": "View sieve + laser database",
        # Ayuda
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
        "btn_filter_groups": "Filter groups",
        "btn_export_image": "Export image",
        "console_header": "Granulometric parameters",
        "btn_export_params": "Export granulometric parameters",
        # Walker/GK
        "btn_group_walker": "Select groups",
        "btn_group_gk": "Select groups",
        # Histograma
        "lbl_hist_width": "Bar width (%):",
        "btn_hist_colors": "Bar colors",
        "btn_hist_labels": "Edit histogram",
        "chk_hist_bars": "Histogram (bars)",
        "chk_hist_poly": "Frequency polygon",
        "chk_hist_cum": "Cumulative curve",
        "btn_export_hist": "Export plot",
        "chk_gmm": "Detect modes (AI)",
        "lbl_gmm_kmax": "max k:",
        "btn_export_gmm": "Export modes (CSV)",  # alias towards unified export
        "lbl_hist_base": "Database:",
        "lbl_hist_sample": "Sample:",
        "btn_plot_hist": "Plot histogram",
        # Valores por defecto de histograma
        "hist_title_def": "Histogram",
        "hist_xlabel_def": "φ",
        "hist_ylabel_def": "wt (%)",
        "hist_ylabel2_def": "Cumulative Frequency",
        # Diálogos genéricos
        "ok": "OK",
        "cancel": "Cancel",
        "close": "Close",
        "info": "Information",
        "error": "Error",
        "success": "Success",
        # Otros textos frecuentes
        "choose_theme_title": "UI theme",
        "theme_dark": "Dark (Fusion dark)",
        "theme_light": "Light (Fusion light)",
        "choose_method_title": "Choose computation method",
        "language_title": "Idioma / Language",
        "save_figure_title": "Save figure",
        "save_params_title": "Save parameters",

        # === Match Sieve ↔ Laser ===
        "match_title": "Match Sieve and Laser samples",
        "match_sieve_samples": "Samples (Sieve)",
        "match_laser_samples": "Samples (Laser)",
        "match_pair_selection": "Pair selection",
        "match_col_origin": "Origin",
        "match_col_phi_star": "Extracted φ (φ*)",
        "match_need_11": "Please assign each sieve sample its matching laser sample.",
        "match_need_origin": "You must complete the 'Origin' field for every row.",
        "match_need_phi": "You must provide φ* (extracted phi) for every row.",

        # === Hybrid histogram (overlay/residual) ===
        "hist_overlay_toggle": "Show overlay & residual",
        "hist_residual_label": "Laser residual: {x}%",

        # === NEW: Method selector (M1/M2/M3) ===
        "hybrid_method_title": "Combine sieve + laser — choose method",
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

        # === NEW: Histogram — original sieve behind hybrid ===
        "hist_show_original_sieve": "Show original sieve (shaded)",
        "hist_show_original_sieve_tip": "Overlay the original sieve histogram (hybrid only).",

        # === NEW: Histogram — Modes detection method (GMM/Weibull) ===
        "lbl_hist_method": "Method:",
        "hist_method_gmm": "GMM",
        "hist_method_weibull": "Weibull",
        "lbl_modes_n": "n subpopulations:",
        "lbl_modes_k": "k (components):",
        "btn_export_modes": "Export modes (CSV)",  # unified key (GMM or Weibull)
    },
}

def tr(key: str, lang: str = None, **fmt) -> str:
    """
    Traduce una clave. Si no existe, cae al idioma alterno y finalmente devuelve la clave.
    Permite formateo con **fmt, por ejemplo tr("hist_residual_label", x="16.4").
    """
    L = _current_lang(lang)
    txt = (
        I18N.get(L, {}).get(key)
        or I18N.get("en", {}).get(key)
        or I18N.get("es", {}).get(key)
        or key
    )
    try:
        return txt.format(**fmt) if fmt else txt
    except Exception:
        return txt

# endregion




# region # === Bloque 1: Importaciones y configuración global ===
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QAction,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QColorDialog,
    QCheckBox,
    QDialog,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QTextEdit,
    QMessageBox,
    QTableView,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    qApp,            # se usa para propagar idioma a nivel app
    QScrollArea,     # lo usan algunos diálogos
    QGridLayout      # lo usan algunos editores (colores)
)
from PyQt5.QtCore import Qt, QAbstractTableModel

# === ML opcional (GMM) ===
try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
# endregion


# region # === Bloque 2: Definición de constantes y diccionarios ===
DARK_STYLESHEET = """
QMainWindow { background-color: #232629; }
QWidget { background-color: #232629; color: #edeef0; }
QTabWidget::pane { border: 1px solid #444; background: #232629; }
QTabBar::tab { background: #2b2e31; color: #edeef0; border-radius: 4px; padding: 6px; }
QTabBar::tab:selected { background: #232629; color: #ffd60a; }
QPushButton { background-color: #393e41; color: #edeef0; border: 1px solid #444; border-radius: 6px; padding: 5px 12px;}
QPushButton:hover { background-color: #515457; }
QLineEdit, QComboBox, QTextEdit { background-color: #2b2e31; color: #edeef0; border: 1px solid #444; }
QGroupBox { border: 1px solid #444; margin-top: 10px; }
QRadioButton, QCheckBox { color: #edeef0; }
QMenuBar, QMenu, QMenu::item { background-color: #232629; color: #edeef0; }
QMenu::item:selected { background-color: #515457; }
QScrollBar { background: #232629; }
QLabel { color: #edeef0; }
"""
LIGHT_STYLESHEET = ""

# --- Archivos candidatos para ICONO y BANNER (branding) ---
# Incluye tu archivo 'icono.nuevo.png' y patrones flexibles por si cambia el nombre.
ICON_CANDIDATES = [
    "icono.nuevo.png", "icon.png", "icono.png", "fw.png", "fw_icon.png",
    "app.png", "logo.png", "logo32.png", "logo64.png", "icon.ico"
]
# Patrones adicionales: busca cualquier icono típico en la carpeta del script
ICON_PATTERNS = ["*icon*.png", "*icono*.png", "*logo*.png", "*fw*.png", "*.ico"]

# Si quieres usar un banner (opcional)
BANNER_CANDIDATES = ["banner.png", "banner.jpg", "banner.jpeg", "logo.png", "fw.png"]

# --- Símbolos de parámetros (comunes a ambos idiomas). ---
SYMBOLS = {
    "mean": "Mφ",
    "median": "Mdφ",
    "sigma": "σφ",
    "skewness": "Skφ",
    "kurtosis": "Kφ"
}

# Métodos disponibles en la UI
METHODS = {
    "Folk & Ward (1957)": "folkward",
    "Inman (1952)": "inman",
    "Trask (1932)": "trask"
}

# Claves internas para imágenes de Walker
WALKER_IMAGES = {
    "All": "WALKER.tif",
    "Fall deposit": "WALKER-FALL-DEPOSIT.tif",
    "Fine-depleted flow": "WALKER-FINE-DEPLETED-FLOW.tif",
    "Surge (Walker) and Surge-Dunes (Gençalioğlu-Kuşcu)": "WALKER-PYROCLASTIC-SURGE.tif",
    "Pyroclastic flow": "WALKER-PYROCLASTIC-FLOW.tif"
}

# Claves internas para imágenes de Gençalioğlu-Kuşcu et al. (2007)
GK_IMAGES = {
    "All": "GK.tif",
    "Flow": "GK-FLOW.tif",
    "Surge (Walker) and Surge-Dunes (Gençalioğlu-Kuşcu)": "GK-SURGEDUNES.tif",
    "Fall": "GK-FALL.tif",
    "Surge Dunes": "GK-SURGEDUNES.tif"
}

# Claves internas para imágenes de Pardo et al. (2009)  <<< NUEVO
PARDO_IMAGES = {
    "All": "PARDO-ALL.tif",
    "Fallout": "PARDO-FALLOUT.tif",
    "PF": "PARDO-PF.tif",
    "PS": "PARDO-PS.tif",
    "Zone 1": "PARDO-ZONE1.tif",
    "Zone 2": "PARDO-ZONE2.tif",
    "Zone 3": "PARDO-ZONE3.tif",
}

# Paleta de colores para grupos
GROUP_COLORS = [
    "#52b788",
    "#4895ef",
    "#e76f51",
    "#ffd60a",
    "#adb5bd",
]

# --- Opciones de presentación específicas para Trask ---
TRASK_DISPLAY = "phi_equiv"  # "phi_equiv" o "raw"

def sigma_axis_label(method: str) -> str:
    """Devuelve el rótulo del eje Y para 'sigma' según el método activo."""
    return "So (Trask)" if (method == "trask" and TRASK_DISPLAY == "raw") else SYMBOLS["sigma"]
# endregion



# region # === Bloque 3: Funciones auxiliares de cálculo ===

def cumulative_distribution(phi, weights):
    '''
    Devuelve la distribución acumulada en % para los pesos.
    Robusto a pesos=0 y NaN. Si total<=0, devuelve NaN en toda la serie.
    '''
    w = np.asarray(weights, dtype=float)
    total = np.nansum(w)
    if not np.isfinite(total) or total <= 0:
        return np.full(w.shape, np.nan, dtype=float)
    return np.cumsum(np.nan_to_num(w, nan=0.0)) * 100.0 / float(total)


def interp_percentile(phi, cum, target):
    '''
    Interpolación lineal robusta sobre la curva acumulada (en %).
    Maneja NaN y bordes; usa np.interp con cum no decreciente.
    '''
    x = np.asarray(phi, dtype=float)
    y = np.asarray(cum, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0 or y.size == 0:
        return np.nan

    # Asegurar orden por acumulada (debe ser no decreciente).
    order = np.argsort(y)
    y = y[order]; x = x[order]

    if target <= y[0]:
        return x[0]
    if target >= y[-1]:
        return x[-1]

    return float(np.interp(target, y, x))


def calculate_parameters(phi, weights, method="folkward"):
    '''
    Cálculo de parámetros con nombres CONSISTENTES:
      devuelve dict con: {"median","sigma","skewness","kurtosis","mean"}

    Métodos:
      - "folkward" (Folk & Ward, 1957)  -> usa percentiles 5-16-25-50-75-84-95
      - "inman"    (Inman, 1952)        -> usa 16-50-84
      - "trask"    (Trask, 1932)        -> So = sqrt(d75/d25) en DIÁMETRO:
                                           So = 2**((phi25 - phi75)/2)
                                           'sigma' reporta por defecto (phi75-phi25)/2 (equivalente en φ).
                                           Si TRASK_DISPLAY == "raw", 'sigma' = So (adimensional).
    '''
    phi = np.asarray(phi, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Limpiar NaN y ordenar por phi
    m = np.isfinite(phi) & np.isfinite(weights)
    phi = phi[m]; weights = weights[m]
    if phi.size == 0 or np.nansum(weights) <= 0:
        return {"median": np.nan, "sigma": np.nan, "skewness": np.nan,
                "kurtosis": np.nan, "mean": np.nan}

    order = np.argsort(phi)
    phi = phi[order]; weights = weights[order]

    # Helper de percentiles
    def _percs(*plist):
        cum = cumulative_distribution(phi, weights)
        if not np.isfinite(cum).any():
            return {p: np.nan for p in plist}
        return {p: interp_percentile(phi, cum, p) for p in plist}

    # --- Folk & Ward (alias "blatt") ---
    if method in ("folkward", "blatt"):
        cum = cumulative_distribution(phi, weights)
        if not np.isfinite(cum).any():
            return {"median": np.nan, "sigma": np.nan, "skewness": np.nan,
                    "kurtosis": np.nan, "mean": np.nan}

        phi5   = interp_percentile(phi, cum, 5)
        phi16  = interp_percentile(phi, cum, 16)
        phi25  = interp_percentile(phi, cum, 25)
        phi50  = interp_percentile(phi, cum, 50)
        phi75  = interp_percentile(phi, cum, 75)
        phi84  = interp_percentile(phi, cum, 84)
        phi95  = interp_percentile(phi, cum, 95)

        percs = [phi5, phi16, phi25, phi50, phi75, phi84, phi95]
        if not all(np.isfinite(percs)):
            return {"median": np.nan, "sigma": np.nan, "skewness": np.nan,
                    "kurtosis": np.nan, "mean": np.nan}

        median = phi50
        d8416 = (phi84 - phi16)
        d955  = (phi95 - phi5)
        d7525 = (phi75 - phi25)

        sigma = (d8416 / 4.0 if d8416 != 0 else np.nan) + (d955 / 6.6 if d955 != 0 else np.nan)
        term1 = ((phi16 + phi84 - 2 * phi50) / (2 * d8416)) if d8416 != 0 else np.nan
        term2 = ((phi5 + phi95  - 2 * phi50) / (2 * d955 )) if d955  != 0 else np.nan
        skewness = (term1 if np.isfinite(term1) else 0) + (term2 if np.isfinite(term2) else 0)
        kurtosis = (d955 / (2.44 * d7525)) if d7525 != 0 else np.nan
        mean     = (phi16 + phi50 + phi84) / 3.0

        return {"median": float(median), "sigma": float(sigma), "skewness": float(skewness),
                "kurtosis": float(kurtosis), "mean": float(mean)}

    # --- Inman (1952) ---
    elif method == "inman":
        p = _percs(16, 50, 84)
        phi16, phi50, phi84 = p[16], p[50], p[84]
        if not all(np.isfinite([phi16, phi50, phi84])):
            return {"median": np.nan, "sigma": np.nan, "skewness": np.nan,
                    "kurtosis": np.nan, "mean": np.nan}
        d = phi84 - phi16
        sigma    = d / 2 if d != 0 else np.nan
        skewness = (phi16 + phi84 - 2*phi50) / d if d != 0 else np.nan
        mean     = (phi16 + phi84) / 2
        return {"median": float(phi50), "sigma": float(sigma), "skewness": float(skewness),
                "kurtosis": np.nan, "mean": float(mean)}

    # --- Trask (1932) ---
    elif method == "trask":
        p = _percs(25, 50, 75, 16, 84)
        phi25, phi50, phi75 = p[25], p[50], p[75]
        phi16, phi84 = p[16], p[84]
        if not all(np.isfinite([phi25, phi50, phi75])):
            return {"median": np.nan, "sigma": np.nan, "skewness": np.nan,
                    "kurtosis": np.nan, "mean": np.nan}

        # So (Trask) en DIÁMETRO: So = sqrt(d75/d25) = 2**((phi25 - phi75)/2)
        so = float(2.0 ** ((phi25 - phi75) / 2.0))

        trask_display = globals().get("TRASK_DISPLAY", "phi_equiv")
        if trask_display == "raw":
            sigma = so
        else:
            sigma = (phi75 - phi25) / 2.0  # equivalente en φ (positivo)

        if all(np.isfinite([phi16, phi84])):
            mean = (phi16 + phi50 + phi84) / 3.0
        else:
            mean = phi50

        return {"median": float(phi50), "sigma": float(sigma), "skewness": np.nan,
                "kurtosis": np.nan, "mean": float(mean)}

    else:
        raise ValueError("Método desconocido")


# === Helper adicional (láser um->phi y agregado por bins) ===
def agrupar_laser_por_phi(df_laser_raw: pd.DataFrame, step: float = 1.0):
    '''
    Espera un DataFrame con columnas: ['Sample', 'Diameter (μm)', '1 (%)', 'Group'].
    Convierte μm -> φ y cuantiza a la grilla del usuario:
      - step=1.0  -> centros en enteros (..., -2, -1, 0, 1, ...)
      - step=0.5  -> centros en múltiplos de 0.5
    Luego suma % por (Sample, phi_centro).
    Devuelve lista de dicts: [{'Sample':..., 'phi':..., 'Wt':...}, ...]
    '''
    required = ["Sample", "Diameter (μm)", "1 (%)"]
    for col in required:
        if col not in df_laser_raw.columns:
            raise ValueError("Falta la columna '{}' en el archivo de láser".format(col))

    df = df_laser_raw.copy()

    # Normalizar decimales coma->punto
    d_um = pd.to_numeric(
        df["Diameter (μm)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    wt = pd.to_numeric(
        df["1 (%)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    samp = df["Sample"].astype(str)

    # Filtrar válidos
    m = np.isfinite(d_um) & (d_um > 0) & np.isfinite(wt)
    d_um = d_um[m]; wt = wt[m]; samp = samp[m]

    # μm -> mm -> phi
    phi = -np.log2(d_um / 1000.0)

    # Cuantización a la grilla solicitada (anclada en 0)
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


# region # === Bloque 4: Funciones de trazado genéricas ===
def plot_ribbon(ax, x, y, color, alpha=0.17, width=0.07, smooth=15):
    # Fallback robusto: si SciPy no está, usa una elipse suave
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
        u_fine = np.linspace(0, 1, len(expanded)*smooth)
        xf, yf = splev(u_fine, tck)
        ax.fill(xf, yf, color=color, alpha=alpha, linewidth=0, zorder=1)
    except Exception:
        if len(x) < 3:
            return
        xm, ym = np.mean(x), np.mean(y)
        rx, ry = np.std(x)*1.8, np.std(y)*1.8
        t = np.linspace(0, 2*np.pi, 200)
        ax.fill(xm + rx*np.cos(t), ym + ry*np.sin(t), alpha=alpha, color=color, zorder=1, linewidth=0)

def get_script_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))
# endregion



# region # === Bloque 5: Diálogos, modelo de tablas y helpers de visualización (versión limpia) ===
"""
Incluye:
  - PandasModel (modelo Qt para DataFrame)
  - Delegate global de 2 decimales para tablas (instalación automática)
  - MatchDialog (emparejar Tamiz↔Láser y fijar φ*)
  - BulkExportHistDialog (NUEVO): exportar múltiples histogramas con el mismo formato

Nota: Los diálogos auxiliares duplicados viven en el Bloque 15:
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

# ---------- 5.0: Modelo Qt para DataFrames (solo lectura) ----------
class PandasModel(QAbstractTableModel):
    """
    Modelo simple de solo lectura para mostrar un pandas.DataFrame en QTableView.
    - Soporta ordenamiento por columna.
    - Guarda el DataFrame en self._df para que otros diálogos puedan leerlo (MatchDialog).
    """
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.reset_index(drop=True).copy()

    # tamaño
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    # datos (texto a mostrar)
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            val = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(val) else str(val)
        return None

    # encabezados
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

    # solo seleccionable
    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    # ordenar por columna
    def sort(self, column, order=Qt.AscendingOrder):
        if column < 0 or column >= len(self._df.columns):
            return
        self.layoutAboutToBeChanged.emit()
        colname = self._df.columns[column]
        # intentar ordenar como numérico (soportando coma decimal)
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


# ---------- 5.1: Formato numérico UI — delegate global de 2 decimales ----------
class _TwoDecimalsDelegate(QStyledItemDelegate):
    """Muestra cualquier valor numérico con 2 decimales (solo visual)."""
    def displayText(self, value, locale):
        try:
            v = float(str(value).replace(",", "."))
            if np.isfinite(v):
                return f"{v:.2f}"
        except Exception:
            return str(value)
        return f"{v:.2f}"

class _TableFormatter(QObject):
    """Instala el delegate en cada QTableWidget/QTableView que se muestre."""
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

# Instalación automática del filtro en toda la app
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


# ---------- 5.2: Diálogo para emparejar Tamiz ↔ Láser (usa PandasModel) ----------
class MatchDialog(QDialog):
    """
    Ventana para emparejar muestras de Tamiz y Láser y fijar el φ extraído (φ*).
    - Multiselección con Ctrl/Shift en ambas listas y emparejado 1–a–1 por orden.
    - φ* SIN valor por defecto (specialValue '—'); es OBLIGATORIO.
    """
    def __init__(self, df_tamiz, df_laser, parent=None):
        super().__init__(parent)

        _es = (_current_lang() == "es")
        title         = tr("match_title")
        lbl_samples_t = tr("match_sieve_samples")
        lbl_samples_l = tr("match_laser_samples")
        btn_pair_txt  = tr("match_pair_selection")
        btn_done_txt  = tr("ok")
        col_left      = base_name_display("Tamiz")
        col_right     = base_name_display("Laser")
        col_phi_star  = tr("match_col_phi_star")

        self.setWindowTitle(title)
        self.resize(1100, 560)

        layout = QHBoxLayout(self)

        # ======= Lista TAMIZ =======
        self.tv_t = QTableView()
        tamiz_samples = (
            df_tamiz.iloc[:, 1].drop_duplicates().astype(str).sort_values().tolist()
        )
        df_t = pd.DataFrame(tamiz_samples, columns=[lbl_samples_t])
        self.model_t = PandasModel(df_t)
        self.tv_t.setModel(self.model_t)
        self.tv_t.setSelectionBehavior(QTableView.SelectRows)
        self.tv_t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        try:
            self.tv_t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception:
            pass
        self.tv_t.setStyleSheet("QTableView::item:selected { background-color:#0078d7; color:white; }")
        layout.addWidget(self.tv_t, 2)

        # ======= Lista LÁSER =======
        self.tv_l = QTableView()
        laser_samples = (
            df_laser["Sample"].drop_duplicates().astype(str).sort_values().tolist()
        )
        df_l = pd.DataFrame(laser_samples, columns=[lbl_samples_l])
        self.model_l = PandasModel(df_l)
        self.tv_l.setModel(self.model_l)
        self.tv_l.setSelectionBehavior(QTableView.SelectRows)
        self.tv_l.setSelectionMode(QAbstractItemView.ExtendedSelection)
        try:
            self.tv_l.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        except Exception:
            pass
        self.tv_l.setStyleSheet("QTableView::item:selected { background-color:#0078d7; color:white; }")
        layout.addWidget(self.tv_l, 2)

        # ======= Tabla de emparejamientos =======
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([col_left, col_right, col_phi_star])
        try:
            self.table.setColumnWidth(0, 260)
            self.table.setColumnWidth(1, 260)
            self.table.setColumnWidth(2, 160)
        except Exception:
            pass
        layout.addWidget(self.table, 3)

        # ======= Botonera derecha =======
        right = QVBoxLayout()
        self.btn_pair = QPushButton(btn_pair_txt)
        self.btn_pair.clicked.connect(self.pair_selected)
        right.addWidget(self.btn_pair)

        self.btn_delete = QPushButton("Eliminar fila" if _es else "Delete row")
        self.btn_delete.clicked.connect(self.delete_current_row)
        right.addWidget(self.btn_delete)

        right.addStretch()
        self.btn_ok = QPushButton(btn_done_txt)
        self.btn_ok.clicked.connect(self.accept)
        right.addWidget(self.btn_ok)
        layout.addLayout(right)

    # utilidades
    def _selected_texts(self, view, model):
        sel = view.selectionModel()
        if not sel:
            return []
        rows = sorted([ix.row() for ix in sel.selectedRows()])
        out = []
        for r in rows:
            try:
                out.append(str(model._df.iloc[r, 0]))
            except Exception:
                pass
        return out

    def pair_selected(self):
        t_list = self._selected_texts(self.tv_t, self.model_t)
        l_list = self._selected_texts(self.tv_l, self.model_l)

        if not t_list or not l_list or (len(t_list) != len(l_list)):
            QMessageBox.warning(self, tr("error"), tr("match_need_11"))
            return

        for t_name, l_name in zip(t_list, l_list):
            if self._already_paired(t_name, l_name):
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(t_name)))
            self.table.setItem(row, 1, QTableWidgetItem(str(l_name)))
            spin = self._make_phi_spinbox()
            self.table.setCellWidget(row, 2, spin)

    def _already_paired(self, t_name, l_name):
        used_t, used_l = set(), set()
        for r in range(self.table.rowCount()):
            it_t = self.table.item(r, 0)
            it_l = self.table.item(r, 1)
            if it_t: used_t.add(it_t.text())
            if it_l: used_l.add(it_l.text())
        return (t_name in used_t) or (l_name in used_l)

    def _make_phi_spinbox(self):
        sp = QDoubleSpinBox()
        sp.setRange(-20.0, 20.0)
        sp.setDecimals(2)
        sp.setSingleStep(0.5)
        sp.setSpecialValueText("—")   # valor "no asignado"
        sp.setMinimum(-20.0)
        sp.setValue(sp.minimum())     # arranca en “—”
        sp.setToolTip("φ* (phi extraído). Obligatorio; no puede quedar en '—'.")
        return sp

    def delete_current_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)

    def accept(self):
        n = self.table.rowCount()
        if n == 0:
            QMessageBox.warning(self, tr("error"), tr("match_need_11"))
            return

        used_t, used_l = set(), set()
        for r in range(n):
            it_t = self.table.item(r, 0)
            it_l = self.table.item(r, 1)
            sp   = self.table.cellWidget(r, 2)
            if it_t is None or it_l is None or sp is None:
                QMessageBox.warning(self, tr("error"), tr("match_need_11"))
                return
            t_name = it_t.text().strip()
            l_name = it_l.text().strip()
            try:
                if float(sp.value()) == float(sp.minimum()):
                    QMessageBox.warning(self, tr("error"), tr("match_need_phi"))
                    return
            except Exception:
                QMessageBox.warning(self, tr("error"), tr("match_need_phi"))
                return
            if t_name in used_t or l_name in used_l:
                QMessageBox.warning(self, tr("error"), tr("match_need_11"))
                return
            used_t.add(t_name); used_l.add(l_name)

        super().accept()


# ---------- 5.3: NUEVO — Diálogo de exportación múltiple de histogramas ----------
class BulkExportHistDialog(QDialog):
    """
    Permite elegir:
      - Base (Tamiz / Laser / Híbrido, o las que reciba)
      - Conjunto de muestras (multiselección con 'Seleccionar todo')
      - Formato (PNG / SVG / PDF)
    Al presionar 'Exportar', pide carpeta de destino y devuelve selección.

    Uso:
      dlg = BulkExportHistDialog(bases, samples_by_base, current_base="Tamiz", parent=self)
      if dlg.exec_() == QDialog.Accepted:
          base, samples, fmt, out_dir = dlg.get_results()
    """
    def __init__(self, bases, samples_by_base, current_base=None, parent=None):
        super().__init__(parent)
        self.setModal(True)

        self._bases = list(bases or [])
        self._samples_by_base = dict(samples_by_base or {})
        self._result = None  # (base, samples, fmt, out_dir)

        _es = (_current_lang() == "es")
        title = "Exportar múltiples histogramas" if _es else "Export multiple histograms"
        lbl_base = "Base" if _es else "Dataset"
        lbl_samples = "Muestras" if _es else "Samples"
        lbl_fmt = "Formato" if _es else "Format"
        btn_export = "Exportar" if _es else "Export"
        btn_cancel = "Cancelar" if _es else "Cancel"
        chk_all = "Seleccionar todo" if _es else "Select all"
        choose_dir_txt = "Elegí la carpeta destino" if _es else "Choose output folder"
        no_samples_txt = "Elegí al menos una muestra." if _es else "Choose at least one sample."
        no_dir_txt = "No se seleccionó carpeta de destino." if _es else "No destination folder selected."

        self.setWindowTitle(title)
        self.resize(560, 520)

        root = QVBoxLayout(self)

        # --- Base ---
        row_base = QHBoxLayout()
        row_base.addWidget(QLabel(lbl_base))
        self.cmb_base = QComboBox()
        self.cmb_base.addItems(self._bases)
        if current_base and current_base in self._bases:
            self.cmb_base.setCurrentText(current_base)
        row_base.addWidget(self.cmb_base)
        root.addLayout(row_base)

        # --- Muestras ---
        root.addWidget(QLabel(lbl_samples))
        top_samples = QHBoxLayout()
        self.chk_all = QCheckBox(chk_all)
        self.chk_all.toggled.connect(self._toggle_all_samples)
        top_samples.addWidget(self.chk_all)
        top_samples.addStretch()
        root.addLayout(top_samples)

        self.lst_samples = QListWidget()
        self.lst_samples.setSelectionMode(QAbstractItemView.MultiSelection)
        root.addWidget(self.lst_samples, 1)

        # --- Formato ---
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel(lbl_fmt))
        self.cmb_fmt = QComboBox()
        self.cmb_fmt.addItems(["PNG", "SVG", "PDF"])
        row_fmt.addWidget(self.cmb_fmt)
        row_fmt.addStretch()
        root.addLayout(row_fmt)

        # --- Botonera ---
        btns = QDialogButtonBox()
        self.btn_ok = btns.addButton(btn_export, QDialogButtonBox.AcceptRole)
        self.btn_cancel = btns.addButton(btn_cancel, QDialogButtonBox.RejectRole)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # señales
        self.cmb_base.currentTextChanged.connect(self._reload_samples)

        # carga inicial
        self._reload_samples()
        # por defecto, marcar todo al iniciar
        self.chk_all.setChecked(True)
        self._toggle_all_samples(True)

        # strings auxiliares
        self._choose_dir_txt = choose_dir_txt
        self._no_samples_txt = no_samples_txt
        self._no_dir_txt = no_dir_txt

    # --- helpers ---
    def _reload_samples(self):
        base = self.cmb_base.currentText()
        samples = self._samples_by_base.get(base, []) or []
        self.lst_samples.clear()
        for s in samples:
            it = QListWidgetItem(str(s))
            self.lst_samples.addItem(it)

    def _toggle_all_samples(self, checked: bool):
        for i in range(self.lst_samples.count()):
            it = self.lst_samples.item(i)
            it.setSelected(checked)

    def _on_accept(self):
        base = self.cmb_base.currentText().strip()
        selected = [self.lst_samples.item(i).text() for i in range(self.lst_samples.count()) if self.lst_samples.item(i).isSelected()]
        if not selected:
            QMessageBox.warning(self, "⚠", self._no_samples_txt)
            return

        out_dir = QFileDialog.getExistingDirectory(self, self._choose_dir_txt, os.getcwd())
        if not out_dir:
            QMessageBox.information(self, "ℹ", self._no_dir_txt)
            return

        fmt = self.cmb_fmt.currentText().strip().lower()  # 'png' | 'svg' | 'pdf'
        self._result = (base, selected, fmt, out_dir)
        self.accept()

    def get_results(self):
        """
        Returns: (base:str, samples:list[str], fmt:str, out_dir:str)
        """
        return self._result

# endregion




# region # === Bloque 6: Canvas de tamaño fijo ===
from PyQt5.QtWidgets import QSizePolicy, QApplication

class FixedSizeFigureCanvas(FigureCanvas):
    """
    Lienzo de Matplotlib con tamaño fijo en píxeles.
    Útil para mostrar imágenes de fondo (Walker / GK) sin deformaciones.
    Limita el tamaño máximo del canvas a un porcentaje de la pantalla disponible
    para evitar ventanas gigantes en monitores pequeños.
    """
    def __init__(self, width, height, dpi=100, *args, **kwargs):
        # Tamaño solicitado (en píxeles) según width/height (en inches) y dpi
        w_px = int(round(width * dpi))
        h_px = int(round(height * dpi))

        # Dimensiones máximas permitidas según la pantalla disponible
        screen = QApplication.primaryScreen()
        if screen is not None:
            ag = screen.availableGeometry()
            max_w = int(ag.width() * 0.92)   # p.ej. 92% del ancho disponible
            max_h = int(ag.height() * 0.92)  # p.ej. 92% del alto disponible
        else:
            # Fallback razonable si no hay pantalla (tests/headless)
            max_w, max_h = 1800, 1000

        # Escala para no exceder pantalla
        scale_w = max_w / w_px if w_px > 0 else 1.0
        scale_h = max_h / h_px if h_px > 0 else 1.0
        scale = min(1.0, scale_w, scale_h)

        # Tamaño final en píxeles
        scaled_w = max(1, int(round(w_px * scale)))
        scaled_h = max(1, int(round(h_px * scale)))

        # Crear figura con el tamaño final (convertido a inches)
        fig = plt.figure(figsize=(scaled_w / dpi, scaled_h / dpi), dpi=dpi)
        super().__init__(fig)

        # Fijar tamaño del canvas para que los layouts no lo redimensionen
        self.setFixedSize(scaled_w, scaled_h)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # (Opcional) Guardar info útil por si se necesita más adelante
        self._requested_px = (w_px, h_px)
        self._scaled_px = (scaled_w, scaled_h)
        self._scale = scale
# endregion




# region # === Bloque 7: Clase MainWindow – Constructor, icono y menú principal (SIN banner) ===
from PyQt5.QtWidgets import QApplication, QScrollArea

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- estado base / defaults ----
        self.lang = _current_lang()
        try:
            qApp.lang = self.lang
        except Exception:
            pass

        self.setWindowTitle(tr("app_title"))

        # Tamaño inicial relativo a la pantalla (evita ventanas gigantes)
        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                ag = screen.availableGeometry()
                w = max(900, int(ag.width() * 0.85))
                h = max(600, int(ag.height() * 0.85))
                self.resize(w, h)
                self.setMinimumSize(900, 600)  # piso razonable
            else:
                self.resize(1200, 800)
                self.setMinimumSize(900, 600)
        except Exception:
            self.resize(1200, 800)
            self.setMinimumSize(900, 600)

        self.theme = "light"
        self.current_method = "folkward"

        # <<< NUEVO: recordar método de unión híbrida seleccionado >>>
        self.hybrid_method = None  # o "m1" si querés recordar el último

        # DataFrames / resultados
        self.df_data = None
        self.param_results = None
        self.group_col = "Group"

        self.df_laser = None
        self.param_results_laser = None

        self.df_hybrid = None
        self.param_results_hybrid = None

        # Grupos y colores
        self.groups = []
        self.group_colors = {}
        self.selected_groups_xy = []
        self.selected_groups_ribbons = []
        self.selected_groups_walker = []
        self.selected_groups_gk = []
        self.selected_groups_pardo = []   # <<< NUEVO

        # Estilo inicial
        try:
            qApp.setStyleSheet(LIGHT_STYLESHEET if self.theme == "light" else DARK_STYLESHEET)
        except Exception:
            pass

        # Icono de la app/ventana (solo icono, sin banner)
        self._apply_icon()

        # Contenedor central simple: SOLO las tabs (sin imagen arriba)
        self._init_central_no_banner()

        # Menú y pestañas
        self.initMenu()
        self.initUI_tabs()

        # === Estado IA/Modos ===
        self._gmm_last = None             # compat
        self._modes_last = None           # <<< NUEVO: estado unificado (GMM o Weibull)
        self.hist_mode_method = "gmm"     # <<< NUEVO: método activo en Histograma ("gmm" | "weibull")

    # ---------- Icono ----------
    def _apply_icon(self):
        """
        Busca un icono en ICON_CANDIDATES (Bloque 2) y lo aplica a nivel app y ventana.
        En Windows conviene .ico. Si no hay, toma .png/.jpg.
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

    # ---------- Centro sin banner ----------
    def _init_central_no_banner(self):
        """
        Crea el QWidget central y un layout vertical que contiene ÚNICAMENTE
        el QTabWidget. Sin flyer, sin banner, sin imagen.
        """
        self._central = QWidget(self)
        self._root = QVBoxLayout(self._central)
        self._root.setContentsMargins(6, 6, 6, 6)
        self._root.setSpacing(6)

        # Solo las tabs
        self.tabs = QTabWidget(self._central)
        self._root.addWidget(self.tabs)

        self.setCentralWidget(self._central)

    # ---------- Menú principal ----------
    def initMenu(self):
        menubar = self.menuBar()

        # --- Archivo / File ---
        menu_file = menubar.addMenu(tr("menu_file"))

        act_load_tamiz = QAction(tr("file_load_sieve"), self)
        act_load_tamiz.triggered.connect(self.load_file)
        menu_file.addAction(act_load_tamiz)

        act_load_laser = QAction(tr("file_load_laser"), self)
        act_load_laser.triggered.connect(self.load_laser_file)
        menu_file.addAction(act_load_laser)

        # Exportar imagen de la pestaña ACTUAL (ahora con 5 tabs)
        act_save = QAction(tr("file_export_image"), self)
        act_save.triggered.connect(
            lambda: self.export_canvas(
                [
                    getattr(self, "canvas_xy", None),
                    getattr(self, "canvas_walker", None),
                    getattr(self, "canvas_gk", None),
                    getattr(self, "canvas_pardo", None),   # <<< NUEVO
                    getattr(self, "canvas_hist", None),
                ][self.tabs.currentIndex()]
            )
        )
        menu_file.addAction(act_save)

        act_combine = QAction(tr("file_combine"), self)
        act_combine.triggered.connect(self.combinar_tamiz_laser)
        menu_file.addAction(act_combine)

        # --- Configuración / Settings ---
        menu_cfg = menubar.addMenu(tr("menu_config"))
        menu_cfg.addAction(QAction(tr("cfg_method"), self, triggered=self.choose_method))
        menu_cfg.addAction(QAction(tr("cfg_colors"), self, triggered=self.edit_colors))
        menu_cfg.addAction(QAction(tr("cfg_groups_all"), self, triggered=self.select_groups_all))
        menu_cfg.addAction(QAction(tr("cfg_ribbons"), self, triggered=self.select_ribbons))
        menu_cfg.addAction(QAction(tr("cfg_theme"), self, triggered=self.choose_theme))

        # Idioma / Language
        lang_menu = menu_cfg.addMenu(tr("cfg_language"))
        self.act_lang_es = QAction(tr("cfg_lang_es"), self, checkable=True)
        self.act_lang_en = QAction(tr("cfg_lang_en"), self, checkable=True)
        self.act_lang_es.setChecked(self.lang == "es")
        self.act_lang_en.setChecked(self.lang == "en")

        def _switch_lang(new_lang: str):
            if new_lang == "es":
                self.act_lang_es.setChecked(True);  self.act_lang_en.setChecked(False)
            else:
                self.act_lang_es.setChecked(False); self.act_lang_en.setChecked(True)
            self.lang = new_lang
            set_app_language(new_lang)
            self.apply_translations()

        self.act_lang_es.triggered.connect(lambda _: _switch_lang("es"))
        self.act_lang_en.triggered.connect(lambda _: _switch_lang("en"))
        lang_menu.addAction(self.act_lang_es)
        lang_menu.addAction(self.act_lang_en)

        # --- Base de datos / Database ---
        menu_db = menubar.addMenu(tr("menu_db"))

        self.act_viewdb_tamiz = QAction(tr("db_view_sieve"), self)
        self.act_viewdb_tamiz.setEnabled(False)
        self.act_viewdb_tamiz.triggered.connect(self.show_tamiz_db_window)
        menu_db.addAction(self.act_viewdb_tamiz)

        self.act_viewdb_laser = QAction(tr("db_view_laser"), self)
        self.act_viewdb_laser.setEnabled(False)
        self.act_viewdb_laser.triggered.connect(self.show_laser_db_window)
        menu_db.addAction(self.act_viewdb_laser)

        self.act_viewdb_hybrid = QAction(tr("db_view_hybrid"), self)
        self.act_viewdb_hybrid.setEnabled(False)
        self.act_viewdb_hybrid.triggered.connect(self.show_hybrid_db_window)
        menu_db.addAction(self.act_viewdb_hybrid)

        # --- Ayuda / Help ---
        menu_help = menubar.addMenu(tr("menu_help"))
        menu_help.addAction(
            QAction(
                tr("help_about"), self,
                triggered=lambda: QMessageBox.information(self, tr("about_title"), tr("about_text"))
            )
        )

        # <<< NUEVO: sincronizar habilitado de acciones del menú DB >>>
        self._update_db_menu_actions()

    # <<< NUEVO: helper para habilitar/deshabilitar “Ver base de datos …” >>>
    def _update_db_menu_actions(self):
        has_tamiz  = (self.df_data   is not None) and (not getattr(self.df_data, "empty", True))
        has_laser  = (self.df_laser  is not None) and (not getattr(self.df_laser, "empty", True))
        has_hybrid = (self.df_hybrid is not None) and (not getattr(self.df_hybrid, "empty", True))
        try:
            self.act_viewdb_tamiz.setEnabled(has_tamiz)
            self.act_viewdb_laser.setEnabled(has_laser)
            self.act_viewdb_hybrid.setEnabled(has_hybrid)
        except Exception:
            pass

    # ---------- Actualización de grupos/colores ----------
    def _update_all_groups_and_colors(self):
        """
        Recolecta todos los grupos visibles desde las bases cargadas (Tamiz/Láser/Híbrido),
        arma la lista self.groups con el sufijo de base para la leyenda y asegura
        self.group_colors + selecciones por defecto.
        """
        bases = []

        # Tamiz
        if getattr(self, "param_results", None) is not None and not getattr(self, "param_results").empty:
            bases.append(("Tamiz", self.param_results, getattr(self, "group_col", "Group")))

        # Láser
        if getattr(self, "param_results_laser", None) is not None and not getattr(self, "param_results_laser").empty:
            bases.append(("Laser", self.param_results_laser, "Group"))

        # Híbrido
        if getattr(self, "param_results_hybrid", None) is not None and not getattr(self, "param_results_hybrid").empty:
            bases.append(("Híbrido", self.param_results_hybrid, getattr(self, "group_col", "Group")))

        # Unificar claves "Grupo (Base)"
        unique_groups = []
        for base_key, df, gcol in bases:
            for grp in sorted(df[gcol].astype(str).unique()):
                clave_disp = f"{grp} ({base_name_display(base_key)})"
                if clave_disp not in unique_groups:
                    unique_groups.append(clave_disp)
        self.groups = unique_groups

        # Inicializar/actualizar diccionario de colores
        if not hasattr(self, "group_colors") or self.group_colors is None:
            self.group_colors = {}
        for i, clave in enumerate(self.groups):
            if clave not in self.group_colors:
                self.group_colors[clave] = GROUP_COLORS[i % len(GROUP_COLORS)]

        # Selecciones por defecto (todas visibles)
        self.selected_groups_xy      = list(self.groups)
        self.selected_groups_ribbons = list(self.groups)
        self.selected_groups_walker  = list(self.groups)
        self.selected_groups_gk      = list(self.groups)
        self.selected_groups_pardo   = list(self.groups)   # <<< NUEVO

    # ---------- Pestañas (sin logos internos) ----------
    def initUI_tabs(self):
        self.current_db_selection = {"tamiz": True, "laser": False, "hybrid": False}

        # === Tab XY ===
        self.tab_xy = QWidget()
        v_xy = QVBoxLayout(self.tab_xy)
        h_controls = QHBoxLayout()

        self.btn_load_tamiz = QPushButton(tr("btn_load_sieve")); self.btn_load_tamiz.clicked.connect(self.load_file)
        h_controls.addWidget(self.btn_load_tamiz)

        self.lbl_x = QLabel(tr("lbl_x")); h_controls.addWidget(self.lbl_x)
        self.cmb_x = QComboBox(); h_controls.addWidget(self.cmb_x)

        self.lbl_y = QLabel(tr("lbl_y")); h_controls.addWidget(self.lbl_y)
        self.cmb_y = QComboBox(); h_controls.addWidget(self.cmb_y)

        opts = ["mean", "median", "sigma", "skewness", "kurtosis"]
        self.cmb_x.addItems(opts); self.cmb_y.addItems(opts)
        self.cmb_x.setCurrentText("mean"); self.cmb_y.setCurrentText("sigma")

        self.btn_plot_xy = QPushButton(tr("btn_plot_xy")); self.btn_plot_xy.clicked.connect(self.plot_xy)
        h_controls.addWidget(self.btn_plot_xy)

        self.btn_select_db = QPushButton(tr("btn_choose_db")); self.btn_select_db.clicked.connect(self.select_db_to_plot)
        h_controls.addWidget(self.btn_select_db)

        self.chk_show_names = QCheckBox(tr("chk_show_names")); self.chk_show_names.setChecked(False)
        self.chk_show_names.toggled.connect(self.plot_xy); h_controls.addWidget(self.chk_show_names)

        self.btn_filter_groups = QPushButton(tr("btn_filter_groups")); self.btn_filter_groups.clicked.connect(self.select_groups_xy)
        h_controls.addWidget(self.btn_filter_groups)

        h_controls.addStretch()

        self.btn_export_xy = QPushButton(tr("btn_export_image"))
        self.btn_export_xy.clicked.connect(lambda: self.export_canvas(self.canvas_xy))
        h_controls.addWidget(self.btn_export_xy)

        v_xy.addLayout(h_controls)

        hdr_console = QHBoxLayout()
        self.lbl_console = QLabel(tr("console_header")); self.lbl_console.setStyleSheet("font-weight: bold;")
        hdr_console.addWidget(self.lbl_console); hdr_console.addStretch()
        self.btn_export_params = QPushButton(tr("btn_export_params")); self.btn_export_params.clicked.connect(self.export_params_to_excel)
        hdr_console.addWidget(self.btn_export_params)
        v_xy.addLayout(hdr_console)

        self.txt_console = QTextEdit(); self.txt_console.setReadOnly(True); self.txt_console.setMaximumHeight(190)
        v_xy.addWidget(self.txt_console)

        self.canvas_xy = FigureCanvas(plt.figure(figsize=(8, 4.5)))
        self.canvas_xy.setFixedHeight(450)
        v_xy.addWidget(self.canvas_xy)

        self.tabs.addTab(self.tab_xy, tr("tab_xy"))

        # === Tab Walker ===
        self.tab_walker = QWidget()
        v_walker = QVBoxLayout(self.tab_walker)
        h_walker = QHBoxLayout()

        self.walker_groupbox = QGroupBox()
        self.walker_buttons  = QButtonGroup()
        walker_hbox = QHBoxLayout(self.walker_groupbox)

        for i, key in enumerate(WALKER_IMAGES.keys()):
            rb = QRadioButton(key)
            rb.setProperty("imgkey", key)
            if i == 0: rb.setChecked(True)
            self.walker_buttons.addButton(rb)
            walker_hbox.addWidget(rb)
        self.walker_buttons.buttonClicked.connect(self.plot_walker)
        h_walker.addWidget(self.walker_groupbox)
        h_walker.addStretch()

        self.btn_export_walker = QPushButton(tr("btn_export_image"))
        self.btn_export_walker.clicked.connect(lambda: self.export_canvas(self.canvas_walker))
        h_walker.addWidget(self.btn_export_walker)
        v_walker.addLayout(h_walker)

        script_dir = get_script_dir()
        walker_img = mpimg.imread(os.path.join(script_dir, WALKER_IMAGES["All"]))
        img_h, img_w = walker_img.shape[:2]; dpi = 100
        self.canvas_walker = FixedSizeFigureCanvas(img_w/dpi, img_h/dpi, dpi)
        v_walker.addWidget(self.canvas_walker)

        self.tabs.addTab(self.tab_walker, tr("tab_walker"))

        # === Tab GK ===
        self.tab_gk = QWidget()
        v_gk = QVBoxLayout(self.tab_gk)
        h_gk = QHBoxLayout()

        self.gk_groupbox = QGroupBox()
        self.gk_buttons  = QButtonGroup()
        gk_hbox = QHBoxLayout(self.gk_groupbox)

        for i, key in enumerate(GK_IMAGES.keys()):
            rb = QRadioButton(key)
            rb.setProperty("imgkey", key)
            if i == 0: rb.setChecked(True)
            self.gk_buttons.addButton(rb)
            gk_hbox.addWidget(rb)
        self.gk_buttons.buttonClicked.connect(self.plot_gk)
        h_gk.addWidget(self.gk_groupbox)
        h_gk.addStretch()

        self.btn_export_gk = QPushButton(tr("btn_export_image"))
        self.btn_export_gk.clicked.connect(lambda: self.export_canvas(self.canvas_gk))
        h_gk.addWidget(self.btn_export_gk)
        v_gk.addLayout(h_gk)

        gk_img = mpimg.imread(os.path.join(script_dir, GK_IMAGES["All"]))
        img_h2, img_w2 = gk_img.shape[:2]; dpi = 100
        self.canvas_gk = FixedSizeFigureCanvas(img_w2/dpi, img_h2/dpi, dpi)
        v_gk.addWidget(self.canvas_gk)

        self.tabs.addTab(self.tab_gk, tr("tab_gk"))

        # === Tab Pardo et al. 2009 (NUEVA) ===
        self.tab_pardo = QWidget()
        v_pardo = QVBoxLayout(self.tab_pardo)
        h_pardo = QHBoxLayout()

        self.pardo_groupbox = QGroupBox()
        self.pardo_buttons  = QButtonGroup()
        pardo_hbox = QHBoxLayout(self.pardo_groupbox)

        # Radios según PARDO_IMAGES
        for i, key in enumerate(PARDO_IMAGES.keys()):
            rb = QRadioButton(key)
            rb.setProperty("imgkey", key)
            if i == 0: rb.setChecked(True)
            self.pardo_buttons.addButton(rb)
            pardo_hbox.addWidget(rb)
        self.pardo_buttons.buttonClicked.connect(self.plot_pardo)
        h_pardo.addWidget(self.pardo_groupbox)
        h_pardo.addStretch()

        self.btn_export_pardo = QPushButton(tr("btn_export_image"))
        self.btn_export_pardo.clicked.connect(lambda: self.export_canvas(self.canvas_pardo))
        h_pardo.addWidget(self.btn_export_pardo)
        v_pardo.addLayout(h_pardo)

        pardo_img = mpimg.imread(os.path.join(script_dir, PARDO_IMAGES["All"]))
        img_hp, img_wp = pardo_img.shape[:2]; dpi = 100
        self.canvas_pardo = FixedSizeFigureCanvas(img_wp/dpi, img_hp/dpi, dpi)
        v_pardo.addWidget(self.canvas_pardo)

        self.tabs.addTab(self.tab_pardo, "Pardo et al. 2009")

        # === Tab Histograma ===
        self.tab_hist = QWidget()
        v_hist = QVBoxLayout(self.tab_hist)

        h_opts = QHBoxLayout()

        self.lbl_hist_width = QLabel(tr("lbl_hist_width"))
        h_opts.addWidget(self.lbl_hist_width)

        self.spn_hist_width = QDoubleSpinBox(); self.spn_hist_width.setRange(1, 100); self.spn_hist_width.setValue(70); self.spn_hist_width.setSingleStep(1)
        h_opts.addWidget(self.spn_hist_width)

        self.btn_hist_colors = QPushButton(tr("btn_hist_colors")); h_opts.addWidget(self.btn_hist_colors)
        self.btn_labels = QPushButton(tr("btn_hist_labels")); h_opts.addWidget(self.btn_labels)

        self.chk_hist = QCheckBox(tr("chk_hist_bars")); self.chk_hist.setChecked(True); h_opts.addWidget(self.chk_hist)
        self.chk_poly = QCheckBox(tr("chk_hist_poly")); self.chk_poly.setChecked(True); h_opts.addWidget(self.chk_poly)
        self.chk_cum = QCheckBox(tr("chk_hist_cum")); self.chk_cum.setChecked(True); h_opts.addWidget(self.chk_cum)

        _es = (_current_lang() == "es")
        self.chk_overlay_residue = QCheckBox(tr("hist_overlay_toggle") if hasattr(self, "tabs") else ("Mostrar superposición y residuo" if _es else "Show overlay & residual"))
        self.chk_overlay_residue.setChecked(False)
        self.chk_overlay_residue.setToolTip(tr("hist_show_original_sieve_tip") if _es else tr("hist_show_original_sieve_tip"))
        self.chk_overlay_residue.toggled.connect(self.plot_histogram)
        h_opts.addWidget(self.chk_overlay_residue)

        self.chk_show_sieve_original = QCheckBox(tr("hist_show_original_sieve"))
        try:
            self.chk_show_sieve_original.setToolTip(tr("hist_show_original_sieve_tip"))
        except Exception:
            pass
        self.chk_show_sieve_original.setChecked(False)
        self.chk_show_sieve_original.toggled.connect(self.plot_histogram)
        h_opts.addWidget(self.chk_show_sieve_original)

        # Export individual
        self.btn_export_hist = QPushButton(tr("btn_export_hist")); h_opts.addWidget(self.btn_export_hist)

        # <<< NUEVO: Exportación múltiple con mismo formato >>>
        txt_bulk = "Exportar más de un histograma con este mismo formato" if _es else "Export multiple histograms with this format"
        self.btn_export_hist_bulk = QPushButton(txt_bulk)
        self.btn_export_hist_bulk.clicked.connect(self.open_bulk_hist_export)
        h_opts.addWidget(self.btn_export_hist_bulk)

        # --- NUEVO: Selector de método (GMM / Weibull) ---
        self.lbl_hist_method = QLabel(tr("lbl_hist_method"))
        h_opts.addWidget(self.lbl_hist_method)

        self.cmb_hist_method = QComboBox()
        self.cmb_hist_method.addItems([tr("hist_method_gmm"), tr("hist_method_weibull")])
        self.cmb_hist_method.setCurrentIndex(0)  # GMM por defecto
        self.cmb_hist_method.currentIndexChanged.connect(self._on_hist_method_changed)
        h_opts.addWidget(self.cmb_hist_method)

        # --- "Play" unificado (IA) ---
        self.chk_gmm = QCheckBox(tr("chk_gmm"))  # etiqueta genérica "Detectar modos (IA)"
        self.chk_gmm.setChecked(False)
        h_opts.addWidget(self.chk_gmm)

        # --- NUEVO: Spinner unificado (k / n) ---
        self.lbl_modes_nk = QLabel(tr("lbl_modes_k"))  # arranca como k (GMM)
        h_opts.addWidget(self.lbl_modes_nk)

        self.spn_modes_nk = QSpinBox()
        self.spn_modes_nk.setRange(1, 6)
        self.spn_modes_nk.setValue(4)
        h_opts.addWidget(self.spn_modes_nk)

        # --- NUEVO: Exportación de modos unificada ---
        self.btn_export_modes = QPushButton(tr("btn_export_modes"))
        self.btn_export_modes.setEnabled(False)
        self.btn_export_modes.clicked.connect(self._on_export_gmm_clicked)  # mismo handler
        h_opts.addWidget(self.btn_export_modes)

        # Alias de compatibilidad (código viejo puede referirse a btn_export_gmm)
        self.btn_export_gmm = self.btn_export_modes

        h_opts.addStretch()

        # <<< NUEVO: envolver las opciones en un scroll horizontal para que no queden fuera de la ventana >>>
        _opts_container = QWidget()
        _opts_container.setLayout(h_opts)
        sc_opts = QScrollArea()
        sc_opts.setWidget(_opts_container)
        sc_opts.setWidgetResizable(True)
        v_hist.addWidget(sc_opts)

        # >>> LÍNEAS AGREGADAS (conectar helper y fijar altura fina) <<<
        self.sc_hist_opts = sc_opts              # <- necesario para que el helper lo ajuste
        self.sc_hist_opts.setFixedHeight(56)     # <- evita la barra “gorda” al entrar

        h_sel = QHBoxLayout()
        self.lbl_hist_base = QLabel(tr("lbl_hist_base")); h_sel.addWidget(self.lbl_hist_base)
        self.cmb_hist_base = QComboBox(); h_sel.addWidget(self.cmb_hist_base)
        self.lbl_hist_sample = QLabel(tr("lbl_hist_sample")); h_sel.addWidget(self.lbl_hist_sample)
        self.cmb_hist_sample = QComboBox(); self.cmb_hist_sample.setMinimumWidth(220); h_sel.addWidget(self.cmb_hist_sample)

        self.btn_plot_hist = QPushButton(tr("btn_plot_hist")); self.btn_plot_hist.clicked.connect(self.plot_histogram)
        h_sel.addWidget(self.btn_plot_hist)

        h_sel.addStretch()
        v_hist.addLayout(h_sel)

        # === Defaults de histograma ===
        # Colores/visuales
        self.hist_bar_fill = "skyblue"
        self.hist_bar_edge = "black"
        self.poly_color    = "gray"
        self.cum_color     = "black"

        self.hist_title   = ""
        self.hist_xlabel  = ""
        self.hist_ylabel  = ""
        self.hist_ylabel2 = ""

        # <<< NUEVO: tamaños de fuentes y tamaño físico del gráfico (modificables desde diálogo) >>>
        self.hist_title_size = 13
        self.hist_label_size = 11
        self.hist_tick_size  = 9
        self.hist_fig_w      = 6.0  # inches
        self.hist_fig_h      = 4.0  # inches

        # Canvas del histograma
        self.canvas_hist = FigureCanvas(plt.figure(figsize=(self.hist_fig_w, self.hist_fig_h)))
        v_hist.addWidget(self.canvas_hist)

        self.tabs.addTab(self.tab_hist, tr("tab_hist"))

        # Conexiones
        self.spn_hist_width.valueChanged.connect(self.plot_histogram)
        self.btn_hist_colors.clicked.connect(self.choose_hist_colors)
        self.btn_labels.clicked.connect(self.choose_hist_labels)
        self.chk_hist.toggled.connect(self.plot_histogram)
        self.chk_poly.toggled.connect(self.plot_histogram)
        self.chk_cum.toggled.connect(self.plot_histogram)
        self.btn_export_hist.clicked.connect(lambda: self.export_canvas(self.canvas_hist))

        # "Play" y parámetros de modos
        self.chk_gmm.toggled.connect(self.plot_histogram)
        self.spn_modes_nk.valueChanged.connect(self.plot_histogram)

        self.cmb_hist_base.currentTextChanged.connect(self._on_hist_base_changed)

        # Inicialización combos histograma + fondos Walker/GK/Pardo
        self._update_hist_bases()
        self._on_hist_base_changed()
        self.plot_walker()
        self.plot_gk()
        self.plot_pardo()   # <<< NUEVO

    # --- NUEVO: cambio de método GMM/Weibull en Histograma ---
    def _on_hist_method_changed(self, idx: int):
        txt = self.cmb_hist_method.currentText().lower()
        # normalizo por display: contiene "gmm" o "weibull"
        if "weibull" in txt:
            self.hist_mode_method = "weibull"
            self.lbl_modes_nk.setText(tr("lbl_modes_n"))
        else:
            self.hist_mode_method = "gmm"
            self.lbl_modes_nk.setText(tr("lbl_modes_k"))

        # Si el "play" está activo, re-calcular al cambiar el método
        if self.chk_gmm.isChecked():
            self.plot_histogram()

    # Wrapper seguro para exportar modos (si aún no pegaste Bloque 14)
    def _on_export_gmm_clicked(self):
        if hasattr(self, "export_gmm_csv_from_state"):
            self.export_gmm_csv_from_state()  # (Bloque 14: soporta GMM/Weibull)
        else:
            QMessageBox.information(
                self, "IA/GMM",
                "Para exportar modos, pega el Bloque 14 (export_gmm_csv_from_state)."
                if _current_lang()=="es" else
                "To export modes, paste Block 14 (export_gmm_csv_from_state)."
            )
# endregion








# region # === Bloque 8: Configuración de pestañas de gráficos (initUI) ===
    def initUI(self):
        self.initMenu()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Inicializar selección de bases (solo Tamiz por defecto)
        self.current_db_selection = {"tamiz": True, "laser": False, "hybrid": False}

        # =========================
        # === 8.1: Pestaña XY ===
        # =========================
        self.tab_xy = QWidget()
        v_xy = QVBoxLayout(self.tab_xy)
        h_controls = QHBoxLayout()

        # logo/banner arriba a la izquierda
        h_controls.addWidget(self._make_logo_label(22))

        self.btn_load_tamiz = QPushButton(tr("btn_load_sieve"))
        self.btn_load_tamiz.clicked.connect(self.load_file)
        h_controls.addWidget(self.btn_load_tamiz)

        # Etiquetas y combos de ejes
        self.lbl_x = QLabel(tr("lbl_x"))
        h_controls.addWidget(self.lbl_x)
        self.cmb_x = QComboBox()
        h_controls.addWidget(self.cmb_x)

        self.lbl_y = QLabel(tr("lbl_y"))
        h_controls.addWidget(self.lbl_y)
        self.cmb_y = QComboBox()
        h_controls.addWidget(self.cmb_y)

        # Pre-cargar opciones para evitar errores antes de cargar datos
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

        # Mostrar nombres de muestra
        self.chk_show_names = QCheckBox(tr("chk_show_names"))
        self.chk_show_names.setChecked(False)
        self.chk_show_names.toggled.connect(self.plot_xy)
        h_controls.addWidget(self.chk_show_names)

        # Botón para filtrar grupos (XY)
        self.btn_filter_groups = QPushButton(tr("btn_filter_groups"))
        self.btn_filter_groups.clicked.connect(self.select_groups_xy)
        h_controls.addWidget(self.btn_filter_groups)

        h_controls.addStretch()

        self.btn_export_xy = QPushButton(tr("btn_export_image"))
        self.btn_export_xy.clicked.connect(lambda: self.export_canvas(self.canvas_xy))  # (Bloque 11)
        h_controls.addWidget(self.btn_export_xy)

        v_xy.addLayout(h_controls)

        # Encabezado de consola + exportar parámetros
        hdr_console = QHBoxLayout()
        self.lbl_console = QLabel(tr("console_header"))
        self.lbl_console.setStyleSheet("font-weight: bold;")
        hdr_console.addWidget(self.lbl_console)
        hdr_console.addStretch()
        self.btn_export_params = QPushButton(tr("btn_export_params"))
        self.btn_export_params.clicked.connect(self.export_params_to_excel)  # (Bloque 13)
        hdr_console.addWidget(self.btn_export_params)
        v_xy.addLayout(hdr_console)

        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumHeight(190)
        v_xy.addWidget(self.txt_console)

        self.canvas_xy = FigureCanvas(plt.figure(figsize=(8, 4.5)))
        self.canvas_xy.setFixedHeight(450)
        v_xy.addWidget(self.canvas_xy)

        self.tabs.addTab(self.tab_xy, tr("tab_xy"))

        # ===========================================
        # === 8.2: Pestaña Walker (1971 and 1983) ===
        # ===========================================
        self.tab_walker = QWidget()
        v_walker = QVBoxLayout(self.tab_walker)
        h_walker = QHBoxLayout()

        # logo/banner
        h_walker.addWidget(self._make_logo_label(22))

        self.walker_groupbox = QGroupBox()
        self.walker_buttons  = QButtonGroup()
        walker_hbox = QHBoxLayout(self.walker_groupbox)

        # Crear radios a partir de las claves de imagen y guardar 'imgkey'
        for i, key in enumerate(WALKER_IMAGES.keys()):
            rb = QRadioButton(key)  # texto inicial (se traduce luego en apply_translations)
            rb.setProperty("imgkey", key)  # clave estable para las imágenes
            if i == 0:
                rb.setChecked(True)
            self.walker_buttons.addButton(rb)
            walker_hbox.addWidget(rb)
        self.walker_buttons.buttonClicked.connect(self.plot_walker)
        h_walker.addWidget(self.walker_groupbox)
        h_walker.addStretch()

        self.btn_export_walker = QPushButton(tr("btn_export_image"))
        self.btn_export_walker.clicked.connect(lambda: self.export_canvas(self.canvas_walker))  # (Bloque 11)
        h_walker.addWidget(self.btn_export_walker)
        v_walker.addLayout(h_walker)

        self.btn_group_walker = QPushButton(tr("btn_group_walker"))
        self.btn_group_walker.clicked.connect(self.select_groups_walker)
        v_walker.addWidget(self.btn_group_walker)

        script_dir = get_script_dir()
        walker_img = mpimg.imread(os.path.join(script_dir, WALKER_IMAGES["All"]))
        img_h, img_w = walker_img.shape[:2]; dpi = 100
        self.canvas_walker = FixedSizeFigureCanvas(img_w/dpi, img_h/dpi, dpi)
        v_walker.addWidget(self.canvas_walker)

        self.tabs.addTab(self.tab_walker, tr("tab_walker"))

        # ================================================
        # === 8.3: Pestaña Gençalioğlu-Kuşcu et al 2007 ===
        # ================================================
        self.tab_gk = QWidget()
        v_gk = QVBoxLayout(self.tab_gk)
        h_gk = QHBoxLayout()

        # logo/banner
        h_gk.addWidget(self._make_logo_label(22))

        self.gk_groupbox = QGroupBox()
        self.gk_buttons  = QButtonGroup()
        gk_hbox = QHBoxLayout(self.gk_groupbox)

        for i, key in enumerate(GK_IMAGES.keys()):
            rb = QRadioButton(key)
            rb.setProperty("imgkey", key)
            if i == 0:
                rb.setChecked(True)
            self.gk_buttons.addButton(rb)
            gk_hbox.addWidget(rb)
        self.gk_buttons.buttonClicked.connect(self.plot_gk)
        h_gk.addWidget(self.gk_groupbox)
        h_gk.addStretch()

        self.btn_export_gk = QPushButton(tr("btn_export_image"))
        self.btn_export_gk.clicked.connect(lambda: self.export_canvas(self.canvas_gk))  # (Bloque 11)
        h_gk.addWidget(self.btn_export_gk)
        v_gk.addLayout(h_gk)

        self.btn_group_gk = QPushButton(tr("btn_group_gk"))
        self.btn_group_gk.clicked.connect(self.select_groups_gk)
        v_gk.addWidget(self.btn_group_gk)

        gk_img = mpimg.imread(os.path.join(script_dir, GK_IMAGES["All"]))
        img_h2, img_w2 = gk_img.shape[:2]; dpi = 100
        self.canvas_gk = FixedSizeFigureCanvas(img_w2/dpi, img_h2/dpi, dpi)
        v_gk.addWidget(self.canvas_gk)

        self.tabs.addTab(self.tab_gk, tr("tab_gk"))

        # =================================
        # === 8.4: Pestaña Histograma  ===
        # =================================
        self.tab_hist = QWidget()
        v_hist = QVBoxLayout(self.tab_hist)

        # Panel de opciones
        h_opts = QHBoxLayout()

        # logo/banner
        h_opts.addWidget(self._make_logo_label(22))

        self.lbl_hist_width = QLabel(tr("lbl_hist_width"))  # "Ancho de barra (%):" / "Bar width (%):"
        h_opts.addWidget(self.lbl_hist_width)

        self.spn_hist_width = QDoubleSpinBox()
        self.spn_hist_width.setRange(1, 100)
        self.spn_hist_width.setValue(70)
        self.spn_hist_width.setSingleStep(1)
        h_opts.addWidget(self.spn_hist_width)

        self.btn_hist_colors = QPushButton(tr("btn_hist_colors"))
        h_opts.addWidget(self.btn_hist_colors)

        self.btn_labels = QPushButton(tr("btn_hist_labels"))
        h_opts.addWidget(self.btn_labels)

        self.chk_hist = QCheckBox(tr("chk_hist_bars"))
        self.chk_hist.setChecked(True)
        h_opts.addWidget(self.chk_hist)

        self.chk_poly = QCheckBox(tr("chk_hist_poly"))
        self.chk_poly.setChecked(True)
        h_opts.addWidget(self.chk_poly)

        self.chk_cum = QCheckBox(tr("chk_hist_cum"))
        self.chk_cum.setChecked(True)
        h_opts.addWidget(self.chk_cum)

        self.btn_export_hist = QPushButton(tr("btn_export_hist"))
        h_opts.addWidget(self.btn_export_hist)

        # IA/GMM
        self.chk_gmm = QCheckBox(tr("chk_gmm"))
        self.chk_gmm.setChecked(False)
        h_opts.addWidget(self.chk_gmm)

        self.lbl_gmm_kmax = QLabel(tr("lbl_gmm_kmax"))  # "k máx:" / "max k:"
        h_opts.addWidget(self.lbl_gmm_kmax)

        self.spn_gmm_kmax = QSpinBox()
        self.spn_gmm_kmax.setRange(1, 6)
        self.spn_gmm_kmax.setValue(4)
        h_opts.addWidget(self.spn_gmm_kmax)

        self.btn_export_gmm = QPushButton(tr("btn_export_gmm"))
        self.btn_export_gmm.setEnabled(False)
        self.btn_export_gmm.clicked.connect(self._on_export_gmm_clicked)  # wrapper seguro
        h_opts.addWidget(self.btn_export_gmm)

        h_opts.addStretch()
        v_hist.addLayout(h_opts)

        # Selección de base y muestra
        h_sel = QHBoxLayout()
        self.lbl_hist_base = QLabel(tr("lbl_hist_base"))   # "Base:" / "Database:"
        h_sel.addWidget(self.lbl_hist_base)
        self.cmb_hist_base = QComboBox()
        h_sel.addWidget(self.cmb_hist_base)

        self.lbl_hist_sample = QLabel(tr("lbl_hist_sample")) # "Muestra:" / "Sample:"
        h_sel.addWidget(self.lbl_hist_sample)
        self.cmb_hist_sample = QComboBox()
        self.cmb_hist_sample.setMinimumWidth(220)
        h_sel.addWidget(self.cmb_hist_sample)

        # *** IMPORTANTE: corrige el texto del botón con la clave i18n correcta ***
        self.btn_plot_hist = QPushButton(tr("btn_plot_hist"))
        self.btn_plot_hist.clicked.connect(self.plot_histogram)
        h_sel.addWidget(self.btn_plot_hist)

        h_sel.addStretch()
        v_hist.addLayout(h_sel)

        # Canvas Histograma
        self.canvas_hist = FigureCanvas(plt.figure(figsize=(6, 4)))
        v_hist.addWidget(self.canvas_hist)

        self.tabs.addTab(self.tab_hist, tr("tab_hist"))

        # === Defaults de histograma ===
        self.hist_bar_fill = "skyblue"
        self.hist_bar_edge = "black"
        self.poly_color    = "gray"
        self.cum_color     = "black"

        # Defaults de labels (se ajustan luego en apply_translations si están vacíos)
        self.hist_title   = ""
        self.hist_xlabel  = ""
        self.hist_ylabel  = ""
        self.hist_ylabel2 = ""

        # Conexiones de opciones
        self.spn_hist_width.valueChanged.connect(self.plot_histogram)
        self.btn_hist_colors.clicked.connect(self.choose_hist_colors)
        self.btn_labels.clicked.connect(self.choose_hist_labels)
        self.chk_hist.toggled.connect(self.plot_histogram)
        self.chk_poly.toggled.connect(self.plot_histogram)
        self.chk_cum.toggled.connect(self.plot_histogram)
        self.btn_export_hist.clicked.connect(lambda: self.export_canvas(self.canvas_hist))  # (Bloque 11)

        self.chk_gmm.toggled.connect(self.plot_histogram)
        self.spn_gmm_kmax.valueChanged.connect(self.plot_histogram)

        # Reaccionar al cambio de base para refrescar muestras
        self.cmb_hist_base.currentTextChanged.connect(self._on_hist_base_changed)

        # Inicializar combos del histograma
        self._update_hist_bases()
        self._on_hist_base_changed()

        # Inicialmente, refrescar Walker/GK para que muestren el fondo
        self.plot_walker()
        self.plot_gk()

    # Wrapper seguro para exportar GMM (si aún no pegaste Bloque 14)
    def _on_export_gmm_clicked(self):
        if hasattr(self, "export_gmm_csv_from_state"):
            self.export_gmm_csv_from_state()  # (Bloque 14)
        else:
            QMessageBox.information(
                self, "IA/GMM",
                "Para exportar modos, pega el Bloque 14 (export_gmm_csv_from_state)."
                if _current_lang()=="es" else
                "To export modes, paste Block 14 (export_gmm_csv_from_state)."
            )
# endregion



# region # === Bloque 9: Métodos de carga de archivos y bases de datos ===

    # ========= Helpers TAMIZ: autodetección y normalización =========

    def _to_num(self, s):
        """
        Convierte a numérico aceptando coma o punto como decimal.
        Soporta Series, arrays y DataFrames (columna por columna).
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
        """True si todos los valores finitos son múltiplos de 0.5."""
        x = np.asarray(arr, dtype=float)
        m = np.isfinite(x)
        if not m.any():
            return False
        return np.all(np.isclose((x[m] * 2) - np.round(x[m] * 2), 0.0, atol=tol))

    def _make_unique(self, names):
        """
        Devuelve lista de nombres única (agrega ' (2)', ' (3)', ... a duplicados).
        No toca el orden.
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
        """True si v se interpreta como número (admite coma o punto)."""
        if pd.isna(v):
            return False
        s = str(v).strip().replace(",", ".")
        try:
            float(s)
            return True
        except Exception:
            return False

    def _fraction_nonnumeric(self, values):
        """Fracción de celdas NO numéricas en una lista/serie."""
        vals = list(values)
        nn = sum(0 if self._is_numeric_like(v) else 1 for v in vals) if vals else 0
        return nn / float(len(vals) or 1)

    def _has_sample_repetition(self, series_like):
        """
        True si hay al menos un 'Sample' que aparece en ≥2 filas.
        (Evita confundir Formato 2 compacto con Formato 1.)
        """
        s = pd.Series(series_like)
        vc = s.value_counts(dropna=False)
        return (vc >= 2).any()

    def _normalize_tamiz_formats(self, df0: pd.DataFrame):
        """
        Detecta y normaliza TAMIZ en dos formatos:

        Formato 1 (por filas, EXACTAMENTE 4 columnas):
          Col0: φ (numérica, múltiplos de 0.5)
          Col1: Sample (puede ser numérica o no)
          Col2: Weight (numérica)
          Col3: Group (numérica o no) — debe ser constante por Sample,
                y debe existir REPETICIÓN de algún Sample (≥2 filas).

        Formato 2 (mixto por columnas):
          df.columns           = encabezados → Col0=Phi, Col1..=nombres de muestra
          df.iloc[0, :]        = fila de grupos (idealmente NO numérica en mayoría)
          df.iloc[1:, 0]       = φ (numérica, múltiplos de 0.5)
          df.iloc[1:, 1..n]    = Weight por muestra (numérico)
          Si la fila de grupos NO es mayormente no numérica, se asume “sin grupos”
          y los datos empiezan desde df.iloc[0, :].

        Devuelve (df_norm, group_col_name) con columnas:
          [phi_col, 'Sample', 'Weight', group_col_name]
        """
        if df0 is None or df0.shape[1] < 1:
            return None, None

        ncols = df0.shape[1]

        # ---------- Intento Formato 1: EXACTAMENTE 4 columnas ----------
        if ncols == 4:
            df = df0.iloc[:, :4].copy()

            # φ y Weight como numéricos (admite coma/punto)
            phi = self._to_num(df.iloc[:, 0])
            wt  = self._to_num(df.iloc[:, 2])

            # Validaciones duras
            if not self._is_multiple_of_half(phi):
                pass
            elif wt.notna().sum() == 0:
                pass
            else:
                # Requiere repetición de algún Sample (≥2 filas con mismo Sample)
                if not self._has_sample_repetition(df.iloc[:, 1]):
                    pass
                else:
                    # Limpiar y ordenar
                    df.iloc[:, 0] = phi
                    df.iloc[:, 2] = wt
                    df = df.dropna(subset=[df.columns[0], df.columns[2]]).copy()
                    df.sort_values([df.columns[1], df.columns[0]], inplace=True, kind="mergesort")
                    df.reset_index(drop=True, inplace=True)

                    # Consistencia de Group por Sample
                    group_col_name = str(df.columns[3])
                    samp_col_name  = str(df.columns[1])
                    for sample, g in df.groupby(samp_col_name):
                        vals = g.iloc[:, 3].astype(str).str.strip()
                        non_empty = vals.replace({"": np.nan, "nan": np.nan}).dropna()
                        uniq = non_empty.unique()
                        if len(uniq) > 1:
                            return None, None  # grupo inconsistente dentro de la muestra
                        fill_val = uniq[0] if len(uniq) == 1 else "Sin Grupo"
                        df.loc[g.index, df.columns[3]] = fill_val

                    # OK Formato 1
                    return df, group_col_name

        # ---------- Intento Formato 2 (incluye caso "compacto" con 4 columnas) ----------
        if df0.shape[0] >= 1 and ncols >= 2:
            # Encabezados reales
            header_names = list(df0.columns)
            phi_col_name = str(header_names[0]) if str(header_names[0]).strip() else str(df0.columns[0])

            # Candidatas a columnas de muestra: 1..n con encabezado válido
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

            # Fila “de grupos” candidata = primera fila de datos
            if df0.shape[0] >= 2:
                groups_row = list(df0.iloc[0, :])
                grp_vals_for_samples = [groups_row[j] for j in sample_cols]
                frac_nonn = self._fraction_nonnumeric(grp_vals_for_samples)
                groups_present = frac_nonn > 0.60  # umbral 60%
                data_start_row = 1 if groups_present else 0
            else:
                groups_present = False
                data_start_row = 0

            # Serie de φ desde data_start_row
            phi_series = self._to_num(df0.iloc[data_start_row:, 0])
            if not self._is_multiple_of_half(phi_series):
                return None, None

            # Nombres de muestra únicos y grupos alineados al orden de columnas
            sample_names = self._make_unique(sample_names_raw)

            # Construir DataFrame ancho con filas desde data_start_row
            wide = df0.iloc[data_start_row:, [0] + sample_cols].copy()
            wide.columns = [phi_col_name] + sample_names

            # Convertir a numérico: φ y todos los pesos
            wide[phi_col_name] = self._to_num(wide[phi_col_name])
            wide[sample_names] = self._to_num(wide[sample_names])

            # Eliminar filas sin φ (NaN). Mantener ceros en pesos.
            wide = wide.dropna(subset=[phi_col_name])

            # Pasar a formato largo
            long = wide.melt(
                id_vars=[phi_col_name],
                value_vars=sample_names,
                var_name="Sample",
                value_name="Weight"
            )
            # Quitar solo NaN en Weight (no ceros)
            long = long.dropna(subset=["Weight"]).copy()

            # Asignar grupos
            if groups_present:
                sample_groups_raw = [groups_row[j] for j in sample_cols]
                norm_groups = []
                for gv in sample_groups_raw:
                    gtxt = str(gv).strip()
                    if gtxt == "" or gtxt.lower() == "nan":
                        gtxt = "Sin Grupo"
                    norm_groups.append(gtxt)
                grp_map = dict(zip(sample_names, norm_groups))
                long["Group"] = long["Sample"].map(grp_map).fillna("Sin Grupo")
            else:
                long["Group"] = "Sin Grupo"

            # Ordenar y devolver
            long.sort_values(["Sample", phi_col_name], inplace=True, kind="mergesort")
            long.reset_index(drop=True, inplace=True)

            return long[[phi_col_name, "Sample", "Weight", "Group"]], "Group"

        # Sin coincidencia
        return None, None

    # ------------------------ TAMIZ: carga con autodetección ------------------------
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
            QMessageBox.warning(self, tr("error"), f"{'No se pudo leer el archivo' if _current_lang()=='es' else 'Could not read file'}:\n{e}")
            return

        df_norm, group_col_name = self._normalize_tamiz_formats(df0)
        if df_norm is None:
            if _current_lang() == "es":
                msg = (
                    "El Excel de tamiz debe cumplir uno de los dos formatos:\n"
                    " • Formato 1 (4 columnas): φ | Sample | Weight | Group "
                    "(φ múltiplos de 0.5; algún Sample con ≥2 filas; Group constante por Sample)\n"
                    " • Formato 2 (mixto por columnas): df.columns = encabezados, primera fila = grupos "
                    "(mayoría NO numérica) y datos desde la fila siguiente; si no hay fila de grupos válida "
                    "se asumen 'Sin Grupo' y los datos empiezan desde la primera fila."
                )
            else:
                msg = (
                    "The sieve Excel must match one of two formats:\n"
                    " • Format 1 (4 columns): φ | Sample | Weight | Group "
                    "(φ multiples of 0.5; at least one Sample with ≥2 rows; Group constant per Sample)\n"
                    " • Format 2 (mixed by columns): df.columns = headers, first row = groups "
                    "(mostly NON-numeric) and data from the next row; if there is no valid groups row, "
                    "'No Group' is assumed and data starts from the first row."
                )
            QMessageBox.warning(self, tr("info"), msg)
            return

        # Guardar datos normalizados
        self.group_col = group_col_name  # en Formato 2 será 'Group'
        self.df_data   = df_norm

        # Calcular parámetros
        params = []
        for sample, grp in self.df_data.groupby(self.df_data.columns[1]):  # 'Sample'
            phi     = grp.iloc[:, 0].tolist()
            weights = grp.iloc[:, 2].tolist()
            gval    = grp.iloc[0,  3]
            p       = calculate_parameters(phi, weights, self.current_method)
            row     = {"Sample": sample, self.group_col: gval, **p}
            params.append(row)

        self.param_results = pd.DataFrame(params)

        # Actualiza listas de grupos, colores y selecciones
        self._update_all_groups_and_colors()

        # Población de combos y refresco de vistas
        self.cmb_x.clear(); self.cmb_y.clear()
        self.cmb_x.addItems(["mean", "median", "sigma", "skewness", "kurtosis"])
        self.cmb_y.addItems(["mean", "median", "sigma", "skewness", "kurtosis"])
        self.cmb_x.setCurrentText("mean"); self.cmb_y.setCurrentText("sigma")

        self.update_console()
        self.plot_xy()
        self.plot_walker()
        self.plot_gk()
        # >>> NUEVO: si existe el canvas de Pardo, repintarlo también
        if hasattr(self, "canvas_pardo"):
            try:
                self.plot_pardo()
            except Exception:
                pass
        # reemplazo: habilita menús de DB según disponibilidad
        self._update_db_menu_actions()

        # También refrescar Histograma
        self._update_hist_samples()

    # ------------------------------- LÁSER (mapeo por nombre; orden libre) -------------------------------
    def _map_laser_columns_by_name(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve un DataFrame con EXACTAMENTE estas columnas y nombres:
          ['Sample', 'Diameter (μm)', '1 (%)', 'Group']
        Detecta columnas por encabezado (tolerando español/inglés),
        acentos y símbolos. Lanza ValueError si falta alguna.
        """
        import re, unicodedata

        def norm(s: str) -> str:
            s = unicodedata.normalize("NFKD", str(s))
            s = "".join(c for c in s if not unicodedata.combining(c))  # quita acentos
            s = s.lower()
            s = s.replace("µ", "u").replace("μ", "u")   # µ/μ -> u
            s = re.sub(r"[^a-z0-9%]+", "", s)          # quita espacios y símbolos
            return s

        # Normalizamos encabezados una sola vez
        norm_map = {col: norm(col) for col in df_in.columns}

        def find_col(pred):
            for col, n in norm_map.items():
                if pred(n):
                    return col
            return None

        # Reglas de detección
        sample_col = find_col(lambda n: any(k in n for k in ("sample", "muestra", "nombre", "name", "id")))
        diam_col   = find_col(lambda n: ("diamet" in n) or ("size" in n) or (n.startswith("d") and "um" in n))
        # Soporta "1 (%)", "%", "wt %", "porcentaje", etc.
        perc_col   = find_col(lambda n: (n in ("1", "1%", "1percent")) or ("porcent" in n) or ("percent" in n) or ("wt" in n) or n.endswith("%"))
        group_col  = find_col(lambda n: any(k in n for k in ("group", "grupo", "facies", "filtro")))

        missing = []
        if sample_col is None: missing.append("Sample/Muestra")
        if diam_col   is None: missing.append("Diameter (μm)/Diámetro (μm)")
        if perc_col   is None: missing.append("1 (%) / wt (%) / % / Porcentaje")
        if group_col  is None: missing.append("Group/Grupo/Facies/Filtro")

        if missing:
            if _current_lang() == "es":
                raise ValueError(
                    "No se pudieron detectar estas columnas por nombre: "
                    + ", ".join(missing)
                    + "\nEncabezados detectados: "
                    + ", ".join(map(str, df_in.columns))
                )
            else:
                raise ValueError(
                    "Could not detect these columns by name: "
                    + ", ".join(missing)
                    + "\nDetected headers: "
                    + ", ".join(map(str, df_in.columns))
                )

        # DF canónico (se ignoran columnas extra)
        df = df_in[[sample_col, diam_col, perc_col, group_col]].copy()
        df.columns = ["Sample", "Diameter (μm)", "1 (%)", "Group"]
        return df

    def load_laser_file(self):
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
            QMessageBox.warning(self, tr("error"), f"{'No se pudo leer el archivo' if _current_lang()=='es' else 'Could not read file'}:\n{e}")
            return
        if df.shape[1] < 3:
            msg = ("El archivo de láser debe tener al menos tres columnas con encabezados reconocibles "
                   "(muestra, diámetro en μm, porcentaje) y una de grupo/facies.") \
                  if _current_lang() == "es" \
                  else ("The laser file must have at least three columns with recognizable headers "
                        "(sample, diameter in μm, percent) and one group/facies column.")
            QMessageBox.warning(self, tr("error"), msg)
            return

        # Paso φ
        dlg = QDialog(self)
        dlg.setWindowTitle("Paso de φ para láser" if _current_lang()=="es" else "φ step for laser")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("¿Procesamiento cada 1 o 0.5 φ?" if _current_lang()=="es" else "Process every 1.0 or 0.5 φ?"))
        rb1 = QRadioButton("1.0 φ"); rb2 = QRadioButton("0.5 φ")
        rb1.setChecked(True)
        layout.addWidget(rb1); layout.addWidget(rb2)
        btn = QPushButton(tr("ok")); btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        if not dlg.exec_():
            return
        step = 1.0 if rb1.isChecked() else 0.5
        self.laser_step = step

        # Mapear columnas por nombre (ES/EN, orden libre)
        try:
            df_laser_raw = self._map_laser_columns_by_name(df)
        except ValueError as e:
            QMessageBox.warning(self, tr("error"), str(e))
            return

        # Validar Group no vacío
        if df_laser_raw["Group"].isnull().any() or (df_laser_raw["Group"].astype(str).str.strip() == "").any():
            msg = "Falta al menos un valor en la columna de grupo/facies (Group/Grupo/Filtro) del archivo de láser." \
                  if _current_lang()=="es" else \
                  "At least one value is missing in the group/facies column (Group) of the laser file."
            QMessageBox.warning(self, tr("error"), msg)
            return

        # Agrupar por φ usando helper global (definido en Bloque 3)
        try:
            datos = agrupar_laser_por_phi(df_laser_raw, step)
        except Exception as e:
            QMessageBox.warning(self, tr("error"), f"{'Ocurrió un error al agrupar por φ' if _current_lang()=='es' else 'An error occurred while binning by φ'}:\n{e}")
            return

        # DF láser canónico
        self.df_laser = pd.DataFrame(datos)  # columnas: Sample, phi, Wt

        # Mapear grupos por muestra (robusto con duplicados)
        group_map = (
            df_laser_raw[["Sample", "Group"]]
            .astype({"Sample": str})
            .dropna()
            .drop_duplicates(subset="Sample", keep="first")
            .set_index("Sample")["Group"]
            .to_dict()
        )
        self.df_laser["Group"] = self.df_laser["Sample"].map(group_map).fillna("Sin Grupo")

        # Orden y estado UI
        self.df_laser.sort_values(["Sample", "phi"], inplace=True)
        self.df_laser.reset_index(drop=True, inplace=True)
        # reemplazo: habilita menús de DB según disponibilidad
        self._update_db_menu_actions()

        # Calcular parámetros de láser
        params_l = []
        for sample, grp in self.df_laser.groupby("Sample"):
            phi_vals = grp["phi"].tolist()
            wt_vals  = grp["Wt"].tolist()
            gval     = grp["Group"].iloc[0]
            p        = calculate_parameters(phi_vals, wt_vals, self.current_method)
            row      = {"Sample": sample, "Group": gval, **p}
            params_l.append(row)
        self.param_results_laser = pd.DataFrame(params_l)

        # Actualiza listas de grupos, colores y selecciones
        self._update_all_groups_and_colors()

        # Mostrar en consola (centralizado)
        self.update_console()

        # >>> NUEVO: si existe el canvas de Pardo, repintarlo también
        if hasattr(self, "canvas_pardo"):
            try:
                self.plot_pardo()
            except Exception:
                pass

        # Refrescar Histograma
        self._update_hist_samples()

    # ----------------------- Ventanas de vista de bases (vista Formato 2) -----------------------
    def _make_format2_view(self, df_long, *, phi_col, sample_col, weight_col, group_col="Group", group_row_label="Grupo"):
        """
        Convierte un DataFrame en formato largo (φ, Sample, Weight, Group) a una vista tipo Formato 2:
          - Columna 0 = 'Phi'
          - Fila 0 = 'Grupo' (una celda por muestra con su grupo/facies)
          - Columnas 1..n = una por muestra con los Wt por φ
        Solo afecta la VISTA (no cambia self.df_*). No usa ningún 'Datox'.
        """
        # Asegurar nombres de muestra como string y respetar orden de aparición
        s_sample = df_long[sample_col].astype(str)
        first_order = list(dict.fromkeys(s_sample.tolist()))

        # Pivot de Wt (índice = φ, columnas = Sample)
        wide = (
            df_long
            .pivot_table(index=phi_col, columns=s_sample, values=weight_col, aggfunc="first")
            .sort_index()
        )
        wide = wide.reindex(columns=first_order)

        # Fila superior con el grupo/facies de cada muestra (primer no-nulo)
        grp_map = (
            df_long
            .groupby(s_sample)[group_col]
            .apply(lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna()) else "")
        )
        header_row = pd.DataFrame([grp_map], index=[group_row_label])  # primera fila = "Grupo" / "Group"

        # Apilar: primera fila = grupos; luego las filas de φ
        out = pd.concat([header_row, wide])

        # Primera columna "Phi": primer rótulo = 'Grupo'/'Group', luego los φ
        first_label = "Grupo" if _current_lang()=="es" else "Group"
        out.insert(0, "Phi", [first_label] + list(wide.index))

        # Reset para mostrar lindo en QTableView
        return out.reset_index(drop=True)

    def show_tamiz_db_window(self):
        """
        Muestra la base TAMIZ en vista tipo Formato 2:
        Columna 1 = Phi ; primera fila = Grupo ; columnas restantes = muestras con Wt.
        """
        if self.df_data is None:
            return

        df_vista = self._make_format2_view(
            self.df_data,
            phi_col=self.df_data.columns[0],      # φ
            sample_col=self.df_data.columns[1],   # Sample
            weight_col=self.df_data.columns[2],   # Weight
            group_col=self.group_col,             # Group (nombre de la 4ta col cargada)
            group_row_label="Grupo" if _current_lang()=="es" else "Group"
        )
        title = "Base de datos de tamiz (vista Formato 2)" if _current_lang()=="es" else "Sieve database (Format 2 view)"
        dlg = DataBaseWindow(title, df_vista, self)
        dlg.exec_()

    def show_laser_db_window(self):
        """
        Muestra la base LÁSER en vista tipo Formato 2 (mismo criterio).
        """
        if self.df_laser is None:
            return

        df_vista = self._make_format2_view(
            self.df_laser,
            phi_col="phi",
            sample_col="Sample",
            weight_col="Wt",
            group_col="Group",
            group_row_label="Grupo" if _current_lang()=="es" else "Group"
        )
        title = "Base de datos láser (vista Formato 2)" if _current_lang()=="es" else "Laser database (Format 2 view)"
        dlg = DataBaseWindow(title, df_vista, self)
        dlg.exec_()

    def show_hybrid_db_window(self):
        """
        Muestra la base HÍBRIDA en vista tipo Formato 2 usando 'Tamiz Sample' como nombre de muestra.
        """
        if self.df_hybrid is None or self.df_hybrid.empty:
            QMessageBox.information(
                self, tr("info"),
                "Todavía no has combinado tamiz + láser." if _current_lang()=="es" else "You haven't combined sieve + laser yet."
            )
            return

        # Detectar nombre de columna de grupo disponible en df_hybrid
        gcol = self.group_col if (hasattr(self, "group_col") and self.group_col in self.df_hybrid.columns) else "Group"

        df_vista = self._make_format2_view(
            self.df_hybrid,
            phi_col="phi",
            sample_col="Tamiz Sample",
            weight_col="wt%",
            group_col=gcol,
            group_row_label="Grupo" if _current_lang()=="es" else "Group"
        )
        title = "Base de datos tamiz + láser (vista Formato 2)" if _current_lang()=="es" else "Sieve + laser database (Format 2 view)"
        dlg = DataBaseWindow(title, df_vista, self)
        dlg.exec_()
# endregion





# region # === Bloque 10: Consola bilingüe, gráficos, superposición/residuo, IA (GMM/Weibull) y combinar bases ===

# --- util interno (nuevo): defaults y scroll/canvas fijo para Histograma ---
def _ensure_hist_defaults(self):
    # tamaños de fuente por defecto
    if not hasattr(self, "hist_title_size"): self.hist_title_size = 13
    if not hasattr(self, "hist_label_size"): self.hist_label_size = 11
    if not hasattr(self, "hist_tick_size"):  self.hist_tick_size  = 9
    # tamaño físico del gráfico en píxeles (fijo, independiente de la ventana)
    if not hasattr(self, "hist_px_h"): self.hist_px_h = 700
    if not hasattr(self, "hist_px_w"): self.hist_px_w = int(round(self.hist_px_h * 1.30))

def _ensure_hist_scroller_and_fixed_canvas(self):
    """
    - Si el canvas no está dentro de un QScrollArea, lo envuelve.
    - Fija el canvas a tamaño en píxeles (hist_px_w / hist_px_h).
    - Afina la altura de la tira de opciones (scroll fino).
    """
    from PyQt5.QtWidgets import QSizePolicy, QScrollArea

    # Asegurar scroll finito en la fila de opciones si existe
    try:
        if hasattr(self, "sc_hist_opts") and self.sc_hist_opts:
            self.sc_hist_opts.setFixedHeight(56)  # antes era muy alto
    except Exception:
        pass

    # Si aún no hay scroller para el gráfico, crear y envolver
    if not hasattr(self, "sc_hist") or self.sc_hist is None:
        try:
            v_hist = self.tab_hist.layout()
            # si el canvas estaba directo en el layout, removerlo y envolverlo
            try:
                v_hist.removeWidget(self.canvas_hist)
            except Exception:
                pass
            sc = QScrollArea(self.tab_hist)
            sc.setWidgetResizable(False)
            # alineá al centro; si preferís arriba-izq: Qt.AlignLeft | Qt.AlignTop
            sc.setAlignment(Qt.AlignCenter)
            sc.setFrameShape(sc.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            sc.setObjectName("sc_hist")
            sc.setStyleSheet("#sc_hist { background: transparent; }")
            sc.setWidget(self.canvas_hist)
            v_hist.addWidget(sc, 1)
            self.sc_hist = sc
        except Exception:
            # si algo falla, seguimos igual (canvas sin scroll), pero no debería.
            self.sc_hist = None

    # Fijar tamaño en píxeles del canvas (independiente de la ventana)
    try:
        dpi = int(self.canvas_hist.figure.get_dpi() or 100)
        self.canvas_hist.figure.set_size_inches(self.hist_px_w / dpi, self.hist_px_h / dpi, forward=True)
        self.canvas_hist.setFixedSize(self.hist_px_w, self.hist_px_h)
        self.canvas_hist.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    except Exception:
        pass

def _apply_hist_canvas_size(self):
    _ensure_hist_defaults(self)
    _ensure_hist_scroller_and_fixed_canvas(self)


# --- 10.0: Aplicar traducciones en caliente ---
def apply_translations(self):
    # Título de la ventana
    self.setWindowTitle(tr("app_title"))

    # Menús y acciones (creadas en initMenu)
    try:
        mb = self.menuBar()
        # Orden: Archivo, Configuración, Base de datos, Ayuda
        mb.actions()[0].setText(tr("menu_file"))
        mb.actions()[1].setText(tr("menu_config"))
        mb.actions()[2].setText(tr("menu_db"))
        mb.actions()[3].setText(tr("menu_help"))

        # Acciones de Archivo
        self.menuBar().actions()[0].menu().actions()[0].setText(tr("file_load_sieve"))
        self.menuBar().actions()[0].menu().actions()[1].setText(tr("file_load_laser"))
        self.menuBar().actions()[0].menu().actions()[2].setText(tr("file_export_image"))
        self.menuBar().actions()[0].menu().actions()[3].setText(tr("file_combine"))

        # Acciones de Configuración
        cfg_menu = self.menuBar().actions()[1].menu().actions()
        cfg_menu[0].setText(tr("cfg_method"))
        cfg_menu[1].setText(tr("cfg_colors"))
        cfg_menu[2].setText(tr("cfg_groups_all"))
        cfg_menu[3].setText(tr("cfg_ribbons"))
        cfg_menu[4].setText(tr("cfg_theme"))
        # Submenú idioma
        cfg_menu[5].setText(tr("cfg_language"))
        self.act_lang_es.setText(tr("cfg_lang_es"))
        self.act_lang_en.setText(tr("cfg_lang_en"))

        # Acciones de Base de datos
        db_menu = self.menuBar().actions()[2].menu().actions()
        db_menu[0].setText(tr("db_view_sieve"))
        db_menu[1].setText(tr("db_view_laser"))
        db_menu[2].setText(tr("db_view_hybrid"))

        # Ayuda
        help_menu = self.menuBar().actions()[3].menu().actions()
        help_menu[0].setText(tr("help_about"))
    except Exception:
        pass

    # Pestañas
    try:
        for i in range(self.tabs.count()):
            if i == 0: self.tabs.setTabText(i, tr("tab_xy"))
            if i == 1: self.tabs.setTabText(i, tr("tab_walker"))
            if i == 2: self.tabs.setTabText(i, tr("tab_gk"))
            if i == 3 and self.tabs.count() >= 5: self.tabs.setTabText(i, tr("tab_pardo"))
            if (i == 4 and self.tabs.count() >= 5) or (i == 3 and self.tabs.count() == 4):
                self.tabs.setTabText(i, tr("tab_hist"))
    except Exception:
        pass

    # Controles de pestaña XY
    try:
        self.btn_load_tamiz.setText(tr("btn_load_sieve"))
        self.lbl_x.setText(tr("lbl_x"))
        self.lbl_y.setText(tr("lbl_y"))
        self.btn_plot_xy.setText(tr("btn_plot_xy"))
        self.btn_select_db.setText(tr("btn_choose_db"))
        self.chk_show_names.setText(tr("chk_show_names"))
        self.btn_filter_groups.setText(tr("btn_filter_groups"))
        self.btn_export_xy.setText(tr("btn_export_image"))
        self.lbl_console.setText(tr("console_header"))
        self.btn_export_params.setText(tr("btn_export_params"))
    except Exception:
        pass

    # Pestañas Walker/GK/Pardo: refrescar radios
    try:
        self._refresh_walker_radio_texts()
        self._refresh_gk_radio_texts()
        if hasattr(self, "pardo_buttons"):
            self._refresh_pardo_radio_texts()
    except Exception:
        pass

    # Pestaña Histograma
    try:
        self.lbl_hist_width.setText(tr("lbl_hist_width"))
        self.btn_hist_colors.setText(tr("btn_hist_colors"))
        # (Se elimina el renombrado manual de btn_labels; ahora viene de i18n)
        self.chk_hist.setText(tr("chk_hist_bars"))
        self.chk_poly.setText(tr("chk_hist_poly"))
        self.chk_cum.setText(tr("chk_hist_cum"))
        self.btn_export_hist.setText(tr("btn_export_hist"))

        # Selector de método + "play" + exportación
        if hasattr(self, "lbl_hist_method"):
            self.lbl_hist_method.setText(tr("lbl_hist_method"))
        if hasattr(self, "cmb_hist_method"):
            cur = self.cmb_hist_method.currentText().lower() if self.cmb_hist_method.count() else ""
            self.cmb_hist_method.blockSignals(True)
            self.cmb_hist_method.clear()
            self.cmb_hist_method.addItems([tr("hist_method_gmm"), tr("hist_method_weibull")])
            self.cmb_hist_method.setCurrentIndex(1 if "weibull" in cur else 0)
            self.cmb_hist_method.blockSignals(False)
        if hasattr(self, "chk_gmm"):
            self.chk_gmm.setText(tr("chk_gmm"))
        if hasattr(self, "lbl_modes_nk"):
            self.lbl_modes_nk.setText(tr("lbl_modes_n") if getattr(self, "hist_mode_method", "gmm") == "weibull" else tr("lbl_modes_k"))
        if hasattr(self, "btn_export_modes"):
            self.btn_export_modes.setText(tr("btn_export_modes"))

        self.lbl_hist_base.setText(tr("lbl_hist_base"))
        self.lbl_hist_sample.setText(tr("lbl_hist_sample"))
        if hasattr(self, "chk_overlay_residue"):
            self.chk_overlay_residue.setText(tr("hist_overlay_toggle"))
        if hasattr(self, "chk_show_sieve_original"):
            self.chk_show_sieve_original.setText(tr("hist_show_original_sieve"))
            try:
                self.chk_show_sieve_original.setToolTip(tr("hist_show_original_sieve_tip"))
            except Exception:
                pass

        if not getattr(self, "hist_title", None):
            self.hist_title = tr("hist_title_def")
        if not getattr(self, "hist_xlabel", None):
            self.hist_xlabel = tr("hist_xlabel_def")
        if not getattr(self, "hist_ylabel", None):
            self.hist_ylabel = tr("hist_ylabel_def")
        if not getattr(self, "hist_ylabel2", None):
            self.hist_ylabel2 = tr("hist_ylabel2_def")
    except Exception:
        pass

    # Re-pintar
    try:
        self.update_console()
        self.plot_xy()
        self.plot_walker()
        self.plot_gk()
        if hasattr(self, "canvas_pardo"):
            self.plot_pardo()
    except Exception:
        pass


def _refresh_walker_radio_texts(self):
    _es = (_current_lang() == "es")
    walker_label_map_es = {
        "All": "Todas",
        "Fall deposit": "Depósito de caída",
        "Fine-depleted flow": "Flujo empobrecido en finos",
        "Surge (Walker) and Surge-Dunes (Gençalioğlu-Kuşcu)": "Surge (Walker) y dunas de surge (Gençalioğlu-Kuşcu)",
        "Pyroclastic flow": "Flujo piroclástico",
    }
    for rb in self.walker_buttons.buttons():
        key = rb.property("imgkey")
        rb.setText(walker_label_map_es.get(key, key) if _es else key)


def _refresh_gk_radio_texts(self):
    _es = (_current_lang() == "es")
    gk_label_map_es = {
        "All": "Todas",
        "Flow": "Flujo",
        "Surge (Walker) and Surge-Dunes (Gençalioğlu-Kuşcu)": "Surge (Walker) y dunas de surge (Gençalioğlu-Kuşcu)",
        "Fall": "Caída",
        "Surge Dunes": "Dunas de surge",
    }
    for rb in self.gk_buttons.buttons():
        key = rb.property("imgkey")
        rb.setText(gk_label_map_es.get(key, key) if _es else key)


def _refresh_pardo_radio_texts(self):
    _es = (_current_lang() == "es")
    pardo_label_map_es = {
        "All": "Todas",
        "Fallout": "Depósito de caída",
        "PF": "Flujo piroclástico (PF)",
        "PS": "Surge piroclástico (PS)",
        "Zone 1": "Zona 1",
        "Zone 2": "Zona 2",
        "Zone 3": "Zona 3",
    }
    for rb in self.pardo_buttons.buttons():
        key = rb.property("imgkey")
        rb.setText(pardo_label_map_es.get(key, key) if _es else key)


# --- 10.1: Consola bilingüe ---
def update_console(self):
    def _section_title(db_key):
        if _current_lang() == "es":
            return f"Parámetros granulométricos {base_name_display(db_key)} ({self.current_method}):\n"
        else:
            return f"Granulometric parameters {base_name_display(db_key)} ({self.current_method}):\n"

    def _entry_lines(row, group_col, db_key):
        lines = []
        base = f"{row[group_col]} ({base_name_display(db_key)})"
        lines.append(f"Sample: {row['Sample']} ({base})")
        for key in ["median", "sigma", "skewness", "kurtosis", "mean"]:
            label = param_label(key)
            val = row.get(key, np.nan)
            try:
                lines.append(f"  {label}: {val:.4f}")
            except Exception:
                lines.append(f"  {label}: {val}")
        lines.append("")
        return "\n".join(lines)

    txt = []
    if getattr(self, "param_results", None) is not None:
        db_key = "Tamiz"
        txt.append(_section_title(db_key))
        for _, r in self.param_results.iterrows():
            txt.append(_entry_lines(r, self.group_col, db_key))
    if getattr(self, "param_results_laser", None) is not None:
        db_key = "Laser"
        txt.append(_section_title(db_key))
        for _, r in self.param_results_laser.iterrows():
            txt.append(_entry_lines(r, "Group", db_key))
    if getattr(self, "param_results_hybrid", None) is not None:
        db_key = "Híbrido"
        txt.append(_section_title(db_key))
        for _, r in self.param_results_hybrid.iterrows():
            txt.append(_entry_lines(r, self.group_col, db_key))

    self.txt_console.setPlainText("\n".join(txt))


# Utilidad: rótulo de eje para 'sigma' con Trask
def _axis_label_for_key(self, key: str) -> str:
    if key == "sigma" and self.current_method == "trask" and TRASK_DISPLAY == "raw":
        return "So (Trask)"
    return param_label(key)


# --- 10.2: XY ---
def plot_xy(self):
    x = self.cmb_x.currentText()
    y = self.cmb_y.currentText()
    fig = self.canvas_xy.figure
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    # Bases cargadas
    bases = []
    if self.current_db_selection.get("tamiz") and getattr(self, "param_results", None) is not None:
        bases.append(("Tamiz", self.param_results, self.group_col))
    if self.current_db_selection.get("laser") and getattr(self, "param_results_laser", None) is not None:
        bases.append(("Laser", self.param_results_laser, "Group"))
    if self.current_db_selection.get("hybrid") and getattr(self, "param_results_hybrid", None) is not None:
        bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

    for name, df_params, group_col in bases:
        for grp in df_params[group_col].unique():
            clave_disp = f"{grp} ({base_name_display(name)})"
            old_key    = f"{grp} ({name})"  # compat con claves viejas
            if clave_disp not in self.selected_groups_xy and old_key not in self.selected_groups_xy:
                continue
            sub = df_params[df_params[group_col] == grp]
            col = self.group_colors.get(old_key, self.group_colors.get(clave_disp, "#888888"))
            if (old_key in self.selected_groups_ribbons or clave_disp in self.selected_groups_ribbons) and len(sub) > 2:
                try:
                    plot_ribbon(ax, sub[x], sub[y], color=col, alpha=0.18, width=0.07, smooth=10)
                except Exception:
                    pass
            ax.scatter(sub[x], sub[y], color=col, label=clave_disp, s=40, edgecolor='k', lw=1.0, zorder=3)
            if self.chk_show_names.isChecked():
                for _, row in sub.iterrows():
                    ax.annotate(
                        str(row['Sample']),
                        (row[x], row[y]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8, alpha=0.75, color=col
                    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    ax.tick_params(length=5, width=1.0, direction='out', colors='black', labelsize=9)
    ax.set_xlabel(self._axis_label_for_key(x), fontsize=12)
    ax.set_ylabel(self._axis_label_for_key(y), fontsize=12)
    ax.set_title(f"{self._axis_label_for_key(y)} vs {self._axis_label_for_key(x)}", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    have_legend = len(handles) > 0
    if have_legend:
        legend = ax.legend(
            handles, labels,
            title=legend_group_base(), frameon=False,
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0., fontsize=9, title_fontsize=9
        )
        for t in legend.get_texts():
            t.set_color('black')
        legend.get_title().set_color('black')

    fig.subplots_adjust(left=0.10, right=(0.74 if have_legend else 0.98), bottom=0.14, top=0.95)
    self.canvas_xy.draw()


# --- 10.3: Walker ---
def plot_walker(self):
    x = "median"; y = "sigma"
    fig = self.canvas_walker.figure
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    btn = self.walker_buttons.checkedButton()
    imgkey = btn.property("imgkey") if btn is not None else "All"
    img = mpimg.imread(os.path.join(get_script_dir(), WALKER_IMAGES.get(imgkey, WALKER_IMAGES["All"])))
    ax.imshow(img, extent=[-4, 6, 0, 5], aspect='auto', zorder=1)

    bases = []
    if self.current_db_selection.get("tamiz") and getattr(self, "param_results", None) is not None:
        bases.append(("Tamiz", self.param_results, self.group_col))
    if self.current_db_selection.get("laser") and getattr(self, "param_results_laser", None) is not None:
        bases.append(("Laser", self.param_results_laser, "Group"))
    if self.current_db_selection.get("hybrid") and getattr(self, "param_results_hybrid", None) is not None:
        bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

    for name, df_params, group_col in bases:
        for grp in df_params[group_col].unique():
            clave_disp = f"{grp} ({base_name_display(name)})"
            old_key    = f"{grp} ({name})"
            if clave_disp not in self.selected_groups_walker and old_key not in self.selected_groups_walker:
                continue
            sub = df_params[df_params[group_col] == grp]
            col = self.group_colors.get(old_key, self.group_colors.get(clave_disp, "#888888"))
            ax.scatter(sub[x], sub[y], color=col, label=clave_disp, s=40, edgecolor='k', lw=1.0, zorder=10)

    ax.set_xlim(-4, 6)
    ax.set_ylim(0, 5)
    ax.set_xlabel(param_label("median"), fontsize=12)
    ax.set_ylabel(self._axis_label_for_key("sigma"), fontsize=12)
    ax.set_xticks(np.arange(-4, 7, 2))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.tick_params(length=6, width=1.1, direction='out', colors='black', labelsize=9)
    for x_ in np.arange(-4, 7, 2):
        ax.plot([x_, x_], [0 - 0.12, 0], color='k', lw=1.0, zorder=30)
    for y_ in np.arange(0, 6, 1):
        ax.plot([-4 - 0.12, -4], [y_, y_], color='k', lw=1.0, zorder=30)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    handles, labels = ax.get_legend_handles_labels()
    have_legend = len(handles) > 0
    if have_legend:
        legend = ax.legend(
            handles, labels,
            title=legend_group_base(), frameon=False,
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0., fontsize=9, title_fontsize=9
        )
        for t in legend.get_texts():
            t.set_color('black')
        legend.get_title().set_color('black')

    fig.subplots_adjust(left=0.10, right=(0.74 if have_legend else 0.98), bottom=0.20, top=0.97)
    self.canvas_walker.draw()


# --- 10.4: Gençalioğlu-Kuşcu (2007) ---
def plot_gk(self):
    x = "median"; y = "sigma"
    fig = self.canvas_gk.figure
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    btn = self.gk_buttons.checkedButton()
    imgkey = btn.property("imgkey") if btn is not None else "All"
    img = mpimg.imread(os.path.join(get_script_dir(), GK_IMAGES.get(imgkey, GK_IMAGES["All"])))
    ax.imshow(img, extent=[-4, 6, 0, 5], aspect='auto', zorder=1)

    bases = []
    if self.current_db_selection.get("tamiz") and getattr(self, "param_results", None) is not None:
        bases.append(("Tamiz", self.param_results, self.group_col))
    if self.current_db_selection.get("laser") and getattr(self, "param_results_laser", None) is not None:
        bases.append(("Laser", self.param_results_laser, "Group"))
    if self.current_db_selection.get("hybrid") and getattr(self, "param_results_hybrid", None) is not None:
        bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

    for name, df_params, group_col in bases:
        for grp in df_params[group_col].unique():
            clave_disp = f"{grp} ({base_name_display(name)})"
            old_key    = f"{grp} ({name})"
            if clave_disp not in self.selected_groups_gk and old_key not in self.selected_groups_gk:
                continue
            sub = df_params[df_params[group_col] == grp]
            col = self.group_colors.get(old_key, self.group_colors.get(clave_disp, "#888888"))
            ax.scatter(sub[x], sub[y], color=col, label=clave_disp, s=40, edgecolor='k', lw=1.0, zorder=10)

    ax.set_xlim(-4, 6)
    ax.set_ylim(0, 5)
    ax.set_xlabel(param_label("median"), fontsize=12)
    ax.set_ylabel(self._axis_label_for_key("sigma"), fontsize=12)
    ax.set_xticks(np.arange(-4, 7, 2))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.tick_params(length=6, width=1.1, direction='out', colors='black', labelsize=9)
    for x_ in np.arange(-4, 7, 2):
        ax.plot([x_, x_], [0 - 0.12, 0], color='k', lw=1.0, zorder=30)
    for y_ in np.arange(0, 6, 1):
        ax.plot([-4 - 0.12, -4], [y_, y_], color='k', lw=1.0, zorder=30)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    handles, labels = ax.get_legend_handles_labels()
    have_legend = len(handles) > 0
    if have_legend:
        legend = ax.legend(
            handles, labels,
            title=legend_group_base(), frameon=False,
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0., fontsize=9, title_fontsize=9
        )
        for t in legend.get_texts():
            t.set_color('black')
        legend.get_title().set_color('black')

    fig.subplots_adjust(left=0.10, right=(0.74 if have_legend else 0.98), bottom=0.20, top=0.97)
    self.canvas_gk.draw()


# --- 10.4 bis: Pardo et al. (2009) — NUEVO ---
def plot_pardo(self):
    x = "median"; y = "sigma"
    fig = self.canvas_pardo.figure
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")

    btn = self.pardo_buttons.checkedButton() if hasattr(self, "pardo_buttons") else None
    if btn is None and hasattr(self, "pardo_buttons"):
        try:
            all_btn = next((b for b in self.pardo_buttons.buttons() if b.property("imgkey") == "All"), None)
            if all_btn is not None:
                all_btn.setChecked(True); btn = all_btn
        except Exception:
            pass
    imgkey = btn.property("imgkey") if btn is not None else "All"
    img = mpimg.imread(os.path.join(get_script_dir(), PARDO_IMAGES.get(imgkey, PARDO_IMAGES["All"])))
    ax.imshow(img, extent=[-5, 5, 0, 5], aspect='auto', zorder=1)

    bases = []
    if self.current_db_selection.get("tamiz") and getattr(self, "param_results", None) is not None:
        bases.append(("Tamiz", self.param_results, self.group_col))
    if self.current_db_selection.get("laser") and getattr(self, "param_results_laser", None) is not None:
        bases.append(("Laser", self.param_results_laser, "Group"))
    if self.current_db_selection.get("hybrid") and getattr(self, "param_results_hybrid", None) is not None:
        bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

    sel = getattr(self, "selected_groups_pardo", None)
    if not sel:
        sel = self.groups or []
        self.selected_groups_pardo = sel.copy() if isinstance(sel, list) else list(sel)

    for name, df_params, group_col in bases:
        for grp in df_params[group_col].unique():
            clave_disp = f"{grp} ({base_name_display(name)})"
            old_key    = f"{grp} ({name})"
            if clave_disp not in sel and old_key not in sel:
                continue
            sub = df_params[df_params[group_col] == grp]
            col = self.group_colors.get(old_key, self.group_colors.get(clave_disp, "#888888"))
            ax.scatter(sub[x], sub[y], color=col, label=clave_disp, s=40, edgecolor='k', lw=1.0, zorder=10)

    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel(param_label("median"), fontsize=12)
    ax.set_ylabel(self._axis_label_for_key("sigma"), fontsize=12)
    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))

    ax.tick_params(length=6, width=1.1, direction='out', colors='black', labelsize=9)
    for x_ in np.arange(-5, 6, 1):
        ax.plot([x_, x_], [0 - 0.12, 0], color='k', lw=1.0, zorder=30)
    for y_ in np.arange(0, 6, 1):
        ax.plot([-5 - 0.12, -5], [y_, y_], color='k', lw=1.0, zorder=30)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles, labels,
            title=legend_group_base(), frameon=False,
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0., fontsize=9, title_fontsize=9
        )
        for t in legend.get_texts():
            t.set_color('black')
        legend.get_title().set_color('black')

    fig.subplots_adjust(left=0.10, right=(0.74 if handles else 0.98), bottom=0.20, top=0.97)
    self.canvas_pardo.draw()


# --- 10.5: Configuración de colores y visibilidad de grupos ---
def edit_colors(self):
    if not self.groups:
        return
    dlg = ColorDialog(self.groups, self.group_colors, self)
    if dlg.exec_():
        self.group_colors = dlg.colors
        self.plot_xy()
        self.plot_walker()
        self.plot_gk()
        if hasattr(self, "canvas_pardo"):
            self.plot_pardo()


def select_ribbons(self):
    if not self.groups:
        return
    title = "Ver/ocultar sombras (XY)" if _current_lang()=="es" else "Show/hide ribbons (XY)"
    dlg = GroupSelectDialog(self.groups, self.selected_groups_ribbons, title, self)
    if dlg.exec_():
        self.selected_groups_ribbons = dlg.get_selected()
        self.plot_xy()


def select_groups_all(self):
    if not self.groups:
        return
    seed = self.selected_groups_xy if self.selected_groups_xy else self.groups
    title = tr("cfg_groups_all")
    dlg = GroupSelectDialog(self.groups, seed, title, self)
    if dlg.exec_():
        sel = dlg.get_selected()
        self.selected_groups_xy = sel.copy()
        self.selected_groups_walker = sel.copy()
        self.selected_groups_gk = sel.copy()
        if hasattr(self, "selected_groups_pardo"):
            self.selected_groups_pardo = sel.copy()
        self.plot_xy(); self.plot_walker(); self.plot_gk()
        if hasattr(self, "canvas_pardo"):
            self.plot_pardo()


def select_groups_walker(self):
    if not self.groups:
        return
    title = tr("btn_group_walker")
    dlg = GroupSelectDialog(self.groups, self.selected_groups_walker, title, self)
    if dlg.exec_():
        self.selected_groups_walker = dlg.get_selected()
        self.plot_walker()


def select_groups_gk(self):
    if not self.groups:
        return
    title = tr("btn_group_gk")
    dlg = GroupSelectDialog(self.groups, self.selected_groups_gk, title, self)
    if dlg.exec_():
        self.selected_groups_gk = dlg.get_selected()
        self.plot_gk()


def select_groups_pardo(self):
    if not self.groups:
        return
    title = tr("btn_group_pardo") if "btn_group_pardo" in tr.__code__.co_consts else "Grupos: Pardo et al. 2009"
    dlg = GroupSelectDialog(self.groups, self.selected_groups_pardo, title, self)
    if dlg.exec_():
        self.selected_groups_pardo = dlg.get_selected()
        self.plot_pardo()


# --- 10.5.1: Elegir bases a graficar ---
def select_db_to_plot(self):
    dlg = QDialog(self)
    dlg.setWindowTitle(tr("btn_choose_db"))
    layout = QVBoxLayout(dlg)

    items = []
    if self.param_results is not None and not self.param_results.empty:
        items.append((base_name_display("Tamiz"),  "tamiz",  self.current_db_selection.get("tamiz", True)))
    if self.param_results_laser is not None and not self.param_results_laser.empty:
        items.append((base_name_display("Laser"),  "laser",  self.current_db_selection.get("laser", False)))
    if self.param_results_hybrid is not None and not self.param_results_hybrid.empty:
        items.append((base_name_display("Híbrido"), "hybrid", self.current_db_selection.get("hybrid", False)))

    if not items:
        QMessageBox.information(self, tr("info"),
            "No hay bases cargadas para graficar." if _current_lang()=="es" else "There are no loaded databases to plot.")
        return

    cbs = {}
    for label, key, checked in items:
        cb = QCheckBox(label); cb.setChecked(checked)
        layout.addWidget(cb); cbs[key] = cb

    btn = QPushButton(tr("ok")); btn.clicked.connect(dlg.accept)
    layout.addWidget(btn)

    if dlg.exec_():
        self.current_db_selection = {"tamiz": False, "laser": False, "hybrid": False}
        for key, cb in cbs.items():
            self.current_db_selection[key] = cb.isChecked()
        self.plot_xy(); self.plot_walker(); self.plot_gk()
        if hasattr(self, "canvas_pardo"):
            self.plot_pardo()


# --- 10.5.2: Filtro de grupos para XY ---
def select_groups_xy(self):
    if not self.groups:
        return
    title = tr("btn_filter_groups")
    dlg = GroupSelectDialog(self.groups, self.selected_groups_xy, title, self)
    if dlg.exec_():
        self.selected_groups_xy = dlg.get_selected()
        self.plot_xy()


# --- 10.6: Lógica de Histograma ---
def _update_hist_bases(self):
    bases = []
    if self.df_data   is not None: bases.append("Tamiz")
    if self.df_laser  is not None: bases.append("Laser")
    if self.df_hybrid is not None: bases.append("Híbrido")

    current = self.cmb_hist_base.currentText()
    self.cmb_hist_base.blockSignals(True)
    self.cmb_hist_base.clear()
    self.cmb_hist_base.addItems([base_name_display(b) for b in bases])
    if current and current in [base_name_display(b) for b in bases]:
        self.cmb_hist_base.setCurrentText(current)
    self.cmb_hist_base.blockSignals(False)


def _fill_hist_samples_for_base(self, base_display: str):
    def _to_key(disp):
        for k in ("Tamiz","Laser","Híbrido"):
            if base_name_display(k) == disp:
                return k
        return disp

    base = _to_key(base_display)

    self.cmb_hist_sample.blockSignals(True)
    self.cmb_hist_sample.clear()

    if not base:
        self.cmb_hist_sample.blockSignals(False)
        return

    if base == "Tamiz":
        df, col = self.df_data, (self.df_data.columns[1] if self.df_data is not None else None)
    elif base == "Laser":
        df, col = self.df_laser, "Sample"
    else:
        df, col = self.df_hybrid, "Tamiz Sample"

    samples = []
    if df is not None and col is not None and col in df.columns:
        samples = [str(s) for s in sorted(df[col].unique())]

    self.cmb_hist_sample.addItems(samples)
    if samples:
        self.cmb_hist_sample.setCurrentIndex(0)
    self.cmb_hist_sample.blockSignals(False)


def _update_hist_samples(self):
    self._update_hist_bases()
    self._on_hist_base_changed()


def _on_hist_base_changed(self, *_):
    base_disp = self.cmb_hist_base.currentText()
    self._fill_hist_samples_for_base(base_disp)


# --- 10.6.1: superposición/residuo para híbridos ---
def _draw_hybrid_overlay(self, ax, sample_name: str, bar_width: float):
    if not hasattr(self, "_hybrid_meta"):  # seguridad
        return
    meta = self._hybrid_meta.get(sample_name)
    if not meta:
        return
    phi_cut = float(meta.get("phi_cut", np.nan))
    resid  = float(meta.get("laser_residue", 0.0))

    try:
        phi_col = self.df_data.columns[0]
        samp_col = self.df_data.columns[1]
        wt_col   = self.df_data.columns[2]
        sieve_sub = self.df_data[self.df_data[samp_col].astype(str) == str(sample_name)]
        sieve_sub = sieve_sub[np.isfinite(sieve_sub[phi_col])]
        mask_removed = sieve_sub[phi_col].astype(float) >= phi_cut
        sieve_removed = sieve_sub[mask_removed]
        if not sieve_removed.empty:
            xs = sieve_removed[phi_col].astype(float).values
            ys = sieve_removed[wt_col].astype(float).values
            ax.bar(
                xs, ys,
                width=bar_width,
                edgecolor="#6c757d",
                facecolor="none",
                hatch="///",
                linewidth=0.0,
                zorder=40,
                align="center"
            )
    except Exception:
        pass

    try:
        if resid > 0:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_text = xlim[0] + 0.70*(xlim[1]-xlim[0])
            y_text = ylim[0] + 0.12*(ylim[1]-ylim[0])
            msg = tr("hist_residual_label", x=f"{resid:.2f}")
            ax.text(
                x_text, y_text, msg,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#6c757d", alpha=0.9),
                zorder=50
            )
    except Exception:
        pass


# --- 10.6.2: sombra del tamiz original detrás del híbrido ---
def _overlay_original_sieve(self, ax, tamiz_sample: str, bar_width: float):
    try:
        if self.df_data is None:
            return
        phi_col = self.df_data.columns[0]
        samp_col = self.df_data.columns[1]
        wt_col   = self.df_data.columns[2]

        sub = self.df_data[self.df_data[samp_col].astype(str) == str(tamiz_sample)]
        if sub.empty:
            return

        xs = sub[phi_col].astype(float).values
        ys = sub[wt_col].astype(float).values

        ax.bar(
            xs, ys,
            width=bar_width,
            edgecolor='black',
            facecolor='black',
            alpha=0.18,
            zorder=5,
            align="center"
        )
    except Exception:
        pass


def plot_histogram(self):
    """Histograma con tamaño FIJO en píxeles (scroll propio) + IA opcional."""
    _apply_hist_canvas_size(self)  # asegura scroll + canvas fijo

    def _to_key(disp):
        for k in ("Tamiz","Laser","Híbrido"):
            if base_name_display(k) == disp:
                return k
        return disp

    base_display = self.cmb_hist_base.currentText()
    base   = _to_key(base_display)
    sample = self.cmb_hist_sample.currentText()

    # Selección de DF/columnas según base
    if   base == "Tamiz":
        df, phi_col, wt_col, samp_col = (
            self.df_data,
            self.df_data.columns[0],
            self.df_data.columns[2],
            self.df_data.columns[1]
        )
    elif base == "Laser":
        df, phi_col, wt_col, samp_col = (self.df_laser, "phi", "Wt", "Sample")
    else:
        df, phi_col, wt_col, samp_col = (self.df_hybrid, "phi", "wt%", "Tamiz Sample")

    if df is None or not sample or sample not in df[samp_col].astype(str).values:
        QMessageBox.warning(self, tr("error"),
            (f"No hay datos para la muestra {sample}.") if _current_lang()=="es" else (f"No data for sample {sample}."))
        return
    sub = df[df[samp_col].astype(str) == sample]
    if sub.empty:
        QMessageBox.warning(self, tr("error"),
            (f"No hay datos para la muestra {sample}.") if _current_lang()=="es" else (f"No data for sample {sample}."))
        return

    fig = self.canvas_hist.figure
    fig.clf()
    ax  = fig.add_subplot(111)

    x = np.array(sub[phi_col], dtype=float)   # φ
    y = np.array(sub[wt_col],   dtype=float)  # wt%

    xu = np.sort(np.unique(x))
    spacing = float(np.min(np.diff(xu))) if len(xu) > 1 else 1.0
    width = spacing * (self.spn_hist_width.value() / 100.0)

    # Sombras de tamiz original (híbrido)
    if base == "Híbrido" and hasattr(self, "chk_show_sieve_original") and self.chk_show_sieve_original.isChecked():
        try:
            self._overlay_original_sieve(ax, tamiz_sample=sample, bar_width=width)
        except Exception:
            pass

    if self.chk_hist.isChecked():
        ax.bar(
            x, y,
            width=width,
            color=self.hist_bar_fill,
            edgecolor=self.hist_bar_edge,
            linewidth=1.0,
            zorder=10,
            align="center"
        )

    if self.chk_poly.isChecked():
        ax.plot(x, y, "-o", color=self.poly_color, linewidth=1.5, zorder=20)

    ax2 = None
    if self.chk_cum.isChecked():
        ax2 = ax.twinx()
        y_ac = np.cumsum(y)
        y_ac = (100 * y_ac / y_ac[-1]) if y_ac[-1] != 0 else y_ac
        ax2.plot(x, y_ac, "-o", color=self.cum_color, linewidth=2.0, zorder=30)
        ax2.set_ylabel(self.hist_ylabel2 or tr("hist_ylabel2_def"), fontsize=self.hist_label_size)
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelsize=self.hist_tick_size)

    # Ticks φ con mismo paso que los datos
    start = np.floor(np.min(x) / spacing) * spacing
    end   = np.ceil (np.max(x) / spacing) * spacing
    n = int(round((end - start) / spacing)) + 1
    phi_ticks = np.round(start + np.arange(n) * spacing, 6)
    decimals = 0 if np.isclose(spacing, 1.0) else 1
    ax.set_xticks(phi_ticks)
    ax.set_xticklabels([f"{v:.{decimals}f}" if decimals else f"{int(round(v))}" for v in phi_ticks], fontsize=self.hist_tick_size)

    # Eje secundario inferior en µm
    def um(phi): return 1000 * (2 ** -float(phi))
    sec = ax.twiny()
    sec.set_xlim(ax.get_xlim())
    sec.set_xticks(phi_ticks)
    sec.set_xticklabels([str(int(round(um(v)))) for v in phi_ticks], fontsize=self.hist_tick_size)
    sec.xaxis.set_ticks_position("bottom")
    sec.xaxis.set_label_position("bottom")
    sec.spines["bottom"].set_position(("outward", 36))
    sec.set_xlabel(size_um_label(), fontsize=self.hist_label_size)
    sec.tick_params(axis='x', labelsize=self.hist_tick_size)

    xlabel = getattr(self, "hist_xlabel", tr("hist_xlabel_def"))
    ylabel = getattr(self, "hist_ylabel", tr("hist_ylabel_def"))
    title  = getattr(self, "hist_title",  sample if sample else tr("hist_title_def"))

    ax.set_xlabel(xlabel if xlabel else tr("hist_xlabel_def"), fontsize=self.hist_label_size)
    ax.set_ylabel(ylabel if ylabel else tr("hist_ylabel_def"), fontsize=self.hist_label_size)
    ax.set_title(title if title else (sample if sample else tr("hist_title_def")),
                 fontsize=self.hist_title_size, weight="bold", pad=8)
    ax.tick_params(axis='x', labelsize=self.hist_tick_size)
    ax.tick_params(axis='y', labelsize=self.hist_tick_size)

    # Superposición/Residuo (híbrido)
    if base == "Híbrido" and hasattr(self, "chk_overlay_residue") and self.chk_overlay_residue.isChecked():
        try:
            self._draw_hybrid_overlay(ax, sample_name=sample, bar_width=width)
        except Exception as e:
            print("Overlay error:", e)

    # IA (GMM / Weibull)
    if hasattr(self, "chk_gmm") and self.chk_gmm.isChecked():
        method = getattr(self, "hist_mode_method", "gmm")
        ncomp = int(self.spn_modes_nk.value()) if hasattr(self, "spn_modes_nk") else 4

        if method == "gmm":
            if not _HAS_SKLEARN:
                QMessageBox.information(
                    self, "IA/GMM",
                    "Instala 'scikit-learn' para usar la detección de modos."
                    if _current_lang()=="es" else
                    "Install 'scikit-learn' to use mode detection."
                )
            else:
                try:
                    if 'gmm_detect_modes' not in globals():
                        raise NameError("gmm_detect_modes not found (paste Block 14)")
                    res = gmm_detect_modes(x, y, kmax=ncomp, random_state=0)

                    xs = np.asarray(res.get("xs", []), dtype=float)
                    if xs.size:
                        ax.plot(xs, res["mix_curve"], linewidth=2.0, alpha=0.9, zorder=35)
                        comps = res.get("components", [])
                        for mu, comp in zip(res.get("means", []), comps):
                            ax.plot(xs, comp, linestyle="--", linewidth=1.4, alpha=0.9, zorder=34)
                            ax.axvline(mu, linestyle=":", linewidth=1.0, alpha=0.8)

                    w_show = np.array(res.get("assigned_weight_frac", res.get("weights", [])), dtype=float) * 100.0
                    modos_txt = ";  ".join([f"{mu:.2f}ϕ ({p:.0f}%)" for mu, p in zip(res.get("means", []), w_show)])
                    base_txt = (f"  |  Modos (k={res.get('k', ncomp)}): {modos_txt}"
                                if _current_lang()=="es"
                                else f"  |  Modes (k={res.get('k', ncomp)}): {modos_txt}")
                    ax.set_title((title if title else str(sample)) + base_txt,
                                 fontsize=self.hist_title_size, weight="bold", pad=8)

                    self._gmm_last = {
                        "base": base,
                        "sample": sample,
                        "k": int(res.get("k", 0)),
                        "means_phi": [float(v) for v in res.get("means", [])],
                        "std_phi":   [float(v) for v in res.get("stds",  [])],
                        "weights":   [float(v) for v in res.get("weights", [])],
                        "assigned_weight_frac": [float(v) for v in res.get("assigned_weight_frac", res.get("weights", []))],
                        "bic": float(res.get("bic", np.nan)),
                        "aic": float(res.get("aic", np.nan)) if "aic" in res else np.nan,
                        "loglik": float(res.get("loglik", np.nan)) if "loglik" in res else np.nan,
                        "rmse": float(res.get("rmse", np.nan)) if "rmse" in res else np.nan,
                        "r2": float(res.get("r2", np.nan)) if "r2" in res else np.nan,
                        "area_data": float(res.get("area_data", np.nan)) if "area_data" in res else np.nan,
                        "area_mix": float(res.get("area_mix", np.nan)) if "area_mix" in res else np.nan,
                        "peak_heights": [float(v) for v in res.get("peak_heights", [])],
                        "fwhm": [float(v) for v in res.get("fwhm", [])],
                        "n_modes": int(res.get("n_modes", res.get("k", 0))),
                    }
                    self._modes_last = {"method": "gmm", **self._gmm_last}
                    if hasattr(self, "btn_export_gmm"):
                        self.btn_export_gmm.setEnabled(True)
                except Exception as e:
                    QMessageBox.warning(
                        self, "IA/GMM",
                        f"Fallo al detectar modos: {e}" if _current_lang()=="es" else f"Mode detection failed: {e}"
                    )
        else:
            try:
                if 'weibull_mixture_detect' not in globals():
                    raise NameError("weibull_mixture_detect not found (paste Block 14)")

                res = weibull_mixture_detect(phi=x, wt=y, n_components=ncomp)

                xs_phi = np.asarray(res.get("xs_phi", []), dtype=float)
                if xs_phi.size:
                    ax.plot(xs_phi, res["mix_curve_phi"], linewidth=2.0, alpha=0.9, zorder=35)
                    comps = res.get("components_phi", [])
                    for mphi, comp in zip(res.get("modes_phi", []), comps):
                        ax.plot(xs_phi, comp, linestyle="--", linewidth=1.4, alpha=0.9, zorder=34)
                        ax.axvline(mphi, linestyle=":", linewidth=1.0, alpha=0.8)

                w_show = np.array(res.get("assigned_weight_frac", res.get("weights", [])), dtype=float) * 100.0
                modos_txt = ";  ".join([f"{phi:.2f}ϕ ({p:.0f}%)" for phi, p in zip(res.get("modes_phi", []), w_show)])
                base_txt = (f"  |  Subpoblaciones (n={res.get('k', ncomp)}): {modos_txt}"
                            if _current_lang()=="es"
                            else f"  |  Subpopulations (n={res.get('k', ncomp)}): {modos_txt}")
                ax.set_title((title if title else str(sample)) + base_txt,
                             fontsize=self.hist_title_size, weight="bold", pad=8)

                self._modes_last = {
                    "method": "weibull",
                    "base": base,
                    "sample": sample,
                    "k": int(res.get("k", ncomp)),
                    "modes_phi": [float(v) for v in res.get("modes_phi", [])],
                    "modes_um":  [float(v) for v in res.get("modes_um",  [])],
                    "sigma_phi": [float(v) for v in res.get("sigma_phi", [])],
                    "weights":   [float(v) for v in res.get("weights", [])],
                    "assigned_weight_frac": [float(v) for v in res.get("assigned_weight_frac", res.get("weights", []))],
                    "bic": float(res.get("bic", np.nan)) if "bic" in res else np.nan,
                    "aic": float(res.get("aic", np.nan)) if "aic" in res else np.nan,
                    "loglik": float(res.get("loglik", np.nan)) if "loglik" in res else np.nan,
                    "rmse": float(res.get("rmse", np.nan)) if "rmse" in res else np.nan,
                    "r2": float(res.get("r2", np.nan)) if "r2" in res else np.nan,
                    "area_data": float(res.get("area_data", np.nan)) if "area_data" in res else np.nan,
                    "area_mix": float(res.get("area_mix", np.nan)) if "area_mix" in res else np.nan,
                    "lambda_um": [float(v) for v in res.get("lambda_um", [])],
                    "k_shape":   [float(v) for v in res.get("k_shape",   [])],
                }
                self._gmm_last = None
                if hasattr(self, "btn_export_gmm"):
                    self.btn_export_gmm.setEnabled(True)
            except Exception as e:
                QMessageBox.warning(
                    self, "IA",
                    f"Fallo en mezcla Weibull: {e}" if _current_lang()=="es" else f"Weibull mixture failed: {e}"
                )
    else:
        if hasattr(self, "btn_export_gmm"):
            self.btn_export_gmm.setEnabled(False)
        self._gmm_last = None
        self._modes_last = None

    # margen derecho más generoso si hay eje derecho
    right_margin = 0.93 if (ax2 is not None) else 0.98
    fig.subplots_adjust(left=0.10, right=right_margin, bottom=0.18, top=0.90)

    # separa un poco el label del eje derecho para que no roce el borde
    if ax2 is not None:
        try:
            ax2.yaxis.labelpad = 10
        except Exception:
            pass

    self.canvas_hist.draw()

# --- 10.7: Colores del histograma ---
def choose_hist_colors(self):
    _es = (_current_lang() == "es")
    dlg = QDialog(self)
    dlg.setWindowTitle("Colores del histograma" if _es else "Histogram colors")
    layout = QVBoxLayout(dlg)

    btn_fill = QPushButton("Color de relleno de barras" if _es else "Bar fill color")
    lbl_fill = QLabel(f"{'Actual' if _es else 'Current'}: {getattr(self, 'hist_bar_fill', 'skyblue')}")
    def set_fill():
        color = QColorDialog.getColor()
        if color.isValid():
            self.hist_bar_fill = color.name()
            lbl_fill.setText(f"{'Actual' if _es else 'Current'}: {self.hist_bar_fill}")
    btn_fill.clicked.connect(set_fill)

    btn_edge = QPushButton("Color de borde de barras" if _es else "Bar edge color")
    lbl_edge = QLabel(f"{'Actual' if _es else 'Current'}: {getattr(self, 'hist_bar_edge', 'black')}")
    def set_edge():
        color = QColorDialog.getColor()
        if color.isValid():
            self.hist_bar_edge = color.name()
            lbl_edge.setText(f"{'Actual' if _es else 'Current'}: {self.hist_bar_edge}")
    btn_edge.clicked.connect(set_edge)

    btn_poly = QPushButton("Color del polígono de frecuencia" if _es else "Frequency polygon color")
    lbl_poly = QLabel(f"{'Actual' if _es else 'Current'}: {getattr(self, 'poly_color', 'gray')}")
    def set_poly():
        color = QColorDialog.getColor()
        if color.isValid():
            self.poly_color = color.name()
            lbl_poly.setText(f"{'Actual' if _es else 'Current'}: {self.poly_color}")
    btn_poly.clicked.connect(set_poly)

    btn_cum = QPushButton("Color de la curva acumulativa" if _es else "Cumulative curve color")
    lbl_cum = QLabel(f"{'Actual' if _es else 'Current'}: {getattr(self, 'cum_color', 'black')}")
    def set_cum():
        color = QColorDialog.getColor()
        if color.isValid():
            self.cum_color = color.name()
            lbl_cum.setText(f"{'Actual' if _es else 'Current'}: {self.cum_color}")
    btn_cum.clicked.connect(set_cum)

    for w in (btn_fill, lbl_fill, btn_edge, lbl_edge, btn_poly, lbl_poly, btn_cum, lbl_cum):
        layout.addWidget(w)
    btn_ok = QPushButton(tr("ok")); btn_ok.clicked.connect(dlg.accept)
    layout.addWidget(btn_ok)

    if dlg.exec_():
        self.plot_histogram()


# --- 10.8: “Editar histograma” (labels + fuentes + tamaño PX) ---
def choose_hist_labels(self):
    _ensure_hist_defaults(self)

    _es = (_current_lang() == "es")
    dlg = QDialog(self)
    dlg.setWindowTitle("Editar histograma" if _es else "Edit histogram")
    form = QFormLayout(dlg)

    if not hasattr(self, "hist_title"):   self.hist_title = ""
    if not hasattr(self, "hist_xlabel"):  self.hist_xlabel = ""
    if not hasattr(self, "hist_ylabel"):  self.hist_ylabel = ""
    if not hasattr(self, "hist_ylabel2"): self.hist_ylabel2 = ""

    txt_title  = QLineEdit(self.hist_title)
    txt_xlabel = QLineEdit(self.hist_xlabel)
    txt_ylabel = QLineEdit(self.hist_ylabel)
    txt_y2     = QLineEdit(self.hist_ylabel2)

    sp_title = QSpinBox(); sp_title.setRange(6, 72); sp_title.setValue(int(self.hist_title_size))
    sp_label = QSpinBox(); sp_label.setRange(6, 48); sp_label.setValue(int(self.hist_label_size))
    sp_tick  = QSpinBox(); sp_tick.setRange(6, 36); sp_tick.setValue(int(self.hist_tick_size))

    sp_w = QSpinBox(); sp_w.setRange(400, 8000); sp_w.setSingleStep(50); sp_w.setValue(int(self.hist_px_w))
    sp_h = QSpinBox(); sp_h.setRange(300, 8000); sp_h.setSingleStep(50); sp_h.setValue(int(self.hist_px_h))

    form.addRow(("Título:" if _es else "Title:"), txt_title)
    form.addRow(("Eje X (φ):" if _es else "X axis (φ):"), txt_xlabel)
    form.addRow(("Eje Y (wt %):" if _es else "Y axis (wt %):"), txt_ylabel)
    form.addRow(("Eje Y₂ (Acumulativa):" if _es else "Y₂ (Cumulative):"), txt_y2)
    form.addRow(("Tamaño título (pt):" if _es else "Title size (pt):"), sp_title)
    form.addRow(("Tamaño labels (pt):" if _es else "Axis labels (pt):"), sp_label)
    form.addRow(("Tamaño ticks (pt):" if _es else "Ticks (pt):"), sp_tick)
    form.addRow(("Ancho figura (px):" if _es else "Figure width (px):"), sp_w)
    form.addRow(("Alto figura (px):"  if _es else "Figure height (px):"), sp_h)

    btn_ok = QPushButton(tr("ok")); btn_ok.clicked.connect(dlg.accept)
    form.addRow(btn_ok)

    if dlg.exec_():
        self.hist_title   = txt_title.text()
        self.hist_xlabel  = txt_xlabel.text()
        self.hist_ylabel  = txt_ylabel.text()
        self.hist_ylabel2 = txt_y2.text()

        self.hist_title_size = int(sp_title.value())
        self.hist_label_size = int(sp_label.value())
        self.hist_tick_size  = int(sp_tick.value())

        self.hist_px_w = int(sp_w.value())
        self.hist_px_h = int(sp_h.value())
        _apply_hist_canvas_size(self)   # aplicar tamaño a canvas/scroll
        self.plot_histogram()


# --- 10.9: Mostrar/ocultar curvas del histograma (API) ---
def toggle_hist_curve(self, show_hist: bool, show_acc: bool, show_poly: bool):
    self.show_histogram = show_hist
    self.show_cumulative = show_acc
    self.show_frequency_polygon = show_poly
    self.plot_histogram()


# --- 10.10: Exportación múltiple (ya integrada con el tamaño fijo) ---
def open_bulk_hist_export(self):
    """Exporta varias muestras con EXACTAMENTE el mismo formato actual."""
    from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QDialogButtonBox, QComboBox

    _es = (_current_lang() == "es")
    dlg = QDialog(self)
    dlg.setWindowTitle("Exportar más de un histograma con este mismo formato" if _es else "Export multiple histograms with this format")
    v = QVBoxLayout(dlg)

    # base por defecto: la actual
    bases_disp = []
    if self.df_data is not None:   bases_disp.append(base_name_display("Tamiz"))
    if self.df_laser is not None:  bases_disp.append(base_name_display("Laser"))
    if self.df_hybrid is not None: bases_disp.append(base_name_display("Híbrido"))

    h_top = QHBoxLayout()
    h_top.addWidget(QLabel("Base:" if _es else "Dataset:"))
    cmb_base = QComboBox(); cmb_base.addItems(bases_disp)
    if self.cmb_hist_base.currentText() in bases_disp:
        cmb_base.setCurrentText(self.cmb_hist_base.currentText())
    h_top.addWidget(cmb_base); h_top.addStretch(1)
    v.addLayout(h_top)

    lst = QListWidget(); lst.setSelectionMode(QAbstractItemView.MultiSelection)
    v.addWidget(lst, 1)

    def fill_samples_for(base_disp):
        lst.clear()
        # reusar helper ya existente
        self._fill_hist_samples_for_base(base_disp)
        for i in range(self.cmb_hist_sample.count()):
            it = QListWidgetItem(self.cmb_hist_sample.itemText(i))
            it.setSelected(True)
            lst.addItem(it)

    fill_samples_for(cmb_base.currentText())
    cmb_base.currentTextChanged.connect(fill_samples_for)

    # formato
    h_fmt = QHBoxLayout()
    h_fmt.addWidget(QLabel("Formato:" if _es else "Format:"))
    cmb_fmt = QComboBox(); cmb_fmt.addItems(["PNG","SVG","PDF"])
    h_fmt.addWidget(cmb_fmt); h_fmt.addStretch(1)
    v.addLayout(h_fmt)

    # botones
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    v.addWidget(buttons)
    buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)

    if dlg.exec_():
        # carpeta
        out_title = "Elegir carpeta de destino" if _es else "Choose destination folder"
        outdir = QFileDialog.getExistingDirectory(self, out_title)
        if not outdir:
            return

        # guardar estado actual
        base0   = self.cmb_hist_base.currentText()
        sample0 = self.cmb_hist_sample.currentText()

        try:
            self.cmb_hist_base.blockSignals(True)
            self.cmb_hist_sample.blockSignals(True)

            self.cmb_hist_base.setCurrentText(cmb_base.currentText())
            self._on_hist_base_changed()

            ext = {"PNG":".png","SVG":".svg","PDF":".pdf"}[cmb_fmt.currentText()]
            dpi = 300

            for i in range(lst.count()):
                it = lst.item(i)
                if not it.isSelected():
                    continue
                self.cmb_hist_sample.setCurrentText(it.text())
                self.plot_histogram()

                # usar tamaño fijo actual del canvas (px) exactamente
                fname = f"{it.text()}_{cmb_base.currentText()}".replace(" ","_")
                path = os.path.join(outdir, fname + ext)
                self.canvas_hist.figure.savefig(path, dpi=dpi, bbox_inches="tight")

            QMessageBox.information(
                self,
                "Exportación" if _es else "Export",
                "Listo. Histograma(s) exportado(s)." if _es else "Done. Histogram(s) exported."
            )
        finally:
            # restaurar
            self.cmb_hist_base.setCurrentText(base0)
            self._on_hist_base_changed()
            if sample0:
                self.cmb_hist_sample.setCurrentText(sample0)
            self.cmb_hist_base.blockSignals(False)
            self.cmb_hist_sample.blockSignals(False)
            self.plot_histogram()


# --- Inyectar métodos en MainWindow ---
for _name in [
    "apply_translations",
    "_refresh_walker_radio_texts",
    "_refresh_gk_radio_texts",
    "_refresh_pardo_radio_texts",
    "update_console",
    "_axis_label_for_key",
    "plot_xy",
    "plot_walker",
    "plot_gk",
    "plot_pardo",
    "edit_colors",
    "select_ribbons",
    "select_groups_all",
    "select_groups_walker",
    "select_groups_gk",
    "select_groups_pardo",
    "select_db_to_plot",
    "select_groups_xy",
    "_update_hist_bases",
    "_fill_hist_samples_for_base",
    "_update_hist_samples",
    "_on_hist_base_changed",
    "_draw_hybrid_overlay",
    "_overlay_original_sieve",
    "plot_histogram",
    "choose_hist_colors",
    "choose_hist_labels",
    "toggle_hist_curve",
    "open_bulk_hist_export",
    "_ensure_hist_defaults",
    "_ensure_hist_scroller_and_fixed_canvas",
    "_apply_hist_canvas_size",
]:
    if _name in globals():
        setattr(MainWindow, _name, globals()[_name])

# endregion









# region # === Bloque 11: Exportación de gráficos, selección de método y tema ===

# --- 11.1: Exportar un canvas a imagen ---
def export_canvas(self, canvas):
    if canvas is None:
        return
    _es = (_current_lang() == "es")
    filters = "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
    file, sel = QFileDialog.getSaveFileName(
        self,
        tr("file_export_image"),
        "",
        filters
    )
    if not file:
        return

    # Añadir extensión si falta
    if "." not in os.path.basename(file):
        if "SVG" in sel:
            file += ".svg"
        elif "PDF" in sel:
            file += ".pdf"
        else:
            file += ".png"

    try:
        canvas.figure.savefig(file, dpi=300, bbox_inches="tight")
        QMessageBox.information(
            self,
            tr("info"),
            ("Imagen guardada en:\n" if _es else "Image saved to:\n") + file
        )
    except Exception as e:
        QMessageBox.warning(
            self,
            tr("error"),
            (f"No se pudo exportar la figura:\n{e}" if _es else f"Could not export figure:\n{e}")
        )

# --- 11.2: Guardar la pestaña actual con un clic del menú Archivo ---
def save_current_tab(self):
    idx = self.tabs.currentIndex()
    if idx == 0:   # XY
        self.export_canvas(self.canvas_xy)
    elif idx == 1: # Walker
        self.export_canvas(self.canvas_walker)
    elif idx == 2: # GK
        self.export_canvas(self.canvas_gk)
    elif idx == 3: # Pardo
        self.export_canvas(self.canvas_pardo)
    elif idx == 4: # Histograma
        self.export_canvas(self.canvas_hist)

# --- 11.3: Elegir método de parámetros (Folk & Ward / Trask) ---
def choose_method(self):
    _es = (_current_lang() == "es")
    dlg = QDialog(self)
    dlg.setWindowTitle(tr("cfg_method"))
    lay = QVBoxLayout(dlg)

    rb_fw = QRadioButton("Folk & Ward (φ)")
    rb_tr = QRadioButton("Trask (So)")
    if getattr(self, "current_method", "folkward") == "trask":
        rb_tr.setChecked(True)
    else:
        rb_fw.setChecked(True)

    lay.addWidget(QLabel("Método para calcular parámetros granulométricos:" if _es else "Method to compute grain-size parameters:"))
    lay.addWidget(rb_fw)
    lay.addWidget(rb_tr)

    btn = QPushButton(tr("ok"))
    btn.clicked.connect(dlg.accept)
    lay.addWidget(btn)

    if not dlg.exec_():
        return

    new_method = "trask" if rb_tr.isChecked() else "folkward"
    if new_method == getattr(self, "current_method", "folkward"):
        return

    self.current_method = new_method
    self._recompute_all_params()

def _recompute_all_params(self):
    """Recalcula parámetros para Tamiz/Láser/Híbrido con el método actual."""
    # Tamiz
    if getattr(self, "df_data", None) is not None:
        phi_col = self.df_data.columns[0]
        samp_col = self.df_data.columns[1]
        wt_col   = self.df_data.columns[2]
        grp_col  = getattr(self, "group_col", "Group")
        out = []
        for s, g in self.df_data.groupby(samp_col):
            p = calculate_parameters(g[phi_col].tolist(), g[wt_col].tolist(), self.current_method)
            out.append({"Sample": s, grp_col: str(g[grp_col].iloc[0]), **p})
        self.param_results = pd.DataFrame(out)

    # Láser
    if getattr(self, "df_laser", None) is not None:
        out = []
        for s, g in self.df_laser.groupby("Sample"):
            p = calculate_parameters(g["phi"].tolist(), g["Wt"].tolist(), self.current_method)
            out.append({"Sample": s, "Group": str(g["Group"].iloc[0]), **p})
        self.param_results_laser = pd.DataFrame(out)

    # Híbrido
    if getattr(self, "df_hybrid", None) is not None:
        out = []
        grp_col = getattr(self, "group_col", "Group")
        for s, g in self.df_hybrid.groupby("Tamiz Sample"):
            p = calculate_parameters(g["phi"].tolist(), g["wt%"].tolist(), self.current_method)
            # g podría no tener group_col (depende del armado); lo protegemos:
            gval = str(g[grp_col].iloc[0]) if (grp_col in g.columns and len(g) > 0) else "Sin Grupo"
            out.append({"Sample": s, grp_col: gval, **p})
        self.param_results_hybrid = pd.DataFrame(out)

    # Actualizar todo
    try: self._update_all_groups_and_colors()
    except Exception: pass
    try: self.update_console()
    except Exception: pass
    try: self.plot_xy()
    except Exception: pass
    try: self.plot_walker()
    except Exception: pass
    try: self.plot_gk()
    except Exception: pass
    try: self.plot_pardo()
    except Exception: pass
    try: self._update_hist_samples()
    except Exception: pass

# --- 11.4: Tema claro/oscuro ---
def choose_theme(self):
    _es = (_current_lang() == "es")
    dlg = QDialog(self)
    dlg.setWindowTitle(tr("cfg_theme"))
    lay = QVBoxLayout(dlg)

    rb_light = QRadioButton("Claro" if _es else "Light")
    rb_dark  = QRadioButton("Oscuro" if _es else "Dark")
    (rb_dark if getattr(self, "theme", "light") == "dark" else rb_light).setChecked(True)

    lay.addWidget(QLabel("Tema de la interfaz:" if _es else "Interface theme:"))
    lay.addWidget(rb_light)
    lay.addWidget(rb_dark)

    btn = QPushButton(tr("ok"))
    btn.clicked.connect(dlg.accept)
    lay.addWidget(btn)

    if not dlg.exec_():
        return

    self.theme = "dark" if rb_dark.isChecked() else "light"
    try:
        qApp.setStyleSheet(DARK_STYLESHEET if self.theme == "dark" else LIGHT_STYLESHEET)
    except Exception:
        pass

# === Vincular métodos del BLOQUE 11 a MainWindow (por si este bloque queda fuera de la clase) ===
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






# region # === Bloque 12 v3: Combinar Tamiz + Láser — Método 1 (original), Método 2 (EDLT+SG), Método 3 (placeholder) ===
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox,
    QListWidget, QListWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt

# ---------- Utilidades ----------
def _get_sieve_columns(self):
    phi_col = self.df_data.columns[0]
    samp_col = self.df_data.columns[1]
    wt_col  = self.df_data.columns[2]
    return phi_col, samp_col, wt_col

def _sanitize_subtables(self, tamiz_sample, laser_sample):
    phi_col, sieve_samp_col, sieve_wt_col = _get_sieve_columns(self)
    sieve_sub = self.df_data[self.df_data[sieve_samp_col].astype(str)==str(tamiz_sample)].copy()
    laser_sub = self.df_laser[self.df_laser["Sample"].astype(str)==str(laser_sample)].copy()

    for c in (phi_col, sieve_wt_col):
        sieve_sub[c] = pd.to_numeric(sieve_sub[c], errors="coerce")
    for c in ("phi","Wt"):
        laser_sub[c] = pd.to_numeric(laser_sub[c], errors="coerce")

    sieve_sub = sieve_sub[np.isfinite(sieve_sub[phi_col]) & np.isfinite(sieve_sub[sieve_wt_col])]
    laser_sub = laser_sub[np.isfinite(laser_sub["phi"]) & np.isfinite(laser_sub["Wt"])]
    return phi_col, sieve_sub, laser_sub, sieve_wt_col

def _attach_group_if_any(self, df_out, sieve_sub):
    try:
        if hasattr(self,"group_col") and self.group_col in self.df_data.columns:
            gval = sieve_sub[self.group_col].iloc[0] if not sieve_sub.empty else None
            df_out[self.group_col] = gval
    except Exception:
        pass
    return df_out

def _try_recompute_everything(self):
    for cand in ("recompute_all", "recalculate_all", "compute_all", "update_all_plots", "refresh_all"):
        fn = getattr(self, cand, None)
        if callable(fn):
            try: fn(); return
            except Exception: pass
    # caídas suaves
    try:
        if getattr(self, "df_hybrid", None) is not None and not self.df_hybrid.empty:
            pass
    except Exception:
        pass
    for cand in ("update_console","plot_xy","plot_walker","plot_gk","_update_hist_samples"):
        fn = getattr(self, cand, None)
        if callable(fn):
            try: fn()
            except Exception: pass

# ---------- Métodos de combinación ----------
def _build_hybrid_method1(self, tamiz_sample, laser_sample, phi_cut):
    """Original: grueso=tamiz(<phi*), finos=láser(>=phi*) escalado a EDLT."""
    phi_col, sieve_sub, laser_sub, sieve_wt_col = _sanitize_subtables(self, tamiz_sample, laser_sample)

    sieve_keep = sieve_sub[sieve_sub[phi_col] <  float(phi_cut)].copy()
    laser_keep = laser_sub[laser_sub["phi"]  >= float(phi_cut)].copy()

    s_coarse = float(sieve_keep[sieve_wt_col].sum()) if not sieve_keep.empty else 0.0
    l_fine   = float(laser_keep["Wt"].sum()) if not laser_keep.empty else 0.0
    target_fine = max(0.0, 100.0 - s_coarse)  # = EDLT si tamiz suma 100
    scale = (target_fine / l_fine) if l_fine > 0 else 0.0

    rows = []
    if not sieve_keep.empty:
        for _, r in sieve_keep.iterrows():
            rows.append({"phi": float(r[phi_col]), "wt%": float(r[sieve_wt_col]),
                         "Tamiz Sample": str(tamiz_sample), "Laser Sample": str(laser_sample)})
    if not laser_keep.empty and scale > 0:
        for _, r in laser_keep.iterrows():
            rows.append({"phi": float(r["phi"]), "wt%": float(r["Wt"]*scale),
                         "Tamiz Sample": str(tamiz_sample), "Laser Sample": str(laser_sample)})

    if not rows:
        return pd.DataFrame(columns=["phi","wt%","Tamiz Sample","Laser Sample"]), {"method":"m1","phi*":float(phi_cut)}

    df_h = pd.DataFrame(rows)
    df_h = df_h.groupby(["Tamiz Sample","Laser Sample","phi"], as_index=False)["wt%"].sum()
    df_h.sort_values(["Tamiz Sample","phi"], inplace=True)
    tot = df_h.groupby("Tamiz Sample")["wt%"].transform("sum")
    tot = tot.replace(0, np.nan)
    df_h["wt%"] = np.where(np.isfinite(tot), df_h["wt%"]*(100.0/tot), df_h["wt%"])
    df_h = _attach_group_if_any(self, df_h, sieve_sub)
    return df_h, {"method":"m1","phi*":float(phi_cut)}

def _build_hybrid_method2(self, tamiz_sample, laser_sample, phi_cut, phi_round=2):
    """
    EDLT + SG:
      - Finos (>=phi*): F_phi = k * L_phi, con k=EDLT/sum(L_finos)
      - SG (<phi* y con coincidencia): S'_phi = S_phi + k*L_phi
      - Gruesos sin láser (<phi* y sin coincidencia) quedan fijos
      - Renormalización protegida: solo (SG+Finos) se escalan por alpha=(100-Tfijo)/Mpre
    """
    phi_col, sieve_sub, laser_sub, sieve_wt_col = _sanitize_subtables(self, tamiz_sample, laser_sample)

    # Estandarizo phi (redondeo para favorecer coincidencias exactas)
    s_df = pd.DataFrame({
        "phi": np.round(sieve_sub[phi_col].astype(float).values, phi_round),
        "S":   sieve_sub[sieve_wt_col].astype(float).values
    }).groupby("phi", as_index=False)["S"].sum()

    l_df = pd.DataFrame({
        "phi": np.round(laser_sub["phi"].astype(float).values, phi_round),
        "L":   laser_sub["Wt"].astype(float).values
    }).groupby("phi", as_index=False)["L"].sum()

    # 1) Finos como Método 1 (EDLT)
    EDLT = float(s_df.loc[s_df["phi"] >= float(phi_cut), "S"].sum())
    L_fines_sum = float(l_df.loc[l_df["phi"] >= float(phi_cut), "L"].sum())
    k = (EDLT / L_fines_sum) if L_fines_sum > 0 else 0.0

    fines = l_df[l_df["phi"] >= float(phi_cut)].copy()
    fines["wt%"] = k * fines["L"]

    # 2) SG: bins con phi < phi* y coincidencia tamiz-láser
    s_coarse = s_df[s_df["phi"] < float(phi_cut)].copy()
    l_coarse = l_df[l_df["phi"] < float(phi_cut)].copy()

    # Conjunto de phis con coincidencia
    ph_coinc = np.intersect1d(s_coarse["phi"].values, l_coarse["phi"].values)
    # Gruesos sin láser (fijos)
    ph_fixed = np.setdiff1d(s_coarse["phi"].values, l_coarse["phi"].values)

    # SG ajustado (solo coincidencia)
    if ph_coinc.size:
        s_co = s_coarse[s_coarse["phi"].isin(ph_coinc)].set_index("phi")
        l_co = l_coarse[l_coarse["phi"].isin(ph_coinc)].set_index("phi")
        sg_adj = (s_co["S"] + k * l_co["L"]).reset_index().rename(columns={0:"wt%"})
        sg_adj.columns = ["phi","wt%"]
    else:
        sg_adj = pd.DataFrame(columns=["phi","wt%"])

    # Fijos: intactos
    fixed = s_coarse[s_coarse["phi"].isin(ph_fixed)].copy()
    fixed = fixed.rename(columns={"S":"wt%"}).loc[:,["phi","wt%"]]

    # 3) Renormalización protegida
    Tfijo = float(fixed["wt%"].sum()) if not fixed.empty else 0.0
    Mpre  = float(sg_adj["wt%"].sum()) + float(fines["wt%"].sum())
    Mtarget = max(0.0, 100.0 - Tfijo)
    alpha = (Mtarget / Mpre) if Mpre > 0 else 1.0

    sg_adj["wt%"] = sg_adj["wt%"] * alpha
    fines["wt%"]  = fines["wt%"]  * alpha

    # Ensamble final
    rows = []
    for _, r in fixed.iterrows():   # gruesos sin láser
        rows.append({"phi": float(r["phi"]), "wt%": float(r["wt%"]),
                     "Tamiz Sample": str(tamiz_sample), "Laser Sample": str(laser_sample)})
    for _, r in sg_adj.iterrows():  # SG ajustado y escalado
        rows.append({"phi": float(r["phi"]), "wt%": float(r["wt%"]),
                     "Tamiz Sample": str(tamiz_sample), "Laser Sample": str(laser_sample)})
    for _, r in fines.iterrows():   # finos escalados a EDLT y renormalizados
        rows.append({"phi": float(r["phi"]), "wt%": float(r["wt%"]),
                     "Tamiz Sample": str(tamiz_sample), "Laser Sample": str(laser_sample)})

    if not rows:
        return pd.DataFrame(columns=["phi","wt%","Tamiz Sample","Laser Sample"]), {
            "method":"m2","phi*":float(phi_cut),"k":float(k),"EDLT":float(EDLT),"alpha":float(alpha)
        }

    df_h = pd.DataFrame(rows)
    # Limpieza/orden y cierre exacto a 100 por redondeos
    df_h = df_h.groupby(["Tamiz Sample","Laser Sample","phi"], as_index=False)["wt%"].sum()
    df_h.sort_values(["Tamiz Sample","phi"], inplace=True)
    tot = float(df_h["wt%"].sum())
    if tot > 0:
        df_h["wt%"] = df_h["wt%"] * (100.0 / tot)

    df_h = _attach_group_if_any(self, df_h, sieve_sub)
    meta = {"method":"m2","phi*":float(phi_cut),"k":float(k),"EDLT":float(EDLT),"alpha":float(alpha)}
    return df_h, meta

def _build_hybrid_method3(self, tamiz_sample, laser_sample, phi_cut):
    """Placeholder. De momento, se comporta como el Método 1."""
    return _build_hybrid_method1(self, tamiz_sample, laser_sample, phi_cut)

def _build_hybrid_for_pair(self, tamiz_sample, laser_sample, phi_cut, method="m1"):
    method = (method or "m1").lower()
    if method == "m1":
        return _build_hybrid_method1(self, tamiz_sample, laser_sample, phi_cut)
    elif method == "m2":
        return _build_hybrid_method2(self, tamiz_sample, laser_sample, phi_cut)
    elif method == "m3":
        return _build_hybrid_method3(self, tamiz_sample, laser_sample, phi_cut)
    # fallback
    return _build_hybrid_method1(self, tamiz_sample, laser_sample, phi_cut)

# ---------- Diálogo ----------
class _CombineDialog(QDialog):
    """Selector de tamiz/láser, φ* y método (M1, M2, M3). Permite acumular pares."""
    def __init__(self, sieve_samples, laser_samples, default_phi=2.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combinar tamiz + láser")
        self.setModal(True)
        self.pairs = []

        v = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Tamiz:"))
        self.cb_tamiz = QComboBox(); self.cb_tamiz.addItems(sieve_samples); row.addWidget(self.cb_tamiz)
        row.addWidget(QLabel("Láser:"))
        self.cb_laser = QComboBox(); self.cb_laser.addItems(laser_samples); row.addWidget(self.cb_laser)
        row.addWidget(QLabel("Método:"))
        self.cb_method = QComboBox()
        self.cb_method.addItems([
            "MÉTODO 1 (original)",
            "MÉTODO 2 (EDLT + SG)",
            "MÉTODO 3 (placeholder)"
        ])
        row.addWidget(self.cb_method)
        v.addLayout(row)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("φ*"))
        self.sp_phi = QDoubleSpinBox(); self.sp_phi.setDecimals(2)
        self.sp_phi.setRange(-10.0, 10.0); self.sp_phi.setSingleStep(0.10)
        self.sp_phi.setValue(float(default_phi))
        r2.addWidget(self.sp_phi)
        self.btn_add = QPushButton("Agregar"); r2.addStretch(1); r2.addWidget(self.btn_add)
        v.addLayout(r2)

        self.lst = QListWidget(); v.addWidget(self.lst)

        r3 = QHBoxLayout()
        self.btn_remove = QPushButton("Quitar")
        a = QHBoxLayout()
        self.btn_ok = QPushButton("Aceptar")
        self.btn_cancel = QPushButton("Cancelar")
        r3.addWidget(self.btn_remove); r3.addStretch(1); r3.addWidget(self.btn_ok); r3.addWidget(self.btn_cancel)
        v.addLayout(r3)

        self.btn_add.clicked.connect(self._on_add)
        self.btn_remove.clicked.connect(self._on_remove)
        self.btn_ok.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self.reject)

    def _on_add(self):
        t = self.cb_tamiz.currentText().strip()
        l = self.cb_laser.currentText().strip()
        m_idx = self.cb_method.currentIndex()
        code = ["m1","m2","m3"][m_idx]
        phi = float(self.sp_phi.value())
        if not t or not l: return
        txt = f"{t} ⟷ {l} | {self.cb_method.currentText()} | φ*={phi:.2f}"
        item = QListWidgetItem(txt)
        payload = {"tamiz":t, "laser":l, "phi":phi, "method":code}
        item.setData(Qt.UserRole, payload)
        # evitar duplicados exactos
        for i in range(self.lst.count()):
            if self.lst.item(i).data(Qt.UserRole) == payload:
                return
        self.lst.addItem(item)

    def _on_remove(self):
        row = self.lst.currentRow()
        if row >= 0: self.lst.takeItem(row)

    def _on_accept(self):
        pairs = []
        for i in range(self.lst.count()):
            data = self.lst.item(i).data(Qt.UserRole)
            if isinstance(data, dict): pairs.append(data)
        if not pairs:
            pairs = [{
                "tamiz": self.cb_tamiz.currentText().strip(),
                "laser": self.cb_laser.currentText().strip(),
                "phi": float(self.sp_phi.value()),
                "method": ["m1","m2","m3"][self.cb_method.currentIndex()],
            }]
        self.pairs = pairs
        self.accept()

# ---------- Orquestador ----------
def combinar_tamiz_laser(self):
    if getattr(self,"df_data",None) is None or self.df_data is None or self.df_data.empty:
        QMessageBox.warning(self, "Error", "Cargá primero la base de Tamiz."); return
    if getattr(self,"df_laser",None) is None or self.df_laser is None or self.df_laser.empty:
        QMessageBox.warning(self, "Error", "Cargá primero la base de Láser."); return

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
        QMessageBox.warning(self, "Error", "No se detectaron nombres de muestra en Tamiz/Láser."); return

    default_phi = 2.00
    dlg = _CombineDialog(sieve_samples, laser_samples, default_phi, self)
    if not dlg.exec_(): return

    pairs = dlg.pairs or []
    if not pairs:
        QMessageBox.information(self, "Info", "No se agregó ningún par Tamiz–Láser."); return

    all_rows, meta = [], {}
    for p in pairs:
        t, l, phi = p.get("tamiz"), p.get("laser"), p.get("phi")
        method = p.get("method","m1")
        try:
            df_h, m = _build_hybrid_for_pair(self, t, l, phi, method=method)
            if not df_h.empty: all_rows.append(df_h)
            meta[str(t)] = m
        except Exception as e:
            print("Combinar error:", t, l, e)

    if not all_rows:
        QMessageBox.warning(self, "Error", "No se pudo construir el híbrido con los pares elegidos."); return

    self.df_hybrid = pd.concat(all_rows, ignore_index=True)
    self._hybrid_meta = meta

    # NUEVO: habilitar menú DB según disponibilidad + UX del histograma en "Híbrido"
    try:
        self._update_db_menu_actions()
    except Exception:
        pass
    try:
        disp = base_name_display("Híbrido")
        i = self.cmb_hist_base.findText(disp)
        if i >= 0:
            self.cmb_hist_base.setCurrentIndex(i)
            if hasattr(self, "_on_hist_base_changed"):
                self._on_hist_base_changed()
    except Exception:
        pass

    _try_recompute_everything(self)

    try:
        QMessageBox.information(self, "Info", "Híbrido generado correctamente.")
    except Exception:
        pass

# --- Vincular en MainWindow ---
for _name in ["combinar_tamiz_laser"]:
    if _name in globals():
        setattr(MainWindow, _name, globals()[_name])
# endregion



# region # === Bloque 13: Exportación de parámetros a Excel ===

def export_params_to_excel(self):
    """
    Exporta a Excel los parámetros calculados de Tamiz, Láser y (si existe) Híbrido.
    Crea además una hoja README con notas breves. Idioma según _current_lang().
    """
    _es = (_current_lang() == "es")

    # Verificar que haya algo para exportar
    has_tamiz  = getattr(self, "param_results", None) is not None
    has_laser  = getattr(self, "param_results_laser", None) is not None
    has_hybrid = getattr(self, "param_results_hybrid", None) is not None

    if not (has_tamiz or has_laser or has_hybrid):
        QMessageBox.information(
            self, tr("info"),
            "No hay parámetros para exportar." if _es else "There are no parameters to export."
        )
        return

    # Diálogo de guardado
    suggested = "Parametros.xlsx" if _es else "Parameters.xlsx"
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

    # Contenidos del README
    readme_rows = (
        [
            "Este archivo fue exportado desde la aplicación.",
            "Hojas: Tamiz, Laser, Hibrido (si existen).",
            "Valores en unidades coherentes con el método seleccionado."
        ] if _es else [
            "This workbook was exported from the application.",
            "Sheets: Sieve, Laser, Hybrid (if present).",
            "Values use units consistent with the selected method."
        ]
    )
    df_readme = pd.DataFrame({"Info": readme_rows})

    # Función interna para escribir con un engine y fallback si no está disponible
    def _write_excel(path):
        # 1) Intentar con xlsxwriter (rápido y robusto)
        try:
            with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
                df_readme.to_excel(xw, index=False, sheet_name="README")
                if has_tamiz:
                    self.param_results.to_excel(xw, index=False, sheet_name=("Tamiz" if _es else "Sieve"))
                if has_laser:
                    self.param_results_laser.to_excel(xw, index=False, sheet_name="Laser")
                if has_hybrid:
                    self.param_results_hybrid.to_excel(xw, index=False, sheet_name=("Hibrido" if _es else "Hybrid"))
            return True
        except Exception:
            pass

        # 2) Fallback: engine por defecto (openpyxl u otro disponible)
        with pd.ExcelWriter(path) as xw:
            df_readme.to_excel(xw, index=False, sheet_name="README")
            if has_tamiz:
                self.param_results.to_excel(xw, index=False, sheet_name=("Tamiz" if _es else "Sieve"))
            if has_laser:
                self.param_results_laser.to_excel(xw, index=False, sheet_name="Laser")
            if has_hybrid:
                self.param_results_hybrid.to_excel(xw, index=False, sheet_name=("Hibrido" if _es else "Hybrid"))
        return True

    # Escribir y reportar resultado
    try:
        _write_excel(file)
        QMessageBox.information(
            self, tr("info"),
            ("Parámetros exportados a:\n" if _es else "Parameters exported to:\n") + file
        )
    except Exception as e:
        QMessageBox.warning(
            self, tr("error"),
            (f"No se pudo exportar a Excel:\n{e}" if _es else f"Could not export to Excel:\n{e}")
        )

# Vincular el método a MainWindow en caso de que este bloque esté fuera de la clase
try:
    setattr(MainWindow, "export_params_to_excel", export_params_to_excel)
except Exception:
    pass

# endregion


# region # === Bloque 14: SFT (Wohletz 1989) — mezcla de subpoblaciones en φ + Exportación CSV ===
import math
import warnings

# ========= SFT (Wohletz 1989) en φ =========
def _sft_component_phi(phi, g, chi, phi_mode=None, phi2=None, normalize=True):
    """
    Curva SFT de una subpoblación en espacio φ:
        dM/dφ ∝ z * exp(-z),   con  z(φ) = chi * 2**(3*(g+1)*(φ - φ2))

    Parámetros
    ----------
    phi : array-like
        Soporte en φ (centros de bin del histograma, por ejemplo).
    g : float
        Parámetro de SFT (> -1 típico). Se recorta a > -0.9999 para estabilidad.
    chi : float
        Escala adimensional (>0). Controla la pendiente/curtosis del lóbulo.
    phi_mode : float, opcional
        Si se pasa, se impone que el modo (máximo) ocurra en φ = phi_mode (equivale a z=1).
        En ese caso se calcula φ2 = phi_mode - log2(1/chi)/(3*(g+1)).
    phi2 : float, opcional
        Ancla de escala. Se usa sólo si no se pasa phi_mode.
    normalize : bool
        Si True, la curva se normaliza a área 1 en el soporte 'phi'.

    Returns
    -------
    y : np.ndarray
        Valores proporcionales de la subpoblación SFT en φ.
    """
    phi = _as_array(phi)
    g = float(max(g, -0.9999))
    chi = float(max(chi, 1e-12))

    if phi_mode is not None:
        denom = 3.0 * (g + 1.0)
        if abs(denom) < 1e-12:
            # límite g → -1: toma phi2≈phi_mode
            phi2_eff = float(phi_mode)
        else:
            phi2_eff = float(phi_mode) - (math.log2(1.0/chi) / denom)
    else:
        phi2_eff = float(0.0 if phi2 is None else phi2)

    z = chi * np.power(2.0, 3.0*(g+1.0)*(phi - phi2_eff))
    # estabilidad numérica
    z = np.clip(z, 0.0, 1e6)
    y = z * np.exp(-z)

    if normalize:
        area = _safe_trapz(y, phi)
        if np.isfinite(area) and area > 0:
            y = y / area
    return y


def sft_mixture_phi(phi, components, weights=None, normalize_each=True):
    """
    Mezcla de subpoblaciones SFT en φ.

    Parameters
    ----------
    phi : array-like (φ)
    components : list[dict]
        Cada dict puede tener claves: {"g", "chi", "phi_mode"} o {"g","chi","phi2"}.
    weights : list[float] or None
        Pesos de mezcla (se normalizan). Si None, iguales.
    normalize_each : bool
        Si True, cada componente se normaliza a área 1 antes de ponderar.

    Returns
    -------
    mix : dict
        {"mix": y_mix, "components": [y1, y2, ...], "weights": [w1,...]}
    """
    phi = _as_array(phi)
    n = len(components)
    if n == 0:
        return {"mix": np.zeros_like(phi), "components": [], "weights": []}

    if weights is None:
        weights = [1.0/n] * n
    w = _as_array(weights)
    w = np.clip(w, 0.0, np.inf)
    if w.sum() <= 0:
        w = np.ones(n) / n
    else:
        w = w / w.sum()

    comps = []
    for comp in components:
        y = _sft_component_phi(
            phi,
            comp.get("g", 0.0),
            comp.get("chi", 1.0),
            phi_mode=comp.get("phi_mode", None),
            phi2=comp.get("phi2", None),
            normalize=normalize_each
        )
        comps.append(y)

    mix = np.sum([wi * yi for wi, yi in zip(w, comps)], axis=0)
    return {"mix": mix, "components": comps, "weights": w.tolist()}

# ========= GMM (Gaussians en φ) =========
def _gaussian_pdf_1d(x, mu, sigma):
    sigma = max(float(sigma), 1e-9)
    c = 1.0 / (math.sqrt(2.0*math.pi) * sigma)
    z = (x - mu) / sigma
    return c * np.exp(-0.5 * z*z)

def gmm_detect_modes(phi, wt, kmax=4, random_state=0):
    """
    Ajusta mezclas gaussianas en φ con k=1..kmax y elige por BIC.
    Requiere scikit-learn (GaussianMixture).

    Parameters
    ----------
    phi : array-like (φ)
    wt : array-like (wt % por bin)
    kmax : int
    random_state : int or None

    Returns
    -------
    dict con claves principales:
      - xs, mix_curve, components, means, stds, weights, k
      - assigned_weight_frac, bic, aic, loglik, rmse, r2, area_data, area_mix
      - peak_heights, fwhm, n_modes
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn no está disponible para GMM.")

    X = _as_array(phi).reshape(-1, 1)
    y = _as_array(wt)
    m = np.isfinite(X[:,0]) & np.isfinite(y) & (y >= 0)
    X = X[m].reshape(-1, 1)
    y = y[m]
    if X.size == 0 or y.sum() <= 0:
        raise ValueError("Datos vacíos o sin peso.")

    # soporte para dibujar (usa exactamente los phi observados)
    xs = X[:,0]
    order = np.argsort(xs)
    xs = xs[order]
    y_sorted = y[order]

    # probar k=1..kmax
    from sklearn.mixture import GaussianMixture
    best = None
    best_k = None
    best_bic = np.inf

    # sample_weight con sklearn (si fallara, fallback simple)
    have_sw = True
    try:
        # prueba seca
        GaussianMixture(1, random_state=random_state).fit(X, sample_weight=y)
    except TypeError:
        have_sw = False

    for k in range(1, int(kmax)+1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            reg_covar=1e-6,
            n_init=5
        )
        if have_sw:
            gm.fit(X, sample_weight=y)
        else:
            # fallback: repetir muestras según y (escalado)
            mult = max(1, int(round(500.0 / max(1.0, len(y)))))
            reps = np.clip(np.round((y / y.sum()) * len(y) * mult), 0, 1000).astype(int)
            X_rep = np.repeat(X, reps, axis=0)
            gm.fit(X_rep)

        bic = gm.bic(X)  # BIC estándar (no pondera y, pero sirve para modelo relativo)
        if bic < best_bic:
            best_bic = bic
            best = gm
            best_k = k

    gm = best
    k = best_k

    # Curva en xs
    # densidad de mezcla en φ (área=1), luego la escalamos al área del histograma
    dens = np.zeros_like(xs, dtype=float)
    comp_curves = []
    for j in range(k):
        mu = gm.means_[j,0]
        sd = math.sqrt(float(gm.covariances_[j][0,0]))
        pj = float(gm.weights_[j])
        dens_j = pj * _gaussian_pdf_1d(xs, mu, sd)
        comp_curves.append(dens_j.copy())
        dens += dens_j

    # escalar al área del histograma
    area_data = _safe_trapz(y_sorted, xs)
    area_mix  = _safe_trapz(dens, xs)
    scale = (area_data / area_mix) if (area_mix and np.isfinite(area_mix) and area_mix > 0) else 1.0
    mix_curve = dens * scale
    comp_curves = [c * scale for c in comp_curves]

    # responsabilidades ponderadas por y (asignación de masa)
    resp = gm.predict_proba(X[order])
    mass_per_comp = (resp.T * y_sorted).sum(axis=1)
    mass_total = y_sorted.sum()
    assigned_frac = (mass_per_comp / mass_total) if mass_total > 0 else np.zeros(k)

    # métricas
    yhat = mix_curve
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = float(np.sqrt(np.nanmean((y_sorted - yhat)**2)))
        ss_res = float(np.nansum((y_sorted - yhat)**2))
        ss_tot = float(np.nansum((y_sorted - np.nanmean(y_sorted))**2))
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    # log-verosimilitud aproximada ponderando y
    # (usar densidad de mezcla NO escalada a área_data para verosimilitud)
    dens_ll = np.maximum(dens, 1e-300)  # evitar log(0)
    loglik = float(np.sum(y_sorted * np.log(dens_ll)))
    p = (k - 1) + 2*k  # pesos (k-1) + (mu, sigma) por comp
    n_eff = float(mass_total)
    aic = float(2*p - 2*loglik)
    bic = float(p * np.log(max(n_eff, 1.0)) - 2*loglik)

    means = [float(gm.means_[j,0]) for j in range(k)]
    stds  = [float(math.sqrt(gm.covariances_[j][0,0])) for j in range(k)]
    weights = [float(w) for w in gm.weights_]

    # alturas de pico (tras escala) y FWHM en φ
    peak_heights = []
    fwhm = []
    for mu, sd, pj, comp in zip(means, stds, weights, comp_curves):
        dens_mu = pj * _gaussian_pdf_1d(mu, mu, sd) * scale
        peak_heights.append(float(dens_mu))
        fwhm.append(float(2.0 * math.sqrt(2.0*math.log(2.0)) * sd))

    return {
        "xs": xs,
        "mix_curve": mix_curve,
        "components": comp_curves,
        "means": means,
        "stds": stds,
        "weights": weights,
        "assigned_weight_frac": assigned_frac.tolist(),
        "k": int(k),
        "bic": bic,
        "aic": aic,
        "loglik": loglik,
        "rmse": rmse,
        "r2": r2,
        "area_data": area_data,
        "area_mix": float(_safe_trapz(mix_curve, xs)),
        "peak_heights": peak_heights,
        "fwhm": fwhm,
        "n_modes": int(k),
    }

# ========= Mezcla Weibull en d (μm), dibujada en φ (vía jacobiano) =========
def _weibull_pdf(d, k, lam):
    k = max(float(k), 1e-6)
    lam = max(float(lam), 1e-9)
    x = np.clip(_as_array(d), 1e-300, np.inf)
    return (k/lam) * np.power(x/lam, k-1.0) * np.exp(-np.power(x/lam, k))

def _weibull_mode_um(k, lam):
    # modo en d para k>1: d_mode = lam * ((k-1)/k)^(1/k)
    if k <= 1.0:
        return np.nan
    return float(lam * ((k-1.0)/k) ** (1.0/k))

def _update_weibull_k_fixed_point(d, w, k, lam, max_iter=50):
    """
    Actualización tipo punto fijo para k con pesos w.
    Fórmulas extendidas al caso ponderado:
        B = sum(w*ln d)/sum(w)
        A = sum(w*d^k*ln d) / sum(w*d^k)
        1/k = A - B
    """
    d = np.clip(_as_array(d), 1e-300, np.inf)
    w = np.clip(_as_array(w), 0.0, np.inf)
    sw = w.sum()
    if sw <= 0:
        return max(k, 1e-3)

    B = float((w * np.log(d)).sum() / sw)
    for _ in range(max_iter):
        dk = np.power(d, k)
        A = float((w * dk * np.log(d)).sum() / max((w * dk).sum(), 1e-300))
        invk = A - B
        if invk <= 0:
            k_new = max(0.2, k * 0.7)
        else:
            k_new = 1.0 / invk
        if abs(k_new - k) < 1e-6:
            return max(k_new, 1e-3)
        k = k_new
    return max(k, 1e-3)

def weibull_mixture_detect(phi, wt, n_components=3, max_iter=200, tol=1e-5, random_state=0):
    """
    Ajusta una mezcla de Weibull en diámetro d (μm) usando EM con pesos del histograma,
    y devuelve curvas transformadas a φ (usando |dd/dφ| = ln(2)*d).

    Parameters
    ----------
    phi : array-like (φ)
    wt : array-like (wt %)
    n_components : int
    max_iter : int
    tol : float

    Returns
    -------
    dict con claves:
      - xs_phi, mix_curve_phi, components_phi
      - modes_phi, modes_um, weights (π), assigned_weight_frac
      - k, lambda_um, sigma_phi (aprox), bic, aic, loglik, rmse, r2, area_data, area_mix
    """
    rng = np.random.RandomState(random_state)
    phi = _as_array(phi)
    y = _as_array(wt)
    m = np.isfinite(phi) & np.isfinite(y) & (y >= 0)
    phi = phi[m]
    y = y[m]
    if phi.size == 0 or y.sum() <= 0:
        raise ValueError("Datos vacíos o sin peso.")

    # pasar a d (μm)
    d = _phi_to_um(phi)
    xs_phi = np.sort(np.unique(phi))
    order = np.argsort(phi)
    phi = phi[order]; y = y[order]; d = d[order]

    # inicialización
    w_norm = y / y.sum()
    qs = np.linspace(0.15, 0.85, n_components)
    qd = _weighted_quantiles(d, w_norm, qs)
    lam = np.maximum(qd, np.median(d))  # semillas razonables
    k_shape = np.full(n_components, 2.0)
    pi = np.full(n_components, 1.0/n_components)

    # EM
    prev_ll = -np.inf
    for it in range(max_iter):
        # E-step: responsabilidades (ponderadas por y)
        dens = np.stack([_weibull_pdf(d, k_shape[j], lam[j]) for j in range(n_components)], axis=1)
        mix_dens = np.dot(dens, pi)
        mix_dens = np.clip(mix_dens, 1e-300, np.inf)
        resp = (dens * pi) / mix_dens[:, None]  # N x K
        # responsabilidades ponderadas por masa y:
        R = resp * y[:, None]

        # M-step: actualizar π
        mass_k = R.sum(axis=0)
        mass_total = y.sum()
        if mass_total <= 0:
            break
        pi = mass_k / mass_total
        pi = np.clip(pi, 1e-9, 1.0)
        pi = pi / pi.sum()

        # actualizar parámetros de cada componente
        for j in range(n_components):
            wj = R[:, j]
            sw = wj.sum()
            if sw <= 0:
                # resembrar suavemente
                k_shape[j] = 2.0
                lam[j] = float(np.average(d, weights=w_norm))
                continue
            # actualizar k mediante punto fijo
            k_new = _update_weibull_k_fixed_point(d, wj, k_shape[j], lam[j])
            k_shape[j] = max(k_new, 1e-3)
            # actualizar λ usando k nuevo
            dk = np.power(d, k_shape[j])
            lam[j] = float(np.power(np.sum(wj * dk) / sw, 1.0 / max(k_shape[j], 1e-9)))

        # log-verosimilitud ponderada
        dens = np.stack([_weibull_pdf(d, k_shape[j], lam[j]) for j in range(n_components)], axis=1)
        mix_dens = np.dot(dens, pi)
        mix_dens = np.clip(mix_dens, 1e-300, np.inf)
        ll = float(np.sum(y * np.log(mix_dens)))

        if abs(ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            break
        prev_ll = ll

    # Curvas en φ (aplicar jacobiano)
    xs_phi = phi  # usamos el mismo soporte del histograma
    dd_dphi = math.log(2.0) * _phi_to_um(xs_phi)  # |dd/dφ| = ln 2 · d
    comp_phi = []
    dens_comp_d = np.stack([_weibull_pdf(_phi_to_um(xs_phi), k_shape[j], lam[j]) for j in range(n_components)], axis=0)
    for j in range(n_components):
        fphi = pi[j] * dens_comp_d[j, :] * dd_dphi
        comp_phi.append(fphi.copy())
    mix_phi = np.sum(comp_phi, axis=0)

    # Escalar al área del histograma
    area_data = _safe_trapz(y, xs_phi)
    area_mix = _safe_trapz(mix_phi, xs_phi)
    scale = (area_data / area_mix) if (area_mix and np.isfinite(area_mix) and area_mix > 0) else 1.0
    mix_phi *= scale
    comp_phi = [c * scale for c in comp_phi]

    # métricas de ajuste en φ
    yhat = mix_phi
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = float(np.sqrt(np.nanmean((y - yhat)**2)))
        ss_res = float(np.nansum((y - yhat)**2))
        ss_tot = float(np.nansum((y - np.nanmean(y))**2))
        r2 = float(1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    # info de modos (en d y en φ)
    modes_um = [ _weibull_mode_um(k_shape[j], lam[j]) for j in range(n_components) ]
    modes_phi = [ float(_um_to_phi(mu)) if np.isfinite(mu) else np.nan for mu in modes_um ]

    # sigma_phi (aprox) por varianza numérica de cada componente (en φ)
    sigma_phi = []
    for comp in comp_phi:
        a = _safe_trapz(comp, xs_phi)
        if not (np.isfinite(a) and a > 0):
            sigma_phi.append(np.nan); continue
        m1 = _safe_trapz(xs_phi * comp, xs_phi) / a
        m2 = _safe_trapz((xs_phi**2) * comp, xs_phi) / a
        var = max(m2 - m1*m1, 0.0)
        sigma_phi.append(float(np.sqrt(var)))

    # criterios de información (aprox) con n_eff = masa total
    p = (n_components - 1) + 2*n_components  # π (K-1) + (k,λ) por comp
    n_eff = float(y.sum())
    aic = float(2*p - 2*prev_ll)
    bic = float(p * np.log(max(n_eff, 1.0)) - 2*prev_ll)

    return {
        "xs_phi": xs_phi,
        "mix_curve_phi": mix_phi,
        "components_phi": comp_phi,
        "weights": pi.tolist(),
        "assigned_weight_frac": (np.sum((resp * y[:,None]), axis=0) / max(y.sum(), 1e-9)).tolist(),
        "k": int(n_components),
        "lambda_um": [float(v) for v in lam.tolist()],
        "k_shape":   [float(v) for v in k_shape.tolist()],
        "modes_um":  [float(v) if np.isfinite(v) else np.nan for v in modes_um],
        "modes_phi": [float(v) if np.isfinite(v) else np.nan for v in modes_phi],
        "sigma_phi": sigma_phi,
        "bic": bic,
        "aic": aic,
        "loglik": float(prev_ll),
        "rmse": float(rmse),
        "r2": float(r2),
        "area_data": float(area_data),
        "area_mix": float(_safe_trapz(mix_phi, xs_phi)),
    }

# ========= Exportación CSV (GMM/Weibull unificado) =========
def export_gmm_csv_from_state(self):
    """
    Usa self._modes_last (preferente) o self._gmm_last para exportar los modos detectados
    (GMM o Weibull) a un CSV plano.
    """
    data = getattr(self, "_modes_last", None)
    if data is None:
        data = getattr(self, "_gmm_last", None)
    if not data:
        QMessageBox.information(self, "IA", "No hay resultados de modos para exportar.")
        return

    # Comun: base y sample
    base = str(data.get("base", ""))
    sample = str(data.get("sample", ""))

    rows = []
    method = data.get("method", "gmm").lower()

    if method == "weibull":
        k = int(data.get("k", 0))
        modes_phi = data.get("modes_phi", [])
        modes_um  = data.get("modes_um",  [])
        sigma_phi = data.get("sigma_phi", [])
        weights   = data.get("weights",   [])
        lam_um    = data.get("lambda_um", [])
        k_shape   = data.get("k_shape",   [])

        for i in range(k):
            rows.append({
                "method": "weibull",
                "base": base,
                "sample": sample,
                "component": i+1,
                "weight_frac": float(weights[i]) if i < len(weights) else np.nan,
                "mode_phi": float(modes_phi[i]) if i < len(modes_phi) else np.nan,
                "mode_um":  float(modes_um[i])  if i < len(modes_um)  else np.nan,
                "sigma_phi": float(sigma_phi[i]) if i < len(sigma_phi) else np.nan,
                "lambda_um": float(lam_um[i])    if i < len(lam_um)    else np.nan,
                "k_shape":   float(k_shape[i])   if i < len(k_shape)   else np.nan,
                "bic": float(data.get("bic", np.nan)),
                "aic": float(data.get("aic", np.nan)),
                "loglik": float(data.get("loglik", np.nan)),
                "rmse": float(data.get("rmse", np.nan)),
                "r2": float(data.get("r2", np.nan)),
                "area_data": float(data.get("area_data", np.nan)),
                "area_mix": float(data.get("area_mix", np.nan)),
            })
    else:
        # GMM
        k = int(data.get("k", 0))
        means = data.get("means", [])
        stds  = data.get("stds",  [])
        weights = data.get("weights", [])
        assigned = data.get("assigned_weight_frac", data.get("weights", []))
        for i in range(k):
            rows.append({
                "method": "gmm",
                "base": base,
                "sample": sample,
                "component": i+1,
                "weight_frac": float(assigned[i]) if i < len(assigned) else (float(weights[i]) if i < len(weights) else np.nan),
                "mean_phi": float(means[i]) if i < len(means) else np.nan,
                "sigma_phi": float(stds[i]) if i < len(stds) else np.nan,
                "bic": float(data.get("bic", np.nan)),
                "aic": float(data.get("aic", np.nan)),
                "loglik": float(data.get("loglik", np.nan)),
                "rmse": float(data.get("rmse", np.nan)),
                "r2": float(data.get("r2", np.nan)),
                "area_data": float(data.get("area_data", np.nan)),
                "area_mix": float(data.get("area_mix", np.nan)),
            })

    df = pd.DataFrame(rows)

    # diálogo de guardado
    suggested = f"modos_{method}_{sample or 'sample'}.csv"
    file, _ = QFileDialog.getSaveFileName(
        self,
        tr("btn_export_modes") if "btn_export_modes" in I18N.get(_current_lang(), {}) else "Exportar modos (CSV)",
        suggested,
        "CSV (*.csv)"
    )
    if not file:
        return
    if not file.lower().endswith(".csv"):
        file += ".csv"

    try:
        df.to_csv(file, index=False, encoding="utf-8")
        QMessageBox.information(self, tr("success"), ("Guardado en:\n" if _current_lang()=="es" else "Saved to:\n") + file)
    except Exception as e:
        QMessageBox.warning(self, tr("error"), f"{'No se pudo exportar' if _current_lang()=='es' else 'Could not export'}:\n{e}")


# === Vincular exportación al MainWindow ===
try:
    setattr(MainWindow, "export_gmm_csv_from_state", export_gmm_csv_from_state)
except Exception:
    pass
# endregion






# region # === Bloque 15: Diálogos auxiliares y widgets (GroupSelect, ColorDialog, DataBaseWindow, FixedSizeFigureCanvas) ===
class GroupSelectDialog(QDialog):
    """Diálogo de selección múltiple de grupos (con checkboxes)."""
    def __init__(self, groups, preselected=None, title="Seleccionar grupos", parent=None):
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
            cb = QCheckBox(g); cb.setChecked((preselected is None) or (g in preselected))
            v.addWidget(cb)
            self._checks[g] = cb
        v.addStretch(1)
        scroll.setWidget(cont)
        layout.addWidget(scroll)

        btns = QHBoxLayout()
        btn_ok = QPushButton(tr("ok")); btn_ok.clicked.connect(self.accept)
        btns.addStretch(1); btns.addWidget(btn_ok)
        layout.addLayout(btns)

    def get_selected(self):
        return [g for g, cb in self._checks.items() if cb.isChecked()]


class ColorDialog(QDialog):
    """Editor simple de colores por grupo."""
    def __init__(self, groups, current_colors: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("cfg_colors"))
        self.colors = dict(current_colors)
        layout = QVBoxLayout(self)

        self._rows = []
        grid = QGridLayout()
        _es = (_current_lang() == "es")
        grid.addWidget(QLabel("Grupo" if _es else "Group"), 0, 0)
        grid.addWidget(QLabel("Color"), 0, 1)
        for i, g in enumerate(groups, start=1):
            lbl = QLabel(g)
            btn = QPushButton(self.colors.get(g, "#888888"))
            btn.clicked.connect(lambda _, key=g, b=btn: self._pick_color(key, b))
            grid.addWidget(lbl, i, 0)
            grid.addWidget(btn, i, 1)
            self._rows.append((g, btn))
        layout.addLayout(grid)

        btn_ok = QPushButton(tr("ok")); btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)

    def _pick_color(self, key, btn):
        col = QColorDialog.getColor()
        if col.isValid():
            self.colors[key] = col.name()
            btn.setText(col.name())


class DataBaseWindow(QDialog):
    """
    Muestra un DataFrame en una tabla con opción de exportar a Excel.
    **Formato de visualización**: todas las celdas numéricas con 2 decimales.
    **Exportar a Excel**: se redondea a 2 decimales para mantener consistencia visual.
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
        btn_export = QPushButton("Exportar a Excel" if _current_lang()=="es" else "Export to Excel")
        btn_export.clicked.connect(self._export_excel)
        btn_close = QPushButton(tr("ok")); btn_close.clicked.connect(self.accept)
        btns.addStretch(1); btns.addWidget(btn_export); btns.addWidget(btn_close)
        layout.addLayout(btns)

        self._populate()

    @staticmethod
    def _fmt_cell(val) -> str:
        """Devuelve string con 2 decimales si es número; caso contrario, tal cual."""
        # Detectar numérico robusto (admite str con coma/punto, NaN, etc.)
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass
        s = str(val).strip()
        # Intento convertir respetando coma como decimal
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
                # Alinear números a la derecha
                try:
                    float(str(val).replace(",", "."))
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                except Exception:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()

    def _export_excel(self):
        _es = (_current_lang() == "es")
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar a Excel" if _es else "Export to Excel",
            "Base.xlsx",
            "Excel files (*.xlsx)"
        )
        if not file:
            return
        if not file.lower().endswith(".xlsx"):
            file += ".xlsx"

        # Copia con redondeo a 2 decimales solo en columnas numéricas
        df_out = self._df.copy()
        try:
            num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                df_out[num_cols] = df_out[num_cols].round(2)
        except Exception:
            # Fallback: intentar convertir cada columna a numérico y redondear
            for col in df_out.columns:
                try:
                    v = pd.to_numeric(df_out[col], errors="coerce")
                    if v.notna().any():
                        df_out[col] = v.round(2)
                except Exception:
                    pass

        try:
            df_out.to_excel(file, index=False)
            QMessageBox.information(self, "OK", ("Guardado en:\n" if _es else "Saved to:\n") + file)
        except Exception as e:
            QMessageBox.warning(self, "Error", (f"No se pudo exportar:\n{e}" if _es else f"Could not export:\n{e}"))


class FixedSizeFigureCanvas(FigureCanvas):
    """
    Lienzo de Matplotlib con tamaño fijo en píxeles.
    Útil para mostrar imágenes de fondo (Walker / GK) sin deformaciones.
    """
    def __init__(self, width_in, height_in, dpi=100):
        fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
        super().__init__(fig)
        w_px = int(round(width_in * dpi))
        h_px = int(round(height_in * dpi))
        self.setFixedSize(w_px, h_px)
# endregion



# region # === Bloque 16: Punto de entrada ===
if __name__ == "__main__":
    import sys, os
    from PyQt5.QtCore import Qt, QCoreApplication
    from PyQt5.QtGui import QPixmap, QIcon
    from PyQt5.QtWidgets import QApplication, QSplashScreen
    import matplotlib as mpl

    # --- HiDPI y pixmaps nítidos (antes de crear la app) ---
    try:
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    # --- Matplotlib: tamaños globales sobrios (coherentes con Bloque 10) ---
    try:
        mpl.rcParams.update({
            "font.size": 9,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        })
    except Exception:
        pass

    # Crear la aplicación Qt
    app = QApplication(sys.argv)

    # Idioma por defecto visible a nivel app (lo usan _current_lang y tr)
    try:
        from PyQt5.QtWidgets import qApp  # ya existe con QApplication
        qApp.lang = DEFAULT_LANG  # viene del Bloque 0
    except Exception:
        pass

    # --- Resolver rutas de recursos (icono y banner/splash) ---
    def _script_dir():
        try:
            return get_script_dir()  # helper del Bloque 4
        except Exception:
            return os.path.dirname(os.path.abspath(__file__))

    base = _script_dir()

    # Helper local: devuelve el primer archivo existente en 'cands'
    def _first_existing(cands):
        for fn in cands:
            p = os.path.join(base, fn)
            if os.path.exists(p):
                return p
        return None

    # ======================
    #   ICONO DE LA APP
    # ======================
    # No pisamos ICON_CANDIDATES global; solo buscamos localmente.
    # Preferimos .ico/.jpg para evitar warnings de perfiles PNG,
    # pero dejamos tu "icono.nuevo.png" como fallback explícito.
    _icon_candidates_local = (
        # preferencia alta: .ico
        "app_icon.ico", "app.ico", "logo.ico", "icon.ico",
        # preferencia luego: .jpg/.jpeg
        "app_icon.jpg", "logo.jpg", "icon.jpg", "app_icon.jpeg", "logo.jpeg", "icon.jpeg",
        # png (incluye tu archivo)
        "icono.nuevo.png", "app_icon.png", "logo.png", "icon.png", "fw.png"
    )
    icon_path = _first_existing(_icon_candidates_local)

    if icon_path:
        try:
            app.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass

    # ======================
    #   SPLASH / BANNER
    # ======================
    # Priorizamos .jpg para minimizar warnings de perfiles PNG.
    _splash_candidates = (
        "banner.jpg", "flyer.jpg", "splash.jpg", "portada.jpg",
        "banner.jpeg", "flyer.jpeg", "splash.jpeg", "portada.jpeg",
        "banner.png", "flyer.png", "splash.png", "portada.png"
    )
    splash = None
    sp_path = _first_existing(_splash_candidates)
    if sp_path:
        try:
            pm = QPixmap(sp_path)
            if not pm.isNull():
                # Escalar suavemente si es enorme
                max_h = 360
                if pm.height() > max_h:
                    pm = pm.scaledToHeight(max_h, Qt.SmoothTransformation)
                splash = QSplashScreen(pm)
                splash.show()
                app.processEvents()
        except Exception:
            splash = None

    # Crear y mostrar la ventana principal
    win = MainWindow()
    if icon_path:
        try:
            win.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass

    win.show()

    # Cerrar splash cuando ya está la ventana
    if splash is not None:
        try:
            splash.finish(win)
        except Exception:
            splash.close()

    # Event loop
    sys.exit(app.exec_())
# endregion
# endregion =================== FIN BLOQUE 16: Punto de entrada ===================
