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
    QSpinBox
)
from PyQt5.QtCore import Qt, QAbstractTableModel

# === ML opcional (GMM) ===
try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

#endregion

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

SYMBOLS = {
    "mean": "Mzφ",
    "median": "Mdφ",
    "sigma": "σφ",
    "skewness": "Skφ",
    "kurtosis": "Kφ"
}

# Métodos disponibles en la UI (por pedido: F&W, Inman y Trask).
METHODS = {
    "Folk & Ward (1957)": "folkward",
    "Inman (1952)": "inman",
    "Trask (1932)": "trask"
}

WALKER_IMAGES = {
    "All": "WALKER.tif",
    "Fall deposit": "WALKER-FALL-DEPOSIT.tif",
    "Fine-depleted flow": "WALKER-FINE-DEPLETED-FLOW.tif",
    "Surge (Walker) and Surge-Dunes (Gençalioğlu-Kuşcu)": "WALKER-PYROCLASTIC-SURGE.tif",
    "Pyroclastic flow": "WALKER-PYROCLASTIC-FLOW.tif"
}

GK_IMAGES = {
    "All": "GK.tif",
    "Flow": "GK-FLOW.tif",
    "Surge (Walker) and Surge-Dunes (Gençalioğlu-Kuşcu)": "GK-SURGEDUNES.tif",
    "Fall": "GK-FALL.tif",
    "Surge Dunes": "GK-SURGEDUNES.tif"
}

GROUP_COLORS = [
    "#52b788",
    "#4895ef",
    "#e76f51",
    "#ffd60a",
    "#adb5bd",
]

# --- Opciones de presentación específicas para Trask ---
# "phi_equiv": muestra σ en unidades φ equivalente (para comparar en los gráficos).
# "raw": muestra So (Trask) sin convertir en los ejes/etiquetas donde corresponda.
TRASK_DISPLAY = "phi_equiv"  # "phi_equiv" o "raw"

def sigma_axis_label(method: str) -> str:
    """
    Devuelve el rótulo del eje Y para 'sigma' según el método activo.
    - Si el método es Trask y se eligió 'raw', el eje debe decir 'So (Trask)'.
    - En cualquier otro caso se usa el símbolo estándar σφ.
    """
    return "So (Trask)" if (method == "trask" and TRASK_DISPLAY == "raw") else SYMBOLS["sigma"]

# endregion

# region === Bloque 3: Funciones auxiliares de cálculo ===
import numpy as np

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
    Cálculo de parámetros con nombres CONSISTENTES en todas las opciones:
      devuelve dict con: {"median","sigma","skewness","kurtosis","mean"}

    Métodos:
      - "folkward" (Folk & Ward, 1957)  -> usa percentiles 5-16-25-50-75-84-95
      - "inman"    (Inman, 1952)        -> usa 16-50-84
      - "trask"    (Trask, 1932)        -> So = sqrt(d75/d25) en DIAMETRO.
                                           En 'sigma' se devuelve por defecto
                                           sigma_phi_equiv = (phi75 - phi25)/2 = -log2(So) >= 0.
                                           Si en el Bloque 2 TRASK_DISPLAY == "raw",
                                           entonces 'sigma' reporta So (adimensional).

    NOTA: Los nombres (mean/median/sigma/skewness/kurtosis) NO cambian nunca.
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

    # Percentiles comunes cuando se necesiten
    def _percs(*plist):
        cum = cumulative_distribution(phi, weights)
        if not np.isfinite(cum).any():
            return {p: np.nan for p in plist}
        out = {p: interp_percentile(phi, cum, p) for p in plist}
        return out

    # --- Folk & Ward (alias Blatt) ---
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

        return {"median": median, "sigma": float(sigma), "skewness": float(skewness),
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
        return {"median": phi50, "sigma": float(sigma), "skewness": float(skewness),
                "kurtosis": np.nan, "mean": float(mean)}

    # --- Trask (1932) ---
    elif method == "trask":
        # Necesitamos cuartiles y (opcionalmente) 16/84 para "mean" homogéneo.
        p = _percs(25, 50, 75, 16, 84)
        phi25, phi50, phi75 = p[25], p[50], p[75]
        phi16, phi84 = p[16], p[84]
        if not all(np.isfinite([phi25, phi50, phi75])):
            return {"median": np.nan, "sigma": np.nan, "skewness": np.nan,
                    "kurtosis": np.nan, "mean": np.nan}

        # So (Trask) en DIAMETRO: So = sqrt(d75/d25) = 2**((phi25 - phi75)/2)
        so = float(2.0 ** ((phi25 - phi75) / 2.0))

        # Cómo reportar 'sigma' según TRASK_DISPLAY (definido en Bloque 2)
        trask_display = globals().get("TRASK_DISPLAY", "phi_equiv")
        if trask_display == "raw":
            sigma = so
        else:
            # sigma equivalente en phi: positivo
            sigma = (phi75 - phi25) / 2.0

        # Trask no define Sk y K en phi; las dejamos NaN.
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
    # Ej.: step=1 -> round(phi) ; step=0.5 -> round(2*phi)/2
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


# region # === Bloque 5: Clases de diálogos personalizados y visualización DataFrame ===

class ColorDialog(QDialog):
    def __init__(self, groups, colors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Elegir colores por grupo")
        self.resize(340, 270)
        self.groups = groups
        self.colors = colors.copy()
        self.layout = QVBoxLayout()
        self.color_buttons = {}

        for grp in groups:
            h = QHBoxLayout()
            lbl = QLabel(str(grp))
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {self.colors.get(grp, '#888888')};")
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, g=grp: self.change_color(g))
            h.addWidget(lbl)
            h.addWidget(btn)
            h.addStretch()
            self.layout.addLayout(h)
            self.color_buttons[grp] = btn

        btn_ok = QPushButton("Aceptar")
        btn_ok.clicked.connect(self.accept)
        self.layout.addWidget(btn_ok)
        self.setLayout(self.layout)

    def change_color(self, grp):
        color = QColorDialog.getColor()
        if color.isValid():
            self.colors[grp] = color.name()
            self.color_buttons[grp].setStyleSheet(f"background-color: {color.name()};")


class GroupSelectDialog(QDialog):
    def __init__(self, groups, selected, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(360, 380)
        self.groups = groups
        self.selected = selected.copy()
        self.layout = QVBoxLayout()
        self.checkboxes = {}

        for grp in groups:
            cb = QCheckBox(str(grp))
            cb.setChecked(grp in self.selected)
            self.layout.addWidget(cb)
            self.checkboxes[grp] = cb

        # Botones: seleccionar / deseleccionar todos + aceptar
        h = QHBoxLayout()
        btn_all = QPushButton("Seleccionar todos")
        btn_none = QPushButton("Deseleccionar todos")
        btn_ok = QPushButton("Aceptar")

        btn_all.clicked.connect(self.select_all)
        btn_none.clicked.connect(self.deselect_all)
        btn_ok.clicked.connect(self.accept)

        h.addWidget(btn_all)
        h.addWidget(btn_none)
        h.addStretch()
        h.addWidget(btn_ok)

        self.layout.addLayout(h)
        self.setLayout(self.layout)

    def select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)

    def deselect_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)

    def get_selected(self):
        return [g for g, cb in self.checkboxes.items() if cb.isChecked()]

#endregion

# region # === Bloque 5.1: Modelo para mostrar DataFrame en QTableView ===
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        val = self._df.iat[index.row(), index.column()]
        return "" if pd.isna(val) else str(val)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return ""
        else:
            try:
                return str(self._df.index[section])
            except Exception:
                return ""

# endregion
# region # === Bloque 5.2: Ventana genérica para mostrar DataFrame ===
class DataBaseWindow(QDialog):
    def __init__(self, title, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(920, 480)
        layout = QVBoxLayout(self)
        table = QTableView()
        table.setSortingEnabled(True)  # si querés evitar que ordenen y muevan la fila "Grupo", podés poner False
        model = PandasModel(df)
        table.setModel(model)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

#endregion

# region # === Bloque 5.3: Diálogo para emparejar muestras Tamiz ↔ Láser (con φₑₓₜᵣₐ) ===
class MatchDialog(QDialog):
    def __init__(self, df_tamiz, df_laser, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Emparejar muestras Tamiz y Láser")
        self.resize(900, 450)
        self.pairs = []        # lista de tuplas (tamiz_sample, laser_sample)
        self.phi_map = {}      # mapa tamiz_sample → φ_extraído

        layout = QHBoxLayout(self)

        # Tabla Tamiz
        self.tv_t = QTableView()
        tamiz_samples = df_tamiz.iloc[:,1].drop_duplicates().tolist()
        df_t = pd.DataFrame(tamiz_samples, columns=["G. Tamizado"])
        self.model_t = PandasModel(df_t)
        self.tv_t.setModel(self.model_t)
        self.tv_t.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(self.tv_t)

        # Tabla Láser
        self.tv_l = QTableView()
        laser_samples = df_laser["Sample"].drop_duplicates().tolist()
        df_l = pd.DataFrame(laser_samples, columns=["G. Láser"])
        self.model_l = PandasModel(df_l)
        self.tv_l.setModel(self.model_l)
        self.tv_l.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(self.tv_l)

        # Tabla de emparejamientos con columna de φₑₓₜᵣₐ
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Tamiz", "Láser", "φₑₓₜᵣₐ"])
        layout.addWidget(self.table)

        # Botones a la derecha
        right = QVBoxLayout()
        btn_pair = QPushButton("Emparejar selección")
        btn_pair.clicked.connect(self.pair_selected)
        right.addWidget(btn_pair)
        right.addStretch()
        btn_ok = QPushButton("Listo")
        btn_ok.clicked.connect(self.accept)
        right.addWidget(btn_ok)
        layout.addLayout(right)

    def pair_selected(self):
        sel_t = self.tv_t.selectionModel().selectedRows()
        sel_l = self.tv_l.selectionModel().selectedRows()
        if not sel_t or not sel_l:
            QMessageBox.warning(self, "Error", "Selecciona una fila en cada tabla")
            return
        t = self.model_t._df.iloc[sel_t[0].row(), 0]
        l = self.model_l._df.iloc[sel_l[0].row(), 0]
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(t)))
        self.table.setItem(row, 1, QTableWidgetItem(str(l)))
        spin = QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)
        spin.setDecimals(2)
        spin.setValue(0.0)
        self.table.setCellWidget(row, 2, spin)
        self.pairs.append((t, l))

#endregion

# region # === Bloque 6: Canvas de tamaño fijo ===
class FixedSizeFigureCanvas(FigureCanvas):
    def __init__(self, width, height, dpi=100, *args, **kwargs):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setFixedSize(int(width * dpi), int(height * dpi))
        self.setSizePolicy(self.sizePolicy().Fixed, self.sizePolicy().Fixed)

# endregion 
# region # === Bloque 7: Clase MainWindow – Constructor y menú principal ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ---- estado base / defaults ----
        self.setWindowTitle("Folk & Ward - Gráfico XY tipo paper + Walker + GK (PyQt5 version)")
        self.resize(1200, 800)

        self.theme = "light"                 # tema por defecto
        self.current_method = "folkward"     # método de cálculo por defecto

        # DataFrames / resultados
        self.df_data = None
        self.param_results = None
        self.group_col = "Group"

        self.df_laser = None
        self.param_results_laser = None

        self.df_hybrid = None
        self.param_results_hybrid = None

        # Grupos y colores
        self.groups = []                     # lista de "Facies (Base)"
        self.group_colors = {}               # dict clave->color hex
        self.selected_groups_xy = []
        self.selected_groups_ribbons = []
        self.selected_groups_walker = []
        self.selected_groups_gk = []

        # Estilo inicial
        try:
            qApp.setStyleSheet(LIGHT_STYLESHEET if self.theme == "light" else DARK_STYLESHEET)
        except Exception:
            pass

        # Construir UI
        self.initUI()

        # === CAMBIO: Estado IA/GMM ===
        self._gmm_last = None
# endregion

    # region # === Bloque 7.1: Menú principal ===
    def initMenu(self):
        menubar = self.menuBar()

        # --- Archivo ---
        menu_file = menubar.addMenu("&Archivo")

        act_load_tamiz = QAction("Cargar archivo Excel de tamizado", self)
        act_load_tamiz.triggered.connect(self.load_file)
        menu_file.addAction(act_load_tamiz)

        act_load_laser = QAction("Cargar archivo Excel de laser", self)
        act_load_laser.triggered.connect(self.load_laser_file)
        menu_file.addAction(act_load_laser)

        act_save = QAction("Exportar imagen actual", self)
        act_save.triggered.connect(self.save_current_tab)
        menu_file.addAction(act_save)

        act_combine = QAction("Combinar tamiz + láser", self)
        act_combine.triggered.connect(self.combinar_tamiz_laser)
        menu_file.addAction(act_combine)

        # --- Configuración ---
        menu_cfg = menubar.addMenu("&Configuración")
        menu_cfg.addAction(QAction("Método de cálculo", self, triggered=self.choose_method))
        menu_cfg.addAction(QAction("Colores de grupos", self, triggered=self.edit_colors))

        # ÚNICA entrada global para visibilidad de grupos
        menu_cfg.addAction(QAction("Grupos visibles (todas las pestañas)", self,
                                   triggered=self.select_groups_all))

        # Sombras (solo XY)
        menu_cfg.addAction(QAction("Sombras (XY)", self, triggered=self.select_ribbons))

        menu_cfg.addAction(QAction("Estilo de interfaz", self, triggered=self.choose_theme))

        # --- Base de datos ---
        menu_db = menubar.addMenu("Base de datos")

        self.act_viewdb_tamiz = QAction("Ver base de datos de tamiz", self)
        self.act_viewdb_tamiz.setEnabled(False)
        self.act_viewdb_tamiz.triggered.connect(self.show_tamiz_db_window)
        menu_db.addAction(self.act_viewdb_tamiz)

        self.act_viewdb_laser = QAction("Ver base de datos de láser", self)
        self.act_viewdb_laser.setEnabled(False)
        self.act_viewdb_laser.triggered.connect(self.show_laser_db_window)
        menu_db.addAction(self.act_viewdb_laser)

        self.act_viewdb_hybrid = QAction("Ver base de datos tamiz + láser", self)
        self.act_viewdb_hybrid.setEnabled(False)
        self.act_viewdb_hybrid.triggered.connect(self.show_hybrid_db_window)
        menu_db.addAction(self.act_viewdb_hybrid)

        # --- Ayuda ---
        menu_help = menubar.addMenu("&Ayuda")
        menu_help.addAction(
            QAction(
                "Acerca de...", self,
                triggered=lambda: QMessageBox.information(
                    self,
                    "Acerca de",
                    "Software libre y gratuito de análisis granulométrico de depósitos piroclásticos.\n\n"
                    "Más información: santiagoabelretamoso@gmail.com\n\n"
                    "Folk and Ward Granulometry or\nFW-Granulometry - Python"
                )
            )
        )

# endregion

    # region # === Bloque 7.2: Actualización de grupos y colores ===
    def _update_all_groups_and_colors(self):
        """
        Actualiza self.groups y self.group_colors para todas las bases presentes,
        usando la clave 'nombre de grupo (base)' para cada combinación.
        """
        bases = []
        if self.param_results is not None:
            bases.append(("Tamiz", self.param_results, self.group_col))
        if self.param_results_laser is not None:
            bases.append(("Laser", self.param_results_laser, "Group"))
        if self.param_results_hybrid is not None:
            bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

        unique_groups = []
        for name, df, group_col in bases:
            for grp in sorted(df[group_col].unique()):
                clave = f"{grp} ({name})"
                if clave not in unique_groups:
                    unique_groups.append(clave)
        self.groups = unique_groups

        # Asignar/Completar colores
        if not self.group_colors:
            for i, clave in enumerate(self.groups):
                self.group_colors[clave] = GROUP_COLORS[i % len(GROUP_COLORS)]
        else:
            for i, clave in enumerate(self.groups):
                if clave not in self.group_colors:
                    self.group_colors[clave] = GROUP_COLORS[i % len(GROUP_COLORS)]

        # Por defecto todos visibles
        self.selected_groups_xy = self.groups.copy()
        self.selected_groups_ribbons = self.groups.copy()
        self.selected_groups_walker = self.groups.copy()
        self.selected_groups_gk = self.groups.copy()

# endregion

        # region # === Bloque 8: Configuración de pestañas de gráficos ===
    def initUI(self):
        self.initMenu()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Inicializar selección de bases (solo Tamiz por defecto)
        self.current_db_selection = {"tamiz": True, "laser": False, "hybrid": False}

        # endregion
        
        # region # === 8.1: Pestaña Gráfico XY ===
        self.tab_xy = QWidget()
        v_xy = QVBoxLayout(self.tab_xy)
        h_controls = QHBoxLayout()

        self.btn_load_tamiz = QPushButton("Cargar archivo Excel de tamizado")
        self.btn_load_tamiz.clicked.connect(self.load_file)
        h_controls.addWidget(self.btn_load_tamiz)

        h_controls.addWidget(QLabel("Eje X:"))
        self.cmb_x = QComboBox();    h_controls.addWidget(self.cmb_x)
        h_controls.addWidget(QLabel("Eje Y:"))
        self.cmb_y = QComboBox();    h_controls.addWidget(self.cmb_y)

        self.btn_plot_xy = QPushButton("Graficar XY")
        self.btn_plot_xy.clicked.connect(self.plot_xy)
        h_controls.addWidget(self.btn_plot_xy)

        self.btn_select_db = QPushButton("Elegir bases a graficar")
        self.btn_select_db.clicked.connect(self.select_db_to_plot)
        h_controls.addWidget(self.btn_select_db)

        # ========================
        # NUEVOS CONTROLES
        # ========================
        # Opción para mostrar nombres de muestra
        self.chk_show_names = QCheckBox("Mostrar nombres")
        self.chk_show_names.setChecked(False)
        self.chk_show_names.toggled.connect(self.plot_xy)
        h_controls.addWidget(self.chk_show_names)

        # Botón para filtrar grupos (columna 4)
        self.btn_filter_groups = QPushButton("Filtrar grupos")
        self.btn_filter_groups.clicked.connect(self.select_groups_xy)
        h_controls.addWidget(self.btn_filter_groups)
        # ========================

        h_controls.addStretch()

        self.btn_export_xy = QPushButton("Exportar imagen")
        self.btn_export_xy.clicked.connect(lambda: self.export_canvas(self.canvas_xy))
        h_controls.addWidget(self.btn_export_xy)

        v_xy.addLayout(h_controls)

        # --- Encabezado de la consola de parámetros + botón de exportar ---
        hdr_console = QHBoxLayout()
        self.lbl_console = QLabel("Cálculo de parámetros granulométricos")
        self.lbl_console.setStyleSheet("font-weight: bold;")
        hdr_console.addWidget(self.lbl_console)
        hdr_console.addStretch()
        self.btn_export_params = QPushButton("Exportar parámetros granulométricos")
        self.btn_export_params.clicked.connect(self.export_params_to_excel)
        hdr_console.addWidget(self.btn_export_params)
        v_xy.addLayout(hdr_console)
        # --- fin encabezado ---

        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumHeight(190)
        v_xy.addWidget(self.txt_console)

        self.canvas_xy = FigureCanvas(plt.figure(figsize=(8, 4.5)))
        self.canvas_xy.setFixedHeight(450)
        v_xy.addWidget(self.canvas_xy)

        self.tabs.addTab(self.tab_xy, "Gráfico XY")

        # endregion
        
        # region # === 8.2: Pestaña Walker (1971 and 1983) ===
        self.tab_walker = QWidget()
        v_walker = QVBoxLayout(self.tab_walker)
        h_walker = QHBoxLayout()

        self.walker_groupbox = QGroupBox()
        self.walker_buttons  = QButtonGroup()
        walker_hbox = QHBoxLayout(self.walker_groupbox)
        for i, name in enumerate(WALKER_IMAGES.keys()):
            rb = QRadioButton(name)
            if i == 0: rb.setChecked(True)
            self.walker_buttons.addButton(rb)
            walker_hbox.addWidget(rb)
        self.walker_buttons.buttonClicked.connect(self.plot_walker)
        h_walker.addWidget(self.walker_groupbox)
        h_walker.addStretch()

        self.btn_export_walker = QPushButton("Exportar imagen")
        self.btn_export_walker.clicked.connect(lambda: self.export_canvas(self.canvas_walker))
        h_walker.addWidget(self.btn_export_walker)
        v_walker.addLayout(h_walker)

        self.btn_group_walker = QPushButton("Seleccionar grupos")
        self.btn_group_walker.clicked.connect(self.select_groups_walker)
        v_walker.addWidget(self.btn_group_walker)

        script_dir = get_script_dir()
        walker_img = mpimg.imread(os.path.join(script_dir, WALKER_IMAGES["All"]))
        img_h, img_w = walker_img.shape[:2]; dpi = 100
        self.canvas_walker = FixedSizeFigureCanvas(img_w/dpi, img_h/dpi, dpi)
        v_walker.addWidget(self.canvas_walker)

        self.tabs.addTab(self.tab_walker, "Walker (1971 and 1983)")

# endregion

        # region # === 8.3: Pestaña Gençalioğlu-Kuşcu et al 2007 ===
        self.tab_gk = QWidget()
        v_gk = QVBoxLayout(self.tab_gk)
        h_gk = QHBoxLayout()

        self.gk_groupbox = QGroupBox()
        self.gk_buttons   = QButtonGroup()
        gk_hbox = QHBoxLayout(self.gk_groupbox)
        for i, name in enumerate(GK_IMAGES.keys()):
            rb = QRadioButton(name)
            if i == 0: rb.setChecked(True)
            self.gk_buttons.addButton(rb)
            gk_hbox.addWidget(rb)
        self.gk_buttons.buttonClicked.connect(self.plot_gk)
        h_gk.addWidget(self.gk_groupbox)
        h_gk.addStretch()

        self.btn_export_gk = QPushButton("Exportar imagen")
        self.btn_export_gk.clicked.connect(lambda: self.export_canvas(self.canvas_gk))
        h_gk.addWidget(self.btn_export_gk)
        v_gk.addLayout(h_gk)

        self.btn_group_gk = QPushButton("Seleccionar grupos")
        self.btn_group_gk.clicked.connect(self.select_groups_gk)
        v_gk.addWidget(self.btn_group_gk)

        gk_img = mpimg.imread(os.path.join(script_dir, GK_IMAGES["All"]))
        img_h2, img_w2 = gk_img.shape[:2]; dpi = 100
        self.canvas_gk = FixedSizeFigureCanvas(img_w2/dpi, img_h2/dpi, dpi)
        v_gk.addWidget(self.canvas_gk)

        self.tabs.addTab(self.tab_gk, "Gençalioğlu-Kuşcu et al 2007")

# endregion


        # region # === 8.4: Pestaña Histograma ===
        self.tab_hist = QWidget()
        v_hist = QVBoxLayout(self.tab_hist)

        # Panel de controles de opciones (siempre visible)
        h_opts = QHBoxLayout()
        h_opts.addWidget(QLabel("Ancho de barra (%):"))
        self.spn_hist_width = QDoubleSpinBox()
        self.spn_hist_width.setRange(1, 100)
        self.spn_hist_width.setValue(70)
        self.spn_hist_width.setSingleStep(1)
        h_opts.addWidget(self.spn_hist_width)

        # Botón para cambiar colores
        self.btn_hist_colors = QPushButton("Colores de barras")
        h_opts.addWidget(self.btn_hist_colors)

        # Botón para editar labels de ejes y título
        self.btn_labels = QPushButton("Labels")
        h_opts.addWidget(self.btn_labels)

        # Checkboxes para visibilidad de componentes
        self.chk_hist = QCheckBox("Histograma (barras)")
        self.chk_hist.setChecked(True)
        h_opts.addWidget(self.chk_hist)

        self.chk_poly = QCheckBox("Polígono de frecuencia")
        self.chk_poly.setChecked(True)
        h_opts.addWidget(self.chk_poly)

        self.chk_cum = QCheckBox("Curva acumulativa")
        self.chk_cum.setChecked(True)
        h_opts.addWidget(self.chk_cum)

        # Botón para exportar
        self.btn_export_hist = QPushButton("Exportar gráfico")
        h_opts.addWidget(self.btn_export_hist)

        # === CAMBIO: Controles IA/GMM ===
        self.chk_gmm = QCheckBox("Detectar modos (IA)")
        self.chk_gmm.setChecked(False)
        h_opts.addWidget(self.chk_gmm)

        h_opts.addWidget(QLabel("k máx:"))
        self.spn_gmm_kmax = QSpinBox()
        self.spn_gmm_kmax.setRange(1, 6)
        self.spn_gmm_kmax.setValue(4)
        h_opts.addWidget(self.spn_gmm_kmax)

        self.btn_export_gmm = QPushButton("Exportar modos (CSV)")
        self.btn_export_gmm.setEnabled(False)
        h_opts.addWidget(self.btn_export_gmm)
        # === FIN CAMBIO ===

        h_opts.addStretch()
        v_hist.addLayout(h_opts)

        # Controles de selección de base y muestra
        h_sel = QHBoxLayout()
        h_sel.addWidget(QLabel("Base:"))
        self.cmb_hist_base = QComboBox()
        h_sel.addWidget(self.cmb_hist_base)

        h_sel.addWidget(QLabel("Muestra:"))
        self.cmb_hist_sample = QComboBox()
        self.cmb_hist_sample.setMinimumWidth(220)
        h_sel.addWidget(self.cmb_hist_sample)

        self.btn_plot_hist = QPushButton("Graficar Histograma")
        self.btn_plot_hist.clicked.connect(self.plot_histogram)
        h_sel.addWidget(self.btn_plot_hist)

        h_sel.addStretch()
        v_hist.addLayout(h_sel)

        # Canvas para el histograma
        self.canvas_hist = FigureCanvas(plt.figure(figsize=(6, 4)))
        v_hist.addWidget(self.canvas_hist)

        # Añadimos la pestaña al QTabWidget
        self.tabs.addTab(self.tab_hist, "Histograma")

        # === Valores por defecto para el histograma ===
        self.hist_bar_fill = "skyblue"
        self.hist_bar_edge = "black"
        self.poly_color    = "gray"    # o "#808080" para gris medio
        self.cum_color     = "black"   # color curva acumulativa

        # === Valores por defecto para los labels ===
        self.hist_title   = "Histograma"
        self.hist_xlabel  = "φ"
        self.hist_ylabel  = "wt (%)"
        self.hist_ylabel2 = "Frecuencia Acumulativa"

        # === Conexiones para panel de opciones ===
        self.spn_hist_width.valueChanged.connect(self.plot_histogram)
        self.btn_hist_colors.clicked.connect(self.choose_hist_colors)
        self.btn_labels.clicked.connect(self.choose_hist_labels)
        self.chk_hist.toggled.connect(self.plot_histogram)
        self.chk_poly.toggled.connect(self.plot_histogram)
        self.chk_cum.toggled.connect(self.plot_histogram)
        self.btn_export_hist.clicked.connect(lambda: self.export_canvas(self.canvas_hist))

        # === CAMBIO: Conexiones IA/GMM ===
        self.chk_gmm.toggled.connect(self.plot_histogram)
        self.spn_gmm_kmax.valueChanged.connect(self.plot_histogram)
        self.btn_export_gmm.clicked.connect(lambda: export_gmm_csv_from_state(self))
        # === FIN CAMBIO ===

        # === NUEVO: reaccionar al cambio de base para refrescar muestras ===
        self.cmb_hist_base.currentTextChanged.connect(self._on_hist_base_changed)

        # Inicializar combos
        self._update_hist_bases()
        self._on_hist_base_changed()  # llena muestras según la base actual

# endregion

    #  region #=== Bloque 9: Métodos de carga de archivos y bases de datos ===

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
            "Seleccionar archivo Excel de tamizado",
            "",
            "Excel files (*.xls *.xlsx)"
        )
        if not file:
            return

        try:
            df0 = pd.read_excel(file, header=0)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo leer el archivo:\n{e}")
            return

        df_norm, group_col_name = self._normalize_tamiz_formats(df0)
        if df_norm is None:
            QMessageBox.warning(
                self,
                "Formato no reconocido",
                "El Excel de tamiz debe cumplir uno de los dos formatos:\n"
                " • Formato 1 (4 columnas): φ | Sample | Weight | Group "
                "(φ múltiplos de 0.5; algún Sample con ≥2 filas; Group constante por Sample)\n"
                " • Formato 2 (mixto por columnas): df.columns = encabezados, primera fila = grupos "
                "(mayoría NO numérica) y datos desde la fila siguiente; si no hay fila de grupos válida "
                "se asumen 'Sin Grupo' y los datos empiezan desde la primera fila."
            )
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
        opts = ["mean", "median", "sigma", "skewness", "kurtosis"]
        self.cmb_x.clear(); self.cmb_y.clear()
        self.cmb_x.addItems(opts); self.cmb_y.addItems(opts)
        self.cmb_x.setCurrentText("mean"); self.cmb_y.setCurrentText("sigma")

        self.update_console()
        self.plot_xy()
        self.plot_walker()
        self.plot_gk()
        self.act_viewdb_tamiz.setEnabled(True)

        # También refrescar Histograma
        self._update_hist_samples()

    # ------------------------------- LÁSER (mapeo por nombre; orden libre) -------------------------------

    def _map_laser_columns_by_name(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve un DataFrame con EXACTAMENTE estas columnas y nombres:
          ['Sample', 'Diameter (μm)', '1 (%)', 'Group']
        Detecta columnas por encabezado (no por posición), tolerando español/inglés,
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
            raise ValueError(
                "No se pudieron detectar estas columnas por nombre: "
                + ", ".join(missing)
                + "\nEncabezados detectados: "
                + ", ".join(map(str, df_in.columns))
            )

        # DF canónico (se ignoran columnas extra)
        df = df_in[[sample_col, diam_col, perc_col, group_col]].copy()
        df.columns = ["Sample", "Diameter (μm)", "1 (%)", "Group"]
        return df

    def load_laser_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo Excel de laser",
            "",
            "Excel files (*.xls *.xlsx)"
        )
        if not file:
            return
        try:
            df = pd.read_excel(file)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo leer el archivo:\n{e}")
            return
        if df.shape[1] < 3:
            QMessageBox.warning(
                self,
                "Error",
                "El archivo de láser debe tener al menos tres columnas con encabezados reconocibles "
                "(muestra, diámetro en μm, porcentaje) y una de grupo/facies."
            )
            return

        # Paso φ
        dlg = QDialog(self)
        dlg.setWindowTitle("Paso de φ para láser")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("¿Procesamiento cada 1 o 0.5 φ?"))
        rb1 = QRadioButton("1.0 φ"); rb2 = QRadioButton("0.5 φ")
        rb1.setChecked(True)
        layout.addWidget(rb1); layout.addWidget(rb2)
        btn = QPushButton("Aceptar"); btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        if not dlg.exec_():
            return
        step = 1.0 if rb1.isChecked() else 0.5
        self.laser_step = step

        # Mapear columnas por nombre (ES/EN, orden libre)
        try:
            df_laser_raw = self._map_laser_columns_by_name(df)
        except ValueError as e:
            QMessageBox.warning(self, "Columnas no detectadas", str(e))
            return

        # Validar Group no vacío
        if df_laser_raw["Group"].isnull().any() or (df_laser_raw["Group"].astype(str).str.strip() == "").any():
            QMessageBox.warning(
                self,
                "Error",
                "Falta al menos un valor en la columna de grupo/facies (Group/Grupo/Filtro) del archivo de láser."
            )
            return

        # Agrupar por φ usando helper global (definido en Bloque 3)
        try:
            datos = agrupar_laser_por_phi(df_laser_raw, step)
        except Exception as e:
            QMessageBox.warning(self, "Error al procesar láser", f"Ocurrió un error al agrupar por φ:\n{e}")
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
        self.act_viewdb_laser.setEnabled(True)

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

        # Mostrar en consola
        txt = "\nParámetros granulométricos (Laser):\n\n"
        for _, r in self.param_results_laser.iterrows():
            txt += f"Sample: {r['Sample']} (Group: {r['Group']})\n"
            txt += f"  Mediana (φ50): {r['median']:.4f}\n"
            txt += f"  Sorting (σ): {r['sigma']:.4f}\n"
            txt += f"  Asimetría: {r['skewness']:.4f}\n"
            txt += f"  Curtosis: {r['kurtosis']:.4f}\n"
            txt += f"  Media (Mz): {r['mean']:.4f}\n\n"
        self.txt_console.append(txt)

        # Y aquí también refrescar Histograma
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
        header_row = pd.DataFrame([grp_map], index=[group_row_label])  # primera fila = "Grupo"

        # Apilar: primera fila = grupos; luego las filas de φ
        out = pd.concat([header_row, wide])

        # Primera columna "Phi": primer rótulo = 'Grupo', luego los φ
        out.insert(0, "Phi", [group_row_label] + list(wide.index))

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
            sample_col=self.df_data.columns[1],   # Sample (puede ser texto o numérico)
            weight_col=self.df_data.columns[2],   # Weight
            group_col=self.group_col,             # Group (nombre de la 4ta col cargada)
            group_row_label="Grupo"
        )
        dlg = DataBaseWindow("Base de datos de tamiz (vista Formato 2)", df_vista, self)
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
            group_row_label="Grupo"
        )
        dlg = DataBaseWindow("Base de datos láser (vista Formato 2)", df_vista, self)
        dlg.exec_()

    def show_hybrid_db_window(self):
        """
        Muestra la base HÍBRIDA en vista tipo Formato 2 usando 'Tamiz Sample' como nombre de muestra.
        """
        if self.df_hybrid is None:
            QMessageBox.information(self, "Base de datos híbrida", "Todavía no has combinado tamiz + láser.")
            return

        df_vista = self._make_format2_view(
            self.df_hybrid,
            phi_col="phi",
            sample_col="Tamiz Sample",
            weight_col="wt%",
            group_col="Group",
            group_row_label="Grupo"
        )
        dlg = DataBaseWindow("Base de datos tamiz + láser (vista Formato 2)", df_vista, self)
        dlg.exec_()
        
        # endregion


# region # === Bloque 10: Métodos de actualización de consola, gráficos y configuración de grupos ===

    # === Bloque 10.1: update_console ===
    def update_console(self):
        txt = ""
        # Tamiz
        if hasattr(self, "param_results") and self.param_results is not None:
            txt += f"Parámetros granulométricos Tamiz ({self.current_method}):\n\n"
            for _, r in self.param_results.iterrows():
                clave = f"{r[self.group_col]} (Tamiz)"
                txt += f"Sample: {r['Sample']} ({clave})\n"
                txt += f"  Mediana (φ50): {r['median']:.4f}\n"
                txt += f"  Sorting (σ): {r['sigma']:.4f}\n"
                txt += f"  Asimetría: {r['skewness']:.4f}\n"
                txt += f"  Curtosis: {r['kurtosis']:.4f}\n"
                txt += f"  Media (Mz): {r['mean']:.4f}\n\n"
        # Laser
        if hasattr(self, "param_results_laser") and self.param_results_laser is not None:
            txt += f"Parámetros granulométricos Laser ({self.current_method}):\n\n"
            for _, r in self.param_results_laser.iterrows():
                clave = f"{r['Group']} (Laser)"
                txt += f"Sample: {r['Sample']} ({clave})\n"
                txt += f"  Mediana (φ50): {r['median']:.4f}\n"
                txt += f"  Sorting (σ): {r['sigma']:.4f}\n"
                txt += f"  Asimetría: {r['skewness']:.4f}\n"
                txt += f"  Curtosis: {r['kurtosis']:.4f}\n"
                txt += f"  Media (Mz): {r['mean']:.4f}\n\n"
        # Híbrido
        if hasattr(self, "param_results_hybrid") and self.param_results_hybrid is not None:
            txt += f"Parámetros granulométricos Híbrido ({self.current_method}):\n\n"
            for _, r in self.param_results_hybrid.iterrows():
                clave = f"{r[self.group_col]} (Híbrido)"
                txt += f"Sample: {r['Sample']} ({clave})\n"
                txt += f"  Mediana (φ50): {r['median']:.4f}\n"
                txt += f"  Sorting (σ): {r['sigma']:.4f}\n"
                txt += f"  Asimetría: {r['skewness']:.4f}\n"
                txt += f"  Curtosis: {r['kurtosis']:.4f}\n"
                txt += f"  Media (Mz): {r['mean']:.4f}\n\n"
        self.txt_console.setPlainText(txt)
        
        # endregion

    # region # === Bloque 10.2: plot_xy ===
    def plot_xy(self):
        x = self.cmb_x.currentText()
        y = self.cmb_y.currentText()
        fig = self.canvas_xy.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        # Preparar bases y claves compuestas
        bases = []
        if self.current_db_selection.get("tamiz") and hasattr(self, "param_results"):
            bases.append(("Tamiz", self.param_results, self.group_col))
        if self.current_db_selection.get("laser") and hasattr(self, "param_results_laser"):
            bases.append(("Laser", self.param_results_laser, "Group"))
        if self.current_db_selection.get("hybrid") and hasattr(self, "param_results_hybrid"):
            bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

        for name, df_params, group_col in bases:
            if df_params is None:
                continue
            for grp in df_params[group_col].unique():
                clave = f"{grp} ({name})"
                if clave not in self.selected_groups_xy:
                    continue
                sub = df_params[df_params[group_col] == grp]
                col = self.group_colors.get(clave, "#888888")
                if clave in self.selected_groups_ribbons and len(sub) > 2:
                    plot_ribbon(ax, sub[x], sub[y], color=col, alpha=0.18, width=0.07, smooth=10)
                ax.scatter(sub[x], sub[y], color=col, label=clave, s=50, edgecolor='k', lw=1.2, zorder=3)
                if self.chk_show_names.isChecked():
                    for _, row in sub.iterrows():
                        ax.annotate(
                            str(row['Sample']),
                            (row[x], row[y]),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8, alpha=0.7, color=col
                        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.1)
        ax.spines['left'].set_linewidth(1.1)
        ax.tick_params(length=6, width=1.1, direction='out', colors='black')
        ax.set_xlabel(SYMBOLS[x], fontsize=16)
        ax.set_ylabel(SYMBOLS[y], fontsize=16)
        ax.set_title(f"{SYMBOLS[y]} vs {SYMBOLS[x]}", fontsize=15)

        # Leyenda solo si hay artistas con etiqueta
        handles, labels = ax.get_legend_handles_labels()
        have_legend = len(handles) > 0
        if have_legend:
            legend = ax.legend(
                handles, labels,
                title="Group (Base)", frameon=False,
                loc='center left', bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.
            )
            for t in legend.get_texts():
                t.set_color('black')
            legend.get_title().set_color('black')

        # Márgenes (más espacio sólo si hay leyenda)
        fig.subplots_adjust(left=0.08, right=(0.72 if have_legend else 0.96), bottom=0.12, top=0.95)
        self.canvas_xy.draw()
        
        # endregion

    # region # === Bloque 10.3: plot_walker ===
    def plot_walker(self):
        x = "median"
        y = "sigma"
        fig = self.canvas_walker.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        key = self.walker_buttons.checkedButton().text()
        img = mpimg.imread(os.path.join(get_script_dir(), WALKER_IMAGES[key]))
        ax.imshow(img, extent=[-4, 6, 0, 5], aspect='auto', zorder=1)

        bases = []
        if self.current_db_selection.get("tamiz") and hasattr(self, "param_results"):
            bases.append(("Tamiz", self.param_results, self.group_col))
        if self.current_db_selection.get("laser") and hasattr(self, "param_results_laser"):
            bases.append(("Laser", self.param_results_laser, "Group"))
        if self.current_db_selection.get("hybrid") and hasattr(self, "param_results_hybrid"):
            bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

        for name, df_params, group_col in bases:
            if df_params is None:
                continue
            for grp in df_params[group_col].unique():
                clave = f"{grp} ({name})"
                if clave not in self.selected_groups_walker:
                    continue
                sub = df_params[df_params[group_col] == grp]
                col = self.group_colors.get(clave, "#888888")
                ax.scatter(sub[x], sub[y], color=col, label=clave, s=50, edgecolor='k', lw=1.2, zorder=10)

        ax.set_xlim(-4, 6)
        ax.set_ylim(0, 5)
        ax.set_xlabel(SYMBOLS["median"], fontsize=16)
        ax.set_ylabel(SYMBOLS["sigma"], fontsize=16)
        ax.set_xticks(np.arange(-4, 7, 2))
        ax.set_yticks(np.arange(0, 6, 1))
        ax.tick_params(length=8, width=1.3, direction='out', colors='black')
        for x_ in np.arange(-4, 7, 2):
            ax.plot([x_, x_], [0 - 0.12, 0], color='k', lw=1.1, zorder=30)
        for y_ in np.arange(0, 6, 1):
            ax.plot([-4 - 0.15, -4], [y_, y_], color='k', lw=1.1, zorder=30)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        # Leyenda
        handles, labels = ax.get_legend_handles_labels()
        have_legend = len(handles) > 0
        if have_legend:
            legend = ax.legend(
                handles, labels,
                title="Group (Base)", frameon=False,
                loc='center left', bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.
            )
            for t in legend.get_texts():
                t.set_color('black')
            legend.get_title().set_color('black')

        fig.subplots_adjust(left=0.08, right=(0.72 if have_legend else 0.96), bottom=0.20, top=0.97)
        self.canvas_walker.draw()
        
        # endregion

    # region # === Bloque 10.4: plot_gk ===
    def plot_gk(self):
        x = "median"
        y = "sigma"
        fig = self.canvas_gk.figure
        fig.clf()
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        key = self.gk_buttons.checkedButton().text()
        img = mpimg.imread(os.path.join(get_script_dir(), GK_IMAGES[key]))
        ax.imshow(img, extent=[-4, 6, 0, 5], aspect='auto', zorder=1)

        bases = []
        if self.current_db_selection.get("tamiz") and hasattr(self, "param_results"):
            bases.append(("Tamiz", self.param_results, self.group_col))
        if self.current_db_selection.get("laser") and hasattr(self, "param_results_laser"):
            bases.append(("Laser", self.param_results_laser, "Group"))
        if self.current_db_selection.get("hybrid") and hasattr(self, "param_results_hybrid"):
            bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

        for name, df_params, group_col in bases:
            if df_params is None:
                continue
            for grp in df_params[group_col].unique():
                clave = f"{grp} ({name})"
                if clave not in self.selected_groups_gk:
                    continue
                sub = df_params[df_params[group_col] == grp]
                col = self.group_colors.get(clave, "#888888")
                ax.scatter(sub[x], sub[y], color=col, label=clave, s=50, edgecolor='k', lw=1.2, zorder=10)

        ax.set_xlim(-4, 6)
        ax.set_ylim(0, 5)
        ax.set_xlabel(SYMBOLS["median"], fontsize=16)
        ax.set_ylabel(SYMBOLS["sigma"], fontsize=16)
        ax.set_xticks(np.arange(-4, 7, 2))
        ax.set_yticks(np.arange(0, 6, 1))
        ax.tick_params(length=8, width=1.3, direction='out', colors='black')
        for x_ in np.arange(-4, 7, 2):
            ax.plot([x_, x_], [0 - 0.12, 0], color='k', lw=1.1, zorder=30)
        for y_ in np.arange(0, 6, 1):
            ax.plot([-4 - 0.15, -4], [y_, y_], color='k', lw=1.1, zorder=30)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        # Leyenda
        handles, labels = ax.get_legend_handles_labels()
        have_legend = len(handles) > 0
        if have_legend:
            legend = ax.legend(
                handles, labels,
                title="Group (Base)", frameon=False,
                loc='center left', bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.
            )
            for t in legend.get_texts():
                t.set_color('black')
            legend.get_title().set_color('black')

        fig.subplots_adjust(left=0.08, right=(0.72 if have_legend else 0.96), bottom=0.20, top=0.97)
        self.canvas_gk.draw()
        
        # endregion

    # region # === Bloque 10.5: Configuración de colores y visibilidad de grupos ===
    def edit_colors(self):
        if not self.groups:
            return
        dlg = ColorDialog(self.groups, self.group_colors, self)
        if dlg.exec_():
            self.group_colors = dlg.colors
            self.plot_xy()
            self.plot_walker()
            self.plot_gk()

    def select_ribbons(self):
        """
        Ver/ocultar sombras (ribbons) en el XY por grupo.
        NOTA: esto solo afecta a la pestaña XY.
        """
        if not self.groups:
            return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_ribbons, "Ver/ocultar sombras (XY)", self)
        if dlg.exec_():
            self.selected_groups_ribbons = dlg.get_selected()
            self.plot_xy()

    def select_groups_all(self):
        """
        Selecciona/filtra grupos y aplica la misma visibilidad
        a TODAS las pestañas (XY, Walker, GK).
        """
        if not self.groups:
            return
        seed = self.selected_groups_xy if self.selected_groups_xy else self.groups
        dlg = GroupSelectDialog(self.groups, seed, "Grupos visibles (todas las pestañas)", self)
        if dlg.exec_():
            sel = dlg.get_selected()
            self.selected_groups_xy = sel.copy()
            self.selected_groups_walker = sel.copy()
            self.selected_groups_gk = sel.copy()
            self.plot_xy(); self.plot_walker(); self.plot_gk()

    def select_groups_walker(self):
        if not self.groups:
            return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_walker, "Seleccionar grupos (Walker)", self)
        if dlg.exec_():
            self.selected_groups_walker = dlg.get_selected()
            self.plot_walker()

    def select_groups_gk(self):
        if not self.groups:
            return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_gk, "Seleccionar grupos (GK)", self)
        if dlg.exec_():
            self.selected_groups_gk = dlg.get_selected()
            self.plot_gk()
            
            # endregion

    # region # === Bloque 10.5.1: Elegir bases a graficar (solo las cargadas) ===
    def select_db_to_plot(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Elegir bases a graficar")
        layout = QVBoxLayout(dlg)

        items = []
        if self.param_results is not None and not self.param_results.empty:
            items.append(("Tamiz",  "tamiz",  self.current_db_selection.get("tamiz", True)))
        if self.param_results_laser is not None and not self.param_results_laser.empty:
            items.append(("Laser",  "laser",  self.current_db_selection.get("laser", False)))
        if self.param_results_hybrid is not None and not self.param_results_hybrid.empty:
            items.append(("Tamiz + Laser", "hybrid", self.current_db_selection.get("hybrid", False)))

        if not items:
            QMessageBox.information(self, "Bases", "No hay bases cargadas para graficar.")
            return

        cbs = {}
        for label, key, checked in items:
            cb = QCheckBox(label); cb.setChecked(checked)
            layout.addWidget(cb); cbs[key] = cb

        btn = QPushButton("Aceptar"); btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)

        if dlg.exec_():
            self.current_db_selection = {"tamiz": False, "laser": False, "hybrid": False}
            for key, cb in cbs.items():
                self.current_db_selection[key] = cb.isChecked()
            self.plot_xy(); self.plot_walker(); self.plot_gk()
            
            # endregion

    # region # === Bloque 10.5.2: Filtro de grupos para XY ===
    def select_groups_xy(self):
        if not self.groups:
            return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_xy, "Seleccionar grupos (XY)", self)
        if dlg.exec_():
            self.selected_groups_xy = dlg.get_selected()
            self.plot_xy()
            
            # endregion

    # region # === Bloque 10.6: Métodos para la pestaña Histograma ===
    def _update_hist_bases(self):
        """Rellena el combo de bases según lo cargado."""
        bases = []
        if self.df_data   is not None: bases.append("Tamiz")
        if self.df_laser  is not None: bases.append("Laser")
        if self.df_hybrid is not None: bases.append("Híbrido")

        current = self.cmb_hist_base.currentText()
        self.cmb_hist_base.blockSignals(True)
        self.cmb_hist_base.clear()
        self.cmb_hist_base.addItems(bases)
        if current in bases:
            self.cmb_hist_base.setCurrentText(current)
        self.cmb_hist_base.blockSignals(False)

    def _fill_hist_samples_for_base(self, base: str):
        """Llena el combo de muestras para la base elegida."""
        self.cmb_hist_sample.blockSignals(True)
        self.cmb_hist_sample.clear()

        if not base:
            self.cmb_hist_sample.blockSignals(False)
            return

        if base == "Tamiz":
            df, col = self.df_data, (self.df_data.columns[1] if self.df_data is not None else None)
        elif base == "Laser":
            df, col = self.df_laser, "Sample"
        else:  # Híbrido
            df, col = self.df_hybrid, "Tamiz Sample"

        samples = []
        if df is not None and col is not None and col in df.columns:
            samples = [str(s) for s in sorted(df[col].unique())]

        self.cmb_hist_sample.addItems(samples)
        if samples:
            self.cmb_hist_sample.setCurrentIndex(0)
        self.cmb_hist_sample.blockSignals(False)

    def _update_hist_samples(self):
        """Actualiza bases y luego refresca muestras de la base actual."""
        self._update_hist_bases()
        self._on_hist_base_changed()

    def _on_hist_base_changed(self, *_):
        """Slot: cuando cambia la base, refrescar listado de muestras."""
        base = self.cmb_hist_base.currentText()
        self._fill_hist_samples_for_base(base)

    def plot_histogram(self):
        """Dibuja histograma, polígono de frecuencia y/o curva acumulativa."""
        base   = self.cmb_hist_base.currentText()
        sample = self.cmb_hist_sample.currentText()

        # seleccionar DataFrame y columnas según base
        if   base == "Tamiz":
            df, phi_col, wt_col, samp_col = (
                self.df_data,
                self.df_data.columns[0],
                self.df_data.columns[2],
                self.df_data.columns[1]
            )
        elif base == "Laser":
            df, phi_col, wt_col, samp_col = (
                self.df_laser,
                "phi",
                "Wt",
                "Sample"
            )
        else:  # Híbrido
            df, phi_col, wt_col, samp_col = (
                self.df_hybrid,
                "phi",
                "wt%",
                "Tamiz Sample"
            )

        # validar
        if df is None or not sample or sample not in df[samp_col].astype(str).values:
            QMessageBox.warning(self, "Error", f"No hay datos para la muestra {sample}.")
            return
        sub = df[df[samp_col].astype(str) == sample]
        if sub.empty:
            QMessageBox.warning(self, "Error", f"No hay datos para la muestra {sample}.")
            return

        # preparar figura
        fig = self.canvas_hist.figure
        fig.clf()
        ax  = fig.add_subplot(111)

        x = np.array(sub[phi_col], dtype=float)
        y = np.array(sub[wt_col],   dtype=float)

        # paso real de los datos (detecta 1.0φ, 0.5φ, etc.)
        xu = np.sort(np.unique(x))
        spacing = float(np.min(np.diff(xu))) if len(xu) > 1 else 1.0

        # ancho de barra en función del porcentaje
        width = spacing * (self.spn_hist_width.value() / 100.0)

        # barras
        if self.chk_hist.isChecked():
            ax.bar(
                x, y,
                width=width,
                color=self.hist_bar_fill,
                edgecolor=self.hist_bar_edge,
                zorder=10,
                align="center"
            )

        # polígono (color independiente)
        if self.chk_poly.isChecked():
            ax.plot(x, y, "-o", color=self.poly_color, zorder=20)

        # acumulativa (color independiente)
        ax2 = None
        if self.chk_cum.isChecked():
            ax2 = ax.twinx()
            y_ac = np.cumsum(y)
            y_ac = (100 * y_ac / y_ac[-1]) if y_ac[-1] != 0 else y_ac
            ax2.plot(x, y_ac, "-o", color=self.cum_color, zorder=30)
            ylabel2 = getattr(self, "hist_ylabel2", "Frecuencia Acumulativa")
            ax2.set_ylabel(ylabel2, fontsize=14)
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='y', labelsize=12)

        # === Ticks de φ con el MISMO paso que los datos (1.0 o 0.5, etc.) ===
        start = np.floor(np.min(x) / spacing) * spacing
        end   = np.ceil (np.max(x) / spacing) * spacing
        n = int(round((end - start) / spacing)) + 1
        phi_ticks = np.round(start + np.arange(n) * spacing, 6)
        decimals = 0 if np.isclose(spacing, 1.0) else 1
        ax.set_xticks(phi_ticks)
        ax.set_xticklabels([f"{v:.{decimals}f}" if decimals else f"{int(round(v))}" for v in phi_ticks])

        # eje secundario inferior con tamaños en µm, alineado con φ
        def um(phi): return 1000 * (2 ** -float(phi))
        sec = ax.twiny()
        sec.set_xlim(ax.get_xlim())
        sec.set_xticks(phi_ticks)
        sec.set_xticklabels([str(int(round(um(v)))) for v in phi_ticks])
        sec.xaxis.set_ticks_position("bottom")
        sec.xaxis.set_label_position("bottom")
        sec.spines["bottom"].set_position(("outward", 40))
        sec.set_xlabel("Tamaño (µm)", fontsize=13)
        sec.tick_params(axis='x', labelsize=12)

        # labels personalizados
        xlabel = getattr(self, "hist_xlabel", "φ")
        ylabel = getattr(self, "hist_ylabel", "wt (%)")
        title  = getattr(self, "hist_title", sample if sample else "Histograma")

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title if title else str(sample), fontsize=16, weight="bold", pad=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # === IA/GMM – overlay y guardado de estado ===
        if hasattr(self, "chk_gmm") and self.chk_gmm.isChecked():
            if not _HAS_SKLEARN:
                QMessageBox.information(self, "IA/GMM", "Instala 'scikit-learn' para usar la detección de modos.")
            else:
                try:
                    kmax = int(self.spn_gmm_kmax.value()) if hasattr(self, "spn_gmm_kmax") else 4
                    res = gmm_detect_modes(x, y, kmax=kmax, random_state=0)

                    # Curvas: mezcla total y componentes
                    xs = res["xs"]
                    ax.plot(xs, res["mix_curve"], linewidth=2.0, alpha=0.9, zorder=35)
                    for mu, sd, wt_frac, comp in zip(res["means"], res["stds"], res["weights"], res["components"]):
                        ax.plot(xs, comp, linestyle="--", linewidth=1.6, alpha=0.9, zorder=34)
                        ax.axvline(mu, linestyle=":", linewidth=1.2, alpha=0.8)

                    # Título con resumen de modas
                    modos_txt = ";  ".join([f"{mu:.2f}ϕ ({p:.0f}%)" for mu, p in zip(res["means"], 100*res["weights"])])
                    ax.set_title((title if title else str(sample)) + f"  |  Modos (k={res['k']}): {modos_txt}",
                                 fontsize=16, weight="bold", pad=12)

                    # Guardar estado para exportación
                    self._gmm_last = {
                        "base": base,
                        "sample": sample,
                        "k": int(res["k"]),
                        "means_phi": [float(v) for v in res["means"]],
                        "std_phi":   [float(v) for v in res["stds"]],
                        "weights":   [float(v) for v in res["weights"]],
                        "bic": float(res["bic"]),
                        "rmse": float(res["rmse"])
                    }
                    if hasattr(self, "btn_export_gmm"):
                        self.btn_export_gmm.setEnabled(True)
                except Exception as e:
                    QMessageBox.warning(self, "IA/GMM", f"Fallo al detectar modos: {e}")
        else:
            if hasattr(self, "btn_export_gmm"):
                self.btn_export_gmm.setEnabled(False)
            self._gmm_last = None
        # === FIN IA/GMM ===

        fig.tight_layout()
        self.canvas_hist.draw()
        
        # endregion

    # region # === Bloque 10.7: Colores del histograma ===
    def choose_hist_colors(self):
        """Seleccionar colores: relleno, borde, polígono y acumulativa."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Colores del histograma")
        layout = QVBoxLayout(dlg)

        btn_fill = QPushButton("Color de relleno de barras")
        lbl_fill = QLabel(f"Actual: {getattr(self, 'hist_bar_fill', 'skyblue')}")
        def set_fill():
            color = QColorDialog.getColor()
            if color.isValid():
                self.hist_bar_fill = color.name()
                lbl_fill.setText(f"Actual: {self.hist_bar_fill}")
        btn_fill.clicked.connect(set_fill)

        btn_edge = QPushButton("Color de borde de barras")
        lbl_edge = QLabel(f"Actual: {getattr(self, 'hist_bar_edge', 'black')}")
        def set_edge():
            color = QColorDialog.getColor()
            if color.isValid():
                self.hist_bar_edge = color.name()
                lbl_edge.setText(f"Actual: {self.hist_bar_edge}")
        btn_edge.clicked.connect(set_edge)

        btn_poly = QPushButton("Color del polígono de frecuencia")
        lbl_poly = QLabel(f"Actual: {getattr(self, 'poly_color', 'gray')}")
        def set_poly():
            color = QColorDialog.getColor()
            if color.isValid():
                self.poly_color = color.name()
                lbl_poly.setText(f"Actual: {self.poly_color}")
        btn_poly.clicked.connect(set_poly)

        btn_cum = QPushButton("Color de la curva acumulativa")
        lbl_cum = QLabel(f"Actual: {getattr(self, 'cum_color', 'black')}")
        def set_cum():
            color = QColorDialog.getColor()
            if color.isValid():
                self.cum_color = color.name()
                lbl_cum.setText(f"Actual: {self.cum_color}")
        btn_cum.clicked.connect(set_cum)

        for w in (btn_fill, lbl_fill, btn_edge, lbl_edge, btn_poly, lbl_poly, btn_cum, lbl_cum):
            layout.addWidget(w)
        btn_ok = QPushButton("Aceptar"); btn_ok.clicked.connect(dlg.accept)
        layout.addWidget(btn_ok)

        if dlg.exec_():
            self.plot_histogram()
            
            # endregion

    # region # === Bloque 10.8: Labels del histograma ===
    def choose_hist_labels(self):
        """Diálogo para editar título y etiquetas de ejes del histograma."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Editar Labels del Histograma")
        form = QFormLayout(dlg)

        if not hasattr(self, "hist_title"):   self.hist_title = ""
        if not hasattr(self, "hist_xlabel"):  self.hist_xlabel = ""
        if not hasattr(self, "hist_ylabel"):  self.hist_ylabel = ""
        if not hasattr(self, "hist_ylabel2"): self.hist_ylabel2 = ""

        txt_title  = QLineEdit(self.hist_title)
        txt_xlabel = QLineEdit(self.hist_xlabel)
        txt_ylabel = QLineEdit(self.hist_ylabel)
        txt_y2     = QLineEdit(self.hist_ylabel2)

        form.addRow("Título:", txt_title)
        form.addRow("Eje X (φ):", txt_xlabel)
        form.addRow("Eje Y (wt %):", txt_ylabel)
        form.addRow("Eje Y₂ (Acumulativa):", txt_y2)

        btn_ok = QPushButton("Aceptar"); btn_ok.clicked.connect(dlg.accept)
        form.addRow(btn_ok)

        if dlg.exec_():
            self.hist_title   = txt_title.text()
            self.hist_xlabel  = txt_xlabel.text()
            self.hist_ylabel  = txt_ylabel.text()
            self.hist_ylabel2 = txt_y2.text()
            self.plot_histogram()

# endregion


    # region # === Bloque 10.9: Ocultar/mostrar curvas del histograma ===
    def toggle_hist_curve(self, show_hist: bool, show_acc: bool, show_poly: bool):
        self.show_histogram = show_hist
        self.show_cumulative = show_acc
        self.show_frequency_polygon = show_poly
        self.plot_histogram()

# endregion

# region # === Bloque 11: Métodos de configuración y exportación ===
    def choose_method(self):
        """
        Cambia el método de cálculo (Folk & Ward, Inman, etc.) y
        **recalcula** los parámetros para todas las bases ya cargadas,
        sin volver a pedir los archivos.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Elegir método de cálculo")
        layout = QVBoxLayout(dlg)

        combo = QComboBox()
        combo.addItems(METHODS.keys())

        # método actual → texto visible
        try:
            current_text = [k for k, v in METHODS.items() if v == self.current_method][0]
        except Exception:
            current_text = list(METHODS.keys())[0]
        combo.setCurrentText(current_text)

        layout.addWidget(QLabel("Método de cálculo estadístico:"))
        layout.addWidget(combo)

        btn_ok = QPushButton("Aceptar")
        btn_ok.clicked.connect(dlg.accept)
        layout.addWidget(btn_ok)

        if not dlg.exec_():
            return

        # 1) Guardar método seleccionado
        self.current_method = METHODS[combo.currentText()]

        # 2) Recalcular parámetros para TODAS las bases presentes
        # --- Tamiz ---
        if self.df_data is not None and not self.df_data.empty:
            params = []
            phi_col   = self.df_data.columns[0]
            samp_col  = self.df_data.columns[1]
            wt_col    = self.df_data.columns[2]
            group_col = self.group_col
            for sample, grp in self.df_data.groupby(samp_col):
                phi     = grp[phi_col].tolist()
                weights = grp[wt_col].tolist()
                gval    = grp.iloc[0][group_col]
                p       = calculate_parameters(phi, weights, self.current_method)
                params.append({"Sample": sample, group_col: gval, **p})
            self.param_results = pd.DataFrame(params)

        # --- Láser ---
        if self.df_laser is not None and not self.df_laser.empty:
            params_l = []
            for sample, grp in self.df_laser.groupby("Sample"):
                phi_vals = grp["phi"].tolist()
                wt_vals  = grp["Wt"].tolist()
                gval     = grp["Group"].iloc[0]
                p        = calculate_parameters(phi_vals, wt_vals, self.current_method)
                params_l.append({"Sample": sample, "Group": gval, **p})
            self.param_results_laser = pd.DataFrame(params_l)

        # --- Híbrido ---
        if self.df_hybrid is not None and not self.df_hybrid.empty:
            params_h = []
            for sample, grp in self.df_hybrid.groupby("Tamiz Sample"):
                phi_vals = grp["phi"].tolist()
                wt_vals  = grp["wt%"].tolist()
                gval     = grp["Group"].iloc[0]
                p        = calculate_parameters(phi_vals, wt_vals, self.current_method)
                params_h.append({"Sample": sample, self.group_col: gval, **p})
            self.param_results_hybrid = pd.DataFrame(params_h)

        # 3) Actualizar grupos/colores, consola y gráficos
        self._update_all_groups_and_colors()
        self.update_console()
        self.plot_xy()
        self.plot_walker()
        self.plot_gk()
        # (la pestaña Histograma se recalcula cuando el usuario presiona su botón)

    def choose_theme(self):
        """Diálogo para alternar entre tema oscuro o claro."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Estilo de interfaz")
        layout = QVBoxLayout(dlg)
        combo = QComboBox()
        combo.addItems(["Oscuro (Fusion dark)", "Claro (Fusion light)"])
        combo.setCurrentIndex(0 if self.theme == "dark" else 1)
        layout.addWidget(QLabel("Selecciona el estilo de la interfaz:"))
        layout.addWidget(combo)
        btn_ok = QPushButton("Aceptar"); btn_ok.clicked.connect(dlg.accept)
        layout.addWidget(btn_ok)
        if dlg.exec_():
            idx = combo.currentIndex()
            if idx == 0 and self.theme != "dark":
                self.theme = "dark"; qApp.setStyleSheet(DARK_STYLESHEET)
            elif idx == 1 and self.theme != "light":
                self.theme = "light"; qApp.setStyleSheet(LIGHT_STYLESHEET)

    def export_canvas(self, canvas):
        """
        Abre un diálogo para guardar la figura.
        Formatos soportados: PNG, TIFF, SVG, PDF.
        """
        file, ext = QFileDialog.getSaveFileName(
            self,
            "Guardar figura",
            "",
            "PNG (*.png);;TIFF (*.tif);;SVG (*.svg);;PDF (*.pdf)"
        )
        if not file:
            return

        if ext == "PNG (*.png)" and not file.lower().endswith(".png"):
            file += ".png"
        elif ext == "TIFF (*.tif)" and not file.lower().endswith(".tif"):
            file += ".tif"
        elif ext == "SVG (*.svg)" and not file.lower().endswith(".svg"):
            file += ".svg"
        elif ext == "PDF (*.pdf)" and not file.lower().endswith(".pdf"):
            file += ".pdf"

        try:
            canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white")
        except ValueError as e:
            QMessageBox.warning(self, "Error al guardar", f"No se pudo guardar la figura en ese formato:\n{e}")

    def save_current_tab(self):
        """Llama a export_canvas según la pestaña activa."""
        idx = self.tabs.currentIndex()
        if   idx == 0: self.export_canvas(self.canvas_xy)
        elif idx == 1: self.export_canvas(self.canvas_walker)
        elif idx == 2: self.export_canvas(self.canvas_gk)
        elif idx == 3: self.export_canvas(self.canvas_hist)
        else:
            pass

# endregion

# region # === Bloque 12: Métodos de bins, procesamiento y combinación tamiz–láser ===
    def obtener_bins_tamiz(self, step=None):
        """Genera bins φ para la base de tamiz."""
        phi = self.df_data.iloc[:, 0].values
        mn, mx = np.floor(phi.min()), np.ceil(phi.max())
        if step is None:
            step = np.min(np.diff(np.sort(np.unique(phi)))) if len(phi) > 1 else 1.0
        bins = np.arange(mn, mx + step, step)
        bins = np.round(bins / step) * step
        return bins

    def procesar_laser(self, df_laser, bins):
        """Prepara la lista (phi, Wt) de láser agrupada previamente."""
        required_cols = ["phi", "Sample", "Wt"]
        if not all(col in df_laser.columns for col in required_cols):
            raise ValueError(f"[procesar_laser] Columnas faltantes: {df_laser.columns}")
        phi_vals = df_laser["phi"].astype(float).values
        wt_vals  = df_laser["Wt"].astype(float).values
        return list(zip(phi_vals, wt_vals))

    def combinar_metodo_1(self, tamiz, laser_phi, phi_extraido):
        """
        Método 1: Hibridar datos eliminando los φ ≥ φ_extraído.
        1) Suma tamiz para φ ≥ φ_extraído -> T_tamiz
        2) Filtra tamiz (φ < umbral) y láser (φ ≥ umbral)
        3) Escala láser para que su suma sea T_tamiz
        4) Concatena y ordena resultado
        """
        T_tamiz = sum(w for p, w in tamiz if p >= phi_extraido)
        tamiz_filtrado  = [(p, w) for p, w in tamiz      if p <  phi_extraido]
        laser_filtrado  = [(p, w) for p, w in laser_phi  if p >= phi_extraido]
        T_laser = sum(w for p, w in laser_filtrado) or 1.0
        factor  = T_tamiz / T_laser
        laser_norm = [(p, w * factor) for p, w in laser_filtrado]
        combinado = tamiz_filtrado + laser_norm
        return sorted(combinado, key=lambda x: x[0])

    def combinar_metodo_2(self, tamiz, laser_phi, phi_extraido):
        """Método 2: Tamiz intacto (φ<umbral), luego suma tamiz+láser y escala al 100%."""
        part1 = [(p, w) for p, w in tamiz if p < phi_extraido]
        X1    = sum(w for _, w in part1)
        pts   = sorted(set(p for p, _ in tamiz if p >= phi_extraido) |
                       set(p for p, _ in laser_phi if p >= phi_extraido))
        part2 = [(p,
                  (next((w for q, w in tamiz     if q==p), 0) +
                   next((w for q, w in laser_phi if q==p), 0)))
                 for p in pts]
        X2 = sum(w for _, w in part2)
        if X2 == 0:
            return part1
        fac    = (100.0 - X1) / X2
        scaled = [(p, w * fac) for p, w in part2]
        return sorted(part1 + scaled, key=lambda x: x[0])

    def combinar_tamiz_laser(self):
        """
        Ventana de emparejamiento y creación de base híbrida.
        Lee φ_extraído de la tabla MatchDialog para cada par muestra.
        """
        if self.df_data is None or self.df_laser is None:
            QMessageBox.warning(self, "Error", "Carga ambas bases de datos primero.")
            return

        dlg_map = MatchDialog(self.df_data, self.df_laser, self)
        if not dlg_map.exec_():
            return

        mapping = []
        for row in range(dlg_map.table.rowCount()):
            sample_t = dlg_map.table.item(row, 0).text()
            sample_l = dlg_map.table.item(row, 1).text()
            phi_extraido = dlg_map.table.cellWidget(row, 2).value()
            mapping.append((sample_t, sample_l, phi_extraido))
        if not mapping:
            QMessageBox.warning(self, "Error", "No se emparejaron muestras.")
            return

        # Selección de método
        dlg2 = QDialog(self); dlg2.setWindowTitle("Método de combinación")
        ly = QVBoxLayout(dlg2); mg = QButtonGroup(dlg2)
        rb1 = QRadioButton("Reemplazar fracción fina por láser escalado")
        rb2 = QRadioButton("Reemplazar fracción fina por láser + tamiz fino escalado")
        rb1.setChecked(True); mg.addButton(rb1); mg.addButton(rb2)
        ly.addWidget(rb1); ly.addWidget(rb2)
        btn_ok = QPushButton("Combinar"); btn_ok.clicked.connect(dlg2.accept); ly.addWidget(btn_ok)
        if not dlg2.exec_():
            return
        metodo = 1 if rb1.isChecked() else 2

        resultados = []
        for sample_t, sample_l, phi_extraido in mapping:
            df_t = self.df_data[self.df_data.iloc[:,1]==sample_t]
            tamiz = list(zip(df_t.iloc[:,0], df_t.iloc[:,2]))
            df_l = self.df_laser[self.df_laser["Sample"]==sample_l]
            laser_phi = list(zip(df_l["phi"], df_l["Wt"]))

            comb = self.combinar_metodo_1(tamiz, laser_phi, phi_extraido) if metodo==1 \
                   else self.combinar_metodo_2(tamiz, laser_phi, phi_extraido)

            for p, w in comb:
                resultados.append({"phi":p, "wt%":w,
                                   "Tamiz Sample":sample_t,
                                   "Laser Sample":sample_l})

        self.df_hybrid = pd.DataFrame(resultados)
        self.df_hybrid["Group"] = [
            self.df_data[self.df_data.iloc[:,1]==s].iloc[0,3]
            for s in self.df_hybrid["Tamiz Sample"]
        ]
        self.act_viewdb_hybrid.setEnabled(True)

        params_h = []
        for sample, grp in self.df_hybrid.groupby("Tamiz Sample"):
            phi_vals = grp["phi"].tolist()
            wt_vals  = grp["wt%"].tolist()
            gval     = grp["Group"].iloc[0]
            p        = calculate_parameters(phi_vals, wt_vals, self.current_method)
            params_h.append({"Sample":sample, self.group_col:gval, **p})
        self.param_results_hybrid = pd.DataFrame(params_h)

        self._update_all_groups_and_colors()
        self._update_hist_samples()
        txt="\nParámetros granulométricos (Híbrido):\n\n"
        for _, r in self.param_results_hybrid.iterrows():
            txt+=f"Sample: {r['Sample']} (Group: {r[self.group_col]})\n"
            txt+=f"  Mediana (φ50): {r['median']:.4f}\n"
            txt+=f"  Sorting (σ): {r['sigma']:.4f}\n"
            txt+=f"  Asimetría: {r['skewness']:.4f}\n"
            txt+=f"  Curtosis: {r['kurtosis']:.4f}\n"
            txt+=f"  Media (Mz): {r['mean']:.4f}\n\n"
        self.txt_console.append(txt)
        QMessageBox.information(self, "Éxito", "Base de datos tamiz+láser creada.")

# endregion

# region # === Bloque 13: Exportar parámetros a Excel (una hoja por base) ===
    def export_params_to_excel(self):
        """
        Exporta parámetros de todas las bases cargadas.
        Crea UNA hoja por base: 'Tamiz', 'Laser' y/o 'Híbrido' (si existen),
        con el mismo formato de la consola (Sample, Group, φ50, σ, Sk, K, Mz).
        """
        bases = []
        if self.param_results is not None and not self.param_results.empty:
            bases.append(("Tamiz", self.param_results, self.group_col))
        if self.param_results_laser is not None and not self.param_results_laser.empty:
            bases.append(("Laser", self.param_results_laser, "Group"))
        if self.param_results_hybrid is not None and not self.param_results_hybrid.empty:
            bases.append(("Híbrido", self.param_results_hybrid, self.group_col))

        if not bases:
            QMessageBox.information(self, "Exportar", "No hay parámetros para exportar.")
            return

        file, _ = QFileDialog.getSaveFileName(self, "Guardar parámetros", "", "Excel (*.xlsx)")
        if not file:
            return
        if not file.lower().endswith(".xlsx"):
            file += ".xlsx"

        def _df_consola(df_params, group_col):
            df = df_params.copy()
            return pd.DataFrame({
                "Sample":         df["Sample"],
                "Group":          df[group_col],
                "Mediana (φ50)":  df["median"],
                "Sorting (σ)":    df["sigma"],
                "Asimetría":      df["skewness"],
                "Curtosis":       df["kurtosis"],
                "Media (Mz)":     df["mean"],
            })

        try:
            with pd.ExcelWriter(file) as xw:
                for base_name, dfp, gcol in bases:
                    _df_consola(dfp, gcol).to_excel(xw, sheet_name=f"{base_name}", index=False)
            QMessageBox.information(self, "Listo", "Parámetros exportados correctamente.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"No se pudo exportar:\n{e}")
            
            
            # endregion

# region # === Bloque 14: IA/GMM – helpers y exportación ===
import math

def _normal_pdf(x, mu, sigma):
    if sigma <= 0 or not np.isfinite(sigma):
        return np.zeros_like(x, dtype=float)
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gmm_detect_modes(phi, wt, *, kmax=4, random_state=0):
    """
    Ajusta un GMM a datos discretos (phi, wt%) usando BIC para elegir k∈[1..kmax].
    Si tu scikit-learn NO soporta 'sample_weight', se usa un *fallback* que
    replica observaciones proporcionalmente a los pesos.

    Devuelve:
      {k, means, stds, weights, bic, rmse, xs, components, mix_curve}
    con las curvas escaladas a unidades de 'wt por bin' usando el spacing de phi.
    """
    x = np.asarray(phi, dtype=float)
    y = np.asarray(wt, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    x = x[m]; y = y[m]
    if x.size < 2 or y.sum() <= 0:
        return {'k':0,'means':[],'stds':[],'weights':[],'bic':np.nan,'rmse':np.nan,'xs':x,'components':[],'mix_curve':np.zeros_like(x)}

    # ordenar por phi
    order = np.argsort(x); x = x[order]; y = y[order]

    # pesos normalizados
    w = y / y.sum()
    X = x.reshape(-1, 1)

    def _fit_gmm_for_k(k):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=k, covariance_type='full',
            reg_covar=1e-6, max_iter=500, tol=1e-5, n_init=10,
            random_state=random_state
        )
        # Intentar con sample_weight; si la versión no lo soporta, replicar X:
        try:
            gmm.fit(X, sample_weight=w)
            bic_val = gmm.bic(X)
        except TypeError:
            # Fallback: replicar observaciones (aprox) según w
            # Escogemos un tamaño total razonable para no explotar memoria.
            total = max(200, int(round(100 * len(X))))
            counts = np.maximum(1, np.round(w * total)).astype(int)
            X_rep = np.repeat(X, counts, axis=0)
            gmm.fit(X_rep)
            bic_val = gmm.bic(X_rep)
        return gmm, bic_val

    # Buscar el mejor k por BIC
    best = None
    for k in range(1, int(kmax) + 1):
        gmm_k, bic_k = _fit_gmm_for_k(k)
        if (best is None) or (bic_k < best['bic']):
            best = {'k': k, 'gmm': gmm_k, 'bic': bic_k}

    g = best['gmm']
    means = g.means_.ravel()
    weights = g.weights_.ravel()

    # stds desde covariances_
    cov = g.covariances_
    if np.ndim(cov) == 3:
        stds = np.sqrt(cov[:, 0, 0].ravel())
    else:
        stds = np.sqrt(np.asarray(cov).ravel())

    # ordenar por media ascendente (para mostrar prolijo)
    idx = np.argsort(means)
    means = means[idx]; stds = stds[idx]; weights = weights[idx]

    # Construir curvas (escala aprox. wt por bin)
    xu = np.sort(np.unique(x))
    spacing = float(np.min(np.diff(xu))) if len(xu) > 1 else 1.0
    xs = np.linspace(x.min() - 0.5 * spacing, x.max() + 0.5 * spacing, 400)

    comps = []
    for mu, sd, pi in zip(means, stds, weights):
        comps.append(pi * _normal_pdf(xs, mu, sd) * (y.sum() * spacing))
    mix_curve = np.sum(comps, axis=0) if comps else np.zeros_like(xs)

    # RMSE contra alturas de bins (evaluado en centros)
    y_hat_bins = np.zeros_like(y, dtype=float)
    for i, xi in enumerate(x):
        dens = np.sum([pi * _normal_pdf(np.array([xi]), mu, sd)
                       for mu, sd, pi in zip(means, stds, weights)], axis=0)
        y_hat_bins[i] = dens[0] * (y.sum() * spacing)
    rmse = float(np.sqrt(np.mean((y - y_hat_bins) ** 2)))

    return {
        'k': int(best['k']),
        'means': means,
        'stds': stds,
        'weights': weights,
        'bic': float(best['bic']),
        'rmse': rmse,
        'xs': xs,
        'components': comps,
        'mix_curve': mix_curve
    }

def export_gmm_csv_from_state(self):
    """Exporta a CSV los modos detectados para la muestra actualmente graficada (self._gmm_last)."""
    if not hasattr(self, '_gmm_last') or not self._gmm_last:
        QMessageBox.information(self, 'IA/GMM', 'No hay resultados de GMM para exportar.')
        return

    state = self._gmm_last
    import pandas as _pd
    rows = []
    for i, (mu, sd, w) in enumerate(zip(state['means_phi'], state['std_phi'], state['weights']), start=1):
        rows.append({
            'Base': state['base'],
            'Sample': state['sample'],
            'k': state['k'],
            'Modo': i,
            'Moda phi (φ_mode)': mu,
            'Dispersión phi (σϕ)': sd,
            'Fracción en masa (%)': 100.0 * w,
            'BIC (modelo)': state['bic'],
            'RMSE (bins)': state['rmse']
        })
    df = _pd.DataFrame(rows)

    file, _ = QFileDialog.getSaveFileName(None, 'Guardar modos detectados', '', 'CSV (*.csv)')
    if not file:
        return
    if not file.lower().endswith('.csv'):
        file += '.csv'
    try:
        df.to_csv(file, index=False)
        QMessageBox.information(None, 'IA/GMM', 'Modos exportados correctamente.')
    except Exception as e:
        QMessageBox.warning(None, 'IA/GMM', f'No se pudo exportar: {e}')

# endregion

# region # === Bloque 15: Main principal ===
