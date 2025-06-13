# === Bloque 1: Importaciones y configuración global ===
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
    QApplication, QMainWindow, QFileDialog, QAction, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QColorDialog,
    QCheckBox, QDialog, QGroupBox, QRadioButton, QButtonGroup, QTextEdit, QMessageBox,
    QTableView, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QAbstractTableModel


# === Bloque 2: Definición de constantes y diccionarios ===
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

METHODS = {
    "Folk & Ward (1957)": "folkward",
    "Blatt et al. (1972)": "blatt",
    "Inman (1952)": "inman"
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

# === Bloque 3: Funciones auxiliares de cálculo ===
import numpy as np

def cumulative_distribution(phi, weights):
    total = sum(weights)
    cum = []
    s = 0
    for w in weights:
        s += w
        cum.append(100 * s / total)
    return cum

def interp_percentile(phi, cum, target):
    if target <= cum[0]:
        return phi[0]
    if target >= cum[-1]:
        return phi[-1]
    for i in range(1, len(cum)):
        if cum[i] >= target:
            p1, p2 = cum[i-1], cum[i]
            phi1, phi2 = phi[i-1], phi[i]
            if p2 == p1:
                return phi[i]
            frac = (target - p1) / (p2 - p1)
            return phi1 + frac * (phi2 - phi1)
    return phi[-1]

def calculate_parameters(phi, weights, method="folkward"):
    phi = np.array(phi)
    weights = np.array(weights)
    if method == "folkward":
        cum = cumulative_distribution(phi, weights)
        phi5   = interp_percentile(phi, cum, 5)
        phi16  = interp_percentile(phi, cum, 16)
        phi25  = interp_percentile(phi, cum, 25)
        phi50  = interp_percentile(phi, cum, 50)
        phi75  = interp_percentile(phi, cum, 75)
        phi84  = interp_percentile(phi, cum, 84)
        phi95  = interp_percentile(phi, cum, 95)
        median   = phi50
        sigma    = (phi84 - phi16) / 4.0 + (phi95 - phi5) / 6.6
        skewness = ((phi16 + phi84 - 2 * phi50) / (2 * (phi84 - phi16))) + ((phi5 + phi95 - 2 * phi50) / (2 * (phi95 - phi5)))
        kurtosis = (phi95 - phi5) / (2.44 * (phi75 - phi25))
        mean     = (phi16 + phi50 + phi84) / 3.0
        return {"median": median, "sigma": sigma, "skewness": skewness, "kurtosis": kurtosis, "mean": mean}
    elif method == "blatt":
        return calculate_parameters(phi, weights, "folkward")
    elif method == "inman":
        cum = cumulative_distribution(phi, weights)
        phi16 = interp_percentile(phi, cum, 16)
        phi50 = interp_percentile(phi, cum, 50)
        phi84 = interp_percentile(phi, cum, 84)
        median = phi50
        sigma = (phi84 - phi16) / 2
        skewness = (phi16 + phi84 - 2*phi50)/(phi84 - phi16) if (phi84-phi16) != 0 else 0
        mean = (phi16 + phi84)/2
        kurtosis = np.nan
        return {"median": median, "sigma": sigma, "skewness": skewness, "kurtosis": kurtosis, "mean": mean}
    else:
        raise ValueError("Método desconocido")

# === NUEVAS FUNCIONES PARA AGRUPAR LASER ===

def phi_limits_from_um(step, phi_min, phi_max):
    """
    Devuelve lista de (phi_centro, um_sup, um_inf) para el rango dado.
    Siempre incluye los bins que cubren desde phi_min hasta phi_max.
    """
    phis = []
    # phi_n:     um < upper y >= lower
    current = np.floor(phi_min / step) * step
    last   = np.ceil(phi_max / step) * step
    while current <= last:
        um_sup = 1000 * 2 ** (-current + (step / 2))
        um_inf = 1000 * 2 ** (-current - (step / 2))
        phis.append( (round(current, 3), um_sup, um_inf) )
        current += step
    return phis

def agrupar_laser_por_phi(df, step):
    """
    Procesa un DataFrame con columnas: [Sample, Diameter_um, Wt]
    Retorna lista de diccionarios con claves: phi, Sample, Wt (sólo para bins válidos)
    """
    out = []
    for sample, grupo in df.groupby(df.columns[0]):
        um = grupo.iloc[:,1].values
        wt = grupo.iloc[:,2].values
        # Calcular phi de cada partícula
        phi_val = -np.log2(um / 1000)
        phi_min, phi_max = phi_val.min(), phi_val.max()
        bins_info = phi_limits_from_um(step, phi_min, phi_max)
        # Para cada bin φ, sumar wt donde um < sup y um >= inf
        wt_bins = []
        for phi_c, um_sup, um_inf in bins_info:
            # En el límite superior, es "menor que"; en el inferior, "mayor o igual"
            mask = (um < um_sup) & (um >= um_inf)
            suma = wt[mask].sum() if np.any(mask) else 0.0
            wt_bins.append((phi_c, suma))
        # Cortar ceros extremos (salvo los internos como pediste)
        vals = [x[1] for x in wt_bins]
        # Indices donde hay valores >0
        nz = [i for i, v in enumerate(vals) if v > 0]
        if nz:
            start, end = nz[0], nz[-1]
            for phi_c, suma in wt_bins[start:end+1]:
                out.append({"phi": phi_c, "Sample": sample, "Wt": suma})
    return out

# === Bloque 4: Funciones de trazado genéricas ===
def plot_ribbon(ax, x, y, color, alpha=0.17, width=0.07, smooth=15):
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
    tck, u = splprep([expanded[:, 0], expanded[:, 1]], s=0, per=True)
    u_fine = np.linspace(0, 1, len(expanded)*smooth)
    x_fine, y_fine = splev(u_fine, tck)
    ax.fill(x_fine, y_fine, color=color, alpha=alpha, linewidth=0, zorder=1)

def get_script_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))
# === Bloque 5.1: Modelo para mostrar DataFrame en QTableView ===
# === Bloque 5: Clases de diálogos personalizados y visualización DataFrame ===

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
        self.resize(320, 350)
        self.groups = groups
        self.selected = selected.copy()
        self.layout = QVBoxLayout()
        self.checkboxes = {}
        for grp in groups:
            cb = QCheckBox(str(grp))
            cb.setChecked(grp in self.selected)
            self.layout.addWidget(cb)
            self.checkboxes[grp] = cb
        btn_all = QPushButton("Seleccionar todos")
        btn_all.clicked.connect(self.select_all)
        btn_ok = QPushButton("Aceptar")
        btn_ok.clicked.connect(self.accept)
        h = QHBoxLayout()
        h.addWidget(btn_all)
        h.addStretch()
        h.addWidget(btn_ok)
        self.layout.addLayout(h)
        self.setLayout(self.layout)

    def select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)

    def get_selected(self):
        return [g for g, cb in self.checkboxes.items() if cb.isChecked()]

# === Bloque 5.1: Modelo para mostrar DataFrame en QTableView ===
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
        return str(self._df.iat[index.row(), index.column()])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._df.columns[section]
        else:
            return str(self._df.index[section])

# === Bloque 5.2: Ventana genérica para mostrar DataFrame ===
class DataBaseWindow(QDialog):
    def __init__(self, title, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 400)
        layout = QVBoxLayout(self)
        table = QTableView()
        table.setSortingEnabled(True)
        model = PandasModel(df)
        table.setModel(model)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

# === Bloque 5.3: Diálogo para emparejar muestras Tamiz ↔ Láser ===
class MatchDialog(QDialog):
    def __init__(self, df_tamiz, df_laser, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Emparejar muestras Tamiz y Láser")
        self.resize(900, 450)
        self.pairs = []  # lista de tuplas (fila_tamiz, fila_laser)

        layout = QHBoxLayout(self)

        # Tabla Tamiz (muestra nombres únicos de muestra)
        self.tv_t = QTableView()
        tamiz_samples = df_tamiz.iloc[:, 1].drop_duplicates().tolist()
        df_t = pd.DataFrame(tamiz_samples, columns=["G. Tamizado"])
        self.model_t = PandasModel(df_t)
        self.tv_t.setModel(self.model_t)
        self.tv_t.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(self.tv_t)

        # Tabla Laser (ahora muestra nombres únicos de muestra LASER, no φ)
        self.tv_l = QTableView()
        # Corrección clave: usar el nombre de muestra, no phi
        laser_samples = df_laser["Sample"].drop_duplicates().tolist()
        df_l = pd.DataFrame(laser_samples, columns=["G. Laser"])
        self.model_l = PandasModel(df_l)
        self.tv_l.setModel(self.model_l)
        self.tv_l.setSelectionBehavior(QTableView.SelectRows)
        layout.addWidget(self.tv_l)

        # Panel de emparejamientos
        right = QVBoxLayout()
        btn_pair = QPushButton("Emparejar selección")
        btn_pair.clicked.connect(self.pair_selected)
        right.addWidget(btn_pair)

        self.txt_pairs = QTextEdit()
        self.txt_pairs.setReadOnly(True)
        right.addWidget(self.txt_pairs)

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
        r_t = sel_t[0].row()
        r_l = sel_l[0].row()
        self.pairs.append((r_t, r_l))
        self._update_text()

    def _update_text(self):
        lines = []
        for rt, rl in self.pairs:
            s_t = self.model_t._df.iloc[rt, 0]
            s_l = self.model_l._df.iloc[rl, 0]
            lines.append(f"{s_t}  ↔  {s_l}")
        self.txt_pairs.setPlainText("\n".join(lines))

# === Bloque 6: Canvas de tamaño fijo ===
class FixedSizeFigureCanvas(FigureCanvas):
    def __init__(self, width, height, dpi=100, *args, **kwargs):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setFixedSize(int(width * dpi), int(height * dpi))
        self.setSizePolicy(self.sizePolicy().Fixed, self.sizePolicy().Fixed)

# === Bloque 6: Clase FixedSizeFigureCanvas ===
class FixedSizeFigureCanvas(FigureCanvas):
    def __init__(self, width, height, dpi=100, *args, **kwargs):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setFixedSize(int(width * dpi), int(height * dpi))
        self.setSizePolicy(
            self.sizePolicy().Fixed, self.sizePolicy().Fixed
        )

# === Bloque 7: Clase MainWindow – Constructor y menú principal ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Folk & Ward - Gráfico XY tipo paper + Walker + GK (PyQt5 version)")
        self.setGeometry(100, 30, 1260, 850)
        self.df_data = None
        self.df_laser = None
        self.df_hybrid = None

        self.group_col = None
        self.groups = []  # lista de claves únicas "nombre (base)"
        self.group_colors = {}  # diccionario con clave "nombre (base)": color
        self.selected_groups_xy = []
        self.selected_groups_ribbons = []
        self.selected_groups_walker = []
        self.selected_groups_gk = []

        self.param_results = None
        self.param_results_laser = None
        self.param_results_hybrid = None

        self.current_method = "folkward"
        self.theme = "dark"

        self.initUI()

    def initMenu(self):
        menubar = self.menuBar()
        # Archivo
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

        # Configuración
        menu_cfg = menubar.addMenu("&Configuración")
        menu_cfg.addAction(QAction("Método de cálculo", self, triggered=self.choose_method))
        menu_cfg.addAction(QAction("Colores de grupos", self, triggered=self.edit_colors))
        menu_cfg.addAction(QAction("Grupos visibles (XY)", self, triggered=self.select_ribbons))
        menu_cfg.addAction(QAction("Grupos visibles (Walker)", self, triggered=self.select_groups_walker))
        menu_cfg.addAction(QAction("Grupos visibles (GK)", self, triggered=self.select_groups_gk))
        menu_cfg.addAction(QAction("Estilo de interfaz", self, triggered=self.choose_theme))

        # Base de datos
        menu_db = menubar.addMenu("Base de datos")
        self.act_viewdb_tamiz = QAction("Ver base de datos de tamiz", self)
        self.act_viewdb_tamiz.setEnabled(False)
        self.act_viewdb_tamiz.triggered.connect(self.show_tamiz_db_window)
        menu_db.addAction(self.act_viewdb_tamiz)

        self.act_viewdb_laser = QAction("Ver base de datos laser", self)
        self.act_viewdb_laser.setEnabled(False)
        self.act_viewdb_laser.triggered.connect(self.show_laser_db_window)
        menu_db.addAction(self.act_viewdb_laser)

        self.act_viewdb_hybrid = QAction("Ver base de datos tamiz + laser", self)
        self.act_viewdb_hybrid.setEnabled(False)
        self.act_viewdb_hybrid.triggered.connect(self.show_hybrid_db_window)
        menu_db.addAction(self.act_viewdb_hybrid)

        # Ayuda
        menu_help = menubar.addMenu("&Ayuda")
        menu_help.addAction(
            QAction(
                "Acerca de...", self,
                triggered=lambda: QMessageBox.information(
                    self,
                    "Acerca de",
                    "Software de análisis granulométrico Folk & Ward - PyQt5"
                )
            )
        )

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

        # Asigna colores por defecto si no hay
        if not self.group_colors:
            for i, clave in enumerate(self.groups):
                self.group_colors[clave] = GROUP_COLORS[i % len(GROUP_COLORS)]
        else:
            # Mantener colores existentes, agregar nuevos si hace falta
            for i, clave in enumerate(self.groups):
                if clave not in self.group_colors:
                    self.group_colors[clave] = GROUP_COLORS[i % len(GROUP_COLORS)]

        # Actualiza los seleccionados (por defecto todos)
        self.selected_groups_xy = self.groups.copy()
        self.selected_groups_ribbons = self.groups.copy()
        self.selected_groups_walker = self.groups.copy()
        self.selected_groups_gk = self.groups.copy()

    # === Bloque 8.1: Pestaña Gráfico XY (multiselección de bases) ===

    def initUI(self):
        self.initMenu()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Inicializar selección de bases (solo Tamiz por defecto)
        self.current_db_selection = {"tamiz": True, "laser": False, "hybrid": False}

        # === 8.1: Pestaña Gráfico XY ===
        self.tab_xy = QWidget()
        v_xy = QVBoxLayout(self.tab_xy)
        h_controls = QHBoxLayout()

        # Cargar Tamiz
        self.btn_load_tamiz = QPushButton("Cargar archivo Excel de tamizado")
        self.btn_load_tamiz.clicked.connect(self.load_file)
        h_controls.addWidget(self.btn_load_tamiz)

        # Ejes X/Y
        h_controls.addWidget(QLabel("Eje X:"))
        self.cmb_x = QComboBox();    h_controls.addWidget(self.cmb_x)
        h_controls.addWidget(QLabel("Eje Y:"))
        self.cmb_y = QComboBox();    h_controls.addWidget(self.cmb_y)

        # Graficar XY
        self.btn_plot_xy = QPushButton("Graficar XY")
        self.btn_plot_xy.clicked.connect(self.plot_xy)
        h_controls.addWidget(self.btn_plot_xy)

        # Elegir bases a graficar
        self.btn_select_db = QPushButton("Elegir bases a graficar")
        self.btn_select_db.clicked.connect(self.select_db_to_plot)
        h_controls.addWidget(self.btn_select_db)

        h_controls.addStretch()

        # Exportar imagen
        self.btn_export_xy = QPushButton("Exportar imagen")
        self.btn_export_xy.clicked.connect(lambda: self.export_canvas(self.canvas_xy))
        h_controls.addWidget(self.btn_export_xy)

        v_xy.addLayout(h_controls)

        # Consola de texto
        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumHeight(190)
        v_xy.addWidget(self.txt_console)

        # Canvas XY
        self.canvas_xy = FigureCanvas(plt.figure(figsize=(8, 4.5)))
        self.canvas_xy.setFixedHeight(450)
        v_xy.addWidget(self.canvas_xy)

        self.tabs.addTab(self.tab_xy, "Gráfico XY")

        # === 8.2: Pestaña Walker (1971 and 1983) ===
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

        # === 8.3: Pestaña Gençalioğlu-Kuşcu et al 2007 ===
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

    def select_db_to_plot(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Elegir bases a graficar")
        layout = QVBoxLayout(dlg)

        cb_t = QCheckBox("Tamiz")
        cb_t.setChecked(self.current_db_selection["tamiz"])
        cb_l = QCheckBox("Laser")
        cb_l.setChecked(self.current_db_selection["laser"])
        cb_l.setEnabled(hasattr(self, "param_results_laser"))
        cb_h = QCheckBox("Tamiz + Laser")
        cb_h.setChecked(self.current_db_selection["hybrid"])
        cb_h.setEnabled(hasattr(self, "param_results_hybrid"))

        layout.addWidget(cb_t)
        layout.addWidget(cb_l)
        layout.addWidget(cb_h)

        btn = QPushButton("Aceptar")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)

        if dlg.exec_():
            self.current_db_selection = {
                "tamiz":  cb_t.isChecked(),
                "laser":  cb_l.isChecked(),
                "hybrid": cb_h.isChecked()
            }
            self.plot_xy()

# === Bloque 9: Métodos de MainWindow (dividido en subbloques) ===

           # === Bloque 9.1: Métodos de carga de archivos y bases de datos ===

    def load_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo Excel de tamizado",
            "",
            "Excel files (*.xls *.xlsx)"
        )
        if not file:
            return
        df = pd.read_excel(file)
        if df.shape[1] < 4:
            QMessageBox.warning(self, "Error", "El archivo de tamizado debe tener al menos cuatro columnas")
            return

        # Determinar columna de grupo
        self.group_col = df.columns[3]

        # Calcular parámetros Folk & Ward
        params = []
        for sample, grp in df.groupby(df.columns[1]):
            phi     = grp.iloc[:, 0].tolist()
            weights = grp.iloc[:, 2].tolist()
            gval    = grp.iloc[0, 3]
            p       = calculate_parameters(phi, weights, self.current_method)
            row     = {"Sample": sample, self.group_col: gval, **p}
            params.append(row)

        # Guardar resultados y datos originales
        self.param_results = pd.DataFrame(params)
        self.df_data       = df

        # ——— Actualiza listas de grupos, colores y selecciones ———
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


    def load_laser_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo Excel de laser",
            "",
            "Excel files (*.xls *.xlsx)"
        )
        if not file:
            return
        df = pd.read_excel(file)
        if df.shape[1] < 4:
            QMessageBox.warning(
                self,
                "Error",
                "El archivo de láser debe tener al menos cuatro columnas:\n"
                "Sample, Diameter (μm), 1 (%) y Group."
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

        # Renombrar y validar columna Group
        df_laser_raw = df.iloc[:, :4].copy()
        df_laser_raw.columns = ["Sample", "Diameter (μm)", "1 (%)", "Group"]
        if df_laser_raw["Group"].isnull().any() or (df_laser_raw["Group"].astype(str).str.strip() == "").any():
            QMessageBox.warning(
                self,
                "Error",
                "Falta al menos un valor en la columna 'Group' del archivo de láser."
            )
            return

        # Agrupar por φ
        datos = agrupar_laser_por_phi(df_laser_raw, step)
        self.df_laser = pd.DataFrame(datos)
        group_map = dict(zip(df_laser_raw["Sample"], df_laser_raw["Group"]))
        self.df_laser["Group"] = self.df_laser["Sample"].map(group_map)
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

        # ——— Actualiza listas de grupos, colores y selecciones ———
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


    def show_tamiz_db_window(self):
        if self.df_data is None:
            return
        df_int = self.df_data.iloc[:, :4].copy()
        df_int.columns = [
            f"{self.df_data.columns[0]} (φ)",
            f"{self.df_data.columns[1]} (Sample)",
            f"{self.df_data.columns[2]} (Weight)",
            f"{self.group_col} (Group)"
        ]
        dlg = DataBaseWindow("Base de datos de tamiz", df_int, self)
        dlg.exec_()

    def show_laser_db_window(self):
        if self.df_laser is None:
            return
        dlg = DataBaseWindow("Base de datos laser", self.df_laser, self)
        dlg.exec_()

    def show_hybrid_db_window(self):
        if self.df_hybrid is None:
            QMessageBox.information(self, "Base de datos híbrida", "Todavía no has combinado tamiz + láser.")
            return
        dlg = DataBaseWindow("Base de datos tamiz + láser", self.df_hybrid, self)
        dlg.exec_()

        # === Bloque 9.2: Métodos de actualización de consola, gráficos y configuración de grupos ===

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

    def plot_xy(self):
        x = self.cmb_x.currentText(); y = self.cmb_y.currentText()
        fig = self.canvas_xy.figure; fig.clf()
        ax = fig.add_subplot(111); ax.set_facecolor("white")

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

        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.1); ax.spines['left'].set_linewidth(1.1)
        ax.tick_params(length=6, width=1.1, direction='out', colors='black')
        ax.set_xlabel(SYMBOLS[x], fontsize=16); ax.set_ylabel(SYMBOLS[y], fontsize=16)
        ax.set_title(f"{SYMBOLS[y]} vs {SYMBOLS[x]}", fontsize=15)
        legend = ax.legend(title="Group (Base)", frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
        if legend:
            for t in legend.get_texts(): t.set_color('black')
            legend.get_title().set_color('black')
        fig.tight_layout(pad=1.7, rect=[0, 0, 0.87, 1])
        self.canvas_xy.draw()

    def plot_walker(self):
        x = "median"
        y = "sigma"
        fig = self.canvas_walker.figure; fig.clf()
        ax = fig.add_subplot(111); ax.set_facecolor("white")
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

        ax.set_xlim(-4, 6); ax.set_ylim(0, 5)
        ax.set_xlabel(SYMBOLS["median"], fontsize=16); ax.set_ylabel(SYMBOLS["sigma"], fontsize=16)
        ax.set_xticks(np.arange(-4, 7, 2)); ax.set_yticks(np.arange(0, 6, 1))
        ax.tick_params(length=8, width=1.3, direction='out', colors='black')
        for x_ in np.arange(-4, 7, 2): ax.plot([x_, x_], [0 - 0.12, 0], color='k', lw=1.1, zorder=30)
        for y_ in np.arange(0, 6, 1): ax.plot([-4 - 0.15, -4], [y_, y_], color='k', lw=1.1, zorder=30)
        for spine in ax.spines.values(): spine.set_linewidth(1.6)
        legend = ax.legend(title="Group (Base)", frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
        if legend:
            for t in legend.get_texts(): t.set_color('black')
            legend.get_title().set_color('black')
        fig.subplots_adjust(left=0.08, right=0.83, bottom=0.20, top=0.97)
        self.canvas_walker.draw()

    def plot_gk(self):
        x = "median"
        y = "sigma"
        fig = self.canvas_gk.figure; fig.clf()
        ax = fig.add_subplot(111); ax.set_facecolor("white")
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

        ax.set_xlim(-4, 6); ax.set_ylim(0, 5)
        ax.set_xlabel(SYMBOLS["median"], fontsize=16); ax.set_ylabel(SYMBOLS["sigma"], fontsize=16)
        ax.set_xticks(np.arange(-4, 7, 2)); ax.set_yticks(np.arange(0, 6, 1))
        ax.tick_params(length=8, width=1.3, direction='out', colors='black')
        for x_ in np.arange(-4, 7, 2): ax.plot([x_, x_], [0 - 0.12, 0], color='k', lw=1.1, zorder=30)
        for y_ in np.arange(0, 6, 1): ax.plot([-4 - 0.15, -4], [y_, y_], color='k', lw=1.1, zorder=30)
        for spine in ax.spines.values(): spine.set_linewidth(1.6)
        legend = ax.legend(title="Group (Base)", frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
        if legend:
            for t in legend.get_texts(): t.set_color('black')
            legend.get_title().set_color('black')
        fig.subplots_adjust(left=0.08, right=0.83, bottom=0.20, top=0.97)
        self.canvas_gk.draw()

    def edit_colors(self):
        # Usa la lista self.groups (que ya tiene las claves compuestas)
        if not self.groups:
            return
        dlg = ColorDialog(self.groups, self.group_colors, self)
        if dlg.exec_():
            self.group_colors = dlg.colors
            self.plot_xy(); self.plot_walker(); self.plot_gk()

    def select_ribbons(self):
        if not self.groups:
            return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_ribbons, "Ver/ocultar sombras (XY)", self)
        if dlg.exec_():
            self.selected_groups_ribbons = dlg.get_selected()
            self.plot_xy()

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

      # === Bloque 9.3: Métodos de configuración y exportación ===

    def choose_method(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Elegir método de cálculo")
        layout = QVBoxLayout(dlg)
        combo = QComboBox()
        combo.addItems(METHODS.keys())
        current = [k for k, v in METHODS.items() if v == self.current_method][0]
        combo.setCurrentText(current)
        layout.addWidget(QLabel("Método de cálculo estadístico:"))
        layout.addWidget(combo)
        btn_ok = QPushButton("Aceptar"); btn_ok.clicked.connect(dlg.accept)
        layout.addWidget(btn_ok)
        if dlg.exec_():
            self.current_method = METHODS[combo.currentText()]
            self.load_file()

    def choose_theme(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Estilo de interfaz")
        layout = QVBoxLayout(dlg)
        combo = QComboBox(); combo.addItems(["Oscuro (Fusion dark)", "Claro (Fusion light)"])
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
        file, ext = QFileDialog.getSaveFileName(
            self, "Guardar figura", "",
            "PNG (*.png);;TIFF (*.tif);;SVG (*.svg);;PDF (*.pdf);;Adobe Illustrator (*.ai)"
        )
        if not file:
            return
        if ext == "PNG (*.png)" and not file.lower().endswith(".png"):
            file += ".png"; canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white"); return
        if ext == "TIFF (*.tif)" and not file.lower().endswith(".tif"):
            file += ".tif"; canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white"); return
        if ext == "SVG (*.svg)" and not file.lower().endswith(".svg"):
            file += ".svg"; canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white"); return
        if ext == "PDF (*.pdf)" and not file.lower().endswith(".pdf"):
            file += ".pdf"; canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white"); return
        if ext == "Adobe Illustrator (*.ai)" and not file.lower().endswith(".ai"):
            file += ".ai"; canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white"); return
        canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white")

    def save_current_tab(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.export_canvas(self.canvas_xy)
        elif idx == 1:
            self.export_canvas(self.canvas_walker)
        elif idx == 2:
            self.export_canvas(self.canvas_gk)
        # === Bloque 9.4: Métodos de bins, procesamiento y combinación tamiz–láser ===

    def obtener_bins_tamiz(self, step=None):
        phi = self.df_data.iloc[:, 0].values
        mn, mx = np.floor(phi.min()), np.ceil(phi.max())
        if step is None:
            step = np.min(np.diff(np.sort(np.unique(phi)))) if len(phi) > 1 else 1.0
        bins = np.arange(mn, mx + step, step)
        bins = np.round(bins / step) * step  # Corrige posibles flotantes
        return bins

    def procesar_laser(self, df_laser, bins):
        required_cols = ["phi", "Sample", "Wt"]
        if not all(col in df_laser.columns for col in required_cols):
            raise ValueError(f"[procesar_laser] Columnas faltantes: {df_laser.columns}")
        phi_vals = df_laser["phi"].astype(float).values
        wt_vals  = df_laser["Wt"].astype(float).values
        # Ya vienen en % y agrupados, se devuelven tal cual
        return list(zip(phi_vals, wt_vals))

    def combinar_metodo_1(self, tamiz, laser_phi, phi_cut):
        """
        Método 1: φ < phi_cut ➔ tamiz; φ >= phi_cut ➔ láser; luego normalizar a 100%.
        """
        below    = [(p, w) for p, w in tamiz      if p <  phi_cut]
        above    = [(p, w) for p, w in laser_phi if p >= phi_cut]
        combined = below + above
        total    = sum(w for _, w in combined) or 1.0
        normalized = [(p, w * 100.0 / total) for p, w in combined]
        return sorted(normalized, key=lambda x: x[0])

    def combinar_metodo_2(self, tamiz, laser_phi, phi_cut):
        """
        Método 2: φ < phi_cut intacto; φ >= phi_cut sumando tamiz+láser y escalando.
        """
        part1 = [(p, w) for p, w in tamiz      if p <  phi_cut]
        X1    = sum(w for _, w in part1)
        pts   = sorted(
            set(p for p, _ in tamiz      if p >= phi_cut) |
            set(p for p, _ in laser_phi if p >= phi_cut)
        )
        part2 = []
        for p in pts:
            w_t = next((w for q, w in tamiz      if q == p), 0.0)
            w_l = next((w for q, w in laser_phi if q == p), 0.0)
            part2.append((p, w_t + w_l))
        X2 = sum(w for _, w in part2)
        if X2 == 0:
            return part1
        fac    = (100.0 - X1) / X2
        scaled = [(p, w * fac) for p, w in part2]
        return sorted(part1 + scaled, key=lambda x: x[0])

    def combinar_tamiz_laser(self):
        if self.df_data is None or self.df_laser is None:
            QMessageBox.warning(self, "Error", "Carga ambas bases de datos primero.")
            return

        dlg_map = MatchDialog(self.df_data, self.df_laser, self)
        if not dlg_map.exec_():
            return
        mapping = dlg_map.pairs
        if not mapping:
            QMessageBox.warning(self, "Error", "No se emparejaron muestras.")
            return

        # Sólo elegir método
        dlg2 = QDialog(self)
        dlg2.setWindowTitle("Método de combinación")
        ly = QVBoxLayout(dlg2)
        mg = QButtonGroup(dlg2)
        rb1 = QRadioButton("Método 1: tamiz<cut + láser≥cut")
        rb2 = QRadioButton("Método 2: tamiz intacto + sumar+escalar")
        rb1.setChecked(True); mg.addButton(rb1); mg.addButton(rb2)
        ly.addWidget(rb1); ly.addWidget(rb2)
        btn_ok = QPushButton("Combinar"); btn_ok.clicked.connect(dlg2.accept)
        ly.addWidget(btn_ok)
        if not dlg2.exec_():
            return

        metodo  = 1 if rb1.isChecked() else 2
        phi_cut = self.laser_step
        bins    = self.obtener_bins_tamiz(step=phi_cut)

        resultados = []
        for rt, rl in mapping:
            sample_t = dlg_map.model_t._df.iloc[rt, 0]
            sample_l = dlg_map.model_l._df.iloc[rl, 0]

            # Datos de tamiz
            df_t  = self.df_data[self.df_data.iloc[:,1] == sample_t]
            tamiz = list(zip(df_t.iloc[:,0], df_t.iloc[:,2]))

            # Datos de láser (filtrado correctamente)
            df_l      = self.df_laser[self.df_laser["Sample"] == sample_l]
            laser_phi = list(zip(df_l["phi"], df_l["Wt"]))

            # Aplicar método
            if metodo == 1:
                comb = self.combinar_metodo_1(tamiz, laser_phi, phi_cut)
            else:
                comb = self.combinar_metodo_2(tamiz, laser_phi, phi_cut)

            for p, w in comb:
                resultados.append({
                    "phi":           p,
                    "wt%":           w,
                    "Tamiz Sample":  sample_t,
                    "Laser Sample":  sample_l
                })

        # Construir DataFrame híbrido y agregar columna Group
        self.df_hybrid = pd.DataFrame(resultados)
        groups = []
        for sample in self.df_hybrid["Tamiz Sample"]:
            grp = self.df_data[self.df_data.iloc[:,1] == sample].iloc[0,3]
            groups.append(grp)
        self.df_hybrid["Group"] = groups
        self.act_viewdb_hybrid.setEnabled(True)

        # Calcular parámetros híbridos
        params_h = []
        for sample, grp in self.df_hybrid.groupby("Tamiz Sample"):
            phi_vals = grp["phi"].tolist()
            wt_vals  = grp["wt%"].tolist()
            gval     = grp["Group"].iloc[0]
            p        = calculate_parameters(phi_vals, wt_vals, self.current_method)
            row      = {"Sample": sample, self.group_col: gval, **p}
            params_h.append(row)
        self.param_results_hybrid = pd.DataFrame(params_h)

        # ——— Actualiza listas de grupos, colores y selecciones ———
        self._update_all_groups_and_colors()

        # Imprimir en consola
        txt = "\nParámetros granulométricos (Híbrido):\n\n"
        for _, r in self.param_results_hybrid.iterrows():
            txt += f"Sample: {r['Sample']} (Group: {r[self.group_col]})\n"
            txt += f"  Mediana (φ50): {r['median']:.4f}\n"
            txt += f"  Sorting (σ): {r['sigma']:.4f}\n"
            txt += f"  Asimetría: {r['skewness']:.4f}\n"
            txt += f"  Curtosis: {r['kurtosis']:.4f}\n"
            txt += f"  Media (Mz): {r['mean']:.4f}\n\n"
        self.txt_console.append(txt)

        QMessageBox.information(self, "Éxito", "Base de datos tamiz+láser creada.")

# === Bloque 10: Main principal ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    global qApp
    qApp = app
    # Tema CLARO por defecto:
    app.setStyleSheet(LIGHT_STYLESHEET)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
