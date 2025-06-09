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
    QCheckBox, QDialog, QGroupBox, QRadioButton, QButtonGroup, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt

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
        return {
            "median": median,
            "sigma": sigma,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "mean": mean
        }
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
        return {
            "median": median,
            "sigma": sigma,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "mean": mean
        }
    else:
        raise ValueError("Método desconocido")

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

# === Bloque 5: Clases de diálogos personalizados ===
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
            btn.clicked.connect(lambda checked, g=grp: self.change_color(g))
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
        self.param_results = None
        self.group_col = None
        self.groups = []
        self.group_colors = {}
        self.selected_groups_xy = []
        self.selected_groups_ribbons = []
        self.selected_groups_walker = []
        self.selected_groups_gk = []
        self.current_method = "folkward"
        self.walker_img_shape = (8, 4.5)
        self.gk_img_shape = (8, 4.5)
        self.theme = "dark"
        self.initUI()

    def initUI(self):
        self.initMenu()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # === Bloque 8A: Pestaña Gráfico XY ===
        self.tab_xy = QWidget()
        v_xy = QVBoxLayout(self.tab_xy)
        h_controls = QHBoxLayout()
        self.btn_load = QPushButton("Cargar archivo Excel")
        self.btn_load.clicked.connect(self.load_file)
        h_controls.addWidget(self.btn_load)
        h_controls.addWidget(QLabel("Eje X:"))
        self.cmb_x = QComboBox()
        h_controls.addWidget(self.cmb_x)
        h_controls.addWidget(QLabel("Eje Y:"))
        self.cmb_y = QComboBox()
        h_controls.addWidget(self.cmb_y)
        self.btn_plot_xy = QPushButton("Graficar XY")
        self.btn_plot_xy.clicked.connect(self.plot_xy)
        h_controls.addWidget(self.btn_plot_xy)
        h_controls.addStretch()
        self.btn_export_xy = QPushButton("Exportar imagen")
        self.btn_export_xy.clicked.connect(lambda: self.export_canvas(self.canvas_xy))
        h_controls.addWidget(self.btn_export_xy)
        v_xy.addLayout(h_controls)
        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumHeight(190)
        v_xy.addWidget(self.txt_console)
        h_colors = QHBoxLayout()
        self.btn_color = QPushButton("Colores de grupos")
        self.btn_color.clicked.connect(self.edit_colors)
        h_colors.addWidget(self.btn_color)
        self.btn_show_ribbons = QPushButton("Ver/Ocultar sombras")
        self.btn_show_ribbons.clicked.connect(self.select_ribbons)
        h_colors.addWidget(self.btn_show_ribbons)
        h_colors.addStretch()
        v_xy.addLayout(h_colors)
        self.canvas_xy = FigureCanvas(plt.figure(figsize=(8, 4.5)))
        self.canvas_xy.setFixedHeight(450)
        v_xy.addWidget(self.canvas_xy)
        self.tabs.addTab(self.tab_xy, "Gráfico XY")

        # === Bloque 8B: Pestaña Walker ===
        self.tab_walker = QWidget()
        v_walker = QVBoxLayout(self.tab_walker)
        h_walker = QHBoxLayout()
        self.walker_groupbox = QGroupBox()
        self.walker_buttons = QButtonGroup()
        self.walker_images = list(WALKER_IMAGES.keys())
        walker_hbox = QHBoxLayout(self.walker_groupbox)
        for i, name in enumerate(self.walker_images):
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
        walker_img_path = os.path.join(script_dir, WALKER_IMAGES["All"])
        walker_img = mpimg.imread(walker_img_path)
        img_h, img_w = walker_img.shape[0], walker_img.shape[1]
        dpi = 100
        width_in = img_w / dpi
        height_in = img_h / dpi
        self.walker_img_shape = (width_in, height_in)
        self.canvas_walker = FixedSizeFigureCanvas(width_in, height_in, dpi)
        v_walker.addWidget(self.canvas_walker)
        self.tabs.addTab(self.tab_walker, "Walker (1971 and 1983)")

        # === Bloque 8C: Pestaña GK ===
        self.tab_gk = QWidget()
        v_gk = QVBoxLayout(self.tab_gk)
        h_gk = QHBoxLayout()
        self.gk_groupbox = QGroupBox()
        self.gk_buttons = QButtonGroup()
        self.gk_images = list(GK_IMAGES.keys())
        gk_hbox = QHBoxLayout(self.gk_groupbox)
        for i, name in enumerate(self.gk_images):
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
        gk_img_path = os.path.join(script_dir, GK_IMAGES["All"])
        gk_img = mpimg.imread(gk_img_path)
        img_h2, img_w2 = gk_img.shape[0], gk_img.shape[1]
        width_in2 = img_w2 / dpi
        height_in2 = img_h2 / dpi
        self.gk_img_shape = (width_in2, height_in2)
        self.canvas_gk = FixedSizeFigureCanvas(width_in2, height_in2, dpi)
        v_gk.addWidget(self.canvas_gk)
        self.tabs.addTab(self.tab_gk, "Gençalioğlu-Kuşcu et al 2007")

# === Bloque 9: Métodos de menú y configuración general ===
    def initMenu(self):
        menubar = self.menuBar()
        menu_file = menubar.addMenu("&Archivo")
        act_load = QAction("Cargar archivo Excel", self)
        act_load.triggered.connect(self.load_file)
        menu_file.addAction(act_load)
        act_save = QAction("Exportar imagen actual", self)
        act_save.triggered.connect(self.save_current_tab)
        menu_file.addAction(act_save)
        menu_cfg = menubar.addMenu("&Configuración")
        act_method = QAction("Método de cálculo", self)
        act_method.triggered.connect(self.choose_method)
        menu_cfg.addAction(act_method)
        act_color = QAction("Colores de grupos", self)
        act_color.triggered.connect(self.edit_colors)
        menu_cfg.addAction(act_color)
        act_group_xy = QAction("Grupos visibles (XY)", self)
        act_group_xy.triggered.connect(self.select_ribbons)
        menu_cfg.addAction(act_group_xy)
        act_group_walker = QAction("Grupos visibles (Walker)", self)
        act_group_walker.triggered.connect(self.select_groups_walker)
        menu_cfg.addAction(act_group_walker)
        act_group_gk = QAction("Grupos visibles (GK)", self)
        act_group_gk.triggered.connect(self.select_groups_gk)
        menu_cfg.addAction(act_group_gk)
        act_theme = QAction("Estilo de interfaz", self)
        act_theme.triggered.connect(self.choose_theme)
        menu_cfg.addAction(act_theme)
        menu_help = menubar.addMenu("&Ayuda")
        act_about = QAction("Acerca de...", self)
        act_about.triggered.connect(lambda: QMessageBox.information(self, "Acerca de", "Software de análisis granulométrico Folk & Ward - PyQt5"))
        menu_help.addAction(act_about)

# === Bloque 10: Carga y actualización de datos ===
    def load_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo Excel", "", "Excel files (*.xls *.xlsx)")
        if not file:
            return
        df = pd.read_excel(file)
        if df.shape[1] < 4:
            QMessageBox.warning(self, "Error", "El archivo debe tener al menos cuatro columnas")
            return
        self.group_col = df.columns[3]
        self.groups = sorted(df[self.group_col].unique())
        self.group_colors = {grp: GROUP_COLORS[i % len(GROUP_COLORS)] for i, grp in enumerate(self.groups)}
        self.selected_groups_xy = self.groups.copy()
        self.selected_groups_ribbons = self.groups.copy()
        self.selected_groups_walker = self.groups.copy()
        self.selected_groups_gk = self.groups.copy()
        param_list = []
        for sample, group in df.groupby(df.columns[1]):
            phi = group.iloc[:,0].tolist()
            weights = group.iloc[:,2].tolist()
            group_val = group.iloc[0, 3]
            params = calculate_parameters(phi, weights, self.current_method)
            row = { "Sample": sample, self.group_col: group_val, **params }
            param_list.append(row)
        self.param_results = pd.DataFrame(param_list)
        self.df_data = df
        param_opts = ["mean", "median", "sigma", "skewness", "kurtosis"]
        self.cmb_x.clear(); self.cmb_y.clear()
        self.cmb_x.addItems(param_opts); self.cmb_y.addItems(param_opts)
        self.cmb_x.setCurrentText("mean")
        self.cmb_y.setCurrentText("sigma")
        self.update_console()
        self.plot_xy()
        self.plot_walker()
        self.plot_gk()

# === Bloque 11: Actualización de consola ===
    def update_console(self):
        if self.param_results is None: return
        txt = f"Parámetros granulométricos ({self.current_method}):\n\n"
        for _, row in self.param_results.iterrows():
            txt += f"Sample: {row['Sample']} ({self.group_col}: {row[self.group_col]})\n"
            txt += f"  Mediana (φ50): {row['median']:.4f}\n"
            txt += f"  Sorting (σ): {row['sigma']:.4f}\n"
            txt += f"  Asimetría: {row['skewness']:.4f}\n"
            txt += f"  Curtosis: {row['kurtosis']:.4f}\n"
            txt += f"  Media (Mz): {row['mean']:.4f}\n\n"
        self.txt_console.setPlainText(txt)

# === Bloque 12: Graficación general ===
    def plot_xy(self):
        if self.param_results is None: return
        x_param = self.cmb_x.currentText()
        y_param = self.cmb_y.currentText()
        fig = self.canvas_xy.figure
        fig.clf()
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        groups = [g for g in self.groups if g in self.selected_groups_xy]
        for grp in groups:
            sub = self.param_results[self.param_results[self.group_col] == grp]
            x = sub[x_param].values
            y = sub[y_param].values
            col = self.group_colors.get(grp, "#888888")
            if grp in self.selected_groups_ribbons and len(x) > 2:
                plot_ribbon(ax, x, y, color=col, alpha=0.18, width=0.07, smooth=10)
        for grp in groups:
            sub = self.param_results[self.param_results[self.group_col] == grp]
            col = self.group_colors.get(grp, "#888888")
            ax.scatter(sub[x_param], sub[y_param], color=col, label=str(grp), s=50, edgecolor='k', lw=1.2, alpha=1, zorder=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.1)
        ax.spines['left'].set_linewidth(1.1)
        ax.tick_params(axis='both', which='major', length=6, width=1.1, direction='out', colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        ax.grid(axis='y', color='#dddddd', linewidth=0.55)
        ax.grid(axis='x', color='#dddddd', linewidth=0.55)
        ax.set_axisbelow(True)
        ax.set_xlabel(SYMBOLS.get(x_param, x_param), fontsize=16, fontweight='normal', labelpad=4)
        ax.set_ylabel(SYMBOLS.get(y_param, y_param), fontsize=16, fontweight='normal', labelpad=4)
        ax.set_title(f"{SYMBOLS.get(y_param, y_param)} vs {SYMBOLS.get(x_param, x_param)} (por {self.group_col})", fontsize=15, fontweight='normal', pad=4)
        legend = ax.legend(title=self.group_col, fontsize=11, title_fontsize=11, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color('black')
            legend.get_title().set_color('black')
        fig.tight_layout(pad=1.7, rect=[0, 0, 0.87, 1])
        self.canvas_xy.draw()

    def plot_walker(self):
        if self.param_results is None: return
        img_key = self.walker_buttons.checkedButton().text()
        script_dir = get_script_dir()
        img_path = os.path.join(script_dir, WALKER_IMAGES[img_key])
        img = mpimg.imread(img_path)
        x_min, x_max = -4, 6
        y_min, y_max = 0, 5
        fig = self.canvas_walker.figure
        fig.clf()
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto', zorder=1)
        groups = [g for g in self.groups if g in self.selected_groups_walker]
        for grp in groups:
            sub = self.param_results[self.param_results[self.group_col] == grp]
            x = sub["median"].values
            y = sub["sigma"].values
            col = self.group_colors.get(grp, "#888888")
            ax.scatter(x, y, color=col, label=str(grp), s=50, edgecolor='k', lw=1.2, alpha=1, zorder=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(SYMBOLS["median"], fontsize=16, fontweight='normal', labelpad=10)
        ax.set_ylabel(SYMBOLS["sigma"], fontsize=16, fontweight='normal', labelpad=10)
        ax.set_xticks(np.arange(x_min, x_max + 1, 2))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.tick_params(axis='both', which='major', length=8, width=1.3, direction='out', colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        for x in np.arange(x_min, x_max + 1, 2):
            ax.plot([x, x], [y_min-0.12, y_min], color='k', lw=1.1, clip_on=False, zorder=30)
        for y in np.arange(y_min, y_max + 1, 1):
            ax.plot([x_min-0.15, x_min], [y, y], color='k', lw=1.1, clip_on=False, zorder=30)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_axisbelow(True)
        ax.grid(False)
        legend = ax.legend(title=self.group_col, fontsize=11, title_fontsize=11, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color('black')
            legend.get_title().set_color('black')
        fig.subplots_adjust(left=0.08, right=0.83, bottom=0.20, top=0.97)
        self.canvas_walker.draw()

    def plot_gk(self):
        if self.param_results is None: return
        img_key = self.gk_buttons.checkedButton().text()
        script_dir = get_script_dir()
        img_path = os.path.join(script_dir, GK_IMAGES[img_key])
        img = mpimg.imread(img_path)
        x_min, x_max = -4, 6
        y_min, y_max = 0, 5
        fig = self.canvas_gk.figure
        fig.clf()
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")
        ax.imshow(img, extent=[x_min, x_max, y_min, y_max], aspect='auto', zorder=1)
        groups = [g for g in self.groups if g in self.selected_groups_gk]
        for grp in groups:
            sub = self.param_results[self.param_results[self.group_col] == grp]
            x = sub["median"].values
            y = sub["sigma"].values
            col = self.group_colors.get(grp, "#888888")
            ax.scatter(x, y, color=col, label=str(grp), s=50, edgecolor='k', lw=1.2, alpha=1, zorder=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(SYMBOLS["median"], fontsize=16, fontweight='normal', labelpad=10)
        ax.set_ylabel(SYMBOLS["sigma"], fontsize=16, fontweight='normal', labelpad=10)
        ax.set_xticks(np.arange(x_min, x_max + 1, 2))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.tick_params(axis='both', which='major', length=8, width=1.3, direction='out', colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        for x in np.arange(x_min, x_max + 1, 2):
            ax.plot([x, x], [y_min-0.12, y_min], color='k', lw=1.1, clip_on=False, zorder=30)
        for y in np.arange(y_min, y_max + 1, 1):
            ax.plot([x_min-0.15, x_min], [y, y], color='k', lw=1.1, clip_on=False, zorder=30)
        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.set_axisbelow(True)
        ax.grid(False)
        legend = ax.legend(title=self.group_col, fontsize=11, title_fontsize=11, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color('black')
            legend.get_title().set_color('black')
        fig.subplots_adjust(left=0.08, right=0.83, bottom=0.20, top=0.97)
        self.canvas_gk.draw()

# === Bloque 13: Gestión de colores y selección de grupos ===
    def edit_colors(self):
        if not self.groups: return
        dlg = ColorDialog(self.groups, self.group_colors, self)
        if dlg.exec_():
            self.group_colors = dlg.colors
            self.plot_xy()
            self.plot_walker()
            self.plot_gk()

    def select_ribbons(self):
        if not self.groups: return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_ribbons, "Ver/ocultar sombras (XY)", self)
        if dlg.exec_():
            self.selected_groups_ribbons = dlg.get_selected()
            self.plot_xy()

    def select_groups_walker(self):
        if not self.groups: return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_walker, "Seleccionar grupos (Walker)", self)
        if dlg.exec_():
            self.selected_groups_walker = dlg.get_selected()
            self.plot_walker()

    def select_groups_gk(self):
        if not self.groups: return
        dlg = GroupSelectDialog(self.groups, self.selected_groups_gk, "Seleccionar grupos (GK)", self)
        if dlg.exec_():
            self.selected_groups_gk = dlg.get_selected()
            self.plot_gk()

# === Bloque 14: Selección de método y tema ===
    def choose_method(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Elegir método de cálculo")
        layout = QVBoxLayout(dlg)
        combo = QComboBox()
        combo.addItems(METHODS.keys())
        current_method = [k for k, v in METHODS.items() if v == self.current_method][0]
        combo.setCurrentText(current_method)
        layout.addWidget(QLabel("Método de cálculo estadístico:"))
        layout.addWidget(combo)
        btn_ok = QPushButton("Aceptar")
        btn_ok.clicked.connect(dlg.accept)
        layout.addWidget(btn_ok)
        if dlg.exec_():
            self.current_method = METHODS[combo.currentText()]
            self.load_file()

    def choose_theme(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Estilo de interfaz")
        layout = QVBoxLayout(dlg)
        combo = QComboBox()
        combo.addItems(["Oscuro (Fusion dark)", "Claro (Fusion light)"])
        combo.setCurrentIndex(0 if self.theme == "dark" else 1)
        layout.addWidget(QLabel("Selecciona el estilo de la interfaz:"))
        layout.addWidget(combo)
        btn_ok = QPushButton("Aceptar")
        btn_ok.clicked.connect(dlg.accept)
        layout.addWidget(btn_ok)
        if dlg.exec_():
            idx = combo.currentIndex()
            if idx == 0 and self.theme != "dark":
                self.theme = "dark"
                qApp.setStyleSheet(DARK_STYLESHEET)
            elif idx == 1 and self.theme != "light":
                self.theme = "light"
                qApp.setStyleSheet(LIGHT_STYLESHEET)

# === Bloque 15: Métodos de exportación de figuras ===
    def export_canvas(self, canvas):
        file, ext = QFileDialog.getSaveFileName(
            self, "Guardar figura", "",
            "PNG (*.png);;TIFF (*.tif);;SVG (*.svg);;PDF (*.pdf);;Adobe Illustrator (*.ai)"
        )
        if not file:
            return
        # Lógica para extensión y formato según selección
        if ext == "PNG (*.png)" and not file.lower().endswith(".png"):
            file += ".png"
            canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white", format="png")
            return
        if ext == "TIFF (*.tif)" and not file.lower().endswith(".tif"):
            file += ".tif"
            canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white", format="tiff")
            return
        if ext == "SVG (*.svg)" and not file.lower().endswith(".svg"):
            file += ".svg"
            canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white", format="svg")
            return
        if ext == "PDF (*.pdf)" and not file.lower().endswith(".pdf"):
            file += ".pdf"
            canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white", format="pdf")
            return
        if ext == "Adobe Illustrator (*.ai)" and not file.lower().endswith(".ai"):
            file += ".ai"
            # Guarda como PDF pero con extensión .ai, editable en Illustrator
            canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white", format="pdf")
            return
        # Si ya tiene la extensión correcta
        canvas.figure.savefig(file, dpi=300, bbox_inches='tight', facecolor="white")

    def save_current_tab(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.export_canvas(self.canvas_xy)
        elif idx == 1:
            self.export_canvas(self.canvas_walker)
        elif idx == 2:
            self.export_canvas(self.canvas_gk)

# === Bloque 16: Main principal ===
if __name__ == "__main__":
    app = QApplication(sys.argv)
    global qApp
    qApp = app
    app.setStyleSheet(DARK_STYLESHEET)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
