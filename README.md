# FW-Granulometry 2.0

Software de escritorio para análisis granulométrico: parámetros estadísticos (Folk & Ward, Inman, Trask), gráficos de granulometría volcánica (Walker, Gençalioğlu-Kuşcu, Pardo), unificación de datos láser con tamizado, análisis de componentes (Componentry-FW), análisis fractal y sistema de complementos.

**Página y descarga:** https://santiagoabelretamoso.github.io/FW-Granulometry/

## Descarga

[Descargar FW-Granulometry 2.0 (ZIP)](https://santiagoabelretamoso.github.io/FW-Granulometry/descargas/FW-Granulometry-2.0.zip). Incluye el programa completo, un instalador para Windows y las instrucciones (`LEEME.txt`).

## Requisitos

- Python 3.9 o superior
- Dependencias: `pip install -r requirements.txt` (PyQt5, numpy, pandas, matplotlib, scipy, openpyxl, xlsxwriter, xlrd para archivos .xls)

## Instalación y uso

### Windows

1. Instalar Python 3.9 o superior desde https://www.python.org/downloads/ y marcar la opción "Add python.exe to PATH".
2. Descomprimir el ZIP y hacer doble clic en `Instalar FW-Granulometry.bat`. Instala los paquetes necesarios y crea un acceso directo en el Escritorio.
3. Abrir el programa con el acceso directo o con `Abrir FW-Granulometry.bat`.

### macOS, Linux o desde el código

1. Descargar o clonar el repositorio. Todos los `.py` tienen que quedar en la misma carpeta.
2. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
3. Ejecutar:
   ```
   python "FW-Granulometry 2.0.py"
   ```

## Archivos

| Archivo | Función |
|---|---|
| `FW-Granulometry 2.0.py` | Programa principal |
| `Graphic.py`, `GraphicVectors.py` | Motor de diagramas vectoriales (Walker, GK, Pardo) |
| `Componentry-FW.py` | Módulo de análisis de componentes y gestor de complementos |
| `FractalAnalysis.py` | Análisis de fragmentación fractal |
| `GdeInput.py` | Complemento opcional: genera archivos `.dat` y ejecuta PYFLOW. Requiere Windows con WSL (Ubuntu-22.04) y PYFLOW instalado |
| `icono.nuevo.png`, `icono.nuevoico.ico`, `flyer.png` | Icono y pantalla de inicio |
| `Instalar FW-Granulometry.bat`, `Abrir FW-Granulometry.bat` | Instalación y apertura en Windows |
| `LEEME.txt` | Instrucciones que acompañan a la descarga |
| `docs/` | Página web (GitHub Pages) y ZIP de descarga |
| `tools/armar_descarga.py` | Arma `docs/descargas/FW-Granulometry-2.0.zip` con los archivos del programa |

## Complementos

Los complementos se instalan desde el menú *Complementos* de Componentry y se copian a `~/.componentry_analysis/`.

## Página web y descarga

La página está en `docs/` y se publica con GitHub Pages. Cada vez que cambia algún script hay que volver a armar el ZIP de descarga:

```
python tools/armar_descarga.py
```

## Contacto

santiagoabelretamoso@gmail.com

## Licencia

GNU General Public License v3.0. Ver [LICENSE](LICENSE).
