# FW-Granulometry 2.0

Software de escritorio para análisis granulométrico: parámetros estadísticos (Folk & Ward, Inman, Trask), gráficos de granulometría volcánica (Walker, Gençalioğlu-Kuşcu, Pardo), unificación de datos láser con tamizado, análisis de componentes (Componentry-FW), análisis fractal y sistema de complementos.

## Requisitos

- Python 3.9 o superior
- Dependencias: `pip install -r requirements.txt`

## Instalación y uso

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

## Complementos

Los complementos se instalan desde el menú *Complementos* de Componentry y se copian a `~/.componentry_analysis/`.

## Contacto

santiagoabelretamoso@gmail.com

## Licencia

GNU General Public License v3.0. Ver [LICENSE](LICENSE).
