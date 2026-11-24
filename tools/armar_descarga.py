"""Arma el ZIP que se descarga desde la página de FW-Granulometry.

Toma los archivos del programa de la raíz del repositorio y los guarda en
docs/descargas/FW-Granulometry-<versión>.zip, dentro de una carpeta
"FW-Granulometry <versión>". Hay que correrlo cada vez que cambia alguno
de los scripts, para que la descarga quede actualizada:

    python tools/armar_descarga.py
"""
import pathlib
import time
import zipfile

VERSION = "2.0"
RAIZ = pathlib.Path(__file__).resolve().parent.parent
CARPETA = f"FW-Granulometry {VERSION}"
SALIDA = RAIZ / "docs" / "descargas" / f"FW-Granulometry-{VERSION}.zip"

ARCHIVOS = [
    "FW-Granulometry 2.0.py",
    "Componentry-FW.py",
    "FractalAnalysis.py",
    "Graphic.py",
    "GraphicVectors.py",
    "GdeInput.py",
    "icono.nuevo.png",
    "icono.nuevoico.ico",
    "requirements.txt",
    "LEEME.txt",
    "Instalar FW-Granulometry.bat",
    "Abrir FW-Granulometry.bat",
    "LICENSE",
]


def main():
    faltan = [a for a in ARCHIVOS if not (RAIZ / a).is_file()]
    if faltan:
        raise SystemExit("Faltan archivos: " + ", ".join(faltan))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SALIDA, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for nombre in ARCHIVOS:
            ruta = RAIZ / nombre
            fecha = time.localtime(ruta.stat().st_mtime)[:6]
            info = zipfile.ZipInfo(f"{CARPETA}/{nombre}", date_time=fecha)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, ruta.read_bytes(), compresslevel=9)
    mb = SALIDA.stat().st_size / (1024 * 1024)
    print(f"{SALIDA.relative_to(RAIZ)}  ({mb:.2f} MB, {len(ARCHIVOS)} archivos)")


if __name__ == "__main__":
    main()
