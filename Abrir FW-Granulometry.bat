@echo off
setlocal
cd /d "%~dp0"

if not exist "FW-Granulometry 2.0.py" goto :sin_carpeta

set "PY="
set "PYW="
py -3 -c "import PyQt5, numpy, pandas, matplotlib, scipy" >nul 2>&1 && set "PY=py -3" && set "PYW=pyw -3"
if not defined PY python -c "import PyQt5, numpy, pandas, matplotlib, scipy" >nul 2>&1 && set "PY=python" && set "PYW=pythonw"
if not defined PY goto :sin_requisitos

start "" %PYW% "FW-Granulometry 2.0.py"
exit /b 0

:sin_carpeta
echo No se encontraron los archivos del programa en esta carpeta.
echo Primero hay que descomprimir el ZIP completo: clic derecho, "Extraer todo".
echo.
pause
exit /b 1

:sin_requisitos
echo Faltan Python o los paquetes que usa FW-Granulometry.
echo Ejecutar primero "Instalar FW-Granulometry.bat".
echo.
pause
exit /b 1
