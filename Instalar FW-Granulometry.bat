@echo off
setlocal
title Instalar FW-Granulometry 2.0
cd /d "%~dp0"

echo.
echo  ==========================================================
echo    FW-Granulometry 2.0 - Instalacion
echo  ==========================================================
echo.

if not exist "FW-Granulometry 2.0.py" goto :sin_carpeta

set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY goto :sin_python

echo  Python encontrado:
%PY% --version
echo.
echo  Instalando los paquetes necesarios. Puede tardar unos minutos...
echo.
%PY% -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto :error_pip
%PY% -c "import PyQt5, numpy, pandas, matplotlib, scipy, openpyxl, xlsxwriter, xlrd" >nul 2>&1
if errorlevel 1 goto :error_pip

echo.
echo  Creando acceso directo en el Escritorio...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d=(Get-Location).Path; $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'),'FW-Granulometry 2.0.lnk')); $s.TargetPath=[IO.Path]::Combine($d,'Abrir FW-Granulometry.bat'); $s.WorkingDirectory=$d; $s.IconLocation=[IO.Path]::Combine($d,'icono.nuevoico.ico')+',0'; $s.WindowStyle=7; $s.Description='FW-Granulometry 2.0'; $s.Save()" >nul 2>&1
if errorlevel 1 goto :sin_acceso
echo  Listo: acceso directo "FW-Granulometry 2.0" en el Escritorio.
goto :fin

:sin_acceso
echo  No se pudo crear el acceso directo.
echo  El programa se puede abrir igual con "Abrir FW-Granulometry.bat".

:fin
echo.
echo  ==========================================================
echo    Instalacion terminada
echo  ==========================================================
echo.
echo  Para abrir el programa: acceso directo "FW-Granulometry 2.0"
echo  del Escritorio, o doble clic en "Abrir FW-Granulometry.bat".
echo.
pause
exit /b 0

:sin_carpeta
echo  No se encontraron los archivos del programa en esta carpeta.
echo  Primero hay que descomprimir el ZIP completo: clic derecho sobre
echo  el archivo, "Extraer todo", y despues ejecutar este archivo desde
echo  la carpeta extraida.
echo.
pause
exit /b 1

:sin_python
echo  No se encontro Python 3.9 o superior en esta computadora.
echo.
echo  1. Instalar Python desde https://www.python.org/downloads/
echo  2. Durante la instalacion, marcar la opcion "Add python.exe to PATH".
echo  3. Volver a ejecutar este archivo.
echo.
start "" "https://www.python.org/downloads/"
pause
exit /b 1

:error_pip
echo.
echo  No se pudieron instalar los paquetes necesarios.
echo  Revisar la conexion a internet y volver a ejecutar este archivo.
echo.
pause
exit /b 1
