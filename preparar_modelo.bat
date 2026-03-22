@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No se ha encontrado el entorno virtual .venv
    echo Revisa que exista la carpeta .venv dentro del proyecto.
    pause
    exit /b 1
)

echo Preparando y entrenando el modelo desde cero...
".venv\Scripts\python.exe" start_training.py --regions all --years 3 --from-scratch --ticker-percentage 1.0

if errorlevel 1 (
    echo.
    echo La preparacion del modelo ha terminado con error.
    pause
)
