@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No se ha encontrado el entorno virtual .venv
    echo Revisa que exista la carpeta .venv dentro del proyecto.
    pause
    exit /b 1
)

set "MPLCONFIGDIR=%TEMP%\predictor_bursatil_tft\matplotlib"
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%" >nul 2>&1

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py

if errorlevel 1 (
    echo.
    echo La aplicacion ha terminado con error.
    pause
)
