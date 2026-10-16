@echo off
title Task-Aware Vision - Web Dashboard
echo =========================================================================
echo    TASK-AWARE VISION  -  WEB DASHBOARD
echo    Opens http://localhost:8000 in your browser
echo =========================================================================
echo.

cd /d "%~dp0"
set VENV_PY=C:\Users\gshub\venvs\pynq\Scripts\python.exe

if not exist "%VENV_PY%" (
    set VENV_PY=python
)

"%VENV_PY%" "%~dp0PJT1\web\server.py" %*

pause
