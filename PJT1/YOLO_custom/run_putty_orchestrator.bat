@echo off
title PuTTY Orchestrator Launcher - Assistive Robotics Vision System
echo =========================================================================
echo    STARTING PUTTY TERMINAL ORCHESTRATOR BRIDGE
echo    Model 1 (Tiny YOLOv2) + Model 2 (Task-Aware Selector)
echo =========================================================================
echo.

cd /d "%~dp0"
set VENV_PY=%~dp0..\..\.venv\Scripts\python.exe

if not exist "%VENV_PY%" (
    set VENV_PY=python
)

echo [+] Launching PuTTY Terminal Bridge Server on port 5000...
echo [+] PuTTY Terminal window will automatically open and connect.
echo.

"%VENV_PY%" "%~dp0putty_bridge.py" --port 5000

pause
