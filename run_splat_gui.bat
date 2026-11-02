@echo off
rem ===========================================================================================
rem  Image-to-Gaussian GUI: generate 360 video candidates from one front image,
rem  approve a candidate (or automatically use the first), then reconstruct a splat.
rem  Existing Image edit and imported 360 video methods remain available.
rem
rem  Double-click this file. A browser tab opens at http://127.0.0.1:8765/
rem  Leave this window open while you work. Close it (or Ctrl-C) to stop the UI.
rem ===========================================================================================
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
set "PY=%~dp0.venv-i2g\Scripts\python3.exe"
set "RC=0"

set "PAUSE_AT_END=0"
echo %CMDCMDLINE% | findstr /i /c:"%~nx0" >nul && set "PAUSE_AT_END=1"

if exist "%PY%" goto :have_python
if exist "%~dp0.venv-i2g\Scripts\python.exe" (
    copy /y "%~dp0.venv-i2g\Scripts\python.exe" "%PY%" >nul
    goto :have_python
)
echo [setup] First run: creating the Python environment (one time, ~5 min, downloads ~3.5 GB)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\v2_setup_env.ps1"
if errorlevel 1 goto :setup_failed
if not exist "%PY%" goto :setup_failed

:have_python
echo Starting the Image-to-Gaussian GUI...
echo If a GUI is already running it will just open the browser.
echo Otherwise leave this window open. A browser tab should open in a moment.
"%PY%" "%~dp0splat_gui.py"
set "RC=%ERRORLEVEL%"
goto :end

:setup_failed
echo [setup] FAILED - see the messages above. Retry from a terminal with:
echo         powershell -ExecutionPolicy Bypass -File scripts\v2_setup_env.ps1
set "RC=1"

:end
if "%PAUSE_AT_END%"=="1" pause
exit /b %RC%
