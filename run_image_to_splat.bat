@echo off
rem ===========================================================================================
rem  Image-to-Gaussian: one front-facing full-body photo in, one 3D Gaussian splat (.ply) out.
rem
rem  Three ways to run it:
rem    1. Double-click this file, drag your photo into the window, press Enter.
rem    2. Drag your photo onto this file.
rem    3. From a terminal, with any option of image_to_splat.py, for example:
rem         run_image_to_splat.bat --input "D:\photos\front.png" --output_dir "D:\splats\me" --preset draft
rem         run_image_to_splat.bat --input "D:\photos\front.png" --from_stage train --set iterations=20000
rem         run_image_to_splat.bat --check
rem         run_image_to_splat.bat --help
rem
rem  Result: <output_dir>\character.ply  (default: output\<photo name>\v2\final\)
rem  View it: drag character.ply onto https://superspl.at/editor
rem ===========================================================================================
setlocal EnableExtensions
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
set "PY=%~dp0.venv-i2g\Scripts\python3.exe"
set "RC=0"

rem keep the window open at the end when started from Explorer (double-click or drag-and-drop)
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
if "%~1"=="" goto :ask
set "FIRST=%~1"
if "%FIRST:~0,1%"=="-" goto :passthrough
rem the first argument is a file (drag-and-drop): use it as --input and pass the rest through
"%PY%" "%~dp0image_to_splat.py" --input %*
set "RC=%ERRORLEVEL%"
goto :end

:passthrough
"%PY%" "%~dp0image_to_splat.py" %*
set "RC=%ERRORLEVEL%"
goto :end

:ask
echo.
echo  Image-to-Gaussian: turn one front-facing full-body photo into a 3D Gaussian splat.
echo  Drag your photo into this window (or type its full path) and press Enter:
set "IMG="
set /p "IMG=> "
if not defined IMG goto :ask
set "IMG=%IMG:"=%"
"%PY%" "%~dp0image_to_splat.py" --input "%IMG%"
set "RC=%ERRORLEVEL%"
goto :end

:setup_failed
echo [setup] FAILED - see the messages above. Retry from a terminal with:
echo         powershell -ExecutionPolicy Bypass -File scripts\v2_setup_env.ps1
set "RC=1"

:end
if "%PAUSE_AT_END%"=="1" pause
exit /b %RC%
