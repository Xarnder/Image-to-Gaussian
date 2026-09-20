@echo off
rem ===========================================================================================
rem  Image-to-Gaussian, batch: a folder of photos in, one 3D Gaussian splat per photo out.
rem
rem  Same pipeline as run_image_to_splat.bat, run once per photo, one at a time so the GPU is
rem  never shared. ComfyUI starts once and stays up for the whole batch.
rem
rem  Three ways to run it:
rem    1. Double-click this file, drag your folder of photos into the window, press Enter.
rem    2. Drag a folder (or a single photo) onto this file.
rem    3. From a terminal, with any option of batch_to_splat.py or image_to_splat.py:
rem         run_batch_to_splat.bat --input "D:\photos\characters"
rem         run_batch_to_splat.bat --input "D:\photos\characters" --output_dir "D:\splats"
rem         run_batch_to_splat.bat --input "D:\photos\characters" --dry_run
rem         run_batch_to_splat.bat --input "D:\photos\characters" --preset draft
rem         run_batch_to_splat.bat --input "D:\photos\characters" --no_orbit
rem         run_batch_to_splat.bat --help
rem
rem  Result per photo: output\<photo name>\v2\final\character.ply  (or under --output_dir),
rem  plus batch_summary.json listing every photo, its status and how long it took.
rem  View one: drag character.ply onto https://superspl.at/editor
rem
rem  Photos that already have a character.ply are skipped, so you can stop and start again and it
rem  carries on where it left off. About 3 h per photo on the standard preset.
rem
rem  Stopping: Ctrl-C stops now (the photo it is on is left unfinished but resumable). To stop
rem  after the current photo instead, put an empty file called stop.txt in the photo folder.
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
rem the first argument is a folder or a photo (drag-and-drop): use it as --input, pass the rest through
"%PY%" "%~dp0batch_to_splat.py" --input %*
set "RC=%ERRORLEVEL%"
goto :end

:passthrough
"%PY%" "%~dp0batch_to_splat.py" %*
set "RC=%ERRORLEVEL%"
goto :end

:ask
echo.
echo  Image-to-Gaussian, batch: every photo in a folder becomes its own 3D Gaussian splat.
echo  Drag the folder of photos into this window (or type its full path) and press Enter:
set "IMG="
set /p "IMG=> "
if not defined IMG goto :ask
set "IMG=%IMG:"=%"
"%PY%" "%~dp0batch_to_splat.py" --input "%IMG%"
set "RC=%ERRORLEVEL%"
goto :end

:setup_failed
echo [setup] FAILED - see the messages above. Retry from a terminal with:
echo         powershell -ExecutionPolicy Bypass -File scripts\v2_setup_env.ps1
set "RC=1"

:end
if "%PAUSE_AT_END%"=="1" pause
exit /b %RC%
