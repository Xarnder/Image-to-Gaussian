@echo off
rem Choose Image edit or a pre-generated 360 MP4 when double-clicking.
rem Drag a photo or MP4 onto this launcher; MP4s use the video pipeline.
rem Video requires manual review and approval before reconstruction.
rem Example: --method video --input "D:\clips\orbit.mp4" --experiment orbit-test --approved-video
rem Image CLI options remain supported. --method image explicitly selects the existing method.
rem Video results: output\<experiment>\<video-name>\video\final\character.ply
rem Image mode accepts a folder; video mode accepts one approved MP4.
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
if "%~1"=="" goto :passthrough
set "FIRST=%~1"
if "%FIRST:~0,1%"=="-" goto :passthrough
"%PY%" "%~dp0run_to_splat.py" --batch-launcher --input %*
set "RC=%ERRORLEVEL%"
goto :end

:passthrough
"%PY%" "%~dp0run_to_splat.py" --batch-launcher %*
set "RC=%ERRORLEVEL%"
goto :end

:setup_failed
echo [setup] FAILED - see the messages above. Retry from a terminal with:
echo         powershell -ExecutionPolicy Bypass -File scripts\v2_setup_env.ps1
set "RC=1"

:end
if "%PAUSE_AT_END%"=="1" pause
exit /b %RC%
