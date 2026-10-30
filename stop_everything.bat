@echo off
rem ===========================================================================================
rem  Image-to-Gaussian: stop EVERYTHING and free the GPU.
rem
rem  Double-click this file (it asks for administrator rights). It stops every Image-to-Gaussian
rem  job - single runs, batches, the GUI, research scripts and test loops - and ComfyUI, so the GPU
rem  memory and the system RAM the models use are completely free. Other programs are left alone.
rem
rem  Finished stages stay on disk: start the GUI or run_batch_to_splat.bat again (without Rebuild)
rem  and each photo continues from where it stopped.
rem
rem  Only look, stop nothing:   stop_everything.bat -ListOnly
rem ===========================================================================================
setlocal
cd /d "%~dp0"

rem relaunch as administrator if needed (stopping processes another window started can need it)
net session >nul 2>&1
if errorlevel 1 (
    echo Asking for administrator rights...
    if "%~1"=="" (
        powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    ) else (
        powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '%*' -Verb RunAs"
    )
    exit /b
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_everything.ps1" %*
echo.
pause
