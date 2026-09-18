# Starts the user's portable ComfyUI headless on 127.0.0.1:8188 for the v2 pipeline, or attaches
# to one that is already running. Does not modify ComfyUI's own .bat files.
# SageAttention is opt-in (-UseSageAttention); all v2 results so far were made without it.
# CUDA_PATH/CUDA_HOME are pointed at Triton's bundled CUDA for the ComfyUI process: machine-wide they
# name a miniconda env whose include folder has Python 3.8 headers, which broke Triton's compiles.
param(
    [double]$ReserveVramGb = 1.5,
    [int]$TimeoutSec = 900,
    [switch]$UseSageAttention
)
. (Join-Path $PSScriptRoot "v2_common.ps1")

if (Test-ComfyUp) {
    Write-Log "ComfyUI already running at $ComfyUrl (attaching)"
    exit 0
}
if (-not (Test-Path $ComfyPython)) {
    throw "ComfyUI python not found: $ComfyPython (set I2G_COMFY_ROOT)"
}

New-Item -ItemType Directory -Force $LogsDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $LogsDir "comfyui_$stamp.log"
$stderr = Join-Path $LogsDir "comfyui_$stamp.err.log"

$comfyArgs = @(
    "-s", "ComfyUI\main.py",
    "--windows-standalone-build",
    "--disable-auto-launch",
    "--listen", $ComfyHost,
    "--port", "$ComfyPort",
    "--reserve-vram", "$ReserveVramGb",
    "--fast"
)
if ($UseSageAttention) {
    $comfyArgs += "--use-sage-attention"
}

$env:CUDA_PATH = Join-Path $ComfyRoot "python_embeded\Lib\site-packages\triton\backends\nvidia"
$env:CUDA_HOME = $env:CUDA_PATH
Write-Log "Starting ComfyUI: $ComfyPython $($comfyArgs -join ' ')"
$proc = Start-Process -FilePath $ComfyPython -ArgumentList $comfyArgs -WorkingDirectory $ComfyRoot `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content -Path (Join-Path $LogsDir "comfyui.pid") -Value $proc.Id

$deadline = (Get-Date).AddSeconds($TimeoutSec)
while ((Get-Date) -lt $deadline) {
    if ($proc.HasExited) {
        throw "ComfyUI exited early (code $($proc.ExitCode)). See $stderr"
    }
    if (Test-ComfyUp) {
        Write-Log "ComfyUI is up at $ComfyUrl (pid $($proc.Id)); logs: $stdout"
        exit 0
    }
    Start-Sleep -Seconds 3
}
throw "ComfyUI did not answer within $TimeoutSec s. See $stdout and $stderr"
