param(
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot
Ensure-Directories -Root $Root

Write-Log "Host nvidia-smi"
Invoke-Native nvidia-smi

Write-Log "Docker GPU passthrough (CUDA 12.8 Noble base)"
Invoke-Native docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

$image = "image-to-gaussian-4danyone:torch"
if ($Rebuild) {
    & (Join-Path $PSScriptRoot "build.ps1") -Service 4danyone -Target torch
}

$exists = docker image inspect $image 2>$null
if (-not $exists) {
    throw "Torch-stage image $image is missing. Run .\scripts\build.ps1 -Service 4danyone -Target torch"
}

Write-Log "Python 3.11 in torch-stage image"
Invoke-Native docker run --rm $image python3.11 --version
Invoke-Native docker run --rm $image python -c "import sys; print(sys.version); assert sys.version_info.releaselevel == 'final' and sys.version_info[:2] == (3, 11)"

Write-Log "PyTorch CUDA smoke test (exact requested prints)"
Invoke-Native docker run --rm --gpus all $image python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

Write-Log "Extended GPU check"
Invoke-Native docker run --rm --gpus all $image python /opt/pipeline/gpu_check.py

Write-Log "GPU verification succeeded. Safe to build the full 4DAnyone stack with: .\scripts\build.ps1 -Service 4danyone -Target full"
Write-Log "Do not download 4DAnyone weights until that full image exists and this smoke test has passed."
