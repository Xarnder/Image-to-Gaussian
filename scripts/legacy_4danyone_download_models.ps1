param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot
Ensure-Directories -Root $Root

$smplx = Join-Path $Root "models\body_models\smplx\SMPLX_NEUTRAL.npz"
if (-not (Test-Path $smplx)) {
    Write-Log "SMPL-X is still missing. Hugging Face 4DAnyone weights can download, but inference cannot start until SMPL-X is installed."
}

Invoke-4danyone $Root python /opt/4danyone/scripts/download_model.py `
    --model_dir /workspace/models `
    --gvhmr_root /opt/4danyone/third_party/GVHMR

Write-Log "Published 4DAnyone / GVHMR / BiRefNet weights are in $Root\models"
