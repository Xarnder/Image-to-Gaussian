param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot
Ensure-Directories -Root $Root

if (-not (Test-Path $ArchivePath)) {
    throw "SMPL-X archive not found: $ArchivePath"
}

$destDir = Join-Path $Root "models\_incoming"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$leaf = [IO.Path]::GetFileName($ArchivePath)
$destFile = Join-Path $destDir $leaf
if ([IO.Path]::GetFullPath($ArchivePath) -ne [IO.Path]::GetFullPath($destFile)) {
    Copy-Item -Force $ArchivePath $destFile
}
$linuxArchive = "/workspace/models/_incoming/$leaf"

Invoke-4danyone $Root python /opt/4danyone/scripts/download_smplx.py `
    --archive_path $linuxArchive `
    --model_dir /workspace/models `
    --gvhmr_root /opt/4danyone/third_party/GVHMR

$installed = Join-Path $Root "models\body_models\smplx\SMPLX_NEUTRAL.npz"
if (-not (Test-Path $installed)) {
    throw "SMPL-X install did not produce $installed"
}
Write-Log "Installed SMPL-X to $installed"
