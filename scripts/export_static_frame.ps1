param(
    [string]$DataDir,
    [string]$OutputDir,
    [int]$FrameIndex = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot

if (-not $DataDir) {
    throw "Pass -DataDir to a finished 4DAnyone output directory (contains cameras.json)."
}
if (-not [IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $Root $DataDir
}
if (-not (Test-Path (Join-Path $DataDir "cameras.json"))) {
    throw "Not a 4DAnyone output directory (missing cameras.json): $DataDir"
}

$rel = $DataDir.Substring($Root.Length).TrimStart("\", "/")
$linuxData = "/workspace/" + ($rel -replace "\\", "/")
$name = Split-Path (Split-Path $DataDir -Parent) -Leaf
if (-not $OutputDir) {
    $OutputDir = Join-Path $Root "output\$name\static_frameset_frame_$("{0:000}" -f $FrameIndex)"
}
elseif (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $Root $OutputDir
}

if ((Test-Path (Join-Path $OutputDir "transforms.json")) -and -not $Force) {
    Write-Log "Static frameset already exists: $OutputDir"
    exit 0
}
if ((Test-Path $OutputDir) -and $Force) {
    Remove-Item -Recurse -Force $OutputDir
}

$relOut = $OutputDir.Substring($Root.Length).TrimStart("\", "/")
$linuxOut = "/workspace/" + ($relOut -replace "\\", "/")

Invoke-4danyone $Root python /opt/4danyone/scripts/export_nerfstudio.py `
    --data_dir $linuxData `
    --output_dir $linuxOut `
    --frame_index $FrameIndex `
    --model_dir /workspace/models `
    --device cuda:0

if (-not (Test-Path (Join-Path $OutputDir "transforms.json"))) {
    throw "Nerfstudio export did not write transforms.json under $OutputDir"
}
Write-Log "Exported static frameset to $OutputDir (no COLMAP)."
