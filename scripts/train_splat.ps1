param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$OutputDir,
    [string]$ExportDir,
    [string]$Method = "splatfacto",
    [int]$MaxNumIterations = 30000,
    [switch]$Force,
    [switch]$SkipExport
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot

if (-not [IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $Root $DataDir
}
if (-not (Test-Path (Join-Path $DataDir "transforms.json"))) {
    throw "Nerfstudio dataset missing transforms.json: $DataDir"
}

$name = Split-Path (Split-Path $DataDir -Parent) -Leaf
if (-not $OutputDir) {
    $OutputDir = Join-Path $Root "output\$name\nerfstudio"
}
elseif (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $Root $OutputDir
}
if (-not $ExportDir) {
    $ExportDir = Join-Path $Root "output\$name\final"
}
elseif (-not [IO.Path]::IsPathRooted($ExportDir)) {
    $ExportDir = Join-Path $Root $ExportDir
}

$relData = "/workspace/" + ($DataDir.Substring($Root.Length).TrimStart("\", "/") -replace "\\", "/")
$relOut = "/workspace/" + ($OutputDir.Substring($Root.Length).TrimStart("\", "/") -replace "\\", "/")
$relExport = "/workspace/" + ($ExportDir.Substring($Root.Length).TrimStart("\", "/") -replace "\\", "/")

$config = Get-ChildItem -Path $OutputDir -Filter config.yml -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($config -and -not $Force) {
    Write-Log "Existing Nerfstudio config: $($config.FullName)"
}
else {
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Invoke-Compose $Root run --rm --no-deps -p 7007:7007 nerfstudio ns-train $Method `
        --data $relData `
        --output-dir $relOut `
        --max-num-iterations $MaxNumIterations `
        --vis viewer `
        --viewer.websocket-port 7007 `
        --viewer.quit-on-train-completion True `
        --pipeline.model.background-color random `
        nerfstudio-data `
        --orientation-method none `
        --center-method none `
        --auto-scale-poses False
    $config = Get-ChildItem -Path $OutputDir -Filter config.yml -Recurse -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

if (-not $config) {
    throw "Training finished but config.yml was not found under $OutputDir"
}

if (-not $SkipExport) {
    New-Item -ItemType Directory -Force -Path $ExportDir | Out-Null
    $linuxConfig = "/workspace/" + ($config.FullName.Substring($Root.Length).TrimStart("\", "/") -replace "\\", "/")
    Invoke-Compose $Root run --rm --no-deps nerfstudio ns-export gaussian-splat `
        --load-config $linuxConfig `
        --output-dir $relExport `
        --output-filename character.ply `
        --ply-color-mode sh_coeffs
    $ply = Join-Path $ExportDir "character.ply"
    if (-not (Test-Path $ply)) {
        throw "Gaussian export did not produce $ply"
    }
    Write-Log "Exported static Gaussian splat: $ply"
}
