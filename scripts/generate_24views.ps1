param(
    [Alias("Input")]
    [string]$InputImage = ".\input\character.png",
    [string]$StaticVideo,
    [string]$OutputDir,
    [int]$Views = 24,
    [int]$FrameIndex = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot

$name = Get-CharacterName $InputImage
$sixView = Join-Path $Root "output\$name\4danyone_6views\contact_sheet.png"
if (-not (Test-Path $sixView)) {
    throw "Six-view contact sheet not found at $sixView. Run test_6views.ps1 and inspect it before the 24-view job."
}

if (-not $OutputDir) {
    $OutputDir = "output\$name\4danyone_${Views}views"
}

$splat = @{
    InputImage = $InputImage
    OutputDir = $OutputDir
    Views = $Views
    FrameIndex = $FrameIndex
    Force = $Force
}
if ($StaticVideo) { $splat.StaticVideo = $StaticVideo }
& (Join-Path $PSScriptRoot "test_6views.ps1") @splat
