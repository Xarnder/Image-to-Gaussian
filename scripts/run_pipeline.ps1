param(
    [Alias("Input")]
    [string]$InputImage = ".\input\character.png",
    [string]$StaticVideo,
    [ValidateSet("prepare", "test", "generate", "export", "train", "all")]
    [string]$Stage = "test",
    [int]$FrameIndex = 0,
    [int]$Views = 24,
    [switch]$Force,
    [switch]$Rebuild,
    [switch]$SkipReview
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot
Ensure-Directories -Root $Root

if ($Rebuild) {
    & (Join-Path $PSScriptRoot "build.ps1") -Service all
}

$resolvedInput = $InputImage
if (-not [IO.Path]::IsPathRooted($resolvedInput)) {
    $resolvedInput = Join-Path $Root $InputImage
}
$name = Get-CharacterName $resolvedInput
$sixDir = "output\$name\4danyone_6views"
$denseDir = "output\$name\4danyone_${Views}views"
$frameset = "output\$name\static_frameset_frame_$("{0:000}" -f $FrameIndex)"

function Invoke-Prepare {
    & (Join-Path $PSScriptRoot "verify_gpu.ps1")
    if ($StaticVideo) {
        Write-Log "Using provided static video: $StaticVideo"
    }
    else {
        & (Join-Path $PSScriptRoot "make_static_video.ps1") -Input $resolvedInput -Force:$Force
    }
    $smplx = Join-Path $Root "models\body_models\smplx\SMPLX_NEUTRAL.npz"
    if (Test-Path $smplx) {
        & (Join-Path $PSScriptRoot "legacy_4danyone_download_models.ps1")
    }
    else {
        Write-Log "Skipping published-weight download until SMPL-X is installed (inference still needs both)."
    }
}

function Invoke-Test {
    Invoke-Prepare
    $testArgs = @{
        InputImage = $resolvedInput
        FrameIndex = $FrameIndex
        Force = $Force
    }
    if ($StaticVideo) { $testArgs.StaticVideo = $StaticVideo }
    & (Join-Path $PSScriptRoot "test_6views.ps1") @testArgs
}

function Invoke-Generate {
    $genArgs = @{
        InputImage = $resolvedInput
        Views = $Views
        FrameIndex = $FrameIndex
        Force = $Force
    }
    if ($StaticVideo) { $genArgs.StaticVideo = $StaticVideo }
    & (Join-Path $PSScriptRoot "generate_24views.ps1") @genArgs
}

function Invoke-Export {
    $source24 = Join-Path $Root $denseDir
    $source6 = Join-Path $Root $sixDir
    $source = $source24
    if (-not (Test-Path (Join-Path $source "cameras.json"))) {
        if (Test-Path (Join-Path $source6 "cameras.json")) {
            Write-Log "24-view output missing; exporting from six-view result $source6"
            $source = $source6
        }
        else {
            throw "4DAnyone output is missing. Run -Stage test or -Stage generate first."
        }
    }
    & (Join-Path $PSScriptRoot "export_static_frame.ps1") -DataDir $source -FrameIndex $FrameIndex -Force:$Force
}

function Invoke-Train {
    $data = Join-Path $Root $frameset
    if (-not (Test-Path (Join-Path $data "transforms.json"))) {
        throw "Static frameset missing. Run -Stage export first."
    }
    & (Join-Path $PSScriptRoot "train_splat.ps1") -DataDir $data -Force:$Force
}

switch ($Stage) {
    "prepare" { Invoke-Prepare }
    "test" { Invoke-Test }
    "generate" { Invoke-Generate }
    "export" { Invoke-Export }
    "train" { Invoke-Train }
    "all" {
        if ($SkipReview) {
            Invoke-Test
            Invoke-Export
            Invoke-Train
        }
        else {
            Write-Log "Quality gate: running six-view test. After you inspect contact_sheet.png, rerun with -Stage generate, then -Stage export, then -Stage train. Or pass -SkipReview to export+train from the six-view result."
            Invoke-Test
        }
    }
}

Write-Log "Stage '$Stage' finished."
