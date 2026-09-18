param(
    [Alias("Input")]
    [string]$InputImage = ".\input\character.png",
    [string]$StaticVideo,
    [string]$OutputDir,
    [int]$Views = 6,
    [int]$FrameIndex = 0,
    [switch]$Force,
    [switch]$SkipVideo,
    [switch]$SkipContactSheet
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot
Ensure-Directories -Root $Root

$resolvedInput = $InputImage
if (-not [IO.Path]::IsPathRooted($resolvedInput)) {
    $resolvedInput = Join-Path $Root $InputImage
}
if (-not (Test-Path $resolvedInput)) {
    throw @"
Character image not found: $resolvedInput

Place a front-facing full-body PNG at input\character.png and rerun.
"@
}

$leaf = [IO.Path]::GetFileName($resolvedInput)
$inputCopy = Join-Path $Root "input\$leaf"
if ([IO.Path]::GetFullPath($resolvedInput) -ne [IO.Path]::GetFullPath($inputCopy)) {
    Copy-Item -Force $resolvedInput $inputCopy
}

$name = Get-CharacterName $resolvedInput
if (-not $OutputDir) {
    $OutputDir = Join-Path $Root "output\$name\4danyone_${Views}views"
}
elseif (-not [IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $Root $OutputDir
}

$marker = Join-Path $OutputDir "cameras.json"
if ((Test-Path $marker) -and -not $Force) {
    Write-Log "Existing 4DAnyone output found at $OutputDir (use -Force to rerun inference)"
}
else {
    if (Test-Path $OutputDir) {
        if ($Force) {
            Write-Log "Removing previous output $OutputDir"
            Remove-Item -Recurse -Force $OutputDir
        }
        else {
            throw "Output directory already exists and is incomplete: $OutputDir (use -Force)"
        }
    }
}

$video = Join-Path $Root "input\${name}_static.mp4"
if ($StaticVideo) {
    $resolvedVideo = $StaticVideo
    if (-not [IO.Path]::IsPathRooted($resolvedVideo)) {
        $resolvedVideo = Join-Path $Root $StaticVideo
    }
    if (-not (Test-Path $resolvedVideo)) {
        throw "Static video not found: $resolvedVideo"
    }
    $destVideo = Join-Path $Root "input\$([IO.Path]::GetFileName($resolvedVideo))"
    if ([IO.Path]::GetFullPath($resolvedVideo) -ne [IO.Path]::GetFullPath($destVideo)) {
        Copy-Item -Force $resolvedVideo $destVideo
    }
    $video = $destVideo
    $SkipVideo = $true
}

if (-not $SkipVideo) {
    & (Join-Path $PSScriptRoot "make_static_video.ps1") -Input $inputCopy -Output $video -Force:$Force
}

$linuxVideo = "/workspace/input/" + [IO.Path]::GetFileName($video)

$smplx = Join-Path $Root "models\body_models\smplx\SMPLX_NEUTRAL.npz"
if (-not (Test-Path $smplx)) {
    throw @"
SMPL-X is not installed at $smplx

This weight is separately licensed. Download models_smplx_v1_1.zip from
https://smpl-x.is.tue.mpg.de/ after accepting the license, then run:

  .\scripts\install_smplx.ps1 -ArchivePath PATH\to\models_smplx_v1_1.zip
"@
}

$relOut = $OutputDir.Substring($Root.Length).TrimStart("\", "/")
$linuxOut = "/workspace/" + ($relOut -replace "\\", "/")

if (-not (Test-Path $marker) -or $Force) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $logFile = Join-Path $Root "logs\${name}_${Views}views_$stamp.log"
    $linuxLog = "/workspace/logs/" + [IO.Path]::GetFileName($logFile)
    Write-Log "Starting $Views-view 4DAnyone inference. Log: $logFile"
    Invoke-4danyone $Root -NeedGpu python /opt/pipeline/run_inference.py `
        --video-path $linuxVideo `
        --output-dir $linuxOut `
        --views-per-layer $Views `
        --log-file $linuxLog `
        --model-dir /workspace/models `
        --gvhmr-root /opt/4danyone/third_party/GVHMR `
        --enable-turbo True `
        --attention-backend auto
}

if (-not $SkipContactSheet) {
    $sheet = Join-Path $OutputDir "contact_sheet.png"
    Invoke-4danyone $Root python /opt/pipeline/contact_sheet.py `
        --input-dir $linuxOut `
        --output "$linuxOut/contact_sheet.png" `
        --frame-index $FrameIndex `
        --temporal-stats
    Write-Log "Review contact sheet: $sheet"
    Write-Log "Do not start 24-view generation until this result looks geometrically consistent."
}
