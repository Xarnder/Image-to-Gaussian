param(
    [Parameter(Mandatory = $true)]
    [string]$Input,
    [string]$Output,
    [int]$Frames = 121,
    [int]$Width = 1080,
    [int]$Height = 1920,
    [int]$Fps = 24,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot
Ensure-Directories -Root $Root

$resolvedInput = $Input
if (-not [IO.Path]::IsPathRooted($resolvedInput)) {
    $resolvedInput = Join-Path $Root $Input
}
if (-not (Test-Path $resolvedInput)) {
    throw "Source image not found: $resolvedInput"
}

$name = Get-CharacterName $resolvedInput
if (-not $Output) {
    $Output = Join-Path $Root "input\${name}_static.mp4"
}
elseif (-not [IO.Path]::IsPathRooted($Output)) {
    $Output = Join-Path $Root $Output
}

if ((Test-Path $Output) -and -not $Force) {
    Write-Log "Static video already exists: $Output (use -Force to recreate)"
    exit 0
}

$inputDir = Join-Path $Root "input"
$leaf = [IO.Path]::GetFileName($resolvedInput)
$copied = Join-Path $inputDir $leaf
if ((Resolve-Path $resolvedInput).Path -ne (Resolve-Path $inputDir).Path) {
    if ((Resolve-Path $resolvedInput).Path -ne $copied) {
        Copy-Item -Force $resolvedInput $copied
    }
}
$linuxInput = "/workspace/input/$leaf"

$linuxOutput = "/workspace/input/" + [IO.Path]::GetFileName($Output)

Invoke-4danyone $Root python /opt/pipeline/make_static_video.py `
    --input $linuxInput `
    --output $linuxOutput `
    --frames $Frames `
    --width $Width `
    --height $Height `
    --fps $Fps

if (-not (Test-Path $Output)) {
    throw "Static video was not created: $Output"
}
Write-Log "Created $Output"
