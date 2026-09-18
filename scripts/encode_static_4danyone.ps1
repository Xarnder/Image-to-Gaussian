param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Input,
    [string]$Output,
    [int]$Frames = 121,
    [int]$Width = 1080,
    [int]$Height = 1920,
    [int]$Fps = 24
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot

$resolvedInput = $Input
if (-not [IO.Path]::IsPathRooted($resolvedInput)) {
    $resolvedInput = Join-Path $Root $Input
}
if (-not (Test-Path $resolvedInput)) {
    throw "Source image not found: $resolvedInput"
}

$py = Join-Path $Root "tools\encode_static_4danyone.py"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $python) {
    throw "python was not found on PATH"
}

$argsList = @(
    $py,
    $resolvedInput,
    "--width", "$Width",
    "--height", "$Height",
    "--frames", "$Frames",
    "--fps", "$Fps"
)
if ($Output) {
    if (-not [IO.Path]::IsPathRooted($Output)) {
        $Output = Join-Path $Root $Output
    }
    $argsList += @("--output", $Output)
}

Invoke-Native $python.Source @argsList
