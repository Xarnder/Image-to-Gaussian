# M0 smoke tests: CUDA, gsplat rasterization, sysmem-fallback policy, ComfyUI API round trip.
param([switch]$SkipFallbackTest)
. (Join-Path $PSScriptRoot "v2_common.ps1")

if (-not (Test-Path $VenvPython)) {
    throw "Missing $VenvPython. Run scripts\v2_setup_env.ps1 first."
}
$checkArgs = @("-m", "i2g.env_check", "--comfy-url", $ComfyUrl)
if ($SkipFallbackTest) {
    $checkArgs += "--skip-fallback-test"
}
Push-Location $RepoRoot
try {
    Invoke-Native $VenvPython @checkArgs
}
finally {
    Pop-Location
}
