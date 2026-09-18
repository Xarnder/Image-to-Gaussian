# Shared paths for the v2 pipeline (native Windows, no Docker). Dot-source this file.
. (Join-Path $PSScriptRoot "common.ps1")

$script:RepoRoot = Get-RepoRoot
# machine-specific paths: env var, then config.local.json (copy config.example.json), then a default
$script:LocalConfig = $null
$configPath = Join-Path $RepoRoot "config.local.json"
if (Test-Path $configPath) {
    $script:LocalConfig = Get-Content $configPath -Raw | ConvertFrom-Json
}
$script:ComfyRoot = if ($env:I2G_COMFY_ROOT) { $env:I2G_COMFY_ROOT } elseif ($LocalConfig -and $LocalConfig.comfy_root) { $LocalConfig.comfy_root } else { "C:\ComfyUI_windows_portable" }
$script:ComfyPython = Join-Path $ComfyRoot "python_embeded\python.exe"
$script:ComfyCustomNodes = Join-Path $ComfyRoot "ComfyUI\custom_nodes"
$script:ComfyHost = "127.0.0.1"
$script:ComfyPort = 8188
$script:ComfyUrl = "http://${ComfyHost}:${ComfyPort}"
$script:SplatKitCache = Join-Path $ComfyCustomNodes "ComfyUI-SplatKit\bin\splat_backend_cache"
$script:ToolsDir = Join-Path $RepoRoot ".tools"
$script:VenvDir = Join-Path $RepoRoot ".venv-i2g"
$script:VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$script:LogsDir = Join-Path $RepoRoot "logs"

function Test-ComfyUp {
    try {
        Invoke-RestMethod -Uri "$ComfyUrl/system_stats" -TimeoutSec 3 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}
