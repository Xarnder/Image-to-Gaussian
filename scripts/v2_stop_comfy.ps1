# Stops the ComfyUI instance started by v2_start_comfy.ps1 (frees VRAM first).
. (Join-Path $PSScriptRoot "v2_common.ps1")

$pidFile = Join-Path $LogsDir "comfyui.pid"
if (-not (Test-Path $pidFile)) {
    Write-Log "No comfyui.pid; nothing started by v2 to stop"
    exit 0
}
$comfyPid = [int](Get-Content $pidFile)
if (Test-ComfyUp) {
    try {
        Invoke-RestMethod -Method Post -Uri "$ComfyUrl/free" -ContentType "application/json" `
            -Body '{"unload_models": true, "free_memory": true}' -TimeoutSec 30 | Out-Null
    }
    catch {
        Write-Log "POST /free failed: $($_.Exception.Message)"
    }
}
$proc = Get-Process -Id $comfyPid -ErrorAction SilentlyContinue
if ($proc -and $proc.Path -eq $ComfyPython) {
    Stop-Process -Id $comfyPid -Confirm:$false
    Write-Log "Stopped ComfyUI (pid $comfyPid)"
}
else {
    Write-Log "pid $comfyPid is not the v2 ComfyUI process; leaving it alone"
}
Remove-Item $pidFile
