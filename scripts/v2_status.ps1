# One-shot status of a running v2 job: stage progress, ComfyUI step progress, GPU load, and a
# stuck/working verdict. Run it as often as you like; it changes nothing.
param([string]$Name = "test-image")
. (Join-Path $PSScriptRoot "v2_common.ps1")

$now = Get-Date
$views = Join-Path $RepoRoot "output\$Name\v2\02_views"
$manifest = Join-Path $views "views.json"
$planned = 22   # 11 angles x 2 seeds (i2g/views.py DEFAULT_VIEWS)
$done = if (Test-Path $manifest) { (Get-Content $manifest -Raw | ConvertFrom-Json).Count } else { 0 }
Write-Host ("Views generated: {0}/{1}" -f $done, $planned)

$jobLog = Get-ChildItem (Join-Path $LogsDir "v2_*.log") -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$jobAge = $null
if ($jobLog) {
    $jobAge = ($now - $jobLog.LastWriteTime).TotalMinutes
    Write-Host ("Pipeline log:    {0} (last write {1:N1} min ago)" -f $jobLog.Name, $jobAge)
    Get-Content $jobLog.FullName -Tail 3 | ForEach-Object { Write-Host "   $_" }
}

$comfyAge = $null
$comfyLog = Get-ChildItem (Join-Path $LogsDir "comfyui_*.err.log") -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($comfyLog) {
    $comfyAge = ($now - $comfyLog.LastWriteTime).TotalMinutes
    $last = (Get-Content $comfyLog.FullName -Tail 1) -split "`r" | Where-Object { $_.Trim() } | Select-Object -Last 1
    Write-Host ("ComfyUI:         last activity {0:N1} min ago: {1}" -f $comfyAge, $last.Trim())
}
if (Test-ComfyUp) {
    $q = Invoke-RestMethod "$ComfyUrl/queue" -TimeoutSec 10
    Write-Host ("ComfyUI queue:   running {0}, pending {1}   (live view: $ComfyUrl)" -f $q.queue_running.Count, $q.queue_pending.Count)
}
else {
    Write-Host "ComfyUI queue:   not reachable at $ComfyUrl"
}

$gpu = (nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits) -split ","
$power = [double]$gpu[2]
Write-Host ("GPU:             {0} MiB used, {1}% util, {2:N0} W" -f $gpu[0].Trim(), $gpu[1].Trim(), $power)

$recent = @($jobAge, $comfyAge) | Where-Object { $_ -ne $null } | Measure-Object -Minimum
if ($power -gt 150) {
    $verdict = "WORKING - GPU is computing"
}
elseif ($recent.Count -gt 0 -and $recent.Minimum -lt 3) {
    $verdict = "WORKING - loading/swapping models (low GPU power is normal here)"
}
else {
    $verdict = "POSSIBLY STUCK - no log activity for {0:N0} min and the GPU is idle" -f $recent.Minimum
}
Write-Host "Verdict:         $verdict"
