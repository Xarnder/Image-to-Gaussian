# Runs v2 pipeline stages in order, one log per stage in logs\v2_<stage>_<time>.log.
# ComfyUI must be running (scripts\v2_start_comfy.ps1). Check progress with scripts\v2_status.ps1.
#   .\scripts\v2_run.ps1 -InputImage input\me.png
#   .\scripts\v2_run.ps1 -Stage register,train -WaitPid 1234   # start after another job exits
param(
    [string]$InputImage = "test-assets\test-image.png",
    [string[]]$Stage = @("ingest", "body", "views", "register", "train"),
    [int]$WaitPid = 0
)
. (Join-Path $PSScriptRoot "v2_common.ps1")

if ($WaitPid) {
    Write-Log "waiting for pid $WaitPid to exit"
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
}
foreach ($s in $Stage) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $log = Join-Path $LogsDir "v2_${s}_$stamp.log"
    Write-Log "stage $s -> $log"
    $p = Start-Process -FilePath $VenvPython -ArgumentList "-u", "-m", "i2g.run", "--image", $InputImage, "--stage", $s `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $log `
        -RedirectStandardError "$log.err" -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        throw "stage $s failed (exit $($p.ExitCode)); see $log and $log.err"
    }
}
Write-Log "stages done: $($Stage -join ', ')"
