# Stop every Image-to-Gaussian job and ComfyUI, so the GPU (and the RAM the models use) is completely free.
#
#   stop_everything.bat              (double-click; asks for administrator rights, then stops everything)
#   powershell -File scripts\stop_everything.ps1 -ListOnly    (only show what would be stopped)
#
# Stops: single runs (image_to_splat.py), batches (batch_to_splat.py), the GUI (splat_gui.py), research
# scripts (i2g.experiment, i2g.reregister, headpass, pacgs), shell loops that drive them, and ComfyUI.
# Nothing else: other Python programs, games and browsers are left alone. Finished stages stay on disk, so
# runs continue where they stopped when you start them again (without Rebuild).
param([switch]$ListOnly)

$ErrorActionPreference = "Continue"
$patterns = @(
    "image_to_splat.py", "batch_to_splat.py", "splat_gui.py",
    "i2g.experiment", "i2g.reregister", "-m headpass", "-m pacgs",
    "ComfyUI\main.py", "ComfyUI/main.py", "run_all.sh"
)

function Find-Jobs {
    # only Python processes (the jobs themselves) and shells that run a test loop script; a terminal that
    # merely mentions image_to_splat.py in a command (grep, an editor) is left alone
    Get-CimInstance Win32_Process | Where-Object {
        $cmd = $_.CommandLine
        $name = $_.Name.ToLower()
        $isPython = $name.StartsWith("python")
        $isLoop = ($name -eq "bash.exe" -or $name -eq "sh.exe") -and $cmd -and $cmd -match "\.sh(\s|`"|$)" -and
                  $cmd.Contains("run_all.sh")
        $_.ProcessId -ne $PID -and $cmd -and ($isLoop -or ($isPython -and ($patterns | Where-Object { $cmd.Contains($_) })))
    }
}

function Show-Gpu {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $line = & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
        Write-Host "  GPU: $line"
    }
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Host ("  RAM: {0:N1} GB free of {1:N1} GB" -f ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB))
}

Write-Host ""
Write-Host "Before:"
Show-Gpu
$jobs = @(Find-Jobs)
if ($jobs.Count -eq 0) {
    Write-Host ""
    Write-Host "Nothing to stop: no Image-to-Gaussian job or ComfyUI is running."
} else {
    Write-Host ""
    Write-Host ("Found {0} process(es):" -f $jobs.Count)
    foreach ($j in $jobs) {
        $cmd = $j.CommandLine
        if ($cmd.Length -gt 150) { $cmd = $cmd.Substring(0, 150) + "..." }
        Write-Host ("  PID {0,-6} {1}" -f $j.ProcessId, $cmd)
    }
    if ($ListOnly) {
        Write-Host ""
        Write-Host "-ListOnly: nothing was stopped."
        exit 0
    }
    Write-Host ""
    Write-Host "Stopping them (and anything they started)..."
    foreach ($j in $jobs) {
        # /T takes the whole tree: a venv launcher's interpreter, ComfyUI's workers
        & taskkill.exe /PID $j.ProcessId /T /F 2>&1 | Out-Null
    }
    Start-Sleep -Seconds 3
    $left = @(Find-Jobs)
    foreach ($j in $left) { Stop-Process -Id $j.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 2
    $left = @(Find-Jobs)
    if ($left.Count -gt 0) {
        Write-Host ("Could not stop: " + (($left | ForEach-Object { $_.ProcessId }) -join ", ") +
                    ". Run this again as administrator, or restart the PC.")
    } else {
        Write-Host "All stopped."
    }
}

# the run lock releases itself when its process ends; this only tidies the note of who held it
$lockInfo = Join-Path (Split-Path $PSScriptRoot -Parent) "cache\gpu.lock.json"
if (-not $ListOnly -and (Test-Path $lockInfo)) { Remove-Item $lockInfo -Force -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "After:"
Show-Gpu
Write-Host ""
Write-Host "The GPU is free. Start the GUI or a .bat again; leave Rebuild unticked and each photo continues"
Write-Host "from its last finished stage."
