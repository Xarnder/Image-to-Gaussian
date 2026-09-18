param(
    [switch]$NoCache,
    [ValidateSet("4danyone", "nerfstudio", "all")]
    [string]$Service = "4danyone",
    [ValidateSet("torch", "full")]
    [string]$Target = "torch"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")
$Root = Get-RepoRoot
Ensure-Directories -Root $Root

function Invoke-DockerBuild {
    param(
        [string]$Dockerfile,
        [string]$TargetName,
        [string]$Tag,
        [string[]]$BuildArgs
    )
    $cmd = @(
        "build",
        "-f", (Join-Path $Root $Dockerfile),
        "--target", $TargetName,
        "-t", $Tag
    )
    if ($NoCache) {
        $cmd += "--no-cache"
    }
    foreach ($item in $BuildArgs) {
        $cmd += @("--build-arg", $item)
    }
    $cmd += $Root
    Invoke-Native docker @cmd
}

if ($Service -eq "nerfstudio") {
    Invoke-Compose $Root build nerfstudio
}
elseif ($Service -eq "all") {
    Invoke-DockerBuild -Dockerfile "Dockerfile.4danyone" -TargetName $Target -Tag "image-to-gaussian-4danyone:$Target" -BuildArgs @(
        "FDANYONE_COMMIT=61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5",
        "GVHMR_COMMIT=6ec3ca39336c50492c0fae65fba2fb831fc7d866"
    )
    if ($Target -eq "full") {
        Invoke-Native docker tag "image-to-gaussian-4danyone:full" "image-to-gaussian-4danyone:61e0ccd"
    }
    Invoke-Compose $Root build nerfstudio
}
else {
    Invoke-DockerBuild -Dockerfile "Dockerfile.4danyone" -TargetName $Target -Tag "image-to-gaussian-4danyone:$Target" -BuildArgs @(
        "FDANYONE_COMMIT=61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5",
        "GVHMR_COMMIT=6ec3ca39336c50492c0fae65fba2fb831fc7d866"
    )
    if ($Target -eq "full") {
        Invoke-Native docker tag "image-to-gaussian-4danyone:full" "image-to-gaussian-4danyone:61e0ccd"
    }
}

Write-Log "Build finished: service=$Service target=$Target"
if ($Target -eq "torch") {
    Write-Log "Next: .\scripts\verify_gpu.ps1  (CUDA smoke test). Do not install the rest of the ML stack or download weights until that prints True and NVIDIA GeForce RTX 4090."
}
