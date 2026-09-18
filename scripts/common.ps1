$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    if ($PSScriptRoot) {
        return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    }
    return (Get-Location).Path
}

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] $Message"
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ArgumentList
    )
    Write-Log ("RUN: " + $FilePath + " " + ($ArgumentList -join " "))
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE : $FilePath $($ArgumentList -join ' ')"
    }
}

function Ensure-Directories {
    param([string]$Root)
    foreach ($name in @("input", "output", "models", "cache\huggingface", "cache\torch", "logs")) {
        $path = Join-Path $Root $name
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path | Out-Null
        }
    }
}

function Invoke-Compose {
    param(
        [string]$Root,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ArgumentList
    )
    if ($ArgumentList.Count -gt 0 -and $ArgumentList[0] -eq "run" -and -not ($ArgumentList -contains "-T")) {
        $ArgumentList = @("run", "-T") + $ArgumentList[1..($ArgumentList.Length - 1)]
    }
    Push-Location $Root
    try {
        Invoke-Native docker compose @ArgumentList
    }
    finally {
        Pop-Location
    }
}

function Invoke-4danyone {
    param(
        [string]$Root,
        [switch]$NeedGpu,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$ArgumentList
    )
    $image = "image-to-gaussian-4danyone:full"
    $inspect = docker image inspect $image 2>$null
    if (-not $inspect) {
        $image = "image-to-gaussian-4danyone:61e0ccd"
        $inspect = docker image inspect $image 2>$null
    }
    if (-not $inspect) {
        throw "4DAnyone image is missing. Wait for the full build to finish, then rerun."
    }
    $run = @("run", "--rm")
    if ($NeedGpu) {
        $run += @("--gpus", "all")
    }
    $run += @(
        "--shm-size", "4g",
        "-e", "HF_HOME=/workspace/cache/huggingface",
        "-e", "HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface",
        "-e", "HF_HUB_CACHE=/workspace/cache/huggingface",
        "-e", "TRANSFORMERS_CACHE=/workspace/cache/huggingface",
        "-e", "TORCH_HOME=/workspace/cache/torch",
        "-e", "XDG_CACHE_HOME=/workspace/cache",
        "-v", "${Root}/input:/workspace/input",
        "-v", "${Root}/output:/workspace/output",
        "-v", "${Root}/models:/workspace/models",
        "-v", "${Root}/cache:/workspace/cache",
        "-v", "${Root}/logs:/workspace/logs",
        "-v", "${Root}/test-assets:/workspace/test-assets",
        "-v", "${Root}/tools:/opt/pipeline:ro",
        "-v", "${Root}/patches/gvhmr_body_model.py:/opt/4danyone/third_party/GVHMR/hmr4d/utils/body_model/body_model.py:ro",
        "-v", "${Root}/patches/official/conditioning.py:/opt/4danyone/fdanyone/model/conditioning.py:ro",
        "-v", "${Root}/patches/official/inference.py:/opt/4danyone/fdanyone/model/inference.py:ro",
        "-v", "${Root}/patches/official/wan_video_dit.py:/opt/4danyone/fdanyone/vendor/diffsynth/models/wan_video_dit.py:ro",
        "-w", "/opt/4danyone",
        $image
    ) + $ArgumentList
    Invoke-Native docker @run
}

function Get-CharacterName {
    param([string]$InputPath)
    return [IO.Path]::GetFileNameWithoutExtension($InputPath)
}

function Convert-ToLinuxPath {
    param([string]$RelativePosix)
    return ($RelativePosix -replace "\\", "/")
}
