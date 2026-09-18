# Creates .venv-i2g: standalone CPython 3.11 + torch 2.8.0+cu128 + gsplat 1.4.0 (native Windows).
# Safe to rerun; -Force recreates the venv.
param([switch]$Force)
. (Join-Path $PSScriptRoot "v2_common.ps1")

$GsplatWheelName = "gsplat-1.4.0+pt28cu128-cp311-cp311-win_amd64.whl"
$GsplatWheelSha256 = "a3026d43405bca2be175c8fbce986ea1e91c40226887011eac6c72964640149f"
$GsplatWheelUrl = "https://github.com/mickmumpitz/ComfyUI-SplatKit/releases/download/gsplat-1.4.0-pt28-cu128/$GsplatWheelName"
$PythonDirName = "cpython-3.11.16-windows-x86_64-none"

$env:UV_CACHE_DIR = Join-Path $RepoRoot "cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $ToolsDir "python"
New-Item -ItemType Directory -Force $ToolsDir, $env:UV_CACHE_DIR, $LogsDir | Out-Null

# uv (the Python package manager): this repo's copy, else one ComfyUI-SplatKit already downloaded,
# else a uv installed on PATH.
$uv = Join-Path $ToolsDir "uv\uv.exe"
if (-not (Test-Path $uv)) {
    $src = Join-Path $SplatKitCache "uv.exe"
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force (Split-Path $uv) | Out-Null
        Copy-Item $src $uv
        Write-Log "Copied uv from SplatKit cache"
    }
    elseif ($onPath) {
        $uv = $onPath.Source
        Write-Log "Using uv from PATH: $uv"
    }
    else {
        throw "uv is not installed. Install it once with:  winget install --id=astral-sh.uv -e   (or: python -m pip install uv), open a new terminal, then run this script again."
    }
}

# Python 3.11: a relocatable python-build-standalone CPython kept inside this repo, so the venv
# does not depend on SplatKit's folder surviving an update.
$pyHome = Join-Path $env:UV_PYTHON_INSTALL_DIR $PythonDirName
$pyExe = Join-Path $pyHome "python.exe"
if (-not (Test-Path $pyExe)) {
    $srcPy = Join-Path $SplatKitCache "python\$PythonDirName"
    if (Test-Path (Join-Path $srcPy "python.exe")) {
        New-Item -ItemType Directory -Force $env:UV_PYTHON_INSTALL_DIR | Out-Null
        Copy-Item $srcPy $env:UV_PYTHON_INSTALL_DIR -Recurse
        Write-Log "Copied standalone CPython 3.11.16 from SplatKit cache"
    }
    else {
        Invoke-Native $uv python install 3.11.16
    }
}
if (-not (Test-Path $pyExe)) {
    throw "Python 3.11 not found at $pyExe"
}

if ($Force -and (Test-Path $VenvDir)) {
    Write-Log "Removing existing venv $VenvDir"
    Remove-Item -Recurse -Force $VenvDir
}
if (-not (Test-Path $VenvPython)) {
    Invoke-Native $uv venv $VenvDir --python $pyExe --seed
}
# python3.exe alongside python.exe, so `python3` works inside the activated environment
$venvPython3 = Join-Path $VenvDir "Scripts\python3.exe"
if (-not (Test-Path $venvPython3)) {
    Copy-Item $VenvPython $venvPython3
}

Invoke-Native $uv pip install --python $VenvPython `
    -r (Join-Path $RepoRoot "requirements.txt") `
    --extra-index-url https://download.pytorch.org/whl/cu128 `
    --index-strategy unsafe-best-match

# gsplat: SplatKit's prebuilt Windows wheel (local copy first, GitHub release as fallback), hash-checked.
$wheel = Join-Path $ToolsDir $GsplatWheelName
if (-not (Test-Path $wheel)) {
    $localWheel = Join-Path $SplatKitCache $GsplatWheelName
    if (Test-Path $localWheel) {
        Copy-Item $localWheel $wheel
    }
    else {
        Write-Log "Downloading $GsplatWheelUrl"
        Invoke-WebRequest -Uri $GsplatWheelUrl -OutFile $wheel
    }
}
$hash = (Get-FileHash $wheel -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $GsplatWheelSha256) {
    Remove-Item $wheel
    throw "gsplat wheel SHA-256 mismatch: $hash"
}
Invoke-Native $uv pip install --python $VenvPython $wheel --no-deps

Invoke-Native $VenvPython -c "import torch, gsplat, cv2, onnxruntime; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('gsplat', gsplat.__version__, 'opencv', cv2.__version__, 'onnxruntime', onnxruntime.__version__)"
Write-Log "v2 environment ready: $VenvPython"
