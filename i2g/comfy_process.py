"""Start, check and stop the user's portable ComfyUI for the pipeline."""
from __future__ import annotations

import os
import subprocess
import time
import urllib.parse
from pathlib import Path

from .comfy_client import ComfyClient
from .errors import PipelineError
from .log import get_logger
from .paths import REPO_ROOT

log = get_logger("comfy")

PID_FILE = REPO_ROOT / "logs" / "comfyui.pid"   # shared with scripts\v2_start_comfy.ps1 / v2_stop_comfy.ps1
NODE_DIR_NAME = "i2g_export"
REQUIRED_NODES = ["LoadImage", "SaveImage", "SAM3DBody_Loader", "SAM3DBody_Predict", "I2G_SaveMHRPose",
                  "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly", "ModelSamplingAuraFlow",
                  "CFGNorm", "TextEncodeQwenImageEditPlus", "FluxKontextMultiReferenceLatentMethod",
                  "EmptySD3LatentImage", "KSampler", "VAEDecode"]


def _model_requirements() -> list[dict]:
    from .body import SAM3D_CHECKPOINT
    from .views import ANGLES_LORA, LIGHTNING_LORA, QWEN_CLIP, QWEN_UNET, QWEN_VAE
    return [
        {"node": "UNETLoader", "input": "unet_name", "file": QWEN_UNET, "folder": "models\\diffusion_models",
         "source": "huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI (split_files/diffusion_models)"},
        {"node": "CLIPLoader", "input": "clip_name", "file": QWEN_CLIP, "folder": "models\\text_encoders",
         "source": "huggingface.co/Comfy-Org/Qwen-Image_ComfyUI (split_files/text_encoders)"},
        {"node": "VAELoader", "input": "vae_name", "file": QWEN_VAE, "folder": "models\\vae",
         "source": "huggingface.co/Comfy-Org/Qwen-Image_ComfyUI (split_files/vae)"},
        {"node": "LoraLoaderModelOnly", "input": "lora_name", "file": ANGLES_LORA, "folder": "models\\loras",
         "source": "huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"},
        {"node": "LoraLoaderModelOnly", "input": "lora_name", "file": LIGHTNING_LORA, "folder": "models\\loras",
         "source": "huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning"},
        {"node": "SAM3DBody_Loader", "input": "model_file", "file": SAM3D_CHECKPOINT, "folder": "models\\detection",
         "source": "the checkpoint ComfyUI's SAM 3D Body nodes use (original weights: facebook/sam-3d-body-dinov3, gated)"},
    ]


def ensure_node_link(comfy_root: Path) -> bool:
    """Junction comfy_nodes/i2g_export into ComfyUI/custom_nodes. Returns True if it had to be created."""
    link = comfy_root / "ComfyUI" / "custom_nodes" / NODE_DIR_NAME
    target = REPO_ROOT / "comfy_nodes" / NODE_DIR_NAME
    if link.exists():
        return False
    log.info("Installing the pipeline's helper node into ComfyUI: %s -> %s", link, target)
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"could not create {link}: {result.stderr.strip() or result.stdout.strip()}",
                            hint="Run scripts\\v2_link_nodes.ps1, or copy the folder comfy_nodes\\i2g_export "
                                 "into ComfyUI\\custom_nodes yourself.")
    return True


def start_comfy(comfy_root: Path, url: str, logs_dir: Path, sage: bool, timeout_s: float,
                make_bar=None, stop_event=None) -> subprocess.Popen:
    """Start ComfyUI hidden and wait until its API answers. Returns the process."""
    python = comfy_root / "python_embeded" / "python.exe"
    main = comfy_root / "ComfyUI" / "main.py"
    for path in (python, main):
        if not path.exists():
            raise PipelineError(f"ComfyUI not found: {path} is missing",
                                hint="Pass --comfy_root pointing at the folder that contains python_embeded "
                                     "and ComfyUI (the portable build).")
    parsed = urllib.parse.urlparse(url)
    host, port = parsed.hostname or "127.0.0.1", str(parsed.port or 8188)
    args = [str(python), "-s", "ComfyUI\\main.py", "--windows-standalone-build", "--disable-auto-launch",
            "--listen", host, "--port", port, "--reserve-vram", "1.5", "--fast"]
    if sage:
        args.append("--use-sage-attention")

    env = os.environ.copy()
    # Machine-wide CUDA_PATH/CUDA_HOME point at a conda env with Python 3.8 headers; that broke Triton's
    # compiles. Use the CUDA bundled with Triton for this ComfyUI process only.
    bundled_cuda = comfy_root / "python_embeded" / "Lib" / "site-packages" / "triton" / "backends" / "nvidia"
    if bundled_cuda.exists():
        env["CUDA_PATH"] = env["CUDA_HOME"] = str(bundled_cuda)

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"comfyui_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log.info("Starting ComfyUI (first start takes ~5-8 min while its custom nodes load); log: %s", log_path)
    log.debug("ComfyUI command: %s", " ".join(args))
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")  # noqa: SIM115 - lives with the process
    proc = subprocess.Popen(args, cwd=str(comfy_root), env=env, stdout=log_file, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    client = ComfyClient(url)
    bar = make_bar(total=None, desc="Starting ComfyUI (5-8 min on first start):", unit="s",
                   bar_format="  {desc} {elapsed}") if make_bar else None
    start = time.monotonic()
    try:
        while time.monotonic() - start < timeout_s:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                raise PipelineError("ComfyUI startup cancelled.")
            if proc.poll() is not None:
                tail = _tail(log_path, 25)
                raise PipelineError(f"ComfyUI exited during startup (exit code {proc.returncode}).\n{tail}",
                                    hint=f"Read {log_path}. If port {port} is already in use, close the other "
                                         "ComfyUI window or pass --comfy_url for it.")
            if client.is_up():
                log.info("ComfyUI is up after %.0f s (pid %d)", time.monotonic() - start, proc.pid)
                return proc
            time.sleep(2)
            if bar is not None:
                bar.update(int(time.monotonic() - start) - bar.n)
    finally:
        if bar is not None:
            bar.close()
    proc.terminate()
    raise PipelineError(f"ComfyUI did not answer within {timeout_s:.0f} s",
                        hint=f"Read {log_path}. Raise --comfy_timeout if startup is just slow.")


def stop_comfy(proc: subprocess.Popen, client: ComfyClient) -> None:
    try:
        client.free()
    except Exception:  # noqa: BLE001 - best effort before killing
        pass
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    PID_FILE.unlink(missing_ok=True)
    log.info("Stopped the ComfyUI instance this run started")


def stop_kept_comfy(comfy_root: Path, client: ComfyClient) -> bool:
    """Stop a ComfyUI that this tool (or scripts\\v2_start_comfy.ps1) started earlier. Never touches others."""
    if not PID_FILE.exists():
        return False
    pid = int(PID_FILE.read_text().strip() or 0)
    expected = str((comfy_root / "python_embeded" / "python.exe").resolve()).lower()
    out = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid} -ErrorAction "
                          "SilentlyContinue).Path"], capture_output=True, text=True).stdout.strip().lower()
    if out != expected:
        PID_FILE.unlink(missing_ok=True)
        return False
    try:
        client.free()
    except Exception:  # noqa: BLE001
        pass
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    PID_FILE.unlink(missing_ok=True)
    log.info("Stopped ComfyUI (pid %d)", pid)
    return True


RESTORE_NODES = ["SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler", "VHS_LoadImagesPath"]


def check_comfy(client: ComfyClient, restore: bool = False) -> dict:
    """Verify the nodes and model files the pipeline needs, using ComfyUI's own file lookup."""
    info = client.object_info()
    if restore:
        from .paths import COMFY_ROOT
        from .restore import SEEDVR2_DIT, SEEDVR2_VAE
        missing = [n for n in RESTORE_NODES if n not in info]
        if missing:
            raise PipelineError(f"--restore needs ComfyUI nodes that are missing: {', '.join(missing)}",
                                hint="Install 'ComfyUI-SeedVR2_VideoUpscaler' (numz) and 'ComfyUI-VideoHelperSuite' "
                                     "with ComfyUI-Manager and restart ComfyUI, or run with --no-restore.")
        for f in (SEEDVR2_DIT, SEEDVR2_VAE):
            if not (COMFY_ROOT / "ComfyUI" / "models" / "SEEDVR2" / f).exists():
                raise PipelineError(rf"--restore needs ComfyUI\models\SEEDVR2\{f}",
                                    hint="Download it from huggingface.co/numz/SeedVR2_comfyUI into that folder "
                                         "(the SeedVR2 node would otherwise fetch it silently), or use --no-restore.")
    missing_nodes = [n for n in REQUIRED_NODES if n not in info]
    if "I2G_SaveMHRPose" in missing_nodes:
        raise PipelineError("ComfyUI is running without the pipeline's helper node (I2G_SaveMHRPose)",
                            hint="Close ComfyUI and run this again (the script installs the node and restarts "
                                 "ComfyUI), or run scripts\\v2_link_nodes.ps1 and restart ComfyUI yourself.")
    if missing_nodes:
        raise PipelineError(f"ComfyUI is missing nodes: {', '.join(missing_nodes)}",
                            hint="Update ComfyUI to v0.35 or newer (update\\update_comfyui.bat in the portable "
                                 "folder); SAM 3D Body and Qwen-Image-Edit nodes are built in from that version.")
    missing_models = []
    for req in _model_requirements():
        spec = info[req["node"]]["input"]["required"][req["input"]]
        options = spec[0]
        if options == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):  # newer (v3) node schema
            options = spec[1].get("options", [])
        names = {Path(str(o).replace("\\", "/")).name for o in options} if isinstance(options, list) else set()
        if req["file"] not in names:
            log.debug("%s.%s options: %s", req["node"], req["input"], options)
            missing_models.append(req)
    if missing_models:
        lines = "\n".join(f"  - {m['file']}  ->  ComfyUI\\{m['folder']}\\   (from {m['source']})"
                          for m in missing_models)
        raise PipelineError(f"ComfyUI cannot see {len(missing_models)} required model file(s):\n{lines}",
                            hint="Run  python3 scripts\\download_models.py  (downloads them into the right "
                                 "folders), then run again. If ComfyUI was already running, restart it so it "
                                 "rescans its model folders.")
    stats = client.system_stats()
    device = (stats.get("devices") or [{}])[0]
    return {"comfyui_version": stats.get("system", {}).get("comfyui_version"),
            "device": device.get("name"), "vram_free_gib": round(device.get("vram_free", 0) / 1024 ** 3, 1)}


def _tail(path: Path, n: int) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join("    " + line for line in lines[-n:])
