"""Image -> MiniMax orbit candidates. No image-edit pipeline dependencies."""
from __future__ import annotations

import copy
import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

from .comfy_client import ComfyClient, _format_error
from .comfy_process import start_comfy
from .gpu_lock import GpuLock
from .paths import COMFY_ROOT, COMFY_URL, LOGS_DIR, OUTPUT_DIR, REPO_ROOT, safe_run_name
from .video_options import video_settings
from .video_sizing import SIZING_VERSION, image_layout, prepare_video_image

WORKFLOW = REPO_ROOT / "workflows" / "minimax_360.json"


def default_prompt():
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    return next(n for n in workflow["nodes"] if n["id"] == 133)["widgets_values_named"]["prompt"]


def generation_prompt(body):
    if body.get("custom_prompt_enabled") is not True:
        return default_prompt()
    prompt = body.get("custom_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a custom MiniMax prompt, or turn off the custom prompt toggle.")
    if len(prompt) > 32000:
        raise ValueError("The custom MiniMax prompt must be at most 32,000 characters.")
    return prompt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def api_workflow(workflow, image, seed, prefix, prompt=None, video_size=None):
    """Compile this versioned GUI graph, honoring disabled and bypassed nodes.

    Named widget values are authoritative; notes are never executed. Links override
    saved widget defaults, including resolution and duration calculations.
    """
    nodes = {n["id"]: n for n in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    def resolve(link_id):
        _, source, slot, *_ = links[link_id]
        node = nodes[source]
        if node.get("mode") == 4:
            output_type = node["outputs"][slot]["type"]
            matching = [i for i in node["inputs"] if i["type"] == output_type and i.get("link") is not None]
            if len(matching) != 1:
                raise ValueError(f"Ambiguous bypass on node {source}")
            return resolve(matching[0]["link"])
        if node.get("mode") == 2:
            raise ValueError(f"Active input uses disabled node {source}")
        return [str(source), slot]

    graph = {}
    for node in nodes.values():
        kind = node["type"]
        if node.get("mode", 0) in (2, 4) or kind in ("MarkdownNote", "Note"):
            continue
        values = node.get("widgets_values_named", {})
        inputs = {}
        for field in node.get("inputs", []):
            key = field["name"]
            if field.get("link") is not None:
                inputs[key] = resolve(field["link"])
            elif key in values and key != "upload":
                inputs[key] = copy.deepcopy(values[key])
        if kind == "LoadImage":
            inputs["image"] = image
        elif kind == "RandomNoise":
            inputs["noise_seed"] = seed
        elif kind == "SaveVideo":
            inputs["filename_prefix"] = prefix
        elif node["id"] == 133 and prompt is not None:
            inputs["prompt"] = prompt
        graph[str(node["id"])] = {"class_type": kind, "inputs": inputs}
    if video_size:
        width, height = video_size
        for node in ("150", "133"):
            graph[node]["inputs"].update(width=width, height=height)
        # The uploaded image is already fitted and padded to these exact dimensions.
        graph["150"]["inputs"]["keep_proportion"] = "resize"
        graph.pop("154", None)
    return graph


class VideoGeneration:
    def __init__(self, session_file=None, handoff_lock=None):
        self.lock = threading.RLock()
        self.handoff_lock = handoff_lock or threading.RLock()
        self.cancel = threading.Event()
        self.state = {"active": False, "status": "idle", "candidates": []}
        self.client = None
        self.prompt_id = None
        self.pending_approval = None
        self.root = None
        self.body = None
        self.session_file = session_file
        if session_file and session_file.exists():
            try:
                session = json.loads(session_file.read_text(encoding="utf-8"))
                root = Path(session["root"]).resolve()
                root.relative_to(OUTPUT_DIR.resolve())
                state = json.loads((root / "generation.json").read_text(encoding="utf-8"))
                self.root, self.body, self.state = root, session["body"], state
                self.state["active"] = False
                self.state["status"] = "review"
                self.state["message"] = "Saved candidates restored. Review or generate more. Any unfinished ComfyUI job must finish before a new batch."
            except (OSError, ValueError, KeyError):
                pass

    def clear(self):
        with self.lock:
            if self.state["active"]:
                raise ValueError("Stop generation before resetting.")
            self.root = self.body = None
            self.state = {"active": False, "status": "idle", "candidates": []}
            if self.session_file:
                self.session_file.unlink(missing_ok=True)

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            result["settings"] = {k: v for k, v in (self.body or {}).items()
                                  if k in ("video_preset", "coverage", "initialization", "iterations", "stride", "max_height", "candidate_count", "custom_prompt_enabled", "custom_prompt")}
            return result

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            if self.root:
                write_json(self.root / "generation.json", self.state)
                if self.session_file:
                    self.session_file.parent.mkdir(parents=True, exist_ok=True)
                    write_json(self.session_file, {"root": str(self.root), "body": self.body})

    def start(self, body, on_auto):
        with self.lock:
            if self.state["active"]:
                raise ValueError("Video generation is already running.")
            video_settings(body)
            prompt = generation_prompt(body)
            source = Path(body.get("input") or "").expanduser().resolve()
            if not source.is_file() or source.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
                raise ValueError("Choose one front-facing PNG, JPG, WebP, BMP or TIFF image.")
            experiment = str(body.get("experiment") or "").strip()
            if not experiment:
                raise ValueError("Enter an experiment name.")
            policy = body.get("generation_policy", "review")
            if policy not in ("review", "first"):
                raise ValueError("Choose review candidates or automatically use the first video.")
            count = int(body.get("candidate_count", 5)) if policy == "review" else 1
            if not 1 <= count <= 20:
                raise ValueError("Generate between 1 and 20 candidates per batch.")
            fingerprint = digest(source) + digest(WORKFLOW) + SIZING_VERSION
            if prompt != default_prompt():
                fingerprint += hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            root = OUTPUT_DIR / safe_run_name(experiment) / safe_run_name(body.get("output_name") or source.stem) / "orbit_candidates" / hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
            root.mkdir(parents=True, exist_ok=True)
            saved = root / "generation.json"
            previous = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else {}
            candidates = [c for c in previous.get("candidates", []) if Path(c["path"]).is_file()]
            self.root, self.body = root, copy.deepcopy(body)
            self.cancel.clear()
            self.pending_approval = None
            self.state = {"active": True, "status": "starting", "message": "Starting ComfyUI…", "input": str(source),
                          "experiment": experiment, "policy": policy, "requested": count, "completed": 0, "started_at": time.time(),
                          "candidates": candidates, "source_sha256": digest(source), "workflow_sha256": digest(WORKFLOW), "prompt": prompt}
            write_json(root / "workflow.json", json.loads(WORKFLOW.read_text(encoding="utf-8")))
            self.update()
            threading.Thread(target=self._generate, args=(source, count, on_auto), daemon=True).start()
            return {"ok": True}

    def stop(self):
        with self.lock:
            self.pending_approval = None
        self.cancel.set()
        self.update(message="Stopping generation; completed candidates are kept.")
        return {"ok": True}

    def approve_early(self, candidate_id, callback):
        """Reserve a completed candidate; the worker hands off after GPU cleanup."""
        with self.lock:
            if not self.state["active"]:
                return False
            if self.pending_approval:
                raise ValueError("An approved video is already waiting for generation to stop.")
            candidate = next((c for c in self.state["candidates"] if c["id"] == candidate_id), None)
            if not candidate or digest(candidate["path"]) != candidate["sha256"]:
                raise ValueError("Candidate is missing or changed. Generate it again.")
            self.pending_approval = (candidate_id, callback)
            self.cancel.set()
            self.update(status="stopping", selected=candidate_id,
                        message="Video approved. Cancelling remaining generation before reconstruction…")
            return True

    def _cancel_prompt(self):
        if not self.prompt_id:
            return
        q = self.client._get_json("/queue")
        if any(row[1] == self.prompt_id for row in q.get("queue_running", [])):
            self.client.interrupt()
        self.client._post_json("/queue", {"delete": [self.prompt_id]})

    def _generate(self, source, count, on_auto):
        gpu = GpuLock()
        selected = None
        try:
            if not gpu.try_acquire("MiniMax 360 video candidates"):
                raise ValueError("Another pipeline is using the GPU. Wait for it to finish.")
            self.client = ComfyClient()
            if not self.client.is_up():
                start_comfy(COMFY_ROOT, COMFY_URL, LOGS_DIR, True, 900, stop_event=self.cancel)
            if self.cancel.is_set():
                return
            if self.client.busy():
                raise ValueError("ComfyUI has another running or queued job. Wait for it to finish.")
            workflow = json.loads((self.root / "workflow.json").read_text(encoding="utf-8"))
            layout = image_layout(source)
            upload_source = prepare_video_image(source, self.root / "source_input.png", layout)
            self.update(image_layout=layout)
            write_json(self.root / "image_layout.json", layout)
            uploaded = self.client.upload_image(upload_source, subfolder="i2g/orbits", name=self.state["source_sha256"] + "_" + SIZING_VERSION + ".png")
            info = self.client.object_info()
            for index in range(count):
                if self.cancel.is_set():
                    break
                seed = secrets.randbits(50)
                candidate_id = "orbit_" + secrets.token_hex(6)
                graph = api_workflow(workflow, uploaded, seed, "video/i2g/" + candidate_id, self.state["prompt"], layout["video_size"])
                missing = sorted({n["class_type"] for n in graph.values()} - info.keys())
                if missing:
                    raise ValueError("ComfyUI is missing workflow nodes: " + ", ".join(missing))
                write_json(self.root / (candidate_id + "_api.json"), graph)
                self.update(status="generating", progress=0, message=f"Generating candidate {index+1} of {count} · seed {seed}")
                ws = self.client._open_ws()
                if ws:
                    ws.settimeout(1)
                try:
                    self.prompt_id = self.client.queue(graph)
                    self.update(prompt_id=self.prompt_id)
                    deadline = time.monotonic() + 7200
                    while True:
                        if self.cancel.is_set():
                            self._cancel_prompt()
                            break
                        entry = self.client.history(self.prompt_id)
                        if entry and entry.get("status", {}).get("status_str") == "error":
                            raise RuntimeError(_format_error(entry["status"]))
                        if entry and entry.get("status", {}).get("completed"):
                            break
                        if time.monotonic() > deadline:
                            self._cancel_prompt()
                            raise TimeoutError("Video generation exceeded two hours; this job was stopped.")
                        if ws:
                            try:
                                raw = ws.recv()
                                msg = json.loads(raw) if isinstance(raw, str) else {}
                                data = msg.get("data", {})
                                if data.get("prompt_id") == self.prompt_id and msg.get("type") == "progress":
                                    self.update(progress=data['value']/max(1, data['max']), message=f"Candidate {index+1}/{count} · sampling {data['value']}/{data['max']}")
                                elif data.get("prompt_id") == self.prompt_id and msg.get("type") == "executing" and data.get("node"):
                                    labels = {"130": "loading text encoder", "129": "loading MiniMax video model", "121": "loading video decoder",
                                              "133": "encoding image and prompt", "127": "sampling video", "124": "decoding video frames", "92": "saving video"}
                                    label = labels.get(str(data["node"]), "preparing workflow")
                                    self.update(message=f"Candidate {index+1}/{count} · {label}")
                            except TimeoutError:
                                pass
                            except Exception as exc:
                                if "timed out" not in str(exc).lower():
                                    ws.close()
                                    ws = None
                        else:
                            self.cancel.wait(1)
                finally:
                    if ws:
                        ws.close()
                if self.cancel.is_set():
                    break
                outputs = [f for f in self.client.output_files(entry) if f.node_id == "92" and f.filename.lower().endswith(".mp4")]
                if len(outputs) != 1:
                    raise RuntimeError("Workflow finished without one MP4 from SaveVideo.")
                path = self.client.download(outputs[0], self.root / (candidate_id + ".mp4"))
                candidate = {"id": candidate_id, "path": str(path), "seed": seed, "sha256": digest(path)}
                candidates = self.snapshot()["candidates"] + [candidate]
                self.update(candidates=candidates, completed=index+1, progress=0)
                selected = candidate
                self.prompt_id = None
            self.update(status="stopped" if self.cancel.is_set() else "review", message="Generation stopped. Completed videos are available." if self.cancel.is_set() else "Review the eyes and the complete orbit, then select a video or generate more.")
        except Exception as exc:
            try:
                self._cancel_prompt()
            except Exception:
                pass
            self.update(status="stopped" if self.cancel.is_set() else "failed", message=str(exc))
        finally:
            cleanup_ok = True
            # Interrupt is asynchronous: wait for our GPU job to leave the queue.
            try:
                if self.cancel.is_set() and self.prompt_id:
                    self._cancel_prompt()
                    deadline = time.monotonic() + 120
                    while True:
                        q = self.client._get_json("/queue")
                        if not any(row[1] == self.prompt_id for row in q.get("queue_running", []) + q.get("queue_pending", [])):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError("Generation has not stopped yet. Wait for ComfyUI, then approve again.")
                        time.sleep(0.25)
            except Exception as exc:
                cleanup_ok = False
                self.update(status="failed", message=str(exc))
            # Release diffusion weights before camera recovery or Gaussian training.
            try:
                if self.client and not self.client.busy():
                    self.client.free()
            except Exception as exc:
                cleanup_ok = False
                self.update(status="failed", message=f"Could not release generation resources: {exc}")
            self.prompt_id = None
            gpu.release()
            with self.handoff_lock:
                self.update(active=False)
                pending, self.pending_approval = self.pending_approval, None
                if pending and cleanup_ok:
                    result = pending[1]()
                    if not result.get("ok"):
                        self.update(status="failed", message=result.get("error", "Could not start reconstruction"))
                elif selected and cleanup_ok and not self.cancel.is_set() and self.state["policy"] == "first" and self.state["status"] == "review":
                    result = on_auto(selected["id"], "automatic-first")
                    if not result.get("ok"):
                        self.update(status="failed", message=result.get("error", "Could not start reconstruction"))

    def selection(self, candidate_id, basis="manual-review"):
        with self.lock:
            if self.state["active"]:
                raise ValueError("Wait for generation to finish, or stop it before choosing a video.")
            candidate = next((c for c in self.state["candidates"] if c["id"] == candidate_id), None)
            if not candidate or digest(candidate["path"]) != candidate["sha256"]:
                raise ValueError("Candidate is missing or changed. Generate it again.")
            write_json(self.root / "selection.json", {**candidate, "basis": basis, "selected_at": time.time()})
            return self.body | {"mode": "video", "input": candidate["path"], "approved_video": True, "approval_basis": basis,
                                "source_image": self.state["input"],
                                "output_name": self.body.get("output_name") or safe_run_name(Path(self.state["input"]).stem)}
