"""Minimal client for the ComfyUI HTTP API.

The v2 pipeline drives the user's ComfyUI headlessly: upload inputs, queue an API-format workflow,
wait for it, download the outputs, then POST /free so the next GPU stage starts from an empty card.

`run()` watches the websocket progress feed and interrupts a job whose sampler steps are far slower
than expected, which is how a VRAM spill into system RAM shows up on Windows (plan.md §5).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from .paths import COMFY_URL

# Keys under which nodes report files in /history outputs ("i2g_files" comes from our own nodes).
OUTPUT_KEYS = ("images", "gifs", "videos", "files", "3d", "i2g_files")


class ComfyError(RuntimeError):
    pass


class WatchdogTripped(ComfyError):
    pass


@dataclass(frozen=True)
class OutputFile:
    node_id: str
    filename: str
    subfolder: str
    type: str


class ComfyClient:
    def __init__(self, base_url: str = COMFY_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    # --- transport -------------------------------------------------------------------------

    def _request(self, method: str, path: str, data: bytes | None = None,
                 headers: dict | None = None, timeout: float | None = None) -> bytes:
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ComfyError(f"{method} {path} -> HTTP {exc.code}: {body[:4000]}") from exc

    def _get_json(self, path: str):
        return json.loads(self._request("GET", path))

    def _post_json(self, path: str, payload: dict, timeout: float | None = None):
        raw = self._request("POST", path, json.dumps(payload).encode("utf-8"),
                            {"Content-Type": "application/json"}, timeout)
        return json.loads(raw) if raw.strip() else {}

    # --- API -------------------------------------------------------------------------------

    def is_up(self) -> bool:
        try:
            self._request("GET", "/system_stats", timeout=3)
            return True
        except (OSError, ComfyError):
            return False

    def system_stats(self) -> dict:
        return self._get_json("/system_stats")

    def object_info(self, node_class: str | None = None) -> dict:
        return self._get_json("/object_info" + (f"/{urllib.parse.quote(node_class)}" if node_class else ""))

    def upload_image(self, path: str | Path, subfolder: str = "i2g", overwrite: bool = True) -> str:
        """Upload into ComfyUI's input folder; returns the name LoadImage expects."""
        path = Path(path)
        boundary = uuid.uuid4().hex
        fields = {"subfolder": subfolder, "overwrite": "true" if overwrite else "false", "type": "input"}
        body = bytearray()
        for key, value in fields.items():
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                     f"{value}\r\n").encode()
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                 f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        body += path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        resp = json.loads(self._request("POST", "/upload/image", bytes(body),
                                        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
                                        timeout=300))
        return f"{resp['subfolder']}/{resp['name']}" if resp.get("subfolder") else resp["name"]

    def queue(self, workflow: dict) -> str:
        resp = self._post_json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        if resp.get("node_errors"):
            raise ComfyError(f"workflow rejected: {json.dumps(resp['node_errors'])[:4000]}")
        return resp["prompt_id"]

    def history(self, prompt_id: str) -> dict | None:
        return self._get_json(f"/history/{prompt_id}").get(prompt_id)

    def wait(self, prompt_id: str, timeout: float = 3600, poll: float = 1.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = self.history(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyError(_format_error(status))
                if status.get("completed"):
                    return entry
            time.sleep(poll)
        raise TimeoutError(f"prompt {prompt_id} did not finish within {timeout} s")

    def run(self, workflow: dict, timeout: float = 3600, max_step_seconds: float | None = None) -> dict:
        """Queue a workflow and block until it finishes; returns its /history entry.

        max_step_seconds: interrupt if two consecutive sampler progress events are further apart
        than this (the first step is exempt because it includes model loading).
        """
        ws = self._open_ws() if max_step_seconds else None
        prompt_id = self.queue(workflow)
        if ws is not None:
            try:
                self._watch(ws, prompt_id, timeout, max_step_seconds)
            finally:
                ws.close()
        return self.wait(prompt_id, timeout)

    @staticmethod
    def output_files(entry: dict) -> list[OutputFile]:
        files = []
        for node_id, out in entry.get("outputs", {}).items():
            for key in OUTPUT_KEYS:
                for item in out.get(key) or []:
                    if isinstance(item, dict) and "filename" in item:
                        files.append(OutputFile(node_id, item["filename"], item.get("subfolder", ""),
                                                item.get("type", "output")))
        return files

    def download(self, item: OutputFile, dest: str | Path) -> Path:
        query = urllib.parse.urlencode({"filename": item.filename, "subfolder": item.subfolder, "type": item.type})
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self._request("GET", f"/view?{query}", timeout=600))
        return dest

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        self._post_json("/free", {"unload_models": unload_models, "free_memory": free_memory}, timeout=120)

    def interrupt(self) -> None:
        self._request("POST", "/interrupt", b"", {"Content-Type": "application/json"})

    # --- websocket watchdog ----------------------------------------------------------------

    def _open_ws(self):
        try:
            import websocket  # websocket-client
        except ImportError:
            return None
        ws_url = self.base_url.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
        ws = websocket.create_connection(ws_url, timeout=10)
        ws.settimeout(5)
        return ws

    def _watch(self, ws, prompt_id: str, timeout: float, max_step_seconds: float) -> None:
        import websocket

        deadline = time.monotonic() + timeout
        last_step = None
        while time.monotonic() < deadline:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                raw = None
            now = time.monotonic()
            if isinstance(raw, str):
                msg = json.loads(raw)
                data = msg.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    last_step = now
                elif kind == "executing":
                    last_step = None  # a new node starts; don't time its loading as a step
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        return
                elif kind in ("execution_success", "execution_error", "execution_interrupted"):
                    return
            if last_step is not None and now - last_step > max_step_seconds:
                self.interrupt()
                raise WatchdogTripped(
                    f"sampler step took > {max_step_seconds:.0f} s; interrupted (VRAM spill into system RAM?)")


def _format_error(status: dict) -> str:
    for event, data in status.get("messages", []):
        if event == "execution_error":
            return (f"node {data.get('node_id')} ({data.get('node_type')}): "
                    f"{data.get('exception_type')}: {data.get('exception_message')}")
    return f"ComfyUI reported an error: {json.dumps(status)[:2000]}"
