"""Serial, unattended image -> orbit -> splat batches."""
import copy
import hashlib
import json
import threading
import time
from pathlib import Path

from .paths import OUTPUT_DIR, safe_run_name, deliverable_path, find_deliverable
from .video_generation import write_json, generation_prompt
from .video_options import video_settings
from .video_sizing import legacy_aspect_mismatch
from .batch_outputs import organize_ply


class GeneratedBatch:
    def __init__(self, session_file=None):
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.after = threading.Event()
        self.state = {"active": False, "items": []}
        self.manifest = None
        self.session_file = session_file
        if session_file and session_file.is_file():
            try:
                manifest = Path(json.loads(session_file.read_text())["manifest"]).resolve()
                manifest.relative_to(OUTPUT_DIR.resolve())
                self.state = json.loads(manifest.read_text(encoding="utf-8"))
                self.manifest = manifest
                if self.state["active"]:
                    self.state.update(active=False, status="stopped", message="Saved batch restored. Start again to continue; completed splats are skipped.")
                    for item in self.state["items"]:
                        if item["status"] == "running":
                            item["status"] = "stopped"
            except (OSError, ValueError, KeyError):
                self.state = {"active": False, "items": []}

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def clear(self):
        with self.lock:
            if self.state["active"]:
                raise ValueError("Stop the folder batch first.")
            self.state = {"active": False, "items": []}
            self.manifest = None
            if self.session_file:
                self.session_file.unlink(missing_ok=True)

    def save(self):
        if self.manifest:
            write_json(self.manifest, self.state)
            if self.session_file:
                self.session_file.parent.mkdir(parents=True, exist_ok=True)
                write_json(self.session_file, {"manifest": str(self.manifest)})

    def start(self, body, photos, process):
        with self.lock:
            if self.state["active"]:
                raise ValueError("A folder batch is already running.")
            if body.get("generation_policy") != "first" and int(body.get("candidate_count", 5)) != 1:
                raise ValueError("Folder batches require automatic first-video selection or exactly one candidate.")
            video_settings(body)
            generation_prompt(body)
            folder = Path(body.get("input", "")).resolve()
            if not folder.is_dir() or not photos:
                raise ValueError("Choose a folder containing front-facing images.")
            if not str(body.get("experiment", "")).strip():
                raise ValueError("Enter an experiment name.")
            experiment = safe_run_name(body["experiment"])
            items = []
            used = set()
            for source in photos:
                name = safe_run_name(source.stem)
                # Hash every ambiguous stem (including case/sanitization collisions).
                if sum(safe_run_name(p.stem).casefold() == name.casefold() for p in photos) > 1:
                    name += "_" + hashlib.sha256(str(source.relative_to(folder)).encode()).hexdigest()[:10]
                if name.casefold() in used:
                    raise ValueError("Duplicate image output names.")
                used.add(name.casefold())
                final = deliverable_path(OUTPUT_DIR / experiment / name / "video" / "final", name, "character.ply", experiment)
                items.append({"input": str(source), "name": name, "ply": str(final), "status": "queued"})
            self.cancel.clear(); self.after.clear()
            self.manifest = OUTPUT_DIR / experiment / "generated_batch.json"
            self.manifest.parent.mkdir(parents=True, exist_ok=True)
            self.state = {"active": True, "input": str(folder), "experiment": body["experiment"],
                          "status": "running", "items": items, "completed": 0, "total": len(items),
                          "message": "Starting automated image batch…", "settings": copy.deepcopy(body)}
            self.save()
            threading.Thread(target=self._run, args=(copy.deepcopy(body), process), daemon=True).start()
            return {"ok": True}

    def stop(self, how="now"):
        self.after.set()
        if how != "after":
            self.cancel.set()
        with self.lock:
            self.state["stop_mode"] = "after" if how == "after" else "now"
            self.state["message"] = "Stopping after this image." if how == "after" else "Stopping batch…"
            self.save()
        return {"ok": True}

    def _run(self, body, process):
        try:
            for item in self.state["items"]:
                if self.after.is_set():
                    break
                existing = find_deliverable(Path(item["ply"]).parent, item["name"], "character.ply", body["experiment"])
                final_dir = Path(item["ply"]).parent
                timing_file = final_dir.parent / 'batch_timing.json'
                item['project_dir'] = str(final_dir.parent)
                combined = OUTPUT_DIR / safe_run_name(body['experiment']) / 'final_ply'
                if existing and legacy_aspect_mismatch(Path(item["input"]), final_dir.parent):
                    existing = None
                if existing and not body.get("redo"):
                    existing = organize_ply(existing, final_dir, combined, body.get('combined_ply', True))
                    with self.lock:
                        if timing_file.is_file():
                            timing = json.loads(timing_file.read_text(encoding='utf-8'))
                            item.update({k: timing[k] for k in ('started_at','finished_at','duration_seconds') if k in timing})
                        item.update(status="skipped", ply=str(existing))
                        self.state["completed"] += 1
                        self.save()
                    continue
                with self.lock:
                    item["status"] = "running"
                    item['started_at'] = time.time()
                    self.state["message"] = f"Processing {item['name']} ({self.state['completed'] + 1}/{self.state['total']})"
                    self.save()
                try:
                    result = process(body | {"input": item["input"], "output_name": item["name"],
                                             "generation_policy": "first", "candidate_count": 1}, self.cancel)
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                with self.lock:
                    item["status"] = "done" if result.get("ok") else "stopped" if self.cancel.is_set() else "failed"
                    if not result.get("ok"):
                        item["error"] = result.get("error", "Image processing failed")
                        self.save()
                        if self.cancel.is_set():
                            break
                        continue
                    if not Path(item["ply"]).is_file():
                        item["status"] = "failed"
                        item["error"] = f"No final PLY was written for {item['name']}"
                        self.save()
                        continue
                    self.state["completed"] += 1
                    item['ply'] = str(organize_ply(Path(item['ply']), final_dir, combined, body.get('combined_ply', True)))
                    item['finished_at'] = time.time()
                    item['duration_seconds'] = max(0, item['finished_at'] - item['started_at'])
                    write_json(timing_file, {k:item[k] for k in ('started_at','finished_at','duration_seconds')})
                    if result.get("warning"):
                        item["warning"] = result["warning"]
                    self.save()
            else:
                failed = sum(i["status"] == "failed" for i in self.state["items"])
                self.state.update(status="failed" if failed else "done", message=f"Batch complete: {self.state['completed']} finished or already saved, {failed} failed. Start again to retry failures; completed splats are skipped.")
            if self.after.is_set():
                self.state.update(status="stopped", message="Batch stopped. Completed splats are kept.")
        except Exception as exc:
            if 'item' in locals():
                item.update(status="failed", error=str(exc))
            self.state.update(status="failed", message=str(exc))
        finally:
            with self.lock:
                self.state["active"] = False
                self.save()
