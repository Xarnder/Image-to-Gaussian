"""Serial, unattended image -> orbit -> splat batches."""
import copy
import hashlib
import json
import threading
import time
import uuid
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
            self.state = {"active": True, "batch_id": uuid.uuid4().hex, "input": str(folder), "experiment": body["experiment"],
                          "status": "running", "items": items, "completed": 0, "total": len(items),
                          "message": "Starting automated image batch…", "settings": copy.deepcopy(body)}
            self.save()
            threading.Thread(target=self._run, args=(copy.deepcopy(body), process), daemon=True).start()
            return {"ok": True}

    def resume(self, batch_id, process):
        with self.lock:
            if (self.state.get("active") or self.state.get("status") != "stopped"
                    or self.state.get("batch_id") != batch_id):
                raise ValueError("This saved batch is no longer available to resume.")
            body = copy.deepcopy(self.state["settings"])
            experiment = safe_run_name(self.state["experiment"])
            finished = 0
            for item in self.state["items"]:
                final = OUTPUT_DIR / experiment / item["name"] / "video" / "final"
                if item["status"] in ("done", "skipped"):
                    ply = find_deliverable(final, item["name"], "character.ply", experiment)
                    if ply:
                        item["ply"] = str(ply)
                        finished += 1
                        continue
                item.update(status="queued", ply=str(deliverable_path(final, item["name"], "character.ply", experiment)))
                for key in ("error", "warning", "started_at", "finished_at", "duration_seconds"):
                    item.pop(key, None)
            self.cancel.clear()
            self.after.clear()
            self.state.pop("stop_mode", None)
            self.state.update(active=True, status="running", completed=finished, message="Resuming the saved queue.")
            self.save()
            threading.Thread(target=self._run, args=(body, process), daemon=True).start()
            return {"ok": True}

    def append(self, batch_id, photos):
        with self.lock:
            if (not self.state['active'] or self.state['status'] != 'running'
                    or self.after.is_set() or batch_id != self.state.get('batch_id')):
                raise ValueError("This batch is no longer accepting images.")
            items = self.state['items']
            known = {Path(item['input']).resolve() for item in items}
            names = {item['name'].casefold() for item in items}
            experiment = safe_run_name(self.state['experiment'])
            added = []
            for source in photos:
                source = source.resolve()
                if source in known:
                    continue
                if not source.is_file():
                    raise ValueError(f"Image no longer exists: {source}")
                name = safe_run_name(source.stem)
                if name.casefold() in names:
                    name += '_' + hashlib.sha256(str(source).casefold().encode()).hexdigest()[:10]
                if name.casefold() in names:
                    raise ValueError(f"Project name already in use: {name}")
                final = deliverable_path(OUTPUT_DIR / experiment / name / 'video' / 'final',
                                         name, 'character.ply', experiment)
                added.append({'input': str(source), 'name': name, 'ply': str(final), 'status': 'queued'})
                known.add(source)
                names.add(name.casefold())
            items.extend(added)
            self.state['total'] = len(items)
            self.save()
            return {'ok': True, 'added': len(added), 'duplicates': len(photos) - len(added)}

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
            index = 0
            while not self.after.is_set():
                with self.lock:
                    if index == len(self.state['items']):
                        failed = sum(i['status'] == 'failed' for i in self.state['items'])
                        self.state.update(status='failed' if failed else 'done',
                                          message=f"Batch complete: {self.state['completed']} finished or already saved, {failed} failed.")
                        break
                    item = self.state['items'][index]
                    index += 1
                    if item["status"] in ("done", "skipped"):
                        continue
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
            if self.after.is_set():
                self.state.update(status="stopped", message="Batch stopped. Press Resume batch to continue the saved queue. Finished splats will be skipped.")
        except Exception as exc:
            if 'item' in locals():
                item.update(status="failed", error=str(exc))
            self.state.update(status="failed", message=str(exc))
        finally:
            with self.lock:
                self.state["active"] = False
                self.save()
