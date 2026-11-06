"""Persistent local catalogue of exported splats and small source-image previews."""
import hashlib
import json
import os
import threading
from pathlib import Path
from PIL import Image, ImageOps
from .paths import OUTPUT_DIR

CACHE = OUTPUT_DIR / ".splat_history"
LOCK = threading.Lock()


def metadata(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def training_seconds(work):
    final = work / "final"
    for path in [*final.glob("*-report.json"), final / "report.json"]:
        report = metadata(path)
        seconds = report.get("training", report).get("train_seconds")
        if isinstance(seconds, (int, float)) and seconds >= 0:
            head = report.get("head_pass") or {}
            return seconds + 60 * head.get("train_minutes", 0)
    return None


def catalogue():
    with LOCK:
        CACHE.mkdir(parents=True, exist_ok=True)
        index = CACHE / "index.json"
        entries = metadata(index)
        found = {}
        combined = []
        for base, dirs, files in os.walk(OUTPUT_DIR):
            folder = Path(base)
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       {"frames", "orbit_candidates", "masks", "colmap", "images", "sparse", "dense"}
                       and not (len(d) > 2 and d[:2].isdigit() and d[2] == "_")]
            if folder.name == "final_ply":
                combined.extend(folder / f for f in files if f.lower().endswith(".ply"))
                dirs.clear()
                continue
            final = next((p for p in (folder, *folder.parents) if p.name == "final"), None)
            if final is None:
                continue
            for f in files:
                if f.lower().endswith("character.ply"):
                    found[folder / f] = final.parent
            if "ply_location.json" in files:
                target = metadata(folder / "ply_location.json").get("path")
                if target and Path(target).is_file():
                    found[Path(target)] = final.parent
        for ply in combined:
            found.setdefault(ply, ply.parent)
        for ply, work in found.items():
            try:
                stat = ply.stat()
                if not ply.resolve().is_relative_to(OUTPUT_DIR.resolve()):
                    continue
                # Project identity + export timestamp survive a move into final_ply.
                identity = f"{work.relative_to(OUTPUT_DIR)}|{stat.st_mtime_ns}|{stat.st_size}"
                key = hashlib.sha256(identity.encode()).hexdigest()[:24]
                old = entries.get(key, {})
                seconds = old.get("training_seconds")
                if seconds is None:
                    seconds = training_seconds(work)
                old["training_seconds"] = seconds
                if old.get("thumbnail") and (CACHE / old["thumbnail"]).is_file():
                    old["ply"] = str(ply)
                    continue
                naming = metadata(work / "output_naming.json")
                ingest = next(work.glob("*_ingest/ingest.json"), None)
                source = naming.get("source_image") or (metadata(ingest).get("source") if ingest else None)
                name = Path(source).name if source else naming.get("name") or (work.parent.name if work.name in ("video","v2") else ply.stem)
                entry = {"name": name, "experiment": naming.get("experiment") or work.relative_to(OUTPUT_DIR).parts[0],
                         "ply": str(ply), "created": stat.st_mtime, "training_seconds": seconds, "thumbnail": None}
                candidates = [Path(source)] if source else []
                candidates.extend([work / "frames" / "frame_0000.png", *work.glob("*_ingest/source_hr_rgba.png")])
                for candidate in candidates:
                    if not candidate.is_file():
                        continue
                    try:
                        with Image.open(candidate) as img:
                            thumb = ImageOps.exif_transpose(img).convert("RGB")
                            thumb.thumbnail((220, 220))
                            thumb.save(CACHE / (key + ".jpg"), quality=85)
                        entry["thumbnail"] = key + ".jpg"
                        break
                    except (OSError, ValueError):
                        continue
                entries[key] = entry
            except OSError:
                continue
        temporary = index.with_suffix(".tmp")
        temporary.write_text(json.dumps(entries), encoding="utf-8")
        temporary.replace(index)
        return sorted((dict(entry, id=key, available=Path(entry["ply"]).is_file())
                       for key, entry in entries.items()), key=lambda e:e["created"], reverse=True)
