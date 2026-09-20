"""Generated orbit frames -> a calibrated training set for the v2 trainer.

Every frame's camera is the orbit camera it was generated for (known by construction). Frames are
matted with BiRefNet; the photo (native resolution, exact camera) stays the first view. Every
`holdout_every`-th unique frame is held out for leave-frames-out evaluation.

QA per frame (nothing is filtered in this first version; the numbers show whether filtering is needed):
  * iou_control: generated matte vs the control proxy's silhouette, at the frame's camera. Low means
    the model did not follow the control (or the proxy lacks hair/clothes, for the mesh proxy).
  * iou_splat:   generated matte vs the v2 splat's silhouette (hair and clothes included). Below 0.9 the
    clip has re-styled the subject and should not be trained on; `summary["restyled"]` flags it.
  * face_id:     ArcFace similarity to the photo where a face is detected (identity drift).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from i2g.log import get_logger

log = get_logger("pacgs.dataset")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def photo_record(subject_run: Path) -> dict:
    records = json.loads((subject_run / "03_qa" / "registration.json").read_text())
    photo = records[0]
    if photo["tag"] != "photo":
        raise ValueError("registration.json must list the photo first")
    return photo | {"selected": True, "accepted": True}


def head_crop(world: dict, cam: dict, width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    """A box around the head, `scale` x the head's own radius, in the frame's pixels: (x0, y0, x1, y1).

    The radius is the largest distance from the head centre to a facial keypoint (about 0.10 m), so
    scale 1.8 covers the skull, the hair and a little of the shoulders."""
    from i2g.cameras import project
    from i2g.evaluate import _head
    from i2g.body import L_EAR, L_EYE, NOSE, R_EAR, R_EYE

    head = _head(world)
    r = float(np.linalg.norm(world["kp3d"][[NOSE, L_EYE, R_EYE, L_EAR, R_EAR]] - head, axis=1).max())
    uv, z = project(head[None], cam["w2c"], cam["K"])
    half = float(cam["K"][1, 1]) * scale * r / max(float(z[0]), 1e-3)
    cx, cy = float(uv[0, 0]), float(uv[0, 1])
    x0, y0 = max(0, int(round(cx - half))), max(0, int(round(cy - half)))
    x1, y1 = min(width, int(round(cx + half))), min(height, int(round(cy + half)))
    if x1 - x0 < 64 or y1 - y0 < 64:            # the head is off the frame or tiny: keep the whole frame
        return 0, 0, width, height
    return x0, y0, x1, y1


def prepare(frames_root: Path, cams: list[dict], out_dir: Path, subject_run: Path, world: dict,
            control_proxy, splat_proxy, holdout_every: int = 8, tag: str = "a",
            crop: float | None = None) -> dict:
    """Matte every unique frame, score it, write records.json (photo first) and qa.json.
    `tag` prefixes the view names, so datasets from several rigs can be trained together.

    `crop` keeps only a box of that many head radii around the head (see `head_crop`) in the *training*
    frames. The head close-up clips also carry the torso, where each clip invents the parts the photo
    never saw - the back of a top, say - and invents them differently; cropping keeps each clip to the
    job it was filmed for and leaves the torso to the body orbit. QA still scores the whole frame."""
    from i2g.evaluate import FaceID
    from i2g.matting import Matter

    from .orbit import is_duplicate

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mattes").mkdir(exist_ok=True)
    # from the frames themselves: zoomed rigs move the principal point, so K does not give the size
    with Image.open(frames_root / f"arc{cams[0]['arc']}" / f"frame_{cams[0]['index']:04d}.png") as first:
        width, height = first.size
    matter = Matter()
    face = FaceID()
    src_hr = np.asarray(Image.open(subject_run / "00_ingest" / "source_hr_rgba.png").convert("RGB"))
    ref = face.detect(src_hr)
    e_ref = face.embed(src_hr, ref[1]) if ref is not None else None

    records, qa, unique = [photo_record(subject_run)], [], 0
    for cam in cams:
        path = frames_root / f"arc{cam['arc']}" / f"frame_{cam['index']:04d}.png"
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if rgb.shape[:2] != (height, width):
            raise ValueError(f"{path} is {rgb.shape[1]}x{rgb.shape[0]}, expected {width}x{height}")
        name = f"{tag}{cam['arc']}f{cam['index']:03d}"
        dup = is_duplicate(cam)
        row = {"name": name, "arc": cam["arc"], "index": cam["index"], "orbit_deg": round(cam["orbit_deg"], 3),
               "azimuth_deg": round(cam["azimuth_deg"], 2), "duplicate": dup}
        alpha = matter(rgb)
        m = alpha > 0.5
        _, a_ctrl = control_proxy.depth(cam["w2c"], cam["K"], width, height)
        _, a_splat = splat_proxy.depth(cam["w2c"], cam["K"], width, height)
        row["iou_control"] = round(_iou(m, a_ctrl > 0.5), 4)
        row["iou_splat"] = round(_iou(m, a_splat > 0.5), 4)
        det = face.detect(rgb, thresh=0.5)
        row["face_id"] = round(float(face.embed(rgb, det[1]) @ e_ref), 4) if (det and e_ref is not None) else None
        qa.append(row)
        if dup:
            continue
        mask_path = out_dir / "mattes" / f"{name}.png"
        frame_path, K = path, cam["K"]
        if crop:
            x0, y0, x1, y1 = head_crop(world, cam, width, height, crop)
            if (x1 - x0, y1 - y0) != (width, height):
                (out_dir / "frames").mkdir(exist_ok=True)
                frame_path = out_dir / "frames" / f"{name}.png"
                Image.fromarray(rgb[y0:y1, x0:x1]).save(frame_path)
                alpha = alpha[y0:y1, x0:x1]
                K = cam["K"].copy()
                K[0, 2] -= x0
                K[1, 2] -= y0
        Image.fromarray((alpha * 255).round().astype(np.uint8)).save(mask_path)
        unique += 1
        records.append({"tag": f"{tag}{cam['arc']}f", "seed": cam["index"], "kind": "video", "path": str(frame_path),
                        "mask": str(mask_path), "w2c": cam["w2c"].tolist(), "K": K.tolist(),
                        "azimuth_deg": cam["azimuth_deg"], "elevation_deg": cam["elevation_deg"],
                        "radius_m": cam["radius_m"], "orbit_deg": cam["orbit_deg"],
                        "selected": True, "accepted": True})
    matter.close()

    video = records[1:]
    holdout = [f"{r['tag']}_s{r['seed']}" for j, r in enumerate(video) if j % holdout_every == holdout_every // 2]
    summary = {"frames": len(cams), "unique": unique, "holdout": holdout, "crop": crop,
               "iou_control_median": round(float(np.median([q["iou_control"] for q in qa])), 4),
               "iou_splat_median": round(float(np.median([q["iou_splat"] for q in qa])), 4),
               "face_id_median": _median([q["face_id"] for q in qa]),
               "loop": _loop_closure(frames_root, cams)}
    # A clip whose matte disagrees with the subject's own silhouette has re-styled them. It happens when
    # the control proxy lacks what the photo has: the body mesh has no hair and no clothes, so from above
    # the model invented a bun and a strapless top (iou_splat 0.82), and those frames then fought every
    # other clip in training. The splat proxy carries both (0.97).
    summary["restyled"] = summary["iou_splat_median"] < 0.9
    if summary["restyled"]:
        log.warning("%s: matte-vs-subject IoU %.3f - the generated frames do not match the person's "
                    "silhouette; their hair or clothing has been re-styled. Generate this rig from the "
                    "splat proxy, not the mesh.", out_dir.name, summary["iou_splat_median"])
    (out_dir / "records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out_dir / "qa.json").write_text(json.dumps({"summary": summary, "frames": qa}, indent=1), encoding="utf-8")
    contact_sheet(frames_root, cams, qa, control_proxy, out_dir / "contact_sheet.jpg", size=(width, height))
    log.info("Prepared %d unique frames (%d held out); matte-vs-control IoU %.3f, matte-vs-splat IoU %.3f, "
             "face id %s", unique, len(holdout), summary["iou_control_median"], summary["iou_splat_median"],
             summary["face_id_median"])
    return summary


def merge(datasets: list[Path], out_dir: Path, boost: dict[str, float] | None = None) -> dict:
    """One training set from several prepared clips: the photo once, then every clip's frames. The clips
    must have been prepared with different tags, or their view names would collide.

    `boost` gives a clip more of the training steps than its frame count would: {"f1_face_l4": 3.0} makes
    each head close-up frame three times as likely to be sampled as a body-orbit frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    boost = boost or {}
    records, holdout, parts = [], [], []
    for d in datasets:
        rows = json.loads((d / "records.json").read_text())
        summary = json.loads((d / "qa.json").read_text())["summary"]
        if not records:
            records.append(rows[0])                       # the photo
        share = boost.get(d.name, 1.0)
        records += [r | {"share": share} for r in rows[1:]] if share != 1.0 else rows[1:]
        holdout += summary["holdout"]
        parts.append({"dataset": str(d), "frames": len(rows) - 1, "holdout": len(summary["holdout"]),
                      "share": share, "iou_control_median": summary["iou_control_median"]})
    names = [f"{r['tag']}_s{r['seed']}" for r in records]
    if len(set(names)) != len(names):
        raise ValueError("view names collide between datasets: prepare each clip with its own --tag")
    summary = {"frames": len(records) - 1, "unique": len(records) - 1, "holdout": holdout, "parts": parts}
    (out_dir / "records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out_dir / "qa.json").write_text(json.dumps({"summary": summary}, indent=1), encoding="utf-8")
    log.info("Merged %d clips: %d frames, %d held out", len(datasets), summary["frames"], len(holdout))
    return summary


def _median(values) -> float | None:
    v = [x for x in values if x is not None]
    return round(float(np.median(v)), 4) if v else None


def _loop_closure(frames_root: Path, cams: list[dict]) -> dict | None:
    """The full-circle frame against the orbit start, when the two share a camera. A rising rig ends
    above where it started, so there is no loop to close and this returns None."""
    ends = [c for c in cams if c["index"] > 0 and abs(c["orbit_deg"] - 360.0) < 1e-6
            and np.allclose(c["w2c"], cams[0]["w2c"], atol=1e-6)]
    if not ends:
        return None
    first = np.asarray(Image.open(frames_root / "arc0" / "frame_0000.png").convert("RGB")).astype(np.float64)
    last_cam = ends[-1]
    last = np.asarray(Image.open(frames_root / f"arc{last_cam['arc']}" / f"frame_{last_cam['index']:04d}.png")
                      .convert("RGB")).astype(np.float64)
    mse = ((first - last) ** 2).mean() / 255.0 ** 2
    return {"psnr_first_vs_last": round(10 * np.log10(1 / max(mse, 1e-12)), 2)}


def contact_sheet(frames_root: Path, cams: list[dict], qa: list[dict], control_proxy, out: Path,
                  size: tuple[int, int], every: int = 5, tile_w: int = 240, cols: int = 8) -> None:
    """Every `every`-th frame with the control proxy's silhouette outline (green) and its QA numbers."""
    picks = list(range(0, len(cams), every))
    width, height = size
    tile_h = int(round(tile_w * height / width))
    rows = (len(picks) + cols - 1) // cols
    sheet = np.full((rows * (tile_h + 34), cols * tile_w, 3), 30, np.uint8)
    for n, j in enumerate(picks):
        cam, q = cams[j], qa[j]
        img = cv2.imread(str(frames_root / f"arc{cam['arc']}" / f"frame_{cam['index']:04d}.png"))
        _, a = control_proxy.depth(cam["w2c"], cam["K"], width, height)
        contours, _ = cv2.findContours((a > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (80, 255, 80), 1, cv2.LINE_AA)
        tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(n, cols)
        y0, x0 = r * (tile_h + 34), c * tile_w
        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = tile
        fid = f" id {q['face_id']:.2f}" if q["face_id"] is not None else ""
        cv2.putText(sheet, f"{q['name']} {q['orbit_deg']:.0f}deg", (x0 + 4, y0 + tile_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"iou {q['iou_control']:.2f}/{q['iou_splat']:.2f}{fid}", (x0 + 4, y0 + tile_h + 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
