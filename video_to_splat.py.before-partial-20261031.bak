"""Approved orbit MP4 -> foreground SfM -> static Gaussian PLY. No image generation.

Run with .venv-i2g/Scripts/python.exe; requires pycolmap in addition to the v2 environment.
Approval is tied to the input SHA256. Stages are cached in a dedicated output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def progress(stage, fraction=0., detail=""):
    print("[video-event] " + json.dumps({"stage": stage, "fraction": fraction, "detail": detail}), flush=True)


class TrainingProgress:
    def __init__(self, total):
        self.total, self.step = total, 0

    def update(self, n=1):
        self.step += n

    def set_postfix_str(self, s="", refresh=True):
        progress("train", self.step/self.total, f"{self.step:,}/{self.total:,} steps; {s}")

    def close(self):
        pass


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def prepare(args, root):
    from i2g.matting import Matter
    manifest = root / "input.json"
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    spec = {"path": str(args.input.resolve()), "sha256": digest,
            "stride": args.stride, "max_height": args.max_height}
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(old[k] != v for k, v in spec.items()):
            raise ValueError("Input/settings changed: use a new output directory")
    if args.approved_video:
        write_json(root / "approval.json", {"sha256": digest, "approved": True,
                   "basis": "User supplied this MP4 as an approved orbit"})
    approval = root / "approval.json"
    approved = json.loads(approval.read_text()) if approval.exists() else {}
    if approved.get("sha256") != digest or approved.get("approved") is not True:
        raise ValueError("Review the video first, then pass --approved-video")
    if (root / "prepared.json").exists():
        return
    for folder in ("frames", "masks", "sfm_masks"):
        (root / folder).mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if count < 16 or fps <= 0:
        raise ValueError("Video could not be decoded or is too short")
    write_json(manifest, spec | {"frame_count": count, "fps": fps})
    matter = Matter()
    rows, tiles = [], []
    try:
        # Omit the repeated loop endpoint.
        for index in range(count - 1):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Decode failed at frame {index}")
            if index % args.stride:
                continue
            h, w = bgr.shape[:2]
            scale = min(1., args.max_height / h)
            rgb = cv2.cvtColor(cv2.resize(bgr, (round(w * scale), round(h * scale))), cv2.COLOR_BGR2RGB)
            name = f"frame_{index:04d}.png"
            mask_path = root / "masks" / name
            if mask_path.exists():
                alpha = np.asarray(Image.open(mask_path)) / 255.
            else:
                alpha = matter(rgb)
                Image.fromarray((alpha * 255).round().astype(np.uint8)).save(mask_path)
            Image.fromarray(rgb).save(root / "frames" / name)
            # Exclude generated background markers from camera estimation.
            mask = cv2.erode((alpha > .8).astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
            Image.fromarray(mask).save(root / "sfm_masks" / (name + ".png"))
            rows.append({"name": name, "index": index, "time": index / fps})
            tile = (rgb * alpha[..., None] + 128 * (1-alpha[..., None])).astype(np.uint8)
            tiles.append(Image.fromarray(tile))
            print(f"Matted frame {index}/{count-1}", flush=True)
            progress("prepare", (index+1)/(count-1), f"Prepared frame {index} of {count-1}")
    finally:
        cap.release()
        matter.close()
    selected = np.linspace(0, len(tiles)-1, min(16, len(tiles))).astype(int)
    sheet = Image.new("RGB", (800, 360 * ((len(selected)+3)//4)), (128,128,128))
    draw = ImageDraw.Draw(sheet)
    for j, k in enumerate(selected):
        tiles[k].thumbnail((200, 335))
        sheet.paste(tiles[k], (j%4*200, j//4*360))
        draw.text((j%4*200+5, j//4*360+338), rows[k]["name"])
    sheet.save(root / "contact_sheet.jpg")
    write_json(root / "prepared.json", rows)


def calibrate(root):
    import pycolmap as p
    if (root / "calibration.json").exists():
        progress("calibrate", 1., "Using validated camera reconstruction")
        return
    db = root / "features.db"
    reader = p.ImageReaderOptions(camera_model="SIMPLE_PINHOLE", mask_path=str(root / "sfm_masks"))
    extraction = p.FeatureExtractionOptions(num_threads=12, max_image_size=1600)
    extraction.sift.max_num_features = 12000
    extraction.sift.peak_threshold = .003
    p.extract_features(db, root / "frames", camera_mode=p.CameraMode.SINGLE,
                       reader_options=reader, extraction_options=extraction, device=p.Device.cpu)
    progress("calibrate", .3, "Foreground features extracted; matching frames")
    matching = p.FeatureMatchingOptions(num_threads=12)
    p.match_exhaustive(db, matching_options=matching, device=p.Device.cpu)
    progress("calibrate", .6, "Frames matched; recovering cameras and sparse geometry")
    opts = p.IncrementalPipelineOptions(num_threads=12, min_model_size=12)
    opts.mapper.init_min_num_inliers = 60
    opts.mapper.init_min_tri_angle = 8.
    opts.mapper.abs_pose_min_num_inliers = 20
    models = p.incremental_mapping(db, root / "frames", root / "sparse", options=opts)
    if not models:
        raise RuntimeError("No connected reconstruction; cameras need another calibration method")
    best_id, model = max(models.items(), key=lambda kv: kv[1].num_reg_images())
    total = len(json.loads((root / "prepared.json").read_text()))
    summary = {"model": str(root / "sparse" / str(best_id)), "registered": model.num_reg_images(),
               "total": total, "points": model.num_points3D(), "reprojection_error": model.compute_mean_reprojection_error()}
    images = sorted(model.images.values(), key=lambda im: im.name)
    center = np.median([point.xyz for point in model.points3D.values()], axis=0)
    up = np.mean([-im.cam_from_world().rotation.matrix()[1] for im in images], axis=0)
    up /= np.linalg.norm(up)
    front = images[0].projection_center()-center
    front -= up*front.dot(up)
    front /= np.linalg.norm(front)
    basis = np.column_stack([np.cross(up, front), up, front])
    centers = np.array([im.projection_center()-center for im in images]) @ basis
    angles = np.degrees(np.unwrap(np.arctan2(centers[:,0], centers[:,2])))
    circular = np.sort(angles % 360)
    summary["orbit_span_degrees"] = float(np.ptp(angles))
    summary["largest_gap_degrees"] = float(np.diff(np.r_[circular, circular[0]+360]).max())
    summary["frame_azimuth_degrees"] = dict(zip([im.name for im in images], angles.tolist()))
    write_json(root / "calibration_attempt.json", summary)
    if model.num_reg_images() < .9 * total:
        raise RuntimeError(f"Only {model.num_reg_images()}/{total} frames registered; refusing an incomplete orbit")
    if summary["orbit_span_degrees"] < 300 or summary["largest_gap_degrees"] > 45:
        raise RuntimeError("Registered cameras do not cover a full orbit; inspect calibration_attempt.json")
    write_json(root / "calibration.json", summary)
    progress("calibrate", 1., f"Validated {model.num_reg_images()}/{total} cameras and full orbit coverage")


def fit(args, root):
    import pycolmap as p
    import torch
    from i2g.splat import SplatConfig, load_views, train, clean, export_ply, render_rgb, psnr, config_dict
    from i2g.cameras import orbit_camera
    torch.cuda.reset_peak_memory_stats()
    summary = json.loads((root / "calibration.json").read_text())
    model = p.Reconstruction(summary["model"])
    imgs = sorted(model.images.values(), key=lambda im: im.name)
    xyz = np.array([pt.xyz for pt in model.points3D.values()])
    colors = np.array([pt.color for pt in model.points3D.values()]) / 255.
    # Orient using the camera up vectors; arbitrary monocular scale becomes nominal 1.7 m height.
    up = np.mean([-im.cam_from_world().rotation.matrix()[1] for im in imgs], axis=0)
    up /= np.linalg.norm(up)
    front = imgs[0].projection_center() - np.median(xyz, axis=0)
    front -= up * front.dot(up)
    front /= np.linalg.norm(front)
    basis = np.column_stack([np.cross(up, front), up, front])
    oriented = xyz @ basis
    lo, hi = np.quantile(oriented, [.01, .99], axis=0)
    center = (lo + hi) / 2
    center[1] = lo[1]
    scale = 1.7 / (hi[1]-lo[1])
    xyz = (oriented-center)*scale
    # A few erroneous triangulations can lie metres away; they must not enlarge the hull grid.
    bounds = (np.stack([lo, hi])-center)*scale
    records = []
    for i, im in enumerate(imgs):
        old = im.cam_from_world().matrix()
        w2c = np.eye(4)
        w2c[:3,:3] = old[:3,:3] @ basis
        w2c[:3,3] = (old[:3,3] + old[:3,:3] @ basis @ center)*scale
        K = model.cameras[im.camera_id].calibration_matrix()
        records.append({"tag": "photo" if i == 0 else "video", "seed": i, "kind": "photo" if i == 0 else "video",
                        "path": str(root/"frames"/im.name), "mask": str(root/"masks"/im.name),
                        "K": K.tolist(), "w2c": w2c.tolist(), "selected": True, "accepted": True})
    write_json(root / "records.json", records)
    target = np.array([0., .85, 0.])
    world = {"vertices": bounds, "target": target}
    cfg = SplatConfig(iterations=args.iterations, refine_stop=int(args.iterations*.8),
        face_weight=1., face_bake=False, face_align=False, face_depth_weight=0.,
        face_mesh_init=False, face_shell=False, interior_face_margin=0.,
        photo_sample_share=1/len(records), video_weight=1., pose_lr=0., color_lr=0.,
        perceptual_weight=.1, ms_min_height=480, half_res_until=1000,
        hull_tolerance=max(2, len(records)//12))
    holdout = [f"video_s{i}" for i in range(5,len(records),10)]
    cfg.exclude = ",".join(holdout)
    views = load_views(records, np.array(records[0]["K"]), cfg, "cuda")
    started = time.perf_counter()
    # Carve a dense surface from the foreground silhouettes, using recovered cameras.
    params, stats = train(views, world, cfg, bar=TrainingProgress(cfg.iterations))
    progress("export", 0., "Cleaning and exporting the trained splat")
    refined = torch.tensor(stats.pop("refined_w2c"), device="cuda")
    cleaning = clean(params, views, refined, cfg)
    final = root / "final"
    final.mkdir(exist_ok=True)
    export_ply(params, final / "character.ply")
    torch.save({k:v.detach().cpu() for k,v in params.items()}, final / "splats.pt")
    progress("export", .15, "PLY saved; evaluating reconstructed views")
    measurements, tiles = [], []
    with torch.no_grad():
        for i, rec in enumerate(records):
            rgb = np.asarray(Image.open(rec["path"]))
            mask = np.asarray(Image.open(rec["mask"])) / 255.
            pred = render_rgb(params, torch.tensor(rec["w2c"],device="cuda",dtype=torch.float32),
                              torch.tensor(rec["K"],device="cuda",dtype=torch.float32),rgb.shape[1],rgb.shape[0])
            measurements.append({"name": imgs[i].name, "held_out": f"video_s{i}" in holdout,
                                 "foreground_psnr": psnr(pred, rgb, mask > .5)})
            progress("export", .15 + .35*(i+1)/len(records), f"Evaluated view {i+1}/{len(records)}")
            if i in np.linspace(0,len(records)-1,8).astype(int):
                ref = (rgb*mask[...,None]+128*(1-mask[...,None])).astype(np.uint8)
                tiles.append(Image.fromarray(np.concatenate([ref,pred],axis=1)))
        sheet = Image.new("RGB", (1600, 760), (128,128,128))
        labels = ImageDraw.Draw(sheet)
        for i,tile in enumerate(tiles):
            tile.thumbnail((400,350)); sheet.paste(tile,(i%4*400,i//4*380))
            labels.text((i%4*400+8,i//4*380+355), "Video frame                 Gaussian render", fill="white")
        sheet.save(final / "comparison.jpg")
        w,h = Image.open(records[0]["path"]).size
        ratio = min(1., 960/h); w,h = round(w*ratio),round(h*ratio)
        K = np.array(records[0]["K"]); K[:2] *= ratio
        distances = [np.linalg.norm(-np.array(r["w2c"])[:3,:3].T @ np.array(r["w2c"])[:3,3]-target) for r in records]
        writer = cv2.VideoWriter(str(final/"turntable_raw.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),24,(w,h))
        if not writer.isOpened():
            raise RuntimeError("Could not open preview encoder")
        for preview_index, az in enumerate(np.linspace(0,360,144,endpoint=False)):
            mat = orbit_camera(target,float(np.median(distances)),float(az),0.)
            rgb = render_rgb(params,torch.tensor(mat,device="cuda",dtype=torch.float32),torch.tensor(K,device="cuda",dtype=torch.float32),w,h)
            writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if (preview_index+1) % 12 == 0:
                progress("export", .5 + .4*(preview_index+1)/144, f"Rendered preview frame {preview_index+1}/144")
        writer.release()
    import subprocess
    import imageio_ffmpeg
    progress("export", .95, "Encoding browser preview and writing quality report")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(final/"turntable_raw.mp4"), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final/"turntable.mp4")], check=True)
    report = {"input": json.loads((root/"input.json").read_text()), "calibration": summary,
              "config": config_dict(cfg), "training": stats, "cleanup": cleaning,
              "views": measurements, "heldout_mean_psnr": float(np.mean([r["foreground_psnr"] for r in measurements if r["held_out"]])),
              "elapsed_seconds": time.perf_counter()-started,
              "peak_allocated_vram_gib": torch.cuda.max_memory_allocated()/1024**3,
              "scale_note": "Monocular scale is arbitrary; normalized to nominal 1.7 m sparse foreground height.",
              "image_generation": False}
    write_json(final/"report.json",report)
    progress("export", 1., "PLY and previews written")
    print(f"Finished: {final/'character.ply'}",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--approved-video",action="store_true")
    parser.add_argument("--stride",type=int,default=2)
    parser.add_argument("--max-height",type=int,default=1280)
    parser.add_argument("--iterations",type=int,default=15000)
    parser.add_argument("--stage",choices=["prepare","calibrate","train","all"],default="all")
    args = parser.parse_args()
    import importlib.util
    if importlib.util.find_spec("pycolmap") is None:
        parser.error("Video reconstruction requires pycolmap. Install it with .venv-i2g/Scripts/python.exe -m pip install pycolmap==4.2.1")
    if args.stride < 1 or args.max_height < 256 or args.iterations < 100:
        parser.error("Invalid stride, height or iteration count")
    root = args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    from i2g.log import setup_logging, get_logger
    from i2g.gpu_lock import acquire
    from i2g.comfy_client import ComfyClient
    setup_logging(root/"run.log")
    lock = acquire("video-to-splat",get_logger("video"))
    try:
        client = ComfyClient()
        if client.is_up():
            queue = client._get_json("/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                raise RuntimeError("ComfyUI is busy; retry after its jobs finish")
            client.free()
        progress("prepare", 0., "Validating approval and preparing frames")
        prepare(args,root)
        if args.stage in ("all","calibrate","train"):
            progress("calibrate", 0., "Recovering cameras from foreground features")
            calibrate(root)
        if args.stage in ("all","train"):
            progress("train", 0., "Building the visual hull and starting training")
            fit(args,root)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
