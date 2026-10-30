"""Compare a completed video reconstruction with its baseline, including PLY renders."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from i2g.evaluate import load_ply
from i2g.splat import render_rgb


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("repeat", type=Path)
    args = parser.parse_args()
    reports = [json.loads((root / "final/report.json").read_text()) for root in (args.baseline, args.repeat)]
    records = [json.loads((root / "records.json").read_text()) for root in (args.baseline, args.repeat)]
    params = [load_ply(root / "final/character.ply") for root in (args.baseline, args.repeat)]
    checkpoint = torch.load(args.repeat / "final/splats.pt", map_location="cuda", weights_only=True)
    errors = {}
    for key in ("means", "scales", "quats", "opacities", "sh0"):
        expected = checkpoint[key]
        if key == "quats": expected = torch.nn.functional.normalize(expected, dim=-1)
        errors[key] = float((params[1][key] - expected).abs().max())
    sheet = Image.new("RGB", (1600, 800), (128, 128, 128))
    faces = Image.new("RGB", (1600, 800), (128, 128, 128))
    for tile_index, index in enumerate(np.linspace(0, len(records[1])-1, 8).astype(int)):
        pair = []
        for model_index in range(2):
            rec = records[model_index][index]
            w, h = Image.open(rec["path"]).size
            rgb = render_rgb(params[model_index], torch.tensor(rec["w2c"], device="cuda", dtype=torch.float32),
                             torch.tensor(rec["K"], device="cuda", dtype=torch.float32), w, h)
            pair.append(Image.fromarray(rgb))
        x, y = tile_index % 4 * 400, tile_index // 4 * 400
        for j, im in enumerate(pair):
            im.thumbnail((200, 365)); sheet.paste(im, (x+j*200, y))
            # Upper quarter of the foreground bounding box gives a consistent head crop.
            rec = records[j][index]
            mask = np.asarray(Image.open(rec["mask"])) > 128
            yy, xx = np.where(mask)
            full = render_rgb(params[j], torch.tensor(rec["w2c"], device="cuda", dtype=torch.float32),
                              torch.tensor(rec["K"], device="cuda", dtype=torch.float32), mask.shape[1], mask.shape[0])
            crop = Image.fromarray(full).crop((int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.min()+(yy.max()-yy.min())*.25)))
            crop.thumbnail((200, 365)); faces.paste(crop, (x+j*200, y))
        for image in (sheet, faces):
            ImageDraw.Draw(image).text((x+8, y+375), f"Original             UI repeat | frame {index*2}", fill="white")
    final = args.repeat / "final"
    sheet.save(final / "baseline_vs_ui.jpg")
    faces.save(final / "baseline_vs_ui_heads.jpg")
    capture = cv2.VideoCapture(str(final / "turntable.mp4"))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); capture.release()
    result = {
        "baseline_heldout_psnr": reports[0]["heldout_mean_psnr"],
        "repeat_heldout_psnr": reports[1]["heldout_mean_psnr"],
        "delta_db": reports[1]["heldout_mean_psnr"]-reports[0]["heldout_mean_psnr"],
        "gaussians": len(params[1]["means"]),
        "all_parameters_finite": all(bool(torch.isfinite(v).all()) for v in params[1].values()),
        "normalized_checkpoint_max_errors": errors,
        "video_frames": frames,
        "ply_bytes": (final / "character.ply").stat().st_size,
        "same_training_config": reports[0]["config"] == reports[1]["config"],
        "same_heldout_names": [v["name"] for v in reports[0]["views"] if v["held_out"]] == [v["name"] for v in reports[1]["views"] if v["held_out"]],
    }
    (final / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["all_parameters_finite"] or max(errors.values()) > 1e-6 or frames != 144:
        raise SystemExit("Export verification failed")


if __name__ == "__main__":
    main()
