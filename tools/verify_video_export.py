"""Reload a video-derived PLY, compare its checkpoint, and render eight observed views."""
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
from i2g.paths import find_deliverable, deliverable_path


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    final = args.root / "final"
    naming_file = args.root / "output_naming.json"
    naming = json.loads(naming_file.read_text()) if naming_file.exists() else {"name":args.root.parent.name,"experiment":args.root.parent.parent.name}
    def source(kind):
        return find_deliverable(final, naming["name"], kind, naming.get("experiment", ""))
    def dest(kind):
        return deliverable_path(final, naming["name"], kind, naming.get("experiment", ""))
    records = json.loads((args.root / "records.json").read_text())
    params = load_ply(source("character.ply"))
    checkpoint = torch.load(source("splats.pt"), map_location="cuda", weights_only=True)
    errors = {}
    for key in ("means", "scales", "quats", "opacities", "sh0"):
        expected = checkpoint[key]
        if key == "quats":
            expected = torch.nn.functional.normalize(expected, dim=-1)
        errors[key] = float((params[key] - expected).abs().max())
    sheet = Image.new("RGB", (2048, 576), (70, 70, 70))
    for tile, index in enumerate(np.linspace(0, len(records)-1, 8).astype(int)):
        rec = records[index]
        source = Image.open(rec["path"]).convert("RGB")
        w, h = source.size
        rendered = Image.fromarray(render_rgb(params, torch.tensor(rec["w2c"], device="cuda", dtype=torch.float32),
                                             torch.tensor(rec["K"], device="cuda", dtype=torch.float32), w, h))
        x, y = tile % 4 * 512, tile // 4 * 288
        for offset, im in enumerate((source, rendered)):
            im.thumbnail((256, 256))
            sheet.paste(im, (x+offset*256, y))
        ImageDraw.Draw(sheet).text((x+8, y+262), f"Video / reloaded PLY | {rec.get('name', index)}", fill="white")
    sheet.save(dest("verified_comparison.jpg"), quality=95)
    cap = cv2.VideoCapture(str(source("turntable.mp4")))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    result = {"gaussians": len(params["means"]),
              "all_parameters_finite": all(bool(torch.isfinite(v).all()) for v in params.values()),
              "normalized_checkpoint_max_errors": errors, "preview_frames": frames,
              "ply_bytes": (source("character.ply")).stat().st_size}
    (dest("verification.json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["all_parameters_finite"] or max(errors.values()) > 1e-6 or frames != 144:
        raise SystemExit("Export verification failed")


if __name__ == "__main__":
    main()
