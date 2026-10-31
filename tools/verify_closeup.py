import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from i2g.evaluate import load_ply
from i2g.splat import render_rgb

root = Path(sys.argv[1])
final = root / 'final'
params = load_ply(final / 'character.ply')
checkpoint = torch.load(final / 'splats.pt', map_location='cuda', weights_only=True)
errors = {}
for key in ['means', 'scales', 'quats', 'opacities', 'sh0']:
    expected = checkpoint[key]
    if key == 'quats':
        expected = torch.nn.functional.normalize(expected, dim=-1)
    errors[key] = float((params[key] - expected).abs().max())
records = json.loads((root / 'records.json').read_text())
report = json.loads((final / 'report.json').read_text())
names = ['frame_0000.png', 'frame_0018.png', 'frame_0036.png', 'frame_0042.png',
         'frame_0076.png', 'frame_0080.png', 'frame_0098.png', 'frame_0110.png']
sheet = Image.new('RGB', (1600, 1000), (128, 128, 128))
draw = ImageDraw.Draw(sheet)
with torch.no_grad():
    for index, name in enumerate(names):
        rec = next(r for r in records if Path(r['path']).name == name)
        source = np.array(Image.open(rec['path']))
        mask = np.array(Image.open(rec['mask'])) / 255.
        ref = (source * mask[..., None] + 128 * (1-mask[..., None])).astype(np.uint8)
        pred = render_rgb(params, torch.tensor(rec['w2c'], device='cuda', dtype=torch.float32),
                          torch.tensor(rec['K'], device='cuda', dtype=torch.float32), source.shape[1], source.shape[0])
        for j, image in enumerate([ref, pred]):
            tile = Image.fromarray(image)
            tile.thumbnail((200, 465))
            sheet.paste(tile, (index % 4 * 400 + j * 200, index // 4 * 500))
        draw.text((index % 4 * 400 + 6, index // 4 * 500 + 475), name + ' | source / PLY', fill='white')
sheet.save(final / 'verified_comparison.jpg')
cap = cv2.VideoCapture(str(final / 'turntable.mp4'))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
result = dict(gaussians=len(params['means']), all_finite=all(bool(torch.isfinite(v).all()) for v in params.values()),
              checkpoint_errors=errors, preview_frames=frames, heldout_psnr=report['heldout_mean_psnr'])
(final / 'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
assert result['all_finite'] and max(errors.values()) < 1e-6 and frames == 144
