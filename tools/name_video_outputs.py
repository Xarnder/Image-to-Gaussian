"""Rename legacy video final files using recorded source-image provenance."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from i2g.paths import OUTPUT_DIR, deliverable_path, safe_run_name

KINDS = ('character.ply', 'splats.pt', 'comparison.jpg', 'turntable.mp4', 'turntable_raw.mp4',
         'report.json', 'verification.json', 'verified_comparison.jpg', 'preview_8_angles.jpg', 'gui_complete.jpg')


def migrate(root=OUTPUT_DIR, apply=False):
    root = root.resolve()
    sources = {}
    for record in root.glob('*/*/orbit_candidates/*/generation.json'):
        try:
            data = json.loads(record.read_text(encoding='utf-8'))
            source = data['input']
            for candidate in data['candidates']:
                sources[str(Path(candidate['path']).resolve()).casefold()] = (source, data['experiment'])
        except (OSError, ValueError, KeyError):
            continue
    results = []
    for manifest in root.glob('*/*/video/input.json'):
        work = manifest.parent
        final = work / 'final'
        if not final.is_dir():
            continue
        data = json.loads(manifest.read_text(encoding='utf-8'))
        provenance = sources.get(str(Path(data['path']).resolve()).casefold())
        if not provenance:
            continue  # Never guess which original image produced a generated video.
        source, experiment = provenance
        name = safe_run_name(Path(source).stem)
        if not work.parent.name.startswith('orbit_'):
            name = work.parent.name  # Preserve batch collision suffixes.
        moves = [(final / kind, deliverable_path(final, name, kind, experiment)) for kind in KINDS if (final / kind).is_file()]
        for old, new in moves:
            old.resolve().relative_to(root)
            new.resolve().relative_to(root)
            if new.exists():
                raise ValueError(f'Refusing to overwrite {new}')
        if not moves:
            continue
        if apply:
            (work / 'output_naming.json').write_text(json.dumps({'name':name, 'experiment':experiment, 'source_image':source}, indent=2), encoding='utf-8')
            for old, new in moves:
                old.rename(new)
        results.append({'work':str(work), 'name':name, 'files':len(moves)})
    if apply:
        for manifest in root.glob('*/generated_batch.json'):
            data = json.loads(manifest.read_text(encoding='utf-8'))
            changed = False
            for item in data.get('items', []):
                old = Path(item['ply'])
                new = deliverable_path(old.parent, item['name'], 'character.ply', data['experiment'])
                if new.is_file():
                    item['ply'] = str(new); changed = True
            if changed:
                manifest.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(migrate(apply=args.apply), indent=2))
