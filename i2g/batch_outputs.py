"""One real batch PLY, with an Explorer shortcut in the project final folder."""
import json
import os
import subprocess
import uuid
from pathlib import Path


def write_png_pair(source, ply):
    """Publish a full-resolution PNG beside the final PLY."""
    import shutil
    from PIL import Image, ImageOps
    from tempfile import NamedTemporaryFile
    target = Path(ply).with_suffix('.png')
    with NamedTemporaryFile(dir=target.parent, delete=False, suffix='.tmp') as tmp:
        temporary = Path(tmp.name)
    try:
        with Image.open(source) as image:
            if image.format == 'PNG':
                shutil.copyfile(source, temporary)
            else:
                oriented = ImageOps.exif_transpose(image)
                oriented.convert('RGBA' if 'A' in oriented.getbands() else 'RGB').save(temporary, 'PNG')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def linked_ply(final):
    try:
        target = Path(json.loads((final / 'ply_location.json').read_text())['path'])
        return target if target.is_file() else None
    except (OSError, ValueError, KeyError):
        return None


def _shortcut(link, target):
    env = os.environ | {'I2G_LINK': str(link), 'I2G_TARGET': str(target)}
    subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        '$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:I2G_LINK); '
        '$s.TargetPath=$env:I2G_TARGET; $s.Save()'], env=env, check=True,
        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def organize_ply(source, final, combined, enabled=True):
    source, final, combined = map(lambda p: Path(p).resolve(), (source, final, combined))
    destination = combined if enabled else final
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    if source != target and (target.exists() or target.with_suffix(".png").exists()):
        target = destination / f'{source.stem}_{uuid.uuid4().hex[:8]}.ply'
    moved = source != target
    if moved:
        source.rename(target)
    moved_png = False
    link = final / (source.name + '.lnk')
    try:
        if moved and source.with_suffix(".png").is_file():
            source.with_suffix(".png").rename(target.with_suffix(".png"))
            moved_png = True
        if enabled:
            _shortcut(link, target)
            (final / 'ply_location.json').write_text(json.dumps({'path': str(target)}), encoding='utf-8')
        else:
            (final / 'ply_location.json').unlink(missing_ok=True)
        for old in final.glob('*.ply.lnk'):
            if not enabled or old != link:
                old.unlink()
    except Exception:
        if moved_png:
            target.with_suffix(".png").rename(source.with_suffix(".png"))
        if moved:
            target.rename(source)
        raise
    return target
