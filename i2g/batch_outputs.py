"""One real batch PLY, with an Explorer shortcut in the project final folder."""
import json
import os
import subprocess
import uuid
from pathlib import Path


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
    if source != target and target.exists():
        target = destination / f'{source.stem}_{uuid.uuid4().hex[:8]}.ply'
    moved = source != target
    if moved:
        source.rename(target)
    link = final / (source.name + '.lnk')
    try:
        if enabled:
            _shortcut(link, target)
            (final / 'ply_location.json').write_text(json.dumps({'path': str(target)}), encoding='utf-8')
        else:
            (final / 'ply_location.json').unlink(missing_ok=True)
        for old in final.glob('*.ply.lnk'):
            if not enabled or old != link:
                old.unlink()
    except Exception:
        if moved:
            target.rename(source)
        raise
    return target
