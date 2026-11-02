import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import splat_gui as gui
from i2g.paths import deliverable_path, find_deliverable
from i2g.video_options import archive_replaced_video
from video_to_splat import final_path
from tools.name_video_outputs import migrate


class OutputNamingTests(unittest.TestCase):
    def test_image_edit_batch_names_are_unique_for_same_stem_extensions(self):
        from batch_to_splat import run_names
        names = run_names([Path('folder/person.png'), Path('folder/person.jpg')])
        self.assertEqual(len(set(names.values())), 2)
    def test_every_export_kind_uses_original_name_and_experiment(self):
        args = SimpleNamespace(name='Original portrait', input=Path('orbit_random.mp4'), experiment='Batch 1')
        for kind in ('character.ply', 'splats.pt', 'report.json', 'comparison.jpg', 'turntable.mp4', 'turntable_raw.mp4'):
            self.assertEqual(final_path(args, Path('work'), kind).name, 'Original portrait-and-Batch 1-' + kind)

    def test_legacy_migration_uses_provenance_and_preserves_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); work = root/'exp'/'orbit_random'/'video'; final = work/'final'; final.mkdir(parents=True)
            clip = root/'candidate.mp4'; clip.touch()
            (work/'input.json').write_text(json.dumps({'path':str(clip)}))
            gen = root/'exp'/'Original'/'orbit_candidates'/'hash'; gen.mkdir(parents=True)
            (gen/'generation.json').write_text(json.dumps({'input':str(root/'Original.png'), 'experiment':'exp', 'candidates':[{'path':str(clip)}]}))
            (final/'character.ply').write_bytes(b'original bytes')
            self.assertEqual(migrate(root)[0]['name'], 'Original')
            migrate(root, True)
            renamed = final/'Original-and-exp-character.ply'
            self.assertEqual(renamed.read_bytes(), b'original bytes')
            self.assertEqual(find_deliverable(final, 'orbit_random', 'character.ply', 'exp'), renamed)
            self.assertEqual(migrate(root), [])

    def test_rebuild_archives_without_overwriting_previous_files(self):
        with tempfile.TemporaryDirectory() as temp, patch('i2g.video_options.OUTPUT_DIR', Path(temp)):
            root = Path(temp); source = root/'new.mp4'; source.write_bytes(b'new video')
            work = root/'exp'/'portrait'/'video'; work.mkdir(parents=True)
            (work/'input.json').write_text(json.dumps({'sha256':hashlib.sha256(b'old').hexdigest(),'stride':2,'max_height':1280}))
            (work/'saved.ply').write_bytes(b'previous result')
            archive = archive_replaced_video({'input':str(source),'experiment':'exp','output_name':'portrait'})
            self.assertEqual((archive/'saved.ply').read_bytes(), b'previous result')
            self.assertFalse(work.exists())

    def test_order_and_recursive_files_reach_shared_batch_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'nested').mkdir()
            for name in ('a.png','b.png','nested/c.png'):
                (root/name).touch()
            photos, warnings = gui.ordered_photos(root, True, ['b.png','nested/c.png','a.png'])
            self.assertFalse(warnings)
            self.assertEqual([p.relative_to(root).as_posix() for p in photos], ['b.png','nested/c.png','a.png'])
