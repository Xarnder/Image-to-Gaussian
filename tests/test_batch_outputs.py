import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from i2g.batch_outputs import organize_ply, linked_ply
from i2g.paths import find_deliverable


class BatchOutputsTests(unittest.TestCase):
    def test_move_shortcut_lookup_and_restore_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); final=root/'photo'/'video'/'final'; final.mkdir(parents=True)
            source=final/'photo-and-test-character.ply'; source.write_bytes(b'ply fixture')
            combined=root/'final_ply'
            with patch('i2g.batch_outputs._shortcut', side_effect=lambda link,target: link.write_text(str(target))):
                target=organize_ply(source,final,combined)
                self.assertFalse(source.exists())
                self.assertEqual(len(list(root.rglob('*.ply'))),1)
                self.assertTrue(Path(str(source)+'.lnk').exists())
                self.assertEqual(find_deliverable(final,'photo','character.ply','test'),target)
                self.assertEqual(organize_ply(target,final,combined),target)
                restored=organize_ply(target,final,combined,False)
                self.assertEqual(restored,source)
                self.assertIsNone(linked_ply(final))
                self.assertEqual(len(list(root.rglob('*.ply'))),1)
                self.assertFalse(list(final.glob('*.lnk')))

    def test_failed_shortcut_rolls_back_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); final=root/'final'; final.mkdir()
            source=final/'test.ply'; source.write_bytes(b'ply')
            with patch('i2g.batch_outputs._shortcut',side_effect=OSError('failed')):
                with self.assertRaises(OSError): organize_ply(source,final,root/'combined')
            self.assertTrue(source.exists())
            self.assertEqual(len(list(root.rglob('*.ply'))),1)
