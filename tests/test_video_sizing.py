import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import random
from pathlib import Path
import numpy as np
from PIL import Image
from i2g.video_sizing import image_layout, prepare_video_image, legacy_aspect_mismatch
from i2g.video_generation import api_workflow, WORKFLOW


class VideoSizingTests(unittest.TestCase):
    def test_dimension_limits_across_random_and_extreme_sizes(self):
        rng = random.Random(360)
        dimensions = [(1,1), (1,20000), (20000,1), (31,33), (32000,32000)]
        dimensions += [(rng.randint(64,20000), rng.randint(64,20000)) for _ in range(500)]
        for width, height in dimensions:
            raw = MagicMock()
            raw.__enter__.return_value = raw
            raw.size = (width, height)
            with patch('i2g.video_sizing.Image.open', return_value=raw), patch('i2g.video_sizing.ImageOps.exif_transpose', return_value=raw):
                layout = image_layout('fixture.png')
            w,h = layout['video_size']; rw,rh = layout['resized_size']
            self.assertTrue(w % 32 == h % 32 == 0)
            self.assertLessEqual(w*h,1248**2)
            self.assertLessEqual(max(w,h),1920)
            self.assertTrue(1 <= rw <= w and 1 <= rh <= h)
            left,top,right,bottom = layout['padding']
            self.assertEqual((rw+left+right,rh+top+bottom),(w,h))
            self.assertLessEqual(abs(left-right),1)
            self.assertLessEqual(abs(top-bottom),1)

    def test_exif_rotation_uses_displayed_aspect(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'rotated.jpg'
            exif=Image.Exif(); exif[274]=6
            Image.new('RGB',(120,240),'red').save(source,exif=exif)
            layout=image_layout(source)
            self.assertEqual(layout['source_size'],[240,120])
            dest=prepare_video_image(source,Path(tmp)/'prepared.png',layout)
            with Image.open(dest) as prepared:
                self.assertEqual(list(prepared.size),layout['video_size'])
                self.assertIsNone(prepared.getexif().get(274))

    def test_mixed_aspects_keep_all_edges_and_use_separate_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sizes = []
            for width, height in [(1122,1402), (1402,1122), (1248,1248), (512,1536), (2000,600), (4000,3000)]:
                source = root / f'{width}x{height}.png'
                im = Image.new('RGB',(width,height),'white')
                im.paste('red',(0,0,width,20)); im.paste('blue',(0,height-20,width,height))
                im.save(source)
                layout = image_layout(source); sizes.append(layout['video_size'])
                dest = prepare_video_image(source, root/'prepared.png', layout)
                with Image.open(dest) as prepared:
                    result = np.array(prepared)
                    self.assertEqual(list(prepared.size), layout['video_size'])
                self.assertTrue(all(n % 32 == 0 for n in layout['video_size']))
                self.assertLessEqual(np.prod(layout['video_size']), 1248**2)
                self.assertFalse(layout['crop'])
                self.assertTrue(np.all(result[0,:,0] > 240))
                self.assertTrue(np.all(result[-1,:,2] > 240))
                rw,rh=layout['resized_size']
                self.assertLess(abs(rw/rh-width/height), .003)
            self.assertNotEqual(sizes[0],sizes[1])

    def test_graph_dimensions_replace_square_resolution_and_crop(self):
        workflow=json.loads(WORKFLOW.read_text(encoding='utf-8'))
        graph=api_workflow(workflow,'prepared.png',1,'out',video_size=[1088,1376])
        self.assertNotIn('154',graph)
        for node in ('150','133'):
            self.assertEqual(graph[node]['inputs']['width'],1088)
            self.assertEqual(graph[node]['inputs']['height'],1376)
        self.assertEqual(graph['150']['inputs']['keep_proportion'],'resize')

    def test_legacy_square_crop_is_not_treated_as_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'frames').mkdir()
            source=root/'portrait.png'; Image.new('RGB',(1122,1402)).save(source)
            Image.new('RGB',(1248,1248)).save(root/'frames/frame_0000.png')
            self.assertTrue(legacy_aspect_mismatch(source,root))
