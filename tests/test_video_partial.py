"""Partial camera coverage must not weaken the default full-orbit checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_to_splat

import numpy as np
import video_to_splat as video
from i2g.video_options import video_command


class PartialVideoTests(unittest.TestCase):
    def summary(self):
        return dict(registered=46, total=62, points=13872,
                    orbit_span_degrees=359.95, largest_gap_degrees=139,
                    frame_azimuth_degrees={str(i): a for i, a in enumerate(
                        [0, 30, 60, 90, 116, 255, 270, 300, 330, 359.95])})

    def test_full_still_rejects_missing_rear(self):
        with self.assertRaisesRegex(RuntimeError, '46/62'):
            video.check_coverage(self.summary(), 'full')

    def test_automatic_falls_back_only_when_partial_is_usable(self):
        summary = video.check_coverage(self.summary(), 'auto')
        self.assertEqual(summary['coverage'], 'partial')
        self.assertIn('46/62', summary['fallback_reason'])
        self.assertIn('missing areas', video.coverage_message(summary))
        for change in [dict(registered=11), dict(points=999), dict(largest_gap_degrees=310)]:
            with self.assertRaises(RuntimeError):
                video.check_coverage(self.summary() | change, 'auto')

    def test_automatic_keeps_full_orbit_when_valid(self):
        summary = video.check_coverage(self.summary() | dict(registered=62, largest_gap_degrees=8), 'auto')
        self.assertEqual(summary['coverage'], 'full')
        self.assertNotIn('fallback_reason', summary)

    def test_automatic_uses_cached_cameras_and_records_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'sparse'; model.mkdir()
            (root / 'calibration_attempt.json').write_text(json.dumps(self.summary() | dict(model=str(model))))
            video.calibrate(root, 'auto')
            saved = json.loads((root / 'calibration.json').read_text())
            self.assertEqual(saved['coverage'], 'partial')
            self.assertIn('fallback_reason', saved)

    def test_partial_accepts_verified_arc_but_rejects_degenerate_data(self):
        self.assertEqual(video.check_coverage(self.summary(), 'partial')['covered_arc_degrees'], 221)
        for change in [dict(registered=11), dict(points=999), dict(largest_gap_degrees=310)]:
            with self.assertRaises(RuntimeError):
                video.check_coverage(self.summary() | change, 'partial')

    def test_partial_preview_never_visits_missing_back(self):
        angles = video.preview_angles(video.check_coverage(self.summary(), 'partial')) % 360
        self.assertEqual(len(angles), 144)
        self.assertTrue(np.all((angles <= 116.001) | (angles >= 254.999)))
        self.assertAlmostEqual(angles[0], angles[-1])

    def test_cached_partial_is_revalidated_for_full_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'sparse'; model.mkdir()
            (root / 'calibration.json').write_text(json.dumps(
                self.summary() | dict(model=str(model), coverage='partial')))
            with self.assertRaises(RuntimeError):
                video.calibrate(root, 'full')
            video.calibrate(root, 'partial')

    def test_gui_option_reaches_pipeline_and_rejects_invalid_value(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'video.mp4'; source.touch()
            body = dict(input=str(source), experiment='partial-test', approved_video=True)
            cmd = video_command('python', body | dict(coverage='partial'))
            self.assertEqual(cmd[cmd.index('--coverage')+1], 'partial')
            with self.assertRaises(ValueError):
                video_command('python', body | dict(coverage='invalid'))

    def test_closeup_preset_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'face.mp4'; source.touch()
            body = dict(input=str(source), experiment='closeup', approved_video=True, video_preset='closeup')
            cmd = video_command('python', body)
            for flag, value in [('--coverage', 'partial'), ('--initialization', 'sparse'), ('--iterations', '7500')]:
                self.assertEqual(cmd[cmd.index(flag)+1], value)
            cmd = video_command('python', body | dict(initialization='hull', iterations=2500))
            self.assertEqual(cmd[cmd.index('--initialization')+1], 'hull')
            self.assertEqual(cmd[cmd.index('--iterations')+1], '2500')
            for field, value in [('video_preset', 'invalid'), ('initialization', 'invalid')]:
                with self.assertRaises(ValueError): video_command('python', body | {field: value})

    def test_interactive_launcher_closeup_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'face.mp4'; source.touch()
            with patch('builtins.input', side_effect=['2', 'interactive-closeup', 'y']), \
                 patch.object(run_to_splat.sys.stdin, 'isatty', return_value=True), \
                 patch.object(run_to_splat.subprocess, 'call', return_value=0) as call:
                self.assertEqual(run_to_splat.main(['--input', str(source)]), 0)
                cmd = call.call_args.args[0]
                self.assertEqual(cmd[cmd.index('--initialization')+1], 'sparse')
                self.assertEqual(cmd[cmd.index('--iterations')+1], '7500')


if __name__ == '__main__':
    unittest.main()
