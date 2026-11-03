import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from i2g.generated_batch import GeneratedBatch
from i2g.video_options import video_command


class GeneratedBatchTests(unittest.TestCase):
    def setUp(self):
        shortcut = patch('i2g.batch_outputs._shortcut', side_effect=lambda link,target: link.write_text(str(target)))
        shortcut.start(); self.addCleanup(shortcut.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photos = [self.root / "face.png", self.root / "face.jpg", self.root / "other.png"]
        for p in self.photos:
            p.write_bytes(b"image")
        self.body = {"input": str(self.root), "experiment": "test", "generation_policy": "first", "candidate_count": 5}
        for target, value in [("i2g.generated_batch.OUTPUT_DIR", self.root / "output"), ("i2g.generated_batch.threading.Thread", None)]:
            p = patch(target, value) if value is not None else patch(target)
            p.start(); self.addCleanup(p.stop)

    def test_multiple_images_are_serial_and_have_separate_plys(self):
        batch = GeneratedBatch()
        seen = []
        def process(body, cancel):
            self.assertEqual(body["generation_policy"], "first")
            self.assertEqual(body["candidate_count"], 1)
            self.assertEqual(sum(i["status"] == "running" for i in batch.state["items"]), 1)
            item = next(i for i in batch.state["items"] if i["input"] == body["input"])
            ply = Path(item["ply"])
            ply.parent.mkdir(parents=True); ply.write_bytes(b"test PLY")
            seen.append(body["input"])
            return {"ok": True}
        batch.start(self.body, self.photos, process)
        batch._run(self.body, process)
        self.assertEqual(seen, [str(p) for p in self.photos])
        self.assertEqual(batch.state["status"], "done")
        self.assertEqual(batch.state["completed"], 3)
        self.assertEqual(len({i["ply"] for i in batch.state["items"]}), 3)
        self.assertTrue(batch.manifest.is_file())
        for item in batch.state['items']:
            self.assertGreaterEqual(item['finished_at'],item['started_at'])
            self.assertEqual(item['duration_seconds'],item['finished_at']-item['started_at'])
            self.assertTrue((Path(item['project_dir'])/'batch_timing.json').is_file())

    def test_append_during_last_item(self):
        batch = GeneratedBatch()
        batch.start(self.body, self.photos[:1], None)
        seen = []
        def process(body, cancel):
            seen.append(Path(body['input']))
            if len(seen) == 1:
                result = batch.append(batch.state['batch_id'], [self.photos[2], self.photos[1], self.photos[0]])
                self.assertEqual(result['added'], 2)
            return {'ok': False, 'error': 'test fixture'}
        batch._run(self.body, process)
        self.assertEqual(seen, [self.photos[0], self.photos[2], self.photos[1]])
        self.assertEqual(len({i['name'] for i in batch.state['items']}), 3)
        with self.assertRaises(ValueError):
            batch.append(batch.state['batch_id'], self.photos)

    def test_review_multiple_rejected_single_allowed(self):
        batch = GeneratedBatch()
        with self.assertRaises(ValueError):
            batch.start(self.body | {"generation_policy": "review"}, self.photos, None)
        batch.start(self.body | {"generation_policy": "review", "candidate_count": 1}, self.photos, None)
        self.assertTrue(batch.state["active"])

    def test_failure_is_recorded_and_remaining_images_are_attempted(self):
        batch = GeneratedBatch()
        batch.start(self.body, self.photos, None)
        batch._run(self.body, lambda *_: {"ok": False, "error": "No full orbit"})
        self.assertFalse(batch.state["active"])
        self.assertEqual([i["status"] for i in batch.state["items"]], ["failed", "failed", "failed"])
        self.assertIn("3 failed", batch.state["message"])
        self.assertEqual(batch.state["items"][0]["error"], "No full orbit")

    def test_stop_after_preserves_completed_output(self):
        batch = GeneratedBatch()
        def process(body, cancel):
            ply = Path(batch.state["items"][0]["ply"])
            ply.parent.mkdir(parents=True); ply.write_bytes(b"test PLY")
            batch.stop("after")
            self.assertFalse(cancel.is_set())
            return {"ok": True}
        batch.start(self.body, self.photos, process)
        batch._run(self.body, process)
        self.assertEqual(batch.state["completed"], 1)
        self.assertEqual(batch.state["status"], "stopped")
        self.assertEqual(batch.state["items"][1]["status"], "queued")

    def test_resume_saved_queue_after_restart_skips_completed_even_with_rebuild(self):
        session = self.root / 'session.json'
        batch = GeneratedBatch(session)
        settings = self.body | {'redo': True}
        seen = []
        def process(body, cancel):
            seen.append(Path(body['input']))
            item = next(i for i in batch.state['items'] if i['input'] == body['input'])
            ply = Path(item['ply'])
            ply.parent.mkdir(parents=True); ply.write_bytes(b'ply')
            if len(seen) == 1:
                batch.append(batch.state['batch_id'], self.photos[2:])
                batch.stop('after')
            return {'ok': True}
        batch.start(settings, self.photos[:2], process)
        batch._run(settings, process)
        batch = GeneratedBatch(session)
        batch.resume(batch.state['batch_id'], process)
        batch._run(settings, process)
        self.assertEqual(seen, self.photos)
        self.assertEqual(batch.state['completed'], 3)
        self.assertEqual(batch.state['status'], 'done')

    def test_stop_now_is_forwarded_and_does_not_advance(self):
        batch = GeneratedBatch()
        def process(body, cancel):
            batch.stop()
            self.assertTrue(cancel.is_set())
            return {"ok": False, "error": "Stopped"}
        batch.start(self.body, self.photos, process)
        batch._run(self.body, process)
        self.assertEqual(batch.state["status"], "stopped")
        self.assertEqual(batch.state["items"][1]["status"], "queued")

    def test_reconstruction_uses_image_name_instead_of_random_video_name(self):
        video = self.root / "orbit_random.mp4"
        video.write_bytes(b"fixture")
        cmd = video_command("python", self.body | {"input": str(video), "output_name": "portrait",
                                                   "approved_video": True, "approval_basis": "automatic-first"})
        output = Path(cmd[cmd.index("--output") + 1])
        self.assertEqual(output.parts[-3:], ("test", "portrait", "video"))

    def test_missing_final_output_is_a_failure(self):
        batch = GeneratedBatch()
        batch.start(self.body, self.photos, None)
        batch._run(self.body, lambda *_: {"ok": True})
        self.assertEqual(batch.state["status"], "failed")
        self.assertEqual(batch.state["completed"], 0)

    def test_finished_images_are_skipped_unless_rebuild_requested(self):
        batch = GeneratedBatch()
        batch.start(self.body, self.photos, None)
        for item in batch.state["items"]:
            ply = Path(item["ply"]); ply.parent.mkdir(parents=True); ply.write_bytes(b"saved")
        calls = []
        batch._run(self.body, lambda body, _: calls.append(body) or {"ok":True})
        self.assertEqual(calls, [])
        self.assertTrue(all(i["status"] == "skipped" for i in batch.state["items"]))
        batch.start(self.body | {"redo": True}, self.photos, None)
        batch._run(self.body | {"redo": True}, lambda body, _: calls.append(body) or {"ok":True})
        self.assertEqual(len(calls), 3)

    def test_batch_restart_restores_original_order_and_folder(self):
        session = self.root / 'session.json'
        batch = GeneratedBatch(session)
        batch.start(self.body, self.photos[::-1], None)
        restored = GeneratedBatch(session)
        self.assertFalse(restored.state['active'])
        self.assertEqual([i['input'] for i in restored.state['items']], [str(p) for p in self.photos[::-1]])
        self.assertEqual(restored.state['input'], str(self.root))
