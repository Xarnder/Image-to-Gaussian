"""Workflow fidelity and candidate review regressions, without GPU inference."""
import json
from PIL import Image
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from i2g.video_generation import WORKFLOW, VideoGeneration, api_workflow, digest, default_prompt, generation_prompt
from i2g.video_options import video_settings


class VideoGenerationTests(unittest.TestCase):
    def test_custom_prompt_changes_only_conditioning_prompt(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        original = api_workflow(workflow, "face.png", 1, "out")
        prompt = "Wide shot, arms folded.\nKeep the original orbit and eye lock."
        edited = api_workflow(workflow, "face.png", 1, "out", prompt)
        self.assertEqual(original["133"]["inputs"]["prompt"], default_prompt())
        self.assertEqual(edited["133"]["inputs"].pop("prompt"), prompt)
        original["133"]["inputs"].pop("prompt")
        self.assertEqual(original, edited)

    def test_prompt_toggle_and_validation(self):
        self.assertEqual(generation_prompt({"custom_prompt": "ignored"}), default_prompt())
        self.assertEqual(generation_prompt({"custom_prompt_enabled": False, "custom_prompt": "ignored"}), default_prompt())
        exact = "  Close up\nPose unchanged.  "
        self.assertEqual(generation_prompt({"custom_prompt_enabled": True, "custom_prompt": exact}), exact)
        for invalid in (None, "", " \n", "x" * 32001):
            with self.assertRaises(ValueError):
                generation_prompt({"custom_prompt_enabled": True, "custom_prompt": invalid})

    def test_custom_prompt_is_saved_restored_and_separates_candidates(self):
        with tempfile.TemporaryDirectory() as tmp, patch("i2g.video_generation.OUTPUT_DIR", Path(tmp)), patch("i2g.video_generation.threading.Thread"):
            source = Path(tmp) / "face.png"; source.write_bytes(b"image")
            session = Path(tmp) / "session.json"
            g = VideoGeneration(session)
            body = {"input": str(source), "experiment": "test"}
            g.start(body, None)
            original_root = g.root
            g.update(active=False)
            g.start(body | {"custom_prompt_enabled": True, "custom_prompt": "Custom orbit prompt"}, None)
            self.assertNotEqual(g.root, original_root)
            self.assertEqual(g.state["prompt"], "Custom orbit prompt")
            restored = VideoGeneration(session)
            self.assertEqual(restored.snapshot()["settings"]["custom_prompt"], "Custom orbit prompt")
            self.assertTrue(restored.snapshot()["settings"]["custom_prompt_enabled"])
            g.update(active=False)
            g.start(body | {"custom_prompt_enabled": False, "custom_prompt": "Custom orbit prompt"}, None)
            self.assertEqual(g.root, original_root)

    def test_exact_prompt_lora_links_and_bypass(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        graph = api_workflow(workflow, "input/portrait.png", 42, "video/test")
        source = next(n for n in workflow["nodes"] if n["id"] == 133)
        self.assertEqual(graph["133"]["inputs"]["prompt"], source["widgets_values_named"]["prompt"])
        self.assertEqual(graph["140"]["inputs"]["lora_name"], "minimax_h3_flf2v_lora_v1.safetensors")
        self.assertEqual(graph["140"]["inputs"]["strength_model"], 1)
        self.assertEqual(graph["133"]["inputs"]["first_frame"], graph["133"]["inputs"]["last_frame"])
        self.assertEqual(graph["126"]["inputs"]["model"], ["140", 0])
        self.assertEqual(graph["133"]["inputs"]["length"], ["134", 1])
        self.assertNotIn("141", graph)
        self.assertNotIn("123", graph)
        self.assertNotIn("142", graph)
        self.assertEqual(graph["131"]["inputs"], {"noise_seed": 42})
        self.assertEqual(graph["149"]["inputs"], {"image": "input/portrait.png"})
        for node in graph.values():
            self.assertFalse(any(s in node["class_type"].lower() for s in ("qwen", "wan", "sam3d", "edit")))

    def test_only_seed_input_output_change(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        a = api_workflow(workflow, "a.png", 1, "a")
        b = api_workflow(workflow, "b.png", 2, "b")
        self.assertEqual([k for k in a if a[k] != b[k]], ["131", "92", "149"])

    def test_review_default_more_and_first(self):
        with tempfile.TemporaryDirectory() as tmp, patch("i2g.video_generation.OUTPUT_DIR", Path(tmp)), patch("i2g.video_generation.threading.Thread") as thread:
            source = Path(tmp) / "face.png"
            Image.new("RGB", (96, 128), "gray").save(source)
            g = VideoGeneration()
            body = {"input": str(source), "experiment": "test"}
            g.start(body, lambda *_: None)
            self.assertEqual(g.state["requested"], 5)
            self.assertEqual(g.state["policy"], "review")
            with self.assertRaises(ValueError):
                g.start(body, lambda *_: None)
            clip = g.root / "candidate.mp4"
            clip.write_bytes(b"video")
            c = {"id": "one", "path": str(clip), "sha256": digest(clip), "seed": 1}
            g.update(active=False, candidates=[c])
            g.start(body | {"generation_policy": "first", "candidate_count": 5}, lambda *_: None)
            self.assertEqual(g.state["requested"], 1)
            self.assertEqual(g.state["candidates"], [c])
            with self.assertRaises(ValueError):
                g.selection("one")
            g.update(active=False)
            chosen = g.selection("one", "automatic-first")
            self.assertEqual(chosen["input"], str(clip))
            self.assertEqual(chosen["mode"], "video")
            self.assertTrue(chosen["approved_video"])
            clip.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                g.selection("one")
            source.write_bytes(b"new image")
            g.start(body, lambda *_: None)
            self.assertEqual(g.state["candidates"], [])

    def test_bad_reconstruction_settings_rejected_before_generation(self):
        for body in ({"iterations": 0}, {"coverage": "invent"}, {"initialization": "unknown"}, {"stride": 0}):
            with self.assertRaises(ValueError):
                video_settings(body)

    def test_completed_batch_waits_for_review_and_auto_selects_new_video(self):
        for policy, count in (("review", 5), ("first", 1)):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp, patch("i2g.video_generation.OUTPUT_DIR", Path(tmp)), patch("i2g.video_generation.threading.Thread"), patch("i2g.video_generation.GpuLock") as lock, patch("i2g.video_generation.ComfyClient") as client:
                source = Path(tmp) / "face.png"
                Image.new("RGB", (96, 128), "gray").save(source)
                c = client.return_value
                c.is_up.return_value = True
                c.busy.return_value = False
                c._open_ws.return_value = None
                c.upload_image.return_value = "i2g/face.png"
                c.queue.return_value = "test-prompt-id"
                c.object_info.return_value = {n["class_type"]: {} for n in api_workflow(json.loads(WORKFLOW.read_text(encoding="utf-8")), "x", 1, "x").values()}
                c.history.return_value = {"status": {"completed": True}}
                c.output_files.return_value = [MagicMock(node_id="92", filename="video.mp4")]
                def download(item, dest):
                    dest.write_bytes(b"completed MP4")
                    return dest
                c.download.side_effect = download
                callback = MagicMock(return_value={"ok": True})
                g = VideoGeneration()
                g.start({"input": str(source), "experiment": "test", "generation_policy": policy}, callback)
                g._generate(source, count, callback)
                self.assertFalse(g.state["active"])
                self.assertEqual(len(g.state["candidates"]), count)
                self.assertEqual(c.queue.call_count, count)
                self.assertEqual(g.state["status"], "review")
                c.free.assert_called_once()
                lock.return_value.release.assert_called_once()
                if policy == "first":
                    callback.assert_called_once_with(g.state["candidates"][0]["id"], "automatic-first")
                else:
                    callback.assert_not_called()
                seeds = [call.args[0]["131"]["inputs"]["noise_seed"] for call in c.queue.call_args_list]
                self.assertEqual(len(set(seeds)), count)

    def test_stop_only_interrupts_owned_prompt(self):
        g = VideoGeneration()
        g.client = MagicMock()
        g.prompt_id = "ours"
        g.client._get_json.return_value = {"queue_running": [[0, "theirs"]], "queue_pending": [[1, "ours"]]}
        g._cancel_prompt()
        g.client.interrupt.assert_not_called()
        g.client._post_json.assert_called_once_with("/queue", {"delete": ["ours"]})
        g.client._get_json.return_value = {"queue_running": [[0, "ours"]]}
        g._cancel_prompt()
        g.client.interrupt.assert_called_once()

    def test_early_approval_interrupts_and_hands_off_after_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, patch("i2g.video_generation.OUTPUT_DIR", Path(tmp)), patch("i2g.video_generation.threading.Thread"), patch("i2g.video_generation.GpuLock") as lock, patch("i2g.video_generation.ComfyClient") as client:
            source = Path(tmp) / "face.png"
            Image.new("RGB", (96, 128), "gray").save(source)
            c = client.return_value
            c.is_up.return_value = True
            c.busy.return_value = False
            c._open_ws.return_value = None
            c.upload_image.return_value = "face.png"
            c.object_info.return_value = {n["class_type"]: {} for n in api_workflow(json.loads(WORKFLOW.read_text(encoding="utf-8")), "x", 1, "x").values()}
            c.queue.side_effect = ["first", "second"]
            c.output_files.return_value = [MagicMock(node_id="92", filename="video.mp4")]
            def download(item, dest):
                dest.write_bytes(b"video")
                return dest
            c.download.side_effect = download
            g = VideoGeneration()
            callback = MagicMock()
            def handoff():
                self.assertFalse(g.state["active"])
                c.free.assert_called_once()
                lock.return_value.release.assert_called_once()
                self.assertEqual(g.selection(g.state["selected"])["input"], g.state["candidates"][0]["path"])
                return {"ok": True}
            callback.side_effect = handoff
            def history(prompt):
                if prompt == "second":
                    self.assertTrue(g.approve_early(g.state["candidates"][0]["id"], callback))
                    with self.assertRaises(ValueError):
                        g.approve_early(g.state["candidates"][0]["id"], callback)
                return {"status": {"completed": True}}
            c.history.side_effect = history
            c._get_json.side_effect = [
                {"queue_running": [[0, "second"]]},
                {"queue_running": []},
            ]
            g.start({"input": str(source), "experiment": "early", "candidate_count": 5}, MagicMock())
            g._generate(source, 5, MagicMock())
            self.assertEqual(c.queue.call_count, 2)
            self.assertEqual(len(g.state["candidates"]), 1)
            c.interrupt.assert_called_once()
            callback.assert_called_once()

    def test_restart_restores_candidates_and_settings_without_starting_gpu(self):
        with tempfile.TemporaryDirectory() as tmp, patch("i2g.video_generation.OUTPUT_DIR", Path(tmp)), patch("i2g.video_generation.threading.Thread"):
            source = Path(tmp) / "face.png"
            Image.new("RGB", (96, 128), "gray").save(source)
            session = Path(tmp) / "session.json"
            g = VideoGeneration(session)
            g.start({"input": str(source), "experiment": "test", "iterations": 7500}, lambda *_: None)
            g.update(candidates=[{"id": "keep", "path": "clip.mp4"}])
            restored = VideoGeneration(session)
            self.assertFalse(restored.state["active"])
            self.assertEqual(restored.state["candidates"][0]["id"], "keep")
            self.assertEqual(restored.snapshot()["settings"]["iterations"], 7500)
            restored.clear()
            self.assertFalse(session.exists())
            self.assertTrue((g.root / "generation.json").exists())


if __name__ == "__main__":
    unittest.main()
