"""Routing and live progress regressions; no GPU inference is performed."""
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import run_to_splat
import splat_gui as gui
from i2g.video_options import video_command, video_output


class VideoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / "an orbit with spaces.mp4"
        self.source.write_bytes(b"input validation fixture")
        self.body = dict(mode="video", input=str(self.source), experiment="test run", approved_video=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_video_recipe_and_isolated_destination(self):
        cmd, mode = gui.build_command(self.body | dict(extra="--no_orbit --set iterations=1", preset="draft"))
        self.assertEqual(mode, "video")
        self.assertEqual(Path(cmd[2]).name, "video_to_splat.py")
        self.assertEqual(cmd[cmd.index("--iterations")+1], "15000")
        self.assertEqual(cmd[cmd.index("--stride")+1], "2")
        self.assertEqual(cmd[cmd.index("--max-height")+1], "1280")
        self.assertEqual(Path(cmd[cmd.index("--output")+1]), video_output("test run", self.source))
        self.assertNotIn("--no_orbit", cmd)

    def test_approval_and_input_type_enforced_on_server(self):
        for approval in (False, None, "true", 1):
            with self.assertRaises(ValueError):
                gui.build_command(self.body | dict(approved_video=approval))
        with self.assertRaises(ValueError):
            gui.build_command(self.body | dict(input=self.temp.name))
        with self.assertRaises(ValueError):
            gui.build_command(self.body | dict(iterations=0))

    def test_image_route_preserved(self):
        for mode, script in (("single", "image_to_splat.py"), ("batch", "batch_to_splat.py")):
            cmd, selected = gui.build_command(self.body | dict(mode=mode, preset="high", no_orbit=True))
            self.assertEqual(selected, mode)
            self.assertEqual(Path(cmd[2]).name, script)
            self.assertIn("--no_orbit", cmd)
            self.assertIn("high", cmd)

    def test_terminal_launchers_dispatch_without_losing_arguments(self):
        with patch.object(run_to_splat.subprocess, "call", return_value=0) as call:
            run_to_splat.main(["--method", "video", "--input", str(self.source), "--experiment", "test run", "--approved-video"])
            cmd = call.call_args.args[0]
            self.assertEqual(Path(cmd[2]).name, "video_to_splat.py")
            self.assertIn(str(self.source), cmd)
            run_to_splat.main(["--batch-launcher", "--method", "image", "--input", self.temp.name, "--preset", "draft"])
            cmd = call.call_args.args[0]
            self.assertEqual(Path(cmd[1]).name, "batch_to_splat.py")
            self.assertEqual(cmd[-2:], ["--preset", "draft"])

    def test_live_progress_arrives_before_child_exit(self):
        received = threading.Event()
        proc = subprocess.Popen([sys.executable, "-u", "-c",
            "import sys; print('progress now',flush=True); sys.stdin.readline()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            with patch.object(gui, "ingest_line", side_effect=lambda kind, text: received.set()):
                worker = threading.Thread(target=gui.pump_output, args=(proc,))
                worker.start()
                self.assertTrue(received.wait(3), "Progress buffered until the child exits")
                self.assertIsNone(proc.poll())
                proc.stdin.write(b"\n"); proc.stdin.flush()
                worker.join(3)
        finally:
            if proc.poll() is None: proc.terminate()
            proc.wait(timeout=5)
            proc.stdin.close(); proc.stdout.close()

    def test_stage_progress_and_output_paths(self):
        state = gui.RunState()
        state.mode = "video"; state.running = True
        state.input_path = str(self.source); state.experiment = "test run"
        state.reset_stages()
        with patch.object(gui, "STATE", state):
            for key in ("prepare", "calibrate", "train"):
                gui.ingest_line("log", "[video-event] " + json.dumps(dict(stage=key, fraction=.4, detail="working")))
            snap = state.snapshot()
            self.assertEqual(snap["stage_index"], 3)
            self.assertEqual(snap["stage_percent"], 40)
            self.assertEqual([s["status"] for s in snap["stages"]], ["done", "done", "running", "pending"])
            self.assertEqual(Path(snap["work_dir"]), video_output("test run", self.source))
            self.assertEqual(snap["preview"]["items"][0]["kind"], "video")
            for fraction in (.15, .5, .9, 1.):
                gui.ingest_line("log", "[video-event] " + json.dumps(dict(stage="export", fraction=fraction, detail="exporting")))
                snap = state.snapshot()
                self.assertEqual(snap["stage_index"], 4)
                self.assertEqual(snap["stage_percent"], round(fraction*100))
            self.assertEqual([s["status"] for s in snap["stages"]], ["done"]*4)
            state.running = False; state.status = "done"
            self.assertEqual(state.snapshot()["stage_percent"], 100)


if __name__ == "__main__":
    unittest.main()
