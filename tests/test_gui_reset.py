import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import splat_gui as gui


class ResetTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(gui, 'GENERATION', gui.VideoGeneration())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_requires_confirmation(self):
        with patch.object(gui, 'STATE', gui.RunState()):
            for value in [None, False, 'true', 1]:
                self.assertFalse(gui.reset_run({'confirmed': value})['ok'])

    def test_rejects_running_or_live_process(self):
        state = gui.RunState()
        state.experiment = 'keep-me'
        with patch.object(gui, 'STATE', state):
            state.running = True
            self.assertFalse(gui.reset_run({'confirmed': True})['ok'])
            state.running = False
            state.proc = Mock(); state.proc.poll.return_value = None
            self.assertFalse(gui.reset_run({'confirmed': True})['ok'])
            self.assertEqual(state.experiment, 'keep-me')

    def test_clears_state_keeps_subscribers_and_saved_files(self):
        state = gui.RunState()
        subscriber = queue.Queue()
        state.subscribers.append(subscriber)
        state.history.append({'type': 'log', 'text': 'old'})
        state.status = 'done'; state.experiment = 'old-project'
        state.tqdm_frac = 1.; state.error = {'message': 'old'}
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'character.ply'; saved.write_bytes(b'completed result')
            state.input_path = str(saved)
            with patch.object(gui, 'STATE', state):
                result = gui.reset_run({'confirmed': True})
            self.assertTrue(result['ok'])
            self.assertEqual(saved.read_bytes(), b'completed result')
            self.assertEqual(result['state']['status'], 'idle')
            self.assertEqual(result['state']['experiment'], '')
            self.assertEqual(result['state']['input'], '')
            self.assertEqual(result['state']['stage_percent'], 0)
            self.assertEqual(result['state']['preview']['items'], [])
            self.assertIs(state.subscribers[0], subscriber)
            self.assertEqual(subscriber.get_nowait()['type'], 'reset')
            self.assertFalse(any(e.get('text') == 'old' for e in state.history))


if __name__ == '__main__':
    unittest.main()
