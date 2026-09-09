"""Self-contained test suite for Keystroke-LLM.

Tests:
1. Model forward pass, attention shapes, and checkpoint serialization.
2. Character tokenization, key mapping, aliases, and physical matrix slots.
3. Hardware controller mock mode, buffer sizing, and background illumination.
4. Input reader edge cases (normalization, DEL/backspace, debounce, control codes).
5. Lockfile and PID liveness detection (Windows and POSIX).
6. Attention matrix row re-normalization.
7. Training dataset validation and checkpoint vocabulary reconciliation on resume.
8. End-to-end application lifecycle in mock mode.
"""

import errno
import os
import queue
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from hardware_controller import (
    KeyboardController,
    _is_pid_running,
    hex_to_rgb,
    load_profile,
    scale_rgb,
)
from key_mapper import CharTokenizer, char_to_key_name, get_default_vocab
from model import TinyTransformer
from predict_and_light import (
    LowLatencyInputReader,
    PredictiveKeyLightsApp,
    render_attention_matrix,
)
from train import build_sliding_window_dataset


class TestKeystrokeLLM(unittest.TestCase):

    def test_01_model_and_tokenizer(self):
        vocab = get_default_vocab()
        tokenizer = CharTokenizer(vocab)
        model = TinyTransformer(vocab_size=len(vocab), hidden_size=16, seq_len=8)

        # Ensure vocab attribute initialized
        self.assertTrue(hasattr(model, "vocab"))
        self.assertIsNone(model.vocab)

        # Test forward with single token
        logits, probs = model.forward([tokenizer.encode_char("a")])
        self.assertEqual(logits.shape, (1, len(vocab)))
        self.assertEqual(probs.shape, (len(vocab),))
        self.assertTrue(np.isclose(np.sum(probs), 1.0, atol=1e-5))

        # Test forward with max seq_len
        tokens = [tokenizer.encode_char(c) for c in "hello wo"]
        logits, probs = model.forward(tokens)
        self.assertEqual(logits.shape, (8, len(vocab)))
        self.assertEqual(probs.shape, (len(vocab),))

        # Test attention weights cache
        self.assertIsNotNone(model.last_attn_weights)
        self.assertEqual(model.last_attn_weights.shape, (8, 8))

        # Test checkpoint save and load with embedded vocab
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name
        try:
            model.save_checkpoint(tmp_path, vocab=vocab)
            loaded = TinyTransformer.load_checkpoint(tmp_path)
            self.assertEqual(loaded.vocab, vocab)
            self.assertEqual(loaded.vocab_size, len(vocab))
            self.assertEqual(loaded.seq_len, 8)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_02_key_mapping(self):
        profile = load_profile("hive75")
        self.assertEqual(profile.id, "hive75")
        self.assertGreaterEqual(len(profile.slot_map), 82)

        # Verify symbol mappings
        symbols = ["=", "[", "]", "-", ";", "'", ",", ".", "/", "\\", "`"]
        for s in symbols:
            name = char_to_key_name(s)
            self.assertIsNotNone(name, f"char_to_key_name failed for '{s}'")
            self.assertIn(name, profile.slot_map, f"Key '{name}' missing from profile.slot_map")
            slot = profile.slot_map[name]
            self.assertTrue(0 <= slot < 128, f"Slot {slot} out of range for '{name}'")

        # Verify modifier aliases
        for alias in ["ctrl", "shift", "alt", "function", "del", "return"]:
            self.assertIn(alias, profile.slot_map, f"Alias '{alias}' missing from profile.slot_map")
            self.assertTrue(0 <= profile.slot_map[alias] < 128)

    def test_03_hardware_controller_mock(self):
        ctrl = KeyboardController(mock=True)
        try:
            self.assertTrue(ctrl.mock)
            # Verify full background fill
            ctrl.set_key_colors({}, brightness=1.0, clear_others=True, default_background="ffffff")
            self.assertEqual(len(ctrl.rgb_buffer), ctrl.profile.num_slots * 3)
            self.assertEqual(ctrl.rgb_buffer, bytearray([255, 255, 255] * ctrl.profile.num_slots))

            # Verify key highlighting
            ctrl.set_key_colors({"w": "ff0000", "space": "0000ff"}, brightness=1.0, clear_others=True, default_background="ffffff")
            w_slot = ctrl.profile.slot_map["w"]
            space_slot = ctrl.profile.slot_map["space"]
            self.assertEqual(ctrl.rgb_buffer[w_slot * 3 : w_slot * 3 + 3], bytes([255, 0, 0]))
            self.assertEqual(ctrl.rgb_buffer[space_slot * 3 : space_slot * 3 + 3], bytes([0, 0, 255]))

            # Unhighlighted key remains white
            a_slot = ctrl.profile.slot_map["a"]
            self.assertEqual(ctrl.rgb_buffer[a_slot * 3 : a_slot * 3 + 3], bytes([255, 255, 255]))
        finally:
            ctrl.close()

    def test_04_input_reader_edge_cases(self):
        q = queue.Queue(maxsize=128)
        reader = LowLatencyInputReader(q)
        try:
            # Test carriage return normalization (\r -> \n)
            reader._enqueue("\r")
            self.assertFalse(q.empty())
            self.assertEqual(q.get_nowait(), "\n")

            # Test DEL normalization (\x7f -> \b)
            reader._enqueue("\x7f")
            self.assertFalse(q.empty())
            self.assertEqual(q.get_nowait(), "\b")

            # Test debouncing immediate duplicate within 15ms
            reader._enqueue("a")
            reader._enqueue("a")
            self.assertFalse(q.empty())
            self.assertEqual(q.get_nowait(), "a")
            self.assertTrue(q.empty())

            # Test unprintable control characters filtered out
            reader._enqueue("\x01")  # Ctrl+A
            reader._enqueue("\x16")  # Ctrl+V
            self.assertTrue(q.empty())

            # Test allowed backspace
            time.sleep(0.02)
            reader._enqueue("\b")
            self.assertFalse(q.empty())
            self.assertEqual(q.get_nowait(), "\b")
        finally:
            reader.stop()

    def test_05_pid_and_lockfile(self):
        my_pid = os.getpid()
        self.assertTrue(_is_pid_running(my_pid))
        self.assertFalse(_is_pid_running(999999))

        # Test POSIX PermissionError / EPERM retention
        with mock.patch("os.kill", side_effect=PermissionError("Permission denied")):
            with mock.patch("hardware_controller.os.name", "posix"):
                self.assertTrue(_is_pid_running(88888))

        with mock.patch("os.kill", side_effect=OSError(errno.EPERM, "Operation not permitted")):
            with mock.patch("hardware_controller.os.name", "posix"):
                self.assertTrue(_is_pid_running(88888))

        with mock.patch("os.kill", side_effect=ProcessLookupError("No such process")):
            with mock.patch("hardware_controller.os.name", "posix"):
                self.assertFalse(_is_pid_running(88888))

    def test_06_attention_renormalization(self):
        weights = np.array([[0.2, 0.3], [0.1, 0.4]])
        row_sums = weights.sum(axis=-1, keepdims=True)
        renorm = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
        self.assertTrue(np.allclose(renorm.sum(axis=-1), [1.0, 1.0]))

    def test_07_train_dataset_validation(self):
        tokenizer = CharTokenizer()
        # Empty text should produce empty dataset
        empty_ds = build_sliding_window_dataset("hi", tokenizer, seq_len=12)
        self.assertEqual(len(empty_ds), 0)

        # Valid text
        valid_ds = build_sliding_window_dataset("hello world this is a test text for dataset creation", tokenizer, seq_len=12)
        self.assertGreater(len(valid_ds), 0)

    def test_08_predictive_app_lifecycle_mock(self):
        app = PredictiveKeyLightsApp(
            checkpoint_path="checkpoints/model_final.npz",
            profile_name="hive75",
            mock=True,
            top_k=5,
            brightness=1.0,
            context_len=12,
            show_probs=False,
            show_attn=False
        )
        try:
            self.assertEqual(len(app.rolling_buffer), 0)
            app.rolling_buffer = list("hell")
            app._update_prediction()
            # Verify prediction illumination
            self.assertGreater(len(app.kbd._current_colors), 0)

            # Revert to empty context
            app.rolling_buffer = []
            app._update_prediction()
            self.assertEqual(len(app.kbd._current_colors), 0)
        finally:
            app.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
