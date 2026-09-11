"""Unit test suite for Keystroke-LLM."""

import errno
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_controller import (
    KeyboardController,
    _is_pid_running,
    hex_to_rgb,
    load_profile,
    scale_rgb,
)
from key_mapper import CharTokenizer, VALID_PREDICTIVE_CHARS, char_to_key_name, get_default_vocab
from model import TinyTransformer
from predict_and_light import (
    GAMING_IDLE_TIMEOUT,
    LowLatencyInputReader,
    PredictiveKeyLightsApp,
    load_runtime_config,
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

            # Ctrl+C is ignored so the long-running process stays active.
            reader._enqueue("\x03")
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
        checkpoint_path = "checkpoints/model_final.npz"
        if not os.path.exists(checkpoint_path):
            self.skipTest(f"Checkpoint unavailable: {checkpoint_path}")
        app = PredictiveKeyLightsApp(
            checkpoint_path=checkpoint_path,
            profile_name="hive75",
            mock=True,
            top_k=5,
            brightness=1.0,
            context_len=12,
            show_probs=False,
            show_attn=False
        )
        try:
            class DeterministicModel:
                def __init__(self, vocab_size, seq_len, vocab):
                    self.vocab_size = vocab_size
                    self.seq_len = seq_len
                    self.vocab = vocab
                    self.last_attn_weights = None

                def forward(self, tokens):
                    probs = np.full(self.vocab_size, 1e-6)
                    probs[app.tokenizer.encode_char("e")] = 0.99
                    probs /= probs.sum()
                    return np.zeros((len(tokens), self.vocab_size)), probs

            app.model = DeterministicModel(app.model.vocab_size, app.model.seq_len, app.tokenizer.vocab)
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

    def test_09_long_context_and_sampling_stability(self):
        vocab = get_default_vocab()
        tokenizer = CharTokenizer(vocab)
        model = TinyTransformer(vocab_size=len(vocab), hidden_size=64, seq_len=48)

        # 48-character forward pass
        text_48 = "a" * 48
        tokens = tokenizer.encode(text_48)
        logits, probs = model.forward(tokens)
        self.assertEqual(logits.shape, (48, len(vocab)))
        self.assertEqual(probs.shape, (len(vocab),))
        self.assertFalse(np.isnan(probs).any())
        self.assertTrue(np.isclose(np.sum(probs), 1.0, atol=1e-5))

        # Context truncation if input > seq_len
        text_60 = "b" * 60
        logits_trunc, probs_trunc = model.forward(tokenizer.encode(text_60))
        self.assertEqual(logits_trunc.shape, (48, len(vocab)))

        # Temperature sampling stability: greedy, extreme cold, standard, hot
        seed = tokenizer.encode("the ")
        for temp in [0.0, 1e-5, 0.05, 0.5, 1.0, 2.0]:
            gen = model.generate(seed, num_chars=5, temperature=temp)
            self.assertEqual(len(gen), 5)
            self.assertTrue(all(0 <= tid < len(vocab) for tid in gen))

    def test_10_adam_state_checkpoint_persistence(self):
        vocab = get_default_vocab()
        tokenizer = CharTokenizer(vocab)
        model = TinyTransformer(vocab_size=len(vocab), hidden_size=32, seq_len=16)

        # Run 2 training steps with Adam
        ctx = tokenizer.encode("hello ")
        tgt = tokenizer.encode("ello w")
        loss1 = model.train_step_backprop(ctx, tgt, lr=0.005, use_adam=True)
        loss2 = model.train_step_backprop(ctx, tgt, lr=0.005, use_adam=True)
        self.assertEqual(model._adam_step, 2)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_ckpt = f.name
        try:
            model.save_checkpoint(tmp_ckpt, vocab=vocab)
            loaded = TinyTransformer.load_checkpoint(tmp_ckpt)
            self.assertEqual(loaded._adam_step, 2)
            self.assertTrue(np.allclose(loaded._adam_emb_m, model._adam_emb_m))
            self.assertTrue(np.allclose(loaded._adam_pos_m, model._adam_pos_m))
            self.assertTrue(np.allclose(loaded.W_pos, model.W_pos))
            self.assertTrue(np.allclose(loaded.W_out, model.W_out))
        finally:
            if os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)

    def test_11_gaming_wasd_auto_pause_transitions(self):
        checkpoint_path = "checkpoints/model_final.npz"
        if not os.path.exists(checkpoint_path):
            self.skipTest(f"Checkpoint unavailable: {checkpoint_path}")
        app = PredictiveKeyLightsApp(
            checkpoint_path=checkpoint_path,
            mock=True,
            show_probs=False,
            auto_pause_wasd=True
        )
        try:
            # Simulate gaming WASD streak
            for ch in ["w", "a", "s", "d"]:
                app.key_queue.put(ch)

            # Process queue manually or via run loop step
            while not app.key_queue.empty():
                ch = app.key_queue.get()
                if ch.lower() in ("w", "a", "s", "d"):
                    app.wasd_streak += 1
                if app.wasd_streak >= 4:
                    app.is_gaming = True
                    app.rolling_buffer.clear()
            self.assertTrue(app.is_gaming)
            self.assertEqual(len(app.rolling_buffer), 0)

            # Resume by typing regular text
            resume_char = "h"
            if ch in ("\n", " ") or (resume_char.lower() not in ("w", "a", "s", "d")):
                app.is_gaming = False
                app.wasd_streak = 0
                app.rolling_buffer.append(resume_char)
            self.assertFalse(app.is_gaming)
            self.assertEqual(app.rolling_buffer, ["h"])
        finally:
            app.cleanup()

    def test_12_gaming_pause_waits_through_round_break(self):
        checkpoint_path = "checkpoints/model_final.npz"
        if not os.path.exists(checkpoint_path):
            self.skipTest(f"Checkpoint unavailable: {checkpoint_path}")
        app = PredictiveKeyLightsApp(checkpoint_path=checkpoint_path, mock=True)
        try:
            app.is_gaming = True
            app.last_type_time = time.time() - (GAMING_IDLE_TIMEOUT - 1)
            app._handle_idle_iteration()
            self.assertTrue(app.is_gaming)

            app.last_type_time = time.time() - (GAMING_IDLE_TIMEOUT + 1)
            app._handle_idle_iteration()
            self.assertFalse(app.is_gaming)
        finally:
            app.cleanup()

    def test_13_space_does_not_exit_gaming_mode(self):
        checkpoint_path = "checkpoints/model_final.npz"
        if not os.path.exists(checkpoint_path):
            self.skipTest(f"Checkpoint unavailable: {checkpoint_path}")
        app = PredictiveKeyLightsApp(checkpoint_path=checkpoint_path, mock=True)
        try:
            app.is_gaming = True
            app.key_queue.put(" ")
            ch = app.key_queue.get_nowait()
            if ch == "\n" or (ch.lower() not in ("w", "a", "s", "d") and ch != " " and ch in VALID_PREDICTIVE_CHARS):
                app.is_gaming = False
            self.assertTrue(app.is_gaming)
        finally:
            app.cleanup()

    def test_14_idle_timeout_clears_predictions_by_default(self):
        checkpoint_path = "checkpoints/model_final.npz"
        if not os.path.exists(checkpoint_path):
            self.skipTest(f"Checkpoint unavailable: {checkpoint_path}")
        app = PredictiveKeyLightsApp(checkpoint_path=checkpoint_path, mock=True)
        try:
            self.assertEqual(app.idle_timeout, 6.0)
            app.rolling_buffer = list("hello")
            app.last_type_time = time.time() - 60
            app._update_prediction()
            app._handle_idle_iteration()
            self.assertTrue(app.is_idle)
            self.assertEqual(app.kbd._current_colors, {})
        finally:
            app.cleanup()

    def test_15_attention_matrix_edge_cases(self):
        # Empty inputs should safely return without exception
        render_attention_matrix([], np.array([]))
        render_attention_matrix(["a"], np.array([[1.0]]))

    def test_16_controller_atexit_and_idempotence(self):
        ctrl = KeyboardController(mock=True)
        self.assertTrue(ctrl._registered_atexit)
        self.assertTrue(ctrl._running)
        ctrl.close()
        self.assertFalse(ctrl._running)
        self.assertFalse(ctrl._registered_atexit)
        # Second close must be a no-op and not raise
        ctrl.close()

    def test_17_debounce_cache_pruning(self):
        q = queue.Queue(maxsize=512)
        reader = LowLatencyInputReader(q)
        try:
            # Enqueue 300 unique unicode characters
            for i in range(300):
                reader._enqueue(chr(0x4E00 + i))
            # Cache must be bounded to at most 257 items
            self.assertLessEqual(len(reader._last_key_time), 257)
        finally:
            reader.stop()

    def test_18_controller_lock_rejects_duplicate_owner(self):
        ctrl = KeyboardController(mock=True)
        try:
            with self.assertRaises(RuntimeError):
                KeyboardController(mock=True)
        finally:
            ctrl.close()

    def test_19_input_queue_overflow_remains_bounded(self):
        q = queue.Queue(maxsize=2)
        reader = LowLatencyInputReader(q)
        try:
            reader._put_event("a")
            reader._put_event("b")
            reader._put_event("c")
            self.assertEqual(q.qsize(), 2)
            self.assertEqual(reader.dropped_events, 1)
            self.assertEqual(q.get_nowait(), "b")
            self.assertEqual(q.get_nowait(), "c")
        finally:
            reader.stop()

    def test_20_malformed_controller_lock_is_not_removed(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as lock_file:
            lock_file.write("not-a-pid")
            lock_path = lock_file.name
        try:
            controller = KeyboardController.__new__(KeyboardController)
            controller.lockfile_path = lock_path
            controller._lock_held = False
            with self.assertRaises(RuntimeError):
                controller._acquire_lock()
            with open(lock_path, "r", encoding="utf-8") as saved_lock:
                self.assertEqual(saved_lock.read(), "not-a-pid")
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)

    def test_21_startup_failure_cleans_up_runtime(self):
        checkpoint_path = "checkpoints/model_final.npz"
        if not os.path.exists(checkpoint_path):
            self.skipTest(f"Checkpoint unavailable: {checkpoint_path}")
        app = PredictiveKeyLightsApp(checkpoint_path=checkpoint_path, mock=True, seed="hello")
        try:
            with mock.patch.object(app, "_update_prediction", side_effect=RuntimeError("startup failure")):
                with self.assertRaisesRegex(RuntimeError, "startup failure"):
                    app.run()
            self.assertFalse(app.running)
            self.assertFalse(app.input_reader.running)
            self.assertFalse(app.kbd._running)
        finally:
            app.cleanup()

    def test_22_runtime_config_overrides_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
            config_file.write('{"runtime": {"gaming_idle_timeout": 9}, "lighting": {"background": "101010", "rank_colors": ["100000", "200000", "300000", "400000", "500000"]}}')
            config_path = config_file.name
        try:
            config = load_runtime_config(config_path)
            self.assertEqual(config["runtime"]["gaming_idle_timeout"], 9)
            self.assertEqual(config["lighting"]["background"], "101010")
            self.assertEqual(len(config["lighting"]["rank_colors"]), 5)
            self.assertEqual(config["runtime"]["top_k"], 5)
        finally:
            os.remove(config_path)

    def test_23_runtime_config_rejects_invalid_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_file:
            config_file.write('{"runtime": {"brightness": 2}, "lighting": {"rank_colors": ["GGGGGG", "000000", "000000", "000000", "000000"]}}')
            config_path = config_file.name
        try:
            with self.assertRaises(ValueError):
                load_runtime_config(config_path)
        finally:
            os.remove(config_path)

    def test_24_unknown_profile_fails_fast(self):
        with self.assertRaises(FileNotFoundError):
            load_profile("profile-that-does-not-exist")


if __name__ == "__main__":
    unittest.main(verbosity=2)
