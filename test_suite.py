"""
test_suite.py - Automated Unit Tests for Predictive Key Lights
==============================================================
Validates math, shape correctness, tokenizer, and hardware controller.
"""

import os
import unittest
import numpy as np

from model import TinyTransformer, softmax, relu, matmul, transpose
from key_mapper import CharTokenizer, char_to_key_name, is_shifted, get_default_vocab
from hardware_controller import KeyboardController, hex_to_rgb, scale_rgb


class TestTransformerMath(unittest.TestCase):
    def test_softmax_numerical_stability(self):
        # Huge numbers that would normally overflow exp()
        huge_x = np.array([[1000.0, 1005.0, 999.0]])
        probs = softmax(huge_x, axis=-1)
        self.assertFalse(np.isnan(probs).any())
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        # Verify order preserved
        self.assertTrue(probs[0, 1] > probs[0, 0] > probs[0, 2])

    def test_causal_mask(self):
        model = TinyTransformer(vocab_size=10, hidden_size=8, seq_len=4)
        _, _ = model.forward([1, 2, 3])
        attn = model.last_attn_weights
        self.assertEqual(attn.shape, (3, 3))
        # Top-right above diagonal must be strictly 0.0
        self.assertEqual(attn[0, 1], 0.0)
        self.assertEqual(attn[0, 2], 0.0)
        self.assertEqual(attn[1, 2], 0.0)
        # Diagonal and lower triangle must be non-zero
        self.assertGreater(attn[0, 0], 0.0)
        self.assertGreater(attn[1, 1], 0.0)

    def test_forward_and_shapes(self):
        vocab_size = 20
        hidden_size = 16
        seq_len = 8
        model = TinyTransformer(vocab_size=vocab_size, hidden_size=hidden_size, seq_len=seq_len)
        tokens = [2, 5, 7, 1]
        logits, probs = model.forward(tokens)
        
        self.assertEqual(logits.shape, (4, vocab_size))
        self.assertEqual(probs.shape, (vocab_size,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_training_heuristic(self):
        model = TinyTransformer(vocab_size=10, hidden_size=8, seq_len=4)
        loss1 = model.train_step_heuristic([1, 2, 3], target_token=5, lr=0.05)
        self.assertGreater(loss1, 0.0)

    def test_training_backprop(self):
        model = TinyTransformer(vocab_size=10, hidden_size=8, seq_len=4)
        loss1 = model.train_step_backprop([1, 2, 3], target_token=5, lr=0.05)
        self.assertGreater(loss1, 0.0)

    def test_checkpoint_roundtrip(self):
        model = TinyTransformer(vocab_size=12, hidden_size=8, seq_len=4)
        ckpt_path = "checkpoints/test_ckpt.npz"
        os.makedirs("checkpoints", exist_ok=True)
        model.save_checkpoint(ckpt_path)
        
        loaded = TinyTransformer.load_checkpoint(ckpt_path)
        self.assertEqual(model.vocab_size, loaded.vocab_size)
        self.assertEqual(model.hidden_size, loaded.hidden_size)
        np.testing.assert_allclose(model.W_emb, loaded.W_emb)
        np.testing.assert_allclose(model.W_out, loaded.W_out)
        
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


class TestKeyMapperAndTokenizer(unittest.TestCase):
    def test_tokenizer_roundtrip(self):
        tokenizer = CharTokenizer()
        sample = "hello world! 123"
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(sample, decoded)

    def test_key_mapping(self):
        self.assertEqual(char_to_key_name('a'), 'a')
        self.assertEqual(char_to_key_name('A'), 'a')
        self.assertEqual(char_to_key_name(' '), 'space')
        self.assertEqual(char_to_key_name('\n'), 'enter')
        self.assertEqual(char_to_key_name('!'), '1')
        self.assertTrue(is_shifted('!'))
        self.assertFalse(is_shifted('1'))


class TestHardwareHelpers(unittest.TestCase):
    def test_color_utilities(self):
        r, g, b = hex_to_rgb("ff8000")
        self.assertEqual((r, g, b), (255, 128, 0))
        
        sr, sg, sb = scale_rgb(r, g, b, 0.5)
        self.assertEqual((sr, sg, sb), (127, 64, 0))

    def test_mock_controller(self):
        ctrl = KeyboardController(profile_name="hive75", mock=True)
        ctrl.set_key_colors({"a": "ff0000", "space": "00ff00"}, brightness=0.8)
        ctrl.clear()
        ctrl.close()


if __name__ == "__main__":
    unittest.main()
