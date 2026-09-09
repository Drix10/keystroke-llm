"""Real-time keystroke capture, Transformer inference, and keyboard lighting."""

import argparse
import os
import queue
import sys
import threading
import time
from typing import List

import numpy as np

from hardware_controller import KeyboardController
from key_mapper import CharTokenizer, char_to_key_name, get_default_vocab
from model import TinyTransformer

# Color palette for predicted keys: Rank 1 is vivid solid red, ranks 2-5 are graded soft red
RANK_COLORS = [
    "FF0000",  # Rank 1: Solid Vivid Red
    "FF3333",  # Rank 2: Bright Red
    "FF5555",  # Rank 3: Soft Red
    "FF7777",  # Rank 4: Light Red
    "FF9999",  # Rank 5: Pale Red
]


def render_attention_matrix(tokens: List[str], attn_weights: np.ndarray):
    """Prints an ASCII causal attention heatmap for the active context."""
    T = len(tokens)
    print("\n--- Attention Weights Matrix ---")
    header = "     " + " ".join([f"{repr(t)[1:-1]:>3}" for t in tokens])
    print(header)
    print("    " + "-" * (len(header) - 4))

    shades = " .:-=+*#%@"
    for i in range(T):
        row = f"{repr(tokens[i])[1:-1]:>3} |"
        for j in range(T):
            if j > i:
                row += "    "
            else:
                val = attn_weights[i, j]
                shade = shades[min(len(shades) - 1, int(val * len(shades)))]
                row += f"  {shade} "
        print(row)

    last_row = attn_weights[-1, :]
    focus = " | ".join([f"{repr(t)[1:-1]}: {w*100:.1f}%" for t, w in zip(tokens, last_row)])
    print(f"\nLast Character Attention Focus:\n  {focus}\n" + "-" * 40)


import _thread


class LowLatencyInputReader:
    """Asynchronous, non-blocking keystroke reader across Linux and Windows."""

    def __init__(self, key_queue: queue.Queue):
        self.key_queue = key_queue
        self.running = True
        self._last_key_time = {}
        self.listener = None
        self._start_capture()

    def _start_capture(self):
        # 1. Global hook via pynput if available
        try:
            from pynput import keyboard

            def on_press(key):
                if not self.running:
                    return False
                ch = None
                try:
                    if hasattr(key, "char") and key.char:
                        ch = key.char
                    elif key == keyboard.Key.space:
                        ch = " "
                    elif key == keyboard.Key.enter:
                        ch = "\n"
                    elif key == keyboard.Key.tab:
                        ch = "\t"
                    elif key == keyboard.Key.backspace:
                        ch = "\b"
                except Exception:
                    pass

                if ch:
                    self._enqueue(ch)

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.daemon = True
            self.listener.start()
        except Exception:
            self.listener = None

        # 2. Worker thread for stdin / msvcrt console input (fallback if pynput is unavailable)
        if self.listener is None:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
        else:
            self.thread = None

    def _enqueue(self, ch: str):
        # Normalize carriage return to newline and DEL to backspace
        if ch == "\r":
            ch = "\n"
        elif ch == "\x7f":
            ch = "\b"

        # Check for Ctrl+C
        if ch == "\x03":
            try:
                _thread.interrupt_main()
            except Exception:
                pass
            return

        # Ignore unprintable control characters except whitespace / backspace
        if ord(ch) < 32 and ch not in ("\n", "\t", "\b"):
            return

        # Debounce: avoid duplicate events if identical key fires within 15ms
        now = time.time()
        last_t = self._last_key_time.get(ch, 0.0)
        if (now - last_t) > 0.015:
            self._last_key_time[ch] = now
            try:
                self.key_queue.put_nowait(ch)
            except queue.Full:
                pass

    def _worker(self):
        # Direct console input reader
        if os.name == "nt":
            import msvcrt
            while self.running:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b'\x03':  # Ctrl+C from console
                        try:
                            _thread.interrupt_main()
                        except Exception:
                            pass
                        break
                    elif ch in (b'\x00', b'\xe0'):
                        msvcrt.getch()  # discard special prefix
                    else:
                        try:
                            decoded = ch.decode("utf-8", errors="ignore")
                            if decoded:
                                self._enqueue(decoded)
                        except Exception:
                            pass
                else:
                    time.sleep(0.01)
            return

        if os.name == "posix":
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            try:
                old = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                try:
                    while self.running:
                        r, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if r:
                            ch = sys.stdin.read(1)
                            if ch:
                                self._enqueue(ch)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass
        if hasattr(self, "thread") and self.thread is not None and self.thread.is_alive():
            try:
                self.thread.join(timeout=0.3)
            except Exception:
                pass


class PredictiveKeyLightsApp:
    """Coordinates keystroke ingestion, Transformer inference, and RGB lighting."""

    def __init__(self, checkpoint_path: str, profile_name: str = "hive75",
                 top_k: int = 5, brightness: float = 1.0,
                 context_len: int = 12, show_probs: bool = False,
                 show_attn: bool = False, mock: bool = False,
                 idle_timeout: float = 6.0):
        self.top_k = min(max(1, top_k), 5)
        self.brightness = max(0.0, min(1.0, brightness))
        self.show_probs = show_probs
        self.show_attn = show_attn
        self.idle_timeout = idle_timeout

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Run train.py first.")

        self.model = TinyTransformer.load_checkpoint(checkpoint_path)
        # Clamp context length to model sequence length to avoid shape assertion failures
        self.context_len = min(max(1, context_len), self.model.seq_len)
        vocab = getattr(self.model, "vocab", None) or get_default_vocab()
        if len(vocab) > self.model.vocab_size:
            raise ValueError(
                f"Vocabulary size ({len(vocab)}) exceeds model capacity ({self.model.vocab_size})."
            )
        self.tokenizer = CharTokenizer(vocab=vocab)
        self.kbd = KeyboardController(profile_name=profile_name, mock=mock)

        self.key_queue = queue.Queue(maxsize=128)
        self.rolling_buffer = []
        self.last_type_time = time.time()
        self.is_idle = False

        self.input_reader = LowLatencyInputReader(self.key_queue)
        self.running = True

    def run(self):
        print("\n" + "=" * 55)
        print("  Predictive Key Lights Active")
        print("  Full-board clean white backlight enabled.")
        print("  Active keystroke capture running (terminal + global).")
        print("  Type in any window, browser, or terminal.")
        print("  Press Ctrl+C to exit.")
        print("=" * 55 + "\n")

        # Initial state: Full keyboard in solid bright white backlight. No red keys until user types!
        self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
        if self.show_probs:
            sys.stdout.write("\r\033[K[Context: (empty)] -> Start typing to see predictions...")
            sys.stdout.flush()

        try:
            while self.running:
                try:
                    first_ch = self.key_queue.get(timeout=0.005)
                    new_chars = [first_ch]
                    while not self.key_queue.empty():
                        try:
                            new_chars.append(self.key_queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    new_chars = []

                if new_chars:
                    self.last_type_time = time.time()
                    self.is_idle = False
                    for ch in new_chars:
                        if ch == "\b":
                            if self.rolling_buffer:
                                self.rolling_buffer.pop()
                        else:
                            self.rolling_buffer.append(ch)

                    if len(self.rolling_buffer) > self.context_len:
                        self.rolling_buffer = self.rolling_buffer[-self.context_len:]

                    if self.rolling_buffer:
                        self._update_prediction()
                    else:
                        # Reverted to empty context via backspace: reset all keys to solid white
                        self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
                        if self.show_probs:
                            sys.stdout.write("\r\033[K[Context: (empty)] -> Start typing to see predictions...")
                            sys.stdout.flush()
                else:
                    # Idle timeout: return to clean white backlight when typing is paused
                    if not self.is_idle and (time.time() - self.last_type_time > self.idle_timeout):
                        self.is_idle = True
                        self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
                        if self.show_probs and self.rolling_buffer:
                            ctx = "".join(self.rolling_buffer)[-12:]
                            sys.stdout.write(f"\r\033[K[Context: {ctx!r:<12}] -> (Idle pause - keys white)")
                            sys.stdout.flush()

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def _update_prediction(self):
        if not self.rolling_buffer:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
            return

        active_chars = self.rolling_buffer[-self.context_len:]
        tokens = [self.tokenizer.encode_char(c) for c in active_chars]
        _, probs = self.model.forward(tokens)

        # Ignore ambiguous or degenerate/NaN probability distributions
        if np.isnan(probs).any() or float(np.nanmax(probs)) < 0.02:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
            return

        top_indices = np.argsort(probs)[::-1][:self.top_k]
        key_colors = {}
        log_items = []

        # Target scheme: All keys normally clean White, predicted keys illuminate in bright Red
        for rank, token_id in enumerate(top_indices):
            ch = self.tokenizer.decode_id(token_id)
            key_name = char_to_key_name(ch)
            color = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]

            if key_name and key_name not in key_colors:
                key_colors[key_name] = color

            label = repr(ch) if ch in (" ", "\n", "\t") else f"'{ch}'"
            log_items.append(f"#{rank+1}: {label} ({probs[token_id]*100:.1f}%)")

        self.kbd.set_key_colors(key_colors, brightness=self.brightness, clear_others=True, default_background="ffffff")

        if self.show_probs:
            ctx = "".join(self.rolling_buffer)[-12:]
            sys.stdout.write(f"\r\033[K[Context: {ctx!r:<12}] -> Next: {', '.join(log_items[:3])}")
            sys.stdout.flush()

        if self.show_attn and self.model.last_attn_weights is not None:
            ctx_tokens = active_chars[-min(8, len(active_chars)):]
            weights = self.model.last_attn_weights[-len(ctx_tokens):, -len(ctx_tokens):]
            row_sums = weights.sum(axis=-1, keepdims=True)
            weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
            render_attention_matrix(ctx_tokens, weights)

    def cleanup(self):
        self.running = False
        self.input_reader.stop()
        self.kbd.close()


def main():
    if os.name == "nt":
        os.system("")  # Enable Windows virtual terminal / ANSI escape sequences
    parser = argparse.ArgumentParser(description="Predictive Key Lights - Real-time Keystroke Lighting")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_final.npz", help="Model checkpoint path")
    parser.add_argument("--profile", type=str, default="hive75", help="Keyboard profile (default: hive75)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predicted keys to illuminate (1-5)")
    parser.add_argument("--brightness", type=float, default=1.0, help="LED brightness scale (0.0 to 1.0)")
    parser.add_argument("--context-len", type=int, default=12, help="Context sequence length")
    parser.add_argument("--show-probs", action="store_true", help="Print live top predictions to terminal")
    parser.add_argument("--show-attn", action="store_true", help="Visualize causal attention matrix")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no physical keyboard required)")
    parser.add_argument("--idle-timeout", type=float, default=6.0, help="Seconds before dimming during typing pause")
    args = parser.parse_args()

    app = PredictiveKeyLightsApp(
        checkpoint_path=args.checkpoint,
        profile_name=args.profile,
        top_k=args.top_k,
        brightness=args.brightness,
        context_len=args.context_len,
        show_probs=args.show_probs,
        show_attn=args.show_attn,
        mock=args.mock,
        idle_timeout=args.idle_timeout
    )
    app.run()


if __name__ == "__main__":
    main()
