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
from key_mapper import (
    CharTokenizer,
    char_to_key_name,
    get_default_vocab,
    VALID_PREDICTIVE_CHARS,
    format_prediction_label,
)
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
    if not tokens or attn_weights is None or attn_weights.size == 0:
        return
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

        # 2. Worker thread for stdin / msvcrt console input (runs alongside pynput for zero-miss capture)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

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
        if len(self._last_key_time) > 256:
            sorted_items = sorted(self._last_key_time.items(), key=lambda item: item[1])
            self._last_key_time = dict(sorted_items[-128:])
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
                 context_len: int = 48, show_probs: bool = False,
                 show_attn: bool = False, mock: bool = False,
                 idle_timeout: float = 6.0, demo: bool = False,
                 seed: str = "", auto_pause_wasd: bool = True):
        self.top_k = min(max(1, top_k), 5)
        self.brightness = max(0.0, min(1.0, brightness))
        self.show_probs = show_probs
        self.show_attn = show_attn
        self.idle_timeout = idle_timeout
        self.demo = demo
        self.seed = seed
        self.auto_pause_wasd = auto_pause_wasd
        self.is_gaming = False
        self.wasd_streak = 0
        self.last_char = None
        self.repeat_count = 0

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
        self.rolling_buffer = list(seed) if seed else []
        self.last_type_time = time.time()
        self.is_idle = False

        self.input_reader = LowLatencyInputReader(self.key_queue)
        self.running = True
        self._demo_stop_event = threading.Event()

        if self.demo:
            self.demo_thread = threading.Thread(target=self._demo_runner, daemon=True)
            self.demo_thread.start()

    def _demo_runner(self):
        demo_text = "The quick brown fox jumps over the lazy dog. How are you today? Thank you very much! "
        if self._demo_stop_event.wait(1.5):
            return
        while self.running and not self._demo_stop_event.is_set():
            for ch in demo_text:
                if not self.running or self._demo_stop_event.is_set():
                    return
                try:
                    self.key_queue.put_nowait(ch)
                except queue.Full:
                    pass
                if self._demo_stop_event.wait(1.2):
                    return
            if self._demo_stop_event.wait(2.0):
                return
            try:
                self.key_queue.put_nowait("\n")
            except queue.Full:
                pass

    def run(self):
        mode_str = "Demo" if self.demo else "Live"
        print(f"\n[PredictiveKeyLights] Active ({mode_str}) | Context: {self.context_len}")
        print("Type in any window. Press Ctrl+C to exit.\n", flush=True)

        # Initial state: If seed provided, predict immediately; otherwise solid white backlight
        if self.rolling_buffer:
            self._update_prediction()
        else:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
            if self.show_probs:
                print("[Context: (empty)] -> Start typing to see predictions...", flush=True)

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
                        # Auto-pause gaming movement (WASD) and repetitive key holding
                        if self.auto_pause_wasd:
                            if ch.lower() in ("w", "a", "s", "d"):
                                self.wasd_streak += 1
                            else:
                                self.wasd_streak = 0

                            if ch == self.last_char and ch not in (" ", "\n", "\b"):
                                self.repeat_count += 1
                            else:
                                self.last_char = ch
                                self.repeat_count = 1

                            # Trigger gaming pause if 4+ consecutive WASD keystrokes or 5+ identical key repeats
                            if (self.wasd_streak >= 4 or self.repeat_count >= 5):
                                if not self.is_gaming:
                                    self.is_gaming = True
                                    self.rolling_buffer.clear()
                                    self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
                                    if self.show_probs:
                                        print("\n[Gaming Mode] WASD detected. Lighting paused.", flush=True)

                        if self.is_gaming:
                            if ch in ("\n", " ") or (ch.lower() not in ("w", "a", "s", "d") and ch in VALID_PREDICTIVE_CHARS):
                                self.is_gaming = False
                                self.wasd_streak = 0
                                self.repeat_count = 1
                                self.rolling_buffer.clear()
                                if self.show_probs:
                                    print("[Typing Resumed] Predictions active.", flush=True)
                            else:
                                continue

                        if ch == "\b":
                            if self.rolling_buffer:
                                self.rolling_buffer.pop()
                        elif ch == "\n":
                            self.rolling_buffer.clear()
                        else:
                            if ch == " " and self.rolling_buffer and self.rolling_buffer[-1] == " ":
                                continue
                            self.rolling_buffer.append(ch)

                    if self.is_gaming:
                        continue

                    if len(self.rolling_buffer) > self.context_len:
                        self.rolling_buffer = self.rolling_buffer[-self.context_len:]

                    if self.rolling_buffer:
                        self._update_prediction()
                    else:
                        self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
                        if self.show_probs:
                            print("[Context: (empty)] -> Start typing to see predictions...", flush=True)
                else:
                    if self.is_gaming and (time.time() - self.last_type_time > 1.2):
                        self.is_gaming = False
                        self.wasd_streak = 0
                        self.repeat_count = 0
                        if self.show_probs:
                            print("[Typing Resumed] Idle timeout.", flush=True)

                    if not self.is_idle and (time.time() - self.last_type_time > self.idle_timeout):
                        self.is_idle = True
                        self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
                        if self.show_probs and self.rolling_buffer:
                            ctx = "".join(self.rolling_buffer)[-16:]
                            print(f"[Context: {ctx!r:<16}] -> (Idle)", flush=True)

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

        if np.isnan(probs).any() or float(np.nanmax(probs)) < 0.02:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
            return

        valid_candidates = []
        for token_id, prob in enumerate(probs):
            ch = self.tokenizer.decode_id(token_id)
            if ch in VALID_PREDICTIVE_CHARS:
                key_name = char_to_key_name(ch)
                if key_name:
                    valid_candidates.append((prob, ch, key_name))

        if not valid_candidates:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background="ffffff")
            return

        key_to_best_char = {}
        key_to_prob = {}
        for prob, ch, key_name in valid_candidates:
            if key_name not in key_to_prob or prob > key_to_best_char[key_name][0]:
                key_to_best_char[key_name] = (prob, ch)
            key_to_prob[key_name] = key_to_prob.get(key_name, 0.0) + prob

        sorted_keys = sorted(key_to_prob.keys(), key=lambda k: key_to_prob[k], reverse=True)[:self.top_k]

        key_colors = {}
        log_items = []

        for rank, key_name in enumerate(sorted_keys):
            color = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]
            key_colors[key_name] = color
            best_prob, best_ch = key_to_best_char[key_name]
            total_prob = key_to_prob[key_name]
            label = format_prediction_label(best_ch)
            log_items.append(f"#{rank+1}: {label} ({total_prob*100:.1f}%)")

        self.kbd.set_key_colors(key_colors, brightness=self.brightness, clear_others=True, default_background="ffffff")

        if self.show_probs:
            ctx = "".join(self.rolling_buffer)[-16:]
            print(f"[Context: {ctx!r:<16}] -> Predicted: {', '.join(log_items[:3])}", flush=True)

        if self.show_attn and self.model.last_attn_weights is not None:
            ctx_tokens = active_chars[-min(8, len(active_chars)):]
            weights = self.model.last_attn_weights[-len(ctx_tokens):, -len(ctx_tokens):]
            row_sums = weights.sum(axis=-1, keepdims=True)
            weights = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)
            render_attention_matrix(ctx_tokens, weights)

    def cleanup(self):
        self.running = False
        if hasattr(self, "_demo_stop_event"):
            self._demo_stop_event.set()
        if hasattr(self, "demo_thread") and self.demo_thread is not None and self.demo_thread.is_alive():
            try:
                self.demo_thread.join(timeout=0.5)
            except Exception:
                pass
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
    parser.add_argument("--context-len", type=int, default=48, help="Context sequence length (default: 48)")
    parser.add_argument("--show-probs", action="store_true", help="Print live top predictions to terminal")
    parser.add_argument("--show-attn", action="store_true", help="Visualize causal attention matrix")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no physical keyboard required)")
    parser.add_argument("--idle-timeout", type=float, default=6.0, help="Seconds before dimming during typing pause")
    parser.add_argument("--demo", action="store_true", help="Run automated typing demo showing predictions live")
    parser.add_argument("--seed", type=str, default="", help="Initial text prompt to seed predictions")
    parser.add_argument("--no-auto-pause-wasd", action="store_true", help="Disable automatic pause on WASD movement / key spam")
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
        idle_timeout=args.idle_timeout,
        demo=args.demo,
        seed=args.seed,
        auto_pause_wasd=not args.no_auto_pause_wasd
    )
    app.run()


if __name__ == "__main__":
    main()
