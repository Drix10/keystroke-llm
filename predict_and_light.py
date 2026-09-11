"""Real-time keystroke capture, Transformer inference, and keyboard lighting."""

import argparse
import json
import os
import queue
import re
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

STARTUP_VALUE_NAME = "KeystrokeLLM"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DEFAULT_RANK_COLORS = [
    "FF0000", "FF3333", "FF5555", "FF7777", "FF9999"
]


def load_runtime_config(config_path: str = DEFAULT_CONFIG_PATH):
    defaults = {
        "runtime": {
            "checkpoint": "checkpoints/model_final.npz",
            "profile": "hive75",
            "top_k": 5,
            "brightness": 1.0,
            "context_len": 48,
            "idle_timeout": 6.0,
            "gaming_idle_timeout": 12.0,
            "auto_pause_wasd": True,
            "queue_size": 1024,
            "debounce_ms": 15,
            "wasd_threshold": 4,
            "repeat_threshold": 5,
            "show_probs": False,
            "show_attn": False,
            "demo": False,
            "seed": "",
        },
        "lighting": {"background": "FFFFFF", "rank_colors": list(DEFAULT_RANK_COLORS)},
        "hardware": {"keepalive_hz": None},
    }
    if not os.path.exists(config_path):
        return defaults
    with open(config_path, "r", encoding="utf-8") as config_file:
        supplied = json.load(config_file)
    for section in defaults:
        if isinstance(supplied.get(section), dict):
            defaults[section].update(supplied[section])
    runtime = defaults["runtime"]
    lighting = defaults["lighting"]
    hardware = defaults["hardware"]
    integer_ranges = {"top_k": (1, 5), "context_len": (1, None), "queue_size": (1, None),
                      "wasd_threshold": (1, None), "repeat_threshold": (1, None)}
    for key, (minimum, maximum) in integer_ranges.items():
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum and value > maximum):
            raise ValueError(f"config runtime.{key} must be an integer in range {minimum}-{maximum or 'infinity'}")
    for key in ("brightness", "idle_timeout", "gaming_idle_timeout", "debounce_ms"):
        value = runtime[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"config runtime.{key} must be a non-negative number")
    for key in ("auto_pause_wasd", "show_probs", "show_attn", "demo"):
        if not isinstance(runtime[key], bool):
            raise ValueError(f"config runtime.{key} must be true or false")
    for key in ("checkpoint", "profile", "seed"):
        if not isinstance(runtime[key], str):
            raise ValueError(f"config runtime.{key} must be a string")
    if hardware["keepalive_hz"] is not None and (
            isinstance(hardware["keepalive_hz"], bool) or
            not isinstance(hardware["keepalive_hz"], (int, float)) or
            hardware["keepalive_hz"] < 0):
        raise ValueError("config hardware.keepalive_hz must be null or a non-negative number")
    colors = [lighting["background"], *lighting["rank_colors"]]
    if len(lighting["rank_colors"]) != 5 or any(
            not isinstance(color, str) or not re.fullmatch(r"[0-9A-Fa-f]{6}", color.lstrip("#"))
            for color in colors):
        raise ValueError("config lighting colors must be six-digit hexadecimal strings")
    return defaults


GAMING_IDLE_TIMEOUT = 12.0


def configure_windows_startup(install: bool):
    """Install or remove the per-user Windows startup entry."""
    if os.name != "nt":
        raise RuntimeError("Windows startup registration is only available on Windows.")

    import winreg

    run_key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key_path, 0, winreg.KEY_SET_VALUE) as run_key:
        if install:
            config_path = os.path.abspath(DEFAULT_CONFIG_PATH)
            executable_path = sys.executable
            if os.name == "nt" and os.path.basename(executable_path).lower() == "python.exe":
                windowless_executable = os.path.join(os.path.dirname(executable_path), "pythonw.exe")
                if os.path.exists(windowless_executable):
                    executable_path = windowless_executable
            command = f'"{executable_path}" "{os.path.abspath(__file__)}" --config "{config_path}"'
            winreg.SetValueEx(run_key, STARTUP_VALUE_NAME, 0, winreg.REG_SZ, command)
            print(f"Installed Windows startup entry: {command}")
        else:
            try:
                winreg.DeleteValue(run_key, STARTUP_VALUE_NAME)
                print("Removed Windows startup entry.")
            except FileNotFoundError:
                print("Windows startup entry was not installed.")


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
        self.debounce_seconds = 0.015
        self.key_queue = key_queue
        self.running = True
        self._last_key_time = {}
        self._debounce_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self.dropped_events = 0
        self.listener = None
        self._start_capture()

    def _start_capture(self):
        # 1. Global hook via pynput if available
        listener_started = False
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
            listener_started = True
        except Exception:
            self.listener = None

        # 2. Use console input only when the global hook is unavailable. Running both
        # sources can deliver the same terminal keystroke twice.
        self.thread = None
        if not listener_started:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def _enqueue(self, ch: str):
        if not ch:
            return
        # Normalize carriage return to newline and DEL to backspace.
        # Different terminals (PowerShell, Windows Terminal, bash) emit different byte
        # codes for Enter (\r vs \n) and Backspace (\x7f vs \b).
        if ch == "\r":
            ch = "\n"
        elif ch == "\x7f":
            ch = "\b"

        # On Windows, msvcrt.getch() intercepts Ctrl+C (b'\x03') as a raw byte instead of
        # raising KeyboardInterrupt. We manually signal interrupt_main() to allow clean exit.
        if ch == "\x03":
            try:
                _thread.interrupt_main()
            except Exception:
                pass
            return

        # Filter out unprintable terminal control sequences while preserving whitespace
        if ord(ch) < 32 and ch not in ("\n", "\t", "\b"):
            return

        # 15ms debounce window prevents mechanical switch contact bounce from registering twice.
        # MEMORY SAFETY FIX: Prune the debounce dictionary when it exceeds 256 keys so long
        # typing sessions don't leak memory over time.
        now = time.time()
        with self._debounce_lock:
            if len(self._last_key_time) > 256:
                sorted_items = sorted(self._last_key_time.items(), key=lambda item: item[1])
                self._last_key_time = dict(sorted_items[-128:])
            last_t = self._last_key_time.get(ch, 0.0)
            if (now - last_t) <= self.debounce_seconds:
                return
            self._last_key_time[ch] = now
        self._put_event(ch)

    def _put_event(self, ch: str):
        with self._event_lock:
            try:
                self.key_queue.put_nowait(ch)
            except queue.Full:
                try:
                    self.key_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.key_queue.put_nowait(ch)
                except queue.Full:
                    pass
                self.dropped_events += 1

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

            try:
                fd = sys.stdin.fileno()
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
            try:
                self.listener.join(timeout=0.5)
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
                 seed: str = "", auto_pause_wasd: bool = True,
                 gaming_idle_timeout: float = GAMING_IDLE_TIMEOUT,
                 queue_size: int = 1024, background_color: str = "FFFFFF",
                 rank_colors=None, keepalive_hz=None, debounce_ms: float = 15,
                 wasd_threshold: int = 4, repeat_threshold: int = 5):
        self.top_k = min(max(1, top_k), 5)
        self.brightness = max(0.0, min(1.0, brightness))
        self.show_probs = show_probs
        self.show_attn = show_attn
        self.idle_timeout = idle_timeout
        self.gaming_idle_timeout = max(0.0, float(gaming_idle_timeout))
        self.background_color = background_color
        self.rank_colors = list(rank_colors or DEFAULT_RANK_COLORS)
        self.wasd_threshold = max(1, int(wasd_threshold))
        self.repeat_threshold = max(1, int(repeat_threshold))
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
        controller_args = {"profile_name": profile_name, "mock": mock}
        if keepalive_hz is not None:
            controller_args["keepalive_hz"] = float(keepalive_hz)
        self.kbd = KeyboardController(**controller_args)

        self.key_queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.rolling_buffer = list(seed) if seed else []
        self.last_type_time = time.time()
        self.is_idle = False

        try:
            self.input_reader = LowLatencyInputReader(self.key_queue)
            self.input_reader.debounce_seconds = max(0.0, float(debounce_ms)) / 1000.0
        except Exception:
            self.kbd.close()
            raise
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
                    self.input_reader._put_event(ch)
                except Exception:
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

        try:
            # Initial state: If seed provided, predict immediately; otherwise solid white backlight
            if self.rolling_buffer:
                self._update_prediction()
            else:
                self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background=self.background_color)
                if self.show_probs:
                    print("[Context: (empty)] -> Start typing to see predictions...", flush=True)
        except BaseException:
            self.cleanup()
            raise

        try:
            while self.running:
                try:
                    # BATCH QUEUE DRAINING:
                    # If the user types rapidly (e.g. 80-120 WPM burst), processing keystrokes
                    # one-by-one would introduce a backlog queue. Instead, we block for 5ms on the
                    # first character, then drain all pending strokes immediately. Forward inference
                    # and USB lighting only execute once for the newest tail character.
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
                        # GAMING AUTO-PAUSE STATE MACHINE:
                        # When playing an FPS or movement-heavy game, spamming W/A/S/D or holding
                        # strafe keys causes a language model to hallucinate random predictions,
                        # turning the keyboard into an annoying disco strobe.
                        # We detect movement patterns: >= 4 WASD keys or >= 5 identical repeats.
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

                            # Trigger gaming pause: clear context buffer and revert to solid white backlight
                            if (self.wasd_streak >= self.wasd_threshold or
                                    self.repeat_count >= self.repeat_threshold):
                                if not self.is_gaming:
                                    self.is_gaming = True
                                    self.rolling_buffer.clear()
                                    self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background=self.background_color)
                                    if self.show_probs:
                                        print("\n[Gaming Mode] WASD detected. Lighting paused.", flush=True)

                        # Once in gaming mode, ignore movement keys.
                        # Ignore game controls such as Space while paused; resume on Enter or real text.
                        if self.is_gaming:
                            if ch == "\n" or (ch.lower() not in ("w", "a", "s", "d") and ch != " " and ch in VALID_PREDICTIVE_CHARS):
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
                        self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background=self.background_color)
                        if self.show_probs:
                            print("[Context: (empty)] -> Start typing to see predictions...", flush=True)
                else:
                    self._handle_idle_iteration()

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def _handle_idle_iteration(self):
        now = time.time()
        if self.is_gaming and now - self.last_type_time > self.gaming_idle_timeout:
            self.is_gaming = False
            self.wasd_streak = 0
            self.repeat_count = 0
            if self.show_probs:
                print("[Typing Resumed] Idle timeout.", flush=True)

        if (self.idle_timeout > 0 and
                not self.is_idle and
                now - self.last_type_time > self.idle_timeout):
            self.is_idle = True
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background=self.background_color)
            if self.show_probs and self.rolling_buffer:
                ctx = "".join(self.rolling_buffer)[-16:]
                print(f"[Context: {ctx!r:<16}] -> (Idle)", flush=True)

    def _update_prediction(self):
        if not self.rolling_buffer:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background=self.background_color)
            return

        active_chars = self.rolling_buffer[-self.context_len:]
        tokens = [self.tokenizer.encode_char(c) for c in active_chars]
        _, probs = self.model.forward(tokens)

        if np.isnan(probs).any() or float(np.nanmax(probs)) < 0.02:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background=self.background_color)
            return

        valid_candidates = []
        for token_id, prob in enumerate(probs):
            ch = self.tokenizer.decode_id(token_id)
            if ch in VALID_PREDICTIVE_CHARS:
                key_name = char_to_key_name(ch)
                if key_name:
                    valid_candidates.append((prob, ch, key_name))

        if not valid_candidates:
            self.kbd.set_key_colors({}, brightness=self.brightness, clear_others=True, default_background=self.background_color)
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
            color = self.rank_colors[min(rank, len(self.rank_colors) - 1)]
            key_colors[key_name] = color
            best_prob, best_ch = key_to_best_char[key_name]
            total_prob = key_to_prob[key_name]
            label = format_prediction_label(best_ch)
            log_items.append(f"#{rank+1}: {label} ({total_prob*100:.1f}%)")

        self.kbd.set_key_colors(key_colors, brightness=self.brightness, clear_others=True, default_background=self.background_color)

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
    parser.add_argument("--install-startup", action="store_true", help="Start this live predictor automatically when you sign in to Windows")
    parser.add_argument("--uninstall-startup", action="store_true", help="Remove the automatic Windows startup entry")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Runtime JSON configuration path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override config checkpoint path")
    parser.add_argument("--profile", type=str, default=None, help="Override config keyboard profile")
    parser.add_argument("--top-k", type=int, default=None, help="Override config prediction count (1-5)")
    parser.add_argument("--brightness", type=float, default=None, help="Override config LED brightness (0.0 to 1.0)")
    parser.add_argument("--context-len", type=int, default=None, help="Override config context sequence length")
    parser.add_argument("--show-probs", dest="show_probs", action="store_true", default=None, help="Print live top predictions to terminal")
    parser.add_argument("--no-show-probs", dest="show_probs", action="store_false", help="Disable probability output for this run")
    parser.add_argument("--show-attn", dest="show_attn", action="store_true", default=None, help="Visualize causal attention matrix")
    parser.add_argument("--no-show-attn", dest="show_attn", action="store_false", help="Disable attention output for this run")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no physical keyboard required)")
    parser.add_argument("--idle-timeout", type=float, default=None, help="Override config idle timeout (0 disables dimming)")
    parser.add_argument("--demo", dest="demo", action="store_true", default=None, help="Run automated typing demo showing predictions live")
    parser.add_argument("--no-demo", dest="demo", action="store_false", help="Disable demo mode for this run")
    parser.add_argument("--seed", type=str, default=None, help="Initial text prompt to seed predictions")
    parser.add_argument("--auto-pause-wasd", dest="auto_pause_wasd", action="store_true", default=None, help="Enable automatic pause on WASD movement / key spam")
    parser.add_argument("--no-auto-pause-wasd", dest="auto_pause_wasd", action="store_false", help="Disable automatic pause on WASD movement / key spam")
    args = parser.parse_args()

    if args.install_startup or args.uninstall_startup:
        if args.install_startup and args.uninstall_startup:
            parser.error("--install-startup and --uninstall-startup cannot be used together")
        configure_windows_startup(install=args.install_startup)
        return

    config = load_runtime_config(args.config)
    runtime = config["runtime"]
    lighting = config["lighting"]
    hardware = config["hardware"]
    checkpoint_path = args.checkpoint or runtime["checkpoint"]
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), checkpoint_path)
    profile_name = args.profile or runtime["profile"]
    top_k = args.top_k if args.top_k is not None else runtime["top_k"]
    brightness = args.brightness if args.brightness is not None else runtime["brightness"]
    context_len = args.context_len if args.context_len is not None else runtime["context_len"]
    idle_timeout = args.idle_timeout if args.idle_timeout is not None else runtime["idle_timeout"]
    if idle_timeout < 0:
        parser.error("--idle-timeout must be 0 or greater")

    app = PredictiveKeyLightsApp(
        checkpoint_path=checkpoint_path,
        profile_name=profile_name,
        top_k=top_k,
        brightness=brightness,
        context_len=context_len,
        show_probs=args.show_probs if args.show_probs is not None else runtime["show_probs"],
        show_attn=args.show_attn if args.show_attn is not None else runtime["show_attn"],
        mock=args.mock,
        idle_timeout=idle_timeout,
        demo=args.demo if args.demo is not None else runtime["demo"],
        seed=args.seed if args.seed is not None else runtime["seed"],
        auto_pause_wasd=args.auto_pause_wasd if args.auto_pause_wasd is not None else runtime["auto_pause_wasd"],
        gaming_idle_timeout=runtime["gaming_idle_timeout"],
        queue_size=runtime["queue_size"],
        background_color=lighting["background"],
        rank_colors=lighting["rank_colors"],
        keepalive_hz=hardware["keepalive_hz"],
        debounce_ms=runtime["debounce_ms"],
        wasd_threshold=runtime["wasd_threshold"],
        repeat_threshold=runtime["repeat_threshold"],
    )
    app.run()


if __name__ == "__main__":
    main()
