"""
predict_and_light.py - Real-Time Keystroke Inference and Keyboard Lighting
==========================================================================
Captures live user keystrokes, runs the educational TinyTransformer model,
predicts the top-K most likely next characters, and dynamically illuminates
the corresponding physical keys on the Kreo Hive keyboard over wired USB.

Features:
  - Asynchronous non-blocking architecture (never blocks or delays typing)
  - Color rank mapping (Rank 1: Cyan, Rank 2: Emerald, Rank 3: Amber, etc.)
  - Attention matrix visualization (ASCII heatmap of context weights)
  - Seamless recovery on USB drop / reconnect
  - Mock mode for development without hardware

Usage:
  python predict_and_light.py --checkpoint checkpoints/model_final.npz --profile hive75 --top-k 5
  python predict_and_light.py --mock --show-probs --show-attn
"""

import argparse
import os
import queue
import sys
import threading
import time
from typing import List, Tuple

import numpy as np

from hardware_controller import KeyboardController
from key_mapper import CharTokenizer, char_to_key_name, get_default_vocab
from model import TinyTransformer

# Vibrant color palette for prediction ranks (Rank 1 -> Rank 5)
RANK_COLORS = [
    "00FFFF",  # Rank 1: Electric Cyan
    "00FF66",  # Rank 2: Vibrant Emerald
    "FFD700",  # Rank 3: Golden Yellow
    "FF007F",  # Rank 4: Vivid Magenta
    "7F00FF",  # Rank 5: Deep Violet
]


def render_attention_matrix(tokens: List[str], attn_weights: np.ndarray):
    """
    Renders an educational ASCII heatmap of the causal attention matrix.
    attn_weights shape: [T, T]
    """
    T = len(tokens)
    print("\n--- Attention Weights Matrix (Causal Focus) ---")
    
    # Print column header
    header = "     " + " ".join([f"{repr(t)[1:-1]:>3}" for t in tokens])
    print(header)
    print("    " + "-" * (len(header) - 4))
    
    # Gradient shades
    shades = " .:-=+*#%@"
    
    for i in range(T):
        row_str = f"{repr(tokens[i])[1:-1]:>3} |"
        for j in range(T):
            if j > i:
                row_str += "    "  # Causal mask (future)
            else:
                val = attn_weights[i, j]
                shade_idx = min(len(shades) - 1, int(val * len(shades)))
                row_str += f"  {shades[shade_idx]} "
        print(row_str)
        
    # Last token focus (what influenced the current prediction)
    last_row = attn_weights[-1, :]
    print("\nLast Character Attention Focus:")
    focus_items = []
    for t, weight in zip(tokens, last_row):
        focus_items.append(f"{repr(t)[1:-1]}: {weight*100:.1f}%")
    print("  " + " | ".join(focus_items))
    print("-" * 48 + "\n")


class LowLatencyInputReader:
    """
    Cross-platform, low-latency keystroke capturer.
    Uses pynput or termios on Linux, msvcrt on Windows, with fallback stdin.
    """
    def __init__(self, key_queue: queue.Queue):
        self.key_queue = key_queue
        self.running = True
        self.thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.thread.start()

    def _reader_worker(self):
        # 1. Try pynput for system-wide background hook
        try:
            from pynput import keyboard

            def on_press(key):
                if not self.running:
                    return False
                try:
                    if hasattr(key, 'char') and key.char is not None:
                        self.key_queue.put(key.char)
                    elif key == keyboard.Key.space:
                        self.key_queue.put(' ')
                    elif key == keyboard.Key.enter:
                        self.key_queue.put('\n')
                    elif key == keyboard.Key.tab:
                        self.key_queue.put('\t')
                    elif key == keyboard.Key.backspace:
                        self.key_queue.put('\b')
                except Exception:
                    pass

            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()
                return
        except (ImportError, Exception):
            pass

        # 2. Linux termios fallback (terminal raw mode)
        if os.name == "posix":
            import select
            import termios
            import tty
            
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self.running:
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch:
                            self.key_queue.put(ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return

        # 3. Windows msvcrt fallback
        if os.name == "nt":
            import msvcrt
            while self.running:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    try:
                        decoded = ch.decode("utf-8")
                        self.key_queue.put(decoded)
                    except Exception:
                        pass
                else:
                    time.sleep(0.01)
            return

    def stop(self):
        self.running = False


class PredictiveKeyLightsApp:
    """
    Main controller coordinating input capture, transformer inference,
    and keyboard RGB hardware updates.
    """
    def __init__(self, checkpoint_path: str, profile_name: str = "hive75",
                 top_k: int = 5, brightness: float = 0.8,
                 context_len: int = 12, show_probs: bool = False,
                 show_attn: bool = False, mock: bool = False,
                 idle_timeout: float = 8.0):
        self.top_k = top_k
        self.brightness = brightness
        self.context_len = context_len
        self.show_probs = show_probs
        self.show_attn = show_attn
        self.idle_timeout = idle_timeout
        
        # Load Tokenizer & Model
        print(f"[Init] Loading model checkpoint from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_path} not found! Please run train.py first to create a model."
            )
            
        self.model = TinyTransformer.load_checkpoint(checkpoint_path)
        self.tokenizer = CharTokenizer(get_default_vocab())
        print(f"[Init] Model loaded: vocab={self.model.vocab_size}, hidden={self.model.hidden_size}, context_window={self.context_len}")
        
        # Initialize Hardware
        self.kbd = KeyboardController(profile_name=profile_name, mock=mock)
        
        # Keystroke state
        self.key_queue = queue.Queue()
        self.rolling_buffer = [" "] * self.context_len
        self.last_type_time = time.time()
        self.is_idle = False
        
        # Start Input Listener
        self.input_reader = LowLatencyInputReader(self.key_queue)
        self.running = True

    def run(self):
        """Main event loop running asynchronous inference and lighting."""
        print("\n" + "=" * 65)
        print("  PREDICTIVE KEY LIGHTS IS ACTIVE!")
        print("  Start typing in any window or in this terminal.")
        print("  Watch the physical keys light up ahead of your keystrokes!")
        print("  Press Ctrl+C to stop.")
        print("=" * 65 + "\n")
        
        # Run initial prediction on space context
        self._update_prediction()
        
        try:
            while self.running:
                # 1. Process all pending keys in queue (drain queue to prevent typing lag)
                new_keystrokes = []
                while not self.key_queue.empty():
                    try:
                        ch = self.key_queue.get_nowait()
                        new_keystrokes.append(ch)
                    except queue.Empty:
                        break
                        
                if new_keystrokes:
                    self.last_type_time = time.time()
                    self.is_idle = False
                    for ch in new_keystrokes:
                        if ch == '\b':  # Backspace handling
                            if len(self.rolling_buffer) > 0:
                                self.rolling_buffer.pop()
                                self.rolling_buffer.insert(0, " ")
                        else:
                            self.rolling_buffer.append(ch)
                            if len(self.rolling_buffer) > self.context_len:
                                self.rolling_buffer.pop(0)
                                
                    # Run inference and update lights
                    self._update_prediction()
                else:
                    # Check for idle timeout
                    if not self.is_idle and (time.time() - self.last_type_time > self.idle_timeout):
                        self.is_idle = True
                        self._handle_idle()
                        
                time.sleep(0.01)  # 10ms pacing (~100Hz tick)
                
        except KeyboardInterrupt:
            print("\n[PredictiveKeyLights] Shutting down...")
        finally:
            self.cleanup()

    def _update_prediction(self):
        """Runs the transformer forward pass and illuminates top-K keys."""
        # 1. Encode context buffer
        token_ids = [self.tokenizer.encode_char(c) for c in self.rolling_buffer]
        
        # 2. Forward pass through TinyTransformer
        _, probs = self.model.forward(token_ids)
        
        # Check for model uncertainty: if maximum probability is very low
        max_prob = float(np.max(probs))
        if max_prob < 0.02:  # Extremely diffuse distribution
            self.kbd.clear()
            return

        # 3. Select Top-K predictions
        top_indices = np.argsort(probs)[::-1][:self.top_k]
        
        key_colors = {}
        log_predictions = []
        
        for rank, token_id in enumerate(top_indices):
            char_pred = self.tokenizer.decode_id(token_id)
            prob = probs[token_id]
            
            # Map character to physical keyboard key name
            key_name = char_to_key_name(char_pred)
            
            # Pick color for this rank
            color_hex = RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]
            
            # If key exists on the keyboard, queue for lighting
            if key_name is not None:
                # Probability-weighted brightness scaling
                key_colors[key_name] = color_hex
                
            repr_str = repr(char_pred) if char_pred in (' ', '\n', '\t') else f"'{char_pred}'"
            log_predictions.append(f"#{rank+1}: {repr_str} ({key_name or 'N/A'}, {prob*100:.1f}%)")
            
        # 4. Light the physical keys
        self.kbd.set_key_colors(key_colors, brightness=self.brightness, clear_others=True)
        
        # 5. Terminal Display
        if self.show_probs:
            current_context = "".join(self.rolling_buffer)[-12:]
            sys.stdout.write(f"\r\033[K[Context: {current_context!r}] -> {', '.join(log_predictions[:3])}")
            sys.stdout.flush()
            
        if self.show_attn and self.model.last_attn_weights is not None:
            render_attention_matrix(self.rolling_buffer[-8:], self.model.last_attn_weights[-8:, -8:])

    def _handle_idle(self):
        """Dims or clears lights when user pauses typing."""
        if self.show_probs:
            sys.stdout.write("\r\033[K[IDLE] Lights dimmed during typing pause.")
            sys.stdout.flush()
        # Dim lights to 15% during idle
        self.kbd.set_key_colors(self.kbd._current_colors, brightness=0.15, clear_others=False)

    def cleanup(self):
        """Restores hardware and closes listeners."""
        self.running = False
        self.input_reader.stop()
        self.kbd.clear()
        self.kbd.close()
        print("[PredictiveKeyLights] Complete. LEDs turned off.")


def main():
    parser = argparse.ArgumentParser(description="Predictive Key Lights - Educational Keystroke Transformer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_final.npz",
                        help="Path to trained model checkpoint (.npz)")
    parser.add_argument("--profile", type=str, default="hive75", choices=["hive75", "hive65"],
                        help="Keyboard profile (hive75 or hive65)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predicted keys to illuminate (1-10)")
    parser.add_argument("--brightness", type=float, default=0.8, help="LED brightness scale (0.0 to 1.0)")
    parser.add_argument("--context-len", type=int, default=12, help="Context sequence length")
    parser.add_argument("--show-probs", action="store_true", help="Print live probabilities in the terminal")
    parser.add_argument("--show-attn", action="store_true", help="Visualize causal attention matrix")
    parser.add_argument("--mock", action="store_true", help="Force mock hardware mode (no physical keyboard required)")
    parser.add_argument("--idle-timeout", type=float, default=6.0, help="Seconds before dimming during pause")
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
