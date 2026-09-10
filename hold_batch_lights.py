"""Script to hold keyboard LED configurations live indefinitely by keys or by slots."""

import argparse
import signal
import sys
import time
from hardware_controller import KeyboardController

def main():
    parser = argparse.ArgumentParser(description="Hold keyboard LEDs live indefinitely.")
    parser.add_argument("--slots", type=str, default="",
                        help="Comma-separated slot indices (0..127) to highlight in RED")
    parser.add_argument("--keys", type=str, default="",
                        help="Comma-separated key names to highlight in RED")
    parser.add_argument("--color", type=str, default="FF0000",
                        help="Hex color for highlighted keys (default: FF0000 vivid red)")
    parser.add_argument("--bg", type=str, default="ffffff",
                        help="Hex color for background keys (default: ffffff white, or off)")
    args = parser.parse_args()

    bg_color = None if args.bg.lower() in ("off", "none", "000000") else args.bg.lower()

    kbd = KeyboardController(profile_name="hive75", mock=False)

    if args.slots:
        slot_list = [int(s.strip()) for s in args.slots.split(",") if s.strip()]
        print("=" * 60)
        print(f"[LIVE SLOT TESTER] Highlighting SLOTS in RED (#{args.color}): {slot_list}")
        print(f"  Background: {'OFF' if not bg_color else f'WHITE (#{bg_color})'}")
        print("=" * 60)
        
        # Directly build the rgb_buffer
        bg_r, bg_g, bg_b = (255, 255, 255) if bg_color else (0, 0, 0)
        buf = bytearray([bg_r, bg_g, bg_b] * kbd.profile.num_slots)
        
        r, g, b = (255, 0, 0)  # Red
        for s in slot_list:
            if 0 <= s < kbd.profile.num_slots:
                buf[s*3 : s*3+3] = bytes([r, g, b])
                
        with kbd._lock:
            kbd.rgb_buffer = bytearray(buf)
            kbd._flush_frame(read_ack=True)
    else:
        target_keys = [k.strip().lower() for k in args.keys.split(",") if k.strip()]
        print("=" * 60)
        print(f"[LIVE KEY TESTER] Highlighting KEYS in RED: {target_keys}")
        print("=" * 60)
        colors = {k: args.color for k in target_keys}
        kbd.set_key_colors(colors, brightness=1.0, clear_others=True, default_background=bg_color)

    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("\n>>> KEYBOARD IS NOW LIT AND HOLDING LIVE INDEFINITELY. <<<")
    print("Look at your physical keyboard right now. You can chat while this stays lit!\n")
    sys.stdout.flush()

    try:
        while running:
            if args.slots:
                with kbd._lock:
                    kbd.rgb_buffer = bytearray(buf)
                    kbd._flush_frame(read_ack=True)
            else:
                kbd.set_key_colors(colors, brightness=1.0, clear_others=True, default_background=bg_color)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        kbd.close()
        print("[Done] Closed.")

if __name__ == "__main__":
    main()
