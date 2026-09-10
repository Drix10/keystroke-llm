"""Script to hold keyboard LED configurations with color-coded slots."""

import argparse
import signal
import sys
import time
from hardware_controller import KeyboardController

def main():
    parser = argparse.ArgumentParser(description="Color-coded slot tester.")
    parser.add_argument("--slots", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--bg", type=str, default="off")
    args = parser.parse_args()

    slot_list = [int(s.strip()) for s in args.slots.split(",") if s.strip()]
    bg_r, bg_g, bg_b = (0, 0, 0) if args.bg.lower() in ("off", "none", "000000") else (255, 255, 255)

    kbd = KeyboardController(profile_name="hive75", mock=False)
    buf = bytearray([bg_r, bg_g, bg_b] * kbd.profile.num_slots)

    # Alternate Red (255,0,0) on even slots, Blue (0,0,255) on odd slots
    color_map = {}
    for s in slot_list:
        if s % 2 == 0:
            color_name = "RED"
            rgb = (255, 0, 0)
        else:
            color_name = "BLUE"
            rgb = (0, 100, 255)
        color_map[s] = color_name
        if 0 <= s < kbd.profile.num_slots:
            buf[s*3 : s*3+3] = bytes(rgb)

    print("=" * 60)
    print(f"[ALTERNATING RED / BLUE SLOT TEST]")
    print(f"  Even slots { [s for s in slot_list if s % 2 == 0] } -> RED")
    print(f"  Odd  slots { [s for s in slot_list if s % 2 != 0] } -> BLUE")
    print(f"  Background: {'OFF (dark)' if bg_r == 0 else 'WHITE'}")
    print("=" * 60)

    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while running:
            with kbd._lock:
                kbd.rgb_buffer = bytearray(buf)
                kbd._flush_frame(read_ack=True)
            time.sleep(1.0)
    finally:
        kbd.close()

if __name__ == "__main__":
    main()
