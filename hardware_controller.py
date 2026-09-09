"""
hardware_controller.py - Hardware Control Layer for Kreo Hive 75 & 65
====================================================================
Self-contained, robust, thread-safe Python driver for Kreo Hive keyboards.
Directly handles Linux HID /dev/hidraw communication with zero extra bloat.

Features:
  - Clean API: set_key_colors({"a": "ff0000", "space": "00ff00"}, brightness=0.8)
  - Background Keep-Alive daemon (1 Hz) to prevent SinoWealth MCU timeout
  - Automatic reconnection if the keyboard resets or disconnects
  - Graceful Mock Mode fallback on non-Linux systems or without hardware
  - Single-instance lockfile to prevent multi-process conflict
"""

import atexit
import glob
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

# Linux fcntl import (None on Windows/mock)
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


# -----------------------------------------------------------------------------
# 1. IOCTL and Packet Construction Helpers
# -----------------------------------------------------------------------------

def _IOC(dirn: int, typ: str, nr: int, size: int) -> int:
    return (dirn << 30) | (size << 16) | (ord(typ) << 8) | nr


def HIDIOCSFEATURE(length: int) -> int:
    """Linux HID ioctl number for SET_FEATURE (Report 6)."""
    return _IOC(3, "H", 0x06, length)


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Converts '#RRGGBB' or 'RRGGBB' string to (r, g, b) tuple 0..255."""
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join([c * 2 for c in s])
    if len(s) != 6:
        return 255, 255, 255
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Converts (r, g, b) tuple 0..255 to hex string 'rrggbb'."""
    return f"{int(r):02x}{int(g):02x}{int(b):02x}"


def scale_rgb(r: int, g: int, b: int, brightness: float) -> Tuple[int, int, int]:
    """Applies brightness scale factor (0.0 to 1.0) to RGB components."""
    factor = max(0.0, min(1.0, float(brightness)))
    return int(r * factor), int(g * factor), int(b * factor)


# -----------------------------------------------------------------------------
# 2. Keyboard Profile Model
# -----------------------------------------------------------------------------

class KeyboardProfile:
    """Loads keyboard layout and slot mapping from JSON."""
    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
            
        self.id = d["id"]
        self.name = d.get("name", self.id)
        self.usb_ids = [u.lower() for u in d["identity"]["usb_ids"]]
        self.report_id = d["identity"].get("report_id", 6)
        
        proto = d["protocol"]
        self.header = bytes.fromhex(proto["header"])
        self.pkt_len = proto.get("pkt_len", 520)
        self.num_slots = proto.get("num_slots", 126)
        self.color_order = proto.get("color_order", "rgb").lower()
        self.keepalive_hz = proto.get("keepalive_hz", 1.0)
        
        lay = d["layout"]
        self.rows = lay["rows"]
        self.cols = lay["cols"]
        formula = lay.get("slot_formula")
        
        self.slot_map: Dict[str, int] = {}
        for kd in lay["keys"]:
            name = kd["name"].lower()
            slot = kd.get("slot")
            if slot is None and formula:
                slot = int(eval(formula, {"__builtins__": {}}, {"col": kd["col"], "row": kd["row"]}))
            if slot is not None:
                self.slot_map[name] = slot


def load_profile(profile_name: str) -> KeyboardProfile:
    """Searches profiles/ directory for matching profile JSON."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
    target_path = os.path.join(base_dir, f"{profile_name}.json")
    if not os.path.exists(target_path):
        target_path = os.path.join(base_dir, "hive75.json")
    if not os.path.exists(target_path):
        target_path = os.path.join(base_dir, "hive65.json")
    return KeyboardProfile(target_path)


# -----------------------------------------------------------------------------
# 3. Main Keyboard Controller
# -----------------------------------------------------------------------------

class KeyboardController:
    """
    Thread-safe controller for Kreo Hive keyboards with automated keep-alive.
    """
    GONE_ERRNOS = (5, 19, 32, 71)  # EIO, ENODEV, EPIPE, EPROTO

    def __init__(self, profile_name: str = "hive75", mock: bool = False, keepalive_hz: float = 1.0):
        self.profile_name = profile_name
        self.profile = load_profile(profile_name)
        self.mock = mock or (not HAS_FCNTL) or (os.name != "posix")
        self.keepalive_hz = keepalive_hz
        self.dev_path: Optional[str] = None
        self.fd: Optional[int] = None
        
        # Raw RGB buffer (3 bytes per slot)
        self.rgb_buffer = bytearray(self.profile.num_slots * 3)
        self._current_colors: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._running = False
        self._keepalive_thread: Optional[threading.Thread] = None
        
        # Lockfile to warn if multiple instances run
        self.lockfile_path = "/tmp/keystroke_llm.lock" if os.name == "posix" else "keystroke_llm.lock"
        self._acquire_instance_lock()
        
        if not self.mock:
            self._connect_hardware()
        else:
            print(f"[HardwareController] Running in MOCK MODE (Profile: {self.profile.name}).")
            
        # Start background keep-alive daemon thread
        self._running = True
        self._keepalive_thread = threading.Thread(target=self._keepalive_worker, daemon=True)
        self._keepalive_thread.start()
        
        atexit.register(self.close)

    def _acquire_instance_lock(self):
        try:
            if os.path.exists(self.lockfile_path):
                print(f"[WARNING] Lockfile {self.lockfile_path} exists! Another instance may be running.", file=sys.stderr)
            with open(self.lockfile_path, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass

    def _release_instance_lock(self):
        try:
            if os.path.exists(self.lockfile_path):
                os.remove(self.lockfile_path)
        except Exception:
            pass

    def _find_device_node(self) -> Optional[str]:
        """Scans /sys/class/hidraw to locate the keyboard's Report 6 interface."""
        desc_marker = bytes([0x85, self.profile.report_id])
        usb_tokens = [f"V0000{u.split(':')[0]}P0000{u.split(':')[1]}".upper() for u in self.profile.usb_ids]
        
        for path in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
            try:
                with open(os.path.join(path, "device", "uevent"), "r", encoding="utf-8", errors="ignore") as f:
                    uevent = f.read().upper()
                with open(os.path.join(path, "device", "report_descriptor"), "rb") as f:
                    desc = f.read()
                if any(tok in uevent for tok in usb_tokens) and desc_marker in desc:
                    return "/dev/" + os.path.basename(path)
            except OSError:
                continue
        return None

    def _connect_hardware(self):
        """Discovers and opens the hidraw character device."""
        try:
            dev = self._find_device_node()
            if not dev:
                print(f"[HardwareController] Device not found on USB (IDs: {self.profile.usb_ids}). Switching to MOCK MODE.")
                self.mock = True
                return
                
            self.dev_path = dev
            self.fd = os.open(self.dev_path, os.O_RDWR)
            print(f"[HardwareController] Connected to {self.profile.name} at {self.dev_path}")
        except PermissionError:
            print("\n[ERROR] Permission denied opening /dev/hidraw node!", file=sys.stderr)
            print("Fix: Copy udev rule and reload permissions:", file=sys.stderr)
            print("  sudo cp 60-keyboardrgb.rules /etc/udev/rules.d/", file=sys.stderr)
            print("  sudo udevadm control --reload && sudo udevadm trigger\n", file=sys.stderr)
            print("[HardwareController] Falling back to MOCK MODE.", file=sys.stderr)
            self.mock = True
        except Exception as e:
            print(f"[HardwareController] Hardware connection error: {e}. Switching to MOCK MODE.", file=sys.stderr)
            self.mock = True

    def _reopen_device(self, timeout: float = 8.0) -> bool:
        """Re-attaches to the device if the firmware resets and re-enumerates."""
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            dev = self._find_device_node()
            if dev:
                try:
                    self.dev_path = dev
                    self.fd = os.open(dev, os.O_RDWR)
                    return True
                except OSError:
                    pass
            time.sleep(0.4)
        return False

    def _flush_hardware(self):
        """Builds SET_FEATURE frame and writes via ioctl."""
        if self.fd is None or not HAS_FCNTL:
            return
            
        # Build 520-byte packet
        pkt = bytearray(self.profile.pkt_len)
        pkt[:len(self.profile.header)] = self.profile.header
        off = len(self.profile.header)
        pkt[off : off + len(self.rgb_buffer)] = self.rgb_buffer
        
        try:
            fcntl.ioctl(self.fd, HIDIOCSFEATURE(len(pkt)), pkt, True)
        except OSError as e:
            if e.errno in self.GONE_ERRNOS:
                print("[HardwareController] USB disconnect / reset detected. Reconnecting...")
                if self._reopen_device():
                    print(f"[HardwareController] Reconnected to {self.dev_path}")
                    try:
                        fcntl.ioctl(self.fd, HIDIOCSFEATURE(len(pkt)), pkt, True)
                    except OSError:
                        pass
            else:
                raise

    def _keepalive_worker(self):
        """Background thread keeping colors latched on SinoWealth MCU."""
        interval = 1.0 / self.keepalive_hz if self.keepalive_hz > 0 else 1.0
        while self._running:
            time.sleep(interval)
            if not self.mock and self.fd is not None:
                with self._lock:
                    try:
                        self._flush_hardware()
                    except Exception:
                        pass

    def set_key_colors(self, key_colors: Dict[str, str], brightness: float = 1.0, clear_others: bool = True):
        """
        Sets arbitrary colors for specific keys by name.
        
        Args:
            key_colors: Mapping of key name to hex color, e.g. {"a": "ff0000", "space": "00ffff"}
            brightness: Overall brightness multiplier (0.0 to 1.0).
            clear_others: If True, all unmentioned keys are turned off.
        """
        self._current_colors = dict(key_colors)
        
        with self._lock:
            if clear_others:
                self.rgb_buffer = bytearray(self.profile.num_slots * 3)
                
            for key_name, hex_code in key_colors.items():
                name = key_name.lower()
                slot = self.profile.slot_map.get(name)
                if slot is not None and slot < self.profile.num_slots:
                    r, g, b = hex_to_rgb(hex_code)
                    r, g, b = scale_rgb(r, g, b, brightness)
                    # Wire byte order is usually RGB
                    idx = slot * 3
                    self.rgb_buffer[idx : idx + 3] = bytes([r, g, b])
                    
            if not self.mock and self.fd is not None:
                try:
                    self._flush_hardware()
                except Exception:
                    pass
            else:
                # Mock display
                lit = [f"[{k}: #{v}]" for k, v in key_colors.items()]
                lit_str = " ".join(lit) if lit else "(all off)"
                sys.stdout.write(f"\r\033[K[MOCK LED] Brightness {brightness*100:.0f}%: {lit_str}")
                sys.stdout.flush()

    def clear(self):
        """Turns off all LEDs."""
        self.set_key_colors({}, brightness=0.0, clear_others=True)

    def close(self):
        """Gracefully shuts down driver and restores keyboard."""
        if not self._running:
            return
        self._running = False
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=1.0)
        try:
            self.clear()
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
        except Exception:
            pass
        self._release_instance_lock()
        print("\n[HardwareController] Closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Module-level convenience helper
_GLOBAL_CONTROLLER: Optional[KeyboardController] = None

def set_key_colors(dict_of_key_to_hex: Dict[str, str], brightness: float = 1.0):
    """
    Convenience function:
      set_key_colors({"a": "ff0000", "space": "00ff00"}, brightness=0.8)
    """
    global _GLOBAL_CONTROLLER
    if _GLOBAL_CONTROLLER is None:
        _GLOBAL_CONTROLLER = KeyboardController()
    _GLOBAL_CONTROLLER.set_key_colors(dict_of_key_to_hex, brightness=brightness, clear_others=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Kreo Hive Keyboard RGB Lighting")
    parser.add_argument("--profile", type=str, default="hive75", choices=["hive75", "hive65"], help="Keyboard profile")
    parser.add_argument("--color", type=str, default=None, help="Set all known keys to hex color (e.g. ff0000)")
    parser.add_argument("--key", nargs="+", help="Pairs of key and hex color (e.g. --key a ff0000 space 00ff00)")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness scale (0.0 to 1.0)")
    parser.add_argument("--mock", action="store_true", help="Force mock mode")
    args = parser.parse_args()

    ctrl = KeyboardController(profile_name=args.profile, mock=args.mock)
    print(f"Testing hardware with profile {args.profile} (Press Ctrl+C to stop)...")
    
    try:
        if args.color:
            color_map = {k: args.color for k in ctrl.profile.slot_map.keys()}
            ctrl.set_key_colors(color_map, brightness=args.brightness)
        elif args.key:
            pairs = {}
            for i in range(0, len(args.key), 2):
                if i + 1 < len(args.key):
                    pairs[args.key[i]] = args.key[i + 1]
            ctrl.set_key_colors(pairs, brightness=args.brightness)
        else:
            # Default test pattern: QWERTY in Cyan, Space in Emerald, Enter in Gold
            ctrl.set_key_colors({
                "w": "00ffff", "a": "00ffff", "s": "00ffff", "d": "00ffff",
                "space": "00ff66", "enter": "ffd700"
            }, brightness=args.brightness)
            
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ctrl.close()
