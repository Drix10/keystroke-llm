"""Hardware control layer for Kreo Hive 75 and Hive 65 mechanical keyboards.

Supports both:
  - Cross-platform HID (Windows & Linux via python-hid/hidapi)
  - Direct Linux HID /dev/hidraw ioctl via SET_FEATURE (Report 6)
Includes background keep-alive to avoid firmware timeouts and automatic reconnect.
"""

import ast
import atexit
import errno
import glob
import json
import os
import sys
import tempfile
import threading
import time
from typing import Dict, Optional, Tuple

try:
    import hid
    HAS_HID = True
except ImportError:
    HAS_HID = False

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


def _ioctl_nr(dirn: int, typ: str, nr: int, size: int) -> int:
    return (dirn << 30) | (size << 16) | (ord(typ) << 8) | nr


def hid_set_feature_cmd(length: int) -> int:
    return _ioctl_nr(3, "H", 0x06, length)


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return 255, 255, 255
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"{int(r):02x}{int(g):02x}{int(b):02x}"


def scale_rgb(r: int, g: int, b: int, brightness: float) -> Tuple[int, int, int]:
    f = max(0.0, min(1.0, float(brightness)))
    return int(r * f), int(g * f), int(b * f)


def _evaluate_slot_formula(formula: str, col: int, row: int) -> int:
    allowed_binary = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    allowed_unary = (ast.UAdd, ast.USub)

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in {"col", "row"}:
            return col if node.id == "col" else row
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binary):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            return left % right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unary):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError("slot formula may only use integer arithmetic with col and row")

    return int(evaluate(ast.parse(formula, mode="eval")))


def _compute_evision_checksum(buf: bytearray) -> bytearray:
    # The EVision V2 microcontroller requires a 16-bit sum checksum across bytes 3..63,
    # stored little-endian at bytes 1 (low) and 2 (high).
    # If the checksum is off by even 1 bit, the keyboard MCU silently drops the report
    # without returning any USB error status to the host.
    chksum = sum(buf[3:64]) & 0xFFFF
    buf[1] = chksum & 0xFF
    buf[2] = (chksum >> 8) & 0xFF
    return buf


class KeyboardProfile:
    """Loads keyboard layout geometry and slot mappings from JSON."""

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.id = data["id"]
        self.name = data.get("name", self.id)
        self.usb_ids = [u.lower() for u in data["identity"]["usb_ids"]]
        self.report_id = data["identity"].get("report_id", 6)

        proto = data["protocol"]
        self.proto_mode = proto.get("mode", "sinowealth")
        self.header = bytes.fromhex(proto.get("header", "0608000001007a01"))
        self.pkt_len = proto.get("pkt_len", 520)
        self.num_slots = proto.get("num_slots", 126)
        self.color_order = proto.get("color_order", "rgb").lower()
        self.keepalive_hz = proto.get("keepalive_hz", 1.0)

        layout = data["layout"]
        formula = layout.get("slot_formula")
        self.slot_map: Dict[str, int] = {}

        for kd in layout["keys"]:
            name = kd["name"].lower()
            slot = kd.get("slot")
            if slot is None and formula:
                slot = _evaluate_slot_formula(formula, kd["col"], kd["row"])
            if slot is not None:
                self.slot_map[name] = slot

        # Common aliases for symbols and modifier keys
        aliases = {
            "=": "equal", "[": "lbracket", "]": "rbracket", "-": "minus",
            ";": "semicolon", "'": "quote", ",": "comma", ".": "period",
            "/": "slash", "\\": "backslash", "`": "grave",
            "ctrl": "lctrl", "shift": "lshift", "alt": "lalt",
            "win": "win", "windows": "win", "cmd": "win",
            "function": "fn", "delete": "del", "return": "enter",
            "esc": "esc", "escape": "esc",
        }
        for alias, target in aliases.items():
            if target in self.slot_map and alias not in self.slot_map:
                self.slot_map[alias] = self.slot_map[target]


def load_profile(profile_name: str = "hive75") -> KeyboardProfile:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
    target = os.path.join(base, f"{profile_name}.json")
    if not os.path.exists(target):
        target = os.path.join(base, "hive75.json")
    return KeyboardProfile(target)


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                kernel32.CloseHandle(h)
                return True
        except Exception:
            pass
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except (OSError, ProcessLookupError) as e:
            if getattr(e, "errno", None) == errno.EPERM:
                return True
            return False


class KeyboardController:
    """Hardware controller with dual HID/ioctl support, keep-alive, and auto-reconnect."""

    DISCONNECT_ERRNOS = tuple({
        errno.EBADF,
        errno.EIO,
        errno.ENODEV,
        errno.EPIPE,
        errno.ENXIO,
        getattr(errno, "EPROTO", 71),
        getattr(errno, "ESHUTDOWN", 108),
    })

    def __init__(self, profile_name: str = "hive75", mock: bool = False, keepalive_hz: float = 1.0):
        self.profile = load_profile(profile_name)
        self.mock = mock
        self.keepalive_hz = keepalive_hz
        self.hid_device = None
        self.dev_path: Optional[str] = None
        self.fd: Optional[int] = None
        self.is_evision = False

        self.rgb_buffer = bytearray(self.profile.num_slots * 3)
        self._current_colors: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._running = True
        self._stop_event = threading.Event()
        self._last_reconnect_attempt = 0.0
        self._keepalive_thread: Optional[threading.Thread] = None

        self.lockfile_path = os.path.join(tempfile.gettempdir(), "keystroke_llm.lock")
        self._lock_held = False
        self._acquire_lock()

        if not self.mock:
            self._connect()
        else:
            print(f"[HardwareController] Running in MOCK MODE ({self.profile.name})")

        self._keepalive_thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive_thread.start()

        self._registered_atexit = True
        atexit.register(self.close)

    def _acquire_lock(self):
        lock_contents = str(os.getpid()).encode("ascii")
        for attempt in range(3):
            try:
                fd = os.open(self.lockfile_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, lock_contents)
                finally:
                    os.close(fd)
                self._lock_held = True
                return
            except FileExistsError:
                try:
                    with open(self.lockfile_path, "r", encoding="utf-8") as f:
                        old_pid = int(f.read().strip())
                except (OSError, ValueError):
                    if attempt < 2:
                        time.sleep(0.05)
                        continue
                    raise RuntimeError("Controller lock file remained unreadable.")
                if _is_pid_running(old_pid):
                    raise RuntimeError(f"Another keyboard controller instance (PID {old_pid}) is active.")
                try:
                    os.remove(self.lockfile_path)
                except FileNotFoundError:
                    pass
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Unable to acquire controller lock: {exc}") from exc
        raise RuntimeError("Unable to acquire controller lock after retries.")

    def _release_lock(self):
        if not getattr(self, "_lock_held", False):
            return
        try:
            if os.path.exists(self.lockfile_path):
                try:
                    with open(self.lockfile_path, "r", encoding="utf-8") as f:
                        file_pid = int(f.read().strip())
                    if file_pid == os.getpid():
                        os.remove(self.lockfile_path)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._lock_held = False

    def _connect_hid(self) -> bool:
        """Attempts connection using cross-platform python-hid."""
        if not HAS_HID:
            return False
        # Clean up any existing device handle first
        if self.hid_device is not None:
            try:
                self.hid_device.close()
            except Exception:
                pass
            self.hid_device = None
        self.is_evision = False

        handle = None
        try:
            devs = hid.enumerate()
            target_path = None
            is_evision = False

            # Priority 1: Evision / Sonix vendor usage page 0xFF1C
            for d in devs:
                vid = f"{d['vendor_id']:04x}"
                pid = f"{d['product_id']:04x}"
                usb_id = f"{vid}:{pid}".lower()
                is_match = (usb_id in self.profile.usb_ids or
                            "kreo hive" in (d.get("product_string") or "").lower())
                if is_match and d.get("usage_page") == 0xFF1C:
                    target_path = d["path"]
                    is_evision = True
                    break

            # Priority 2: Standard Interface 0 (e.g. SinoWealth 258a:010c)
            if not target_path:
                for d in devs:
                    vid = f"{d['vendor_id']:04x}"
                    pid = f"{d['product_id']:04x}"
                    usb_id = f"{vid}:{pid}".lower()
                    is_match = (usb_id in self.profile.usb_ids or
                                "kreo hive" in (d.get("product_string") or "").lower())
                    if is_match and d.get("interface_number") == 0:
                        target_path = d["path"]
                        break

            if not target_path:
                for d in devs:
                    if "kreo hive" in (d.get("product_string") or "").lower():
                        target_path = d["path"]
                        break

            if target_path:
                handle = hid.device()
                handle.open_path(target_path)
                if not self._running or self._stop_event.is_set():
                    handle.close()
                    return False
                self.hid_device = handle
                handle = None
                self.is_evision = is_evision
                mode_desc = "Evision 0xFF1C (64B chunks)" if is_evision else "SinoWealth (Feature Reports)"
                print(f"[HardwareController] Connected via USB HID to {self.profile.name} [{mode_desc}]")
                return True
        except Exception as e:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
            print(f"[HardwareController] HID connection attempt: {e}")
        return False

    def _connect_linux_hidraw(self) -> bool:
        """Attempts Linux /dev/hidraw character device connection."""
        if not HAS_FCNTL or os.name != "posix":
            return False
        if self.fd is not None:
            old_fd = self.fd
            try:
                os.close(old_fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    return False
            self.fd = None

        try:
            desc_marker = bytes([0x85, self.profile.report_id])
            tokens = [f"V0000{u.split(':')[0]}P0000{u.split(':')[1]}".upper() for u in self.profile.usb_ids]

            for path in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
                try:
                    with open(os.path.join(path, "device", "uevent"), "r", errors="ignore") as f:
                        uevent = f.read().upper()
                    with open(os.path.join(path, "device", "report_descriptor"), "rb") as f:
                        desc = f.read()
                    if any(t in uevent for t in tokens) and desc_marker in desc:
                        dev = "/dev/" + os.path.basename(path)
                        self.dev_path = dev
                        fd = os.open(self.dev_path, os.O_RDWR)
                        if not self._running or self._stop_event.is_set():
                            os.close(fd)
                            return False
                        self.fd = fd
                        self.is_evision = False
                        print(f"[HardwareController] Connected via hidraw to {self.profile.name} at {self.dev_path}")
                        return True
                except OSError:
                    continue
        except PermissionError:
            print("\n[Error] Permission denied on /dev/hidraw. Run:", file=sys.stderr)
            print("  sudo cp 60-keyboardrgb.rules /etc/udev/rules.d/", file=sys.stderr)
            print("  sudo udevadm control --reload && sudo udevadm trigger\n", file=sys.stderr)
        except Exception as e:
            print(f"[HardwareController] Linux hidraw failed: {e}")
        return False

    def _connect(self):
        with self._connection_lock:
            if not self._running or self._stop_event.is_set():
                return
            self._connect_locked()

    def _connect_locked(self):
        # 1. Try HID first (works on Windows, macOS, Linux with python-hid)
        if self._connect_hid():
            return

        # 2. Try Linux /dev/hidraw ioctl
        self.is_evision = False
        if self._connect_linux_hidraw():
            return

        print(f"[HardwareController] Physical keyboard not detected ({self.profile.usb_ids}). Operating without active hardware connection (retrying in background).")

    def _send_evision_frame(self, read_ack: bool = True):
        """Streams dynamic RGB colors using EVision CMD 0x12."""
        if self.hid_device is None:
            return

        try:
            buf_len = len(self.rgb_buffer)
            idx = 0
            while idx < buf_len:
                # 64-byte HID report: 8 header bytes + up to 56 RGB payload bytes
                pktsz = min(buf_len - idx, 56)
                chunk = self.rgb_buffer[idx : idx + pktsz]
                pkt = bytearray(64)
                pkt[0] = 0x04  # Report ID
                pkt[3] = 0x12  # EVISION_V2_CMD_SEND_DYNAMIC_COLORS
                pkt[4] = pktsz
                pkt[5] = idx & 0xFF        # Buffer byte offset (low)
                pkt[6] = (idx >> 8) & 0xFF # Buffer byte offset (high)
                pkt[7] = 0x00
                pkt[8 : 8 + pktsz] = chunk
                pkt = _compute_evision_checksum(pkt)
                self.hid_device.write(list(pkt))

                # CRITICAL BUG FIX: Draining the ACK packet
                # The EVision firmware replies with a 64-byte ACK packet for every chunk.
                # If we don't read and drain this packet, the OS USB pipe buffer chokes after ~1 minute
                # of rapid writes, causing write latency to spike from 1ms to 200ms+ or throwing EPIPE.
                if read_ack:
                    self.hid_device.read(64, 20)  # Read ACK report within 20ms timeout
                idx += pktsz
        except Exception:
            # THREAD SAFETY FIX: Never attempt inline reconnect while holding self._lock.
            # In early versions, calling _connect() here created a lock inversion with the
            # keepalive thread. Instead, cleanly close the stale handle and let the keepalive
            # thread handle reconnection sequentially.
            handle = self.hid_device
            self.hid_device = None
            self.is_evision = False
            try:
                handle.close()
            except Exception:
                pass

    def _flush_frame(self, read_ack: bool = True):
        if self.is_evision:
            self._send_evision_frame(read_ack=read_ack)
            return

        pkt = bytearray(self.profile.pkt_len)
        pkt[:len(self.profile.header)] = self.profile.header
        off = len(self.profile.header)
        pkt[off : off + len(self.rgb_buffer)] = self.rgb_buffer

        # Path A: python-hid
        if self.hid_device is not None:
            try:
                self.hid_device.send_feature_report(pkt)
                return
            except Exception:
                # Mark disconnected — keepalive thread owns reconnection to avoid lock races.
                handle = self.hid_device
                self.hid_device = None
                self.is_evision = False
                try:
                    handle.close()
                except Exception:
                    pass

        # Path B: Linux /dev/hidraw
        if self.fd is not None and HAS_FCNTL:
            try:
                fcntl.ioctl(self.fd, hid_set_feature_cmd(len(pkt)), pkt, True)
            except OSError as e:
                if e.errno in self.DISCONNECT_ERRNOS:
                    try:
                        os.close(self.fd)
                    except OSError:
                        pass
                    self.fd = None
                else:
                    raise

    def _keepalive_loop(self):
        # The EVision microcontroller has an internal watchdog timer. If dynamic color reports
        # stop arriving for ~1000ms, the MCU assumes host software died and drops back to factory
        # rainbow breathing mode. Streaming at 10 Hz (100ms interval) keeps the MCU locked in
        # dynamic host-controlled lighting mode without overloading the USB bus.
        while self._running and not self._stop_event.is_set():
            if self.keepalive_hz <= 0:
                if self._stop_event.wait(1.0):
                    break
                continue
            if self.is_evision:
                interval = 0.1  # 10 Hz keepalive for EVision V2
            else:
                interval = 1.0 / self.keepalive_hz
            if self._stop_event.wait(interval):
                break
            if not self._running or self._stop_event.is_set():
                break

            if not self.mock:
                if self.hid_device is not None or self.fd is not None:
                    with self._lock:
                        try:
                            self._flush_frame(read_ack=True)
                        except Exception:
                            pass
                else:
                    # Single reconnection owner: retry connection sequentially every 2.0s
                    # without racing against worker threads.
                    now = time.time()
                    if now - self._last_reconnect_attempt > 2.0:
                        self._last_reconnect_attempt = now
                        self._connect()

    def set_key_colors(self, key_colors: Dict[str, str], brightness: float = 1.0, clear_others: bool = True, default_background: Optional[str] = "ffffff"):
        """Sets RGB colors for specified keys.
        
        Args:
            key_colors: Mapping from key name (e.g. 'w', 'space') to hex color (e.g. 'ff0000').
            brightness: Overall brightness multiplier (0.0 to 1.0).
            clear_others: If True, resets unspecified keys to default_background (or off).
            default_background: Hex color for unhighlighted keys (default 'ffffff' white, or None for black/off).
        """
        with self._lock:
            self._current_colors = dict(key_colors)
            if clear_others:
                if default_background:
                    bg_r, bg_g, bg_b = hex_to_rgb(default_background)
                    bg_r, bg_g, bg_b = scale_rgb(bg_r, bg_g, bg_b, brightness)
                    # Fill all 128 hardware slots unconditionally. In early builds, only keys
                    # explicitly mapped in slot_map were filled, leaving unmapped hardware slots
                    # (like Del, PgUp, right modifiers) dark and patchy.
                    self.rgb_buffer = bytearray([bg_r, bg_g, bg_b] * self.profile.num_slots)
                else:
                    self.rgb_buffer = bytearray(self.profile.num_slots * 3)

            for key_name, hex_code in key_colors.items():
                name = key_name.lower()
                slot = self.profile.slot_map.get(name)
                if slot is not None and slot < self.profile.num_slots:
                    r, g, b = hex_to_rgb(hex_code)
                    r, g, b = scale_rgb(r, g, b, brightness)
                    idx = slot * 3
                    self.rgb_buffer[idx : idx + 3] = bytes([r, g, b])

            if not self.mock and (self.hid_device is not None or self.fd is not None):
                try:
                    self._flush_frame()
                except Exception:
                    pass
            else:
                lit = [f"[{k}: #{v}]" for k, v in key_colors.items()]
                text = " ".join(lit) if lit else "(all white)"
                sys.stdout.write(f"\r\033[K[MOCK LED] Brightness {brightness*100:.0f}%: {text}")
                sys.stdout.flush()

    def clear(self):
        self.set_key_colors({}, brightness=0.0, clear_others=True, default_background=None)

    def restore_default_mode(self):
        """Restores the keyboard to its default breathing rainbow animation."""
        if self.is_evision and self.hid_device is not None:
            try:
                param = [0x01, 0x04, 0x03, 0x00, 0x00, 0xFF, 0xFF, 0xFF]
                pkt = bytearray([0x04, 0x00, 0x00, 0x06, 8, 0, 0, 0] + param + [0] * 48)
                pkt = _compute_evision_checksum(pkt)
                self.hid_device.write(list(pkt))
            except Exception:
                pass

    def close(self):
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        if getattr(self, "_registered_atexit", False):
            try:
                atexit.unregister(self.close)
            except Exception:
                pass
            self._registered_atexit = False

        forced_abort = False
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=1.0)
            if self._keepalive_thread.is_alive():
                forced_abort = True
                self._abort_transports()
                self._keepalive_thread.join(timeout=1.0)

        with self._connection_lock:
            if not forced_abort:
                self._restore_default_mode_bounded()
            self._abort_transports_locked()

        self._release_lock()
        print("\n[HardwareController] Closed (restored default lighting).")

    def _restore_default_mode_bounded(self):
        restore_thread = threading.Thread(target=self.restore_default_mode, daemon=True)
        restore_thread.start()
        restore_thread.join(timeout=0.25)

    def _abort_transports(self):
        with self._connection_lock:
            self._abort_transports_locked()

    def _abort_transports_locked(self):
        """Close transport handles; caller must hold _connection_lock."""
        handle = self.hid_device
        self.hid_device = None
        self.is_evision = False
        fd = self.fd
        self.fd = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


_GLOBAL_CONTROLLER: Optional[KeyboardController] = None

def set_key_colors(dict_of_key_to_hex: Dict[str, str], brightness: float = 1.0):
    global _GLOBAL_CONTROLLER
    if _GLOBAL_CONTROLLER is None:
        _GLOBAL_CONTROLLER = KeyboardController()
    _GLOBAL_CONTROLLER.set_key_colors(dict_of_key_to_hex, brightness=brightness, clear_others=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Kreo Hive Keyboard RGB Lighting")
    parser.add_argument("--profile", type=str, default="hive75", help="Keyboard profile (default: hive75)")
    parser.add_argument("--color", type=str, default=None, help="Set all keys to hex color (e.g. ff0000)")
    parser.add_argument("--key", nargs="+", help="Pairs of key name and hex color (e.g. --key a ff0000 space 00ff00)")
    parser.add_argument("--brightness", type=float, default=1.0, help="Brightness scale (0.0 to 1.0)")
    parser.add_argument("--mock", action="store_true", help="Force mock mode")
    args = parser.parse_args()

    ctrl = KeyboardController(profile_name=args.profile, mock=args.mock)
    print(f"Testing {args.profile} RGB (Press Ctrl+C to stop)...")

    try:
        if args.color:
            colors = {k: args.color for k in ctrl.profile.slot_map.keys()}
            ctrl.set_key_colors(colors, brightness=args.brightness)
        elif args.key:
            pairs = {args.key[i]: args.key[i + 1] for i in range(0, len(args.key) - 1, 2)}
            ctrl.set_key_colors(pairs, brightness=args.brightness, default_background="ffffff")
        else:
            print("Lighting WASD + Space in red (#FF0000) over white (#FFFFFF)...")
            ctrl.set_key_colors({
                "w": "ff0000", "a": "ff0000", "s": "ff0000", "d": "ff0000", "space": "ff0000"
            }, brightness=args.brightness, clear_others=True, default_background="ffffff")

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.close()
