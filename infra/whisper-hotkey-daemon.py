#!/usr/bin/env python3
"""Whisper Hotkey Daemon.

Creates a persistent uinput virtual keyboard supporting KEY_F24 and listens on a
local Unix socket for press/release triggers from Solaar. Whisper Typer watches
the virtual keyboard for push-to-talk press and release.

The dedicated keyboard is disabled in X only. Whisper Typer still reads its
evdev events directly, but desktop applications no longer receive F24 and hide
the mouse pointer as if the user had typed. If X isolation is unavailable,
disable F24 auto-repeat as a fallback against flicker. Reapply the guard at
start-up and on an idle tick for device/keymap changes.
"""

import os
import socket
import subprocess
import time

import evdev

from pathlib import Path
import select
import threading

SOCKET_PATH = f"/run/user/{os.getuid()}/whisper-hotkey.sock"

# X keycode for F24 on the standard evdev keymap (evdev code 194 + 8).
F24_KEYCODE = int(os.environ.get("WHISPER_F24_KEYCODE", "202"))

# How often the idle loop re-asserts the repeat guard.
GUARD_INTERVAL = 5.0


def guard_hotkey_device(device_name="whisper-gesture-keyboard") -> bool:
    """Keep the dedicated hotkey out of X; leave direct evdev readers intact."""
    if not os.environ.get("DISPLAY"):
        return False
    try:
        result = subprocess.run(
            ["xinput", "disable", device_name],
            timeout=2,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return disable_f24_repeat()


def disable_f24_repeat() -> bool:
    """Turn off X auto-repeat for the F24 keycode (idempotent)."""
    if not os.environ.get("DISPLAY"):
        return False
    try:
        result = subprocess.run(
            ["xset", "-r", str(F24_KEYCODE)],
            timeout=2,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


class GestureState:
    """Thread-safe state manager for push-to-talk press and release.

    Deduplicates events arriving concurrently from direct hidraw and the Unix socket.
    """

    def __init__(self, ui: evdev.UInput):
        self.ui = ui
        self.lock = threading.Lock()
        self.is_down = False

    def emit(self, down: bool):
        with self.lock:
            if down and not self.is_down:
                self.is_down = True
                self.ui.write(evdev.ecodes.EV_KEY, evdev.ecodes.KEY_F24, 1)
                self.ui.syn()
            elif not down and self.is_down:
                self.is_down = False
                self.ui.write(evdev.ecodes.EV_KEY, evdev.ecodes.KEY_F24, 0)
                self.ui.syn()


def find_receiver_hidraw() -> str | None:
    """Locate the Logitech Bolt receiver hidraw device node."""
    hidraw_dir = Path("/sys/class/hidraw")
    if not hidraw_dir.exists():
        return None
    for p in hidraw_dir.iterdir():
        try:
            uevent = (p / "device/uevent").read_text()
            if "0000046D:0000C548" in uevent and "input2" in uevent:
                return f"/dev/{p.name}"
        except Exception:
            pass
    return None


def hidraw_listener(state: GestureState, stop_event: threading.Event):
    """Directly monitor Logitech receiver for Mouse Gesture Button (0x00C3).

    Bypasses Solaar entirely so voice typing has sub-millisecond latency and
    never drops out even if Solaar's GUI process or listener thread freezes.
    """
    feature_index = 0x09  # Default on MX Master 3S for REPROG_CONTROLS_V4
    while not stop_event.is_set():
        dev_path = find_receiver_hidraw()
        if not dev_path or not os.path.exists(dev_path):
            time.sleep(1.0)
            continue

        try:
            fd = os.open(dev_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            time.sleep(1.0)
            continue

        try:
            while not stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 1.0)
                if not r:
                    continue
                try:
                    data = os.read(fd, 64)
                except OSError:
                    break
                if not data:
                    break

                # HID++ Long Report (0x11), Device Index 2 (MX Master 3S)
                if len(data) >= 6 and data[0] == 0x11 and data[1] == 0x02:
                    if data[2] == feature_index and data[3] == 0x00:
                        cid = (data[4] << 8) | data[5]
                        if cid == 0x00C3:  # Mouse Gesture Button pressed
                            state.emit(True)
                        elif cid == 0x0000:  # Button released
                            state.emit(False)
        except Exception:
            pass
        finally:
            state.emit(False)
            try:
                os.close(fd)
            except OSError:
                pass
            time.sleep(1.0)


def main() -> None:
    # Remove stale socket if present
    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

    # Create virtual keyboard device with KEY_A and KEY_ENTER (so whisper-typer
    # recognises it) and KEY_F24.
    cap = {
        evdev.ecodes.EV_KEY: [
            evdev.ecodes.KEY_A,
            evdev.ecodes.KEY_ENTER,
            evdev.ecodes.KEY_F24,
        ]
    }
    ui = evdev.UInput(cap, name="whisper-gesture-keyboard")
    state = GestureState(ui)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    server.settimeout(GUARD_INTERVAL)

    guard_hotkey_device()

    # Start direct hardware listener in background
    stop_event = threading.Event()
    listener_thread = threading.Thread(
        target=hidraw_listener, args=(state, stop_event), daemon=True
    )
    listener_thread.start()

    try:
        while True:
            try:
                data, _ = server.recvfrom(64)
            except socket.timeout:
                guard_hotkey_device()
                continue
            msg = data.decode().strip()
            if msg in ("1", "press", "down"):
                state.emit(True)
            elif msg in ("0", "release", "up"):
                state.emit(False)
    finally:
        stop_event.set()
        ui.close()
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    main()
