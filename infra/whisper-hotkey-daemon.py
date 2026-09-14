#!/usr/bin/env python3
"""Whisper Hotkey Daemon.

Creates a persistent uinput virtual keyboard supporting KEY_F24 and listens on a
local Unix socket for press/release triggers from Solaar. Whisper Typer watches
the virtual keyboard for push-to-talk press and release.

It also keeps X's auto-repeat disabled for the F24 keycode. X re-enables repeat
whenever the keymap is reloaded (a layout switch, `setxkbmap`, or Input Remapper
refreshing its mapping), and once it is back on, holding the gesture button
repeats F24 roughly 30 times a second, which shows up as flicker in the focused
application. So the guard is applied at start-up, before every press, and on an
idle tick.
"""

import os
import socket
import subprocess
import time

import evdev

SOCKET_PATH = f"/run/user/{os.getuid()}/whisper-hotkey.sock"

# X keycode for F24 on the standard evdev keymap (evdev code 194 + 8).
F24_KEYCODE = int(os.environ.get("WHISPER_F24_KEYCODE", "202"))

# How often the idle loop re-asserts the repeat guard.
GUARD_INTERVAL = 5.0


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

    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o600)
    server.settimeout(GUARD_INTERVAL)

    disable_f24_repeat()

    try:
        while True:
            try:
                data, _ = server.recvfrom(64)
            except socket.timeout:
                disable_f24_repeat()
                continue
            msg = data.decode().strip()
            if msg in ("1", "press", "down"):
                # Re-assert before the hold starts: this is the moment the
                # repeat would become audible/visible in the focused app.
                disable_f24_repeat()
                ui.write(evdev.ecodes.EV_KEY, evdev.ecodes.KEY_F24, 1)
                ui.syn()
            elif msg in ("0", "release", "up"):
                ui.write(evdev.ecodes.EV_KEY, evdev.ecodes.KEY_F24, 0)
                ui.syn()
    finally:
        ui.close()
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    main()
