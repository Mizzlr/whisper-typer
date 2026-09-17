#!/usr/bin/env python3
"""Whisper Hotkey Daemon.

Creates a persistent uinput virtual keyboard supporting KEY_F24 and listens on a
local Unix socket for press/release triggers from Solaar. Whisper Typer watches
the virtual keyboard for push-to-talk press and release.

The dedicated keyboard is disabled in X only. Whisper Typer still reads its
evdev events directly, but desktop applications no longer receive F24 and hide
the mouse pointer as if the user had typed. If X isolation is unavailable,
disable F24 auto-repeat as a fallback against flicker. Reapply the guard at
start-up, before every press, and on an idle tick for device/keymap changes.
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

    guard_hotkey_device()

    try:
        while True:
            try:
                data, _ = server.recvfrom(64)
            except socket.timeout:
                guard_hotkey_device()
                continue
            msg = data.decode().strip()
            if msg in ("1", "press", "down"):
                # X must stop consuming this device before emitting the key;
                # Whisper Typer's direct evdev reader still receives it.
                guard_hotkey_device()
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
