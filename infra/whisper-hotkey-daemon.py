#!/usr/bin/env python3
"""
Whisper Hotkey Daemon:
Creates a persistent uinput virtual keyboard supporting KEY_F24,
and listens on a local Unix socket for press/release triggers from Solaar.
"""
import os
import sys
import socket
import select
import evdev

SOCKET_PATH = f"/run/user/{os.getuid()}/whisper-hotkey.sock"

def main():
    # Remove stale socket if present
    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except OSError:
            pass

    # Create virtual keyboard device with KEY_A and KEY_ENTER (so whisper-typer recognises it) and KEY_F24
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

    try:
        while True:
            data, _ = server.recvfrom(64)
            msg = data.decode().strip()
            if msg in ("1", "press", "down"):
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
