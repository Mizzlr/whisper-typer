#!/usr/bin/env python3
import os
import sys
import socket

if len(sys.argv) < 2:
    sys.exit(1)

cmd = sys.argv[1].encode()
sock_path = f"/run/user/{os.getuid()}/whisper-hotkey.sock"

try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.sendto(cmd, sock_path)
    s.close()
except Exception:
    pass
