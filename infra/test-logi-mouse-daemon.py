#!/usr/bin/env python3
"""Integration test for logi-mouse-daemon using synthetic uinput devices.

Creates a fake "mouse", runs the daemon against it, and checks the translated
events that come out of the daemon's own uinput mouse. Nothing here touches the
real receiver, so it is safe to run while you work.

The synthetic sink is disabled in X as soon as it appears, so its button events
cannot leak into the session.

Usage: infra/test-logi-mouse-daemon.py [--bin PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import evdev
from evdev import ecodes as e

SOURCE_NAME = "logi-mouse-test-source"
SINK_NAME = "logi-mouse-test-sink"
RUN_DIR = f"/run/user/{os.getuid()}"


class Failure(Exception):
    pass


def wait_for_device(name: str, timeout: float = 5.0) -> evdev.InputDevice:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            if dev.name == name:
                return dev
        time.sleep(0.05)
    raise Failure(f"device {name!r} never appeared")


def drain(dev: evdev.InputDevice, timeout: float = 0.6) -> list[tuple[int, int, int]]:
    """Collect EV_KEY/EV_REL events from a device until it goes quiet."""
    out: list[tuple[int, int, int]] = []
    while True:
        ready, _, _ = select.select([dev.fd], [], [], timeout)
        if not ready:
            return out
        for event in dev.read():
            if event.type in (e.EV_KEY, e.EV_REL):
                out.append((event.type, event.code, event.value))
        timeout = 0.15


def send(src: evdev.UInput, *events: tuple[int, int, int]) -> None:
    for type_, code, value in events:
        src.write(type_, code, value)
    src.syn()


def expect(label: str, got: list[tuple[int, int, int]], want: list[tuple[int, int, int]]) -> None:
    if got != want:
        raise Failure(f"{label}: got {got}, want {want}")
    print(f"  [PASS] {label}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default=None, help="path to logi-mouse-daemon")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    binary = args.bin or os.path.join(repo, "target", "debug", "logi-mouse-daemon")
    if not os.path.exists(binary):
        raise Failure(f"binary not found: {binary} (run cargo build --bin logi-mouse-daemon)")

    tmp = tempfile.mkdtemp(prefix="logi-mouse-test-")
    sock_path = os.path.join(RUN_DIR, f"logi-mouse-test-{os.getpid()}.sock")
    journal_dir = os.path.join(tmp, "journal")
    proc: subprocess.Popen | None = None
    src: evdev.UInput | None = None
    sink: evdev.InputDevice | None = None

    caps = {
        e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_SIDE, e.BTN_EXTRA],
        e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL],
    }

    try:
        src = evdev.UInput(caps, name=SOURCE_NAME)
        source_path = wait_for_device(SOURCE_NAME).path
        print(f"synthetic source: {source_path}")

        proc = subprocess.Popen(
            [
                binary,
                "--source-path", source_path,
                "--sink-name", SINK_NAME,
                "--socket-path", sock_path,
                "--log-dir", journal_dir,
                "--paste-keycode", "52",
                "--no-x-hygiene",
                "--swap-timeout", "5",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        sink = wait_for_device(SINK_NAME)
        # Keep the synthetic pointer out of the real session.
        subprocess.run(["xinput", "disable", SINK_NAME], capture_output=True)
        print(f"synthetic sink: {sink.path} (disabled in X)")

        expect(
            "Back press/release -> KEY_ENTER",
            (send(src, (e.EV_KEY, e.BTN_SIDE, 1)), drain(sink))[1],
            [(e.EV_KEY, e.KEY_ENTER, 1)],
        )
        send(src, (e.EV_KEY, e.BTN_SIDE, 0))
        expect("Back release -> KEY_ENTER up", drain(sink), [(e.EV_KEY, e.KEY_ENTER, 0)])

        send(src, (e.EV_KEY, e.BTN_EXTRA, 1))
        expect(
            "Forward press -> Ctrl + paste key",
            drain(sink),
            [(e.EV_KEY, e.KEY_LEFTCTRL, 1), (e.EV_KEY, 52, 1)],
        )
        send(src, (e.EV_KEY, e.BTN_EXTRA, 0))
        expect(
            "Forward release -> paste up + Ctrl up",
            drain(sink),
            [(e.EV_KEY, 52, 0), (e.EV_KEY, e.KEY_LEFTCTRL, 0)],
        )

        send(src, (e.EV_KEY, e.BTN_LEFT, 1))
        expect("Left passes through", drain(sink), [(e.EV_KEY, e.BTN_LEFT, 1)])
        send(src, (e.EV_KEY, e.BTN_LEFT, 0))
        drain(sink)

        send(src, (e.EV_REL, e.REL_WHEEL, 1))
        expect("Wheel passes through", drain(sink), [(e.EV_REL, e.REL_WHEEL, 1)])

        control = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        control.sendto(b"swap-on", sock_path)
        time.sleep(0.3)
        send(src, (e.EV_KEY, e.BTN_LEFT, 1))
        expect("Left swapped to right during screenshot mode", drain(sink), [(e.EV_KEY, e.BTN_RIGHT, 1)])
        send(src, (e.EV_KEY, e.BTN_LEFT, 0))
        expect("Swapped release matches", drain(sink), [(e.EV_KEY, e.BTN_RIGHT, 0)])

        control.sendto(b"swap-off", sock_path)
        time.sleep(0.3)
        send(src, (e.EV_KEY, e.BTN_LEFT, 1))
        expect("Left restored after swap-off", drain(sink), [(e.EV_KEY, e.BTN_LEFT, 1)])
        send(src, (e.EV_KEY, e.BTN_LEFT, 0))
        drain(sink)
        control.close()

        # Journal must contain the remaps for later forensic use.
        records = []
        for name in os.listdir(journal_dir):
            with open(os.path.join(journal_dir, name)) as handle:
                records += [json.loads(line) for line in handle if line.strip()]
        remaps = [r for r in records if r.get("action") == "remap"]
        notes = [r for r in records if r.get("kind") == "state"]
        if not remaps:
            raise Failure("journal recorded no remap entries")
        if not any("swap" in str(n.get("msg", "")) for n in notes):
            raise Failure("journal recorded no swap state changes")
        print(f"  [PASS] journal has {len(remaps)} remap entries and {len(notes)} state notes")

        return 0
    finally:
        if sink is not None:
            sink.close()
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if proc.returncode not in (0, -15):
                print(proc.stdout.read() if proc.stdout else "", file=sys.stderr)
        if src is not None:
            src.close()
        for path in (sock_path,):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failure as err:
        print(f"FAILED: {err}", file=sys.stderr)
        sys.exit(1)
