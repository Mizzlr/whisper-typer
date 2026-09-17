#!/usr/bin/python3
"""X isolation and fallback checks; --device also probes a private uinput key."""
import importlib.util
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('hotkey', Path(__file__).with_name('whisper-hotkey-daemon.py'))
hotkey = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hotkey)


class GuardTests(unittest.TestCase):
    def test_no_display_needs_no_desktop_commands(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(hotkey.subprocess, 'run') as run:
            self.assertFalse(hotkey.guard_hotkey_device())
            run.assert_not_called()

    def test_successful_isolation_needs_no_repeat_fallback(self):
        with patch.dict(os.environ, DISPLAY=':test'), patch.object(hotkey.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertTrue(hotkey.guard_hotkey_device())
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], ['xinput', 'disable', 'whisper-gesture-keyboard'])

    def test_failed_isolation_preserves_repeat_guard(self):
        for failure in (subprocess.CompletedProcess([], 1), FileNotFoundError(), subprocess.TimeoutExpired('xinput', 2)):
            with self.subTest(failure=failure), patch.dict(os.environ, DISPLAY=':test'), patch.object(hotkey.subprocess, 'run', side_effect=[failure, subprocess.CompletedProcess([], 0)]) as run:
                self.assertTrue(hotkey.guard_hotkey_device())
                self.assertEqual(run.call_args.args[0], ['xset', '-r', '202'])


def probe_device():
    import evdev
    from evdev import ecodes as e
    from Xlib import display
    # No KEY_ENTER: the running dictation monitor cannot discover this fixture.
    name = f'whisper-isolation-test-{os.getpid()}'
    with evdev.UInput({e.EV_KEY: [e.KEY_A, e.KEY_Z, e.KEY_SPACE, e.KEY_F24]}, name=name) as source:
        reader = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            hotkey.guard_hotkey_device(name)
            props = subprocess.run(['xinput', 'list-props', name], capture_output=True, text=True)
            if props.returncode == 0 and 'Device Enabled' in props.stdout and props.stdout.split('Device Enabled', 1)[1].splitlines()[0].rstrip().endswith('0'):
                for path in evdev.list_devices():
                    device = evdev.InputDevice(path)
                    if device.name == name:
                        reader = device
                        break
                    device.close()
                if reader is not None: break
            time.sleep(.05)
        else:
            raise AssertionError('Synthetic keyboard was not isolated; no keys emitted.')
        d = display.Display()
        try:
            for value in (1, 0):
                source.write(e.EV_KEY, e.KEY_F24, value); source.syn()
                self_events = []
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    if not select.select([reader.fd], [], [], .05)[0]: continue
                    self_events.extend((event.code, event.value) for event in reader.read() if event.type == e.EV_KEY)
                    if (e.KEY_F24, value) in self_events: break
                assert (e.KEY_F24, value) in self_events, 'Direct evdev event was lost.'
                assert not (d.query_keymap()[202 // 8] & (1 << (202 % 8))), 'F24 leaked into X.'
        finally:
            d.close()
            reader.close()
    print('PASS: evdev receives press/release; desktop receives neither.')


if __name__ == '__main__':
    if '--device' in sys.argv:
        probe_device()
    else:
        unittest.main()
