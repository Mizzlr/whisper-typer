#!/usr/bin/python3
"""Local White Noise & Ambient Audio Engine for Whisper Typer.

Provides high-quality procedural noise (Brown, Pink, White, Rain) and
seamless natural audio sample looping (River & Rain ASMR) with real-time volume
control, sleep timer, Unix domain socket daemon, and Python/CLI clients.
"""
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import signal as dsp_signal

SOCKET_DIR = Path.home() / ".cache/whisper-typer"
SOCKET_PATH = SOCKET_DIR / "white_noise.sock"
CONFIG_DIR = Path.home() / ".config/whisper-typer"
CONFIG_PATH = CONFIG_DIR / "white_noise.json"
AMBIENT_DIR = Path.home() / ".cache/whisper-typer/ambient"

SAMPLE_RATE = 48000
BUFFER_FRAMES = 2048

PRESETS = {
    "river_rain": {
        "title": "River & Rain",
        "description": "Natural river & rain ASMR recording",
        "type": "sample",
    },
    "mellow_rain": {
        "title": "Mellow Rain",
        "description": "Original river & rain with sharp sounds cut out (tinnitus aid)",
        "type": "sample",
    },
    "brown": {
        "title": "Brown Noise",
        "description": "Deep, warm, low-frequency rumble (focus/sleep)",
        "type": "procedural",
    },
    "pink": {
        "title": "Pink Noise",
        "description": "Balanced 1/f spectrum, natural rainfall",
        "type": "procedural",
    },
    "white": {
        "title": "White Noise",
        "description": "Full-spectrum crisp masking noise",
        "type": "procedural",
    },
    "rain": {
        "title": "Gentle Rain",
        "description": "Synthetic rain shower & soothing water stream",
        "type": "procedural",
    },
    "custom": {
        "title": "Custom File",
        "description": "Local audio file with seamless crossfade",
        "type": "custom",
    },
}


def make_seamless_loop(arr: np.ndarray, crossfade_samples: int = 96000) -> np.ndarray:
    """Apply equal-power sine/cosine crossfade to create a perfectly seamless loop."""
    length = len(arr)
    crossfade = min(crossfade_samples, max(length // 4, 1024))
    loop_length = length - crossfade
    if loop_length <= 0:
        return arr.astype(np.float32)

    loop_buf = arr[:loop_length].copy()
    alpha = np.linspace(0.0, 1.0, crossfade, endpoint=False).reshape(-1, 1)
    w_start = np.sin(alpha * (np.pi / 2)).astype(np.float32)
    w_end = np.cos(alpha * (np.pi / 2)).astype(np.float32)

    loop_buf[:crossfade] = w_end * arr[loop_length : loop_length + crossfade] + w_start * arr[:crossfade]
    return loop_buf.astype(np.float32)


def generate_brown_noise(duration_sec: int = 30) -> np.ndarray:
    """Generate high-quality Brown (Brownian / red) noise."""
    n = SAMPLE_RATE * duration_sec
    white = np.random.normal(0, 1.0, (n, 2)).astype(np.float32)
    b = [0.08]
    a = [1.0, -0.994]
    brown = dsp_signal.lfilter(b, a, white, axis=0)
    max_val = np.max(np.abs(brown))
    if max_val > 0:
        brown = brown / max_val * 0.45
    return make_seamless_loop(brown.astype(np.float32), SAMPLE_RATE * 2)


def generate_pink_noise(duration_sec: int = 30) -> np.ndarray:
    """Generate high-quality Pink (1/f) noise via multi-pole IIR filter."""
    n = SAMPLE_RATE * duration_sec
    white = np.random.normal(0, 1.0, (n, 2)).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = dsp_signal.lfilter(b, a, white, axis=0)
    max_val = np.max(np.abs(pink))
    if max_val > 0:
        pink = pink / max_val * 0.4
    return make_seamless_loop(pink.astype(np.float32), SAMPLE_RATE * 2)


def generate_white_noise(duration_sec: int = 30) -> np.ndarray:
    """Generate pure Gaussian White noise."""
    n = SAMPLE_RATE * duration_sec
    white = np.random.normal(0, 0.18, (n, 2)).astype(np.float32)
    return make_seamless_loop(white, SAMPLE_RATE * 2)


def generate_rain_noise(duration_sec: int = 30) -> np.ndarray:
    """Generate synthetic gentle rain with organic water stream modulation."""
    n = SAMPLE_RATE * duration_sec
    white = np.random.normal(0, 1.0, (n, 2)).astype(np.float32)
    sos = dsp_signal.butter(4, [150, 4500], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    base_rain = dsp_signal.sosfilt(sos, white, axis=0)
    t = np.linspace(0, duration_sec, n, endpoint=False).reshape(-1, 1)
    mod = 0.75 + 0.25 * np.sin(2 * np.pi * 0.2 * t) * np.cos(2 * np.pi * 0.08 * t + 0.7)
    rain = base_rain * mod
    max_val = np.max(np.abs(rain))
    if max_val > 0:
        rain = rain / max_val * 0.4
    return make_seamless_loop(rain.astype(np.float32), SAMPLE_RATE * 2)


def load_audio_file(filepath: Path | str) -> np.ndarray:
    """Decode any audio file into float32 stereo numpy array using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(filepath),
        "-f",
        "f32le",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw, _ = proc.communicate()
    if not raw:
        raise ValueError(f"Failed to decode audio file {filepath}")
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
    return make_seamless_loop(arr, SAMPLE_RATE * 2)


class WhiteNoiseEngine:
    """Threaded audio synthesis and looping playback engine with PyAudio."""

    def __init__(self):
        self.lock = threading.Lock()
        self.playing = False
        self.preset = "river_rain"
        self.volume = 0.20
        self.current_vol = 0.0
        self.custom_file = ""
        self.timer_minutes = 0
        self.timer_end_time = 0.0
        self.stop_requested = threading.Event()
        self.audio_thread = None
        self.timer_thread = None
        self.buffers = {}
        self.load_settings()

    def load_settings(self):
        try:
            if CONFIG_PATH.is_file():
                data = json.loads(CONFIG_PATH.read_text())
                self.preset = data.get("preset", "river_rain")
                if self.preset not in PRESETS:
                    self.preset = "river_rain"
                self.volume = max(0.0, min(1.0, float(data.get("volume", 0.20))))
                self.custom_file = data.get("custom_file", "")
        except Exception:
            pass

    def save_settings(self):
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "preset": self.preset,
                "volume": round(self.volume, 3),
                "custom_file": self.custom_file,
            }
            tmp = CONFIG_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(CONFIG_PATH)
        except Exception:
            pass

    def get_or_create_buffer(self, preset: str) -> np.ndarray:
        if preset in self.buffers:
            return self.buffers[preset]

        buf = None
        if preset == "river_rain":
            sample_path = AMBIENT_DIR / "river-rain.opus"
            if sample_path.is_file():
                try:
                    buf = load_audio_file(sample_path)
                except Exception:
                    pass
            if buf is None:
                # Fallback to procedural rain if sample not found
                buf = generate_rain_noise()
        elif preset == "mellow_rain":
            sample_path = AMBIENT_DIR / "river-rain.opus"
            if sample_path.is_file():
                try:
                    raw_buf = load_audio_file(sample_path)
                    # Gentle 2nd-order Butterworth low-pass at 2800 Hz:
                    # removes sharp splash clicks & harsh treble while preserving
                    # full natural stream body, warmth, depth, and pacing.
                    sos = dsp_signal.butter(2, 2800, btype="lowpass", fs=SAMPLE_RATE, output="sos")
                    filtered = dsp_signal.sosfilt(sos, raw_buf, axis=0).astype(np.float32)
                    orig_rms = np.sqrt(np.mean(raw_buf**2))
                    filt_rms = np.sqrt(np.mean(filtered**2))
                    if filt_rms > 0:
                        buf = filtered * (orig_rms / filt_rms)
                    else:
                        buf = filtered
                except Exception:
                    pass
            if buf is None:
                buf = generate_rain_noise()
        elif preset == "brown":
            buf = generate_brown_noise()
        elif preset == "pink":
            buf = generate_pink_noise()
        elif preset == "white":
            buf = generate_white_noise()
        elif preset == "rain":
            buf = generate_rain_noise()
        elif preset == "custom":
            if self.custom_file and Path(self.custom_file).is_file():
                try:
                    buf = load_audio_file(self.custom_file)
                except Exception:
                    pass
            if buf is None:
                buf = generate_brown_noise()
        else:
            buf = generate_brown_noise()

        self.buffers[preset] = buf
        return buf

    def play(self):
        with self.lock:
            if self.playing:
                return
            self.playing = True
            self.stop_requested.clear()
            self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True, name="white-noise-audio")
            self.audio_thread.start()
            if self.timer_minutes > 0 and (self.timer_thread is None or not self.timer_thread.is_alive()):
                self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True, name="white-noise-timer")
                self.timer_thread.start()

    def pause(self):
        with self.lock:
            if not self.playing:
                return
            self.playing = False
            self.stop_requested.set()

    def toggle(self) -> bool:
        if self.playing:
            self.pause()
            return False
        else:
            self.play()
            return True

    def set_volume(self, vol: float):
        with self.lock:
            self.volume = max(0.0, min(1.0, float(vol)))
            self.save_settings()

    def set_preset(self, preset: str):
        if preset not in PRESETS:
            return
        with self.lock:
            self.preset = preset
            self.save_settings()

    def set_custom_file(self, filepath: str):
        with self.lock:
            self.custom_file = str(filepath)
            self.buffers.pop("custom", None)
            self.preset = "custom"
            self.save_settings()

    def set_timer(self, minutes: int):
        with self.lock:
            self.timer_minutes = max(0, int(minutes))
            if self.timer_minutes > 0:
                self.timer_end_time = time.time() + (self.timer_minutes * 60)
                if self.playing and (self.timer_thread is None or not self.timer_thread.is_alive()):
                    self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True, name="white-noise-timer")
                    self.timer_thread.start()
            else:
                self.timer_end_time = 0.0

    def timer_remaining(self) -> int:
        if self.timer_minutes <= 0 or self.timer_end_time <= 0:
            return 0
        rem = int(self.timer_end_time - time.time())
        return max(0, rem)

    def status(self) -> dict:
        with self.lock:
            return {
                "playing": self.playing,
                "preset": self.preset,
                "volume": round(self.volume, 3),
                "timer_minutes": self.timer_minutes,
                "timer_remaining": self.timer_remaining(),
                "presets": PRESETS,
                "custom_file": self.custom_file,
            }

    def _timer_loop(self):
        while not self.stop_requested.is_set():
            time.sleep(1)
            with self.lock:
                if not self.playing:
                    break
                if self.timer_minutes > 0 and self.timer_end_time > 0:
                    if time.time() >= self.timer_end_time:
                        self.playing = False
                        self.stop_requested.set()
                        self.timer_minutes = 0
                        self.timer_end_time = 0.0
                        break

    def _audio_loop(self):
        # Mute ALSA stderr output during PyAudio initialization
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            import pyaudio
            p = pyaudio.PyAudio()
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull)

        stream = None
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=BUFFER_FRAMES,
            )

            current_preset = None
            buf = None
            buf_len = 0
            idx = 0
            gain = 0.0  # Smooth fade-in

            while not self.stop_requested.is_set():
                with self.lock:
                    target_vol = self.volume
                    active_preset = self.preset

                if active_preset != current_preset:
                    buf = self.get_or_create_buffer(active_preset)
                    buf_len = len(buf)
                    current_preset = active_preset
                    idx = 0

                # Read chunk with loop wrap
                end_idx = idx + BUFFER_FRAMES
                if end_idx <= buf_len:
                    chunk = buf[idx:end_idx].copy()
                    idx = end_idx % buf_len
                else:
                    part1 = buf[idx:]
                    part2_len = BUFFER_FRAMES - len(part1)
                    part2 = buf[:part2_len]
                    chunk = np.vstack([part1, part2])
                    idx = part2_len

                # Smooth volume ramp
                target_gain = target_vol
                if gain != target_gain:
                    gain_curve = np.linspace(gain, target_gain, len(chunk)).reshape(-1, 1)
                    chunk *= gain_curve
                    gain = target_gain
                else:
                    chunk *= gain

                stream.write(chunk.tobytes())

            # Fade out before closing stream
            if gain > 0:
                fade_steps = 1024
                fade_curve = np.linspace(gain, 0.0, fade_steps).reshape(-1, 1)
                fade_chunk = (buf[idx : idx + fade_steps] if idx + fade_steps <= buf_len else buf[:fade_steps]) * fade_curve
                stream.write(fade_chunk.astype(np.float32).tobytes())

        except Exception as e:
            pass
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            p.terminate()


class WhiteNoiseServer:
    """Unix Domain Socket server to expose the audio engine to UI and CLI."""

    def __init__(self, engine: WhiteNoiseEngine):
        self.engine = engine
        self.running = False
        self.server_sock = None

    def start(self):
        SOCKET_DIR.mkdir(parents=True, exist_ok=True)
        if SOCKET_PATH.exists():
            try:
                # Test if another server is already active
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(str(SOCKET_PATH))
                s.close()
                print("Server already running.")
                sys.exit(0)
            except OSError:
                SOCKET_PATH.unlink(missing_ok=True)

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_sock.bind(str(SOCKET_PATH))
        self.server_sock.listen(16)
        self.running = True

        def cleanup(*_):
            self.running = False
            self.engine.pause()
            if self.server_sock:
                self.server_sock.close()
            SOCKET_PATH.unlink(missing_ok=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        print(f"White Noise daemon running on {SOCKET_PATH}")
        while self.running:
            try:
                conn, _ = self.server_sock.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
            except OSError:
                break

    def _handle_client(self, conn: socket.socket):
        with conn:
            try:
                data = conn.recv(4096)
                if not data:
                    return
                req = json.loads(data.decode("utf-8"))
                cmd = req.get("cmd", "status")

                if cmd == "play":
                    self.engine.play()
                elif cmd == "pause":
                    self.engine.pause()
                elif cmd == "toggle":
                    self.engine.toggle()
                elif cmd == "set_volume":
                    self.engine.set_volume(req.get("volume", 0.5))
                elif cmd == "set_preset":
                    self.engine.set_preset(req.get("preset", "river_rain"))
                elif cmd == "set_custom_file":
                    self.engine.set_custom_file(req.get("filepath", ""))
                elif cmd == "set_timer":
                    self.engine.set_timer(req.get("minutes", 0))

                resp = self.engine.status()
                conn.sendall(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                try:
                    conn.sendall(json.dumps({"error": str(e)}).encode("utf-8"))
                except Exception:
                    pass


class WhiteNoiseClient:
    """Robust client that connects to the background daemon, auto-starting it if needed."""

    def __init__(self, autostart: bool = True):
        self.autostart = autostart

    def _connect(self) -> socket.socket:
        t0 = time.time()
        daemon_started = False
        while time.time() - t0 < 4.0:
            if SOCKET_PATH.exists():
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.settimeout(3.0)
                    sock.connect(str(SOCKET_PATH))
                    return sock
                except OSError:
                    pass

            if not self.autostart:
                break

            if not daemon_started:
                script_path = Path(__file__).resolve()
                subprocess.Popen(
                    [sys.executable, str(script_path), "--daemon"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setpgrp,
                )
                daemon_started = True

            time.sleep(0.1)

        raise ConnectionError("Could not connect to White Noise daemon")

    def send_cmd(self, cmd: str, **kwargs) -> dict:
        sock = self._connect()
        with sock:
            payload = {"cmd": cmd, **kwargs}
            sock.sendall(json.dumps(payload).encode("utf-8"))
            data = sock.recv(4096)
            return json.loads(data.decode("utf-8"))

    def status(self) -> dict:
        return self.send_cmd("status")

    def play(self) -> dict:
        return self.send_cmd("play")

    def pause(self) -> dict:
        return self.send_cmd("pause")

    def toggle(self) -> dict:
        return self.send_cmd("toggle")

    def set_volume(self, volume: float) -> dict:
        return self.send_cmd("set_volume", volume=volume)

    def set_preset(self, preset: str) -> dict:
        return self.send_cmd("set_preset", preset=preset)

    def set_custom_file(self, filepath: str) -> dict:
        return self.send_cmd("set_custom_file", filepath=str(filepath))

    def set_timer(self, minutes: int) -> dict:
        return self.send_cmd("set_timer", minutes=int(minutes))


def main():
    parser = argparse.ArgumentParser(description="Whisper Typer White Noise Tool")
    parser.add_argument("--daemon", action="store_true", help="Run background daemon")
    parser.add_argument("--ui", action="store_true", help="Launch standalone White Noise UI")
    parser.add_argument("command", nargs="?", choices=["play", "pause", "toggle", "status", "preset", "volume", "timer", "ui"], help="CLI command")
    parser.add_argument("arg", nargs="?", help="Command argument (preset name, volume 0-100, timer minutes)")

    args = parser.parse_args()

    if args.daemon:
        engine = WhiteNoiseEngine()
        server = WhiteNoiseServer(engine)
        server.start()
        return

    if args.ui or args.command == "ui" or (len(sys.argv) == 1 and sys.stdout.isatty()):
        from white_noise_ui import launch_ui
        launch_ui()
        return

    client = WhiteNoiseClient(autostart=True)
    cmd = args.command or "status"

    if cmd == "status":
        s = client.status()
        state = "PLAYING" if s.get("playing") else "PAUSED"
        preset_info = PRESETS.get(s.get("preset"), {})
        title = preset_info.get("title", s.get("preset"))
        vol = int(s.get("volume", 0) * 100)
        rem = s.get("timer_remaining", 0)
        timer_str = f" (Timer: {rem // 60}m {rem % 60}s)" if rem > 0 else ""
        print(f"White Noise: {state} | Sound: {title} | Volume: {vol}%{timer_str}")

    elif cmd == "play":
        s = client.play()
        print(f"White Noise: PLAYING ({s.get('preset')})")

    elif cmd == "pause":
        s = client.pause()
        print("White Noise: PAUSED")

    elif cmd == "toggle":
        s = client.toggle()
        state = "PLAYING" if s.get("playing") else "PAUSED"
        print(f"White Noise: {state}")

    elif cmd == "preset":
        if not args.arg:
            print("Available presets:", ", ".join(PRESETS.keys()))
            return
        preset = args.arg.lower()
        s = client.set_preset(preset)
        print(f"Preset set to: {s.get('preset')}")

    elif cmd == "volume":
        if not args.arg:
            print("Specify volume 0 to 100")
            return
        vol = float(args.arg) / 100.0
        s = client.set_volume(vol)
        v = round(s.get("volume", 0) * 100, 1)
        pct = f"{int(v)}%" if v.is_integer() else f"{v:.1f}%"
        print(f"Volume set to: {pct}")

    elif cmd == "timer":
        if not args.arg:
            print("Specify timer minutes (0 to cancel)")
            return
        mins = int(args.arg)
        s = client.set_timer(mins)
        print(f"Timer set to: {mins} minutes")


if __name__ == "__main__":
    main()
