#!/usr/bin/python3
"""Comprehensive test suite for Whisper Typer Local White Noise system."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from white_noise import (
    SAMPLE_RATE,
    PRESETS,
    make_seamless_loop,
    generate_brown_noise,
    generate_pink_noise,
    generate_white_noise,
    generate_rain_noise,
    WhiteNoiseEngine,
    WhiteNoiseClient,
)
from white_noise_ui import WhiteNoiseUI


class WhiteNoiseTests(unittest.TestCase):
    def test_procedural_generators(self):
        """Test that procedural generators generate stereo float32 non-empty buffers."""
        brown = generate_brown_noise(duration_sec=2)
        self.assertEqual(brown.shape[1], 2)
        self.assertEqual(brown.dtype, np.float32)
        self.assertTrue(np.max(np.abs(brown)) > 0.05)

        pink = generate_pink_noise(duration_sec=2)
        self.assertEqual(pink.shape[1], 2)
        self.assertEqual(pink.dtype, np.float32)
        self.assertTrue(np.max(np.abs(pink)) > 0.05)

        white = generate_white_noise(duration_sec=2)
        self.assertEqual(white.shape[1], 2)
        self.assertEqual(white.dtype, np.float32)
        self.assertTrue(np.max(np.abs(white)) > 0.05)

        rain = generate_rain_noise(duration_sec=2)
        self.assertEqual(rain.shape[1], 2)
        self.assertEqual(rain.dtype, np.float32)
        self.assertTrue(np.max(np.abs(rain)) > 0.05)

    def test_make_seamless_loop(self):
        """Test equal-power crossfade loop boundary continuity."""
        raw = np.linspace(0.0, 10.0, 48000 * 4).reshape(-1, 2).astype(np.float32)
        looped = make_seamless_loop(raw, crossfade_samples=4800)
        self.assertLess(len(looped), len(raw))
        self.assertEqual(looped.dtype, np.float32)

    def test_engine_state_and_settings(self):
        """Test engine setting modification and timer logic without starting audio stream."""
        engine = WhiteNoiseEngine()
        engine.set_volume(0.85)
        self.assertAlmostEqual(engine.volume, 0.85, places=2)

        engine.set_preset("brown")
        self.assertEqual(engine.preset, "brown")

        engine.set_timer(15)
        self.assertEqual(engine.timer_minutes, 15)
        self.assertGreater(engine.timer_remaining(), 800)

        engine.set_timer(0)
        self.assertEqual(engine.timer_minutes, 0)
        self.assertEqual(engine.timer_remaining(), 0)

        status = engine.status()
        self.assertEqual(status["preset"], "brown")
        self.assertAlmostEqual(status["volume"], 0.85, places=2)

    def test_client_daemon_communication(self):
        """Test WhiteNoiseClient socket commands."""
        client = WhiteNoiseClient(autostart=True)
        status = client.status()
        self.assertIn("playing", status)
        self.assertIn("preset", status)
        self.assertIn("volume", status)

        client.set_preset("pink")
        s = client.status()
        self.assertEqual(s["preset"], "pink")

        client.set_volume(0.42)
        s = client.status()
        self.assertAlmostEqual(s["volume"], 0.42, places=2)

        # Restore default
        client.set_preset("river_rain")
        client.set_volume(0.50)

    def test_ui_instantiation(self):
        """Test that WhiteNoiseUI window builds cleanly."""
        import tkinter as tk
        root = tk.Tk()
        app = WhiteNoiseUI(root)
        root.update()
        self.assertIn(app.theme, ["dark", "light"])
        self.assertIsNotNone(app.play_button)
        self.assertIsNotNone(app.slider)
        app._toggle_theme()
        root.update()
        app._on_close()

    def test_dictation_window_integration(self):
        """Test that DictationWindow toolbar contains noise button and popover works."""
        import tkinter as tk
        import importlib.util
        spec = importlib.util.spec_from_file_location("dictation_window", Path(__file__).with_name("dictation-window.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        root = tk.Tk()
        with tempfile.TemporaryDirectory() as td:
            store = mod.DictationStore(td, Path(td) / "reviews.jsonl")
            app = mod.DictationWindow(root, store, Path(td) / "settings.json")
            root.update()
            self.assertIsNotNone(app.noise_button)
            self.assertTrue(app.noise_button.cget("text").startswith("🌊"))

            # Test popover
            app.show_white_noise_popover()
            root.update()
            self.assertIsNotNone(app.white_noise_popover)
            self.assertTrue(app.white_noise_popover.winfo_exists())
            app.show_white_noise_popover()
            self.assertIsNone(app.white_noise_popover)
            app.close()


if __name__ == "__main__":
    unittest.main()
