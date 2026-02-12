"""Entry point for whisper-input."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import Config
from .service import DictationService, OutputMode


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Suppress deprecation warnings from transformers
    import warnings

    warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speech-to-text dictation service with hotkey activation"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to config file (default: ./config.yaml)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--test-hotkey",
        action="store_true",
        help="Test hotkey detection and exit",
    )
    parser.add_argument(
        "--mode",
        choices=["ollama", "whisper", "both"],
        default="ollama",
        help="Output mode: ollama (default), whisper (raw only), both (corrected + [raw])",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Disable Ollama processing entirely",
    )
    parser.add_argument(
        "--enable-wakeword",
        action="store_true",
        help="Enable wake-word activation (e.g., 'Hey Jarvis')",
    )
    parser.add_argument(
        "--wakeword-model",
        default=None,
        help="Wake-word model to use (default: hey_jarvis)",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        const="today",
        metavar="DATE",
        help="Show productivity report and exit (today, YYYY-MM-DD, or 'list' to show available dates)",
    )
    # Code Speaker TTS options
    parser.add_argument(
        "--enable-tts",
        action="store_true",
        help="Enable Code Speaker TTS (Kokoro) for Claude Code events",
    )
    parser.add_argument(
        "--tts-voice",
        default=None,
        help="Set TTS voice (e.g., 'af_heart', 'bf_emma', 'am_adam')",
    )
    parser.add_argument(
        "--list-tts-voices",
        action="store_true",
        help="List available TTS voices and exit",
    )
    parser.add_argument(
        "--test-tts",
        metavar="TEXT",
        help="Speak given text using Kokoro TTS and exit",
    )
    return parser.parse_args()


def list_audio_devices():
    """List available audio input devices."""
    from .recorder import AudioRecorder
    from .config import AudioConfig, RecordingConfig

    recorder = AudioRecorder(AudioConfig(), RecordingConfig())
    devices = recorder.list_devices()
    recorder.close()

    print("\nAvailable audio input devices:")
    print("-" * 50)
    for device in devices:
        print(f"  Index {device['index']}: {device['name']}")
        print(
            f"           Channels: {device['channels']}, Rate: {device['sample_rate']}Hz"
        )
    print()


async def test_hotkey():
    """Test hotkey detection."""
    from .config import HotkeyConfig
    from .hotkey import HotkeyMonitor

    print("\nTesting hotkey detection...")
    print("Press Win+Alt to test. Press Ctrl+C to exit.\n")

    def on_press():
        print(">>> Hotkey PRESSED")

    def on_release():
        print("<<< Hotkey RELEASED")

    config = HotkeyConfig()
    monitor = HotkeyMonitor(config, on_press, on_release)

    try:
        await monitor.run()
    except KeyboardInterrupt:
        print("\nTest complete.")
    finally:
        monitor.stop()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    # Handle utility commands
    if args.list_devices:
        list_audio_devices()
        return 0

    if args.test_hotkey:
        asyncio.run(test_hotkey())
        return 0

    if args.report:
        from .history import generate_report, list_available_dates

        if args.report == "list":
            dates = list_available_dates()
            if not dates:
                print("No history records found.")
            else:
                print("Available dates:")
                for d in dates:
                    print(f"  {d}")
        else:
            print(generate_report(args.report))
        return 0

    if args.list_tts_voices:
        print("\nAvailable Kokoro TTS voices:")
        print("-" * 40)
        print("  American Female: af_heart, af_bella, af_nova, af_sarah, af_sky")
        print("  American Male:   am_adam, am_michael, am_echo")
        print("  British Female:  bf_emma, bf_alice, bf_lily")
        print("  British Male:    bm_george, bm_lewis")
        print()
        return 0

    if args.test_tts:

        async def _test_tts():
            from .config import TTSConfig
            from .code_speaker.tts import KokoroTTS

            tts_config = TTSConfig(enabled=True, voice=args.tts_voice or "af_heart")
            tts = KokoroTTS(tts_config)
            await tts.load_model()
            result = await tts.speak(args.test_tts)
            print(f'Spoke: "{args.test_tts}"')
            print(
                f"  Generate: {result.generate_ms:.0f}ms, Playback: {result.playback_ms:.0f}ms"
            )
            tts.unload_model()

        asyncio.run(_test_tts())
        return 0

    # Load configuration
    config = Config.load(args.config)

    # Apply CLI overrides to config
    if args.enable_wakeword:
        config.wakeword.enabled = True
    if args.wakeword_model:
        config.wakeword.model = args.wakeword_model
    if args.enable_tts:
        config.tts.enabled = True
    if args.tts_voice:
        config.tts.voice = args.tts_voice

    # Map CLI mode to OutputMode
    mode_map = {
        "ollama": OutputMode.OLLAMA_ONLY,
        "whisper": OutputMode.WHISPER_ONLY,
        "both": OutputMode.BOTH,
    }
    output_mode = mode_map[args.mode]

    # Create and run service
    service = DictationService(
        config, output_mode=output_mode, disable_ollama=args.no_ollama
    )

    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        pass  # Clean shutdown handled by signal handler

    return 0


if __name__ == "__main__":
    sys.exit(main())
