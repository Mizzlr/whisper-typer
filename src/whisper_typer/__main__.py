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
        "-c", "--config",
        type=Path,
        help="Path to config file (default: ./config.yaml)",
    )
    parser.add_argument(
        "-v", "--verbose",
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
        print(f"           Channels: {device['channels']}, Rate: {device['sample_rate']}Hz")
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

    # Load configuration
    config = Config.load(args.config)

    # Apply CLI overrides to config
    if args.enable_wakeword:
        config.wakeword.enabled = True
    if args.wakeword_model:
        config.wakeword.model = args.wakeword_model

    # Map CLI mode to OutputMode
    mode_map = {
        "ollama": OutputMode.OLLAMA_ONLY,
        "whisper": OutputMode.WHISPER_ONLY,
        "both": OutputMode.BOTH,
    }
    output_mode = mode_map[args.mode]

    # Create and run service
    service = DictationService(config, output_mode=output_mode, disable_ollama=args.no_ollama)

    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        pass  # Clean shutdown handled by signal handler

    return 0


if __name__ == "__main__":
    sys.exit(main())
