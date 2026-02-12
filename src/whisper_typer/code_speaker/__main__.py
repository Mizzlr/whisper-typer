"""Standalone Code Speaker TTS server.

Runs the TTS HTTP API independently, for use alongside whisper-typer-rs.
Usage: python3 -m whisper_typer.code_speaker [--port 8767] [--voice af_heart]
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

# Add parent to path so we can import from whisper_typer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Code Speaker TTS Server")
    parser.add_argument("--port", type=int, default=8767, help="HTTP API port")
    parser.add_argument("--voice", default=None, help="TTS voice (e.g., am_michael)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--model-path", default="", help="Path to kokoro-v1.0.onnx")
    parser.add_argument(
        "--max-direct-chars",
        type=int,
        default=150,
        help="Summarize text longer than this",
    )
    parser.add_argument(
        "--reminder-interval",
        type=int,
        default=300,
        help="Seconds between reminders",
    )
    parser.add_argument(
        "--ollama-model", default="qwen2.5:1.5b", help="Ollama model for summarization"
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama API endpoint",
    )
    args = parser.parse_args()

    asyncio.run(_run(args))


async def _run(args):
    from whisper_typer.config import TTSConfig
    from whisper_typer.code_speaker.tts import KokoroTTS
    from whisper_typer.code_speaker.summarizer import OllamaSummarizer
    from whisper_typer.code_speaker.reminder import ReminderManager
    from whisper_typer.code_speaker.api import TTSApiServer
    from whisper_typer.code_speaker.history import TTSRecord, save_tts_record

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("code_speaker")

    # Build TTS config
    tts_config = TTSConfig(
        enabled=True,
        voice=args.voice or "am_michael",
        speed=args.speed,
        api_port=args.port,
        max_direct_chars=args.max_direct_chars,
        reminder_interval=args.reminder_interval,
        model_path=args.model_path,
    )

    # Load model
    logger.info("Loading Kokoro TTS model...")
    tts = KokoroTTS(tts_config)
    await tts.load_model()

    # Summarizer
    summarizer = OllamaSummarizer(model=args.ollama_model, host=args.ollama_host)

    # Reminder
    reminder = ReminderManager(interval=tts_config.reminder_interval)

    # History callback
    def on_tts_complete(**kwargs):
        from datetime import datetime

        record = TTSRecord(
            timestamp=datetime.now().isoformat(),
            event_type=kwargs.get("event_type", "unknown"),
            input_text_chars=len(kwargs.get("input_text", "")),
            summarized=kwargs.get("summarized", False),
            summary_text=kwargs.get("spoken_text", ""),
            ollama_latency_ms=int(kwargs.get("ollama_ms", 0)),
            kokoro_latency_ms=int(kwargs.get("kokoro_ms", 0)),
            playback_duration_ms=int(kwargs.get("playback_ms", 0)),
            total_latency_ms=int(kwargs.get("total_ms", 0)),
            voice=tts_config.voice,
            cancelled=kwargs.get("cancelled", False),
            reminder_count=reminder.reminder_count if reminder else 0,
        )
        save_tts_record(record)

    # Voice idle event (always set — no STT in this standalone mode,
    # the Rust service controls gating via /cancel)
    voice_idle = asyncio.Event()
    voice_idle.set()

    loop = asyncio.get_event_loop()

    # Start API server
    api = TTSApiServer(
        tts=tts,
        summarizer=summarizer,
        reminder=reminder,
        loop=loop,
        port=tts_config.api_port,
        on_tts_complete=on_tts_complete,
        voice_idle=voice_idle,
    )
    api.start()
    logger.info(
        f"Code Speaker TTS server running on port {tts_config.api_port} "
        f"(voice: {tts_config.voice}, speed: {tts_config.speed})"
    )

    # Wait for shutdown signal
    stop = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop.set()

    loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    loop.add_signal_handler(signal.SIGINT, _signal_handler)

    await stop.wait()

    # Cleanup
    logger.info("Shutting down...")
    await tts.cancel_and_wait()
    api.stop()
    tts.unload_model()
    logger.info("Code Speaker stopped")


if __name__ == "__main__":
    main()
