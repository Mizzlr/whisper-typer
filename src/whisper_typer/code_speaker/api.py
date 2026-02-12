"""HTTP API server for code_speaker TTS.

Provides endpoints for the Claude Code hook script to trigger TTS.
Runs on a background thread, uses the main asyncio event loop for TTS operations.
"""

import asyncio
import json
import logging
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from .reminder import ReminderManager
from .summarizer import OllamaSummarizer
from .tts import KokoroTTS, SpeakResult

logger = logging.getLogger(__name__)


class TTSRequestHandler(BaseHTTPRequestHandler):
    """Handle incoming TTS API requests."""

    server: "TTSHTTPServer"

    def log_message(self, format, *args):
        logger.debug(f"API: {format % args}")

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/status":
            self._send_json({
                "speaking": self.server.tts.is_speaking,
                "voice": self.server.tts.config.voice,
                "model_loaded": self.server.tts.is_loaded,
                "reminder_active": self.server.reminder.is_active,
                "reminder_count": self.server.reminder.reminder_count,
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, 400)
            return

        if self.path == "/speak":
            text = data.get("text", "")
            summarize = data.get("summarize", False)
            event_type = data.get("event_type", "unknown")
            start_reminder = data.get("start_reminder", False)

            if not text.strip():
                self._send_json({"error": "empty text"}, 400)
                return

            logger.info(f"HTTP /speak: event={event_type} text={len(text)} chars summarize={summarize}")

            # Fire-and-forget: schedule TTS on the event loop
            future = asyncio.run_coroutine_threadsafe(
                self.server.handle_speak(text, summarize, event_type, start_reminder),
                self.server.loop,
            )
            logger.info(f"HTTP /speak: coroutine scheduled, future={future}")
            self._send_json({"status": "speaking"})

        elif self.path == "/cancel":
            self.server.tts.cancel()
            self._send_json({"status": "cancelled"})

        elif self.path == "/cancel-reminder":
            count = self.server.reminder.cancel()
            self.server.tts.cancel()
            self._send_json({"status": "cancelled", "reminders_fired": count})

        else:
            self._send_json({"error": "not found"}, 404)


class TTSHTTPServer(HTTPServer):
    """HTTP server with references to TTS components."""

    def __init__(
        self,
        port: int,
        tts: KokoroTTS,
        summarizer: OllamaSummarizer,
        reminder: ReminderManager,
        loop: asyncio.AbstractEventLoop,
        on_tts_complete=None,
    ):
        super().__init__(("127.0.0.1", port), TTSRequestHandler)
        self.tts = tts
        self.summarizer = summarizer
        self.reminder = reminder
        self.loop = loop
        self.on_tts_complete = on_tts_complete

    async def handle_speak(
        self,
        text: str,
        summarize: bool,
        event_type: str,
        start_reminder: bool,
    ):
        """Handle a speak request: summarize → speak → start reminder."""
        logger.info(f"handle_speak START [{event_type}]: text={len(text)} chars, summarize={summarize}")
        t_total_start = time.perf_counter()
        ollama_ms = 0.0
        spoken_text = text
        summarized = False

        try:
            # Cancel any existing reminder
            self.reminder.cancel()
            # Cancel any current speech
            self.tts.cancel()
            await asyncio.sleep(0.05)  # Brief pause for sounddevice to stop

            # Summarize long text
            max_chars = self.tts.config.max_direct_chars
            if summarize and len(text) > max_chars:
                spoken_text, ollama_ms = await self.summarizer.summarize(text)
                summarized = True
                logger.info(f"Summarized {len(text)} chars → {len(spoken_text)} chars ({ollama_ms:.0f}ms)")

            # Speak
            result: SpeakResult = await self.tts.speak(spoken_text)

            total_ms = (time.perf_counter() - t_total_start) * 1000
            logger.info(
                f"TTS complete [{event_type}]: "
                f"ollama={ollama_ms:.0f}ms gen={result.generate_ms:.0f}ms "
                f"play={result.playback_ms:.0f}ms total={total_ms:.0f}ms"
            )

            # Start reminder if requested and not cancelled
            if start_reminder and not result.cancelled:
                self.reminder.start(spoken_text, self.tts.speak)

            # Callback for history/stats
            if self.on_tts_complete:
                self.on_tts_complete(
                    event_type=event_type,
                    input_text=text,
                    spoken_text=spoken_text,
                    summarized=summarized,
                    ollama_ms=ollama_ms,
                    kokoro_ms=result.generate_ms,
                    playback_ms=result.playback_ms,
                    total_ms=total_ms,
                    cancelled=result.cancelled,
                )

        except Exception as e:
            logger.exception(f"TTS speak error: {e}")


class TTSApiServer:
    """Manages the TTS HTTP API server on a background thread."""

    def __init__(
        self,
        tts: KokoroTTS,
        summarizer: OllamaSummarizer,
        reminder: ReminderManager,
        loop: asyncio.AbstractEventLoop,
        port: int = 8767,
        on_tts_complete=None,
    ):
        self._server = TTSHTTPServer(
            port=port,
            tts=tts,
            summarizer=summarizer,
            reminder=reminder,
            loop=loop,
            on_tts_complete=on_tts_complete,
        )
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the API server on a background thread."""
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()
        logger.info(f"TTS API server started on port {self._server.server_port}")

    def stop(self):
        """Stop the API server."""
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("TTS API server stopped")
