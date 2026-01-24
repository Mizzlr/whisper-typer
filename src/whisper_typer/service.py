"""Main service orchestration."""

import asyncio
import json
import logging
import signal
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from .config import Config

# Shared state file for MCP control
STATE_FILE = Path.home() / ".cache" / "whisper-typer" / "state.json"
from .hotkey import HotkeyMonitor
from .notifier import Notifier
from .processor import OllamaProcessor
from .recorder import AudioRecorder
from .silence_detector import SilenceDetector
from .transcriber import WhisperTranscriber
from .typer import TextTyper
from .wakeword import WakeWordDetector

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service state machine states."""
    IDLE = "idle"
    RECORDING = "recording"        # Hotkey-triggered recording
    WAKE_RECORDING = "wake_recording"  # Wake-word triggered recording
    PROCESSING = "processing"


class OutputMode(Enum):
    """Output mode for transcription."""
    BOTH = "both"           # Ollama output + [Whisper raw] in brackets
    WHISPER_ONLY = "whisper_only"  # Only raw Whisper output
    OLLAMA_ONLY = "ollama_only"    # Only Ollama output (no brackets)


class DictationService:
    """Main service that orchestrates all components."""

    def __init__(self, config: Config, output_mode: OutputMode = OutputMode.OLLAMA_ONLY, disable_ollama: bool = False):
        self.config = config
        self.state = ServiceState.IDLE
        self.running = False
        self.output_mode = output_mode
        self.ollama_enabled = config.ollama.enabled and not disable_ollama
        self._last_state_check = 0
        self._write_mcp_state()  # Initialize state file

        # Initialize components
        self.hotkey = HotkeyMonitor(
            config.hotkey,
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
        )
        self.recorder = AudioRecorder(config.audio, config.recording)
        self.transcriber = WhisperTranscriber(config.whisper)
        self.processor = OllamaProcessor(config.ollama)
        self.typer = TextTyper(config.typer)
        self.notifier = Notifier(config.feedback)

        # Wake-word components
        self.wakeword = WakeWordDetector(
            config.wakeword,
            on_detected=self._on_wakeword_detected,
            sample_rate=config.audio.sample_rate,
        )
        self.silence_detector = SilenceDetector(
            config.silence,
            sample_rate=config.audio.sample_rate,
        )

        # For async coordination
        self._process_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mcp_thread: Optional[threading.Thread] = None

    def _write_mcp_state(self):
        """Write current state to MCP state file."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "output_mode": self.output_mode.value,
            "ollama_enabled": self.ollama_enabled,
            "recent_transcriptions": getattr(self, "_recent_transcriptions", []),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

    def _read_mcp_state(self):
        """Read and apply state from MCP state file (if changed externally)."""
        if not STATE_FILE.exists():
            return

        try:
            state = json.loads(STATE_FILE.read_text())
            new_mode = OutputMode(state.get("output_mode", "ollama_only"))
            new_ollama = state.get("ollama_enabled", True)

            if new_mode != self.output_mode or new_ollama != self.ollama_enabled:
                self.output_mode = new_mode
                self.ollama_enabled = new_ollama
                logger.info(f"═══ MCP state change: mode={new_mode.value}, ollama={new_ollama} ═══")
        except Exception as e:
            logger.debug(f"Error reading MCP state: {e}")

    def _add_transcription(self, text: str):
        """Add transcription to recent list for MCP access."""
        if not hasattr(self, "_recent_transcriptions"):
            self._recent_transcriptions = []
        self._recent_transcriptions.append(text)
        self._recent_transcriptions = self._recent_transcriptions[-20:]  # Keep last 20
        self._write_mcp_state()


    def _on_hotkey_press(self):
        """Called when hotkey combination is pressed."""
        if self.state != ServiceState.IDLE:
            logger.debug(f"Ignoring hotkey press, state is {self.state}")
            return

        t_start = time.perf_counter()
        logger.info("Hotkey pressed - starting recording")
        self.state = ServiceState.RECORDING
        self.notifier.recording_started()
        self.recorder.start()
        t_elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(f"Recording active (startup: {t_elapsed:.0f}ms)")

    def _on_hotkey_release(self):
        """Called when hotkey combination is released."""
        if self.state != ServiceState.RECORDING:
            logger.debug(f"Ignoring hotkey release, state is {self.state}")
            return

        logger.info("Hotkey released - stopping recording")
        self.state = ServiceState.PROCESSING
        self.notifier.recording_stopped()

        # Stop recording and get audio data
        audio_data = self.recorder.stop()

        # Schedule async processing
        if self._loop:
            self._process_task = self._loop.create_task(
                self._process_audio(audio_data)
            )

    def _on_wakeword_detected(self):
        """Called when wake-word is detected."""
        if self.state != ServiceState.IDLE:
            logger.debug(f"Ignoring wake-word, state is {self.state}")
            return

        logger.info("Wake-word detected - starting recording")
        self.state = ServiceState.WAKE_RECORDING
        self.notifier.recording_started()
        self.silence_detector.reset()
        self.recorder.start()

    def _on_silence_detected(self):
        """Called when silence is detected during wake-recording."""
        if self.state != ServiceState.WAKE_RECORDING:
            return

        logger.info("Silence detected - stopping wake-recording")
        self.state = ServiceState.PROCESSING
        self.notifier.recording_stopped()

        audio_data = self.recorder.stop()

        if self._loop:
            self._process_task = self._loop.create_task(
                self._process_audio(audio_data)
            )

    def _audio_subscriber(self, audio_chunk):
        """Process audio from recorder stream for wake-word and silence detection."""
        import numpy as np

        # Wake-word detection when idle
        if self.state == ServiceState.IDLE:
            if self.wakeword.enabled and self.wakeword.process_audio(audio_chunk):
                # Schedule callback in main loop (we're in audio thread)
                if self._loop:
                    self._loop.call_soon_threadsafe(self._on_wakeword_detected)

        # Silence detection during wake-recording
        elif self.state == ServiceState.WAKE_RECORDING:
            if self.silence_detector.process_chunk(audio_chunk):
                # Schedule stop in main loop
                if self._loop:
                    self._loop.call_soon_threadsafe(self._on_silence_detected)

    async def _process_audio(self, audio_data: bytes):
        """Process recorded audio: transcribe, fix grammar, type."""
        t_start = time.perf_counter()

        # Check for MCP state changes
        self._read_mcp_state()

        try:
            audio_duration = len(audio_data) / (self.config.audio.sample_rate * 2)  # 2 bytes per sample
            logger.info(f"{'─'*50}")
            logger.info(f"Processing {len(audio_data):,} bytes ({audio_duration:.1f}s audio)")

            if len(audio_data) < 1000:
                logger.warning("⚠️  Recording too short, ignoring")
                self.notifier.error("Recording too short")
                return

            # Check for silence (prevents Whisper hallucinations like "Thank you")
            if self.recorder.is_silent(audio_data):
                logger.warning("⚠️  Audio is silent, ignoring (prevents hallucination)")
                self.notifier.error("No speech detected")
                return

            # Convert to numpy array
            audio = self.recorder.get_audio_as_numpy(audio_data)

            # Transcribe with Whisper
            t_whisper_start = time.perf_counter()
            text = self.transcriber.transcribe(
                audio,
                sample_rate=self.config.audio.sample_rate,
            )
            t_whisper = (time.perf_counter() - t_whisper_start) * 1000

            if not text.strip():
                logger.warning("⚠️  No speech detected")
                self.notifier.error("No speech detected")
                return

            # Filter common Whisper hallucinations
            hallucinations = {
                "thank you", "thank you.", "thanks.", "thanks",
                "thanks for watching", "thanks for watching.",
                "subscribe", "like and subscribe",
                "you", "bye", "bye.", "goodbye", "goodbye.",
            }
            if text.strip().lower() in hallucinations:
                logger.warning(f"Filtered hallucination: '{text}'")
                self.notifier.error("No speech detected")
                return

            # Process with Ollama (if enabled and not in whisper-only mode)
            t_ollama_start = time.perf_counter()
            if self.ollama_enabled and self.output_mode != OutputMode.WHISPER_ONLY:
                processed_text = await self.processor.process(text)
                t_ollama = (time.perf_counter() - t_ollama_start) * 1000
            else:
                processed_text = text
                t_ollama = 0

            # Build final output based on mode
            if self.output_mode == OutputMode.BOTH and processed_text != text:
                final_text = f"{processed_text} [{text}]"
            elif self.output_mode == OutputMode.WHISPER_ONLY:
                final_text = text
            else:
                final_text = processed_text

            # Type the result
            t_type_start = time.perf_counter()
            self.typer.type_text(final_text)
            t_type = (time.perf_counter() - t_type_start) * 1000

            t_total = (time.perf_counter() - t_start) * 1000

            # Log with latencies
            logger.info(f"─── Dictation Complete ───")
            logger.info(f"  Whisper [{t_whisper:.0f}ms]: \"{text}\"")
            if self.ollama_enabled and t_ollama > 0:
                logger.info(f"  Ollama  [{t_ollama:.0f}ms]: \"{processed_text}\"")
            logger.info(f"  Typed   [{t_type:.0f}ms]: {len(final_text)} chars")
            logger.info(f"  Total: {t_total:.0f}ms | Audio: {audio_duration:.1f}s | Speed: {(audio_duration*1000)/t_total:.1f}x")

            self.notifier.transcription_complete(final_text)
            self._add_transcription(final_text)

        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            self.notifier.error(str(e)[:100])

        finally:
            self.state = ServiceState.IDLE

    def _start_mcp_server(self, port: int = 8766):
        """Start MCP server on a background thread."""
        from .mcp_server import mcp, set_service

        # Connect MCP server to this service for direct state access
        set_service(self)

        def run_mcp():
            logger.info(f"Starting MCP server on port {port}")
            mcp.run(transport="sse", port=port)

        self._mcp_thread = threading.Thread(target=run_mcp, daemon=True)
        self._mcp_thread.start()
        logger.info(f"MCP server thread started (port {port})")

    async def run(self):
        """Run the dictation service."""
        self._loop = asyncio.get_running_loop()
        self.running = True

        # Start MCP server on background thread
        self._start_mcp_server(port=8766)

        # Set up signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._loop.add_signal_handler(sig, self._handle_signal)

        # Pre-load Whisper model
        logger.info("Loading Whisper model (this may take a moment)...")
        self.transcriber.load_model()
        logger.info("Whisper model loaded")

        # Load wake-word model if enabled
        if self.config.wakeword.enabled:
            logger.info("Loading wake-word model...")
            self.wakeword.load_model()

        # Pre-open audio stream for low-latency recording
        logger.info("Opening audio stream...")
        self.recorder.open_stream()

        # Subscribe to audio stream for wake-word and silence detection
        self.recorder.add_subscriber(self._audio_subscriber)

        # Start audio stream immediately if wake-word is enabled (for continuous listening)
        if self.wakeword.enabled:
            logger.info("Starting continuous audio stream for wake-word detection...")
            self.recorder.stream.start_stream()

        # Check Ollama availability
        if self.config.ollama.enabled:
            if await self.processor.check_model_available():
                logger.info(f"Ollama model ready: {self.config.ollama.model}")
            else:
                logger.warning(
                    f"Ollama model '{self.config.ollama.model}' not found. "
                    f"Pull it with: ollama pull {self.config.ollama.model}"
                )

        # Check typing backend
        if not self.typer.check_backend_ready():
            logger.warning("Typing backend may not work correctly")

        # Service is ready
        self.notifier.service_ready()
        mode_str = self.output_mode.value.replace("_", " ").title()
        wakeword_str = f"{self.config.wakeword.model}" if self.wakeword.enabled else "Disabled"
        logger.info(f"════════════════════════════════════════")
        logger.info(f"  WHISPER INPUT READY")
        logger.info(f"  Whisper:  {self.config.whisper.model.split('/')[-1]}")
        logger.info(f"  Ollama:   {self.config.ollama.model if self.ollama_enabled else 'Disabled'}")
        logger.info(f"  Mode:     {mode_str}")
        logger.info(f"  WakeWord: {wakeword_str}")
        logger.info(f"  Hotkeys:  Win+Alt, Ctrl+Alt, PgDn+Right, PgDn+Down")
        logger.info(f"════════════════════════════════════════")

        # Start hotkey monitoring
        try:
            await self.hotkey.run()
        except asyncio.CancelledError:
            logger.info("Service cancelled")
        finally:
            await self.shutdown()

    def _handle_signal(self):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self.running = False
        # Cancel the hotkey monitoring
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

    async def shutdown(self):
        """Clean up resources."""
        logger.info("Shutting down...")

        # Cancel any pending processing
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()

        # Clean up components
        self.hotkey.stop()
        self.recorder.remove_subscriber(self._audio_subscriber)
        self.recorder.close()
        self.transcriber.unload_model()
        self.wakeword.unload_model()
        await self.processor.close()
        self.notifier.close()

        logger.info("Shutdown complete")
