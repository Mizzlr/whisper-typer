"""Main service orchestration."""

import asyncio
import logging
import signal
from enum import Enum
from typing import Optional

from .config import Config
from .hotkey import HotkeyMonitor
from .notifier import Notifier
from .processor import OllamaProcessor
from .recorder import AudioRecorder
from .transcriber import WhisperTranscriber
from .typer import TextTyper

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service state machine states."""
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class DictationService:
    """Main service that orchestrates all components."""

    def __init__(self, config: Config):
        self.config = config
        self.state = ServiceState.IDLE
        self.running = False

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

        # For async coordination
        self._process_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _on_hotkey_press(self):
        """Called when hotkey combination is pressed."""
        if self.state != ServiceState.IDLE:
            logger.debug(f"Ignoring hotkey press, state is {self.state}")
            return

        logger.info("Hotkey pressed - starting recording")
        self.state = ServiceState.RECORDING
        self.notifier.recording_started()
        self.recorder.start()

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

    async def _process_audio(self, audio_data: bytes):
        """Process recorded audio: transcribe, fix grammar, type."""
        try:
            if len(audio_data) < 1000:
                logger.warning("Recording too short, ignoring")
                self.notifier.error("Recording too short")
                return

            # Check for silence (prevents Whisper hallucinations like "Thank you")
            if self.recorder.is_silent(audio_data):
                logger.warning("Audio is silent, ignoring (prevents hallucination)")
                self.notifier.error("No speech detected")
                return

            # Convert to numpy array
            audio = self.recorder.get_audio_as_numpy(audio_data)

            # Transcribe with Whisper
            text = self.transcriber.transcribe(
                audio,
                sample_rate=self.config.audio.sample_rate,
            )

            if not text.strip():
                logger.warning("No speech detected")
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

            # Process with Ollama (if enabled)
            processed_text = await self.processor.process(text)

            # Append raw Whisper output in brackets if Ollama changed it
            if self.config.ollama.enabled and processed_text != text:
                final_text = f"{processed_text} [{text}]"
            else:
                final_text = processed_text

            # Type the result
            logger.info(f"Final text to type: '{final_text}'")
            self.typer.type_text(final_text)
            logger.info(f"Typing complete ({len(final_text)} chars)")
            self.notifier.transcription_complete(final_text)

        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            self.notifier.error(str(e)[:100])

        finally:
            self.state = ServiceState.IDLE

    async def run(self):
        """Run the dictation service."""
        self._loop = asyncio.get_running_loop()
        self.running = True

        # Set up signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._loop.add_signal_handler(sig, self._handle_signal)

        # Pre-load Whisper model
        logger.info("Loading Whisper model (this may take a moment)...")
        self.transcriber.load_model()
        logger.info("Whisper model loaded")

        # Pre-open audio stream for low-latency recording
        logger.info("Opening audio stream...")
        self.recorder.open_stream()

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
        logger.info("Dictation service ready. Hold Win+Alt to dictate.")

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
        self.recorder.close()
        self.transcriber.unload_model()
        await self.processor.close()
        self.notifier.close()

        logger.info("Shutdown complete")
