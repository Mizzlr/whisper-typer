"""Kokoro TTS engine with async speak/cancel and sentence streaming."""

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import TTSConfig

logger = logging.getLogger(__name__)

# Sentence boundary pattern: split on .!? followed by whitespace
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SpeakResult:
    """Result of a speak operation with timing info."""

    generate_ms: float
    playback_ms: float
    cancelled: bool
    text_spoken: str


class KokoroTTS:
    """Text-to-speech using Kokoro-82M ONNX model."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self._kokoro = None
        self._cancel_event = asyncio.Event()
        self._speaking = False
        self._speak_lock = asyncio.Lock()
        self._playback_done = threading.Event()
        self._playback_done.set()  # Not playing initially
        # Serializes all sounddevice operations (play/wait/stop) to prevent
        # PortAudio double-free when cancel races with playback teardown.
        self._sd_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._kokoro is not None

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def load_model(self):
        """Load the Kokoro ONNX model. Blocking call wrapped in to_thread."""
        if self._kokoro is not None:
            return

        def _load():
            from kokoro_onnx import Kokoro

            # Find model files
            model_dir = (
                Path(self.config.model_path) if self.config.model_path else Path.cwd()
            )
            onnx_path = model_dir / "kokoro-v1.0.onnx"
            voices_path = model_dir / "voices-v1.0.bin"

            # Fallback: check whisper-typer root
            if not onnx_path.exists():
                wt_root = Path(__file__).parent.parent.parent.parent
                onnx_path = wt_root / "kokoro-v1.0.onnx"
                voices_path = wt_root / "voices-v1.0.bin"

            if not onnx_path.exists():
                raise FileNotFoundError(
                    "Kokoro model not found. Download with:\n"
                    "  wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx\n"
                    "  wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
                )

            logger.info(f"Loading Kokoro model from {onnx_path}")
            return Kokoro(str(onnx_path), str(voices_path))

        t0 = time.perf_counter()
        self._kokoro = await asyncio.to_thread(_load)
        t_load = (time.perf_counter() - t0) * 1000
        logger.info(f"Kokoro model loaded in {t_load:.0f}ms")

    async def speak(self, text: str) -> SpeakResult:
        """Speak text aloud with sentence-level streaming.

        Acquires a lock so only one speak() runs at a time.
        Generates audio for each sentence and plays them sequentially.
        Checks cancel event between sentences for quick interruption.
        """
        if not self._kokoro:
            await self.load_model()

        async with self._speak_lock:
            return await self._speak_inner(text)

    async def _speak_inner(self, text: str) -> SpeakResult:
        """Inner speak implementation (caller must hold _speak_lock)."""
        self._cancel_event.clear()
        self._speaking = True
        total_gen_ms = 0.0
        total_play_ms = 0.0
        cancelled = False

        try:
            sentences = SENTENCE_SPLIT.split(text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                return SpeakResult(0, 0, False, "")

            for i, sentence in enumerate(sentences):
                if self._cancel_event.is_set():
                    cancelled = True
                    logger.info(f"Cancelled before sentence {i + 1}/{len(sentences)}")
                    break

                # Generate audio
                t_gen_start = time.perf_counter()
                samples, sample_rate = await asyncio.to_thread(
                    self._kokoro.create,
                    sentence,
                    voice=self.config.voice,
                    speed=self.config.speed,
                )
                gen_ms = (time.perf_counter() - t_gen_start) * 1000
                total_gen_ms += gen_ms

                if self._cancel_event.is_set():
                    cancelled = True
                    logger.info(
                        f"Cancelled after generating sentence {i + 1}/{len(sentences)}"
                    )
                    break

                # Play audio
                t_play_start = time.perf_counter()
                await self._play_audio(samples, sample_rate)
                play_ms = (time.perf_counter() - t_play_start) * 1000
                total_play_ms += play_ms

                # Check if playback was cancelled (sd.stop() called)
                if self._cancel_event.is_set():
                    cancelled = True
                    logger.info(
                        f"Cancelled during playback of sentence {i + 1}/{len(sentences)}"
                    )
                    break

                duration = len(samples) / sample_rate
                logger.debug(
                    f"Sentence {i + 1}/{len(sentences)}: "
                    f"gen={gen_ms:.0f}ms play={duration:.1f}s"
                )

        except asyncio.CancelledError:
            cancelled = True
        except Exception as e:
            logger.warning(f"TTS speak failed: {e}")
        finally:
            self._speaking = False

        return SpeakResult(
            generate_ms=total_gen_ms,
            playback_ms=total_play_ms,
            cancelled=cancelled,
            text_spoken=text,
        )

    async def _play_audio(self, samples, sample_rate: int):
        """Play audio samples through speakers with proper cancellation.

        Uses _sd_lock only around sd.play() to prevent overlapping playback.
        sd.wait() runs outside the lock so cancel() can call sd.stop() immediately.
        """
        import sounddevice as sd

        self._playback_done.clear()

        def _play():
            try:
                with self._sd_lock:
                    sd.play(samples, sample_rate)
                # Wait outside the lock — allows cancel() to call sd.stop() immediately
                sd.wait()
            except Exception:
                pass
            finally:
                self._playback_done.set()

        await asyncio.to_thread(_play)

    def cancel(self):
        """Cancel current speech immediately.

        Sets the cancel event first (so the speak loop stops between sentences),
        then calls sd.stop() to interrupt playback. sd.stop() is safe to call
        from any thread — it signals PortAudio to stop the stream.
        """
        self._cancel_event.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        # Wait for the _play thread to fully exit
        self._playback_done.wait(timeout=0.2)
        self._speaking = False
        logger.info("TTS cancelled")

    async def cancel_and_wait(self):
        """Cancel current speech and wait for the speak lock to be released."""
        self.cancel()
        # If speak() is holding the lock, wait for it to finish
        # (it will see the cancel event and exit quickly)
        async with self._speak_lock:
            pass

    def list_voices(self) -> list[str]:
        """Return available Kokoro voices."""
        if not self._kokoro:
            return []
        return sorted(self._kokoro.get_voices())

    def unload_model(self):
        """Free the ONNX model."""
        self._kokoro = None
        logger.info("Kokoro model unloaded")
