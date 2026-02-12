"""Silence detection for automatic recording stop."""

import logging
import time

import numpy as np

from .config import SilenceConfig

logger = logging.getLogger(__name__)


class SilenceDetector:
    """Detect silence in audio stream for auto-stop during wake-word recording."""

    def __init__(self, config: SilenceConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate
        self._speech_started = False
        self._silence_start: float | None = None
        self._recording_start: float = 0

    def reset(self):
        """Reset state for new recording."""
        self._speech_started = False
        self._silence_start = None
        self._recording_start = time.time()

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Process audio chunk and check if recording should stop.

        Args:
            audio_chunk: Float32 audio samples normalized to [-1, 1]

        Returns:
            True if silence threshold exceeded (should stop recording)
        """
        now = time.time()
        elapsed = now - self._recording_start

        # Check max recording duration
        if elapsed >= self.config.max_recording_duration:
            logger.info(
                f"Max recording duration reached ({self.config.max_recording_duration}s)"
            )
            return True

        # Don't check silence until minimum duration passed
        if elapsed < self.config.min_speech_duration:
            return False

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_chunk**2))
        is_silent = rms < self.config.threshold

        if is_silent:
            if self._silence_start is None:
                self._silence_start = now
            elif now - self._silence_start >= self.config.duration:
                logger.debug(f"Silence detected for {self.config.duration}s - stopping")
                return True
        else:
            self._silence_start = None
            self._speech_started = True

        return False
