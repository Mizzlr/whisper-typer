"""Wake-word detection using OpenWakeWord."""

import logging
import time
from typing import Callable, Optional

import numpy as np

from .config import WakeWordConfig

logger = logging.getLogger(__name__)

INT16_MAX_F = 32768.0
_WAKEWORD_COOLDOWN_S = 2.0


class WakeWordDetector:
    """Detect wake-word from audio stream using OpenWakeWord."""

    def __init__(
        self,
        config: WakeWordConfig,
        on_detected: Callable[[], None],
        sample_rate: int = 16000,
    ):
        self.config = config
        self.on_detected = on_detected
        self.sample_rate = sample_rate
        self.model: Optional[object] = None
        self._cooldown_until: float = 0
        self._enabled = config.enabled
        self._model_name: Optional[str] = None

    def load_model(self):
        """Load OpenWakeWord model."""
        if not self.config.enabled:
            logger.info("Wake-word detection disabled")
            return

        try:
            import openwakeword
            from openwakeword.model import Model

            # Get pretrained model paths
            model_paths = openwakeword.get_pretrained_model_paths()

            # Find the matching model
            target_model = self.config.model.lower()
            matching_paths = [p for p in model_paths if target_model in p.lower()]

            if matching_paths:
                self.model = Model(wakeword_model_paths=matching_paths)
                self._model_name = matching_paths[0].split("/")[-1].replace(".onnx", "")
                logger.info(f"Wake-word model loaded: {self._model_name}")
            else:
                # Load all models and filter by name during detection
                self.model = Model()
                self._model_name = target_model
                logger.info(f"Wake-word models loaded (filtering for: {target_model})")

        except ImportError:
            logger.error(
                "openwakeword not installed. Run: pip install openwakeword onnxruntime"
            )
            self._enabled = False
        except Exception as e:
            logger.error(f"Failed to load wake-word model: {e}")
            self._enabled = False

    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """Process audio chunk and check for wake-word.

        Args:
            audio_chunk: Float32 audio samples normalized to [-1, 1]

        Returns:
            True if wake-word detected
        """
        if not self._enabled or self.model is None:
            return False

        # Check cooldown
        if time.time() < self._cooldown_until:
            return False

        # OpenWakeWord expects int16 audio
        audio_int16 = (audio_chunk * INT16_MAX_F).astype(np.int16)

        # Get predictions
        predictions = self.model.predict(audio_int16)

        # Check if configured model's prediction exceeds threshold
        target = self.config.model.lower()
        for model_name, score in predictions.items():
            # Match model name (partial match)
            if target in model_name.lower() and score >= self.config.threshold:
                logger.info(f"Wake-word detected: {model_name} (score: {score:.3f})")
                self._cooldown_until = time.time() + _WAKEWORD_COOLDOWN_S
                return True

        return False

    def unload_model(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None

    @property
    def enabled(self) -> bool:
        """Check if wake-word detection is enabled and ready."""
        return self._enabled and self.model is not None
