"""Whisper transcription using transformers pipeline."""

import logging

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .config import WhisperConfig

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Transcribe audio using Whisper model."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self.pipe = None
        self.device = config.device
        self.torch_dtype = torch.float16 if config.device == "cuda" else torch.float32

    def load_model(self):
        """Load Whisper model and create pipeline."""
        logger.info(f"Loading Whisper model: {self.config.model}")
        logger.info(f"Device: {self.device}, dtype: {self.torch_dtype}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            logger.info(f"Model loaded on CUDA: {torch.cuda.get_device_name()}")
        elif self.device == "mps" and torch.backends.mps.is_available():
            model = model.to("mps")
            logger.info("Model loaded on MPS (Apple Silicon)")
        else:
            self.device = "cpu"
            logger.info("Model loaded on CPU")

        processor = AutoProcessor.from_pretrained(self.config.model)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device if self.device != "cpu" else -1,
        )

        logger.info("Whisper pipeline ready")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio array to text."""
        if self.pipe is None:
            self.load_model()

        if len(audio) == 0:
            return ""

        # Ensure audio is the right shape and type
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        audio = audio.astype(np.float32)

        logger.debug(f"Transcribing {len(audio) / sample_rate:.1f}s of audio")

        result = self.pipe(
            {"raw": audio, "sampling_rate": sample_rate},
            return_timestamps=True,  # Required for audio > 30 seconds
        )

        logger.debug(f"Whisper raw result: {result}")
        text = result["text"].strip()
        logger.debug(f"Whisper transcription: '{text}'")
        return text

    def unload_model(self):
        """Unload model to free memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")
