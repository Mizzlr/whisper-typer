"""Audio recording with PyAudio."""

import logging
import threading
from typing import Callable, Optional

import numpy as np
import pyaudio

from .config import AudioConfig, RecordingConfig

logger = logging.getLogger(__name__)

# Audio normalization: int16 range is [-32768, 32767], divide by this for [-1, 1] float32
INT16_MAX_F = 32768.0

# Type alias for audio subscriber callbacks
AudioSubscriber = Callable[[np.ndarray], None]


class AudioRecorder:
    """Record audio from microphone using PyAudio.

    Keeps the audio stream open to minimize latency on recording start.
    """

    def __init__(self, audio_config: AudioConfig, recording_config: RecordingConfig):
        self.audio_config = audio_config
        self.recording_config = recording_config
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.buffer: list[bytes] = []
        self.is_recording = False
        self.lock = threading.Lock()
        self.max_frames = int(
            recording_config.max_duration
            * audio_config.sample_rate
            / audio_config.chunk_size
        )
        # Subscribers receive audio chunks for wake-word detection, silence detection, etc.
        self._subscribers: list[AudioSubscriber] = []
        self._subscribers_lock = threading.Lock()

    def add_subscriber(self, callback: AudioSubscriber):
        """Add a subscriber to receive audio chunks."""
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def remove_subscriber(self, callback: AudioSubscriber):
        """Remove an audio subscriber."""
        with self._subscribers_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - runs in separate thread."""
        if status:
            logger.warning(f"PyAudio status: {status}")

        # Convert to numpy for subscribers
        audio_np = (
            np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / INT16_MAX_F
        )

        # Notify all subscribers (wake-word detector, silence detector, etc.)
        with self._subscribers_lock:
            for subscriber in self._subscribers:
                try:
                    subscriber(audio_np)
                except Exception as e:
                    logger.error(f"Audio subscriber error: {e}")

        with self.lock:
            if self.is_recording:
                self.buffer.append(in_data)
                # Safety limit
                if len(self.buffer) >= self.max_frames:
                    logger.warning("Max recording duration reached")
                    self.is_recording = False

        return (None, pyaudio.paContinue)

    def open_stream(self):
        """Open audio stream (call once at startup for low latency)."""
        if self.stream is not None:
            return  # Already open

        if self.audio is None:
            self.audio = pyaudio.PyAudio()

        device_index = self.audio_config.device_index
        if device_index is not None:
            logger.info(f"Using audio device index: {device_index}")

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.audio_config.channels,
            rate=self.audio_config.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.audio_config.chunk_size,
            stream_callback=self._audio_callback,
            start=False,  # Don't start immediately
        )
        logger.info("Audio stream opened (ready for low-latency recording)")

    def start(self):
        """Start recording audio."""
        if self.stream is None:
            self.open_stream()

        # Start stream if not already running (keep it running always)
        if not self.stream.is_active():
            self.stream.start_stream()

        with self.lock:
            self.buffer = []
            self.is_recording = True

        logger.info("Recording started")

    def stop(self) -> bytes:
        """Stop recording and return audio data."""
        with self.lock:
            self.is_recording = False
            audio_data = b"".join(self.buffer)
            self.buffer = []

        # Keep stream running for instant restart (no stop_stream call)
        logger.info(f"Recording stopped, captured {len(audio_data)} bytes")
        return audio_data

    def get_audio_as_numpy(self, audio_data: bytes) -> np.ndarray:
        """Convert raw audio bytes to numpy array for Whisper."""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Normalize to [-1, 1] range as float32
        return audio_array.astype(np.float32) / INT16_MAX_F

    def is_silent(self, audio_data: bytes, threshold: float = 0.01) -> bool:
        """Check if audio is mostly silence based on RMS energy.

        Args:
            audio_data: Raw audio bytes
            threshold: RMS threshold below which audio is considered silent
                       (0.01 works well for typical microphone input)

        Returns:
            True if audio is silent, False if speech detected
        """
        audio = self.get_audio_as_numpy(audio_data)
        rms = np.sqrt(np.mean(audio**2))
        logger.debug(f"Audio RMS energy: {rms:.4f} (threshold: {threshold})")
        return rms < threshold

    def list_devices(self) -> list[dict]:
        """List available audio input devices."""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                        "sample_rate": int(info["defaultSampleRate"]),
                    }
                )
        return devices

    def close(self):
        """Clean up PyAudio resources."""
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None
