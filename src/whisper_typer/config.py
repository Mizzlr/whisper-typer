"""Configuration management for Whisper Input."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class HotkeyConfig:
    combo: list[str] = field(default_factory=lambda: ["KEY_LEFTMETA", "KEY_LEFTALT"])
    alt_combos: list[list[str]] = field(default_factory=list)  # Alternative hotkey combos


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    device_index: Optional[int] = None
    channels: int = 1
    chunk_size: int = 1024


@dataclass
class RecordingConfig:
    max_duration: float = 120.0


@dataclass
class WhisperConfig:
    model: str = "distil-whisper/distil-large-v3"
    device: str = "cuda"


@dataclass
class OllamaConfig:
    enabled: bool = True
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"


@dataclass
class TyperConfig:
    backend: str = "ydotool"


@dataclass
class FeedbackConfig:
    notifications: bool = True
    sounds: bool = False


@dataclass
class WakeWordConfig:
    enabled: bool = False
    model: str = "hey_jarvis"
    threshold: float = 0.5


@dataclass
class SilenceConfig:
    threshold: float = 0.01
    duration: float = 1.5
    min_speech_duration: float = 0.5
    max_recording_duration: float = 30.0


@dataclass
class TTSConfig:
    enabled: bool = False
    voice: str = "af_heart"
    speed: float = 1.0
    api_port: int = 8767
    max_direct_chars: int = 150
    reminder_interval: int = 300
    model_path: str = ""  # Empty = auto-detect in working dir


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    typer: TyperConfig = field(default_factory=TyperConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    silence: SilenceConfig = field(default_factory=SilenceConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration from YAML file."""
        if path is None:
            # Look for config in standard locations
            candidates = [
                Path.cwd() / "config.yaml",
                Path.home() / ".config" / "whisper-input" / "config.yaml",
                Path("/etc/whisper-input/config.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break

        if path is None or not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            hotkey=HotkeyConfig(**data.get("hotkey", {})),
            audio=AudioConfig(**data.get("audio", {})),
            recording=RecordingConfig(**data.get("recording", {})),
            whisper=WhisperConfig(**data.get("whisper", {})),
            ollama=OllamaConfig(**data.get("ollama", {})),
            typer=TyperConfig(**data.get("typer", {})),
            feedback=FeedbackConfig(**data.get("feedback", {})),
            wakeword=WakeWordConfig(**data.get("wakeword", {})),
            silence=SilenceConfig(**data.get("silence", {})),
            tts=TTSConfig(**data.get("tts", {})),
        )
