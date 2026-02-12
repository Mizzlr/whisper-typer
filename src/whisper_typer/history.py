"""History tracking and productivity reporting for WhisperTyper."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# History directory in home folder
HISTORY_DIR = Path.home() / ".whisper-typer-history"


@dataclass
class TranscriptionRecord:
    """Record of a single transcription."""
    timestamp: str  # ISO 8601 format
    whisper_text: str
    ollama_text: Optional[str]
    final_text: str
    output_mode: str
    whisper_latency_ms: int
    ollama_latency_ms: Optional[int]
    typing_latency_ms: int
    total_latency_ms: int
    audio_duration_s: float
    char_count: int
    word_count: int
    speed_ratio: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptionRecord":
        """Create from dictionary."""
        return cls(**data)


def _get_history_file(date: str = "today") -> Path:
    """Get the history file path for a given date."""
    if date == "today":
        date = datetime.now().strftime("%Y-%m-%d")
    return HISTORY_DIR / f"{date}.jsonl"


def save_record(record: TranscriptionRecord) -> None:
    """Append a transcription record to the daily history file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_file = _get_history_file("today")

    try:
        with open(history_file, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        logger.debug(f"Saved transcription record to {history_file}")
    except Exception as e:
        logger.error(f"Failed to save history record: {e}")


def load_records(date: str = "today") -> list[TranscriptionRecord]:
    """Load all transcription records for a given date."""
    history_file = _get_history_file(date)

    if not history_file.exists():
        return []

    records = []
    try:
        with open(history_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(TranscriptionRecord.from_dict(data))
    except Exception as e:
        logger.error(f"Failed to load history records: {e}")

    return records


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _truncate(text: str, max_len: int = 50) -> str:
    """Truncate text for table display."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def generate_report(date: str = "today") -> str:
    """Generate a Markdown productivity report for a given date."""
    records = load_records(date)

    if date == "today":
        display_date = datetime.now().strftime("%Y-%m-%d")
    else:
        display_date = date

    if not records:
        return f"# WhisperTyper Report - {display_date}\n\nNo transcriptions recorded."

    # Calculate statistics
    total_chars = sum(r.char_count for r in records)
    total_words = sum(r.word_count for r in records)
    total_audio = sum(r.audio_duration_s for r in records)
    total_processing = sum(r.total_latency_ms for r in records) / 1000

    whisper_latencies = [r.whisper_latency_ms for r in records]
    ollama_latencies = [r.ollama_latency_ms for r in records if r.ollama_latency_ms]
    typing_latencies = [r.typing_latency_ms for r in records]

    avg_whisper = sum(whisper_latencies) / len(whisper_latencies) if whisper_latencies else 0
    avg_ollama = sum(ollama_latencies) / len(ollama_latencies) if ollama_latencies else 0
    avg_typing = sum(typing_latencies) / len(typing_latencies) if typing_latencies else 0
    avg_speed = sum(r.speed_ratio for r in records) / len(records) if records else 0

    # Build report
    lines = [
        f"# WhisperTyper Report - {display_date}",
        "",
        "## Summary",
        f"- **Transcriptions**: {len(records)}",
        f"- **Total characters**: {total_chars:,}",
        f"- **Total words**: {total_words:,}",
        f"- **Total audio**: {_format_duration(total_audio)}",
        f"- **Total processing time**: {_format_duration(total_processing)}",
        f"- **Average speed ratio**: {avg_speed:.1f}x",
        "",
        "## Latency Averages",
        f"- Whisper: {avg_whisper:.0f}ms",
    ]

    if ollama_latencies:
        lines.append(f"- Ollama: {avg_ollama:.0f}ms")
    lines.append(f"- Typing: {avg_typing:.0f}ms")

    # Transcription log table
    lines.extend([
        "",
        "## Transcription Log",
        "",
        "| Time | Whisper | Ollama | Chars | Speed |",
        "|------|---------|--------|-------|-------|",
    ])

    for r in records:
        # Extract time from timestamp
        try:
            ts = datetime.fromisoformat(r.timestamp)
            time_str = ts.strftime("%H:%M:%S")
        except Exception:
            time_str = r.timestamp[:8]

        whisper_display = _truncate(r.whisper_text, 30)
        if r.ollama_text and r.ollama_text != r.whisper_text:
            ollama_display = _truncate(r.ollama_text, 30)
        else:
            ollama_display = "-"

        lines.append(
            f"| {time_str} | {whisper_display} | {ollama_display} | {r.char_count} | {r.speed_ratio:.1f}x |"
        )

    return "\n".join(lines)


def list_available_dates() -> list[str]:
    """List all dates with history records."""
    if not HISTORY_DIR.exists():
        return []

    dates = []
    for f in HISTORY_DIR.glob("*.jsonl"):
        # Extract date from filename (YYYY-MM-DD.jsonl)
        date = f.stem
        dates.append(date)

    return sorted(dates, reverse=True)
