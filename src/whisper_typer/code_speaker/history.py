"""TTS history tracking and reporting for code_speaker."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HISTORY_DIR = Path.home() / ".code-speaker-history"


@dataclass
class TTSRecord:
    """Record of a single TTS event with per-stage latency."""
    timestamp: str
    event_type: str           # stop, permission, notification, manual
    input_text_chars: int     # Original text length
    summarized: bool          # Whether Ollama was used
    summary_text: str         # What was actually spoken
    ollama_latency_ms: int    # Summarization time (0 if not summarized)
    kokoro_latency_ms: int    # Audio generation time
    playback_duration_ms: int # Playback time
    total_latency_ms: int     # End-to-end TTS pipeline
    voice: str                # Kokoro voice used
    cancelled: bool = False
    claude_session_ms: Optional[int] = None  # Time Claude spent thinking
    reminder_count: int = 0   # Reminders fired before user responded


def save_tts_record(record: TTSRecord):
    """Save a TTS record to today's history file."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    filepath = HISTORY_DIR / f"{today}-tts.jsonl"

    with open(filepath, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_tts_records(target_date: str = "today") -> list[TTSRecord]:
    """Load TTS records for a given date."""
    if target_date == "today":
        target_date = date.today().isoformat()

    filepath = HISTORY_DIR / f"{target_date}-tts.jsonl"
    records = []

    if not filepath.exists():
        return records

    with open(filepath) as f:
        for line in f:
            try:
                data = json.loads(line)
                records.append(TTSRecord(**data))
            except (json.JSONDecodeError, TypeError):
                continue

    return records


def list_tts_dates() -> list[str]:
    """List dates with TTS history."""
    if not HISTORY_DIR.exists():
        return []

    dates = []
    for f in sorted(HISTORY_DIR.glob("*-tts.jsonl")):
        # Extract date from filename like "2026-02-12-tts.jsonl"
        name = f.stem.replace("-tts", "")
        dates.append(name)

    return dates


def generate_tts_report(target_date: str = "today") -> str:
    """Generate a TTS latency and usage report."""
    records = load_tts_records(target_date)

    if not records:
        return f"No TTS events recorded for {target_date}."

    if target_date == "today":
        target_date = date.today().isoformat()

    # Count by event type
    by_type = {}
    for r in records:
        by_type.setdefault(r.event_type, []).append(r)

    # Compute averages
    total = len(records)
    non_cancelled = [r for r in records if not r.cancelled]

    def avg(values):
        return sum(values) / len(values) if values else 0

    avg_ollama = avg([r.ollama_latency_ms for r in non_cancelled if r.summarized])
    avg_kokoro = avg([r.kokoro_latency_ms for r in non_cancelled])
    avg_playback = avg([r.playback_duration_ms for r in non_cancelled])
    avg_total = avg([r.total_latency_ms for r in non_cancelled])

    # Claude session duration (stop events only)
    claude_durations = [
        r.claude_session_ms for r in records
        if r.event_type == "stop" and r.claude_session_ms is not None
    ]
    avg_claude = avg(claude_durations)

    # Reminder stats
    reminder_counts = [r.reminder_count for r in records if r.reminder_count > 0]
    avg_reminders = avg(reminder_counts) if reminder_counts else 0

    # Build report
    lines = [
        f"=== Code Speaker TTS Report: {target_date} ===",
        "",
        f"Total events: {total}",
    ]

    for etype, recs in sorted(by_type.items()):
        lines.append(f"  {etype}: {len(recs)}")

    lines.extend([
        "",
        "--- Latency (non-cancelled events) ---",
        f"  Ollama summarize: {avg_ollama:.0f}ms avg"
        f" ({sum(1 for r in non_cancelled if r.summarized)} summarized)",
        f"  Kokoro generate:  {avg_kokoro:.0f}ms avg",
        f"  Playback:         {avg_playback:.0f}ms avg",
        f"  Total pipeline:   {avg_total:.0f}ms avg",
    ])

    if claude_durations:
        lines.extend([
            "",
            "--- Claude Session Duration ---",
            f"  Avg thinking time: {avg_claude/1000:.1f}s",
            f"  Events with timing: {len(claude_durations)}",
        ])

    if reminder_counts:
        lines.extend([
            "",
            "--- Reminders ---",
            f"  Events with reminders: {len(reminder_counts)}",
            f"  Avg reminders before response: {avg_reminders:.1f}",
            f"  Max reminders: {max(reminder_counts)}",
        ])

    # Cancelled events
    cancelled = [r for r in records if r.cancelled]
    if cancelled:
        lines.append(f"\nCancelled events: {len(cancelled)}")

    return "\n".join(lines)
