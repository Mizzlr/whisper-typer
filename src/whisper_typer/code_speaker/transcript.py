"""Claude Code JSONL transcript parser."""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_last_assistant_text(
    transcript_path: str, max_chars: int = 2000
) -> Optional[str]:
    """Extract the last assistant text message from a Claude Code JSONL transcript.

    Walks the file backwards to find the most recent assistant message
    with text content.
    """
    try:
        with open(transcript_path) as f:
            lines = f.readlines()

        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if entry.get("type") == "assistant":
                    content = entry.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            text = block["text"][:max_chars]
                            return text
            except (json.JSONDecodeError, KeyError):
                continue

    except FileNotFoundError:
        logger.warning(f"Transcript not found: {transcript_path}")
    except Exception as e:
        logger.warning(f"Failed to parse transcript: {e}")

    return None


def extract_last_user_timestamp(transcript_path: str) -> Optional[str]:
    """Extract the timestamp of the last user message (for Claude session duration)."""
    try:
        with open(transcript_path) as f:
            lines = f.readlines()

        for line in reversed(lines):
            try:
                entry = json.loads(line)
                if entry.get("type") == "user" and "timestamp" in entry:
                    return entry["timestamp"]
            except (json.JSONDecodeError, KeyError):
                continue

    except Exception as e:
        logger.warning(f"Failed to extract user timestamp: {e}")

    return None
