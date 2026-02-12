"""MCP Server for WhisperTyper control using FastMCP."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from fastmcp import FastMCP

if TYPE_CHECKING:
    from .service import DictationService

# Shared state file for IPC (fallback when service not connected)
STATE_FILE = Path.home() / ".cache" / "whisper-typer" / "state.json"

# Reference to the running service (set when embedded in service)
_service: Optional["DictationService"] = None

# Default state
DEFAULT_STATE = {
    "output_mode": "ollama_only",
    "ollama_enabled": True,
    "recent_transcriptions": [],
}


def set_service(service: "DictationService"):
    """Set the service reference for direct state access."""
    global _service
    _service = service


def ensure_state_file():
    """Ensure state file exists with defaults."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps(DEFAULT_STATE, indent=2))


def read_state() -> dict:
    """Read current state (from service if available, else file)."""
    if _service is not None:
        return {
            "output_mode": _service.output_mode.value,
            "ollama_enabled": _service.ollama_enabled,
            "recent_transcriptions": getattr(_service, "_recent_transcriptions", []),
        }
    ensure_state_file()
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return DEFAULT_STATE.copy()


def write_state(state: dict):
    """Write state (to service if available, else file)."""
    if _service is not None:
        from .service import OutputMode
        mode_map = {
            "ollama_only": OutputMode.OLLAMA_ONLY,
            "whisper_only": OutputMode.WHISPER_ONLY,
            "both": OutputMode.BOTH,
        }
        _service.output_mode = mode_map.get(state.get("output_mode", "ollama_only"), OutputMode.OLLAMA_ONLY)
        _service.ollama_enabled = state.get("ollama_enabled", True)
        return

    ensure_state_file()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def update_state(**kwargs):
    """Update specific state values."""
    state = read_state()
    state.update(kwargs)
    write_state(state)
    return state


# Create FastMCP server
mcp = FastMCP("whisper-typer")


@mcp.tool()
def whisper_set_mode(mode: Literal["ollama", "whisper", "both"]) -> str:
    """Set the output mode for WhisperTyper.

    Args:
        mode: Output mode - 'ollama' (corrected text only), 'whisper' (raw transcription only), 'both' (corrected + [raw] in brackets)
    """
    mode_map = {
        "ollama": "ollama_only",
        "whisper": "whisper_only",
        "both": "both",
    }
    update_state(output_mode=mode_map.get(mode, "ollama_only"))
    return f"Output mode set to: {mode}"


@mcp.tool()
def whisper_enable_ollama() -> str:
    """Enable Ollama processing for grammar/spelling correction."""
    update_state(ollama_enabled=True, output_mode="ollama_only")
    return "Ollama enabled. Mode set to: ollama"


@mcp.tool()
def whisper_disable_ollama() -> str:
    """Disable Ollama processing, use raw Whisper output only."""
    update_state(ollama_enabled=False, output_mode="whisper_only")
    return "Ollama disabled. Mode set to: whisper"


@mcp.tool()
def whisper_get_status() -> str:
    """Get current WhisperTyper status and configuration."""
    state = read_state()
    mode_display = state.get("output_mode", "unknown").replace("_", " ").title()
    return f"""WhisperTyper Status:
- Output Mode: {mode_display}
- Ollama Enabled: {state.get('ollama_enabled', False)}
- Recent Transcriptions: {len(state.get('recent_transcriptions', []))}"""


@mcp.tool()
def whisper_get_recent(count: int = 5) -> str:
    """Get recent transcriptions.

    Args:
        count: Number of recent transcriptions to return (default: 5)
    """
    state = read_state()
    recent = state.get("recent_transcriptions", [])[-count:]
    if not recent:
        return "No recent transcriptions."
    return "Recent transcriptions:\n" + "\n".join(f"- {t}" for t in recent)


@mcp.tool()
def whisper_get_daily_report(date: str = "today") -> str:
    """Get Markdown productivity report for a specific date.

    Args:
        date: Date to get report for - 'today' (default), 'list' (show available dates), or YYYY-MM-DD format
    """
    from .history import generate_report, list_available_dates

    if date == "list":
        dates = list_available_dates()
        if not dates:
            return "No history records found."
        return "Available dates:\n" + "\n".join(f"- {d}" for d in dates)

    return generate_report(date)


# ─── code_speaker TTS tools ────────────────────────────────────────────

@mcp.tool()
def code_speaker_speak(text: str) -> str:
    """Speak text aloud using Kokoro TTS.

    Args:
        text: The text to speak aloud
    """
    import httpx
    try:
        port = 8767
        if _service and hasattr(_service, "tts_config"):
            port = _service.tts_config.api_port
        resp = httpx.post(
            f"http://localhost:{port}/speak",
            json={"text": text, "summarize": False, "event_type": "manual"},
            timeout=5.0,
        )
        return f"Speaking: {text[:80]}..."
    except Exception as e:
        return f"TTS error: {e}"


@mcp.tool()
def code_speaker_set_voice(voice: str) -> str:
    """Set the TTS voice for code_speaker.

    Args:
        voice: Voice name (e.g., 'af_heart', 'bf_emma', 'am_adam')
    """
    if _service and hasattr(_service, "kokoro_tts"):
        _service.kokoro_tts.config.voice = voice
        return f"Voice set to: {voice}"
    state = read_state()
    state["tts_voice"] = voice
    write_state(state)
    return f"Voice set to: {voice}"


@mcp.tool()
def code_speaker_enable() -> str:
    """Enable code_speaker TTS output."""
    if _service and hasattr(_service, "tts_config"):
        _service.tts_config.enabled = True
        return "Code Speaker TTS enabled"
    state = read_state()
    state["tts_enabled"] = True
    write_state(state)
    return "Code Speaker TTS enabled"


@mcp.tool()
def code_speaker_disable() -> str:
    """Disable code_speaker TTS output."""
    if _service and hasattr(_service, "tts_config"):
        _service.tts_config.enabled = False
        return "Code Speaker TTS disabled"
    state = read_state()
    state["tts_enabled"] = False
    write_state(state)
    return "Code Speaker TTS disabled"


@mcp.tool()
def code_speaker_voices() -> str:
    """List available TTS voices for code_speaker."""
    if _service and hasattr(_service, "kokoro_tts"):
        voices = _service.kokoro_tts.list_voices()
        if voices:
            return "Available voices:\n" + "\n".join(f"- {v}" for v in voices)
    return (
        "Available voice prefixes:\n"
        "- af_* (American female): af_heart, af_bella, af_nova, af_sarah\n"
        "- am_* (American male): am_adam, am_michael, am_echo\n"
        "- bf_* (British female): bf_emma, bf_alice, bf_lily\n"
        "- bm_* (British male): bm_george, bm_lewis\n"
        "Use code_speaker_set_voice to change."
    )


@mcp.tool()
def code_speaker_report(date: str = "today") -> str:
    """Get unified Voice I/O report (STT + TTS statistics).

    Args:
        date: Date for report - 'today' (default), 'list', or YYYY-MM-DD
    """
    from .code_speaker.history import generate_tts_report, list_tts_dates

    if date == "list":
        dates = list_tts_dates()
        if not dates:
            return "No TTS history records found."
        return "Available dates:\n" + "\n".join(f"- {d}" for d in dates)

    # Get both STT and TTS reports
    from .history import generate_report
    stt_report = generate_report(date)
    tts_report = generate_tts_report(date)

    return f"{stt_report}\n\n{'='*50}\n\n{tts_report}"


if __name__ == "__main__":
    # When run standalone (e.g., by Claude Code), use stdio transport
    # When embedded in service, use SSE transport (set by service.py)
    mcp.run(transport="stdio")
