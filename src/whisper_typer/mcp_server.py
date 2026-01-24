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


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    mcp.run(transport="sse", port=port)
