"""MCP Server for Whisper Input control."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

# Shared state file for IPC with the dictation service
STATE_FILE = Path.home() / ".cache" / "whisper-input" / "state.json"

# Default state
DEFAULT_STATE = {
    "output_mode": "ollama_only",  # ollama_only, whisper_only, both
    "ollama_enabled": True,
    "recent_transcriptions": [],
}


def ensure_state_file():
    """Ensure state file exists with defaults."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        STATE_FILE.write_text(json.dumps(DEFAULT_STATE, indent=2))


def read_state() -> dict:
    """Read current state."""
    ensure_state_file()
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return DEFAULT_STATE.copy()


def write_state(state: dict):
    """Write state to file."""
    ensure_state_file()
    STATE_FILE.write_text(json.dumps(state, indent=2))


def update_state(**kwargs):
    """Update specific state values."""
    state = read_state()
    state.update(kwargs)
    write_state(state)
    return state


# Create MCP server
server = Server("whisper-input")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="whisper_set_mode",
            description="Set the output mode for Whisper Input. Modes: 'ollama' (corrected text only), 'whisper' (raw transcription only), 'both' (corrected + [raw] in brackets)",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["ollama", "whisper", "both"],
                        "description": "Output mode to set",
                    }
                },
                "required": ["mode"],
            },
        ),
        Tool(
            name="whisper_enable_ollama",
            description="Enable Ollama processing for grammar/spelling correction",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="whisper_disable_ollama",
            description="Disable Ollama processing, use raw Whisper output only",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="whisper_get_status",
            description="Get current Whisper Input status and configuration",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="whisper_get_recent",
            description="Get recent transcriptions",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "Number of recent transcriptions to return (default: 5)",
                        "default": 5,
                    }
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "whisper_set_mode":
        mode = arguments.get("mode", "ollama")
        mode_map = {
            "ollama": "ollama_only",
            "whisper": "whisper_only",
            "both": "both",
        }
        state = update_state(output_mode=mode_map.get(mode, "ollama_only"))
        return [TextContent(type="text", text=f"Output mode set to: {mode}")]

    elif name == "whisper_enable_ollama":
        state = update_state(ollama_enabled=True, output_mode="ollama_only")
        return [TextContent(type="text", text="Ollama enabled. Mode set to: ollama")]

    elif name == "whisper_disable_ollama":
        state = update_state(ollama_enabled=False, output_mode="whisper_only")
        return [TextContent(type="text", text="Ollama disabled. Mode set to: whisper")]

    elif name == "whisper_get_status":
        state = read_state()
        mode_display = state.get("output_mode", "unknown").replace("_", " ").title()
        status = f"""Whisper Input Status:
- Output Mode: {mode_display}
- Ollama Enabled: {state.get('ollama_enabled', False)}
- Recent Transcriptions: {len(state.get('recent_transcriptions', []))}"""
        return [TextContent(type="text", text=status)]

    elif name == "whisper_get_recent":
        count = arguments.get("count", 5)
        state = read_state()
        recent = state.get("recent_transcriptions", [])[-count:]
        if not recent:
            return [TextContent(type="text", text="No recent transcriptions.")]
        text = "Recent transcriptions:\n" + "\n".join(f"- {t}" for t in recent)
        return [TextContent(type="text", text=text)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
