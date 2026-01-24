# WhisperTyper

Speech-to-text dictation service for Linux using Whisper and Ollama.

Press a hotkey, speak, and your words are transcribed and typed into any application.

## Features

- **Instant recording** - Always-running audio stream for ~10ms latency
- **Whisper transcription** - Uses distil-whisper/distil-large-v3 for fast, accurate ASR
- **Ollama correction** - Optional grammar/spelling correction via local LLM
- **Clipboard paste** - Works with any keyboard layout (Dvorak, Colemak, etc.)
- **MCP integration** - Control from Claude Code via MCP server
- **Desktop notifications** - Visual feedback via libnotify

## Installation

```bash
# Clone the repository
git clone https://github.com/Mizzlr/whisper-typer.git
cd whisper-typer

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev python3-gi xdotool xclip libnotify-bin

# For evdev hotkey detection (required)
sudo usermod -aG input $USER
# Log out and back in for group change to take effect
```

## Usage

### Run directly

```bash
# Activate venv
source .venv/bin/activate

# Run with default settings
python -m whisper_typer

# Run with verbose logging
python -m whisper_typer --verbose

# Run with specific output mode
python -m whisper_typer --mode whisper    # Raw transcription only
python -m whisper_typer --mode ollama     # Ollama-corrected only (default)
python -m whisper_typer --mode both       # Show both outputs
```

### Run as systemd service

```bash
# Copy service file
mkdir -p ~/.config/systemd/user
cp systemd/whisper-input.service ~/.config/systemd/user/whisper-typer.service

# Edit paths if needed
# vim ~/.config/systemd/user/whisper-typer.service

# Enable and start
systemctl --user daemon-reload
systemctl --user enable whisper-typer.service
systemctl --user start whisper-typer.service

# View logs
journalctl --user -u whisper-typer.service -f
```

## Hotkeys

Default hotkey combinations (configurable in `config.yaml`):

| Hotkey | Description |
|--------|-------------|
| Win + Alt | Primary hotkey |
| Ctrl + Alt | Alternative |
| Page Down + Down Arrow | Alternative |

Press and hold to record, release to transcribe.

## Configuration

Create `config.yaml` in the working directory:

```yaml
hotkey:
  combo: win+alt
  alt_combos:
    - ctrl+alt
    - pagedown+down

ollama:
  model: qwen2.5:1.5b
  host: http://127.0.0.1:11434

whisper:
  model: distil-whisper/distil-large-v3
```

## MCP Integration

WhisperTyper includes an MCP server for control from Claude Code.

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "whisper-typer": {
      "command": "/path/to/whisper-typer/.venv/bin/python",
      "args": ["-m", "whisper_typer.mcp_server"]
    }
  }
}
```

Available MCP tools:
- `whisper_set_mode` - Set output mode (whisper/ollama/both)
- `whisper_enable_ollama` / `whisper_disable_ollama` - Toggle Ollama
- `whisper_get_status` - Get current configuration
- `whisper_get_recent` - Get recent transcriptions

## Architecture

```
whisper_typer/
├── service.py      # Main orchestration, state machine
├── hotkey.py       # evdev-based hotkey detection
├── recorder.py     # PyAudio recording with always-running stream
├── transcriber.py  # Whisper ASR via HuggingFace transformers
├── processor.py    # Ollama integration for correction
├── typer.py        # xdotool + xclip for clipboard paste
├── mcp_server.py   # MCP server for external control
└── notifier.py     # Desktop notifications
```

State machine: `IDLE → RECORDING → PROCESSING → IDLE`

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for Whisper)
- Ollama running locally (optional, for text correction)
- X11 display server (for xdotool/xclip)

## License

MIT
