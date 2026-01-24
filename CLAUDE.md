# WhisperTyper

Speech-to-text dictation service for Linux using Whisper and Ollama.

## Architecture

- **service.py** - Main orchestration, state machine (IDLE → RECORDING → PROCESSING)
- **hotkey.py** - evdev-based hotkey detection (Win+Alt, Ctrl+Alt, Page Down combos)
- **recorder.py** - PyAudio recording with always-running stream for low latency
- **transcriber.py** - Whisper ASR via HuggingFace transformers (distil-whisper/distil-large-v3)
- **processor.py** - Ollama integration for grammar/spelling correction
- **typer.py** - xdotool + xclip for clipboard paste output
- **mcp_server.py** - MCP server for external control from Claude Code
- **notifier.py** - Desktop notifications via libnotify

## Key Design Decisions

- **Clipboard paste** instead of character-by-character typing (works with Dvorak)
- **Always-running audio stream** for instant recording start (~10ms latency)
- **Silence detection** via RMS energy to prevent Whisper hallucinations
- **State file sync** (`~/.cache/whisper-typer/state.json`) for MCP control

## Running

```bash
# Development
cd src && python3 -m whisper_typer --verbose

# Service
systemctl --user start whisper-typer.service
journalctl --user -u whisper-typer.service -f
```

## CLI Options

- `--mode ollama|whisper|both` - Output mode (default: ollama)
- `--no-ollama` - Disable Ollama entirely
- `--verbose` - Show debug logs

## Configuration

Edit `config.yaml`:
- `hotkey.combo` - Primary hotkey (default: Win+Alt)
- `hotkey.alt_combos` - Additional hotkey combinations
- `ollama.model` - LLM model for correction (default: qwen2.5:1.5b)
- `ollama.host` - Ollama API endpoint (default: http://127.0.0.1:11434)

## MCP Tools

When loaded in Claude Code, provides:
- `whisper_set_mode` - Set output mode
- `whisper_enable_ollama` / `whisper_disable_ollama`
- `whisper_get_status` - Current configuration
- `whisper_get_recent` - Recent transcriptions
