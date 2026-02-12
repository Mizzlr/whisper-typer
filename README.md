# WhisperTyper

Speech-to-text dictation service for Linux using Whisper and Ollama.

Press a hotkey (or say a wake word), speak, and your words are transcribed and typed into any application.

## Features

- **Instant recording** - Always-running audio stream for ~10ms startup latency
- **Whisper transcription** - Uses distil-whisper/distil-large-v3 for fast, accurate ASR
- **Ollama correction** - Optional grammar/spelling correction via local LLM
- **Wake word activation** - Hands-free recording with OpenWakeWord (e.g., "Hey Jarvis")
- **Silence detection** - Auto-stops recording after speech ends (wake word mode)
- **Hallucination filtering** - Blocks common Whisper phantom outputs ("Thank you", "Bye", etc.)
- **Multiple typing backends** - ydotool, dotool, or xdotool with auto-detection
- **Code Speaker TTS** - Text-to-speech output via Kokoro ONNX for Claude Code events
- **MCP integration** - Control from Claude Code via MCP server
- **Productivity reports** - Daily transcription history with latency breakdowns
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

# Enable wake word activation
python -m whisper_typer --enable-wakeword
python -m whisper_typer --enable-wakeword --wakeword-model hey_jarvis

# Enable Code Speaker TTS
python -m whisper_typer --enable-tts --tts-voice af_heart

# View productivity report
python -m whisper_typer --report            # Today's report
python -m whisper_typer --report 2026-02-12  # Specific date
python -m whisper_typer --report list        # List available dates
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
| Page Down + Right Arrow | Alternative |
| Page Down + Down Arrow | Alternative |

Press and hold to record, release to transcribe.

## Configuration

Create `config.yaml` in the working directory:

```yaml
hotkey:
  combo: [KEY_LEFTMETA, KEY_LEFTALT]
  alt_combos:
    - [KEY_LEFTCTRL, KEY_LEFTALT]
    - [KEY_PAGEDOWN, KEY_DOWN]

audio:
  sample_rate: 16000
  # device_index: null  # null = default device

whisper:
  model: distil-whisper/distil-large-v3
  device: cuda  # cuda, cpu, or mps

ollama:
  enabled: true
  model: qwen2.5:1.5b
  host: http://127.0.0.1:11434

typer:
  backend: auto  # auto, ydotool, dotool, or xdotool

wakeword:
  enabled: false
  model: hey_jarvis
  threshold: 0.5

silence:
  threshold: 0.01
  duration: 1.5

tts:
  enabled: false
  voice: af_heart
  speed: 1.0
  api_port: 8767
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
- `whisper_report` - View productivity reports
- `code_speaker_speak` - Speak text aloud via Kokoro TTS
- `code_speaker_set_voice` - Change TTS voice
- `code_speaker_report` - View TTS latency reports

## Architecture

```
                    ┌─────────────┐
Wake Word ─────────▶│             │
(OpenWakeWord)      │             │
                    │   Service   │──▶ Notifier (libnotify)
Hotkey ────────────▶│   (state    │
(evdev)             │   machine)  │──▶ MCP Server (FastMCP)
                    │             │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Recorder      Ollama       Typer
         (PyAudio)     Client      (ydotool/
              │        (httpx)      xdotool)
              ▼            │
         Transcriber       ├──▶ Processor (grammar)
         (Whisper)         └──▶ Summarizer (TTS)
                                     │
                                     ▼
                               Code Speaker
                               (Kokoro TTS)
```

### State Machine

```
                    ┌──(hotkey pressed)──▶ RECORDING ──(hotkey released)──┐
                    │                                                     │
IDLE ◀──(done)──────┤                                                     ├──▶ PROCESSING
                    │                                                     │
                    └──(wake word)──▶ WAKE_RECORDING ──(silence)─────────┘
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| Service | `service.py` | Main orchestration and state machine |
| Ollama Client | `ollama_client.py` | Shared async HTTP client for Ollama API |
| Hotkey | `hotkey.py` | evdev-based global hotkey detection |
| Recorder | `recorder.py` | PyAudio with always-running stream, subscriber pattern |
| Transcriber | `transcriber.py` | Whisper ASR via HuggingFace transformers |
| Processor | `processor.py` | Ollama grammar/spelling correction |
| Typer | `typer.py` | Multi-backend text typing (ydotool/dotool/xdotool) |
| Wake Word | `wakeword.py` | OpenWakeWord-based voice activation |
| Silence Detector | `silence_detector.py` | RMS-based silence detection for auto-stop |
| MCP Server | `mcp_server.py` | FastMCP server for Claude Code control |
| Notifier | `notifier.py` | Desktop notifications via libnotify |
| History | `history.py` | Transcription logging and productivity reports |
| TTS Engine | `code_speaker/tts.py` | Kokoro ONNX text-to-speech |
| TTS API | `code_speaker/api.py` | HTTP API server for TTS requests |
| Summarizer | `code_speaker/summarizer.py` | Ollama text summarization for TTS |
| Reminder | `code_speaker/reminder.py` | Escalating TTS reminders |
| Config | `config.py` | Dataclass-based YAML configuration |

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for Whisper)
- Ollama running locally (optional, for text correction)
- X11 display server (for xdotool/xclip) or Wayland (for ydotool/dotool)

## Roadmap

### Streaming Transcription

Currently, transcription happens after the hotkey is released. Future work could add real-time streaming output as you speak.

| Approach | Library | Latency | Notes |
|----------|---------|---------|-------|
| VAD-based chunking | [silero-vad](https://github.com/snakers4/silero-vad) + [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ~1-2s | Transcribe on natural pauses |
| LocalAgreement | [whisper_streaming](https://github.com/ufal/whisper_streaming) | ~3.3s | Emit when 2+ iterations agree |
| SimulStreaming | [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) | <1s | SOTA 2025, AlignAtt policy |

**Recommended starting point**: Silero-VAD + faster-whisper, since we already have silence detection infrastructure.

## License

MIT
