# whisper-typer-rs

Rust port of WhisperTyper — speech-to-text dictation service for Linux using Whisper and Ollama, with native Kokoro TTS.

## Architecture

```
src/
├── main.rs             # CLI entry, async runtime setup
├── config.rs           # YAML config loading
├── service.rs          # State machine: IDLE → RECORDING → PROCESSING
├── hotkey.rs           # evdev global hotkey detection
├── recorder.rs         # cpal always-open audio stream (~10ms latency)
├── transcriber.rs      # whisper-rs ASR (GGML, GPU via CUDA)
├── processor.rs        # Ollama grammar/spelling correction
├── typer.rs            # Clipboard paste via arboard + enigo
├── history.rs          # JSONL transcription records
├── notifier.rs         # D-Bus desktop notifications
├── mcp_server.rs       # MCP server (SSE transport, port 8766)
└── code_speaker/       # Native Kokoro TTS
    ├── tts.rs          # ONNX inference (ort, CUDA) + rodio playback
    ├── api.rs          # HTTP API server (port 8767)
    ├── summarizer.rs   # Ollama text summarization for long TTS
    ├── reminder.rs     # Periodic reminder manager
    ├── transcript.rs   # Transcript tracking
    └── history.rs      # TTS event history (JSONL)
```

## Build

```bash
# CPU only
cargo build --release

# With CUDA (Whisper GPU + Kokoro TTS GPU)
cargo build --release --features cuda
```

Requires: CUDA toolkit, cuDNN 9 (for Kokoro ONNX CUDA EP).

## Run

```bash
# Direct
./target/release/whisper-typer-rs --mode ollama --verbose

# As systemd user service
systemctl --user start whisper-typer-rs.service
journalctl --user -u whisper-typer-rs -f
```

### CLI Options

| Flag | Description |
|---|---|
| `--config PATH` | Custom config.yaml path |
| `--mode ollama\|whisper\|both` | Output mode (default: ollama) |
| `--no-ollama` | Disable Ollama correction |
| `--verbose` | Debug logging |

## Configuration

Loaded from `./config.yaml` (or `~/.config/whisper-input/config.yaml`). Key sections:

```yaml
hotkey:
  combo: [KEY_LEFTMETA, KEY_LEFTALT]

whisper:
  model_path: models/ggml-distil-large-v3.bin

ollama:
  model: qwen2.5:1.5b
  host: http://127.0.0.1:11434

tts:
  enabled: true
  voice: am_michael
  speed: 1.0
  api_port: 8767

mcp:
  enabled: true
  port: 8766
```

## GPU Usage

All three inference workloads run on GPU:

| Component | Framework | GPU Memory |
|---|---|---|
| Whisper ASR | whisper-rs (ggml CUDA) | ~1519 MiB |
| Kokoro TTS | ort (ONNX CUDA EP) | ~524 MiB |
| Ollama LLM | ollama (external) | ~1920 MiB |

## MCP Tools

Exposed via SSE on port 8766 for Claude Code integration:

- `whisper_set_mode` / `whisper_get_status` / `whisper_get_recent`
- `whisper_enable_ollama` / `whisper_disable_ollama`
- `whisper_get_daily_report`
- `whisper_teach` — add vocabulary terms for Whisper bias
- `whisper_add_correction` — add post-transcription corrections
- `code_speaker_speak` / `code_speaker_set_voice` / `code_speaker_voices`
- `code_speaker_enable` / `code_speaker_disable` / `code_speaker_report`

## Per-Project Speech Config

See [../.whisper/README.md](../.whisper/README.md) for vocabulary and corrections setup.
