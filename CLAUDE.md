# WhisperTyper

Speech-to-text dictation service for Linux using Whisper and Ollama.

## IMPORTANT: Python version is DEPRECATED — DO NOT MODIFY

The Python implementation (`src/whisper_typer/`) is **fully deprecated and must not be modified**. All code changes go in the **Rust rewrite** at `whisper-typer-rs/`. The Python code exists only as historical reference.

| | Python (deprecated) | Rust (active) |
|---|---|---|
| Service | `whisper-typer.service` | `whisper-typer-rs.service` |
| Binary | `python -m whisper_typer` | `whisper-typer-rs/target/release/whisper-typer-rs` |
| MCP server | `mcp_server.py` (FastMCP/SSE) | `mcp_server.rs` (rmcp/SSE on port 8766) |
| TTS | External Kokoro process | Native Kokoro ONNX (`code_speaker/`) |

## Architecture (Rust)

- **service.rs** - Main orchestration, state machine (IDLE → RECORDING → PROCESSING)
- **hotkey.rs** - evdev-based hotkey detection (Win+Alt, Ctrl+Alt, Page Down combos)
- **recorder.rs** - cpal-based audio recording with always-running stream
- **transcriber.rs** - Whisper ASR via whisper-rs (ggml-distil-large-v3.bin, CUDA)
- **processor.rs** - Ollama integration for grammar/spelling correction
- **typer.rs** - xdotool + xclip/arboard for clipboard paste output
- **mcp_server.rs** - MCP server using rmcp (SSE transport on port 8766)
- **code_speaker/** - Native Kokoro TTS (ONNX, port 8767)
- **history.rs** - JSONL transcription history and productivity reports
- **bin/tts_hook.rs** - Claude Code hook binary for TTS events

## Key Design Decisions

- **Clipboard paste** instead of character-by-character typing (works with Dvorak)
- **Always-running audio stream** for instant recording start (~10ms latency)
- **Silence detection** via RMS energy to prevent Whisper hallucinations
- **State file sync** (`~/.cache/whisper-typer/state.json`) for MCP control
- **Native TTS** via Kokoro ONNX (no external process needed)

## Running

```bash
# Development (Rust)
cd whisper-typer-rs && cargo run --release

# Service
systemctl --user start whisper-typer-rs.service
journalctl --user -u whisper-typer-rs -f
```

## Configuration

Edit `config.yaml`:
- `hotkey.combo` - Primary hotkey (default: Win+Alt)
- `hotkey.alt_combos` - Additional hotkey combinations
- `ollama.model` - LLM model for correction (default: qwen2.5:1.5b)
- `ollama.host` - Ollama API endpoint (default: http://127.0.0.1:11434)

## MCP Tools

When loaded in Claude Code (SSE on `http://localhost:8766/sse`), provides:
- `whisper_set_mode` - Set output mode
- `whisper_enable_ollama` / `whisper_disable_ollama`
- `whisper_get_status` - Current configuration
- `whisper_get_recent` - Recent transcriptions
- `whisper_get_daily_report` - Productivity reports from JSONL history
- `code_speaker_speak` / `code_speaker_set_voice` / `code_speaker_enable` / `code_speaker_disable`
- `code_speaker_voices` / `code_speaker_report`
- `whisper_teach` / `whisper_add_correction`

## Known Issues

- **HTTP 410 from MCP**: rmcp SSE sessions expire when Claude Code reconnects (e.g., after `/clear` or new session). The stale session ID returns 410 "Gone". Workaround: restart `whisper-typer-rs.service`.
- **Verbose whisper_init_state logs**: Each transcription emits 7 lines of buffer allocation info from the whisper.cpp backend. This is cosmetic noise, not an error.
