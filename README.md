# WhisperTyper RS

Speech-to-text dictation, voice journaling, and Claude-Code voice feedback for Linux. Pure Rust.

Press a hotkey, speak, and your words are transcribed (Whisper, CUDA) and pasted into the focused application — optionally grammar-corrected by a local LLM (Ollama). A separate TUI app records and filters a daily voice journal. Claude Code lifecycle events get spoken aloud through Kokoro TTS.

## What's in this repo

This repo builds three binaries from one Rust workspace:

| Binary | Purpose |
|---|---|
| `whisper-typer-rs` | Background dictation service: hotkey → Whisper → Ollama → paste. Hosts the MCP HTTP server (port 8766) and the Kokoro TTS HTTP API (port 8767). |
| `tts-hook` | Standalone binary invoked by Claude Code on lifecycle events (SessionStart, Stop, Notification, PermissionRequest, UserPromptSubmit). Speaks short status phrases via the TTS API. |
| `voice-journal` | TUI for personal voice journaling. Captures audio, runs VAD, transcribes via the running service's `/transcribe` endpoint, applies regex + LLM hallucination filtering, appends to `~/voice-journal/journal_YYYY-MM-DD.md`. |

## Architecture

```
                      ┌──────────────────┐
   Hotkey  ─evdev──▶  │                  │
   (Win+Alt etc.)     │  whisper-typer-rs│ ──▶ Whisper (CUDA, distil-large-v3)
                      │   state machine  │
   Claude Code ──┐    │   IDLE→REC→PROC  │ ──▶ Ollama /api/generate (gemma4:e2b)
   (MCP HTTP)    │    │                  │
                 ├──▶ │  ┌─────────────┐ │ ──▶ Typer (xdotool / arboard paste)
   Voice Journal │    │  │ MCP server  │ │
   (HTTP)        │    │  │ port 8766   │ │
                 │    │  └─────────────┘ │
                 │    │  ┌─────────────┐ │
   Claude Code ──┘    │  │ TTS API     │ │ ──▶ Kokoro ONNX → rodio playback
   (tts-hook bin)     │  │ port 8767   │ │
                      │  └─────────────┘ │
                      └──────────────────┘
```

## Installation

```bash
git clone https://github.com/Mizzlr/whisper-typer.git
cd whisper-typer
./infra/install.sh
```

`install.sh` does five things: adds you to the `input` group, installs the udev rule for `/dev/uinput`, runs `cargo build --release`, copies the three binaries to `~/.local/bin/`, and installs the systemd user service as `whisper-typer-rs.service`.

You'll need to:
- Log out and back in (for `input` group to take effect)
- `ollama pull gemma4:e2b` (default LLM)
- Place Whisper and Kokoro models under `models/` (paths are `models/ggml-distil-large-v3.bin` and `models/kokoro-v1.0.onnx` by default — see `config.yaml`)

System packages: a recent Rust toolchain, CUDA + cuDNN if you want GPU Whisper, `xdotool` and `xclip` for X11 paste, ALSA dev headers (`libasound2-dev`) for `cpal`.

## Running

```bash
# Service (after install)
systemctl --user start whisper-typer-rs
journalctl --user -fu whisper-typer-rs

# Dev (foreground, verbose)
cargo run --release -- --verbose

# Voice journal (interactive TUI)
voice-journal
# or
./target/release/voice-journal
```

## Hotkeys

Configured in `config.yaml`. Defaults:

| Hotkey | Notes |
|---|---|
| Win + Alt | Primary |
| Ctrl + Alt | Alternative |
| Right Ctrl + Right Alt | Alternative |
| Page Down + Right Arrow | Alternative |
| Page Down + Down Arrow | Alternative |

Press and hold to record, release to transcribe. Wake-word activation (`alexa` / `hey_jarvis`) is configurable but not currently implemented in the Rust path — the config is parsed and ignored.

## Configuration

`config.yaml` (repo root) controls hotkeys, audio capture, Whisper model, Ollama, typer backend, silence detection, and Kokoro TTS:

```yaml
ollama:
  enabled: true
  model: "gemma4:e2b"            # gemma4:e2b for grammar correction
  host: "http://127.0.0.1:11434"
  keep_alive: 3600
  skip_threshold: 5              # skip Ollama on utterances ≤ N words

whisper:
  model: "models/ggml-distil-large-v3.bin"
  device: "cuda"                 # cuda | cpu | mps

tts:
  enabled: true                  # native Kokoro TTS
  voice: "am_michael"            # any af_*, am_*, bf_*, bm_* preset
  speed: 1.0
  api_port: 8767
```

## MCP integration

The service exposes an MCP HTTP server on `http://localhost:8766/mcp` that Claude Code can connect to. Add to `.mcp.json`:

```json
{
  "mcpServers": {
    "whisper-typer": {
      "type": "http",
      "url": "http://localhost:8766/mcp"
    }
  }
}
```

Tools exposed:

| Tool | Purpose |
|---|---|
| `whisper_set_mode` | Switch output mode: `whisper` / `ollama` / `both` |
| `whisper_enable_ollama` / `whisper_disable_ollama` | Toggle grammar correction |
| `whisper_get_status` | Current mode + Ollama state |
| `whisper_get_recent` | Last N transcriptions |
| `whisper_get_daily_report` | Productivity report (WPM, latencies) |
| `whisper_teach` | Add term to vocabulary (`.whisper/vocabulary.txt`) — used as Whisper initial prompt |
| `whisper_add_correction` | Add wrong→right substitution (`.whisper/corrections.yaml`) — injected into Ollama prompt |
| `code_speaker_speak` | Enqueue text to TTS |
| `code_speaker_set_voice` | Persist Kokoro voice across restarts |
| `code_speaker_enable` / `code_speaker_disable` | Toggle TTS playback |
| `code_speaker_voices` | List available voice presets |
| `code_speaker_report` | Same daily report unified across STT + TTS |

## Voice Journal

`voice-journal` is a separate TUI binary tuned for long dictation sessions. It captures audio, runs VAD, sends each utterance to the running service's `/transcribe` endpoint (port 8767, reuses the loaded Whisper model — no second model load), and runs each transcribed chunk through a two-stage hallucination filter.

**VAD (voice activity detection)**

| Mode | Default | Trigger | Notes |
|---|---|---|---|
| RMS energy threshold | yes | always (default) | Cheap, distinguishes silence from sound but not speech from background ambient/podcast |
| Silero VAD ONNX (v5) | no | `WHISPER_VOICE_JOURNAL_VAD=silero` | 1.8 MB ONNX model, ~1ms/frame on CPU. Distinguishes intentional speech from background noise. Eliminates most podcast-bleed hallucinations at source. Requires `models/silero_vad.onnx` on disk; download from <https://github.com/snakers4/silero-vad>. Path overridable via `WHISPER_VAD_MODEL`. Falls back to RMS with stderr warning if the model can't be loaded. |

**Hallucination filter (two stages)**

1. **Regex pass** — fast, deterministic. Rules live in `~/voice-journal/hallucinations.txt`. Catches known echo patterns, podcast bleed, named-entity garble.
2. **LLM pass** — Ollama (gemma4:e2b) chat API with few-shot prompt. Catches novel hallucinations the regex doesn't know about. Runs in ~0.4s per chunk on a warm model. Auto-disables if Ollama is unreachable; can be force-disabled via `WHISPER_VOICE_JOURNAL_LLM=0`.

Filtered chunks appear inline with a `[filtered]` (regex) or `[filtered-llm]` (LLM) prefix and don't write to the journal file. Output: `~/voice-journal/journal_YYYY-MM-DD.md`.

## Layout

```
src/
├── main.rs                  # service entry point
├── service.rs               # state machine, voice gate
├── hotkey.rs                # evdev monitoring
├── recorder.rs              # cpal audio capture
├── transcriber.rs           # whisper-rs ASR
├── processor.rs             # Ollama grammar correction
├── typer.rs                 # arboard + enigo paste
├── mcp_server.rs            # rmcp HTTP server (port 8766)
├── history.rs               # JSONL transcription history
├── config.rs                # YAML loader
├── code_speaker/            # native Kokoro TTS
│   ├── mod.rs
│   ├── tts.rs               # ONNX inference + rodio playback
│   └── api.rs               # axum HTTP API (port 8767)
└── bin/
    ├── tts_hook.rs          # Claude Code lifecycle listener
    └── voice_journal.rs     # voice journal TUI

infra/
├── install.sh               # one-shot setup script
├── systemd/
│   └── whisper-typer.service
└── udev/
    └── 99-uinput.rules

models/                      # Whisper, Kokoro, Silero-VAD model files (gitignored)
config.yaml                  # runtime configuration
```

## Common operations

```bash
# Rebuild + redeploy after pulling
cargo build --release
install -m 0755 target/release/{whisper-typer-rs,tts-hook,voice-journal} ~/.local/bin/
systemctl --user restart whisper-typer-rs

# Check service health
systemctl --user status whisper-typer-rs
curl -s http://localhost:8766/mcp -X POST -d '{}' | head    # MCP probe
curl -s http://localhost:8767/status                        # TTS probe

# Tail TTS hook events
tail -f ~/.cache/whisper-typer/tts-hook.log
```

## Known issues

- **MCP HTTP 410 after `/clear`**: rmcp Streamable HTTP sessions expire when Claude Code reconnects. Workaround: `systemctl --user restart whisper-typer-rs`.
- **Wake word**: `wakeword:` block in `config.yaml` is parsed but no Rust module currently implements activation. Carryover from the Python era.
- **Verbose `whisper_init_state` startup**: 7 lines of GPU buffer allocation per service start. Cosmetic, not an error.

## License

MIT.
