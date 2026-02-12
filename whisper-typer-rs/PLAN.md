# whisper-typer-rs: Rust Port Implementation Plan

Phased plan for porting WhisperTyper from Python to Rust.
Crate location: `whisper-typer-rs/` subfolder (coexists with the Python codebase).

**Scope:** Full voice input pipeline (hotkey → record → transcribe → correct → paste).
Wake-word detection is excluded.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    whisper-typer-rs                       │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐            │
│  │  hotkey   │  │ recorder │  │ transcriber │            │
│  │  (evdev)  │  │  (cpal)  │  │(whisper-rs) │            │
│  └────┬─────┘  └────┬─────┘  └──────┬──────┘            │
│       │              │               │                   │
│       ▼              ▼               ▼                   │
│  ┌───────────────────────────────────────────┐           │
│  │               service.rs                  │           │
│  │    State Machine: IDLE → RECORDING →      │           │
│  │    PROCESSING → IDLE                      │           │
│  └──────────┬───────────────┬────────────────┘           │
│             │               │                            │
│             ▼               ▼                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  processor   │  │    typer     │  │   notifier   │   │
│  │  (reqwest →  │  │ (arboard +  │  │(notify-rust) │   │
│  │   Ollama)    │  │   enigo)    │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   history    │  │    config    │  │  mcp_server  │   │
│  │   (JSONL)    │  │ (serde_yml) │  │   (rmcp)     │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              code_speaker (TTS)                   │    │
│  │  tts.rs (ort + Kokoro ONNX) │ api.rs (axum)     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Rust Crate Dependencies

| Component | Python Library | Rust Crate | Purpose |
|-----------|---------------|------------|---------|
| Async runtime | asyncio | **tokio** | Async I/O, timers, channels |
| Hotkey | evdev (Python) | **evdev** (Rust) | Keyboard input events |
| Audio capture | PyAudio | **cpal** | PCM recording (always-open stream) |
| ASR | transformers + distil-whisper | **whisper-rs** | Whisper.cpp bindings (CUDA) |
| Ollama client | httpx | **reqwest** | Async HTTP for grammar correction |
| Clipboard | xclip subprocess | **arboard** | Cross-platform clipboard |
| Key simulation | xdotool subprocess | **enigo** | Simulate Ctrl+V paste |
| Notifications | libnotify (PyGObject) | **notify-rust** | Desktop notifications (D-Bus) |
| Config | PyYAML | **serde_yml** | YAML config parsing |
| History | json (stdlib) | **serde_json** | JSONL read/write |
| MCP server | fastmcp | **rmcp** | Model Context Protocol (SSE) |
| HTTP server | http.server | **axum** | TTS API + MCP SSE transport |
| TTS (Kokoro) | kokoro-onnx | **ort** + custom | ONNX model inference |
| Audio playback | sounddevice | **rodio** or **cpal** | TTS audio output |
| Logging | logging (stdlib) | **tracing** | Structured logging |
| CLI args | argparse | **clap** | CLI argument parsing |

---

## Phase 1: Skeleton + Hotkey + Recording

**Goal:** Detect hotkey press/release and capture audio to a WAV file.
This validates the evdev + cpal stack and the state machine core.

### Modules
- `main.rs` — CLI args (clap), tokio runtime setup
- `config.rs` — Load `config.yaml` with serde_yml, config structs
- `hotkey.rs` — evdev keyboard monitoring, combo detection, press/release callbacks
- `recorder.rs` — cpal always-open stream, start/stop buffering, PCM → f32
- `service.rs` — State machine (IDLE → RECORDING), wires hotkey + recorder

### Deliverables
- `cargo run` starts, detects keyboards, opens audio stream
- Win+Alt press → starts recording, release → stops, saves WAV
- Prints audio stats (duration, sample count, RMS energy)

### Key Decisions
- Use tokio channels (`mpsc`) for hotkey → service communication
- cpal callback thread writes to a `Arc<Mutex<Vec<f32>>>` ring buffer
- Silence detection: inline RMS check in the recorder (port `silence_detector.py`)

### Estimated Complexity
~600 lines. Mostly boilerplate + evdev/cpal setup.

---

## Phase 2: Whisper Transcription

**Goal:** Transcribe captured audio using whisper-rs (whisper.cpp).

### Modules
- `transcriber.rs` — whisper-rs model loading, audio → text

### Deliverables
- After recording stops, audio is transcribed via whisper.cpp
- Prints transcribed text to stdout
- Supports CUDA acceleration (feature flag)

### Key Decisions
- Use `distil-large-v3` GGML weights from HuggingFace
- Model loaded once at startup (like Python version)
- `tokio::task::spawn_blocking` for the whisper inference call
- Audio resampled to 16kHz mono f32 (whisper.cpp requirement)

### Model Download
```bash
wget https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin
```

### Estimated Complexity
~200 lines. whisper-rs API is straightforward.

---

## Phase 3: Ollama Correction + Text Typing

**Goal:** Send transcribed text to Ollama for grammar correction, then paste into active window.

### Modules
- `processor.rs` — reqwest async POST to Ollama `/api/generate`
- `typer.rs` — arboard clipboard + enigo Ctrl+V simulation
- `notifier.rs` — notify-rust desktop notifications

### Deliverables
- Full pipeline: hotkey → record → transcribe → correct → paste
- Desktop notification on transcription complete
- Graceful degradation if Ollama is unavailable

### Key Decisions
- Same Ollama prompt as Python version
- Typer: set clipboard with arboard, then simulate Ctrl+Shift+V with enigo
- Fallback: if enigo fails, shell out to `xdotool` (like Python)
- Output modes: ollama-only, whisper-only, both (matching Python)

### Estimated Complexity
~300 lines across 3 modules.

---

## Phase 4: History + Config Parity

**Goal:** Full feature parity with Python for the core dictation pipeline.

### Modules
- `history.rs` — JSONL read/write in `~/.whisper-typer-history/`
- Enhanced `config.rs` — All config fields from Python version

### Deliverables
- `TranscriptionRecord` struct matching Python's dataclass fields exactly
- Daily JSONL files, compatible with existing Python history
- Markdown report generation (`generate_report()`)
- Full config.yaml compatibility (same keys, same defaults)

### History Record Schema (matching Python)
```rust
struct TranscriptionRecord {
    timestamp: String,          // ISO 8601
    whisper_text: String,
    ollama_text: Option<String>,
    final_text: String,
    output_mode: String,
    whisper_latency_ms: f64,
    ollama_latency_ms: Option<f64>,
    typing_latency_ms: f64,
    total_latency_ms: f64,
    audio_duration_s: f64,
    char_count: usize,
    word_count: usize,
    speed_ratio: f64,
}
```

### Estimated Complexity
~250 lines. Mostly serde structs + formatting.

---

## Phase 5: MCP Server

**Goal:** Expose MCP tools for Claude Code integration.

### Modules
- `mcp_server.rs` — rmcp-based MCP server with SSE transport

### MCP Tools (matching Python)
- `whisper_set_mode` — Set output mode
- `whisper_enable_ollama` / `whisper_disable_ollama`
- `whisper_get_status` — Current configuration
- `whisper_get_recent` — Recent transcriptions
- `whisper_get_daily_report` — Productivity report

### Key Decisions
- Use rmcp procedural macros for tool definitions
- SSE transport on stdio (matching Claude Code expectations)
- State sync via tokio channels (no file-based IPC — Rust binary is single-process)

### Estimated Complexity
~250 lines.

---

## Phase 6: Code Speaker (TTS)

**Goal:** Port the Kokoro TTS engine and HTTP API for Claude Code hooks.

### Modules
- `code_speaker/tts.rs` — ort ONNX inference for Kokoro-82M, sentence streaming
- `code_speaker/api.rs` — axum HTTP server (speak, cancel, set-voice, status)
- `code_speaker/summarizer.rs` — Ollama summarization for long text
- `code_speaker/reminder.rs` — Periodic reminder playback

### Deliverables
- HTTP API on port 8767 (same as Python)
- Compatible with existing `tts-hook.sh` (no hook changes needed)
- Sentence-level streaming with cancel support
- Voice-active gate (defer TTS during recording)

### Key Decisions
- Use `ort` crate directly to load `kokoro-v1.0.onnx`
- Study `kokorox`/`kokoroxide` for phonemizer implementation
- Audio playback via rodio `Sink` (supports cancel via `sink.stop()`)
- `tokio::sync::Mutex` for speak lock (replaces asyncio.Lock)
- `tokio_util::sync::CancellationToken` for cancel events

### Estimated Complexity
~500 lines. Most complex phase due to ONNX + audio + HTTP.

---

## Phase 7: Claude Code Hook Binary

**Goal:** Replace `tts-hook.sh` with a native Rust binary for Claude Code hooks.

The current shell script spawns `curl`, `jq`, and `python3` on every hook event.
A compiled Rust binary eliminates all subprocess overhead — instant startup, no
interpreter, direct HTTP calls.

### Binary
- `whisper-typer-hook` — Single binary invoked by Claude Code hooks

### Behavior (matching tts-hook.sh)
- Reads hook event JSON from stdin
- Parses `hook_event_name` field
- **SessionStart**: Check TTS API, speak "Claude Code is ready"
- **Stop**: Read transcript JSONL, extract last assistant text, summarize + speak
- **PermissionRequest**: Speak "Claude needs permission to use {tool}"
- **Notification**: Handle idle_prompt / permission_prompt
- **UserPromptSubmit**: POST `/cancel-reminder` to cancel TTS + reminders

### Key Decisions
- Use `reqwest::blocking` (not async) — hooks are short-lived processes
- Parse JSON with `serde_json` (replaces jq + python3)
- Read transcript file directly (replaces python3 subprocess)
- All HTTP calls have connect timeout (300ms) + total timeout (1s–3s)
- Exit cleanly if TTS API is unreachable (no error output)

### Claude Code Settings
```json
{
  "hooks": {
    "Stop": [{ "command": "whisper-typer-hook", "timeout": 10 }]
  }
}
```

### Estimated Complexity
~200 lines. Straightforward stdin → parse → HTTP POST.

---

## Phase 8: Systemd Integration + Polish

**Goal:** Production-ready binary with systemd service file.

### Deliverables
- `whisper-typer-rs.service` systemd unit file
- Structured logging with tracing (JSON + console output)
- Signal handling (SIGTERM, SIGINT) for graceful shutdown
- Error recovery (reconnect evdev devices, retry Ollama)
- CLI flags matching Python (`--mode`, `--no-ollama`, `--verbose`, `--enable-tts`)

### Estimated Complexity
~200 lines of integration + service file.

---

## Build & Development

### Prerequisites
```bash
# System dependencies
sudo apt install cmake build-essential libasound2-dev  # cpal ALSA backend
sudo apt install libclang-dev                          # whisper-rs bindgen

# Optional: CUDA
sudo apt install nvidia-cuda-toolkit  # For whisper-rs CUDA feature

# Whisper model
wget https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin

# Kokoro TTS model (for Phase 6)
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Cargo.toml Features
```toml
[features]
default = ["tts"]
cuda = ["whisper-rs/cuda"]    # GPU-accelerated Whisper
tts = ["ort", "rodio"]        # Code Speaker TTS (optional)
```

### Binaries
```toml
[[bin]]
name = "whisper-typer-rs"
path = "src/main.rs"

[[bin]]
name = "whisper-typer-hook"
path = "src/hook.rs"
```

### Running
```bash
cd whisper-typer-rs
cargo run --release            # CPU Whisper
cargo run --release -F cuda    # GPU Whisper
cargo run --release --bin whisper-typer-hook  # Hook binary
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| whisper-rs distil-whisper quality | Test with short dictation (<30s). Fall back to standard large-v3 if needed |
| ort 2.0 RC API instability | Pin exact version in Cargo.toml |
| enigo Wayland support | Fallback to xdotool subprocess (same as Python) |
| rmcp SSE maturity | Test with Claude Code early in Phase 5 |
| Kokoro phonemizer in Rust | Use kokorox phonemizer or shell out to espeak |
| Build complexity (cmake, CUDA) | Document thoroughly, provide Dockerfile |

---

## Timeline

Each phase is independently testable. Phases 1–4 deliver a working dictation tool.
Phases 5–7 add Claude Code integration and production polish.

| Phase | Description | Dependencies |
|-------|------------|-------------|
| 1 | Skeleton + Hotkey + Recording | None |
| 2 | Whisper Transcription | Phase 1 |
| 3 | Ollama + Typing | Phase 2 |
| 4 | History + Config | Phase 3 |
| 5 | MCP Server | Phase 4 |
| 6 | Code Speaker TTS | Phase 4 |
| 7 | Claude Code Hook Binary | Phase 6 |
| 8 | Systemd + Polish | Phase 5, 6, 7 |
