# WhisperTyper RS

[![CI](https://github.com/Mizzlr/whisper-typer/actions/workflows/ci.yml/badge.svg)](https://github.com/Mizzlr/whisper-typer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Local-first speech-to-text dictation, voice journaling, and Claude Code voice
feedback for Linux. The application code is Rust; Whisper, Ollama, and the
model files remain separate local dependencies.

Press a hotkey, speak, and your words are transcribed (Whisper, CUDA) and pasted into the focused application — optionally grammar-corrected by a local LLM (Ollama). A separate TUI app records and filters a daily voice journal. Claude Code lifecycle events get spoken aloud through Kokoro TTS.

## What's in this repo

This repo builds four binaries from one Rust package:

| Binary | Purpose |
|---|---|
| `whisper-typer-rs` | Background dictation service: hotkey → Whisper → Ollama → paste. Hosts the MCP HTTP server (port 8766) and the Kokoro TTS HTTP API (port 8767). |
| `tts-hook` | Standalone binary invoked by Claude Code on lifecycle events (SessionStart, Stop, Notification, PermissionRequest, UserPromptSubmit). Speaks short status phrases via the TTS API. |
| `voice-journal` | TUI for personal voice journaling. Captures audio, runs VAD, transcribes via the running service's `/transcribe` endpoint, applies regex + LLM hallucination filtering, appends to `~/voice-journal/journal_YYYY-MM-DD.md`. |
| `whisper-benchmark` | Reports p50/p95 latency and correction fallback counts from recent local history. |

## Architecture

```
                      ┌──────────────────┐
   Hotkey  ─evdev──▶  │                  │
   (Win+Alt etc.)     │  whisper-typer-rs│ ──▶ Whisper (CUDA, distil-large-v3)
                      │   state machine  │
   Claude Code ──┐    │   IDLE→REC→PROC  │ ──▶ Ollama /api/generate (granite4.1:3b)
   (MCP HTTP)    │    │                  │
                 ├──▶ │  ┌─────────────┐ │ ──▶ Typer (xdotool+xclip / enigo)
   Voice Journal │    │  │ MCP server  │ │
   (HTTP)        │    │  │ port 8766   │ │
                 │    │  └─────────────┘ │
                 │    │  ┌─────────────┐ │
   Claude Code ──┘    │  │ TTS API     │ │ ──▶ Kokoro ONNX → rodio playback
   (tts-hook bin)     │  │ port 8767   │ │
                      │  └─────────────┘ │
                      └──────────────────┘
```

## Requirements

- Linux with PipeWire or ALSA audio
- A recent stable Rust toolchain
- `ollama`, `xdotool`, and `xclip`
- ALSA and X11 development headers (`libasound2-dev`, `libx11-dev`,
  `libxdo-dev`, and `pkg-config` on Debian/Ubuntu)
- Optional but recommended: NVIDIA CUDA and cuDNN for low-latency Whisper

Whisper, Kokoro, and Silero model files are not committed to this repository.
Review the paths in `config.example.yaml` before installing.

## Installation

```bash
git clone https://github.com/Mizzlr/whisper-typer.git
cd whisper-typer
./infra/install.sh
```

`install.sh` does five things: adds you to the `input` group, installs the udev rule for `/dev/uinput`, runs `cargo build --release`, copies the four binaries to `~/.local/bin/`, and installs the systemd user services.

After the script completes:

- Log out and back in (for `input` group to take effect)
- `ollama pull granite4.1:3b` (default local correction model)
- Place `ggml-distil-large-v3.bin`, `kokoro-v1.0.onnx`,
  `voices-v1.0.bin`, and `tokenizer.json` under `models/`
- Optionally place `silero_vad.onnx` under `models/` for Voice Journal VAD
- Start both services with
  `systemctl --user start whisper-typer-rs voice-journal`

## Running

```bash
# Service (after install)
systemctl --user start whisper-typer-rs
journalctl --user -fu whisper-typer-rs

# Dev (foreground, verbose)
cargo run --release -- --verbose

# Voice journal (interactive TUI; auto-uses Silero VAD when models/silero_vad.onnx exists)
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

The hotkey and focused-window paste paths are deliberately conservative because
they sit directly in the text-input path. Grammar correction is advisory: if
Ollama times out, produces malformed JSON, repeats text, removes a URL or
number, changes a protected project name, or rewrites too aggressively, the
service types the original Whisper transcription instead.

## Configuration

Copy `config.example.yaml` when starting a new setup. Configuration is loaded
from an explicit `--config` path, the repo's `config.yaml`, or
`~/.config/whisper-typer/config.yaml`, in that order. Malformed or unsafe values
fail startup with a clear error instead of silently using defaults.

```yaml
ollama:
  enabled: true
  model: "granite4.1:3b"
  host: "http://127.0.0.1:11434"
  keep_alive: 3600
  skip_threshold: 5              # skip Ollama on utterances ≤ N words

whisper:
  model: "models/ggml-distil-large-v3.bin"
  device: "cuda"                 # cuda | cpu | mps

tts:
  enabled: true                  # native Kokoro TTS
  voice: "af_bella"              # any af_*, am_*, bf_*, bm_* preset
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
| `code_speaker_speak` | Enqueue text to TTS |
| `code_speaker_set_voice` | Persist Kokoro voice across restarts |
| `code_speaker_enable` / `code_speaker_disable` | Toggle TTS playback |
| `code_speaker_voices` | List available voice presets |
| `code_speaker_report` | Same daily report unified across STT + TTS |

## TTS hook (Claude Code lifecycle voice notifications)

`tts-hook` is the standalone binary Claude Code invokes on session events. It reads event JSON from stdin and POSTs short phrases to the TTS API at `http://127.0.0.1:8767`. Per-event behavior:

| Event | Speaks | Notes |
|---|---|---|
| `SessionStart` | `"Claude Code is ready."` | Skipped for `source=resume` and `source=compact` (only fresh starts). Focus session only. |
| `UserPromptSubmit` | (silent) | Claims focus for this session, captures the user's prompt to `~/.tts-hook-history/.last-prompt-{short_id}` for later use, and notifies the TTS API to interrupt focus speech and re-queue any deferred non-focus items. |
| `Stop` | `"<label>" task done.` (focus) or `"<label>" {project} done.` (non-focus) | The label is extracted from the saved user prompt: leading filler ("okay", "by the way", "so", "actually", "well", ...) is stripped, then the first ~6 words / 32 chars become the label. If no prompt is on file, falls back to `Task done.` / `{project} done.`. Per-session dedup prevents Claude Code's multi-Stop bursts from speaking the same announcement twice. |
| `PermissionRequest` | `"{project} needs permission."` | Always speaks regardless of focus. |
| `Notification` (`permission_prompt`) | `"Permission needed."` | Focus session only. |

The label feature lets you tell which task finished when several Claude Code sessions complete back-to-back. Sessions whose `cwd` is under `~/.claude-mem/` are muted (background observer noise).

Per-session state files live under `~/.tts-hook-history/`:
- `.focus-session` — current focus session ID (6h expiry)
- `.last-stop-{short_id}` — dedup token for Stop events
- `.last-prompt-{short_id}` — most recent prompt for label extraction
- `YYYY-MM-DD.jsonl` — full event log for debugging

## Voice Journal

`voice-journal` is a separate recorder tuned for long dictation sessions. It captures audio, runs VAD, sends each utterance to the running service's `/transcribe` endpoint (port 8767, reuses the loaded Whisper model — no second model load), and runs each transcribed chunk through a two-stage hallucination filter. Running `voice-journal` manually opens the TUI; the systemd service runs `voice-journal --headless` as a normal long-lived background process. Debug capture is available with `voice-journal --debug` or `WHISPER_VOICE_JOURNAL_DEBUG=1 voice-journal`; it writes sidecars next to the journal: `journal_YYYY-MM-DD_HHMMSS.mic.wav` for the raw mic stream and `journal_YYYY-MM-DD_HHMMSS.vad.csv` for RMS, Silero probability, threshold, voiced decision, capture state, and utterance events.

**VAD (voice activity detection)**

The detection chain has four stacked gates, each addressing a different false-positive mode observed in real journaling sessions. The first three are entry-only (they decide *whether to start* an utterance); the fourth runs at finalize time (decides whether to *send* it to Whisper).

| Gate | Default | Purpose |
|---|---|---|
| **Silero VAD ONNX (v5)** with hysteresis | enabled when model present | Probability ≥ `enter` (default `0.5`) is needed to start speech; once started, probability ≥ `stay` (default `0.35`) keeps the utterance voiced. Hysteresis prevents probabilities bouncing across a single threshold from chopping a sentence into fragments. Falls back to RMS energy thresholding if the model can't be loaded. Override via `WHISPER_VOICE_JOURNAL_SILERO_THRESHOLD` and `WHISPER_VOICE_JOURNAL_SILERO_STAY_THRESHOLD`. RMS rescue threshold for very-near-field speech that Silero scored low: `WHISPER_VOICE_JOURNAL_RMS_RESCUE_THRESHOLD` (default `0.05`). |
| **Speech-start streak** | 4 frames (≈128ms) | Requires N consecutive ≥enter-threshold frames before declaring speech_start. Single keystroke clicks (≈20–50ms broadband transients) and brief desk taps cannot sustain 4 frames; speech onsets do. Once `in_speech` is true, the streak requirement drops to 1 frame so naturally short pauses inside a sentence don't restart the count. |
| **Keystroke gate** | 250ms after each key event | A background thread reads `/dev/input/event*` for any key press/repeat and stamps an atomic timestamp. The audio callback forces `voiced=false` for `KEYSTROKE_GATE_MS` after each event, so typing on a mechanical keyboard near the mic can't open the speech gate. Only applied while `!in_speech` — if you're already speaking, typing won't truncate the utterance. Needs `input` group membership (same as the hotkey monitor). The TUI status changes to `Typing` while the gate is active. |
| **Minimum voiced time** | 250ms | Pre-roll plus silence-tail puff up every utterance to ≈1.2s of bytes — byte-length alone can't distinguish a real sentence from a single transient followed by silence. Voiced sample count is tracked separately, and utterances containing less than `MIN_VOICED_MS` of actually-voiced audio are dropped before being sent to Whisper. Visible in the TUI as `dropped low-voiced: N`. |

**TUI flicker telemetry**

The status panel shows live VAD diagnostics so you can see whether speech is being captured cleanly or chopped. Format:

```
Flicker: 268 (15.6/min) | voiced: 0ms (max 3413ms) | gated: 3000 | dropped low-voiced: 0 | queue drops: 0
```

- `transitions/min` should stay low during silence/typing and spike only when you speak.
- `max voiced run` should reach into the seconds during real sentences. If it stays below ~500ms, the stay threshold is too high.
- `gated` counts callbacks the keystroke gate suppressed. Climbs while you type without speaking.
- `dropped low-voiced` counts utterances skipped before Whisper was called.
- `queue drops` counts completed utterances discarded because the bounded
  transcription queue was already full. A non-zero value means the downstream
  transcription/filter path could not keep up.

**Hallucination filter (two stages, post-Whisper)**

1. **Regex pass** — fast, deterministic. Rules live in `~/voice-journal/hallucinations.txt`. Catches known echo patterns, podcast bleed, named-entity garble.
2. **LLM pass** — Ollama (`granite4.1:3b`) chat API with a few-shot prompt. Catches novel hallucinations the regex doesn't know about. Auto-disables if Ollama is unreachable; can be force-disabled via `WHISPER_VOICE_JOURNAL_LLM=0`.

**Output files**

- `~/voice-journal/journal_YYYY-MM-DD.md` — clean journal. Real speech only.
- `~/voice-journal/journal_YYYY-MM-DD.unfiltered.md` — sibling file capturing **every** transcribed utterance, including `[filtered]`, `[filtered-llm]`, and `[error]` lines. Use this to audit what the filters are catching without re-running audio.

**Whisper-typer integration**

When the `whisper-typer-rs` service is running, voice-journal tails its history file at `~/.whisper-typer-history/YYYY-MM-DD.jsonl` and injects each new dictation into both the live TUI (cyan `[dictated]` line) and the journal + unfiltered files. This means dictations performed via the global hotkey appear in the same daily journal alongside ambient capture — single source of truth for "what was said today." The integration is one-way (read-only on whisper-typer's side) and falls back silently if the file or service isn't present.

## Layout

```
src/
├── lib.rs                   # shared modules for all binaries
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
├── runtime_settings.rs      # live MCP/service settings + atomic persistence
├── code_speaker/            # native Kokoro TTS
│   ├── mod.rs
│   ├── tts.rs               # ONNX inference + rodio playback
│   └── api.rs               # axum HTTP API (port 8767)
└── bin/
    ├── tts_hook.rs          # Claude Code lifecycle listener
    ├── voice_journal.rs     # voice journal TUI
    └── benchmark.rs         # history latency summary

infra/
├── install.sh               # one-shot setup script
├── systemd/
│   ├── whisper-typer.service
│   └── voice-journal.service
└── udev/
    └── 99-uinput.rules

models/                      # Whisper, Kokoro, Silero-VAD model files (gitignored)
config.yaml                  # runtime configuration
```

## Common operations

```bash
# Rebuild + redeploy after pulling
cargo build --release
install -m 0755 target/release/{whisper-typer-rs,tts-hook,voice-journal,whisper-benchmark} ~/.local/bin/
systemctl --user restart whisper-typer-rs voice-journal

# Check service health
systemctl --user status whisper-typer-rs
curl -s http://localhost:8766/mcp -X POST -d '{}' | head    # MCP probe
curl -s http://localhost:8767/status                        # TTS probe

# Tail TTS hook events
tail -f ~/.cache/whisper-typer/tts-hook.log

# Summarize the latest seven history days
whisper-benchmark 7

# Public-repository verification
cargo test --all-targets --no-default-features
cargo clippy --all-targets --no-default-features -- -D warnings
```

## Known issues

- **MCP HTTP 410 after `/clear`**: rmcp Streamable HTTP sessions expire when Claude Code reconnects. Workaround: `systemctl --user restart whisper-typer-rs`.
- **Deprecated compatibility keys**: `wakeword`, `feedback`, and
  `ollama.audio_mode` remain parse-compatible for existing configurations but
  are inactive and emit startup warnings.
- **Verbose `whisper_init_state` startup**: 7 lines of GPU buffer allocation per service start. Cosmetic, not an error.

## Privacy and maintenance

Audio processing is local by default. Transcription history under
`~/.whisper-typer-history/`, voice journals under `~/voice-journal/`, and TTS
hook history may contain sensitive text; retention and deletion are currently
user-managed. Do not publish these files with bug reports. See `SECURITY.md`
for the full data-handling notes and `docs/MODEL_RECOMMENDATIONS.md` before
adding any hosted model backend.

## License

MIT.
