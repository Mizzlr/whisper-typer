# Architecture and reliability boundaries

The service is split into four layers:

1. `main.rs` loads and validates configuration, initializes models, and starts
   HTTP services.
2. `service.rs` owns the dictation state machine and chooses the text output.
3. `transcriber.rs` and `processor.rs` perform speech recognition and optional
   conservative grammar correction.
4. `typer.rs` delivers final text to the focused application.

Runtime MCP controls and the dictation loop share `RuntimeSettings`; changing a
mode takes effect on the next utterance and is atomically persisted. Grammar
correction is advisory: structured output is checked for stutter, large length
changes, removed URLs/numbers, and protected domain terms. Any failure returns
the original Whisper transcription.

`voice-journal` is a separate capture pipeline. Its audio callback sends
completed utterances through a bounded queue so a slow HTTP or LLM request
cannot grow memory without limit or stall the real-time callback. It reuses the
main service's loaded Whisper model through the loopback transcription API.

## Reliability invariants

- Whisper is the authoritative ASR path. Deprecated direct-to-Ollama audio
  settings remain parse-compatible but cannot cause an utterance to be dropped.
- Optional grammar correction must fail open to the original transcription.
- Runtime mode changes are applied from shared in-memory state and persisted by
  atomic rename; the MCP server and dictation loop never race to rewrite the
  state file independently.
- The real-time Voice Journal audio callback never waits on HTTP or LLM work.
  Its bounded queue drops excess completed utterances and records that event in
  telemetry instead of allowing unbounded memory growth.
- MCP and TTS listeners bind to loopback and are not intended as network-facing
  services.

## Runtime data

| Data | Default location | Notes |
|---|---|---|
| Runtime mode and recent entries | `~/.cache/whisper-typer/state.json` | Private atomic JSON state |
| Dictation history | `~/.whisper-typer-history/*.jsonl` | Contains dictated text and latency metadata |
| Voice Journal | `~/voice-journal/` | Markdown journal and optional debug audio/CSV |
| TTS hook history | `~/.tts-hook-history/` | Claude Code event metadata and prompt-derived labels |

## Change-sensitive paths

Hotkey press/release monitoring and focused-window text delivery are the most
operationally sensitive components. Changes to `hotkey.rs` or `typer.rs` need
manual desktop testing in addition to unit and compile checks. The system must
always keep raw Whisper text as a fallback when optional processing fails.
