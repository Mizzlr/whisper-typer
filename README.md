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

## Runtime architecture

```
 Global hotkey (evdev)
          │ press/release
          ▼
 ┌────────────────────── whisper-typer-rs ──────────────────────┐
 │ recorder → Whisper ASR → corrections → output-mode selection │
 │                                      │                       │
 │                                      ▼                       │
 │                         focused-window paste                 │
 │                         (xdotool+xclip / enigo)               │
 │                                                              │
 │ MCP server :8766                 Local HTTP API :8767         │
 │ runtime controls + reports       TTS queue + /transcribe      │
 └──────────────┬───────────────────────────────┬────────────────┘
                │                               │
     Claude Code / MCP             ┌────────────┴─────────────┐
                                   │                          │
                           tts-hook binary            voice-journal
                                   │                  bounded queue
                                   ▼                          │
                         Kokoro → rodio              daily Markdown
```

The service loads one Whisper model and shares it with the dictation loop and
the loopback `/transcribe` endpoint. `voice-journal` therefore does not load a
second Whisper model. MCP tools use shared `RuntimeSettings`; TTS-related MCP
tools proxy to the local API on port 8767. Both listeners bind to loopback.

See [the architecture notes](docs/ARCHITECTURE.md) for reliability boundaries,
runtime data locations, and the components that require manual desktop testing.

## Requirements

- Linux with PipeWire or ALSA audio
- A recent stable Rust toolchain
- `ollama`, `xdotool`, and `xclip`
- ALSA and X11 development headers (`libasound2-dev`, `libx11-dev`,
  `libxdo-dev`, and `pkg-config` on Debian/Ubuntu)
- Optional but recommended: NVIDIA CUDA and cuDNN for low-latency Whisper

Whisper, Kokoro, and Silero model weights are not committed to this repository.
Kokoro tokenizer metadata is tracked under `models/`; review the remaining
paths in `config.example.yaml` before installing.

## Installation

```bash
git clone https://github.com/Mizzlr/whisper-typer.git
cd whisper-typer
./infra/install.sh
```

`install.sh` does five things: adds you to the `input` group, installs the udev rule for `/dev/uinput`, runs `cargo build --release`, copies the five binaries to `~/.local/bin/` and ONNX runtime providers to `~/.local/lib/whisper-typer/`, and installs the systemd user services.

After the script completes:

- Log out and back in (for `input` group to take effect)
- `ollama pull granite4.2:3b` (default local correction model)
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
service keeps the cleaned, punctuated transcript instead.

Offline spelling cleanup uses `symspell` to propose candidates and `spellbook`
to validate English word forms from the system Hunspell dictionary. It runs
before punctuation and does not maintain typo-to-word mappings. Currently only
an unambiguous deletion of an accidentally duplicated letter is applied:
`dataaset` becomes `dataset`, while valid inflections such as `fills` and
`trees` stay unchanged. Code, paths, URLs, numbers, acronyms, and existing local
protection rules are preserved. Install `hunspell-en-us` and enable `spelling`
in the config to use it. Missing data disables this advisory pass with a warning.
Leading doubled letters and rare short candidate words are deferred to context.
Candidates shorter than six letters require at least 100,000 occurrences in the
optional frequency data; longer valid stems can work without that supplement.

Enable `ollama.background_review: true` for immediate paste and optional grammar
suggestions. The fast spelling and punctuation output pastes first; a bounded
background queue runs the configured grammar corrector directly, without a judge.
Each dictation keeps its own timestamped result, so continuing to talk preserves
earlier suggestions. Closing the review window does not stop dictation or review.

The Python Tkinter floating window (`infra/dictation-window.py`) compares the
already-pasted text with background grammar corrections using `difflib`:
red for removed words and green for added words. Punctuation and capitalization
alone are ignored. Unchanged dictations show one text row; grammar edits show
original and corrected rows. All **Copy** buttons align on the right of the timestamp.
Click Copy, then paste wherever you want. It does not replace text in another application or
change the clipboard when new suggestions arrive. The window has a **Keep on top** toggle,
is movable and resizable, and remembers its position.
Triage Desk's milky cream light theme is the default; the **◐** button switches
between light and dark without rebuilding cards or changing the reading position.
The choice is saved with the window settings. The initial view contains
the last 15 minutes. Scrolling down appends older entries in batches of 30;
unchanged cards and thumbnails are reused across updates. Returning to the top
releases older cards and archive days, keeping the live view small.
archive days load only as needed. Raw ASR remains in history; historical entries without a
separate pre-grammar baseline show their final text without an inferred diff.
Each compact header shows a gray dot for no grammar edit, green for a word edit,
and amber while checking, followed by time until paste / grammar round trip in
milliseconds. Background grammar time is separate from paste latency because
review starts after paste. Detailed judge outcomes/timings remain in history.
The top row shows today's date and a live clock updated each second. Entries
show only their time; older days get a single date heading when scrolling back.
Dictations show a stable daily `#N`. Hover the outcome dot for details. A
validated suggestion that was rejected is distinguished from an unavailable
model; the original pasted text remains unchanged.

The **Dictations**, **Recordings**, and **Clipboard** buttons independently
show or hide each category and remember their settings. Hiding recordings
does not stop capture. Controls wrap below the clock in narrower windows.
Each enabled category falls back to older items when it has no recent entries,
so recent clipboard copies do not hide recordings or block dictation history.
The **Clipboard** toggle adds text copies and clickable
image thumbnails to the same timeline. Copies are deduplicated; immediate
dictation pastes appear once. Copying a matching dictation preserves its full
card: the original/corrected diff, daily number, and latency statistics remain
visible. Dictation/recording text stays in its own category, including copies
captured by the clipboard monitor when that category is hidden. Text previews
are compact, but Copy restores the full exact text.
Clicking an image thumbnail opens a viewer inside the app. Use +/− or the
wheel to zoom, drag to pan, and Fit (or 0) to reset. Shift+wheel or Left/Right
pans horizontally; arrow keys and both scrollbars also pan the image. Close or Escape returns
to the timeline. Copy is available inside the viewer too and restores the
original full-resolution image, including parts outside the current view. Capturing/history updates never overwrite it.
Images have their own daily `#N`. Recopies preserve their first-capture time,
number, and position; dictations also stay in their original timeline order.
New captures and recent images from the last 15 minutes are named and described
in the background by Qwen3 VL 2B on
White Wolf. The thumbnail stays on the left; its screen title and one or two
sentences appear on the right. **Describe** beside Copy runs this on demand
for historical images. It reads **Describing…** while processing and
**Described** afterward. Already described or currently processing images do not
queue another model call. Cached descriptions
survive restarts. The original image and clipboard stay intact if vision is
unavailable. Existing images are numbered during migration; older images are described on demand.
The choice is remembered. Capture continues with the toggle off, while the
window is running. It starts with the current clipboard; past unrecorded text
cannot be recovered. Recent saved screenshots (last 24 hours) from
`~/Pictures/Screenshots` and `/tmp/codex-clipboard-*.png` are also imported.

The recent history window covers the last 24 hours without deleting saved dictations.
The window initially shows the last 15 minutes; if that view is empty, it
automatically searches older history to show up to 20 items from the selected
categories. Scrolling can continue into earlier archives. Finished sessions with no accepted transcript are labeled
**Empty recording** and omit the unused transcript pane. Scrolling progressively loads
12 nearby entries in either direction, with at most 100 cards mounted; distant
widgets are destroyed. An ongoing recording remains available past the cutoff.

Clipboard ownership notifications are asynchronous, serviced every 20 ms;
image encoding, thumbnails, and persistence use a bounded background worker.
New clipboard copies take priority over screenshot imports. Imported file
signatures survive restarts, and cached image thumbnails are reused.
Screenshot discovery runs every 2 seconds. A private SQLite database at
`~/.cache/whisper-typer/clipboard/history.sqlite` holds the latest 200 unique
items, with private PNGs under `clipboard/images`. No clipboard content is sent
to grammar models. The module requires system `python3-gi`, GTK 3 introspection,
`python3-pil`, and `python3-xlib` for returning keyboard focus after image viewing. GUI/clipboard tests use an isolated `xvfb-run` display.

Rust pushes dictations and grammar results to `http://127.0.0.1:8768/events`
through a bounded background queue and persistent HTTP connection. Incoming
messages wake Tk's event loop through a socket; normal updates do not wait for a
file poll. The JSONL files remain authoritative for startup and five-second
recovery checks. `GET /health` exposes delivery counts and dispatch timing,
without transcript text. This loopback endpoint rejects browser-origin writes.
Configure `ui.enabled` and `ui.endpoint` to enable pushes.

The small red **Record** button starts hands-free, copy-only dictation. Voice
Journal's VAD splits speech at pauses or every 25 seconds and reuses the existing
ASR servers, spelling rules, and punctuator. It does not load another Whisper
model or paste chunks into the active app. **Stop** ends microphone capture and
finishes pending chunks; a single card and **Copy** button contain the full
session. Sessions are private JSONL under `~/.cache/whisper-typer/recordings`.
Completed sessions survive window restarts; failed transcriptions retain their
private WAV for recovery. Closing the window also stops session capture.
The existing ambient Voice Journal service runs independently.
Recording panes fit their displayed content, occupy at most about half the window height, and scroll
independently. New segments follow the tail unless you scroll back to read.
The live status shows Listening, Speaking, Silence (milliseconds), and
Transcribing; segment timestamps indicate the audio boundary and its reason
(pause, 25-second limit, or Stop). Silero receives the preceding 64 samples
required by its model wrapper; its speech score takes priority over loudness.
Confident non-speech frames update a noise floor for adaptive energy fallback.
Recordings use the same configured hallucination filters as Voice Journal,
retaining filtered raw text only in the unfiltered journal/session evidence.
**Download** saves the complete timestamped transcript unchanged. **Summarize**
processes a snapshot with the configured Ollama model in a background worker;
long meetings use bounded portions and a merged summary. Summary view shows the
whether newer transcript text is missing from the summary and can copy the result. Summarize also generates a
short 3–5 word topic title beside the Summary tab, saved with the summary;
completed visible recordings without titles are named automatically in a bounded
background queue, including historical sessions, without re-summarizing. Topic
titles are cached privately under `recordings/titles`. Private summaries persist
under `recordings/summaries`; summary failure leaves capture and text intact.
During an explicit session its capture is paused to avoid duplicate audio.
The recorder also appends tagged session chunks to `~/voice-journal/journal_DATE.md`
and raw chunks to `journal_DATE.unfiltered.md`, then pushes them into the window.
Recording temporarily suppresses hook announcements and TTS playback, discards
pending speech, and preserves the user's existing TTS enabled/disabled preference.
A process-held OS lock clears automatically on Stop, normal exit, or crash.
The window appears in GNOME overview as a normal app and declines automatic focus
requests; Copy and mouse scrolling still work without activating it during
workspace changes. Tooltips also decline focus.
Install `infra/dictation-window-show.py` as `~/.local/bin/whisper-typer-window`
and `infra/whisper-typer-window.desktop` under `~/.local/share/applications/`.
Search for **Whisper Typer** with Super to open/reveal the existing instance.

Install `infra/dictation-window.py`, `infra/clipboard_history.py`,
`infra/ui_events.py`, `infra/recording_session.py`, `infra/recording_view.py`,
`infra/recording_summary.py`, `infra/image_caption.py`, and `infra/image_view.py` under
`~/.local/lib/whisper-typer/` and install
the release `voice-journal` binary there as `voice-journal-recorder`. Install
`infra/systemd/whisper-dictation-window.service` as a user unit. Start/reopen with
`systemctl --user start whisper-dictation-window.service`. It uses system Python
and Tkinter, plus PyYAML and python3-xlib for summaries and launcher activation.
Original history remains in `~/.whisper-typer-history`; background
results are private JSONL in `~/.cache/whisper-typer/grammar-review.jsonl`.
Set `ollama.background_review: false` to restore synchronous correction; the
pre-deployment binary/config are also saved for a complete revert.

Any dictation containing the whole word **sorry** (case-insensitive) bypasses
the grammar judgment stage and short-text skip threshold. In Ollama mode it
also bypasses immediate paste/background review: the grammar LLM resolves the
correction before paste. Its prompt handles single-word and phrase replacements,
including numbers and names:

- “Payouts for September, sorry, October” → “Payouts for October”
- “Meet at five, sorry, at six” → “Meet at six”
- “Visit New York, sorry, Los Angeles” → “Visit Los Angeles”
- “Send 5 SOL, sorry, 6 SOL” → “Send 6 SOL”
- “When I said that, sorry, this” → “When I said this”

The LLM preserves genuine apologies, quoted uses, explicit references to the
word “sorry,” and incomplete or ambiguous corrections. The LLM receives the
transcription first, then labeled contexts in spoken order (before/after each
marker), and is instructed to retain the replacement after “sorry.” Corrections
also apply inside sentences about dictation. Validation rejects edits that keep
the old wording while discarding the supplied replacement, or strip only
“sorry” without replacing an abandoned phrase. Validation allows the abandoned local
phrase to be removed while protecting facts outside that repair. If all
attempts fail or time out, the original dictation is preserved. After two invalid edits,
clear terminal single-word replacements have a constrained fallback: the same
LLM chooses a literal last-word replacement, with optional article adjustment,
then restores punctuation and capitalization while preserving those words.
This fallback excludes compound names, broader replacements, apologies, and
literal references; those retain the original if normal correction fails. All
attempts share the existing correction timeout budget. Raw ASR stays in private
history. Explicit Whisper mode or disabled Ollama still skips the LLM. This
corrects the current dictation before delivery; it does not edit earlier pasted
or sent messages.

An optional `ollama.grammar_gate` asks TypeSafe's Jev (`provider: typesafe`)
for typed error probabilities via `/v1/systemone`. Rust skips rewriting only
when the probability of a required repair is at most `clean_threshold` (0.2).
Uncertain, failed, or timed-out judgments use the existing validated Granite
correction pass. Jev never generates replacement text. With `provider: race`, Jev and the configured `ollama.model` on `ollama.host`
judge concurrently. The first valid decision wins; a fast failed response does
not win. The pending request is dropped, though a provider may finish already
accepted work. If both judges fail, the normal correction pass runs. History
records `grammar_gate_provider` so the winner is observable. A legacy
`provider: ollama` gate remains available for offline setups.
For a fast local Granite 350M judge, set `provider: ollama`, `model: granite4:350m`,
`host` to its Ollama endpoint, and `decision_format: pass_repair`. This short
classifier returns PASS or REPAIR; PASS keeps the input unchanged, while REPAIR
runs the separately configured `ollama.model` corrector. It does not authorize
direct prefix removal. Invalid or timed-out decisions also run the corrector.
Use `keep_alive: -1` and an Ollama loaded-model limit of at least two to keep
the judge and corrector hot together. The White Wolf warm unit preloads both.

The TypeSafe key is read at startup from `api_key_file`, normally
`~/.config/typesafe/api-key`; keep this private file outside the repository.
Only its path belongs in configuration. TypeSafe receives the cleaned,
punctuated transcript and any proposed prefix candidate, not audio.
See [TypeSafe's API reference](https://docs.typesafe.ai/api).
The judge has its own timeout; `ollama.correction_timeout_ms` bounds the entire
correction including a retry. The gate only runs when Ollama processing and an
Ollama/both output mode are enabled. A model's decision is advisory and can miss
an error or flag acceptable text; no dictionary/model files are fetched during
dictation.
For a repeated single-letter opening followed by a sentence-capitalized word,
Rust proposes its exact removal. The judge
must explicitly confirm that it is an artifact before Rust executes the edit;
if the remaining text is clean, no longer rewrite call is needed. The model
cannot request arbitrary edits through this mechanism. Initials, language names,
and shortcut keys require contextual preservation.

History preserves raw ASR and records spelling edits, spelling latency, the
judge decision/latency, and whether correction was accepted. Replay history
without microphone capture or focused-window typing:

```bash
cargo run --release --example spelling_benchmark -- --date-prefix 2026-09-
cargo run --release --example spelling_benchmark -- --text 'S Summarize the progress.' --grammar
```

The optional `--report /absolute/private/path.json` saves edit evidence with
mode 0600 and refuses to overwrite an existing report. Whole-text replay latency
and candidate-generator comparisons are reported separately. The default
generator comparison uses the same Hunspell stems for both libraries; Spellbook
also understands affixes and performs a broader suggestion search. These are
different algorithms, so the timing comparison does not imply equal accuracy.
The optional frequency file can be provisioned with
`python3 infra/fetch-spelling-data.py`; its source revision and checksum are
pinned. Hunspell stems are sufficient without this extra file.

## Configuration

Copy `config.example.yaml` when starting a new setup. Configuration is loaded
from an explicit `--config` path, the repo's `config.yaml`, or
`~/.config/whisper-typer/config.yaml`, in that order. Malformed or unsafe values
fail startup with a clear error instead of silently using defaults.

```yaml
ollama:
  enabled: true
  model: "granite4.2:3b"
  host: "http://127.0.0.1:11434"
  keep_alive: 3600
  skip_threshold: 5              # skip Ollama on utterances ≤ N words
  correction_timeout_ms: 5000     # correction + retry budget; cleaned text on timeout
  grammar_gate:
    enabled: false
    provider: "race"             # first valid Jev/Granite judgment wins
    model: "jev-latest"
    host: "https://api.typesafe.ai"
    api_key_file: "~/.config/typesafe/api-key"
    timeout_ms: 1500
    clean_threshold: 0.2
    fragment_threshold: 0.9

spelling:
  enabled: false
  aff_path: "/usr/share/hunspell/en_US.aff"
  dic_path: "/usr/share/hunspell/en_US.dic"
  frequency_path: ""             # optional external word-frequency data

whisper:
  model: "models/ggml-distil-large-v3.bin"
  device: "cuda"                 # cuda | cpu | mps

tts:
  enabled: true                  # native Kokoro TTS
  voice: "af_bella"              # any af_*, am_*, bf_*, bm_* preset
  speed: 1.0
  api_port: 8767
```

Set `ollama.skip_threshold: 0` when the judge should assess short utterances too.

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
2. **LLM pass** — Ollama (`granite4.2:3b`) chat API with a few-shot prompt. Catches novel hallucinations the regex doesn't know about. Auto-disables if Ollama is unreachable; can be force-disabled via `WHISPER_VOICE_JOURNAL_LLM=0`.

**Output files**

- `~/voice-journal/journal_YYYY-MM-DD.md` — clean journal. Real speech only.
- `~/voice-journal/journal_YYYY-MM-DD.unfiltered.md` — sibling file capturing **every** transcribed utterance, including `[filtered]`, `[filtered-llm]`, and `[error]` lines. Use this to audit what the filters are catching without re-running audio.

**Whisper-typer integration**

When the `whisper-typer-rs` service is running, voice-journal tails its history file at `~/.whisper-typer-history/YYYY-MM-DD.jsonl` and injects each new dictation into both the live TUI (cyan `[dictated]` line) and the journal + unfiltered files. This means dictations performed via the global hotkey appear in the same daily journal alongside ambient capture — single source of truth for "what was said today." The integration is one-way (read-only on whisper-typer's side) and falls back silently if the file or service isn't present.

## Repository structure

The Rust package groups implementation files by responsibility while preserving
the original public module names through re-exports in `lib.rs`. That keeps the
organization visible on GitHub without forcing binaries or integrations to
change imports.

```
whisper-typer/
├── src/
│   ├── main.rs                 # daemon startup and component wiring
│   ├── lib.rs                  # shared module exports for all binaries
│   ├── config.rs               # YAML loading, defaults, validation
│   ├── dictation/
│   │   ├── mod.rs              # dictation module boundary
│   │   ├── service.rs          # state machine and voice gate
│   │   ├── hotkey.rs           # evdev monitoring and reconnects
│   │   ├── recorder.rs         # cpal microphone capture
│   │   └── typer.rs            # focused-window clipboard delivery
│   ├── speech/
│   │   ├── mod.rs              # speech module boundary
│   │   ├── transcriber.rs      # shared whisper-rs model and inference
│   │   ├── vad.rs              # voice-activity helpers
│   │   └── processor.rs        # correction and safety validation
│   ├── interfaces/
│   │   ├── mod.rs              # integration module boundary
│   │   └── mcp_server.rs       # MCP controls and reports on port 8766
│   ├── persistence/
│   │   ├── mod.rs              # persistence module boundary
│   │   ├── runtime_settings.rs # live controls and atomic persistence
│   │   └── history.rs          # transcription JSONL and reports
│   ├── code_speaker/
│   │   ├── mod.rs              # TTS module exports
│   │   ├── tts.rs              # Kokoro ONNX inference and playback
│   │   ├── api.rs              # TTS + transcription API on port 8767
│   │   └── history.rs          # spoken-event history and reports
│   └── bin/
│       ├── tts_hook.rs         # Claude Code lifecycle hook client
│       ├── voice_journal.rs    # journal TUI/headless recorder
│       └── benchmark.rs        # local latency summary
├── infra/
│   ├── install.sh              # build, deploy, udev, and user services
│   ├── install-mouse-stack.sh  # deploy the MX Master 3S Solaar/Input Remapper stack
│   ├── verify-mouse-stack.sh   # verify that stack against the live session
│   ├── test-logi-mouse-daemon.py # end-to-end test against synthetic uinput devices
│   ├── hooks/tts-hook.sh       # older shell-hook implementation
│   ├── systemd/                # daemon and journal unit templates
│   └── udev/99-uinput.rules    # input-device permissions
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MOUSE_INPUT_STACK.md    # MX Master 3S input stack and recovery runbook
│   ├── MODEL_RECOMMENDATIONS.md
│   └── reports/                # dated engineering evaluations
├── models/tokenizer.json       # tracked Kokoro tokenizer metadata
├── config.example.yaml         # portable configuration template
├── config.yaml                 # deployment configuration for this checkout
├── .mcp.json                   # local MCP connection example
├── .github/workflows/ci.yml    # tests and strict Clippy
├── Cargo.toml / Cargo.lock
├── CONTRIBUTING.md / SECURITY.md / LICENSE
└── README.md
```

Generated model weights, `target/`, local virtual environments, histories, and
journals are intentionally outside version control.

The reorganization is intentionally structural: public paths such as
`whisper_typer_rs::history` remain available. The hotkey and typer files were
moved intact because their press/release and xclip-owner behavior is sensitive
to functional changes.

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

Compare the shipping judges on a private sample of today's history (half spread
since 06:00 local, half most recent). Both receive identical domain-corrected,
spell-cleaned, punctuated inputs; clients retain their connections across runs:

```bash
cargo run --release --example judge_benchmark -- --date 2026-09-17 --samples 24 --repeats 3 --report ~/.cache/whisper-typer/judge-benchmark.json
```

The private report includes each input, decision, elapsed time, observed race
winner and disagreements. These are judge timings, not full transcription or
typing timings; no labeled accuracy is implied. The benchmark uses a 5-second
observation budget per judge so slower replies remain visible.

Compare other local judges using the exact inputs from that report:

```bash
cargo run --release --example judge_benchmark -- --date 2026-09-17 --model-input-report ~/.cache/whisper-typer/judge-benchmark.json --models qwen3.5:0.8b --models smollm2:360m --models granite4.1:3b --repeats 3 --report ~/.cache/whisper-typer/small-judge-benchmark.json
```

Use `--model-host` to select an isolated Ollama server. Confirm all models remain
resident on the GPU before measuring; a server configured for one loaded model
will otherwise include repeated loading costs. This comparison runs candidates
sequentially in rotating order and includes 16 separately labeled policy checks.
Repeated-input caching affects later rounds, which are reported separately.
