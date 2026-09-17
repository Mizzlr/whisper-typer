# Architecture and reliability boundaries

The service is split into four layers:

1. `main.rs` loads and validates configuration, initializes models, and starts
   HTTP services.
2. `dictation/service.rs` owns the dictation state machine and chooses the text
   output.
3. `speech/transcriber.rs`, `speech/spelling.rs`, and `speech/processor.rs`
   perform speech recognition, offline spelling cleanup, and optional grammar
   judging/correction.
4. `dictation/typer.rs` delivers final text to the focused application.

With `ollama.background_review: true`, the foreground delivers the spelling and
punctuation result immediately. A bounded queue independently uses the validated
corrector without the judge. Jobs/results retain the original dictation timestamp;
new speech never cancels an earlier suggestion. A private JSONL event stores raw
ASR, pasted text, suggested text, correction outcome, and grammar duration. The
original transcription record is never rewritten. Queue overflow skips grammar
only, with an explicit saved status, while preserving the original paste.

The separate Tkinter window loads daily history and grammar-result files, then
receives live loopback HTTP pushes. Rust's bounded publisher uses a persistent
client independently of typing and background review. A socketpair and Tk's
file handler wake the UI's main thread immediately; widgets are never updated
from the HTTP worker threads. Files provide startup and recovery state.
`difflib.SequenceMatcher` compares case-folded words in the already-pasted text
and grammar result. Only word edits receive highlights; punctuation/casing alone
retain the original pasted text on one row. Grammar changes use two rows, with
Copy buttons in a common right-aligned column beside each timestamp. Historical
judge decisions remain in the history data. It copies only on a user's **Copy** click, keeps suggestions
available for later use, and never changes another application's text. The window
is independently movable/resizable, remembers its Keep on top choice, and saves geometry. Per-entry
headers show a colored outcome dot and paste / grammar round-trip milliseconds.
Only the last 15 minutes appear initially. Scrolling down appends batches of 30
without rebuilding existing widgets; archive days load on demand. The window
keeps its reading anchor during incoming corrections and dictations.
Cards are reconciled by stable keys; only changed or reordered cards are
rebuilt, with other buttons and thumbnail objects retained. Reading anchors
count text indices including embedded windows. Returning to the top discards
older rendered cards and releases loaded archive days.

The optional Clipboard view merges cached text/image copies with dictations,
deduplicating automatic pastes and preserving the original dictation card when
a corrected suggestion is copied again. GTK ownership events request contents
asynchronously; the Tk event loop services GLib every 20 ms. A single bounded
worker writes exact text or normalized PNGs/thumbnails to a private, 200-item
SQLite history outside the repository. Image copying uses GTK's image selection
API, providing image MIME targets to other applications. Screenshot discovery
checks recent saved files every 2 seconds. No subprocess clipboard owners are
spawned, and no clipboard contents enter the model pipeline. The window reads
only newly appended dictation/result bytes every five seconds for recovery; view toggles use loaded
records. Daily dictation ordinals come from the full loaded day's history.

Hands-free capture is an optional `voice-journal --ui-session` mode launched by
the window. It shares the journal's VAD, WAV transport, and transcription code,
uses the existing ASR endpoints, and never touches the clipboard or paste path.
Its bounded audio queue transcribes while capture continues. Stop closes the
microphone before flushing the final partial frame and draining queued chunks.
JSON events persist to a private session file and travel over stdout into the
same Tk socket wakeup. Ordered chunk IDs aggregate idempotently into one card
with one Copy action. The ambient service runs independently.
Explicit sessions also apply the ambient journal's text hallucination filters.
Silero's shared wrapper prepends 64 context samples to each 512-sample frame.
The session detector trusts successful speech predictions and learns a noise
floor from confident non-speech for its energy fallback. Live meter events
travel over stdout at 4 Hz without entering journal/session files; persisted
chunks include their audio timestamps and pause/limit/stop boundary reason.
A fixed-height inner transcript scrolls independently and follows new text
unless the user is reading earlier segments. Export preserves the transcript;
on-demand Ollama summaries snapshot included chunk IDs, run off the Tk thread,
and persist separately without changing capture or transcript text.
Explicit sessions append their raw and cleaned chunks to the daily Voice Journal
before sending the same events to the window. While a session's process holds
the shared recording lock, ambient capture pauses and clears its partial VAD
buffer to avoid duplicate speech. The TTS API, queue, and playback loop also
respect the lock. Recording clears pending speech but leaves the user's enabled
preference intact; releasing the lock, including on process death, ends the
temporary inhibition. The normal window is included in GNOME overview; a local
`POST /show` and desktop launcher reveal the existing instance. Background
updates never activate it. The window and its tooltips decline
window-manager focus requests so workspace switches do not activate them.

Runtime MCP controls and the dictation loop share `RuntimeSettings`; changing a
mode takes effect on the next utterance and is atomically persisted. Grammar
correction is advisory: structured output is checked for stutter, large length
changes, removed URLs/numbers, and protected domain terms. Any failure returns
the cleaned, punctuated input to that pass. Raw ASR is retained separately in
history. The optional grammar judge uses TypeSafe Jev's typed probabilities; Rust
converts them into correction need and approval of a narrowly proposed
repeated-letter prefix removal. Grammar is evaluated independently on the
original and candidate texts, and only a confident prefix decision selects the
candidate. With `provider: race`, Jev and the configured Ollama corrector model judge
concurrently; the first valid decision wins and the pending request is dropped.
An early failure waits for the other judge, and both failures fall back to the
normal corrector. Winner provider and latency are recorded in history.
The Ollama gate supports the existing structured boolean contract and a short
`decision_format: pass_repair` contract for Granite 350M. PASS preserves input;
REPAIR or failure invokes the separately configured corrector. The short
classifier cannot authorize direct prefix removal. White Wolf keeps both models
resident with `keep_alive: -1` and a loaded-model limit of two; its warm unit
loads both at startup. Jev credentials
are loaded from a private external file and are never logged or configured
inline. The gate sends text, not audio, to TypeSafe. Clean text
bypasses rewriting, while flagged or uncertain decisions use the existing
corrector. Only a proposed prefix edit may be executed directly by Rust.
Judge latency and spelling edits are recorded independently. Separate budgets
bound judging and the full correction including retry.

`voice-journal` is a separate capture pipeline. Its audio callback sends
completed utterances through a bounded queue so a slow HTTP or LLM request
cannot grow memory without limit or stall the real-time callback. It prefers
White Wolf ASR and punctuation, with independent warm fallbacks on Black Beast.
The unfiltered journal is written before punctuation so it remains exact ASR
evidence even when optional post-processing changes or fails.

## Reliability invariants

- Whisper is the authoritative ASR path. Deprecated direct-to-Ollama audio
  settings remain parse-compatible but cannot cause an utterance to be dropped.
- Optional grammar correction must fail open to its cleaned input. A failed
  offline spelling-data load disables cleanup without blocking dictation.
- Remote ASR and punctuation fail independently to their local Black Beast
  equivalents; one remote-stage failure must not disable the other stage.
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
operationally sensitive components. Changes to `dictation/hotkey.rs` or
`dictation/typer.rs` need manual desktop testing in addition to unit and compile
checks. The system must always retain raw ASR for inspection and keep the last
usable transcript when an optional processing stage fails.
