# UI performance review — September 18, 2026

## Measured changes

Baseline: session checkpoint `7869b33`. Measurements use isolated Xvfb, 5,000
synthetic records, 100 mounted cards, 20 warm samples for row collection and
single-card updates, and five samples for idle polls and fullscreen/restore.
This measures local UI work, not speech/model/network latency. The benchmark
asserts identical before/after row hashes and the existing 100-card limit.

| Workload | Operation | Before ms | After ms |
| --- | --- | ---: | ---: |
| Plain text | Collect rows | 21.783 | 12.037 |
| Plain text | Update one corrected card | 31.435 | 21.790 |
| Plain text | Idle recovery poll | 18.861 | 3.797 |
| Plain text | Fullscreen and restore | 31.185 | 36.486 |
| 20% reviewed | Collect rows | 32.120 | 17.144 |
| 20% reviewed | Update one corrected card | 43.269 | 22.216 |
| 20% reviewed | Idle recovery poll | 18.669 | 3.484 |
| 20% reviewed | Fullscreen and restore | 37.918 | 37.071 |

Fullscreen timing is mixed; these measurements do not establish a resize
speedup. Reproduce with `python3 infra/benchmark-ui.py`; private raw evidence is
`~/.cache/whisper-typer/ui-optimization-benchmark-final.json`.

## Data and rendering lifecycle

- Rust sends dictations and later grammar reviews over persistent loopback HTTP.
  A socket wakeup dispatches them on Tk's owning thread. An out-of-order review
  joins by dictation timestamp. HTTP 202 means queued, not rendered; health exposes
  event counts and dispatch/render timing separately.
- JSONL files remain the durable dictation/review history. Incremental offsets and
  partial UTF-8 lines support recovery without rereading complete daily files.
  The five-second recovery poll now renders only changed data, expired pending
  reviews, expired history, or an invalidated view. Pending reviews still expire
  after two minutes even when no more events arrive.
- Warm row caching avoids repeated word comparisons and status/timing formatting.
  The cache signature covers displayed record/review values, daily ordinal and
  pending timeout. Returned copies prevent view annotations from contaminating
  the cache. Removing records prunes cached rows. Parsed timestamps use an LRU
  limited to 16,384 entries. Collection remains linear in loaded history and
  sorts timestamps; it is not a constant-time database query.
- Only 100 timeline cards are mounted. Existing unchanged widgets and thumbnails
  survive updates. Coalesced paging, reading anchors and release of older daily
  archives preserve scroll position and bound mounted work. The initial view is
  recent data, with up to 20 fallback items per selected category; explicit
  scrolling can load older history. This does not delete stored history.
- Clipboard items and image descriptions use private SQLite WAL storage. Text
  ownership is checked against all loaded voice categories and targeted older
  history days, independent of toggle selection. Copying does not reorder cards.
- Capture, image decoding/resizing, vision descriptions, recording titles and
  summaries run outside Tk. Workers send primitive events/results back; Tk
  widgets and PhotoImages stay on the main thread. Image resizing keeps only the
  latest pending request/result and caps output size. Title work has a bounded
  queue and avoids duplicate requests; summary/caption failures preserve source.
- Recording panes append timestamped chunks, fit short content and cap long
  content. Very long wrapped paragraphs now go directly to the height cap instead
  of an expensive exact pixel count. Transcript and summary wheel input hands
  off to the outer timeline at each edge. Inside the pane it stays independent.
  Live tail following pauses on manual upward scrolling and resumes at the end.
- Shutdown cancels widget-owned timers and keeps the event loop alive until an
  active recorder persists its final chunks. The regression harness explicitly
  collects destroyed Tk interpreters on the main thread to avoid cross-test
  background-thread garbage collection.

## Limits and follow-up

Tk remains appropriate for the measured workloads; this review does not justify
an immediate Qt rewrite. Large initial daily-file reads and up to 50 saved
recording files still happen at startup, and SQL/history ownership work during
collection can grow with retained history. The HTTP queue is bounded to 256, but
its dispatch drain is not limited by a time budget. These are future profiling
candidates, not changes made without evidence in this batch.

Validation covers independent category combinations, partial JSONL reads, review
ordering, caching invalidation, daily numbering, pending timeout, short and long
recording panes, wheel handoff at both edges, live-tail behavior, fullscreen,
paging/anchors, theme changes, image zoom/pan/copy, captions, titles and summaries.
All GUI checks run on Xvfb; benchmark artifacts contain synthetic data.

## Terminator popup mitigation

The installed Terminator maps `preferences_keybindings` to Ctrl+Shift+K by
default. Only that shortcut was disabled in `~/.config/terminator/config`; the
menu remains available. A private backup was retained, and parsed configuration
comparison found no other setting changes. Existing preferences panels were
closed using their Close action, without restarting terminal processes. A scoped
configuration reload attempt in the existing terminal produced no new panel.
The exact gesture/input path responsible for the shortcut remains unconfirmed;
this mitigation must not be presented as proof of that root cause.

## Rust transcription/grammar safeguards

ASR unknown-token cleanup runs immediately after transcription in dictation,
explicit recording and ambient journal paths. The normal path borrows input;
raw history remains available. Case-insensitive unknown-marker checks and exact
percent-symbol counts reject corrupt punctuation before it is accepted, using
the existing endpoint/original-text fallback. Grammar uses the same marker and
percent checks. Both normal/retry prompts forbid changing speaker/recipient;
validation also compares personal-reference family counts, allowing contractions
and adjacent repeated starts. This conservative guard can reject valid rewrites
that add/remove references; it keeps the original rather than silently adopting
a different viewpoint. It is not a general semantic-equivalence proof.

Additional static check: Clippy passes the library but the all-binary strict run
reports pre-existing unnecessary_unwrap in tts_hook.rs:635 and four
needless_borrow findings in voice_journal/ui_recorder.rs. They are not suppressed
or described as passing checks.

Automatic scrolling remains under investigation. Solaar owns thumb-wheel
Ctrl/Super+PageUp/Down mappings; the main Rust mouse daemon forwards wheel
events. The user has not yet identified whether the unexpected movement is a
workspace change or content scroll. Configuration reload attempts briefly
activated Terminator; those actions stopped and were disclosed to the user.
No speculative mouse mapping change was made.
