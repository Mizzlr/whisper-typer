# Recording and window improvements — September 17

- [x] Fix Silero's missing 64-sample context and verify saved speech scores.
- [x] Learn the background noise floor for adaptive loudness fallback.
- [x] Test speech and pauses with continuous white noise.
- [x] Apply Voice Journal's existing hallucination filters to recordings.
- [x] Filter restored transcript views with the same native rules without rewriting raw history.
- [x] Show live speech/silence/transcription status and segment boundary reasons.
- [x] Give each session an independent scroll pane, capped near half the window height.
- [x] Follow new transcript segments while preserving a reader's scroll position.
- [x] Export the entire original transcript with timestamps.
- [x] Summarize a snapshot through configured Granite; retain full original text.
- [x] Save summaries privately and keep recording active if summarization fails.
- [x] Make the window visible in GNOME overview and reliably reopen it via its launcher.
- [x] Deploy the updated window/helpers and Voice Journal binary without interrupting capture.
- [x] Verify live window properties, launcher, binary revisions and service health.
- [x] Investigate inner wheel direction; the initial inversion was superseded by matching the working outer history.
- [x] Route wheel events across each entire recording card, including completed/restored recordings; real X11 and generated wheel events passed; installed hashes/live health verified.
- [x] Fix reproduced inner-pane snap-back on long transcripts: manual scrolling holds the position; status-only updates never scroll; auto-follow resumes at the bottom. Verified and deployed after capture finished.
- [x] Add Triage Desk's milky cream light theme with a compact light/dark switch; 23 UI checks passed, preview inspected, installed hashes/live health verified.
- [x] Remove the inner wheel inversion so it matches the working outer history; 14 UI checks passed and actual desktop wheel-up/down moved the completed pane after deployment.
- [x] Caption clipboard screenshots through Qwen VL on White Wolf; show short titles beside thumbnails without blocking the UI.
- [x] Add on-demand Describe for historical images, and place screen title plus 1–2 sentence description beside each thumbnail.
- [x] Add persistent daily image numbers, preserving same-day deduplication and advancing through retention.
- [x] Make screenshot clicks toggle an in-app enlarged view; Copy remains explicit, and zoom preserves clipboard/timeline position.
- [x] Preserve first-capture timestamps and timeline order when existing clipboard items or dictations are copied again.

Evidence: saved speech median Silero score rose from 0.0005 to 0.9982 after
correcting context. The white-noise test rejects noise alone and detects speech
pauses above the old energy gate. UI checks cover independent scrolling, tail
following, exact Top/Copy alignment, export, summary snapshots and failure.
All 93 checks passed. The live window has normal window type, no skip-taskbar
flag, the expected launcher class and its automatic-focus rejection intact.
The launcher reveals the same process. A current 2048 ms microphone background
sample yielded zero accepted segments and zero errors. Restored views were
filtered without changing the hashes of the saved raw session files.
Private deployment evidence: `~/.cache/whisper-typer/adaptive-meeting-deployment.json`.

- [x] Default history to 24 hours and progressively mount at most 100 nearby cards in both directions; the later archive fallback below allows older items.
- [x] Skip Describe for images already described or already processing.

Latest UI batch: 38 isolated checks passed, including an 800-entry sliding
window, both loading directions, a 25-hour exclusion, repeated Describe while
busy/after success, large-image zoom fit, exact image copying and stable recopy
order. All five installed helper hashes match source; the live UI PID and
loopback health PID match with zero restarts. Private deployment evidence:
`~/.cache/whisper-typer/bounded-image-timeline-deployment.json`.

- [x] Fix wheel-event backlog from progressive loading: coalesce requests, recheck direction/edge, and reduce each batch to 12 cards; benchmark and verify live.

Scroll regression: a 32-event wheel burst used to queue 32 renders and 609 ms
of page work; coalesced paging does one 12-card load in 15 ms on the same
800-entry fixture. Direction reversal cancels stale paging; scrollbar bursts
are also coalesced. All 39 UI/clipboard checks passed. Installed source hash,
service PID, loopback health and zero restarts verified. Evidence:
`~/.cache/whisper-typer/scroll-coalescing-deployment.json`.

- [x] Add zoom controls, pointer-centered wheel zoom, drag panning and Fit/Close/Escape for the in-app screenshot viewer; verify responsive resize and unchanged timeline/clipboard.

- [x] Investigate fullscreen resize stalls and reduce repeated card layout work; verify fullscreen/restore responsiveness.
- [x] Label completed screenshot captions Described.

- [x] Add independent Dictations/Recordings view toggles beside Clipboard, remember choices, and wrap controls cleanly on narrow windows.

Viewer/filter/layout batch: 44 isolated checks passed, including larger-than-
window zoom, pointer anchoring, two-axis drag, Fit, Close, real X11 Escape and
previous-application focus restoration; independent saved category toggles;
fullscreen/restore with stable widgets and settled line heights; recording
manual-scroll preservation. Visible regions repaint after layout changes.
Actual-history tests with 100 cards completed fullscreen/restore in 12–39 ms
and coalesced 32 wheel events into one 12-card load (15 ms page work). Four
installed helper hashes, matching service/health PID and zero restarts verified.
Latest user screenshot confirms Clipboard-only filtering, completed Described
labels and clean visible card rendering. Private deployment evidence:
`~/.cache/whisper-typer/image-viewer-filters-layout-deployment.json`.

- [x] If selected categories have no recent items, automatically search older history for up to 20 items; allow continued scrolling into archives while keeping at most 100 cards mounted.
- [x] Label finished recordings with no accepted transcript Empty recording and keep their cards compact.

- [x] Add Copy inside the screenshot viewer and verify original pixels/resolution, open viewer and stable timeline after zoomed copy.

- [x] Make horizontal image panning accessible through drag, Shift+wheel, arrow keys and the bottom scrollbar; prevent repeat drags from resetting zoom.

Archive fallback and compact empty recordings deployed with 50 checks; the
viewer Copy/pan batch passed 51 isolated checks. Zoomed copying preserved original
pixels and resolution. Shift+wheel, Left/Right, scrollbar wheel and repeated drag
changed horizontal position without changing scale or the outer timeline.
Tk 8.6 rejects button 6/7 bindings, so dedicated horizontal wheel bindings were
omitted. Installed helper hashes, matching UI service/health PID, zero restarts
and no callback tracebacks verified. Private deployment evidence:
`~/.cache/whisper-typer/history-fallback-deployment.json` and
`~/.cache/whisper-typer/viewer-copy-pan-deployment.json`.

- [x] Generate and save a short recording topic title with each summary, and show it beside the Summary tab.
- [x] Fit recording panes to their displayed transcript/summary content, keeping half-window height as a maximum and preserving independent scrolling.

Recording title/content-fit batch: all 53 checks passed. A live synthetic model
probe returned the title Whisper Typer Image Viewer from configured granite4.1:3b
and a nonempty summary in one request. Topic titles survive private summary
storage/reload. Short transcript and summary panes shrink; long transcripts grow
to the cap and preserve existing manual-scroll tests. Four installed helper
hashes, matching service/health PID, zero restarts and no callback tracebacks
verified. Evidence: `~/.cache/whisper-typer/recording-title-fit-deployment.json`.

- [x] Automatically name completed visible recordings that lack titles, including historical sessions, without replacing transcript or summary.
- [x] Remove Through chunk numbers; show Summary is behind only when newer transcript chunks are not summarized.

Automatic-title batch: 55 checks passed, including historical summary/title cache,
duplicate request suppression, bounded title-only transport, and conditional
summary freshness text. Live UI has six titled recording cards, no pending title
jobs, matching service/health PID and zero restarts. All five installed hashes
match; all 15 pre-existing transcript/summary files retain their original hashes.
Also fixed shutdown timer cancellation to leave callback cleanup with its owning
widget. Evidence: `~/.cache/whisper-typer/auto-recording-titles-deployment.json`.

- [x] Verify all eight Clipboard/Dictations/Recordings combinations, including mixed-age histories and clipboard duplicates, and make each category's filtering/fallback independent.

Category independence batch: 57 checks passed. All eight combinations also passed
against actual saved history in isolated Xvfb. Clipboard + Recordings includes
older recording cards despite recent clipboard entries; dictation archive loading
works independently. Explicit render paths respect the Clipboard toggle. A copied
dictation is suppressed only while its originating category is selected, keeping
the clipboard accessible when that category is hidden. The 100-card cap and saved
toggle settings remain. Source/installed hash, service/health PID, zero restarts,
and live category counts against saved toggle choices verified. Evidence:
`~/.cache/whisper-typer/independent-categories-deployment.json` and
`~/.cache/whisper-typer/independent-category-live-history-checks.json`.

- [x] Keep voice-owned clipboard duplicates hidden when Dictations/Recordings are disabled; verify the user's Clipboard-only screenshot against saved data.

Screenshot follow-up supersedes the prior rule that allowed copied dictations to
appear as Clipboard items when Dictations was hidden. Ownership now stays with
the voice category regardless of toggle/window state. Recording chunks and full
session copies are included. Older clipboard text is checked against incrementally
read history days without loading/rendering those dictation cards. All 58 checks
passed, including all eight toggle combinations and appended archive ownership.
Actual saved data matched 164 voice-owned clipboard duplicates; isolated recent
Clipboard-only rendering retained the latest screenshot. Source/installed hash,
service/health PID, zero restarts and no callback errors verified. Evidence:
`~/.cache/whisper-typer/clipboard-voice-ownership-deployment.json` and
`~/.cache/whisper-typer/clipboard-voice-ownership-live-history.json`.

- [x] Review current changes for credentials and commit the session checkpoint (7869b33).
- [x] Profile and review the UI data/render/scroll/background-job lifecycle; optimize measured costs and verify end to end.
- [ ] Confirm the original gesture/input cause of the Terminator preferences popup; the shortcut mitigation is saved and scoped invocation opened no new panel.
- [x] Hand off recording/summary wheel input to the outer timeline at both edges, preserving independent scrolling inside each pane.

- [x] Disable the Terminator keybindings-preferences shortcut and close unwanted preferences panels without restarting terminal sessions.

Only preferences_keybindings was changed in the private local configuration;
parsed settings match the backup otherwise. The scoped shortcut invocation did
not open another panel. The original gesture input cause remains unconfirmed.

- [x] Clean ASR unknown-token artifacts in Rust before all correction stages, preserving raw history and percentage symbols.
- [x] Preserve personal references in grammar correction; reject the reported you-are to I-am change.
- [ ] Trace unexpected automatic workspace/content scrolling and fix the verified source.

Optimization/correction batch: 61 UI checks, 50 Rust library tests, 16 binary
unit tests and five native recorder replay checks passed. Strict library Clippy
passed; the all-binary run has five existing findings, recorded in
UI_PERFORMANCE_REVIEW.md. Final synthetic reviewed workload: collect 32.120 to
17.144 ms, one-card update 43.269 to 22.216 ms, idle poll 18.669 to 3.484 ms;
fullscreen results mixed. All five installed hashes and running binary hashes
match, service/health PID agrees, and all three services have zero failure
restarts. A real dictation/review advanced live push counters after deployment.
Real Granite probes preserved the reported personal reference and both percent
symbols. Evidence is private in ~/.cache/whisper-typer/ui-rust-optimization-*.json,
ui-optimization-benchmark-final.json and grammar-person-percent-check.json.

Automatic movement remains unconfirmed pending whether it is workspace or
content scrolling. Brief Terminator activations from configuration reload
attempts were disclosed and stopped; no speculative mouse remapping was made.

- [x] Show a muted, disabled Summarized action for current recording summaries; re-enable Summarize when new transcript chunks arrive.

Summary-action batch: 38 UI checks passed, including historical reload, stale
summary recovery and unchanged summary text with advancing coverage. The UI
helper was deployed independently; existing recordings and summaries are preserved.

- [x] Build and install Folio as a separate local Qt reading desk for repo-resolved paths, Markdown, CSV/TSV, text, PDFs and pasted content.
- [x] Render Mermaid and LaTeX math offline with pinned local renderer assets.
- [x] Add selected-cell/row/column statistics and copying for CSV and Markdown tables.

Folio batch: 13 behavioral checks passed in isolated Xvfb, including actual
Mermaid SVG/MathJax output, Markdown table statistics, code copying, CSV decimal
statistics and TSV copying, PDF extraction/paging/zoom/pan, and asynchronous
pasted-path loading. Live discovery found 16 Git repos in 13.47 ms; two exact
path probes resolved in 1.83 and 32.83 ms. Installed through tools/folio/install.py
with checksum-pinned offline assets and an Applications launcher. Whisper Typer
services were not changed. Folio starts on demand; no boot autostart was added.

- [x] Simplify Folio to automatic text/image paste, dated file groups, content-only reading and Escape to return to a cleared input.
- [x] Preserve original dump/screenshots privately, browse folders, and add Recent/Downloads views.
- [x] Resolve terminal-wrapped/shortened paths and fuzzy report names from recent adhoc dates, including untracked files.
- [x] Replace private report identifiers in new tests with synthetic fixtures; review unpublished changes for data artifacts and credentials before publication.

Minimal Folio batch: 17 behavioral checks passed in isolated Xvfb. Checks cover
automatic paste, old/new history persistence, Escape and reopening, screenshot
retention/zoom/copy, folder expansion, recent/download views, fuzzy untracked
reports, and existing Markdown/CSV/PDF rendering. Regression fixtures are synthetic
and private live probes remain outside the repository. Runtime history is private
SQLite outside Git; renderer assets and screenshots are not staged.

- [x] Wrap Folio text/source lines to the viewport and show original line numbers in a visible gutter, including zoom/resize updates.

Text-readability batch: 18 isolated GUI/behavior checks passed. Long prose and
unbroken words wrap without horizontal scrolling; original line count and copied
contents stay unchanged. Synthetic fixtures only.

- [x] Increase Folio CSV selection/header/statistics contrast; use explicit dark selection foregrounds for table and text views.

Contrast batch: existing 18 behavioral checks passed; selected cells and statistics
were visually checked on a synthetic CSV in isolated Xvfb. No private file data
was used in the regression fixture or preview.

- [x] Replace Folio boxed file rows with plain links, muted timestamp headers and a thin separator between paste groups.

Plain-list batch: three relevant group/folder/image interaction checks passed in
isolated Xvfb; synthetic list preview verified without private data. History view
also resets the window title after leaving a document.
