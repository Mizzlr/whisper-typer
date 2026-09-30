# Ambient White Noise Integration, Typer Latency Optimization & UI Hardening

**Date**: 2026-09-30  
**Topic**: Ambient audio player integration (ASMR River & Rain, Mellow Rain for tinnitus aid), LLM backtick artifact filtering, dictation latency tuning, and Terminator pane divider cleanup.

---

## 1. Ambient White Noise Tool & Whisper Typer Integration

* **Background & Motivation**: The user requested a local ambient sound generator accessible directly from the Whisper Typer floating window UI and as a standalone app, specifically tuned to aid tinnitus.
* **Audio Engineering**:
  * Extracted and looped the user's preferred YouTube ASMR River & Rain stream (`~/.cache/whisper-typer/ambient/river-rain.opus`).
  * Created a dedicated **Mellow Rain** profile applying a 2,800 Hz low-pass filter (24 dB/oct rolloff) with RMS loudness equalization to eliminate high-frequency acoustic sharpness that irritates tinnitus.
  * Set startup default volume to **20%** across the CLI engine, standalone UI, and dictation pill integration.
* **Architecture & UI**:
  * Implemented headless CLI engine in `infra/white_noise.py` supporting daemon mode, dynamic volume adjustments, profile switching, and state reporting.
  * Developed GTK3 standalone control applet in `infra/white_noise_ui.py` and desktop entry `infra/whisper-white-noise.desktop`.
  * Integrated dedicated headphone icon (`🎧`) into the Whisper Typer pill bar in `infra/dictation-window.py` with an interactive popover containing volume slider, play/pause toggle, and sound profile picker.
  * Added auto-dismissal of popover menus upon window dragging (`moved()`) or focus loss so the UI never detaches awkwardly.
* **Tests**: All 6 unit tests in `infra/test-white-noise.py` pass.

---

## 2. Dictation Latency & LLM Backtick Artifact Dropping

* **Issue 1 — 6 Backticks Paste**: Occasionally Ollama / Granite models emitted hallucinated markdown code fences (e.g. ```` ``` ```` or whitespace-padded backticks) during pauses.
* **Fix**:
  * Added `is_backtick_garbage()` in `src/dictation/service.rs` to detect and drop spurious backtick hallucinations before clipboard injection.
  * Enforced double-check in `src/dictation/typer.rs` to prevent empty or backtick-only strings from ever reaching `xdotool` or clipboard paste.
  * Short-circuited Ollama retries on `wrapped_or_explained_output` and `introduced_unknown_token` in `src/speech/processor.rs` to prevent slow multi-second retries when corrections fail formatting rules.
* **Issue 2 — Typing Latency**:
  * Increased Ollama `clean_threshold` in `config.yaml` from `0.30` to `0.45`, enabling clean speech to instantly bypass the LLM and paste in ~90ms while reserving Ollama only for genuine transcription errors.
* **Tests**: 60 unit tests pass in `whisper_typer_rs`.

---

## 3. Terminator Pane Splitter & Scrollbar Remediation

* **Issue**: Moving panes or workspaces in Terminator displayed a thick, dark vertical column between adjacent horizontal splits.
* **Root Causes**:
  1. Default `scrollbar_position = right` in Terminator caused dark GTK VTE scrollbars to appear against the cream background (`#ffffdd`).
  2. Terminator splitters set `wide_handle = True` in `paned.py`, drawing a wide GTK handle.
* **Resolution**:
  * Configured `handle_size = 0` and `scrollbar_position = hidden` in `~/.config/terminator/config`.
  * Added CSS overrides in `~/.config/gtk-3.0/gtk.css` targeting `.terminator-terminal-window separator` and `paned > separator` to zero out separator dimensions, borders, and margins.

---

## 4. Voice Journal Recorder Sync & Jev Deprecation

* **Issue — Recorder Not Starting**: Clicking **● Record** in the Whisper Typer floating window immediately failed and returned to idle without recording.
* **Root Cause**:
  * When `provider: modernbert` was added to `config.yaml` on September 26, the compiled binary at `~/.local/lib/whisper-typer/voice-journal-recorder` had not been updated since September 18.
  * The older binary failed validation on launch with `ConfigError("unknown grammar gate provider")`.
  * Because `recording_session.py` redirected `stderr` to `/dev/null`, the startup error was suppressed.
* **Fixes & Remediation**:
  * Rebuilt and deployed the release binary to `~/.local/lib/whisper-typer/voice-journal-recorder` and `~/.local/bin/voice-journal`.
  * Updated `infra/install.sh` to install `voice-journal-recorder` into `~/.local/lib/whisper-typer/` on deployment so future builds stay synchronized.
  * Updated `infra/recording_session.py` to pipe `stderr` and surface actionable startup errors in the UI event bridge if the subprocess fails.
  * Restarted `voice-journal.service` and `whisper-dictation-window.service`.
* **Deprecation of TypeSafe Jev**:
  * Replaced Jev defaults and examples in `config.example.yaml`, `config.yaml`, and `src/config.rs` with local ModernBERT (`provider: modernbert`).
  * Removed obsolete `api_key_file` entries from active configs and updated `README.md` and `docs/ARCHITECTURE.md` to establish local ModernBERT and Ollama as the standard grammar gate.
* **Verification**: All 61 Rust tests and Python recording lifecycle tests pass cleanly.
