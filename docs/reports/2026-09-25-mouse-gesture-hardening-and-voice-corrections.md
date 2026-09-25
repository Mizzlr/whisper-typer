# Mouse Gesture Hardening and Voice Typing Corrections

**Date**: 2026-09-25  
**Topic**: System responsiveness, Logitech MX Master 3S gesture push-to-talk resilience, and voice typing typo/stutter corrections.

---

## 1. System Responsiveness & Memory Thrashing

* **Issue**: System load spiked to 97.20 with 64.6% I/O wait (`wa`). 42.8 GB of 46.5 GB swap was consumed during heavy concurrent builds (`cargo test --workspace` with multiple `ld.mold` linkers), swapping out interactive processes (`whisper-typer-rs`, `solaar`, `gnome-shell`).
* **Resolution**: Paged swap cleanly back into RAM via `sudo swapoff -a && sudo swapon -a`. System load dropped to healthy idle levels with 0.0% I/O wait.

---

## 2. Mouse Gesture Push-to-Talk Resilience

* **Issue**: Logitech MX Master 3S thumb gesture button voice typing stopped responding intermittently.
* **Root Cause**: Solaar's internal HID++ listener thread on `/dev/hidraw1` died during system memory pressure, leaving Solaar in an apparent "active" state where the GUI showed `Diverted`, but no physical button presses were forwarded.
* **Permanent Architecture Hardening**:
  1. **Direct HID++ Listener Thread**: Updated `infra/whisper-hotkey-daemon.py` with a background thread reading `/dev/hidraw1` directly for HID++ 2.0 report `0x11`, feature `0x09` (`REPROG_CONTROLS_V4`) and CID `0x00C3` (Gesture Button). This provides zero-latency push-to-talk independent of Solaar process health.
  2. **Watchdog Health Check**: Added `check_solaar_listener_thread()` in `infra/mouse-button-guard` to verify every 10 seconds that Solaar maintains an active listener thread in `do_select`.
  3. **Verification**: `infra/verify-mouse-stack.sh` passed all 29/29 diagnostic checks.

---

## 3. Voice Typing Typo & Stutter Corrections

* **Audit**: Analyzed 2,189 voice dictations from `2026-09-20` through `2026-09-25` across `~/.whisper-typer-history/`.
* **Findings**:
  * **Searcher Clarification**: "King 7" / "syncing 7" was confirmed as an active searcher entity with API keys and settlements (not an acoustic hallucination for "working").
  * **Acoustic CTC Blurs**: Granite Speech 5 TurboCTC lack of language model caused phonetic merges (e.g. "for these kind of stutters" -> "for these kind ofutters").
  * **Suffix Stutters**: Morphological repetitions (e.g. "testinging" -> "testing", "Trailil blazer" -> "Trailblazer") bypassed offline spell checkers due to strict edit-distance-1 limits.
* **Corrections Deployed**:
  * Added 67 rules to `~/.config/whisper-typer/corrections.tsv` covering market data (Zerodha, OHLCV, write amplification, demerger), Solana infrastructure (SOL, WSOL, db-sync, GBX, Blockdaemon, Firedancer), system tools (status backfiller, burndown, Pushover, config.toml, quiet hours, caret), team members (Emmanuel, Gautham, Xiao, Giorgio), and stutters (testinging, Trailblazer, kinds of stutters, Solaar).
  * Added `*.csv` and `*.tsv` patterns to `.gitignore` to guarantee local data tables remain private and uncommitted.

---

## 4. Latency & Instant Typing

* Restored `background_review: true` in `config.yaml`.
* Pre-compiled in-memory regex pass executes in 0.1ms, enabling immediate paste in ~70ms with zero perceptible lag.
