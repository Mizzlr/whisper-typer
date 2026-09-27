# Voice Typing Vocabulary Additions & Hotkey Hardening

**Date**: 2026-09-27  
**Topic**: Domain vocabulary corrections (skews, status backfiller, white wolf) and multi-keyboard modifier hotkey resilience.

---

## 1. Domain Vocabulary Corrections

* **Audit & Need**: Spoken dictation logs revealed recurring acoustic and language-model transcription distortions:
  * Volatility/options/trading "skews" mistranscribed as "SKUs" / "skus" / "skues", and singular "skew" as "SKU" / "sku".
  * Database tooling "status backfiller" mistranscribed across varied spacing/casing as "status backf filler", "status BACKF filler", "status back filler", or "backf filler".
  * Remote host "white wolf" acoustically confused as "vitals" or "white tools".
* **Rules Added to `~/.config/whisper-typer/corrections.tsv`**:
  * `replace\t(?i)\b(?:skus|skues)\b\tskews`
  * `replace\t(?i)\bsku\b\tskew`
  * `replace\t(?i)\bstatus\s+(?:backf?\s*filler|backflow\s*status|bachelor)\b\tstatus backfiller`
  * `replace\t(?i)\bbackf\s*filler\b\tbackfiller`
  * `replace\t(?i)\b(?:vitals|white\s+tools?)\b\twhite wolf`
* **Reference Template**: Added `corrections.example.tsv` to the repository root and updated `.gitignore` (`!*.example.tsv`) so repository users have a clean template of regex replacement and span protection rules.
* **Unit Tests**: Added regression tests in `src/dictation/service.rs` (`applies_vocabulary_corrections`). All tests pass.

---

## 2. Keyboard Hotkey Hardening

* **Issue**: Following service restart, keyboard hotkey was reported non-responsive while mouse gesture (`KEY_F24`) remained functional.
* **Root Causes**:
  1. **Dual HID Endpoints**: Physical keyboard (EVISION USB-STDHID) splits keys across `/dev/input/event5` (standard keys) and `/dev/input/event3` (modifiers `KEY_LEFTMETA`, `KEY_LEFTALT`).
  2. **Strict Modifier Matching**: `config.yaml` only configured `[KEY_LEFTMETA, KEY_LEFTALT]`. Any press of `Ctrl+Alt`, `AltGr` (Right Alt), or Right Win was unhandled.
  3. **Startup Gap**: 10-second CUDA/Kokoro/Whisper initialization window drops transitions if keys are held during startup.
* **Resolution**:
  * Expanded `config.yaml` alternate combos to 9 variants including `Ctrl+Alt` (`KEY_LEFTCTRL + KEY_LEFTALT`), Right Ctrl/Alt, Right Meta, and PageDown combos.
  * Verified live dictations succeeded across both mouse and keyboard triggers.
