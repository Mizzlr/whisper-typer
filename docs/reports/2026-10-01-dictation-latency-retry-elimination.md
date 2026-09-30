# Dictation Latency Optimization & Ollama Retry Elimination

**Date**: 2026-10-01  
**Topic**: Diagnosis and resolution of multi-second dictation latency spikes (2.5s – 2.9s) caused by validation mismatches and redundant Ollama retry loops.

---

## 1. Problem Statement & Root Cause Analysis

* **Symptom**: Recent long dictations were taking a couple of seconds (2.55s – 2.89s) to paste, showing grammar latencies of 2311ms and 2725ms in the Whisper Typer UI.
* **Root Causes**:
  1. **Ordinal & Number Token Mismatch**: Spoken ordinals like `"1st"` were written as `"first"` by Ollama. The existing numeric validator (`significant_tokens` checking for digits) saw `["1st"]` in the original and `[]` in the candidate, triggering a false-positive `changed_numeric_fact` rejection. Similar rejections occurred with hyphenated compound modifiers (e.g. `"14 day"` vs `"14-day"`).
  2. **Inanimate Pronoun ("it") False Positive**: Spoken glitches or stutters involving `"it"` (e.g. `"continue it regenerate"` corrected by Ollama to `"continue to regenerate"`) changed the count of `"it"`. Because `"it"` was grouped under personal pronouns in `personal_pronouns()`, this triggered a `changed_personal_reference` rejection.
  3. **Futile Ollama Retry Loop**: In `OllamaProcessor::correct`, validation failures other than empty/unknown tokens triggered a sequential retry to Ollama (`warn!("Rejected Ollama correction ({reason}); retrying once")`). The retry took an additional 1.2s – 1.5s, produced the same mismatch, and fell back to uncorrected text—wasting 2.5s to 2.9s total.

---

## 2. Implementation Details

* **File**: `src/speech/processor.rs`
* **Normalized Numeric & Ordinal Validation**:
  * Added `numeric_tokens()` and `normalize_number_word()` to parse words and digit tokens.
  * Canonicalizes ordinals (`1st` $\leftrightarrow$ `first`, `2nd` $\leftrightarrow$ `second`, `3rd` $\leftrightarrow$ `third`, etc.), cardinal number words (`one` $\leftrightarrow$ `1`, `two` $\leftrightarrow$ `2`), hyphenated numbers (`14 day` $\leftrightarrow$ `14-day`), and decimals/commas.
  * Replaced raw digit token comparison with `numeric_tokens(original) != numeric_tokens(candidate)`.
* **Personal Pronouns Scoped to Human Personas**:
  * Removed inanimate `"it" | "its" | "itself"` from `personal_pronouns()` reference families (`counts` reduced from 7 to 6).
  * Legitimate speech slip repairs of `"it"` are now accepted while full persona protection (`I`, `you`, `we`, `he`, `she`, `they`) remains intact.
* **Immediate Fallback on Structural Validation Errors**:
  * Added `changed_numeric_fact`, `changed_personal_reference`, `changed_url`, `large_length_change`, and `removed_protected_term` to the immediate fallback list in `OllamaProcessor::correct`.
  * Eliminates redundant sequential Ollama inferences for unrecoverable structural errors.
* **Regression Tests**:
  * Added assertions in `preserves_personal_references_in_requests_and_accepts_contractions` covering ordinal normalization (`1st` $\leftrightarrow$ `first`), hyphenated spans (`14 day` $\leftrightarrow$ `14-day`), decimal repair (`+one.8` $\leftrightarrow$ `+1.8`), speech slip repair (`continue it regenerate` $\rightarrow$ `continue to regenerate`), and tamper rejection (`Send 28 SOL` $\rightarrow$ `Send 29 SOL`).

---

## 3. Verification & Deployment

* **Unit Tests**: All 61 unit tests in `whisper_typer_rs` passed cleanly.
* **Build & Deployment**:
  * Built release binary with CUDA support: `cargo build --release --bin whisper-typer-rs`.
  * Installed binary to `~/.local/bin/whisper-typer-rs`.
  * Restarted `whisper-typer-rs.service` via `systemctl --user restart whisper-typer-rs`.
  * Verified service status (**active (running)**) and journal logs confirming all keyboards monitored and ready.
