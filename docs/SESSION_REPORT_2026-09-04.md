# Whisper Typer and Voice Journal session report — 2026-09-04

## Outcome

Whisper Typer now uses White Wolf's resident Granite Speech 5.0 TurboCTC ASR,
with Black Beast's resident Distil-Whisper kept as an automatic fallback.
Black Beast performs optional CUDA punctuation/truecasing and uses a persistent
in-process X11 clipboard for low-latency, Dvorak-safe paste. Voice Journal keeps
the raw ASR evidence separately from curated output and retains private WAV
samples for seven days.

## Implemented

- Added configurable remote ASR with lossless WAV transport and local fallback.
- Added configurable fail-open punctuation/truecasing.
- Added a permanently warm Rust Whisper HTTP server and White Wolf units/config.
- Added the Granite Speech 5 TurboCTC server and service definition.
- Added the Black Beast CUDA punctuation server and loopback-only service.
- Added private per-dictation WAV capture, atomic writes, metadata, and a
  seven-day retention policy.
- Made Voice Journal accept segmented or plain ASR responses, use remote-first
  ASR with local fallback, preserve raw dictated text in the unfiltered stream,
  and exclude filler-only `okay`/`ok`/`yeah` runs only from curated output.
- Replaced per-dictation `xclip` process churn with a persistent `arboard`
  clipboard owner; retained `xclip` as a fallback and retained `xdotool` for the
  keyboard-layout-aware paste shortcut.
- Updated the local Granite model references from 4.1 to 4.2 without promoting
  Granite 4.2 to live correction.
- Documented White Wolf inference, model measurements, and MX Master 3S/Solaar
  controls.
- Added a local global Codex instruction at `~/.codex/AGENTS.md` to interpret
  voice-dictated requests in context while verifying precision-sensitive
  identifiers and values. This personal instruction is intentionally outside
  the public repository.

## Verification

- Rust library suite: 15/15 passed.
- Live Whisper Typer service: active, zero restarts after the paste deployment.
- Live Voice Journal service: active.
- Live punctuation service: active; CUDA execution provider reported first.
- White Wolf Granite Speech endpoint: healthy.
- Deployed Whisper Typer, Voice Journal, and punctuation artifacts matched the
  corresponding build/source hashes at verification time.
- Two live paste samples measured 17.1 ms each, down from 183–222 ms; complete
  post-recording processing measured 67 ms and 90 ms.

## Privacy and safety

The GitHub repository is public. Audio under `~/.whisper-typer-history/`, Voice
Journal data, evaluation manifests, and recovered dictation documents are
private runtime evidence and must not be committed. The repository ignores
recovered-dictation Markdown files explicitly; model binaries and Python cache
artifacts are also ignored.

White Wolf's isolated single-DIMM test passed 18 hours 30 minutes with no
hardware incidents. This does not certify the removed DIMMs or existing
ClickHouse data, so ClickHouse, Keeper, Redpanda, and RustFS remain outside the
scope of this inference cutover.
