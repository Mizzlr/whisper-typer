# Codebase health and model evaluation

Date: 2026-08-21

Hardware: NVIDIA RTX 3060 12 GB

Production baseline: Distil-Whisper large-v3, Granite 4.1 3B, Kokoro v1.0

Runtime implementation commit: `2c65be5` (`refactor: harden dictation runtime
and observability`)

## Scope

This session reviewed and hardened the public repository without changing the
main `hotkey.rs` or `typer.rs` paths. It also compared smaller local grammar
models and evaluated Voxtral Mini 3B as a possible ASR replacement.

## Delivered changes

- Added conservative Ollama correction validation, one retry, and raw-Whisper
  fallback for malformed, repeated, fact-changing, or over-edited responses.
- Added correction acceptance/fallback and model timing fields to local history.
- Replaced competing MCP/service state-file writers with shared runtime state
  and atomic persistence.
- Kept deprecated configuration keys parse-compatible while removing the
  utterance-dropping direct-Ollama audio path.
- Added explicit configuration validation and aligned defaults with the shipped
  example.
- Bounded the Voice Journal transcription queue, exposed queue drops, reused an
  HTTP client with timeouts, and made keyboard-device watchers reconnect
  independently.
- Added the `whisper-benchmark` history summary binary.
- Added CI, public contribution/security guidance, architecture documentation,
  an example configuration, a stable Rust toolchain file, and an MIT license.
- Hardened the user services and made installed binaries under `~/.local/bin`
  the deployment source of truth.

## Measured model results

### Correction models

The production-shaped test used Ollama `/api/generate`, the repository system
prompt, deterministic decoding, and the same constrained JSON schema used by
the service.

| Model | Warm median | Observed behavior |
|---|---:|---|
| Granite 4.1 3B | ~573 ms | Most consistent command and punctuation behavior |
| LFM2.5-2.6B QAD-Q4_0 | ~485 ms | Faster median, but ~1.8 s tail, prompt echo, and missed command rewrites |
| Qwen3.5 2B | ~488 ms in the earlier six-case run | Faster than Granite but weaker punctuation |
| Qwen3.5 4B | ~731 ms in the earlier six-case run | Slower than Granite |

The validator correctly protects against plausible-looking semantic changes;
for example, the tested models changed `3.14 PM` to `3:14 PM`.

### Voxtral Mini 3B

The CUDA llama.cpp Q4_K_M path transcribed a 203-second English sample in
10.55 seconds (~19x realtime). The deployed Distil-Whisper path completed the
same sample in 3.96 seconds (~51x realtime). Voxtral added wrapper prose and
made at least one error that Whisper avoided.

The native vLLM checkpoint was not deployed. Its approximately 9.5 GB BF16 GPU
requirement exceeds the project's roughly 3 GB ASR memory target, and desktop
GPU usage left insufficient free VRAM without heavy CPU offload.

## Verification and deployment

- Release binaries were installed under `~/.local/bin` and matched the local
  `target/release` artifacts by SHA-256.
- `whisper-typer-rs.service` and `voice-journal.service` were active after the
  model experiments were rolled back.
- The TTS status endpoint reported the Kokoro model loaded with `af_bella`.
- Granite 4.1 3B was reloaded and warmed after testing.
- The Voxtral checkpoints, isolated vLLM environment, llama.cpp build, and
  temporary evaluation audio were removed after the memory constraint ruled
  out deployment. LFM2.5 remains available in Ollama as an unloaded experiment.

## Reproduction notes

- Local latency history: `whisper-benchmark 7`
- Public checks: `cargo test --all-targets --no-default-features` and
  `cargo clippy --all-targets --no-default-features -- -D warnings`
- Service health: `systemctl --user is-active whisper-typer-rs voice-journal`
- TTS health: `curl -fsS http://127.0.0.1:8767/status`

Model weights, audio samples, transcripts, local correction tables, and cache
directories are intentionally excluded from the repository.
