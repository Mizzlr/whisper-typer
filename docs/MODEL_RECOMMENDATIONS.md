# Model recommendations

These recommendations are based on measurements from the production request
shape, not general leaderboard scores. Results below were collected on an RTX
3060 12 GB and should be re-measured on different hardware. Last reviewed:
2026-08-21.

## Production default

- Speech-to-text: keep Distil-Whisper large-v3 on CUDA. It is already warm,
  local, private, and fast on the measured workload.
- Grammar correction: keep `granite4.1:3b` through local Ollama. The correction
  guard falls back to raw Whisper text when the model changes facts or produces
  malformed output.
- Text-to-speech: keep Kokoro v1.0 and the preferred local voice. It avoids a
  network round trip and is already integrated with interruption/voice gating.

## Local evaluations

### Grammar correction

| Model | Result | Decision |
|---|---|---|
| Granite 4.1 3B | Warm median about 573 ms across the constrained-JSON correction suite; handled the command-rewrite cases consistently | Production default |
| LFM2.5-2.6B QAD-Q4_0 | Warm median about 485 ms, but tail latency reached about 1.8 s; echoed prompt instructions and missed required command rewrites | Installed experiment only |
| Qwen3.5 2B | About 17% faster than Granite in an earlier local run, with weaker punctuation behavior | Do not promote without a larger corpus |
| Qwen3.5 4B | Slower than Granite in the same local run | Not recommended for this latency-sensitive path |

All models tried to reinterpret `3.14 PM` as `3:14 PM`; the correction validator
therefore remains necessary even when model output looks fluent.

### Speech recognition

Voxtral Mini 3B was evaluated through the current CUDA llama.cpp audio path on
a 203-second English sample. It completed in 10.55 seconds (about 19x realtime)
versus 3.96 seconds (about 51x realtime) for the deployed Distil-Whisper path.
It also added wrapper prose and made a transcription error that Whisper avoided.

vLLM is the native serving path recommended by the Voxtral model card, but the
BF16 checkpoint requires roughly 9.5 GB of GPU memory before serving overhead.
That does not satisfy this project's roughly 3 GB ASR memory target on the RTX
3060. Keep Voxtral as an offline experiment rather than a deployed backend.

References: [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B),
[LFM2.5 GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF),
[Voxtral Mini 3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), and
[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/).

## Evaluation protocol

Use 50–100 consented, redacted utterances that cover short commands, long
dictation, numbers, URLs, and project names. Record word error rate, changed
facts, correction fallback rate, p50/p95 end-to-end latency, and cost. Do not
replace the default based on a few anecdotal samples. `whisper-benchmark 7`
summarizes recent local latency; provider comparisons should use the same
corpus and scoring rules.
