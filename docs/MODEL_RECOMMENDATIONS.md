# Model recommendations

These recommendations are based on measurements from the production request
shape, not general leaderboard scores. Results below were collected on an RTX
3060 12 GB and should be re-measured on different hardware. Last reviewed:
2026-09-04.

## Production default

- Speech-to-text: Granite Speech 5.0 TurboCTC on White Wolf is the current
  trial primary. Keep Distil-Whisper large-v3 warm on Black Beast as automatic
  fallback when the remote endpoint is unavailable.
- Grammar correction: disabled during the raw Granite Speech 5 TurboCTC trial.
  `granite4.2:3b` is installed as the local Ollama successor, but must not be
  promoted until it passes the saved-corpus suite. The correction guard falls
  back to raw ASR text when a model changes facts or produces malformed output.
- Text-to-speech: keep Kokoro v1.0 and the preferred local voice. It avoids a
  network round trip and is already integrated with interruption/voice gating.

## Local evaluations

### Grammar correction

| Model | Result | Decision |
|---|---|---|
| Granite 4.2 3B | Initial 2026-09-04 structured-output smoke test over-generated invented instructions and took 6.9 s | Installed experiment only; correction remains disabled |
| Granite 4.1 3B | Warm median about 573 ms across the constrained-JSON correction suite; handled the command-rewrite cases consistently | Historical baseline; removed locally after the 4.2 download |
| LFM2.5-2.6B QAD-Q4_0 | Warm median about 485 ms, but tail latency reached about 1.8 s; echoed prompt instructions and missed required command rewrites | Installed experiment only |
| Qwen3.5 2B | About 17% faster than Granite in an earlier local run, with weaker punctuation behavior | Do not promote without a larger corpus |
| Qwen3.5 4B | Slower than Granite in the same local run | Not recommended for this latency-sensitive path |

All models tried to reinterpret `3.14 PM` as `3:14 PM`; the correction validator
therefore remains necessary even when model output looks fluent.

### Punctuation and truecasing

`1-800-BAD-CODE/punctuation_fullstop_truecase_english` was evaluated through
its portable ONNX model on Black Beast. Across the same 23 genuine raw
TurboCTC transcripts, CPU execution measured 72.47 ms median and 124.99 ms
p95; NVIDIA CUDA execution measured 30.76 ms median and 36.65 ms p95 while
using about 386 MiB VRAM. It restored ordinary questions well, but made
domain-casing and segmentation errors such as
`BlackBe`, `The Voice dictations`, and awkward `Ubuntu, Nvidia` punctuation.
It is therefore enabled only as an experimental live trial, with raw ASR
preserved by Voice Journal and fail-open behavior if the punctuation service
is unavailable. Do not treat it as a validated replacement until a terminology
layer and human-scored corpus show that the combined output is safer than raw
TurboCTC.

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
