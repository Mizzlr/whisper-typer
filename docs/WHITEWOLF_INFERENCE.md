# White Wolf inference node

White Wolf (`192.168.0.103`) hosts persistent GPU inference services for
Whisper Typer. The node currently has one 32 GiB DIMM in slot A2 and an NVIDIA
RTX 3060 with 12 GiB VRAM.

## Current services

| Service | Endpoint | Resident model |
|---|---|---|
| `granite-speech5.service` (user unit) | `http://192.168.0.103:8769` | Granite Speech 5.0 470M TurboCTC |
| `punctuation-blackbeast.service` (Black Beast user unit) | `http://127.0.0.1:8770` | `punctuation_fullstop_truecase_english` |

White Wolf's ASR listener is restricted by its firewall to Black Beast at
`192.168.0.100`; Black Beast's punctuation listener is loopback-only. Both
models load and warm during service startup. The earlier White Wolf
Distil-Whisper service on port 8768 and Ollama grammar service on port 11434
are retained as deployment options but disabled during this trial.

Repository deployment inputs:

- `infra/systemd/whisper-asr-whitewolf.service`
- `infra/systemd/ollama-whitewolf.conf`
- `infra/systemd/ollama-whitewolf-warm.service`
- `infra/whitewolf-asr-config.yaml`

## API checks

```bash
curl -fsS http://192.168.0.103:8769/health
curl -fsS http://127.0.0.1:8770/health
curl -fsS -F audio=@sample.wav http://192.168.0.103:8769/transcribe | jq
```

The ASR endpoint accepts a multipart field named `audio` and returns the text,
audio duration, server latency, and timestamped segments. Its request limit is
32 MiB.

## Evaluation corpus

Captured audio remains private and is retained locally for seven days under:

```text
~/.whisper-typer-history/audio/YYYY-MM-DD/*.wav
```

The synchronized evaluation copy on White Wolf is private to the user:

```text
~/.local/share/whisper-asr/corpus/audio/
~/.local/share/whisper-asr/corpus/manifest.jsonl
```

The manifest contains only records that have a corresponding WAV. Do not
commit either the audio or transcript manifest to Git.

### 2026-09-03 baseline

- 21 WAV samples, 9.48 MiB, totaling 310.6 seconds of speech
- 21/21 remote ASR outputs matched the current local Whisper output after
  case and whitespace normalization
- server latency: 370 ms median, 835 ms p95, 847 ms maximum
- end-to-end HTTP latency: 374 ms median, 852 ms p95, 860 ms maximum
- aggregate ASR real-time factor: 0.0307 (about 32.5x real time)
- warm Granite grammar request: 235 ms end to end on the checked sample

This baseline checks reproducibility against the current transcription, not
word-error accuracy against a human-authored reference transcript.

### 2026-09-04 model evaluation and cutover

The expanded evaluation set contains 115 linked WAVs totaling 1,316.04
seconds (21 minutes 56 seconds). Transcript comparisons below use the existing
Distil-Whisper v3 outputs as a reference, not human-authored ground truth.

| Speech model | Whole-corpus wall time | Throughput | Difference from v3 | Decision |
|---|---:|---:|---:|---|
| NVIDIA Nemotron 3.5 ASR Streaming 0.6B q8 | 7.26 s | 181x real time | 15.17% word edits | Reject for final text; dropped words in spot checks |
| NVIDIA Parakeet TDT 0.6B v3 q8 | 4.75 s | 277x real time | 21.22% word edits | Reject; introduced repeated words |
| Distil-Whisper large-v3.5 | 47.27 s of summed server latency | 360 ms median/823 ms p95 per request | 5.92% word edits | Deployed on White Wolf |

On the same 115 requests, Distil-Whisper large-v3 took 367 ms median and 830 ms
p95. Version 3.5 is therefore a quality-oriented update on this RTX 3060, not
a 2-3x speedup.

The grammar comparison used the production structured-JSON prompt over 60
captured dictations:

| Grammar model | Valid responses | Median HTTP latency | Exact match to Granite | Edits from input | Decision |
|---|---:|---:|---:|---:|---|
| Granite 4.1 3B Q4_K_M | 60/60 | 268 ms | baseline | 14 | Keep in production |
| Qwen 3.5 0.8B | 60/60 | 193 ms | 31/60 | 73 | Reject; over-edits meaning/domain terms |
| Qwen 3.5 2B | 60/60 | 322 ms | 36/60 | 67 | Reject; slower and less conservative |

An experimental grammar cutover sent five live requests from Black Beast to
White Wolf, with 267 ms median grammar latency. It was reverted at the user's
request. Production Whisper Typer therefore continues to run both local
Distil-Whisper v3 ASR and local Granite correction on Black Beast. White Wolf
is an evaluation host only until a candidate passes audio-ground-truth review.

### 2026-09-04 expanded backend comparison

Each backend processed the same 115 clips (1,316.02 seconds). `Difference`
means normalized word-edit distance from the saved Whisper v3 transcript; it
is not human-ground-truth WER, so a difference can be either an improvement or
a regression.

| Model and backend | Median | P95 | Corpus throughput | Difference | Exact clips |
|---|---:|---:|---:|---:|---:|
| Distil-Whisper v3, whisper.cpp baseline | 367 ms | 830 ms | about 27x | baseline | 115/115 |
| Distil-Whisper v3.5, whisper.cpp | 360 ms | 823 ms | about 28x | 5.92% | 57/115 |
| Distil-Whisper v3, faster-whisper FP16 | 282 ms | 631 ms | 35.57x | 1.84% | 86/115 |
| Whisper large-v3-turbo, faster-whisper FP16 | 296 ms | 686 ms | 33.10x | 6.97% | 47/115 |
| Granite Speech 4.1 2B BF16 | 368 ms | 1,564 ms | 20.67x | 13.54% | 24/115 |
| Granite Speech 5.0 470M TurboCTC BF16 | 13 ms | 34 ms | 504.02x | 11.12% | 40/115 |

Granite Speech 4.1 allocated about 4.4 GiB of VRAM by itself; Granite Speech
5.0 allocated about 0.9 GiB. For fair memory isolation, the resident ASR and
Ollama services were stopped before the 4.1 run and restored afterward.

The strongest low-risk engineering candidate is faster-whisper with
Distil-Whisper. Granite Speech 5.0 is the strongest latency candidate, but its
domain-word and repetition regressions require a human-checked evaluation set.

### Granite Speech 5.0 live trial

At the user's request, production dictation was subsequently switched to an
isolated Granite Speech 5.0 TurboCTC service on White Wolf:

- endpoint: `http://192.168.0.103:8769/transcribe`
- Black Beast performs capture and typing; White Wolf performs ASR
- Voice Journal also uses this endpoint for ambient chunks and accepts its
  plain-text response shape; it retries Black Beast's local `/transcribe`
  endpoint if White Wolf is unavailable
- Ollama grammar correction and the local TSV replacement layer are disabled
- Voice Journal always retains the raw transcript in its `.unfiltered.md`
  stream; its semantic LLM filter is disabled for this trial
- local Distil-Whisper stays loaded as automatic fallback if remote ASR fails
- the prior White Wolf Whisper and Ollama services are disabled during the trial

A 5.97-second saved clip took 14.3 ms for server inference and 18.4 ms for the
complete LAN request. TurboCTC was the only GPU process on White Wolf and used
about 1.1 GiB of the 12 GiB VRAM. The experiment is deliberately collecting raw
TurboCTC output; do not add post-processing until its error modes are reviewed.
Standalone filler-only runs composed of `okay`, `ok`, and `yeah` are now
excluded from the curated Voice Journal and live view, while remaining intact
in the unfiltered evidence stream. Real utterances containing those words are
preserved.

### Low-latency desktop paste

Black Beast keeps X11 clipboard ownership inside the long-running Whisper
Typer process and uses `xdotool` only to emit the Dvorak-safe paste shortcut.
The legacy `xclip` synchronization path remains an automatic fallback.

The first two live requests after deployment measured a 0.0 ms clipboard
handoff and a 17.1 ms paste, with no fallback. This reduced the typing stage
from 183–222 ms to 17 ms and reduced complete post-recording latency to 67–90
ms on those samples.

### Live punctuation trial

Black Beast now runs a resident CUDA service for
`1-800-BAD-CODE/punctuation_fullstop_truecase_english` on
`http://127.0.0.1:8770`. Dictation uses it after ASR and fails open to the raw
transcript if the service is unavailable. The first live requests measured
37–41 ms model latency. Ollama correction remains disabled. Voice Journal's
unfiltered stream stores `whisper_text` (the exact raw ASR), while its curated
dictated entries use the final punctuated text.

## Hardware-safety boundary

The single-DIMM A2 memory test was stopped after 18 hours 30 minutes with zero
hardware incidents and zero errors. That is strong evidence for this isolated
configuration, but it does not certify the removed DIMMs or all four-DIMM
operation.

ClickHouse, ClickHouse Keeper, Redpanda, and RustFS remain stopped. Do not
restore storage workloads until the previously reported ClickHouse checksum
damage has been audited independently; a clean RAM test cannot repair or
validate existing files.
