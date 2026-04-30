# Rust Ecosystem Research for Speech-to-Text Dictation Application

Research completed 2026-02-12. This document maps each Python dependency in WhisperTyper
to its Rust equivalent(s), with notes on maturity, CUDA/GPU support, and known issues.

---

## 1. Whisper ASR (replaces HuggingFace transformers + distil-whisper/distil-large-v3)

### Option A: whisper-rs (RECOMMENDED)
- **Crate**: [whisper-rs](https://crates.io/crates/whisper-rs) (v0.15.1)
- **What**: Rust bindings to whisper.cpp (ggml-org/whisper.cpp)
- **CUDA**: Yes, via `cuda` feature flag. Also supports ROCm/hipBLAS, Vulkan, Metal, OpenBLAS.
- **distil-whisper support**: GGML-format weights available at `distil-whisper/distil-large-v3-ggml` on HuggingFace. whisper.cpp has initial distil-whisper support, though chunk-based transcription is not implemented, which can reduce quality on long audio. For short dictation segments (< 30s), this is not an issue.
- **Maturity**: High. Actively maintained on Codeberg. Used by many projects.
- **Known issues**: Build requires cmake and C++ toolchain. CUDA build needs CUDA toolkit installed.
- **Performance**: whisper.cpp with CUDA achieves real-time factors well under 1x on modern GPUs. distil-large-v3-turbo achieves ~216x real-time on GPU.

### Option B: candle (HuggingFace)
- **Crate**: [candle-core](https://crates.io/crates/candle-core), candle-transformers
- **What**: Pure-Rust ML framework from HuggingFace with Whisper example included.
- **CUDA**: Yes, via `cuda` feature. Also supports Metal, MKL, cuDNN.
- **Maturity**: Medium-High. Active development, but Whisper example may not match whisper.cpp quality/speed.
- **Trade-off**: More Rust-native, compiles to single binary, but whisper-rs (via whisper.cpp) has more optimizations and wider testing.

### Option C: whisper-burn
- **Crate**: [whisper-burn](https://github.com/Gadersd/whisper-burn)
- **What**: Whisper implemented in the Burn deep learning framework.
- **CUDA**: Yes, via Burn's WGPU or CUDA backends.
- **Maturity**: Lower. Research/experimental quality. Burn itself is actively developed (v0.20+).

### Recommendation
**whisper-rs** for production use. Best optimized, most tested, direct GGML model support, proven CUDA acceleration. Use distil-large-v3-ggml weights from HuggingFace.

---

## 2. ONNX Runtime (replaces Python onnxruntime for Kokoro TTS)

### ort (RECOMMENDED)
- **Crate**: [ort](https://crates.io/crates/ort) (v2.0.0-rc.11)
- **What**: Safe Rust wrapper for ONNX Runtime 1.23. Maintained by pyke.io.
- **CUDA**: Yes, via `cuda` feature flag. Also supports TensorRT, OpenVINO, DirectML, ROCm, CoreML, XNNPACK, QNN, WebGPU, and many more execution providers.
- **Maturity**: High. Production-recommended despite RC version. Extensive feature flags. Used by kokorox and other TTS projects.
- **Key features**: `download-binaries` feature auto-downloads ONNX Runtime shared libraries. `ModelCompiler` for ahead-of-time optimization. Training support.
- **Known issues**: v2.0 API is not yet fully stable (RC), but is recommended over v1.x for new projects.

### Recommendation
**ort** is the clear winner. No real alternatives exist for ONNX in Rust. The `cuda` feature provides GPU acceleration out of the box.

---

## 3. Audio Capture (replaces PyAudio)

### cpal (RECOMMENDED)
- **Crate**: [cpal](https://crates.io/crates/cpal) (RustAudio project)
- **What**: Cross-platform low-level audio I/O in pure Rust. Supports ALSA, PulseAudio, JACK, WASAPI, CoreAudio.
- **Maturity**: High. The foundational Rust audio library. Used by rodio internally.
- **Features**: Direct PCM capture, configurable sample rate/format/channels, callback-based or blocking. JACK support for ultra-low latency. Always-running stream pattern (matching current WhisperTyper design) is natural with cpal's callback model.
- **Known issues**: ALSA backend can have latency spikes. JACK backend is recommended for lowest latency.

### rodio
- **Crate**: [rodio](https://crates.io/crates/rodio)
- **What**: Higher-level audio playback library built on cpal.
- **Note**: rodio is primarily for **playback**, not capture. Use cpal directly for recording.

### Recommendation
**cpal** for audio capture. Its callback-based stream model maps directly to the always-running audio stream pattern in the current Python implementation. For the 16kHz mono f32 PCM that Whisper expects, cpal can be configured precisely.

---

## 4. Audio Playback (replaces sounddevice for TTS output)

### rodio (RECOMMENDED)
- **Crate**: [rodio](https://crates.io/crates/rodio)
- **What**: High-level audio playback built on cpal. Spawns background thread, handles mixing, supports WAV/MP3/OGG/FLAC decoding.
- **Maturity**: High. Standard choice for Rust audio playback.
- **Features**: Simple `Sink` API for queuing audio, volume control, pause/resume, automatic format conversion.

### cpal (alternative)
- Use cpal directly if you need lower-level control over playback (e.g., streaming PCM from TTS model without intermediate file).

### Recommendation
**rodio** for simple playback of TTS audio. If streaming TTS output sample-by-sample, use **cpal** directly for lower latency.

---

## 5. evdev (replaces Python evdev for hotkey detection)

### evdev (RECOMMENDED)
- **Crate**: [evdev](https://crates.io/crates/evdev) (emberian/evdev)
- **What**: Pure Rust re-implementation of libevdev. Reads input events from /dev/input devices.
- **Maturity**: High. Stable API, actively maintained.
- **Features**: Async support (tokio), device enumeration, uinput support, key state queries.
- **Note**: Works identically to the Python evdev library. Same /dev/input interface, same event types.

### hotkey-listener (alternative)
- **Crate**: [hotkey-listener](https://crates.io/crates/hotkey-listener)
- **What**: Higher-level hotkey detection built on evdev. Supports modifier combos like Shift+F8.
- **Note**: May simplify the hotkey combo detection logic vs raw evdev.

### evdev-rs (alternative)
- **Crate**: [evdev-rs](https://github.com/ndesh26/evdev-rs)
- **What**: Rust bindings to libevdev (C library). Closer to C API.
- **Note**: The pure-Rust `evdev` crate is preferred as it has no C dependency.

### Recommendation
**evdev** crate for direct port. The API maps almost 1:1 to the Python evdev library. Consider **hotkey-listener** if you want built-in modifier key combo handling.

---

## 6. xdotool/xclip (replaces subprocess calls for clipboard paste)

### For Clipboard: arboard (RECOMMENDED)
- **Crate**: [arboard](https://crates.io/crates/arboard) (maintained by 1Password)
- **What**: Cross-platform clipboard access (text + images). Supports X11 by default, Wayland via `wayland-data-control` feature.
- **Maturity**: High. Backed by 1Password.
- **Known issues**: Wayland support requires data-control protocol, not universally available.

### For Keyboard Simulation: enigo (RECOMMENDED)
- **Crate**: [enigo](https://crates.io/crates/enigo) (v0.5.0)
- **What**: Cross-platform keyboard/mouse simulation. Simulates Ctrl+V paste, key presses, etc.
- **Maturity**: Medium-High. X11 support is solid. Wayland support exists behind feature flags but has known bugs.
- **Note**: For the "clipboard paste" pattern (set clipboard then simulate Ctrl+V), combine arboard + enigo.

### Alternative: x11-clipboard
- **Crate**: [x11-clipboard](https://crates.io/crates/x11-clipboard)
- **What**: Low-level X11 clipboard access. X11-only.
- **Note**: arboard is preferred as it abstracts over X11/Wayland.

### Alternative: subprocess xdotool
- Simply call `xdotool` via `std::process::Command`. This is what the Python version does and is the simplest port.
- **Pro**: Zero new dependencies, known-working.
- **Con**: Requires xdotool installed, subprocess overhead.

### Recommendation
**Phase 1**: Keep subprocess calls to xdotool/xclip for simplest port.
**Phase 2**: Replace with arboard (clipboard) + enigo (key simulation) for pure-Rust solution.

---

## 7. Ollama HTTP Client (replaces Python httpx/requests)

### reqwest (RECOMMENDED)
- **Crate**: [reqwest](https://crates.io/crates/reqwest)
- **What**: Ergonomic async HTTP client. De facto standard in Rust.
- **Maturity**: Very High. Used everywhere.
- **Features**: JSON serialization via `json` feature, streaming responses, connection pooling, TLS.

### ollama-rs (higher-level alternative)
- **Crate**: [ollama-rs](https://github.com/pepperoni21/ollama-rs)
- **What**: Dedicated Ollama API client built on reqwest. Typed request/response structs.
- **Maturity**: Medium. Actively maintained.

### ollama-rest (alternative)
- **Crate**: [ollama-rest](https://crates.io/crates/ollama-rest)
- **What**: Async Ollama REST API bindings using reqwest + tokio.

### Recommendation
**reqwest** directly. The Ollama API is simple (POST /api/generate with JSON body). A dedicated client library adds unnecessary abstraction for such a simple API. Use `reqwest::Client` with `serde` for JSON serialization.

---

## 8. MCP Server (replaces Python MCP server)

### rmcp - Official SDK (RECOMMENDED)
- **Crate**: [rmcp](https://crates.io/crates/rmcp) (v0.13.0)
- **What**: Official Rust SDK for Model Context Protocol, maintained at modelcontextprotocol/rust-sdk.
- **Maturity**: Medium-High. Official SDK, actively developed.
- **Features**:
  - Multiple transport mechanisms (stdio, SSE, streamable-HTTP)
  - Procedural macros for automatic tool definition with JSON schema
  - JSON-RPC 2.0 message handling
  - Task lifecycle support (create, inspect, await, cancel)
  - tokio-based async
- **Note**: This is the *official* SDK from the MCP project. Replaces earlier community efforts.

### Alternatives
- **mcp-protocol-sdk** (v0.5.1): Community SDK with multiple transport support.
- **pmcp**: Complete MCP ecosystem with CLI toolkit.
- **mcpr**: Another Rust MCP implementation.

### Recommendation
**rmcp** as the official SDK. The procedural macros for tool definition will make defining whisper_set_mode, whisper_get_status, etc. very ergonomic. SSE transport support matches the current Python implementation.

---

## 9. Desktop Notifications (replaces libnotify/gi.repository.Notify)

### notify-rust (RECOMMENDED)
- **Crate**: [notify-rust](https://crates.io/crates/notify-rust) (v4.x)
- **What**: Desktop notifications for Linux/BSD/macOS using pure Rust D-Bus client.
- **Maturity**: High. Stable, well-maintained.
- **Features**: XDG notification spec support (KDE, GNOME, XFCE, LXDE, Mate), image attachments, action buttons, urgency levels, timeouts.
- **Note**: Pure Rust D-Bus implementation means no C library dependency on Linux.

### Recommendation
**notify-rust**. Direct replacement, no alternatives needed.

---

## 10. YAML Config (replaces Python PyYAML)

### serde_yml (RECOMMENDED)
- **Crate**: [serde_yml](https://crates.io/crates/serde_yml)
- **What**: Continuation of serde_yaml after its deprecation. Serde-based YAML serialization/deserialization.
- **IMPORTANT**: The original **serde_yaml** (by dtolnay) was **deprecated and archived** in March 2024. Do NOT use it.
- **Maturity**: Medium. Fork/continuation of the original. API-compatible.

### Alternative: serde with TOML
- Consider switching config format to TOML (serde_toml is very mature) or keeping YAML with serde_yml.

### Recommendation
**serde_yml** for YAML compatibility with existing config files. If starting fresh, TOML via the `toml` crate would be more idiomatic Rust.

---

## 11. Async Runtime

### tokio (RECOMMENDED)
- **Crate**: [tokio](https://crates.io/crates/tokio)
- **What**: The dominant async runtime for Rust. Used by >60% of async Rust projects.
- **Maturity**: Very High. Industry standard.
- **Features**: Multi-threaded runtime, fs/io/net/process/signal handling, timers, channels, synchronization primitives.
- **Note**: async-std development has stalled. The community has consolidated around tokio. All major libraries (reqwest, axum, ort, rmcp) use tokio.

### Recommendation
**tokio**. Non-negotiable -- every dependency in this stack uses tokio.

---

## 12. HTTP Server (for TTS API, MCP SSE transport)

### axum (RECOMMENDED)
- **Crate**: [axum](https://crates.io/crates/axum) (v0.8.x, v0.9 in development)
- **What**: HTTP framework from the tokio team. Built on hyper + tower.
- **Maturity**: Very High. Officially part of the tokio ecosystem.
- **Features**: SSE support built-in (`axum::response::Sse`), middleware via tower, WebSocket support, extractors, routing.
- **Note**: SSE streaming is well-documented and battle-tested in axum.

### Alternatives
- **actix-web**: Mature but uses its own runtime (can conflict with tokio).
- **warp**: Filter-based API, less popular than axum now.

### Recommendation
**axum**. Best integration with tokio ecosystem. Built-in SSE support is essential for MCP transport and TTS streaming.

---

## 13. Text-to-Speech (Kokoro ONNX model inference)

### kokorox / kokoroxide (RECOMMENDED)
- **Projects**:
  - [kokorox](https://github.com/WismutHansen/kokorox) - Rust Kokoro TTS library + koko CLI
  - [kokoroxide](https://lib.rs/crates/kokoroxide) - High-performance Rust Kokoro TTS using ONNX Runtime
  - [Kokoros](https://github.com/lucasjinreal/Kokoros) - Rust Kokoro with HTTP API streaming and OpenAI-compatible server
- **What**: Native Rust Kokoro TTS inference using ort (ONNX Runtime).
- **Maturity**: Medium. Multiple active implementations.
- **Features**: kokorox includes built-in phonemizer (no external dependencies), end-to-end inference. Kokoros includes HTTP API with streaming support.
- **ONNX Models**: Available at `onnx-community/Kokoro-82M-v1.0-ONNX` on HuggingFace.
- **Known issues**: ONNX Runtime shutdown crashes reported on some Linux distros when interrupting the program.

### DIY with ort
- Load Kokoro ONNX model directly with the `ort` crate. Requires implementing the phonemizer and audio post-processing yourself.

### Recommendation
**kokoroxide** or **kokorox** as a library dependency, or study their code and build a custom integration using **ort** directly. The Kokoros project's HTTP API streaming is closest to the current WhisperTyper Code Speaker architecture.

---

## 14. Wake Word Detection (replaces openwakeword)

### Option A: oww-rs
- **Crate**: [oww-rs](https://crates.io/crates/oww-rs) (v0.0.1)
- **What**: Minimalistic OpenWakeWord inference in Rust using ONNX Runtime.
- **Maturity**: Very Low. Version 0.0.1. May be abandoned or early stage.
- **Note**: Uses the same ONNX models as Python openwakeword.

### Option B: Rustpotter (RECOMMENDED for native Rust)
- **Crate**: [rustpotter](https://github.com/GiviMAD/rustpotter)
- **What**: Personal on-device wake word detection in pure Rust. Uses MFCC-based similarity or neural network classification.
- **Maturity**: Medium. Used in Home Assistant / openHAB ecosystems.
- **Features**: Two detection methods -- wakeword references (audio similarity) and wakeword models (NN classification). No external runtime dependency.

### Option C: DIY with ort
- Load openwakeword ONNX models directly with `ort` crate. The models are small and inference is straightforward (audio features -> classification).

### Recommendation
**Rustpotter** for a native Rust solution, or **DIY with ort** to reuse existing openwakeword ONNX models. oww-rs is too immature.

---

## Summary: Recommended Rust Stack

| Component | Python Library | Rust Replacement | Crate Version | CUDA |
|---|---|---|---|---|
| Whisper ASR | transformers + distil-whisper | **whisper-rs** | 0.15.1 | Yes |
| ONNX Runtime | onnxruntime | **ort** | 2.0.0-rc.11 | Yes |
| Audio Capture | PyAudio | **cpal** | latest | N/A |
| Audio Playback | sounddevice | **rodio** (or cpal) | latest | N/A |
| Hotkey Detection | evdev (Python) | **evdev** (Rust) | latest | N/A |
| Clipboard | xclip subprocess | **arboard** | latest | N/A |
| Key Simulation | xdotool subprocess | **enigo** | 0.5.0 | N/A |
| Ollama Client | httpx/requests | **reqwest** | latest | N/A |
| MCP Server | mcp Python SDK | **rmcp** | 0.13.0 | N/A |
| Notifications | libnotify/GI | **notify-rust** | 4.x | N/A |
| YAML Config | PyYAML | **serde_yml** | latest | N/A |
| Async Runtime | asyncio | **tokio** | latest | N/A |
| HTTP Server | Flask/FastAPI | **axum** | 0.8.x | N/A |
| TTS (Kokoro) | kokoro-onnx (Python) | **kokorox/kokoroxide** | latest | Via ort |
| Wake Word | openwakeword | **rustpotter** or DIY+ort | latest | N/A |

## Key Risks and Considerations

1. **whisper-rs + distil-whisper**: The chunk-based transcription strategy is not implemented in whisper.cpp for distil models. For short dictation (<30s), this is fine. For long audio, quality may suffer vs Python transformers pipeline.

2. **serde_yaml deprecation**: The most popular YAML crate is deprecated. serde_yml is the successor but has less community testing. Consider TOML migration.

3. **Wayland**: Both arboard and enigo have Wayland support behind feature flags, but it's less mature than X11. If targeting Wayland-only desktops, thorough testing is needed.

4. **ort 2.0 RC**: While recommended for new projects, the API may change before stable release. Pin the version carefully.

5. **MCP rmcp**: The official SDK is relatively new (v0.13.0). Documentation may be sparse compared to the Python SDK.

6. **Build complexity**: whisper-rs requires cmake + C++ toolchain + optional CUDA toolkit. ort downloads ONNX Runtime binaries. Consider documenting build requirements carefully.

7. **Binary size**: A single Rust binary with all these dependencies will be large (likely 50-100MB+). Consider feature flags to disable unused backends.
