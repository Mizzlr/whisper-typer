//! Code Speaker: native Kokoro TTS for Claude Code voice notifications.
//!
//! Replaces the Python code-speaker service with native Rust implementation.
//! Components:
//! - `tts`: Kokoro ONNX model inference + rodio playback with cancellation
//! - `api`: Axum HTTP server (port 8767) for hook integration
//! - `summarizer`: Ollama text summarization for long TTS input
//! - `reminder`: Periodic reminder manager
//! - `history`: TTS event history and reporting

pub mod api;
#[allow(dead_code)]
pub mod history;
pub mod reminder;
pub mod summarizer;
#[allow(dead_code)]
pub mod transcript;
pub mod tts;
