//! Shared WhisperTyper library used by the service and support binaries.

pub mod code_speaker;
pub mod config;

mod dictation;
mod interfaces;
mod persistence;
mod speech;

// Preserve the original public module paths while the implementation files are
// grouped by responsibility. Existing binaries and downstream code can keep
// using `whisper_typer_rs::history`, `crate::hotkey`, and the other flat paths.
pub use dictation::{hotkey, recorder, service, typer};
pub use interfaces::mcp_server;
pub use persistence::{history, runtime_settings};
pub use speech::{processor, punctuation, remote_asr, transcriber, vad};
