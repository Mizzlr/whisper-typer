//! Live runtime settings shared by dictation and MCP.
//!
//! The in-memory state is authoritative. A private, atomically replaced JSON
//! file preserves settings and recent transcriptions across service restarts.

use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, RwLock};
use tracing::{debug, warn};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Ollama,
    Whisper,
    Both,
}

impl OutputMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "whisper" | "whisper_only" => Some(Self::Whisper),
            "both" => Some(Self::Both),
            "ollama" | "ollama_only" => Some(Self::Ollama),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama_only",
            Self::Whisper => "whisper_only",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeSnapshot {
    pub output_mode: OutputMode,
    pub ollama_enabled: bool,
    pub recent_transcriptions: Vec<String>,
}

pub struct RuntimeSettings {
    state: RwLock<RuntimeSnapshot>,
    path: PathBuf,
    persist_lock: Mutex<()>,
}

impl RuntimeSettings {
    pub fn load(
        initial_mode: OutputMode,
        initial_ollama_enabled: bool,
        use_persisted_controls: bool,
    ) -> Self {
        let path = state_file();
        let persisted = fs::read_to_string(&path)
            .ok()
            .and_then(|contents| serde_json::from_str::<serde_json::Value>(&contents).ok());

        let output_mode = if use_persisted_controls {
            persisted
                .as_ref()
                .and_then(|state| state["output_mode"].as_str())
                .and_then(OutputMode::parse)
                .unwrap_or(initial_mode)
        } else {
            initial_mode
        };
        let ollama_enabled = if use_persisted_controls {
            persisted
                .as_ref()
                .and_then(|state| state["ollama_enabled"].as_bool())
                .unwrap_or(initial_ollama_enabled)
        } else {
            initial_ollama_enabled
        };
        let recent_transcriptions = persisted
            .as_ref()
            .and_then(|state| state["recent_transcriptions"].as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_owned))
                    .rev()
                    .take(20)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            })
            .unwrap_or_default();

        let settings = Self {
            state: RwLock::new(RuntimeSnapshot {
                output_mode,
                ollama_enabled,
                recent_transcriptions,
            }),
            path,
            persist_lock: Mutex::new(()),
        };
        settings.persist();
        settings
    }

    pub fn snapshot(&self) -> RuntimeSnapshot {
        self.state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn set_mode(&self, mode: OutputMode) {
        self.state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .output_mode = mode;
        self.persist();
    }

    pub fn set_ollama_enabled(&self, enabled: bool) {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.ollama_enabled = enabled;
        if !enabled {
            state.output_mode = OutputMode::Whisper;
        } else if state.output_mode == OutputMode::Whisper {
            state.output_mode = OutputMode::Ollama;
        }
        drop(state);
        self.persist();
    }

    pub fn add_transcription(&self, text: &str) {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.recent_transcriptions.push(text.to_string());
        let excess = state.recent_transcriptions.len().saturating_sub(20);
        if excess > 0 {
            state.recent_transcriptions.drain(..excess);
        }
        drop(state);
        self.persist();
    }

    fn persist(&self) {
        let _persist_guard = self
            .persist_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let snapshot = self.snapshot();
        let value = json!({
            "output_mode": snapshot.output_mode.as_str(),
            "ollama_enabled": snapshot.ollama_enabled,
            "recent_transcriptions": snapshot.recent_transcriptions,
        });
        let Some(parent) = self.path.parent() else {
            return;
        };
        if let Err(error) = fs::create_dir_all(parent) {
            warn!("Failed to create runtime state directory: {error}");
            return;
        }

        let temp_path = parent.join(format!(".state.json.tmp-{}", std::process::id()));
        let mut options = fs::OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        options.mode(0o600);

        let result = (|| -> Result<(), String> {
            let mut file = options
                .open(&temp_path)
                .map_err(|error| format!("open temporary state: {error}"))?;
            serde_json::to_writer_pretty(&mut file, &value)
                .map_err(|error| format!("serialize state: {error}"))?;
            file.write_all(b"\n")
                .map_err(|error| format!("write state newline: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("sync temporary state: {error}"))?;
            fs::rename(&temp_path, &self.path)
                .map_err(|error| format!("replace state: {error}"))?;
            Ok(())
        })();

        if let Err(error) = result {
            let _ = fs::remove_file(&temp_path);
            warn!("Failed to persist runtime state: {error}");
        } else {
            debug!("Persisted runtime state to {}", self.path.display());
        }
    }
}

fn state_file() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/whisper-typer/state.json")
}

#[cfg(test)]
mod tests {
    use super::OutputMode;

    #[test]
    fn output_mode_accepts_public_and_persisted_names() {
        assert_eq!(OutputMode::parse("whisper"), Some(OutputMode::Whisper));
        assert_eq!(OutputMode::parse("whisper_only"), Some(OutputMode::Whisper));
        assert_eq!(OutputMode::parse("both"), Some(OutputMode::Both));
        assert_eq!(OutputMode::parse("ollama_only"), Some(OutputMode::Ollama));
        assert_eq!(OutputMode::parse("typo"), None);
    }
}
