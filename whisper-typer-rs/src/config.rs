//! Configuration management for whisper-typer-rs.
//!
//! Loads config from YAML files in standard locations, matching
//! the Python whisper-typer config.yaml format exactly.

use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct HotkeyConfig {
    pub combo: Vec<String>,
    pub alt_combos: Vec<Vec<String>>,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            combo: vec!["KEY_LEFTMETA".into(), "KEY_LEFTALT".into()],
            alt_combos: vec![],
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub device_index: Option<u32>,
    pub channels: u16,
    pub chunk_size: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            device_index: None,
            channels: 1,
            chunk_size: 1024,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RecordingConfig {
    pub max_duration: f64,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            max_duration: 120.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WhisperConfig {
    pub model: String,
    pub device: String,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model: "distil-whisper/distil-large-v3".into(),
            device: "cuda".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OllamaConfig {
    pub enabled: bool,
    pub model: String,
    pub host: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "llama3.2:3b".into(),
            host: "http://localhost:11434".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TyperConfig {
    pub backend: String,
}

impl Default for TyperConfig {
    fn default() -> Self {
        Self {
            backend: "ydotool".into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FeedbackConfig {
    pub notifications: bool,
    pub sounds: bool,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            notifications: true,
            sounds: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SilenceConfig {
    pub threshold: f32,
    pub duration: f64,
    pub min_speech_duration: f64,
    pub max_recording_duration: f64,
}

impl Default for SilenceConfig {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            duration: 1.5,
            min_speech_duration: 0.5,
            max_recording_duration: 30.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TTSConfig {
    pub enabled: bool,
    pub voice: String,
    pub speed: f32,
    pub api_port: u16,
    pub max_direct_chars: usize,
    pub reminder_interval: u64,
    pub model_path: String,
}

impl Default for TTSConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            voice: "af_heart".into(),
            speed: 1.0,
            api_port: 8767,
            max_direct_chars: 150,
            reminder_interval: 300,
            model_path: String::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct McpConfig {
    pub enabled: bool,
    pub port: u16,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8766,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub hotkey: HotkeyConfig,
    pub audio: AudioConfig,
    pub recording: RecordingConfig,
    pub whisper: WhisperConfig,
    pub ollama: OllamaConfig,
    pub typer: TyperConfig,
    pub feedback: FeedbackConfig,
    pub silence: SilenceConfig,
    pub tts: TTSConfig,
    pub mcp: McpConfig,
}

impl Config {
    /// Load configuration from YAML file.
    ///
    /// Searches standard locations if no path is provided:
    /// 1. ./config.yaml
    /// 2. ~/.config/whisper-input/config.yaml
    /// 3. /etc/whisper-input/config.yaml
    pub fn load(path: Option<&Path>) -> Self {
        let resolved = path.map(PathBuf::from).or_else(|| {
            let candidates = [
                std::env::current_dir().ok().map(|d| d.join("config.yaml")),
                dirs::home_dir().map(|h| h.join(".config/whisper-input/config.yaml")),
                Some(PathBuf::from("/etc/whisper-input/config.yaml")),
            ];
            candidates.into_iter().flatten().find(|p| p.exists())
        });

        let Some(config_path) = resolved else {
            info!("No config file found, using defaults");
            return Self::default();
        };

        match std::fs::read_to_string(&config_path) {
            Ok(contents) => match serde_yml::from_str(&contents) {
                Ok(config) => {
                    info!("Loaded config from {}", config_path.display());
                    config
                }
                Err(e) => {
                    tracing::warn!("Failed to parse {}: {e}, using defaults", config_path.display());
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!("Failed to read {}: {e}, using defaults", config_path.display());
                Self::default()
            }
        }
    }
}
