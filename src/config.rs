//! Configuration management for whisper-typer-rs.
//!
//! Loads config from YAML files in standard locations, matching
//! the Python whisper-typer config.yaml format exactly.

use serde::Deserialize;
use std::fmt;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

#[derive(Debug)]
pub struct ConfigError(String);

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

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
            model: "models/ggml-distil-large-v3.bin".into(),
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
    pub keep_alive: i64,
    pub skip_threshold: usize,
    /// When true, bypass Whisper and send audio directly to Ollama (requires audio-capable model).
    pub audio_mode: bool,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "granite4.1:3b".into(),
            host: "http://localhost:11434".into(),
            keep_alive: 3600,
            skip_threshold: 0,
            audio_mode: false,
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
            backend: "xdotool".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct FeedbackConfig {
    pub notifications: bool,
    pub sounds: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WakewordConfig {
    pub enabled: bool,
    pub model: String,
    pub threshold: f32,
}

impl Default for WakewordConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "alexa".into(),
            threshold: 0.5,
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
    pub model_path: String,
}

impl Default for TTSConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            voice: "af_heart".into(),
            speed: 1.0,
            api_port: 8767,
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

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CorrectionsConfig {
    pub enabled: bool,
    pub path: String,
}

impl Default for CorrectionsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "~/.config/whisper-typer/corrections.tsv".into(),
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
    pub wakeword: WakewordConfig,
    pub silence: SilenceConfig,
    pub tts: TTSConfig,
    pub mcp: McpConfig,
    pub corrections: CorrectionsConfig,
}

impl Config {
    /// Load configuration from YAML file.
    ///
    /// Searches standard locations if no path is provided:
    /// 1. An explicitly supplied `--config` path
    /// 2. `./config.yaml`
    /// 3. `~/.config/whisper-typer/config.yaml`
    /// 4. Legacy whisper-input locations
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let resolved = path.map(PathBuf::from).or_else(|| {
            let candidates = [
                std::env::current_dir().ok().map(|d| d.join("config.yaml")),
                dirs::home_dir().map(|h| h.join(".config/whisper-typer/config.yaml")),
                dirs::home_dir().map(|h| h.join(".config/whisper-input/config.yaml")),
                Some(PathBuf::from("/etc/whisper-input/config.yaml")),
            ];
            candidates.into_iter().flatten().find(|p| p.exists())
        });

        let Some(config_path) = resolved else {
            info!("No config file found, using defaults");
            return Ok(Self::default());
        };

        let contents = std::fs::read_to_string(&config_path).map_err(|error| {
            ConfigError(format!(
                "Failed to read configuration {}: {error}",
                config_path.display()
            ))
        })?;
        let config: Self = serde_yml::from_str(&contents).map_err(|error| {
            ConfigError(format!(
                "Failed to parse configuration {}: {error}",
                config_path.display()
            ))
        })?;
        config.validate()?;
        config.log_deprecations();
        info!("Loaded config from {}", config_path.display());
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.audio.sample_rate != 16_000 {
            return Err(ConfigError(format!(
                "audio.sample_rate must be 16000, got {}",
                self.audio.sample_rate
            )));
        }
        if self.audio.channels != 1 {
            return Err(ConfigError(format!(
                "audio.channels must be 1, got {}",
                self.audio.channels
            )));
        }
        if self.audio.chunk_size == 0 {
            return Err(ConfigError(
                "audio.chunk_size must be greater than zero".into(),
            ));
        }
        if self.recording.max_duration <= 0.0 || self.silence.max_recording_duration <= 0.0 {
            return Err(ConfigError(
                "recording durations must be greater than zero".into(),
            ));
        }
        if self.silence.threshold < 0.0 || self.silence.duration < 0.0 {
            return Err(ConfigError(
                "silence threshold and duration cannot be negative".into(),
            ));
        }
        if self.silence.min_speech_duration < 0.0 {
            return Err(ConfigError(
                "silence.min_speech_duration cannot be negative".into(),
            ));
        }
        if self.tts.speed <= 0.0 {
            return Err(ConfigError("tts.speed must be greater than zero".into()));
        }
        if self.mcp.enabled && self.tts.enabled && self.mcp.port == self.tts.api_port {
            return Err(ConfigError(
                "mcp.port and tts.api_port must be different".into(),
            ));
        }
        Ok(())
    }

    fn log_deprecations(&self) {
        if self.ollama.audio_mode {
            warn!("ollama.audio_mode is deprecated and will use the reliable Whisper path");
        }
        if self.wakeword.enabled {
            warn!("wakeword configuration is deprecated and is not active in the Rust service");
        }
        if self.feedback.notifications || self.feedback.sounds {
            warn!("feedback configuration is deprecated and currently has no effect");
        }
        if self.typer.backend != "xdotool" && self.typer.backend != "enigo" {
            warn!(
                "Unknown typer backend '{}'; the existing enigo fallback behavior will apply",
                self.typer.backend
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use std::path::Path;

    #[test]
    fn rejects_audio_rates_that_whisper_cannot_consume() {
        let mut config = Config::default();
        config.audio.sample_rate = 48_000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn accepts_default_configuration() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn repository_configuration_is_valid() {
        Config::load(Some(Path::new("config.yaml"))).expect("valid repository config");
    }
}
