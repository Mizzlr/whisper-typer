//! Main service orchestration with state machine.
//!
//! IDLE → RECORDING → PROCESSING → IDLE

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use chrono::Local;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::Config;
use crate::history::{self, TranscriptionRecord};
use crate::hotkey::{HotkeyEvent, HotkeyMonitor};
use crate::notifier::Notifier;
use crate::processor::OllamaProcessor;
use crate::recorder::AudioRecorder;
use crate::transcriber::WhisperTranscriber;
use crate::typer::TextTyper;

/// MCP state file path.
fn state_file() -> PathBuf {
    dirs::home_dir()
        .expect("No home directory")
        .join(".cache/whisper-typer/state.json")
}

/// Output mode: what gets typed into the active window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Only Ollama-corrected text
    Ollama,
    /// Only raw Whisper transcription
    Whisper,
    /// Corrected text + [raw] in brackets
    Both,
}

impl OutputMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "whisper" => Self::Whisper,
            "both" => Self::Both,
            _ => Self::Ollama,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Ollama => "ollama_only",
            Self::Whisper => "whisper_only",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceState {
    Idle,
    Recording,
    Processing,
}

impl std::fmt::Display for ServiceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "IDLE"),
            Self::Recording => write!(f, "RECORDING"),
            Self::Processing => write!(f, "PROCESSING"),
        }
    }
}

pub struct DictationService {
    config: Config,
    state: ServiceState,
    recorder: AudioRecorder,
    transcriber: WhisperTranscriber,
    processor: OllamaProcessor,
    typer: TextTyper,
    notifier: Notifier,
    output_mode: OutputMode,
    recent_transcriptions: Vec<String>,
}

impl DictationService {
    pub fn new(
        config: Config,
        transcriber: WhisperTranscriber,
        output_mode: OutputMode,
    ) -> Self {
        let recorder = AudioRecorder::new(
            config.audio.clone(),
            config.recording.clone(),
            config.silence.clone(),
        );
        let processor = OllamaProcessor::new(config.ollama.clone());
        let typer = TextTyper::new(&config.typer);
        let notifier = Notifier::new(config.feedback.notifications);

        let svc = Self {
            config,
            state: ServiceState::Idle,
            recorder,
            transcriber,
            processor,
            typer,
            notifier,
            output_mode,
            recent_transcriptions: Vec::new(),
        };
        svc.write_mcp_state();
        svc
    }

    /// Write current state to MCP state file for external control.
    fn write_mcp_state(&self) {
        let path = state_file();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let state = json!({
            "output_mode": self.output_mode.as_str(),
            "ollama_enabled": self.config.ollama.enabled,
            "recent_transcriptions": self.recent_transcriptions,
        });

        if let Err(e) = fs::write(&path, serde_json::to_string_pretty(&state).unwrap()) {
            warn!("Failed to write MCP state: {e}");
        }
    }

    /// Add a transcription to the recent list and update state file.
    fn add_transcription(&mut self, text: &str) {
        self.recent_transcriptions.push(text.to_string());
        if self.recent_transcriptions.len() > 20 {
            let excess = self.recent_transcriptions.len() - 20;
            self.recent_transcriptions.drain(..excess);
        }
        self.write_mcp_state();
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Open audio stream (always-on for low latency)
        self.recorder.open_stream()?;

        // Create hotkey channel
        let (hotkey_tx, mut hotkey_rx) = mpsc::channel::<HotkeyEvent>(16);

        // Start hotkey monitor in background
        let hotkey_monitor = HotkeyMonitor::new(&self.config.hotkey, hotkey_tx);
        tokio::spawn(async move {
            hotkey_monitor.run().await;
        });

        info!("Service ready — press hotkey to start recording (mode: {:?})", self.output_mode);

        // Auto-stop poll interval
        let mut auto_stop_interval = tokio::time::interval(tokio::time::Duration::from_millis(100));

        loop {
            tokio::select! {
                event = hotkey_rx.recv() => {
                    match event {
                        Some(HotkeyEvent::Pressed) => self.on_hotkey_press(),
                        Some(HotkeyEvent::Released) => self.on_hotkey_release().await,
                        None => {
                            warn!("Hotkey channel closed");
                            break;
                        }
                    }
                }
                _ = auto_stop_interval.tick() => {
                    // Check for silence-triggered auto-stop
                    if self.state == ServiceState::Recording && self.recorder.should_auto_stop() {
                        info!("Auto-stop triggered by silence detection");
                        self.on_hotkey_release().await;
                    }
                }
            }
        }

        Ok(())
    }

    fn on_hotkey_press(&mut self) {
        if self.state != ServiceState::Idle {
            return;
        }

        self.state = ServiceState::Recording;
        self.recorder.start();
        info!("State: IDLE → RECORDING");
    }

    async fn on_hotkey_release(&mut self) {
        if self.state != ServiceState::Recording {
            return;
        }

        let t_start = Instant::now();
        self.state = ServiceState::Processing;
        info!("State: RECORDING → PROCESSING");

        let samples = self.recorder.stop();

        if samples.is_empty() {
            info!("No audio captured, returning to IDLE");
            self.state = ServiceState::Idle;
            return;
        }

        // Check if audio is too quiet (silence)
        if AudioRecorder::is_silent(&samples, self.config.silence.threshold) {
            info!("Audio is silent, skipping transcription");
            self.state = ServiceState::Idle;
            return;
        }

        let audio_duration = samples.len() as f64 / self.recorder.sample_rate() as f64;
        info!("Captured {:.1}s of audio ({} samples)", audio_duration, samples.len());

        // --- Whisper transcription ---
        let t_whisper_start = Instant::now();
        let transcriber = self.transcriber.clone();
        let raw_text = match tokio::task::spawn_blocking(move || transcriber.transcribe(&samples)).await {
            Ok(Ok(result)) => {
                info!(
                    "Transcription ({:.0}ms): \"{}\"",
                    result.latency_ms, result.text
                );
                result.text
            }
            Ok(Err(e)) => {
                warn!("Transcription failed: {e}");
                self.state = ServiceState::Idle;
                info!("State: PROCESSING → IDLE");
                return;
            }
            Err(e) => {
                warn!("Transcription task panicked: {e}");
                self.state = ServiceState::Idle;
                info!("State: PROCESSING → IDLE");
                return;
            }
        };
        let t_whisper = t_whisper_start.elapsed().as_secs_f64() * 1000.0;

        if raw_text.is_empty() {
            info!("Empty transcription, returning to IDLE");
            self.state = ServiceState::Idle;
            info!("State: PROCESSING → IDLE");
            return;
        }

        // --- Ollama correction ---
        let t_ollama_start = Instant::now();
        let (processed_text, ollama_text) = match self.output_mode {
            OutputMode::Whisper => (None, None),
            OutputMode::Ollama | OutputMode::Both => {
                let corrected = self.processor.process(&raw_text).await;
                info!("Ollama corrected: \"{}\"", corrected);
                (Some(corrected.clone()), Some(corrected))
            }
        };
        let t_ollama = t_ollama_start.elapsed().as_secs_f64() * 1000.0;

        // Build final output
        let final_text = match self.output_mode {
            OutputMode::Whisper => format!("{raw_text} "),
            OutputMode::Ollama => format!("{} ", processed_text.as_deref().unwrap_or(&raw_text)),
            OutputMode::Both => {
                format!("{} [{raw_text}] ", processed_text.as_deref().unwrap_or(&raw_text))
            }
        };

        // --- Type into active window ---
        let t_type_start = Instant::now();
        self.typer.type_text(&final_text);
        let t_type = t_type_start.elapsed().as_secs_f64() * 1000.0;

        let t_total = t_start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "  Whisper: {:.0}ms | Ollama: {:.0}ms | Typing: {:.0}ms | Total: {:.0}ms | Audio: {:.1}s | Speed: {:.1}x",
            t_whisper, t_ollama, t_type, t_total, audio_duration,
            if t_total > 0.0 { (audio_duration * 1000.0) / t_total } else { 0.0 }
        );

        self.notifier.notify("Whisper Typer", &final_text);
        self.add_transcription(&final_text);

        // --- Save history record ---
        let speed_ratio = if t_total > 0.0 {
            ((audio_duration * 1000.0) / t_total * 10.0).round() / 10.0
        } else {
            0.0
        };

        let record = TranscriptionRecord {
            timestamp: Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string(),
            whisper_text: raw_text,
            ollama_text: if t_ollama > 0.0 { ollama_text } else { None },
            final_text: final_text.clone(),
            output_mode: self.output_mode.as_str().to_string(),
            whisper_latency_ms: t_whisper as i64,
            ollama_latency_ms: if t_ollama > 0.0 { Some(t_ollama as i64) } else { None },
            typing_latency_ms: t_type as i64,
            total_latency_ms: t_total as i64,
            audio_duration_s: (audio_duration * 100.0).round() / 100.0,
            char_count: final_text.len(),
            word_count: final_text.split_whitespace().count(),
            speed_ratio,
        };
        history::save_record(&record);

        self.state = ServiceState::Idle;
        info!("State: PROCESSING → IDLE");
    }
}
