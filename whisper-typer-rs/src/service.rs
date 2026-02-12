//! Main service orchestration with state machine.
//!
//! IDLE → RECORDING → PROCESSING → IDLE

use std::path::PathBuf;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::Config;
use crate::hotkey::{HotkeyEvent, HotkeyMonitor};
use crate::recorder::AudioRecorder;

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
}

impl DictationService {
    pub fn new(config: Config) -> Self {
        let recorder = AudioRecorder::new(
            config.audio.clone(),
            config.recording.clone(),
            config.silence.clone(),
        );

        Self {
            config,
            state: ServiceState::Idle,
            recorder,
        }
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

        info!("Service ready — press hotkey to start recording");

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

        let duration = samples.len() as f64 / self.recorder.sample_rate() as f64;
        info!("Captured {:.1}s of audio ({} samples)", duration, samples.len());

        // Phase 1: Save as WAV for testing
        self.save_wav(&samples);

        // TODO Phase 2: Transcribe with whisper-rs
        // TODO Phase 3: Process with Ollama, type with arboard+enigo

        self.state = ServiceState::Idle;
        info!("State: PROCESSING → IDLE");
    }

    /// Save audio samples to a WAV file for Phase 1 testing.
    fn save_wav(&self, samples: &[f32]) {
        let path = PathBuf::from("/tmp/whisper-typer-rs-last.wav");
        let spec = hound::WavSpec {
            channels: self.config.audio.channels,
            sample_rate: self.config.audio.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        match hound::WavWriter::create(&path, spec) {
            Ok(mut writer) => {
                for &sample in samples {
                    // f32 [-1, 1] → i16
                    let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    if writer.write_sample(s).is_err() {
                        break;
                    }
                }
                if writer.finalize().is_ok() {
                    info!("Saved WAV to {}", path.display());
                }
            }
            Err(e) => {
                warn!("Failed to save WAV: {e}");
            }
        }
    }
}
