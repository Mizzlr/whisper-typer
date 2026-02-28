//! Main service orchestration with state machine.
//!
//! IDLE → RECORDING → PROCESSING → IDLE
//!
//! Voice gate: TTS is suppressed during recording/processing.
//! When the user presses the hotkey, any active TTS is cancelled immediately.
//! TTS waits for `voice_idle` before playing so it never talks over the user.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use chrono::Local;
use serde_json::json;
use tokio::sync::{mpsc, Notify};
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::history::{self, TranscriptionRecord};
use crate::hotkey::{HotkeyEvent, HotkeyMonitor};
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

/// Shared voice gate for TTS coordination.
///
/// - `is_idle` is true when the user is NOT recording/processing (TTS may play).
/// - `idle_notify` is signalled when transitioning back to idle (TTS waiters wake up).
/// - `cancel_tts` is signalled when recording starts (active TTS should stop immediately).
#[derive(Clone)]
pub struct VoiceGate {
    /// True when voice input is idle (safe for TTS to play).
    pub is_idle: Arc<AtomicBool>,
    /// Notified when voice input returns to idle.
    pub idle_notify: Arc<Notify>,
    /// Notified when TTS should be cancelled (user started speaking).
    pub cancel_notify: Arc<Notify>,
}

impl VoiceGate {
    pub fn new() -> Self {
        Self {
            is_idle: Arc::new(AtomicBool::new(true)),
            idle_notify: Arc::new(Notify::new()),
            cancel_notify: Arc::new(Notify::new()),
        }
    }



    /// Signal that voice input has started (suppress + cancel TTS).
    fn begin_voice_input(&self) {
        self.is_idle.store(false, Ordering::Relaxed);
        self.cancel_notify.notify_waiters();
        debug!("Voice gate: closed (TTS suppressed)");
    }

    /// Signal that voice input has ended (TTS may resume).
    fn end_voice_input(&self) {
        self.is_idle.store(true, Ordering::Relaxed);
        self.idle_notify.notify_waiters();
        debug!("Voice gate: opened (TTS may play)");
    }
}

/// Load vocabulary terms from `.whisper/vocabulary.txt` (one term per line).
/// Returns a comma-separated string suitable as Whisper initial prompt.
fn load_vocabulary() -> String {
    let path = PathBuf::from(".whisper/vocabulary.txt");
    match fs::read_to_string(&path) {
        Ok(contents) => {
            let terms: Vec<&str> = contents
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();
            if !terms.is_empty() {
                info!("Loaded {} vocabulary terms from .whisper/vocabulary.txt", terms.len());
            }
            terms.join(", ")
        }
        Err(_) => String::new(),
    }
}

/// Remove trailing "Thank you[.!]?" if the preceding text has more than 10 words.
/// Whisper commonly appends a spoken "Thank you" at the end of real dictations.
fn strip_trailing_thankyou(text: &str) -> &str {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();
    for suffix in &["thank you.", "thank you!", "thank you"] {
        if lower.ends_with(suffix) {
            let preceding = trimmed[..trimmed.len() - suffix.len()].trim();
            if preceding.split_whitespace().count() > 10 {
                debug!("Stripped trailing 'Thank you' from dictation");
                return preceding;
            }
        }
    }
    trimmed
}

/// Load correction mappings from `.whisper/corrections.yaml` (wrong: right).
fn load_corrections() -> HashMap<String, String> {
    let path = PathBuf::from(".whisper/corrections.yaml");
    match fs::read_to_string(&path) {
        Ok(contents) => {
            match serde_yml::from_str::<HashMap<String, String>>(&contents) {
                Ok(map) => {
                    if !map.is_empty() {
                        info!("Loaded {} corrections from .whisper/corrections.yaml", map.len());
                    }
                    map
                }
                Err(e) => {
                    warn!("Failed to parse .whisper/corrections.yaml: {e}");
                    HashMap::new()
                }
            }
        }
        Err(_) => HashMap::new(),
    }
}

pub struct DictationService {
    config: Config,
    state: ServiceState,
    recorder: AudioRecorder,
    transcriber: WhisperTranscriber,
    processor: OllamaProcessor,
    typer: TextTyper,
    output_mode: OutputMode,
    recent_transcriptions: Vec<String>,
    voice_gate: VoiceGate,
    tts_cancel_client: reqwest::Client,
    /// Whisper initial prompt from .whisper/vocabulary.txt
    vocabulary: Arc<RwLock<String>>,
    /// Ollama correction mappings from .whisper/corrections.yaml
    corrections: Arc<RwLock<HashMap<String, String>>>,
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
        let voice_gate = VoiceGate::new();

        // Short-timeout client for fire-and-forget TTS cancel calls
        let tts_cancel_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .expect("Failed to create TTS cancel client");

        let vocabulary = Arc::new(RwLock::new(load_vocabulary()));
        let corrections = Arc::new(RwLock::new(load_corrections()));

        let svc = Self {
            config,
            state: ServiceState::Idle,
            recorder,
            transcriber,
            processor,
            typer,
            output_mode,
            recent_transcriptions: Vec::new(),
            voice_gate,
            tts_cancel_client,
            vocabulary,
            corrections,
        };
        svc.write_mcp_state();
        svc
    }

    /// Get a clone of the voice gate for sharing with TTS components.
    pub fn voice_gate(&self) -> VoiceGate {
        self.voice_gate.clone()
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

    /// Cancel any active TTS playback via HTTP (fire-and-forget).
    /// Works whether TTS is running in-process (Phase 6) or externally.
    fn cancel_tts(&self) {
        let tts_port = self.config.tts.api_port;
        let client = self.tts_cancel_client.clone();
        tokio::spawn(async move {
            let url = format!("http://127.0.0.1:{tts_port}/cancel");
            match client.post(&url).send().await {
                Ok(_) => debug!("TTS cancel sent"),
                Err(_) => {} // TTS not running — that's fine
            }
        });
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
                        Some(HotkeyEvent::Pressed) => self.on_hotkey_press().await,
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

    async fn on_hotkey_press(&mut self) {
        if self.state != ServiceState::Idle {
            return;
        }

        // Cancel any active TTS — user starts speaking, Claude stops talking
        self.cancel_tts();

        // Close voice gate — suppress new TTS until voice input completes
        self.voice_gate.begin_voice_input();

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
            self.transition_to_idle();
            return;
        }

        // Check if audio is too quiet (silence)
        if AudioRecorder::is_silent(&samples, self.config.silence.threshold) {
            info!("Audio is silent, skipping transcription");
            self.transition_to_idle();
            return;
        }

        let audio_duration = samples.len() as f64 / self.recorder.sample_rate() as f64;
        info!("Captured {:.1}s of audio ({} samples)", audio_duration, samples.len());

        // Check if vocabulary/corrections need reloading (MCP tools set flags in state file)
        self.check_whisper_reload();

        // --- Whisper transcription ---
        let t_whisper_start = Instant::now();
        let transcriber = self.transcriber.clone();
        let vocab = self.vocabulary.read().unwrap().clone();
        let vocab_prompt = if vocab.is_empty() { None } else { Some(vocab) };
        let raw_text = match tokio::task::spawn_blocking(move || {
            transcriber.transcribe(&samples, vocab_prompt.as_deref())
        }).await {
            Ok(Ok(result)) => {
                info!(
                    "Transcription ({:.0}ms): \"{}\"",
                    result.latency_ms, result.text
                );
                result.text
            }
            Ok(Err(e)) => {
                warn!("Transcription failed: {e}");
                self.transition_to_idle();
                return;
            }
            Err(e) => {
                warn!("Transcription task panicked: {e}");
                self.transition_to_idle();
                return;
            }
        };
        let t_whisper = t_whisper_start.elapsed().as_secs_f64() * 1000.0;

        if raw_text.is_empty() {
            info!("Empty transcription, returning to IDLE");
            self.transition_to_idle();
            return;
        }

        // Filter common Whisper hallucinations (ported from Python service.py)
        const HALLUCINATIONS: &[&str] = &[
            "thank you",
            "thank you.",
            "thanks.",
            "thanks",
            "thanks for watching",
            "thanks for watching.",
            "subscribe",
            "like and subscribe",
            "you",
            "bye",
            "bye.",
            "goodbye",
            "goodbye.",
        ];

        let normalized = raw_text.trim().to_lowercase();
        if HALLUCINATIONS.contains(&normalized.as_str()) {
            info!("Filtered hallucination: '{raw_text}'");
            self.transition_to_idle();
            return;
        }

        // --- Ollama correction ---
        let t_ollama_start = Instant::now();
        let corrections = self.corrections.read().unwrap().clone();
        let corrections_ref = if corrections.is_empty() { None } else { Some(&corrections) };
        let word_count = raw_text.split_whitespace().count();
        let skip_threshold = self.config.ollama.skip_threshold;
        let (processed_text, ollama_text) = match self.output_mode {
            OutputMode::Whisper => (None, None),
            _ if skip_threshold > 0 && word_count <= skip_threshold => {
                info!("Skipped Ollama ({word_count} words <= {skip_threshold} threshold)");
                (Some(raw_text.clone()), None)
            }
            OutputMode::Ollama | OutputMode::Both => {
                let corrected = self.processor.process(&raw_text, corrections_ref).await;
                info!("Ollama corrected: \"{}\"", corrected);
                (Some(corrected.clone()), Some(corrected))
            }
        };
        let t_ollama = t_ollama_start.elapsed().as_secs_f64() * 1000.0;

        // Strip trailing "Thank you" — common speech artifact when user ends dictation
        let raw_clean = strip_trailing_thankyou(&raw_text);
        let processed_clean = processed_text.as_deref().map(|t| strip_trailing_thankyou(t));

        // Build final output
        let final_text = match self.output_mode {
            OutputMode::Whisper => format!("{raw_clean} "),
            OutputMode::Ollama => format!("{} ", processed_clean.unwrap_or(raw_clean)),
            OutputMode::Both => {
                format!("{} [{raw_clean}] ", processed_clean.unwrap_or(raw_clean))
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

        self.transition_to_idle();
    }

    /// Transition to IDLE and open the voice gate (allow TTS to play again).
    /// All paths back to IDLE must go through this method.
    fn transition_to_idle(&mut self) {
        self.state = ServiceState::Idle;
        self.voice_gate.end_voice_input();
        info!("State: → IDLE");
    }

    /// Check state file for vocabulary/corrections reload flags (set by MCP tools).
    fn check_whisper_reload(&self) {
        let state = {
            let path = state_file();
            match fs::read_to_string(&path) {
                Ok(contents) => serde_json::from_str::<serde_json::Value>(&contents).unwrap_or_default(),
                Err(_) => return,
            }
        };

        if state.get("vocabulary_updated").and_then(|v| v.as_bool()) == Some(true) {
            let new_vocab = load_vocabulary();
            *self.vocabulary.write().unwrap() = new_vocab;
            // Clear the flag
            let path = state_file();
            if let Ok(contents) = fs::read_to_string(&path) {
                if let Ok(mut s) = serde_json::from_str::<serde_json::Value>(&contents) {
                    s.as_object_mut().map(|o| o.remove("vocabulary_updated"));
                    let _ = fs::write(&path, serde_json::to_string_pretty(&s).unwrap());
                }
            }
        }

        if state.get("corrections_updated").and_then(|v| v.as_bool()) == Some(true) {
            let new_corrections = load_corrections();
            *self.corrections.write().unwrap() = new_corrections;
            // Clear the flag
            let path = state_file();
            if let Ok(contents) = fs::read_to_string(&path) {
                if let Ok(mut s) = serde_json::from_str::<serde_json::Value>(&contents) {
                    s.as_object_mut().map(|o| o.remove("corrections_updated"));
                    let _ = fs::write(&path, serde_json::to_string_pretty(&s).unwrap());
                }
            }
        }
    }
}
