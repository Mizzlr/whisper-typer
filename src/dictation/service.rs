//! Main service orchestration with state machine.
//!
//! IDLE → RECORDING → PROCESSING → IDLE
//!
//! Voice gate: TTS is suppressed during recording/processing.
//! When the user presses the hotkey, any active TTS is cancelled immediately.
//! TTS waits for `voice_idle` before playing so it never talks over the user.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use chrono::Local;
use regex::{Captures, Regex};
use tokio::sync::{mpsc, Notify};
use tracing::{debug, info, warn};

use crate::config::Config;
use crate::history::{self, TranscriptionRecord};
use crate::hotkey::{HotkeyEvent, HotkeyMonitor};
use crate::processor::{is_pathological_stutter, CorrectionMetadata, OllamaProcessor};
use crate::punctuation::PunctuationClient;
use crate::recorder::AudioRecorder;
use crate::remote_asr::RemoteAsrClient;
use crate::runtime_settings::{OutputMode, RuntimeSettings};
use crate::transcriber::WhisperTranscriber;
use crate::typer::TextTyper;

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

impl Default for VoiceGate {
    fn default() -> Self {
        Self::new()
    }
}

/// Remove trailing Whisper hallucination phrases if the preceding text has more than 10 words.
/// Whisper commonly appends stray phrases like "Thank you", "I'm gonna", "I'm sorry" at the
/// end of real dictations.
fn strip_trailing_hallucination(text: &str) -> &str {
    const TRAILING_SUFFIXES: &[&str] = &[
        "thank you.",
        "thank you!",
        "thank you",
        "i'm gonna.",
        "i'm gonna",
        "i'm going to.",
        "i'm going to",
        "i'm sorry.",
        "i'm sorry",
    ];
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();
    for suffix in TRAILING_SUFFIXES {
        if lower.ends_with(suffix) {
            let preceding = trimmed[..trimmed.len() - suffix.len()].trim();
            if preceding.split_whitespace().count() > 10 {
                debug!(
                    "Stripped trailing '{}' from dictation",
                    &trimmed[trimmed.len() - suffix.len()..]
                );
                return preceding;
            }
        }
    }
    trimmed
}

#[derive(Clone, Default)]
struct VoiceCorrections {
    replacements: Vec<VoiceReplacement>,
    protectors: Vec<Regex>,
}

#[derive(Clone)]
struct VoiceReplacement {
    pattern: Regex,
    replacement: String,
}

impl VoiceCorrections {
    fn load(config: &Config) -> Self {
        if !config.corrections.enabled {
            return Self::default();
        }

        let path = expand_home_path(&config.corrections.path);
        let Ok(contents) = fs::read_to_string(&path) else {
            debug!(
                "Voice typing corrections file not loaded: {}",
                path.display()
            );
            return Self::default();
        };

        let mut corrections = Self::default();
        for (idx, raw_line) in contents.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts = line.splitn(3, '\t').collect::<Vec<_>>();
            match parts.as_slice() {
                ["protect", pattern] => match Regex::new(pattern) {
                    Ok(regex) => corrections.protectors.push(regex),
                    Err(e) => warn!(
                        "Invalid voice correction protector at {}:{}: {e}",
                        path.display(),
                        idx + 1
                    ),
                },
                ["replace", pattern, replacement] => {
                    match Regex::new(pattern) {
                        Ok(regex) => corrections.replacements.push(VoiceReplacement {
                            pattern: regex,
                            replacement: (*replacement).to_string(),
                        }),
                        Err(e) => warn!(
                            "Invalid voice correction replacement at {}:{}: {e}",
                            path.display(),
                            idx + 1
                        ),
                    }
                }
                [pattern, replacement] => match Regex::new(pattern) {
                    Ok(regex) => corrections.replacements.push(VoiceReplacement {
                        pattern: regex,
                        replacement: (*replacement).to_string(),
                    }),
                    Err(e) => warn!(
                        "Invalid voice correction replacement at {}:{}: {e}",
                        path.display(),
                        idx + 1
                    ),
                },
                _ => warn!(
                    "Invalid voice correction row at {}:{}: expected 'replace<TAB>regex<TAB>replacement' or 'protect<TAB>regex'",
                    path.display(),
                    idx + 1
                ),
            }
        }

        info!(
            "Loaded {} voice typing correction(s), {} protection rule(s)",
            corrections.replacements.len(),
            corrections.protectors.len()
        );
        corrections
    }

    fn apply(&self, text: &str) -> String {
        let mut corrected = text.to_string();
        for replacement in &self.replacements {
            let protected_spans = self.protected_spans(&corrected);
            corrected = replacement
                .pattern
                .replace_all(&corrected, |caps: &Captures<'_>| {
                    let m = caps.get(0).expect("whole match");
                    if protected_spans
                        .iter()
                        .any(|(start, end)| spans_overlap(m.start(), m.end(), *start, *end))
                    {
                        m.as_str().to_string()
                    } else {
                        replacement.replacement.clone()
                    }
                })
                .into_owned();
        }
        corrected
    }

    fn protected_spans(&self, text: &str) -> Vec<(usize, usize)> {
        self.protectors
            .iter()
            .flat_map(|regex| regex.find_iter(text).map(|m| (m.start(), m.end())))
            .collect()
    }
}

fn expand_home_path(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

fn spans_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

pub struct DictationService {
    config: Config,
    state: ServiceState,
    recorder: AudioRecorder,
    transcriber: WhisperTranscriber,
    remote_asr: Option<RemoteAsrClient>,
    punctuation: Option<PunctuationClient>,
    processor: OllamaProcessor,
    typer: TextTyper,
    runtime_settings: Arc<RuntimeSettings>,
    voice_gate: VoiceGate,
    tts_cancel_client: reqwest::Client,
    voice_corrections: VoiceCorrections,
}

impl DictationService {
    pub fn new(
        config: Config,
        transcriber: WhisperTranscriber,
        runtime_settings: Arc<RuntimeSettings>,
    ) -> Self {
        let recorder = AudioRecorder::new(
            config.audio.clone(),
            config.recording.clone(),
            config.silence.clone(),
        );
        let processor = OllamaProcessor::new(config.ollama.clone());
        let typer = TextTyper::new(&config.typer);
        let voice_gate = VoiceGate::new();
        let voice_corrections = VoiceCorrections::load(&config);
        let remote_asr = if config.remote_asr.enabled {
            match RemoteAsrClient::new(config.remote_asr.clone()) {
                Ok(client) => Some(client),
                Err(error) => {
                    warn!("Remote ASR disabled: {error}");
                    None
                }
            }
        } else {
            None
        };
        let punctuation = if config.punctuation.enabled {
            match PunctuationClient::new(config.punctuation.clone()) {
                Ok(client) => Some(client),
                Err(error) => {
                    warn!("Punctuation processing disabled: {error}");
                    None
                }
            }
        } else {
            None
        };

        // Short-timeout client for fire-and-forget TTS cancel calls
        let tts_cancel_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
            .expect("Failed to create TTS cancel client");

        Self {
            config,
            state: ServiceState::Idle,
            recorder,
            transcriber,
            remote_asr,
            punctuation,
            processor,
            typer,
            runtime_settings,
            voice_gate,
            tts_cancel_client,
            voice_corrections,
        }
    }

    /// Get a clone of the voice gate for sharing with TTS components.
    pub fn voice_gate(&self) -> VoiceGate {
        self.voice_gate.clone()
    }

    /// Add a transcription to the recent list and update state file.
    fn add_transcription(&self, text: &str) {
        self.runtime_settings.add_transcription(text);
    }

    /// Cancel any active TTS playback via HTTP (fire-and-forget).
    /// Works whether TTS is running in-process (Phase 6) or externally.
    fn cancel_tts(&self) {
        let tts_port = self.config.tts.api_port;
        let client = self.tts_cancel_client.clone();
        tokio::spawn(async move {
            let url = format!("http://127.0.0.1:{tts_port}/cancel");
            if client.post(&url).send().await.is_ok() {
                debug!("TTS cancel sent");
            }
        });
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Open audio stream; retry because PipeWire/WirePlumber may not finish
        // routing before this service starts, even with After=pipewire.service.
        for attempt in 1u32..=12 {
            match self.recorder.open_stream() {
                Ok(()) => break,
                Err(e) if attempt < 12 => {
                    warn!("Audio device not ready (attempt {attempt}/12): {e}. Retrying in 5s...");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
                Err(e) => return Err(e.into()),
            }
        }

        // Create hotkey channel
        let (hotkey_tx, mut hotkey_rx) = mpsc::channel::<HotkeyEvent>(16);

        // Start hotkey monitor in background
        let hotkey_monitor = HotkeyMonitor::new(&self.config.hotkey, hotkey_tx);
        tokio::spawn(async move {
            hotkey_monitor.run().await;
        });

        info!(
            "Service ready — press hotkey to start recording (mode: {:?})",
            self.runtime_settings.snapshot().output_mode
        );

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

        let recorded_at = Local::now();
        let sample_rate = self.recorder.sample_rate();
        let sample_count = samples.len();
        let audio_duration = sample_count as f64 / sample_rate as f64;
        info!(
            "Captured {:.1}s of audio ({} samples)",
            audio_duration,
            samples.len()
        );

        let runtime = self.runtime_settings.snapshot();
        let output_mode = runtime.output_mode;

        // Persist the exact buffer before ASR/correction/typing so benchmark
        // inputs remain faithful even if later processing fails.
        let audio_artifact = if self.config.recording.save_audio {
            match history::save_audio(
                &samples,
                sample_rate,
                &recorded_at,
                self.config.recording.audio_retention_days,
            ) {
                Ok(artifact) => {
                    info!("Saved dictation audio to {}", artifact.path.display());
                    Some(artifact)
                }
                Err(e) => {
                    warn!("Failed to save dictation audio; continuing transcription: {e}");
                    None
                }
            }
        } else {
            None
        };

        let (raw_text, t_whisper, t_punctuation, t_ollama, processed_text, ollama_text);
        let mut correction_metadata: Option<CorrectionMetadata> = None;
        // The deprecated Ollama audio mode intentionally follows this reliable
        // Whisper path. No utterance may be dropped merely because an
        // experimental multimodal request failed.
        {
            let t_whisper_start = Instant::now();
            let remote_result = match self.remote_asr.clone() {
                Some(remote) => match remote.transcribe(&samples, sample_rate).await {
                    Ok(result) => {
                        info!("Remote ASR succeeded in {:.0}ms", result.latency_ms);
                        Some(Ok(result))
                    }
                    Err(error) if remote.fallback_local() => {
                        warn!("Remote ASR failed; using local Whisper fallback: {error}");
                        None
                    }
                    Err(error) => Some(Err(error)),
                },
                None => None,
            };
            let transcription = match remote_result {
                Some(result) => result,
                None => {
                    let transcriber = self.transcriber.clone();
                    match tokio::task::spawn_blocking(move || {
                        transcriber.transcribe(&samples, None)
                    })
                    .await
                    {
                        Ok(Ok(result)) => Ok(result),
                        Ok(Err(error)) => Err(format!("local Whisper failed: {error}")),
                        Err(error) => Err(format!("local Whisper task panicked: {error}")),
                    }
                }
            };
            raw_text = match transcription {
                Ok(result) => {
                    info!(
                        "Transcription ({:.0}ms): \"{}\"",
                        result.latency_ms, result.text
                    );
                    result.text
                }
                Err(error) => {
                    warn!("Transcription failed: {error}");
                    self.transition_to_idle();
                    return;
                }
            };
            t_whisper = t_whisper_start.elapsed().as_secs_f64() * 1000.0;

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
                "i'm gonna",
                "i'm gonna.",
                "i'm sorry",
                "i'm sorry.",
                "i'm going to",
                "i'm going to.",
                "the president",
                "the president.",
            ];

            let normalized = raw_text.trim().to_lowercase();
            if HALLUCINATIONS.contains(&normalized.as_str()) {
                info!("Filtered hallucination: '{raw_text}'");
                self.transition_to_idle();
                return;
            }

            // Restore punctuation and truecasing after ASR. Failure is
            // deliberately fail-open: typing the raw transcript is better
            // than dropping or delaying the user's dictation.
            let t_punctuation_start = Instant::now();
            let punctuated_text = match self.punctuation.clone() {
                Some(client) => match client.process(&raw_text).await {
                    Ok(result) => {
                        info!(
                            "Punctuation succeeded in {:.0}ms: \"{}\"",
                            result.latency_ms, result.text
                        );
                        result.text
                    }
                    Err(error) => {
                        warn!("Punctuation failed; using raw ASR text: {error}");
                        raw_text.clone()
                    }
                },
                None => raw_text.clone(),
            };
            t_punctuation = t_punctuation_start.elapsed().as_secs_f64() * 1000.0;

            // --- Ollama correction ---
            let t_ollama_start = Instant::now();
            let word_count = punctuated_text.split_whitespace().count();
            let skip_threshold = self.config.ollama.skip_threshold;
            let (pt, ot) = match output_mode {
                OutputMode::Whisper => (Some(punctuated_text.clone()), None),
                _ if !runtime.ollama_enabled => (Some(punctuated_text.clone()), None),
                _ if skip_threshold > 0 && word_count <= skip_threshold => {
                    info!("Skipped Ollama ({word_count} words <= {skip_threshold} threshold)");
                    (Some(punctuated_text.clone()), None)
                }
                OutputMode::Ollama | OutputMode::Both => {
                    let correction = self.processor.process(&punctuated_text).await;
                    info!(
                        "Ollama correction accepted={}: \"{}\"",
                        correction.metadata.accepted, correction.text
                    );
                    let corrected = correction.text;
                    correction_metadata = Some(correction.metadata);
                    (Some(corrected.clone()), Some(corrected))
                }
            };
            t_ollama = t_ollama_start.elapsed().as_secs_f64() * 1000.0;
            processed_text = pt;
            ollama_text = ot;
        }

        // Strip trailing hallucination phrases — common speech artifacts when user ends dictation
        let raw_clean = self
            .voice_corrections
            .apply(strip_trailing_hallucination(&raw_text));
        let processed_clean = processed_text
            .as_deref()
            .map(strip_trailing_hallucination)
            .map(|text| self.voice_corrections.apply(text));

        let selected_text = processed_clean.as_deref().unwrap_or(&raw_clean);
        if is_pathological_stutter(selected_text) {
            warn!("Dropping dictation with repeated stutter text: '{selected_text}'");
            self.transition_to_idle();
            return;
        }

        // Build final output
        let final_text = match output_mode {
            OutputMode::Whisper => format!("{selected_text} "),
            OutputMode::Ollama => {
                format!("{selected_text} ")
            }
            OutputMode::Both => {
                format!("{} [{raw_clean}] ", selected_text)
            }
        };

        // --- Type into active window ---
        let t_type_start = Instant::now();
        self.typer.type_text(&final_text);
        let t_type = t_type_start.elapsed().as_secs_f64() * 1000.0;

        let t_total = t_start.elapsed().as_secs_f64() * 1000.0;

        info!(
            "  ASR: {:.0}ms | Punctuation: {:.0}ms | Ollama: {:.0}ms | Typing: {:.0}ms | Total: {:.0}ms | Audio: {:.1}s | Speed: {:.1}x",
            t_whisper, t_punctuation, t_ollama, t_type, t_total, audio_duration,
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
            timestamp: recorded_at.format("%Y-%m-%dT%H:%M:%S%.6f").to_string(),
            whisper_text: raw_text,
            ollama_text,
            final_text: final_text.clone(),
            output_mode: output_mode.as_str().to_string(),
            whisper_latency_ms: t_whisper as i64,
            ollama_latency_ms: correction_metadata.as_ref().map(|_| t_ollama as i64),
            correction_accepted: correction_metadata
                .as_ref()
                .map(|metadata| metadata.accepted),
            correction_fallback_reason: correction_metadata
                .as_ref()
                .and_then(|metadata| metadata.fallback_reason.clone()),
            ollama_load_ms: correction_metadata
                .as_ref()
                .and_then(|metadata| metadata.load_ms),
            ollama_prompt_eval_ms: correction_metadata
                .as_ref()
                .and_then(|metadata| metadata.prompt_eval_ms),
            ollama_eval_ms: correction_metadata
                .as_ref()
                .and_then(|metadata| metadata.eval_ms),
            typing_latency_ms: t_type as i64,
            total_latency_ms: t_total as i64,
            audio_duration_s: (audio_duration * 100.0).round() / 100.0,
            char_count: final_text.len(),
            word_count: final_text.split_whitespace().count(),
            speed_ratio,
            audio_path: audio_artifact
                .as_ref()
                .map(|artifact| artifact.path.to_string_lossy().into_owned()),
            audio_sample_rate_hz: audio_artifact.as_ref().map(|_| sample_rate),
            audio_samples: audio_artifact.as_ref().map(|_| sample_count),
            audio_bytes: audio_artifact.as_ref().map(|artifact| artifact.bytes),
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
}

#[cfg(test)]
mod tests {
    use super::{VoiceCorrections, VoiceReplacement};
    use regex::Regex;

    #[test]
    fn applies_replacements() {
        let corrections = VoiceCorrections {
            replacements: vec![VoiceReplacement {
                pattern: Regex::new(r"(?i)\btea\s+two\b").unwrap(),
                replacement: "T2".into(),
            }],
            protectors: vec![],
        };

        assert_eq!(
            corrections.apply("Compare tea two against baseline."),
            "Compare T2 against baseline."
        );
    }

    #[test]
    fn skips_replacements_inside_protected_spans() {
        let corrections = VoiceCorrections {
            replacements: vec![VoiceReplacement {
                pattern: Regex::new(r"(?i)\bthing\b").unwrap(),
                replacement: "term".into(),
            }],
            protectors: vec![Regex::new(r"(?i)\bliteral thing\b").unwrap()],
        };

        assert_eq!(
            corrections.apply("Fix thing but keep literal thing."),
            "Fix term but keep literal thing."
        );
    }

    #[test]
    fn expands_home_paths() {
        let expanded = super::expand_home_path("~/.config/whisper-typer/corrections.tsv");
        assert!(expanded.is_absolute());
    }
}
