//! Whisper ASR transcription using whisper-rs (whisper.cpp bindings).
//!
//! Loads a GGML model once at startup, then transcribes f32 audio
//! samples (16kHz mono) to text on demand.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::config::WhisperConfig;

/// Thread-safe wrapper around WhisperContext.
/// WhisperContext is Send+Sync, so we wrap it in Arc for sharing.
#[derive(Clone)]
pub struct WhisperTranscriber {
    ctx: Arc<WhisperContext>,
    model_path: PathBuf,
}

/// Result of a transcription with timing info.
pub struct TranscribeResult {
    pub text: String,
    pub latency_ms: f64,
}

/// One timestamped segment, as emitted by `transcribe_segments`.
#[derive(Debug, Clone)]
pub struct TranscribeSegment {
    pub text: String,
    pub start_s: f64,
    pub end_s: f64,
}

impl WhisperTranscriber {
    /// Load the Whisper GGML model.
    pub fn load(config: &WhisperConfig) -> Result<Self, String> {
        let model_path = Self::find_model(&config.model)?;

        info!("Loading Whisper model from {}", model_path.display());
        let t0 = Instant::now();

        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(model_path.to_str().unwrap(), params)
            .map_err(|e| format!("Failed to load Whisper model: {e}"))?;

        let load_ms = t0.elapsed().as_millis();
        info!("Whisper model loaded in {load_ms}ms");

        Ok(Self {
            ctx: Arc::new(ctx),
            model_path,
        })
    }

    /// Pre-warm CUDA state by running a dummy inference.
    ///
    /// `WhisperContext::new_with_params` only loads model weights (fast).
    /// The heavy CUDA buffer allocation (`whisper_init_state`) is deferred
    /// to the first `create_state()` call. Without pre-warming, this 10+
    /// minute GPU init storm happens on the first hotkey press, which can
    /// starve the evdev hotkey monitor and make it unresponsive.
    pub fn warm_up(&self) -> Result<(), String> {
        info!("Pre-warming Whisper CUDA state (this may take several minutes)...");
        let t0 = Instant::now();

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("CUDA warm-up create_state failed: {e}"))?;

        // Run a tiny inference to fully initialize all compute buffers
        let dummy_audio = vec![0.0f32; 16000]; // 1 second of silence
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_single_segment(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, &dummy_audio)
            .map_err(|e| format!("CUDA warm-up inference failed: {e}"))?;

        let elapsed = t0.elapsed();
        info!(
            "Whisper CUDA state pre-warmed in {:.1}s",
            elapsed.as_secs_f64()
        );
        Ok(())
    }

    /// Transcribe audio samples (f32, 16kHz, mono) to text.
    /// If `initial_prompt` is provided, it biases Whisper toward recognizing those terms.
    pub fn transcribe(
        &self,
        samples: &[f32],
        initial_prompt: Option<&str>,
    ) -> Result<TranscribeResult, String> {
        let t0 = Instant::now();

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("Failed to create whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(true);
        params.set_token_timestamps(false);

        if let Some(prompt) = initial_prompt {
            params.set_initial_prompt(prompt);
        }

        state
            .full(params, samples)
            .map_err(|e| format!("Whisper inference failed: {e}"))?;

        // Collect all segments into a single string
        let n_segments = state.full_n_segments();

        let mut text = String::new();
        for i in 0..n_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str_lossy() {
                    let trimmed = segment_text.trim();
                    if !trimmed.is_empty() {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(trimmed);
                    }
                }
            }
        }

        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let audio_duration = samples.len() as f64 / 16000.0;
        let rtf = if audio_duration > 0.0 {
            latency_ms / (audio_duration * 1000.0)
        } else {
            0.0
        };

        info!(
            "Transcribed {:.1}s audio in {:.0}ms (RTF: {:.2}x): \"{}\"",
            audio_duration,
            latency_ms,
            rtf,
            truncate_preview(&text, 80)
        );

        Ok(TranscribeResult { text, latency_ms })
    }

    /// Long-form transcription with native segment timestamps. Used by the
    /// `/transcribe` HTTP endpoint that the voice-journal GUI calls.
    pub fn transcribe_segments(&self, samples: &[f32]) -> Result<Vec<TranscribeSegment>, String> {
        let t0 = Instant::now();
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("Failed to create whisper state: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);
        params.set_token_timestamps(false);

        state
            .full(params, samples)
            .map_err(|e| format!("Whisper inference failed: {e}"))?;

        let n_segments = state.full_n_segments();
        let mut segments = Vec::with_capacity(n_segments as usize);
        for i in 0..n_segments {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            let Ok(text) = seg.to_str_lossy() else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.is_empty() {
                continue;
            }
            let start_s = seg.start_timestamp() as f64 / 100.0;
            let end_s = seg.end_timestamp() as f64 / 100.0;
            segments.push(TranscribeSegment {
                text: trimmed.to_string(),
                start_s,
                end_s,
            });
        }

        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let audio_duration = samples.len() as f64 / 16000.0;
        info!(
            "Segmented transcription: {:.1}s audio → {} segments in {:.0}ms",
            audio_duration,
            segments.len(),
            latency_ms,
        );
        Ok(segments)
    }

    #[allow(dead_code)]
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Find the GGML model file.
    fn find_model(model_name: &str) -> Result<PathBuf, String> {
        // Check if it's a direct path to an existing file
        let direct = PathBuf::from(model_name);
        if direct.exists() && direct.extension().is_some() {
            return Ok(direct);
        }

        // Common GGML model filenames to search for
        let filenames = [
            format!("ggml-{}.bin", model_name.replace('/', "-")),
            "ggml-distil-large-v3.bin".to_string(),
            "ggml-large-v3-turbo.bin".to_string(),
            "ggml-large-v3.bin".to_string(),
            "ggml-base.bin".to_string(),
        ];

        // Search locations
        let search_dirs: Vec<PathBuf> = [
            std::env::current_dir().ok(),
            std::env::current_dir().ok().map(|d| d.join("models")),
            dirs::home_dir().map(|h| h.join(".cache/whisper")),
            dirs::home_dir().map(|h| h.join("whisper-typer")),
            dirs::home_dir().map(|h| h.join("whisper-typer/models")),
        ]
        .into_iter()
        .flatten()
        .collect();

        for dir in &search_dirs {
            for filename in &filenames {
                let path = dir.join(filename);
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        Err(format!(
            "Whisper GGML model not found. Download with:\n  \
             wget https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin\n\
             Searched in: {:?}",
            search_dirs
        ))
    }
}

fn truncate_preview(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
