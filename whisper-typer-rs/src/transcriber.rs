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

    /// Transcribe audio samples (f32, 16kHz, mono) to text.
    pub fn transcribe(&self, samples: &[f32]) -> Result<TranscribeResult, String> {
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
            dirs::home_dir().map(|h| h.join(".cache/whisper")),
            dirs::home_dir().map(|h| h.join("whisper-typer")),
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
