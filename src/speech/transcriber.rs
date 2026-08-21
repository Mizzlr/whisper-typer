//! Whisper ASR transcription using whisper-rs (whisper.cpp bindings).
//!
//! Two-plane architecture (2026-05-03):
//! - **Dictation plane** (`transcribe`): used by service.rs for hotkey-driven
//!   dictation. Owns its own permanent `WhisperState` allocated once at
//!   startup, reused across calls. Behind a Mutex but in practice
//!   uncontended (one caller).
//! - **API plane** (`transcribe_segments`): used by the `/transcribe` HTTP
//!   handler that voice-journal hits. Owns a separate permanent
//!   `WhisperState`. Multiple concurrent HTTP requests serialize through
//!   the plane's Mutex.
//!
//! Two independent states + the shared `WhisperContext` (model weights)
//! means voice-journal's batch transcribe calls never block whisper-typer's
//! low-latency hotkey path. GPU memory becomes deterministic at startup
//! (~1.5 GB model + 2 × ~0.37 GB state = ~2.24 GB) instead of churning
//! ~370 MB allocations per call as the previous design did.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::info;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::config::WhisperConfig;

#[derive(Clone)]
pub struct WhisperTranscriber {
    dictation_state: Arc<Mutex<WhisperState>>,
    api_state: Arc<Mutex<WhisperState>>,
}

pub struct TranscribeResult {
    pub text: String,
    pub latency_ms: f64,
}

#[derive(Debug, Clone)]
pub struct TranscribeSegment {
    pub text: String,
    pub start_s: f64,
    pub end_s: f64,
}

impl WhisperTranscriber {
    pub fn load(config: &WhisperConfig) -> Result<Self, String> {
        let model_path = Self::find_model(&config.model)?;

        info!("Loading Whisper model from {}", model_path.display());
        let t0 = Instant::now();

        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(model_path.to_str().unwrap(), params)
            .map_err(|e| format!("Failed to load Whisper model: {e}"))?;

        info!("Whisper model loaded in {}ms", t0.elapsed().as_millis());

        // Allocate the dictation plane's state. Whisper-typer's hotkey path
        // owns this one — it must never wait on an API caller.
        info!("Allocating dictation plane WhisperState...");
        let warm_t0 = Instant::now();
        let mut dictation_state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create dictation state: {e}"))?;
        Self::prewarm_state(&mut dictation_state, true)?;
        info!(
            "Dictation plane ready in {:.1}s",
            warm_t0.elapsed().as_secs_f64()
        );

        // Allocate the API plane's state. Voice-journal and any other caller
        // of /transcribe share this one; concurrent calls serialize via the
        // plane's Mutex.
        info!("Allocating API plane WhisperState...");
        let warm_t0 = Instant::now();
        let mut api_state = ctx
            .create_state()
            .map_err(|e| format!("Failed to create API state: {e}"))?;
        Self::prewarm_state(&mut api_state, false)?;
        info!(
            "API plane ready in {:.1}s",
            warm_t0.elapsed().as_secs_f64()
        );

        // Drop our owning reference — both states already hold internal
        // pointers into the context's CUDA buffers, and the WhisperContext
        // outlives the states by way of the C++ wrapper's reference counting.
        // (whisper-rs 0.15.1's WhisperState has no lifetime parameter.)
        drop(ctx);

        Ok(Self {
            dictation_state: Arc::new(Mutex::new(dictation_state)),
            api_state: Arc::new(Mutex::new(api_state)),
        })
    }

    fn prewarm_state(state: &mut WhisperState, single_segment: bool) -> Result<(), String> {
        let dummy = vec![0.0f32; 16000];
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("en"));
        params.set_single_segment(single_segment);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        state
            .full(params, &dummy)
            .map(|_| ())
            .map_err(|e| format!("Pre-warm inference failed: {e}"))
    }

    /// Backwards-compatible no-op. Pre-warming now happens inside `load()`.
    pub fn warm_up(&self) -> Result<(), String> {
        Ok(())
    }

    /// Transcribe audio samples (f32, 16kHz, mono) on the dictation plane.
    /// Used by the hotkey-driven path; never contends with API callers.
    pub fn transcribe(
        &self,
        samples: &[f32],
        initial_prompt: Option<&str>,
    ) -> Result<TranscribeResult, String> {
        let mut state = self
            .dictation_state
            .lock()
            .map_err(|_| "dictation state lock poisoned".to_string())?;
        let t0 = Instant::now();

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

        let n_segments = state.full_n_segments();
        let mut text = String::new();
        for i in 0..n_segments {
            let Some(segment) = state.get_segment(i) else {
                continue;
            };
            let Ok(segment_text) = segment.to_str_lossy() else {
                continue;
            };
            let trimmed = segment_text.trim();
            if trimmed.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(trimmed);
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

    /// Long-form transcription with native segment timestamps on the API
    /// plane. Used by the `/transcribe` HTTP endpoint that voice-journal
    /// calls. Concurrent HTTP requests serialize via the plane's Mutex —
    /// they never block the dictation plane.
    pub fn transcribe_segments(&self, samples: &[f32]) -> Result<Vec<TranscribeSegment>, String> {
        let mut state = self
            .api_state
            .lock()
            .map_err(|_| "API state lock poisoned".to_string())?;
        let t0 = Instant::now();

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

    /// Find the GGML model file.
    fn find_model(model_name: &str) -> Result<PathBuf, String> {
        let direct = PathBuf::from(model_name);
        if direct.exists() && direct.extension().is_some() {
            return Ok(direct);
        }

        let filenames = [
            format!("ggml-{}.bin", model_name.replace('/', "-")),
            "ggml-distil-large-v3.bin".to_string(),
            "ggml-large-v3-turbo.bin".to_string(),
            "ggml-large-v3.bin".to_string(),
            "ggml-base.bin".to_string(),
        ];

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
        let end = s.floor_char_boundary(max);
        format!("{}...", &s[..end])
    }
}
