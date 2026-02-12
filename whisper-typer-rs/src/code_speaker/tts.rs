//! Kokoro TTS engine: text → phonemes → ONNX inference → audio playback.
//!
//! Pipeline:
//! 1. Text → sentences (split on .!?)
//! 2. Sentence → phonemes (misaki-rs G2P)
//! 3. Phonemes → token IDs (tokenizer.json vocabulary)
//! 4. Token IDs + voice style + speed → ONNX inference → f32 audio (24kHz)
//! 5. Audio → rodio Sink playback with cancellation

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ndarray::{Array2, Array3};
use ndarray_npy::NpzReader;
use regex::Regex;
use ort::value::Tensor;
use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, OutputStreamBuilder, Sink};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

use crate::config::TTSConfig;

const SAMPLE_RATE: u32 = 24000;
const MAX_TOKENS: usize = 510; // Voice style array first dimension

/// Result of a speak operation with timing breakdown.
pub struct SpeakResult {
    pub generate_ms: f64,
    pub playback_ms: f64,
    pub cancelled: bool,
    pub text_spoken: String,
}

/// Loaded voice style data: shape (510, 1, 256) f32.
struct VoiceData {
    /// Style vectors indexed by token count. shape: (510, 256)
    styles: Array2<f32>,
}

/// Native Kokoro TTS engine.
pub struct KokoroTtsEngine {
    // ONNX model (Mutex because ort 2.0 Session::run needs &mut)
    session: Mutex<Option<ort::session::Session>>,

    // Phonemizer (misaki-rs G2P)
    phonemizer: Option<misaki_rs::G2P>,

    // Tokenizer vocabulary: char → token ID
    vocab: HashMap<char, i64>,

    // Voices: name → style data
    voices: HashMap<String, VoiceData>,

    // Current voice and speed
    voice: Mutex<String>,
    speed: f32,

    // Audio output (kept alive for process lifetime)
    // In rodio 0.21, OutputStream is the handle — no separate OutputStreamHandle
    output_stream: Option<OutputStream>,

    // State
    cancel_flag: Arc<AtomicBool>,
    speaking: Arc<AtomicBool>,
    speak_lock: AsyncMutex<()>,
    active_sink: Arc<Mutex<Option<Sink>>>,

    // Paths
    model_path: PathBuf,
    voices_path: PathBuf,
    tokenizer_path: PathBuf,
}

impl KokoroTtsEngine {
    pub fn new(config: &TTSConfig) -> Self {
        let base_dir = std::env::current_dir().unwrap_or_default();

        let model_path = if config.model_path.is_empty() {
            base_dir.join("kokoro-v1.0.onnx")
        } else {
            PathBuf::from(&config.model_path)
        };

        let voices_path = base_dir.join("voices-v1.0.bin");
        let tokenizer_path = base_dir.join("tokenizer.json");

        Self {
            session: Mutex::new(None),
            phonemizer: None,
            vocab: HashMap::new(),
            voices: HashMap::new(),
            voice: Mutex::new(config.voice.clone()),
            speed: config.speed,
            output_stream: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            speaking: Arc::new(AtomicBool::new(false)),
            speak_lock: AsyncMutex::new(()),
            active_sink: Arc::new(Mutex::new(None)),
            model_path,
            voices_path,
            tokenizer_path,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.session.lock().unwrap().is_some()
    }

    pub fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::Relaxed)
    }

    pub fn current_voice(&self) -> String {
        self.voice.lock().unwrap().clone()
    }

    pub fn set_voice(&self, voice: &str) -> bool {
        if self.voices.contains_key(voice) {
            *self.voice.lock().unwrap() = voice.to_string();
            info!("Voice changed to: {voice}");
            true
        } else {
            warn!("Unknown voice: {voice}");
            false
        }
    }

    pub fn list_voices(&self) -> Vec<String> {
        let mut names: Vec<String> = self.voices.keys().cloned().collect();
        names.sort();
        names
    }

    /// Load the ONNX model, tokenizer, voices, and phonemizer.
    /// This is blocking and should be called in spawn_blocking.
    pub fn load_model_sync(&mut self) -> Result<(), String> {
        let t0 = Instant::now();

        // 1. Load tokenizer vocabulary
        info!("Loading tokenizer from {}", self.tokenizer_path.display());
        self.vocab = load_tokenizer(&self.tokenizer_path)?;
        info!("Tokenizer loaded: {} tokens", self.vocab.len());

        // 2. Load voice styles from NPZ
        info!("Loading voices from {}", self.voices_path.display());
        self.voices = load_voices(&self.voices_path)?;
        info!("Loaded {} voices", self.voices.len());

        // 3. Load ONNX model
        info!("Loading ONNX model from {}", self.model_path.display());
        let session = ort::session::Session::builder()
            .map_err(|e| format!("Failed to create ONNX session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| format!("Failed to set thread count: {e}"))?
            .commit_from_file(&self.model_path)
            .map_err(|e| format!("Failed to load ONNX model: {e}"))?;
        *self.session.lock().unwrap() = Some(session);

        // 4. Initialize phonemizer (misaki-rs G2P)
        info!("Initializing misaki-rs phonemizer...");
        let phonemizer = misaki_rs::G2P::new(misaki_rs::Language::EnglishUS);
        self.phonemizer = Some(phonemizer);

        // 5. Initialize audio output (rodio 0.21)
        let stream = OutputStreamBuilder::open_default_stream()
            .map_err(|e| format!("Failed to open audio output: {e}"))?;
        self.output_stream = Some(stream);

        let load_ms = t0.elapsed().as_millis();
        info!("Kokoro TTS loaded in {load_ms}ms");

        Ok(())
    }

    /// Load model asynchronously.
    pub async fn load_model(&mut self) -> Result<(), String> {
        // load_model_sync needs &mut self which can't cross spawn_blocking boundary easily,
        // so we call it directly (it blocks the async task briefly, which is acceptable at startup)
        self.load_model_sync()
    }

    /// Speak text aloud with sentence-level streaming and cancellation.
    pub async fn speak(&self, text: &str) -> SpeakResult {
        let _guard = self.speak_lock.lock().await;
        self.cancel_flag.store(false, Ordering::Relaxed);
        self.speaking.store(true, Ordering::Relaxed);

        let result = self.speak_inner(text).await;

        self.speaking.store(false, Ordering::Relaxed);
        result
    }

    async fn speak_inner(&self, text: &str) -> SpeakResult {
        // Split on sentence boundaries (.!? followed by whitespace)
        // Rust regex doesn't support lookbehind, so split manually
        let sentences = split_sentences(text.trim());

        if sentences.is_empty() {
            return SpeakResult {
                generate_ms: 0.0,
                playback_ms: 0.0,
                cancelled: false,
                text_spoken: String::new(),
            };
        }

        let mut total_gen_ms = 0.0;
        let mut total_play_ms = 0.0;
        let mut cancelled = false;

        for (i, sentence) in sentences.iter().enumerate() {
            // Check cancel before generation
            if self.cancel_flag.load(Ordering::Relaxed) {
                cancelled = true;
                info!("Cancelled before sentence {}/{}", i + 1, sentences.len());
                break;
            }

            // Generate audio for this sentence
            let t_gen = Instant::now();
            let samples = match self.generate_audio(sentence) {
                Ok(s) => s,
                Err(e) => {
                    warn!("TTS generation failed for sentence {}: {e}", i + 1);
                    continue;
                }
            };
            let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
            total_gen_ms += gen_ms;

            // Check cancel after generation
            if self.cancel_flag.load(Ordering::Relaxed) {
                cancelled = true;
                info!("Cancelled after generating sentence {}/{}", i + 1, sentences.len());
                break;
            }

            if samples.is_empty() {
                continue;
            }

            // Play audio
            let t_play = Instant::now();
            let was_cancelled = self.play_audio(samples).await;
            let play_ms = t_play.elapsed().as_secs_f64() * 1000.0;
            total_play_ms += play_ms;

            if was_cancelled {
                cancelled = true;
                info!("Cancelled during playback of sentence {}/{}", i + 1, sentences.len());
                break;
            }

            let duration = play_ms / 1000.0;
            debug!(
                "Sentence {}/{}: gen={gen_ms:.0}ms play={duration:.1}s",
                i + 1,
                sentences.len()
            );
        }

        SpeakResult {
            generate_ms: total_gen_ms,
            playback_ms: total_play_ms,
            cancelled,
            text_spoken: text.to_string(),
        }
    }

    /// Generate audio samples for a single sentence.
    fn generate_audio(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut session_guard = self.session.lock().unwrap();
        let session = session_guard.as_mut().ok_or("Model not loaded")?;
        let phonemizer = self.phonemizer.as_ref().ok_or("Phonemizer not loaded")?;

        // 1. Text → phonemes via misaki-rs G2P
        let (phonemes, _tokens) = phonemizer
            .g2p(text)
            .map_err(|e| format!("Phonemization failed: {e}"))?;

        if phonemes.is_empty() {
            return Ok(Vec::new());
        }

        // 2. Phonemes → token IDs
        let mut token_ids: Vec<i64> = Vec::with_capacity(phonemes.len() + 2);
        token_ids.push(0); // Start padding
        for ch in phonemes.chars() {
            if let Some(&id) = self.vocab.get(&ch) {
                token_ids.push(id);
            }
            // Skip unknown characters silently
        }
        token_ids.push(0); // End padding

        let n_tokens = token_ids.len().min(MAX_TOKENS);
        token_ids.truncate(n_tokens);

        // 3. Get voice style vector for this token count
        let voice_name = self.voice.lock().unwrap().clone();
        let voice_data = self
            .voices
            .get(&voice_name)
            .ok_or_else(|| format!("Voice not found: {voice_name}"))?;

        // Index into style array by token count (clamped to max)
        let style_idx = (n_tokens.saturating_sub(2)).min(voice_data.styles.nrows() - 1);
        let style_vec: Vec<f32> = voice_data.styles.row(style_idx).to_vec();

        // 4. Build ONNX input tensors (ort 2.0: must convert to Tensor values)
        let tokens_array =
            ndarray::Array2::from_shape_vec((1, n_tokens), token_ids.clone())
                .map_err(|e| format!("Failed to create tokens tensor: {e}"))?;
        let tokens_tensor = Tensor::from_array(tokens_array)
            .map_err(|e| format!("Failed to create tokens ort tensor: {e}"))?;

        let style_array =
            ndarray::Array2::from_shape_vec((1, 256), style_vec)
                .map_err(|e| format!("Failed to create style tensor: {e}"))?;
        let style_tensor = Tensor::from_array(style_array)
            .map_err(|e| format!("Failed to create style ort tensor: {e}"))?;

        let speed_array = ndarray::Array1::from_vec(vec![self.speed]);
        let speed_tensor = Tensor::from_array(speed_array)
            .map_err(|e| format!("Failed to create speed ort tensor: {e}"))?;

        // 5. Run ONNX inference
        let outputs = session
            .run(ort::inputs![
                "tokens" => tokens_tensor,
                "style" => style_tensor,
                "speed" => speed_tensor
            ])
            .map_err(|e| format!("ONNX inference failed: {e}"))?;

        // 6. Extract audio samples from output
        // ort 2.0: try_extract_tensor returns (&Shape, &[T]) tuple
        let first_output = outputs
            .iter()
            .next()
            .ok_or("No output tensor from model")?;

        let (_shape, audio_slice) = first_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract audio tensor: {e}"))?;

        let samples: Vec<f32> = audio_slice.iter().copied().collect();
        debug!(
            "Generated {} samples ({:.1}s)",
            samples.len(),
            samples.len() as f32 / SAMPLE_RATE as f32
        );

        Ok(samples)
    }

    /// Play audio samples through rodio. Returns true if cancelled during playback.
    async fn play_audio(&self, samples: Vec<f32>) -> bool {
        let stream = match &self.output_stream {
            Some(s) => s,
            None => {
                warn!("No audio output stream");
                return false;
            }
        };

        // rodio 0.21: Sink::connect_new takes &Mixer
        let sink = Sink::connect_new(stream.mixer());
        let source = SamplesBuffer::new(1, SAMPLE_RATE, samples);
        sink.append(source);

        // Store sink for cancel access
        *self.active_sink.lock().unwrap() = Some(sink);

        // Poll for completion or cancellation
        let cancel_flag = self.cancel_flag.clone();
        let active_sink = self.active_sink.clone();

        let was_cancelled = tokio::task::spawn_blocking(move || {
            loop {
                // Check if sink is done
                let is_empty = {
                    let guard = active_sink.lock().unwrap();
                    match guard.as_ref() {
                        Some(s) => s.empty(),
                        None => true,
                    }
                };

                if is_empty {
                    return false;
                }

                // Check cancel
                if cancel_flag.load(Ordering::Relaxed) {
                    if let Some(sink) = active_sink.lock().unwrap().take() {
                        sink.stop();
                    }
                    return true;
                }

                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        })
        .await
        .unwrap_or(false);

        // Clean up sink reference
        *self.active_sink.lock().unwrap() = None;

        was_cancelled
    }

    /// Cancel current speech immediately.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
        if let Some(sink) = self.active_sink.lock().unwrap().take() {
            sink.stop();
        }
        self.speaking.store(false, Ordering::Relaxed);
        info!("TTS cancelled");
    }

    /// Cancel and wait for speak lock to be released.
    pub async fn cancel_and_wait(&self) {
        self.cancel();
        // Wait for any in-progress speak to finish
        let _guard = self.speak_lock.lock().await;
    }
}

// --- Helper functions ---

/// Load tokenizer vocabulary from tokenizer.json.
fn load_tokenizer(path: &Path) -> Result<HashMap<char, i64>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read tokenizer: {e}"))?;

    let data: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|e| format!("Failed to parse tokenizer JSON: {e}"))?;

    let vocab = data["model"]["vocab"]
        .as_object()
        .ok_or("Missing model.vocab in tokenizer.json")?;

    let mut map = HashMap::new();
    for (token, id) in vocab {
        let id = id.as_i64().ok_or("Token ID is not an integer")?;
        // Each token should be a single character
        if let Some(ch) = token.chars().next() {
            map.insert(ch, id);
        }
    }

    Ok(map)
}

/// Load all voice styles from an NPZ file.
fn load_voices(path: &Path) -> Result<HashMap<String, VoiceData>, String> {
    let file = fs::File::open(path)
        .map_err(|e| format!("Failed to open voices file: {e}"))?;

    let mut npz = NpzReader::new(file)
        .map_err(|e| format!("Failed to read NPZ voices file: {e}"))?;

    let names: Vec<String> = npz
        .names()
        .map_err(|e| format!("Failed to list NPZ entries: {e}"))?
        .into_iter()
        .map(|n| n.trim_end_matches(".npy").to_string())
        .collect();

    let mut voices = HashMap::new();
    for name in &names {
        let npy_name = format!("{name}.npy");
        let arr: Array3<f32> = npz
            .by_name(&npy_name)
            .map_err(|e| format!("Failed to read voice '{name}': {e}"))?;

        // Shape is (510, 1, 256). Squeeze the middle dimension to (510, 256).
        let dim0 = arr.shape()[0];
        let dim2 = arr.shape()[2];
        let styles = arr
            .into_shape_with_order((dim0, dim2))
            .map_err(|e| format!("Failed to reshape voice '{name}': {e}"))?;

        voices.insert(name.clone(), VoiceData { styles });
    }

    Ok(voices)
}

/// Split text into sentences at .!? boundaries.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        if (b == b'.' || b == b'!' || b == b'?')
            && i + 1 < bytes.len()
            && bytes[i + 1].is_ascii_whitespace()
        {
            let end = i + 1;
            let s = text[start..end].trim();
            if !s.is_empty() {
                sentences.push(s);
            }
            start = end;
        }
    }

    // Remainder
    let s = text[start..].trim();
    if !s.is_empty() {
        sentences.push(s);
    }

    sentences
}
