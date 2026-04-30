//! Ollama text processing for grammar and punctuation correction.
//!
//! Sends transcribed text to Ollama's /api/generate endpoint for
//! grammar/spelling fixes. Falls back gracefully if Ollama is unavailable.
//!
//! Also supports direct audio-to-text mode via Ollama's multimodal chat API,
//! bypassing Whisper entirely (requires an audio-capable model like gemma4).

use std::io::Cursor;

use base64::Engine;
use hound::{SampleFormat, WavSpec, WavWriter};
use reqwest::Client;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::config::OllamaConfig;

const PROMPT_TEMPLATE: &str = r#"Fix this speech transcription. Correct:
- Grammar and punctuation
- Misspelled names
- Technical terms
- Every sentence must end with a full stop or question mark

Output ONLY the corrected text, nothing else.

Text: {text}

Corrected:"#;

const AUDIO_PROMPT: &str =
    "Transcribe this audio word for word. Output ONLY the transcription, nothing else.";

pub struct OllamaProcessor {
    config: OllamaConfig,
    client: Client,
}

impl OllamaProcessor {
    pub fn new(config: OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self { config, client }
    }

    /// Process text through Ollama for grammar correction.
    /// Returns the original text if Ollama is disabled or unavailable.
    pub async fn process(&self, text: &str) -> String {
        if !self.config.enabled || text.trim().is_empty() {
            return text.to_string();
        }

        let prompt = PROMPT_TEMPLATE.replace("{text}", text);
        debug!("Sending to Ollama model '{}': {}", self.config.model, text);

        let body = json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": false,
            "think": false,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": 0.1,
                "num_predict": 500
            }
        });

        let url = format!("{}/api/generate", self.config.host);

        match self.client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    warn!("Ollama returned status {}", resp.status());
                    return text.to_string();
                }
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let result = data["response"].as_str().unwrap_or("").trim().to_string();
                        if result.is_empty() {
                            warn!("Ollama returned empty response, using original text");
                            text.to_string()
                        } else {
                            debug!("Ollama output: '{result}'");
                            result
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse Ollama response: {e}");
                        text.to_string()
                    }
                }
            }
            Err(e) => {
                self.log_request_error(&e, "Ollama request");
                text.to_string()
            }
        }
    }

    /// Send audio directly to Ollama for transcription + correction in a single pass.
    /// Bypasses Whisper entirely. Requires an audio-capable model (e.g. gemma4).
    /// Returns None on failure so the caller can fall back to the Whisper path.
    pub async fn process_audio(&self, samples: &[f32], sample_rate: u32) -> Option<String> {
        if !self.config.enabled || samples.is_empty() {
            return None;
        }

        // Encode PCM f32 samples as WAV in memory
        let wav_bytes = encode_wav(samples, sample_rate);
        let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

        let audio_duration = samples.len() as f64 / sample_rate as f64;
        info!(
            "Audio mode: sending {:.1}s ({} bytes WAV) to Ollama model '{}'",
            audio_duration,
            wav_bytes.len(),
            self.config.model
        );

        let prompt = AUDIO_PROMPT.to_string();

        // Use /api/chat with multimodal message (audio sent via images field)
        let body = json!({
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [audio_b64]
            }],
            "stream": false,
            "think": false,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": 0.1,
                "num_predict": 500
            }
        });

        let url = format!("{}/api/chat", self.config.host);

        match self.client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    warn!("Ollama audio mode returned status {}", resp.status());
                    return None;
                }
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let result = data["message"]["content"]
                            .as_str()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if result.is_empty() {
                            warn!("Ollama audio mode returned empty response");
                            None
                        } else {
                            debug!("Ollama audio output: '{result}'");
                            Some(result)
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse Ollama audio response: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                self.log_request_error(&e, "Ollama audio request");
                None
            }
        }
    }

    fn log_request_error(&self, e: &reqwest::Error, label: &str) {
        if e.is_connect() {
            warn!("Cannot connect to Ollama at {}", self.config.host);
        } else if e.is_timeout() {
            warn!("{label} timed out");
        } else {
            warn!("{label} failed: {e}");
        }
    }
}

/// Encode f32 PCM samples as 16-bit WAV in memory.
fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut buf, spec).expect("WAV writer creation failed");
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let val = (clamped * 32767.0) as i16;
            writer.write_sample(val).expect("WAV write failed");
        }
        writer.finalize().expect("WAV finalize failed");
    }
    buf.into_inner()
}
