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

const PROMPT_TEMPLATE: &str = r#"Fix punctuation, capitalization, and obvious speech-recognition grammar errors in this transcription. The speaker is dictating instructions to the recipient.

Rules:
- Preserve every word unless a minimal change is required to fix an obvious recognition or grammar error
- Never polish, summarize, or make optional stylistic rewrites
- Treat action requests as commands to the recipient, never as actions the speaker will perform
- If an accidental "I" is the subject of an action request, remove it to restore the outward command: "now I generate the PDF" -> "now generate the PDF"; "then I send the invoice" -> "then send the invoice"
- Preserve genuine first-person context, including the speaker's intent, opinion, approval, or situation: "I want", "I think", "I agree", "I approve", "from my side", and "let me"
- Fix obvious homophones (their/there, its/it's)
- Preserve meaning, facts, numbers, domain terms, and names. Do not invent information

Output ONLY the corrected text, nothing else.

Text: {text}

Corrected:"#;

const RETRY_PROMPT_TEMPLATE: &str = r#"Retry the correction from the ORIGINAL transcription below.

Your previous answer had repeated stutter words or phrases. Do not introduce any repeated words or phrases that are not present in the original.

Rules:
- Preserve every word unless a minimal change is required to fix an obvious recognition or grammar error
- Never polish, summarize, or make optional stylistic rewrites
- Treat action requests as commands to the recipient, never as actions the speaker will perform
- If an accidental "I" is the subject of an action request, remove it to restore the outward command: "now I generate the PDF" -> "now generate the PDF"; "then I send the invoice" -> "then send the invoice"
- Preserve genuine first-person context, including the speaker's intent, opinion, approval, or situation: "I want", "I think", "I agree", "I approve", "from my side", and "let me"
- Fix obvious homophones (their/there, its/it's)
- Preserve meaning, facts, numbers, domain terms, and names. Do not invent information
- Output ONLY the corrected text, nothing else

Original transcription: {text}

Corrected:"#;

const AUDIO_PROMPT: &str =
    "Transcribe this audio word for word. Output ONLY the transcription, nothing else.";

const AUDIO_RETRY_PROMPT: &str = concat!(
    "Transcribe this audio word for word. Your previous answer had repeated stutter words ",
    "or phrases. Do not introduce repeated words or phrases. Output ONLY the transcription, ",
    "nothing else."
);

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

        let Some(result) = self.generate(&prompt, "Ollama request").await else {
            return text.to_string();
        };

        if ollama_output_has_stutter(text, &result) {
            warn!("Ollama output had repeated stutter text; retrying once");
            let retry_prompt = RETRY_PROMPT_TEMPLATE.replace("{text}", text);
            let Some(retry) = self.generate(&retry_prompt, "Ollama retry request").await else {
                return text.to_string();
            };

            if ollama_output_has_stutter(text, &retry) {
                warn!("Ollama retry still had repeated stutter text, using original text");
                text.to_string()
            } else {
                retry
            }
        } else {
            result
        }
    }

    async fn generate(&self, prompt: &str, label: &str) -> Option<String> {
        let body = json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": false,
            "think": false,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": 1024
            }
        });

        let url = format!("{}/api/generate", self.config.host);

        match self.client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    warn!("Ollama returned status {}", resp.status());
                    return None;
                }
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let result = data["response"].as_str().unwrap_or("").trim().to_string();
                        if result.is_empty() {
                            warn!("Ollama returned empty response");
                            None
                        } else {
                            debug!("Ollama output: '{result}'");
                            Some(result)
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse Ollama response: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                self.log_request_error(&e, label);
                None
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

        // Use /api/chat with multimodal message (audio sent via images field)
        let Some(result) = self
            .chat_audio(&audio_b64, AUDIO_PROMPT, "Ollama audio request")
            .await
        else {
            return None;
        };

        if is_pathological_stutter(&result) {
            warn!("Ollama audio output had repeated stutter text; retrying once");
            let Some(retry) = self
                .chat_audio(&audio_b64, AUDIO_RETRY_PROMPT, "Ollama audio retry request")
                .await
            else {
                return None;
            };

            if is_pathological_stutter(&retry) {
                warn!("Ollama audio retry still had repeated stutter text; dropping utterance");
                None
            } else {
                Some(retry)
            }
        } else {
            Some(result)
        }
    }

    async fn chat_audio(&self, audio_b64: &str, prompt: &str, label: &str) -> Option<String> {
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
                self.log_request_error(&e, label);
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

#[derive(Debug, Default, PartialEq, Eq)]
struct RepetitionStats {
    max_word_run: usize,
    max_phrase_repeats: usize,
}

fn ollama_output_has_stutter(original: &str, candidate: &str) -> bool {
    let candidate_stats = repetition_stats(candidate);
    if stats_are_pathological(&candidate_stats) {
        return true;
    }

    let original_stats = repetition_stats(original);
    (candidate_stats.max_word_run >= 3
        && candidate_stats.max_word_run > original_stats.max_word_run)
        || (candidate_stats.max_phrase_repeats >= 3
            && candidate_stats.max_phrase_repeats > original_stats.max_phrase_repeats)
}

pub(crate) fn is_pathological_stutter(text: &str) -> bool {
    stats_are_pathological(&repetition_stats(text))
}

fn stats_are_pathological(stats: &RepetitionStats) -> bool {
    stats.max_word_run >= 6 || stats.max_phrase_repeats >= 4
}

fn repetition_stats(text: &str) -> RepetitionStats {
    let tokens = normalized_tokens(text);
    RepetitionStats {
        max_word_run: max_word_run(&tokens),
        max_phrase_repeats: max_phrase_repeats(&tokens),
    }
}

fn normalized_tokens(text: &str) -> Vec<String> {
    let mut normalized = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '\'' {
            normalized.extend(ch.to_lowercase());
        } else {
            normalized.push(' ');
        }
    }

    normalized
        .split_whitespace()
        .map(|token| token.trim_matches('\''))
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn max_word_run(tokens: &[String]) -> usize {
    let mut max_run = 0;
    let mut current_run = 0;
    let mut previous = "";

    for token in tokens {
        if token == previous {
            current_run += 1;
        } else {
            previous = token;
            current_run = 1;
        }
        max_run = max_run.max(current_run);
    }

    max_run
}

fn max_phrase_repeats(tokens: &[String]) -> usize {
    let mut max_repeats = 0;

    for phrase_len in 2..=4 {
        if tokens.len() < phrase_len * 2 {
            continue;
        }

        for start in 0..=tokens.len() - phrase_len * 2 {
            let phrase = &tokens[start..start + phrase_len];
            let mut repeats = 1;
            let mut next = start + phrase_len;

            while next + phrase_len <= tokens.len() && tokens[next..next + phrase_len] == *phrase {
                repeats += 1;
                next += phrase_len;
            }

            max_repeats = max_repeats.max(repeats);
        }
    }

    max_repeats
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

#[cfg(test)]
mod tests {
    use super::{
        is_pathological_stutter, ollama_output_has_stutter, repetition_stats, RepetitionStats,
    };

    #[test]
    fn detects_dash_and_comma_word_stutter() {
        assert!(is_pathological_stutter(
            "The source-the-the-the-the-the-the algorithm broke."
        ));
        assert!(is_pathological_stutter(
            "So the, the, the, the, the, the thing failed."
        ));
    }

    #[test]
    fn detects_ollama_added_short_stutter_without_rejecting_original_repetition() {
        assert!(ollama_output_has_stutter(
            "The flat table is query perspective.",
            "The flat table is query perspective perspective perspective."
        ));
        assert!(!ollama_output_has_stutter(
            "The flat table is query perspective perspective perspective.",
            "The flat table is query perspective perspective perspective."
        ));
    }

    #[test]
    fn detects_repeated_phrase_stutter() {
        assert_eq!(
            repetition_stats("source accounting and all that and all that and all that"),
            RepetitionStats {
                max_word_run: 1,
                max_phrase_repeats: 3,
            }
        );
        assert!(ollama_output_has_stutter(
            "source accounting and all that",
            "source accounting and all that and all that and all that"
        ));
    }
}
