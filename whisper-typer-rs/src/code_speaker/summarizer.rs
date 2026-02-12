//! Ollama text summarization for TTS.
//!
//! Summarizes long text into 1-2 sentences before speaking.
//! Falls back to truncation if Ollama is unavailable.

use std::time::Instant;

use reqwest::Client;
use serde_json::json;
use tracing::{info, warn};

const SUMMARIZE_PROMPT: &str = r#"Summarize this in 1-2 short sentences suitable for text-to-speech. Be concise and conversational. Output ONLY the summary, nothing else.

Text: {text}

Summary:"#;

const MAX_INPUT_CHARS: usize = 2000;

pub struct OllamaSummarizer {
    model: String,
    host: String,
    client: Client,
}

impl OllamaSummarizer {
    pub fn new(model: &str, host: &str) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            model: model.to_string(),
            host: host.to_string(),
            client,
        }
    }

    /// Summarize text for TTS. Returns (summary, latency_ms).
    pub async fn summarize(&self, text: &str) -> (String, f64) {
        let t_start = Instant::now();

        // Truncate long input
        let input = if text.len() > MAX_INPUT_CHARS {
            &text[..MAX_INPUT_CHARS]
        } else {
            text
        };

        let prompt = SUMMARIZE_PROMPT.replace("{text}", input);

        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": 0.3,
                "num_predict": 200
            }
        });

        let url = format!("{}/api/generate", self.host);
        let latency_ms;

        match self.client.post(&url).json(&body).send().await {
            Ok(resp) => {
                latency_ms = t_start.elapsed().as_secs_f64() * 1000.0;
                if !resp.status().is_success() {
                    warn!("Ollama summarizer returned status {}", resp.status());
                    return (Self::fallback_truncate(text), latency_ms);
                }
                match resp.json::<serde_json::Value>().await {
                    Ok(data) => {
                        let result = data["response"]
                            .as_str()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if result.is_empty() {
                            warn!("Ollama summarizer returned empty response");
                            (Self::fallback_truncate(text), latency_ms)
                        } else {
                            info!("Summarized {} chars → {} chars ({latency_ms:.0}ms)", text.len(), result.len());
                            (result, latency_ms)
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse Ollama summarizer response: {e}");
                        (Self::fallback_truncate(text), latency_ms)
                    }
                }
            }
            Err(e) => {
                latency_ms = t_start.elapsed().as_secs_f64() * 1000.0;
                warn!("Ollama summarizer request failed: {e}");
                (Self::fallback_truncate(text), latency_ms)
            }
        }
    }

    /// Fallback: take first 2 sentences.
    fn fallback_truncate(text: &str) -> String {
        let text = text.trim();
        let mut count = 0;
        let mut end = text.len();
        for (i, b) in text.bytes().enumerate() {
            if b == b'.' || b == b'!' || b == b'?' {
                count += 1;
                if count >= 2 {
                    end = i + 1;
                    break;
                }
            }
        }
        text[..end].to_string()
    }
}
