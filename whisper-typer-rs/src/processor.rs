//! Ollama text processing for grammar and punctuation correction.
//!
//! Sends transcribed text to Ollama's /api/generate endpoint for
//! grammar/spelling fixes. Falls back gracefully if Ollama is unavailable.

use std::collections::HashMap;

use reqwest::Client;
use serde_json::json;
use tracing::{debug, warn};

use crate::config::OllamaConfig;

const PROMPT_TEMPLATE: &str = r#"Fix this speech transcription. Correct:
- Grammar and punctuation
- Misspelled names
- Technical terms
- Every sentence must end with a full stop or question mark

Output ONLY the corrected text, nothing else.

Text: {text}

Corrected:"#;

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
    /// If `corrections` is provided, known substitutions are appended to the prompt.
    pub async fn process(&self, text: &str, corrections: Option<&HashMap<String, String>>) -> String {
        if !self.config.enabled || text.trim().is_empty() {
            return text.to_string();
        }

        let mut prompt = PROMPT_TEMPLATE.replace("{text}", text);

        // Inject per-project corrections into the prompt
        if let Some(corrections) = corrections {
            if !corrections.is_empty() {
                let mut section = String::from("\n\nKnown corrections (apply these substitutions):\n");
                for (wrong, right) in corrections {
                    section.push_str(&format!("- \"{wrong}\" → \"{right}\"\n"));
                }
                // Insert before the "Text:" line
                prompt = prompt.replacen("\nText:", &format!("{section}\nText:"), 1);
            }
        }
        debug!("Sending to Ollama model '{}': {}", self.config.model, text);

        let body = json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": false,
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
                        let result = data["response"]
                            .as_str()
                            .unwrap_or("")
                            .trim()
                            .to_string();
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
                if e.is_connect() {
                    warn!("Cannot connect to Ollama at {}", self.config.host);
                } else if e.is_timeout() {
                    warn!("Ollama request timed out");
                } else {
                    warn!("Ollama request failed: {e}");
                }
                text.to_string()
            }
        }
    }
}
