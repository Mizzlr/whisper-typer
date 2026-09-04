//! Optional punctuation and truecasing client.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::PunctuationConfig;

#[derive(Clone)]
pub struct PunctuationClient {
    config: PunctuationConfig,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct PunctuationRequest<'a> {
    text: &'a str,
}

#[derive(Deserialize)]
struct PunctuationResponse {
    text: String,
    latency_ms: Option<f64>,
}

pub struct PunctuationResult {
    pub text: String,
    pub latency_ms: f64,
}

impl PunctuationClient {
    pub fn new(config: PunctuationConfig) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .map_err(|error| format!("build punctuation client: {error}"))?;
        Ok(Self { config, client })
    }

    pub async fn process(&self, text: &str) -> Result<PunctuationResult, String> {
        let started = Instant::now();
        let response = self
            .client
            .post(&self.config.url)
            .json(&PunctuationRequest { text })
            .send()
            .await
            .map_err(|error| format!("punctuation request: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("punctuation service returned {status}: {body}"));
        }
        let response: PunctuationResponse = response
            .json()
            .await
            .map_err(|error| format!("decode punctuation response: {error}"))?;
        let corrected = response.text.trim().to_string();
        if corrected.is_empty() {
            return Err("punctuation service returned empty text".into());
        }
        Ok(PunctuationResult {
            text: corrected,
            latency_ms: response
                .latency_ms
                .unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0),
        })
    }
}
