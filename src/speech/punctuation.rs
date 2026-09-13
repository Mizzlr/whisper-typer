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
    pub used_fallback: bool,
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
        match self.process_at(&self.config.url, text).await {
            Ok(mut result) => {
                result.used_fallback = false;
                Ok(result)
            }
            Err(primary_error) => {
                let Some(fallback_url) = self
                    .config
                    .fallback_url
                    .as_deref()
                    .filter(|url| *url != self.config.url)
                else {
                    return Err(primary_error);
                };
                match self.process_at(fallback_url, text).await {
                    Ok(mut result) => {
                        result.used_fallback = true;
                        Ok(result)
                    }
                    Err(fallback_error) => Err(format!(
                        "primary failed ({primary_error}); fallback failed ({fallback_error})"
                    )),
                }
            }
        }
    }

    async fn process_at(&self, url: &str, text: &str) -> Result<PunctuationResult, String> {
        let started = Instant::now();
        let response = self
            .client
            .post(url)
            .json(&PunctuationRequest { text })
            .send()
            .await
            .map_err(|error| format!("punctuation request to {url}: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "punctuation service {url} returned {status}: {body}"
            ));
        }
        let response: PunctuationResponse = response
            .json()
            .await
            .map_err(|error| format!("decode punctuation response: {error}"))?;
        let corrected = response.text.trim().to_string();
        if corrected.is_empty() {
            return Err("punctuation service returned empty text".into());
        }
        // The punctuation tokenizer can turn symbols such as '%' into <unk>.
        // Reject corrupt output so the fallback endpoint or original ASR text
        // is used, rather than typing model tokens or losing percentages.
        if corrected.matches("<unk>").count() > text.matches("<unk>").count()
            || corrected.matches('%').count() != text.matches('%').count()
        {
            return Err("punctuation service corrupted symbols".into());
        }
        Ok(PunctuationResult {
            text: corrected,
            latency_ms: response
                .latency_ms
                .unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0),
            used_fallback: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use axum::{http::StatusCode, routing::post, Json, Router};
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn rejects_corrupted_percentages_but_accepts_preserved_symbols() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route(
                    "/punctuate",
                    post(|Json(request): Json<serde_json::Value>| async move {
                        let text = match request["text"].as_str().unwrap() {
                            "are you 100% sure" => "Are you 100<unk>? Sure.",
                            "it is 25%" => "It is 25.",
                            "unknown symbol" => "Unknown <unk> symbol.",
                            _ => "It is 12.5% complete.",
                        };
                        Json(json!({"text": text}))
                    }),
                ),
            )
            .await
            .unwrap();
        });
        let client = PunctuationClient::new(PunctuationConfig {
            enabled: true,
            url: format!("http://{addr}/punctuate"),
            fallback_url: None,
            timeout_ms: 500,
        })
        .unwrap();
        for text in ["are you 100% sure", "it is 25%", "unknown symbol"] {
            assert!(client
                .process(text)
                .await
                .err()
                .unwrap()
                .contains("corrupted symbols"));
        }
        assert_eq!(
            client.process("it is 12.5% complete").await.unwrap().text,
            "It is 12.5% complete."
        );
        server.abort();
    }

    #[tokio::test]
    async fn falls_back_when_primary_returns_an_error() {
        let primary_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let primary_addr = primary_listener.local_addr().unwrap();
        let primary = tokio::spawn(async move {
            axum::serve(
                primary_listener,
                Router::new().route(
                    "/punctuate",
                    post(|| async { StatusCode::SERVICE_UNAVAILABLE }),
                ),
            )
            .await
            .unwrap();
        });

        let fallback_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let fallback_addr = fallback_listener.local_addr().unwrap();
        let fallback = tokio::spawn(async move {
            axum::serve(
                fallback_listener,
                Router::new().route(
                    "/punctuate",
                    post(|| async {
                        Json(json!({
                            "text": "Fallback worked.",
                            "latency_ms": 1.0
                        }))
                    }),
                ),
            )
            .await
            .unwrap();
        });

        let client = PunctuationClient::new(PunctuationConfig {
            enabled: true,
            url: format!("http://{primary_addr}/punctuate"),
            fallback_url: Some(format!("http://{fallback_addr}/punctuate")),
            timeout_ms: 500,
        })
        .unwrap();
        let result = client.process("fallback worked").await.unwrap();

        assert_eq!(result.text, "Fallback worked.");
        assert!(result.used_fallback);
        primary.abort();
        fallback.abort();
    }
}
