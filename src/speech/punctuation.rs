//! Optional punctuation and truecasing client.

use std::borrow::Cow;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::PunctuationConfig;

pub fn unknown_marker_count(text: &str) -> usize {
    text.as_bytes().windows(5).filter(|part| part.eq_ignore_ascii_case(b"<unk>")).count()
}

/// Remove ASR tokenizer artifacts before other text processing. Ordinary
/// transcripts borrow their input; adjacent words never get glued together.
/// Percent signs and all other actual content remain unchanged.
pub fn clean_asr_text(text: &str) -> Cow<'_, str> {
    let positions: Vec<_> = text.as_bytes().windows(5).enumerate()
        .filter_map(|(offset, part)| part.eq_ignore_ascii_case(b"<unk>").then_some(offset))
        .collect();
    if positions.is_empty() { return Cow::Borrowed(text); }
    let mut cleaned = String::with_capacity(text.len());
    let mut cursor = 0;
    for offset in positions {
        if offset < cursor { continue; }
        cleaned.push_str(&text[cursor..offset]);
        cursor = offset + 5;
        if cleaned.ends_with([' ', '\t']) {
            while matches!(text.as_bytes().get(cursor), Some(b' ' | b'\t')) { cursor += 1; }
        } else if cleaned.chars().last().is_some_and(char::is_alphanumeric)
            && text[cursor..].chars().next().is_some_and(char::is_alphanumeric) {
            cleaned.push(' ');
        }
    }
    cleaned.push_str(&text[cursor..]);
    Cow::Owned(cleaned.trim().to_string())
}

pub fn validate_punctuation_symbols(original: &str, corrected: &str) -> Result<(), &'static str> {
    if unknown_marker_count(corrected) > unknown_marker_count(original)
        || corrected.matches('%').count() != original.matches('%').count() {
        Err("punctuation service corrupted symbols")
    } else { Ok(()) }
}

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
        validate_punctuation_symbols(text, &corrected).map_err(str::to_string)?;
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

    #[test]
    fn cleans_unknown_markers_before_processing_without_losing_symbols() {
        assert!(matches!(clean_asr_text("Are you 100% sure?"), Cow::Borrowed(_)));
        for (input, expected) in [
            ("Can you give <Unk>me links?", "Can you give me links?"),
            ("to<UNK>can", "to can"),
            ("one <unk> two", "one two"),
            ("<uNk><UNK>", ""),
            ("你好<Unk>世界 12.5%", "你好 世界 12.5%"),
            ("100<Unk>% complete", "100% complete"),
            ("<unknown> is a tag", "<unknown> is a tag"),
        ] { assert_eq!(clean_asr_text(input), expected); }
        assert!(validate_punctuation_symbols("100% complete", "100<Unk> complete").is_err());
        assert!(validate_punctuation_symbols("Give me links", "Give <UNK>me links.").is_err());
        assert!(validate_punctuation_symbols("12.5% complete", "12.5% complete.").is_ok());
    }

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
                            "mixed case symbol" => "Mixed <Unk> case symbol.",
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
        for text in ["are you 100% sure", "it is 25%", "unknown symbol", "mixed case symbol"] {
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
