//! Ollama text processing for grammar and punctuation correction.
//!
//! Sends transcribed text to Ollama's /api/generate endpoint for
//! grammar/spelling fixes. Falls back gracefully if Ollama is unavailable.
//!
use reqwest::Client;
use serde_json::json;
use tracing::{debug, warn};

use crate::config::OllamaConfig;

const PROMPT_TEMPLATE: &str = r#"Fix punctuation, capitalization, and obvious speech-recognition grammar errors in the user-provided transcription. The speaker is dictating instructions to the recipient.

Rules:
- Preserve every word unless a minimal change is required to fix an obvious recognition or grammar error
- Never polish, summarize, or make optional stylistic rewrites
- Treat action requests as commands to the recipient, never as actions the speaker will perform
- If an accidental "I" is the subject of an action request, remove it to restore the outward command: "now I generate the PDF" -> "now generate the PDF"; "then I send the invoice" -> "then send the invoice"
- Preserve genuine first-person context, including the speaker's intent, opinion, approval, or situation: "I want", "I think", "I agree", "I approve", "from my side", and "let me"
- Fix obvious homophones (their/there, its/it's)
- Preserve meaning, facts, numbers, domain terms, and names. Do not invent information

Return a JSON object with exactly one string field named corrected_text."#;

const RETRY_PROMPT_TEMPLATE: &str = r#"Retry correction of the user-provided ORIGINAL transcription.

Your previous answer had repeated stutter words or phrases. Do not introduce any repeated words or phrases that are not present in the original.

Rules:
- Preserve every word unless a minimal change is required to fix an obvious recognition or grammar error
- Never polish, summarize, or make optional stylistic rewrites
- Treat action requests as commands to the recipient, never as actions the speaker will perform
- If an accidental "I" is the subject of an action request, remove it to restore the outward command: "now I generate the PDF" -> "now generate the PDF"; "then I send the invoice" -> "then send the invoice"
- Preserve genuine first-person context, including the speaker's intent, opinion, approval, or situation: "I want", "I think", "I agree", "I approve", "from my side", and "let me"
- Fix obvious homophones (their/there, its/it's)
- Preserve meaning, facts, numbers, domain terms, and names. Do not invent information
- Return a JSON object with exactly one string field named corrected_text"#;

const PROTECTED_TERMS: &[&str] = &[
    "Astralane",
    "ClickHouse",
    "Pingora",
    "Hermes",
    "Dagster",
    "Helius",
    "Jito",
    "Solscan",
    "Solana",
    "MEV",
    "Rust",
    "Python",
];

pub struct OllamaProcessor {
    config: OllamaConfig,
    client: Client,
}

#[derive(Debug, Clone, Default)]
pub struct CorrectionMetadata {
    pub accepted: bool,
    pub fallback_reason: Option<String>,
    pub load_ms: Option<i64>,
    pub prompt_eval_ms: Option<i64>,
    pub eval_ms: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct CorrectionResult {
    pub text: String,
    pub metadata: CorrectionMetadata,
}

struct GeneratedCorrection {
    text: String,
    metadata: CorrectionMetadata,
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
    pub async fn process(&self, text: &str) -> CorrectionResult {
        if !self.config.enabled || text.trim().is_empty() {
            return CorrectionResult {
                text: text.to_string(),
                metadata: CorrectionMetadata {
                    fallback_reason: Some("disabled_or_empty".into()),
                    ..CorrectionMetadata::default()
                },
            };
        }

        debug!("Sending to Ollama model '{}': {}", self.config.model, text);

        let Some(result) = self.generate(PROMPT_TEMPLATE, text, "Ollama request").await else {
            return fallback_result(text, "request_failed", None);
        };

        match validate_correction(text, &result.text) {
            Ok(()) => CorrectionResult {
                text: result.text,
                metadata: CorrectionMetadata {
                    accepted: true,
                    ..result.metadata
                },
            },
            Err(reason) => {
                warn!("Rejected Ollama correction ({reason}); retrying once");
                let Some(retry) = self
                    .generate(RETRY_PROMPT_TEMPLATE, text, "Ollama retry request")
                    .await
                else {
                    return fallback_result(text, "retry_request_failed", Some(result.metadata));
                };

                match validate_correction(text, &retry.text) {
                    Ok(()) => CorrectionResult {
                        text: retry.text,
                        metadata: CorrectionMetadata {
                            accepted: true,
                            ..retry.metadata
                        },
                    },
                    Err(retry_reason) => {
                        warn!(
                            "Rejected Ollama retry ({retry_reason}); using original transcription"
                        );
                        fallback_result(text, retry_reason, Some(retry.metadata))
                    }
                }
            }
        }
    }

    async fn generate(
        &self,
        system: &str,
        transcription: &str,
        label: &str,
    ) -> Option<GeneratedCorrection> {
        let body = json!({
            "model": self.config.model,
            "system": system,
            "prompt": transcription,
            "stream": false,
            "think": false,
            "format": correction_schema(),
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": 512
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
                        let raw = data["response"].as_str().unwrap_or("").trim();
                        let result = serde_json::from_str::<serde_json::Value>(raw)
                            .ok()
                            .and_then(|value| value["corrected_text"].as_str().map(str::to_owned));
                        let Some(result) = result.filter(|value| !value.trim().is_empty()) else {
                            warn!("Ollama returned empty response");
                            return None;
                        };
                        debug!("Ollama output: '{result}'");
                        Some(GeneratedCorrection {
                            text: result,
                            metadata: CorrectionMetadata {
                                load_ms: nanos_to_ms(data["load_duration"].as_i64()),
                                prompt_eval_ms: nanos_to_ms(data["prompt_eval_duration"].as_i64()),
                                eval_ms: nanos_to_ms(data["eval_duration"].as_i64()),
                                ..CorrectionMetadata::default()
                            },
                        })
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

fn correction_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "corrected_text": { "type": "string" }
        },
        "required": ["corrected_text"],
        "additionalProperties": false
    })
}

fn nanos_to_ms(nanos: Option<i64>) -> Option<i64> {
    nanos.map(|value| value / 1_000_000)
}

fn fallback_result(
    original: &str,
    reason: &str,
    metadata: Option<CorrectionMetadata>,
) -> CorrectionResult {
    let mut metadata = metadata.unwrap_or_default();
    metadata.accepted = false;
    metadata.fallback_reason = Some(reason.to_string());
    CorrectionResult {
        text: original.to_string(),
        metadata,
    }
}

fn validate_correction(original: &str, candidate: &str) -> Result<(), &'static str> {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return Err("empty_correction");
    }
    if candidate.contains("```")
        || candidate
            .to_ascii_lowercase()
            .starts_with("corrected_text:")
    {
        return Err("wrapped_or_explained_output");
    }
    if ollama_output_has_stutter(original, candidate) {
        return Err("introduced_stutter");
    }

    let original_words = original.split_whitespace().count();
    let candidate_words = candidate.split_whitespace().count();
    if original_words >= 4 {
        let minimum = (original_words / 2).max(1);
        let maximum = original_words.saturating_mul(3) / 2 + 3;
        if candidate_words < minimum || candidate_words > maximum {
            return Err("large_length_change");
        }
    }

    if significant_tokens(original, |token| {
        token.chars().any(|ch| ch.is_ascii_digit())
    }) != significant_tokens(candidate, |token| {
        token.chars().any(|ch| ch.is_ascii_digit())
    }) {
        return Err("changed_numeric_fact");
    }
    if significant_tokens(original, |token| {
        token.starts_with("http://") || token.starts_with("https://")
    }) != significant_tokens(candidate, |token| {
        token.starts_with("http://") || token.starts_with("https://")
    }) {
        return Err("changed_url");
    }

    let original_tokens = normalized_tokens(original);
    let candidate_tokens = normalized_tokens(candidate);
    if PROTECTED_TERMS.iter().any(|term| {
        let term = term.to_ascii_lowercase();
        original_tokens.contains(&term) && !candidate_tokens.contains(&term)
    }) {
        return Err("removed_protected_term");
    }

    Ok(())
}

fn significant_tokens<F>(text: &str, predicate: F) -> Vec<String>
where
    F: Fn(&str) -> bool,
{
    let mut tokens = text
        .split_whitespace()
        .map(|token| {
            token.trim_matches(|ch: char| {
                matches!(
                    ch,
                    '.' | ','
                        | ';'
                        | ':'
                        | '!'
                        | '?'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '\''
                        | '"'
                )
            })
        })
        .filter(|token| predicate(token))
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    tokens.sort();
    tokens
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

#[cfg(test)]
mod tests {
    use super::{
        is_pathological_stutter, ollama_output_has_stutter, repetition_stats, validate_correction,
        RepetitionStats,
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

    #[test]
    fn correction_validation_preserves_facts_and_domain_terms() {
        assert!(validate_correction(
            "Send 28 SOL to Jito at https://example.com.",
            "Send 29 SOL to Jito at https://example.com."
        )
        .is_err());
        assert!(validate_correction(
            "Check the ClickHouse table now.",
            "Check the database table now."
        )
        .is_err());
        assert!(validate_correction(
            "send 28 SOL to Jito at https://example.com.",
            "Send 28 SOL to Jito at https://example.com."
        )
        .is_ok());
        assert!(validate_correction("Send 28 now.", "Send 28, now!").is_ok());
        assert!(validate_correction("I trust this result.", "I trust this result!").is_ok());
        assert!(
            validate_correction("Open https://example.com now.", "Open the website now.").is_err()
        );
    }
}
