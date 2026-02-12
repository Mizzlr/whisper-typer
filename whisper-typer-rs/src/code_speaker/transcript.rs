//! Claude Code JSONL transcript parser.
//!
//! Extracts assistant text and user timestamps from Claude Code session
//! transcripts for use in TTS reminders and context-aware notifications.

use std::fs;
use std::path::Path;

use tracing::warn;

/// Extract the last assistant text message from a Claude Code JSONL transcript.
///
/// Walks the file backwards to find the most recent assistant message
/// with text content, truncated to `max_chars`.
pub fn extract_last_assistant_text(transcript_path: &Path, max_chars: usize) -> Option<String> {
    let contents = match fs::read_to_string(transcript_path) {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to read transcript {}: {e}", transcript_path.display());
            return None;
        }
    };

    for line in contents.lines().rev() {
        let entry: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if entry.get("type").and_then(|t| t.as_str()) != Some("assistant") {
            continue;
        }

        let content = entry
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_array());

        if let Some(blocks) = content {
            for block in blocks {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        let truncated: String = text.chars().take(max_chars).collect();
                        return Some(truncated);
                    }
                }
            }
        }
    }

    None
}

/// Extract the timestamp of the last user message from a Claude Code JSONL transcript.
pub fn extract_last_user_timestamp(transcript_path: &Path) -> Option<String> {
    let contents = match fs::read_to_string(transcript_path) {
        Ok(c) => c,
        Err(e) => {
            warn!("Failed to read transcript {}: {e}", transcript_path.display());
            return None;
        }
    };

    for line in contents.lines().rev() {
        let entry: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if entry.get("type").and_then(|t| t.as_str()) == Some("user") {
            if let Some(ts) = entry.get("timestamp").and_then(|t| t.as_str()) {
                return Some(ts.to_string());
            }
        }
    }

    None
}
