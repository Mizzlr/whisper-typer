//! tts-hook: Claude Code hook binary for TTS notifications.
//!
//! Reads event JSON from stdin, sends HTTP requests to the TTS API.
//! Replaces hooks/tts-hook.sh with zero subprocess overhead.
//! Logs all events to ~/.tts-hook-history/YYYY-MM-DD.jsonl.

use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde::{Deserialize, Serialize};

const TTS_API: &str = "http://127.0.0.1:8767";

// --- Event JSON from Claude Code ---

#[derive(Deserialize)]
#[allow(clippy::struct_field_names)]
struct HookEvent {
    hook_event_name: Option<String>,
    source: Option<String>,
    transcript_path: Option<String>,
    tool_name: Option<String>,
    notification_type: Option<String>,
}

// --- TTS API request ---

#[derive(Serialize)]
struct SpeakRequest {
    text: String,
    summarize: bool,
    event_type: String,
    start_reminder: bool,
}

// --- TTS API status response ---

#[derive(Deserialize)]
struct StatusResponse {
    model_loaded: Option<bool>,
}

// --- Transcript JSONL parsing ---

#[derive(Deserialize)]
struct TranscriptEntry {
    #[serde(rename = "type")]
    entry_type: Option<String>,
    message: Option<TranscriptMessage>,
}

#[derive(Deserialize)]
struct TranscriptMessage {
    content: Option<Vec<ContentBlock>>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: Option<String>,
    text: Option<String>,
}

// --- History record ---

#[derive(Serialize)]
struct HistoryRecord {
    timestamp: String,
    event: String,
    action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_chars: Option<usize>,
    duration_ms: u64,
    tts_api_up: bool,
}

fn history_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tts-hook-history")
}

fn save_record(record: &HistoryRecord) {
    let dir = history_dir();
    let _ = fs::create_dir_all(&dir);

    // Date from timestamp (first 10 chars: YYYY-MM-DD)
    let date = &record.timestamp[..10];
    let path = dir.join(format!("{date}.jsonl"));

    if let Ok(json) = serde_json::to_string(record) {
        if let Ok(mut file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(file, "{json}");
        }
    }
}

fn now_timestamp() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%.3f")
        .to_string()
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let t0 = Instant::now();

    // Read stdin
    let mut input = String::new();
    if std::io::stdin().read_to_string(&mut input).is_err() {
        return;
    }

    // Parse event
    let event: HookEvent = match serde_json::from_str(&input) {
        Ok(e) => e,
        Err(_) => return,
    };

    let event_name = match &event.hook_event_name {
        Some(name) => name.clone(),
        None => return,
    };

    // Build HTTP client with short timeouts
    let client = Client::builder()
        .connect_timeout(Duration::from_millis(300))
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap_or_else(|_| Client::new());

    // Quick connectivity check — exit cleanly if TTS API is down
    let tts_api_up = client
        .get(format!("{TTS_API}/status"))
        .send()
        .await
        .is_ok();

    if !tts_api_up {
        save_record(&HistoryRecord {
            timestamp: now_timestamp(),
            event: event_name,
            action: "skipped".into(),
            detail: Some("TTS API unreachable".into()),
            text: None,
            text_chars: None,
            duration_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
            tts_api_up: false,
        });
        return;
    }

    let (action, detail, text) = match event_name.as_str() {
        "SessionStart" => handle_session_start(&client, &event).await,
        "Stop" => handle_stop(&client, &event).await,
        "PermissionRequest" => handle_permission(&client, &event).await,
        "Notification" => handle_notification(&client, &event).await,
        "UserPromptSubmit" => handle_user_prompt_submit(&client).await,
        _ => ("ignored".into(), Some("unknown event".into()), None),
    };

    let text_chars = text.as_ref().map(String::len);
    save_record(&HistoryRecord {
        timestamp: now_timestamp(),
        event: event_name,
        action,
        detail,
        text,
        text_chars,
        duration_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
        tts_api_up: true,
    });
}

async fn handle_session_start(
    client: &Client,
    event: &HookEvent,
) -> (String, Option<String>, Option<String>) {
    // Skip resume and compaction restarts
    if let Some(source) = &event.source {
        if source == "resume" || source == "compact" {
            return (
                "skipped".into(),
                Some(format!("source={source}")),
                None,
            );
        }
    }

    // Check if model is loaded
    let Ok(resp) = client.get(format!("{TTS_API}/status")).send().await else {
        return ("skipped".into(), Some("status request failed".into()), None);
    };
    let Ok(status) = resp.json::<StatusResponse>().await else {
        return ("skipped".into(), Some("status parse error".into()), None);
    };

    if status.model_loaded != Some(true) {
        return (
            "skipped".into(),
            Some("model not loaded".into()),
            None,
        );
    }

    let text = "Claude Code is ready.".to_string();
    let _ = client
        .post(format!("{TTS_API}/speak"))
        .json(&SpeakRequest {
            text: text.clone(),
            summarize: false,
            event_type: "session_start".into(),
            start_reminder: false,
        })
        .send()
        .await;

    ("spoke".into(), None, Some(text))
}

async fn handle_stop(
    client: &Client,
    event: &HookEvent,
) -> (String, Option<String>, Option<String>) {
    let transcript_path = match &event.transcript_path {
        Some(p) if !p.is_empty() => p,
        _ => return ("skipped".into(), Some("no transcript path".into()), None),
    };

    let Some(text) = extract_last_assistant_text(transcript_path) else {
        return (
            "skipped".into(),
            Some("no assistant text found".into()),
            None,
        );
    };

    let _ = client
        .post(format!("{TTS_API}/speak"))
        .json(&SpeakRequest {
            text: text.clone(),
            summarize: true,
            event_type: "stop".into(),
            start_reminder: true,
        })
        .send()
        .await;

    ("spoke".into(), None, Some(text))
}

async fn handle_permission(
    client: &Client,
    event: &HookEvent,
) -> (String, Option<String>, Option<String>) {
    let tool = event.tool_name.as_deref().unwrap_or("unknown tool");

    let text = format!("Claude needs permission to use {tool}.");

    let _ = client
        .post(format!("{TTS_API}/speak"))
        .json(&SpeakRequest {
            text: text.clone(),
            summarize: false,
            event_type: "permission".into(),
            start_reminder: true,
        })
        .send()
        .await;

    ("spoke".into(), Some(tool.into()), Some(text))
}

async fn handle_notification(
    client: &Client,
    event: &HookEvent,
) -> (String, Option<String>, Option<String>) {
    let (text, event_type) = match event.notification_type.as_deref() {
        Some("idle_prompt") => ("Claude is waiting for your input.", "notification"),
        Some("permission_prompt") => ("Permission needed.", "permission"),
        Some(other) => {
            return (
                "skipped".into(),
                Some(format!("unknown notification: {other}")),
                None,
            )
        }
        None => return ("skipped".into(), Some("no notification_type".into()), None),
    };

    let _ = client
        .post(format!("{TTS_API}/speak"))
        .json(&SpeakRequest {
            text: text.into(),
            summarize: false,
            event_type: event_type.into(),
            start_reminder: true,
        })
        .send()
        .await;

    (
        "spoke".into(),
        event.notification_type.clone(),
        Some(text.into()),
    )
}

async fn handle_user_prompt_submit(
    client: &Client,
) -> (String, Option<String>, Option<String>) {
    let _ = client
        .post(format!("{TTS_API}/cancel-reminder"))
        .send()
        .await;

    ("cancel_reminder".into(), None, None)
}

/// Read a JSONL transcript file and extract the last assistant text block.
/// Returns up to 2000 characters.
fn extract_last_assistant_text(path: &str) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;

    for line in content.lines().rev() {
        let entry: TranscriptEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.entry_type.as_deref() != Some("assistant") {
            continue;
        }

        let message = entry.message?;
        let content_blocks = message.content?;

        for block in &content_blocks {
            if block.block_type.as_deref() != Some("text") {
                continue;
            }
            if let Some(text) = &block.text {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    let truncated: String = trimmed.chars().take(2000).collect();
                    return Some(truncated);
                }
            }
        }
    }

    None
}
