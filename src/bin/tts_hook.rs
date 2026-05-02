//! tts-hook: Claude Code hook binary for TTS notifications.
//!
//! Reads event JSON from stdin, sends HTTP requests to the TTS API.
//! Speaks fixed 2-3 word phrases — task done vs. waiting for input —
//! never the full assistant text. Multi-session focus tracking still
//! applies: non-focus sessions get a `{project} done.` announcement.
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
    session_id: Option<String>,
    cwd: Option<String>,
    source: Option<String>,
    transcript_path: Option<String>,
    tool_name: Option<String>,
    notification_type: Option<String>,
    /// Set by UserPromptSubmit. Captured per-session and replayed on Stop
    /// to disambiguate "task done" announcements when multiple Claude Code
    /// sessions finish around the same time.
    prompt: Option<String>,
}

// --- TTS API request ---

#[derive(Serialize)]
struct SpeakRequest {
    text: String,
    event_type: String,
    session_id: String,
}

#[derive(Serialize)]
struct UserInputRequest {
    session_id: String,
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

// --- Focus session state ---

#[derive(Serialize, Deserialize)]
struct FocusState {
    session_id: String,
    project: String,
    timestamp: String,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project: Option<String>,
    is_focus: bool,
}

fn history_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".tts-hook-history")
}

// --- Focus session tracking ---

fn focus_file() -> PathBuf {
    history_dir().join(".focus-session")
}

/// Read current focus session. Returns None if file missing or stale (>6h).
fn read_focus() -> Option<FocusState> {
    let content = fs::read_to_string(focus_file()).ok()?;
    let state: FocusState = serde_json::from_str(&content).ok()?;

    // Expire focus after 6 hours (handles overnight staleness)
    if let Ok(ts) = chrono::NaiveDateTime::parse_from_str(&state.timestamp, "%Y-%m-%dT%H:%M:%S%.3f")
    {
        let age = chrono::Local::now().naive_local() - ts;
        if age > chrono::Duration::hours(6) {
            return None;
        }
    }

    Some(state)
}

/// Write focus session.
fn write_focus(session_id: &str, project: &str) {
    let state = FocusState {
        session_id: session_id.to_string(),
        project: project.to_string(),
        timestamp: now_timestamp(),
    };
    let _ = fs::create_dir_all(history_dir());
    if let Ok(json) = serde_json::to_string(&state) {
        let _ = fs::write(focus_file(), json);
    }
}

/// Check if this session is the focus session.
/// Returns true if: (a) session matches focus, or (b) no focus exists.
fn is_focus_session(session_id: &str) -> bool {
    match read_focus() {
        Some(focus) => focus.session_id == session_id,
        None => true,
    }
}

// --- Per-session dedup ---

/// Dedup file for a specific session (first 8 chars of UUID).
fn dedup_file_for(session_id: &str) -> PathBuf {
    let short_id = &session_id[..session_id.len().min(8)];
    history_dir().join(format!(".last-stop-{short_id}"))
}

/// Check if this Stop text was already spoken for this session.
fn is_duplicate_stop(session_id: &str, text: &str) -> bool {
    let path = dedup_file_for(session_id);
    let previous = fs::read_to_string(&path).unwrap_or_default();
    let _ = fs::write(&path, text);
    previous == text
}

/// Clean up dedup + last-prompt files older than 24 hours and the legacy
/// single dedup file.
fn cleanup_stale_files() {
    let dir = history_dir();

    // Remove legacy single dedup file
    let old_dedup = dir.join(".last-stop-text");
    if old_dedup.exists() {
        let _ = fs::remove_file(&old_dedup);
    }

    // Remove stale per-session dedup + prompt files
    let Ok(entries) = fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.starts_with(".last-stop-") && !name_str.starts_with(".last-prompt-") {
            continue;
        }
        if let Ok(meta) = entry.metadata() {
            if let Ok(modified) = meta.modified() {
                if modified.elapsed().unwrap_or_default() > Duration::from_secs(24 * 3600) {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
}

// --- Per-session prompt store ---

fn last_prompt_file_for(session_id: &str) -> PathBuf {
    let short_id = &session_id[..session_id.len().min(8)];
    history_dir().join(format!(".last-prompt-{short_id}"))
}

fn save_last_prompt(session_id: &str, prompt: &str) {
    if session_id.is_empty() || prompt.trim().is_empty() {
        return;
    }
    let _ = fs::create_dir_all(history_dir());
    let _ = fs::write(last_prompt_file_for(session_id), prompt);
}

fn read_last_prompt(session_id: &str) -> Option<String> {
    if session_id.is_empty() {
        return None;
    }
    fs::read_to_string(last_prompt_file_for(session_id)).ok()
}

/// Pull a short, speakable label out of a free-form user prompt. Strips
/// leading filler ("okay", "by the way", "so", ...) and returns the first
/// few content words capped at MAX_LABEL_LEN. Returns None if nothing
/// substantive remains. The label gets quoted into the TTS line so a
/// listener can tell which task finished when several finish back-to-back.
fn extract_short_label(prompt: &str) -> Option<String> {
    const MAX_LABEL_LEN: usize = 32;
    const FILLERS: &[&str] = &[
        "by the way,", "by the way",
        "first of all,", "first of all",
        "actually,", "actually",
        "please,", "please",
        "okay,", "okay", "ok,", "ok",
        "well,", "well",
        "hey,", "hey",
        "now,", "now",
        "so,", "so",
        "alright,", "alright",
    ];

    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut start_byte = 0usize;
    loop {
        let lower_rem = trimmed[start_byte..].to_ascii_lowercase();
        let mut advanced = false;
        for f in FILLERS {
            if lower_rem.starts_with(f) {
                start_byte += f.len();
                while let Some(ch) = trimmed[start_byte..].chars().next() {
                    if ch.is_whitespace() || ch == ',' {
                        start_byte += ch.len_utf8();
                    } else {
                        break;
                    }
                }
                advanced = true;
                break;
            }
        }
        if !advanced {
            break;
        }
    }

    let body = &trimmed[start_byte.min(trimmed.len())..];
    let words: Vec<&str> = body.split_whitespace().take(6).collect();
    if words.is_empty() {
        return None;
    }
    let mut label = words.join(" ");
    if label.len() > MAX_LABEL_LEN {
        let cut = label[..MAX_LABEL_LEN]
            .rfind(' ')
            .unwrap_or(MAX_LABEL_LEN);
        label.truncate(cut);
    }
    let trimmed_label = label.trim_end_matches(|c: char| !c.is_alphanumeric());
    if trimmed_label.is_empty() {
        None
    } else {
        Some(trimmed_label.to_string())
    }
}

/// Compose the spoken text for a Stop event using the per-session prompt
/// if available. Falls back to the prior fixed phrasing when no prompt is
/// known (fresh session, prompt file pruned, etc.).
fn task_done_text(session_id: &str, project: &str, is_focus: bool) -> String {
    let label = read_last_prompt(session_id)
        .as_deref()
        .and_then(extract_short_label);
    match (label, is_focus) {
        (Some(l), true) => format!("\"{l}\" task done."),
        (Some(l), false) => format!("\"{l}\" {project} done."),
        (None, true) => "Task done.".to_string(),
        (None, false) => format!("{project} done."),
    }
}

// --- Helpers ---

/// Extract project name from cwd (last path component).
fn project_name(event: &HookEvent) -> String {
    event
        .cwd
        .as_deref()
        .and_then(|p| std::path::Path::new(p).file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// True if this Claude Code session is a non-interactive background worker
/// (e.g. claude-mem observers) that should not generate any TTS or claim focus.
/// Currently muted: anything under ~/.claude-mem/.
fn is_muted_project(event: &HookEvent) -> bool {
    let Some(cwd) = event.cwd.as_deref() else {
        return false;
    };
    let Some(home) = dirs::home_dir() else {
        return false;
    };
    let muted_root = home.join(".claude-mem");
    std::path::Path::new(cwd).starts_with(&muted_root)
}

fn save_record(record: &HistoryRecord) {
    let dir = history_dir();
    let _ = fs::create_dir_all(&dir);

    let date = &record.timestamp[..10];
    let path = dir.join(format!("{date}.jsonl"));

    if let Ok(json) = serde_json::to_string(record) {
        if let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{json}");
        }
    }
}

fn now_timestamp() -> String {
    chrono::Local::now()
        .format("%Y-%m-%dT%H:%M:%S%.3f")
        .to_string()
}

// --- Main ---

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let t0 = Instant::now();

    // Clean up stale dedup files
    cleanup_stale_files();

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

    // Extract session context
    let session_id = event.session_id.clone().unwrap_or_default();
    let project = project_name(&event);
    let is_focus = is_focus_session(&session_id);

    // Skip non-interactive background sessions (claude-mem observers, etc.)
    // before doing anything else — they must not claim focus or trigger TTS.
    if is_muted_project(&event) {
        save_record(&HistoryRecord {
            timestamp: now_timestamp(),
            event: event_name,
            action: "skipped".into(),
            detail: Some(format!("muted project ({project})")),
            text: None,
            text_chars: None,
            duration_ms: u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
            tts_api_up: false,
            session_id: Some(session_id),
            cwd: event.cwd.clone(),
            project: Some(project),
            is_focus,
        });
        return;
    }

    // Build HTTP client with short timeouts
    let client = Client::builder()
        .connect_timeout(Duration::from_millis(300))
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap_or_else(|_| Client::new());

    // Quick connectivity check — exit cleanly if TTS API is down
    let tts_api_up = client.get(format!("{TTS_API}/status")).send().await.is_ok();

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
            session_id: Some(session_id),
            cwd: event.cwd.clone(),
            project: Some(project),
            is_focus,
        });
        return;
    }

    let (action, detail, text) = match event_name.as_str() {
        "SessionStart" => {
            handle_session_start(&client, &event, &session_id, &project, is_focus).await
        }
        "Stop" => handle_stop(&client, &event, &session_id, &project, is_focus).await,
        "PermissionRequest" => handle_permission(&client, &event, &session_id, &project).await,
        "Notification" => {
            handle_notification(&client, &event, &session_id, &project, is_focus).await
        }
        "UserPromptSubmit" => {
            handle_user_prompt_submit(&client, &event, &session_id, &project).await
        }
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
        session_id: Some(session_id),
        cwd: event.cwd.clone(),
        project: Some(project),
        is_focus,
    });
}

// --- Event handlers ---

async fn handle_session_start(
    client: &Client,
    event: &HookEvent,
    session_id: &str,
    project: &str,
    is_focus: bool,
) -> (String, Option<String>, Option<String>) {
    // Skip resume and compaction restarts
    if let Some(source) = &event.source {
        if source == "resume" || source == "compact" {
            return ("skipped".into(), Some(format!("source={source}")), None);
        }
    }

    // Only announce for focus session (or first session when no focus exists)
    if !is_focus {
        return (
            "skipped".into(),
            Some(format!("non-focus start ({project})")),
            None,
        );
    }

    // Check if model is loaded
    let Ok(resp) = client.get(format!("{TTS_API}/status")).send().await else {
        return ("skipped".into(), Some("status request failed".into()), None);
    };
    let Ok(status) = resp.json::<StatusResponse>().await else {
        return ("skipped".into(), Some("status parse error".into()), None);
    };

    if status.model_loaded != Some(true) {
        return ("skipped".into(), Some("model not loaded".into()), None);
    }

    let text = "Claude Code is ready.".to_string();
    let _ = client
        .post(format!("{TTS_API}/speak"))
        .json(&SpeakRequest {
            text: text.clone(),
            event_type: "session_start".into(),
            session_id: session_id.into(),
        })
        .send()
        .await;

    ("spoke".into(), None, Some(text))
}

async fn handle_stop(
    client: &Client,
    event: &HookEvent,
    session_id: &str,
    project: &str,
    is_focus: bool,
) -> (String, Option<String>, Option<String>) {
    let transcript_path = match &event.transcript_path {
        Some(p) if !p.is_empty() => p,
        _ => return ("skipped".into(), Some("no transcript path".into()), None),
    };

    let text = match extract_last_assistant_text(transcript_path) {
        Some(t) => t,
        None => {
            return (
                "skipped".into(),
                Some("no assistant text found".into()),
                None,
            )
        }
    };

    // Per-session dedup — Claude Code can fire multiple Stop events for one turn
    if is_duplicate_stop(session_id, &text) {
        return (
            "skipped".into(),
            Some("duplicate stop text".into()),
            Some(text),
        );
    }

    let short_text = task_done_text(session_id, project, is_focus);
    if is_focus {
        let _ = client
            .post(format!("{TTS_API}/speak"))
            .json(&SpeakRequest {
                text: short_text.clone(),
                event_type: "stop".into(),
                session_id: session_id.into(),
            })
            .send()
            .await;

        (
            "spoke".into(),
            Some(format!("focus ({project})")),
            Some(short_text),
        )
    } else {
        // Non-focus session: short announcement (queued, plays after current speech)
        let _ = client
            .post(format!("{TTS_API}/speak"))
            .json(&SpeakRequest {
                text: short_text.clone(),
                event_type: "background_stop".into(),
                session_id: session_id.into(),
            })
            .send()
            .await;

        (
            "queued_background".into(),
            Some(format!("non-focus ({project})")),
            Some(short_text),
        )
    }
}

async fn handle_permission(
    client: &Client,
    event: &HookEvent,
    session_id: &str,
    project: &str,
) -> (String, Option<String>, Option<String>) {
    let tool = event.tool_name.as_deref().unwrap_or("unknown tool");
    let text = format!("{project} needs permission.");

    let _ = client
        .post(format!("{TTS_API}/speak"))
        .json(&SpeakRequest {
            text: text.clone(),
            event_type: "permission".into(),
            session_id: session_id.into(),
        })
        .send()
        .await;

    (
        "spoke".into(),
        Some(format!("{project}/{tool}")),
        Some(text),
    )
}

async fn handle_notification(
    client: &Client,
    event: &HookEvent,
    session_id: &str,
    project: &str,
    is_focus: bool,
) -> (String, Option<String>, Option<String>) {
    // Skip non-focus notifications
    if !is_focus {
        return (
            "skipped".into(),
            Some(format!("non-focus notification ({project})")),
            None,
        );
    }

    let (text, event_type) = match event.notification_type.as_deref() {
        Some("idle_prompt") => {
            // Skip — Claude Code's idle_prompt fires false positives.
            return (
                "skipped".into(),
                Some("idle_prompt (redundant)".into()),
                None,
            );
        }
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
            event_type: event_type.into(),
            session_id: session_id.into(),
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
    event: &HookEvent,
    session_id: &str,
    project: &str,
) -> (String, Option<String>, Option<String>) {
    // Claim focus for this session
    if !session_id.is_empty() {
        write_focus(session_id, project);
    }

    // Save the prompt so the next Stop event can quote a short label from it.
    if let Some(prompt) = event.prompt.as_deref() {
        save_last_prompt(session_id, prompt);
    }

    // Notify TTS API: interrupt focus speech + re-queue deferred non-focus items
    let _ = client
        .post(format!("{TTS_API}/user-input"))
        .json(&UserInputRequest {
            session_id: session_id.into(),
        })
        .send()
        .await;

    ("user_input".into(), Some(format!("focus={project}")), None)
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
