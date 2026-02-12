//! MCP Server for WhisperTyper control using rmcp.
//!
//! Exposes tools for Claude Code integration:
//! - whisper_set_mode, whisper_enable_ollama, whisper_disable_ollama
//! - whisper_get_status, whisper_get_recent, whisper_get_daily_report
//! - code_speaker_speak, code_speaker_set_voice, code_speaker_enable/disable
//! - code_speaker_voices, code_speaker_report

use std::collections::{HashMap, HashSet};
use std::fs;
use std::future::Future;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use rmcp::handler::server::tool::{Parameters, ToolRouter};
use rmcp::model::{CallToolResult, Content, ServerCapabilities, ServerInfo};
use rmcp::transport::sse_server::SseServerConfig;
use rmcp::transport::SseServer;
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler};
use serde::Deserialize;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::history;

/// State file path (shared with service.rs).
fn state_file() -> PathBuf {
    dirs::home_dir()
        .expect("No home directory")
        .join(".cache/whisper-typer/state.json")
}

fn read_state() -> serde_json::Value {
    let path = state_file();
    if path.exists() {
        if let Ok(contents) = fs::read_to_string(&path) {
            if let Ok(state) = serde_json::from_str(&contents) {
                return state;
            }
        }
    }
    json!({
        "output_mode": "ollama_only",
        "ollama_enabled": true,
        "recent_transcriptions": []
    })
}

fn write_state(state: &serde_json::Value) {
    let path = state_file();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(e) = fs::write(&path, serde_json::to_string_pretty(state).unwrap()) {
        warn!("Failed to write MCP state: {e}");
    }
}

fn update_state(updates: serde_json::Value) -> serde_json::Value {
    let mut state = read_state();
    if let (Some(state_obj), Some(updates_obj)) = (state.as_object_mut(), updates.as_object()) {
        for (k, v) in updates_obj {
            state_obj.insert(k.clone(), v.clone());
        }
    }
    write_state(&state);
    state
}

// --- Tool parameter structs ---

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct SetModeRequest {
    #[schemars(description = "Output mode - 'ollama' (corrected text only), 'whisper' (raw transcription only), 'both' (corrected + [raw] in brackets)")]
    pub mode: String,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct GetRecentRequest {
    #[schemars(description = "Number of recent transcriptions to return (default: 5)")]
    pub count: Option<usize>,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct GetDailyReportRequest {
    #[schemars(description = "Date to get report for - 'today' (default), 'list' (show available dates), or YYYY-MM-DD format")]
    pub date: Option<String>,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct SpeakRequest {
    #[schemars(description = "The text to speak aloud")]
    pub text: String,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct SetVoiceRequest {
    #[schemars(description = "Voice name (e.g., 'af_heart', 'bf_emma', 'am_adam')")]
    pub voice: String,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct ReportRequest {
    #[schemars(description = "Date for report - 'today' (default), 'list', or YYYY-MM-DD")]
    pub date: Option<String>,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct TeachRequest {
    #[schemars(description = "Vocabulary terms to teach Whisper (comma-separated, e.g., 'Ollama, Kokoro, ndarray')")]
    pub terms: String,
}

#[derive(Debug, Deserialize, rmcp::schemars::JsonSchema)]
pub struct AddCorrectionRequest {
    #[schemars(description = "The wrong/misrecognized text")]
    pub wrong: String,
    #[schemars(description = "The correct text")]
    pub right: String,
}

// --- MCP Server handler ---

#[derive(Clone)]
pub struct WhisperTyperMcp {
    tts_port: u16,
    http_client: reqwest::Client,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl WhisperTyperMcp {
    pub fn new(tts_port: u16) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            tts_port,
            http_client,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Set the output mode for WhisperTyper.\n\nArgs:\n    mode: Output mode - 'ollama' (corrected text only), 'whisper' (raw transcription only), 'both' (corrected + [raw] in brackets)")]
    async fn whisper_set_mode(
        &self,
        Parameters(req): Parameters<SetModeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mode_internal = match req.mode.as_str() {
            "whisper" => "whisper_only",
            "both" => "both",
            _ => "ollama_only",
        };
        update_state(json!({ "output_mode": mode_internal }));
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Output mode set to: {}",
            req.mode
        ))]))
    }

    #[tool(description = "Enable Ollama processing for grammar/spelling correction.")]
    async fn whisper_enable_ollama(&self) -> Result<CallToolResult, McpError> {
        update_state(json!({ "ollama_enabled": true, "output_mode": "ollama_only" }));
        Ok(CallToolResult::success(vec![Content::text(
            "Ollama enabled. Mode set to: ollama",
        )]))
    }

    #[tool(description = "Disable Ollama processing, use raw Whisper output only.")]
    async fn whisper_disable_ollama(&self) -> Result<CallToolResult, McpError> {
        update_state(json!({ "ollama_enabled": false, "output_mode": "whisper_only" }));
        Ok(CallToolResult::success(vec![Content::text(
            "Ollama disabled. Mode set to: whisper",
        )]))
    }

    #[tool(description = "Get current WhisperTyper status and configuration.")]
    async fn whisper_get_status(&self) -> Result<CallToolResult, McpError> {
        let state = read_state();
        let mode = state["output_mode"]
            .as_str()
            .unwrap_or("unknown")
            .replace('_', " ");
        let ollama = state["ollama_enabled"].as_bool().unwrap_or(false);
        let recent_count = state["recent_transcriptions"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);

        let status = format!(
            "WhisperTyper Status:\n- Output Mode: {}\n- Ollama Enabled: {}\n- Recent Transcriptions: {}",
            mode, ollama, recent_count
        );
        Ok(CallToolResult::success(vec![Content::text(status)]))
    }

    #[tool(description = "Get recent transcriptions.\n\nArgs:\n    count: Number of recent transcriptions to return (default: 5)")]
    async fn whisper_get_recent(
        &self,
        Parameters(req): Parameters<GetRecentRequest>,
    ) -> Result<CallToolResult, McpError> {
        let state = read_state();
        let count = req.count.unwrap_or(5);
        let recent: Vec<&str> = state["recent_transcriptions"]
            .as_array()
            .map(|a| {
                a.iter()
                    .rev()
                    .take(count)
                    .filter_map(|v| v.as_str())
                    .collect()
            })
            .unwrap_or_default();

        if recent.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No recent transcriptions.",
            )]));
        }

        let text = format!(
            "Recent transcriptions:\n{}",
            recent
                .iter()
                .map(|t| format!("- {t}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(description = "Get Markdown productivity report for a specific date.\n\nArgs:\n    date: Date to get report for - 'today' (default), 'list' (show available dates), or YYYY-MM-DD format")]
    async fn whisper_get_daily_report(
        &self,
        Parameters(req): Parameters<GetDailyReportRequest>,
    ) -> Result<CallToolResult, McpError> {
        let date = req.date.as_deref().unwrap_or("today");

        if date == "list" {
            let dates = history::list_available_dates();
            if dates.is_empty() {
                return Ok(CallToolResult::success(vec![Content::text(
                    "No history records found.",
                )]));
            }
            let text = format!(
                "Available dates:\n{}",
                dates.iter().map(|d| format!("- {d}")).collect::<Vec<_>>().join("\n")
            );
            return Ok(CallToolResult::success(vec![Content::text(text)]));
        }

        let report = history::generate_report(date);
        Ok(CallToolResult::success(vec![Content::text(report)]))
    }

    // --- Code Speaker TTS tools ---

    #[tool(description = "Speak text aloud using Kokoro TTS.\n\nArgs:\n    text: The text to speak aloud")]
    async fn code_speaker_speak(
        &self,
        Parameters(req): Parameters<SpeakRequest>,
    ) -> Result<CallToolResult, McpError> {
        let url = format!("http://127.0.0.1:{}/speak", self.tts_port);
        match self
            .http_client
            .post(&url)
            .json(&json!({
                "text": req.text,
                "summarize": false,
                "event_type": "manual"
            }))
            .send()
            .await
        {
            Ok(_) => {
                let preview = if req.text.len() > 80 {
                    format!("{}...", &req.text[..80])
                } else {
                    req.text
                };
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Speaking: {preview}"
                ))]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "TTS error: {e}"
            ))])),
        }
    }

    #[tool(description = "Set the TTS voice for code_speaker.\n\nArgs:\n    voice: Voice name (e.g., 'af_heart', 'bf_emma', 'am_adam')")]
    async fn code_speaker_set_voice(
        &self,
        Parameters(req): Parameters<SetVoiceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let url = format!("http://127.0.0.1:{}/set-voice", self.tts_port);
        match self
            .http_client
            .post(&url)
            .json(&json!({ "voice": req.voice }))
            .send()
            .await
        {
            Ok(resp) => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    if data["status"].as_str() == Some("ok") {
                        return Ok(CallToolResult::success(vec![Content::text(format!(
                            "Voice set to: {}",
                            req.voice
                        ))]));
                    }
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Error: {}",
                        data["error"].as_str().unwrap_or("unknown")
                    ))]));
                }
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Voice set to: {}",
                    req.voice
                ))]))
            }
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to set voice: {e}"
            ))])),
        }
    }

    #[tool(description = "Enable code_speaker TTS output.")]
    async fn code_speaker_enable(&self) -> Result<CallToolResult, McpError> {
        let url = format!("http://127.0.0.1:{}/enable", self.tts_port);
        match self.http_client.post(&url).send().await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(
                "Code Speaker TTS enabled",
            )])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to enable TTS: {e}"
            ))])),
        }
    }

    #[tool(description = "Disable code_speaker TTS output.")]
    async fn code_speaker_disable(&self) -> Result<CallToolResult, McpError> {
        let url = format!("http://127.0.0.1:{}/disable", self.tts_port);
        match self.http_client.post(&url).send().await {
            Ok(_) => Ok(CallToolResult::success(vec![Content::text(
                "Code Speaker TTS disabled",
            )])),
            Err(e) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to disable TTS: {e}"
            ))])),
        }
    }

    #[tool(description = "List available TTS voices for code_speaker.")]
    async fn code_speaker_voices(&self) -> Result<CallToolResult, McpError> {
        let text = "Available voice prefixes:\n\
            - af_* (American female): af_heart, af_bella, af_nova, af_sarah\n\
            - am_* (American male): am_adam, am_michael, am_echo\n\
            - bf_* (British female): bf_emma, bf_alice, bf_lily\n\
            - bm_* (British male): bm_george, bm_lewis\n\
            Use code_speaker_set_voice to change.";
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(description = "Get unified Voice I/O report (STT + TTS statistics).\n\nArgs:\n    date: Date for report - 'today' (default), 'list', or YYYY-MM-DD")]
    async fn code_speaker_report(
        &self,
        Parameters(req): Parameters<ReportRequest>,
    ) -> Result<CallToolResult, McpError> {
        let date = req.date.as_deref().unwrap_or("today");

        if date == "list" {
            let dates = history::list_available_dates();
            if dates.is_empty() {
                return Ok(CallToolResult::success(vec![Content::text(
                    "No history records found.",
                )]));
            }
            let text = format!(
                "Available dates:\n{}",
                dates.iter().map(|d| format!("- {d}")).collect::<Vec<_>>().join("\n")
            );
            return Ok(CallToolResult::success(vec![Content::text(text)]));
        }

        let report = history::generate_report(date);
        Ok(CallToolResult::success(vec![Content::text(report)]))
    }

    // --- Vocabulary & Corrections tools ---

    #[tool(description = "Teach WhisperTyper vocabulary terms for better speech recognition. Terms are added to .whisper/vocabulary.txt and used as Whisper initial prompt.\n\nArgs:\n    terms: Comma-separated vocabulary terms (e.g., 'Ollama, Kokoro, ndarray')")]
    async fn whisper_teach(
        &self,
        Parameters(req): Parameters<TeachRequest>,
    ) -> Result<CallToolResult, McpError> {
        let new_terms: Vec<String> = req
            .terms
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();

        if new_terms.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No terms provided.",
            )]));
        }

        let vocab_path = PathBuf::from(".whisper/vocabulary.txt");

        // Read existing terms
        let mut existing: HashSet<String> = if vocab_path.exists() {
            fs::read_to_string(&vocab_path)
                .unwrap_or_default()
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect()
        } else {
            HashSet::new()
        };

        // Add new terms (deduplicate)
        let mut added = Vec::new();
        for term in &new_terms {
            if existing.insert(term.clone()) {
                added.push(term.as_str());
            }
        }

        // Write back
        if let Some(parent) = vocab_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let mut sorted: Vec<&String> = existing.iter().collect();
        sorted.sort();
        let contents = sorted.iter().map(|t| t.as_str()).collect::<Vec<_>>().join("\n");
        if let Err(e) = fs::write(&vocab_path, format!("{contents}\n")) {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Failed to write vocabulary file: {e}"
            ))]));
        }

        // Signal service to reload
        update_state(json!({ "vocabulary_updated": true }));

        let msg = if added.is_empty() {
            format!("All {} terms already existed in vocabulary.", new_terms.len())
        } else {
            format!(
                "Added {} new term(s): {}. Total vocabulary: {} terms.",
                added.len(),
                added.join(", "),
                existing.len()
            )
        };
        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    #[tool(description = "Add a speech correction mapping. When Whisper misrecognizes a word, this teaches Ollama the correct replacement. Stored in .whisper/corrections.yaml.\n\nArgs:\n    wrong: The misrecognized text\n    right: The correct replacement")]
    async fn whisper_add_correction(
        &self,
        Parameters(req): Parameters<AddCorrectionRequest>,
    ) -> Result<CallToolResult, McpError> {
        if req.wrong.trim().is_empty() || req.right.trim().is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "Both 'wrong' and 'right' must be non-empty.",
            )]));
        }

        let corrections_path = PathBuf::from(".whisper/corrections.yaml");

        // Read existing corrections
        let mut corrections: HashMap<String, String> = if corrections_path.exists() {
            let contents = fs::read_to_string(&corrections_path).unwrap_or_default();
            serde_yml::from_str(&contents).unwrap_or_default()
        } else {
            HashMap::new()
        };

        let wrong = req.wrong.trim().to_string();
        let right = req.right.trim().to_string();
        let is_update = corrections.contains_key(&wrong);
        corrections.insert(wrong.clone(), right.clone());

        // Write back as YAML
        if let Some(parent) = corrections_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        match serde_yml::to_string(&corrections) {
            Ok(yaml) => {
                if let Err(e) = fs::write(&corrections_path, &yaml) {
                    return Ok(CallToolResult::success(vec![Content::text(format!(
                        "Failed to write corrections file: {e}"
                    ))]));
                }
            }
            Err(e) => {
                return Ok(CallToolResult::success(vec![Content::text(format!(
                    "Failed to serialize corrections: {e}"
                ))]));
            }
        }

        // Signal service to reload
        update_state(json!({ "corrections_updated": true }));

        let action = if is_update { "Updated" } else { "Added" };
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{action} correction: \"{wrong}\" → \"{right}\". Total corrections: {}.",
            corrections.len()
        ))]))
    }
}

#[tool_handler]
impl ServerHandler for WhisperTyperMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("WhisperTyper speech-to-text control interface. Use whisper_* tools to control dictation, code_speaker_* tools for TTS.".into()),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            ..Default::default()
        }
    }
}

/// Start the MCP SSE server on the given port (runs in background).
pub async fn start_mcp_server(port: u16, tts_port: u16) {
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();

    let config = SseServerConfig {
        bind: addr,
        sse_path: "/sse".to_string(),
        post_path: "/message".to_string(),
        ct: CancellationToken::new(),
        sse_keep_alive: Some(Duration::from_secs(15)),
    };

    match SseServer::serve_with_config(config).await {
        Ok(sse_server) => {
            info!("MCP SSE server listening on http://{addr}/sse");
            let tts_port_clone = tts_port;
            sse_server.with_service(move || WhisperTyperMcp::new(tts_port_clone));
        }
        Err(e) => {
            warn!("Failed to start MCP server on {addr}: {e}");
        }
    }
}
