//! HTTP API server for Code Speaker TTS.
//!
//! Provides the same endpoint contract as the Python code-speaker service
//! for backward compatibility with tts-hook.sh and MCP tools.
//! Runs on port 8767 (configurable) using axum.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::service::VoiceGate;

use super::history::{save_tts_record, TTSRecord};
use super::reminder::ReminderManager;
use super::summarizer::OllamaSummarizer;
use super::tts::KokoroTtsEngine;

#[derive(Clone)]
pub struct TtsApiState {
    pub tts: Arc<KokoroTtsEngine>,
    pub summarizer: Arc<OllamaSummarizer>,
    pub reminder: Arc<ReminderManager>,
    pub voice_gate: VoiceGate,
    pub max_direct_chars: usize,
}

// --- Request/Response types ---

#[derive(Deserialize)]
struct SpeakRequest {
    text: String,
    #[serde(default)]
    summarize: bool,
    #[serde(default = "default_event_type")]
    event_type: String,
    #[serde(default)]
    start_reminder: bool,
}

fn default_event_type() -> String {
    "unknown".to_string()
}

#[derive(Deserialize)]
struct SetVoiceRequest {
    voice: String,
}

#[derive(Serialize)]
struct StatusResponse {
    speaking: bool,
    voice: String,
    model_loaded: bool,
    reminder_active: bool,
    reminder_count: u32,
}

#[derive(Serialize)]
struct SimpleResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reminders_fired: Option<u32>,
}

/// Build the axum router.
pub fn router(state: TtsApiState) -> Router {
    Router::new()
        .route("/status", get(handle_status))
        .route("/speak", post(handle_speak))
        .route("/set-voice", post(handle_set_voice))
        .route("/cancel", post(handle_cancel))
        .route("/cancel-reminder", post(handle_cancel_reminder))
        .with_state(state)
}

/// Start the TTS API server as a background tokio task.
pub async fn start_tts_api(state: TtsApiState, port: u16) {
    let app = router(state);
    let addr = format!("127.0.0.1:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            warn!("Failed to bind TTS API on {addr}: {e}");
            return;
        }
    };
    info!("TTS API server listening on {addr}");

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            warn!("TTS API server error: {e}");
        }
    });
}

// --- Handlers ---

async fn handle_status(State(state): State<TtsApiState>) -> Json<StatusResponse> {
    Json(StatusResponse {
        speaking: state.tts.is_speaking(),
        voice: state.tts.current_voice(),
        model_loaded: state.tts.is_loaded(),
        reminder_active: state.reminder.is_active(),
        reminder_count: state.reminder.reminder_count(),
    })
}

async fn handle_speak(
    State(state): State<TtsApiState>,
    Json(req): Json<SpeakRequest>,
) -> Json<SimpleResponse> {
    if req.text.trim().is_empty() {
        return Json(SimpleResponse {
            status: "error".into(),
            error: Some("empty text".into()),
            voice: None,
            reminders_fired: None,
        });
    }

    let preview: String = req.text.chars().take(80).collect();
    info!(
        "HTTP /speak [{}]: \"{}{}\" ({} chars, summarize={})",
        req.event_type,
        preview.replace('\n', " "),
        if req.text.len() > 80 { "..." } else { "" },
        req.text.len(),
        req.summarize,
    );

    // Fire-and-forget: spawn the speak pipeline
    let tts = state.tts.clone();
    let summarizer = state.summarizer.clone();
    let reminder = state.reminder.clone();
    let voice_gate = state.voice_gate.clone();
    let max_direct_chars = state.max_direct_chars;

    tokio::spawn(async move {
        do_speak(
            tts,
            summarizer,
            reminder,
            voice_gate,
            max_direct_chars,
            req.text,
            req.summarize,
            req.event_type,
            req.start_reminder,
        )
        .await;
    });

    Json(SimpleResponse {
        status: "speaking".into(),
        voice: None,
        error: None,
        reminders_fired: None,
    })
}

async fn handle_set_voice(
    State(state): State<TtsApiState>,
    Json(req): Json<SetVoiceRequest>,
) -> Json<SimpleResponse> {
    if state.tts.set_voice(&req.voice) {
        Json(SimpleResponse {
            status: "ok".into(),
            voice: Some(req.voice),
            error: None,
            reminders_fired: None,
        })
    } else {
        Json(SimpleResponse {
            status: "error".into(),
            error: Some(format!("Unknown voice: {}", req.voice)),
            voice: None,
            reminders_fired: None,
        })
    }
}

async fn handle_cancel(State(state): State<TtsApiState>) -> Json<SimpleResponse> {
    state.tts.cancel();
    Json(SimpleResponse {
        status: "cancelled".into(),
        voice: None,
        error: None,
        reminders_fired: None,
    })
}

async fn handle_cancel_reminder(State(state): State<TtsApiState>) -> Json<SimpleResponse> {
    let count = state.reminder.cancel();
    state.tts.cancel();
    Json(SimpleResponse {
        status: "cancelled".into(),
        voice: None,
        error: None,
        reminders_fired: Some(count),
    })
}

/// Execute the speak pipeline: cancel previous → summarize → voice gate → speak → reminder.
async fn do_speak(
    tts: Arc<KokoroTtsEngine>,
    summarizer: Arc<OllamaSummarizer>,
    reminder: Arc<ReminderManager>,
    voice_gate: VoiceGate,
    max_direct_chars: usize,
    text: String,
    summarize: bool,
    event_type: String,
    start_reminder: bool,
) {
    let t_total = std::time::Instant::now();

    // Cancel any existing reminder
    reminder.cancel();

    // Cancel any in-flight speech
    tts.cancel();

    // Optionally summarize long text
    let mut spoken_text = text.clone();
    let mut ollama_ms = 0.0;
    let mut summarized = false;

    if summarize && text.len() > max_direct_chars {
        let (summary, ms) = summarizer.summarize(&text).await;
        spoken_text = summary;
        ollama_ms = ms;
        summarized = true;
    }

    // Wait for voice input to finish before playing
    if !voice_gate.is_voice_idle() {
        info!("TTS deferred — waiting for voice input to complete");
        match tokio::time::timeout(Duration::from_secs(60), voice_gate.wait_for_idle()).await {
            Ok(_) => info!("TTS resumed — voice input complete"),
            Err(_) => warn!("TTS gate timeout (60s) — speaking anyway"),
        }
    }

    // Speak
    let result = tts.speak(&spoken_text).await;

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    info!(
        "TTS complete [{}]: ollama={ollama_ms:.0}ms gen={:.0}ms play={:.0}ms total={total_ms:.0}ms",
        event_type, result.generate_ms, result.playback_ms,
    );

    // Start reminder if requested and not cancelled
    if start_reminder && !result.cancelled {
        reminder.start(spoken_text.clone(), tts.clone());
    }

    // Save history record
    let voice = tts.current_voice();
    let record = TTSRecord {
        timestamp: chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.6f").to_string(),
        event_type,
        input_text_chars: text.len(),
        summarized,
        summary_text: spoken_text,
        ollama_latency_ms: ollama_ms as i64,
        kokoro_latency_ms: result.generate_ms as i64,
        playback_duration_ms: result.playback_ms as i64,
        total_latency_ms: total_ms as i64,
        voice,
        cancelled: result.cancelled,
        reminder_count: reminder.reminder_count(),
    };
    save_tts_record(&record);
}
