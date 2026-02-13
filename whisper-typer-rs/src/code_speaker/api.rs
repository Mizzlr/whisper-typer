//! HTTP API server for Code Speaker TTS.
//!
//! Speak requests are queued and processed sequentially — each one plays
//! to completion before the next starts. Cancel operations increment a
//! generation counter to skip stale queued jobs.
//!
//! Interrupted items are saved to a deferred list. When the user finishes
//! speaking (POST /user-input), non-focus deferred items are re-queued
//! so they're not silently lost.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::history::{save_tts_record, TTSRecord};
use super::reminder::ReminderManager;
use super::summarizer::OllamaSummarizer;
use super::tts::KokoroTtsEngine;

const MAX_DEFERRED: usize = 20;

/// Short session ID for logging (first 8 chars or "manual" if empty).
fn short_sid(sid: &str) -> &str {
    if sid.is_empty() {
        "manual"
    } else {
        &sid[..sid.len().min(8)]
    }
}

#[derive(Clone)]
pub struct TtsApiState {
    pub tts: Arc<KokoroTtsEngine>,
    pub summarizer: Arc<OllamaSummarizer>,
    pub reminder: Arc<ReminderManager>,
    pub enabled: Arc<AtomicBool>,
    pub queue_tx: mpsc::Sender<SpeakJob>,
    pub generation: Arc<AtomicU64>,
    pub deferred: Arc<Mutex<Vec<SpeakJob>>>,
}

/// A queued speak request with generation stamp for cancellation.
#[derive(Clone)]
pub struct SpeakJob {
    pub text: String,
    pub summarize: bool,
    pub event_type: String,
    pub start_reminder: bool,
    pub generation: u64,
    pub session_id: String,
    /// How many times this job has been deferred+re-queued. Max 1 retry.
    pub retries: u32,
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
    #[serde(default)]
    session_id: String,
}

fn default_event_type() -> String {
    "unknown".to_string()
}

#[derive(Deserialize)]
struct SetVoiceRequest {
    voice: String,
}

#[derive(Deserialize)]
struct UserInputRequest {
    #[serde(default)]
    session_id: String,
}

#[derive(Serialize)]
struct StatusResponse {
    enabled: bool,
    speaking: bool,
    voice: String,
    model_loaded: bool,
    reminder_active: bool,
    reminder_count: u32,
    queue_depth: usize,
    deferred_count: usize,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    requeued: Option<usize>,
}

impl SimpleResponse {
    fn ok(status: &str) -> Self {
        Self {
            status: status.into(),
            voice: None,
            error: None,
            reminders_fired: None,
            requeued: None,
        }
    }

    fn err(message: impl Into<String>) -> Self {
        Self {
            status: "error".into(),
            error: Some(message.into()),
            voice: None,
            reminders_fired: None,
            requeued: None,
        }
    }
}

/// Build the axum router.
pub fn router(state: TtsApiState) -> Router {
    Router::new()
        .route("/status", get(handle_status))
        .route("/speak", post(handle_speak))
        .route("/set-voice", post(handle_set_voice))
        .route("/cancel", post(handle_cancel))
        .route("/cancel-reminder", post(handle_cancel_reminder))
        .route("/user-input", post(handle_user_input))
        .route("/enable", post(handle_enable))
        .route("/disable", post(handle_disable))
        .with_state(state)
}

/// Start the TTS API server and queue consumer as background tokio tasks.
pub async fn start_tts_api(state: TtsApiState, port: u16, queue_rx: mpsc::Receiver<SpeakJob>) {
    spawn_queue_consumer(
        queue_rx,
        state.tts.clone(),
        state.summarizer.clone(),
        state.reminder.clone(),
        state.generation.clone(),
        state.deferred.clone(),
    );

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

/// Spawn a background task that processes speak jobs sequentially from the queue.
fn spawn_queue_consumer(
    mut rx: mpsc::Receiver<SpeakJob>,
    tts: Arc<KokoroTtsEngine>,
    summarizer: Arc<OllamaSummarizer>,
    reminder: Arc<ReminderManager>,
    generation: Arc<AtomicU64>,
    deferred: Arc<Mutex<Vec<SpeakJob>>>,
) {
    tokio::spawn(async move {
        while let Some(job) = rx.recv().await {
            let sid = short_sid(&job.session_id).to_string();
            let deferred_count = deferred.lock().unwrap().len();
            let current_gen = generation.load(Ordering::Relaxed);

            if job.generation != current_gen {
                let job_gen = job.generation;
                let retries = job.retries;
                // Only defer first-time items (retries == 0). Already-retried items are dropped.
                if retries == 0 {
                    let evt = job.event_type.clone();
                    let mut def = deferred.lock().unwrap();
                    if def.len() < MAX_DEFERRED {
                        def.push(job);
                    }
                    info!(
                        "Queue: DEFERRED stale [{evt}] sid={sid} (gen {job_gen} != {current_gen}) [deferred={}]",
                        def.len(),
                    );
                } else {
                    info!(
                        "Queue: DROPPED re-queued stale [{}] sid={sid} (retries={retries}) [deferred={deferred_count}]",
                        job.event_type,
                    );
                }
                continue;
            }

            let text_preview: String = job.text.chars().take(60).collect();
            let text_ellipsis = if job.text.len() > 60 { "..." } else { "" };
            info!(
                "Queue: NEXT [{}] sid={} retries={} \"{text_preview}{text_ellipsis}\" [deferred={}]",
                job.event_type, sid, job.retries, deferred_count,
            );

            // Clone before speaking in case we get cancelled mid-speech
            let backup = job.clone();
            let event_type = job.event_type.clone();

            let cancelled = do_speak(
                &tts,
                &summarizer,
                &reminder,
                job.text,
                job.summarize,
                job.event_type,
                job.start_reminder,
            )
            .await;

            if cancelled {
                // Only defer first-time cancelled items. Already-retried items are dropped.
                if backup.retries == 0 {
                    let mut def = deferred.lock().unwrap();
                    if def.len() < MAX_DEFERRED {
                        def.push(backup);
                    }
                    info!(
                        "Queue: DEFERRED cancelled [{}] sid={} [deferred={}]",
                        event_type, sid, def.len(),
                    );
                } else {
                    info!(
                        "Queue: DROPPED re-queued cancelled [{}] sid={} (retries={}) [deferred={}]",
                        event_type, sid, backup.retries, deferred_count,
                    );
                }
            } else {
                let new_deferred = deferred.lock().unwrap().len();
                info!(
                    "Queue: DONE [{}] sid={} [deferred={}]",
                    event_type, sid, new_deferred,
                );
            }
        }
    });
}

// --- Handlers ---

async fn handle_status(State(state): State<TtsApiState>) -> Json<StatusResponse> {
    Json(StatusResponse {
        enabled: state.enabled.load(Ordering::Relaxed),
        speaking: state.tts.is_speaking(),
        voice: state.tts.current_voice(),
        model_loaded: state.tts.is_loaded(),
        reminder_active: state.reminder.is_active(),
        reminder_count: state.reminder.reminder_count(),
        queue_depth: state.queue_tx.max_capacity() - state.queue_tx.capacity(),
        deferred_count: state.deferred.lock().unwrap().len(),
    })
}

async fn handle_speak(
    State(state): State<TtsApiState>,
    Json(req): Json<SpeakRequest>,
) -> Json<SimpleResponse> {
    if !state.enabled.load(Ordering::Relaxed) {
        return Json(SimpleResponse::ok("disabled"));
    }

    if req.text.trim().is_empty() {
        return Json(SimpleResponse::err("empty text"));
    }

    let preview: String = req.text.chars().take(80).collect();
    let sid = short_sid(&req.session_id);
    let deferred_count = state.deferred.lock().unwrap().len();
    let queue_depth = state.queue_tx.max_capacity() - state.queue_tx.capacity();

    info!(
        "HTTP /speak [{}] sid={}: \"{}{}\" ({} chars, summarize={}) [queue={}, deferred={}]",
        req.event_type,
        sid,
        preview.replace('\n', " "),
        if req.text.len() > 80 { "..." } else { "" },
        req.text.len(),
        req.summarize,
        queue_depth,
        deferred_count,
    );

    let job = SpeakJob {
        text: req.text,
        summarize: req.summarize,
        event_type: req.event_type,
        start_reminder: req.start_reminder,
        generation: state.generation.load(Ordering::Relaxed),
        session_id: req.session_id,
        retries: 0,
    };

    match state.queue_tx.try_send(job) {
        Ok(()) => Json(SimpleResponse::ok("queued")),
        Err(_) => {
            warn!("Speak queue full (capacity=20), dropping job");
            Json(SimpleResponse::err("queue full"))
        }
    }
}

async fn handle_set_voice(
    State(state): State<TtsApiState>,
    Json(req): Json<SetVoiceRequest>,
) -> Json<SimpleResponse> {
    if state.tts.set_voice(&req.voice) {
        Json(SimpleResponse {
            voice: Some(req.voice),
            ..SimpleResponse::ok("ok")
        })
    } else {
        Json(SimpleResponse::err(format!("Unknown voice: {}", req.voice)))
    }
}

/// Hard cancel: stop speech, skip queue, save interrupted items for re-queue.
/// Called by hotkey press (user starts recording).
async fn handle_cancel(State(state): State<TtsApiState>) -> Json<SimpleResponse> {
    let new_gen = state.generation.fetch_add(1, Ordering::Relaxed) + 1;
    let reminder_count = state.reminder.cancel();
    let was_speaking = state.tts.is_speaking();
    state.tts.cancel();
    let queue_depth = state.queue_tx.max_capacity() - state.queue_tx.capacity();
    let deferred_count = state.deferred.lock().unwrap().len();
    info!(
        "CANCEL: gen→{} was_speaking={} reminders_cancelled={} [queue={}, deferred={}]",
        new_gen, was_speaking, reminder_count, queue_depth, deferred_count,
    );
    Json(SimpleResponse::ok("cancelled"))
}

/// Cancel reminders only. Does not affect the queue or current speech.
async fn handle_cancel_reminder(State(state): State<TtsApiState>) -> Json<SimpleResponse> {
    let count = state.reminder.cancel();
    Json(SimpleResponse {
        reminders_fired: Some(count),
        ..SimpleResponse::ok("cancelled")
    })
}

/// User finished speaking/typing. Re-queue deferred non-focus items.
/// Called by UserPromptSubmit hook event.
async fn handle_user_input(
    State(state): State<TtsApiState>,
    Json(req): Json<UserInputRequest>,
) -> Json<SimpleResponse> {
    // Cancel reminders (user is actively engaging)
    state.reminder.cancel();

    // Re-queue deferred items that aren't from the user's current session
    let items: Vec<SpeakJob> = {
        let mut def = state.deferred.lock().unwrap();
        def.drain(..).collect()
    };

    let current_gen = state.generation.load(Ordering::Relaxed);
    let mut requeued = 0usize;
    let mut discarded = 0usize;

    for mut job in items {
        if job.session_id.is_empty() || job.session_id == req.session_id {
            // Focus session or manual (MCP) — user is there, they'll see the output
            discarded += 1;
            continue;
        }
        // Re-queue with current generation and incremented retry count
        job.generation = current_gen;
        job.retries += 1;
        if state.queue_tx.try_send(job).is_ok() {
            requeued += 1;
        }
    }

    let queue_depth = state.queue_tx.max_capacity() - state.queue_tx.capacity();
    info!(
        "USER-INPUT sid={}: requeued={} discarded={} [queue={}, deferred=0]",
        short_sid(&req.session_id),
        requeued,
        discarded,
        queue_depth,
    );

    Json(SimpleResponse {
        requeued: Some(requeued),
        ..SimpleResponse::ok("ok")
    })
}

async fn handle_enable(State(state): State<TtsApiState>) -> Json<SimpleResponse> {
    state.enabled.store(true, Ordering::Relaxed);
    info!("TTS enabled");
    Json(SimpleResponse::ok("enabled"))
}

async fn handle_disable(State(state): State<TtsApiState>) -> Json<SimpleResponse> {
    state.enabled.store(false, Ordering::Relaxed);
    state.generation.fetch_add(1, Ordering::Relaxed);
    state.tts.cancel();
    state.reminder.cancel();
    // Clear deferred on disable — user wants silence
    state.deferred.lock().unwrap().clear();
    info!("TTS disabled (do-not-disturb)");
    Json(SimpleResponse::ok("disabled"))
}

/// Execute the speak pipeline: cancel reminder → interrupt stale speech → summarize → speak → reminder.
/// Returns true if the speech was cancelled mid-playback.
async fn do_speak(
    tts: &Arc<KokoroTtsEngine>,
    summarizer: &Arc<OllamaSummarizer>,
    reminder: &Arc<ReminderManager>,
    text: String,
    summarize: bool,
    event_type: String,
    start_reminder: bool,
) -> bool {
    let t_total = std::time::Instant::now();

    // Cancel any existing reminder
    reminder.cancel();

    // Interrupt any in-flight speech (e.g., from a reminder that fired between queue items)
    tts.interrupt();

    // Clear stale cancel flag — generation counter handles queue-level cancellation.
    // Without this, a previous cancel() (hotkey when nothing was playing) poisons
    // the next speak with an immediate bail.
    tts.clear_cancel();

    // Summarize if requested — trust the caller's judgement
    let (spoken_text, ollama_ms, summarized) =
        if summarize {
            let (summary, ms) = summarizer.summarize(&text).await;
            (summary, ms, true)
        } else {
            (text.clone(), 0.0, false)
        };

    info!("Queue: PLAYING [{event_type}] ({} chars{})", spoken_text.len(), if summarized { ", summarized" } else { "" });

    // Speak (voice gate wait happens inside tts.speak())
    let result = tts.speak(&spoken_text).await;

    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
    info!(
        "TTS complete [{}]: ollama={ollama_ms:.0}ms gen={:.0}ms play={:.0}ms total={total_ms:.0}ms cancelled={}",
        event_type, result.generate_ms, result.playback_ms, result.cancelled,
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

    result.cancelled
}
