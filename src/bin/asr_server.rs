//! Dedicated, permanently warm Whisper HTTP service for remote inference.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use serde::Serialize;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use whisper_typer_rs::code_speaker::api::decode_wav_16k_mono;
use whisper_typer_rs::config::Config;
use whisper_typer_rs::transcriber::WhisperTranscriber;

#[derive(Debug, Parser)]
#[command(
    name = "whisper-asr-server",
    about = "Permanently warm Whisper HTTP inference service"
)]
struct Args {
    /// Whisper Typer YAML configuration containing the model path.
    #[arg(short, long)]
    config: PathBuf,

    /// Address exposed by the HTTP server. Prefer a private LAN address.
    #[arg(long, default_value = "127.0.0.1:8768")]
    listen: SocketAddr,
}

#[derive(Clone)]
struct AppState {
    transcriber: Arc<WhisperTranscriber>,
    model: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    model: String,
}

#[derive(Serialize)]
struct SegmentResponse {
    start_s: f64,
    end_s: f64,
    text: String,
}

#[derive(Serialize)]
struct TranscribeResponse {
    text: String,
    duration_s: f64,
    latency_ms: f64,
    segments: Vec<SegmentResponse>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,ort=warn"))
        .init();
    let args = Args::parse();
    let config = Config::load(Some(&args.config))?;

    info!(
        "Loading permanently resident model {}",
        config.whisper.model
    );
    let whisper_config = config.whisper.clone();
    let transcriber =
        tokio::task::spawn_blocking(move || WhisperTranscriber::load(&whisper_config)).await??;
    let state = AppState {
        transcriber: Arc::new(transcriber),
        model: config.whisper.model,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/transcribe", post(transcribe))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    info!("Whisper ASR ready on http://{}", args.listen);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ready",
        model: state.model,
    })
}

async fn transcribe(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    let mut wav_bytes = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("audio") {
            match field.bytes().await {
                Ok(bytes) => wav_bytes = Some(bytes.to_vec()),
                Err(error) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        format!("failed to read audio upload: {error}"),
                    )
                        .into_response();
                }
            }
            break;
        }
    }

    let Some(wav_bytes) = wav_bytes else {
        return (StatusCode::BAD_REQUEST, "missing multipart field 'audio'").into_response();
    };
    let samples = match decode_wav_16k_mono(&wav_bytes) {
        Ok(samples) => samples,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, format!("invalid WAV: {error}")).into_response();
        }
    };

    let duration_s = samples.len() as f64 / 16_000.0;
    let transcriber = state.transcriber.clone();
    let started = Instant::now();
    let result =
        tokio::task::spawn_blocking(move || transcriber.transcribe_segments(&samples)).await;
    let segments = match result {
        Ok(Ok(segments)) => segments,
        Ok(Err(error)) => {
            warn!("Whisper inference failed: {error}");
            return (StatusCode::INTERNAL_SERVER_ERROR, error).into_response();
        }
        Err(error) => {
            warn!("Whisper worker failed: {error}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Whisper worker failed: {error}"),
            )
                .into_response();
        }
    };
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    let text = segments
        .iter()
        .map(|segment| segment.text.trim())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let segments = segments
        .into_iter()
        .map(|segment| SegmentResponse {
            start_s: segment.start_s,
            end_s: segment.end_s,
            text: segment.text,
        })
        .collect();

    Json(TranscribeResponse {
        text,
        duration_s,
        latency_ms,
        segments,
    })
    .into_response()
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
