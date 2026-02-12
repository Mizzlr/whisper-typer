//! whisper-typer-rs: Speech-to-text dictation service for Linux.

mod code_speaker;
mod config;
mod history;
mod hotkey;
mod mcp_server;
mod notifier;
mod processor;
mod recorder;
mod service;
mod transcriber;
mod typer;

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "whisper-typer-rs", about = "Speech-to-text dictation service")]
struct Args {
    /// Path to config.yaml
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Output mode: ollama, whisper, or both
    #[arg(short, long, default_value = "ollama")]
    mode: String,

    /// Disable Ollama processing (same as --mode whisper)
    #[arg(long)]
    no_ollama: bool,

    /// Enable verbose (debug) logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging (suppress noisy ort/rmcp internals)
    let filter = if args.verbose {
        EnvFilter::new("debug,ort=info,rmcp=info")
    } else {
        EnvFilter::new("info,ort=warn,rmcp=warn")
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    info!("whisper-typer-rs starting");

    // Load config
    let config = config::Config::load(args.config.as_deref());
    info!("Config loaded: {:?}", config.hotkey);

    // Determine output mode
    let output_mode = if args.no_ollama {
        service::OutputMode::Whisper
    } else {
        service::OutputMode::from_str(&args.mode)
    };
    info!("Output mode: {:?}", output_mode);

    // Load Whisper model (blocking, takes a few seconds)
    info!("Loading Whisper model...");
    let transcriber = tokio::task::spawn_blocking({
        let whisper_config = config.whisper.clone();
        move || transcriber::WhisperTranscriber::load(&whisper_config)
    })
    .await??;

    // Start MCP server (background task)
    if config.mcp.enabled {
        let mcp_port = config.mcp.port;
        let tts_port = config.tts.api_port;
        mcp_server::start_mcp_server(mcp_port, tts_port).await;
    }

    // Run the service
    let mut service = service::DictationService::new(config.clone(), transcriber, output_mode);

    // Start native TTS server (replaces Python code-speaker.service)
    if config.tts.enabled {
        let voice_gate = service.voice_gate();

        info!("Loading Kokoro TTS model...");
        let mut tts_engine = code_speaker::tts::KokoroTtsEngine::new(&config.tts);
        // Connect TTS to voice gate so it waits during recording, stops on cancel
        tts_engine.set_voice_gate(voice_gate.is_idle.clone(), voice_gate.idle_notify.clone());
        match tts_engine.load_model_sync() {
            Ok(()) => {
                let tts = Arc::new(tts_engine);
                let summarizer = Arc::new(code_speaker::summarizer::OllamaSummarizer::new(
                    &config.ollama.model,
                    &config.ollama.host,
                ));
                let reminder = Arc::new(code_speaker::reminder::ReminderManager::new(
                    config.tts.reminder_interval,
                ));

                let api_state = code_speaker::api::TtsApiState {
                    tts,
                    summarizer,
                    reminder,
                    max_direct_chars: config.tts.max_direct_chars,
                };
                code_speaker::api::start_tts_api(api_state, config.tts.api_port).await;
                info!(
                    "Native TTS server started on port {} (voice: {}, speed: {})",
                    config.tts.api_port, config.tts.voice, config.tts.speed
                );
            }
            Err(e) => {
                tracing::warn!("Failed to load TTS model: {e}");
                info!("TTS disabled — continuing without voice output");
            }
        }
    }

    service.run().await?;

    Ok(())
}
