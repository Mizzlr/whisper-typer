//! whisper-typer-rs: Speech-to-text dictation service for Linux.

use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;
use whisper_typer_rs::{code_speaker, config, mcp_server, runtime_settings, service, transcriber};

#[derive(Parser, Debug)]
#[command(name = "whisper-typer-rs", about = "Speech-to-text dictation service")]
struct Args {
    /// Path to config.yaml
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Output mode: ollama, whisper, or both
    #[arg(short, long)]
    mode: Option<String>,

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
    let config = config::Config::load(args.config.as_deref())?;
    info!("Config loaded: {:?}", config.hotkey);

    // Determine output mode
    let explicit_mode = args
        .mode
        .as_deref()
        .map(|mode| {
            runtime_settings::OutputMode::parse(mode).ok_or_else(|| {
                format!("invalid output mode {mode:?}; expected ollama, whisper, or both")
            })
        })
        .transpose()?;
    let output_mode = if args.no_ollama || !config.ollama.enabled {
        runtime_settings::OutputMode::Whisper
    } else {
        explicit_mode.unwrap_or(runtime_settings::OutputMode::Ollama)
    };
    info!("Output mode: {:?}", output_mode);

    let runtime_settings = Arc::new(runtime_settings::RuntimeSettings::load(
        output_mode,
        config.ollama.enabled && !args.no_ollama,
        explicit_mode.is_none() && !args.no_ollama && config.ollama.enabled,
    ));

    // Load Whisper model and pre-warm CUDA state.
    // Pre-warming forces whisper_init_state() to allocate GPU buffers NOW,
    // not on the first hotkey press (which would cause a 10+ minute CPU storm
    // that starves the evdev hotkey monitor).
    info!("Loading Whisper model...");
    let transcriber = tokio::task::spawn_blocking({
        let whisper_config = config.whisper.clone();
        move || {
            let t = transcriber::WhisperTranscriber::load(&whisper_config)?;
            t.warm_up()?;
            Ok::<_, String>(t)
        }
    })
    .await??;

    // Run the service. Transcriber is Arc-backed + Clone, so cloning here
    // keeps a handle for the /transcribe HTTP endpoint while the original
    // moves into DictationService.
    let transcriber_for_http = transcriber.clone();
    let mut service =
        service::DictationService::new(config.clone(), transcriber, runtime_settings.clone());

    // Start MCP server (background task)
    if config.mcp.enabled {
        let mcp_port = config.mcp.port;
        let tts_port = config.tts.api_port;
        mcp_server::start_mcp_server(mcp_port, tts_port, runtime_settings).await;
    }

    // Start native TTS server (replaces Python code-speaker.service)
    if config.tts.enabled {
        let voice_gate = service.voice_gate();

        info!("Loading Kokoro TTS model...");
        let mut tts_engine = code_speaker::tts::KokoroTtsEngine::new(&config.tts);
        // Connect TTS to voice gate so it waits during recording, stops on cancel
        tts_engine.set_voice_gate(voice_gate.is_idle.clone(), voice_gate.idle_notify.clone());
        match tts_engine.load_model_sync() {
            Ok(()) => {
                // Apply persisted voice override (survives restarts)
                if let Some(voice) = code_speaker::api::load_persisted_voice() {
                    if tts_engine.set_voice(&voice) {
                        info!("Restored persisted voice: {voice}");
                    }
                }

                let tts = Arc::new(tts_engine);

                let (queue_tx, queue_rx) = tokio::sync::mpsc::channel(20);
                let api_state = code_speaker::api::TtsApiState {
                    tts,
                    enabled: Arc::new(std::sync::atomic::AtomicBool::new(true)),
                    queue_tx,
                    generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
                    discard_before: Arc::new(std::sync::atomic::AtomicU64::new(0)),
                    deferred: Arc::new(std::sync::Mutex::new(Vec::new())),
                    transcriber: transcriber_for_http.clone(),
                };
                code_speaker::api::start_tts_api(api_state, config.tts.api_port, queue_rx).await;
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
