//! whisper-typer-rs: Speech-to-text dictation service for Linux.

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

    // Initialize logging
    let filter = if args.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
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
    let mut service = service::DictationService::new(config, transcriber, output_mode);
    service.run().await?;

    Ok(())
}
