//! whisper-typer-rs: Speech-to-text dictation service for Linux.

mod config;
mod hotkey;
mod recorder;
mod service;

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

    // Run the service
    let mut service = service::DictationService::new(config);
    service.run().await?;

    Ok(())
}
