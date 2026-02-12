//! History tracking and productivity reporting.
//!
//! Stores transcription records as daily JSONL files in ~/.whisper-typer-history/,
//! compatible with the Python whisper-typer history format.

use chrono::Local;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use tracing::{debug, error};

/// Directory for history JSONL files.
fn history_dir() -> PathBuf {
    dirs::home_dir()
        .expect("No home directory")
        .join(".whisper-typer-history")
}

/// Get the history file path for a given date.
fn history_file(date: &str) -> PathBuf {
    let date_str = if date == "today" {
        Local::now().format("%Y-%m-%d").to_string()
    } else {
        date.to_string()
    };
    history_dir().join(format!("{date_str}.jsonl"))
}

/// Record of a single transcription, matching the Python format exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionRecord {
    pub timestamp: String,
    pub whisper_text: String,
    pub ollama_text: Option<String>,
    pub final_text: String,
    pub output_mode: String,
    pub whisper_latency_ms: i64,
    pub ollama_latency_ms: Option<i64>,
    pub typing_latency_ms: i64,
    pub total_latency_ms: i64,
    pub audio_duration_s: f64,
    pub char_count: usize,
    pub word_count: usize,
    pub speed_ratio: f64,
}

/// Append a transcription record to the daily history file.
pub fn save_record(record: &TranscriptionRecord) {
    let dir = history_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        error!("Failed to create history dir: {e}");
        return;
    }

    let path = history_file("today");
    match fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut file) => {
            match serde_json::to_string(record) {
                Ok(json) => {
                    if let Err(e) = writeln!(file, "{json}") {
                        error!("Failed to write history record: {e}");
                    } else {
                        debug!("Saved transcription record to {}", path.display());
                    }
                }
                Err(e) => error!("Failed to serialize record: {e}"),
            }
        }
        Err(e) => error!("Failed to open history file: {e}"),
    }
}

/// Load all transcription records for a given date.
pub fn load_records(date: &str) -> Vec<TranscriptionRecord> {
    let path = history_file(date);
    if !path.exists() {
        return Vec::new();
    }

    let mut records = Vec::new();
    match fs::File::open(&path) {
        Ok(file) => {
            for line in std::io::BufReader::new(file).lines() {
                if let Ok(line) = line {
                    let line = line.trim().to_string();
                    if !line.is_empty() {
                        match serde_json::from_str::<TranscriptionRecord>(&line) {
                            Ok(record) => records.push(record),
                            Err(e) => debug!("Skipping malformed history line: {e}"),
                        }
                    }
                }
            }
        }
        Err(e) => error!("Failed to load history records: {e}"),
    }

    records
}

/// List all dates with history records (newest first).
pub fn list_available_dates() -> Vec<String> {
    let dir = history_dir();
    if !dir.exists() {
        return Vec::new();
    }

    let mut dates: Vec<String> = fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".jsonl") {
                Some(name.trim_end_matches(".jsonl").to_string())
            } else {
                None
            }
        })
        .collect();

    dates.sort_by(|a, b| b.cmp(a)); // newest first
    dates
}

fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else {
        let minutes = (seconds / 60.0) as u64;
        let secs = seconds % 60.0;
        if minutes < 60 {
            format!("{minutes}m {secs:.0}s")
        } else {
            let hours = minutes / 60;
            let mins = minutes % 60;
            format!("{hours}h {mins}m")
        }
    }
}

fn truncate(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len.saturating_sub(3)])
    }
}

/// Generate a Markdown productivity report for a given date.
pub fn generate_report(date: &str) -> String {
    let records = load_records(date);

    let display_date = if date == "today" {
        Local::now().format("%Y-%m-%d").to_string()
    } else {
        date.to_string()
    };

    if records.is_empty() {
        return format!(
            "# WhisperTyper Report - {display_date}\n\nNo transcriptions recorded."
        );
    }

    let total_chars: usize = records.iter().map(|r| r.char_count).sum();
    let total_words: usize = records.iter().map(|r| r.word_count).sum();
    let total_audio: f64 = records.iter().map(|r| r.audio_duration_s).sum();
    let total_processing: f64 =
        records.iter().map(|r| r.total_latency_ms as f64).sum::<f64>() / 1000.0;

    let whisper_latencies: Vec<i64> = records.iter().map(|r| r.whisper_latency_ms).collect();
    let ollama_latencies: Vec<i64> = records
        .iter()
        .filter_map(|r| r.ollama_latency_ms)
        .collect();
    let typing_latencies: Vec<i64> = records.iter().map(|r| r.typing_latency_ms).collect();

    let avg_whisper = if whisper_latencies.is_empty() {
        0.0
    } else {
        whisper_latencies.iter().sum::<i64>() as f64 / whisper_latencies.len() as f64
    };
    let avg_ollama = if ollama_latencies.is_empty() {
        0.0
    } else {
        ollama_latencies.iter().sum::<i64>() as f64 / ollama_latencies.len() as f64
    };
    let avg_typing = if typing_latencies.is_empty() {
        0.0
    } else {
        typing_latencies.iter().sum::<i64>() as f64 / typing_latencies.len() as f64
    };
    let avg_speed: f64 =
        records.iter().map(|r| r.speed_ratio).sum::<f64>() / records.len() as f64;

    let mut lines = vec![
        format!("# WhisperTyper Report - {display_date}"),
        String::new(),
        "## Summary".to_string(),
        format!("- **Transcriptions**: {}", records.len()),
        format!("- **Total characters**: {total_chars}"),
        format!("- **Total words**: {total_words}"),
        format!("- **Total audio**: {}", format_duration(total_audio)),
        format!(
            "- **Total processing time**: {}",
            format_duration(total_processing)
        ),
        format!("- **Average speed ratio**: {avg_speed:.1}x"),
        String::new(),
        "## Latency Averages".to_string(),
        format!("- Whisper: {avg_whisper:.0}ms"),
    ];

    if !ollama_latencies.is_empty() {
        lines.push(format!("- Ollama: {avg_ollama:.0}ms"));
    }
    lines.push(format!("- Typing: {avg_typing:.0}ms"));

    lines.extend([
        String::new(),
        "## Transcription Log".to_string(),
        String::new(),
        "| Time | Whisper | Ollama | Chars | Speed |".to_string(),
        "|------|---------|--------|-------|-------|".to_string(),
    ]);

    for r in &records {
        let time_str = if r.timestamp.len() >= 19 {
            // Extract HH:MM:SS from ISO 8601 timestamp
            &r.timestamp[11..19]
        } else {
            &r.timestamp[..8.min(r.timestamp.len())]
        };

        let whisper_display = truncate(&r.whisper_text, 30);
        let ollama_display = match &r.ollama_text {
            Some(text) if text != &r.whisper_text => truncate(text, 30),
            _ => "-".to_string(),
        };

        lines.push(format!(
            "| {time_str} | {whisper_display} | {ollama_display} | {} | {:.1}x |",
            r.char_count, r.speed_ratio
        ));
    }

    lines.join("\n")
}
