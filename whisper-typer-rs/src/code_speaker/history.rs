//! TTS event history and reporting.
//!
//! Stores TTS records in JSONL files at ~/.code-speaker-history/{date}-tts.jsonl.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use chrono::Local;
use serde::{Deserialize, Serialize};
use tracing::warn;

fn history_dir() -> PathBuf {
    dirs::home_dir()
        .expect("No home directory")
        .join(".code-speaker-history")
}

fn history_file(date: &str) -> PathBuf {
    history_dir().join(format!("{date}-tts.jsonl"))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TTSRecord {
    pub timestamp: String,
    pub event_type: String,
    pub input_text_chars: usize,
    pub summarized: bool,
    pub summary_text: String,
    pub ollama_latency_ms: i64,
    pub kokoro_latency_ms: i64,
    pub playback_duration_ms: i64,
    pub total_latency_ms: i64,
    pub voice: String,
    pub cancelled: bool,
    pub reminder_count: u32,
}

pub fn save_tts_record(record: &TTSRecord) {
    let dir = history_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        warn!("Failed to create TTS history dir: {e}");
        return;
    }

    let date = Local::now().format("%Y-%m-%d").to_string();
    let path = history_file(&date);

    let mut file = match fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => f,
        Err(e) => {
            warn!("Failed to open TTS history file: {e}");
            return;
        }
    };

    match serde_json::to_string(record) {
        Ok(line) => {
            if let Err(e) = writeln!(file, "{line}") {
                warn!("Failed to write TTS history record: {e}");
            }
        }
        Err(e) => warn!("Failed to serialize TTS record: {e}"),
    }
}

pub fn load_tts_records(date: &str) -> Vec<TTSRecord> {
    let path = history_file(date);
    let contents = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    contents
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

pub fn list_tts_dates() -> Vec<String> {
    let dir = history_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut dates: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with("-tts.jsonl") {
                Some(name.trim_end_matches("-tts.jsonl").to_string())
            } else {
                None
            }
        })
        .collect();
    dates.sort();
    dates
}

pub fn generate_tts_report(date: &str) -> String {
    let records = load_tts_records(date);
    if records.is_empty() {
        return format!("No TTS records for {date}.");
    }

    let total = records.len();
    let cancelled = records.iter().filter(|r| r.cancelled).count();
    let summarized = records.iter().filter(|r| r.summarized).count();

    let avg_kokoro: f64 =
        records.iter().map(|r| r.kokoro_latency_ms as f64).sum::<f64>() / total as f64;
    let avg_total: f64 =
        records.iter().map(|r| r.total_latency_ms as f64).sum::<f64>() / total as f64;

    // Event type breakdown
    let mut event_counts = std::collections::HashMap::new();
    for r in &records {
        *event_counts.entry(r.event_type.as_str()).or_insert(0) += 1;
    }

    let mut report = format!(
        "# TTS Report for {date}\n\n\
        - Total events: {total}\n\
        - Cancelled: {cancelled}\n\
        - Summarized: {summarized}\n\
        - Avg Kokoro latency: {avg_kokoro:.0}ms\n\
        - Avg total latency: {avg_total:.0}ms\n\n\
        ## Event Types\n"
    );

    for (event, count) in &event_counts {
        report.push_str(&format!("- {event}: {count}\n"));
    }

    report
}
