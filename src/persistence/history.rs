//! History tracking and productivity reporting.
//!
//! Stores transcription records as daily JSONL files in ~/.whisper-typer-history/,
//! compatible with the Python whisper-typer history format.

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufWriter, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{debug, error, warn};

#[derive(Debug, Clone)]
pub struct AudioArtifact {
    pub path: PathBuf,
    pub bytes: u64,
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_accepted: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ollama_load_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ollama_prompt_eval_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ollama_eval_ms: Option<i64>,
    pub typing_latency_ms: i64,
    pub total_latency_ms: i64,
    pub audio_duration_s: f64,
    pub char_count: usize,
    pub word_count: usize,
    pub speed_ratio: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_sample_rate_hz: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_samples: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_bytes: Option<u64>,
}

/// Save the exact non-silent microphone buffer used by Whisper as a private,
/// lossless 16-bit mono WAV. Audio failures never block transcription.
pub fn save_audio(
    samples: &[f32],
    sample_rate: u32,
    recorded_at: &DateTime<Local>,
    retention_days: u64,
) -> Result<AudioArtifact, String> {
    let audio_root = history_dir().join("audio");
    let day_dir = audio_root.join(recorded_at.format("%Y-%m-%d").to_string());
    create_private_dir(&audio_root)?;
    prune_expired_audio(&audio_root, retention_days);
    // Pruning removes empty date directories, so create today's directory only
    // after pruning has completed.
    create_private_dir(&day_dir)?;

    let stem = recorded_at.format("%Y%m%dT%H%M%S%.6f").to_string();
    let final_path = day_dir.join(format!("{stem}.wav"));
    let temporary_path = day_dir.join(format!(".{stem}.wav.part"));

    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temporary_path)
        .map_err(|e| format!("create {}: {e}", temporary_path.display()))?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(BufWriter::new(file), spec)
        .map_err(|e| format!("open WAV writer: {e}"))?;

    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        if let Err(e) = writer.write_sample(pcm) {
            drop(writer);
            let _ = fs::remove_file(&temporary_path);
            return Err(format!("write WAV sample: {e}"));
        }
    }
    if let Err(e) = writer.finalize() {
        let _ = fs::remove_file(&temporary_path);
        return Err(format!("finalize WAV: {e}"));
    }
    fs::rename(&temporary_path, &final_path)
        .map_err(|e| format!("publish {}: {e}", final_path.display()))?;
    let _ = fs::set_permissions(&final_path, fs::Permissions::from_mode(0o600));
    let bytes = fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);

    Ok(AudioArtifact {
        path: final_path,
        bytes,
    })
}

fn create_private_dir(path: &Path) -> Result<(), String> {
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
        .map_err(|e| format!("create private directory {}: {e}", path.display()))?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|e| format!("secure directory {}: {e}", path.display()))
}

fn prune_expired_audio(audio_root: &Path, retention_days: u64) {
    if retention_days == 0 {
        return;
    }
    let ttl = Duration::from_secs(retention_days.saturating_mul(86_400));
    let Ok(day_dirs) = fs::read_dir(audio_root) else {
        return;
    };

    for day_dir in day_dirs.flatten().filter(|entry| entry.path().is_dir()) {
        let path = day_dir.path();
        let Ok(files) = fs::read_dir(&path) else {
            continue;
        };
        for file in files.flatten() {
            let file_path = file.path();
            let is_capture = file_path.extension().is_some_and(|ext| ext == "wav")
                || file_path.extension().is_some_and(|ext| ext == "part");
            if !is_capture {
                continue;
            }
            let expired = file
                .metadata()
                .and_then(|metadata| metadata.modified())
                .and_then(|modified| modified.elapsed().map_err(std::io::Error::other))
                .is_ok_and(|age| age > ttl);
            if expired {
                if let Err(e) = fs::remove_file(&file_path) {
                    warn!("Failed to prune expired audio {}: {e}", file_path.display());
                }
            }
        }
        let _ = fs::remove_dir(path);
    }
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
        Ok(mut file) => match serde_json::to_string(record) {
            Ok(json) => {
                if let Err(e) = writeln!(file, "{json}") {
                    error!("Failed to write history record: {e}");
                } else {
                    debug!("Saved transcription record to {}", path.display());
                }
            }
            Err(e) => error!("Failed to serialize record: {e}"),
        },
        Err(e) => error!("Failed to open history file: {e}"),
    }
}

/// Load all transcription records for a given date.
pub fn load_records(date: &str) -> Vec<TranscriptionRecord> {
    let path = history_file(date);
    if !path.exists() {
        return Vec::new();
    }

    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            error!("Failed to load history records: {e}");
            return Vec::new();
        }
    };

    std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(
            |line| match serde_json::from_str::<TranscriptionRecord>(line.trim()) {
                Ok(record) => Some(record),
                Err(e) => {
                    debug!("Skipping malformed history line: {e}");
                    None
                }
            },
        )
        .collect()
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
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.strip_suffix(".jsonl").map(str::to_owned)
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
        return format!("# WhisperTyper Report - {display_date}\n\nNo transcriptions recorded.");
    }

    let total_chars: usize = records.iter().map(|r| r.char_count).sum();
    let total_words: usize = records.iter().map(|r| r.word_count).sum();
    let total_audio: f64 = records.iter().map(|r| r.audio_duration_s).sum();
    let total_processing: f64 = records
        .iter()
        .map(|r| r.total_latency_ms as f64)
        .sum::<f64>()
        / 1000.0;

    let avg = |xs: &[i64]| -> f64 {
        if xs.is_empty() {
            0.0
        } else {
            xs.iter().sum::<i64>() as f64 / xs.len() as f64
        }
    };

    let whisper_latencies: Vec<i64> = records.iter().map(|r| r.whisper_latency_ms).collect();
    let ollama_latencies: Vec<i64> = records.iter().filter_map(|r| r.ollama_latency_ms).collect();
    let typing_latencies: Vec<i64> = records.iter().map(|r| r.typing_latency_ms).collect();

    let avg_whisper = avg(&whisper_latencies);
    let avg_ollama = avg(&ollama_latencies);
    let avg_typing = avg(&typing_latencies);
    let avg_speed: f64 = records.iter().map(|r| r.speed_ratio).sum::<f64>() / records.len() as f64;

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
