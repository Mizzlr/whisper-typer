use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Local;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, StreamConfig};
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use hound::{SampleFormat, WavSpec, WavWriter};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use regex::Regex;
use serde::Deserialize;
use serde_json::json;

#[path = "../vad.rs"]
mod vad;
use vad::SileroVad;

const TRANSCRIBE_URL: &str = "http://127.0.0.1:8767/transcribe";
const SAMPLE_RATE: u32 = 16_000;
const HALLUCINATION_CONFIG_NAME: &str = "hallucinations.txt";
const VAD_RMS_THRESHOLD: f32 = 0.012;
const SILERO_VAD_THRESHOLD: f32 = 0.5;
const SILERO_VAD_MODEL_REL_PATH: &str = "models/silero_vad.onnx";
const SILENCE_FINALIZE_MS: u32 = 900;
const MIN_UTTERANCE_MS: u32 = 500;
const MAX_UTTERANCE_MS: u32 = 25_000;
const PRE_ROLL_MS: u32 = 200;

/// Voice-activity-detection backend. RMS is the cheap default; Silero VAD
/// is opt-in via WHISPER_VOICE_JOURNAL_VAD=silero.
#[derive(Clone)]
enum VadMode {
    Rms,
    Silero(Arc<SileroVad>),
}

// Ollama-based hallucination filter (second pass after regex).
// Ablation on journal_2026-04-30.md: 38 additional drops regex missed
// (podcast bleed, novel phoneme clusters), 0.39s avg latency on gemma4:e2b.
const OLLAMA_HOST: &str = "http://127.0.0.1:11434";
const OLLAMA_MODEL: &str = "gemma4:e2b";
const OLLAMA_TIMEOUT_SEC: u64 = 8;
const OLLAMA_KEEP_ALIVE_SEC: u64 = 3600;

const LLM_SYSTEM_PROMPT: &str = concat!(
    "You filter Whisper voice-transcription hallucinations for a Solana MEV ",
    "engineer's voice journal. For each input, return ONLY the cleaned text ",
    "(deduplicate echoed phrases) OR the literal token [HALLUCINATION] if ",
    "the input is entirely noise.\n\n",
    "Domain terms to preserve verbatim: Astralane Quant, ClickHouse, Pingora, ",
    "Hermes, Dagster, backrun, arb, slot, sig, bundle, Helius, Jito, Solscan, ",
    "API key, searcher, ES slot, Sujith, Telegram, Slack, recipe, chatter, ",
    "send bundle, MEV, validator, Rust, Python.\n\n",
    "ALWAYS treat as [HALLUCINATION] if the input contains any of these ",
    "phoneme clusters (they are never real speech in this domain): ",
    "tath done, taff done, taston, tastom, photone, astronom, ackoned, ",
    "quote down, quantone, quintan, govety, salere, sarkham, pooley, ",
    "kata'i, mawr, edgilly, edgerley, brahmin, clipper ship, alan forbes, ",
    "state street deposit, harvard graduate, massachusetts investors trust, ",
    "andrew jackson, federal reserve.\n\n",
    "ALSO drop: long passages about 1920s American banking, finance history, ",
    "or stock market crashes (those are background podcast capture, not ",
    "user speech). Drop fragments under 4 words that are pure filler.\n\n",
    "No commentary, no explanation, no quotes around the output."
);

// Few-shot pairs anchoring the dedupe-vs-drop distinction. Picked from the
// gemma4:e2b ablation as the cases the model most often gets wrong without
// explicit demonstration.
const FEW_SHOT: &[(&str, &str)] = &[
    ("Yeah, do it. Yeah, do it.", "Yeah, do it."),
    ("Task done. Task done.", "Task done."),
    ("Tath done. Tath done.", "[HALLUCINATION]"),
    ("Taff done. Taff done.", "[HALLUCINATION]"),
    (
        "Astraline, the Quintan, Aston, Ackoned down. Astralane, the Quantone.",
        "[HALLUCINATION]",
    ),
    ("Astralane, Quantone. Astralane, Quantone.", "[HALLUCINATION]"),
    (
        "I'm here. I'm here. I'm here. I'm here. Come here. Come here.",
        "[HALLUCINATION]",
    ),
    (
        "1891, a group of directors from Boston, his third national bank, \
         file a charter for a new institution.",
        "[HALLUCINATION]",
    ),
    (
        "The President The President I'm going to The President",
        "[HALLUCINATION]",
    ),
    ("Test on. Tested. Tested.", "[HALLUCINATION]"),
    (
        "Please check if that background task has completed.",
        "Please check if that background task has completed.",
    ),
    (
        "Okay, check if the slot to ARB analysis is done for all the slots \
         of 28 April for both the vortex validator and this app order flow.",
        "Okay, check if the slot to ARB analysis is done for all the slots \
         of 28 April for both the vortex validator and this app order flow.",
    ),
];

fn ms_to_samples(ms: u32) -> usize {
    (SAMPLE_RATE as u64 * ms as u64 / 1000) as usize
}

#[derive(Debug, Deserialize, Clone)]
struct AsrSegment {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct AsrResponse {
    segments: Vec<AsrSegment>,
}

fn wav_bytes(samples: &[f32]) -> Result<Vec<u8>, String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut cur = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cur, spec).map_err(|e| e.to_string())?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let v = (clamped * i16::MAX as f32) as i16;
            writer.write_sample(v).map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())?;
    }
    Ok(cur.into_inner())
}

fn transcribe(samples: &[f32]) -> Result<String, String> {
    let wav = wav_bytes(samples)?;
    let part = reqwest::blocking::multipart::Part::bytes(wav)
        .file_name("chunk.wav")
        .mime_str("audio/wav")
        .map_err(|e| e.to_string())?;
    let form = reqwest::blocking::multipart::Form::new().part("audio", part);
    let resp = reqwest::blocking::Client::new()
        .post(TRANSCRIBE_URL)
        .multipart(form)
        .send()
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("transcribe failed: HTTP {}", resp.status()));
    }
    let parsed: AsrResponse = resp.json().map_err(|e| e.to_string())?;
    let text = parsed
        .segments
        .iter()
        .map(|s| s.text.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(text)
}

fn output_file_for_session() -> io::Result<PathBuf> {
    let base = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("voice-journal");
    fs::create_dir_all(&base)?;
    let date = Local::now().format("%Y-%m-%d");
    Ok(base.join(format!("journal_{date}.md")))
}

fn hallucination_config_path() -> PathBuf {
    let base = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("voice-journal");
    let _ = fs::create_dir_all(&base);
    base.join(HALLUCINATION_CONFIG_NAME)
}

fn default_hallucination_patterns() -> &'static str {
    "# One regex per line. Lines starting with # are comments.\n\
\\b(the\\s+){2,}\\b\n\
\\b(president\\s+){2,}\\b\n\
\\b(city\\s+){2,}\\b\n\
\\b(the|president|city)(\\s+\\1){2,}\\b\n"
}

fn load_hallucination_filters() -> Vec<Regex> {
    let path = hallucination_config_path();
    if !path.exists() {
        let legacy = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".config/whisper-typer/voice-journal-hallucinations.txt");
        if legacy.exists() {
            let _ = fs::copy(&legacy, &path);
        }
    }
    if !path.exists() {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&path, default_hallucination_patterns());
    }

    let content = fs::read_to_string(&path).unwrap_or_default();
    content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter_map(|pattern| Regex::new(&format!("(?i){pattern}")).ok())
        .collect()
}

fn is_hallucination(text: &str, filters: &[Regex]) -> bool {
    let lowered = text.to_lowercase();
    let normalized = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() || c == '\'' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let apostrophe_folded = normalized.replace('\'', "");
    let lower = normalized.as_str();
    let words = lower.split_whitespace().collect::<Vec<_>>();

    // Built-in guardrail for low-signal chunk fragments.
    if words.is_empty() {
        return true;
    }
    if words.len() == 1 {
        let w = words[0].trim_matches(|c: char| !c.is_alphanumeric());
        if w.len() <= 2 {
            return true;
        }
        if matches!(
            w,
            "the" | "a" | "an" | "and" | "or" | "but" | "okay" | "ok" | "hmm"
        ) {
            return true;
        }
    }

    if filters
        .iter()
        .any(|re| re.is_match(&normalized) || re.is_match(&apostrophe_folded))
    {
        return true;
    }

    // Single-phrase guardrails for common hallucinations, constrained to short utterances.
    let short = words.len() <= 8;
    if short
        && (normalized.contains("the president")
            || normalized.contains("the city")
            || normalized.contains("the road")
            || normalized.contains("im sorry")
            || normalized.contains("i m sorry")
            || normalized.contains("im going to")
            || normalized.contains("i m going to"))
    {
        return true;
    }

    false
}

/// Second-pass hallucination filter using Ollama chat API with few-shot prompt.
/// Stays off if Ollama is unreachable or `WHISPER_VOICE_JOURNAL_LLM=0` is set.
struct OllamaFilter {
    enabled: bool,
    client: reqwest::blocking::Client,
}

impl OllamaFilter {
    fn new() -> Self {
        if std::env::var("WHISPER_VOICE_JOURNAL_LLM").as_deref() == Ok("0") {
            return Self {
                enabled: false,
                client: reqwest::blocking::Client::new(),
            };
        }
        let client = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(OLLAMA_TIMEOUT_SEC))
            .build()
        {
            Ok(c) => c,
            Err(_) => return Self {
                enabled: false,
                client: reqwest::blocking::Client::new(),
            },
        };
        let url = format!("{OLLAMA_HOST}/api/tags");
        let enabled = client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        Self { enabled, client }
    }

    /// Returns Some(true) if the model declares this a hallucination,
    /// Some(false) if it should be kept, or None if the call failed
    /// (in which case the caller should keep the chunk as a safe default).
    fn check(&self, text: &str) -> Option<bool> {
        if !self.enabled {
            return None;
        }
        let mut messages = Vec::with_capacity(FEW_SHOT.len() * 2 + 2);
        messages.push(json!({"role": "system", "content": LLM_SYSTEM_PROMPT}));
        for (u, a) in FEW_SHOT {
            messages.push(json!({"role": "user", "content": *u}));
            messages.push(json!({"role": "assistant", "content": *a}));
        }
        messages.push(json!({"role": "user", "content": text}));

        let body = json!({
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": false,
            "think": false,
            "keep_alive": OLLAMA_KEEP_ALIVE_SEC,
            "options": {
                "temperature": 0.1,
                "num_predict": 200,
            }
        });

        let url = format!("{OLLAMA_HOST}/api/chat");
        let resp = self.client.post(&url).json(&body).send().ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let data: serde_json::Value = resp.json().ok()?;
        let content = data["message"]["content"].as_str()?.trim();
        let lc = content.to_lowercase();
        let is_hall = content.is_empty()
            || lc == "[hallucination]"
            || lc.starts_with("[hallucination]");
        Some(is_hall)
    }
}

fn start_capture(
    tx: Sender<Vec<f32>>,
    level: Arc<Mutex<f32>>,
    status: Arc<Mutex<String>>,
    paused: Arc<AtomicBool>,
    vad_mode: VadMode,
) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("no default input device found")?;
    let cfg = StreamConfig {
        channels: 1,
        sample_rate: SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Fixed(1024),
    };

    let mut in_speech = false;
    let mut utterance: Vec<f32> = Vec::with_capacity(ms_to_samples(10_000));
    let mut pre_roll: std::collections::VecDeque<f32> =
        std::collections::VecDeque::with_capacity(ms_to_samples(PRE_ROLL_MS));
    let mut silence_run_samples: usize = 0;
    let silence_limit = ms_to_samples(SILENCE_FINALIZE_MS);
    let min_utt = ms_to_samples(MIN_UTTERANCE_MS);
    let max_utt = ms_to_samples(MAX_UTTERANCE_MS);
    let pre_roll_cap = ms_to_samples(PRE_ROLL_MS);

    // Silero state held inside the closure so it persists across cpal callbacks.
    // frame_buf accumulates partial frames; last_silero_voiced caches the most
    // recent frame's voiced verdict so the state machine keeps a sticky value
    // when a callback lands without a complete frame.
    let mut frame_buf: Vec<f32> = Vec::with_capacity(vad::FRAME_SAMPLES * 4);
    let mut last_silero_voiced = false;

    let stream = device
        .build_input_stream(
            &cfg,
            move |data: &[f32], _| {
                if paused.load(Ordering::Relaxed) {
                    if let Ok(mut s) = status.lock() {
                        *s = "Paused".to_string();
                    }
                    return;
                }
                let rms = if data.is_empty() {
                    0.0
                } else {
                    let sum: f32 = data.iter().map(|s| s * s).sum();
                    (sum / data.len() as f32).sqrt()
                };
                if let Ok(mut l) = level.lock() {
                    *l = rms;
                }
                let voiced = match &vad_mode {
                    VadMode::Rms => rms >= VAD_RMS_THRESHOLD,
                    VadMode::Silero(silero) => {
                        frame_buf.extend_from_slice(data);
                        while frame_buf.len() >= vad::FRAME_SAMPLES {
                            let frame: Vec<f32> =
                                frame_buf.drain(..vad::FRAME_SAMPLES).collect();
                            match silero.predict(&frame) {
                                Ok(p) => last_silero_voiced = p >= SILERO_VAD_THRESHOLD,
                                Err(e) => eprintln!("silero predict error: {e}"),
                            }
                        }
                        last_silero_voiced
                    }
                };
                if let Ok(mut s) = status.lock() {
                    *s = if voiced {
                        "Recording".to_string()
                    } else {
                        "Silence".to_string()
                    };
                }

                if in_speech {
                    utterance.extend_from_slice(data);
                    if voiced {
                        silence_run_samples = 0;
                    } else {
                        silence_run_samples = silence_run_samples.saturating_add(data.len());
                    }

                    let hit_silence_end = silence_run_samples >= silence_limit;
                    let hit_max_len = utterance.len() >= max_utt;
                    if hit_silence_end || hit_max_len {
                        if utterance.len() >= min_utt {
                            let send_chunk = std::mem::take(&mut utterance);
                            let _ = tx.send(send_chunk);
                        } else {
                            utterance.clear();
                        }
                        silence_run_samples = 0;
                        in_speech = false;
                        if let VadMode::Silero(silero) = &vad_mode {
                            silero.reset_state();
                            frame_buf.clear();
                            last_silero_voiced = false;
                        }
                    }
                } else {
                    for &s in data {
                        if pre_roll.len() == pre_roll_cap {
                            pre_roll.pop_front();
                        }
                        pre_roll.push_back(s);
                    }
                    if voiced {
                        in_speech = true;
                        silence_run_samples = 0;
                        utterance.extend(pre_roll.iter().copied());
                        pre_roll.clear();
                        utterance.extend_from_slice(data);
                    }
                }
            },
            move |err| {
                eprintln!("audio stream error: {err}");
            },
            None,
        )
        .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}

fn spawn_transcriber(
    rx: Receiver<Vec<f32>>,
    tx: Sender<String>,
    path: PathBuf,
    hallucination_filters: Vec<Regex>,
    llm_filter: Arc<OllamaFilter>,
) {
    std::thread::spawn(move || {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("failed to open journal file");
        for chunk in rx {
            match transcribe(&chunk) {
                Ok(text) if !text.trim().is_empty() => {
                    if is_hallucination(&text, &hallucination_filters) {
                        let _ = tx.send(format!("[filtered] {}", text.trim()));
                        continue;
                    }
                    if matches!(llm_filter.check(&text), Some(true)) {
                        let _ = tx.send(format!("[filtered-llm] {}", text.trim()));
                        continue;
                    }
                    let ts = Local::now().format("%H:%M:%S");
                    let line = format!("[{ts}] {text}");
                    let _ = writeln!(file, "{line}");
                    let _ = file.flush();
                    let _ = tx.send(line);
                }
                Ok(_) => {}
                Err(e) => {
                    let _ = tx.send(format!("[error] {e}"));
                }
            }
        }
    });
}

/// Resolve the VAD model path. Tries (in order): WHISPER_VAD_MODEL env var,
/// CWD-relative `models/silero_vad.onnx`, then `~/whisper-typer/models/...`.
/// Returns None if no candidate exists on disk.
fn resolve_silero_model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("WHISPER_VAD_MODEL") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    let cwd_rel = PathBuf::from(SILERO_VAD_MODEL_REL_PATH);
    if cwd_rel.exists() {
        return Some(cwd_rel);
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join("whisper-typer").join(SILERO_VAD_MODEL_REL_PATH);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Build the VAD mode from env. Default RMS; opt into Silero via
/// WHISPER_VOICE_JOURNAL_VAD=silero. Falls back to RMS with a stderr warning
/// if the model can't be found or fails to load.
fn build_vad_mode() -> VadMode {
    if std::env::var("WHISPER_VOICE_JOURNAL_VAD").as_deref() != Ok("silero") {
        return VadMode::Rms;
    }
    let Some(path) = resolve_silero_model_path() else {
        eprintln!(
            "VAD: Silero requested but model not found (set WHISPER_VAD_MODEL or place at {SILERO_VAD_MODEL_REL_PATH}); falling back to RMS"
        );
        return VadMode::Rms;
    };
    match SileroVad::load(&path) {
        Ok(v) => {
            eprintln!("VAD: Silero loaded from {}", path.display());
            VadMode::Silero(Arc::new(v))
        }
        Err(e) => {
            eprintln!("VAD: Silero load failed ({e}); falling back to RMS");
            VadMode::Rms
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = output_file_for_session()?;
    if !out.exists() {
        let mut bootstrap = File::create(&out)?;
        writeln!(
            bootstrap,
            "# Voice Journal\n\nStarted: {}\n",
            Local::now().to_rfc3339()
        )?;
    }

    let hallucination_filters = load_hallucination_filters();
    let llm_filter = Arc::new(OllamaFilter::new());
    let llm_label = if llm_filter.enabled {
        format!("LLM filter: {OLLAMA_MODEL} (active)")
    } else {
        "LLM filter: disabled".to_string()
    };

    let vad_mode = build_vad_mode();
    let vad_label = match &vad_mode {
        VadMode::Rms => format!("VAD: RMS (threshold {VAD_RMS_THRESHOLD})"),
        VadMode::Silero(_) => {
            format!("VAD: Silero ONNX (prob threshold {SILERO_VAD_THRESHOLD})")
        }
    };

    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
    let (line_tx, line_rx) = mpsc::channel::<String>();
    let level = Arc::new(Mutex::new(0.0f32));
    let status = Arc::new(Mutex::new(String::from("Silence")));
    let paused = Arc::new(AtomicBool::new(false));
    let _stream = start_capture(
        audio_tx,
        level.clone(),
        status.clone(),
        paused.clone(),
        vad_mode,
    )?;
    spawn_transcriber(
        audio_rx,
        line_tx,
        out.clone(),
        hallucination_filters,
        llm_filter,
    );

    enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    terminal.clear()?;

    let started = Instant::now();
    let mut lines: Vec<String> = Vec::new();
    let mut scroll_offset: usize = 0; // 0 = follow latest; increases as user scrolls up

    loop {
        while let Ok(l) = line_rx.try_recv() {
            lines.push(l);
            if lines.len() > 200 {
                lines.drain(0..100);
            }
            // Auto-follow latest only when user is already at bottom.
            if scroll_offset == 0 {
                scroll_offset = 0;
            }
        }
        let rms = *level.lock().unwrap_or_else(|e| e.into_inner());
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(5), Constraint::Min(1)])
                .split(f.area());

            let header = vec![
                Line::from(format!(
                    "Voice Journal TUI | elapsed: {}s | press q/esc stop | p pause/resume",
                    started.elapsed().as_secs(),
                )),
                Line::from(format!("Output: {}", out.display())),
                Line::from(vad_label.clone()),
                Line::from(llm_label.clone()),
                Line::from(format!(
                    "Status: {} | Mic RMS: {:.4}",
                    status.lock().map(|s| s.clone()).unwrap_or_else(|_| "Unknown".to_string()),
                    rms
                ))
                .style(Style::default().fg(Color::Green)),
            ];
            let p1 = Paragraph::new(header).block(Block::default().borders(Borders::ALL).title("Status"));
            f.render_widget(p1, chunks[0]);

            let visible_rows = chunks[1].height.saturating_sub(2) as usize;
            let total = lines.len();
            let end = total.saturating_sub(scroll_offset);
            let start = end.saturating_sub(visible_rows);
            let tail = lines[start..end]
                .iter()
                .map(|line| {
                    if line.starts_with("[filtered-llm]") {
                        Line::from(line.clone()).style(Style::default().fg(Color::Magenta))
                    } else if line.starts_with("[filtered]") {
                        Line::from(line.clone()).style(Style::default().fg(Color::DarkGray))
                    } else {
                        Line::from(line.clone())
                    }
                })
                .collect::<Vec<_>>();
            let p2 = Paragraph::new(tail)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title("Live Transcript"));
            f.render_widget(p2, chunks[1]);
        })?;

        if event::poll(Duration::from_millis(150))? {
            if let Event::Key(k) = event::read()? {
                if matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                    break;
                }
                if matches!(k.code, KeyCode::Char('p') | KeyCode::Char('P')) {
                    let next = !paused.load(Ordering::Relaxed);
                    paused.store(next, Ordering::Relaxed);
                    if let Ok(mut s) = status.lock() {
                        *s = if next {
                            "Paused".to_string()
                        } else {
                            "Silence".to_string()
                        };
                    }
                }
                match k.code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        let max_offset = lines.len().saturating_sub(1);
                        scroll_offset = (scroll_offset + 1).min(max_offset);
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        scroll_offset = scroll_offset.saturating_sub(1);
                    }
                    KeyCode::PageUp => {
                        let step = 10usize;
                        let max_offset = lines.len().saturating_sub(1);
                        scroll_offset = (scroll_offset + step).min(max_offset);
                    }
                    KeyCode::PageDown => {
                        let step = 10usize;
                        scroll_offset = scroll_offset.saturating_sub(step);
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    terminal.clear()?;
    println!("Voice journal saved to {}", out.display());
    Ok(())
}
