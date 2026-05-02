use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
/// Hysteresis: once `in_speech` is true, a frame counts as voiced if its
/// probability is ≥ this lower bar. Without hysteresis, probs that bounce
/// between 0.4 and 0.6 during normal speech repeatedly cross a single
/// threshold and chop the utterance into many short fragments.
const SILERO_STAY_THRESHOLD: f32 = 0.35;
const SILERO_RMS_RESCUE_THRESHOLD: f32 = 0.05;
const SILERO_SPEECH_HOLD_FRAMES: u8 = 8;
/// Require this many consecutive ≥enter-threshold frames before starting an
/// utterance. Filters out single-frame transients (keystrokes, taps) that
/// previously triggered 1.2s chunks of "[filtered] Thank you" hallucinations
/// during ambient silence.
const SILERO_SPEECH_START_FRAMES: u8 = 4;
/// After any key event on any keyboard, force VAD to silence for this window.
/// Mechanical keystrokes are 20–50ms broadband transients that fool Silero;
/// gating during typing is what Krisp/Zoom call "keyboard noise suppression."
/// Only applied when not already `in_speech` — once a real utterance is in
/// progress, typing on the keyboard mustn't truncate it.
const KEYSTROKE_GATE_MS: u64 = 250;
/// Drop an utterance before sending to Whisper if it doesn't contain at least
/// this much voiced audio. Pre-roll + silence-tail puff up the byte length
/// of every utterance to ~1.2s, so byte-length alone can't tell a real
/// sentence apart from a single transient followed by silence.
const MIN_VOICED_MS: u32 = 250;
const SILERO_VAD_MODEL_REL_PATH: &str = "models/silero_vad.onnx";
const SILENCE_FINALIZE_MS: u32 = 900;
const MIN_UTTERANCE_MS: u32 = 500;
const MAX_UTTERANCE_MS: u32 = 25_000;
const PRE_ROLL_MS: u32 = 200;

/// Voice-activity-detection backend. Silero is preferred when the ONNX model is
/// available; RMS remains as a cheap fallback and debug override.
#[derive(Clone)]
enum VadMode {
    Rms,
    Silero {
        detector: Arc<SileroVad>,
        /// Probability ≥ this enters speech (after streak gate).
        threshold: f32,
        /// Probability ≥ this keeps an in-progress utterance voiced.
        /// Lower than `threshold` to give hysteresis.
        stay_threshold: f32,
        rms_rescue_threshold: f32,
    },
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

/// Sibling of the journal that captures EVERY transcribed utterance —
/// including `[filtered]`, `[filtered-llm]`, and `[error]` lines that the
/// clean journal drops. Used to audit false positives from the regex/LLM
/// filters and Silero VAD without re-running audio.
fn unfiltered_path_for(journal: &PathBuf) -> PathBuf {
    let parent = journal
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = journal
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("journal");
    parent.join(format!("{stem}.unfiltered.md"))
}

fn debug_paths_for_session(journal_path: &PathBuf) -> io::Result<(PathBuf, PathBuf)> {
    let parent = journal_path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "journal path has no parent"))?;
    let stem = journal_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("journal");
    let stamp = Local::now().format("%H%M%S");
    Ok((
        parent.join(format!("{stem}_{stamp}.mic.wav")),
        parent.join(format!("{stem}_{stamp}.vad.csv")),
    ))
}

fn debug_enabled() -> bool {
    let cli_enabled = std::env::args().skip(1).any(|arg| arg == "--debug" || arg == "-d");
    let env_enabled = std::env::var("WHISPER_VOICE_JOURNAL_DEBUG")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    cli_enabled || env_enabled
}

struct VadDebugRecorder {
    wav: WavWriter<BufWriter<File>>,
    csv: BufWriter<File>,
    callback_index: u64,
    sample_index: u64,
}

struct VadDebugRow<'a> {
    rms: f32,
    silero_prob: Option<f32>,
    silero_threshold: Option<f32>,
    rms_rescue_threshold: Option<f32>,
    voiced: bool,
    in_speech_before: bool,
    in_speech_after: bool,
    silence_run_samples: usize,
    utterance_samples: usize,
    speech_hold_frames: u8,
    event: &'a str,
}

impl VadDebugRecorder {
    fn create(wav_path: &PathBuf, csv_path: &PathBuf) -> io::Result<Self> {
        let spec = WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let wav_file = BufWriter::new(File::create(wav_path)?);
        let wav = WavWriter::new(wav_file, spec).map_err(|e| io::Error::other(e.to_string()))?;
        let mut csv = BufWriter::new(File::create(csv_path)?);
        writeln!(
            csv,
            "callback_index,start_sample,samples,rms,silero_prob,silero_threshold,rms_rescue_threshold,voiced,in_speech_before,in_speech_after,silence_run_samples,utterance_samples,speech_hold_frames,event"
        )?;
        Ok(Self {
            wav,
            csv,
            callback_index: 0,
            sample_index: 0,
        })
    }

    fn record(&mut self, samples: &[f32], row: VadDebugRow<'_>) -> io::Result<()> {
        let start_sample = self.sample_index;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let v = (clamped * i16::MAX as f32) as i16;
            self.wav
                .write_sample(v)
                .map_err(|e| io::Error::other(e.to_string()))?;
        }
        self.sample_index = self.sample_index.saturating_add(samples.len() as u64);

        let silero_prob = row
            .silero_prob
            .map(|p| format!("{p:.6}"))
            .unwrap_or_default();
        let silero_threshold = row
            .silero_threshold
            .map(|t| format!("{t:.6}"))
            .unwrap_or_default();
        let rms_rescue_threshold = row
            .rms_rescue_threshold
            .map(|t| format!("{t:.6}"))
            .unwrap_or_default();
        writeln!(
            self.csv,
            "{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{}",
            self.callback_index,
            start_sample,
            samples.len(),
            row.rms,
            silero_prob,
            silero_threshold,
            rms_rescue_threshold,
            row.voiced,
            row.in_speech_before,
            row.in_speech_after,
            row.silence_run_samples,
            row.utterance_samples,
            row.speech_hold_frames,
            row.event
        )?;
        self.callback_index = self.callback_index.saturating_add(1);
        if self.callback_index % 50 == 0 {
            let _ = self.csv.flush();
            let _ = self.wav.flush();
        }
        Ok(())
    }
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

/// Spawn one std::thread per detected keyboard that polls `fetch_events` and
/// stamps the current epoch-ms into `last_keypress_ms` on any press/repeat.
/// Auto-rediscovers if all watchers exit (device disconnect or transient
/// kernel hiccup). Permission needs the user in the `input` group, same as
/// whisper-typer's hotkey monitor.
fn spawn_keystroke_watcher(last_keypress_ms: Arc<AtomicU64>) {
    use evdev::{Device, EventType, Key};
    std::thread::spawn(move || loop {
        let keyboards: Vec<Device> = evdev::enumerate()
            .filter_map(|(_, dev)| {
                let keys = dev.supported_keys()?;
                if keys.contains(Key::KEY_A) && keys.contains(Key::KEY_ENTER) {
                    Some(dev)
                } else {
                    None
                }
            })
            .collect();

        if keyboards.is_empty() {
            // Silent retry — we can't print here without corrupting the TUI's
            // ratatui output. The user can tell the watcher is alive by
            // watching the `keystroke suppressed` counter in the TUI.
            std::thread::sleep(Duration::from_secs(5));
            continue;
        }

        let mut handles = Vec::new();
        for mut dev in keyboards {
            let last = last_keypress_ms.clone();
            handles.push(std::thread::spawn(move || loop {
                match dev.fetch_events() {
                    Ok(events) => {
                        let mut had_event = false;
                        for ev in events {
                            // value 1 = press, 2 = autorepeat. Releases (0)
                            // also indicate user activity but we focus on
                            // press/repeat which align with the audio click.
                            if ev.event_type() == EventType::KEY
                                && (ev.value() == 1 || ev.value() == 2)
                            {
                                had_event = true;
                            }
                        }
                        if had_event {
                            let now_ms = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_millis() as u64)
                                .unwrap_or(0);
                            last.store(now_ms, Ordering::Relaxed);
                        } else {
                            std::thread::sleep(Duration::from_millis(8));
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(8));
                    }
                    Err(_) => break,
                }
            }));
        }
        for h in handles {
            let _ = h.join();
        }
        std::thread::sleep(Duration::from_secs(2));
    });
}

fn start_capture(
    tx: Sender<Vec<f32>>,
    level: Arc<Mutex<f32>>,
    silero_prob: Arc<Mutex<Option<f32>>>,
    debug_recorder: Option<Arc<Mutex<VadDebugRecorder>>>,
    status: Arc<Mutex<String>>,
    paused: Arc<AtomicBool>,
    vad_mode: VadMode,
    transitions: Arc<AtomicU64>,
    longest_voiced_ms: Arc<AtomicU64>,
    voiced_run_ms: Arc<AtomicU64>,
    last_keypress_ms: Arc<AtomicU64>,
    keystroke_suppressions: Arc<AtomicU64>,
    low_voiced_drops: Arc<AtomicU64>,
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
    // frame_buf accumulates partial frames; speech_hold_frames keeps speech
    // sticky across brief probability dips so the last frame in a callback
    // cannot erase an earlier voiced frame. voiced_streak gates speech_start
    // on consecutive frames to reject single-frame transients.
    let mut frame_buf: Vec<f32> = Vec::with_capacity(vad::FRAME_SAMPLES * 4);
    let mut speech_hold_frames: u8 = 0;
    let mut voiced_streak: u8 = 0;
    // Per-callback flicker tracking — every voiced↔unvoiced flip increments
    // `transitions`, and the longest sustained voiced run (ms) is exposed
    // for the TUI so the user can see when speech is being chopped up.
    let mut prev_voiced = false;
    let mut voiced_run_samples: u64 = 0;
    // Voiced-content audit for the current in-progress utterance. Reset on
    // speech_start, used at finalize to drop barely-voiced blips before
    // they reach Whisper.
    let mut voiced_samples_in_utt: usize = 0;
    let min_voiced_samples = ms_to_samples(MIN_VOICED_MS);

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
                let in_speech_before = in_speech;
                let mut callback_silero_prob: Option<f32> = None;
                let mut silero_threshold_for_row: Option<f32> = None;
                let mut rms_rescue_threshold_for_row: Option<f32> = None;
                let voiced = match &vad_mode {
                    VadMode::Rms => {
                        if let Ok(mut p) = silero_prob.lock() {
                            *p = None;
                        }
                        rms >= VAD_RMS_THRESHOLD
                    }
                    VadMode::Silero {
                        detector,
                        threshold,
                        stay_threshold,
                        rms_rescue_threshold,
                    } => {
                        silero_threshold_for_row = Some(*threshold);
                        rms_rescue_threshold_for_row = Some(*rms_rescue_threshold);
                        frame_buf.extend_from_slice(data);
                        let mut callback_voiced = false;
                        while frame_buf.len() >= vad::FRAME_SAMPLES {
                            let frame: Vec<f32> =
                                frame_buf.drain(..vad::FRAME_SAMPLES).collect();
                            match detector.predict(&frame) {
                                Ok(p) => {
                                    callback_silero_prob = Some(
                                        callback_silero_prob.map_or(p, |max_prob| max_prob.max(p)),
                                    );
                                    // Hysteresis: while in_speech, the bar to
                                    // remain voiced is `stay_threshold`; while
                                    // not in speech, we need the higher
                                    // `threshold` AND a multi-frame streak.
                                    let active_threshold = if in_speech_before {
                                        *stay_threshold
                                    } else {
                                        *threshold
                                    };
                                    if p >= active_threshold {
                                        voiced_streak = voiced_streak.saturating_add(1);
                                        let required = if in_speech_before {
                                            1
                                        } else {
                                            SILERO_SPEECH_START_FRAMES
                                        };
                                        if voiced_streak >= required {
                                            speech_hold_frames = SILERO_SPEECH_HOLD_FRAMES;
                                        }
                                    } else {
                                        voiced_streak = 0;
                                    }
                                    if speech_hold_frames > 0 {
                                        callback_voiced = true;
                                        speech_hold_frames = speech_hold_frames.saturating_sub(1);
                                    }
                                }
                                Err(e) => eprintln!("silero predict error: {e}"),
                            }
                        }
                        if rms >= *rms_rescue_threshold {
                            speech_hold_frames = SILERO_SPEECH_HOLD_FRAMES;
                            callback_voiced = true;
                        }
                        if let Some(p) = callback_silero_prob {
                            if let Ok(mut prob) = silero_prob.lock() {
                                *prob = Some(p);
                            }
                        }
                        callback_voiced || speech_hold_frames > 0
                    }
                };
                // Keystroke gate: any key event in the last
                // KEYSTROKE_GATE_MS forces voiced=false and clears
                // hold/streak so the gate's exit doesn't immediately latch
                // residual state. Applied after Silero predict so the LSTM
                // still sees the audio (state stays consistent), but the
                // VAD output ignores typing-induced transients.
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                let last_kp = last_keypress_ms.load(Ordering::Relaxed);
                let in_keystroke_window =
                    last_kp > 0 && now_ms.saturating_sub(last_kp) < KEYSTROKE_GATE_MS;
                // Only gate ENTRY into speech. Once we're already in speech,
                // keystrokes are background and the user's voice should
                // dominate; gating mid-utterance would chop sentences when
                // the user types while speaking.
                let voiced = if in_keystroke_window && !in_speech_before {
                    if voiced {
                        keystroke_suppressions.fetch_add(1, Ordering::Relaxed);
                    }
                    speech_hold_frames = 0;
                    voiced_streak = 0;
                    false
                } else {
                    voiced
                };

                if let Ok(mut s) = status.lock() {
                    *s = if voiced {
                        "Recording".to_string()
                    } else if in_keystroke_window {
                        "Typing".to_string()
                    } else {
                        "Silence".to_string()
                    };
                }

                if voiced != prev_voiced {
                    transitions.fetch_add(1, Ordering::Relaxed);
                }
                if voiced {
                    voiced_run_samples = voiced_run_samples
                        .saturating_add(data.len() as u64);
                } else {
                    if prev_voiced && voiced_run_samples > 0 {
                        let run_ms = voiced_run_samples * 1000 / SAMPLE_RATE as u64;
                        if run_ms > longest_voiced_ms.load(Ordering::Relaxed) {
                            longest_voiced_ms.store(run_ms, Ordering::Relaxed);
                        }
                    }
                    voiced_run_samples = 0;
                }
                voiced_run_ms.store(
                    voiced_run_samples * 1000 / SAMPLE_RATE as u64,
                    Ordering::Relaxed,
                );
                prev_voiced = voiced;

                let mut event = "";
                if in_speech {
                    utterance.extend_from_slice(data);
                    if voiced {
                        silence_run_samples = 0;
                        voiced_samples_in_utt =
                            voiced_samples_in_utt.saturating_add(data.len());
                    } else {
                        silence_run_samples = silence_run_samples.saturating_add(data.len());
                    }

                    let hit_silence_end = silence_run_samples >= silence_limit;
                    let hit_max_len = utterance.len() >= max_utt;
                    if hit_silence_end || hit_max_len {
                        let has_min_bytes = utterance.len() >= min_utt;
                        let has_min_voiced = voiced_samples_in_utt >= min_voiced_samples;
                        if has_min_bytes && has_min_voiced {
                            let send_chunk = std::mem::take(&mut utterance);
                            event = if hit_max_len {
                                "chunk_sent_max_len"
                            } else {
                                "chunk_sent_silence"
                            };
                            let _ = tx.send(send_chunk);
                        } else {
                            utterance.clear();
                            event = if !has_min_voiced {
                                low_voiced_drops.fetch_add(1, Ordering::Relaxed);
                                "drop_low_voiced"
                            } else {
                                "drop_short"
                            };
                        }
                        silence_run_samples = 0;
                        voiced_samples_in_utt = 0;
                        in_speech = false;
                        if let VadMode::Silero { detector, .. } = &vad_mode {
                            detector.reset_state();
                            frame_buf.clear();
                            speech_hold_frames = 0;
                            voiced_streak = 0;
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
                        event = "speech_start";
                        silence_run_samples = 0;
                        voiced_samples_in_utt = data.len();
                        utterance.extend(pre_roll.iter().copied());
                        pre_roll.clear();
                        utterance.extend_from_slice(data);
                    }
                }
                if let Some(debug_recorder) = &debug_recorder {
                    if let Ok(mut recorder) = debug_recorder.lock() {
                        let _ = recorder.record(
                            data,
                            VadDebugRow {
                                rms,
                                silero_prob: callback_silero_prob,
                                silero_threshold: silero_threshold_for_row,
                                rms_rescue_threshold: rms_rescue_threshold_for_row,
                                voiced,
                                in_speech_before,
                                in_speech_after: in_speech,
                                silence_run_samples,
                                utterance_samples: utterance.len(),
                                speech_hold_frames,
                                event,
                            },
                        );
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

/// Tail today's whisper-typer JSONL history and inject each new dictation
/// as a `[dictated]` line into both the live TUI channel and the journal +
/// unfiltered files. Date rolls naturally because the path is recomputed
/// every poll cycle. If the file disappears (path moved, day rolled,
/// service stopped), the tailer transparently reopens.
fn spawn_dictation_tailer(
    line_tx: Sender<String>,
    journal_path: PathBuf,
    unfiltered_path: PathBuf,
) {
    #[derive(Deserialize)]
    struct DictationEntry {
        #[serde(default)]
        timestamp: String,
        #[serde(default)]
        final_text: String,
    }

    fn jsonl_path_for_today() -> PathBuf {
        let date = Local::now().format("%Y-%m-%d");
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".whisper-typer-history")
            .join(format!("{date}.jsonl"))
    }

    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader, Seek, SeekFrom};

        let mut current_path = jsonl_path_for_today();
        let mut last_offset: u64 = match fs::metadata(&current_path) {
            // Start at end so we don't replay history on restart.
            Ok(meta) => meta.len(),
            Err(_) => 0,
        };

        loop {
            let path = jsonl_path_for_today();
            if path != current_path {
                // Date rolled. Start the new file from byte 0.
                current_path = path.clone();
                last_offset = 0;
            }
            let new_lines: Vec<String> = match File::open(&current_path) {
                Ok(file) => {
                    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
                    if len < last_offset {
                        last_offset = 0; // file truncated/rotated
                    }
                    let mut reader = BufReader::new(file);
                    if reader.seek(SeekFrom::Start(last_offset)).is_err() {
                        std::thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                    let mut buf = String::new();
                    let mut collected = Vec::new();
                    while let Ok(n) = reader.read_line(&mut buf) {
                        if n == 0 {
                            break;
                        }
                        // Only emit complete lines (ending in \n). A partial
                        // last line means whisper-typer is mid-write — leave
                        // it for the next poll.
                        if buf.ends_with('\n') {
                            collected.push(buf.trim_end().to_string());
                            last_offset = last_offset.saturating_add(n as u64);
                        } else {
                            break;
                        }
                        buf.clear();
                    }
                    collected
                }
                Err(_) => Vec::new(),
            };

            if !new_lines.is_empty() {
                let mut journal = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&journal_path)
                    .ok();
                let mut unfiltered = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&unfiltered_path)
                    .ok();
                for raw in new_lines {
                    let entry: DictationEntry = match serde_json::from_str(&raw) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };
                    if entry.final_text.trim().is_empty() {
                        continue;
                    }
                    let hms = entry
                        .timestamp
                        .split('T')
                        .nth(1)
                        .and_then(|t| t.split('.').next())
                        .unwrap_or("--:--:--");
                    let line = format!("[{hms}] [dictated] {}", entry.final_text.trim());
                    if let Some(f) = journal.as_mut() {
                        let _ = writeln!(f, "{line}");
                        let _ = f.flush();
                    }
                    if let Some(f) = unfiltered.as_mut() {
                        let _ = writeln!(f, "{line}");
                        let _ = f.flush();
                    }
                    let _ = line_tx.send(line);
                }
            }

            std::thread::sleep(Duration::from_millis(500));
        }
    });
}

fn spawn_transcriber(
    rx: Receiver<Vec<f32>>,
    tx: Sender<String>,
    path: PathBuf,
    unfiltered_path: PathBuf,
    hallucination_filters: Vec<Regex>,
    llm_filter: Arc<OllamaFilter>,
) {
    std::thread::spawn(move || {
        let mut journal = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("failed to open journal file");
        let mut unfiltered = OpenOptions::new()
            .create(true)
            .append(true)
            .open(unfiltered_path)
            .expect("failed to open unfiltered journal file");
        for chunk in rx {
            match transcribe(&chunk) {
                Ok(text) if !text.trim().is_empty() => {
                    let trimmed = text.trim();
                    let ts = Local::now().format("%H:%M:%S");
                    if is_hallucination(trimmed, &hallucination_filters) {
                        let _ = writeln!(unfiltered, "[{ts}] [filtered] {trimmed}");
                        let _ = unfiltered.flush();
                        let _ = tx.send(format!("[filtered] {trimmed}"));
                        continue;
                    }
                    if matches!(llm_filter.check(trimmed), Some(true)) {
                        let _ = writeln!(unfiltered, "[{ts}] [filtered-llm] {trimmed}");
                        let _ = unfiltered.flush();
                        let _ = tx.send(format!("[filtered-llm] {trimmed}"));
                        continue;
                    }
                    let line = format!("[{ts}] {trimmed}");
                    let _ = writeln!(journal, "{line}");
                    let _ = journal.flush();
                    let _ = writeln!(unfiltered, "{line}");
                    let _ = unfiltered.flush();
                    let _ = tx.send(line);
                }
                Ok(_) => {}
                Err(e) => {
                    let ts = Local::now().format("%H:%M:%S");
                    let _ = writeln!(unfiltered, "[{ts}] [error] {e}");
                    let _ = unfiltered.flush();
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

fn silero_threshold() -> f32 {
    match std::env::var("WHISPER_VOICE_JOURNAL_SILERO_THRESHOLD") {
        Ok(v) => match v.parse::<f32>() {
            Ok(threshold) if (0.0..=1.0).contains(&threshold) => threshold,
            _ => {
                eprintln!(
                    "VAD: invalid WHISPER_VOICE_JOURNAL_SILERO_THRESHOLD={v:?}; using {SILERO_VAD_THRESHOLD}"
                );
                SILERO_VAD_THRESHOLD
            }
        },
        Err(_) => SILERO_VAD_THRESHOLD,
    }
}

fn silero_stay_threshold(enter: f32) -> f32 {
    let default = SILERO_STAY_THRESHOLD.min(enter);
    match std::env::var("WHISPER_VOICE_JOURNAL_SILERO_STAY_THRESHOLD") {
        Ok(v) => match v.parse::<f32>() {
            Ok(t) if (0.0..=1.0).contains(&t) => t.min(enter),
            _ => {
                eprintln!(
                    "VAD: invalid WHISPER_VOICE_JOURNAL_SILERO_STAY_THRESHOLD={v:?}; using {default}"
                );
                default
            }
        },
        Err(_) => default,
    }
}

fn silero_rms_rescue_threshold() -> f32 {
    match std::env::var("WHISPER_VOICE_JOURNAL_RMS_RESCUE_THRESHOLD") {
        Ok(v) => match v.parse::<f32>() {
            Ok(threshold) if threshold >= 0.0 => threshold,
            _ => {
                eprintln!(
                    "VAD: invalid WHISPER_VOICE_JOURNAL_RMS_RESCUE_THRESHOLD={v:?}; using {SILERO_RMS_RESCUE_THRESHOLD}"
                );
                SILERO_RMS_RESCUE_THRESHOLD
            }
        },
        Err(_) => SILERO_RMS_RESCUE_THRESHOLD,
    }
}

/// Build the VAD mode from env. Default is auto: use Silero when the ONNX model
/// exists, otherwise use RMS. Set WHISPER_VOICE_JOURNAL_VAD=rms to force RMS,
/// or WHISPER_VOICE_JOURNAL_VAD=silero to require Silero and warn on fallback.
fn build_vad_mode() -> VadMode {
    let requested = std::env::var("WHISPER_VOICE_JOURNAL_VAD").unwrap_or_else(|_| "auto".into());
    let requested = requested.to_ascii_lowercase();
    if requested == "rms" {
        eprintln!("VAD: RMS forced by WHISPER_VOICE_JOURNAL_VAD=rms");
        return VadMode::Rms;
    }

    let Some(path) = resolve_silero_model_path() else {
        if requested == "silero" {
            eprintln!(
                "VAD: Silero requested but model not found (set WHISPER_VAD_MODEL or place at {SILERO_VAD_MODEL_REL_PATH}); falling back to RMS"
            );
        } else {
            eprintln!(
                "VAD: Silero model not found (set WHISPER_VAD_MODEL or place at {SILERO_VAD_MODEL_REL_PATH}); using RMS"
            );
        }
        return VadMode::Rms;
    };

    match SileroVad::load(&path) {
        Ok(v) => {
            eprintln!("VAD: Silero loaded from {}", path.display());
            let enter = silero_threshold();
            VadMode::Silero {
                detector: Arc::new(v),
                threshold: enter,
                stay_threshold: silero_stay_threshold(enter),
                rms_rescue_threshold: silero_rms_rescue_threshold(),
            }
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
    let unfiltered_out = unfiltered_path_for(&out);
    if !unfiltered_out.exists() {
        let mut bootstrap = File::create(&unfiltered_out)?;
        writeln!(
            bootstrap,
            "# Voice Journal (unfiltered)\n\n\
             Captures every transcribed utterance — including [filtered], \
             [filtered-llm], and [error] lines — for false-positive auditing.\n\n\
             Started: {}\n",
            Local::now().to_rfc3339()
        )?;
    }
    let (debug_recorder, debug_label) = if debug_enabled() {
        let (debug_wav, debug_csv) = debug_paths_for_session(&out)?;
        let recorder = Arc::new(Mutex::new(VadDebugRecorder::create(
            &debug_wav,
            &debug_csv,
        )?));
        (
            Some(recorder),
            Some(format!(
                "Debug: {} | {}",
                debug_wav.display(),
                debug_csv.display()
            )),
        )
    } else {
        (None, None)
    };

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
        VadMode::Silero {
            threshold,
            stay_threshold,
            rms_rescue_threshold,
            ..
        } => {
            format!(
                "VAD: Silero ONNX (enter {threshold}, stay {stay_threshold}, RMS rescue {rms_rescue_threshold})"
            )
        }
    };

    let (audio_tx, audio_rx) = mpsc::channel::<Vec<f32>>();
    let (line_tx, line_rx) = mpsc::channel::<String>();
    let level = Arc::new(Mutex::new(0.0f32));
    let silero_prob = Arc::new(Mutex::new(None::<f32>));
    let status = Arc::new(Mutex::new(String::from("Silence")));
    let paused = Arc::new(AtomicBool::new(false));
    let transitions = Arc::new(AtomicU64::new(0));
    let longest_voiced_ms = Arc::new(AtomicU64::new(0));
    let voiced_run_ms = Arc::new(AtomicU64::new(0));
    let last_keypress_ms = Arc::new(AtomicU64::new(0));
    let keystroke_suppressions = Arc::new(AtomicU64::new(0));
    let low_voiced_drops = Arc::new(AtomicU64::new(0));
    spawn_keystroke_watcher(last_keypress_ms.clone());
    let _stream = start_capture(
        audio_tx,
        level.clone(),
        silero_prob.clone(),
        debug_recorder,
        status.clone(),
        paused.clone(),
        vad_mode,
        transitions.clone(),
        longest_voiced_ms.clone(),
        voiced_run_ms.clone(),
        last_keypress_ms.clone(),
        keystroke_suppressions.clone(),
        low_voiced_drops.clone(),
    )?;
    spawn_transcriber(
        audio_rx,
        line_tx.clone(),
        out.clone(),
        unfiltered_out.clone(),
        hallucination_filters,
        llm_filter,
    );
    spawn_dictation_tailer(line_tx, out.clone(), unfiltered_out.clone());

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
        let prob = *silero_prob.lock().unwrap_or_else(|e| e.into_inner());
        let total_transitions = transitions.load(Ordering::Relaxed);
        let elapsed_secs = started.elapsed().as_secs_f64().max(1.0);
        let transitions_per_min = total_transitions as f64 * 60.0 / elapsed_secs;
        let cur_voiced_ms = voiced_run_ms.load(Ordering::Relaxed);
        let max_voiced_ms = longest_voiced_ms.load(Ordering::Relaxed);
        let ks_suppressions = keystroke_suppressions.load(Ordering::Relaxed);
        let lv_drops = low_voiced_drops.load(Ordering::Relaxed);
        terminal.draw(|f| {
            let status_height = if debug_label.is_some() { 10 } else { 9 };
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(status_height), Constraint::Min(1)])
                .split(f.area());

            let mut header = vec![
                Line::from(format!(
                    "Voice Journal TUI | elapsed: {}s | press q/esc stop | p pause/resume",
                    started.elapsed().as_secs(),
                )),
                Line::from(format!("Output: {}", out.display())),
                Line::from(format!("Unfiltered: {}", unfiltered_out.display())),
            ];
            if let Some(label) = &debug_label {
                header.push(Line::from(label.clone()));
            }
            header.extend([
                Line::from(vad_label.clone()),
                Line::from(llm_label.clone()),
                Line::from(match prob {
                    Some(p) => format!(
                        "Status: {} | Mic RMS: {:.4} | Silero prob: {:.3}",
                        status
                            .lock()
                            .map(|s| s.clone())
                            .unwrap_or_else(|_| "Unknown".to_string()),
                        rms,
                        p
                    ),
                    None => format!(
                        "Status: {} | Mic RMS: {:.4}",
                        status
                            .lock()
                            .map(|s| s.clone())
                            .unwrap_or_else(|_| "Unknown".to_string()),
                        rms
                    ),
                })
                .style(Style::default().fg(Color::Green)),
                Line::from(format!(
                    "Flicker: {total_transitions} ({transitions_per_min:.1}/min) | voiced: {cur_voiced_ms}ms (max {max_voiced_ms}ms) | gated: {ks_suppressions} | dropped low-voiced: {lv_drops}"
                ))
                .style(Style::default().fg(Color::Yellow)),
            ]);
            let p1 = Paragraph::new(header).block(Block::default().borders(Borders::ALL).title("Status"));
            f.render_widget(p1, chunks[0]);

            let visible_rows = chunks[1].height.saturating_sub(2) as usize;
            let total = lines.len();
            let end = total.saturating_sub(scroll_offset);
            let start = end.saturating_sub(visible_rows);
            let tail = lines[start..end]
                .iter()
                .map(|line| {
                    if line.contains("[dictated]") {
                        Line::from(line.clone()).style(Style::default().fg(Color::Cyan))
                    } else if line.starts_with("[filtered-llm]") {
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
