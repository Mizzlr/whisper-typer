//! Voice Journal — live, VAD-chunked transcription GUI on top of
//! whisper-typer-rs's `/transcribe` endpoint.
//!
//! Pipeline:
//!   cpal mic callback (audio thread)
//!     ├─ RMS-based VAD: tracks speech/silence, accumulates the current utterance
//!     └─ on ≥600ms silence after speech: finalize → mpsc → worker
//!   worker thread:
//!     - POSTs the utterance WAV to /transcribe
//!     - sends the result back to the UI thread via mpsc
//!   egui UI thread:
//!     - renders the live RMS meter + each utterance line (pending → text)
//!     - "Stop" turns the mic off but lets pending utterances finish landing
//!
//! Whisper isn't streaming, so the natural granularity is one utterance = one
//! pause-bounded chunk. Transcription latency is dominated by Whisper itself
//! (~tens of ms on the 3060 for short clips), so the UX feels near-live.

use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Local};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use eframe::egui;
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::Deserialize;

const TRANSCRIBE_URL: &str = "http://127.0.0.1:8767/transcribe";
const SR: u32 = 16_000;
const SR_F: f32 = 16_000.0;

// VAD tuning
const VAD_RMS_THRESHOLD: f32 = 0.012;
const SILENCE_TO_FINALIZE_MS: u32 = 600;
const MIN_UTTERANCE_MS: u32 = 250;
const MAX_UTTERANCE_MS: u32 = 25_000;
const PRE_ROLL_MS: u32 = 200;

fn ms_to_samples(ms: u32) -> usize {
    (SR as u64 * ms as u64 / 1000) as usize
}

// --- HTTP types ---

#[derive(Debug, Deserialize, Clone)]
struct AsrSegment {
    #[serde(default)]
    start_s: f64,
    #[serde(default)]
    #[allow(dead_code)] // accepted from server, currently unrendered
    end_s: f64,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct AsrResponse {
    duration_s: f64,
    latency_ms: f64,
    segments: Vec<AsrSegment>,
}

// --- Pipeline messages ---

struct UtteranceJob {
    id: u64,
    samples: Vec<f32>,
    /// Wallclock time of the first sample in this utterance.
    started_at: DateTime<Local>,
    /// Seconds since recording began (for in-session timeline display).
    offset_s: f64,
}

enum WorkerMsg {
    Finalized {
        id: u64,
        result: Result<AsrResponse, String>,
    },
}

// --- VAD / audio state ---

/// Shared between the cpal callback and the UI thread.
struct AudioState {
    /// Live RMS reading of the most recent callback frame, for the level meter.
    last_rms: f32,
    /// Currently tracking an active utterance (between speech onset and silence-finalize).
    in_speech: bool,
    /// Samples since the last speech sample. When this exceeds the silence threshold,
    /// we finalize the utterance.
    silence_run: usize,
    /// Audio for the current utterance (includes pre-roll once speech started).
    utterance: Vec<f32>,
    /// Timestamp of the first sample in `utterance`.
    utterance_started_at: Option<DateTime<Local>>,
    utterance_offset_s: f64,
    /// Rolling pre-roll buffer of the last PRE_ROLL_MS so we don't clip the first phoneme.
    pre_roll: VecDeque<f32>,
    /// Total samples seen since recording started — used for offset_s.
    samples_total: u64,
    /// Monotonic ID for utterances.
    next_id: u64,
    /// Time (Instant) recording started, for the elapsed timer.
    rec_started: Option<Instant>,
}

impl AudioState {
    fn new() -> Self {
        Self {
            last_rms: 0.0,
            in_speech: false,
            silence_run: 0,
            utterance: Vec::with_capacity(SR as usize * 5),
            utterance_started_at: None,
            utterance_offset_s: 0.0,
            pre_roll: VecDeque::with_capacity(ms_to_samples(PRE_ROLL_MS)),
            samples_total: 0,
            next_id: 0,
            rec_started: None,
        }
    }

    fn reset_session(&mut self) {
        // Reset per-session state but keep `next_id` monotonic across stop/start
        // so the UI can preserve previous utterances and index into them by id.
        self.last_rms = 0.0;
        self.in_speech = false;
        self.silence_run = 0;
        self.utterance.clear();
        self.utterance_started_at = None;
        self.utterance_offset_s = 0.0;
        self.pre_roll.clear();
        self.samples_total = 0;
        self.rec_started = Some(Instant::now());
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Push samples through the VAD state machine. Returns Some(job) when an
/// utterance has been finalized and should be sent for transcription.
fn process_chunk(state: &mut AudioState, chunk: &[f32]) -> Option<UtteranceJob> {
    let energy = rms(chunk);
    state.last_rms = energy;
    let now_total = state.samples_total;
    state.samples_total = state.samples_total.saturating_add(chunk.len() as u64);

    let is_voiced = energy >= VAD_RMS_THRESHOLD;

    let pre_roll_capacity = ms_to_samples(PRE_ROLL_MS);
    if state.in_speech {
        // Accumulate into the active utterance
        state.utterance.extend_from_slice(chunk);
        if is_voiced {
            state.silence_run = 0;
        } else {
            state.silence_run = state.silence_run.saturating_add(chunk.len());
        }

        // Force-flush if utterance is too long
        if state.utterance.len() >= ms_to_samples(MAX_UTTERANCE_MS) {
            return finalize(state);
        }

        // Silence threshold reached → finalize
        if state.silence_run >= ms_to_samples(SILENCE_TO_FINALIZE_MS) {
            return finalize(state);
        }
    } else {
        // Maintain pre-roll
        for &s in chunk {
            if state.pre_roll.len() == pre_roll_capacity {
                state.pre_roll.pop_front();
            }
            state.pre_roll.push_back(s);
        }
        if is_voiced {
            // Speech onset
            state.in_speech = true;
            state.silence_run = 0;
            // Bake the pre-roll into the utterance
            state.utterance.extend(state.pre_roll.iter().copied());
            state.pre_roll.clear();
            state.utterance_started_at = Some(Local::now());
            // offset ≈ samples seen so far, minus the pre-roll we just folded in
            let pre_samples = state.utterance.len() as f64;
            state.utterance_offset_s = (now_total as f64 - pre_samples) / SR as f64;
            // Append the current voiced chunk (pre-roll already includes... no it doesn't include this chunk)
            state.utterance.extend_from_slice(chunk);
        }
    }
    None
}

fn finalize(state: &mut AudioState) -> Option<UtteranceJob> {
    if state.utterance.len() < ms_to_samples(MIN_UTTERANCE_MS) {
        // Too short to be real speech — drop it
        state.utterance.clear();
        state.in_speech = false;
        state.silence_run = 0;
        state.utterance_started_at = None;
        return None;
    }
    let id = state.next_id;
    state.next_id = state.next_id.saturating_add(1);
    let samples = std::mem::take(&mut state.utterance);
    let started_at = state.utterance_started_at.take().unwrap_or_else(Local::now);
    let offset_s = state.utterance_offset_s;
    state.in_speech = false;
    state.silence_run = 0;
    Some(UtteranceJob {
        id,
        samples,
        started_at,
        offset_s,
    })
}

// --- Recording lifecycle ---

struct RecorderHandles {
    _stream: Stream,
    /// Set to false to ask the worker to drain and exit.
    worker_alive: Arc<AtomicBool>,
}

fn start_recording(
    audio_state: Arc<Mutex<AudioState>>,
    job_tx: Sender<UtteranceJob>,
) -> Result<Stream, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("no default input device")?;
    let config = StreamConfig {
        channels: 1,
        sample_rate: SampleRate(SR),
        buffer_size: cpal::BufferSize::Fixed(1024),
    };
    {
        let mut s = audio_state.lock().unwrap();
        s.reset_session();
    }
    let state = audio_state.clone();
    let stream = device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut st = state.lock().unwrap();
                if let Some(job) = process_chunk(&mut st, data) {
                    drop(st);
                    if let Err(e) = job_tx.send(job) {
                        eprintln!("dropping utterance {}: channel disconnected", e.0.id);
                    }
                }
            },
            |err| eprintln!("audio stream error: {err}"),
            None,
        )
        .map_err(|e| format!("build_input_stream: {e}"))?;
    stream.play().map_err(|e| format!("stream.play: {e}"))?;
    Ok(stream)
}

fn spawn_worker(
    job_rx: Receiver<UtteranceJob>,
    result_tx: Sender<WorkerMsg>,
    alive: Arc<AtomicBool>,
    egui_ctx: egui::Context,
) {
    std::thread::spawn(move || {
        let client = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("worker: failed to build client: {e}");
                return;
            }
        };
        while alive.load(Ordering::Relaxed) {
            let job = match job_rx.recv_timeout(Duration::from_millis(250)) {
                Ok(j) => j,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            let id = job.id;
            let result = transcribe_one(&client, &job);
            let _ = result_tx.send(WorkerMsg::Finalized { id, result });
            egui_ctx.request_repaint();
        }
    });
}

fn transcribe_one(
    client: &reqwest::blocking::Client,
    job: &UtteranceJob,
) -> Result<AsrResponse, String> {
    let wav = samples_to_wav_bytes(&job.samples)?;
    let part = reqwest::blocking::multipart::Part::bytes(wav)
        .file_name(format!("utt-{}.wav", job.id))
        .mime_str("audio/wav")
        .map_err(|e| format!("part: {e}"))?;
    let form = reqwest::blocking::multipart::Form::new().part("audio", part);
    let resp = client
        .post(TRANSCRIBE_URL)
        .multipart(form)
        .send()
        .map_err(|e| format!("post: {e}"))?;
    if !resp.status().is_success() {
        let code = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(format!("server {code}: {body}"));
    }
    resp.json::<AsrResponse>().map_err(|e| format!("parse: {e}"))
}

// --- Persistence ---

fn journal_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("voice-journal")
}

fn samples_to_wav_bytes(samples: &[f32]) -> Result<Vec<u8>, String> {
    let mut buf: Vec<u8> = Vec::with_capacity(samples.len() * 2 + 44);
    {
        let cursor = Cursor::new(&mut buf);
        let spec = WavSpec {
            channels: 1,
            sample_rate: SR,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut w = WavWriter::new(cursor, spec).map_err(|e| format!("wav new: {e}"))?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            w.write_sample(v).map_err(|e| format!("wav write: {e}"))?;
        }
        w.finalize().map_err(|e| format!("wav finalize: {e}"))?;
    }
    Ok(buf)
}

fn append_to_daily(line: &str) {
    let dir = journal_dir();
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{}.md", Local::now().format("%Y-%m-%d")));
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

// --- UI ---

#[derive(Debug, Clone)]
enum UtteranceUi {
    Pending { started_at: DateTime<Local> },
    Done {
        started_at: DateTime<Local>,
        text: String,
        latency_ms: f64,
    },
    Failed {
        started_at: DateTime<Local>,
        error: String,
    },
}

#[derive(Debug, Clone)]
enum TimelineItem {
    SessionStart { at: DateTime<Local> },
    SessionStop { at: DateTime<Local> },
    Utterance { id: u64, state: UtteranceUi },
}

struct App {
    audio_state: Arc<Mutex<AudioState>>,
    stream: Option<Stream>,
    recording: bool,
    session_started: Option<DateTime<Local>>,
    session_header_written: bool,

    job_tx: Sender<UtteranceJob>,
    result_rx: Receiver<WorkerMsg>,
    worker_alive: Arc<AtomicBool>,

    /// Persistent across stop/start cycles. Cleared only by the Clear button.
    timeline: Vec<TimelineItem>,
    /// Maps utterance id → index in `timeline` so result deliveries can locate
    /// the correct row regardless of intervening SessionStart/Stop markers.
    by_id: HashMap<u64, usize>,
    /// Highest utterance id we've already pushed into the timeline.
    next_observed_id: u64,
    /// Currently in-flight utterances (dispatched, no result yet).
    pending_count: u64,

    status: String,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // --- Theme & font tweaks: pure-black panels, slightly larger text ---
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(8, 8, 8);
        visuals.window_fill = egui::Color32::from_rgb(8, 8, 8);
        visuals.extreme_bg_color = egui::Color32::BLACK;
        visuals.faint_bg_color = egui::Color32::from_rgb(16, 16, 16);
        visuals.override_text_color = Some(egui::Color32::from_rgb(225, 225, 225));
        cc.egui_ctx.set_visuals(visuals);

        let mut style = (*cc.egui_ctx.style()).clone();
        for (_, fid) in style.text_styles.iter_mut() {
            fid.size *= 1.15;
        }
        // Enable text selection on plain Labels so the user can highlight
        // any transcript line and Ctrl+C it to the clipboard.
        style.interaction.selectable_labels = true;
        cc.egui_ctx.set_style(style);

        let audio_state = Arc::new(Mutex::new(AudioState::new()));
        let (job_tx, job_rx) = channel::<UtteranceJob>();
        let (result_tx, result_rx) = channel::<WorkerMsg>();
        let alive = Arc::new(AtomicBool::new(true));
        spawn_worker(job_rx, result_tx, alive.clone(), cc.egui_ctx.clone());

        Self {
            audio_state,
            stream: None,
            recording: false,
            session_started: None,
            session_header_written: false,
            job_tx,
            result_rx,
            worker_alive: alive,
            timeline: Vec::new(),
            by_id: HashMap::new(),
            next_observed_id: 0,
            pending_count: 0,
            status: format!("Ready. Endpoint: {TRANSCRIBE_URL}"),
        }
    }

    fn toggle_recording(&mut self) {
        if self.recording {
            self.stop();
        } else {
            self.start();
        }
    }

    fn start(&mut self) {
        // Don't wipe the timeline — sessions accumulate. Push a marker so
        // the UI can show where this session begins.
        let now = Local::now();
        match start_recording(self.audio_state.clone(), self.job_tx.clone()) {
            Ok(stream) => {
                self.stream = Some(stream);
                self.recording = true;
                self.timeline.push(TimelineItem::SessionStart { at: now });
                self.session_started = Some(now);
                append_to_daily(&format!("\n## ▶ session start · {}", now.format("%H:%M:%S")));
                self.status = "Recording — speak; pauses end utterances.".into();
            }
            Err(e) => {
                self.status = format!("Mic error: {e}");
            }
        }
    }

    fn stop(&mut self) {
        if let Some(s) = self.stream.take() {
            drop(s);
        }
        // Force-flush any in-progress utterance the VAD is still holding
        if let Some(job) = {
            let mut st = self.audio_state.lock().unwrap();
            if st.in_speech {
                finalize(&mut *st)
            } else {
                None
            }
        } {
            let _ = self.job_tx.send(job);
        }
        self.recording = false;
        let now = Local::now();
        self.timeline.push(TimelineItem::SessionStop { at: now });
        append_to_daily(&format!("_■ session stop · {}_\n", now.format("%H:%M:%S")));
        self.status = "Stopped — pending transcripts will land as they finish.".into();
    }

    fn drain_results(&mut self) {
        while let Ok(WorkerMsg::Finalized { id, result }) = self.result_rx.try_recv() {
            let Some(&idx) = self.by_id.get(&id) else {
                continue;
            };
            let TimelineItem::Utterance { state: prev, .. } = &self.timeline[idx] else {
                continue;
            };
            let started_at = match prev {
                UtteranceUi::Pending { started_at }
                | UtteranceUi::Done { started_at, .. }
                | UtteranceUi::Failed { started_at, .. } => *started_at,
            };
            let new_state = match result {
                Ok(resp) => {
                    let text = if resp.segments.is_empty() {
                        String::from("(no speech)")
                    } else {
                        resp.segments
                            .iter()
                            .map(|s| s.text.trim())
                            .filter(|t| !t.is_empty())
                            .collect::<Vec<_>>()
                            .join(" ")
                    };
                    // Append to today's markdown
                    if !self.session_header_written {
                        if let Some(s) = self.session_started {
                            append_to_daily(&format!("\n## {}", s.format("%Y-%m-%d %H:%M:%S")));
                        }
                        self.session_header_written = true;
                    }
                    append_to_daily(&format!("- `{}` {text}", started_at.format("%H:%M:%S")));
                    UtteranceUi::Done {
                        started_at,
                        text,
                        latency_ms: resp.latency_ms,
                    }
                }
                Err(error) => UtteranceUi::Failed { started_at, error },
            };
            self.timeline[idx] = TimelineItem::Utterance { id, state: new_state };
            self.pending_count = self.pending_count.saturating_sub(1);
        }
    }

    /// Polled each frame: the audio thread bumps `next_id` when it emits an
    /// utterance, so we can backfill a Pending row immediately and display
    /// "transcribing…" until the worker result lands.
    fn observe_dispatches(&mut self) {
        let next_id = self.audio_state.lock().unwrap().next_id;
        while self.next_observed_id < next_id {
            let id = self.next_observed_id;
            let idx = self.timeline.len();
            self.timeline.push(TimelineItem::Utterance {
                id,
                state: UtteranceUi::Pending {
                    started_at: Local::now(),
                },
            });
            self.by_id.insert(id, idx);
            self.next_observed_id = self.next_observed_id.saturating_add(1);
            self.pending_count = self.pending_count.saturating_add(1);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.recording || self.pending_count > 0 {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
        self.observe_dispatches();
        self.drain_results();

        let live_rms = self.audio_state.lock().unwrap().last_rms;

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let label = if self.recording {
                    "⏹  Stop"
                } else {
                    "●  Record"
                };
                let btn = egui::Button::new(egui::RichText::new(label).strong())
                    .min_size(egui::vec2(120.0, 32.0));
                if ui.add(btn).clicked() {
                    self.toggle_recording();
                }

                if ui
                    .add_enabled(!self.recording, egui::Button::new("Open Today"))
                    .clicked()
                {
                    let path = journal_dir()
                        .join(format!("{}.md", Local::now().format("%Y-%m-%d")));
                    if let Err(e) = open_in_editor(&path) {
                        self.status = format!("Open failed: {e}");
                    }
                }

                if ui
                    .add_enabled(!self.recording, egui::Button::new("Clear"))
                    .clicked()
                {
                    self.timeline.clear();
                    self.by_id.clear();
                    // Don't reset next_observed_id — IDs from the audio thread
                    // are monotonic and we'd lose pending results otherwise.
                    self.pending_count = 0;
                    self.session_header_written = false;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.recording {
                        if let Some(t) = self.audio_state.lock().unwrap().rec_started {
                            let secs = t.elapsed().as_secs();
                            ui.label(
                                egui::RichText::new(format!(
                                    "● {:02}:{:02}",
                                    secs / 60,
                                    secs % 60
                                ))
                                .color(egui::Color32::from_rgb(220, 60, 60))
                                .strong()
                                .monospace(),
                            );
                        }
                    } else if self.pending_count > 0 {
                        ui.label(format!("transcribing {} pending…", self.pending_count));
                    }
                });
            });

            // Live mic level meter
            ui.horizontal(|ui| {
                ui.label("Level:");
                let bar_w = 300.0;
                let level = (live_rms / 0.10).clamp(0.0, 1.0);
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(bar_w, 12.0), egui::Sense::hover());
                let painter = ui.painter();
                painter.rect_filled(
                    rect,
                    2.0,
                    egui::Color32::from_rgb(40, 40, 40),
                );
                let mut filled = rect;
                filled.set_width(rect.width() * level);
                let voiced = live_rms >= VAD_RMS_THRESHOLD;
                let color = if voiced {
                    egui::Color32::from_rgb(60, 200, 90)
                } else {
                    egui::Color32::from_rgb(120, 120, 120)
                };
                painter.rect_filled(filled, 2.0, color);
                ui.monospace(format!("{:.4}", live_rms));
            });
            ui.add_space(4.0);
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.add_space(2.0);
            ui.label(&self.status);
            ui.add_space(2.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    if self.timeline.is_empty() {
                        ui.weak("Press Record. Each pause finalizes a sentence.");
                    }
                    let timeline_len = self.timeline.len();
                    for idx in 0..timeline_len {
                        let item = &self.timeline[idx];
                        let response = match item {
                            TimelineItem::SessionStart { at } => {
                                ui.scope(|ui| {
                                    draw_session_separator(ui, at, "▶ session start");
                                })
                                .response
                            }
                            TimelineItem::SessionStop { at } => {
                                ui.scope(|ui| {
                                    draw_session_separator(ui, at, "■ session stop");
                                })
                                .response
                            }
                            TimelineItem::Utterance { state: utt, .. } => {
                                ui.horizontal_top(|ui| match utt {
                                    UtteranceUi::Pending { started_at, .. } => {
                                        ui.add_sized(
                                            [90.0, 0.0],
                                            egui::Label::new(
                                                egui::RichText::new(format!(
                                                    "[{}]",
                                                    started_at.format("%H:%M:%S")
                                                ))
                                                .monospace()
                                                .color(egui::Color32::from_rgb(140, 140, 140)),
                                            ),
                                        );
                                        ui.add(egui::Spinner::new());
                                        ui.label(
                                            egui::RichText::new("transcribing…")
                                                .italics()
                                                .color(egui::Color32::from_rgb(160, 160, 160)),
                                        );
                                    }
                                    UtteranceUi::Done {
                                        started_at,
                                        text,
                                        latency_ms,
                                        ..
                                    } => {
                                        ui.add_sized(
                                            [90.0, 0.0],
                                            egui::Label::new(
                                                egui::RichText::new(format!(
                                                    "[{}]",
                                                    started_at.format("%H:%M:%S")
                                                ))
                                                .monospace()
                                                .color(egui::Color32::from_rgb(120, 200, 220)),
                                            ),
                                        );
                                        ui.add(
                                            egui::Label::new(format!(
                                                "{text}  ({:.0}ms)",
                                                latency_ms
                                            ))
                                            .wrap(),
                                        );
                                    }
                                    UtteranceUi::Failed {
                                        started_at, error, ..
                                    } => {
                                        ui.add_sized(
                                            [90.0, 0.0],
                                            egui::Label::new(
                                                egui::RichText::new(format!(
                                                    "[{}]",
                                                    started_at.format("%H:%M:%S")
                                                ))
                                                .monospace()
                                                .color(egui::Color32::from_rgb(220, 100, 100)),
                                            ),
                                        );
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(format!("✗ {error}"))
                                                    .color(egui::Color32::from_rgb(
                                                        220, 120, 120,
                                                    )),
                                            )
                                            .wrap(),
                                        );
                                    }
                                })
                                .response
                            }
                        };
                        // Hover-only responses ignore right-clicks; promote to
                        // click-sensitive so the context menu actually fires.
                        let response = response.interact(egui::Sense::click());
                        let timeline_ref = &self.timeline;
                        response.context_menu(|ui| {
                            if ui.button("Copy session").clicked() {
                                let text = collect_session_text(timeline_ref, idx);
                                ui.ctx().copy_text(text);
                                ui.close_menu();
                            }
                            if let TimelineItem::Utterance {
                                state: UtteranceUi::Done { text, started_at, .. },
                                ..
                            } = &timeline_ref[idx]
                            {
                                if ui.button("Copy line").clicked() {
                                    ui.ctx().copy_text(format!(
                                        "[{}] {text}",
                                        started_at.format("%H:%M:%S")
                                    ));
                                    ui.close_menu();
                                }
                            }
                            if ui.button("Copy all").clicked() {
                                let text = collect_all_text(timeline_ref);
                                ui.ctx().copy_text(text);
                                ui.close_menu();
                            }
                        });
                    }
                });
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.worker_alive.store(false, Ordering::Relaxed);
    }
}

/// Extract the text of every finalized utterance in the session containing
/// `from_idx`. Walks backward to the most recent SessionStart and forward to
/// the next SessionStop (or end of timeline).
fn collect_session_text(timeline: &[TimelineItem], from_idx: usize) -> String {
    if timeline.is_empty() {
        return String::new();
    }
    // If the user right-clicked on a SessionStop, jump back one so we walk
    // into the session it closes.
    let mut start = from_idx.min(timeline.len() - 1);
    if matches!(timeline[start], TimelineItem::SessionStop { .. }) && start > 0 {
        start -= 1;
    }
    while start > 0 && !matches!(timeline[start], TimelineItem::SessionStart { .. }) {
        start -= 1;
    }
    let mut end = (start + 1).min(timeline.len());
    while end < timeline.len() {
        if matches!(timeline[end], TimelineItem::SessionStop { .. }) {
            break;
        }
        end += 1;
    }
    let mut out = String::new();
    for item in &timeline[start..end] {
        if let TimelineItem::Utterance {
            state: UtteranceUi::Done {
                started_at, text, ..
            },
            ..
        } = item
        {
            out.push_str(&format!("[{}] {}\n", started_at.format("%H:%M:%S"), text));
        }
    }
    out
}

/// Concatenate every finalized utterance from the entire timeline. Includes
/// `## session ...` headers so multi-session copies still read sensibly.
fn collect_all_text(timeline: &[TimelineItem]) -> String {
    let mut out = String::new();
    for item in timeline {
        match item {
            TimelineItem::SessionStart { at } => {
                out.push_str(&format!(
                    "\n## ▶ session start · {}\n",
                    at.format("%H:%M:%S")
                ));
            }
            TimelineItem::SessionStop { at } => {
                out.push_str(&format!("■ session stop · {}\n", at.format("%H:%M:%S")));
            }
            TimelineItem::Utterance {
                state: UtteranceUi::Done {
                    started_at, text, ..
                },
                ..
            } => {
                out.push_str(&format!("[{}] {}\n", started_at.format("%H:%M:%S"), text));
            }
            _ => {}
        }
    }
    out
}

fn draw_session_separator(ui: &mut egui::Ui, at: &DateTime<Local>, label: &str) {
    ui.add_space(8.0);
    let avail = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(avail, 1.0), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 0.0, egui::Color32::from_rgb(45, 55, 75));
    ui.add_space(2.0);
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("{label} · {}", at.format("%H:%M:%S")))
                .color(egui::Color32::from_rgb(120, 150, 180))
                .monospace()
                .small(),
        );
    });
    ui.add_space(2.0);
}

fn open_in_editor(path: &std::path::Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("not yet — {} doesn't exist", path.display()));
    }
    std::process::Command::new("xdg-open")
        .arg(path)
        .spawn()
        .map_err(|e| format!("xdg-open: {e}"))?;
    Ok(())
}

fn _unused_ref_to_keep_constants() -> f32 {
    // Keep SR_F referenced in case future tunables use it.
    SR_F
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 650.0])
            .with_title("Voice Journal — Live"),
        ..Default::default()
    };
    eframe::run_native(
        "Voice Journal — Live",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
