//! Copy-only sessions using Voice Journal's VAD and existing ASR endpoints.
use super::*;
use std::collections::VecDeque;
use std::io::BufRead;
use std::os::unix::fs::OpenOptionsExt;
use whisper_typer_rs::{config::Config, service::VoiceCorrections, spelling::SpellCorrector};

pub(super) fn filter_recordings() -> Result<(), Box<dyn std::error::Error>> {
    // Reuse the exact journal rules for restored UI transcripts. No microphone,
    // model calls, file edits, or transcript content in diagnostic output.
    let input: serde_json::Value = serde_json::from_reader(io::stdin().lock())?;
    let filters = load_hallucination_filters();
    let mut rejected = serde_json::Map::new();
    for session in input.as_array().ok_or("Expected session list")? {
        let identity = session["id"].as_str().ok_or("Missing session ID")?;
        let ids: Vec<_> = session["chunks"]
            .as_array()
            .ok_or("Missing chunks")?
            .iter()
            .filter(|chunk| is_hallucination(chunk["text"].as_str().unwrap_or(""), &filters))
            .map(|chunk| chunk["seq"].clone())
            .collect();
        rejected.insert(identity.to_string(), json!(ids));
    }
    println!("{}", serde_json::Value::Object(rejected));
    Ok(())
}

struct Segmenter {
    pending: VecDeque<f32>,
    preroll: VecDeque<f32>,
    speech: Vec<f32>,
    voiced: usize,
    streak: usize,
    silence: usize,
    mode: VadMode,
    last_rms: f32,
    last_probability: Option<f32>,
    speech_started_at: Option<chrono::DateTime<Local>>,
    noise_floor: f32,
    noise_observations: usize,
}
struct AudioChunk {
    samples: Vec<f32>,
    started_at: chrono::DateTime<Local>,
    ended_at: chrono::DateTime<Local>,
    end_reason: &'static str,
}
impl Segmenter {
    fn new(mode: VadMode) -> Self {
        Self {
            pending: VecDeque::new(),
            preroll: VecDeque::new(),
            speech: Vec::new(),
            voiced: 0,
            streak: 0,
            silence: 0,
            mode,
            last_rms: 0.0,
            last_probability: None,
            speech_started_at: None,
            noise_floor: 0.0,
            noise_observations: 0,
        }
    }
    fn feed(&mut self, samples: &[f32]) -> Vec<AudioChunk> {
        self.pending.extend(samples);
        let mut chunks = Vec::new();
        while self.pending.len() >= vad::FRAME_SAMPLES {
            let frame: Vec<_> = self.pending.drain(..vad::FRAME_SAMPLES).collect();
            let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
            self.last_rms = rms;
            let (probability, bar, rescue) = match &self.mode {
                VadMode::Rms => (None, 1.0, VAD_RMS_THRESHOLD),
                VadMode::Silero {
                    detector,
                    threshold,
                    stay_threshold,
                    rms_rescue_threshold,
                } => {
                    let bar = if self.speech.is_empty() {
                        *threshold
                    } else {
                        *stay_threshold
                    };
                    (detector.predict(&frame).ok(), bar, *rms_rescue_threshold)
                }
            };
            self.last_probability = probability;
            let voiced = self.classify(rms, probability, bar, rescue);
            if let Some(chunk) = self.frame(&frame, voiced) {
                chunks.push(chunk);
            }
        }
        chunks
    }
    fn classify(&mut self, rms: f32, probability: Option<f32>, bar: f32, rescue: f32) -> bool {
        // Learn from confident non-speech frames, never from identified speech.
        // Fast downward / slow upward adaptation follows changing white noise
        // without letting a brief syllable rapidly raise the fallback threshold.
        if probability
            .map(|p| p < 0.1)
            .unwrap_or(self.speech.is_empty())
        {
            if self.noise_observations == 0 {
                self.noise_floor = rms;
            } else {
                let rate = if rms < self.noise_floor { 0.2 } else { 0.02 };
                self.noise_floor += (rms - self.noise_floor) * rate;
            }
            self.noise_observations += 1;
        }
        probability
            .map(|p| p >= bar)
            .unwrap_or(rms >= rescue.max(self.noise_floor * 2.5))
    }
    fn frame(&mut self, frame: &[f32], voiced: bool) -> Option<AudioChunk> {
        if self.speech.is_empty() {
            self.preroll.extend(frame);
            while self.preroll.len() > ms_to_samples(PRE_ROLL_MS) {
                self.preroll.pop_front();
            }
            self.streak = if voiced { self.streak + 1 } else { 0 };
            if self.streak < SILERO_SPEECH_START_FRAMES as usize {
                return None;
            }
            self.speech.extend(self.preroll.drain(..));
            self.speech_started_at = Some(
                Local::now()
                    - chrono::Duration::milliseconds(
                        (self.speech.len() as u64 * 1000 / SAMPLE_RATE as u64) as i64,
                    ),
            );
            self.voiced = self.streak * frame.len();
        } else {
            self.speech.extend_from_slice(frame);
            if voiced {
                self.voiced += frame.len();
            }
        }
        self.silence = if voiced {
            0
        } else {
            self.silence + frame.len()
        };
        if self.silence >= ms_to_samples(SILENCE_FINALIZE_MS)
            || self.speech.len() >= ms_to_samples(MAX_UTTERANCE_MS)
        {
            return self.take_chunk(if self.silence >= ms_to_samples(SILENCE_FINALIZE_MS) {
                "pause"
            } else {
                "limit"
            });
        }
        None
    }
    fn finish(&mut self) -> Option<AudioChunk> {
        // Stop includes the last partial frame instead of discarding the tail.
        if !self.speech.is_empty() {
            self.speech.extend(self.pending.drain(..));
        }
        self.take_chunk("stop")
    }
    fn take_chunk(&mut self, end_reason: &'static str) -> Option<AudioChunk> {
        let chunk = std::mem::take(&mut self.speech);
        let enough = self.voiced >= ms_to_samples(MIN_VOICED_MS);
        self.voiced = 0;
        self.streak = 0;
        self.silence = 0;
        self.preroll.clear();
        let ended_at = Local::now();
        let started_at = self.speech_started_at.take().unwrap_or(ended_at);
        if enough && !chunk.is_empty() {
            Some(AudioChunk {
                samples: chunk,
                started_at,
                ended_at,
                end_reason,
            })
        } else {
            None
        }
    }
    fn activity(&self) -> serde_json::Value {
        json!({"state":if self.speech.is_empty() { "listening" } else if self.silence>0 { "silence" } else { "speaking" },
               "silence_ms":self.silence as f64*1000.0/SAMPLE_RATE as f64,
               "rms":self.last_rms,"speech_probability":self.last_probability,
               "noise_floor":self.noise_floor,
               "vad_mode":if matches!(&self.mode,VadMode::Silero{..}) { "Silero" } else { "RMS" }})
    }
}

struct Events {
    session: String,
    file: Mutex<File>,
    journal: Mutex<RotatingJournalWriter>,
}
impl Events {
    fn activity(&self, activity: serde_json::Value) {
        // Meter events are ephemeral; do not grow journal/session files at 4 Hz.
        let _file = self.file.lock().unwrap();
        let event = json!({"type":"recording","payload":{"session_id":self.session,
            "timestamp":Local::now().to_rfc3339(),"status":"activity","activity":activity}});
        let stdout = io::stdout();
        let mut out = stdout.lock();
        let _ = writeln!(out, "{event}").and_then(|_| out.flush());
    }
    fn emit(&self, status: &str, details: serde_json::Value) {
        let mut payload = details;
        payload["session_id"] = json!(self.session);
        payload["timestamp"] = json!(Local::now().to_rfc3339());
        payload["status"] = json!(status);
        let mut event = json!({"type":"recording", "payload":payload});
        // Serialize file and pipe writes together so start/chunk/stop stay ordered.
        let mut file = self.file.lock().unwrap();
        let ts = Local::now().format("%H:%M:%S");
        let label = format!("[{ts}] [recording {}", self.session);
        let payload = &event["payload"];
        let saved = if status == "chunk" || status == "filtered" {
            let seq = payload["seq"].as_u64().unwrap_or(0);
            let mut journal = self.journal.lock().unwrap();
            let raw_saved = journal.write_unfiltered(&format!(
                "{label} #{seq}] {}",
                payload["raw_text"].as_str().unwrap_or("")
            ));
            if status == "filtered" {
                raw_saved
            } else {
                raw_saved.and_then(|_| {
                    journal.write_journal(&format!(
                        "{label} #{seq}] {}",
                        payload["text"].as_str().unwrap_or("")
                    ))
                })
            }
        } else if matches!(status, "started" | "stopped" | "error") {
            self.journal.lock().unwrap().write_journal(&format!(
                "{label} {status}]{}",
                payload["message"]
                    .as_str()
                    .map(|m| format!(" {m}"))
                    .unwrap_or_default()
            ))
        } else {
            Ok(())
        };
        if saved.is_err() {
            eprintln!("Cannot append recording to daily Voice Journal");
        }
        event["payload"]["journal_saved"] = json!(saved.is_ok());
        let line = serde_json::to_string(&event).unwrap();
        if writeln!(file, "{line}").and_then(|_| file.flush()).is_err() {
            eprintln!("Cannot persist recording event");
        }
        let stdout = io::stdout();
        let mut out = stdout.lock();
        let _ = writeln!(out, "{line}").and_then(|_| out.flush());
    }
}

fn argument(name: &str) -> Option<String> {
    let args: Vec<_> = std::env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let session = argument("--ui-session").ok_or("missing session ID")?;
    let output = PathBuf::from(argument("--output").ok_or("missing output path")?);
    let config_path = argument("--config").map(PathBuf::from);
    let config = Config::load(config_path.as_deref())?;
    let _recording_guard = if argument("--input-wav").is_none() {
        Some(whisper_typer_rs::recording::RecordingGuard::acquire()?)
    } else {
        None
    };
    let journal = RotatingJournalWriter::open_at(
        argument("--journal-dir")
            .map(PathBuf::from)
            .unwrap_or_else(voice_journal_dir),
    )?;
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&output)?;
    let events = Arc::new(Events {
        session,
        file: Mutex::new(file),
        journal: Mutex::new(journal),
    });
    if let Err(error) = capture(config, &output, events.clone()) {
        events.emit("error", json!({"message":error}));
    }
    events.emit("stopped", json!({}));
    Ok(())
}

fn capture(config: Config, output: &Path, events: Arc<Events>) -> Result<(), String> {
    // Persistent clients and the existing warm ASR server; no Whisper model is loaded here.
    let remote = reqwest::blocking::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_millis(750))
        .timeout(Duration::from_secs(
            config.remote_asr.timeout_seconds.max(1),
        ))
        .build()
        .map_err(|e| e.to_string())?;
    let local = reqwest::blocking::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(35))
        .build()
        .map_err(|e| e.to_string())?;
    let punctuation = reqwest::blocking::Client::builder()
        .no_proxy()
        .timeout(Duration::from_millis(config.punctuation.timeout_ms.max(1)))
        .build()
        .map_err(|e| e.to_string())?;
    let (tx, rx) = mpsc::sync_channel::<AudioChunk>(32);
    let worker_events = events.clone();
    let audio_prefix = output.with_extension("");
    let worker = std::thread::spawn(move || {
        // Index spelling rules while capture starts, rather than delaying the
        // microphone until the dictionaries have finished loading.
        let corrections = VoiceCorrections::load(&config);
        let spelling = SpellCorrector::load(&config.spelling);
        let hallucination_filters = load_hallucination_filters();
        for (index, chunk) in rx.into_iter().enumerate() {
            let samples = &chunk.samples;
            let seq = index + 1;
            worker_events.activity(json!({"transcribing":true}));
            let audio = audio_prefix.with_extension(format!("{seq}.wav"));
            // Keep a recoverable WAV if a service fails; remove it after successful ASR.
            let saved = wav_bytes(&samples).and_then(|bytes| {
                let mut file = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .mode(0o600)
                    .open(&audio)
                    .map_err(|e| e.to_string())?;
                file.write_all(&bytes).map_err(|e| e.to_string())
            });
            if saved.is_err() {
                worker_events.emit("error", json!({"message":"Cannot save recording audio"}));
            }
            let started = Instant::now();
            let raw = if config.remote_asr.enabled {
                match transcribe_at(&remote, &samples, &config.remote_asr.url) {
                    Ok(text) => Ok(text),
                    Err(_) if config.remote_asr.fallback_local => {
                        transcribe_at(&local, &samples, LOCAL_TRANSCRIBE_URL)
                    }
                    Err(error) => Err(error),
                }
            } else {
                transcribe_at(&local, &samples, LOCAL_TRANSCRIBE_URL)
            };
            match raw {
                Ok(raw) if !raw.trim().is_empty() => {
                    if is_hallucination(&raw, &hallucination_filters) {
                        worker_events.emit(
                            "filtered",
                            json!({"seq":seq,"raw_text":raw,"end_reason":chunk.end_reason}),
                        );
                        worker_events.activity(json!({"transcribing":false,"filtered":true}));
                        let _ = fs::remove_file(&audio);
                        continue;
                    }
                    let mut text = corrections.apply(&raw);
                    if let Some(spelling) = &spelling {
                        text = spelling
                            .apply(&text, corrections.protectors())
                            .text
                            .into_owned();
                    }
                    if config.punctuation.enabled {
                        let repaired = punctuate_at(&punctuation, &text, &config.punctuation.url)
                            .or_else(|error| match &config.punctuation.fallback_url {
                                Some(url) => punctuate_at(&punctuation, &text, url),
                                None => Err(error),
                            });
                        if let Ok(repaired) = repaired {
                            text = repaired;
                        }
                    }
                    worker_events.emit(
                        "chunk",
                        json!({"seq":seq,"text":text,"raw_text":raw,
                        "speech_started_at":chunk.started_at.to_rfc3339(),"speech_ended_at":chunk.ended_at.to_rfc3339(),
                        "end_reason":chunk.end_reason,"audio_ms":samples.len() as f64*1000.0/SAMPLE_RATE as f64,
                        "latency_ms":started.elapsed().as_secs_f64()*1000.0}),
                    );
                    let _ = fs::remove_file(&audio);
                }
                _ => worker_events.emit(
                    "error",
                    json!({"message":"Transcription failed; audio saved for recovery", "seq":seq,
                    "audio_path":audio.to_string_lossy()}),
                ),
            }
            worker_events.activity(json!({"transcribing":false}));
        }
    });
    let segmenter = Arc::new(Mutex::new(Segmenter::new(build_vad_mode())));
    // Deterministic replay uses the same segmentation/transcription path without
    // opening a microphone or changing the active desktop's clipboard.
    if let Some(path) = argument("--input-wav") {
        let result = (|| {
            let mut wav = hound::WavReader::open(path).map_err(|e| e.to_string())?;
            if wav.spec().sample_rate != SAMPLE_RATE
                || wav.spec().channels != 1
                || wav.spec().bits_per_sample != 16
                || wav.spec().sample_format != SampleFormat::Int
            {
                return Err("Replay requires mono 16-bit PCM at 16 kHz".to_string());
            }
            events.emit("started", json!({}));
            let mut block = Vec::with_capacity(1024);
            for sample in wav.samples::<i16>() {
                block.push(sample.map_err(|e| e.to_string())? as f32 / i16::MAX as f32);
                if block.len() == 1024 {
                    for chunk in segmenter.lock().unwrap().feed(&block) {
                        tx.send(chunk).map_err(|e| e.to_string())?;
                    }
                    block.clear();
                }
            }
            for chunk in segmenter.lock().unwrap().feed(&block) {
                tx.send(chunk).map_err(|e| e.to_string())?;
            }
            events.emit("finishing", json!({}));
            if let Some(chunk) = segmenter.lock().unwrap().finish() {
                tx.send(chunk).map_err(|e| e.to_string())?;
            }
            Ok(())
        })();
        drop(tx);
        worker.join().map_err(|_| "Transcription worker failed")?;
        return result;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let capture_error = Arc::new(Mutex::new(None::<String>));
    let stream_result = (|| {
        let device = cpal::default_host()
            .default_input_device()
            .ok_or("No microphone found")?;
        let cfg = StreamConfig {
            channels: 1,
            sample_rate: SampleRate(SAMPLE_RATE),
            buffer_size: cpal::BufferSize::Fixed(1024),
        };
        let state = segmenter.clone();
        let sender = tx.clone();
        let callback_stop = stop.clone();
        let callback_error = capture_error.clone();
        let stream_error = capture_error.clone();
        let error_stop = stop.clone();
        let stream = device
            .build_input_stream(
                &cfg,
                move |samples: &[f32], _| {
                    if callback_stop.load(Ordering::Relaxed) {
                        return;
                    }
                    for chunk in state.lock().unwrap().feed(samples) {
                        if sender.try_send(chunk).is_err() {
                            *callback_error.lock().unwrap() =
                                Some("Transcription queue full; recording stopped".into());
                            callback_stop.store(true, Ordering::Relaxed);
                        }
                    }
                },
                move |_| {
                    *stream_error.lock().unwrap() = Some("Microphone disconnected".into());
                    error_stop.store(true, Ordering::Relaxed);
                },
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        Ok::<_, String>(stream)
    })();
    match stream_result {
        Ok(stream) => {
            events.emit("started", json!({}));
            let stdin_stop = stop.clone();
            std::thread::spawn(move || {
                let mut line = String::new();
                let _ = io::stdin().lock().read_line(&mut line);
                stdin_stop.store(true, Ordering::Relaxed);
            });
            while !stop.load(Ordering::Relaxed) {
                let activity = segmenter.lock().unwrap().activity();
                events.activity(activity);
                std::thread::sleep(Duration::from_millis(250));
            }
            let _ = stream.pause();
            drop(stream); // Stop microphone before draining the transcription queue.
            events.emit("finishing", json!({}));
            if let Some(chunk) = segmenter.lock().unwrap().finish() {
                let _ = tx.send(chunk);
            }
        }
        Err(error) => {
            *capture_error.lock().unwrap() = Some(error);
        }
    }
    drop(tx);
    worker.join().map_err(|_| "Transcription worker failed")?;
    let result = capture_error.lock().unwrap().take();
    if let Some(error) = result {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silence_never_creates_transcripts() {
        let mut state = Segmenter::new(VadMode::Rms);
        assert!(state.feed(&vec![0.0; SAMPLE_RATE as usize * 2]).is_empty());
        assert!(state.finish().is_none());
    }
    #[test]
    fn stop_flushes_partial_frame_and_speech_tail() {
        let mut state = Segmenter::new(VadMode::Rms);
        state.feed(&vec![0.0; 16000]);
        assert!(state.feed(&vec![0.1; 16000 + 123]).is_empty());
        let chunk = state.finish().unwrap();
        assert!(chunk.samples.len() >= 16000 + 123);
        assert!(
            chunk.samples.len() <= 16000 + 123 + ms_to_samples(PRE_ROLL_MS) + vad::FRAME_SAMPLES
        );
        assert!(chunk.samples[chunk.samples.len() - 123..]
            .iter()
            .all(|sample| *sample == 0.1));
        assert_eq!(chunk.end_reason, "stop");
        assert!(state.finish().is_none());
    }
    #[test]
    fn pauses_and_long_speech_create_separate_bounded_chunks() {
        let mut state = Segmenter::new(VadMode::Rms);
        state.feed(&vec![0.0; 16000]);
        state.feed(&vec![0.1; 16000]);
        let pauses = state.feed(&vec![0.0; 16000]);
        assert_eq!(pauses.len(), 1);
        assert_eq!(pauses[0].end_reason, "pause");
        let chunks = state.feed(&vec![0.1; SAMPLE_RATE as usize * 30]);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].samples.len() <= ms_to_samples(MAX_UTTERANCE_MS) + vad::FRAME_SAMPLES);
        assert_eq!(chunks[0].end_reason, "limit");
        assert!(state.finish().is_some());
    }
    #[test]
    fn adaptive_floor_rejects_steady_noise_without_overriding_speech() {
        let mut state = Segmenter::new(VadMode::Rms);
        for _ in 0..100 {
            assert!(!state.classify(0.04, Some(0.01), 0.5, 0.012));
        }
        assert!(!state.classify(0.04, None, 0.5, 0.012));
        assert!(state.classify(0.03, Some(0.9), 0.5, 0.012));
        assert!(state.classify(0.12, None, 0.5, 0.012));
        for _ in 0..100 {
            state.classify(0.01, Some(0.01), 0.5, 0.012);
        }
        assert!(state.noise_floor < 0.011);
    }
    #[test]
    fn silero_detects_pauses_in_speech_with_continuous_white_noise() {
        let Ok(path) = std::env::var("WHISPER_VAD_TEST_WAV") else {
            return;
        };
        let mut wav = hound::WavReader::open(path).unwrap();
        let mut state = Segmenter::new(VadMode::Silero {
            detector: Arc::new(SileroVad::load(Path::new("models/silero_vad.onnx")).unwrap()),
            threshold: 0.5,
            stay_threshold: 0.35,
            rms_rescue_threshold: 0.012,
        });
        let mut seed = 17u32;
        let mut noise = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32 * 0.025
        };
        let lead: Vec<_> = (0..48000).map(|_| noise()).collect();
        assert!(state.feed(&lead).is_empty());
        let speech: Vec<_> = wav
            .samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / 32768.0 + noise())
            .collect();
        let mut chunks = state.feed(&speech);
        let tail: Vec<_> = (0..48000).map(|_| noise()).collect();
        chunks.extend(state.feed(&tail));
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|chunk| chunk.end_reason == "pause"));
        assert!(
            state.finish().is_none(),
            "White noise kept an utterance open"
        );
        assert!(
            state.noise_floor > 0.012,
            "Background was louder than the old fixed gate"
        );
    }
}
