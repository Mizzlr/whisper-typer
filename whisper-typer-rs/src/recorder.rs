//! Audio recording with cpal.
//!
//! Keeps the audio stream open for low-latency recording start.
//! Captures 16kHz mono f32 audio suitable for Whisper.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::config::{AudioConfig, RecordingConfig, SilenceConfig};

/// Audio recorder with always-open stream for low-latency start.
pub struct AudioRecorder {
    config: AudioConfig,
    #[allow(dead_code)]
    recording_config: RecordingConfig,
    #[allow(dead_code)]
    silence_config: SilenceConfig,
    /// Shared state between the audio callback thread and the main thread.
    shared: Arc<SharedState>,
    /// The cpal stream handle. Kept alive to maintain the always-open stream.
    _stream: Option<Stream>,
}

struct SharedState {
    inner: Mutex<RecorderInner>,
}

struct RecorderInner {
    is_recording: bool,
    buffer: Vec<f32>,
    max_samples: usize,
    // Silence detection state
    silence_start: Option<Instant>,
    recording_start: Option<Instant>,
    should_auto_stop: bool,
}

impl AudioRecorder {
    pub fn new(
        audio_config: AudioConfig,
        recording_config: RecordingConfig,
        silence_config: SilenceConfig,
    ) -> Self {
        let max_samples =
            (recording_config.max_duration * audio_config.sample_rate as f64) as usize;

        let shared = Arc::new(SharedState {
            inner: Mutex::new(RecorderInner {
                is_recording: false,
                buffer: Vec::with_capacity(max_samples),
                max_samples,
                silence_start: None,
                recording_start: None,
                should_auto_stop: false,
            }),
        });

        Self {
            config: audio_config,
            recording_config,
            silence_config,
            shared,
            _stream: None,
        }
    }

    /// Open the audio stream. Call once at startup.
    pub fn open_stream(&mut self) -> Result<(), String> {
        if self._stream.is_some() {
            return Ok(());
        }

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input audio device available")?;

        info!(
            "Using audio device: {}",
            device.name().unwrap_or("unknown".into())
        );

        let stream_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.chunk_size),
        };

        let shared = Arc::clone(&self.shared);
        let silence_threshold = self.silence_config.threshold;
        let silence_duration = self.silence_config.duration;
        let min_speech_duration = self.silence_config.min_speech_duration;
        let max_recording_duration = self.silence_config.max_recording_duration;

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                    let mut inner = shared.inner.lock().unwrap();

                    if !inner.is_recording {
                        return;
                    }

                    // Append samples to buffer
                    let remaining = inner.max_samples.saturating_sub(inner.buffer.len());
                    let to_copy = data.len().min(remaining);
                    inner.buffer.extend_from_slice(&data[..to_copy]);

                    if inner.buffer.len() >= inner.max_samples {
                        warn!("Max recording duration reached");
                        inner.is_recording = false;
                        inner.should_auto_stop = true;
                        return;
                    }

                    // Silence detection
                    if let Some(rec_start) = inner.recording_start {
                        let elapsed = rec_start.elapsed().as_secs_f64();

                        // Check max recording duration
                        if elapsed >= max_recording_duration {
                            info!("Max recording duration reached ({max_recording_duration}s)");
                            inner.should_auto_stop = true;
                            return;
                        }

                        // Don't check silence until minimum speech duration
                        if elapsed < min_speech_duration {
                            return;
                        }

                        // RMS energy
                        let rms = rms_energy(data);
                        let is_silent = rms < silence_threshold;

                        if is_silent {
                            let silence_start =
                                inner.silence_start.get_or_insert_with(Instant::now);
                            if silence_start.elapsed().as_secs_f64() >= silence_duration {
                                debug!("Silence detected for {silence_duration}s — auto-stopping");
                                inner.should_auto_stop = true;
                            }
                        } else {
                            inner.silence_start = None;
                        }
                    }
                },
                move |err| {
                    warn!("Audio stream error: {err}");
                },
                None, // timeout
            )
            .map_err(|e| format!("Failed to build input stream: {e}"))?;

        stream.play().map_err(|e| format!("Failed to start audio stream: {e}"))?;
        info!("Audio stream opened (ready for low-latency recording)");

        self._stream = Some(stream);
        Ok(())
    }

    /// Start recording audio.
    pub fn start(&self) {
        let mut inner = self.shared.inner.lock().unwrap();
        inner.buffer.clear();
        inner.is_recording = true;
        inner.silence_start = None;
        inner.recording_start = Some(Instant::now());
        inner.should_auto_stop = false;
        info!("Recording started");
    }

    /// Stop recording and return captured audio samples (f32, mono, 16kHz).
    pub fn stop(&self) -> Vec<f32> {
        let mut inner = self.shared.inner.lock().unwrap();
        inner.is_recording = false;
        let samples = std::mem::take(&mut inner.buffer);
        let duration = samples.len() as f64 / self.config.sample_rate as f64;
        info!("Recording stopped: {:.1}s ({} samples)", duration, samples.len());
        samples
    }

    /// Check if auto-stop was triggered by silence detection.
    pub fn should_auto_stop(&self) -> bool {
        self.shared.inner.lock().unwrap().should_auto_stop
    }

    /// Check if currently recording.
    #[allow(dead_code)]
    pub fn is_recording(&self) -> bool {
        self.shared.inner.lock().unwrap().is_recording
    }

    /// Check if audio data is silent.
    pub fn is_silent(samples: &[f32], threshold: f32) -> bool {
        let rms = rms_energy(samples);
        debug!("Audio RMS energy: {rms:.4} (threshold: {threshold})");
        rms < threshold
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

/// Calculate RMS energy of audio samples.
fn rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
