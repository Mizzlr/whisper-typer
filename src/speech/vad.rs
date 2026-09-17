//! Silero VAD speech-detection wrapper.
//!
//! Wraps the Silero VAD v5 ONNX model
//! (<https://github.com/snakers4/silero-vad>) for real-time speech-vs-noise
//! discrimination. Used by `voice-journal` as an opt-in upgrade over RMS
//! energy thresholding, which cannot distinguish "loud noise" (background
//! podcast / TV / breathing) from "intentional human speech directed at
//! the microphone."
//!
//! Model contract (v5):
//!   inputs:
//!     - input  f32 [batch=1, 576]    — 64 context + 512 new samples at 16 kHz
//!     - state  f32 [2, 1, 128]       — LSTM hidden state, persists across frames
//!     - sr     i64 scalar            — 16000
//!   outputs:
//!     - output f32 [1, 1]            — speech probability in [0, 1]
//!     - stateN f32 [2, 1, 128]       — next LSTM state
//!
//! Callers supply exactly 512 new samples (32 ms). Silero's official v5 wrapper
//! prepends the preceding 64 samples; omitting this context suppresses scores.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

pub const FRAME_SAMPLES: usize = 512;
pub const SAMPLE_RATE: i64 = 16_000;

pub struct SileroVad {
    session: Mutex<Session>,
    state: Mutex<RecurrentState>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silence_and_reset_use_context_correctly() {
        let path = Path::new("models/silero_vad.onnx");
        if !path.exists() {
            return;
        }
        let vad = SileroVad::load(path).unwrap();
        assert!(vad.predict(&[0.0; 511]).is_err());
        let initial = vad.predict(&[0.0; FRAME_SAMPLES]).unwrap();
        for _ in 0..40 {
            assert!(vad.predict(&[0.0; FRAME_SAMPLES]).unwrap() < 0.5);
        }
        vad.reset_state();
        assert_eq!(initial, vad.predict(&[0.0; FRAME_SAMPLES]).unwrap());
    }
    #[test]
    fn known_speech_reaches_threshold_with_official_context() {
        let Ok(path) = std::env::var("WHISPER_VAD_TEST_WAV") else {
            return;
        };
        let vad = SileroVad::load(Path::new("models/silero_vad.onnx")).unwrap();
        let mut wav = hound::WavReader::open(path).unwrap();
        assert_eq!(wav.spec().sample_rate, 16000);
        let samples: Vec<_> = wav
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let probabilities: Vec<_> = samples
            .chunks_exact(FRAME_SAMPLES)
            .map(|s| vad.predict(s).unwrap())
            .collect();
        assert!(!probabilities.is_empty());
        assert!(probabilities.iter().filter(|p| **p >= 0.5).count() * 5 > probabilities.len());
    }
}
struct RecurrentState {
    hidden: Array3<f32>,
    context: [f32; 64],
}

impl SileroVad {
    pub fn load(path: &Path) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| format!("ort intra threads: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| format!("ort inter threads: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("commit_from_file({}): {e}", path.display()))?;
        Ok(Self {
            session: Mutex::new(session),
            state: Mutex::new(RecurrentState {
                hidden: Array3::<f32>::zeros((2, 1, 128)),
                context: [0.0; 64],
            }),
        })
    }

    /// Run one frame through the model. `samples.len()` MUST equal `FRAME_SAMPLES`.
    /// Returns the speech probability in [0, 1].
    pub fn predict(&self, samples: &[f32]) -> Result<f32, String> {
        if samples.len() != FRAME_SAMPLES {
            return Err(format!(
                "expected {FRAME_SAMPLES} samples, got {}",
                samples.len()
            ));
        }

        let mut recurrent = self.state.lock().unwrap();
        let mut contextual_samples = recurrent.context.to_vec();
        contextual_samples.extend_from_slice(samples);
        let input = Array2::from_shape_vec((1, FRAME_SAMPLES + 64), contextual_samples)
            .map_err(|e| format!("input shape: {e}"))?;
        let sr = Array1::from_elem(1, SAMPLE_RATE);

        let state_in = recurrent.hidden.clone();
        let input_t = Tensor::from_array(input).map_err(|e| format!("input tensor: {e}"))?;
        let state_t = Tensor::from_array(state_in).map_err(|e| format!("state tensor: {e}"))?;
        let sr_t = Tensor::from_array(sr).map_err(|e| format!("sr tensor: {e}"))?;

        // Note: the `MutexGuard` must outlive the SessionOutputs view, so we
        // bind it explicitly here instead of inlining inside a block.
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "input" => input_t,
                "state" => state_t,
                "sr"    => sr_t,
            ])
            .map_err(|e| format!("session.run: {e}"))?;

        let prob_view = outputs["output"]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract output: {e}"))?;
        let prob = prob_view
            .iter()
            .next()
            .copied()
            .ok_or_else(|| "empty output tensor".to_string())?;

        let next_state = outputs["stateN"]
            .try_extract_array::<f32>()
            .map_err(|e| format!("extract stateN: {e}"))?;
        recurrent.hidden = next_state
            .to_owned()
            .into_dimensionality()
            .map_err(|e| format!("stateN dimensionality: {e}"))?;
        recurrent
            .context
            .copy_from_slice(&samples[FRAME_SAMPLES - 64..]);

        Ok(prob)
    }

    /// Reset LSTM state. Call between utterances so silence-tail context
    /// from one utterance doesn't bleed into the next.
    pub fn reset_state(&self) {
        let mut s = self.state.lock().unwrap();
        s.hidden.fill(0.0);
        s.context.fill(0.0);
    }
}
