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
//!     - input  f32 [batch=1, 512]    — 32 ms of 16 kHz mono PCM
//!     - state  f32 [2, 1, 128]       — LSTM hidden state, persists across frames
//!     - sr     i64 scalar            — 16000
//!   outputs:
//!     - output f32 [1, 1]            — speech probability in [0, 1]
//!     - stateN f32 [2, 1, 128]       — next LSTM state
//!
//! Frame size for 16 kHz must be exactly 512 samples; the model rejects
//! anything else. Callers buffer audio and emit one prediction per 32 ms frame.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

pub const FRAME_SAMPLES: usize = 512;
pub const SAMPLE_RATE: i64 = 16_000;

pub struct SileroVad {
    session: Mutex<Session>,
    state: Mutex<Array3<f32>>,
}

impl SileroVad {
    pub fn load(path: &Path) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .commit_from_file(path)
            .map_err(|e| format!("commit_from_file({}): {e}", path.display()))?;
        Ok(Self {
            session: Mutex::new(session),
            state: Mutex::new(Array3::<f32>::zeros((2, 1, 128))),
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

        let input = Array2::from_shape_vec((1, FRAME_SAMPLES), samples.to_vec())
            .map_err(|e| format!("input shape: {e}"))?;
        let sr = Array1::from_elem(1, SAMPLE_RATE);

        let state_in = self.state.lock().unwrap().clone();
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
        *self.state.lock().unwrap() = next_state.to_owned().into_dimensionality().map_err(|e| {
            format!("stateN dimensionality: {e}")
        })?;

        Ok(prob)
    }

    /// Reset LSTM state. Call between utterances so silence-tail context
    /// from one utterance doesn't bleed into the next.
    pub fn reset_state(&self) {
        let mut s = self.state.lock().unwrap();
        s.fill(0.0);
    }
}
