//! Optional remote ASR client with WAV transport.

use std::io::Cursor;
use std::time::Instant;

use reqwest::multipart::{Form, Part};
use serde::Deserialize;

use crate::config::RemoteAsrConfig;
use crate::transcriber::TranscribeResult;

#[derive(Clone)]
pub struct RemoteAsrClient {
    config: RemoteAsrConfig,
    client: reqwest::Client,
}

#[derive(Deserialize)]
struct RemoteResponse {
    text: String,
    latency_ms: Option<f64>,
}

impl RemoteAsrClient {
    pub fn new(config: RemoteAsrConfig) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|error| format!("build remote ASR client: {error}"))?;
        Ok(Self { config, client })
    }

    pub fn fallback_local(&self) -> bool {
        self.config.fallback_local
    }

    pub async fn transcribe(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<TranscribeResult, String> {
        let wav = encode_wav(samples, sample_rate)?;
        let part = Part::bytes(wav)
            .file_name("dictation.wav")
            .mime_str("audio/wav")
            .map_err(|error| format!("build remote ASR upload: {error}"))?;
        let started = Instant::now();
        let response = self
            .client
            .post(&self.config.url)
            .multipart(Form::new().part("audio", part))
            .send()
            .await
            .map_err(|error| format!("remote ASR request: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("remote ASR returned {status}: {body}"));
        }
        let response: RemoteResponse = response
            .json()
            .await
            .map_err(|error| format!("decode remote ASR response: {error}"))?;
        let text = response.text.trim().to_string();
        if text.is_empty() {
            return Err("remote ASR returned an empty transcript".into());
        }
        Ok(TranscribeResult {
            text,
            latency_ms: response
                .latency_ms
                .unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0),
        })
    }
}

fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let mut cursor = Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|error| format!("create WAV payload: {error}"))?;
        for sample in samples {
            let pcm = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
            writer
                .write_sample(pcm)
                .map_err(|error| format!("encode WAV payload: {error}"))?;
        }
        writer
            .finalize()
            .map_err(|error| format!("finalize WAV payload: {error}"))?;
    }
    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    use super::encode_wav;

    #[test]
    fn wav_transport_preserves_shape_and_rate() {
        let samples = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let bytes = encode_wav(&samples, 16_000).expect("encode WAV");
        let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).expect("read WAV");
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.duration(), samples.len() as u32);
    }
}
