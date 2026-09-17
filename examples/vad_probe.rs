//! Report only speech-detector statistics for a private mono 16 kHz WAV.
use whisper_typer_rs::vad::{SileroVad, FRAME_SAMPLES};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let detector = SileroVad::load(std::path::Path::new("models/silero_vad.onnx")).unwrap();
    let mut wav = hound::WavReader::open(&args[1]).unwrap();
    assert_eq!(wav.spec().sample_rate, 16000);
    assert_eq!(wav.spec().channels, 1);
    let samples: Vec<_> = wav
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect();
    let mut probabilities = Vec::new();
    let mut errors = 0;
    for frame in samples.chunks_exact(FRAME_SAMPLES) {
        match detector.predict(frame) {
            Ok(p) => probabilities.push(p),
            Err(_) => errors += 1,
        }
    }
    probabilities.sort_by(f32::total_cmp);
    if probabilities.is_empty() {
        panic!("No predictions; errors={errors}");
    }
    println!(
        "frames={} errors={} median={:.4} max={:.4} fraction_speech={:.3}",
        probabilities.len(),
        errors,
        probabilities[probabilities.len() / 2],
        probabilities.last().unwrap(),
        probabilities.iter().filter(|p| **p >= 0.5).count() as f32 / probabilities.len() as f32
    );
}
