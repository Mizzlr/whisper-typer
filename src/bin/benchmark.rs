//! Summarize recent real-world latency from local transcription history.

use whisper_typer_rs::history;

fn percentile(values: &mut [i64], percentile: f64) -> i64 {
    if values.is_empty() {
        return 0;
    }
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index]
}

fn main() {
    let days = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|days| *days > 0)
        .unwrap_or(7);
    let dates = history::list_available_dates()
        .into_iter()
        .take(days)
        .collect::<Vec<_>>();
    let records = dates
        .iter()
        .flat_map(|date| history::load_records(date))
        .collect::<Vec<_>>();

    if records.is_empty() {
        println!("No transcription history found.");
        return;
    }

    let mut total = records
        .iter()
        .map(|record| record.total_latency_ms)
        .collect::<Vec<_>>();
    let mut whisper = records
        .iter()
        .map(|record| record.whisper_latency_ms)
        .collect::<Vec<_>>();
    let mut ollama = records
        .iter()
        .filter_map(|record| record.ollama_latency_ms)
        .collect::<Vec<_>>();
    let fallbacks = records
        .iter()
        .filter(|record| record.correction_accepted == Some(false))
        .count();

    println!(
        "WhisperTyper benchmark ({} records, {} day(s))",
        records.len(),
        dates.len()
    );
    println!(
        "total:   p50={}ms p95={}ms",
        percentile(&mut total.clone(), 0.50),
        percentile(&mut total, 0.95)
    );
    println!(
        "whisper: p50={}ms p95={}ms",
        percentile(&mut whisper.clone(), 0.50),
        percentile(&mut whisper, 0.95)
    );
    if !ollama.is_empty() {
        println!(
            "ollama:  p50={}ms p95={}ms",
            percentile(&mut ollama.clone(), 0.50),
            percentile(&mut ollama, 0.95)
        );
    }
    println!("correction fallbacks: {fallbacks}");
}
