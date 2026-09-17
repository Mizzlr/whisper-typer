//! Replay private dictation history without audio capture or typing.
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use regex::Regex;
use serde_json::{json, Value};
use whisper_typer_rs::{config::Config, processor::OllamaProcessor, spelling::SpellCorrector};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "config.yaml")]
    config: PathBuf,
    #[arg(long, default_value = "2026-09-")]
    date_prefix: String,
    /// Evaluate a single string instead of history.
    #[arg(long)]
    text: Option<String>,
    /// Exercise the configured real model; otherwise replay is entirely offline.
    #[arg(long)]
    grammar: bool,
    /// Store private edit evidence outside the repository.
    #[arg(long)]
    report: Option<PathBuf>,
}

fn read(path: &str) -> std::io::Result<String> {
    let path = path
        .strip_prefix("~/")
        .map(|p| dirs::home_dir().unwrap().join(p))
        .unwrap_or_else(|| path.into());
    fs::read_to_string(path)
}

fn percentile(values: &[f64], percent: usize) -> f64 {
    values[(values.len() - 1) * percent / 100]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = Config::load(Some(&args.config))?;
    let started = Instant::now();
    let corrector = SpellCorrector::load(&config.spelling)
        .ok_or("spelling must be enabled and data readable")?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let protectors = read(&config.corrections.path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            line.strip_prefix("protect\t")
                .and_then(|s| Regex::new(s).ok())
        })
        .collect::<Vec<_>>();
    if let Some(text) = args.text {
        let cleaned = corrector.apply(&text, &protectors);
        let mut result =
            json!({ "original": text, "spelling_text": cleaned.text, "edits": cleaned.edits });
        if args.grammar {
            let started = Instant::now();
            let correction = OllamaProcessor::new(config.ollama)
                .process(&cleaned.text)
                .await;
            result["final_text"] = json!(correction.text);
            result["accepted"] = json!(correction.metadata.accepted);
            result["fallback_reason"] = json!(correction.metadata.fallback_reason);
            result["gate"] = json!(correction.metadata.grammar_gate_decision);
            result["gate_provider"] = json!(correction.metadata.grammar_gate_provider);
            result["gate_ms"] = json!(correction.metadata.grammar_gate_latency_ms);
            result["grammar_wall_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        }
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    let mut paths = fs::read_dir(dirs::home_dir().unwrap().join(".whisper-typer-history"))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().is_some_and(|s| s == "jsonl")
                && path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with(&args.date_prefix)
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut records = Vec::new();
    for path in paths {
        for (line_number, line) in fs::read_to_string(&path)?.lines().enumerate() {
            let Ok(record) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if let Some(text) = record["whisper_text"].as_str() {
                records.push((path.clone(), line_number + 1, text.to_owned()));
            }
        }
    }
    if records.is_empty() {
        return Err("no matching dictations".into());
    }
    corrector.apply("a speccified chartt and dataaset", &protectors);
    let mut timing = Vec::new();
    let mut evidence = Vec::new();
    let mut edits = 0;
    for (path, line, text) in &records {
        let started = Instant::now();
        let result = corrector.apply(text, &protectors);
        timing.push(started.elapsed().as_secs_f64() * 1000.0);
        if !result.edits.is_empty() {
            edits += result.edits.len();
            evidence.push(json!({ "file": path, "line": line, "original": text, "corrected": result.text, "edits": result.edits }));
        }
    }
    timing.sort_by(f64::total_cmp);

    // Compare suggestion generation for the same unique unknown words with
    // duplicated letters. This isolates generators, not whole-text latency.
    let dictionary = spellbook::Dictionary::new(
        &read(&config.spelling.aff_path)?,
        &read(&config.spelling.dic_path)?,
    )
    .map_err(|error| error.to_string())?;
    let mut index = symspell::SymSpellBuilder::<symspell::UnicodeStringStrategy>::default()
        .max_dictionary_edit_distance(1)
        .build()?;
    for line in read(&config.spelling.dic_path)?.lines().skip(1) {
        let word = line
            .split_whitespace()
            .next()
            .unwrap_or("")
            .split('/')
            .next()
            .unwrap_or("");
        if !word.is_empty()
            && word.bytes().all(|b| b.is_ascii_lowercase())
            && dictionary.check(word)
        {
            index.load_dictionary_line(&format!("{word} 1"), 0, 1, " ");
        }
    }
    let words = Regex::new(r"\b[a-z]{4,64}\b")?;
    let unknown = records
        .iter()
        .flat_map(|(_, _, text)| words.find_iter(text).map(|m| m.as_str().to_owned()))
        .filter(|word| word.as_bytes().windows(2).any(|p| p[0] == p[1]) && !dictionary.check(word))
        .collect::<BTreeSet<_>>();
    let mut symspell_times = Vec::new();
    let mut spellbook_times = Vec::new();
    let mut suggestions = Vec::new();
    for word in &unknown {
        let started = Instant::now();
        std::hint::black_box(index.lookup(word, symspell::Verbosity::Closest, 1));
        symspell_times.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = Instant::now();
        dictionary.suggest(word, &mut suggestions);
        std::hint::black_box(&suggestions);
        spellbook_times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    symspell_times.sort_by(f64::total_cmp);
    spellbook_times.sort_by(f64::total_cmp);
    let summary = json!({
        "records": records.len(), "changed_records": evidence.len(), "edits": edits, "load_ms": load_ms,
        "whole_text_ms": { "p50": percentile(&timing, 50), "p95": percentile(&timing, 95), "p99": percentile(&timing, 99), "max": timing.last() },
        "candidate_generator": { "unique_unknown_words": unknown.len(),
            "symspell_p50_ms": (!unknown.is_empty()).then(|| percentile(&symspell_times, 50)),
            "symspell_p99_ms": (!unknown.is_empty()).then(|| percentile(&symspell_times, 99)),
            "spellbook_p50_ms": (!unknown.is_empty()).then(|| percentile(&spellbook_times, 50)),
            "spellbook_p99_ms": (!unknown.is_empty()).then(|| percentile(&spellbook_times, 99))
        }
    });
    if let Some(path) = args.report {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(
            serde_json::to_string_pretty(&json!({"summary": summary, "changes": evidence}))?
                .as_bytes(),
        )?;
    }
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
