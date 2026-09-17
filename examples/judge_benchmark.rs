//! Private, reproducible comparison using the shipping judge requests.
use std::{fs, path::PathBuf, time::Instant, os::unix::fs::OpenOptionsExt, io::Write};
use clap::Parser;
use regex::Regex;
use serde_json::{json, Value};
use whisper_typer_rs::{config::Config, processor::OllamaProcessor, punctuation::PunctuationClient, spelling::SpellCorrector};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "config.yaml")]
    config: PathBuf,
    #[arg(long)]
    date: String,
    #[arg(long, default_value_t = 24)]
    samples: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long)]
    report: PathBuf,
    /// Verify the actual shipping race on already-prepared private inputs.
    #[arg(long)]
    race_input_report: Option<PathBuf>,
    /// Compare local models on the exact saved benchmark inputs, without typing.
    #[arg(long)]
    model_input_report: Option<PathBuf>,
    #[arg(long)]
    models: Vec<String>,
    /// Isolated Ollama host for model comparisons; leaves live routing untouched.
    #[arg(long)]
    model_host: Option<String>,
}

fn read_private(path: &str) -> std::io::Result<String> {
    fs::read_to_string(path.strip_prefix("~/").map(|p| dirs::home_dir().unwrap().join(p)).unwrap_or_else(|| path.into()))
}

fn stats(values: &[f64]) -> Value {
    if values.is_empty() { return Value::Null; }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let p = |percent: usize| v[((v.len() * percent).div_ceil(100).max(1) - 1).min(v.len()-1)];
    json!({"count":v.len(), "p50_ms":p(50), "p95_ms":p(95), "min_ms":v[0], "max_ms":v[v.len()-1]})
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.samples < 2 || args.repeats == 0 { return Err("need at least two samples and one repeat".into()); }
    // Refuse overwriting evidence; create a private file before any calls.
    let mut report_file = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&args.report)?;
    let config = Config::load(Some(&args.config))?;
    if let Some(input_report) = args.model_input_report {
        if args.models.is_empty() { return Err("specify at least one --models value".into()); }
        let inputs: Value = serde_json::from_str(&fs::read_to_string(&input_report)?)?;
        let samples=inputs["samples"].as_array().ok_or("missing prepared samples")?;
        let processors=args.models.iter().map(|model| {
            let mut c=config.ollama.clone();
            c.grammar_gate.provider="ollama".into();
            c.grammar_gate.decision_format="structured".into();
            c.grammar_gate.model=model.clone();
            c.grammar_gate.host=args.model_host.clone().unwrap_or_else(||config.ollama.host.clone());
            c.grammar_gate.timeout_ms=5000;
            c.keep_alive=600;
            OllamaProcessor::new(c)
        }).collect::<Vec<_>>();
        let mut runs=Vec::new();
        for round in 0..args.repeats {
            for (sample,row) in samples.iter().enumerate() {
                // Sequential, rotating order: avoid benchmarking GPU contention
                // between the candidates themselves.
                for offset in 0..processors.len() {
                    let i=(offset+sample+round)%processors.len();
                    let t=Instant::now();
                    let decision=processors[i].judge_only(row["judge_input"].as_str().ok_or("missing input")?).await;
                    runs.push(json!({"model":args.models[i],"sample":sample,"round":round,"elapsed_ms":t.elapsed().as_secs_f64()*1000.0,"decision":decision}));
                }
            }
            eprintln!("Completed local-model round {}/{}",round+1,args.repeats);
        }
        // Deliberately labeled diagnostics, separate from unlabeled dictations.
        let diagnostics=[
            ("Summarize the progress.",false,false),
            ("Use the C compiler and press Ctrl plus V.",false,false),
            ("C compiler builds this.",false,false),
            ("Check ClickHouse for the 169 missing entries.",false,false),
            ("For now, ignore that.",false,false),
            ("Like.",false,false),
            ("And you can ignore Elos. For now. Just ming is what is considered.",true,false),
            ("Ignore that. For now. Keep this.",true,false),
            ("The results is ready.",true,false),
            ("She have two files.",true,false),
            ("S Summarize the progress.",false,true),
            ("W What can we do?",false,true),
            ("Y You can check it.",false,true),
            ("Press X to exit.",false,false),
            ("Use Rust and Qwen3.5:0.8b.",false,false),
            ("The URL is https://example.com/chartt and the count is 28.",false,false),
        ];
        let mut checks=Vec::new();
        for (i,processor) in processors.iter().enumerate() {
            for (text,needs,remove) in diagnostics {
                let t=Instant::now();let decision=processor.judge_only(text).await;
                let matched=decision.valid && decision.needs_correction==needs && decision.remove_leading_fragment==remove;
                checks.push(json!({"model":args.models[i],"input":text,"expected_needs_correction":needs,"expected_remove_leading_fragment":remove,"decision":decision,"matched":matched,"elapsed_ms":t.elapsed().as_secs_f64()*1000.0}));
            }
        }
        let summary=args.models.iter().map(|model| {
            let model_runs=runs.iter().filter(|r|r["model"]==*model).collect::<Vec<_>>();
            let durations=model_runs.iter().filter(|r|r["decision"]["valid"]==true).map(|r|r["elapsed_ms"].as_f64().unwrap()).collect::<Vec<_>>();
            let baseline=|sample:usize| inputs["runs"].as_array().unwrap().iter().find(|r|r["sample"]==sample && r["round"]==0).unwrap()["granite"].clone();
            let agreement=model_runs.iter().filter(|r|{
                let b=baseline(r["sample"].as_u64().unwrap() as usize);
                r["decision"]["valid"]==true && r["decision"]["needs_correction"]==b["needs_correction"] && r["decision"]["remove_leading_fragment"]==b["remove_leading_fragment"]
            }).count();
            let rounds=(0..args.repeats).map(|round|stats(&model_runs.iter().filter(|r|r["round"]==round && r["decision"]["valid"]==true).map(|r|r["elapsed_ms"].as_f64().unwrap()).collect::<Vec<_>>())).collect::<Vec<_>>();
            json!({"model":model,"latency":stats(&durations),"rounds":rounds,"failures":model_runs.len()-durations.len(),"agreement_with_original_granite":agreement,"runs":model_runs.len(),"diagnostic_matches":checks.iter().filter(|c|c["model"]==*model && c["matched"]==true).count(),"diagnostic_count":diagnostics.len()})
        }).collect::<Vec<_>>();
        let report=json!({"method":"Same prepared historical judge inputs and shipping structured two-boolean prompt; thinking disabled, 32 output tokens, persistent clients, sequential rotating model order, five-second budget, preloaded models. Three rounds include repeated-input/prefix caching. Historical agreement is not accuracy; 16 hand-labeled policy diagnostics are a small separate check.","input_report":input_report,"summary":summary,"samples":samples,"runs":runs,"diagnostics":checks});
        report_file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
        println!("{}",serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    if let Some(input_report) = args.race_input_report {
        let inputs: Value = serde_json::from_str(&fs::read_to_string(input_report)?)?;
        let mut race_config=config.ollama.clone();
        race_config.grammar_gate.provider="race".into();
        let race=OllamaProcessor::new(race_config);
        let mut runs=Vec::new();
        for (sample,row) in inputs["samples"].as_array().ok_or("missing prepared samples")?.iter().enumerate() {
            let t=Instant::now();
            let decision=race.judge_only(row["judge_input"].as_str().ok_or("missing judge input")?).await;
            runs.push(json!({"sample":sample,"elapsed_ms":t.elapsed().as_secs_f64()*1000.0,"decision":decision}));
        }
        let durations=runs.iter().filter(|r|r["decision"]["valid"]==true).map(|r|r["elapsed_ms"].as_f64().unwrap()).collect::<Vec<_>>();
        let summary=json!({"actual_race":stats(&durations), "jev_wins":runs.iter().filter(|r|r["decision"]["provider"]=="typesafe" && r["decision"]["valid"]==true).count(), "granite_wins":runs.iter().filter(|r|r["decision"]["provider"]=="ollama" && r["decision"]["valid"]==true).count(), "failures":runs.iter().filter(|r|r["decision"]["valid"]!=true).count()});
        report_file.write_all(serde_json::to_string_pretty(&json!({"method":"Actual shipping race: first valid decision, loser future dropped; same prepared inputs, one persistent processor", "summary":summary,"runs":runs}))?.as_bytes())?;
        println!("{}",serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    let history = dirs::home_dir().unwrap().join(".whisper-typer-history").join(format!("{}.jsonl", args.date));
    let secret_like = Regex::new(r"(?i)(?:apikey_[a-z0-9]{12,}|sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._-]{16,})")?;
    let rows = fs::read_to_string(&history)?.lines()
        .map(serde_json::from_str::<Value>).collect::<Result<Vec<_>,_>>()?
        .into_iter().filter(|r| r["timestamp"].as_str().is_some_and(|t| t >= format!("{}T06:00:00", args.date).as_str())
            && r["whisper_text"].as_str().is_some_and(|t| !t.trim().is_empty() && !secret_like.is_match(t)))
        .collect::<Vec<_>>();
    if rows.len() < args.samples { return Err("not enough eligible morning-onward records".into()); }
    let latest = args.samples / 2;
    let spread = args.samples - latest;
    let earlier = rows.len() - latest;
    let mut indices = (0..spread).map(|i| i * (earlier-1) / (spread-1).max(1)).collect::<Vec<_>>();
    indices.extend(earlier..rows.len());
    let rules = read_private(&config.corrections.path)?;
    let protectors = rules.lines().filter_map(|l| l.strip_prefix("protect\t").and_then(|p| Regex::new(p).ok())).collect::<Vec<_>>();
    let replacements = rules.lines().filter_map(|l| {
        let p = l.splitn(3, '\t').collect::<Vec<_>>();
        match p.as_slice() {
            ["replace", pattern, replacement] => Some((Regex::new(pattern).ok()?, (*replacement).to_owned())),
            [pattern, replacement] if !pattern.starts_with('#') && *pattern != "protect" => Some((Regex::new(pattern).ok()?, (*replacement).to_owned())),
            _ => None,
        }
    }).collect::<Vec<_>>();
    let spelling = SpellCorrector::load(&config.spelling).ok_or("spelling unavailable")?;
    let punctuator = PunctuationClient::new(config.punctuation.clone())?;
    let mut samples = Vec::new();
    for index in indices {
        let row = &rows[index];
        let original = row["whisper_text"].as_str().unwrap();
        let mut cleaned = original.to_owned();
        if config.corrections.enabled {
            for (regex, replacement) in &replacements {
                let spans = protectors.iter().flat_map(|p| p.find_iter(&cleaned).map(|m| (m.start(),m.end()))).collect::<Vec<_>>();
                cleaned = regex.replace_all(&cleaned, |caps: &regex::Captures<'_>| {
                    let m=caps.get(0).unwrap();
                    if spans.iter().any(|&(a,b)| m.start()<b && a<m.end()) { m.as_str().to_owned() } else { replacement.clone() }
                }).into_owned();
            }
        }
        let cleaned = spelling.apply(&cleaned, &protectors).text.into_owned();
        let (input, punctuation_failed) = match punctuator.process(&cleaned).await {
            Ok(result) => (result.text,false), Err(_) => (cleaned,true),
        };
        samples.push(json!({"timestamp":row["timestamp"], "raw_text":original, "judge_input":input, "words":input.split_whitespace().count(), "punctuation_failed":punctuation_failed}));
    }
    let mut jev_config = config.ollama.clone();
    jev_config.grammar_gate.provider="typesafe".into();
    // Longer observation budget measures slow replies rather than censoring them.
    jev_config.grammar_gate.timeout_ms=5000;
    let mut granite_config = config.ollama.clone();
    granite_config.grammar_gate.provider="ollama".into();
    granite_config.grammar_gate.model=config.ollama.model.clone();
    granite_config.grammar_gate.host=config.ollama.host.clone();
    granite_config.grammar_gate.timeout_ms=5000;
    let jev=OllamaProcessor::new(jev_config);
    let granite=OllamaProcessor::new(granite_config);
    let mut runs=Vec::new();
    for round in 0..args.repeats {
        for (sample, row) in samples.iter().enumerate() {
            let input=row["judge_input"].as_str().unwrap();
            let (j,g)=tokio::join!(async { let t=Instant::now(); let result=jev.judge_only(input).await; (result,t.elapsed().as_secs_f64()*1000.0) },
                async { let t=Instant::now(); let result=granite.judge_only(input).await; (result,t.elapsed().as_secs_f64()*1000.0) });
            let winner = match (j.0.valid,g.0.valid) {
                (true,true) => if j.1<g.1 {"typesafe"} else {"ollama"},
                (true,false) => "typesafe", (false,true) => "ollama", _ => "neither",
            };
            let race_ms = if winner=="typesafe" {j.1} else if winner=="ollama" {g.1} else {j.1.max(g.1)};
            let disagree = j.0.valid && g.0.valid && (j.0.needs_correction != g.0.needs_correction || j.0.remove_leading_fragment != g.0.remove_leading_fragment);
            runs.push(json!({"sample":sample,"round":round,"connection":"reused_except_first_pair", "jev":j.0,"jev_ms":j.1,"granite":g.0,"granite_ms":g.1,"winner":winner,"race_ms":race_ms,"disagree":disagree}));
        }
        eprintln!("Completed paired round {}/{}", round+1,args.repeats);
    }
    let durations = |field: &str, provider: &str| runs.iter().filter(|r| r[provider]["valid"]==true).map(|r| r[field].as_f64().unwrap()).collect::<Vec<_>>();
    let races=runs.iter().filter(|r| r["winner"]!="neither").map(|r|r["race_ms"].as_f64().unwrap()).collect::<Vec<_>>();
    let summary=json!({"samples":samples.len(),"paired_runs":runs.len(),"jev":stats(&durations("jev_ms","jev")),"granite":stats(&durations("granite_ms","granite")),"race":stats(&races),
        "jev_wins":runs.iter().filter(|r|r["winner"]=="typesafe").count(),"granite_wins":runs.iter().filter(|r|r["winner"]=="ollama").count(),
        "jev_failures":runs.iter().filter(|r|r["jev"]["valid"]!=true).count(),"granite_failures":runs.iter().filter(|r|r["granite"]["valid"]!=true).count(),
        "disagreements":runs.iter().filter(|r|r["disagree"]==true).count(),"first_pair":runs.first()});
    let report=json!({"method":"24 samples (half spread since 06:00 local, half latest), three paired rounds; shipping judge prompts, identical post-domain/spelling/punctuation inputs; persistent Rust reqwest clients. Race latency is observed earliest valid completed judge, not an accuracy score or full typing latency.",
        "date":args.date,"history":history,"jev_model":config.ollama.grammar_gate.model,"granite_model":config.ollama.model,"judge_observation_timeout_ms":5000,"summary":summary,"samples":samples,"runs":runs});
    report_file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!("{}",serde_json::to_string_pretty(&summary)?);
    Ok(())
}
