//! Bounded background grammar queue. Suggestions never delay or replace a paste.
use crate::{
    config::OllamaConfig,
    processor::{is_pathological_stutter, OllamaProcessor},
};
use serde_json::{json, Value};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    os::unix::fs::OpenOptionsExt,
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::{info, warn};
use super::ui::UiPublisher;

#[derive(Clone)]
pub struct ReviewJob {
    pub timestamp: String,
    pub original: String,
    pub pasted: String,
}
pub struct BackgroundReviewer {
    tx: mpsc::Sender<ReviewJob>,
    task: JoinHandle<()>,
    path: PathBuf,
    publisher: UiPublisher,
}
impl Drop for BackgroundReviewer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
impl BackgroundReviewer {
    pub fn new(
        config: OllamaConfig,
        enabled: Arc<dyn Fn() -> bool + Send + Sync>,
        postprocess: Arc<dyn Fn(&str) -> String + Send + Sync>,
        publisher: UiPublisher,
    ) -> Self {
        let path = dirs::cache_dir()
            .expect("cache directory")
            .join("whisper-typer/grammar-review.jsonl");
        Self::at_path(config, enabled, postprocess, path, publisher)
    }
    fn at_path(
        mut config: OllamaConfig,
        enabled: Arc<dyn Fn() -> bool + Send + Sync>,
        postprocess: Arc<dyn Fn(&str) -> String + Send + Sync>,
        path: PathBuf,
        publisher: UiPublisher,
    ) -> Self {
        config.enabled = true;
        config.grammar_gate.enabled = false;
        config.background_review = false;
        let model = config.model.clone();
        let processor = OllamaProcessor::new(config);
        let (tx, mut rx) = mpsc::channel::<ReviewJob>(32);
        let output = path.clone();
        let events = publisher.clone();
        let task = tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                if !enabled() {
                    save(
                        &output,
                        json!({"dictation_timestamp":job.timestamp,"status":"skipped","reason":"disabled"}),
                        &events,
                    );
                    continue;
                }
                let timer = Instant::now();
                let correction = processor.process(job.pasted.trim_end()).await;
                let candidate = postprocess(&correction.text);
                let accepted = correction.metadata.accepted
                    && !candidate.trim().is_empty()
                    && !is_pathological_stutter(&candidate)
                    && enabled();
                let corrected = if accepted {
                    candidate.trim_end().to_string()
                } else {
                    job.pasted.trim_end().to_string()
                };
                let changed = has_edit(&job.pasted, &corrected);
                save(
                    &output,
                    json!({"dictation_timestamp":job.timestamp,"timestamp":chrono::Local::now().to_rfc3339(),"original":job.original,"pasted":job.pasted,"corrected":corrected,"status":if changed {"changed"} else {"unchanged"},"model":model,"accepted":accepted,"fallback_reason":correction.metadata.fallback_reason,"grammar_latency_ms":timer.elapsed().as_secs_f64()*1000.0,"load_ms":correction.metadata.load_ms}),
                    &events,
                );
                info!(
                    "Background grammar review: {} ({}ms)",
                    if changed {
                        "suggestion saved"
                    } else {
                        "no edits"
                    },
                    timer.elapsed().as_millis()
                );
            }
        });
        Self { tx, task, path, publisher }
    }
    pub fn enqueue(&self, job: ReviewJob) {
        if let Err(error) = self.tx.try_send(job) {
            let job = error.into_inner();
            save(
                &self.path,
                json!({"dictation_timestamp":job.timestamp,"status":"skipped","reason":"review_queue_full_or_closed"}),
                &self.publisher,
            );
            warn!("Background grammar queue busy; original paste retained");
        }
    }
}
fn has_edit(original: &str, candidate: &str) -> bool {
    original.split_whitespace().collect::<Vec<_>>()
        != candidate.split_whitespace().collect::<Vec<_>>()
}
fn save(path: &PathBuf, value: Value, publisher: &UiPublisher) {
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            warn!("Cannot create review history directory");
            return;
        }
    }
    match OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)
    {
        Ok(mut file) => {
            if writeln!(file, "{value}").is_err() {
                warn!("Cannot append grammar review history");
            } else {
                publisher.publish("grammar_review", &value);
            }
        }
        Err(_) => warn!("Cannot open grammar review history"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whitespace_alone_is_not_a_grammar_edit() {
        assert!(!has_edit("Two files. ", "Two   files."));
        assert!(has_edit("She have files.", "She has files."));
    }
    #[tokio::test]
    async fn background_worker_keeps_each_dictation_and_bypasses_judge() {
        use axum::{routing::post, Json, Router};
        let app = Router::new().route(
            "/api/generate",
            post(|Json(body): Json<Value>| async move {
                assert_eq!(body["model"], "corrector-test");
                assert!(body["system"].as_str().unwrap().contains("Fix punctuation"));
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                Json(json!({"response":"{\"corrected_text\":\"She has two files.\"}"}))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let host = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let path = std::env::temp_dir().join(format!(
            "whisper-background-test-{}.jsonl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let config = OllamaConfig {
            model: "corrector-test".into(),
            host,
            grammar_gate: crate::config::GrammarGateConfig {
                enabled: true,
                model: "judge-must-not-run".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let worker = BackgroundReviewer::at_path(
            config,
            Arc::new(|| true),
            Arc::new(|s| s.into()),
            path.clone(),
            UiPublisher::default(),
        );
        let t = Instant::now();
        for timestamp in ["first", "second"] {
            worker.enqueue(ReviewJob {
                timestamp: timestamp.into(),
                original: "she have two files".into(),
                pasted: "She have two files. ".into(),
            });
        }
        assert!(t.elapsed() < std::time::Duration::from_millis(30));
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if fs::read_to_string(&path).is_ok_and(|s| s.lines().count() == 2) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let rows = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows[0]["dictation_timestamp"], "first");
        assert_eq!(rows[1]["dictation_timestamp"], "second");
        assert!(rows
            .iter()
            .all(|r| r["corrected"] == "She has two files." && r["status"] == "changed"));
        drop(worker);
        server.abort();
        fs::remove_file(path).unwrap();
    }
}
