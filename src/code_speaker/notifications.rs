//! Durable, idempotent handoff of voice notifications to Triage Desk.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const TIMELINE_URL: &str = "http://127.0.0.1:3088/api/notifications";
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn now() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 1000.0
}

pub fn unique_id() -> String {
    format!(
        "{}-{}-{}",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Notification {
    pub event_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub session_name: String,
    #[serde(default)]
    pub agent: String,
    pub event_type: String,
    pub text: String,
    pub occurred_at: f64,
    pub turn_started_at: Option<f64>,
    pub turn_completed_at: Option<f64>,
    pub dnd_enabled: Option<bool>,
    pub delivery: String,
    pub phase: u8,
}

pub fn outbox_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/whisper-typer/triage-outbox")
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_millis(300))
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap_or_default()
}

/// Atomic disk handoff precedes networking. Never gate speech on the dashboard.
pub fn enqueue(event: &Notification) {
    if let Err(error) = write_pending(&outbox_dir(), event) {
        tracing::warn!("Cannot persist Triage notification: {error}");
    }
}

fn write_pending(dir: &Path, event: &Notification) -> std::io::Result<PathBuf> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let id = unique_id();
    let temporary = dir.join(format!(".{id}"));
    let target = dir.join(format!("{id}.json"));
    let mut file = std::fs::File::create(&temporary)?;
    file.write_all(&serde_json::to_vec(event)?)?;
    file.sync_all()?;
    std::fs::rename(&temporary, &target)?;
    Ok(target)
}

/// Hooks hand over immediately; failed requests remain for the speaker worker.
pub async fn forward(event: &Notification) {
    match write_pending(&outbox_dir(), event) {
        Ok(path) => {
            deliver(&client(), &path).await;
        }
        Err(error) => tracing::warn!("Cannot persist Triage notification: {error}"),
    }
}

async fn deliver(client: &reqwest::Client, path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return true;
    };
    match client
        .post(TIMELINE_URL)
        .header("Content-Type", "application/json")
        .body(bytes)
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => {
            let _ = std::fs::remove_file(path);
            true
        }
        Ok(response)
            if response.status().is_client_error() && response.status().as_u16() != 429 =>
        {
            tracing::warn!(
                "Triage rejected notification {}: {}",
                path.display(),
                response.status()
            );
            let _ = std::fs::rename(path, path.with_extension("rejected"));
            true // Preserve invalid data for diagnosis without blocking other events.
        }
        _ => false,
    }
}

pub fn start_forwarder() {
    tokio::spawn(async {
        let client = client();
        loop {
            if let Ok(entries) = std::fs::read_dir(outbox_dir()) {
                // Bound work per pass; a failed upstream gets one attempt per interval.
                let mut paths: Vec<_> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|e| e == "json"))
                    .take(100)
                    .collect();
                paths.sort();
                for path in paths {
                    if !deliver(&client, &path).await {
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disk_handoff_preserves_muted_event_and_identity() {
        let dir = std::env::temp_dir().join(format!("triage-outbox-test-{}", unique_id()));
        let event = Notification {
            event_id: "same-turn".into(),
            session_id: "session".into(),
            session_name: "Named session".into(),
            agent: "claude".into(),
            event_type: "stop".into(),
            text: "Task done".into(),
            occurred_at: now(),
            turn_started_at: Some(1.0),
            turn_completed_at: Some(4.0),
            dnd_enabled: Some(true),
            delivery: "muted".into(),
            phase: 1,
        };
        let path = write_pending(&dir, &event).unwrap();
        let restored: Notification = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(restored.event_id, "same-turn");
        assert_eq!(restored.dnd_enabled, Some(true));
        assert_eq!(restored.text, "Task done");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
