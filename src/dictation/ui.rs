//! Best-effort local UI pushes, isolated from the typing path.
use crate::config::UiConfig;
use reqwest::{Client, Url};
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

#[derive(Clone, Default)]
pub struct UiPublisher {
    tx: Option<mpsc::Sender<Value>>,
}
impl UiPublisher {
    pub fn new(config: &UiConfig) -> Self {
        if !config.enabled {
            return Self::default();
        }
        let Ok(url) = Url::parse(&config.endpoint) else {
            warn!("UI push disabled: invalid endpoint");
            return Self::default();
        };
        if url.scheme() != "http"
            || !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"))
            || !url.username().is_empty()
            || url.password().is_some()
        {
            warn!("UI push disabled: endpoint must be local HTTP");
            return Self::default();
        }
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .tcp_nodelay(true)
            .timeout(Duration::from_millis(config.timeout_ms.max(1)))
            .build()
            .expect("UI HTTP client");
        let (tx, mut rx) = mpsc::channel::<Value>(128);
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                let started = Instant::now();
                match client.post(url.clone()).json(&event).send().await {
                    Ok(response) if response.status().is_success() => {
                        // Drain the small acknowledgment to reuse the connection.
                        if response.bytes().await.is_ok() {
                            info!(
                                "UI push delivered: {} ({:.1}ms)",
                                event["type"].as_str().unwrap_or("unknown"),
                                started.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                    }
                    _ => debug!("UI push unavailable; history remains available for recovery"),
                }
            }
        });
        info!("Local UI push enabled");
        Self { tx: Some(tx) }
    }
    pub fn publish(&self, kind: &str, payload: &impl Serialize) {
        if let Some(tx) = &self.tx {
            if tx.try_send(json!({"type":kind,"payload":payload})).is_err() {
                warn!("UI push queue full; history will recover updates");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::State, routing::post, Json, Router};

    #[tokio::test]
    async fn preserves_event_order_without_waiting_for_a_slow_ui() {
        let (events_tx, mut events_rx) = mpsc::channel::<Value>(2);
        let app = Router::new()
            .route(
                "/events",
                post(
                    |State(tx): State<mpsc::Sender<Value>>, Json(event): Json<Value>| async move {
                        tokio::time::sleep(Duration::from_millis(80)).await;
                        tx.send(event).await.unwrap();
                        axum::http::StatusCode::ACCEPTED
                    },
                ),
            )
            .with_state(events_tx);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/events", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let publisher = UiPublisher::new(&UiConfig {
            enabled: true,
            endpoint,
            timeout_ms: 500,
        });
        let started = Instant::now();
        publisher.publish(
            "dictation",
            &json!({"timestamp":"first","final_text":"She have files."}),
        );
        publisher.clone().publish(
            "grammar_review",
            &json!({"dictation_timestamp":"first","corrected":"She has files."}),
        );
        assert!(started.elapsed() < Duration::from_millis(30));
        let first = tokio::time::timeout(Duration::from_secs(2), events_rx.recv())
            .await
            .unwrap()
            .unwrap();
        let second = tokio::time::timeout(Duration::from_secs(2), events_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first["type"], "dictation");
        assert_eq!(second["type"], "grammar_review");
        assert_eq!(second["payload"]["corrected"], "She has files.");
        server.abort();
    }

    #[test]
    fn remote_endpoints_are_disabled_without_a_runtime() {
        let publisher = UiPublisher::new(&UiConfig {
            enabled: true,
            endpoint: "https://example.com/events".into(),
            timeout_ms: 200,
        });
        assert!(publisher.tx.is_none());
    }
}
