//! Desktop notifications via notify-rust (D-Bus).

use notify_rust::Notification;
use tracing::{debug, warn};

pub struct Notifier {
    enabled: bool,
}

impl Notifier {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn notify(&self, summary: &str, body: &str) {
        if !self.enabled {
            return;
        }

        debug!("Notification: {summary}");

        if let Err(e) = Notification::new()
            .summary(summary)
            .body(body)
            .icon("audio-input-microphone")
            .timeout(3000)
            .show()
        {
            warn!("Failed to show notification: {e}");
        }
    }
}
