//! Periodic TTS reminder manager.
//!
//! Repeats a TTS notification at intervals until cancelled.
//! Used for "Claude is waiting" reminders after task completion.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::JoinHandle;
use tracing::info;

use super::tts::KokoroTtsEngine;

pub struct ReminderManager {
    interval_secs: u64,
    task: Mutex<Option<JoinHandle<()>>>,
    active: Arc<AtomicBool>,
    count: Arc<AtomicU32>,
}

impl ReminderManager {
    pub fn new(interval: u64) -> Self {
        Self {
            interval_secs: interval,
            task: Mutex::new(None),
            active: Arc::new(AtomicBool::new(false)),
            count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    pub fn reminder_count(&self) -> u32 {
        self.count.load(Ordering::Relaxed)
    }

    /// Start repeating reminders. Cancels any existing reminder first.
    pub fn start(&self, text: String, tts: Arc<KokoroTtsEngine>) {
        self.cancel();
        self.active.store(true, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);

        let active = self.active.clone();
        let count = self.count.clone();
        let interval = self.interval_secs;

        let handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                if !active.load(Ordering::Relaxed) {
                    break;
                }
                let n = count.fetch_add(1, Ordering::Relaxed) + 1;
                info!("Reminder #{n}: speaking");
                let _ = tts.speak(&text).await;
            }
        });

        *self.task.lock().unwrap() = Some(handle);
    }

    /// Cancel all pending reminders. Returns count of reminders fired.
    pub fn cancel(&self) -> u32 {
        self.active.store(false, Ordering::Relaxed);
        if let Some(handle) = self.task.lock().unwrap().take() {
            handle.abort();
        }
        self.count.swap(0, Ordering::Relaxed)
    }
}
