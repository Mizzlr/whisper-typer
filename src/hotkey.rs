//! Global hotkey detection using evdev.
//!
//! Monitors all keyboard devices for configurable key combos.
//! Sends press/release events via a tokio channel.

use crate::config::HotkeyConfig;
use evdev::{Device, EventType, InputEventKind, Key};
use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Events sent from the hotkey monitor to the service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyEvent {
    Pressed,
    Released,
}

/// Resolve a key name like "KEY_LEFTMETA" to an evdev Key code.
fn resolve_key(name: &str) -> Option<Key> {
    match Key::from_str(name) {
        Ok(k) => Some(k),
        Err(_) => {
            warn!("Unknown key name: {name}");
            None
        }
    }
}

/// Shared state for tracking pressed keys across devices.
struct HotkeyState {
    pressed_keys: HashSet<Key>,
    hotkey_active: bool,
}

pub struct HotkeyMonitor {
    combos: Vec<HashSet<Key>>,
    state: Arc<Mutex<HotkeyState>>,
    tx: mpsc::Sender<HotkeyEvent>,
}

impl HotkeyMonitor {
    pub fn new(config: &HotkeyConfig, tx: mpsc::Sender<HotkeyEvent>) -> Self {
        let combos: Vec<HashSet<Key>> = std::iter::once(&config.combo)
            .chain(config.alt_combos.iter())
            .map(|combo| combo.iter().filter_map(|s| resolve_key(s)).collect::<HashSet<_>>())
            .filter(|combo| !combo.is_empty())
            .collect();

        info!("Hotkey combos: {} configured", combos.len());

        Self {
            combos,
            state: Arc::new(Mutex::new(HotkeyState {
                pressed_keys: HashSet::new(),
                hotkey_active: false,
            })),
            tx,
        }
    }

    /// Find all keyboard input devices, returning their /dev/input path alongside.
    pub fn find_keyboards() -> Vec<(PathBuf, Device)> {
        evdev::enumerate()
            .filter_map(|(path, device)| {
                let keys = device.supported_keys()?;
                if keys.contains(Key::KEY_A) && keys.contains(Key::KEY_ENTER) {
                    Some((path, device))
                } else {
                    None
                }
            })
            .collect()
    }

    fn any_combo_active(combos: &[HashSet<Key>], pressed: &HashSet<Key>) -> bool {
        combos.iter().any(|combo| combo.is_subset(pressed))
    }

    /// Monitor a single device for key events.
    async fn monitor_device(
        device: Device,
        combos: Vec<HashSet<Key>>,
        state: Arc<Mutex<HotkeyState>>,
        tx: mpsc::Sender<HotkeyEvent>,
    ) {
        let name = device.name().unwrap_or("unknown").to_string();
        debug!("Monitoring {name}");

        let mut events = match device.into_event_stream() {
            Ok(stream) => stream,
            Err(e) => {
                warn!("Cannot create event stream for {name}: {e}");
                return;
            }
        };

        loop {
            match events.next_event().await {
                Ok(event) => {
                    if event.event_type() != EventType::KEY {
                        continue;
                    }

                    let key = match event.kind() {
                        InputEventKind::Key(k) => k,
                        _ => continue,
                    };

                    let value = event.value();
                    // 0 = release, 1 = press, 2 = repeat
                    let mut state = state.lock().unwrap();

                    match value {
                        1 => {
                            state.pressed_keys.insert(key);
                        }
                        0 => {
                            state.pressed_keys.remove(&key);
                        }
                        _ => continue, // ignore repeats
                    }

                    let now_active = Self::any_combo_active(&combos, &state.pressed_keys);

                    if now_active && !state.hotkey_active {
                        state.hotkey_active = true;
                        debug!("Hotkey pressed");
                        let _ = tx.try_send(HotkeyEvent::Pressed);
                    } else if !now_active && state.hotkey_active {
                        state.hotkey_active = false;
                        debug!("Hotkey released");
                        let _ = tx.try_send(HotkeyEvent::Released);
                    }
                }
                Err(e) => {
                    warn!("Device {name} disconnected: {e}");
                    break;
                }
            }
        }
    }

    /// Start monitoring all keyboards with hotplug-aware reconnect.
    ///
    /// Polls /dev/input every 2s for new keyboard devices and spawns a
    /// per-device monitor task for anything not already tracked. Tolerates
    /// partial disconnects (e.g. one keyboard goes away while another stays
    /// connected) — the previous all-or-nothing watchdog could leave the
    /// system stuck monitoring only the surviving keyboards forever.
    pub async fn run(self) {
        let tracked: Arc<Mutex<HashSet<PathBuf>>> = Arc::new(Mutex::new(HashSet::new()));
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut warned_empty = false;

        loop {
            interval.tick().await;

            let mut spawned = 0;
            for (path, device) in Self::find_keyboards() {
                {
                    let t = tracked.lock().unwrap();
                    if t.contains(&path) {
                        continue;
                    }
                }

                let name = device.name().unwrap_or("unknown").to_string();
                info!("Found keyboard: {name} at {path:?}");

                tracked.lock().unwrap().insert(path.clone());

                let combos = self.combos.clone();
                let state = Arc::clone(&self.state);
                let tx = self.tx.clone();
                let tracked_for_task = Arc::clone(&tracked);
                let task_path = path.clone();

                tokio::spawn(async move {
                    Self::monitor_device(device, combos, state.clone(), tx).await;
                    tracked_for_task.lock().unwrap().remove(&task_path);
                    // Clear cross-device pressed state — a key held during
                    // disconnect would otherwise stay "pressed" forever.
                    let mut s = state.lock().unwrap();
                    s.pressed_keys.clear();
                    s.hotkey_active = false;
                });
                spawned += 1;
            }

            let active = tracked.lock().unwrap().len();
            if spawned > 0 {
                info!("Monitoring {active} keyboard(s) ({spawned} new)");
                warned_empty = false;
            } else if active == 0 && !warned_empty {
                warn!("No keyboards found (are you in the 'input' group?). Polling...");
                warned_empty = true;
            }
        }
    }
}
