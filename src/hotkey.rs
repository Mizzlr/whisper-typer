//! Global hotkey detection using evdev.
//!
//! Monitors all keyboard devices for configurable key combos.
//! Sends press/release events via a tokio channel.

use crate::config::HotkeyConfig;
use evdev::{Device, EventType, InputEventKind, Key};
use std::collections::HashSet;
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

    /// Find all keyboard input devices.
    pub fn find_keyboards() -> Vec<Device> {
        evdev::enumerate()
            .filter_map(|(_path, device)| {
                let keys = device.supported_keys()?;
                if keys.contains(Key::KEY_A) && keys.contains(Key::KEY_ENTER) {
                    info!(
                        "Found keyboard: {} at {:?}",
                        device.name().unwrap_or("unknown"),
                        device.physical_path()
                    );
                    Some(device)
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

    /// Start monitoring all keyboards with auto-reconnect.
    ///
    /// If all device monitors exit (disconnect, evdev stream failure, etc.),
    /// the watchdog re-enumerates keyboards and reconnects after a brief delay.
    /// This prevents permanent hotkey loss after transient system stress.
    pub async fn run(self) {
        loop {
            let keyboards = Self::find_keyboards();
            if keyboards.is_empty() {
                warn!("No keyboards found (are you in the 'input' group?). Retrying in 5s...");
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }

            info!("Monitoring {} keyboard(s)", keyboards.len());

            let mut handles = Vec::new();
            for device in keyboards {
                let combos = self.combos.clone();
                let state = Arc::clone(&self.state);
                let tx = self.tx.clone();
                handles.push(tokio::spawn(Self::monitor_device(
                    device, combos, state, tx,
                )));
            }

            // Wait for all monitors to exit
            for handle in handles {
                let _ = handle.await;
            }

            // All device monitors exited — reset state and reconnect
            {
                let mut state = self.state.lock().unwrap();
                state.pressed_keys.clear();
                state.hotkey_active = false;
            }
            warn!("All keyboard monitors exited — reconnecting in 2s...");
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }
}
