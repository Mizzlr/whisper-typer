//! Global hotkey detection using evdev.
//!
//! Monitors all keyboard devices for configurable key combos.
//! Sends press/release events via a tokio channel.

use crate::config::HotkeyConfig;
use evdev::{Device, EventType, InputEventKind, Key};
use std::collections::HashSet;
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
    // evdev::Key has a from_str-like constructor via the key code number.
    // We need to map string names to codes manually for common keys.
    let key = match name {
        "KEY_LEFTMETA" => Key::KEY_LEFTMETA,
        "KEY_RIGHTMETA" => Key::KEY_RIGHTMETA,
        "KEY_LEFTALT" => Key::KEY_LEFTALT,
        "KEY_RIGHTALT" => Key::KEY_RIGHTALT,
        "KEY_LEFTCTRL" => Key::KEY_LEFTCTRL,
        "KEY_RIGHTCTRL" => Key::KEY_RIGHTCTRL,
        "KEY_LEFTSHIFT" => Key::KEY_LEFTSHIFT,
        "KEY_RIGHTSHIFT" => Key::KEY_RIGHTSHIFT,
        "KEY_PAGEDOWN" => Key::KEY_PAGEDOWN,
        "KEY_PAGEUP" => Key::KEY_PAGEUP,
        "KEY_RIGHT" => Key::KEY_RIGHT,
        "KEY_LEFT" => Key::KEY_LEFT,
        "KEY_UP" => Key::KEY_UP,
        "KEY_DOWN" => Key::KEY_DOWN,
        "KEY_SPACE" => Key::KEY_SPACE,
        "KEY_ENTER" => Key::KEY_ENTER,
        "KEY_TAB" => Key::KEY_TAB,
        "KEY_ESC" => Key::KEY_ESC,
        "KEY_A" => Key::KEY_A,
        "KEY_B" => Key::KEY_B,
        "KEY_C" => Key::KEY_C,
        "KEY_D" => Key::KEY_D,
        "KEY_E" => Key::KEY_E,
        "KEY_F" => Key::KEY_F,
        "KEY_G" => Key::KEY_G,
        "KEY_H" => Key::KEY_H,
        "KEY_I" => Key::KEY_I,
        "KEY_J" => Key::KEY_J,
        "KEY_K" => Key::KEY_K,
        "KEY_L" => Key::KEY_L,
        "KEY_M" => Key::KEY_M,
        "KEY_N" => Key::KEY_N,
        "KEY_O" => Key::KEY_O,
        "KEY_P" => Key::KEY_P,
        "KEY_Q" => Key::KEY_Q,
        "KEY_R" => Key::KEY_R,
        "KEY_S" => Key::KEY_S,
        "KEY_T" => Key::KEY_T,
        "KEY_U" => Key::KEY_U,
        "KEY_V" => Key::KEY_V,
        "KEY_W" => Key::KEY_W,
        "KEY_X" => Key::KEY_X,
        "KEY_Y" => Key::KEY_Y,
        "KEY_Z" => Key::KEY_Z,
        "KEY_F1" => Key::KEY_F1,
        "KEY_F2" => Key::KEY_F2,
        "KEY_F3" => Key::KEY_F3,
        "KEY_F4" => Key::KEY_F4,
        "KEY_F5" => Key::KEY_F5,
        "KEY_F6" => Key::KEY_F6,
        "KEY_F7" => Key::KEY_F7,
        "KEY_F8" => Key::KEY_F8,
        "KEY_F9" => Key::KEY_F9,
        "KEY_F10" => Key::KEY_F10,
        "KEY_F11" => Key::KEY_F11,
        "KEY_F12" => Key::KEY_F12,
        _ => {
            warn!("Unknown key name: {name}");
            return None;
        }
    };
    Some(key)
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
        let mut combos = Vec::new();

        // Primary combo
        let primary: HashSet<Key> = config.combo.iter().filter_map(|s| resolve_key(s)).collect();
        if !primary.is_empty() {
            combos.push(primary);
        }

        // Alternate combos
        for alt in &config.alt_combos {
            let combo: HashSet<Key> = alt.iter().filter_map(|s| resolve_key(s)).collect();
            if !combo.is_empty() {
                combos.push(combo);
            }
        }

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
        let mut keyboards = Vec::new();

        let devices = evdev::enumerate();

        for (_path, device) in devices {
            let supported = device.supported_keys();
            if let Some(keys) = supported {
                if keys.contains(Key::KEY_A) && keys.contains(Key::KEY_ENTER) {
                    info!("Found keyboard: {} at {:?}", device.name().unwrap_or("unknown"), device.physical_path());
                    keyboards.push(device);
                }
            }
        }

        keyboards
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
                        1 => { state.pressed_keys.insert(key); }
                        0 => { state.pressed_keys.remove(&key); }
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

    /// Start monitoring all keyboards. Runs until all devices disconnect.
    pub async fn run(self) {
        let keyboards = Self::find_keyboards();
        if keyboards.is_empty() {
            panic!(
                "No keyboards found. Make sure you're in the 'input' group: \
                 sudo usermod -aG input $USER"
            );
        }

        info!("Monitoring {} keyboard(s)", keyboards.len());

        let mut handles = Vec::new();
        for device in keyboards {
            let combos = self.combos.clone();
            let state = Arc::clone(&self.state);
            let tx = self.tx.clone();
            handles.push(tokio::spawn(Self::monitor_device(device, combos, state, tx)));
        }

        // Wait for all monitors (they run until device disconnect)
        for handle in handles {
            let _ = handle.await;
        }
    }
}
