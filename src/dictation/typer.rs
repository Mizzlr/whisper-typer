//! Text typing using clipboard paste.
//!
//! Keeps clipboard ownership in-process with arboard, then simulates
//! Ctrl+Shift+V with the configured backend. Falls back to xclip if the
//! persistent clipboard is unavailable.

use arboard::Clipboard;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::config::TyperConfig;

const LIVE_XCLIP_STATES: &str = "R,S,D,T,t,W,I";
const XDOTOOL_PASTE_ARGS: [&str; 5] = ["key", "--clearmodifiers", "--delay", "0", "ctrl+shift+v"];

pub struct TextTyper {
    backend: TypingBackend,
    // On Linux the process that owns the X11 clipboard must stay alive to
    // serve paste requests. Keeping this handle for the lifetime of the typer
    // avoids spawning and synchronizing several xclip processes per utterance.
    clipboard: Option<Clipboard>,
}

enum TypingBackend {
    Enigo,
    Xdotool,
}

impl TextTyper {
    pub fn new(config: &TyperConfig) -> Self {
        let backend = if config.backend == "xdotool" {
            // User explicitly wants xdotool
            TypingBackend::Xdotool
        } else {
            // Default: try enigo (arboard + enigo), fallback to xdotool
            TypingBackend::Enigo
        };

        info!(
            "Text typer initialized (backend: {})",
            match &backend {
                TypingBackend::Enigo => "enigo",
                TypingBackend::Xdotool => "xdotool",
            }
        );

        let clipboard = match Clipboard::new() {
            Ok(clipboard) => Some(clipboard),
            Err(error) => {
                warn!("Persistent clipboard unavailable; xclip fallback will be used: {error}");
                None
            }
        };

        Self { backend, clipboard }
    }

    /// Type text into the currently focused window via clipboard paste.
    pub fn type_text(&mut self, text: &str) {
        if text.is_empty() {
            warn!("Empty text, nothing to type");
            return;
        }

        debug!("Typing {} characters", text.len());

        match &self.backend {
            TypingBackend::Enigo => {
                if let Err(e) = self.type_with_enigo(text) {
                    warn!("Enigo failed: {e}, falling back to xdotool");
                    if let Err(e2) = self.type_with_xdotool(text) {
                        warn!("xdotool fallback also failed: {e2}");
                    }
                }
            }
            TypingBackend::Xdotool => {
                if let Err(e) = self.type_with_xdotool(text) {
                    warn!("xdotool failed: {e}");
                }
            }
        }
    }

    fn type_with_enigo(&mut self, text: &str) -> Result<(), String> {
        use enigo::{Direction, Enigo, Key, Keyboard, Settings};

        // Set clipboard
        let clipboard = self
            .clipboard
            .as_mut()
            .ok_or_else(|| "Persistent clipboard is unavailable".to_string())?;
        clipboard
            .set_text(text)
            .map_err(|e| format!("Failed to set clipboard: {e}"))?;

        // Small delay for clipboard sync
        thread::sleep(Duration::from_millis(10));

        // Simulate Ctrl+Shift+V
        let mut enigo =
            Enigo::new(&Settings::default()).map_err(|e| format!("Failed to init enigo: {e}"))?;
        enigo
            .key(Key::Control, Direction::Press)
            .map_err(|e| format!("Key press failed: {e}"))?;
        enigo
            .key(Key::Shift, Direction::Press)
            .map_err(|e| format!("Key press failed: {e}"))?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| format!("Key click failed: {e}"))?;
        enigo
            .key(Key::Shift, Direction::Release)
            .map_err(|e| format!("Key release failed: {e}"))?;
        enigo
            .key(Key::Control, Direction::Release)
            .map_err(|e| format!("Key release failed: {e}"))?;

        debug!("Typed via enigo clipboard paste");
        Ok(())
    }

    fn type_with_xdotool(&mut self, text: &str) -> Result<(), String> {
        let clipboard_started = Instant::now();
        let used_fallback = match self.clipboard.as_mut() {
            Some(clipboard) => match clipboard.set_text(text) {
                Ok(()) => false,
                Err(error) => {
                    warn!("Persistent clipboard write failed; using xclip fallback: {error}");
                    Self::set_clipboard_with_xclip(text)?;
                    true
                }
            },
            None => {
                Self::set_clipboard_with_xclip(text)?;
                true
            }
        };
        let clipboard_ms = clipboard_started.elapsed().as_secs_f64() * 1000.0;

        // Paste with xdotool. A zero inter-key delay is safe for this single
        // shortcut and avoids xdotool's default delay between synthetic events.
        let paste_started = Instant::now();
        let status = Command::new("xdotool")
            .args(XDOTOOL_PASTE_ARGS)
            .status()
            .map_err(|e| format!("xdotool failed: {e}"))?;

        if !status.success() {
            return Err("xdotool exited with non-zero status".to_string());
        }

        let paste_ms = paste_started.elapsed().as_secs_f64() * 1000.0;
        info!(
            "Typing stages: clipboard={clipboard_ms:.1}ms paste={paste_ms:.1}ms fallback={used_fallback}"
        );
        debug!("Typed via persistent clipboard + xdotool paste");
        Ok(())
    }

    fn set_clipboard_with_xclip(text: &str) -> Result<(), String> {
        Self::clear_xclip_owners();

        // xclip forks a selection owner process after stdin closes. Use the
        // default (unlimited) loop count so the owner PERSISTS and keeps serving
        // the clipboard until the next dictation's clear_xclip_owners() replaces
        // it. A finite -loops N is fatal here: our wait_for_clipboard_text()
        // verification reads the selection several times, exhausting the loops,
        // so xclip exits and the clipboard goes empty before xdotool can paste
        // (and stays empty, breaking later manual paste into other apps). The
        // pkill guard in clear_xclip_owners() still caps us at one live xclip.
        let mut child = Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn xclip: {e}"))?;

        if let Some(stdin) = child.stdin.take() {
            use std::io::Write;
            let mut stdin = stdin;
            stdin
                .write_all(text.as_bytes())
                .map_err(|e| format!("Failed to write to xclip: {e}"))?;
        }
        child.wait().map_err(|e| format!("xclip failed: {e}"))?;

        Self::wait_for_clipboard_text(text)?;
        Ok(())
    }

    fn clear_xclip_owners() {
        let _ = Command::new("pkill").args(["-x", "xclip"]).status();

        for _ in 0..20 {
            let has_live_xclip = Command::new("pgrep")
                .args(["-x", "-r", LIVE_XCLIP_STATES, "xclip"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false);

            if !has_live_xclip {
                return;
            }

            thread::sleep(Duration::from_millis(10));
        }

        warn!("Timed out waiting for old live xclip clipboard owner to exit");
    }

    fn wait_for_clipboard_text(text: &str) -> Result<(), String> {
        for _ in 0..20 {
            match Command::new("xclip")
                .args(["-selection", "clipboard", "-out"])
                .output()
            {
                Ok(output) if output.status.success() && output.stdout == text.as_bytes() => {
                    return Ok(());
                }
                Ok(_) | Err(_) => {
                    thread::sleep(Duration::from_millis(10));
                }
            }
        }

        Err("Timed out waiting for clipboard to contain new text".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{LIVE_XCLIP_STATES, XDOTOOL_PASTE_ARGS};

    #[test]
    fn xdotool_paste_has_no_artificial_key_delay() {
        assert!(XDOTOOL_PASTE_ARGS
            .windows(2)
            .any(|pair| pair == ["--delay", "0"]));
    }

    #[test]
    fn xclip_cleanup_ignores_zombie_processes() {
        assert!(!LIVE_XCLIP_STATES.split(',').any(|state| state == "Z"));
    }

    #[test]
    fn xclip_cleanup_still_waits_for_running_or_sleeping_processes() {
        let states: Vec<&str> = LIVE_XCLIP_STATES.split(',').collect();
        assert!(states.contains(&"R"));
        assert!(states.contains(&"S"));
    }
}
