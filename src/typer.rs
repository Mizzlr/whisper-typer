//! Text typing using clipboard paste.
//!
//! Sets clipboard with arboard, then simulates Ctrl+Shift+V with enigo.
//! Falls back to xdotool + xclip if enigo fails.

use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;
use tracing::{debug, info, warn};

use crate::config::TyperConfig;

pub struct TextTyper {
    backend: TypingBackend,
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

        Self { backend }
    }

    /// Type text into the currently focused window via clipboard paste.
    pub fn type_text(&self, text: &str) {
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

    fn type_with_enigo(&self, text: &str) -> Result<(), String> {
        use arboard::Clipboard;
        use enigo::{Direction, Enigo, Key, Keyboard, Settings};

        // Set clipboard
        let mut clipboard =
            Clipboard::new().map_err(|e| format!("Failed to open clipboard: {e}"))?;
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

    fn type_with_xdotool(&self, text: &str) -> Result<(), String> {
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

        // Paste with xdotool
        let status = Command::new("xdotool")
            .args(["key", "--clearmodifiers", "ctrl+shift+v"])
            .status()
            .map_err(|e| format!("xdotool failed: {e}"))?;

        if !status.success() {
            return Err("xdotool exited with non-zero status".to_string());
        }

        debug!("Typed via xdotool clipboard paste");
        Ok(())
    }

    fn clear_xclip_owners() {
        let _ = Command::new("pkill").args(["-x", "xclip"]).status();

        for _ in 0..20 {
            let has_xclip = Command::new("pgrep")
                .args(["-x", "xclip"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false);

            if !has_xclip {
                return;
            }

            thread::sleep(Duration::from_millis(10));
        }

        warn!("Timed out waiting for old xclip clipboard owner to exit");
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
