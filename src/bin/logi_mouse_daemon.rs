//! Logitech mouse input daemon.
//!
//! Stage 1 of consolidating the MX Master 3S input stack into one process. It
//! replaces Input Remapper for this mouse and folds in the screenshot button
//! swap and the recovery hygiene that used to live in separate scripts.
//!
//! Responsibilities
//! ----------------
//! * Grab the receiver's mouse node (default `Logitech USB Receiver Mouse`) so
//!   the raw node stops driving X.
//! * Create a uinput mouse mirroring the source capabilities and forward motion,
//!   wheel and button events through it.
//! * Remap the side buttons: Back -> `KEY_ENTER` (submit), Forward -> `Ctrl` plus
//!   the key that types `v` (paste), resolved from the live X keymap so a Dvorak
//!   layout pastes the right character.
//! * Serve a unix socket so the screenshot launcher can swap logical left and
//!   right for the duration of a Flameshot overlay, with an auto-expiry so a
//!   lost `swap-off` cannot leave the buttons inverted.
//! * Journal every button and wheel event plus every remap decision, so a
//!   misbehaving control can be reconstructed after the fact.
//! * Repair the X button state of the raw node on session start and end. A stale
//!   button-down there is what used to wedge all clicks after an injector
//!   restart.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use evdev::uinput::{VirtualDevice, VirtualDeviceBuilder};
use evdev::{AttributeSet, Device, EventType, InputEvent, Key, RelativeAxisType};
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

const BTN_LEFT: u16 = Key::BTN_LEFT.code();
const BTN_RIGHT: u16 = Key::BTN_RIGHT.code();
const BTN_BACK: u16 = Key::BTN_SIDE.code();
const BTN_FORWARD: u16 = Key::BTN_EXTRA.code();
const KEY_ENTER: u16 = Key::KEY_ENTER.code();
const KEY_CTRL: u16 = Key::KEY_LEFTCTRL.code();
const REL_WHEEL: u16 = RelativeAxisType::REL_WHEEL.0;
const REL_HWHEEL: u16 = RelativeAxisType::REL_HWHEEL.0;
const REL_WHEEL_HI_RES: u16 = RelativeAxisType::REL_WHEEL_HI_RES.0;
const REL_HWHEEL_HI_RES: u16 = RelativeAxisType::REL_HWHEEL_HI_RES.0;

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[derive(Parser, Debug, Clone)]
#[command(
    name = "logi-mouse-daemon",
    version,
    about = "Own the Logitech mouse node and remap its side buttons"
)]
struct Args {
    /// evdev device name to own
    #[arg(long, default_value = "Logitech USB Receiver Mouse")]
    device_name: String,

    /// Explicit source device path; overrides --device-name (used by tests)
    #[arg(long)]
    source_path: Option<PathBuf>,

    /// Name of the uinput mouse presented to X
    #[arg(long, default_value = "logi-mouse forwarded")]
    sink_name: String,

    /// Control socket path
    #[arg(long)]
    socket_path: Option<PathBuf>,

    /// evdev keycode used for the paste keystroke (default: from the X keymap)
    #[arg(long)]
    paste_keycode: Option<u16>,

    /// Directory for the JSONL event journal (default: ~/.whisper-typer-history/mouse)
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Seconds before a screenshot swap expires on its own
    #[arg(long, default_value_t = 300)]
    swap_timeout: u64,

    /// Also journal relative motion events (high volume)
    #[arg(long)]
    trace_motion: bool,

    /// Skip the `xinput disable/enable` hygiene pass on the raw node
    #[arg(long)]
    no_x_hygiene: bool,

    /// Send a control command to a running daemon and exit
    /// (`swap-on`, `swap-off`, `status`)
    #[arg(long, value_name = "COMMAND")]
    control: Option<String>,
}

/// Translation state that must survive between events.
#[derive(Default)]
struct State {
    /// Screenshot mode: logical left and right are swapped.
    swap: bool,
    /// The Forward button is physically held, so its release must undo Ctrl.
    forward_down: bool,
    /// Physical code -> emitted code for every button currently held.
    held: HashMap<u16, u16>,
}

fn key_event(code: u16, value: i32) -> InputEvent {
    InputEvent::new(EventType::KEY, code, value)
}

/// Map one source event to zero or more events for the uinput mouse.
///
/// Pure apart from `state`, which is what makes the remaps unit-testable.
fn translate(state: &mut State, ev: InputEvent, paste: u16) -> Vec<InputEvent> {
    if ev.event_type() != EventType::KEY {
        return vec![ev];
    }
    let code = ev.code();
    let value = ev.value();
    match code {
        BTN_BACK => match value {
            1 => vec![key_event(KEY_ENTER, 1)],
            0 => vec![key_event(KEY_ENTER, 0)],
            _ => Vec::new(),
        },
        BTN_FORWARD => match value {
            1 => {
                state.forward_down = true;
                vec![key_event(KEY_CTRL, 1), key_event(paste, 1)]
            }
            0 => {
                if state.forward_down {
                    state.forward_down = false;
                    vec![key_event(paste, 0), key_event(KEY_CTRL, 0)]
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        },
        BTN_LEFT | BTN_RIGHT => {
            let other = if code == BTN_LEFT { BTN_RIGHT } else { BTN_LEFT };
            match value {
                1 => {
                    let emitted = if state.swap { other } else { code };
                    state.held.insert(code, emitted);
                    vec![key_event(emitted, 1)]
                }
                0 => {
                    // Release whatever was pressed, even if the swap flipped
                    // while the button was held.
                    let emitted = state.held.remove(&code).unwrap_or(code);
                    vec![key_event(emitted, 0)]
                }
                _ => Vec::new(),
            }
        }
        _ => vec![ev],
    }
}

fn key_name(code: u16) -> String {
    match code {
        BTN_LEFT => "BTN_LEFT".into(),
        BTN_RIGHT => "BTN_RIGHT".into(),
        BTN_BACK => "BTN_SIDE(back)".into(),
        BTN_FORWARD => "BTN_EXTRA(forward)".into(),
        KEY_ENTER => "KEY_ENTER".into(),
        KEY_CTRL => "KEY_LEFTCTRL".into(),
        other => format!("code:{other}"),
    }
}

fn rel_name(code: u16) -> String {
    match code {
        REL_WHEEL => "REL_WHEEL".into(),
        REL_HWHEEL => "REL_HWHEEL".into(),
        REL_WHEEL_HI_RES => "REL_WHEEL_HI_RES".into(),
        REL_HWHEEL_HI_RES => "REL_HWHEEL_HI_RES".into(),
        other => format!("code:{other}"),
    }
}

/// JSONL event journal: one line per button/wheel event and per remap decision.
struct Journal {
    dir: Option<PathBuf>,
    trace_motion: bool,
    open: Option<(String, File)>,
}

impl Journal {
    fn new(dir: Option<PathBuf>, trace_motion: bool) -> Self {
        Self {
            dir,
            trace_motion,
            open: None,
        }
    }

    fn path_for(&self, day: &str) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(format!("{day}.jsonl")))
    }

    fn write_line(&mut self, line: &str) {
        let Some(dir) = self.dir.clone() else { return };
        if std::fs::create_dir_all(&dir).is_err() {
            return;
        }
        let day = chrono::Local::now().format("%Y-%m-%d").to_string();
        let stale = match &self.open {
            Some((open_day, _)) => open_day != &day,
            None => true,
        };
        if stale {
            let Some(path) = self.path_for(&day) else { return };
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => self.open = Some((day.clone(), f)),
                Err(err) => {
                    debug!(error = %err, "cannot open mouse journal");
                    self.open = None;
                    return;
                }
            }
        }
        if let Some((_, file)) = self.open.as_mut() {
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
        }
    }

    fn record_event(&mut self, dev: &str, ev: &InputEvent, out: &[InputEvent], action: &str) {
        let is_key = ev.event_type() == EventType::KEY;
        let is_wheel = ev.event_type() == EventType::RELATIVE
            && matches!(
                ev.code(),
                REL_WHEEL | REL_HWHEEL | REL_WHEEL_HI_RES | REL_HWHEEL_HI_RES
            );
        if !is_key && !is_wheel && !self.trace_motion {
            return;
        }
        let ts = chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let (kind, code) = if is_key {
            ("key", key_name(ev.code()))
        } else if is_wheel {
            ("wheel", rel_name(ev.code()))
        } else {
            ("motion", format!("rel:{}", ev.code()))
        };
        let outputs: Vec<String> = out
            .iter()
            .map(|o| {
                if o.event_type() == EventType::KEY {
                    format!("{}{}", key_name(o.code()), if o.value() == 1 { "+" } else { "-" })
                } else {
                    format!("{}={}", rel_name(o.code()), o.value())
                }
            })
            .collect();
        let record = serde_json::json!({
            "ts": ts,
            "kind": kind,
            "dev": dev,
            "code": code,
            "code_raw": ev.code(),
            "value": ev.value(),
            "action": action,
            "out": outputs,
        });
        self.write_line(&record.to_string());
    }

    fn note(&mut self, kind: &str, msg: &str) {
        let ts = chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let record = serde_json::json!({ "ts": ts, "kind": kind, "msg": msg });
        self.write_line(&record.to_string());
    }
}

fn default_socket_path() -> PathBuf {
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/run/user/{uid}/logi-mouse.sock"))
}

/// Client side of the control socket: used by the screenshot launcher.
fn send_control(socket_path: &std::path::Path, command: &str) -> io::Result<()> {
    let socket = std::os::unix::net::UnixDatagram::unbound()?;
    socket.send_to(command.as_bytes(), socket_path)?;
    Ok(())
}

fn default_log_dir() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".whisper-typer-history/mouse")
}

/// Resolve the keycode that types `v` under the live X keymap.
///
/// X keycodes are evdev codes plus 8, and a Dvorak layout puts `v` on a
/// different physical key than QWERTY, so this is resolved rather than
/// hard-coded.
fn paste_keycode_from_xmodmap() -> Option<u16> {
    let out = Command::new("xmodmap").arg("-pke").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some("keycode") {
            continue;
        }
        let xcode: u16 = match parts.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        if parts.next() != Some("=") {
            continue;
        }
        if parts.next() == Some("v") {
            return xcode.checked_sub(8);
        }
    }
    None
}

fn resolve_paste_keycode(explicit: Option<u16>) -> u16 {
    if let Some(code) = explicit {
        return code;
    }
    match paste_keycode_from_xmodmap() {
        Some(code) => code,
        None => {
            warn!("could not read the X keymap; falling back to keycode 52 (Dvorak v)");
            52
        }
    }
}

fn open_source(args: &Args) -> io::Result<Device> {
    if let Some(path) = &args.source_path {
        return Device::open(path);
    }
    for (path, device) in evdev::enumerate() {
        if device.name() == Some(args.device_name.as_str()) {
            debug!(path = %path.display(), name = %args.device_name, "source device found");
            return Ok(device);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("no device named {}", args.device_name),
    ))
}

/// Capabilities for the uinput mouse: mirror the source and add the keys this
/// daemon synthesises.
fn mirror_capabilities(
    device: &Device,
    paste: u16,
) -> (AttributeSet<Key>, AttributeSet<RelativeAxisType>) {
    let mut keys = AttributeSet::<Key>::new();
    if let Some(source_keys) = device.supported_keys() {
        for key in source_keys.iter() {
            keys.insert(key);
        }
    }
    keys.insert(Key::KEY_ENTER);
    keys.insert(Key::KEY_LEFTCTRL);
    keys.insert(Key::new(paste));

    let mut axes = AttributeSet::<RelativeAxisType>::new();
    if let Some(source_axes) = device.supported_relative_axes() {
        for axis in source_axes.iter() {
            axes.insert(axis);
        }
    }
    (keys, axes)
}

fn build_sink(
    name: &str,
    keys: &AttributeSet<Key>,
    axes: &AttributeSet<RelativeAxisType>,
) -> io::Result<VirtualDevice> {
    VirtualDeviceBuilder::new()?
        .name(name)
        .with_keys(keys)?
        .with_relative_axes(axes)?
        .build()
}

/// Clear X's button state for the raw node.
///
/// While this daemon grabs the node, X keeps whatever button state it last saw.
/// If a release was swallowed by a restart, that stale "down" makes every later
/// click look like part of a drag, so the state is re-initialised on every
/// session boundary.
fn x_hygiene(device_name: &str, journal: &Mutex<Journal>) {
    if std::env::var_os("DISPLAY").is_none() {
        return;
    }
    let disabled = Command::new("xinput").args(["disable", device_name]).output();
    let enabled = Command::new("xinput").args(["enable", device_name]).output();
    let ok = matches!(&disabled, Ok(o) if o.status.success())
        && matches!(&enabled, Ok(o) if o.status.success());
    if ok {
        debug!(device = device_name, "x input state re-initialised");
    } else {
        warn!(device = device_name, "xinput hygiene pass failed");
    }
    if let Ok(mut j) = journal.lock() {
        j.note(
            "state",
            &format!("x_hygiene device={device_name} ok={ok}"),
        );
    }
}

fn release_held(state: &mut State, sink: &mut VirtualDevice, journal: &Mutex<Journal>) {
    let held: Vec<u16> = state.held.drain().map(|(_, emitted)| emitted).collect();
    for code in &held {
        let _ = sink.emit(&[key_event(*code, 0)]);
    }
    if state.forward_down {
        let _ = sink.emit(&[key_event(KEY_CTRL, 0)]);
        state.forward_down = false;
    }
    if !held.is_empty() {
        info!(count = held.len(), "released buttons still held at shutdown");
        if let Ok(mut j) = journal.lock() {
            j.note("state", &format!("released_on_exit count={}", held.len()));
        }
    }
}

extern "C" fn on_signal(_sig: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    let handler = on_signal as *const () as libc::sighandler_t;
    unsafe {
        libc::signal(libc::SIGTERM, handler);
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGHUP, handler);
    }
}

/// Serve `swap-on` / `swap-off` from the screenshot launcher.
fn spawn_control_socket(
    path: PathBuf,
    swap: Arc<AtomicBool>,
    journal: Arc<Mutex<Journal>>,
) {
    thread::spawn(move || {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::remove_file(&path);
        let socket = match std::os::unix::net::UnixDatagram::bind(&path) {
            Ok(s) => s,
            Err(err) => {
                warn!(error = %err, path = %path.display(), "cannot bind control socket");
                return;
            }
        };
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        let _ = socket.set_read_timeout(Some(Duration::from_millis(500)));
        info!(path = %path.display(), "control socket ready");

        let mut buf = [0u8; 64];
        loop {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break;
            }
            match socket.recv(&mut buf) {
                Ok(n) => {
                    let cmd = String::from_utf8_lossy(&buf[..n]).trim().to_string();
                    let change = match cmd.as_str() {
                        "swap-on" | "1" | "on" => {
                            swap.store(true, Ordering::SeqCst);
                            Some("swap=on")
                        }
                        "swap-off" | "0" | "off" => {
                            swap.store(false, Ordering::SeqCst);
                            Some("swap=off")
                        }
                        "status" => None,
                        other => {
                            debug!(command = other, "ignoring control message");
                            None
                        }
                    };
                    if let Some(change) = change {
                        info!(command = change, "control");
                        if let Ok(mut j) = journal.lock() {
                            j.note("state", change);
                        }
                    }
                }
                Err(err)
                    if matches!(
                        err.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) => {}
                Err(err) => {
                    warn!(error = %err, "control socket receive failed");
                    thread::sleep(Duration::from_millis(200));
                }
            }
        }
        let _ = std::fs::remove_file(&path);
    });
}

fn run(args: &Args, journal: Arc<Mutex<Journal>>, swap: Arc<AtomicBool>) {
    let paste = resolve_paste_keycode(args.paste_keycode);
    info!(paste_keycode = paste, "paste keystroke keycode");

    loop {
        if SHUTDOWN.load(Ordering::SeqCst) {
            return;
        }

        let mut device = match open_source(args) {
            Ok(device) => device,
            Err(err) => {
                info!(error = %err, "source device not available; retrying");
                thread::sleep(Duration::from_millis(1000));
                continue;
            }
        };
        let name = device.name().unwrap_or("(unnamed)").to_string();
        let paste_code = paste;
        let (keys, axes) = mirror_capabilities(&device, paste_code);

        if !args.no_x_hygiene {
            x_hygiene(&args.device_name, &journal);
        }

        // Grab before creating the sink so X never sees both pointers at once.
        if let Err(err) = device.grab() {
            warn!(error = %err, device = %name, "cannot grab source device");
            thread::sleep(Duration::from_millis(1000));
            continue;
        }
        let mut sink = match build_sink(&args.sink_name, &keys, &axes) {
            Ok(sink) => sink,
            Err(err) => {
                warn!(error = %err, "cannot create uinput mouse");
                let _ = device.ungrab();
                thread::sleep(Duration::from_millis(1000));
                continue;
            }
        };
        info!(device = %name, sink = %args.sink_name, "owning mouse node");
        if let Ok(mut j) = journal.lock() {
            j.note("state", &format!("session_start device={name}"));
        }

        let fd = device.as_raw_fd();
        let mut state = State::default();
        let mut swap_deadline: Option<Instant> = None;
        let mut session_ok = true;

        'session: loop {
            if SHUTDOWN.load(Ordering::SeqCst) {
                break 'session;
            }

            // Apply and expire the screenshot swap.
            let requested = swap.load(Ordering::SeqCst);
            if requested && !state.swap {
                state.swap = true;
                swap_deadline = Some(Instant::now() + Duration::from_secs(args.swap_timeout));
                if let Ok(mut j) = journal.lock() {
                    j.note("state", "swap_active");
                }
            } else if !requested && state.swap {
                state.swap = false;
                swap_deadline = None;
                if let Ok(mut j) = journal.lock() {
                    j.note("state", "swap_cleared");
                }
            }
            if let Some(deadline) = swap_deadline {
                if state.swap && Instant::now() >= deadline {
                    state.swap = false;
                    swap_deadline = None;
                    swap.store(false, Ordering::SeqCst);
                    warn!("screenshot swap expired without swap-off");
                    if let Ok(mut j) = journal.lock() {
                        j.note("state", "swap_expired");
                    }
                }
            }

            let mut pollfd = libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let rc = unsafe { libc::poll(&mut pollfd, 1, 250) };
            if rc < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                warn!(error = %err, "poll failed");
                session_ok = false;
                break 'session;
            }
            if rc == 0 {
                continue;
            }

            let events = match device.fetch_events() {
                Ok(events) => events,
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => continue,
                Err(err) => {
                    warn!(error = %err, device = %name, "source read failed; reconnecting");
                    session_ok = false;
                    break 'session;
                }
            };

            for ev in events {
                let out = translate(&mut state, ev, paste_code);
                let action = if out.is_empty() {
                    "suppressed"
                } else if out.len() == 1 && out[0].code() == ev.code() && out[0].value() == ev.value() {
                    "passthrough"
                } else {
                    "remap"
                };
                if !out.is_empty() {
                    if let Err(err) = sink.emit(&out) {
                        warn!(error = %err, "uinput write failed");
                    }
                }
                if let Ok(mut j) = journal.lock() {
                    j.record_event(&name, &ev, &out, action);
                }
                if action == "remap" || action == "suppressed" {
                    debug!(
                        code = %key_name(ev.code()),
                        value = ev.value(),
                        action,
                        out = out.len(),
                        "input event translated"
                    );
                }
            }
        }

        release_held(&mut state, &mut sink, &journal);
        drop(sink);
        let _ = device.ungrab();
        if !args.no_x_hygiene {
            x_hygiene(&args.device_name, &journal);
        }
        if let Ok(mut j) = journal.lock() {
            j.note("state", &format!("session_end ok={session_ok}"));
        }
        if session_ok {
            return;
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn main() -> io::Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();
    let socket_path = args.socket_path.clone().unwrap_or_else(default_socket_path);

    if let Some(command) = args.control.as_deref() {
        send_control(&socket_path, command)?;
        return Ok(());
    }

    install_signal_handlers();

    let log_dir = args.log_dir.clone().unwrap_or_else(default_log_dir);
    let journal = Arc::new(Mutex::new(Journal::new(
        Some(log_dir.clone()),
        args.trace_motion,
    )));
    let swap = Arc::new(AtomicBool::new(false));

    info!(
        device = %args.device_name,
        sink = %args.sink_name,
        socket = %socket_path.display(),
        journal = %log_dir.display(),
        "logi-mouse-daemon starting"
    );
    if let Ok(mut j) = journal.lock() {
        j.note("state", "daemon_start");
    }

    spawn_control_socket(socket_path, swap.clone(), journal.clone());
    run(&args, journal.clone(), swap.clone());

    if let Ok(mut j) = journal.lock() {
        j.note("state", "daemon_stop");
    }
    info!("logi-mouse-daemon stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PASTE: u16 = 52; // Dvorak `v`

    fn press(code: u16) -> InputEvent {
        key_event(code, 1)
    }
    fn release(code: u16) -> InputEvent {
        key_event(code, 0)
    }
    fn codes(events: &[InputEvent]) -> Vec<(u16, i32)> {
        events.iter().map(|e| (e.code(), e.value())).collect()
    }

    #[test]
    fn back_button_submits() {
        let mut state = State::default();
        assert_eq!(
            codes(&translate(&mut state, press(BTN_BACK), PASTE)),
            vec![(KEY_ENTER, 1)]
        );
        assert_eq!(
            codes(&translate(&mut state, release(BTN_BACK), PASTE)),
            vec![(KEY_ENTER, 0)]
        );
    }

    #[test]
    fn forward_button_pastes() {
        let mut state = State::default();
        assert_eq!(
            codes(&translate(&mut state, press(BTN_FORWARD), PASTE)),
            vec![(KEY_CTRL, 1), (PASTE, 1)]
        );
        assert_eq!(
            codes(&translate(&mut state, release(BTN_FORWARD), PASTE)),
            vec![(PASTE, 0), (KEY_CTRL, 0)]
        );
        // A stray release must not emit a second Ctrl-up.
        assert!(translate(&mut state, release(BTN_FORWARD), PASTE).is_empty());
    }

    #[test]
    fn forward_repeat_is_suppressed() {
        let mut state = State::default();
        let _ = translate(&mut state, press(BTN_FORWARD), PASTE);
        assert!(translate(&mut state, key_event(BTN_FORWARD, 2), PASTE).is_empty());
    }

    #[test]
    fn passthrough_for_other_buttons_and_motion() {
        let mut state = State::default();
        assert_eq!(
            codes(&translate(&mut state, press(BTN_LEFT), PASTE)),
            vec![(BTN_LEFT, 1)]
        );
        let motion = InputEvent::new(EventType::RELATIVE, 0, 5);
        let out = translate(&mut state, motion, PASTE);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].code(), motion.code());
    }

    #[test]
    fn screenshot_swap_tracks_press_and_release() {
        let mut state = State::default();
        state.swap = true;
        assert_eq!(
            codes(&translate(&mut state, press(BTN_LEFT), PASTE)),
            vec![(BTN_RIGHT, 1)]
        );
        // Swap flips while the button is held: the release still matches.
        state.swap = false;
        assert_eq!(
            codes(&translate(&mut state, release(BTN_LEFT), PASTE)),
            vec![(BTN_RIGHT, 0)]
        );
        assert_eq!(
            codes(&translate(&mut state, press(BTN_RIGHT), PASTE)),
            vec![(BTN_RIGHT, 1)]
        );
    }

    #[test]
    fn paste_keycode_is_configurable() {
        let mut state = State::default();
        let out = translate(&mut state, press(BTN_FORWARD), 47); // QWERTY `v`
        assert_eq!(codes(&out), vec![(KEY_CTRL, 1), (47, 1)]);
    }
}
