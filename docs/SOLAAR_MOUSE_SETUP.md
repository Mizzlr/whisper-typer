# MX Master 3S, Solaar, and screenshot controls

This workstation uses an MX Master 3S under X11 with Solaar and Input
Remapper. The active keyboard layout is Dvorak, so the physical `Ctrl+Z`
combination is the logical GNOME shortcut `Ctrl+;`.

> The complete stack record — ownership rules, mirrored configs under `infra/`,
> deployment, verification and the recovery runbook — lives in
> [MOUSE_INPUT_STACK.md](MOUSE_INPUT_STACK.md). This page is the short
> operational summary.

## Current behavior

| Control | Behavior |
|---|---|
| Main wheel | Permanently free-spinning |
| Smart Shift button above the main wheel | Launch right-drag Flameshot selection |
| Physical `Ctrl+Z` | Launch right-drag Flameshot selection |
| Thumb wheel | Page/tab navigation through the existing Solaar rules |
| Forward button | Mapped to paste (`Ctrl` + the key that types `v`) by `logi-mouse-daemon` |
| Back button | Mapped to `KEY_ENTER` (submit) by `logi-mouse-daemon` |
| Hidden Gesture button | Diverted in Solaar to trigger Whisper Typer push-to-talk (`KEY_F24`) via `whisper-hotkey-daemon` |

Ownership split: `logi-mouse-daemon` owns the ordinary buttons that already emit
standard evdev codes (left, right, middle, Back, Forward). Solaar owns the exotic
controls that only exist as HID++ controls (Gesture button, Smart Shift, thumb
wheel). The side buttons must stay **undiverted** in Solaar — a diverted button
never reaches `/dev/input/event7`, so the daemon can no longer see it.

Run `systemctl --user status logi-mouse-daemon` and
`tail -f ~/.whisper-typer-history/mouse/$(date +%F).jsonl` to watch what the
daemon does with each button.

The screenshot launcher is installed at
`~/.local/bin/flameshot-right-drag`. It finds the forwarded Input Remapper
pointer, saves the original X11 button map, swaps physical left and right,
starts Flameshot, and restores the original map through a shell trap. A
runtime lock prevents overlapping screenshot sessions.

GNOME binds the wrapper at:

```text
org.gnome.settings-daemon.plugins.media-keys.custom-keybinding
/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/
binding: <Control>semicolon
command: /home/mizzlr/.local/bin/flameshot-right-drag
```

Solaar rules live in `~/.config/solaar/rules.yaml`:

```yaml
---
- Key: [Smart Shift, pressed]
- Execute:
  - /home/mizzlr/.local/bin/flameshot-right-drag
...
---
- Key: [Mouse Gesture Button, pressed]
- Execute:
  - /home/mizzlr/.local/bin/whisper-hotkey
  - press
...
---
- Key: [Mouse Gesture Button, released]
- Execute:
  - /home/mizzlr/.local/bin/whisper-hotkey
  - release
...
```

The gesture hotkey daemon is managed by `whisper-hotkey-daemon.service`
(installed at `~/.local/bin/whisper-hotkey-daemon`). It creates a persistent
`uinput` virtual keyboard (`whisper-gesture-keyboard`) with `KEY_F24`, so that
Whisper Typer can monitor push-to-talk press and release without X11 clipboard
or cursor-blinking interference.

## Required Solaar state

For the MX Master 3S:

```text
scroll-ratchet = Freespinning
smart-shift = 1
Smart Shift diversion = Diverted
Forward Button diversion = Regular
Mouse Gesture Button diversion = Diverted
thumb wheel diversion = enabled
```

The persisted keyed diversion map in `~/.config/solaar/config.yaml` keeps the
Gesture button (`0xc3`) and Smart Shift (`0xc4`) diverted, and leaves the
Forward button (`0x56`) regular so Input Remapper receives it:

```yaml
divert-keys: {0x52: 0x0, 0x53: 0x0, 0x56: 0x0, 0xc3: 0x1, 0xc4: 0x1}
reprogrammable-keys: {0x50: 0x50, 0x51: 0x51, 0x52: 0x52, 0x53: 0x53, 0x56: 0x56, 0xc3: 195, 0xc4: 0xc4}
```

The Input Remapper preset
`~/.config/input-remapper-2/presets/Logitech USB Receiver/Whisper mouse.json` is
kept on disk as the rollback path, but it is no longer autoloaded — the daemon
owns the mouse node now. It still holds both side-button mappings:

```json
{"input_combination": [{"type": 1, "code": 275, "origin_hash": "3053316a9883deb9b2680fdf4ec5566b"}],
 "target_uinput": "keyboard", "output_symbol": "KEY_ENTER", "mapping_type": "key_macro"}
{"input_combination": [{"type": 1, "code": 276, "origin_hash": "3053316a9883deb9b2680fdf4ec5566b"}],
 "target_uinput": "keyboard", "output_symbol": "Control_L + v", "mapping_type": "key_macro"}
```

The `origin_hash` binds the preset to one specific device node
(`md5(capabilities + name)`), so extra mice on the machine are unaffected.
Input Remapper resolves macro symbols through its `xmodmap.json`, which is
layout-aware — on Dvorak the symbol `v` resolves to keycode 52.

> [!IMPORTANT]
> Do not map `KEY_F24` to a mouse button. Whisper Typer's fallback hotkey has
> exactly one producer: the `whisper-hotkey-daemon`. A second F24 producer
> (Solaar paste rule, stale Input Remapper mapping) makes one button press emit
> two hotkeys and dictation flaps.

> [!IMPORTANT]
> In `reprogrammable-keys`, the Mouse Gesture Button (`0xc3`) must remain set to
> `195` (`Gesture Button Navigation`) and never mapped to `0x56` (`Mouse Forward Button`).
> Mapping `0xc3` to `0x56` causes the mouse hardware to alias the gesture button with
> the forward button, making them execute the same paste action and causing cursor-flicker
> during dictation.

Verify the complete live device state with `solaar show`. With Solaar
1.1.11, do not use `solaar config "MX Master 3S" divert-keys "Smart Shift"`
as a read-only query: that incomplete keyed-setting command can serialize the
key name as a scalar and break the persisted diversion map.

## Recovery checks

If the pointer moves and scrolls but **clicks do nothing**, the grabbed
receiver node is holding a stale button-down state: Input Remapper restarted
between a button press and its release, so X treats every later click as part
of an ongoing drag. Check and clear it with:

```bash
xinput query-state "Logitech USB Receiver Mouse" | grep 'button\[1\]'
xinput disable "Logitech USB Receiver Mouse"
xinput enable  "Logitech USB Receiver Mouse"
```

`mouse-button-guard.service` (`~/.local/bin/mouse-button-guard`) watches for
this and clears it within a couple of seconds: it only acts when the raw node
claims a button is held while the forwarded node — the one X actually reads —
reports it released.

```bash
systemctl --user status app-solaar@autostart.service
solaar show
xinput get-button-map "input-remapper Logitech USB Receiver Mouse forwarded"
gsettings get \
  org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/ \
  command
```

Outside a screenshot overlay, the X11 button map must begin with `1 2 3`.
During the overlay it temporarily begins with `3 2 1`.
