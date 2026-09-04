# MX Master 3S, Solaar, and screenshot controls

This workstation uses an MX Master 3S under X11 with Solaar and Input
Remapper. The active keyboard layout is Dvorak, so the physical `Ctrl+Z`
combination is the logical GNOME shortcut `Ctrl+;`.

## Current behavior

| Control | Behavior |
|---|---|
| Main wheel | Permanently free-spinning |
| Smart Shift button above the main wheel | Launch right-drag Flameshot selection |
| Physical `Ctrl+Z` | Launch right-drag Flameshot selection |
| Thumb wheel | Page/tab navigation through the existing Solaar rules |
| Forward button | Diverted through the existing Solaar/Input Remapper setup |

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

Solaar rules live in `~/.config/solaar/rules.yaml`. The Smart Shift rule is:

```yaml
---
- Key: [Smart Shift, pressed]
- Execute:
  - /home/mizzlr/.local/bin/flameshot-right-drag
...
```

## Required Solaar state

For the MX Master 3S:

```text
scroll-ratchet = Freespinning
smart-shift = 1
Smart Shift diversion = Diverted
thumb wheel diversion = enabled
```

The persisted keyed diversion map in `~/.config/solaar/config.yaml` keeps the
Forward button (`0x56`) and Smart Shift (`0xc4`) diverted:

```yaml
divert-keys: {0x52: 0x0, 0x53: 0x0, 0x56: 0x1, 0xc3: 0x0, 0xc4: 0x1}
```

Verify the complete live device state with `solaar show`. With Solaar
1.1.11, do not use `solaar config "MX Master 3S" divert-keys "Smart Shift"`
as a read-only query: that incomplete keyed-setting command can serialize the
key name as a scalar and break the persisted diversion map.

## Recovery checks

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
