#!/usr/bin/env bash
# Deploy the MX Master 3S input stack (Solaar + Input Remapper + helpers) from
# this repo into the live session. Idempotent: safe to re-run.
#
# See docs/MOUSE_INPUT_STACK.md for the architecture and recovery runbook.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
SOLAAR_DIR="$HOME/.config/solaar"
PRESET_DIR="$HOME/.config/input-remapper-2/presets/Logitech USB Receiver"
UNIT_DIR="$HOME/.config/systemd/user"

say() { printf '  %s\n' "$*"; }

say "helper scripts -> $BIN_DIR"
mkdir -p "$BIN_DIR"
install -m 0755 "$REPO_DIR/infra/flameshot-right-drag"       "$BIN_DIR/flameshot-right-drag"
install -m 0755 "$REPO_DIR/infra/mouse-button-guard"         "$BIN_DIR/mouse-button-guard"
install -m 0755 "$REPO_DIR/infra/whisper-hotkey.py"          "$BIN_DIR/whisper-hotkey"
install -m 0755 "$REPO_DIR/infra/whisper-hotkey-daemon.py"   "$BIN_DIR/whisper-hotkey-daemon"

say "Solaar rules -> $SOLAAR_DIR/rules.yaml"
mkdir -p "$SOLAAR_DIR"
install -m 0664 "$REPO_DIR/infra/solaar/rules.yaml" "$SOLAAR_DIR/rules.yaml"

say "Input Remapper preset -> $PRESET_DIR/Whisper mouse.json"
mkdir -p "$PRESET_DIR"
install -m 0664 "$REPO_DIR/infra/input-remapper/Whisper mouse.json" \
  "$PRESET_DIR/Whisper mouse.json"

say "systemd user units -> $UNIT_DIR"
mkdir -p "$UNIT_DIR"
install -m 0644 "$REPO_DIR/infra/systemd/mouse-button-guard.service"   "$UNIT_DIR/mouse-button-guard.service"
install -m 0644 "$REPO_DIR/infra/systemd/whisper-hotkey-daemon.service" "$UNIT_DIR/whisper-hotkey-daemon.service"
systemctl --user daemon-reload
systemctl --user enable --now whisper-hotkey-daemon.service mouse-button-guard.service

say "restarting Solaar so keyed settings (diversion, reprogrammable keys) are re-applied"
systemctl --user restart app-solaar@autostart.service

say "reloading Input Remapper presets"
input-remapper-control --command stop-all >/dev/null 2>&1 || true
input-remapper-control --command autoload

say "expected Solaar device settings live in $REPO_DIR/infra/solaar/mx-master-3s-settings.yaml"
printf '\nDeployed. Verify with: %s/infra/verify-mouse-stack.sh\n' "$REPO_DIR"
