#!/usr/bin/env bash
# Deploy the MX Master 3S input stack.
#
# Stage 1: `logi-mouse-daemon` owns the receiver mouse node (side-button remaps,
# screenshot button swap, X-state hygiene). Input Remapper is kept installed but
# its autoload entry is removed so it never injects again. Solaar keeps the
# HID++ side (gesture, Smart Shift, thumb wheel) for now.
#
# Idempotent. See docs/MOUSE_INPUT_STACK.md.
#
# Usage: infra/install-mouse-stack.sh [--no-build]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
SOLAAR_DIR="$HOME/.config/solaar"
PRESET_DIR="$HOME/.config/input-remapper-2/presets/Logitech USB Receiver"
IR_CONFIG="$HOME/.config/input-remapper-2/config.json"
UNIT_DIR="$HOME/.config/systemd/user"
BUILD=1

[[ "${1:-}" == "--no-build" ]] && BUILD=0

say() { printf '  %s\n' "$*"; }

if (( BUILD )); then
  say "building logi-mouse-daemon"
  ( cd "$REPO_DIR" && cargo build --release --no-default-features --bin logi-mouse-daemon )
fi

say "helper scripts and daemon -> $BIN_DIR"
mkdir -p "$BIN_DIR"
install -m 0755 "$REPO_DIR/target/release/logi-mouse-daemon" "$BIN_DIR/logi-mouse-daemon"
install -m 0755 "$REPO_DIR/infra/flameshot-right-drag"      "$BIN_DIR/flameshot-right-drag"
install -m 0755 "$REPO_DIR/infra/mouse-button-guard"        "$BIN_DIR/mouse-button-guard"
install -m 0755 "$REPO_DIR/infra/whisper-hotkey.py"         "$BIN_DIR/whisper-hotkey"
install -m 0755 "$REPO_DIR/infra/whisper-hotkey-daemon.py"  "$BIN_DIR/whisper-hotkey-daemon"

say "Solaar rules -> $SOLAAR_DIR/rules.yaml"
mkdir -p "$SOLAAR_DIR"
install -m 0664 "$REPO_DIR/infra/solaar/rules.yaml" "$SOLAAR_DIR/rules.yaml"

say "systemd user units -> $UNIT_DIR"
mkdir -p "$UNIT_DIR"
for unit in logi-mouse-daemon.service mouse-button-guard.service whisper-hotkey-daemon.service; do
  install -m 0644 "$REPO_DIR/infra/systemd/$unit" "$UNIT_DIR/$unit"
done
systemctl --user daemon-reload

# Retire Input Remapper: keep the package and preset for rollback, but stop the
# injection and drop the autoload entry so it cannot grab the mouse again.
say "retiring Input Remapper injection"
input-remapper-control --command stop-all >/dev/null 2>&1 || true
if [[ -f "$IR_CONFIG" ]]; then
  python3 - "$IR_CONFIG" "$REPO_DIR/infra/input-remapper/Whisper mouse.json" "$PRESET_DIR/Whisper mouse.json" <<'PY'
import json, os, shutil, sys
config_path, repo_preset, live_preset = sys.argv[1:4]
os.makedirs(os.path.dirname(live_preset), exist_ok=True)
shutil.copyfile(repo_preset, live_preset)
with open(config_path) as fh:
    config = json.load(fh)
if config.get("autoload"):
    shutil.copyfile(config_path, config_path + ".pre-logi-mouse")
    config["autoload"] = {}
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=4)
        fh.write("\n")
PY
fi

say "starting daemon and guard"
systemctl --user enable --now whisper-hotkey-daemon.service mouse-button-guard.service logi-mouse-daemon.service

say "restarting Solaar so keyed settings are re-applied"
systemctl --user restart app-solaar@autostart.service

printf '\nDeployed. Verify with: %s/infra/verify-mouse-stack.sh\n' "$REPO_DIR"
