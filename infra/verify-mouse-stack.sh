#!/usr/bin/env bash
# Read-only verification of the MX Master 3S input stack.
# Exits non-zero if any check fails. See docs/MOUSE_INPUT_STACK.md.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
SOLAAR_DIR="$HOME/.config/solaar"
PRESET_DIR="$HOME/.config/input-remapper-2/presets/Logitech USB Receiver"
UNIT_DIR="$HOME/.config/systemd/user"

RAW="Logitech USB Receiver Mouse"
FWD="input-remapper Logitech USB Receiver Mouse forwarded"

pass=0; fail=0
ok()   { printf '  [PASS] %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=$((fail+1)); }
check(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

echo "services"
for svc in app-solaar@autostart whisper-hotkey-daemon whisper-typer-rs mouse-button-guard; do
  check "$svc active" "systemctl --user is-active --quiet $svc"
done
check "input-remapper active" "systemctl is-active --quiet input-remapper"

echo "deployed files match repo"
check "flameshot-right-drag"  "cmp -s '$REPO_DIR/infra/flameshot-right-drag' '$BIN_DIR/flameshot-right-drag'"
check "mouse-button-guard"    "cmp -s '$REPO_DIR/infra/mouse-button-guard' '$BIN_DIR/mouse-button-guard'"
check "whisper-hotkey client" "cmp -s '$REPO_DIR/infra/whisper-hotkey.py' '$BIN_DIR/whisper-hotkey'"
check "whisper-hotkey daemon" "cmp -s '$REPO_DIR/infra/whisper-hotkey-daemon.py' '$BIN_DIR/whisper-hotkey-daemon'"
check "solaar rules.yaml"     "cmp -s '$REPO_DIR/infra/solaar/rules.yaml' '$SOLAAR_DIR/rules.yaml'"
check "input-remapper preset" "cmp -s '$REPO_DIR/infra/input-remapper/Whisper mouse.json' '$PRESET_DIR/Whisper mouse.json'"
check "mouse-button-guard.service" "cmp -s '$REPO_DIR/infra/systemd/mouse-button-guard.service' '$UNIT_DIR/mouse-button-guard.service'"

echo "live MX Master 3S state (solaar show)"
show="$(timeout 30 solaar show 2>/dev/null)"
check "Forward button regular (Input Remapper owns it)" \
  "printf '%s' \"\$show\" | grep -q 'Key/Button Diversion *: .*Forward Button:Regular'"
check "Back button regular" \
  "printf '%s' \"\$show\" | grep -q 'Key/Button Diversion *: .*Back Button:Regular'"
check "Gesture button diverted" \
  "printf '%s' \"\$show\" | grep -q 'Mouse Gesture Button:Diverted'"
check "Smart Shift diverted" \
  "printf '%s' \"\$show\" | grep -q 'Smart Shift:Diverted'"
check "gesture button not aliased to Forward" \
  "printf '%s' \"\$show\" | grep -q 'Mouse Gesture Button:Gesture Button Navigation'"
check "thumb wheel diverted" \
  "printf '%s' \"\$show\" | grep -q 'Thumb Wheel Diversion *: True'"
check "main wheel free-spinning" \
  "printf '%s' \"\$show\" | grep -q 'Scroll Wheel Ratcheted *: Freespinning'"

echo "X pointer state"
raw_id="$(xinput list --id-only "$RAW" 2>/dev/null | head -n1)"
fwd_id="$(xinput list --id-only "$FWD" 2>/dev/null | head -n1)"
for label in "raw:$raw_id" "forwarded:$fwd_id"; do
  name="${label%%:*}"; id="${label##*:}"
  if [[ -z "$id" ]]; then bad "$name pointer present"; continue; fi
  ok "$name pointer present (id $id)"
  check "$name button map starts 1 2 3" \
    "xinput get-button-map $id | tr -s ' ' | grep -q '^1 2 3 '"
done

stuck=""
for id in $(xinput list --id-only 2>/dev/null); do
  s="$(xinput query-state "$id" 2>/dev/null | grep -o 'button\[[0-9]*\]=down' | tr '\n' ' ')"
  [[ -n "$s" ]] && stuck+="id$id:$s "
done
if [[ -z "$stuck" ]]; then ok "no stuck pointer buttons"; else bad "stuck pointer buttons: $stuck"; fi

echo "Input Remapper injection"
check "hotkey socket present" "test -S /run/user/$(id -u)/whisper-hotkey.sock"
python3 - <<'PY'
import sys
try:
    import evdev, fcntl
except Exception:
    print("  [SKIP] evdev not available for grab check"); sys.exit(0)
try:
    d = evdev.InputDevice("/dev/input/event7")
except Exception as e:
    print(f"  [FAIL] cannot open /dev/input/event7: {e}"); sys.exit(1)
try:
    fcntl.ioctl(d.fd, 0x40044590, 1)
    fcntl.ioctl(d.fd, 0x40044590, 0)
    print("  [FAIL] injector does not hold the grab on event7"); sys.exit(1)
except OSError:
    print("  [PASS] injector holds the grab on event7"); sys.exit(0)
PY
[[ $? -eq 0 ]] && pass=$((pass+1)) || fail=$((fail+1))

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
