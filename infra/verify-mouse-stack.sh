#!/usr/bin/env bash
# Read-only verification of the MX Master 3S input stack (stage 1: the daemon
# owns the mouse node; Solaar keeps the HID++ side).
# Exits non-zero if any check fails. See docs/MOUSE_INPUT_STACK.md.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$HOME/.local/bin"
SOLAAR_DIR="$HOME/.config/solaar"
UNIT_DIR="$HOME/.config/systemd/user"
IR_CONFIG="$HOME/.config/input-remapper-2/config.json"

RAW="Logitech USB Receiver Mouse"
SINK="logi-mouse forwarded"

pass=0; fail=0
ok()   { printf '  [PASS] %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  [FAIL] %s\n' "$1"; fail=$((fail+1)); }
check(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

echo "services"
for svc in app-solaar@autostart logi-mouse-daemon mouse-button-guard whisper-hotkey-daemon whisper-typer-rs; do
  check "$svc active" "systemctl --user is-active --quiet $svc"
done

echo "deployed artifacts"
check "logi-mouse-daemon installed" "test -x '$BIN_DIR/logi-mouse-daemon'"
check "flameshot-right-drag matches repo" "cmp -s '$REPO_DIR/infra/flameshot-right-drag' '$BIN_DIR/flameshot-right-drag'"
check "mouse-button-guard matches repo" "cmp -s '$REPO_DIR/infra/mouse-button-guard' '$BIN_DIR/mouse-button-guard'"
check "whisper-hotkey client matches repo" "cmp -s '$REPO_DIR/infra/whisper-hotkey.py' '$BIN_DIR/whisper-hotkey'"
check "whisper-hotkey daemon matches repo" "cmp -s '$REPO_DIR/infra/whisper-hotkey-daemon.py' '$BIN_DIR/whisper-hotkey-daemon'"
check "solaar rules.yaml matches repo" "cmp -s '$REPO_DIR/infra/solaar/rules.yaml' '$SOLAAR_DIR/rules.yaml'"
check "logi-mouse-daemon.service matches repo" "cmp -s '$REPO_DIR/infra/systemd/logi-mouse-daemon.service' '$UNIT_DIR/logi-mouse-daemon.service'"
check "mouse-button-guard.service matches repo" "cmp -s '$REPO_DIR/infra/systemd/mouse-button-guard.service' '$UNIT_DIR/mouse-button-guard.service'"

echo "input remapper retired"
check "no input-remapper autoload entries" \
  "! python3 -c \"import json,sys; sys.exit(0 if json.load(open('$IR_CONFIG')).get('autoload') else 1)\""
if xinput list --name-only 2>/dev/null | grep -q '^input-remapper'; then
  printf '  [INFO] input-remapper still exposes idle uinput nodes (package kept for rollback)\n'
fi

echo "daemon owns the mouse node"
check "control socket present" "test -S /run/user/$(id -u)/logi-mouse.sock"
check "uinput sink present" "xinput list --name-only 2>/dev/null | grep -qx '$SINK'"
sink_id="$(xinput list --id-only "pointer:$SINK" 2>/dev/null | head -n1)"
if [[ -n "$sink_id" ]]; then
  check "sink button map starts 1 2 3" \
    "xinput get-button-map '$sink_id' | tr -s ' ' | grep -q '^1 2 3 '"
else
  bad "sink pointer device not resolvable by name"
fi

daemon_pid="$(systemctl --user show -p MainPID --value logi-mouse-daemon 2>/dev/null)"
python3 - "$daemon_pid" <<'PY'
import subprocess, sys
daemon_pid = sys.argv[1]
raw_name = "Logitech USB Receiver Mouse"
try:
    import evdev, fcntl
except Exception:
    print("  [SKIP] evdev unavailable for grab check"); sys.exit(0)
paths = []
for path in evdev.list_devices():
    try:
        dev = evdev.InputDevice(path)
    except OSError:
        continue
    if dev.name == raw_name:
        paths.append(path)
if not paths:
    print(f"  [FAIL] raw node {raw_name!r} not present"); sys.exit(1)

owned = False
if daemon_pid and daemon_pid != "0":
    try:
        fds = subprocess.run(
            ["ls", "-l", f"/proc/{daemon_pid}/fd"],
            capture_output=True, text=True, check=True,
        ).stdout
        owned = any(path in fds for path in paths)
    except subprocess.CalledProcessError:
        owned = False
if not owned:
    print(f"  [FAIL] daemon pid {daemon_pid} does not hold the raw node"); sys.exit(1)
print(f"  [PASS] daemon (pid {daemon_pid}) holds the raw node")

held = False
for path in paths:
    dev = evdev.InputDevice(path)
    try:
        fcntl.ioctl(dev.fd, 0x40044590, 1)
        fcntl.ioctl(dev.fd, 0x40044590, 0)
    except OSError:
        held = True
    finally:
        dev.close()
if held:
    print("  [PASS] raw node grab is exclusive"); sys.exit(0)
print("  [FAIL] raw node is not grabbed"); sys.exit(1)
PY
[[ $? -eq 0 ]] && pass=$((pass+1)) || fail=$((fail+1))

stuck=""
for id in $(xinput list --id-only 2>/dev/null); do
  s="$(xinput query-state "$id" 2>/dev/null | grep -o 'button\[[0-9]*\]=down' | tr '\n' ' ')"
  [[ -n "$s" ]] && stuck+="id$id:$s "
done
if [[ -z "$stuck" ]]; then ok "no stuck pointer buttons"; else bad "stuck pointer buttons: $stuck"; fi

echo "live MX Master 3S state (solaar show)"
show="$(timeout 30 solaar show 2>/dev/null)"
check "Forward button regular (daemon owns it)" \
  "printf '%s' \"\$show\" | grep -q 'Key/Button Diversion *: .*Forward Button:Regular'"
check "Back button regular" \
  "printf '%s' \"\$show\" | grep -q 'Key/Button Diversion *: .*Back Button:Regular'"
check "Gesture button diverted" "printf '%s' \"\$show\" | grep -q 'Mouse Gesture Button:Diverted'"
check "Smart Shift diverted" "printf '%s' \"\$show\" | grep -q 'Smart Shift:Diverted'"
check "gesture button not aliased to Forward" \
  "printf '%s' \"\$show\" | grep -q 'Mouse Gesture Button:Gesture Button Navigation'"
check "thumb wheel diverted" "printf '%s' \"\$show\" | grep -q 'Thumb Wheel Diversion *: True'"
check "main wheel free-spinning" "printf '%s' \"\$show\" | grep -q 'Scroll Wheel Ratcheted *: Freespinning'"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ $fail -eq 0 ]]
