#!/bin/bash
set -e

# Whisper Typer RS — installation script.
# Builds the Rust workspace, deploys binaries to ~/.local/bin, installs the
# systemd user service, and configures udev rules for evdev hotkey access.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$INFRA_DIR/.." && pwd)"

echo "=== Whisper Typer RS Setup ==="
echo "Repo: $REPO_DIR"
echo

if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Run as a regular user, not root.${NC}"
    exit 1
fi

# 1. Input group (required for evdev hotkey + uinput key simulation)
echo -e "${YELLOW}[1/5] Input group membership...${NC}"
if groups "$USER" | grep -q '\binput\b'; then
    echo -e "${GREEN}  $USER is in input group${NC}"
else
    sudo usermod -aG input "$USER"
    echo -e "${YELLOW}  Added $USER to input group — log out and back in${NC}"
fi

# 2. udev rule for /dev/uinput
echo -e "${YELLOW}[2/5] udev rules...${NC}"
UDEV_RULE="/etc/udev/rules.d/99-uinput.rules"
if [ -f "$UDEV_RULE" ]; then
    echo -e "${GREEN}  $UDEV_RULE already exists${NC}"
else
    sudo cp "$INFRA_DIR/udev/99-uinput.rules" "$UDEV_RULE"
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo -e "${GREEN}  udev rules installed${NC}"
fi

# 3. Build the Rust workspace
echo -e "${YELLOW}[3/5] Building Rust binaries (release)...${NC}"
cd "$REPO_DIR"
cargo build --release
echo -e "${GREEN}  Built: whisper-typer-rs, tts-hook, voice-journal${NC}"

# 4. Deploy binaries to ~/.local/bin (so Claude Code hooks can find tts-hook)
echo -e "${YELLOW}[4/5] Deploying binaries to ~/.local/bin/...${NC}"
mkdir -p "$HOME/.local/bin"
install -m 0755 "$REPO_DIR/target/release/whisper-typer-rs" "$HOME/.local/bin/whisper-typer-rs"
install -m 0755 "$REPO_DIR/target/release/tts-hook"         "$HOME/.local/bin/tts-hook"
install -m 0755 "$REPO_DIR/target/release/voice-journal"    "$HOME/.local/bin/voice-journal"
echo -e "${GREEN}  Installed to ~/.local/bin/${NC}"

# 5. systemd user service
echo -e "${YELLOW}[5/5] systemd user service...${NC}"
mkdir -p "$HOME/.config/systemd/user"
install -m 0644 "$INFRA_DIR/systemd/whisper-typer.service" \
    "$HOME/.config/systemd/user/whisper-typer-rs.service"
systemctl --user daemon-reload
systemctl --user enable whisper-typer-rs.service
echo -e "${GREEN}  Service installed (whisper-typer-rs.service)${NC}"

echo
echo "=== Setup Complete ==="
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Log out and back in (for input group to take effect)"
echo "  2. Pull the Ollama model: ollama pull gemma4:e2b"
echo "  3. Download Whisper + Kokoro models into models/ (see README.md)"
echo "  4. Start the service: systemctl --user start whisper-typer-rs"
echo "  5. Tail logs:         journalctl --user -fu whisper-typer-rs"
echo
echo -e "${GREEN}Usage:${NC} Hold Win+Alt while speaking, release to transcribe and type."
