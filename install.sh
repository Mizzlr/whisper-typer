#!/bin/bash
set -e

echo "=== Whisper Input Setup ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Check if running as regular user
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please run this script as a regular user, not root.${NC}"
    exit 1
fi

# 2. Add user to input group (required for evdev and uinput)
echo -e "${YELLOW}[1/7] Checking input group membership...${NC}"
if groups $USER | grep -q '\binput\b'; then
    echo -e "${GREEN}  User $USER is already in input group${NC}"
else
    echo "  Adding $USER to input group..."
    sudo usermod -aG input $USER
    echo -e "${YELLOW}  NOTE: You will need to log out and back in for this change to take effect${NC}"
fi

# 3. Create udev rules for uinput
echo -e "${YELLOW}[2/7] Setting up udev rules...${NC}"
UDEV_RULE="/etc/udev/rules.d/99-uinput.rules"
if [ -f "$UDEV_RULE" ]; then
    echo -e "${GREEN}  udev rule already exists${NC}"
else
    echo "  Creating uinput udev rule..."
    sudo cp "$SCRIPT_DIR/udev/99-uinput.rules" "$UDEV_RULE"
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo -e "${GREEN}  udev rules configured${NC}"
fi

# 4. Install ydotool
echo -e "${YELLOW}[3/7] Checking ydotool...${NC}"
if command -v ydotool &> /dev/null; then
    echo -e "${GREEN}  ydotool is already installed${NC}"
else
    echo "  Installing ydotool..."
    sudo apt-get update
    sudo apt-get install -y ydotool
fi

# 5. Enable ydotoold service (if available - older versions don't need it)
echo -e "${YELLOW}[4/7] Configuring ydotoold service...${NC}"
if command -v ydotoold &> /dev/null; then
    if systemctl --user is-active ydotoold.service &> /dev/null; then
        echo -e "${GREEN}  ydotoold is already running${NC}"
    else
        systemctl --user enable ydotoold.service 2>/dev/null || true
        systemctl --user start ydotoold.service 2>/dev/null || true
        echo -e "${GREEN}  ydotoold service enabled${NC}"
    fi
else
    echo -e "${GREEN}  Using ydotool 0.1.x (no daemon required)${NC}"
fi

# 6. Install system dependencies
echo -e "${YELLOW}[5/7] Installing system dependencies...${NC}"
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    portaudio19-dev \
    libgirepository1.0-dev \
    gir1.2-notify-0.7 \
    ffmpeg

echo -e "${GREEN}  System dependencies installed${NC}"

# 7. Install Python package
echo -e "${YELLOW}[6/7] Installing whisper-input Python package...${NC}"
cd "$SCRIPT_DIR"
pip install -e . --break-system-packages 2>/dev/null || pip install -e .
echo -e "${GREEN}  Python package installed${NC}"

# 8. Install systemd service
echo -e "${YELLOW}[7/7] Installing systemd user service...${NC}"
mkdir -p ~/.config/systemd/user
cp "$SCRIPT_DIR/systemd/whisper-input.service" ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable whisper-input.service
echo -e "${GREEN}  systemd service installed${NC}"

echo
echo "=== Setup Complete ==="
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Log out and back in (for input group to take effect)"
echo "  2. Pull the Ollama model: ollama pull llama3.2:3b"
echo "  3. Test manually: python -m whisper_input"
echo "  4. Or start the service: systemctl --user start whisper-input"
echo
echo -e "${GREEN}Usage:${NC} Hold Win+Alt while speaking, release to transcribe and type."
echo
