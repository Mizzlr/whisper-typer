#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for command_name in codex uv xdotool xprop xrandr import tesseract; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

if ! /usr/bin/python3 -c 'import pyatspi' >/dev/null 2>&1; then
  echo "Missing python3-pyatspi. Install it with: sudo apt-get install python3-pyatspi" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" || -z "${XAUTHORITY:-}" ]]; then
  echo "DISPLAY and XAUTHORITY must be set from the graphical X11 session." >&2
  exit 1
fi

uv sync --project "$project_dir" --group dev

if codex mcp get ubuntu-computer-use >/dev/null 2>&1; then
  codex mcp remove ubuntu-computer-use
fi

codex mcp add ubuntu-computer-use \
  --env "DISPLAY=$DISPLAY" \
  --env "XAUTHORITY=$XAUTHORITY" \
  --env "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-x11}" \
  -- uv run --project "$project_dir" ubuntu-computer-use-mcp

codex mcp get ubuntu-computer-use
