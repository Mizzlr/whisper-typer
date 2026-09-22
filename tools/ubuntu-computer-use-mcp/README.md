# Ubuntu Computer Use MCP

A local Model Context Protocol server that lets Codex inspect and operate an
Ubuntu X11 desktop while keeping explicit track of workspaces, monitors,
windows, terminal panes, and accessible GUI controls.

This project lives inside Whisper Typer because both tools are local desktop
infrastructure. It does not store screenshots, OCR output, window titles, or
accessibility snapshots on disk.

## Control model

The server resolves targets in this order:

1. **Workspace** — discover and switch by zero-based workspace number.
2. **Monitor** — report each monitor's global geometry and primary status.
3. **Window** — address by X11 window ID, or by a title/class query only when
   that query has exactly one match.
4. **Pane or tab** — inspect terminal processes by PID, TTY, and working
   directory; inspect graphical panes, tabs, and controls through AT-SPI.
5. **Coordinates** — click relative to the selected window after focus is
   verified. Global coordinates remain available for desktop-level UI.

All mutating operations share a process-independent lock. A workflow can save
the current workspace, focused window, and pointer, do its work, and restore
that context afterward. Typing never presses Enter implicitly.

## MCP tools

| Tool | Purpose |
|---|---|
| desktop_state | Workspaces, monitors, active window, pointer, and windows |
| observe / observe_region | Screenshot a selected window or desktop region |
| read_screen | Local Tesseract OCR |
| focus_window / switch_workspace | Verified navigation |
| click / type_text / press_keys / scroll | Serialized input operations |
| accessibility_tree | Named GUI controls, tabs, panes, bounds, and actions |
| activate_accessible | Invoke one exact, unique accessible control |
| terminal_panes | Visible pane bounds plus TTY, foreground process, and CWD |
| capture_desktop_context / restore_desktop_context | Save and restore user context |
| wait_for_window | Wait for an unambiguous application window |

## Requirements

- Ubuntu running an X11 session
- Python 3.10 or newer and uv
- xdotool, x11-utils, ImageMagick, Tesseract, and python3-pyatspi

Install the Ubuntu packages with:

```bash
sudo apt-get install xdotool x11-utils imagemagick tesseract-ocr python3-pyatspi
```

Install the Python environment and run the tests:

```bash
uv sync --group dev
uv run pytest
```

Register the server in Codex:

```bash
./scripts/register-codex.sh
```

The registration captures the current DISPLAY and XAUTHORITY values because a
desktop server cannot connect to X11 without them. Start a new Codex session
after registration so it discovers the new MCP tools.

## Operating sequence

For reliable work across several windows and workspaces:

1. Call desktop_state and select a unique window ID.
2. Call capture_desktop_context.
3. Focus the target and observe it or inspect its accessibility tree.
4. Perform one bounded action.
5. Observe again and verify the visible result.
6. Restore the saved context.

Window IDs and accessibility paths can change when an application restarts or
rebuilds its UI. Refresh state before acting rather than carrying them across
long workflows. Context tokens exist only for the lifetime of the MCP server.
Terminator shares one application PID across its windows, so the tool reports
window-exact pane regions and labels its process/TTY inventory as
application-wide instead of pretending that association is exact.

## Current boundary

Version 0.1 supports X11. Wayland deliberately fails at the display-command
layer until a portal-based capture and input backend is added. Some
applications expose incomplete AT-SPI trees; screenshots and OCR are the local
fallback in those cases.
