from __future__ import annotations

import json
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP, Image

from .backend import X11Desktop

mcp = FastMCP("ubuntu-computer-use")
desktop = X11Desktop()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _target(window_id: str = "", query: str = ""):
    return desktop.find_window(window_id=window_id or None, query=query)


@mcp.tool()
def desktop_state() -> str:
    """Inspect workspaces, monitors, pointer, active window and all managed windows."""
    return _json(desktop.state())


@mcp.tool()
def observe(window_id: str = "", query: str = "") -> Image:
    """Capture the full desktop, or one unambiguous window selected by ID/title/class.

    Prefer a window capture after desktop_state; full-desktop captures can span
    several monitors and contain unrelated private content.
    """
    window = _target(window_id, query) if window_id or query else None
    return Image(data=desktop.screenshot(window=window), format="png")


@mcp.tool()
def observe_region(x: int, y: int, width: int, height: int) -> Image:
    """Capture a rectangle in global desktop coordinates."""
    return Image(data=desktop.screenshot(region=(x, y, width, height)), format="png")


@mcp.tool()
def read_screen(window_id: str = "", query: str = "") -> str:
    """OCR the full desktop or one selected window with local Tesseract."""
    window = _target(window_id, query) if window_id or query else None
    image = desktop.screenshot(window=window)
    proc = desktop.runner.run(["tesseract", "stdin", "stdout"], timeout=30, input_bytes=image, binary=True)
    return proc.stdout.decode("utf-8", errors="replace").strip()


@mcp.tool()
def focus_window(window_id: str = "", query: str = "") -> str:
    """Switch workspace, focus one unambiguous window, and verify focus."""
    return _json(desktop.focus(_target(window_id, query)).as_dict())


@mcp.tool()
def switch_workspace(workspace: int) -> str:
    """Switch to a zero-based workspace and verify the result."""
    return _json({"current_workspace": desktop.switch_workspace(workspace)})


@mcp.tool()
def click(
    x: int,
    y: int,
    window_id: str = "",
    query: str = "",
    button: int = 1,
    count: int = 1,
) -> str:
    """Click at window-relative coordinates, or global coordinates without a target.

    Use desktop_state and observe immediately beforehand. A query must resolve to
    exactly one window; otherwise this tool refuses to guess.
    """
    window = _target(window_id, query) if window_id or query else None
    return _json(desktop.click(x, y, window=window, button=button, count=count))


@mcp.tool()
def type_text(text: str, window_id: str = "", query: str = "", delay_ms: int = 5) -> str:
    """Focus a selected window and type literal text without pressing Enter."""
    return _json(desktop.type_text(text, window=_target(window_id, query), delay_ms=delay_ms))


@mcp.tool()
def press_keys(keys: str, window_id: str = "", query: str = "") -> str:
    """Focus a window and press xdotool key names, e.g. 'ctrl+l' or 'Tab Return'."""
    return _json(desktop.press_keys(keys, window=_target(window_id, query)))


@mcp.tool()
def scroll(direction: str, steps: int = 3, window_id: str = "", query: str = "") -> str:
    """Scroll in the focused window or a selected target."""
    window = _target(window_id, query) if window_id or query else None
    return _json(desktop.scroll(direction, steps, window=window))


@mcp.tool()
def wait_for_window(query: str, timeout_seconds: float = 10) -> str:
    """Wait for exactly one visible window matching title or application class."""
    return _json(desktop.wait_for_window(query, timeout_seconds).as_dict())


@mcp.tool()
def capture_desktop_context() -> str:
    """Save current workspace, focused window and pointer for later restoration."""
    token, context = desktop.capture_context()
    return _json({"token": token, **asdict(context), "active_window": hex(context.active_window) if context.active_window else None})


@mcp.tool()
def restore_desktop_context(token: str) -> str:
    """Restore and consume a token returned by capture_desktop_context."""
    context = desktop.restore_context(token)
    return _json({**asdict(context), "active_window": hex(context.active_window) if context.active_window else None})


@mcp.tool()
def terminal_panes(window_id: str = "", query: str = "") -> str:
    """Inspect terminal panes, TTY sessions, foreground processes and working directories.

    For apps such as Terminator that share one process across windows, visible
    pane bounds are window-exact while TTY sessions are labeled application-wide.
    """
    window = _target(window_id, query) if window_id or query else None
    return _json(desktop.terminal_panes(window))


@mcp.tool()
def accessibility_tree(
    window_id: str = "",
    query: str = "",
    max_depth: int = 6,
    max_nodes: int = 500,
) -> str:
    """Inspect named GUI controls, tabs and panes in one selected window.

    This uses Ubuntu AT-SPI semantics and includes screen bounds and supported
    actions where the application exposes them.
    """
    return _json(desktop.accessibility_tree(_target(window_id, query), max_depth, max_nodes))


@mcp.tool()
def activate_accessible(
    name: str,
    window_id: str = "",
    query: str = "",
    role: str = "",
    action: str = "",
) -> str:
    """Invoke an exact, unique AT-SPI control by accessible name and optional role/action."""
    return _json(desktop.activate_accessible(_target(window_id, query), name, role, action))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
