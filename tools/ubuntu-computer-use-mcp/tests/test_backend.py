from __future__ import annotations

import subprocess
from contextlib import contextmanager

import pytest

from ubuntu_computer_use_mcp.backend import DesktopError, Window, X11Desktop


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], str] | None = None):
        self.responses = responses or {}
        self.calls: list[list[str]] = []

    def run(self, args, **kwargs):
        self.calls.append(args)
        value = self.responses.get(tuple(args), "")
        return subprocess.CompletedProcess(args, 0, stdout=value, stderr="")


def sample_window(window_id=0x123, title="Editor — Notes", workspace=2):
    return Window(window_id, 101, workspace, "code, Code", title, 3000, 20, 1000, 800)


def test_unquote_preserves_utf8_and_parses_wm_class():
    assert X11Desktop._unquote('"(1) ‎⁨bruce <> Astralane⁩ – (45186)"') == "(1) ‎⁨bruce <> Astralane⁩ – (45186)"
    assert X11Desktop._unquote('"Navigator", "firefox"') == "Navigator, firefox"


def test_monitor_topology_parses_offsets_and_primary():
    runner = FakeRunner({
        ("xrandr", "--listmonitors"): "Monitors: 2\n 0: +*DP-5 3440/798x1440/334+3000+0  DP-5\n 1: +DP-1 1080/527x1920/296+1920+0  DP-1\n"
    })
    monitors = X11Desktop(runner).monitors()
    assert monitors == [
        {"index": 0, "name": "DP-5", "primary": True, "x": 3000, "y": 0, "width": 3440, "height": 1440},
        {"index": 1, "name": "DP-1", "primary": False, "x": 1920, "y": 0, "width": 1080, "height": 1920},
    ]


def test_ambiguous_window_query_refuses_to_guess(monkeypatch):
    desktop = X11Desktop(FakeRunner())
    monkeypatch.setattr(desktop, "windows", lambda: [sample_window(), sample_window(0x456, "Editor — Code")])
    with pytest.raises(DesktopError, match="ambiguous"):
        desktop.find_window(query="Editor")


def test_click_uses_one_lock_and_window_relative_coordinates(monkeypatch):
    runner = FakeRunner()
    desktop = X11Desktop(runner)
    window = sample_window()
    depth = 0
    max_depth = 0

    @contextmanager
    def tracking_lock():
        nonlocal depth, max_depth
        depth += 1
        max_depth = max(max_depth, depth)
        yield
        depth -= 1

    monkeypatch.setattr(desktop, "operation_lock", tracking_lock)
    monkeypatch.setattr(desktop, "active_window_id", lambda: window.window_id)
    monkeypatch.setattr(desktop, "window", lambda _: window)

    result = desktop.click(25, 40, window=window, button=1, count=1)

    assert max_depth == 1
    assert (result["x"], result["y"]) == (3025, 60)
    assert ["xdotool", "set_desktop", "2"] in runner.calls
    assert ["xdotool", "windowactivate", "--sync", str(window.window_id)] in runner.calls
    assert ["xdotool", "mousemove", "--sync", "3025", "60", "click", "--repeat", "1", "--delay", "100", "1"] in runner.calls


def test_click_rejects_coordinates_outside_window_before_input(monkeypatch):
    runner = FakeRunner()
    desktop = X11Desktop(runner)
    window = sample_window()
    monkeypatch.setattr(desktop, "active_window_id", lambda: window.window_id)
    monkeypatch.setattr(desktop, "window", lambda _: window)
    with pytest.raises(DesktopError, match="outside"):
        desktop.click(window.width, 10, window=window, button=1, count=1)
    assert not any("mousemove" in call for call in runner.calls)


def test_type_text_uses_stdin_and_requires_explicit_return(monkeypatch):
    runner = FakeRunner()
    desktop = X11Desktop(runner)
    window = sample_window()
    monkeypatch.setattr(desktop, "active_window_id", lambda: window.window_id)
    monkeypatch.setattr(desktop, "window", lambda _: window)

    desktop.type_text("--literal $HOME", window=window)
    type_call = next(call for call in runner.calls if call[:2] == ["xdotool", "type"])
    assert type_call[-2:] == ["--file", "-"]
    assert "--literal $HOME" not in type_call

    with pytest.raises(DesktopError, match="line breaks"):
        desktop.type_text("command\n", window=window)
