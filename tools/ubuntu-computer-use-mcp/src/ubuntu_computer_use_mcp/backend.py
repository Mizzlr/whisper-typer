from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


class DesktopError(RuntimeError):
    pass


@dataclass(frozen=True)
class Window:
    window_id: int
    pid: int | None
    workspace: int | None
    wm_class: str
    title: str
    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict:
        data = asdict(self)
        data["window_id"] = hex(self.window_id)
        return data


@dataclass(frozen=True)
class DesktopContext:
    workspace: int
    active_window: int | None
    pointer_x: int
    pointer_y: int


class Runner:
    def run(
        self,
        args: list[str],
        *,
        timeout: float = 5,
        input_bytes: bytes | None = None,
        binary: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                args,
                input=input_bytes,
                capture_output=True,
                text=not binary,
                timeout=timeout,
                check=check,
            )
        except FileNotFoundError as exc:
            raise DesktopError(f"Required command is not installed: {args[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise DesktopError(f"Command timed out after {timeout}s: {args[0]}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
            raise DesktopError(f"{args[0]} failed: {(stderr or '').strip()}") from exc


class X11Desktop:
    _PROPERTY_VALUE = re.compile(r"=\s*(.*)$")
    _HEX_ID = re.compile(r"0x[0-9a-fA-F]+")
    _GEOMETRY = re.compile(r"^(X|Y|WIDTH|HEIGHT)=(-?\d+)$", re.MULTILINE)
    _SAFE_KEYS = re.compile(r"^[A-Za-z0-9_+\- ]+$")

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner or Runner()
        self.contexts: dict[str, DesktopContext] = {}
        self.lock_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "ubuntu-computer-use.lock"

    def _text(self, args: list[str], timeout: float = 5, check: bool = True) -> str:
        return self.runner.run(args, timeout=timeout, check=check).stdout.strip()

    @contextmanager
    def operation_lock(self, timeout: float = 3) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise DesktopError("Another desktop-control operation is active")
                    time.sleep(0.05)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _root_ids(self) -> list[int]:
        raw = self._text(["xprop", "-root", "_NET_CLIENT_LIST_STACKING"])
        return [int(value, 16) for value in self._HEX_ID.findall(raw)]

    def _property(self, window_id: int, name: str) -> str:
        raw = self._text(["xprop", "-id", str(window_id), name], check=False)
        match = self._PROPERTY_VALUE.search(raw)
        if not match or "not found" in raw.lower():
            return ""
        return match.group(1).strip()

    @staticmethod
    def _unquote(value: str) -> str:
        quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', value)
        if quoted:
            return ", ".join(item.replace(r'\"', '"').replace(r"\\", "\\") for item in quoted)
        return value

    def window(self, window_id: int) -> Window | None:
        geometry = self._text(["xdotool", "getwindowgeometry", "--shell", str(window_id)], check=False)
        fields = {key: int(value) for key, value in self._GEOMETRY.findall(geometry)}
        if not {"X", "Y", "WIDTH", "HEIGHT"}.issubset(fields):
            return None
        pid_raw = self._property(window_id, "_NET_WM_PID")
        workspace_raw = self._property(window_id, "_NET_WM_DESKTOP")
        title = self._unquote(self._property(window_id, "_NET_WM_NAME") or self._property(window_id, "WM_NAME"))
        wm_class = self._unquote(self._property(window_id, "WM_CLASS"))
        pid_match = re.search(r"\d+", pid_raw)
        workspace_match = re.search(r"\d+", workspace_raw)
        return Window(
            window_id=window_id,
            pid=int(pid_match.group()) if pid_match else None,
            workspace=int(workspace_match.group()) if workspace_match else None,
            wm_class=wm_class,
            title=title,
            x=fields["X"],
            y=fields["Y"],
            width=fields["WIDTH"],
            height=fields["HEIGHT"],
        )

    def windows(self) -> list[Window]:
        return [window for wid in self._root_ids() if (window := self.window(wid))]

    def find_window(
        self,
        *,
        window_id: str | int | None = None,
        query: str = "",
        workspace: int | None = None,
    ) -> Window:
        if window_id is not None:
            wid = int(window_id, 0) if isinstance(window_id, str) else window_id
            window = self.window(wid)
            if not window:
                raise DesktopError(f"Window does not exist: {window_id}")
            return window
        if not query:
            raise DesktopError("Provide window_id or a title/class query")
        needle = query.casefold()
        matches = [
            window
            for window in self.windows()
            if needle in f"{window.title} {window.wm_class}".casefold()
            and (workspace is None or window.workspace == workspace)
        ]
        if not matches:
            raise DesktopError(f"No window matches {query!r}")
        if len(matches) > 1:
            summary = [{"window_id": hex(w.window_id), "workspace": w.workspace, "title": w.title} for w in matches]
            raise DesktopError(f"Window query is ambiguous: {json.dumps(summary, ensure_ascii=False)}")
        return matches[0]

    def active_window_id(self) -> int | None:
        raw = self._text(["xdotool", "getactivewindow"], check=False)
        return int(raw) if raw.isdigit() and int(raw) else None

    def current_workspace(self) -> int:
        return int(self._text(["xdotool", "get_desktop"]))

    def desktop_count(self) -> int:
        return int(self._text(["xdotool", "get_num_desktops"]))

    def monitors(self) -> list[dict]:
        raw = self._text(["xrandr", "--listmonitors"])
        monitors: list[dict] = []
        pattern = re.compile(r"^\s*(\d+):\s+([^ ]+)\s+(\d+)/\d+x(\d+)/\d+\+(-?\d+)\+(-?\d+)\s+(\S+)$")
        for line in raw.splitlines()[1:]:
            match = pattern.match(line)
            if match:
                index, flags, width, height, x, y, name = match.groups()
                monitors.append({"index": int(index), "name": name, "primary": "*" in flags, "x": int(x), "y": int(y), "width": int(width), "height": int(height)})
        return monitors

    def pointer(self) -> tuple[int, int]:
        raw = self._text(["xdotool", "getmouselocation", "--shell"])
        values = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        return int(values["X"]), int(values["Y"])

    def state(self) -> dict:
        active = self.active_window_id()
        return {
            "session_type": os.environ.get("XDG_SESSION_TYPE", ""),
            "display": os.environ.get("DISPLAY", ""),
            "current_workspace": self.current_workspace(),
            "workspace_count": self.desktop_count(),
            "active_window": hex(active) if active else None,
            "pointer": dict(zip(("x", "y"), self.pointer())),
            "monitors": self.monitors(),
            "windows": [window.as_dict() for window in self.windows()],
        }

    def screenshot(self, window: Window | None = None, region: tuple[int, int, int, int] | None = None) -> bytes:
        args = ["import", "-silent", "-window", str(window.window_id) if window else "root"]
        if region:
            x, y, width, height = region
            if width <= 0 or height <= 0:
                raise DesktopError("Region width and height must be positive")
            args += ["-crop", f"{width}x{height}+{x}+{y}"]
        args.append("png:-")
        return self.runner.run(args, timeout=15, binary=True).stdout

    def _focus_unlocked(self, window: Window) -> Window:
        if window.workspace is not None and window.workspace != 0xFFFFFFFF:
            self._text(["xdotool", "set_desktop", str(window.workspace)])
        self._text(["xdotool", "windowactivate", "--sync", str(window.window_id)])
        active = self.active_window_id()
        if active != window.window_id:
            raise DesktopError(f"Focus verification failed: expected {hex(window.window_id)}, got {hex(active) if active else None}")
        return self.window(window.window_id) or window

    def focus(self, window: Window) -> Window:
        with self.operation_lock():
            return self._focus_unlocked(window)

    def switch_workspace(self, workspace: int) -> int:
        count = self.desktop_count()
        if not 0 <= workspace < count:
            raise DesktopError(f"Workspace must be between 0 and {count - 1}")
        with self.operation_lock():
            self._text(["xdotool", "set_desktop", str(workspace)])
        actual = self.current_workspace()
        if actual != workspace:
            raise DesktopError(f"Workspace verification failed: expected {workspace}, got {actual}")
        return actual

    def click(self, x: int, y: int, *, window: Window | None, button: int, count: int) -> dict:
        if button not in range(1, 10) or count not in range(1, 6):
            raise DesktopError("button must be 1-9 and count must be 1-5")
        absolute_x, absolute_y = x, y
        with self.operation_lock():
            if window:
                window = self._focus_unlocked(window)
                if not 0 <= x < window.width or not 0 <= y < window.height:
                    raise DesktopError(f"Window-relative point ({x}, {y}) is outside {window.width}x{window.height}")
                absolute_x, absolute_y = window.x + x, window.y + y
            self._text(["xdotool", "mousemove", "--sync", str(absolute_x), str(absolute_y), "click", "--repeat", str(count), "--delay", "100", str(button)])
        return {"x": absolute_x, "y": absolute_y, "button": button, "count": count, "active_window": hex(self.active_window_id() or 0)}

    def type_text(self, text: str, *, window: Window, delay_ms: int = 5) -> dict:
        if len(text) > 10_000:
            raise DesktopError("Text is limited to 10,000 characters per call")
        if "\n" in text or "\r" in text:
            raise DesktopError("type_text does not accept line breaks; use press_keys with Return explicitly")
        if not 0 <= delay_ms <= 1000:
            raise DesktopError("delay_ms must be between 0 and 1000")
        with self.operation_lock():
            self._focus_unlocked(window)
            self.runner.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", str(delay_ms), "--file", "-"],
                input_bytes=text.encode("utf-8"),
                binary=True,
                timeout=max(5, len(text) * delay_ms / 1000 + 5),
            )
        return {"characters": len(text), "window_id": hex(window.window_id), "pressed_enter": False}

    def press_keys(self, keys: str, *, window: Window) -> dict:
        sequences = keys.split()
        if not sequences or len(sequences) > 20 or not self._SAFE_KEYS.fullmatch(keys):
            raise DesktopError("Use at most 20 xdotool key names/combinations, separated by spaces")
        with self.operation_lock():
            self._focus_unlocked(window)
            self._text(["xdotool", "key", "--clearmodifiers", *sequences])
        return {"keys": sequences, "window_id": hex(window.window_id)}

    def scroll(self, direction: str, steps: int, *, window: Window | None = None) -> dict:
        buttons = {"up": 4, "down": 5, "left": 6, "right": 7}
        if direction not in buttons or not 1 <= steps <= 100:
            raise DesktopError("direction must be up/down/left/right and steps must be 1-100")
        with self.operation_lock():
            if window:
                self._focus_unlocked(window)
            self._text(["xdotool", "click", "--repeat", str(steps), "--delay", "40", str(buttons[direction])])
        return {"direction": direction, "steps": steps}

    def capture_context(self) -> tuple[str, DesktopContext]:
        x, y = self.pointer()
        context = DesktopContext(self.current_workspace(), self.active_window_id(), x, y)
        token = str(uuid.uuid4())
        self.contexts[token] = context
        return token, context

    def restore_context(self, token: str) -> DesktopContext:
        context = self.contexts.get(token)
        if not context:
            raise DesktopError("Unknown or already-restored context token")
        restore_window = bool(context.active_window and self.window(context.active_window))
        with self.operation_lock():
            self._text(["xdotool", "set_desktop", str(context.workspace)])
            if restore_window:
                self._text(["xdotool", "windowactivate", "--sync", str(context.active_window)])
            self._text(["xdotool", "mousemove", "--sync", str(context.pointer_x), str(context.pointer_y)])
            actual_workspace = self.current_workspace()
            actual_window = self.active_window_id()
            actual_pointer = self.pointer()
            if actual_workspace != context.workspace:
                raise DesktopError(f"Context workspace restore failed: expected {context.workspace}, got {actual_workspace}")
            if restore_window and actual_window != context.active_window:
                raise DesktopError("Context window restore failed")
            if actual_pointer != (context.pointer_x, context.pointer_y):
                raise DesktopError(f"Context pointer restore failed: expected {(context.pointer_x, context.pointer_y)}, got {actual_pointer}")
        self.contexts.pop(token, None)
        return context

    def wait_for_window(self, query: str, timeout_seconds: float) -> Window:
        if not 0 < timeout_seconds <= 120:
            raise DesktopError("timeout_seconds must be between 0 and 120")
        deadline = time.monotonic() + timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            try:
                return self.find_window(query=query)
            except DesktopError as exc:
                last_error = str(exc)
            time.sleep(0.1)
        raise DesktopError(f"Timed out waiting for {query!r}: {last_error}")

    def terminal_panes(self, window: Window | None = None) -> dict:
        raw = self._text(["ps", "-eo", "pid=,ppid=,pgid=,tpgid=,tty=,comm=,args="], timeout=10)
        records: dict[int, dict] = {}
        children: dict[int, list[int]] = {}
        for line in raw.splitlines():
            parts = line.strip().split(None, 6)
            if len(parts) < 7:
                continue
            pid, ppid, pgid, tpgid = map(int, parts[:4])
            records[pid] = {
                "pid": pid,
                "ppid": ppid,
                "pgid": pgid,
                "tpgid": tpgid,
                "tty": parts[4],
                "command": parts[5],
                "args": parts[6],
            }
            children.setdefault(ppid, []).append(pid)
        roots = [window.pid] if window and window.pid else [w.pid for w in self.windows() if w.pid]
        owners: dict[int, tuple[int, int]] = {}
        for root in roots:
            stack = [(root, 0)]
            while stack:
                pid, depth = stack.pop()
                owners.setdefault(pid, (root, depth))
                stack.extend((child, depth + 1) for child in children.get(pid, []))
        tty_groups: dict[tuple[int, str], list[tuple[dict, int]]] = {}
        for pid, (owner, depth) in owners.items():
            record = records.get(pid)
            if not record or record["tty"] == "?":
                continue
            tty_groups.setdefault((owner, record["tty"]), []).append((record, depth))
        sessions = []
        for (owner, tty), members in tty_groups.items():
            root_record, _ = min(members, key=lambda item: (item[1], item[0]["pid"]))
            foreground_pid = next((item[0]["tpgid"] for item in members if item[0]["tpgid"] > 0), None)
            foreground = records.get(foreground_pid) if foreground_pid else None
            try:
                cwd = os.readlink(f"/proc/{root_record['pid']}/cwd")
            except OSError:
                cwd = None
            sessions.append(
                {
                    "tty": tty,
                    "shell_pid": root_record["pid"],
                    "shell": root_record["command"],
                    "cwd": cwd,
                    "foreground_pid": foreground_pid,
                    "foreground_command": foreground["command"] if foreground else None,
                    "process_count": len(members),
                    "application_pid": owner,
                }
            )
        regions = []
        if window and window.pid:
            try:
                regions = self._accessibility(
                    ["terminals", "--pid", str(window.pid), "--window-title", window.title]
                )
            except DesktopError:
                regions = []
        shared_application = bool(window and sum(1 for candidate in self.windows() if candidate.pid == window.pid) > 1)
        return {
            "window": window.as_dict() if window else None,
            "visible_panes": regions,
            "terminal_sessions": sorted(sessions, key=lambda item: (item["application_pid"], item["tty"])),
            "session_scope": "application" if shared_application else "window",
            "scope_note": (
                "The terminal application shares one process across windows; visible_panes are exact for the selected window, while TTY sessions cover that application."
                if shared_application else "TTY sessions are descendants of the selected window process."
            ),
        }

    def _accessibility(self, args: list[str], timeout: float = 15) -> object:
        helper = Path(__file__).with_name("atspi_helper.py")
        raw = self._text(["/usr/bin/python3", str(helper), *args], timeout=timeout)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DesktopError(f"Accessibility helper returned invalid data: {raw[:200]}") from exc
        if isinstance(result, dict) and result.get("error"):
            raise DesktopError(str(result["error"]))
        return result

    def accessibility_tree(self, window: Window, max_depth: int = 6, max_nodes: int = 500) -> object:
        if not 1 <= max_depth <= 20 or not 1 <= max_nodes <= 5000:
            raise DesktopError("max_depth must be 1-20 and max_nodes must be 1-5000")
        if not window.pid:
            raise DesktopError("The selected window does not expose a process ID")
        return self._accessibility(
            [
                "snapshot",
                "--pid", str(window.pid),
                "--window-title", window.title,
                "--max-depth", str(max_depth),
                "--max-nodes", str(max_nodes),
            ]
        )

    def activate_accessible(self, window: Window, name: str, role: str = "", action: str = "") -> object:
        if not window.pid:
            raise DesktopError("The selected window does not expose a process ID")
        if not name.strip() or len(name) > 500:
            raise DesktopError("Provide an accessible name between 1 and 500 characters")
        with self.operation_lock():
            self._focus_unlocked(window)
            return self._accessibility(
                [
                    "activate",
                    "--pid", str(window.pid),
                    "--window-title", window.title,
                    "--name", name,
                    "--role", role,
                    "--action", action,
                ]
            )
