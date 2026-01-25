"""Text typing using ydotool/dotool/xdotool."""

import logging
import os
import shutil
import subprocess
from pathlib import Path

from .config import TyperConfig

logger = logging.getLogger(__name__)


def detect_display() -> tuple[str | None, str | None]:
    """
    Detect the active X11 display and Xauthority file.

    Returns:
        Tuple of (DISPLAY, XAUTHORITY) or (None, None) if not found.
    """
    # Check if current DISPLAY already works
    current_display = os.environ.get("DISPLAY")
    current_xauth = os.environ.get("XAUTHORITY")

    if current_display and _test_display(current_display, current_xauth):
        logger.debug(f"Current DISPLAY={current_display} is working")
        return current_display, current_xauth

    # Try to detect from running X sessions
    uid = os.getuid()

    # Common Xauthority locations
    xauth_paths = [
        f"/run/user/{uid}/gdm/Xauthority",
        f"/run/user/{uid}/.mutter-Xwaylandauth.*",
        os.path.expanduser("~/.Xauthority"),
    ]

    # Find valid Xauthority
    xauthority = None
    for path_pattern in xauth_paths:
        if "*" in path_pattern:
            # Glob pattern
            matches = list(Path(path_pattern).parent.glob(Path(path_pattern).name))
            if matches:
                xauthority = str(matches[0])
                break
        elif os.path.exists(path_pattern):
            xauthority = path_pattern
            break

    # Try displays :0 through :3
    for display_num in range(4):
        display = f":{display_num}"
        if _test_display(display, xauthority):
            logger.info(f"Detected working display: DISPLAY={display}, XAUTHORITY={xauthority}")
            return display, xauthority

    # Try to get display from 'w' command (shows logged-in users and their displays)
    try:
        result = subprocess.run(
            ["w", "-hs"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                possible_display = parts[2]
                if possible_display.startswith(":"):
                    if _test_display(possible_display, xauthority):
                        logger.info(f"Detected display from 'w': {possible_display}")
                        return possible_display, xauthority
    except Exception as e:
        logger.debug(f"Failed to detect display from 'w' command: {e}")

    logger.warning("Could not detect a working X11 display")
    return None, None


def _test_display(display: str, xauthority: str | None) -> bool:
    """Test if a display is accessible."""
    env = os.environ.copy()
    env["DISPLAY"] = display
    if xauthority:
        env["XAUTHORITY"] = xauthority

    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True,
            env=env,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


class TextTyper:
    """Type text into the focused window using system tools."""

    def __init__(self, config: TyperConfig):
        self.config = config
        self.backend = self._detect_backend()
        self._display_env: dict[str, str] | None = None
        logger.info(f"Using typing backend: {self.backend}")

        # Pre-detect display for X11-based backends
        if self.backend == "xdotool":
            self._setup_display_env()

    def _setup_display_env(self):
        """Setup display environment for X11 tools."""
        display, xauthority = detect_display()
        if display:
            self._display_env = os.environ.copy()
            self._display_env["DISPLAY"] = display
            if xauthority:
                self._display_env["XAUTHORITY"] = xauthority
            logger.info(f"X11 environment: DISPLAY={display}, XAUTHORITY={xauthority}")
        else:
            logger.warning("No working X11 display found, xdotool may fail")

    def _detect_backend(self) -> str:
        """Detect available typing backend."""
        if self.config.backend != "auto":
            # Verify requested backend exists
            if shutil.which(self.config.backend):
                return self.config.backend
            else:
                logger.warning(
                    f"Requested backend '{self.config.backend}' not found, auto-detecting"
                )

        # Auto-detect: prefer ydotool > dotool > xdotool
        for tool in ["ydotool", "dotool", "xdotool"]:
            if shutil.which(tool):
                return tool

        raise RuntimeError(
            "No typing backend found. Install one of: ydotool, dotool, xdotool\n"
            "For ydotool: sudo apt install ydotool && systemctl --user enable --now ydotoold"
        )

    def type_text(self, text: str):
        """Type text into the currently focused window."""
        if not text:
            logger.warning("Empty text, nothing to type")
            return

        logger.debug(f"Typing {len(text)} characters")

        try:
            if self.backend == "ydotool":
                self._type_ydotool(text)
            elif self.backend == "dotool":
                self._type_dotool(text)
            elif self.backend == "xdotool":
                self._type_xdotool(text)
            else:
                raise RuntimeError(f"Unknown backend: {self.backend}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Typing failed: {e}")
            raise RuntimeError(f"Failed to type text: {e}")

    def _type_ydotool(self, text: str):
        """Type using ydotool."""
        # Try with --clearmodifiers first (newer versions), fall back to simple syntax
        try:
            result = subprocess.run(
                ["ydotool", "type", "--clearmodifiers", "--", text],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(f"ydotool (new) stdout: {result.stdout}, stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.debug(f"ydotool new syntax failed: {e.stderr}, trying old syntax")
            # Older ydotool (0.1.x) has simpler syntax
            result = subprocess.run(
                ["ydotool", "type", text],
                capture_output=True,
                text=True,
            )
            logger.debug(f"ydotool (old) returncode: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
            if result.returncode != 0:
                raise RuntimeError(f"ydotool failed: {result.stderr}")

    def _type_dotool(self, text: str):
        """Type using dotool (reads from stdin)."""
        # dotool uses a simple command format
        subprocess.run(
            ["dotool"],
            input=f"type {text}",
            text=True,
            check=True,
            capture_output=True,
        )

    def _type_xdotool(self, text: str):
        """Type using xdotool (X11 only) via clipboard for instant paste."""
        import time

        # Re-detect display if not set or if it stops working
        if not self._display_env:
            self._setup_display_env()
            if not self._display_env:
                raise RuntimeError("No working X11 display available")

        # Copy text to clipboard
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text,
            text=True,
            check=True,
            env=self._display_env,
        )

        # Small delay to ensure clipboard is ready
        time.sleep(0.01)

        # Paste using Ctrl+Shift+V (works in terminals)
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
            check=True,
            capture_output=True,
            env=self._display_env,
        )

    def check_backend_ready(self) -> bool:
        """Check if the typing backend is ready to use."""
        if self.backend == "ydotool":
            # Check if ydotoold daemon is running (needed for newer versions)
            # Older versions (0.1.x) don't need a daemon
            result = subprocess.run(
                ["systemctl", "--user", "is-active", "ydotoold"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # Check if we have the older version that doesn't need daemon
                help_result = subprocess.run(
                    ["ydotool", "help"],
                    capture_output=True,
                    text=True,
                )
                # Older version outputs to stderr and includes "recorder" command
                if "recorder" in help_result.stderr:
                    # Older version (0.1.x) - no daemon needed
                    logger.debug("Using ydotool 0.1.x (no daemon required)")
                    return True
                else:
                    logger.warning(
                        "ydotoold service not running. Start it with: "
                        "systemctl --user start ydotoold"
                    )
                    return False
        return True
