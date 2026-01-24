"""Text typing using ydotool/dotool/xdotool."""

import logging
import shutil
import subprocess

from .config import TyperConfig

logger = logging.getLogger(__name__)


class TextTyper:
    """Type text into the focused window using system tools."""

    def __init__(self, config: TyperConfig):
        self.config = config
        self.backend = self._detect_backend()
        logger.info(f"Using typing backend: {self.backend}")

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

        logger.info(f"Typing {len(text)} characters")

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

        # Copy text to clipboard
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text,
            text=True,
            check=True,
        )

        # Small delay to ensure clipboard is ready
        time.sleep(0.01)

        # Paste using Ctrl+Shift+V (works in terminals)
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", "ctrl+shift+v"],
            check=True,
            capture_output=True,
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
