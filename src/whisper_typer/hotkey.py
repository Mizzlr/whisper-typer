"""Global hotkey detection using evdev."""

import asyncio
import logging
from typing import Callable, Optional

from evdev import InputDevice, categorize, ecodes, list_devices

from .config import HotkeyConfig

logger = logging.getLogger(__name__)


class HotkeyMonitor:
    """Monitor for global hotkey combinations using evdev."""

    def __init__(
        self,
        config: HotkeyConfig,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ):
        self.config = config
        self.on_press = on_press
        self.on_release = on_release
        # Build list of all hotkey combos (primary + alternates)
        self.all_combos: list[set[int]] = []
        self.all_combos.append(self._resolve_key_codes(config.combo))
        for alt_combo in config.alt_combos:
            self.all_combos.append(self._resolve_key_codes(alt_combo))
        self.pressed_keys: set[int] = set()
        self.hotkey_active = False
        self.devices: list[InputDevice] = []
        logger.info(f"Hotkey combos: {len(self.all_combos)} configured")

    def _resolve_key_codes(self, key_names: list[str]) -> set[int]:
        """Convert key names to evdev key codes."""
        codes = set()
        for name in key_names:
            if hasattr(ecodes, name):
                codes.add(getattr(ecodes, name))
            else:
                logger.warning(f"Unknown key name: {name}")
        return codes

    def _any_combo_active(self) -> bool:
        """Check if any hotkey combo is currently pressed."""
        for combo in self.all_combos:
            if combo.issubset(self.pressed_keys):
                return True
        return False

    def find_keyboards(self) -> list[InputDevice]:
        """Find all keyboard input devices."""
        keyboards = []
        for path in list_devices():
            try:
                device = InputDevice(path)
                capabilities = device.capabilities()
                # Check if device has key events
                if ecodes.EV_KEY in capabilities:
                    keys = capabilities[ecodes.EV_KEY]
                    # Check if it has typical keyboard keys
                    if ecodes.KEY_A in keys and ecodes.KEY_ENTER in keys:
                        keyboards.append(device)
                        logger.info(f"Found keyboard: {device.name} at {device.path}")
            except (PermissionError, OSError) as e:
                logger.debug(f"Cannot access {path}: {e}")
        return keyboards

    async def _monitor_device(self, device: InputDevice):
        """Monitor a single input device for key events."""
        logger.debug(f"Monitoring {device.name}")
        try:
            async for event in device.async_read_loop():
                if event.type == ecodes.EV_KEY:
                    await self._handle_key_event(event)
        except OSError as e:
            logger.warning(f"Device {device.name} disconnected: {e}")

    async def _handle_key_event(self, event):
        """Handle a key press/release event."""
        key_event = categorize(event)
        key_code = event.code

        if key_event.keystate == key_event.key_down:
            self.pressed_keys.add(key_code)
        elif key_event.keystate == key_event.key_up:
            self.pressed_keys.discard(key_code)

        # Check if any hotkey combo is active
        hotkey_now_active = self._any_combo_active()

        if hotkey_now_active and not self.hotkey_active:
            # Hotkey just activated
            self.hotkey_active = True
            logger.debug("Hotkey pressed")
            try:
                self.on_press()
            except Exception as e:
                logger.exception(f"Error in on_press callback: {e}")

        elif not hotkey_now_active and self.hotkey_active:
            # Hotkey just released
            self.hotkey_active = False
            logger.debug("Hotkey released")
            try:
                self.on_release()
            except Exception as e:
                logger.exception(f"Error in on_release callback: {e}")

    async def run(self):
        """Start monitoring all keyboards for hotkey."""
        self.devices = self.find_keyboards()
        if not self.devices:
            raise RuntimeError(
                "No keyboards found. Make sure you're in the 'input' group: "
                "sudo usermod -aG input $USER"
            )

        # Monitor all keyboards concurrently
        tasks = [self._monitor_device(device) for device in self.devices]
        await asyncio.gather(*tasks)

    def stop(self):
        """Stop monitoring and close devices."""
        for device in self.devices:
            try:
                device.close()
            except Exception:
                pass
        self.devices = []
