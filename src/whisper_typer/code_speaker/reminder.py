"""Escalating reminder system for code_speaker TTS notifications."""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ReminderManager:
    """Manages repeating TTS reminders until user responds.

    After the initial TTS notification, repeats the same message
    every `interval` seconds until cancelled (by UserPromptSubmit).
    """

    def __init__(self, interval: int = 300):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._speak_fn = None
        self._reminder_count = 0

    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def reminder_count(self) -> int:
        return self._reminder_count

    def start(self, text: str, speak_fn):
        """Start repeating reminders for given text.

        Args:
            text: The text to speak on each reminder.
            speak_fn: Async callable that speaks the text.
        """
        self.cancel()
        self._speak_fn = speak_fn
        self._reminder_count = 0
        self._task = asyncio.ensure_future(self._reminder_loop(text))
        logger.info(f"Reminder started: every {self.interval}s")

    async def _reminder_loop(self, text: str):
        """Loop: sleep → speak → repeat."""
        try:
            while True:
                await asyncio.sleep(self.interval)
                self._reminder_count += 1
                logger.info(f"Reminder #{self._reminder_count}: speaking")
                if self._speak_fn:
                    await self._speak_fn(text)
        except asyncio.CancelledError:
            logger.info(f"Reminder cancelled after {self._reminder_count} reminders")

    def cancel(self):
        """Cancel all pending reminders."""
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        count = self._reminder_count
        self._reminder_count = 0
        return count
