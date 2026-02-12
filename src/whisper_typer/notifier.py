"""Desktop notifications using libnotify."""

import logging

from .config import FeedbackConfig

logger = logging.getLogger(__name__)

# Try to import Notify, but don't fail if not available
try:
    import gi

    gi.require_version("Notify", "0.7")
    from gi.repository import Notify

    HAS_NOTIFY = True
except (ImportError, ValueError):
    HAS_NOTIFY = False
    logger.warning("PyGObject/libnotify not available, notifications disabled")


class Notifier:
    """Show desktop notifications for service state."""

    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.initialized = False
        self._init_notify()

    def _init_notify(self):
        """Initialize the notification system."""
        if not self.config.notifications:
            return

        if not HAS_NOTIFY:
            self.config.notifications = False
            return

        try:
            Notify.init("Whisper Input")
            self.initialized = True
            logger.debug("Notification system initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize notifications: {e}")
            self.config.notifications = False

    def show(self, title: str, body: str = "", icon: str = "audio-input-microphone"):
        """Show a notification."""
        if not self.config.notifications or not self.initialized:
            # Log instead
            logger.info(f"[Notification] {title}: {body}")
            return

        try:
            notification = Notify.Notification.new(title, body, icon)
            notification.set_urgency(Notify.Urgency.LOW)
            notification.show()
        except Exception as e:
            logger.warning(f"Failed to show notification: {e}")

    def recording_started(self):
        """Notification when recording starts."""
        self.show("Recording", "Speak now... Release keys when done.", "media-record")

    def recording_stopped(self):
        """Notification when recording stops."""
        self.show("Processing", "Transcribing your speech...", "system-run")

    def transcription_complete(self, text: str):
        """Notification when transcription is done and typed."""
        # Show a preview of what was typed
        preview = text[:80] + "..." if len(text) > 80 else text
        self.show("Typed", preview, "dialog-ok")

    def error(self, message: str):
        """Notification for errors."""
        self.show("Error", message, "dialog-error")

    def service_ready(self):
        """Notification when service is ready."""
        self.show(
            "Whisper Input Ready", "Hold Win+Alt to dictate", "audio-input-microphone"
        )

    def close(self):
        """Clean up notification system."""
        if self.initialized and HAS_NOTIFY:
            try:
                Notify.uninit()
            except Exception:
                pass
