# Voice journal hotkey-gate recovery

Date: 2026-08-28

## Incident

The continuous ambient journal stopped producing entries after 10:07 IST on
2026-08-26, although `voice-journal.service` remained active. Dictations still
appeared because the independent history tailer continued copying
`whisper-typer-rs` results into the daily journal.

The microphone source was live, unmuted, and producing signal. A live stack
trace showed the ambient transcriber waiting for VAD chunks. The capture path
could leave its hotkey suppression latch active indefinitely when an evdev
key-release transition was missed; because the main process did not exit,
systemd could not detect or recover the partial failure.

## Fix

The keyboard supervisor now reconciles its event-derived state every two
seconds with the kernel's current key bitmap. It also removes state belonging
to disconnected devices. This clears stale hotkey suppression while retaining
support for hotkey combinations spanning multiple keyboards.

## Verification

- `cargo test --bin voice-journal`: 2 passed
- `cargo clippy --bin voice-journal -- -D warnings`: passed
- `cargo build --release --bin voice-journal`: passed
- Deployed and release binary SHA-256 matched:
  `7bb82a0c60f960fdab7dc4109f8f146a5671ce63780c4f50c3c1206558429481`
- `voice-journal.service` restarted successfully and loaded Silero VAD.
- The deployed capture stream was live, uncorked, and unmuted, with current-day
  journal and unfiltered files open for append.
