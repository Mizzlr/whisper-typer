# Voice journal hotkey-gate recovery

Date: 2026-08-28

## Incident

The continuous ambient journal stopped producing entries after 10:07 IST on
2026-08-26, although `voice-journal.service` remained active. Dictations still
appeared because the independent history tailer continued copying
`whisper-typer-rs` results into the daily journal.

The microphone source was live, unmuted, and producing signal. A live stack
trace showed the ambient transcriber waiting for VAD chunks. The first repair
closed a real stale-state risk in the hotkey suppression latch, but an
end-to-end retest showed that this was not the primary failure.

Non-content capture telemetry then established the root cause: the webcam
microphone produced normal speech around 0.012-0.025 RMS, Silero scored it near
0.001, and the RMS rescue threshold was 0.050. Both detectors therefore
rejected real far-field speech before an utterance could reach Whisper.

## Fix

The keyboard supervisor now reconciles its event-derived state every two
seconds with the kernel's current key bitmap. It also removes state belonging
to disconnected devices. This clears stale hotkey suppression while retaining
support for hotkey combinations spanning multiple keyboards.

The RMS rescue threshold is now aligned with the recorder's established 0.012
RMS speech threshold. A once-per-minute, non-content health line reports VAD
state and counters so future partial failures are visible in service logs.

## Verification

- `cargo test --bin voice-journal`: 2 passed
- `cargo clippy --bin voice-journal -- -D warnings`: passed
- `cargo build --release --bin voice-journal`: passed
- Deployed and release binary SHA-256 matched:
  `7bb82a0c60f960fdab7dc4109f8f146a5671ce63780c4f50c3c1206558429481`
- `voice-journal.service` restarted successfully and loaded Silero VAD.
- The deployed capture stream was live, uncorked, and unmuted, with current-day
  journal and unfiltered files open for append.
- Post-fix telemetry observed repeated recording transitions and a longest
  voiced run of 4.458 seconds.
- An accepted ambient entry was appended at 18:09:07 IST, proving the complete
  microphone-to-VAD-to-Whisper-to-journal path.
