# Security and privacy

Please report vulnerabilities privately through GitHub's security advisory
feature rather than a public issue.

WhisperTyper reads microphone audio, global keyboard events, focused-window
text destinations, and locally stored transcription history. Its default
speech recognition, grammar correction, and speech synthesis paths are local.
Do not attach journals, history JSONL, recordings, correction files, or service
logs to public issues without reviewing and redacting them first.

The MCP and TTS HTTP listeners are intended for loopback use. Do not expose
ports 8766 or 8767 to an untrusted network. Runtime state is written under
`~/.cache/whisper-typer/`; history and voice journals contain dictated text and
should be protected like other personal documents.

Before sharing diagnostics, inspect and redact:

- `~/.whisper-typer-history/`
- `~/voice-journal/`
- `~/.tts-hook-history/`
- `~/.config/whisper-typer/corrections.tsv`
- Voice Journal `.mic.wav` and `.vad.csv` debug sidecars

The repository intentionally ignores common model formats and build output.
Contributors are still responsible for reviewing `git status` before every
push; ignore rules are not a substitute for checking staged content.
