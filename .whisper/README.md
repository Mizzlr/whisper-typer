# .whisper/ — Per-Project Speech Recognition Config

Two files customize WhisperTyper for this project:

## vocabulary.txt

One term per line. Fed to Whisper as `initial_prompt` to bias the decoder toward these terms.

**Be conservative.** Only add terms Whisper genuinely misrecognizes. Adding too many
(especially short or common-sounding words) causes false matches — e.g. "rodio" in the
vocabulary made Whisper hear "rodio" when the user said "README".

Good candidates: names Whisper mangles (Ollama, Kokoro, misaki-rs).
Bad candidates: common English words or short technical terms (axum, ort, cpal).

```
Ollama
Kokoro
```

## corrections.yaml

Post-transcription corrections applied by Ollama. Safer than vocabulary.txt because
it only replaces exact text matches after Whisper has already transcribed.

```yaml
Olamar: Ollama
Kakarot: Kokoro
```

## MCP tools

From Claude Code, use these to add entries at runtime (no file editing needed):

- `whisper_teach` — add vocabulary terms (comma-separated)
- `whisper_add_correction` — add a wrong/right correction pair

Changes take effect on the next dictation (hot-reloaded via state file).

## Tips

- Start with corrections.yaml — it's the safe option
- Only promote a term to vocabulary.txt if Ollama correction alone isn't enough
- Test after adding terms — dictate normal sentences to check for false matches
