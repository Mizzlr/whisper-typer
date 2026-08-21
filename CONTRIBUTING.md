# Contributing

Thank you for improving WhisperTyper. Keep changes small, observable, and easy
to roll back: this project sits directly in the user's text-input path.

Before opening a pull request, run:

```bash
cargo test --all-targets --no-default-features
cargo clippy --all-targets --no-default-features -- -D warnings
```

Format Rust files you change. A repository-wide formatting pass should be a
separate pull request so it does not obscure behavior changes in sensitive
input paths.

Do not commit model files, recordings, transcripts, API keys, local correction
tables, or generated `target/` content. New behavior should include a focused
test where practical. Changes to hotkey handling or text injection require an
explicit manual test plan for both press/release behavior and the target
desktop environment.

Use `config.example.yaml` when documenting options. Keep personal values in
`~/.config/whisper-typer/config.yaml` or an explicitly supplied config file.

Commit `Cargo.lock` for reproducible application builds. Never add local model
weights or generated benchmark recordings to Git; document the model name,
quantization, hardware, request shape, and sample count instead.
