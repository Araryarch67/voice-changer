# Verification — 2026-11-12

Host: Linux x86_64; Rust 1.99.1. This records checks actually executed in the workspace, rather than a claim that all targets are validated.

- `cargo check`: default features, no-default-features, and GUI/audio without ONNX passed.
- `cargo build --release` passed; the Linux executable also passed a `--help` smoke check.
- `cargo fmt --check` passed.
- `cargo clippy --all-targets -- -D warnings` passed.
- All **12 tests** passed, including the normally ignored ONNX and null-audio tests, with ONNX Runtime 1.22 and a null-only ALSA configuration explicitly supplied.
- The ONNX tests execute synthetic embedder, pitch, and voice graphs through the actual Rust inference pipeline. They verify model contracts, pitch-control propagation, duration, silence and retrieval-bank error handling. They do not validate perceptual voice quality.
- The CPAL test opened only ALSA null input/output devices, processed audio and shut down its worker. It did not record a physical microphone or play through physical speakers.
- GUI screenshots were rendered using WGPU in private headless Weston compositors at 1100×760 and 390×844. The captures are `preview-desktop.png` and `preview-mobile.png`. These are Linux renders of two layouts, not screenshots from Android/iOS devices.
- macOS/iOS plists parsed successfully; packaging shell scripts passed `bash -n`.

## Update — FAISS index retrieval (2026-11-12)

- `src/retrieval.rs` parses FAISS `IndexIVFFlat` (`IwFl` / `IxF2` / `ilar`, full and sparse list tables) without linking FAISS.
- `cargo test --test retrieval -- --include-ignored`: 4 passed. Fixtures in `tests/fixtures/ivf-*` were written by the repository's FAISS 1.7.3; search ids and distances match FAISS for nprobe 1 and 3 and for empty probed lists (-1). The real `voice-rvc/kobo` index (768-d, 17711 rows, 454 lists) loads, and its reconstruction sum matches FAISS `reconstruct_n`.
- Release `--convert` with the local Kobo ONNX export plus its real `.index` ran at index ratio 0 and 0.75 on a 2.5 s synthetic 48 kHz signal; both produced full-length output, the ratio changed the output, and wall time was unchanged (about 4.2 s on CPU, offline). Voice quality was not evaluated.
- Zeta (`voice-rvc/zeta`, 40 kHz, 109 speakers) was first exported with the former `tools/export_rvc.py --realtime` to `models/zeta/voice-realtime.onnx`, with `models/zeta/model.json` using its own index (768-d, 18033 rows, 462 lists). Kobo and Zeta both converted the same signal at index ratio 0 and 0.75, and the app's model discovery lists both.
- fmt, clippy (`-D warnings`), `--no-default-features` check and the full test suite (including ignored ONNX and null-audio tests) passed after the change.

Windows, macOS, Android and iOS builds have not been executed. Mobile permissions, lifecycle, device routing, static ONNX Runtime integration and signing need validation on the respective toolchains/devices. Real trained RVC models and end-to-end voice quality/latency have not been validated.

See the README for repeatable commands and the explicit model contract.

## Update — Rust checkpoint export (2026-11-12)

- `--export-rvc` (`src/pth.rs`, `src/export.rs`, `templates/`) exported `voice-rvc/kobo` (48 kHz, realtime and plain) and `voice-rvc/zeta` (40 kHz, realtime) in about 0.45 s each.
- Compared with the Python exports in `models/`, with Python used only as an oracle: node lists, attributes, inputs and outputs are identical, and `onnx.checker` passes. 249 of 353 initializers are bit-identical; the 104 weight-norm products differ by at most 2.4e-7.
- `cargo test -- --include-ignored` (with ONNX Runtime and null ALSA): 23 passed. This includes exporting both real checkpoints in Rust and running them through the realtime pipeline (finite, non-silent, full-length output). The 32 kHz templates were generated, but no 32 kHz checkpoint was available to test.
- clippy (`-D warnings`), fmt and `--no-default-features` check passed.
- `tools/export_rvc.py` was removed. `models/kobo/voice-realtime.onnx`, `models/kobo/voice.onnx` and `models/zeta/voice-realtime.onnx` were regenerated with `--export-rvc`. Both manifests then converted the test signal at index ratio 0 and 0.75.
