# Voice Changer (Rust)

A native Rust real-time RVC voice changer: an egui desktop app with CPAL audio and ONNX Runtime inference, with no Python process or server.

It is a port of [w-okada/voice-changer](https://github.com/w-okada/voice-changer) (MIT). The application, build instructions and model format are in [`native-rust/`](native-rust/README.md).

Release archives for Linux, Windows and macOS are built by `.github/workflows/release.yml` when a `v*` tag is pushed.
