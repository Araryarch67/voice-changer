# Voice Changer (Rust)

This repository is a Rust port of [w-okada/voice-changer](https://github.com/w-okada/voice-changer), a real-time RVC voice changer. The application lives in `native-rust/`; start with `native-rust/README.md`.

## Layout

| Path | What it is |
|---|---|
| `native-rust/` | The Rust application (egui GUI, CPAL audio, ONNX Runtime inference). All new work goes here. |
| `reference/` | The original Python server and web client. Git-ignored, read-only, may be absent. |
| `voice-rvc/` | Local RVC checkpoints (`.pth`) and FAISS indexes. Git-ignored. |
| `.github/workflows/release-linux.yml`, `release-windows.yml` | Build and publish the release downloads, one workflow per OS, so Linux is published as soon as it is ready and Windows follows. |

## Using `reference/`

`reference/` is the behaviour the port is measured against. Before porting or changing a feature, read how the original does it and match its behaviour unless there is a stated reason not to. Never edit it, and never add it to git.

If `reference/` is missing (a fresh clone), get it with `git clone https://github.com/w-okada/voice-changer reference`.

Where to look:

| Topic | Reference path |
|---|---|
| RVC inference pipeline (features, pitch, index retrieval, protect) | `reference/server/voice_changer/RVC/pipeline/Pipeline.py` |
| Pitch detectors (dio, harvest, crepe, rmvpe, fcpe) | `reference/server/voice_changer/RVC/pitchExtractor/` |
| Embedders (HuBERT, ContentVec) | `reference/server/voice_changer/RVC/embedder/` |
| ONNX export of RVC checkpoints | `reference/server/voice_changer/RVC/onnxExporter/` |
| Block buffering, crossfade and SOLA | `reference/server/voice_changer/VoiceChangerV2.py` |
| Settings and their defaults | `reference/server/voice_changer/RVC/RVCSettings.py` |
| Model slots | `reference/server/voice_changer/ModelSlotManager.py` |
| Recording | `reference/server/voice_changer/IORecorder.py` |
| UI options and their names (GAIN, TUNE, INDEX, CHUNK, EXTRA, NOISE, F0 Det., GPU) | `reference/client/demo/src/components/demo/components2/` |
| Other engines (not ported) | `reference/server/voice_changer/{MMVCv13,MMVCv15,SoVitsSvc40,DDSP_SVC,DiffusionSVC,Beatrice,LLVC}` |

Names differ between the two: the reference's CHUNK is `block_ms`, EXTRA is `context_ms`, TUNE is `semitones`, INDEX is `index_ratio`, and S.Thresh is `silence_threshold` in `native-rust/src/config.rs`.

## Working in `native-rust/`

Run these from `native-rust/` before calling a change done:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

- `models/`, `runtime/` and `target/` are git-ignored. Voices are `models/<name>/model.json` manifests; `runtime/` holds the ONNX Runtime shared library.
- The UI is Material 3. Colours, type, icons and components are in `src/m3.rs`; screens are in `src/app.rs`. Every option shows a plain-language hint, so add one with any new control.
- GPU uses the CUDA execution provider and needs a GPU build of ONNX Runtime with matching CUDA and cuDNN libraries. CPU must keep working without them.
- `tools/preview.sh` renders the GUI headlessly to a PNG; use it to check UI changes.
