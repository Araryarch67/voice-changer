#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${PLATFORM_NAME:-iphoneos}" == "iphonesimulator" ]]; then
    if [[ "${CURRENT_ARCH:-arm64}" == "x86_64" ]]; then
        rust_target=x86_64-apple-ios
    else
        rust_target=aarch64-apple-ios-sim
    fi
else
    rust_target=aarch64-apple-ios
fi
export PATH="$HOME/.cargo/bin:$PATH"
if [[ "${VC_WITH_ONNX:-0}" == "1" ]]; then
    : "${ORT_LIB_LOCATION:?Set ORT_LIB_LOCATION to an ONNX Runtime 1.22 static build for this iOS target}"
    export ORT_LIB_PROFILE=Release
    rust_features=gui,audio,onnx-static
else
    # UI/audio-only build is useful before packaging model/runtime binaries.
    rust_features=gui,audio
fi
cargo build --manifest-path "$project_dir/Cargo.toml" --lib --release \
    --target "$rust_target" --no-default-features --features "$rust_features"
mkdir -p "$project_dir/target/ios"
cp "$project_dir/target/$rust_target/release/libvoice_changer.a" "$project_dir/target/ios/"
