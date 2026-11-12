#!/usr/bin/env bash
# Render only this application in a private headless compositor.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
preview_dir="$(mktemp -d /tmp/voice-changer-preview.XXXXXX)"
chmod 700 "$preview_dir"
weston_pid=""
cleanup() {
    if [[ -n "$weston_pid" ]]; then kill "$weston_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
XDG_RUNTIME_DIR="$preview_dir" weston --backend=headless --renderer=pixman --fake-seat --no-config \
    --width="${2:-1100}" --height="${3:-760}" --socket=vc-preview --shell=kiosk-shell.so --log="$preview_dir/weston.log" &
weston_pid=$!
for _ in {1..100}; do
    [[ -S "$preview_dir/vc-preview" ]] && break
    sleep 0.05
done
env -u DISPLAY XDG_RUNTIME_DIR="$preview_dir" WAYLAND_DISPLAY=vc-preview \
    XDG_DATA_HOME="$preview_dir/data" WGPU_BACKEND=vulkan \
    timeout 40 "$project_dir/target/debug/examples/capture" "${1:-$project_dir/docs/preview.png}" "${2:-1100}" "${3:-760}" "${@:4}"
