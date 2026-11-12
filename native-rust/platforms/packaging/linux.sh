#!/usr/bin/env bash
set -euo pipefail

version="$1"
payload="$2"
output="$3"
scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/opt/voice-changer" "$stage/usr/bin" "$stage/usr/share/applications"
cp -a "$payload/." "$stage/opt/voice-changer/"
install -m 755 "$scripts_dir/voice-changer" "$stage/usr/bin/voice-changer"
install -m 644 "$scripts_dir/voice-changer.desktop" "$stage/usr/share/applications/voice-changer.desktop"

common=(
  -s dir -n voice-changer -v "$version" --iteration 1
  --license MIT --maintainer 'Araryarch67 <Araryarch67@users.noreply.github.com>'
  --url https://github.com/Araryarch67/voice-changer
  --description 'Real-time RVC voice changer with a native desktop interface'
  --architecture x86_64 --chdir "$stage"
)
fpm "${common[@]}" -t deb --package "$output/voice-changer-linux-x64.deb" \
  --depends 'libc6 (>= 2.35)' --depends libgcc-s1 --depends libstdc++6 \
  --depends libasound2 --depends libudev1 --depends libxkbcommon0 \
  --depends libwayland-client0 --depends libx11-6 --depends libvulkan1 opt usr
fpm "${common[@]}" -t rpm --package "$output/voice-changer-linux-x64.rpm" \
  --depends 'glibc >= 2.35' --depends libgcc --depends libstdc++ \
  --depends alsa-lib --depends 'libudev.so.1()(64bit)' --depends libxkbcommon \
  --depends wayland --depends libX11 --depends vulkan-loader opt usr

# Inspect both packages before uploading them.
dpkg-deb --info "$output/voice-changer-linux-x64.deb"
dpkg-deb --contents "$output/voice-changer-linux-x64.deb"
rpm -qip "$output/voice-changer-linux-x64.rpm"
rpm -qlp "$output/voice-changer-linux-x64.rpm"
