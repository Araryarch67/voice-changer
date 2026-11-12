#!/usr/bin/env bash
# Renders the app icon from assets/icon.svg (and icon-small.svg for 16-32 px) into
# every format the app and its packages use. Needs rsvg-convert and Python Pillow.
# Run it after changing either SVG and commit the results.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../assets"
for size in 16 24 32 48 64 128 256 512; do
    source=icon.svg
    [ "$size" -le 32 ] && source=icon-small.svg
    mkdir -p "hicolor/${size}x${size}/apps"
    rsvg-convert -w "$size" -h "$size" "$source" -o "hicolor/${size}x${size}/apps/voice-changer.png"
done
mkdir -p hicolor/scalable/apps
cp icon.svg hicolor/scalable/apps/voice-changer.svg
python3 -I - <<'EOF'
from PIL import Image
png = lambda size: Image.open(f"hicolor/{size}x{size}/apps/voice-changer.png").convert("RGBA")
sizes = [16, 24, 32, 48, 64, 128, 256]
png(256).save("icon.ico", sizes=[(s, s) for s in sizes], append_images=[png(s) for s in sizes])
png(512).save("icon.icns")
EOF
echo "Icons written to $(pwd)"
