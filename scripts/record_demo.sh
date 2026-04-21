#!/usr/bin/env bash
# Capture demo.py output as a GIF for the GitHub README.
#
# Three options — use whichever installs cleanly:
#
#   Option A: VHS (recommended — scriptable, reproducible)
#     brew install vhs
#     vhs demo.tape
#
#   Option B: asciinema + svg-term (no agg needed)
#     brew install asciinema
#     npm install -g svg-term-cli
#     bash scripts/record_demo.sh --svg
#
#   Option C: QuickTime + ffmpeg (manual recording)
#     brew install ffmpeg
#     bash scripts/record_demo.sh --ffmpeg path/to/recording.mov
#
# After any option, add to README.md:
#   ![Demo](assets/demo.gif)   or   ![Demo](assets/demo.svg)

set -e
mkdir -p assets

MODE="${1:-auto}"

# ── Option A: VHS ────────────────────────────────────────────────────────────
if command -v vhs &>/dev/null || [ "$MODE" = "--vhs" ]; then
    echo "Using VHS..."
    vhs demo.tape
    echo "✓ GIF saved to assets/demo.gif"
    exit 0
fi

# ── Option B: asciinema + svg-term ───────────────────────────────────────────
if [ "$MODE" = "--svg" ]; then
    if ! command -v asciinema &>/dev/null; then
        echo "Install: brew install asciinema"
        exit 1
    fi
    if ! command -v svg-term &>/dev/null; then
        echo "Install: npm install -g svg-term-cli"
        exit 1
    fi
    echo "Recording with asciinema..."
    asciinema rec assets/demo.cast \
        --command "python3 demo.py --fast" \
        --title "Autonomy Learning Loop" \
        --overwrite
    echo "Converting to SVG..."
    svg-term --in assets/demo.cast --out assets/demo.svg \
        --window --no-cursor --profile Monokai \
        --width 110 --height 40
    echo "✓ SVG saved to assets/demo.svg"
    echo "Add to README: ![Demo](assets/demo.svg)"
    exit 0
fi

# ── Option C: ffmpeg (convert existing .mov or .mp4 recording) ───────────────
if [ "$MODE" = "--ffmpeg" ]; then
    INPUT="$2"
    if [ -z "$INPUT" ]; then
        echo "Usage: bash scripts/record_demo.sh --ffmpeg path/to/recording.mov"
        exit 1
    fi
    if ! command -v ffmpeg &>/dev/null; then
        echo "Install: brew install ffmpeg"
        exit 1
    fi
    echo "Converting $INPUT to GIF..."
    ffmpeg -i "$INPUT" \
        -vf "fps=15,scale=1100:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
        -loop 0 assets/demo.gif
    echo "✓ GIF saved to assets/demo.gif"
    exit 0
fi

# ── No tool found ─────────────────────────────────────────────────────────────
echo "No recording tool found. Install one of:"
echo ""
echo "  VHS (easiest):          brew install vhs  &&  vhs demo.tape"
echo "  asciinema + svg-term:   brew install asciinema && npm i -g svg-term-cli"
echo "  ffmpeg (from QuickTime): brew install ffmpeg"
echo ""
echo "Then re-run:  bash scripts/record_demo.sh"
