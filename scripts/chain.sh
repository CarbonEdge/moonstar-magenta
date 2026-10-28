#!/bin/bash
# One command: Magenta sketch -> ACE-Step cover. Run from moonstar-magenta/ in Git Bash.
#   scripts/chain.sh NAME "caption words" [seconds=180]
# Env (defaults = best recipe found by ear): STRENGTH=0.6  RESET=60  MAG_TEMP=1.0  CFG=6  SEED=7
#   TIMELINE=data/samples/set1.wav   follow a long reference's tonal arc (see fetch_audio.py --spread)
#   PROMPT="text or file: blend"     used instead of TIMELINE; defaults to the caption
# Needs Docker (moonstar-magenta image) and the ACE-Step server (cd D:/dev/ACE-Step-1.5 && uv run acestep-api).
# Writes data/chain/mag_NAME.wav (sketch) and data/chain/ace_NAME.wav (final).
set -euo pipefail
name=$1; caption=$2; secs=${3:-180}
strength=${STRENGTH:-0.6}; reset=${RESET:-60}; temp=${MAG_TEMP:-1.0}; cfg=${CFG:-6}; seed=${SEED:-7}
export MSYS_NO_PATHCONV=1
url=${ACESTEP_URL:-http://localhost:8001}

curl -sf -m 5 "$url/health" >/dev/null || { echo "ACE-Step server not reachable at $url" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker is not running" >&2; exit 1; }
mkdir -p data/chain

if [ -n "${TIMELINE:-}" ]; then src=(--timeline "$TIMELINE"); else src=(--prompt "0:${PROMPT:-$caption}"); fi

echo "[1/2] Magenta sketch ($secs s, reset every ${reset}s, temp $temp)"
docker run --rm --gpus all -e PYTHONPATH=/app \
  -v "C:\Users\HomePc\Documents\Magenta:/root/Documents/Magenta" \
  -v "$PWD/moonstar_magenta:/app/moonstar_magenta" -v "$PWD/data:/app/data" moonstar-magenta \
  python -m moonstar_magenta.stream --seconds "$secs" --seed "$seed" --cfg "$cfg" --temperature "$temp" \
  --reset-every "$reset" "${src[@]}" --out "data/chain/mag_$name.wav" 2>&1 | grep "chunks=\|Error\|Traceback"

echo "[2/2] ACE-Step cover (strength $strength)"
scripts/ace_cover.sh "data/chain/mag_$name.wav" "data/chain/ace_$name.wav" "$strength" "$caption, polished studio mix" "$secs"
