#!/bin/bash
# One command: Magenta sketch -> ACE-Step cover. Run from moonstar-magenta/ in Git Bash.
#   scripts/chain.sh NAME "style prompt" [seconds=30] [strength=0.3]
# Needs Docker (moonstar-magenta image) and the ACE-Step server (cd D:/dev/ACE-Step-1.5 && uv run acestep-api).
# Writes data/chain/mag_NAME.wav (sketch) and data/chain/ace_NAME.wav (final). 0.3 won the by-ear comparison.
set -euo pipefail
name=$1; prompt=$2; secs=${3:-30}; strength=${4:-0.3}
export MSYS_NO_PATHCONV=1
url=${ACESTEP_URL:-http://localhost:8001}

curl -sf -m 5 "$url/health" >/dev/null || { echo "ACE-Step server not reachable at $url" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker is not running" >&2; exit 1; }
mkdir -p data/chain

echo "[1/2] Magenta sketch ($secs s)"
docker run --rm --gpus all -e PYTHONPATH=/app \
  -v "C:\Users\HomePc\Documents\Magenta:/root/Documents/Magenta" \
  -v "$PWD/moonstar_magenta:/app/moonstar_magenta" -v "$PWD/data:/app/data" moonstar-magenta \
  python -m moonstar_magenta.stream --seconds "$secs" --seed 7 --cfg 6 \
  --prompt "0:$prompt" --out "data/chain/mag_$name.wav" 2>&1 | grep "chunks="

echo "[2/2] ACE-Step cover (strength $strength)"
scripts/ace_cover.sh "data/chain/mag_$name.wav" "data/chain/ace_$name.wav" "$strength" "$prompt, polished studio mix" "$secs"
