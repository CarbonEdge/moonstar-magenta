#!/bin/bash
# Cover/remix an audio file with a local ACE-Step 1.5 server (uv run acestep-api, default localhost:8001).
#   scripts/ace_cover.sh SRC.wav OUT.wav STRENGTH "caption" [duration_sec=30]
# STRENGTH 0..1: higher keeps more of the source. Uploads as multipart: JSON paths outside the temp dir are rejected.
set -euo pipefail
src=$1; out=$2; strength=$3; caption=$4; dur=${5:-30}
url=${ACESTEP_URL:-http://localhost:8001}
field() { python -c "import sys,json;print($1)"; }

tid=$(curl -sf -m 60 -X POST "$url/release_task" \
  -F "src_audio=@$src" -F "task_type=cover" -F "audio_cover_strength=$strength" \
  -F "prompt=$caption" -F "lyrics=[Instrumental]" -F "audio_duration=$dur" \
  -F "audio_format=wav" -F "batch_size=1" -F "thinking=false" | field "json.load(sys.stdin)['data']['task_id']")
echo "task=$tid"
for _ in $(seq 1 180); do
  r=$(curl -sf -m 30 -X POST "$url/query_result" -H 'Content-Type: application/json' -d "{\"task_id_list\":[\"$tid\"]}")
  st=$(echo "$r" | field "json.load(sys.stdin)['data'][0]['status']")
  if [ "$st" = 1 ]; then
    f=$(echo "$r" | field "json.loads(json.load(sys.stdin)['data'][0]['result'])[0]['file']")
    mkdir -p "$(dirname "$out")"; curl -sf "$url$f" -o "$out"; echo "saved $out"; exit 0
  elif [ "$st" = 2 ]; then echo "ACE task failed: ${r:0:500}" >&2; exit 1; fi
  sleep 5
done
echo "timeout waiting for $tid" >&2; exit 1
