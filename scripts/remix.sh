#!/bin/bash
# Stem remix: keep the original vocals + drums, replace bass/other with a new ACE-Step bed.
#   scripts/remix.sh URL_OR_WAV NAME ["bed caption"]      (run from moonstar-magenta/ in Git Bash)
# Env: STRENGTH=0.5 (bed cover strength; 0.5 beat 0.7 by ear)  INTRO=5 (seconds of original bed kept,
#      then 2s crossfade; hides ACE not knowing a vocal is about to start)  BPM=auto-detected
# Needs: ACE-Step server (cd D:/dev/ACE-Step-1.5 && uv run acestep-api), demucs, librosa, ffmpeg.
# Writes data/remix/NAME_remix.wav. Best on a continuous vocal song, not a stitched DJ-set sample.
set -euo pipefail
in=$1; name=$2
caption=${3:-new lush synth pads and warm bass, instrumental only, no drums, no vocals, no percussion, same chords}
strength=${STRENGTH:-0.5}; intro=${INTRO:-5}
url=${ACESTEP_URL:-http://localhost:8001}
curl -sf -m 5 "$url/health" >/dev/null || { echo "ACE-Step server not reachable at $url" >&2; exit 1; }
mkdir -p data/samples data/remix

if [[ "$in" == http* ]]; then
  python scripts/fetch_audio.py "$in" "$name" --start 0 --seconds 600 >/dev/null
  src=data/samples/$name.wav
else
  src=$in
fi
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src")
dur=${dur%.*}
bpm=${BPM:-$(python -c "import librosa,numpy as np;y,sr=librosa.load('$src',sr=22050);print(round(float(np.atleast_1d(librosa.beat.beat_track(y=y,sr=sr)[0])[0])))")}
echo "[1/4] ${dur}s, ${bpm} BPM"

echo "[2/4] Demucs stems"
python -m demucs -n htdemucs -d cuda -o data/stems "$src" >/dev/null 2>&1
stem=data/stems/htdemucs/$(basename "${src%.*}")
r=data/remix
ffmpeg -y -loglevel error -i "$stem/other.wav" -i "$stem/bass.wav" -filter_complex "amix=inputs=2:normalize=0" "$r/${name}_bed_src.wav"

echo "[3/4] ACE bed (strength $strength)"
BPM=$bpm scripts/ace_cover.sh "$r/${name}_bed_src.wav" "$r/${name}_bed.wav" "$strength" "$caption" "$((dur + 1))"

echo "[4/4] mix"
ffmpeg -y -loglevel error -i "$r/${name}_bed_src.wav" -i "$r/${name}_bed.wav" -filter_complex \
  "[0]atrim=0:$((intro + 2)),asetpts=PTS-STARTPTS[a];[1]atrim=$intro,asetpts=PTS-STARTPTS[b];[a][b]acrossfade=d=2:c1=tri:c2=tri[x];[x]apad=whole_dur=$dur[bed]" \
  -map "[bed]" "$r/${name}_bed_fixed.wav"
ffmpeg -y -loglevel error -i "$stem/vocals.wav" -i "$stem/drums.wav" -i "$r/${name}_bed_fixed.wav" -filter_complex \
  "[0][1][2]amix=inputs=3:normalize=0:duration=first,alimiter=limit=0.95" "$r/${name}_remix.wav"
echo "saved $r/${name}_remix.wav"
