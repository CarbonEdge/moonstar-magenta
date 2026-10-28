# moonstar-magenta

Local [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) experiments. Sibling of `moonstar-music` (batch, ACE-Step); this repo explores *live / steerable* generation.

## Status (2026-10-28): working pipeline, being tuned by ear

- Chunked streaming on GPU: 1 s chunks, state carried between chunks, ~0.22 s per chunk, 0 underruns.
- Live browser playback with mid-stream prompt changes (`moonstar_magenta.live`, Web Audio player).
- Long-reference style following (`--timeline`) and drift control (`--reset-every`).
- Magenta -> ACE-Step cover chain (`scripts/chain.sh`), optional lyrics.
- Current state, open questions and next steps: `docs/2026-10-28-checkpoint.md`.

## Architecture

```
YouTube URL --scripts/fetch_audio.py--> data/samples/*.wav   (--spread N: N windows across a long set, order kept)
                                              |
        text prompt / file: blend / --timeline (per-10s style embeddings, smoothed, followed over time)
                                              v
  moonstar_magenta.stream  (Docker, JAX GPU, MagentaRT2, 1 s chunks, --reset-every N + crossfade)
        |                         \
        | WAV sketch               `-- moonstar_magenta.live: HTTP + Web Audio player, prompt box
        v
  scripts/ace_cover.sh  --multipart--> ACE-Step 1.5 API (host, localhost:8001, task_type=cover)
        |                                  strength 0.6, optional LYRICS_FILE
        v
  data/chain/ace_NAME.wav (final)         scripts/chain.sh runs the whole path in one command
```

Two GPU processes (Magenta container, ACE-Step server) share one 16 GB card; run them one after the other. Everything under `data/` (samples, lyrics, outputs) is gitignored.

## Why Docker

- Upstream real-time streaming is **Apple Silicon only**; on NVIDIA only offline JAX generation is supported (we chunk it ourselves).
- JAX CUDA wheels are **Linux-only**. Native Windows JAX = CPU only (0.26x real time).
- So the GPU path runs in Docker Desktop (`--gpus all`, WSL2 backend). Docker already works on this machine.

## Results (RTX 4080 Super; real time = 25 steps/s)

| Run | Speed | Real time? |
|---|---|---|
| `mrt2_small`, CPU (native Windows) | 6.4 steps/s | No (0.26x) |
| `mrt2_small`, GPU (Docker), 10 s clip | 76.7 steps/s (13 ms/step) | Yes (~3x) |
| `mrt2_small`, GPU, chunked streaming (20 s) | ~0.22 s per 1 s chunk | Yes (~4.6x), 0 underruns |
| `mrt2_base`, GPU (Docker), 10 s clip | 25.0 steps/s (40 ms/step) | Borderline (~1.0x), single cold run |

## Quick start

All commands from `moonstar-magenta/`, in Git Bash.

### 1. One-time: model weights (shared by native and Docker)

Weights live in `C:\Users\<you>\Documents\Magenta\magenta-rt-v2\`. Already downloaded for `mrt2_small` and `mrt2_base`. To fetch again:

```bash
uv venv --python 3.12 && source .venv/Scripts/activate
uv pip install -e ".[dev]"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8       # mrt crashes on cp1252 without this
mrt models init
mrt checkpoints download mrt2_small               # JAX needs the raw checkpoint
```

### 2. Build the image (one-time, ~11 GB)

```bash
docker build -t moonstar-magenta .
```

If `docker build` hangs (seen once, cause unknown), skip it and mount the source instead (see "Run without rebuilding").

### 3. Run chunked streaming on the GPU

```bash
export MSYS_NO_PATHCONV=1                         # stops Git Bash mangling /root/... paths
mkdir -p data
docker run --rm --gpus all \
  -v "C:\Users\HomePc\Documents\Magenta:/root/Documents/Magenta" \
  -v "$PWD/data:/app/data" \
  moonstar-magenta python -m moonstar_magenta.stream \
  --seconds 20 --prompt "0:lo-fi hip hop, mellow piano" --prompt "10:driving techno, heavy bass"
```

- Output: `data/stream.wav`. Prompt format is `start_sec:text`, repeatable.
- Prints one line per prompt switch, then `chunks=… warm mean=… underruns=…`.

### 4. Plain benchmark (no streaming)

```bash
docker run --rm --gpus all -v "C:\Users\HomePc\Documents\Magenta:/root/Documents/Magenta" \
  moonstar-magenta mrt jax generate --prompt "lo-fi" --duration 10.0 --model=mrt2_small
```

`mrt2_base` only fits with preallocation off:

```bash
docker run --rm --gpus all -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  -v "C:\Users\HomePc\Documents\Magenta:/root/Documents/Magenta" \
  moonstar-magenta mrt jax generate --prompt "lo-fi" --duration 10.0 --model=mrt2_base
```

### Run without rebuilding

The image bakes in a copy of `moonstar_magenta/`. After editing code, mount the live source over it:

```bash
docker run --rm --gpus all -e PYTHONPATH=/app \
  -v "C:\Users\HomePc\Documents\Magenta:/root/Documents/Magenta" \
  -v "$PWD/moonstar_magenta:/app/moonstar_magenta" -v "$PWD/data:/app/data" \
  moonstar-magenta python -m moonstar_magenta.stream --seconds 20
```

### Native (CPU only, tests and timing baseline)

```bash
source .venv/Scripts/activate
python -m pytest -q
PYTHONUTF8=1 python -m moonstar_magenta.bench mrt2_small 4     # writes data/bench.json
```

## Live playback (browser)

```bash
export MSYS_NO_PATHCONV=1
docker run -d --name mm-live --gpus all -p 8765:8765 -e PYTHONPATH=/app \
  -v "C:\Users\HomePc\Documents\Magenta:/root/Documents/Magenta" \
  -v "$PWD/moonstar_magenta:/app/moonstar_magenta" -v "$PWD/data:/app/data" \
  moonstar-magenta python -m moonstar_magenta.live --seed 7 --cfg 6 \
  --prompt "0.7*file:data/samples/ref1.wav | 0.3*lo-fi hip hop"
```

Open http://localhost:8765, type a new prompt (same blend syntax) and press Enter. Stop with `docker rm -f mm-live`.

- Generation stays ~`--lead` seconds (default 2) ahead of playback, so a prompt change is heard after about that long.
- `--seed` only affects text-prompt embedding (mapper noise). Upstream `generate()` has no sampling seed, so audio is not reproducible run to run.
- Prompt change is a hard switch (no crossfade yet).

## Audio-conditioned prompts and comparison

Prompts can be audio or a weighted blend: `--prompt "0:0.7*file:data/samples/ref1.wav | 0.3*funky bass"` (weights normalised, `file:` paths are read inside the container, so mount `data/`).

```bash
uv pip install -e ".[analysis]"                    # yt-dlp + librosa, host only
python scripts/fetch_audio.py URL ref1 --start 30 --seconds 30   # -> data/samples/ref1.wav (needs ffmpeg)
python scripts/compare.py data/samples/ref1.wav data/ab_*.wav    # tempo / key / brightness vs reference
```

`compare.py` is a blunt objective check (chroma cosine, tempo, spectral centroid). It does not replace listening.

First result (ref1 vs 20 s outputs, `mrt2_small` unless noted): cfg 6 gave the best key match (chroma cosine 0.92) and closest brightness; `mrt2_base` was the worst match (0.81, too dark) and slower.

## Gotchas

- `mrt` on Windows needs `PYTHONUTF8=1`.
- The JAX path needs `mrt checkpoints download <model>`, not just `mrt models download`.
- `mrt2_base` OOMs on 16 GB unless `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform`. Windows itself holds ~1.2 GB of the card.
- `mrt2_base` ran at exactly 1.0x on one cold 10 s run. No pacing in upstream `generate.py`, so it's a real number, but too tight to trust.
- Git Bash rewrites `/root/...` paths in `docker run` unless `MSYS_NO_PATHCONV=1`.
- `from magenta_rt.config import MUSICCOCA` (not `magenta_rt.musiccoca`).

## Next steps

Done since the spike: listening (works well with simple prompts), live browser playback with prompt changes, `--timeline`, `--reset-every`, the ACE chain and lyrics. Open items (full list in `docs/2026-10-28-checkpoint.md`):

1. Listen to the 10-minute chain output and the lyrics clips.
2. Smooth live prompt transitions (crossfade or interpolate embeddings instead of a hard switch).
3. Cover mode in the `moonstar-music` ACE provider (it is text2music only today).
4. Re-test `mrt2_base` with the reset recipe (it matched the reference worse and was borderline real time before).
5. Expose temperature, top-k, cfg as live knobs.
6. `docker-compose.yml` with the GPU and volume flags so the long `docker run` isn't needed.

## Chain: Magenta -> ACE-Step cover

Magenta gives a quick style sketch; an ACE-Step 1.5 cover turns it into a coherent track.

```bash
# 1. sketch (30 s) with Magenta in Docker -> data/chain/mag_x.wav  (see "Run chunked streaming")
# 2. start ACE-Step:  cd D:/dev/ACE-Step-1.5 && uv run acestep-api     (localhost:8001)
scripts/ace_cover.sh data/chain/mag_x.wav data/chain/ace_x_s05.wav 0.5 "same style words as the Magenta prompt, polished studio mix"
```

Strength 0..1 is how much of the source is kept (0.5 and 0.8 tried). Only one of Magenta or ACE should hold the GPU at a time. Judge by ear: chroma metrics did not predict quality.

## Long references (DJ sets)

A 30 s clip misses the arc of a set. Sample the whole video instead:

```bash
python scripts/fetch_audio.py URL set1 --spread 12 --seconds 30     # 12 windows evenly spaced, order kept (6 min)
```

Magenta embeds long audio in 10 s clips. By default they are averaged into one style (`--prompt "0:file:data/samples/set1.wav"`).
To follow the set's tonal arc instead, use `--timeline` (the clips' style drifts from start to end over `--seconds`):

```bash
python -m moonstar_magenta.stream --seconds 180 --cfg 6 --timeline data/samples/set1.wav --timeline-smooth 3 --out data/set/mag_timeline.wav
scripts/ace_cover.sh data/set/mag_timeline.wav data/set/ace_timeline.wav 0.3 "electronic DJ set, club mix, polished studio mix" 180
```

`--reset-every N` restarts Magenta's carried state every N seconds (0.5 s equal-power crossfade, `--fade`) to fight the quality drop seen after ~80 s of continuous generation. Untested by ear until compared against the no-reset runs.

## Best recipe so far (judged by ear, 2026-10-28)

```bash
TIMELINE=data/samples/set1.wav scripts/chain.sh myset "electronic DJ set, club mix, driving groove" 180
```

Timeline over a spread-sampled long set + `--reset-every 60` + temperature 1.0, then an ACE cover at strength **0.6**. Findings: timeline beat pooled; resets fixed the quality drop after ~80 s; at 0.3 ACE made a good sketch worse, 0.6 won narrowly (0.8 roughly tied with the raw sketch, results vary by section). Note: `TEMP` is a Windows env var, so the temperature override is `MAG_TEMP`.

## Lyrics

`LYRICS_FILE=data/lyrics/song.txt scripts/chain.sh NAME "style words" SECONDS` sings those lyrics (ACE tags like `[Verse]`/`[Chorus]` help). Keep lyric files under `data/` (gitignored); do not commit third-party lyrics. Magenta supplies only an instrumental sketch, so the vocal comes entirely from ACE.
