# moonstar-magenta

Local [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) experiments. Sibling of `moonstar-music` (batch, ACE-Step); this repo explores *live / steerable* generation.

## Status: spike (works, not yet live)

- Chunked streaming on GPU works: 1 s chunks, state carried between chunks, prompt switchable mid-stream, 0 underruns.
- It writes a WAV file. No live playback, no live prompt input yet.
- Output quality not yet judged by ear.

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

1. **Listen** to `data/stream.wav`: is it musical, and is the prompt switch at 10 s smooth or jarring? This decides whether anything below is worth building.
2. **Rebuild the image** so `stream.py` is baked in (`docker build` hung once; retry when Docker Desktop is idle).
3. **Make it live:**
   - Stream chunks to a browser (small HTTP/WebSocket server in the container, port-mapped) or to local speakers.
   - Accept prompt changes while it plays (stdin or a web text box), instead of a fixed schedule.
   - Needs a small playback buffer: stay a few chunks ahead of the playhead.
4. **Smooth prompt transitions:** crossfade or interpolate between style embeddings rather than a hard switch.
5. **Re-test `mrt2_base`:** 60 s warm run, steady-state steps/s, VRAM headroom. Only worth it if quality is clearly better than small.
6. **Other controls** upstream exposes: temperature, top-k, CFG scales (`musiccoca`, `notes`, `drums`). Expose them as live knobs.
7. **Connect to the rest of the workbench (optional):** a `moonstar-music` provider mode, or a gateway-driven "DJ" that changes prompts over time. Keep it a pure HTTP client; no imports across repos.
8. **Housekeeping:** add a `docker-compose.yml` with the GPU and volume flags so the long `docker run` isn't needed; add `moonstar-magenta` to the workbench root `CLAUDE.md` project table.
