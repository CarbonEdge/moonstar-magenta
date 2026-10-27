# moonstar-magenta

Local [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) experiments. Sibling of `moonstar-music` (batch, ACE-Step); this repo explores *live / steerable* generation.

## Status: spike

- Real-time streaming is **Apple Silicon only** upstream; on NVIDIA only offline JAX generation is supported.
- JAX CUDA wheels are **Linux-only** — on native Windows JAX runs on CPU. GPU needs WSL2 Ubuntu.
- Goal of the spike: measure real-time factor of `mrt2_small` / `mrt2_base` on the RTX 4080 Super.

## Run

```bash
uv venv --python 3.12 && source .venv/Scripts/activate
uv pip install -e ".[dev]"
mrt models init && mrt models download
python -m moonstar_magenta.bench mrt2_small 10     # writes data/bench.json
```

## Results (RTX 4080 Super, 10 s clip; real time = 25 steps/s)

| Run | Speed | Real time? |
|---|---|---|
| `mrt2_small`, CPU (native Windows) | 6.4 steps/s | No (0.26x) |
| `mrt2_small`, GPU (Docker) | 76.7 steps/s (13 ms/step) | Yes (~3x) |
| `mrt2_base`, GPU (Docker) | 25.0 steps/s (40 ms/step) | Borderline (~1.0x), single cold run |

- Native Windows JAX is CPU-only (CUDA wheels are Linux-only), so run in Docker with `--gpus all`.
- `mrt2_base` OOMs unless `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
- Windows needs `PYTHONUTF8=1` for `mrt`; the JAX path needs `mrt checkpoints download <model>`.
- Output quality not yet judged by ear.

## GPU / streaming

```bash
docker build -t moonstar-magenta .
MSYS_NO_PATHCONV=1 docker run --rm --gpus all \
  -v "C:\Users\<you>\Documents\Magenta:/root/Documents/Magenta" -v "$PWD/data:/app/data" \
  moonstar-magenta python -m moonstar_magenta.stream \
  --seconds 20 --prompt "0:lo-fi piano" --prompt "10:driving techno"
```
