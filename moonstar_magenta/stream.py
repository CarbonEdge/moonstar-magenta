"""Chunked streaming on top of MagentaRT2System.generate(state=...).

Generates 1 s chunks, carrying `state` between calls, switching the style prompt on a
schedule (or from a prompt file polled each chunk). Writes one continuous WAV and reports
per-chunk real-time factor: a chunk must finish in < chunk_sec of wall time to keep up.
"""
import argparse
import pathlib
import time

import numpy as np

from moonstar_magenta.blend import blend, parse_spec

FRAMES_PER_SEC = 25


def prompt_at(schedule: list[tuple[float, str]], t: float) -> str:
    """Last prompt whose start time <= t. `schedule` is sorted [(start_sec, prompt), ...]."""
    current = schedule[0][1]
    for start, text in schedule:
        if start <= t:
            current = text
    return current


def parse_schedule(items: list[str]) -> list[tuple[float, str]]:
    """'0:lofi piano' '8:driving techno' -> [(0.0, 'lofi piano'), (8.0, 'driving techno')]."""
    out = []
    for item in items:
        start, _, text = item.partition(":")
        out.append((float(start), text.strip()))
    return sorted(out)


def underruns(chunk_walls: list[float], chunk_sec: float) -> int:
    """Chunks that took longer than the audio they produced (would glitch live playback)."""
    return sum(1 for w in chunk_walls if w > chunk_sec)


def embed_spec(mrt, spec: str, Waveform, seed: int = 0):
    """Embed one prompt spec: text, 'file:<path>' audio, or a weighted '|' blend of both."""
    parts = []
    for weight, text in parse_spec(spec):
        if text.startswith("file:"):
            wav = Waveform.from_file(text[5:])
            parts.append((weight, mrt.embed_style(wav)))
        else:
            parts.append((weight, mrt.embed_style(text, use_mapper=True, seed=seed)))
    return parts[0][1] if len(parts) == 1 else blend(parts)


def smooth_timeline(emb: np.ndarray, window: int) -> np.ndarray:
    """Moving average over time (edge-padded) so the style drifts instead of jumping clip to clip."""
    if window <= 1 or len(emb) < 2:
        return emb
    pad = window // 2
    padded = np.pad(emb, ((pad, window - 1 - pad), (0, 0)), mode="edge")
    return np.stack([padded[i:i + window].mean(0) for i in range(len(emb))])


def embedding_at(timeline: np.ndarray, frac: float) -> np.ndarray:
    """Linear interpolation along a (n, dim) timeline at position frac in [0, 1]."""
    pos = min(max(frac, 0.0), 1.0) * (len(timeline) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(timeline) - 1)
    return timeline[lo] * (1 - (pos - lo)) + timeline[hi] * (pos - lo)


def main() -> None:
    ap = argparse.ArgumentParser("stream")
    ap.add_argument("--model", default="mrt2_small")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--chunk-sec", type=float, default=1.0)
    ap.add_argument("--prompt", action="append", default=["0:lo-fi hip hop, mellow piano"],
                    help="start_sec:prompt, repeatable")
    ap.add_argument("--temperature", type=float, default=None, help="default 1.3")
    ap.add_argument("--top-k", type=int, default=None, help="default 40")
    ap.add_argument("--cfg", type=float, default=None, help="musiccoca guidance scale")
    ap.add_argument("--timeline", help="long audio file: follow its per-10s style embeddings from start to end over --seconds")
    ap.add_argument("--timeline-smooth", type=int, default=3, help="moving-average window (clips) over the timeline")
    ap.add_argument("--seed", type=int, default=0, help="mapper noise seed for text prompts")
    ap.add_argument("--out", default="data/stream.wav")
    args = ap.parse_args()

    from magenta_rt import MagentaRT2Jax
    from magenta_rt.audio import Waveform
    from magenta_rt.config import MUSICCOCA

    schedule = parse_schedule(args.prompt[1:] if len(args.prompt) > 1 else args.prompt)
    mrt = MagentaRT2Jax(size=args.model)
    frames = int(args.chunk_sec * FRAMES_PER_SEC)
    embeddings: dict[str, object] = {}
    state, chunks, walls, last = None, [], [], None

    timeline = None
    if args.timeline:
        raw = np.asarray(mrt.embed_style(Waveform.from_file(args.timeline), pool_across_time=False))
        timeline = smooth_timeline(raw.reshape(-1, raw.shape[-1]), args.timeline_smooth)
        print(f"timeline: {len(timeline)} clips from {args.timeline}", flush=True)

    t = 0.0
    while t < args.seconds:
        prompt = prompt_at(schedule, t)
        if timeline is not None:
            emb = embedding_at(timeline, t / args.seconds)
            prompt = f"timeline {t / args.seconds:4.0%}"
        else:
            if prompt not in embeddings:
                embeddings[prompt] = embed_spec(mrt, prompt, Waveform, args.seed)
            emb = embeddings[prompt]
        if prompt != last and timeline is None:
            print(f"[{t:5.1f}s] prompt -> {prompt!r}", flush=True)
            last = prompt
        t0 = time.perf_counter()
        wav, state = mrt.generate(conditioning={MUSICCOCA.key: emb},
                                  frames=frames, state=state,
                                  temperature=args.temperature, top_k=args.top_k,
                                  cfg_scales={"musiccoca": args.cfg} if args.cfg else None)
        wall = time.perf_counter() - t0
        walls.append(wall)
        chunks.append(wav.samples)
        t += args.chunk_sec

    samples = np.concatenate(chunks, axis=0)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Waveform(samples, mrt._sample_rate).write(str(out))
    warm = walls[1:] or walls  # first chunk includes warm-up
    print(f"chunks={len(walls)} audio={args.seconds}s  first={walls[0]:.2f}s  "
          f"warm mean={np.mean(warm):.3f}s max={np.max(warm):.3f}s  "
          f"underruns={underruns(walls, args.chunk_sec)}  saved={out}")


if __name__ == "__main__":
    main()
