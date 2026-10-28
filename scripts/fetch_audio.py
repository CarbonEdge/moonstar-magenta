"""Download audio from a YouTube URL to data/samples/<name>.wav (run on the host, not in Docker).

    python scripts/fetch_audio.py URL name [--start 30 --seconds 20]
    python scripts/fetch_audio.py URL name --spread 12 --seconds 30   # 12 windows evenly spaced over the whole video

--spread is for long DJ sets: it keeps the order of the set, so the result is a compressed
version of its tonal arc (use it with `stream --timeline`).
"""
import argparse
import pathlib
import subprocess
import tempfile

import yt_dlp


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", *args], check=True, capture_output=True)


def _duration(path: pathlib.Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def window_starts(total: float, n: int, seconds: float) -> list[float]:
    """n window start times evenly spread over [0, total - seconds]."""
    if n < 2 or total <= seconds:
        return [0.0]
    span = total - seconds
    return [span * i / (n - 1) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("name")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="clip length to keep (per window with --spread)")
    ap.add_argument("--spread", type=int, default=0, help="take N windows evenly spaced over the whole video")
    args = ap.parse_args()

    out = pathlib.Path("data/samples") / f"{args.name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        opts = {"format": "bestaudio/best", "outtmpl": f"{tmp}/src.%(ext)s", "quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([args.url])
        src = next(pathlib.Path(tmp).glob("src.*"))
        if args.spread:
            total = _duration(src)
            starts = window_starts(total, args.spread, args.seconds)
            parts = []
            for i, s in enumerate(starts):
                part = pathlib.Path(tmp) / f"w{i:03d}.wav"
                _ffmpeg("-ss", str(s), "-t", str(args.seconds), "-i", str(src), "-ar", "48000", "-ac", "2", str(part))
                parts.append(part)
            lst = pathlib.Path(tmp) / "list.txt"
            lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
            _ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out))
            print(f"video={total / 60:.1f} min, {len(starts)} windows x {args.seconds:.0f}s, "
                  f"starts(min)={[round(s / 60, 1) for s in starts]}")
        else:
            _ffmpeg("-ss", str(args.start), "-t", str(args.seconds), "-i", str(src), "-ar", "48000", "-ac", "2", str(out))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
