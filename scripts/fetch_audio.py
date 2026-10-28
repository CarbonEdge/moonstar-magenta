"""Download audio from a YouTube URL to data/samples/<name>.wav (run on the host, not in Docker).

    python scripts/fetch_audio.py URL name [--start 30 --seconds 20]
"""
import argparse
import pathlib
import subprocess
import tempfile

import yt_dlp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("name")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=30.0, help="clip length to keep")
    args = ap.parse_args()

    out = pathlib.Path("data/samples") / f"{args.name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        opts = {"format": "bestaudio/best", "outtmpl": f"{tmp}/src.%(ext)s", "quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([args.url])
        src = next(pathlib.Path(tmp).glob("src.*"))
        subprocess.run(["ffmpeg", "-y", "-ss", str(args.start), "-t", str(args.seconds),
                        "-i", str(src), "-ar", "48000", "-ac", "2", str(out)],
                       check=True, capture_output=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
