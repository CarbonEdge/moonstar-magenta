"""Objective comparison of generated clips against a reference clip (run on the host).

    python scripts/compare.py data/samples/ref1.wav data/ab_*.wav
"""
import argparse
import pathlib

import librosa
import numpy as np

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def profile(path: str) -> dict:
    y, sr = librosa.load(path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1)
    chroma /= chroma.sum()
    return {
        "dur": len(y) / sr,
        "tempo": float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0]),
        "centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
        "rms": float(librosa.feature.rms(y=y).mean()),
        "chroma": chroma,
    }


def chroma_cosine(ref: np.ndarray, other: np.ndarray, shift: int = 0) -> float:
    o = np.roll(other, shift)
    return float(ref @ o / np.linalg.norm(ref) / np.linalg.norm(o))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("clips", nargs="+")
    args = ap.parse_args()

    ref = profile(args.reference)
    top = ",".join(NOTES[i] for i in np.argsort(ref["chroma"])[::-1][:3])
    print(f"ref  {pathlib.Path(args.reference).name:24s} tempo={ref['tempo']:6.1f} "
          f"cent={ref['centroid']:6.0f} top={top}")
    for clip in args.clips:
        p = profile(clip)
        print(f"     {pathlib.Path(clip).name:24s} tempo={p['tempo']:6.1f} cent={p['centroid']:6.0f} "
              f"rms={p['rms']:.3f} key_match={chroma_cosine(ref['chroma'], p['chroma']):.3f}")


if __name__ == "__main__":
    main()
