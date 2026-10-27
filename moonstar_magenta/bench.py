"""Timing harness: shell out to `mrt jax generate` and report real-time factor (RTF).

RTF = audio_seconds / wall_seconds. RTF >= 1.0 means faster than real time.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time


def rtf(audio_sec: float, wall_sec: float) -> float:
    return audio_sec / wall_sec if wall_sec > 0 else 0.0


def run_once(prompt: str, duration: float, model: str, out: pathlib.Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-c", "from magenta_rt.cli import main; main()", "jax", "generate", "--prompt", prompt, "--duration", str(duration), f"--model={model}"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    wall = time.perf_counter() - t0
    m = re.search(r"([\d.]+) steps/s", proc.stderr + proc.stdout)
    return {
        "steps_per_sec": float(m.group(1)) if m else None,  # 25 steps/s == real time
        "model": model, "duration": duration, "wall_sec": round(wall, 2),
        "rtf": round(rtf(duration, wall), 3), "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-500:], "stderr_tail": proc.stderr[-500:],
    }


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "mrt2_small"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    result = run_once("lo-fi hip hop, mellow piano", duration, model, pathlib.Path("data/bench.json"))
    pathlib.Path("data").mkdir(exist_ok=True)
    pathlib.Path("data/bench.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
