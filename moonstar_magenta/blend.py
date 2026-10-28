"""Prompt specs that mix text and audio: '0.7*file:data/samples/a.wav | 0.3*funky bass'."""
import re

import numpy as np

_WEIGHT = re.compile(r"^\s*(\d*\.?\d+)\s*\*\s*(.*)$")


def parse_spec(spec: str) -> list[tuple[float, str]]:
    """'0.7*file:a.wav | funk' -> [(0.7, 'file:a.wav'), (1.0, 'funk')]. Weights default to 1."""
    parts = []
    for raw in spec.split("|"):
        m = _WEIGHT.match(raw)
        weight, text = (float(m.group(1)), m.group(2)) if m else (1.0, raw)
        if text.strip():
            parts.append((weight, text.strip()))
    return parts


def blend(weighted: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Weighted mean of embeddings (weights are normalised)."""
    total = sum(w for w, _ in weighted)
    return sum(w * np.asarray(e) for w, e in weighted) / total
