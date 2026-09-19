#!/usr/bin/env python3
# Copyright 2026 Addis Pulse Studio. Licensed under the Apache License, Version 2.0.
"""How closely a generated soundtrack follows a reference audio track, and whether it drifts.

    python tools/check_av_alignment.py <reference audio> <generated audio or video> [...more generated]

Needs ffmpeg on PATH and numpy. Both tracks are decoded to 16 kHz mono and compared by their 10 ms RMS loudness
envelope (the audio VAE regenerates the waveform, so sample-level phase is not expected to match): correlation at zero
lag, the best lag within +-500 ms, and the best lag of each half separately -- equal half lags mean no drift.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

RATE, HOP = 16000, 160


def decode(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def envelope(x: np.ndarray) -> np.ndarray:
    n = len(x) // HOP
    return np.sqrt((x[: n * HOP].reshape(n, HOP) ** 2).mean(axis=1))


def best_lag(a: np.ndarray, b: np.ndarray, window: int) -> tuple[int, float]:
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    scores = {}
    for k in range(-window, window + 1):
        x, y = a[max(0, k): m + min(0, k)], b[max(0, -k): m - max(0, k)]
        if len(x) > 10 and x.std() > 0 and y.std() > 0:
            scores[k] = float(np.corrcoef(x, y)[0, 1])
    k = max(scores, key=scores.get)
    return k, scores[k]


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    ref = envelope(decode(argv[1]))
    for path in argv[2:]:
        gen = envelope(decode(path))
        m = min(len(ref), len(gen))
        r, g = ref[:m], gen[:m]
        corr0 = float(np.corrcoef(r, g)[0, 1])
        lag, corr = best_lag(r, g, 50)
        h = m // 2
        lag1, _ = best_lag(r[:h], g[:h], 30)
        lag2, _ = best_lag(r[h:], g[h:], 30)
        print(f"{path}\n  envelope corr @0 ms {corr0:+.3f} | best lag {lag * 10:+d} ms (corr {corr:+.3f}) | "
              f"half lags {lag1 * 10:+d} / {lag2 * 10:+d} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
