"""Synthetic CTG generation for testing/demos. NOT for clinical use."""
from __future__ import annotations

import numpy as np
from .io import Signal


def _running(x, w):
    if w < 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def synth_epoch(
    kind: str = "normal",
    minutes: float = 30.0,
    hz: float = 4.0,
    seed: int | None = 0,
) -> Signal:
    """Generate a synthetic 30-min epoch.

    kind: 'normal' | 'tachy' | 'late_decels' | 'prolonged' | 'low_var' | 'noisy'
    """
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * hz)
    t = np.arange(n) / hz

    base = {"normal": 140, "tachy": 172, "late_decels": 145,
            "prolonged": 140, "low_var": 138, "noisy": 142}.get(kind, 140)

    # Physiological FHR = slow baseline wander + fast beat-to-beat variability.
    slow = 4.0 * np.sin(2 * np.pi * 0.01 * t)            # very slow wander
    fast_amp = 0.6 if kind == "low_var" else 2.6          # short-term variability sd
    fast = fast_amp * rng.normal(0, 1, n)
    fast = _running(fast, int(hz * 3))                    # smooth to beat scale
    fhr = base + slow + fast

    # contractions every ~3 min
    toco = np.zeros(n)
    contraction_times = np.arange(60, minutes * 60, 180)
    for ct in contraction_times:
        c = 40 * np.exp(-((t - ct) ** 2) / (2 * 25 ** 2))
        toco += c

    if kind == "late_decels":
        for ct in contraction_times:
            # decel lagging contraction peak by ~30 s
            dip = 30 * np.exp(-((t - (ct + 30)) ** 2) / (2 * 20 ** 2))
            fhr -= dip
    if kind == "prolonged":
        ct = minutes * 60 * 0.6
        dip = 45 * np.exp(-((t - ct) ** 2) / (2 * 120 ** 2))  # ~6 min wide
        fhr -= dip
    if kind == "noisy":
        drop = rng.random(n) < 0.6
        fhr[drop] = np.nan

    toco += rng.normal(0, 1.5, n)
    toco = np.clip(toco, 0, None)

    return Signal(fhr=fhr, hz=hz, toco=toco, meta={"synthetic": kind})
