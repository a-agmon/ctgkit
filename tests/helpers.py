"""Helpers to build controlled synthetic CTG traces for testing.

These let tests inject a specific feature (a decel of known duration, a
contraction train, low variability for N minutes) so behaviour can be
asserted precisely. NOT for clinical use.
"""
from __future__ import annotations

import numpy as np
from ctgkit import from_arrays


def base_trace(minutes=30, hz=4.0, baseline=140.0, var_sd=2.6, seed=0):
    n = int(minutes * 60 * hz)
    rng = np.random.default_rng(seed)
    fast = var_sd * np.convolve(rng.normal(0, 1, n), np.ones(int(hz * 3)) / (hz * 3), mode="same")
    slow = 4.0 * np.sin(2 * np.pi * 0.01 * np.arange(n) / hz)
    return baseline + slow + fast


def add_contractions(n, hz, every_s=180, width_s=50, amp=40.0):
    t = np.arange(n) / hz
    toco = np.zeros(n)
    for ct in np.arange(60, n / hz, every_s):
        toco += amp * np.exp(-((t - ct) ** 2) / (2 * (width_s / 2.5) ** 2))
    return toco


def trace_with_decel(width_s, depth=50.0, at_frac=0.5, with_toco=False,
                     lag_s=0.0, minutes=30, hz=4.0, baseline=140.0, seed=0):
    """Single deceleration of given width/depth. If lag_s>0 and with_toco,
    the dip nadir lags the contraction peak (late-style)."""
    fhr = base_trace(minutes, hz, baseline, seed=seed)
    n = len(fhr)
    t = np.arange(n) / hz
    toco = add_contractions(n, hz) if with_toco else None
    ct = minutes * 60 * at_frac
    sigma = width_s / 2.5
    fhr -= depth * np.exp(-((t - (ct + lag_s)) ** 2) / (2 * sigma ** 2))
    return from_arrays(fhr=fhr, hz=hz, toco=toco)


def trace_low_variability(low_minutes, minutes=30, hz=4.0, baseline=140.0, seed=0):
    """Near-flat variability (sd~0.15) for the first `low_minutes`, normal after."""
    n = int(minutes * 60 * hz)
    rng = np.random.default_rng(seed)
    cut = int(low_minutes * 60 * hz)
    sd = np.where(np.arange(n) < cut, 0.15, 2.6)
    fast = np.convolve(rng.normal(0, 1, n) * sd, np.ones(int(hz * 3)) / (hz * 3), mode="same")
    fhr = baseline + 4.0 * np.sin(2 * np.pi * 0.01 * np.arange(n) / hz) + fast
    return from_arrays(fhr=fhr, hz=hz, toco=add_contractions(n, hz))


def trace_recurrent_late(n_contractions=10, minutes=30, hz=4.0, seed=0):
    """Late decels on most contractions: nadir lags each contraction peak."""
    n = int(minutes * 60 * hz)
    fhr = base_trace(minutes, hz, seed=seed)
    t = np.arange(n) / hz
    toco = add_contractions(n, hz)
    for ct in np.arange(60, n / hz, 180)[:n_contractions]:
        fhr -= 25 * np.exp(-((t - (ct + 30)) ** 2) / (2 * 20 ** 2))   # lag 30 s
    return from_arrays(fhr=fhr, hz=hz, toco=toco)


def trace_tachysystole(minutes=30, hz=4.0, fhr_abnormal=False, seed=0):
    """>5 contractions per 10 min. Optionally with abnormal FHR."""
    n = int(minutes * 60 * hz)
    baseline = 175.0 if fhr_abnormal else 140.0
    fhr = base_trace(minutes, hz, baseline=baseline, seed=seed)
    toco = add_contractions(n, hz, every_s=90)   # ~6.7 / 10 min
    return from_arrays(fhr=fhr, hz=hz, toco=toco)
