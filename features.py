"""Feature extraction mapping cleanly to guideline constructs.

All detectors are deterministic and explainable. Each returns rich event
records (type, start, duration, depth, recovery, alignment to contractions)
so the rules engine and the concern objects can cite specific evidence.

NOTE: timing of decelerations relative to contractions ('late' vs 'variable')
is only asserted when a usable toco channel exists. Without toco, decels are
surfaced as 'timing uncertain'. This is the conservative, clinically honest
choice from the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .preprocess import Clean


@dataclass
class Deceleration:
    start_min: float
    duration_min: float
    depth_bpm: float
    nadir_min: float
    # morphology and prolonged/severe are INDEPENDENT axes. A deceleration can
    # be variable-shaped AND prolonged at the same time. `dtype` describes
    # shape/timing only; `prolonged`/`severe` are duration/depth driven and are
    # set regardless of morphology so a long deep dip is never missed.
    morphology: str = "uncertain"   # early | late | variable | uncertain
    timing: str = "uncertain"       # uncertain | aligned | unaligned
    aligned_to_contraction: bool | None = None
    reduced_var_within: bool = False
    prolonged: bool = False         # duration >= PROLONGED_MIN_S
    prolonged_severe: bool = False  # duration >= PROLONGED_SEVERE_S
    severe_depth: bool = False      # depth >= SEVERE_DEPTH_BPM

    @property
    def dtype(self) -> str:
        """Back-compat label. Prolonged dominates the displayed type, but the
        underlying morphology is preserved in `morphology`."""
        if self.prolonged:
            return "prolonged"
        return self.morphology


# Duration thresholds (seconds). Prolonged is detected by DURATION ALONE,
# independent of whether the dip also looks variable/late or relates to a
# contraction. Severe-prolonged crosses the ~5-min critical line.
PROLONGED_MIN_S = 120.0          # >= 2 min: prolonged
PROLONGED_SEVERE_S = 300.0       # >= 5 min: severe/critical acute event
SEVERE_DEPTH_BPM = 60.0          # deep deceleration regardless of duration


@dataclass
class Acceleration:
    start_min: float
    duration_min: float
    amp_bpm: float


@dataclass
class Contraction:
    start_min: float
    duration_min: float
    peak: float


@dataclass
class Features:
    baseline_bpm: float | None
    baseline_slope_bpm_per_min: float | None
    variability_bpm: float | None
    variability_low_min: float          # minutes with variability < 5
    accelerations: list[Acceleration] = field(default_factory=list)
    decelerations: list[Deceleration] = field(default_factory=list)
    contractions: list[Contraction] = field(default_factory=list)
    contractions_per_10min: float | None = None
    tachysystole: bool = False
    tachysystole_low_confidence: bool = False  # toco present but quality-rejected
    early_tachysystole_with_fhr_change: bool = False
    sinusoidal: bool = False
    has_prolonged_any: bool = False     # any decel >= 2 min (duration-driven)
    has_prolonged_gt5: bool = False     # any decel >= 5 min (severe acute event)
    has_acute_event_ge3: bool = False
    max_decel_duration_min: float = 0.0
    quick_recovery: bool = True
    n_recurrent_late: int = 0
    n_complicated_variable: int = 0
    toco_available: bool = False


def _smooth(x: np.ndarray, hz: float, win_s: float) -> np.ndarray:
    w = max(1, int(win_s * hz))
    kern = np.ones(w) / w
    filled = np.copy(x)
    # forward/back fill NaNs for smoothing only
    idx = np.where(np.isfinite(filled))[0]
    if len(idx) == 0:
        return filled
    filled = np.interp(np.arange(len(filled)), idx, filled[idx])
    return np.convolve(filled, kern, mode="same")


def estimate_baseline(fhr: np.ndarray, hz: float, win_min: float = 10.0):
    """Baseline = stable mean over rolling window, excluding excursions.
    Returns (baseline_bpm, slope_bpm_per_min)."""
    finite = fhr[np.isfinite(fhr)]
    if len(finite) < hz * 60:
        return None, None
    base_track = _smooth(fhr, hz, 60.0)
    # exclude samples far from the running mean (accels/decels)
    med = np.nanmedian(base_track)
    keep = np.abs(base_track - med) < 25
    baseline = float(np.nanmean(base_track[keep])) if keep.any() else float(med)

    # slope over the epoch (first third vs last third of valid baseline track)
    n = len(base_track)
    first = np.nanmean(base_track[: n // 3])
    last = np.nanmean(base_track[2 * n // 3:])
    total_min = n / hz / 60.0
    slope = float((last - first) / max(total_min * (2 / 3), 1e-6))
    return round(baseline, 1), round(slope, 2)


def estimate_variability(fhr: np.ndarray, hz: float):
    """Per-minute short-term variability band; returns (mean_bpm, minutes_below_5).

    Uses the amplitude of the de-trended signal where 'trend' is a longer
    (~30 s) running mean, so genuine baseline wander is removed and only the
    short-term oscillation that clinicians read as 'variability' remains.
    The band width is taken as ~4x the robust standard deviation of the
    residual, which maps onto the visual peak-to-trough bandwidth.
    """
    spm = int(hz * 60)
    if spm <= 0:
        return None, 0.0
    bands = []
    low_minutes = 0.0
    for s in range(0, len(fhr), spm):
        seg = fhr[s:s + spm]
        mask = np.isfinite(seg)
        if mask.sum() < spm * 0.5:
            continue
        seg = seg.copy()
        idx = np.where(mask)[0]
        seg = np.interp(np.arange(len(seg)), idx, seg[idx])
        trend = _smooth(seg, hz, 30.0)        # remove baseline wander
        detr = seg - trend[: len(seg)]
        # robust sigma via MAD, band ~= 4 sigma (≈ visual peak-to-trough)
        mad = np.median(np.abs(detr - np.median(detr)))
        sigma = 1.4826 * mad
        amp = float(4.0 * sigma)
        bands.append(amp)
        if amp < 5.0:
            low_minutes += 1.0
    if not bands:
        return None, 0.0
    return round(float(np.mean(bands)), 1), low_minutes


def detect_accelerations(fhr: np.ndarray, hz: float, baseline: float | None):
    if baseline is None:
        return []
    track = _smooth(fhr, hz, 15.0)
    above = track >= baseline + 15
    return _segments_to_events(above, hz, fhr, baseline, kind="accel", min_s=15)


def detect_decelerations(fhr: np.ndarray, hz: float, baseline: float | None,
                         contractions: list["Contraction"], toco_ok: bool):
    if baseline is None:
        return []
    track = _smooth(fhr, hz, 10.0)
    below = track <= baseline - 15
    decels: list[Deceleration] = []
    for seg in _bool_runs(below):
        s, e = seg
        dur_min = (e - s) / hz / 60.0
        dur_s = dur_min * 60.0
        if dur_s < 15:                  # < 15 s ignore
            continue
        seg_vals = track[s:e]
        depth = float(baseline - np.nanmin(seg_vals))
        nadir = s + int(np.nanargmin(seg_vals))
        start_min = s / hz / 60.0
        nadir_min = nadir / hz / 60.0

        # 1) DURATION/DEPTH flags — set independently of morphology so a long
        #    or deep deceleration is NEVER missed because it also looked variable.
        prolonged = dur_s >= PROLONGED_MIN_S
        prolonged_severe = dur_s >= PROLONGED_SEVERE_S
        severe_depth = depth >= SEVERE_DEPTH_BPM

        # 2) MORPHOLOGY/TIMING — best-effort shape classification, requires toco
        #    for late/variable. This never suppresses the prolonged flags above.
        morphology = "uncertain"
        timing = "uncertain"
        aligned = None
        if toco_ok and contractions:
            aligned, morphology = _classify_decel_timing(
                start_min, nadir_min, dur_min, contractions)
            timing = "aligned" if aligned else "unaligned"

        # variability within decel
        within = fhr[s:e]
        within = within[np.isfinite(within)]
        red_var = bool(len(within) and (np.percentile(within, 90) - np.percentile(within, 10) < 5))

        decels.append(Deceleration(
            start_min=round(start_min, 2),
            duration_min=round(dur_min, 2),
            depth_bpm=round(depth, 1),
            nadir_min=round(nadir_min, 2),
            morphology=morphology,
            timing=timing,
            aligned_to_contraction=aligned,
            reduced_var_within=red_var,
            prolonged=prolonged,
            prolonged_severe=prolonged_severe,
            severe_depth=severe_depth,
        ))
    return decels


def detect_contractions(toco: np.ndarray | None, hz: float):
    if toco is None:
        return [], None, False
    track = _smooth(toco, hz, 15.0)
    finite = track[np.isfinite(track)]
    if len(finite) < hz * 60:
        return [], None, False
    thr = np.nanpercentile(track, 50) + 0.5 * (np.nanpercentile(track, 90) - np.nanpercentile(track, 50))
    peaks = track >= thr
    contractions: list[Contraction] = []
    for s, e in _bool_runs(peaks):
        dur_min = (e - s) / hz / 60.0
        if dur_min * 60 < 20:
            continue
        contractions.append(Contraction(
            start_min=round(s / hz / 60.0, 2),
            duration_min=round(dur_min, 2),
            peak=round(float(np.nanmax(track[s:e])), 1),
        ))
    total_min = len(toco) / hz / 60.0
    per10 = round(len(contractions) / max(total_min, 1e-6) * 10, 1) if total_min else None
    # tachysystole: >5 in any rolling 10-min window
    tachy = _rolling_tachysystole(contractions, total_min)
    return contractions, per10, tachy


def detect_sinusoidal(fhr: np.ndarray, hz: float, baseline: float | None) -> bool:
    """Crude sinusoidal detector: regular ~3-5 cycles/min oscillation,
    amplitude 5-15 bpm, sustained. Flagged conservatively for review."""
    if baseline is None:
        return False
    track = _smooth(fhr, hz, 5.0)
    track = track[np.isfinite(track)]
    if len(track) < hz * 60 * 10:        # need >= 10 min
        return False
    centered = track - np.mean(track)
    # FFT to look for dominant 0.05-0.0833 Hz (3-5 cpm) component
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / hz)
    mag = np.abs(np.fft.rfft(centered))
    band = (freqs >= 0.05) & (freqs <= 0.0833)
    if not band.any():
        return False
    dominant = mag[band].max() / (mag[1:].mean() + 1e-9)
    amp = np.percentile(centered, 90) - np.percentile(centered, 10)
    # true sinusoidal has a very regular, narrow-band oscillation that
    # dominates the whole spectrum; ordinary variability does not.
    total_power = np.sum(mag[1:] ** 2)
    band_power = np.sum(mag[band] ** 2)
    band_ratio = band_power / (total_power + 1e-9)
    return bool(dominant > 25 and band_ratio > 0.5 and 5 <= amp <= 30)


# ---- helpers ----
def _bool_runs(mask: np.ndarray):
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _segments_to_events(mask, hz, fhr, baseline, kind, min_s):
    events = []
    for s, e in _bool_runs(mask):
        dur_min = (e - s) / hz / 60.0
        if dur_min * 60 < min_s:
            continue
        seg = fhr[s:e]
        seg = seg[np.isfinite(seg)]
        if not len(seg):
            continue
        amp = float(np.nanmax(seg) - baseline) if kind == "accel" else float(baseline - np.nanmin(seg))
        events.append(Acceleration(round(s / hz / 60.0, 2), round(dur_min, 2), round(amp, 1)))
    return events


def _classify_decel_timing(start_min, nadir_min, dur_min, contractions):
    """Return (aligned_bool, dtype). 'late' if nadir lags contraction peak;
    'variable' if abrupt and not aligned; 'early' if mirrors contraction."""
    nearest = min(contractions, key=lambda c: abs(c.start_min + c.duration_min / 2 - nadir_min))
    peak_min = nearest.start_min + nearest.duration_min / 2
    lag = nadir_min - peak_min
    aligned = abs(lag) < 1.5
    if aligned and lag > 0.25:
        return True, "late"
    if aligned and abs(lag) <= 0.25:
        return True, "early"
    return False, "variable"


def _rolling_tachysystole(contractions, total_min):
    if not contractions or total_min < 10:
        return False
    starts = np.array([c.start_min for c in contractions])
    for t in np.arange(0, max(total_min - 10, 0) + 0.1, 1.0):
        if np.sum((starts >= t) & (starts < t + 10)) > 5:
            return True
    return False


def extract_features(clean: Clean) -> Features:
    hz = clean.hz
    fhr = clean.fhr
    baseline, slope = estimate_baseline(fhr, hz)
    variability, low_var_min = estimate_variability(fhr, hz)

    # TOCO is only trustworthy for decel timing if it passed the quality gate.
    # A present-but-degraded toco must NOT be used to type late/variable decels.
    toco_ok = "toco" in clean.quality.accepted_channels
    toco_present = clean.toco is not None
    # Contractions can still be counted from a present channel (used only for
    # tachysystole burden), but timing classification requires toco_ok.
    contractions, per10, tachy = detect_contractions(clean.toco, hz) if toco_present else ([], None, False)
    # If tachysystole was derived from a present-but-quality-rejected toco, the
    # contraction count is unreliable: flag it lower-confidence so the alert
    # layer treats it as supporting-only, not a firm finding.
    tachy_low_conf = bool(tachy and toco_present and not toco_ok)
    accels = detect_accelerations(fhr, hz, baseline)
    decels = detect_decelerations(fhr, hz, baseline, contractions, toco_ok)
    sinusoidal = detect_sinusoidal(fhr, hz, baseline)

    # derived deceleration summaries. Use morphology for shape and the explicit
    # duration flags for prolonged/acute, so a long variable-shaped dip still
    # counts as prolonged.
    n_late = sum(1 for d in decels if d.morphology == "late")
    n_recurrent_late = n_late if (len(contractions) and n_late >= max(1, int(0.5 * len(contractions)))) else 0
    n_complicated_var = sum(
        1 for d in decels
        if d.morphology == "variable" and (d.severe_depth or d.duration_min >= 1 or d.reduced_var_within)
    )
    has_prolonged_any = any(d.prolonged for d in decels)            # >= 2 min
    has_prolonged = any(d.prolonged_severe for d in decels)          # >= 5 min
    has_acute3 = any(d.duration_min * 60 >= 180 for d in decels)     # >= 3 min
    quick_recovery = not any(d.duration_min * 60 >= 180 and d.reduced_var_within for d in decels)
    max_decel = max((d.duration_min for d in decels), default=0.0)

    f = Features(
        baseline_bpm=baseline,
        baseline_slope_bpm_per_min=slope,
        variability_bpm=variability,
        variability_low_min=low_var_min,
        accelerations=accels,
        decelerations=decels,
        contractions=contractions,
        contractions_per_10min=per10,
        tachysystole=tachy,
        tachysystole_low_confidence=tachy_low_conf,
        sinusoidal=sinusoidal,
        has_prolonged_any=has_prolonged_any,
        has_prolonged_gt5=has_prolonged,
        has_acute_event_ge3=has_acute3,
        max_decel_duration_min=round(max_decel, 2),
        quick_recovery=quick_recovery,
        n_recurrent_late=n_recurrent_late,
        n_complicated_variable=n_complicated_var,
        toco_available=toco_ok,
    )
    return f
