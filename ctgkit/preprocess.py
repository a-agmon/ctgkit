"""Preprocessing and signal-quality gate.

Design rule (from the spec): poor signal quality may RAISE caution but must
NEVER reassure. The quality gate decides which channels are usable; the
pipeline never lets the rules engine output 'none' on inadequate FHR.

Preprocessing sequence:
    1. range filtering (drop physiologically implausible FHR)
    2. spike suppression
    3. short-gap interpolation (long gaps stay missing)
    4. toco flatline-artifact detection
    5. rolling signal-quality scoring
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .io import Signal
from .models import QualityReport

FHR_MIN, FHR_MAX = 50.0, 220.0          # implausible outside this -> artifact
MAX_GAP_INTERP_S = 4.0                   # interpolate gaps up to this length

# Confidence/usability is judged on the RAW (pre-interpolation) usable fraction.
# Interpolation can cosmetically fill scattered dropout to ~100% usable; trusting
# that would falsely upgrade confidence on a poor trace. So raw quality caps it.
USABLE_THRESHOLD = 0.80                  # FHR channel accepted only if raw >= this
HIGH_CONF = 0.95                         # raw usable >= 0.95 -> high
MED_CONF = 0.80                          # raw usable >= 0.80 -> medium (else low, no category)
# If too much of the *accepted* signal had to be interpolated, downgrade a notch.
MAX_INTERP_FRACTION = 0.20
# Coverage salvage: when the whole epoch fails the raw-usability gate, analyze
# the longest contiguous run of usable minutes if it is at least this long.
# This rescues traces with a localized dropout (e.g. second-stage sensor loss)
# without trusting diffusely scattered loss, which fails the per-minute test
# uniformly and is still rejected.
ANALYZE_MIN_WINDOW_MIN = 10.0


@dataclass
class Clean:
    fhr: np.ndarray              # cleaned, with NaN for unusable samples
    toco: np.ndarray | None
    mhr: np.ndarray | None
    fhr2: np.ndarray | None
    quality: QualityReport
    hz: float


def _range_filter(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = x.copy()
    out[(out < lo) | (out > hi)] = np.nan
    out[out == 0] = np.nan          # 0 = signal loss for HR channels
    return out


def _suppress_spikes(x: np.ndarray, hz: float, max_jump_bpm: float = 25.0) -> np.ndarray:
    """Remove single-sample jumps larger than physiologically plausible
    beat-to-beat change at this resolution."""
    out = x.copy()
    finite = np.isfinite(out)
    idx = np.where(finite)[0]
    for j in range(1, len(idx) - 1):
        i = idx[j]
        prev, nxt = out[idx[j - 1]], out[idx[j + 1]]
        if abs(out[i] - prev) > max_jump_bpm and abs(out[i] - nxt) > max_jump_bpm:
            out[i] = np.nan
    return out


def _interp_short_gaps(x: np.ndarray, hz: float, max_gap_s: float) -> np.ndarray:
    out = x.copy()
    n = len(out)
    max_gap = int(round(max_gap_s * hz))
    finite = np.isfinite(out)
    i = 0
    while i < n:
        if not finite[i]:
            j = i
            while j < n and not finite[j]:
                j += 1
            gap = j - i
            if 0 < i and j < n and gap <= max_gap:
                out[i:j] = np.linspace(out[i - 1], out[j], gap + 2)[1:-1]
            i = j
        else:
            i += 1
    return out


def _toco_flatline(toco: np.ndarray, hz: float,
                   win_s: float = 60.0, std_thresh: float = 1.0) -> np.ndarray:
    """Detect flat non-zero toco segments (sensor displacement artifact).
    Returns boolean mask of artifact samples."""
    n = len(toco)
    w = max(1, int(win_s * hz))
    mask = np.zeros(n, dtype=bool)
    for s in range(0, n, w):
        seg = toco[s:s + w]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= w // 2 and np.std(seg) < std_thresh and np.mean(seg) > 1.0:
            mask[s:s + w] = True
    return mask


def _usable_fraction(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.isfinite(x)))


def _best_usable_window(fhr_raw: np.ndarray, hz: float,
                        min_min: float, thresh: float):
    """Longest contiguous run of 'good' minutes (per-minute raw usable fraction
    >= thresh). Returns (start, end) sample indices, or None if no run reaches
    `min_min` minutes. A localized dropout leaves a long good run (salvageable);
    diffuse scattered loss makes every minute fall below `thresh`, so no window
    qualifies and the trace is rejected as before."""
    spm = int(hz * 60)
    if spm <= 0:
        return None
    n_min = len(fhr_raw) // spm
    if n_min == 0:
        return None
    good = [_usable_fraction(fhr_raw[m * spm:(m + 1) * spm]) >= thresh
            for m in range(n_min)]
    best_s = best_len = 0
    i = 0
    while i < n_min:
        if good[i]:
            j = i
            while j < n_min and good[j]:
                j += 1
            if j - i > best_len:
                best_len, best_s = j - i, i
            i = j
        else:
            i += 1
    if best_len < int(min_min):
        return None
    return best_s * spm, (best_s + best_len) * spm


def preprocess(sig: Signal) -> Clean:
    notes: list[str] = []
    hz = sig.hz

    # --- coverage salvage: if the whole epoch fails the raw-usability gate,
    # restrict analysis to the longest contiguous usable window (if any). All
    # channels are sliced to the same window so timing stays aligned. ---
    fhr_raw_full = _suppress_spikes(_range_filter(sig.fhr, FHR_MIN, FHR_MAX), hz)
    a, b = 0, len(sig.fhr)
    if _usable_fraction(fhr_raw_full) < USABLE_THRESHOLD:
        win = _best_usable_window(fhr_raw_full, hz, ANALYZE_MIN_WINDOW_MIN, USABLE_THRESHOLD)
        if win is not None:
            a, b = win
            notes.append(
                f"analysis limited to best usable window {a/hz/60:.0f}-{b/hz/60:.0f} min "
                "(localized signal loss elsewhere)")
    in_fhr = sig.fhr[a:b]
    in_fhr2 = sig.fhr2[a:b] if sig.fhr2 is not None else None
    in_mhr = sig.mhr[a:b] if sig.mhr is not None else None
    in_toco = sig.toco[a:b] if sig.toco is not None else None

    # --- FHR cleaning, tracking RAW (pre-interpolation) usability ---
    fhr_raw = _suppress_spikes(_range_filter(in_fhr, FHR_MIN, FHR_MAX), hz)
    raw_usable = {"fhr": _usable_fraction(fhr_raw)}
    fhr = _interp_short_gaps(fhr_raw, hz, MAX_GAP_INTERP_S)
    # fraction of all samples that were filled in by interpolation
    interp_frac = {"fhr": float(np.mean(np.isfinite(fhr) & ~np.isfinite(fhr_raw)))}

    fhr2 = None
    if in_fhr2 is not None:
        fhr2_raw = _suppress_spikes(_range_filter(in_fhr2, FHR_MIN, FHR_MAX), hz)
        raw_usable["fhr2"] = _usable_fraction(fhr2_raw)
        fhr2 = _interp_short_gaps(fhr2_raw, hz, MAX_GAP_INTERP_S)
        interp_frac["fhr2"] = float(np.mean(np.isfinite(fhr2) & ~np.isfinite(fhr2_raw)))

    mhr = None
    if in_mhr is not None:
        mhr = _range_filter(in_mhr, FHR_MIN, FHR_MAX)
        raw_usable["mhr"] = _usable_fraction(mhr)
        interp_frac["mhr"] = 0.0

    toco = None
    if in_toco is not None:
        toco = in_toco.copy()
        flat = _toco_flatline(toco, hz)
        if flat.any():
            toco[flat] = np.nan
            notes.append(f"toco flatline artifact in {100*flat.mean():.0f}% of trace")
        raw_usable["toco"] = _usable_fraction(toco)
        interp_frac["toco"] = 0.0

    usable = {"fhr": _usable_fraction(fhr)}
    if toco is not None:
        usable["toco"] = _usable_fraction(toco)
    if mhr is not None:
        usable["mhr"] = _usable_fraction(mhr)
    if fhr2 is not None:
        usable["fhr2"] = _usable_fraction(fhr2)

    # maternal/fetal confusion check: if fhr tracks mhr closely, flag it
    if mhr is not None:
        both = np.isfinite(fhr) & np.isfinite(mhr)
        if both.sum() > hz * 60:
            close = np.mean(np.abs(fhr[both] - mhr[both]) < 5.0)
            if close > 0.5:
                notes.append("possible maternal/fetal signal confusion (FHR≈MHR)")

    # --- acceptance and confidence are judged on RAW usability ---
    fhr_raw_uf = raw_usable["fhr"]
    accepted = []
    for c, u in raw_usable.items():
        if c == "mhr":
            continue
        if u >= USABLE_THRESHOLD:
            accepted.append(c)

    if "fhr" in accepted:
        if fhr_raw_uf >= HIGH_CONF:
            conf = "high"
        elif fhr_raw_uf >= MED_CONF:
            conf = "medium"
        else:
            conf = "low"
        # heavy interpolation downgrades confidence one notch
        if interp_frac["fhr"] > MAX_INTERP_FRACTION:
            conf = {"high": "medium", "medium": "low", "low": "low"}[conf]
            notes.append(
                f"{interp_frac['fhr']*100:.0f}% of FHR was interpolated — confidence downgraded")
    else:
        conf = "low"

    low_conf = conf == "low"
    if low_conf:
        notes.append(
            f"raw FHR usable fraction {fhr_raw_uf*100:.0f}% — reduced confidence")

    quality = QualityReport(
        accepted_channels=accepted,
        usable_fraction=usable,
        raw_usable_fraction=raw_usable,
        interpolated_fraction=interp_frac,
        confidence=conf,
        low_confidence=low_conf,
        notes=notes,
    )
    return Clean(fhr=fhr, toco=toco, mhr=mhr, fhr2=fhr2, quality=quality, hz=hz)
