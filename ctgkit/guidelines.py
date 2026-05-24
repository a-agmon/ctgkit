"""Guideline packs — versioned, pluggable, deterministic category logic.

Each pack maps extracted Features -> canonical Category (1/2/3) using the
thresholds summarized in the design doc. Packs are intentionally faithful to
their guideline family; they do NOT try to harmonize severity distribution.
The alert layer (alerts.py) is where anti-fatigue logic lives.

Thresholds reflect FIGO (30-min review; duration-qualified variability/decel
rules), NICE (white/amber/red feature thresholds), ACOG/NICHD (Category I/II/III),
and SOGC (duration-based normal/atypical/abnormal). Where guidelines disagree
on borderline baselines (100-110 bpm), the conservative default never returns
Category 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .features import Features
from .models import Category


@dataclass
class GuidelinePack:
    name: str
    version: str
    classify: Callable[[Features], Category]
    feature_labels: dict           # canonical -> guideline-specific label


def _baseline_band(bl: float | None):
    """Return 'normal' | 'borderline' | 'abnormal' for baseline level."""
    if bl is None:
        return "unknown"
    if 110 <= bl <= 160:
        return "normal"
    if (100 <= bl < 110) or (160 < bl <= 180):
        return "borderline"
    return "abnormal"          # <100 or >180


def _common_cat3(f: Features) -> bool:
    """Patterns nearly all packs treat as the abnormal extreme."""
    if f.sinusoidal:
        return True
    if f.has_prolonged_gt5:                 # any decel >= 5 min, regardless of shape
        return True
    if f.has_acute_event_ge3 and not f.quick_recovery:   # >=3 min, poor recovery
        return True
    # absent/very low variability with recurrent decels
    if (f.variability_bpm is not None and f.variability_bpm < 3
            and (f.n_recurrent_late or f.n_complicated_variable)):
        return True
    bl = f.baseline_bpm
    if bl is not None and (bl < 100 or bl > 180) and (f.n_recurrent_late or f.has_acute_event_ge3):
        return True
    return False


def _moderate_variability(f: Features) -> bool:
    return (f.variability_bpm is not None
            and 6 <= f.variability_bpm <= 25
            and f.variability_low_min < 30)


def reassuring_compensation(f: Features) -> bool:
    """Preserved autonomic reserve.

    Accelerations together with sustained moderate variability are the
    strongest bedside evidence that the fetus is NOT acidotic *now*. This is
    the clinical modifier that separates 'recurrent late decels in a
    compensating fetus' (suspicious) from 'recurrent late decels on a flat
    trace' (pathological). It never overrides a hard pathological feature —
    those are handled by `_common_cat3`.

    Requires >=2 accelerations: a single transient is not reactivity, and a
    lone (possibly spurious) acceleration must not be enough to stand down a
    recurrent-decel pattern.
    """
    return len(f.accelerations) >= 2 and _moderate_variability(f)


def classify(pack: GuidelinePack, f: Features) -> Category:
    """Run the pack's guideline-faithful rule, then apply ONE shared clinical
    modifier.

    A category that is abnormal *only* on deceleration morphology — i.e. it
    does not trip `_common_cat3`'s hard pathological features (sinusoidal,
    prolonged >=5 min, >=3 min with poor recovery, variability <3 with decels,
    extreme baseline with decels) — is downgraded to indeterminate when the
    fetus shows reassuring compensation. This encodes the 'are variability and
    accelerations preserved?' question that all four guideline families weigh
    but the raw per-feature decel rules omit.
    """
    cat = pack.classify(f)
    if (cat == Category.ABNORMAL
            and not _common_cat3(f)
            and reassuring_compensation(f)):
        return Category.INDETERMINATE
    return cat


# ---------- FIGO ----------
def _figo(f: Features) -> Category:
    if _common_cat3(f):
        return Category.ABNORMAL
    bl = _baseline_band(f.baseline_bpm)
    var = f.variability_bpm
    # pathological: <100 baseline, reduced var >50 min, repetitive late/prolonged
    if bl == "abnormal" or f.n_recurrent_late or f.n_complicated_variable >= 1:
        return Category.ABNORMAL if (f.n_recurrent_late or f.has_acute_event_ge3) else Category.INDETERMINATE
    # suspicious: lacks at least one normal characteristic
    suspicious = (
        bl == "borderline"
        or (var is not None and (var < 5 and f.variability_low_min >= 30))
        or len([d for d in f.decelerations if d.morphology in ("late", "variable")]) > 0
        or f.has_prolonged_any
        or f.tachysystole
    )
    if suspicious:
        return Category.INDETERMINATE
    return Category.REASSURING


# ---------- NICE ----------
def _nice(f: Features) -> Category:
    """White/amber/red per feature, then combine: pathological if any red or
    >=2 amber; suspicious if exactly one amber."""
    if _common_cat3(f):
        return Category.ABNORMAL
    reds = ambers = 0
    bl = f.baseline_bpm
    if bl is not None:
        if bl < 100 or bl > 180:
            reds += 1
        elif (100 <= bl < 110) or (160 < bl <= 180):
            ambers += 1
    var = f.variability_bpm
    if var is not None:
        if f.variability_low_min > 50 or var > 25:
            reds += 1
        elif f.variability_low_min >= 30:
            ambers += 1
    # decelerations
    if f.n_recurrent_late or any(d.morphology == "late" for d in f.decelerations):
        if f.n_recurrent_late:
            reds += 1
        else:
            ambers += 1
    if f.n_complicated_variable >= 1:
        ambers += 1
    if f.tachysystole:
        ambers += 1

    if reds >= 1 or ambers >= 2:
        return Category.ABNORMAL if reds >= 1 else Category.INDETERMINATE
    if ambers == 1:
        return Category.INDETERMINATE
    return Category.REASSURING


# ---------- ACOG / NICHD ----------
def _acog(f: Features) -> Category:
    """Category I requires ALL of: baseline 110-160, moderate variability,
    no late/variable decels (accels optional). Category III is the abnormal
    extreme. Everything else is Category II (deliberately broad)."""
    if _common_cat3(f):
        return Category.ABNORMAL
    bl = f.baseline_bpm
    var = f.variability_bpm
    cat1 = (
        bl is not None and 110 <= bl <= 160
        and var is not None and 6 <= var <= 25
        and not any(d.morphology in ("late", "variable") for d in f.decelerations)
        and not f.has_acute_event_ge3
        and not f.has_prolonged_any
    )
    if cat1:
        return Category.REASSURING
    return Category.INDETERMINATE


# ---------- SOGC ----------
def _sogc(f: Features) -> Category:
    if _common_cat3(f):
        return Category.ABNORMAL
    bl = f.baseline_bpm
    var = f.variability_bpm
    abnormal = (
        (bl is not None and (bl < 100 or bl > 160) and f.variability_low_min > 80)
        or f.n_recurrent_late
        or any(d.prolonged and 3 <= d.duration_min < 10 for d in f.decelerations)
        or (var is not None and f.variability_low_min > 80)
    )
    if abnormal:
        return Category.ABNORMAL
    atypical = (
        (bl is not None and ((100 <= bl < 110) or bl > 160))
        or (var is not None and (f.variability_low_min < 80 and f.variability_low_min >= 1) )
        or any(d.morphology in ("late", "variable") for d in f.decelerations)
        or any(d.prolonged and 2 <= d.duration_min < 3 for d in f.decelerations)
        or f.tachysystole
    )
    if atypical:
        return Category.INDETERMINATE
    return Category.REASSURING


GUIDELINE_PACKS: dict[str, GuidelinePack] = {
    "figo": GuidelinePack("figo", "2015", _figo,
                          {"category": "normal/suspicious/pathological"}),
    "nice": GuidelinePack("nice", "2022", _nice,
                          {"category": "normal/suspicious/pathological"}),
    "acog": GuidelinePack("acog", "nichd-2008", _acog,
                          {"category": "I/II/III"}),
    "sogc": GuidelinePack("sogc", "2020", _sogc,
                          {"category": "normal/atypical/abnormal"}),
}


def list_packs() -> list[str]:
    return list(GUIDELINE_PACKS.keys())


def get_pack(name: str) -> GuidelinePack:
    key = name.strip().lower()
    if key not in GUIDELINE_PACKS:
        raise KeyError(f"unknown guideline pack '{name}'. Available: {list_packs()}")
    return GUIDELINE_PACKS[key]
