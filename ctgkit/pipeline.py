"""Pipeline orchestration: the single public analyze() entry point."""
from __future__ import annotations

from typing import Optional, Union

from .version import __version__
from .io import Signal, load_csv
from .preprocess import preprocess
from .features import extract_features
from .guidelines import get_pack, classify
from .alerts import score_and_alert
from .models import EpochResult, FeatureSummary


def analyze(
    signal: Union[Signal, str],
    guideline: str = "figo",
    metadata: Optional[dict] = None,
    hz: float = 4.0,
    strict_epoch: bool = False,
    epoch_tolerance_min: float = 2.0,
    previous: Optional[EpochResult] = None,
) -> EpochResult:
    """Analyze one ~30-minute CTG epoch.

    Parameters
    ----------
    signal : Signal | str
        A loaded Signal, or a path to a CSV file.
    guideline : str
        Pack name: 'figo' (default), 'nice', 'acog', or 'sogc'.
    metadata : dict, optional
        Clinical context for risk stacking, e.g. {'oxytocin': True,
        'meconium': True, 'fever': False, 'gestational_age_weeks': 39}.
    strict_epoch : bool
        If True, raise on duration outside tolerance; else warn.
    previous : EpochResult, optional
        Prior epoch result for trend computation.

    Returns
    -------
    EpochResult
    """
    meta = dict(metadata or {})
    if isinstance(signal, str):
        signal = load_csv(signal, hz=hz, meta=meta)
    if signal.meta:
        meta = {**signal.meta, **meta}

    warnings = signal.validate_epoch(
        strict=strict_epoch, tolerance_min=epoch_tolerance_min
    )

    pack = get_pack(guideline)
    clean = preprocess(signal)

    if "fhr" not in clean.quality.accepted_channels:
        alert, concerns, score, trend = score_and_alert(
            None, None, clean.quality, meta, previous
        )
        return EpochResult(
            category=None, alert=alert, concerns=concerns,
            confidence=clean.quality.confidence, trend=trend, features=None,
            quality=clean.quality, guideline_pack=pack.name,
            pack_version=pack.version, library_version=__version__,
            epoch_minutes=round(signal.duration_min, 2), sampling_hz=signal.hz,
            alert_score=score, warnings=warnings,
        )

    feats = extract_features(clean)
    category = classify(pack, feats)
    alert, concerns, score, trend = score_and_alert(
        category, feats, clean.quality, meta, previous
    )

    decel_types: dict[str, int] = {}
    for d in feats.decelerations:
        decel_types[d.dtype] = decel_types.get(d.dtype, 0) + 1

    fsum = FeatureSummary(
        baseline_bpm=feats.baseline_bpm,
        baseline_slope_bpm_per_min=feats.baseline_slope_bpm_per_min,
        variability_bpm=feats.variability_bpm,
        n_accelerations=len(feats.accelerations),
        n_decelerations=len(feats.decelerations),
        decel_types=decel_types,
        contractions_per_10min=feats.contractions_per_10min,
        tachysystole=feats.tachysystole,
        sinusoidal=feats.sinusoidal,
    )

    return EpochResult(
        category=category, alert=alert, concerns=concerns,
        confidence=clean.quality.confidence, trend=trend, features=fsum,
        quality=clean.quality, guideline_pack=pack.name,
        pack_version=pack.version, library_version=__version__,
        epoch_minutes=round(signal.duration_min, 2), sampling_hz=signal.hz,
        alert_score=score, warnings=warnings,
    )
