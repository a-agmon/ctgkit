"""Recommended service configuration for deployment.

The library's `analyze()` defaults are tuned for *exploration* (lenient epoch
handling, FIGO default). A running service should be stricter and explicit.
This module documents and packages the recommended production posture so a
deployment has one safe, auditable entry point instead of scattering config.

Use `RECOMMENDED` as a starting point and override the guideline per site.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from .pipeline import analyze
from .io import Signal
from .models import EpochResult


# ---------------------------------------------------------------------------
# 30-minute epoch contract.
#
# The guideline families are written around a 30-minute formal review window
# (FIGO requires re-evaluation at least every 30 min). The service therefore
# treats 30 minutes as a hard contract, accepting only a small jitter band so
# that real-world export rounding does not reject otherwise-valid epochs.
#
#   expected:  30 min
#   tolerance: +/- 2 min  -> accepts 28..32 min
#   outside that band with strict_epoch=True -> SignalError (refuse to score)
#
# Rationale for refusing (rather than warning) in a service: a sub-30-min trace
# cannot satisfy the duration-based guideline timers (e.g. reduced variability
# for >50 min, prolonged patterns), so silently scoring it risks a falsely
# reassuring result. Better to reject and ask the caller to supply a full epoch.
# ---------------------------------------------------------------------------
EPOCH_EXPECTED_MIN = 30.0
EPOCH_TOLERANCE_MIN = 2.0


@dataclass(frozen=True)
class ServiceConfig:
    """Production posture. Frozen so a running service can log it verbatim."""
    guideline: str = "acog"            # explicit; override per site (see note)
    strict_epoch: bool = True          # refuse off-length epochs, don't warn
    epoch_tolerance_min: float = EPOCH_TOLERANCE_MIN
    hz: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


# The recommended default. `guideline` should be set to the jurisdiction's
# pack at deployment time. ACOG is used here as an explicit, conservative
# placeholder (its broad Category II keeps more traces under surveillance);
# UK/EU sites would set "nice" or "figo", Canadian sites "sogc".
RECOMMENDED = ServiceConfig()


def analyze_service(
    signal: Signal | str,
    config: ServiceConfig = RECOMMENDED,
    metadata: Optional[dict] = None,
    previous: Optional[EpochResult] = None,
) -> EpochResult:
    """Analyze using the recommended service posture.

    Equivalent to calling `analyze(...)` with strict_epoch=True, an explicit
    guideline, and the documented 30-min +/- 2 contract. Raises SignalError on
    an off-length epoch rather than scoring it.

    Example
    -------
    >>> from ctgkit.service_config import analyze_service, ServiceConfig
    >>> cfg = ServiceConfig(guideline="nice")          # site default
    >>> result = analyze_service("epoch.csv", cfg,
    ...                          metadata={"oxytocin": True})
    """
    return analyze(
        signal,
        guideline=config.guideline,
        metadata=metadata,
        hz=config.hz,
        strict_epoch=config.strict_epoch,
        epoch_tolerance_min=config.epoch_tolerance_min,
        previous=previous,
    )
