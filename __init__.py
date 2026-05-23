"""ctgkit — fetal heart-rate epoch analysis (decision support only).

This library accepts a ~30-minute CTG/EFM signal, validates it, extracts
guideline-aligned features, assigns a guideline category, computes an
*orthogonal* alert level, and returns a structured list of concerns.

IMPORTANT: This is decision-support software. It does not diagnose, treat,
or replace clinical judgement. Outputs must be reviewed by a qualified
clinician. See README / spec for regulatory framing.
"""

from .models import (
    AlertLevel,
    Category,
    Concern,
    EpochResult,
    Trend,
)
from .io import load_csv, Signal, from_arrays
from .pipeline import analyze
from .guidelines import GUIDELINE_PACKS, list_packs
from .service_config import analyze_service, ServiceConfig, RECOMMENDED
from .version import __version__


def plot(*args, **kwargs):
    """Lazy wrapper so matplotlib stays an optional dependency.

    Accepts a Signal or a CSV path. Safe to call repeatedly.
    See ctgkit.plot.plot_epoch for full signature.
    """
    from .plotting import plot_epoch
    return plot_epoch(*args, **kwargs)


__all__ = [
    "analyze",
    "analyze_service",
    "ServiceConfig",
    "RECOMMENDED",
    "plot",
    "load_csv",
    "from_arrays",
    "Signal",
    "EpochResult",
    "Concern",
    "Category",
    "AlertLevel",
    "Trend",
    "GUIDELINE_PACKS",
    "list_packs",
]

