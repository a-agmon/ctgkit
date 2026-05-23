"""Core data models for ctgkit.

These are deliberately plain dataclasses/enums so results are easy to
serialize (JSON), audit, and version. Every analysis output is fully
reproducible from (input signal + guideline pack version + library version).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Category(int, Enum):
    """Canonical 3-tier category, mapped from the chosen guideline pack.

    1 -> reassuring  (ACOG I / FIGO normal / NICE normal / SOGC normal)
    2 -> indeterminate (ACOG II / FIGO suspicious / NICE suspicious / SOGC atypical)
    3 -> abnormal    (ACOG III / FIGO pathological / NICE pathological / SOGC abnormal)
    """
    REASSURING = 1
    INDETERMINATE = 2
    ABNORMAL = 3


class AlertLevel(str, Enum):
    """Alert is ORTHOGONAL to category. Category is guideline-faithful;
    alert answers 'is escalation justified now, given persistence/trend/quality
    /risk stacking?'. This split is the core anti-alert-fatigue design."""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


class Trend(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class Concern:
    """A single structured, human-reviewable finding.

    Designed so a clinician can independently review the basis of an alert
    (definition met, when it started, how long, supporting channels, trend).
    """
    label: str                       # machine key, e.g. "recurrent_late_decels"
    title: str                       # human-readable
    severity: Severity
    detail: str                      # one-line explanation
    start_min: Optional[float] = None
    duration_min: Optional[float] = None
    trend: Trend = Trend.UNKNOWN
    supporting_channels: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)  # numbers behind it

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["trend"] = self.trend.value
        return d


@dataclass
class QualityReport:
    accepted_channels: list[str]
    usable_fraction: dict[str, float]      # per channel, AFTER cleaning (0..1)
    raw_usable_fraction: dict[str, float]  # per channel, BEFORE interpolation (0..1)
    interpolated_fraction: dict[str, float]  # fraction of samples filled by interp
    confidence: str                        # "high" | "medium" | "low"
    low_confidence: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSummary:
    baseline_bpm: Optional[float]
    baseline_slope_bpm_per_min: Optional[float]
    variability_bpm: Optional[float]
    n_accelerations: int
    n_decelerations: int
    decel_types: dict[str, int]
    contractions_per_10min: Optional[float]
    tachysystole: bool
    sinusoidal: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochResult:
    """The single object returned by analyze()."""
    category: Optional[Category]
    alert: AlertLevel
    concerns: list[Concern]
    confidence: str
    trend: Trend
    features: Optional[FeatureSummary]
    quality: QualityReport
    guideline_pack: str
    pack_version: str
    library_version: str
    epoch_minutes: float
    sampling_hz: float
    alert_score: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": int(self.category) if self.category is not None else None,
            "alert": self.alert.value,
            "concerns": [c.to_dict() for c in self.concerns],
            "confidence": self.confidence,
            "trend": self.trend.value,
            "features": self.features.to_dict() if self.features else None,
            "quality": self.quality.to_dict(),
            "guideline_pack": self.guideline_pack,
            "pack_version": self.pack_version,
            "library_version": self.library_version,
            "epoch_minutes": self.epoch_minutes,
            "sampling_hz": self.sampling_hz,
            "alert_score": self.alert_score,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        cat = f"Category {int(self.category)}" if self.category else "Category unknown"
        lines = [
            f"{cat}  |  ALERT: {self.alert.value.upper()}  |  confidence: {self.confidence}",
            f"trend: {self.trend.value}  |  pack: {self.guideline_pack} v{self.pack_version}",
        ]
        if self.concerns:
            lines.append("concerns:")
            for c in self.concerns:
                dur = f" ({c.duration_min:.0f} min)" if c.duration_min else ""
                lines.append(f"  - [{c.severity.value}] {c.title}{dur}: {c.detail}")
        else:
            lines.append("concerns: none")
        if self.warnings:
            lines.append("warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)
