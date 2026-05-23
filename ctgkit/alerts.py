"""Alert layer — ORTHOGONAL to category.

The category answers 'what does the guideline call this trace?'.
The alert answers 'is escalation justified now?' using persistence, trend,
acute events, risk stacking, and signal quality. A score model keeps this
auditable; thresholds are starting targets to be tuned per site.

Risk stacking uses optional clinical metadata (oxytocin, meconium, fever,
prolonged ROM, etc.) which raises the escalation threshold downward.
"""
from __future__ import annotations

from .features import Features
from .guidelines import reassuring_compensation
from .models import AlertLevel, Category, Concern, Severity, Trend, QualityReport

# scoring constants (tunable, documented in spec)
WARNING_THRESHOLD = 40
CRITICAL_THRESHOLD = 100

_RISK_WEIGHTS = {
    "oxytocin": 12, "meconium": 10, "fever": 12, "sepsis": 18,
    "prolonged_rom": 8, "preeclampsia": 10, "diabetes": 6,
    "growth_restriction": 14, "prematurity": 8, "previous_cesarean": 8,
    "slow_progress": 6,
}


def derive_contextual_risk_factors(meta: dict) -> list[tuple[str, int, str]]:
    out = []
    for key, weight in _RISK_WEIGHTS.items():
        if meta.get(key):
            label = key.replace("_", " ")
            out.append((key, weight, f"contextual risk: {label}"))
    return out


def score_and_alert(
    category: Category | None,
    f: Features | None,
    quality: QualityReport,
    meta: dict,
    prev: "EpochResult | None" = None,
):
    concerns: list[Concern] = []
    score = 0.0

    # No usable FHR -> warning (uncertainty), never 'none'
    if category is None or f is None:
        concerns.append(Concern(
            "signal_quality_risk", "Insufficient fetal heart signal",
            Severity.MODERATE,
            "FHR signal quality too low to assign a category; review trace.",
            supporting_channels=quality.accepted_channels,
            evidence={"usable_fraction": quality.usable_fraction},
        ))
        return AlertLevel.WARNING, concerns, 50.0, Trend.UNKNOWN

    # base contribution from category
    if category == Category.ABNORMAL:
        score += 100
    elif category == Category.INDETERMINATE:
        score += 30

    # ---- acute events (can stand alone as critical) ----
    if f.has_prolonged_gt5:
        score += 100
        d = max((x for x in f.decelerations if x.prolonged_severe),
                key=lambda x: x.duration_min, default=None)
        concerns.append(Concern(
            "prolonged_deceleration", "Prolonged deceleration ≥ 5 min",
            Severity.HIGH, "Severe acute event; prompt review/escalation.",
            start_min=d.start_min if d else None,
            duration_min=d.duration_min if d else None,
            supporting_channels=["fhr"],
            evidence={"depth_bpm": d.depth_bpm if d else None,
                      "morphology": d.morphology if d else None},
        ))
    elif f.has_acute_event_ge3 and not f.quick_recovery:
        score += 70
        d = max((x for x in f.decelerations if x.duration_min * 60 >= 180),
                key=lambda x: x.duration_min, default=None)
        concerns.append(Concern(
            "prolonged_deceleration", "Deceleration ≥ 3 min without quick recovery",
            Severity.HIGH, "Acute event crossing escalation threshold.",
            start_min=d.start_min if d else None,
            duration_min=d.duration_min if d else None,
            supporting_channels=["fhr"],
            evidence={"morphology": d.morphology if d else None},
        ))
    elif f.has_prolonged_any:
        # any decel >= 2 min, even variable-shaped, is an escalation concern
        score += 40
        d = max((x for x in f.decelerations if x.prolonged),
                key=lambda x: x.duration_min, default=None)
        concerns.append(Concern(
            "prolonged_deceleration", "Prolonged deceleration ≥ 2 min",
            Severity.HIGH, "Prolonged dip detected by duration regardless of shape.",
            start_min=d.start_min if d else None,
            duration_min=d.duration_min if d else None,
            supporting_channels=["fhr"],
            evidence={"depth_bpm": d.depth_bpm if d else None,
                      "morphology": d.morphology if d else None},
        ))

    if f.sinusoidal:
        score += 100
        concerns.append(Concern(
            "sinusoidal_pattern", "Possible sinusoidal pattern",
            Severity.HIGH, "Flagged for urgent clinician confirmation.",
            supporting_channels=["fhr"],
        ))

    # ---- recurrent / morphology concerns ----
    if f.n_recurrent_late:
        score += 35
        concerns.append(Concern(
            "recurrent_late_decels", "Recurrent late decelerations",
            Severity.HIGH, "Strongest recurrent hypoxia pattern in most frameworks.",
            supporting_channels=["fhr", "toco"] if f.toco_available else ["fhr"],
            evidence={"n_late": f.n_recurrent_late},
        ))
    if f.n_complicated_variable:
        score += 20
        concerns.append(Concern(
            "complicated_variable_decels", "Complicated variable decelerations",
            Severity.MODERATE, "Variable decels with concerning characteristics.",
            supporting_channels=["fhr", "toco"] if f.toco_available else ["fhr"],
            evidence={"n": f.n_complicated_variable},
        ))

    # baseline
    bl = f.baseline_bpm
    if bl is not None:
        if bl > 160:
            sev = Severity.MODERATE if bl <= 180 else Severity.HIGH
            score += 12 if bl <= 180 else 25
            concerns.append(Concern(
                "persistent_tachycardia", "Baseline tachycardia",
                sev, f"Baseline {bl:.0f} bpm.", supporting_channels=["fhr"],
                evidence={"baseline_bpm": bl}))
        elif bl < 110:
            sev = Severity.MODERATE if bl >= 100 else Severity.HIGH
            score += 12 if bl >= 100 else 30
            concerns.append(Concern(
                "low_baseline", "Low baseline",
                sev, f"Baseline {bl:.0f} bpm.", supporting_channels=["fhr"],
                evidence={"baseline_bpm": bl}))

    # rising baseline trend
    if f.baseline_slope_bpm_per_min is not None and f.baseline_slope_bpm_per_min > 0.5:
        score += 8
        concerns.append(Concern(
            "rising_baseline", "Rising baseline",
            Severity.LOW, f"+{f.baseline_slope_bpm_per_min:.1f} bpm/min over epoch.",
            trend=Trend.WORSENING, supporting_channels=["fhr"]))

    # variability
    if f.variability_bpm is not None:
        if f.variability_low_min >= 50:
            score += 25
            concerns.append(Concern(
                "reduced_variability", "Reduced variability",
                Severity.HIGH, f"Variability <5 bpm for {f.variability_low_min:.0f} min.",
                duration_min=f.variability_low_min, supporting_channels=["fhr"]))
        elif f.variability_low_min >= 30:
            score += 12
            concerns.append(Concern(
                "reduced_variability", "Reduced variability",
                Severity.MODERATE, f"Variability <5 bpm for {f.variability_low_min:.0f} min.",
                duration_min=f.variability_low_min, supporting_channels=["fhr"]))

    # contractions / tachysystole
    if f.tachysystole:
        if f.tachysystole_low_confidence:
            # toco present but quality-rejected: count is unreliable. Treat as a
            # low-severity, supporting-only signal and do not let it move the score.
            concerns.append(Concern(
                "tachysystole", "Possible tachysystole (low-confidence toco)",
                Severity.LOW,
                ">5 contractions per 10 min, but toco quality is poor — "
                "count unreliable, not used for scoring.",
                supporting_channels=["toco"],
                evidence={"per_10min": f.contractions_per_10min,
                          "toco_quality": "rejected"}))
        else:
            score += 10
            concerns.append(Concern(
                "tachysystole", "Tachysystole",
                Severity.MODERATE, ">5 contractions per 10 min.",
                supporting_channels=["toco"],
                evidence={"per_10min": f.contractions_per_10min}))

    # ---- trend vs previous epoch ----
    trend = Trend.UNKNOWN
    if prev is not None and prev.category is not None:
        if int(category) > int(prev.category):
            trend = Trend.WORSENING
            score += 20
        elif int(category) < int(prev.category):
            trend = Trend.IMPROVING
            score -= 10
        else:
            trend = Trend.STABLE

    # ---- risk stacking ----
    for key, weight, label in derive_contextual_risk_factors(meta):
        score += weight
        concerns.append(Concern(
            "contextual_clinical_risk", label.replace("contextual risk: ", "").title(),
            Severity.INFO, label, supporting_channels=[],
            evidence={"weight": weight}))

    # ---- signal quality: only raises caution ----
    if quality.low_confidence:
        score += 10
        concerns.append(Concern(
            "signal_quality_risk", "Reduced signal confidence",
            Severity.LOW, "Lower usable-signal fraction; interpret with caution.",
            supporting_channels=quality.accepted_channels,
            evidence={"usable_fraction": quality.usable_fraction}))

    # ---- protective features (auditability only, no score effect) ----
    # Surface the 'are variability/accelerations preserved?' answer explicitly so
    # a reviewer can see why a decel-morphology concern was not treated as the
    # abnormal extreme. The category modifier (guidelines.classify) is what
    # actually keeps such a trace out of the critical band.
    if reassuring_compensation(f):
        concerns.append(Concern(
            "protective_features", "Preserved variability and accelerations",
            Severity.INFO,
            "Accelerations with moderate variability — strongest bedside "
            "evidence against current fetal acidosis.",
            supporting_channels=["fhr"],
            evidence={"n_accelerations": len(f.accelerations),
                      "variability_bpm": f.variability_bpm}))

    score = max(score, 0.0)
    if score >= CRITICAL_THRESHOLD:
        alert = AlertLevel.CRITICAL
    elif score >= WARNING_THRESHOLD:
        alert = AlertLevel.WARNING
    else:
        alert = AlertLevel.NONE

    # safety floor: never 'none' on an abnormal category
    if category == Category.ABNORMAL and alert == AlertLevel.NONE:
        alert = AlertLevel.WARNING

    # rank concerns by severity
    order = {Severity.HIGH: 0, Severity.MODERATE: 1, Severity.LOW: 2, Severity.INFO: 3}
    concerns.sort(key=lambda c: order[c.severity])
    return alert, concerns, score, trend
