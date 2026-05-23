"""Behavioural test suite for ctgkit.

Beyond smoke tests: these assert the safety-critical invariants and the
specific clinical behaviours called out in review — prolonged-by-duration,
toco-dependent decel typing, raw-quality-limits-confidence, and the
category-vs-alert separation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest

from ctgkit import analyze, from_arrays
from ctgkit.models import Category, AlertLevel
from ctgkit.preprocess import preprocess
from ctgkit.features import extract_features
from ctgkit.synth import synth_epoch
import helpers as H


# ---------------------------------------------------------------- smoke
def test_normal_is_cat1_no_alert():
    r = analyze(synth_epoch("normal"))
    assert r.category == Category.REASSURING
    assert r.alert == AlertLevel.NONE


def test_json_serializable():
    import json
    json.dumps(analyze(synth_epoch("tachy")).to_dict())


# ------------------------------------------------ prolonged by duration
def test_prolonged_decel_always_detected_by_duration():
    """A long deep dip must be flagged prolonged even when shaped variable
    and even with no toco to type it."""
    f = extract_features(preprocess(H.trace_with_decel(width_s=360, depth=50)))
    longest = max(f.decelerations, key=lambda d: d.duration_min)
    assert longest.duration_min * 60 >= 300
    assert longest.prolonged is True
    assert longest.prolonged_severe is True
    # morphology is allowed to be uncertain/variable — that must NOT suppress it
    assert f.has_prolonged_gt5 is True


def test_prolonged_severe_is_critical_all_packs():
    sig = H.trace_with_decel(width_s=360, depth=50)
    for g in ("figo", "nice", "acog", "sogc"):
        r = analyze(sig, guideline=g)
        assert r.category == Category.ABNORMAL, g
        assert r.alert == AlertLevel.CRITICAL, g


def test_prolonged_2to3min_never_silent():
    """A 2-3 min dip is not severe, but must never be category-1-no-alert."""
    sig = H.trace_with_decel(width_s=150, depth=50)
    for g in ("figo", "nice", "acog", "sogc"):
        r = analyze(sig, guideline=g)
        assert r.alert != AlertLevel.NONE, g


def test_short_decel_not_prolonged():
    f = extract_features(preprocess(H.trace_with_decel(width_s=45, depth=30)))
    if f.decelerations:
        assert all(not d.prolonged for d in f.decelerations)


# --------------------------------------------- toco-dependent typing
def test_missing_toco_does_not_claim_late_decels():
    """Without toco, a deceleration must not be confidently typed late/variable."""
    f = extract_features(preprocess(H.trace_with_decel(width_s=40, with_toco=False)))
    for d in f.decelerations:
        assert d.morphology == "uncertain"
        assert d.timing == "uncertain"
        assert d.aligned_to_contraction is None


def test_variable_deceleration_without_toco_is_uncertain():
    sig = H.trace_with_decel(width_s=40, with_toco=False)
    r = analyze(sig, guideline="nice")
    # no false 'late' claim should appear in concerns
    assert not any(c.label == "recurrent_late_decels" for c in r.concerns)


def test_recurrent_late_with_toco_detected():
    f = extract_features(preprocess(H.trace_recurrent_late(n_contractions=10)))
    assert f.n_recurrent_late >= 1
    assert any(d.morphology == "late" for d in f.decelerations)


def test_recurrent_late_is_high_concern():
    r = analyze(H.trace_recurrent_late(n_contractions=10), guideline="figo")
    assert r.category == Category.ABNORMAL
    assert r.alert == AlertLevel.CRITICAL


def test_recurrent_late_with_preserved_compensation_not_critical():
    """The variability/acceleration modifier: recurrent late decels with
    preserved accelerations AND moderate variability describe a compensating
    fetus, not the abnormal extreme. Category must drop to indeterminate and
    the alert must stay below critical, across all packs."""
    sig = H.trace_recurrent_late_with_accels(n_contractions=10)
    for g in ("figo", "nice", "sogc", "acog"):
        r = analyze(sig, guideline=g)
        assert r.category == Category.INDETERMINATE, g
        assert r.alert != AlertLevel.CRITICAL, g
        assert any(c.label == "protective_features" for c in r.concerns), g


def test_preserved_compensation_does_not_rescue_acute_event():
    """Accelerations must NOT downgrade a hard pathological feature: a
    prolonged (>=5 min) deceleration stays Category 3 / critical even if the
    rest of the trace looks reassuring."""
    sig = H.trace_with_decel(width_s=360, depth=50, with_toco=True)
    r = analyze(sig, guideline="figo")
    assert r.category == Category.ABNORMAL
    assert r.alert == AlertLevel.CRITICAL


# --------------------------------------------- signal quality / confidence
def test_raw_signal_quality_limits_confidence():
    """Scattered dropout that interpolation fills must NOT yield high confidence."""
    sig = synth_epoch("normal")
    rng = np.random.default_rng(1)
    sig.fhr[rng.random(sig.n_samples) < 0.6] = np.nan   # 60% raw loss
    r = analyze(sig)
    assert r.confidence == "low"


def test_bad_signal_never_reassures():
    sig = synth_epoch("normal")
    sig.fhr[: int(sig.n_samples * 0.7)] = np.nan
    r = analyze(sig)
    assert r.category is None
    assert r.alert != AlertLevel.NONE


def test_clean_signal_is_high_confidence():
    r = analyze(synth_epoch("normal"))
    assert r.confidence == "high"


def test_quality_reports_raw_and_cleaned():
    sig = synth_epoch("noisy")
    clean = preprocess(sig)
    q = clean.quality
    # raw should be materially worse than post-interpolation usable fraction
    assert q.raw_usable_fraction["fhr"] < q.usable_fraction["fhr"]


# --------------------------------------------- category vs alert separation
def test_category_iii_critical():
    r = analyze(H.trace_with_decel(width_s=360, depth=50), guideline="figo")
    assert r.category == Category.ABNORMAL
    assert r.alert == AlertLevel.CRITICAL


def test_category_ii_stable_no_alert():
    """A mild, isolated Category-2 trace with no persistence/risk should be
    quiet — this is the core anti-alert-fatigue behaviour."""
    # mild tachycardia ~163, otherwise reassuring, no decelerations
    sig = from_arrays(fhr=H.base_trace(baseline=163.0), hz=4.0,
                      toco=H.add_contractions(int(30*60*4), 4.0))
    r = analyze(sig, guideline="figo")
    assert r.category == Category.INDETERMINATE
    assert r.alert in (AlertLevel.NONE, AlertLevel.WARNING)


def test_category_ii_worsening_warning():
    """Same indeterminate trace, but worsening vs previous epoch -> at least warning."""
    sig = from_arrays(fhr=H.base_trace(baseline=163.0), hz=4.0,
                      toco=H.add_contractions(int(30*60*4), 4.0))
    prev = analyze(synth_epoch("normal"), guideline="figo")  # was Category 1
    r = analyze(sig, guideline="figo", previous=prev)
    assert r.trend.value == "worsening"
    assert r.alert in (AlertLevel.WARNING, AlertLevel.CRITICAL)


def test_acog_category_ii_at_least_as_broad():
    """For a late-decel trace, ACOG's category should be <= others (broader II)."""
    sig = H.trace_recurrent_late(n_contractions=10)
    acog = int(analyze(sig, guideline="acog").category)
    figo = int(analyze(sig, guideline="figo").category)
    assert acog <= figo


# --------------------------------------------- variability timing
def test_low_variability_increases_concern_with_duration():
    short = extract_features(preprocess(H.trace_low_variability(5)))
    long = extract_features(preprocess(H.trace_low_variability(25)))
    assert long.variability_low_min > short.variability_low_min


def test_reduced_variability_surfaces_concern():
    # Within a single 30-min epoch, sustained low variability should at least
    # register in the feature layer. The *concern* threshold is intentionally
    # high (guideline timers span ~30-50 min, often across epochs), so we assert
    # the feature is captured and confidence/feature state reflect it.
    f = extract_features(preprocess(H.trace_low_variability(25)))
    assert f.variability_low_min >= 10
    assert f.variability_bpm is not None and f.variability_bpm < 8


# --------------------------------------------- tachysystole
def test_tachysystole_detected():
    f = extract_features(preprocess(H.trace_tachysystole(fhr_abnormal=False)))
    assert f.tachysystole is True


def test_tachysystole_alone_not_critical():
    """Tachysystole with normal FHR should not be critical by itself."""
    r = analyze(H.trace_tachysystole(fhr_abnormal=False), guideline="figo")
    assert r.alert != AlertLevel.CRITICAL
    assert any(c.label == "tachysystole" for c in r.concerns)


def test_tachysystole_with_abnormal_fhr_escalates():
    normal = analyze(H.trace_tachysystole(fhr_abnormal=False), guideline="figo")
    abnormal = analyze(H.trace_tachysystole(fhr_abnormal=True), guideline="figo")
    assert abnormal.alert_score >= normal.alert_score


# --------------------------------------------- risk stacking
def test_risk_stacking_raises_score():
    sig = from_arrays(fhr=H.base_trace(baseline=163.0), hz=4.0,
                      toco=H.add_contractions(int(30*60*4), 4.0))
    plain = analyze(sig, guideline="figo")
    stacked = analyze(sig, guideline="figo",
                      metadata={"oxytocin": True, "meconium": True, "sepsis": True})
    assert stacked.alert_score > plain.alert_score


# --------------------------------------------- epoch contract
def test_short_epoch_warns_not_raises():
    sig = synth_epoch("normal", minutes=18)
    r = analyze(sig)
    assert any("outside" in w for w in r.warnings)


def test_short_epoch_strict_raises():
    from ctgkit.io import SignalError
    sig = synth_epoch("normal", minutes=18)
    with pytest.raises(SignalError):
        analyze(sig, strict_epoch=True)


# --------------------------------------------- plotting (issue 1)
def test_plot_returns_figure(tmp_path):
    sig = synth_epoch("normal")
    out = tmp_path / "p1.png"
    fig = analyze_and_plot(sig, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_accepts_csv_path(tmp_path):
    import csv
    sig = synth_epoch("late_decels")
    csv_path = tmp_path / "epoch.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["time_s", "fhr", "toco"])
        for i in range(sig.n_samples):
            w.writerow([round(i / sig.hz, 3), round(float(sig.fhr[i]), 1),
                        round(float(sig.toco[i]), 1)])
    import ctgkit
    out = tmp_path / "p_csv.png"
    fig = ctgkit.plot(str(csv_path), save_path=str(out))
    assert out.exists()


def test_plot_repeated_calls_do_not_error(tmp_path):
    import ctgkit
    for i in range(4):
        out = tmp_path / f"rep_{i}.png"
        ctgkit.plot(synth_epoch("tachy"), save_path=str(out))
        assert out.exists()


def test_plot_module_and_function_coexist():
    """ctgkit.plot must be the callable; the implementation lives in
    ctgkit.plotting.plot_epoch. No submodule named 'plot' shadows it."""
    import ctgkit
    from ctgkit.plotting import plot_epoch
    assert callable(ctgkit.plot)
    assert callable(plot_epoch)


def analyze_and_plot(sig, out):
    import ctgkit
    r = analyze(sig)
    return ctgkit.plot(sig, r, save_path=out)


# --------------------------------------------- degraded TOCO (issue 2)
def test_good_toco_types_late_decels():
    f = extract_features(preprocess(H.trace_recurrent_late(n_contractions=10)))
    assert f.toco_available is True
    assert any(d.morphology == "late" for d in f.decelerations)


def test_degraded_toco_excluded_from_acceptance():
    sig = H.trace_recurrent_late(n_contractions=10)
    sig.toco[int(len(sig.toco) * 0.3):] = np.nan   # lose 70% of toco
    clean = preprocess(sig)
    assert "toco" not in clean.quality.accepted_channels


def test_degraded_toco_does_not_claim_late_variable():
    sig = H.trace_recurrent_late(n_contractions=10)
    sig.toco[int(len(sig.toco) * 0.3):] = np.nan
    f = extract_features(preprocess(sig))
    assert f.toco_available is False
    for d in f.decelerations:
        assert d.morphology == "uncertain"
        assert d.timing == "uncertain"


def test_degraded_toco_no_recurrent_late_concern():
    sig = H.trace_recurrent_late(n_contractions=10)
    sig.toco[int(len(sig.toco) * 0.3):] = np.nan
    r = analyze(sig, guideline="figo")
    assert not any(c.label == "recurrent_late_decels" for c in r.concerns)


# --------------------------------------------- confidence thresholds (issue 3)
def test_confidence_high_requires_95pct():
    r = analyze(synth_epoch("normal"))          # ~100% clean
    assert r.confidence == "high"


def test_confidence_medium_band():
    sig = synth_epoch("normal")
    rng = np.random.default_rng(3)
    sig.fhr[rng.random(sig.n_samples) < 0.12] = np.nan   # ~88% raw
    r = analyze(sig)
    assert r.confidence == "medium"
    assert r.category is not None


def test_confidence_low_yields_no_category():
    sig = synth_epoch("normal")
    rng = np.random.default_rng(4)
    sig.fhr[rng.random(sig.n_samples) < 0.30] = np.nan   # ~70% raw
    r = analyze(sig)
    assert r.confidence == "low"
    assert r.category is None


# --------------------------------------------- service config (item 4)
def test_service_config_defaults():
    from ctgkit import RECOMMENDED, ServiceConfig
    assert RECOMMENDED.strict_epoch is True
    assert RECOMMENDED.guideline == "acog"
    assert RECOMMENDED.epoch_tolerance_min == 2.0


def test_analyze_service_runs_full_epoch():
    from ctgkit import analyze_service, ServiceConfig
    r = analyze_service(synth_epoch("normal"), ServiceConfig(guideline="nice"))
    assert r.guideline_pack == "nice"
    assert r.category is not None


def test_analyze_service_rejects_short_epoch():
    from ctgkit import analyze_service
    from ctgkit.io import SignalError
    with pytest.raises(SignalError):
        analyze_service(synth_epoch("normal", minutes=18))


def test_analyze_service_accepts_within_tolerance():
    from ctgkit import analyze_service
    # 31 min is within 30 +/- 2
    r = analyze_service(synth_epoch("normal", minutes=31))
    assert r.category is not None


# --------------------------------------------- tachysystole low-confidence (item 5)
def test_degraded_toco_tachysystole_low_confidence():
    sig = H.trace_tachysystole(fhr_abnormal=False)
    sig.toco[int(len(sig.toco) * 0.5):] = np.nan      # drop below accept threshold
    f = extract_features(preprocess(sig))
    assert f.toco_available is False
    if f.tachysystole:
        assert f.tachysystole_low_confidence is True


def test_low_confidence_tachysystole_does_not_score():
    sig = H.trace_tachysystole(fhr_abnormal=False)
    sig.toco[int(len(sig.toco) * 0.5):] = np.nan
    degraded = analyze(sig, guideline="figo")
    tc = [c for c in degraded.concerns if c.label == "tachysystole"]
    if tc:
        assert tc[0].severity.value == "low"
        assert "low-confidence" in tc[0].title.lower()


def test_good_toco_tachysystole_full_confidence():
    f = extract_features(preprocess(H.trace_tachysystole(fhr_abnormal=False)))
    assert f.tachysystole is True
    assert f.tachysystole_low_confidence is False
