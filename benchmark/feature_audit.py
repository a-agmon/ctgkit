"""Feature signal audit: which individual detectors separate acidemic babies?

The composite alert score is near-chance on real data. Before calibrating
weights (which assumes the features carry signal) or reworking detectors (which
assumes they don't), measure each feature's *univariate* association with the
outcome. For every usable record we extract the full Features and compute the
rank-AUC of each feature against pH<=7.05 and the composite adverse label.

AUC interpretation (label = adverse):
  ~0.50         no information
  >0.50         higher feature value -> worse outcome
  <0.50         lower  feature value -> worse outcome (e.g. variability, accels)
  |AUC-0.50|    discrimination strength, regardless of direction

Run:  python benchmark/feature_audit.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctgkit.io import from_arrays            # noqa: E402
from ctgkit.preprocess import preprocess     # noqa: E402
from ctgkit.features import extract_features  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNALS = os.path.join(HERE, "data", "signals")
OUTCOMES = os.path.join(HERE, "data", "outcomes.csv")
HZ = 4.0
N_EPOCH = int(30 * 60 * HZ)


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def rank_auc(scores, labels):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    n1 = int(y.sum())
    n2 = len(s) - n1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def feature_vector(f) -> dict:
    return {
        "baseline_bpm": f.baseline_bpm,
        "baseline_slope": f.baseline_slope_bpm_per_min,
        "variability_bpm": f.variability_bpm,
        "variability_low_min": f.variability_low_min,
        "n_accelerations": len(f.accelerations),
        "n_decelerations": len(f.decelerations),
        "n_recurrent_late": f.n_recurrent_late,
        "n_complicated_variable": f.n_complicated_variable,
        "max_decel_min": f.max_decel_duration_min,
        "contractions_per_10min": f.contractions_per_10min,
        "tachysystole": int(f.tachysystole),
        "has_prolonged_any": int(f.has_prolonged_any),
        "sinusoidal": int(f.sinusoidal),
    }


def main():
    outcomes = {}
    with open(OUTCOMES) as fh:
        for row in csv.DictReader(fh):
            outcomes[row["record"]] = row
    records = sorted(r[:-4] for r in os.listdir(SIGNALS) if r.endswith(".csv"))

    feats: dict[str, list] = {}
    y_acid, y_comp = [], []
    n_used = 0
    for rec in records:
        if rec not in outcomes:
            continue
        fhr, uc = [], []
        with open(os.path.join(SIGNALS, f"{rec}.csv")) as fh:
            rd = csv.reader(fh); next(rd, None)
            for r in rd:
                if len(r) >= 3:
                    fhr.append(float(r[1])); uc.append(float(r[2]))
        fhr = np.asarray(fhr[-N_EPOCH:], float)
        uc = np.asarray(uc[-N_EPOCH:], float)
        clean = preprocess(from_arrays(fhr=fhr, hz=HZ, toco=uc))
        if "fhr" not in clean.quality.accepted_channels:
            continue
        f = extract_features(clean)
        fv = feature_vector(f)
        ph = _f(outcomes[rec].get("pH"))
        bdecf = _f(outcomes[rec].get("BDecf"))
        apg5 = _f(outcomes[rec].get("Apgar5"))
        morb = any((_f(outcomes[rec].get(k)) or 0) > 0 for k in ("Seizures", "HIE", "Intubation"))
        acid = ph is not None and ph <= 7.05
        comp = bool(acid or (bdecf is not None and bdecf >= 12)
                    or (apg5 is not None and apg5 < 7) or morb)
        for k, v in fv.items():
            feats.setdefault(k, []).append(v if v is not None else np.nan)
        y_acid.append(acid); y_comp.append(comp)
        n_used += 1

    print(f"usable records audited: {n_used}  "
          f"(pH<=7.05: {sum(y_acid)}  composite: {sum(y_comp)})\n")
    print(f"{'feature':24} {'AUC_pH':>7} {'AUC_comp':>9} "
          f"{'median_norm':>12} {'median_adv':>11}")
    print("-" * 70)

    rows = []
    for k, vals in feats.items():
        v = np.asarray(vals, float)
        ok = np.isfinite(v)
        auc_a = rank_auc(v[ok], np.asarray(y_acid)[ok])
        auc_c = rank_auc(v[ok], np.asarray(y_comp)[ok])
        adv = np.asarray(y_comp)[ok]
        med_norm = np.median(v[ok][~adv]) if (~adv).any() else float("nan")
        med_adv = np.median(v[ok][adv]) if adv.any() else float("nan")
        rows.append((k, auc_a, auc_c, med_norm, med_adv))

    rows.sort(key=lambda r: abs((r[2] or 0.5) - 0.5), reverse=True)
    for k, auc_a, auc_c, mn, ma in rows:
        sa = f"{auc_a:.3f}" if auc_a is not None else "  n/a"
        sc = f"{auc_c:.3f}" if auc_c is not None else "  n/a"
        print(f"{k:24} {sa:>7} {sc:>9} {mn:>12.2f} {ma:>11.2f}")
    print("\nrows sorted by |AUC_comp - 0.5| (discrimination strength).")


if __name__ == "__main__":
    main()
