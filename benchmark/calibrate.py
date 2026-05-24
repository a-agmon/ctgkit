"""Calibration ceiling diagnostic (read-only; nothing here is wired into ctgkit).

The hand-weighted additive alert score reaches ~0.61 AUC on CTU-UHB, yet the
audit showed single features separating better than that. This asks: with the
features we already extract, what *cross-validated* AUC could a simple fitted
model reach? That tells us whether re-weighting is worth doing and how many
features it can support given only ~21 acidaemic cases.

We deliberately keep models tiny (1-3 features) and L2-regularised, and report
stratified K-fold CV AUC (out-of-fold), so the numbers are honest rather than
in-sample. Pure numpy: no sklearn/scipy dependency.

Run:  python benchmark/calibrate.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctgkit import analyze                   # noqa: E402
from ctgkit.io import from_arrays            # noqa: E402
from ctgkit.preprocess import preprocess     # noqa: E402
from ctgkit.features import extract_features  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNALS = os.path.join(HERE, "data", "signals")
OUTCOMES = os.path.join(HERE, "data", "outcomes.csv")
HZ = 4.0
N_EPOCH = int(30 * 60 * HZ)
RNG = np.random.default_rng(0)


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def rank_auc(scores, labels):
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    n1 = int(y.sum())
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(s) - n1))


def fit_logistic(X, y, l2=1.0, iters=2000, lr=0.1):
    """Plain L2-regularised logistic regression via gradient descent."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b


def cv_auc(X, y, l2=1.0, k=5, repeats=20):
    """Stratified K-fold, repeated; returns out-of-fold AUC mean/std."""
    y = y.astype(float)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    aucs = []
    for r in range(repeats):
        rng = np.random.default_rng(r)
        pp, nn = rng.permutation(pos), rng.permutation(neg)
        oof = np.full(len(y), np.nan)
        for f in range(k):
            te = np.concatenate([pp[f::k], nn[f::k]])
            tr = np.setdiff1d(np.arange(len(y)), te)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            w, b = fit_logistic(Xtr, y[tr], l2=l2)
            oof[te] = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
        aucs.append(rank_auc(oof, y))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    outcomes = {}
    with open(OUTCOMES) as fh:
        for row in csv.DictReader(fh):
            outcomes[row["record"]] = row
    records = sorted(r[:-4] for r in os.listdir(SIGNALS) if r.endswith(".csv"))

    cols = ["baseline_slope", "max_decel_min", "n_complicated_variable",
            "has_prolonged_any", "n_recurrent_late", "variability_bpm",
            "variability_low_min", "n_accelerations"]
    rows, heur, y_acid = [], [], []
    for rec in records:
        if rec not in outcomes:
            continue
        fhr, uc = [], []
        with open(os.path.join(SIGNALS, f"{rec}.csv")) as fh:
            rd = csv.reader(fh); next(rd, None)
            for r in rd:
                if len(r) >= 3:
                    fhr.append(float(r[1])); uc.append(float(r[2]))
        sig = from_arrays(fhr=np.asarray(fhr[-N_EPOCH:], float), hz=HZ,
                          toco=np.asarray(uc[-N_EPOCH:], float))
        clean = preprocess(sig)
        if "fhr" not in clean.quality.accepted_channels:
            continue
        f = extract_features(clean)
        rows.append([
            f.baseline_slope_bpm_per_min or 0.0,
            f.max_decel_duration_min or 0.0,
            float(f.n_complicated_variable),
            float(f.has_prolonged_any),
            float(f.n_recurrent_late),
            f.variability_bpm or 0.0,
            f.variability_low_min,
            float(len(f.accelerations)),
        ])
        heur.append(analyze(sig, guideline="figo").alert_score)
        ph = _f(outcomes[rec].get("pH"))
        y_acid.append(ph is not None and ph <= 7.05)

    X = np.asarray(rows, float)
    y = np.asarray(y_acid)
    print(f"usable={len(y)}  acidaemic(pH<=7.05)={int(y.sum())}\n")
    print(f"reference: hand-weighted alert_score AUC = {rank_auc(heur, y):.3f}\n")

    print(f"{'model (features)':52} {'CV-AUC':>8} {'+/-':>6}")
    print("-" * 70)
    models = [
        ("baseline_slope", [0]),
        ("baseline_slope + max_decel_min", [0, 1]),
        ("baseline_slope + max_decel_min + n_complicated_variable", [0, 1, 2]),
        ("+ has_prolonged + n_recurrent_late", [0, 1, 2, 3, 4]),
        ("all 8 features (overfit check)", list(range(8))),
    ]
    for name, idx in models:
        m, s = cv_auc(X[:, idx], y, l2=2.0)
        print(f"{name:52} {m:>8.3f} {s:>6.3f}")
    print("\nCV = 5-fold stratified, 20 repeats, L2=2.0, features standardised on train folds.")


if __name__ == "__main__":
    main()
