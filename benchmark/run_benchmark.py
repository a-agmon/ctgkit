"""Run ctgkit over the CTU-UHB benchmark and score it against birth outcomes.

For each record we analyze the LAST 30 minutes before delivery (the terminal
CTG is what relates to cord-blood acid-base status at birth), map FHR/UC into a
ctgkit Signal at 4 Hz, and run analyze() under each guideline pack.

We then compare ctgkit's output against objective neonatal outcomes:
  * primary label  : umbilical artery pH <= 7.05  (clinically significant
                     acidemia, the standard CTU-UHB endpoint)
  * composite label: pH <= 7.05  OR  base deficit (BDecf) >= 12
                     OR  Apgar-5 < 7  OR  HIE/seizures/intubation

"Positive" predictions are evaluated under three definitions so the
category/alert split is visible: alert==critical, alert in {warning,critical},
and category==3 (ABNORMAL). We also report the rank-AUC of the continuous
alert_score, which is threshold-free.

Run:  python benchmark/run_benchmark.py [guideline ...]
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctgkit import analyze, from_arrays  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNALS = os.path.join(HERE, "data", "signals")
OUTCOMES = os.path.join(HERE, "data", "outcomes.csv")
RESULTS = os.path.join(HERE, "results")

HZ = 4.0
EPOCH_MIN = 30.0
PACKS = ["figo", "nice", "acog", "sogc"]


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_outcomes() -> dict[str, dict]:
    out = {}
    with open(OUTCOMES) as f:
        for row in csv.DictReader(f):
            out[row["record"]] = row
    return out


def labels_for(row: dict) -> dict:
    ph = _f(row.get("pH"))
    bdecf = _f(row.get("BDecf"))
    apgar5 = _f(row.get("Apgar5"))
    morb = any((_f(row.get(k)) or 0) > 0 for k in ("Seizures", "HIE", "Intubation"))
    acidemic = ph is not None and ph <= 7.05
    composite = bool(acidemic
                     or (bdecf is not None and bdecf >= 12)
                     or (apgar5 is not None and apgar5 < 7)
                     or morb)
    return {"pH": ph, "acidemic_705": acidemic, "composite_adverse": composite,
            "ph_known": ph is not None}


def load_signal(record: str):
    secs, fhr, uc = [], [], []
    with open(os.path.join(SIGNALS, f"{record}.csv")) as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            if len(r) < 3:
                continue
            fhr.append(float(r[1]))
            uc.append(float(r[2]))
    fhr = np.asarray(fhr, dtype=float)
    uc = np.asarray(uc, dtype=float)
    n_epoch = int(EPOCH_MIN * 60 * HZ)
    if len(fhr) > n_epoch:                      # terminal 30 min
        fhr, uc = fhr[-n_epoch:], uc[-n_epoch:]
    return from_arrays(fhr=fhr, hz=HZ, toco=uc)


def rank_auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.asarray(scores)
    for v in np.unique(s):                       # average ranks over ties
        m = s == v
        ranks[m] = ranks[m].mean()
    rpos = sum(r for r, y in zip(ranks, labels) if y)
    n1, n2 = len(pos), len(neg)
    return (rpos - n1 * (n1 + 1) / 2) / (n1 * n2)


def confusion(pred_pos: list[bool], label: list[bool]) -> dict:
    tp = sum(p and y for p, y in zip(pred_pos, label))
    fp = sum(p and not y for p, y in zip(pred_pos, label))
    fn = sum((not p) and y for p, y in zip(pred_pos, label))
    tn = sum((not p) and (not y) for p, y in zip(pred_pos, label))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sens": sens, "spec": spec, "ppv": ppv, "npv": npv}


def main(argv: list[str]) -> None:
    packs = [p for p in argv if p in PACKS] or PACKS
    outcomes = load_outcomes()
    records = sorted(r[:-4] for r in os.listdir(SIGNALS) if r.endswith(".csv"))

    rows = []
    for i, rec in enumerate(records, 1):
        if rec not in outcomes:
            continue
        lab = labels_for(outcomes[rec])
        sig = load_signal(rec)
        row = {"record": rec, **lab}
        for pk in packs:
            r = analyze(sig, guideline=pk)
            row[f"{pk}_cat"] = int(r.category) if r.category is not None else 0
            row[f"{pk}_alert"] = r.alert.value
            row[f"{pk}_score"] = round(r.alert_score, 1)
        rows.append(row)
        if i % 100 == 0:
            print(f"  ...{i}/{len(records)} analyzed")

    os.makedirs(RESULTS, exist_ok=True)
    pred_path = os.path.join(RESULTS, "predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    n_ph = sum(r["ph_known"] for r in rows)
    n_acid = sum(r["acidemic_705"] for r in rows)
    n_comp = sum(r["composite_adverse"] for r in rows)
    print("\n" + "=" * 72)
    print(f"CTU-UHB benchmark — {n} records, terminal {EPOCH_MIN:.0f} min @ {HZ:.0f} Hz")
    print(f"prevalence: pH<=7.05 = {n_acid}/{n_ph} ({100*n_acid/n_ph:.1f}%)  |  "
          f"composite adverse = {n_comp}/{n} ({100*n_comp/n:.1f}%)")
    print("=" * 72)

    for label_key, label_name in [("acidemic_705", "pH<=7.05"),
                                  ("composite_adverse", "composite adverse")]:
        y = [r[label_key] for r in rows]
        print(f"\n### outcome = {label_name}")
        for pk in packs:
            usable_rows = [r for r in rows if r[f"{pk}_cat"] != 0]
            usable = len(usable_rows)
            # AUC over ALL records penalises the score for cases it declined to
            # assess (no usable signal -> quality warning); the usable-only AUC
            # is the honest discrimination on traces actually interpreted.
            auc_all = rank_auc([r[f"{pk}_score"] for r in rows], y)
            auc_use = rank_auc([r[f"{pk}_score"] for r in usable_rows],
                               [r[label_key] for r in usable_rows])
            # Alert-fatigue metric = PAGE-WORTHY load (warning+critical). WATCH
            # is flagged-but-not-paged and QUALITY is a technical channel; both
            # are excluded from the page load but still surface the trace, so
            # nothing is hidden.
            alerts = [r[f"{pk}_alert"] for r in rows]
            quality = [a == "quality" for a in alerts]
            watch = [a == "watch" for a in alerts]
            pageworthy = [a in ("warning", "critical") for a in alerts]
            n_quality = sum(quality)
            adv_quality = sum(1 for r, q in zip(rows, quality) if q and r[label_key])
            n_page = sum(pageworthy)
            n_watch = sum(watch)
            defs = {
                "alert=critical":            [a == "critical" for a in alerts],
                "page-worthy (warn+crit)":   pageworthy,
                "flagged (watch+warn+crit)": [a in ("watch", "warning", "critical") for a in alerts],
                "category=3 (ABNORMAL)":     [r[f"{pk}_cat"] == 3 for r in rows],
            }
            au = f"{auc_use:.3f}" if auc_use is not None else "n/a"
            print(f"\n  [{pk}]  usable-category {usable}/{n}   "
                  f"score-AUC {auc_all:.3f} all / {au} usable")
            print(f"    page-worthy {n_page}/{n} ({100*n_page/n:.0f}%)   "
                  f"watch {n_watch}/{n}   quality(technical) {n_quality}/{n} (adverse {adv_quality})")
            print(f"    {'positive rule':28}  sens   spec   ppv    npv    TP/FP/FN/TN")
            for name, pred in defs.items():
                c = confusion(pred, y)
                print(f"    {name:28}  {c['sens']:.2f}   {c['spec']:.2f}   "
                      f"{c['ppv']:.2f}   {c['npv']:.2f}   "
                      f"{c['tp']}/{c['fp']}/{c['fn']}/{c['tn']}")

    print(f"\nper-record predictions -> {pred_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
