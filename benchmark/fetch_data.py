"""Fetch and prepare the CTU-UHB intrapartum CTG benchmark dataset.

Source
------
CTU-UHB Intrapartum Cardiotocography Database (PhysioNet, v1.0.0):
552 records, FHR + uterine activity (UC) at 4 Hz, the last <=90 min before
delivery, each with delivery outcomes (umbilical artery pH, base deficit,
Apgar, neonatal morbidity). Reference: Chudacek et al., "Open access
intrapartum CTG database", BMC Pregnancy and Childbirth, 2014.

PhysioNet itself is not reachable from this environment's network policy, so we
pull the same data from a public GitHub mirror that has already converted the
WFDB records to CSV (signals/<id>.csv with columns seconds,FHR,UC and a
transposed ann_db.csv of outcomes):
    https://github.com/fabiom91/CTU-CHB_Physionet.org  (database.zip)

This script is idempotent: it downloads the archive once (cached under
_cache/), extracts the per-record signal CSVs into data/signals/, and writes a
tidy one-row-per-record data/outcomes.csv. Only outcomes.csv is committed; the
150 MB of raw signal CSVs are reconstructed here on demand.
"""
from __future__ import annotations

import csv
import io
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SIGNALS = os.path.join(DATA, "signals")
CACHE = os.path.join(HERE, "_cache")
ZIP_PATH = os.path.join(CACHE, "database.zip")
OUTCOMES = os.path.join(DATA, "outcomes.csv")

ZIP_URL = "https://raw.githubusercontent.com/fabiom91/CTU-CHB_Physionet.org/master/database.zip"


def _download() -> None:
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(ZIP_PATH) and os.path.getsize(ZIP_PATH) > 1_000_000:
        print(f"[cache] using {ZIP_PATH} ({os.path.getsize(ZIP_PATH)//1_000_000} MB)")
        return
    print(f"[download] {ZIP_URL}")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "ctgkit-benchmark"})
    with urllib.request.urlopen(req, timeout=120) as r, open(ZIP_PATH, "wb") as out:
        out.write(r.read())
    print(f"[download] saved {os.path.getsize(ZIP_PATH)//1_000_000} MB")


def _extract_signals(zf: zipfile.ZipFile) -> int:
    os.makedirs(SIGNALS, exist_ok=True)
    n = 0
    for name in zf.namelist():
        if "__MACOSX" in name or not name.endswith(".csv"):
            continue
        if "/signals/" not in name:
            continue
        rec = os.path.basename(name)
        with zf.open(name) as src, open(os.path.join(SIGNALS, rec), "wb") as dst:
            dst.write(src.read())
        n += 1
    return n


def _write_outcomes(zf: zipfile.ZipFile) -> int:
    """ann_db.csv is transposed (columns = record ids, rows = fields).
    Re-orient to one row per record with a leading `record` column."""
    ann_name = next(n for n in zf.namelist()
                    if n.endswith("ann_db.csv") and "__MACOSX" not in n)
    with zf.open(ann_name) as f:
        rows = list(csv.reader(io.TextIOWrapper(f, "utf-8")))
    fields = [r[0].strip() for r in rows]          # row labels (first is blank)
    record_ids = rows[0][1:]                        # header after the blank cell
    fields[0] = "record"
    out_rows = []
    for col, rec in enumerate(record_ids, start=1):
        row = {"record": rec}
        for ri in range(1, len(rows)):
            row[fields[ri]] = rows[ri][col]
        out_rows.append(row)
    with open(OUTCOMES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    return len(out_rows)


def main() -> None:
    have_signals = (os.path.isdir(SIGNALS)
                    and len([x for x in os.listdir(SIGNALS) if x.endswith(".csv")]) >= 552)
    if have_signals and os.path.exists(OUTCOMES):
        print("[skip] dataset already prepared")
        return
    _download()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        ns = _extract_signals(zf)
        no = _write_outcomes(zf)
    print(f"[prepare] {ns} signal records -> {SIGNALS}")
    print(f"[prepare] {no} outcome rows   -> {OUTCOMES}")


if __name__ == "__main__":
    sys.exit(main())
