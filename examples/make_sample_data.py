"""Generate examples/sample_ctg.csv — the demo fixture used by the notebook.

The fixture deliberately models a *realistic export*:

  * a ``timestamp`` column of ISO-8601 datetime **strings** (not numeric seconds),
  * sampled at **2 Hz** (so the inferred rate is clearly NOT ctgkit's 4.0 default),
  * **38 minutes** long (longer than the 30 + 2 min epoch contract, so the
    notebook has something to window down).

Re-run this script to regenerate the CSV:  ``python examples/make_sample_data.py``
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ctgkit.synth import synth_epoch

GEN_HZ = 2.0          # sampling rate used to lay down timestamps (2 Hz, not 4)
MINUTES = 38.0        # > 32 so the notebook must trim to the most recent 30 min
START = datetime(2024, 6, 1, 9, 0, 0)
OUT = Path(__file__).with_name("sample_ctg.csv")


def main() -> None:
    # A synthetic late-deceleration trace makes the downstream analysis interesting.
    sig = synth_epoch("late_decels", minutes=MINUTES, hz=GEN_HZ, seed=7)

    lines = ["timestamp,fhr,toco"]
    for i, (fhr, toco) in enumerate(zip(sig.fhr, sig.toco)):
        # ISO-8601 with milliseconds -> every row has a consistent, parseable format.
        ts = (START + timedelta(seconds=i / GEN_HZ)).isoformat(timespec="milliseconds")
        fhr_cell = "" if np.isnan(fhr) else f"{fhr:.1f}"   # blank == signal loss
        lines.append(f"{ts},{fhr_cell},{toco:.1f}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}  ({sig.n_samples:,} rows, {MINUTES:.0f} min @ {GEN_HZ} Hz)")


if __name__ == "__main__":
    main()
