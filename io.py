"""Input handling: load a CTG signal, validate it, and enforce the
~30-minute epoch contract.

Accepted input is a CSV (the most portable interchange format for exported
waveforms). The library is signal-agnostic underneath, so a Signal can also
be built directly from numpy arrays for streaming/edge use.

Expected CSV columns (case-insensitive, flexible aliases):
    time_s        seconds from start (optional; if absent, derived from hz)
    fhr           primary fetal heart rate, bpm        [REQUIRED]
    fhr2          secondary fetal channel, bpm         [optional]
    toco / ua     uterine activity / tocodynamometer   [optional but preferred]
    mhr           maternal heart rate, bpm             [optional]

Missing samples may be blank or 0; 0 in FHR is treated as signal loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Column name aliases -> canonical name
_ALIASES = {
    "fhr": "fhr", "fhr1": "fhr", "fhr_primary": "fhr", "hr": "fhr", "bpm": "fhr",
    "fhr2": "fhr2", "fhr_secondary": "fhr2",
    "toco": "toco", "ua": "toco", "uc": "toco", "uterine": "toco", "uterine_activity": "toco",
    "mhr": "mhr", "maternal_hr": "mhr", "mhr_bpm": "mhr",
    "time": "time_s", "time_s": "time_s", "t": "time_s", "seconds": "time_s", "sec": "time_s",
}

DEFAULT_HZ = 4.0           # CTU-UHB / common monitor export rate
EPOCH_MINUTES = 30.0
EPOCH_TOLERANCE_MIN = 2.0  # accept 28..32 min by default


class SignalError(ValueError):
    """Raised when input cannot be loaded or fails the epoch contract."""


@dataclass
class Signal:
    """Container for one CTG epoch. Channels are 1-D float arrays of equal
    length, aligned to a common time base at `hz`."""
    fhr: np.ndarray
    hz: float = DEFAULT_HZ
    fhr2: Optional[np.ndarray] = None
    toco: Optional[np.ndarray] = None
    mhr: Optional[np.ndarray] = None
    time_s: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.fhr)

    @property
    def duration_min(self) -> float:
        return self.n_samples / self.hz / 60.0

    def channels(self) -> dict[str, np.ndarray]:
        out = {"fhr": self.fhr}
        if self.fhr2 is not None:
            out["fhr2"] = self.fhr2
        if self.toco is not None:
            out["toco"] = self.toco
        if self.mhr is not None:
            out["mhr"] = self.mhr
        return out

    def validate_epoch(
        self,
        expected_min: float = EPOCH_MINUTES,
        tolerance_min: float = EPOCH_TOLERANCE_MIN,
        strict: bool = False,
    ) -> list[str]:
        """Check the 30-minute contract. Returns a list of warning strings.
        Raises SignalError if strict=True and duration is out of tolerance."""
        warnings: list[str] = []
        dur = self.duration_min
        lo, hi = expected_min - tolerance_min, expected_min + tolerance_min
        if not (lo <= dur <= hi):
            msg = (f"epoch duration {dur:.1f} min is outside the "
                   f"{lo:.0f}-{hi:.0f} min window for a {expected_min:.0f}-min epoch")
            if strict:
                raise SignalError(msg)
            warnings.append(msg)
        # length sanity across channels
        for name, arr in self.channels().items():
            if len(arr) != self.n_samples:
                raise SignalError(f"channel '{name}' length {len(arr)} != fhr length {self.n_samples}")
        return warnings


def _canonical(colname: str) -> Optional[str]:
    key = colname.strip().lower().replace(" ", "_")
    return _ALIASES.get(key)


def load_csv(
    path: str,
    hz: float = DEFAULT_HZ,
    meta: Optional[dict] = None,
) -> Signal:
    """Load a CTG epoch from a CSV file.

    If a time column is present and reasonably regular, `hz` is inferred from
    it and overrides the passed value.
    """
    import csv as _csv

    cols: dict[str, list[float]] = {}
    header_map: dict[int, str] = {}
    with open(path, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise SignalError("CSV is empty")
        for i, h in enumerate(header):
            canon = _canonical(h)
            if canon:
                header_map[i] = canon
                cols.setdefault(canon, [])
        if "fhr" not in cols:
            raise SignalError(
                f"no FHR column found. Columns seen: {header}. "
                "Expected one of fhr/fhr1/fhr_primary/hr."
            )
        for row in reader:
            if not row:
                continue
            for i, canon in header_map.items():
                val = row[i].strip() if i < len(row) else ""
                cols[canon].append(float(val) if val not in ("", "nan", "NaN") else np.nan)

    arrs = {k: np.asarray(v, dtype=float) for k, v in cols.items()}

    # infer hz from time column if regular
    inferred_hz = hz
    if "time_s" in arrs and len(arrs["time_s"]) > 1:
        dt = np.diff(arrs["time_s"])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) and np.std(dt) < 0.25 * np.mean(dt):
            inferred_hz = 1.0 / float(np.mean(dt))

    sig = Signal(
        fhr=arrs["fhr"],
        hz=inferred_hz,
        fhr2=arrs.get("fhr2"),
        toco=arrs.get("toco"),
        mhr=arrs.get("mhr"),
        time_s=arrs.get("time_s"),
        meta=meta or {},
    )
    return sig


def from_arrays(
    fhr,
    hz: float = DEFAULT_HZ,
    toco=None,
    fhr2=None,
    mhr=None,
    meta: Optional[dict] = None,
) -> Signal:
    """Build a Signal directly from arrays (streaming / edge use)."""
    return Signal(
        fhr=np.asarray(fhr, dtype=float),
        hz=hz,
        toco=None if toco is None else np.asarray(toco, dtype=float),
        fhr2=None if fhr2 is None else np.asarray(fhr2, dtype=float),
        mhr=None if mhr is None else np.asarray(mhr, dtype=float),
        meta=meta or {},
    )
