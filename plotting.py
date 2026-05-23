"""Optional plotting. Renders a clinician-style two-panel trace (FHR + toco)
with the baseline, detected events, and the alert banner overlaid.

Requires matplotlib (optional dependency). Import is local so the core
library works without it.
"""
from __future__ import annotations

from typing import Optional, Union
import numpy as np

from .io import Signal, load_csv
from .pipeline import analyze
from .models import EpochResult, AlertLevel

_ALERT_COLOR = {
    AlertLevel.NONE: "#2e7d32",
    AlertLevel.WARNING: "#f9a825",
    AlertLevel.CRITICAL: "#c62828",
}


def plot_epoch(
    signal: Union[Signal, str],
    result: Optional[EpochResult] = None,
    guideline: str = "figo",
    metadata: Optional[dict] = None,
    hz: float = 4.0,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot the epoch. If `result` is None, analyze() is run first.

    `signal` may be a Signal or a path to a CSV. Safe to call repeatedly.
    Returns the matplotlib Figure.
    """
    import matplotlib
    # Only force a headless backend if we're saving without showing AND no
    # interactive backend is already active. Switching backends after pyplot
    # is initialised raises on repeated calls, so guard it.
    if save_path and not show:
        current = matplotlib.get_backend().lower()
        if "agg" not in current and "inline" not in current:
            try:
                matplotlib.use("Agg", force=True)
            except Exception:
                pass
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if isinstance(signal, str):
        signal = load_csv(signal, hz=hz, meta=metadata)

    if result is None:
        result = analyze(signal, guideline=guideline, metadata=metadata)

    hz = signal.hz
    t = np.arange(signal.n_samples) / hz / 60.0  # minutes

    has_toco = signal.toco is not None
    fig, axes = plt.subplots(
        2 if has_toco else 1, 1, figsize=(13, 7 if has_toco else 4.5),
        sharex=True, gridspec_kw={"height_ratios": [3, 1]} if has_toco else None,
    )
    ax_fhr = axes[0] if has_toco else axes

    # FHR
    fhr = signal.fhr.copy()
    fhr[(fhr < 50) | (fhr > 220) | (fhr == 0)] = np.nan
    ax_fhr.plot(t, fhr, lw=0.8, color="#1565c0")
    ax_fhr.set_ylim(50, 210)
    ax_fhr.set_ylabel("FHR (bpm)")
    ax_fhr.axhspan(110, 160, color="#a5d6a7", alpha=0.25, zorder=0)  # normal band
    for y in (110, 160):
        ax_fhr.axhline(y, color="#888", lw=0.6, ls="--")

    # baseline + events
    if result.features and result.features.baseline_bpm:
        ax_fhr.axhline(result.features.baseline_bpm, color="#37474f",
                       lw=1.0, ls=":", label=f"baseline {result.features.baseline_bpm:.0f}")

    # mark concern windows with start/duration
    for c in result.concerns:
        if c.start_min is not None and c.duration_min:
            color = "#c62828" if c.severity.value == "high" else "#f9a825"
            ax_fhr.add_patch(Rectangle(
                (c.start_min, 52), c.duration_min, 156,
                color=color, alpha=0.12, zorder=0))

    # alert banner
    col = _ALERT_COLOR[result.alert]
    cat = f"Category {int(result.category)}" if result.category else "Category ?"
    title = (f"{cat}  |  ALERT: {result.alert.value.upper()}  "
             f"|  {result.guideline_pack.upper()} v{result.pack_version}  "
             f"|  confidence: {result.confidence}")
    ax_fhr.set_title(title, color="white", fontsize=12, fontweight="bold",
                     backgroundcolor=col, pad=10)

    if result.features and result.features.baseline_bpm:
        ax_fhr.legend(loc="upper right", fontsize=8)

    # toco
    if has_toco:
        ax_toco = axes[1]
        toco = signal.toco
        ax_toco.plot(t, toco, lw=0.8, color="#6a1b9a")
        ax_toco.set_ylabel("Toco / UA")
        ax_toco.set_xlabel("Time (minutes)")
    else:
        ax_fhr.set_xlabel("Time (minutes)")

    # concern list as figure text
    if result.concerns:
        lines = [f"• [{c.severity.value}] {c.title}" for c in result.concerns[:6]]
        fig.text(0.012, 0.012, "\n".join(lines), fontsize=8, va="bottom",
                 family="monospace", color="#333")

    fig.tight_layout(rect=[0, 0.06 if result.concerns else 0, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    elif save_path:
        # saved but not shown: close to avoid accumulating figures across calls
        plt.close(fig)
    return fig
