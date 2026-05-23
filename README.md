# ctgkit

A Python library that reads a 30-minute fetal heart-rate recording and tells you, in structured form, **how concerning it looks and why** — a guideline category, an alert level, and a list of specific issues with the evidence behind each one. Optional plotting included.

> ⚠️ **Decision support only.** This is not a diagnostic device. Its outputs are meant to be read and overruled by a qualified clinician. It is not validated for clinical use as-is. See [Regulatory note](#regulatory-note).

---

## What is this, if you're not a clinician?

You don't need an obstetrics background to use this library, but you need to know roughly what it's looking at. Here's the 5-minute version.

During labour, hospitals continuously record two signals from the mother's abdomen:

- **FHR — fetal heart rate**, in beats per minute (bpm). A healthy baby's heart usually sits somewhere around **110–160 bpm** and *wiggles* constantly from second to second.
- **Toco — uterine activity** (contractions). Each contraction is a bump in this signal.

Together these two traces are called a **CTG** (cardiotocograph) or **EFM** (electronic fetal monitoring). Clinicians stare at the paper/screen and ask: *is the baby tolerating labour, or showing signs of oxygen stress?* They read it through a handful of repeatable concepts:

| Concept | Plain meaning | Why it matters |
|---|---|---|
| **Baseline** | the average heart rate, ignoring brief spikes/dips | too high (tachycardia) or too low (bradycardia) is a warning sign |
| **Variability** | how much the rate wiggles second-to-second | a healthy nervous system produces lots of wiggle (5–25 bpm). A *flat* trace can mean the baby is asleep — or in trouble |
| **Acceleration** | a brief jump *up* in rate | generally reassuring |
| **Deceleration** | a brief dip *down* | the important one. Its **shape and timing relative to contractions** tells you a lot — a dip that lags *after* a contraction ("late") is more worrying than one that mirrors it |
| **Contractions / tachysystole** | the bumps; "tachysystole" = too many, too close | too-frequent contractions reduce the baby's oxygen supply |

The catch: there is **no single agreed rulebook**. Four major medical bodies — **FIGO** (international), **NICE** (UK), **ACOG/NICHD** (US), and **SOGC** (Canada) — each publish their own thresholds for turning these features into a verdict. They mostly agree, but draw the lines in different places. `ctgkit` implements all four as interchangeable "guideline packs."

### What the library actually produces

For each 30-minute recording, it returns three things, deliberately kept separate:

1. **Category (1 / 2 / 3)** — the guideline's verdict.
   `1` = reassuring, `2` = indeterminate (watch it), `3` = abnormal (act). This is computed by a deterministic rules engine, faithful to whichever guideline you pick.

2. **Alert level (`none` / `warning` / `critical`)** — *not* the same as the category. This answers a different question: *"should someone be paged right now?"* It factors in how long a problem has persisted, whether things are getting worse, signal quality, and the mother's risk factors.

3. **Concerns** — a list of specific, machine-readable findings (`recurrent_late_decels`, `reduced_variability`, …), each with a start time, duration, severity, and the numbers that triggered it.

### Why category and alert are separate (the one design idea to understand)

You might expect "Category 3 → critical alert, always." We deliberately don't do that. The reason is **alert fatigue**: the guidelines' middle tier (especially the US "Category II") is enormous — a huge fraction of all recordings land there. If every Category 2 fired a warning, clinicians would be buzzed constantly and start ignoring the system, which is dangerous.

So `ctgkit` keeps the **category** faithful to the medical guideline, and computes the **alert** with a separate scorer that only escalates when escalation is actually justified (persistence, worsening trend, acute events, stacked risk factors). A broad-but-stable Category 2 stays quiet; a Category 2 that's deteriorating gets a warning. **The guideline you choose changes the category far more than it changes the alerting behaviour** — the alert scorer is guideline-agnostic by design.

That's the whole mental model. The rest is API.

---

## Install

```bash
pip install -e .            # core (numpy, scipy)
pip install -e ".[plot]"    # + matplotlib for plotting
```

Python 3.9+.

---

## Quickstart

```python
import ctgkit

# Analyze a 30-minute recording from a CSV
result = ctgkit.analyze("epoch.csv", guideline="figo")

print(result.summary())
# Category 3  |  ALERT: CRITICAL  |  confidence: high
# trend: unknown  |  pack: figo v2015
# concerns:
#   - [high] Recurrent late decelerations: Strongest recurrent hypoxia pattern...

result.alert.value      # 'critical'
result.category          # Category.ABNORMAL  (== 3)
result.to_dict()         # fully JSON-serializable, audit-ready
```

With clinical context (improves the alert layer) and a plot:

```python
result = ctgkit.analyze(
    "epoch.csv",
    guideline="nice",
    metadata={"oxytocin": True, "meconium": True},  # risk factors
)
ctgkit.plot("epoch.csv", result, save_path="trace.png")
```

If you have the raw arrays (e.g. from a live monitor) instead of a file:

```python
sig = ctgkit.from_arrays(fhr=fhr_array, hz=4.0, toco=toco_array)
result = ctgkit.analyze(sig, guideline="acog")
```

Try it without any data using the built-in synthetic generator:

```python
from ctgkit.synth import synth_epoch
ctgkit.analyze(synth_epoch("late_decels")).summary()
```

More plotting examples (all equivalent ways in; safe to call repeatedly):

```python
# from a Signal you already analyzed
ctgkit.plot(sig, result, save_path="trace.png")

# straight from a CSV path (analyzes internally)
ctgkit.plot("epoch.csv", save_path="trace.png")

# interactive window instead of saving
ctgkit.plot(sig, result, show=True)

# loop over many epochs without leaking figures
for path in epoch_paths:
    ctgkit.plot(path, save_path=path.replace(".csv", ".png"))
```

---

## Recommended service configuration

`analyze()`'s defaults are tuned for *exploration* (lenient epoch handling, FIGO default). A **running service** should be stricter and explicit. Use `analyze_service()` / `ServiceConfig`, which packages the recommended production posture in one auditable place:

```python
from ctgkit import analyze_service, ServiceConfig

# Pick the jurisdiction's pack at deploy time (see note below).
config = ServiceConfig(
    guideline="acog",      # explicit; or "nice" / "figo" / "sogc" per site
    strict_epoch=True,     # refuse off-length epochs instead of warning
    epoch_tolerance_min=2  # 30-min contract, accept 28..32 min
)

result = analyze_service("epoch.csv", config, metadata={"oxytocin": True})
```

What the recommended posture enforces:

| Setting | Recommended | Why |
|---|---|---|
| `strict_epoch` | **`True`** | A sub-30-min trace can't satisfy the duration-based guideline timers (e.g. reduced variability for >50 min), so scoring it risks a *falsely reassuring* result. The service refuses (`SignalError`) and asks for a full epoch. |
| `guideline` | **explicit** (e.g. `"acog"`) | Never rely on the library default in production. Set the jurisdiction's pack so the category and the audit record are unambiguous. |
| epoch window | **30 min ± 2** (28–32) | Matches the FIGO/NICE 30-min formal-review cadence; the ±2 min band only absorbs export rounding, not genuinely short/long traces. |

`ctgkit.RECOMMENDED` is a ready-made `ServiceConfig` (defaults shown above) — override `guideline` for your site.

```python
from ctgkit import analyze_service, RECOMMENDED
result = analyze_service(sig, RECOMMENDED)        # ACOG, strict, 30±2
```

---

## Input format

A CSV with a header. **Only the FHR column is required.** Columns are matched case-insensitively with common aliases.

```csv
time_s,fhr,toco
0.00,142.3,1.2
0.25,141.8,1.4
...
```

| Column | Aliases | Required? | Notes |
|---|---|---|---|
| `fhr` | `fhr1`, `fhr_primary`, `hr`, `bpm` | **yes** | fetal heart rate, bpm |
| `toco` | `ua`, `uc`, `uterine` | preferred | uterine activity. Without it, decelerations can't be reliably typed as "late" vs "variable" |
| `mhr` | `maternal_hr` | optional | maternal heart rate; used to catch maternal/fetal signal confusion |
| `time_s` | `time`, `t`, `sec` | optional | if regular, sampling rate is inferred from it |

Missing samples may be blank or `0` (a `0` heart rate is treated as signal loss).

### A note on TOCO quality

The toco channel is held to the same quality gate as FHR (raw usable ≥ 0.80). This has two consequences worth understanding:

- **Decel timing needs a quality-passing toco.** `late` vs `variable` typing is only asserted when toco is *accepted*. If toco is absent **or present but degraded**, decelerations are surfaced as `morphology="uncertain"` / `timing="uncertain"` rather than mislabeled — and prolonged dips are still caught by **duration** regardless (see below).
- **Tachysystole from a degraded toco is lower-confidence.** Contraction *counting* can still run on a partial toco, so tachysystole may be detected — but if the toco failed the quality gate, the count is unreliable. In that case the concern is downgraded (`severity=low`, titled "Possible tachysystole (low-confidence toco)") and **does not contribute to the alert score**. `result.features.tachysystole_low_confidence` exposes this flag.

In short: degraded toco never *invents* timing information, and never lets an unreliable contraction count drive escalation.

### The 30-minute check

The guidelines are written around a **30-minute review window**, so that's the contract. By default `ctgkit.analyze()` accepts **28–32 minutes** (30 ± 2).

- **Too short or too long?** With `analyze()`'s exploratory defaults it still analyzes and adds a note to `result.warnings`. Duration-based rules are measured against *actual* elapsed time, so a short trace simply may not trip them.
- **In a service, use `analyze_service()`** (or `analyze(..., strict_epoch=True)`), which **refuses** off-length epochs with a `SignalError` rather than risk a falsely reassuring score. See [Recommended service configuration](#recommended-service-configuration).
- *(Planned: auto-window long recordings to the most recent 30 min.)*

---

## Output: `EpochResult`

| Field | What it is |
|---|---|
| `category` | `Category.{REASSURING, INDETERMINATE, ABNORMAL}` (1/2/3), or `None` if the FHR signal is too poor to classify |
| `alert` | `AlertLevel.{NONE, WARNING, CRITICAL}` |
| `concerns` | list of `Concern` objects (label, title, severity, start/duration, evidence) |
| `confidence` | `'high'` (raw FHR ≥ 0.95) / `'medium'` (≥ 0.80) / `'low'` (< 0.80 → no category), driven by raw signal quality |
| `trend` | `improving` / `stable` / `worsening` vs the `previous` epoch |
| `features` | extracted numbers: baseline, variability, decel types, etc. |
| `quality` | accepted channels + usable fractions |
| `guideline_pack`, `pack_version`, `library_version` | for reproducibility / audit |

`.summary()` → human-readable string. `.to_dict()` → JSON.

**One guarantee worth knowing:** if the FHR signal is too degraded to trust, the library returns `category=None`, `alert='warning'`, `confidence='low'` — it will **never** return a reassuring "all clear" on bad data.

---

## How it fits together

```
CSV / arrays
   → validate 30-min epoch
   → preprocess + quality gate   (range filter, despike, gap-fill, score quality)
   → extract features            (baseline, variability, decels, contractions…)
   → guideline category          (FIGO / NICE / ACOG / SOGC — pluggable)
   → alert scorer (separate!)    (persistence, trend, risk stacking, quality)
   → EpochResult
```

| Module | Responsibility |
|---|---|
| `io.py` | `Signal` container, CSV loader, 30-min contract |
| `preprocess.py` | cleaning + the quality gate ("bad signal never reassures") |
| `features.py` | deterministic, explainable feature detectors |
| `guidelines.py` | the four guideline packs (versioned) |
| `alerts.py` | the guideline-agnostic alert scorer + concern objects |
| `pipeline.py` | `analyze()` orchestration |
| `service_config.py` | `analyze_service()` / `ServiceConfig` — recommended production posture |
| `plotting.py` | optional matplotlib trace (`ctgkit.plot` wraps `plotting.plot_epoch`) |
| `synth.py` | synthetic data for demos/tests (not for clinical use) |

Full design rationale, scoring weights, and roadmap are in [`SPEC.md`](SPEC.md).

---

## Testing / CI

```bash
pip install -e ".[plot]" pytest
python -m pytest -q
```

49 tests covering the safety invariants — *a degraded signal never produces a reassuring result*, *prolonged decelerations are caught by duration regardless of shape*, *degraded TOCO never yields late/variable claims and makes tachysystole low-confidence*, and *the service config refuses off-length epochs*. CI runs the same command across Python 3.9–3.12 (`.github/workflows/ci.yml`).

---

## Status & limitations

- **v0.1.2, reference implementation.** End-to-end runnable; **42 tests pass** (`python -m pytest -q`). Detectors are clinically reasonable but **tuned against synthetic traces, not yet validated on real annotated data** (e.g. CTU-UHB) — do not treat as clinical-grade.
- Signal quality is judged on the **raw** usable fraction (≥0.80 to accept FHR); interpolation cannot upgrade confidence. Prolonged decelerations are detected by **duration independent of morphology**.
- Guideline packs currently share the same underlying feature detectors and differ in how they *combine* results; fully per-pack thresholds are a v0.2 item.
- No ML yet — by design. v0.1 is a deterministic rules engine; ML for signal-quality, morphology, and Category-2 prioritization is a planned hybrid layer.

## Regulatory note

Software that interprets fetal monitoring signals to flag compromise is, in most jurisdictions, likely regulated as **medical device software (SaMD)** — even when framed as decision support. `ctgkit` is built to make that path defensible (full output traceability, explainable concerns, human-in-the-loop intent), but it is **not** a cleared device and must not be used to make clinical decisions without appropriate validation and regulatory clearance.
