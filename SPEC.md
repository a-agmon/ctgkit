# `ctgkit` — Plan & Specification

A Python library that ingests a ~30-minute fetal heart-rate (CTG/EFM) signal, validates it, processes it, and returns a **guideline category**, an **orthogonal alert level**, and a **structured list of concerns** — with optional plotting.

> **Decision support only.** This library does not diagnose, treat, or replace clinical judgement. Every output is designed to be independently reviewable by a clinician. See §9 for regulatory framing.

This spec is **grounded in a working reference implementation** — the design decisions below are reflected in runnable code (`ctgkit/`), a synthetic-data generator, and a passing test suite.

---

## 1. Design thesis (why it's built this way)

The source analysis makes one decision that drives the whole library: **category must not equal alert.** The guideline families (FIGO, NICE, ACOG/NICHD, SOGC) are built for *trace interpretation*, not for managing alert burden. The single biggest source of alert fatigue is the enormous "intermediate" space — especially ACOG Category II, into which a very large share of tracings fall. A service that turns every Category II into a warning is unusable.

So `ctgkit` is split into two layers:

1. **A deterministic rules engine** that produces the guideline-faithful category (1/2/3). This is the source of truth and is fully auditable.
2. **A separate alert scorer** that asks a *different* question — *"is escalation justified now, given persistence, trend, signal quality, and risk stacking?"* — and produces `none` / `watch` / `warning` / `critical` (where `watch` = flagged for review but not page-worthy).

This is the most defensible near-term design and the central anti-fatigue mechanism. ML is deliberately *not* the spine of v0.1; it's a future layer (§10) for signal quality, morphology, and Category-2 prioritization.

---

## 2. Canonical mapping

The library normalizes every guideline to a 3-tier canonical category:

| Canonical | FIGO | NICE | ACOG/NICHD | SOGC | Meaning |
|---|---|---|---|---|---|
| **1** REASSURING | normal | normal | Category I | normal | no evidence of compromise now |
| **2** INDETERMINATE | suspicious | suspicious | Category II | atypical | needs surveillance / context / trend |
| **3** ABNORMAL | pathological | pathological | Category III | abnormal | prompt escalation / review |

The mapping is clinically natural but **not equivalent in severity distribution** — which is exactly why the category is preserved per-pack and the alert is computed separately. The reference implementation demonstrates this: a recurrent-late-deceleration trace classifies as Category 3 under FIGO/NICE/SOGC but Category 2 under ACOG — yet the alert layer keeps the ACOG case at `warning` rather than silencing it.

**Compensation modifier.** After a pack assigns its category, a single shared rule (`guidelines.classify`) asks the question the raw per-feature decel rules omit: *are variability and accelerations preserved?* A category that is abnormal **only** on deceleration morphology — i.e. it does not trip a hard pathological feature (sinusoidal, prolonged ≥5 min, ≥3 min with poor recovery, variability <3 with decels, extreme baseline with decels) — is downgraded to **indeterminate** when accelerations are present **and** variability is moderate. This encodes the clinical fact that accelerations plus moderate variability are the strongest bedside evidence the fetus is not acidotic now, so recurrent late decels in a *compensating* fetus are suspicious rather than the abnormal extreme. The alert layer surfaces this as an INFO `protective_features` concern for auditability.

---

## 3. Public API

The whole library is reachable through one function, plus loaders and an optional plotter.

```python
import ctgkit

# From a CSV path:
result = ctgkit.analyze("epoch.csv", guideline="figo")

# From arrays (streaming / edge use):
sig = ctgkit.from_arrays(fhr=fhr_array, hz=4.0, toco=toco_array)
result = ctgkit.analyze(sig, guideline="nice",
                        metadata={"oxytocin": True, "meconium": True})

print(result.summary())
result.to_dict()          # JSON-serializable, fully auditable

# Optional plot (matplotlib):
ctgkit.plot(sig, result, save_path="trace.png")
```

For a **deployed service**, use the recommended posture instead of bare `analyze()`:

```python
from ctgkit import analyze_service, ServiceConfig
cfg = ServiceConfig(guideline="acog")   # explicit; strict_epoch=True; 30±2 min
result = analyze_service("epoch.csv", cfg, metadata={"oxytocin": True})
# raises SignalError on an off-length epoch rather than scoring it
```

### `analyze(...)` signature

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `signal` | `Signal \| str` | — | A loaded `Signal`, or a path to a CSV |
| `guideline` | `str` | `"figo"` | `"figo"`, `"nice"`, `"acog"`, or `"sogc"` |
| `metadata` | `dict` | `None` | Clinical context for risk stacking (see §6) |
| `hz` | `float` | `4.0` | Sampling rate if not inferable from CSV |
| `strict_epoch` | `bool` | `False` | Raise vs. warn when duration is out of tolerance |
| `epoch_tolerance_min` | `float` | `2.0` | Accept 28–32 min for a 30-min epoch |
| `previous` | `EpochResult` | `None` | Prior epoch, enabling trend computation |

For services, `analyze_service(signal, config=RECOMMENDED, metadata=None, previous=None)` wraps `analyze()` with `strict_epoch=True`, an explicit guideline, and the documented 30 ± 2 min contract (§4). `ServiceConfig` is frozen so a running service can log the exact posture used.

---

## 4. Input format & the 30-minute check

### CSV (the portable interchange format)

The loader is column-flexible (case-insensitive aliases). Only **FHR is required**.

| Canonical column | Accepted aliases | Required? |
|---|---|---|
| `fhr` | `fhr1`, `fhr_primary`, `hr`, `bpm` | **yes** |
| `toco` | `ua`, `uc`, `uterine`, `uterine_activity` | preferred |
| `mhr` | `maternal_hr` | optional |
| `fhr2` | `fhr_secondary` | optional |
| `time_s` | `time`, `t`, `sec`, `seconds` | optional |

```csv
time_s,fhr,toco
0.00,142.3,1.2
0.25,141.8,1.4
...
```

If a regular `time_s` column is present, the sampling rate is **inferred** from it and overrides `hz`. Missing samples may be blank or `0` (a `0` in any HR channel is treated as signal loss, not a real heartbeat).

### The epoch contract

`Signal.validate_epoch()` enforces the ~30-minute window:

- Default acceptance: **28–32 min** (`30 ± tolerance`).
- `strict_epoch=False` (default): out-of-window durations produce a **warning** string in `result.warnings` but still analyze.
- `strict_epoch=True`: raises `SignalError`.
- Channel-length mismatches always raise (data-integrity error).

This honours the FIGO/NICE **30-minute formal review cadence** while staying tolerant of real-world export jitter.

---

## 5. Processing pipeline

```
CSV / arrays
   │
   ▼
[1] Validate epoch (30-min contract, channel lengths)
   │
   ▼
[2] Preprocess + quality gate         ← preprocess.py
      range filter (50–220 bpm) → spike suppression →
      short-gap interpolation (≤4 s) → toco flatline-artifact mask →
      per-channel usable-fraction → confidence (high/med/low)
   │
   ▼  (if FHR unusable → no category, alert=quality [technical channel], never "none")
   │
[3] Feature extraction                ← features.py
      baseline (rolling 10-min, excursions excluded) + slope
      short-term variability (MAD-based band, minutes <5)
      accelerations / decelerations (typed) / contractions
      tachysystole (rolling 10-min) / sinusoidal (spectral)
   │
   ▼
[4] Guideline category                ← guidelines.py  (pluggable packs)
   │
   ▼
[5] Alert scorer (ORTHOGONAL)         ← alerts.py
      category base + acute events + persistence + trend +
      risk stacking + quality penalty  →  none / watch / warning / critical
   │
   ▼
EpochResult  (category, alert, concerns, confidence, trend, features, quality)
```

### Quality gate — the one inviolable rule

**Poor signal quality may raise caution but must never reassure.** Confidence is judged on the **raw FHR usable fraction** (before interpolation), so cosmetically gap-filling a poor trace can't upgrade it:

- **high** — raw FHR usable ≥ 0.95
- **medium** — raw FHR usable ≥ 0.80
- **low** — raw FHR usable < 0.80 → channel not accepted → `category=None`, `alert=quality`

(Heavy interpolation, >20% of samples, downgrades confidence one notch.) If FHR is unusable the library returns `category=None`, `alert=quality`, `confidence=low` — a technical/equipment notice (check the transducer / consider a scalp electrode), kept OUT of the clinical alert stream so unreadable traces don't drive alert fatigue. It does *not* emit a comforting "Category 1 / none." Enforced by a hard safety floor and tests.

The **toco** channel is gated the same way: it is used to type decelerations as `late`/`variable` **only when it passes the quality gate** (`"toco" in accepted_channels`). A present-but-degraded toco is excluded, and decels fall back to `timing-uncertain` rather than being mislabeled.

### Feature detectors (all deterministic & explainable)

- **Baseline** — rolling 10-min mean with accel/decel excursions excluded; epoch slope to catch rising trend.
- **Variability** — short-term band via robust (MAD-based) residual after removing baseline wander; tracks *minutes below 5 bpm* so guideline duration-timers can fire.
- **Decelerations** — typed as `early` / `late` / `variable` / `prolonged`. **Critically:** `late` vs `variable` is only asserted when a usable toco channel exists; without toco, decels are surfaced as **timing-uncertain** rather than confidently mislabeled (the conservative choice from the source analysis).
- **Contractions / tachysystole** — rolling 10-min count; `>5 / 10 min` flags tachysystole.
- **Sinusoidal** — narrow-band spectral check (3–5 cpm dominance) tuned to avoid firing on ordinary variability; flagged for **clinician confirmation**, never auto-asserted.

---

## 6. Alert scoring & concerns

The alert is an additive, **auditable** score (thresholds are tunable starting targets, see §8). Contributions:

| Source | Effect |
|---|---|
| Category 3 / Category 2 | +100 / +30 base |
| Prolonged decel >5 min | +100 (stand-alone critical) |
| Acute decel ≥3 min without quick recovery | +70 |
| Sinusoidal pattern | +100 |
| Recurrent late decels | +35 |
| Complicated variable decels | +20 |
| Baseline tachy/brady (banded) | +12 to +30 |
| Reduced variability (timer-banded) | +12 / +25 |
| Tachysystole | +10 |
| Worsening trend vs previous epoch | +20 |
| Risk stacking (per factor) | +6 to +18 |
| Low signal confidence | +10 (caution only) |

`score ≥ 100 → critical`, `≥ 40 → warning`, else `none`. **Safety floor:** an ABNORMAL category can never resolve to `none`.

### Risk stacking (`metadata`)

Optional clinical context lowers the escalation threshold, per NICE/SOGC guidance on intrapartum risk factors:

```python
metadata = {
  "oxytocin": True, "meconium": True, "fever": False, "sepsis": False,
  "prolonged_rom": True, "preeclampsia": False, "diabetes": False,
  "growth_restriction": False, "prematurity": False,
  "previous_cesarean": False, "slow_progress": False,
}
```

### Concern objects — built for independent review

Each concern is a structured record, not a sentence — so a clinician can inspect the basis of any alert (the FDA CDS "independent review" expectation):

```python
Concern(
  label="recurrent_late_decels",         # machine key
  title="Recurrent late decelerations",  # human-readable
  severity=HIGH,
  detail="Strongest recurrent hypoxia pattern in most frameworks.",
  start_min=..., duration_min=..., trend=...,
  supporting_channels=["fhr", "toco"],
  evidence={"n_late": 9},                # the numbers behind it
)
```

Concern labels include: `persistent_tachycardia`, `low_baseline`, `rising_baseline`, `reduced_variability`, `recurrent_late_decels`, `complicated_variable_decels`, `prolonged_deceleration`, `sinusoidal_pattern`, `tachysystole`, `signal_quality_risk`, `contextual_clinical_risk`.

---

## 7. Output object

`analyze()` returns an `EpochResult`:

| Field | Type | Notes |
|---|---|---|
| `category` | `Category \| None` | `None` only when FHR unusable |
| `alert` | `AlertLevel` | `none` / `watch` / `warning` / `critical` (clinical; `watch` = flagged, not page-worthy) + `quality` (unreadable trace, technical) |
| `concerns` | `list[Concern]` | severity-ranked |
| `confidence` | `str` | `high` / `medium` / `low` |
| `trend` | `Trend` | vs `previous` epoch |
| `features` | `FeatureSummary` | baseline, variability, decel types, etc. |
| `quality` | `QualityReport` | accepted channels, usable fractions, notes |
| `guideline_pack`, `pack_version`, `library_version` | `str` | **reproducibility/audit trail** |
| `alert_score`, `warnings` | | raw score + epoch warnings |

`.to_dict()` is JSON-serializable; `.summary()` gives a human-readable digest. Every output is **reproducible** from `(input + pack version + library version)` — the foundation for post-market investigation and regulatory traceability.

Example `.summary()` output (from the reference implementation):

```
Category 3  |  ALERT: CRITICAL  |  confidence: high
trend: worsening  |  pack: figo v2015
concerns:
  - [high] Recurrent late decelerations (...): Strongest recurrent hypoxia pattern...
  - [info] Oxytocin: contextual risk: oxytocin
```

---

## 8. Plotting

`ctgkit.plot(signal, result=None, save_path=..., show=...)` renders a clinician-style two-panel trace:

- **FHR panel** — signal, baseline line, shaded 110–160 normal band, concern windows highlighted by severity, and a colored alert banner (green/amber/red).
- **Toco panel** — uterine activity (shown only when present).
- **Concern list** — printed beneath the figure.

`matplotlib` is an **optional** dependency (lazy-imported), so the analysis core runs without it.

---

## 9. Regulatory framing (build it in from day one)

Software that analyzes fetal monitoring signals to detect/predict compromise is, in most jurisdictions, likely to be treated as **medical device software / SaMD** — *even framed as decision support* — because its intended purpose is medical detection during labor. The library is therefore designed to make the regulatory position defensible:

- **Traceability** — every output reproducible from stored inputs + versioned pack + versioned library.
- **Explainability** — structured concerns with evidence, supporting independent clinician review.
- **Human-in-the-loop** — outputs are advisory; the design assumes acknowledge / agree / disagree / defer workflows feeding quality monitoring.

Baseline lifecycle expectations to plan for (not bolt on): IEC 62304 (software lifecycle), ISO 14971 (risk), IEC 62366-1 (usability), IMDRF/EU clinical evaluation, and FDA/IEC 81001-5-1 cybersecurity. Jurisdiction is currently **unspecified** and determines the default pack and classification path — flagged as an open question below.

---

## 10. Roadmap

| Phase | Scope | Status in this spec |
|---|---|---|
| **v0.1 (this)** | CSV/array ingest, epoch validation, quality gate, deterministic feature extraction, 4 guideline packs, orthogonal alert scorer, structured concerns, plotting, tests | **Implemented & runnable** |
| **v0.2** | YAML/JSON externalized guideline packs; hospital policy overlays; per-epoch state machine for true rolling (1–5 min) updates beneath the 30-min contract | designed-for (packs are already pluggable) |
| **v0.3** | ML layer: signal-quality model, morphology detector (trained on CTU-UHB annotation set / FHRMA), Category-2 prioritizer, risk calibration. Hybrid, never replacing the rules spine | future |
| **v0.4** | Streaming/edge deployment adapters, audit log sink, post-market analytics hooks | future |

### Data strategy for the ML layer (v0.3+)

Per the source analysis: prototype on **CTU-UHB** (552 recordings, 4 Hz, PhysioNet) + the **CTU-UHB annotation dataset** for event morphology; add **FHRMA** and **CTGDL** for pretraining; then calibrate on **site/partner data**, because cross-center generalization is the central hard problem in CTG AI. Public raw intrapartum datasets are too small/single-center for production alone.

---

## 11. Open questions (materially affect design)

1. **Regulatory jurisdiction** — sets the default guideline pack and the device-classification path.
2. **Available labeled local data** — the single biggest determinant of how ambitious the ML layer can be. Without it, the strong rules engine + small ML for quality/events is the right product; defer risk-model calibration to silent prospective data.
3. **Target hardware / data interface** — exported waveforms → batch design is fine; true bedside monitoring → stream-native, edge-capable, with a heavier validation burden.

---

## 12. Package layout

```
ctgkit/
├── __init__.py        # public API: analyze, analyze_service, plot, load_csv, ...
├── version.py
├── io.py              # Signal container, CSV loader, 30-min epoch contract
├── preprocess.py      # range filter, spike/gap handling, toco artifact, quality gate
├── features.py        # baseline, variability, accel/decel/contraction detectors
├── guidelines.py      # FIGO / NICE / ACOG / SOGC packs (versioned, pluggable)
├── alerts.py          # orthogonal alert scorer + concern generation
├── pipeline.py        # analyze() orchestration
├── service_config.py  # analyze_service() / ServiceConfig — recommended posture
├── plotting.py        # optional matplotlib trace (ctgkit.plot wraps plotting.plot_epoch)
├── synth.py           # synthetic CTG generator (testing/demo only)
tests/
├── helpers.py         # builders for controlled synthetic traces
├── test_smoke.py      # core invariants incl. "bad signal never reassures"
└── test_behaviour.py  # behavioural suite (prolonged-by-duration, toco gating, service config, ...)
.github/workflows/ci.yml   # runs `python -m pytest -q` on Python 3.9-3.12
setup.py
```

Run the tests with:

```bash
python -m pytest -q
```

---

*Reference implementation status: end-to-end runnable; **53/53 tests pass** (`python -m pytest -q`); verified across normal, tachycardia, recurrent-late-decel (with and without preserved compensation), prolonged-decel, low-variability, signal-loss, and degraded-toco cases, plus all four guideline packs, CSV round-trip, plotting (Signal or CSV path, repeated calls), the recommended service config, and tachysystole confidence under poor toco.*

---

## 13. Review outcomes (v0.1.1)

A clinical/engineering review surfaced five issues. Status:

### Fixed — these were correctness/safety bugs

**(1) Signal quality was too permissive, and confidence was measured *after* interpolation.**
This was the most serious bug: scattered dropout (41% raw usable) was interpolated to ~100% and reported as `high` confidence, defeating the "bad signal never reassures" guarantee. Fixed in `preprocess.py`:
- Acceptance and confidence are now judged on the **raw usable fraction** (before interpolation). `QualityReport` now carries `raw_usable_fraction`, `usable_fraction` (cleaned), and `interpolated_fraction` separately.
- Threshold raised: FHR channel accepted only at **≥0.80 raw usable** (was 0.50).
- Heavy interpolation (>20% of samples) downgrades confidence one notch.
- Verified: 41% raw → `confidence: low`, `category: None`.

**(2) Prolonged decelerations could be missed when also variable-shaped.**
`dtype` was a single field, so a long dip typed `variable` never set `prolonged`. Restructured `Deceleration` so **morphology and duration are independent axes**: `morphology` (early/late/variable/uncertain) is separate from the duration-driven flags `prolonged` (≥2 min), `prolonged_severe` (≥5 min), and `severe_depth` (≥60 bpm). Prolonged is now detected by duration alone, regardless of shape or toco availability. Verified: a 6-min variable-shaped dip is Category 3 / critical across all four packs.

### Acknowledged — tracked for v0.2+, not faked in v0.1

**(3) Guideline rules should be external, versioned config, not hardcoded Python.**
Agreed. The packs are already isolated behind a pluggable `GuidelinePack` interface, which makes the migration mechanical. v0.2 will externalize thresholds to versioned files so clinicians can audit/tune them and each case records the exact rule version used:
```
rules/
  acog_nichd_3tier.yaml
  nice_ng229.yaml
  figo_2015.yaml
  <site>_low_noise_alerts_v1.yaml   # hospital policy overlay
```
The Python packs become thin interpreters of these files. (Not implemented in v0.1 — doing it properly means a schema + validation, which is a v0.2 deliverable rather than a stub.)

**(4) Feature detectors are clinically reasonable but unvalidated.**
Stated plainly in the README and here: baseline, MAD-based variability, decel typing, contraction and sinusoidal detection are **prototype-grade, tuned on synthetic traces only**. No benchmark validation against CTU-UHB or annotated clinical data has been done. This is a v0.1 prototype and must not be treated as clinical-grade until validated on real annotated traces (see §10 data strategy).

**(5) Tests were smoke-only.**
Expanded to a behavioural suite (`tests/test_behaviour.py` + `tests/helpers.py`), covering: prolonged-by-duration, prolonged-2-3-min-never-silent, missing-toco-does-not-claim-late, variable-without-toco-is-uncertain, recurrent-late-with-toco, raw-quality-limits-confidence, category-II-stable-quiet vs category-II-worsening-warning, category-III-critical, low-variability duration scaling, tachysystole with/without FHR abnormality, risk stacking, and the epoch contract. One requested case — reduced variability **30 vs 50 min** — can't be expressed in a single 30-min epoch (the guideline timers span multiple epochs); it's correctly a persistence/multi-epoch test, deferred to the v0.2 state machine.

---

## 14. Review outcomes (v0.1.2)

A second review pass surfaced five follow-ups, all fixed. Total suite now **42/42 tests passing** via `python -m pytest -q`.

**(1) `ctgkit.plot` name collision and ergonomics.**
The plotting function `plot` lived in a module also named `plot.py`; once the submodule was imported, `ctgkit.plot` resolved to the *module*, shadowing the callable. Fixed by renaming the module to **`plotting.py`** with the implementation in `plot_epoch`; `ctgkit.plot(...)` is now unambiguously the function. Also: `plot()` now accepts a **CSV path** as well as a `Signal`, and is **safe to call repeatedly** (guarded backend switching; figures are closed after a save-without-show to avoid leaks). Covered by four new tests.

**(2) Usable-TOCO enforcement.**
Decel timing now gates on `clean.quality.accepted_channels` rather than `clean.toco is not None`. A present-but-degraded toco (raw usable < 0.80) is excluded from acceptance, `toco_available` is `False`, and decels fall back to `morphology="uncertain"`, `timing="uncertain"` — no false `late`/`variable` claims. Contraction *counting* (for tachysystole burden) may still use a present channel, but *timing classification* requires a quality-passing toco. Covered by four new tests.

**(3) Meaningful confidence levels.**
Thresholds on raw FHR usable fraction: **high ≥ 0.95**, **medium ≥ 0.80**, **low < 0.80** (low ⇒ not accepted ⇒ `category=None`). Covered by three new tests.

**(4) Stale SPEC test counts** updated (this section and the status line).

**(5) CI command** documented (§12): `python -m pytest -q`.

---

## 15. Review outcomes (v0.1.3)

A third pass added deployment hardening and documentation polish. Suite now **49/49 passing**.

**(1) README version/test counts** refreshed to v0.1.2 → v0.1.3 and 49 tests; stale v0.1.1/31-test text removed.

**(2) `__pycache__` / cache artifacts** excluded from the repo and archive (`.gitignore` covers `__pycache__/`, `*.pyc`, `.pytest_cache/`; packaging strips them with `--exclude`).

**(3) CI matrix** now includes Python **3.10** (full range 3.9–3.12).

**(4) Recommended service configuration** added as a first-class, auditable entry point — `ctgkit.service_config` with `ServiceConfig` and `analyze_service()`. The recommended posture is **`strict_epoch=True`** (refuse off-length epochs rather than risk a falsely reassuring score), an **explicit guideline** (`"acog"` placeholder; site-selected at deploy), and the documented **30 min ± 2** contract. `ctgkit.RECOMMENDED` is a ready-made config. Covered by four new tests.

**(5) Degraded-TOCO → tachysystole lower-confidence.** Decel timing already required a quality-passing toco (v0.1.2). Now tachysystole derived from a present-but-quality-rejected toco is flagged `features.tachysystole_low_confidence=True`, the concern downgrades to `severity=low` ("Possible tachysystole (low-confidence toco)"), and it **contributes 0 to the alert score** — an unreliable contraction count can no longer drive escalation. Documented in the README TOCO note. Covered by three new tests.
