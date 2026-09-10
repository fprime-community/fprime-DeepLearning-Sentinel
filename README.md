# fprime-DeepLearning-Sentinel

**fprime-sentinel** is a reusable F' (F Prime) flight-software component that **warns before a
limit trips**. A small neural forecaster, trained on a mission's own nominal telemetry, predicts
the watched channels each cycle; sustained divergence between prediction and reality is raised as
a standard F' event. The forecast is conditioned on **every kind of context the telemetry
carries** - a channel's own history, the commands the spacecraft was sent, and the other sensors
on the same subsystem. It warns; it never commands. The detection method is Hundman et al., KDD
2018 (JPL's telemanom); the reusable flight packaging is this project's contribution.

**(!) Re-framed 2026-09-09** (`docs/DECISIONS.md` D57). This line used to read "a reusable F'
component for **cross-channel** anomaly detection ... the signature of a broken cross-channel
relationship". Sensor-to-sensor is now one of three contexts rather than the whole thesis, and
the reasons are in `Objective.md` 1.1 with the conditions they carry. **Earlier than a limit
check, with minimal false alarms, is the claim.**

## Status

**Phase 1 closed - 2026-08-29.** Architecture: **GRU** (`docs/DECISIONS.md` D28). Transfer
validated on an independent spacecraft (D29). What Phase 2 inherits is in
[docs/PHASE2.md](docs/PHASE2.md).

**Phase 2 under way.** The `model.bin` format is frozen at version 1 (D30,
[docs/MODEL_FILE.md](docs/MODEL_FILE.md)) and the C++ inference core in `flight/` matches the
NumPy reference to **1.8e-07** at the flown shape, with the crossing flag exact
(`docs/MODELS.md` 19.8). **The F' component is done** (tag `wi9`): `Sentinel::Monitor` builds in
F' v4.3.0's own Ref and all **11/11** loader refusal codes degrade to the Level 1 statistical
baseline without failing the topology (D32-D37). Work items 9.5 to 9.14 are the science that
followed, and `docs/STATUS.md` section 7 is the ordered list of what remains.

| Phase | Scope | Gate | State |
|---|---|---|---|
| 1 | Python - prove the mathematics | Match or beat a telemanom baseline reproduced on this harness, with a multivariate forecaster, plus evidence-based architecture selection | **CLOSED** (tag `wi7`) |
| 2 | C++ - the flight component | Tests green, flight-rule compliance clean | next |
| 3 | C++ - integration and demo in the F' Ref deployment, **pulled forward** | Limit alarms silent while Sentinel warns, with time-to-limit **measured** | next, and the only venue for the sensor-to-sensor claim |
| 4 | C++ - hardware envelope | Comfortable margins documented | - |
| 5 | C++ - in-flight retraining of a **shadow** model under human approval | No heap after init, no exceptions, and the shadow measurably better before any swap is offered | scoped, [docs/PHASE5.md](docs/PHASE5.md) |

The ten-minute overview - goal, results, learnings, roadmap - is [docs/STATUS.md](docs/STATUS.md).

## Headline results

Every figure is `k/n` read from a committed run artifact under `runs/` (gitignored outputs; the
path is the identifier). `n < 20` denominators are UNDERPOWERED by the harness's own rule. The
gate set is `m1-g8.9.10` (12 channels, 46 anomalies, 32 headline-cell, 48 rare nominal events);
`m1-ss5` is its six-channel subset, always reported beside it. Honest lead is measured from the
first threshold crossing; 0.0 means the detector fires at the labelled event boundary.

| Set | Detector | F0.5 | recall | headline-cell (MVGS) | precision | rare-event FA | nominal-step FA | honest lead | artifact |
|---|---|---|---|---|---|---|---|---|---|
| `m1-g8.9.10` | `rstd` (the floor) | **0.676** (was 0.250, D37) | **34/46** (was 3/46) | **25/32** (was 3/32) | 84/127 (was 6/7) | 3/48 (was 1/48) | 0.024% (was 0.000%) | +0.0 (was -1,512) | `runs/m1-g8.9.10/rstd/2026-09-01T220201Z-70632603.json` |
| `m1-g8.9.10` | `lstm-quantile` | **0.838** | 26/46 | 21/32 | 40/42 | 2/48 | 0.002% | +0.0 | `runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json` |
| `m1-g8.9.10` | **`gru-quantile`** (selected) | 0.804 | **27/46** | **22/32** | 139/157 | **1/48** | 0.001% | +0.0 | `runs/m1-g8.9.10/gru-quantile/2026-08-28T222635Z-6d146f5d.json` |
| `m1-g8.9.10` | `tcn-quantile` | 0.411 | 9/46 | 9/32 | 21/37 | 3/48 | 0.00002% | -4.0 | `runs/m1-g8.9.10/tcn-quantile/2026-08-29T162030Z-c48bd47d.json` |
| `m1-ss5` | `lstm-quantile` | **0.885** | 27/42 | 21/31 | 127/130 | 2/48 | 0.293% | +0.0 | same artifact as the gate row |
| `m1-ss5` | `gru-quantile` | 0.593 | 26/42 | 21/31 | 101/172 | 3/48 | 0.014% | +0.0 | same artifact as the gate row |
| `m1-ss5` | `tcn-quantile` | 0.649 | 16/42 | 13/31 | 108/137 | 3/48 | 0.297% | -0.5 | same artifact as the gate row |

**(!) The `rstd` and `mavg` floor rows were re-scored 2026-09-01 and moved a long way.**
`baselines._rolling` accumulated its prefix sums in float32 and lost the statistic it
computed; corrected, the floor's headline-cell recall goes from 3/32 to **25/32** and its
F0.5 from 0.250 to **0.676**. The forecaster rows are unchanged and were not recomputed --
nothing in their path calls `_rolling`. Every moved figure carries its old value in
parentheses with D37. What this does to the project's central claim is stated plainly in
`docs/RESULTS.md` 1 and `docs/MODELS.md` 21.8: **the claim as it was written is
falsified.** The gate metric's ordering is not: `gru-quantile` still clears the corrected
floor by 0.128 of F0.5, at a third of its alarm rate.

**Transfer - the held-back sets, scored once** (`docs/RESULTS.md` section 6k, D29). Recall is
disabled on `m2-ss1` by design; its scorecard is the adoption number.

| Set | Detector | rare-event FA | nominal-step FA | artifact |
|---|---|---|---|---|
| `m2-ss1` (Mission 2, 424 rare events) | `lstm-quantile` | **4/424 (0.94%)** | **0 / 4,155,841** | `runs/m2-ss1/lstm-quantile/2026-08-29T204415Z-2717441a.json` |
| `m2-ss1` | **`gru-quantile`** | **4/424 (0.94%)** | **0** | `runs/m2-ss1/gru-quantile/2026-08-29T204415Z-6d146f5d.json` |
| `m2-ss1` | `tcn-quantile` | 6/424 (1.42%) | 0 | `runs/m2-ss1/tcn-quantile/2026-08-29T204415Z-c48bd47d.json` |
| `m2-ss1` | `lstm-gru-or` (union) | 8/424 (1.89%) | 0 | `runs/m2-ss1/lstm-gru-or/2026-08-29T204415Z-a9e0d056.json` |
| `m2-ss1` | `rstd` / `mavg` (floors) | **22/424** (was 84/424) / **208/424** (was 122/424) | **0.003%** (was 17.30%) / 0.001% (was 0) | `runs/m2-ss1/{rstd,mavg}/2026-09-01T220510Z-*.json` |
| `m1-g3` (Mission 1 group 3; 11 anomalies, 13 rare - UNDERPOWERED) | `lstm-quantile`, `gru-quantile` | 8/13, 10/13 | **29.9%, 28.7%** - fold 0 clean (0 steps), folds 1-2 a calibration collapse (87% of one window) | `runs/m1-g3/{lstm-quantile,gru-quantile}/2026-08-29T223625Z-*.json` |

## What is claimed, and what is retired

From [Objective.md section 1.1](Objective.md), which governs every figure quoted anywhere:

- **Retired: "+26 timesteps of early warning."** That figure was dated from the start of an alarm
  range that `error_buffer` had widened backwards from a crossing that had already happened.
  Measured from the first crossing, the median lead is **0.0** on the gate set (D21,
  `docs/RESULTS.md` 6f). No wall-clock claim - no hours, no "~4h" - is made from Phase 1 evidence.
- **The claim, restated 2026-09-03:** **Sentinel catches anomalies a limit check can never
  see.** On NASA's SMAP/MSL telemetry, **39 of 43** labelled contextual anomalies stay entirely
  inside their channel's historical range; **no per-channel statistic reaches a flyable alarm
  rate there** (a rolling standard deviation at 5,000x its threshold still alarms on 15.17% of
  nominal steps); and the forecaster **under a dynamic threshold** operates at **0.68%** and
  catches **10 of 38** -- the baseline now being improved (D46, D48, `docs/MODELS.md` 26.18).
- **(!) The figure is a measured ceiling, not a work in progress (D62, 2026-09-09).** Nine
  improvement arms have been measured against that 10 of 38 -- commands, dimensionless
  guards, a probabilistic head, a 32-configuration label-free forecaster grid, a
  transition-aware floor, per-channel calibration, a 3-seed ensemble and gradient-boosted
  trees -- and **none beats it**; the best reaches 8/38 and four cannot reach its alarm rate
  at all. The pipeline is frozen on the configuration that produced it.
- **(!) Read the configuration with it.** That result is `gru-telemanom`, the published
  **dynamic** threshold, scored **per channel, univariate, without commands** -- **not** the
  `gru-quantile` configuration that flies today. The component ships both rules and a mission
  selects one, because which rule is needed is a property of the regime, not the method. And
  10 of 38 is **26%**, on one dataset, with no floor available to compare against there.
- **Subordinate, and why both rules ship:** on ESA-ADB's stationary folds a calibrated
  per-channel range check is **sufficient and better** -- 34/46 and 25/32 at an equal or lower
  alarm rate against 27/46 and 22/32, and the forecaster never speaks first in 53 caught events
  (D44). Full record in `docs/RESULTS.md` 6l and 6m.
- **Also kept:** nothing else in an F' deployment watches the relationships between channels at
  all, and on an independent spacecraft the same recipe alarms on one rare event in a hundred
  and on no nominal timestep (D29). Both still hold.
- **(!) Withdrawn 2026-09-02:** the sentence that used to sit here -- "a forecaster over the
  channel set finds 28 of 32 headline-cell events where a per-channel statistic finds 3" -- is
  withdrawn pending re-measurement (D37, D38). The 3 was an arithmetic defect and is 25; the 28
  was `lstm-telemanom`'s and the flying detector scores 22/32; and the flying detector catches a
  strict subset of the corrected floor's events. The two are compared in `docs/RESULTS.md` 6l:
  the forecaster leads the gate metric on `m1-g8.9.10` (0.804 against 0.676) at a third of the
  alarm rate and trails on `m1-ss5` (0.593 against 0.663). Re-run under work item 9.7.
- **Unmeasured by design:** the break-to-limit-trip lead. ESA-ADB carries no dictionary limits and
  an anonymised clock; that number is Phase 3's, on the F' Ref deployment, on a real clock.
- **Measured against the recipe itself:** on a later period of the same spacecraft (`m1-g3`,
  folds 1-2) the noise floor calibrated on early data sat under 87% of a later window's nominal
  residual. Thresholds are parameters with a provenance, recalibrated in orbit - the first thing
  Phase 2 inherits (D29).

## How to review this repository

Reading order: this README, then [Objective.md section 1.1](Objective.md), then
[docs/RESULTS.md](docs/RESULTS.md) (every number, both channel sets, `k/n`, both figures wherever
a correction moved one), then [docs/NARRATIVE.md](docs/NARRATIVE.md) (what happened in order,
mistakes included), then [docs/DECISIONS.md](docs/DECISIONS.md) (**D1 to D57**: why, what else was
considered, what settled it; superseded entries marked, never deleted).

- [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) is the self-contained account of Phase 1 for a
  newcomer; [docs/INDEX.md](docs/INDEX.md) is one sentence per document.
- The [Releases](https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/releases)
  `wi1` to `wi7` are the milestone tour, one per work item, each linking into the documents at
  that tag. Work items 9.5 to 9.14 are untagged, and are recorded in `CHANGELOG.md` 0.6.1 to
  0.6.14 with the artifact behind every figure.
- **Branches.** `main` carries one commit per project checkpoint - the reviewable snapshot. The
  complete development history, decision by decision, lives on `dev` (tags `wi1`-`wi9`, the
  Releases). All work lands on `dev`; `main` advances only by a new snapshot commit at an
  approved checkpoint. Neither branch is ever rewritten.

No number in any document was written without an artifact under `runs/` to read it from.

## How it works

1. **Normalise: identity.** ESA-ADB is min-max scaled within each channel group; per-channel
   rescaling would erase the amplitude ratios between related channels that the method exists to
   watch (D2; `tests/test_no_per_channel_scaler.py`).
2. **Forecast.** One multivariate model over the channel set - the GRU, two layers of 80, 250-step
   lookback, ten-step-ahead head - predicts every channel from every channel; the forecast for a
   timestep is the mean of the up-to-ten predictions made before it.
3. **Residual.** Per channel, the absolute difference between the forecast and reality.
4. **Smooth and reduce.** An EWMA (span 105) per channel, then the maximum across channels.
5. **Threshold.** One label-free cut: the 99.9th percentile of that statistic over the mission's
   own anomaly-masked nominal fitting window - the measured noise floor, never a chosen alarm
   budget (D25, `docs/HARNESS.md` section 1). The crossing is the emission. Warn-only.

Training runs in PyTorch; scoring and the Phase 2 C++ run from the plain-NumPy reference in
`src/sentinel_models/reference.py`, held to torch at 1e-5 by test.

## Repository map

| Path | What it holds |
|---|---|
| `src/sentinel_data/` | ingest toolkit: Zenodo download, parquet transcode, R2 client with per-attempt operation accounting, manifest |
| `src/sentinel_eval/` | the referee: catalog, streaming reads, labels, grid, splits, metrics, harness, scorecard, tasks, ops ledger, CLI. Never imports a model |
| `src/sentinel_models/` | the players: baselines, the LSTM/GRU/TCN trainer, the NumPy reference, telemanom's detection stack, detectors, registry |
| `src/sentinel_export/` | the `model.bin` writer and the reader that mirrors the flight one; format frozen at version 1 (D30), specified in `docs/MODEL_FILE.md` |
| `scripts/` | analysis and pod scripts; every one that touches R2 refuses the held-back sets and writes its artifact before the ledger |
| `flight/` | the C++ inference core: the GRU forward pass and the frozen decision layer, the `model.bin` reader, and the golden vectors. No exceptions, no STL, no allocation |
| `tests/` | 607 tests at zero R2 operations, including the layering rule, the reference-equivalence assertion and the C++ suite |
| `docs/` | the documents - see `docs/INDEX.md` |
| `third_party/` | telemanom's published source, pinned at commit `2e6c5b6c`, **vendored as evidence and never a dependency** (D53). Nothing imports it and nothing executes it; `docs/TELEMANOM_EXCERPTS.md` indexes every citation into it |
| `runs/` | weights and scorecards, gitignored outputs; never data |

## How to run

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # R2 credentials, only for tasks that read the bucket

# the verification set, zero R2 operations
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_no_list.py
PYTHONPATH=src .venv/bin/python -m sentinel_eval selftest        # oracle 1.0, silent 0
make -C flight test                                              # the C++ core

# one scoring run on the offline fixture, zero R2 operations
PYTHONPATH=src .venv/bin/python -m sentinel_eval run synthetic --detector gru-smoke --detector rstd --no-sweep
```

A Mission-1 task (`run m1-g8.9.10 ...`) costs 15 Class B and 1 Class A operations on the bucket
and scores both paired channel sets from one load. Fitting the production configuration takes
minutes on a GPU or an hour on a laptop; scoring runs on the laptop through the NumPy reference.

## Provenance rules

- **Manifest-addressed reads only, never LIST, never glob**; `scripts/check_no_list.py` and
  `tests/test_no_list.py` enforce it at source level.
- **50,000 Class A and 50,000 Class B operations per calendar month, hard**; a per-run tripwire at
  1,000; every operation counted on a `before-send` hook into `_manifest/ops_ledger.json`, read at
  run start and committed at run end. Every artifact records what it spent.
- **No telemetry on local disk.** Weights and scorecards under `runs/` are outputs and may persist.
- **The held-back protocol.** `m1-g3` and `m2-ss1` were nominated before any decision-layer tuning
  began, refused by every analysis script, and scored exactly once at the close of Phase 1 with
  the predictions committed first (`docs/MODELS.md` section 18). They are spent; the results are
  `docs/RESULTS.md` section 6k, whatever they said.
- **No number enters a document that was not read from an artifact.** Pre-register, keep wrong
  predictions beside their outcomes, report mistakes openly.

## Citation and data

- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. *Detecting Spacecraft
  Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
- ESA-ADB, the European Space Agency Anomaly Detection Benchmark (Airbus Defence and Space, KP Labs,
  ESOC). Zenodo, DOI 10.5281/zenodo.15237121, CC BY 3.0 IGO. Never committed here; staged to the
  project's R2 bucket as checksummed parquet with a provenance manifest (`docs/DATA.md`).

## Licence

Not yet selected. Intended for community release to the F' ecosystem.
