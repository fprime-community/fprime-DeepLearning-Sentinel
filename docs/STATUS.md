# Project status and roadmap

**2026-09-01 - Phase 2 in progress - WI8 complete (flight inference core + frozen model
file); WI9 next. Phase 1 closed 2026-08-29 (tag `wi7`). A ten-minute read; every number is
read from a named artifact under `runs/`, or from a test that pins it.**

## 1. Goal

A reusable F' (F Prime) flight-software component that watches the relationships between
telemetry channels and warns before a limit trips - hours earlier is the design target Phase 3
measures; Phase 1 makes no wall-clock claim (Objective.md section 1.1). It warns; it never
commands. The detection method is Hundman et al., KDD 2018 (JPL's telemanom); the reusable
flight packaging is this project's contribution.

## 2. Why it matters

- Hundman et al., studying real expert-confirmed anomalies from the SMAP satellite and the
  Curiosity rover, found 41% were contextual - every channel inside its limits while the
  combination or the trajectory was wrong (Objective.md section 2.3).
- Onboard protection today is a per-channel limit check - a thermostat; nothing on the
  spacecraft watches the relationships between channels (Objective.md section 2.1).
- The method was published in 2018 and no reusable flight component followed; that packaging is
  this project's contribution.

## 3. How it works

1. Normalise: identity - per-channel rescaling would erase the amplitude ratios between related
   channels that the method exists to watch (D2).
2. Forecast: one multivariate GRU over the channel set - two layers of 80, 250-step lookback,
   ten-step-ahead head, mean-aggregated - predicts every channel from every channel.
3. Residual: per channel, the absolute difference between forecast and reality.
4. Smooth and reduce: an EWMA (span 105) per channel, then the maximum across channels.
5. Threshold: one label-free cut - the 99.9th percentile of that statistic over the mission's
   own anomaly-masked nominal window, the measured noise floor, never an alarm budget (D25).
   The crossing is the emission. Warn-only.

Training runs on the ground in PyTorch. Flight inference is C++ transcribed from the NumPy
reference in `src/sentinel_models/reference.py` - static memory, no exceptions. The model is
frozen in flight; nothing learns online (Objective.md section 11). Thresholds are recalibrated
in orbit without retraining (Objective.md 14.10).

## 4. Where we are

**Phase 1 CLOSED 2026-08-29 (tag `wi7`). Architecture: GRU (D28). Transfer validated on an
independent spacecraft (D29). Phase 2 in progress: WI8 complete (tag `wi8`) - the flight
inference core matches the reference at 1.8e-07 and the model file format is frozen (D30,
D31, `docs/MODELS.md` 19.8). WI9, the F' component, is next.**

Every figure is `k/n`, read from the artifact named on its row. MVGS is ESA-ADB's
Multivariate/Global/Subsequence class - the headline cell, the cross-channel anomaly class this
project exists to catch (`docs/HARNESS.md` section 1). Any denominator under 20 would be
stamped UNDERPOWERED by the harness's own rule; none appears here.

| Set | Detector | F0.5 | recall | precision | headline cell (MVGS) | rare-event FA | nominal-step FA | artifact |
|---|---|---|---|---|---|---|---|---|
| `m1-g8.9.10` (gate) | `rstd` (the floor) | 0.250 | 3/46 | 6/7 | 3/32 | 1/48 | 0.000% | `runs/m1-g8.9.10/rstd/2026-08-25T*.json` |
| `m1-g8.9.10` | `lstm-quantile` | 0.838 | 26/46 | 40/42 | 21/32 | 2/48 | 0.002% | `runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json` |
| `m1-g8.9.10` | **`gru-quantile`** (selected) | 0.804 | 27/46 | 139/157 | 22/32 | 1/48 | 0.001% | `runs/m1-g8.9.10/gru-quantile/2026-08-28T222635Z-6d146f5d.json` |
| `m1-g8.9.10` | `tcn-quantile` | 0.411 | 9/46 | 21/37 | 9/32 | 3/48 | 0.00002% | `runs/m1-g8.9.10/tcn-quantile/2026-08-29T162030Z-c48bd47d.json` |
| `m2-ss1` (transfer, 424 rare events) | **`gru-quantile`** | - | - | - | - | 4/424 (0.94%) | 0 / 4,155,841 | `runs/m2-ss1/gru-quantile/2026-08-29T204415Z-6d146f5d.json` |

**(!) The `rstd` floor row is under correction.** `baselines._rolling` accumulates its
prefix sums in float32 and loses the statistic it computes (D37, `docs/MODELS.md` 20.6):
measured 7.6584e+00 of error on a true sigma of 3.0, and 2,852 spurious exact zeros on
this project's own fixture. The figures below are what the artifact says and are left
standing; the repair and the re-score are scoped in `docs/MODELS.md` 21 and run before
work item 10. Nothing in the forecaster rows calls `_rolling`.

On the `m2-ss1` row "-" means unmeasured: recall is disabled on the transfer set by design, so
its scorecard is the adoption number (D29, `docs/RESULTS.md` 6k). The paired subset `m1-ss5`,
the union and floor rows on Mission 2, and the `m1-g3` folds are in README and
`docs/RESULTS.md` 6h to 6k.

## 5. What we did, and why

- [x] Stood the repository up documentation-first - decisions recorded before code: D1 to D29,
      superseded entries marked, never deleted (`docs/DECISIONS.md`).
- [x] Ingested ESA-ADB - 3 real ESA missions, 11.53 GB in 234 checksummed objects in R2
      (CHANGELOG 0.2.0) - because real labelled failures are the only way to prove catching.
- [x] Built the evaluation harness first - an honest referee before any player; it caught five
      defects in our own measurement design before a single model trained (`docs/NARRATIVE.md`
      section 2).
- [x] Reproduced the published LSTM baseline faithfully, then corrected it - the published
      `min_delta` had kept every fit at its first epoch (about fortyfold better fixed, D17) and
      the published dynamic threshold degenerated under the better forecaster (D18).
- [x] Froze the decision layer - one calibrated global quantile; whitening built, measured,
      declined: the quantile matched it event for event at a fourteen-times-lower nominal-step
      rate (D25).
- [x] Trained the GRU - seven of eleven weak events recovered on fold 0 by an elevenfold lower
      noise floor, six lost on fold 1; the only architecture with no stall on any fold (D26,
      `docs/RESULTS.md` 6h and 6i).
- [x] Built the TCN, size-matched, under its own pre-registration - measured: 9/46, a strict
      subset of both cells, floor above both (built D27, declined at the gate D28).
- [x] Ran the architecture gate - criterion 1 a tie inside the resolution the harness refuses
      to call a finding; the GRU selected on criteria 2 to 5 and trainability, the smaller,
      cheaper, simpler, single-state cell that did not stall - 71,160 parameters, 22.35%
      smaller (D28).
- [x] Scoped the LSTM+GRU union - an OR and nothing cleverer - and sealed-tested it: declined,
      its 8/424 exactly the members' 4 + 4, disjoint, past the pre-registered 1.5x line (D29).
- [x] Ran the held-back transfer exam once - the recipe transfers across spacecraft (4/424 rare
      false alarms, 0 nominal-step alarms in 4,155,841) and a floor fixed in the past collapsed
      on a later period of the same spacecraft (D29).
- [x] Checkpointed the repository - `main` carries one commit per checkpoint, the reviewable
      snapshot; the complete history lives on `dev` (tags `wi1`-`wi7`, the Releases).

## 6. What we learned

- Faithful reproduction produced the bug: a published constant silently disabled training, and
  every fit kept its first epoch (D17).
- The published dynamic threshold degenerates under a good forecaster: of 5,684,580 windows,
  92.6% of those that selected a z chose the range minimum - `mu + 2.5*sigma` (D18).
- A calibrated global quantile matched the whitened relationship test event for event, fourteen
  times quieter per nominal step and simpler to fly (D25).
- "+26 timesteps early" was an artifact; measured from the first crossing, the flying
  detector's median lead is 0.0 - true early warning is Phase 3's measurement (D21,
  Objective.md 1.1).
- Thresholds are parameters with a provenance: the recipe held across spacecraft, and a floor
  fixed in the past sat under 86.7% of one later window's nominal residual - recalibrate in
  orbit (D29).

## 7. What's next

PHASE 2 - flight C++ (gate: tests green, flight-rule compliance clean)

- [x] GRU inference in C++ from `src/sentinel_models/reference.py` - static memory, no
      exceptions - **done 2026-09-01**: `flight/` matches the NumPy reference at
      **1.8e-07** worst case over seven weight sets including the four cached production
      fits, with the crossing flag exact on every step (`docs/MODELS.md` 19.8).
- [x] `model.bin` format frozen - gate order named in the header, both bias vectors unsummed,
      normalisation and thresholds outside the weights as PrmDb-style parameters, replaceable
      without retraining (Objective.md 14.10) - **done 2026-09-01** (D30,
      `docs/MODEL_FILE.md`): seven files round-trip Python to C++ to Python byte-identically,
      and the parameter block carries its own CRC so a recalibration in orbit never touches
      the 278.0 KiB of weights.
- [ ] Level 1 safe failure mode, Phase 2's first obligation - done when a corrupt file, CRC or
      version mismatch degrades to the statistical baseline with an event, never failing the
      topology (Objective.md 14.10, D5). **Half built**: the loader refuses a bad magic,
      version, header CRC, static CRC, param CRC, shape, size or truncation with its own
      status code and no exception (16 cases), and a refused model emits nothing. The
      degrade-with-an-event is the F' component's and is open.
- [ ] F' component skeleton - ports, telemetry, the warning event naming the channel - done
      when it builds in an F' Ref deployment.
- [ ] `_rolling`'s float32 accumulation, corrected and re-scored (work item 9.5, D37,
      `docs/MODELS.md` 21) - done when the floor is republished from a new artifact with the
      old figure beside it. Runs before the recalibration path, because the floor is the
      denominator of the headline comparison.
- [ ] In-orbit threshold recalibration path - file uplink, human-approved reload - done when it
      is exercised end to end on the Ref.

TOOLKIT - the mission-facing training package

- [ ] One-command training - healthy telemetry in, `model.bin` out, the guards built in
      (relative early stopping, the first-epoch guard, the anomaly-masked calibration window,
      data-sufficiency grading; Objective.md 10.2) - done when it runs on the synthetic fixture
      with zero cloud operations.
- [ ] Pre-launch sanity report - how often the detector fired on held-out healthy data, a
      sanity report and not a target (`docs/HARNESS.md` section 1) - done when it prints at the
      end of every calibration.
- [ ] Tier ladder (Objective.md 14.10, D5) - Level 1 statistical baseline, Level 2 small
      pretrained model fine-tuned, Level 3 full mission-specific training - done when each tier
      is selectable and documented.
- [ ] Quickstart for a mission engineer with no ML background - done when a newcomer produces a
      `model.bin` from the fixture in under an hour.

PHASE 3 - integration (gate: limit alarms silent while Sentinel warns, with time-to-limit)

- [ ] Real clock, real dictionary limits - done when the break-to-limit-trip lead is measured
      and the early-warning claim is earned or bounded (Objective.md 1.1).

PHASE 4 - hardware envelope (gate: comfortable margins documented)

- [ ] CPU, memory and timing measured on a representative flight-class board (Objective.md
      section 12).

HOUSEKEEPING

- [ ] Licence - "Not yet selected. Intended for community release to the F' ecosystem."
      (README) - done before any public release.
- [ ] Pre-publication sweep of the history for anything sensitive - done before the repository
      goes public.
- [ ] Open decisions carried (`docs/PHASE2.md` section 6) - D14, the weight-cache key is still
      positional; D21 first reach, `error_buffer`'s effect on alarm width; D23, the decision
      layer is channel-blind and k-of-n was never re-derived; D6, the command-conditioning
      ablation has never run on trained weights; and Objective.md 13 item 8, the injected-fault
      sensitivity study.

## 8. Where things live

- Code: `src/sentinel_data` (ingest), `src/sentinel_eval` (the referee; never imports a model),
  `src/sentinel_models` (the players and the NumPy reference), `src/sentinel_export` (the
  `model.bin` writer and reader), `flight/` (the C++ inference core and its golden vectors),
  `scripts/`, `tests/` (473 tests, zero R2 operations).
- Data: R2 bucket `fprime-sentinel-data`, manifest-addressed reads only, never LIST; ceiling
  50,000 operations per class per month, tripwire 1,000, every operation in the ledger
  (`docs/DATA.md`). No data is ever committed to the repository.
- Outputs: `runs/`, gitignored; every number in every document is read from an artifact there.
- Documents: `docs/INDEX.md` is the map, one sentence per document; `docs/PHASE1_REPORT.md` is
  the full Phase 1 story; `docs/PHASE2.md` is what the C++ phase inherits;
  `docs/MODEL_FILE.md` is the normative file format.
- History: all work lands on `dev` (tags `wi1`-`wi7` and their Releases); `main` advances only
  by one snapshot commit per approved checkpoint; neither branch is ever rewritten.

## 9. Verify in four commands

```bash
.venv/bin/python -m pytest -q                                    # 473 tests
.venv/bin/python scripts/check_no_list.py
PYTHONPATH=src .venv/bin/python -m sentinel_eval selftest        # oracle 1.0, silent 0
make -C flight test                                              # the C++ core

# one scoring run on the offline fixture, zero R2 operations
PYTHONPATH=src .venv/bin/python -m sentinel_eval run synthetic --detector gru-smoke --detector rstd --no-sweep
```

Everything above costs zero bucket operations; a Mission-1 scoring task costs 15 Class B and
1 Class A.
