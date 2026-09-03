# Project status and roadmap

**2026-09-01 - Phase 2 in progress - WI8 complete (flight inference core + frozen model
file); **WI9 complete** (the F' component and the Level 1 safe-failure mode, tag `wi9`);
**WI9.5 complete** (the `rstd` correctness fix, and it falsified a published claim);
**WI9.6 complete** -- the corrected floor is not leaking, and the audit found two things
worse than the correction did: the forecaster catches a **strict subset** of the floor's
events on Mission 1 (D38), and the headline cell is **3/32 contextual**, with all three
missed by both detectors (D39). **2026-09-02: the false headline comparison is
withdrawn from all ten documents that carried it** and replaced by a holding note, not
by a new claim -- `docs/RESULTS.md` 6l is the corrected comparison on both channel sets,
and it shows the forecaster leading the gate metric on `m1-g8.9.10` (0.804 to 0.676) and
**trailing on `m1-ss5` (0.593 to 0.663)**, which no document had stated. Work item 9.7
is pre-registered (`docs/MODELS.md` 23). **2026-09-03: work item 9.8 ran, and it is
the most serious result the project has produced.** At a matched alarm rate a
**per-channel range check** built from the fitting window's own min/max catches
**34/46** against the forecaster's 27/46 on the gate set at a *lower* false-alarm
rate, F0.5 0.934 against 0.804 -- and in **0 of 53** caught events does the
forecaster speak first (D44). At every one of those crossings the channel that
raised the alarm is already past 100 sigma of its own training distribution. Under
the definition a limit check would actually hold -- training min/max, not the
0.1/99.9 band -- the contextual class is 6/32 and 18/31, and the flying detector
catches **none** of the gate set's six at its own operating point (D43). **And the
precursor test came back empty**: all 157 of the forecaster's alarm ranges on the gate
set begin inside a labelled span, so its nominal-flagged steps are overhang rather than
independent alarms and there is no precursor population to test (D45). **Studies A
and B ran earlier**, at 1 Class A
and 15 Class B: at a **matched alarm rate the nesting reverses** -- held to the
forecaster's alarm rate the floor finds 7 of 32 headline-cell events, not 25, and the
forecaster catches 20 the floor misses, so D38's strict subset is a calibration
artifact (D41). No cross-channel reduction recovers anything `max` misses, and D23
closes as answered no (D42). **Neither result rehabilitates the contextual claim**:
D39 and D40 stand and the three contextual events are caught by nothing at any
operating point. The checkpoint snapshot to `main` is held until the new headline is
written from 9.7's results.
Phase 1 closed 2026-08-29 (tag
`wi7`). A ten-minute read; every number is read from a named artifact under `runs/`, or
from a test that pins it.**

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
D31, `docs/MODELS.md` 19.8). WI9 complete - `Sentinel::Monitor` builds in F' v4.3.0's own Ref
deployment and Level 1 degrades on all 11 refusal codes without failing the topology (D32 to
D37, `docs/MODELS.md` 20.9). WI9.5, the `_rolling` correctness fix, is next.**

Every figure is `k/n`, read from the artifact named on its row. MVGS is ESA-ADB's
Multivariate/Global/Subsequence class - the headline cell, the cross-channel anomaly class this
project exists to catch (`docs/HARNESS.md` section 1). Any denominator under 20 would be
stamped UNDERPOWERED by the harness's own rule; none appears here.

| Set | Detector | F0.5 | recall | precision | headline cell (MVGS) | rare-event FA | nominal-step FA | artifact |
|---|---|---|---|---|---|---|---|---|
| `m1-g8.9.10` (gate) | `rstd` (the floor) | **0.676** (was 0.250, D37) | **34/46** (was 3/46) | 84/127 (was 6/7) | **25/32** (was 3/32) | 3/48 (was 1/48) | 0.024% (was 0.000%) | `runs/m1-g8.9.10/rstd/2026-09-01T220201Z-70632603.json` |
| `m1-g8.9.10` | `lstm-quantile` | 0.838 | 26/46 | 40/42 | 21/32 | 2/48 | 0.002% | `runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json` |
| `m1-g8.9.10` | **`gru-quantile`** (selected) | 0.804 | 27/46 | 139/157 | 22/32 | 1/48 | 0.001% | `runs/m1-g8.9.10/gru-quantile/2026-08-28T222635Z-6d146f5d.json` |
| `m1-g8.9.10` | `tcn-quantile` | 0.411 | 9/46 | 21/37 | 9/32 | 3/48 | 0.00002% | `runs/m1-g8.9.10/tcn-quantile/2026-08-29T162030Z-c48bd47d.json` |
| `m2-ss1` (transfer, 424 rare events) | **`gru-quantile`** | - | - | - | - | 4/424 (0.94%) | 0 / 4,155,841 | `runs/m2-ss1/gru-quantile/2026-08-29T204415Z-6d146f5d.json` |

**(!) The `rstd` and `mavg` floor rows were re-scored 2026-09-01 and moved a long way.**
`baselines._rolling` accumulated its prefix sums in float32 and lost the statistic it
computed; corrected, the floor's headline-cell recall goes from 3/32 to **25/32** and its
F0.5 from 0.250 to **0.676**. The forecaster rows are unchanged and were not recomputed --
nothing in their path calls `_rolling`. Every moved figure carries its old value in
parentheses with D37. What this does to the project's central claim is stated plainly in
`docs/RESULTS.md` 1 and `docs/MODELS.md` 21.8: **the claim as it was written is
falsified.** The gate metric's ordering is not: `gru-quantile` still clears the corrected
floor by 0.128 of F0.5, at a third of its alarm rate.

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
- A defect in the floor is a defect in every comparison drawn against it. `baselines._rolling`
  accumulated in float32 and lost the statistic; corrected, the floor went from F0.5 0.250 to
  0.676 and from 3/32 to 25/32 headline-cell events, and the project's most-quoted sentence --
  a per-channel statistic finds three, a forecaster finds twenty-eight -- turned out to be
  about arithmetic (D37). The forecaster still wins the gate metric; the ratio it was
  advertised with was never the gate metric.
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
- [x] Level 1 safe failure mode, Phase 2's first obligation - a corrupt file, CRC or version
      mismatch degrades to the statistical baseline with an event, never failing the topology
      (Objective.md 14.10, D5) - **done 2026-09-01**: all **11/11** refusal codes degrade to
      BASELINE with the code named in the event, **0/11** fail the topology, and the component
      served 200 ticks after a refusal in test. Watched working in a live deployment, not only
      asserted (`docs/MODELS.md` 20.11).
- [x] F' component skeleton - ports, telemetry, the warning event naming the channel - **done
      2026-09-01**: `Sentinel::Monitor`, passive on a `Svc.Sched` tick (D32), builds in this
      project's own deployment **and** in F' v4.3.0's own Ref, which moved to
      `TestDeploymentsProject/Ref` (`docs/MODELS.md` 20.2 correction 3). Consumed as an F'
      library, so a mission adopts it with one `library_locations` line and nothing copied.
- [x] `_rolling`'s float32 accumulation, corrected and re-scored (work item 9.5, D37,
      `docs/MODELS.md` 21) - **done 2026-09-01**, at 41 Class B and 3 Class A. The floor is
      republished everywhere from new artifacts with the old figure beside it. **The
      pre-registered falsification fired**: the corrected floor's headline-cell recall is
      **25/32**, past `gru-quantile`'s 22/32, so the claim that a per-channel statistic
      cannot see the cross-channel class is dead. On the gate metric D3 fixed in advance --
      event-wise F0.5, never bare recall -- `gru-quantile` still clears the corrected floor
      0.804 to 0.676, at a third of its alarm rate (`docs/RESULTS.md` 1a,
      `docs/MODELS.md` 21.10).
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
- [x] Withdraw the false headline comparison from every document that carried it -
      **done 2026-09-02**. The ten sites that asserted "28 of 32 ... where a per-channel
      statistic finds 3" as live now carry a holding note and keep the withdrawn sentence
      in quotation. `docs/RESULTS.md` 6l is the corrected comparison on both sets, and
      `tests/test_documents_are_current.py` keeps the retired figures from reappearing.
- [ ] Write the new headline, from work item 9.7's results and not before. The
      early-warning claim was retired at Objective.md 1.1 and the contextual-class claim
      is now in the same position (D39). No replacement claim is stated meanwhile: the
      measured state of the current decision layer is recorded in `docs/RESULTS.md` 6l
      and `docs/MODELS.md` 22, and it is not a front-page claim.
- [x] Work item 9.7 studies A and B - **done 2026-09-02** at 1 Class A and 15 Class B,
      one bundle load, cached weights, nothing refitted (`docs/MODELS.md` 23.15, D41,
      D42). Seven of nine predictions settled; M4 and M9 are unresolved because the
      sweep recorded nominal-step rate and not event-wise precision, a defect in the
      script named in 23.15.
- [x] Work item 9.7 studies C and D scoped with costs, not built - **done 2026-09-02**
      (`docs/MODELS.md` 23.14). Every leave-one-out form costs about C times the
      arithmetic; an injected set may never be stored as values under Rule 1.
- [x] Close M4 and M9 - **done 2026-09-03** in work item 9.8's single load
      (`docs/MODELS.md` 24.9). M4 refuted: neither detector catches a contextual event
      even swept to precision 0.5. M9 held on the gate set, closed analytically -- no
      Study B arm can reach F0.5 0.804 at any precision. `l2`'s F0.5 on `m1-ss5` is
      still uncomputed and is the one piece left.
- [x] Work item 9.9 study 2, the precursor test - **done 2026-09-03** at 1 Class A and
      15 Class B (`docs/MODELS.md` 25.8, D45). S2 refuted on both sets and the study
      failed on power rather than effect: 0 and 6 eligible alarm starts against a
      predicted 20. The permutation was verified on planted signal first.
- [ ] Where the forecaster's 157 alarm ranges begin, split between anomaly and
      rare-event spans. Not instrumented; one bundle load (D45 consequence 3).
- [ ] Work item 9.9 study 1, SMAP/MSL - staged. Stage 1 is ingest plus the per-event
      visibility diagnostic only: of the labelled contextual anomalies, how many stay
      inside their own training min/max. Stage 2 runs only on approval. Not a test of
      the cross-channel thesis - Objective.md 9.2 stands.
- [x] Work item 9.8, parts 1 to 4 - **done 2026-09-03** at 1 Class A and 15 Class B
      (`docs/MODELS.md` 24.9, D43, D44, `docs/RESULTS.md` 6m). Two of the three
      predictions written against interest fired.
- [ ] A scoring set whose events are selected for the contextual property rather than
      assumed to have it (D39 consequence 5). ESA-ADB may not contain one at `n >= 20`.
- [ ] Open decisions carried (`docs/PHASE2.md` section 6) - D14, the weight-cache key is still
      positional; D21 first reach, `error_buffer`'s effect on alarm width; D23, the decision
      layer is channel-blind and k-of-n was never re-derived; D6, the command-conditioning
      ablation has never run on trained weights; and Objective.md 13 item 8, the injected-fault
      sensitivity study.

## 8. Where things live

- Code: `src/sentinel_data` (ingest), `src/sentinel_eval` (the referee; never imports a model),
  `src/sentinel_models` (the players, the NumPy reference and the Level 1 baseline reference),
  `src/sentinel_export` (the `model.bin` writer and reader), `flight/` (the C++ inference core,
  the Level 1 baseline and their golden vectors), `fprime/` (the F' library: the component, a
  deployment, and `settings.ini`; the framework checkout and tool venv under it are gitignored
  and rebuilt by `scripts/fprime_setup.sh` - see `docs/FPRIME.md`), `scripts/`, `tests/`
  (508 tests, zero R2 operations).
- Data: R2 bucket `fprime-sentinel-data`, manifest-addressed reads only, never LIST; ceiling
  50,000 operations per class per month, tripwire 1,000, every operation in the ledger
  (`docs/DATA.md`). No data is ever committed to the repository.
- Outputs: `runs/`, gitignored; every number in every document is read from an artifact there.
- Documents: `docs/INDEX.md` is the map, one sentence per document; `docs/PHASE1_REPORT.md` is
  the full Phase 1 story; `docs/PHASE2.md` is what the C++ phase inherits;
  `docs/MODEL_FILE.md` is the normative file format.
- History: all work lands on `dev` (tags `wi1`-`wi9` and their Releases); `main` advances only
  by one snapshot commit per approved checkpoint; neither branch is ever rewritten.

## 9. Verify in four commands

```bash
.venv/bin/python -m pytest -q                                    # 508 tests
.venv/bin/python scripts/check_no_list.py
PYTHONPATH=src .venv/bin/python -m sentinel_eval selftest        # oracle 1.0, silent 0
make -C flight test                                              # the C++ core and Level 1
make -C flight lint                                              # clang-tidy, three configs

# the F' half, after one-time setup with scripts/fprime_setup.sh
cd fprime && . fprime-venv/bin/activate
cd Sentinel/Monitor && fprime-util check                         # the component, 10 tests
cd ../../SentinelRef && fprime-util build                        # the deployment
bash scripts/fprime_ref_patch.sh                                 # and F's own Ref

# one scoring run on the offline fixture, zero R2 operations
PYTHONPATH=src .venv/bin/python -m sentinel_eval run synthetic --detector gru-smoke --detector rstd --no-sweep
```

Everything above costs zero bucket operations; a Mission-1 scoring task costs 15 Class B and
1 Class A.
