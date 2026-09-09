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
**2026-09-08: telemanom's source is vendored and read, and it corrects five of this
project's own readings.** `third_party/telemanom/`, pinned. The published algorithm
**clips each window to its newest batch** and is therefore causal after its opening
window, so rungs 1c, 1c-i and 1c-ii were built to reproduce a union that does not exist
in the source; `1a+1b` is the faithful arm and D52 is superseded at its premise. The
published precision denominator is **matched events plus unmatched ranges**, not ranges
over ranges, so every precision figure in the ladder is the generous statistic. And the
target is not "~91 candidate ranges" -- inverting Table 2 gives **84 true positives and
12 false positives**, and on MSL, where our population matches the paper's exactly at 36,
**the paper has 2 false positives and this reproduction has 58**. Five mechanisms in the
source are absent here. The corrections are in D53 and `docs/MODELS.md` 26.29 and 26.30, and **work item 9.10 is
pre-registered** (`docs/MODELS.md` 27): a complete faithful port as the reference ceiling,
with a cumulative five-rung ladder beneath it in one load.
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
- [x] Write the new headline - **done 2026-09-03**, from work items 9.7 to 9.9
      together. **Sentinel catches anomalies a limit check can never see**: on SMAP/MSL
      39 of 43 labelled contextual anomalies stay entirely in range, no per-channel
      statistic reaches a flyable alarm rate there, and the forecaster under a dynamic
      threshold operates at 0.68% and catches 10 of 38 - the baseline now being
      improved. The configuration travels with it: `gru-telemanom`, per channel,
      univariate, no commands, **not** the `gru-quantile` that flies today. Applied at
      Objective.md 1.1, README, `docs/RESULTS.md` 7 and `docs/PHASE1_REPORT.md`.
- [ ] (superseded) Write the new headline, from work item 9.7's results and not before. The
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
- [x] Work item 9.9 study 1 stage 1 - **done 2026-09-03**, 164 Class A to ingest and
      165 Class B to read back (`docs/MODELS.md` 26.6, D46). **39 of 43** labelled
      contextual sequences stay strictly inside their channel's training min/max
      against **11 of 61** point, a 72.7-point gap - the population ESA-ADB does not
      have, six and a half times D43's 6/32. Wu & Keogh's triviality critique is
      confirmed for the point class and refuted for the contextual one.
- [x] Work item 9.9 study 1 **stages 2 and 3** - **run 2026-09-03, and both closed
      without a detector comparison** (`docs/MODELS.md` 26.13, 26.16, D47, D48). On
      SMAP/MSL this decision layer cannot be calibrated by either route: the train
      split does not transfer (`rstd` at 50x still alarms on 15.57% of nominal time),
      and a commissioning window short enough to exist makes the 99.9th percentile a
      short-window maximum, which is worse - 32.51% against 10.47%. Both failures have
      one root: the percentile rule needs a long, representative calibration window
      and this dataset provides neither.
- [x] **Work item 9.9 study 1 stage 4 (Route 1) - done 2026-09-03** (`docs/MODELS.md`
      26.18, D49). The published dynamic threshold passes the gate at **0.6838%** where
      both static routes failed, and the forecaster catches **10 of 38** in-range
      contextual sequences - the project's first measurement of the in-limits claim,
      and below the 12-28 predicted. Neither `rstd` nor the range check can be brought
      to that alarm rate at all: `rstd` at **5,000x** its threshold still alarms on
      15.17%, so no comparison against them is available.
- [x] **D6 closed (D49)**: on the only data carrying commands, conditioning on them
      makes the detector **worse** at a matched rate - 6/38 against 10/38 in-range
      contextual, 33/100 against 47/100 overall. Every telemanom-minus-commands figure
      in this project loses nothing by the omission.
- [x] **Work item 9.9 stage 5 - done 2026-09-03** (`docs/MODELS.md` 26.19, 26.20, D50).
      The commissioned hypothesis was untestable: our event-wise rule **is** Hundman's
      overlap rule, in the same code path, after his published pruning. What the run
      measured instead is where the reproduction gap lives. Precision reproduces within
      **11.2** points (76.3% against 87.5%); recall is **35.1** short (44.9% against
      80.0%); and the paper's own pruning ablation reproduces in direction on both
      datasets. **Pruning costs the paper 7.3 points of SMAP recall and nothing on MSL;
      it costs this reproduction 32.3 and 22.2. Pruning triples the gap.**
- [x] **Rung 1, pruning - done 2026-09-03** (`docs/MODELS.md` 26.21, 26.22). The
      divergence was located by reading telemanom's `errors.py` beside ours: their
      extra ladder rung excludes buffered anomaly shoulders, ours does not, so ours is
      always higher and prunes more. Confirmed on **1,721 of 1,721** calls. Fixing it
      recovers **15.3** points of recall (44.9% -> 60.2%) but costs **32.1** of
      precision (76.3% -> 44.2%), against the paper's 80.0/87.5. **Real, material, and
      not the whole gap** - at least one further divergence sits upstream of pruning,
      in candidate-sequence generation. **Not adopted into the source** (D8).
- [x] **Rung 1b - done 2026-09-03** (`docs/MODELS.md` 26.23, 26.24, D51). Cross-window
      tracking is a real divergence, confirmed on **1,721 of 1,721** calls, and worth
      **one candidate range in 147**. Directionally right, numerically negligible.
- [x] **Rung 1c - run 2026-09-03, and its structural check refuted** (`docs/MODELS.md`
      26.25, 26.26). The arm changed **two** things at once - telemanom's forward window
      geometry *and* the unclipped union - so Z4's superset check failed on 40 of 75
      channels and the resulting 69.4% recall is real but **unattributable**. The stop
      fired. Recall 60.2% -> 69.4%, precision 44.5% -> 41.1%, ranges 146 -> **207**
      against the paper's ~91.
- [x] **Rungs 1c-i and 1c-ii - done 2026-09-04** (`docs/MODELS.md` 26.27, 26.28, D52).
      Q4 held 75/75. The decomposition is clean: **aggregation +15.3 points, forward
      geometry -6.1**. The causal arm is the best arm. **The flyable reproduction
      `1a+1b+1c-ii` reaches 75.5% recall against the paper's 80.0%** - within 4.5
      points - at 44.3% precision against 87.5% with **221** ranges against ~91.
      26.25.1's non-causal caveat is **withdrawn on evidence** (D52).
- [x] **telemanom vendored and read - done 2026-09-08**, zero operations (D53,
      `docs/MODELS.md` 26.29, 26.30). The source is pinned at `third_party/telemanom/`
      because five published readings cited `errors.py` line numbers with no copy kept.
      **It clips to the newest `batch_size`** (`errors.py:355-359`), so it is causal and
      the cross-window accumulator unions **disjoint** batches - there is no union over
      thirty overlapping verdicts anywhere in it. **`1a+1b` is the faithful arm and
      `1c-ii` is a departure.** Precision is `TP/(TP+FP)` with TP deduplicated per
      matched event, so `1c-ii` is **74/197 = 37.6%**, not 98/221 = 44.3%. Both numbers
      kept everywhere. The licence is **BSD 3-Clause**, not Apache-2.0.
- [ ] **The target restated: 12 false positives, not ~91 ranges.** `~91` was `80.0/87.5`
      and is derived in no document. On MSL the denominators match the paper's exactly
      (36 to 36) and the gap is **2 against 58**. Five mechanisms in the source are not
      implemented here: a magnitude conjunct, a whole-window bail-out, the two
      `find_epsilon` guards (deviation 8), an inverse pass, and `adjust_window_size`.
- [x] **Work item 9.10 - run 2026-09-08, and its gate refuted** (`docs/MODELS.md` 27.8),
      1 Class A and 165 Class B, weight store **+0**. **G1 failed and the stop fired, and
      the way it failed is the result.** The two gates exercising code in this repository
      reproduce **exactly** - `telemanom.py` as-is at 44/98 and 45/59, and stage 4's swept
      arm at multiplier 0.551, 0.6820%, 47/100 and **10/38**. The two gates depending on
      the study script that was never committed do **not**: 1a+1b measured 57/98 and 110
      ranges against a recorded 59/98 and 146, and 1c-ii reproduced its recall exactly and
      missed its range count by 16. The failure is isolated to the half with no source to
      check against. Populations confirmed at 37 of 98 (LSTM) and 38 of 100 (GRU).
- [ ] **The divergence is located and is not a defect in either implementation**
      (27.8). The lost script built 1a+1b on `channel_ratios`; this port builds it on
      telemanom's own loop. They differ in series-level singleton dropping and in
      `channel_ratios`' 315-step opening suppression, which the measured `error_window`
      of **54 to 432** lets reach past the warm-up. Two faithful readings of the same
      prose differ by 2 events and 36 ranges.
- [x] **The withheld arms released and adjudicated - 2026-09-08, zero new operations**
      (`docs/MODELS.md` 27.9, D54). **The faithful port reproduces the paper's precision
      and does not reproduce its recall**: F sits at **89.5% precision against 87.5%**,
      on **4 false positives against a scaled target of 11**, and at **34.7% recall
      against 80.0%**. That is the inverse of D50, where precision was 11.2 points short
      and recall 35.1. On MSL, the one exactly like-for-like population, **the paper
      catches 25 of 36 with 2 false alarms and this reproduction catches 3 with 1.**
- [x] **The committed port supersedes the lost script (D54).** 1a+1b is now 57/98 and
      62/110 against the recorded 59/98 and 65/146; 1c-ii is 74/98 and 100/237 against
      74/98 and 98/221. Both kept everywhere. From here the port **is** 1a+1b and 1c-ii.
- [ ] **D51 consequence 2 is discharged and the forecaster rungs open.** The candidate
      count now matches - 38 predicted units against 96, 4 false positives against 12 -
      so lookback, cell type, seed ensembles and epoch policy are reachable for the first
      time. **What remains is the forecaster**, by elimination rather than by guess.
- [ ] **S1 and S2 remain unadjudicated for that run** - a defect in the instrument, not
      in the arms - and are **instrumented for the next**: per-channel alarm ranges are
      retained in the artifact and the checks are computed in the script. **S3 held**:
      windows-per-index 3.0-7.2 proportional against 31.0 published. **L1 is settled as
      a property of the data**, not a transcription defect: `tests/test_smap_rungs_port.py`
      builds windows where each mechanism must fire and shows it does.
- [x] **The public-benchmark survey landed - `docs/RESEARCH.md` Part IV, 2026-09-08.**
      Pinet et al. (arXiv:2606.02670, MiLeTS at KDD 2026) find that across eight public
      benchmarks **no cross-channel rupture occurs without an accompanying univariate
      deviation**, and that channel-dependent modelling brings no measurable gain - an
      **independent replication of D42 and D23**, the first this project has. Their test
      is deviation from normal history; D46's is the training min/max a limit check
      actually holds, so the two are compatible and the section says so. **Phase 3's
      physics testbed is now the only venue** for the cross-channel claim, not a
      convenience. Verified citation by citation; what could not be verified is marked.
- [ ] **Work item 9.11 pre-registered, not run** (`docs/MODELS.md` 28): the training
      reproduction. Eight training differences read from `third_party/telemanom/`
      `modeling.py`, `channel.py` and `config.yaml`, and the largest is **T-a**:
      telemanom's `aggregate_predictions` defaults to **`method='first'`** and is called
      with no method, so its forecast is the **single one-step-ahead prediction** - where
      ours averages ten, and `windows.py:214` says telemanom averages them. **Ours
      smooths the residual tenfold.** Also located: `lstm_batch_size` 64 against our 70
      (two different published constants conflated), a random rather than chronological
      validation split, a per-fold sequence budget where the paper trains full epochs,
      **command inputs the paper used and we never have**, an off-by-`n_predictions`
      target length, and weight restoration the paper does not do. **T4 predicts against
      our own D17**: on (-1,1)-scaled data the published absolute `min_delta` may be
      correct, and 20 or fewer of 81 channels should stall. This arm **refits**, so the
      weight store grows by a pre-registered +81 and a 3-channel timing smoke reports
      per-fit wall clock before the full fit starts.
- [x] **Work item 9.11 - run 2026-09-08** (`docs/MODELS.md` 28.7). **T4 held with zero
      stalls in 81 channels**: the published absolute `min_delta` causes none where our
      relative rule stalls six, so **Arm T is the first arm to score the whole 104-sequence
      population**. D17's replacement is necessary on ESA-ADB's ~1e-4 loss and unnecessary
      on (-1,1)-scaled data -- the dimensionless-constants argument holding in both
      directions on the same code. **T1 refuted**: 46/104 (44.2%) against a 55-85 band,
      and 42/98 restricted to Arm F's own population against F's 34. **The published
      training buys eight events and costs four false alarms**, at 83.6% precision
      against the paper's 87.5%. T2 and T5 held; S1 and S2 now adjudicated at 75/75 on
      every rung. The magnitude conjunct bound on **23 indices in 83,780 window-passes**,
      settling L1 as a property of the data.
- [ ] **(!) MSL did not move at all.** Every point of the gain is SMAP's; MSL is 3/36
      before and after. On the one population that matches the paper's exactly the paper
      catches **25 of 36 with 2 false alarms** and this reproduction catches **3 with 2**,
      and eight training changes moved it by zero events.
- [ ] **The next difference is named from the source and not run** (28.7): telemanom
      computes its residual over the **supervised region only** and then replaces the
      first `l_s` smoothed samples with the mean of the first `2*l_s`
      (`errors.py:48-64`), where ours runs the EWMA across the **full** test array
      including the padded warm-up. Our `e_s` enters the scored region carrying smoothing
      state from padded residuals, and `mean_e_s` and `sd_e_s` set every epsilon.
- [x] **The ledger was short by 165 Class B, and it was corrected 2026-09-08.** The first
      full run was killed by the OS for low memory after the fits and before `ops.commit`.
      77 of 78 fits survived on disk; the operations did not. The ledger read 2,446 for
      2026-09 against a true figure of **2,611**. **(!) This line read "Correcting it costs
      1 Class A and has not been done" until 2026-09-09, and it was stale**: the correction
      was made on 2026-09-08 at 1 Class A and 1 Class B, and `docs/MODELS.md` 30.3 and 31.6
      both record the corrected ledger. It is visible in the artifacts as a +1 Class A and
      +166 Class B step between `2026-09-08T201450Z-wi910-port.json` (189 / 2,446) and
      `2026-09-08T210915Z-wi910-port.json` (191 / 2,619) across an intervening run that
      spent 7 Class B. **The month now stands at 196 Class A and 2,974 Class B of 50,000
      each**, read from `runs/smap-msl/_forensics/2026-09-08T220104Z-wi910-port.json` and
      not transcribed. Recorded rather than silently amended, as August's shortfall was.
- [x] **Work item 9.12 (Arm R) - run 2026-09-08** (`docs/MODELS.md` 29.4), 165 Class B,
      weight store +0. **The residual rung changes nothing**: Arm R equals Arm T in every
      cell, so R1 is refuted and it is not the cause. **R2 refuted but the stop is
      discharged on evidence**: the 13 `offset` events scatter from -503 to +427 with
      **none within 250 +/- 40**, so the frames still agree and the classifier was loose.
      **R4 refuted, and it refutes 28.8's own hypothesis**: `max(e_s)` on MSL is **0.7323**
      at the median and **0 of 27** channels fall below the 0.05 floor. The residuals are
      large, not small -- the forecaster is doing badly on MSL, which is a different and
      more ordinary problem. R5 held.
- [ ] **On MSL no named guard fires** in the published window regime - coverage 0,
      sequence cap 0, magnitude 0 - so what silences 23 of 27 channels is downstream, in
      **pruning at `p = 0.13`**. Named as the arm after 30's. An instrument flaw is
      recorded with it: the fallback counter double-counts the inverse pass, so 56% is
      nearer 12%.
- [ ] **Work item 9.13 pre-registered, not run** (`docs/MODELS.md` 30, D55): dimensionless
      guards. Each absolute filter replaced by `mean(e_s) + 1*sd(e_s)`, the multiplier
      fixed in advance and not swept; the coverage and sequence caps left alone because
      they are already scale-free. G1 MSL past 15/36, G2 MSL false positives 6 or fewer,
      G3 SMAP not below 43/68, G4 a superset on every channel and a stop.
- [x] **D55: absolute constants in data units do not transfer.** Three instances in three
      stages of one method - D17's `min_delta` disabling training on ESA-ADB, T4 measuring
      **the same constant correct** on (-1,1) data, and the candidate filters - plus one
      hypothesis measured and refuted. **The toolkit ships dimensionless equivalents**;
      an absolute form is a per-mission override with its provenance attached.
- [ ] **Work item 9.14 pre-registered, not run** (`docs/MODELS.md` 31, D56):
      **`gru-zscore`, a probabilistic forecaster.** The same GRU with a head emitting
      `mu` and `log sigma^2` per channel, a Gaussian NLL loss on nominal data, and the
      statistic `z = (x - mu)/sigma` under D25's unchanged label-free threshold. **`z` is
      dimensionless by construction, so D55 is satisfied structurally** rather than by
      choosing better constants. **This stops chasing Table 2**: the reproduction answered
      what it was asked -- the gap is not the scoring rule, the commands, pruning's rung,
      cross-window tracking, the aggregation, the window regime or the training
      configuration, each measured and closed -- **it is the residual itself.**
      H1 MSL 22/36 or better from A0's 16/36; H2 at 2 or fewer MSL false positives, the
      paper's own; H3 the ESA regression gate at F0.5 0.804, a separate read priced after
      H1 and H2 report; H4 the design claim, that sigma is not approximately constant;
      H5 held-out nominal NLL improving on 70 of 81 channels.
- [ ] **(!) The commission asked for a threshold "calibrated at a fixed nominal rate" and
      31.2 declines it**, because `docs/HARNESS.md` section 1 struck the alarm budget:
      the threshold is a noise floor, not a dial. Calibration is D25's label-free quantile
      and the nominal rate is **reported, never targeted**; the matched-rate figure is a
      comparison device (D41, D44) and is labelled as one.
- [ ] **28.7's OOM is carried into 31.6 as three pre-registered mitigations**: the parent
      releases raw arrays once a job is queued, every completed fit is checkpointed as it
      lands, and a measured free-memory gate refuses to start below 4 GB.
- [x] **Work item 9.14 (`gru-zscore`) - run 2026-09-08, and H4's stop fired**
      (`docs/MODELS.md` 31.8), 165 Class B. **Sigma collapsed to a per-channel constant**:
      coefficient of variation **median 0.0504**, above 0.25 on **11 of 79** channels
      against a prediction of more than half. In 31.5's own words, written before the run,
      the head "learned a global scale, `z` is `|x - mu|` divided by a constant, and the
      arm is the old detector with extra parameters." **MSL 5/36 against A0's 16/36 and
      the paper's 25/36** (H1 refuted), with 27 false alarms against 2 (H2 refuted).
      **H5 held on 79 of 79 channels** - the likelihood objective trained, median best
      epoch 34 of 35, so this is not an optimisation failure. The model could have learned
      a varying sigma and did not.
- [ ] **D48 for the fourth arm running.** At D25's label-free threshold `gru-zscore`
      alarms on **19.41%** of nominal time; swept it saturates at the grid maximum and is
      still 1.18% with 0 of 39 in-range contextual. A constant divisor cannot repair a
      distribution shift. **Stage 4's 10/38 still stands unreplaced.**
- [ ] **(!) A defect in that arm**: the weight store grew by **0**, not the pre-registered
      +81, because `build_zscore` fits through `lstm.train` directly and the cache lives
      in `ForecastDetector.fit`. Arm H's weights are never persisted and it is
      reproducible only from its seed. Two more caught by the smoke first: the relative
      stopping rule raises the bar on a negative loss (fixed sign-safely, identical for
      every non-negative loss, pinned by test), and `Weights` refused a doubled head --
      **a Gaussian head is not representable in `model.bin` version 1** (D30).
- [ ] **Named, not registered**: why sigma stayed constant. A single univariate channel
      gives the likelihood no reason to vary it with state; the multivariate channel set,
      a variance term the loss cannot trivially satisfy, and capacity are each one arm.
- [ ] **The port has no dial**, so matching its alarm rate to stage 4's 0.6838% needs a
      multiplier on epsilon - a deviation from the source, plumbed through and labelled,
      and pre-registered before it is used.
- [ ] **N1 refuted**, so the 38 still has no replacement: the port's nominal-step rate is
      1.5639% against stage 4's 0.6838%, and by 27.3's rule no comparison is drawn.
      **Stage 4's 10/38 stands unreplaced.**
- [ ] (superseded 2026-09-08) **The remaining gap is precision, not recall**: 221 candidate ranges against ~91.
      A question about how many candidates survive, which the four rungs so far have not
      addressed. Named, not run.
- [ ] (superseded) **Rungs 1c-i and 1c-ii, named and not run**: forward windows with clipped
      verdicts, and trailing windows with unioned verdicts, as two arms - so the window
      geometry and the aggregation become separately attributable. **26.25.1's flag
      applies**: the forward geometry is the non-causal half and cannot be shipped, so
      if it carries the recall the advantage is benchmark-only.
- [ ] (superseded) **Rung 1c, named from the source and not run**: telemanom unions each window's
      surviving anomalies across all overlapping windows; we clip each sequence to the
      judged segment, so an index is judged once rather than ~30 times. It changes
      **which indices are anomalous at all**. Candidates stand at **146 against the
      paper's ~91**.
- [ ] **The improvement ladder on the 38**, one lever per pre-registration, each
      measured against stage 4's **10/38** baseline, target published-parity recall
      (~80%) on the in-range population. **D50 names the first lever: pruning**, ahead
      of lookback, cell type, seed ensembles and epoch policy, because those are guesses
      and pruning is measured.
- [ ] **Route B, scoped and not run** (`docs/MODELS.md` 26.17.4): the commissioning
      window's median + k*MAD, a robust small-sample threshold for the case 26.16
      failed. No `k` is chosen.
- [ ] **The in-limits claim is measured but not settled.** 10/38 is 26%, on one
      dataset, with no available comparison against a floor. It is not nothing and it
      is not a vindication.
- [ ] (superseded) **The question stage 1 built a population for.** 39 of 43 labelled
      contextual sequences are genuinely in range (D46), six and a half times
      ESA-ADB's, and no operating point could be obtained to score them at. The
      obstacle is calibration, not detection.
- [ ] **D6 is still open.** The command ablation is correctly wired and ran twice,
      both times at an alarm rate that makes the comparison meaningless.
- [ ] (superseded) stage 2 as originally scoped:
      `gru-quantile` with the command columns as exogenous inputs (D6), against `rstd`
      and a calibrated range check, on the paper's own split, at a matched nominal
      rate, on the 39 and on all 104. Still not a test of the cross-channel thesis -
      Objective.md 9.2 stands.
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
  (605 tests, zero R2 operations).
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
.venv/bin/python -m pytest -q                                    # 605 tests
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
