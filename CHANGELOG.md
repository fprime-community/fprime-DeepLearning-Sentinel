# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0 the version
tracks documentation and Phase 1 research milestones rather than a released flight component.

## [Unreleased]

### Planned - work item 10

- The in-orbit threshold recalibration path: file uplink, human-approved reload. The hook is
  noted in the component's FPP and SDD and no code implements it. It makes the component
  `queued` (D32 consequence 2).

## [0.6.6] - 2026-09-03 - Work item 9.9 study 1 stage 1: a contextual population that is real

SMAP/MSL ingested and the visibility diagnostic run. 164 Class A to upload, 165 Class B
to read back; the month stands at 173 Class A and 269 Class B of 50,000 each.

### Added - the dataset, 2026-09-03

- **`smap-msl/v1/` in R2**, 162 arrays and the canonical `labeled_anomalies.csv`, under
  its own `_manifest/smap_msl.json`. `_manifest/manifest.json` is untouched: the ingest
  writes only under `smap-msl/v1/` and its own manifest, because `manifest.py`
  hard-codes the single key `esa-adb` and `Catalog.load` reads it. Recorded in
  `docs/HARNESS.md` 5a's register of authorised additions, additive-only.
- **Every array verified against the canonical labels before upload** -- fetched from
  `khundman/telemanom` directly rather than from the redistribution -- 82 of 82 rows,
  0 mismatches, with the script written to abort before the first put.
- **A labelling defect recorded, not deduplicated silently.** `P-2` appears twice with
  conflicting spans, so the dataset has 81 unique channels rather than the 82 its own
  label file implies. Carried in the manifest's `labelling_defects` field.

### Added - D46, the finding

- **The labelled contextual class is genuinely in range.** 39 of 43 contextual
  sequences stay strictly inside their channel's training min/max, against 11 of 61
  point -- a **72.7-point gap**, in the direction the labels claim, on both spacecraft
  (SMAP 26/26, MSL 13/17). Against D43's 6/32 on ESA-ADB's headline cell this is a
  population six and a half times larger and five times denser.
- **Wu & Keogh's triviality critique, answered precisely.** Confirmed for the point
  class -- 82% breach their range and a one-liner finds them. Refuted for the
  contextual class -- 91% do not. Both halves recorded, because reporting only the
  second would be the selective reading their paper is about.
- **Objective.md 9.2 stands and is better specified.** SMAP/MSL is still unusable for
  the cross-channel claim and is now demonstrably usable for the in-limits claim.
  No figure here may be quoted as cross-channel evidence.

### Predictions - one wrong in the comfortable direction

- **V1 refuted**: 15 to 30 in-range contextual predicted, **39** measured. The band was
  set expecting the labels not to hold up; they held up. Recorded loudest for that
  reason. **V2 refuted by one** (11 of 61 against a predicted 10). **V3, V4, V5 held**,
  V5 being the gate at n >= 20 that decides whether stage 2 is worth running.

### Held - stage 2 does not run without approval

- `gru-quantile` with the command columns as exogenous inputs (D6, open since work item
  4), against `rstd` and a calibrated range check, on the paper's own split, at a
  matched nominal rate. Its falsification will be stated against the measured
  matched-rate multiplier, not "by construction" -- on `m1-ss5` that multiplier was
  0.672, tighter than the training range.

## [0.6.5] - 2026-09-03 - Work item 9.9 study 2: the precursor test came back empty

Pre-registered at `docs/MODELS.md` 25 before a figure existed, then run in one bundle
load: 1 Class A and 15 Class B, cached weights, nothing refitted during the run.

### Added - the precursor test, 2026-09-03

- **D45 -- every alarm the forecaster raises begins inside a labelled span.** All 157
  of `gru-quantile`'s alarm ranges on `m1-g8.9.10` start inside a labelled anomaly or
  rare-event span; **none** begins in quiet, in-range nominal time. Its 142
  nominal-flagged steps are the tails of alarms that started inside a labelled span and
  ran past its end. So its false-alarm count is spent on where alarms *end*, not where
  they begin -- a different defect from the one the metrics imply, and a smaller one.
- **The study failed on power, not on effect, and that is a prediction refuted.** S2
  predicted at least 20 eligible alarm starts per set and measured **0** and **6**. The
  primary test could not run; at n=6 the subset is UNDERPOWERED and no p-value is
  quoted from it, exactly as 25.3 said in advance.
- **The permutation was verified before it was spent**: on planted precursors it
  returns p = 0.009 under the circular-shift null at a rate ratio of 3.98, and 0.72 on
  randomly placed alarms of the same count. The null is the data's, not the test's.
- **No precursor population means the early-warning argument gains nothing here.** D44
  stands: no measured lead over a rate-matched range check, and now no nominal-period
  alarms that could have been early either.

### Known gaps

- **Where those 157 ranges begin** -- the split between anomaly and rare-event spans --
  is not instrumented. One bundle load. Named rather than left to be found.
- **The weight store is 89 files, not the 86 quoted since work item 9.6.** The three
  additions are synthetic-fixture weights written by this section's offline dry runs,
  which pass `--allow-fit` because the fixture has no cache. The in-run invariant held:
  the count was identical before and after the Mission 1 run, so nothing was refitted on
  Mission 1 data. Recorded because a quoted invariant that quietly changes is worse than
  one that moves for a stated reason.

## [0.6.4] - 2026-09-03 - Phase 2 work item 9.8: the yardstick the forecaster does not clear

Pre-registered at `docs/MODELS.md` 24 before a figure existed, then run in one bundle
load: 1 Class A and 15 Class B, cached weights, nothing refitted, weight store unchanged
at 86 files. All six reproduction checks passed before anything new was read.

### Added - work item 9.8, 2026-09-03

- **D44 -- at a matched alarm rate a per-channel range check beats the forecaster on
  both sets, and the forecaster never speaks first.** The envelope is the fitting
  window's own per-channel min/max under `train_mask`, widened until its nominal-step
  rate matches. On `m1-g8.9.10` it catches **34/46 and 25/32 at 0.0000% nominal, F0.5
  0.934**, against `gru-quantile`'s 27/46, 22/32, 0.0013% and 0.804; on `m1-ss5`
  34/42 and 25/31. The catches are nested against the forecaster -- only-GRU **0** at
  the matched point on both sets. And in **0 of 53** caught events does the forecaster
  fire first: median lead +0.0, mean -105.1 and -0.3, with the range check earlier on
  14 of them by up to 2,638 timesteps. This is the most serious result the project has
  produced, and it retires the last proxy the early-warning argument had on Phase 1
  evidence.
- **The mechanism is amplitude.** At every one of those 53 first crossings the channel
  that raised the alarm is already outside 3 sigma of its own anomaly-masked
  fitting-window distribution -- median 103.8 and 102.9, minimum 6.6, none below 3.
  Stated with its caveat: ESA-ADB is min-max scaled per group, so 3 sigma is a low bar
  at this scale. The bar was fixed before the numbers existed and was not moved.
- **D43 -- contextual is the training-window min/max, not the 0.1/99.9 band.** A real
  limit sits outside a channel's historical range, so the tightest bar any limit could
  hold is min/max, and D39 measured a quantity tighter than the claim it was testing.
  The class is **6/32** and **18/31**, reproducing 22.11 and D40 exactly from a
  different script. Over the gate set's six the flying detector catches **none** at its
  own operating point and **one** -- `id_89` -- only when loosened to the floor's alarm
  rate. D39 consequence 2 survives the change of definition; D39 and D40 are re-read and
  neither is edited.
- **`docs/RESULTS.md` 6m**, the per-event tables, and `scripts/reduction_and_curve.py`
  extended with the per-channel bar, the range check, the amplitude z-scores and
  event-wise F0.5 along the sweep.

### Fixed - a defect this section's own cross-check caught

- **The envelope breach was computed with `>=` where the published definition is
  strict.** An event that *touches* a channel's historical extreme has not *left* it,
  and ESA-ADB's per-group min-max scaling makes exact boundary values common: the first
  run disagreed with the floor audit on 2 gate-set and 14 `m1-ss5` events, every one at
  a reach of exactly 1.000. Under the strict rule the counts reproduce the floor audit
  exactly, 11/11 and 24/24. No re-run was needed -- the disagreement can only occur at
  `w = 1.0`, and no matched operating point sits there. The script now carries a
  `STRICT` rule with the reason.

### Closed - carried from work item 9.7

- **M4 refuted**: swept to the alarm rate where event-wise precision first falls below
  0.5, neither detector catches any of the three 0.1/99.9-contextual events on either
  set. **M9 held on the gate set**, closed analytically: event-wise F0.5 at perfect
  precision is `1.25R/(0.25+R)`, so no Study B arm above 9/46 recall can reach 0.804.
  `l2`'s F0.5 on `m1-ss5` is not computed and is named as the one gap.

### Predictions - two of the three written against interest fired

- **Refuted**: N4 (positive median lead -- measured +0.0), N5 (60% breaching after our
  emission -- measured 0 of 53), N6 (raw z inside 3 sigma -- measured 0 of 53), N7 (3
  misses recovered by the per-channel bar -- measured 0 on the gate set), N1 as written.
- **Held**: N2 by one event on a loosened detector, N8, N3 at one of its two matched
  points, M9.

### Held - still not written

- **The headline.** It is written once, from work items 9.7 and 9.8 together, for
  approval. Nothing is written to a claim site meanwhile, and `main` stays held.

## [0.6.3] - 2026-09-02 - Phase 2 work item 9.7: the comparison made like for like

The false headline withdrawn, and the two studies that could be run, run. One bundle
load for both, 1 Class A and 15 Class B, cached weights, nothing refitted, weight
store unchanged at 86 files.

### Changed - the headline comparison is withdrawn, 2026-09-02

- **Ten sites asserted "a forecaster finds 28 of 32 headline-cell events where a
  per-channel statistic finds 3" as live.** All ten are withdrawn, each keeping the
  sentence in quotation. Three things were wrong with it: the 3 was float32
  accumulation and is 25 (D37); the 28 of 32 is `lstm-telemanom`'s, a detector
  disqualified at 22 of 48 commanded manoeuvres, while the detector that flies scores
  22/32; and D38 measured the flying detector's catches as a strict subset of the
  floor's. What replaces them is a holding note, **not a new claim**.
- **What is kept, separately**: nothing else in an F' deployment watches the
  relationships between channels. That is a claim about the ecosystem, verified in
  Objective.md 3, and no correction touches it.
- **`docs/RESULTS.md` 6l** is new -- the corrected floor beside the flying detector,
  which no section carried, because 6h to 6k compare the architectures against each
  other and the correction came later.

### Fixed - a one-sided figure, found while writing 6l

- **The gate-metric result is not the same on the two channel sets and only one had
  ever been quoted.** On `m1-g8.9.10` the forecaster leads 0.804 to 0.676; on
  `m1-ss5` the corrected floor leads **0.663 to 0.593**. The alarm-rate advantage is
  3.71x on the gate set -- the "third of the alarm rate" every document repeated --
  and 1.40x on `m1-ss5`. Both were inside the same artifacts, each carrying both sets.
  D38 consequence 1 gets a rider; the selection does not move, since D28 never
  involved `rstd`.

### Added - work item 9.7 studies A and B, 2026-09-02

- **D41 -- at a matched alarm rate the nesting reverses.** D38 compared the two at
  their own thresholds, where the floor alarms on eighteen times more nominal steps.
  Held to the forecaster's rate the floor finds **7 of 32** headline-cell events, not
  25; allowed the floor's rate the forecaster reaches **26/32** against 25/32. The
  four sets follow: at matched-quiet on the gate set, only-GRU **20** and only-`rstd`
  **0**. At every matched point on both sets only-`rstd` is 0 or 2. **D38's strict
  subset is a calibration artifact**, and its counts remain exactly right for the
  frozen configuration.
- **D42 -- D23 closes as answered no.** `k`-of-`n` re-derived at k=2 and k=3, plus L2
  and sum, under the frozen calibration recipe, all five from one forecast pass so
  the `max` arm **is** `gru-quantile` and reproduces its scorecard. Every arm's catch
  set is a strict subset of `max`'s on both sets; not one recovers any of the 19
  gate-set misses, the seven fold-1 events or the three contextual events, and every
  arm runs at a *lower* nominal rate, so it is not a threshold handicap. The one
  improvement is `l2` on `m1-ss5`: the same 26/42 and 21/31 at 1,124 nominal steps
  against 1,536, 27% fewer -- and the same arm collapses to 9/46 on the gate set. The
  mechanism inverts D23's premise: `max` wins because cross-channel evidence is
  sparse. All four arms are sign-blind, so a signed reduction remains untested.
- **Neither result rehabilitates the contextual claim.** D39 and D40 stand: the
  headline cell is 3/32 contextual and nothing catches the three, at any operating
  point or under any reduction.
- **`docs/MODELS.md` 23.14** costs studies C and D without building either. Every
  leave-one-out form costs about C times the arithmetic, whatever the parameter
  count; a version-1 `model.bin` would load design (b) and silently run it once
  instead of twelve times, which the reserved-field refusal rule closes. And Rule 1
  forbids storing an injected set as values, so D stores a regenerable recipe.

### Known gaps - named, not left to be found

- **M4 and M9 are unresolved.** The sweep recorded events caught and nominal steps
  flagged, not alarm ranges classified, so event-wise F0.5 along the curve was never
  computed. A defect in `scripts/reduction_and_curve.py`, not in the design: 23.8
  asked for both and one was instrumented. One further bundle load closes it.
- **The new headline is not written.** No replacement claim is stated; the measured
  state is in `docs/RESULTS.md` 6l and `docs/MODELS.md` 23.15, and `main` is held.

## [0.6.2] - 2026-09-01 - Phase 2 work item 9.6: auditing the corrected floor

The audit of a result that overturned a thesis, pre-registered at `e31533d` before a
figure was computed. It found the floor clean and two things worse than the correction
had been. Zero new capability; one bundle load, 15 Class B and 1 Class A against a
budget of 20; cached weights, nothing refitted, the weight store unchanged at 86 files.

### Added - work item 9.6, 2026-09-01

- **`docs/MODELS.md` 22**, fourteen predictions L1 to L14 with their falsifications, and
  `scripts/floor_audit.py`. Artifact
  `runs/m1-g8.9.10/_forensics/2026-09-01T230303Z-floor-audit.json`. The audit reproduces
  every published scorecard count exactly -- 27/46, 34/46, 22/32, 25/32 -- before it says
  anything new.
- **The corrected floor is not leaking, and work item 9.5's numbers stand as measured.**
  L1 to L4 all hold: perturbing a future sample changes rows before `t` by exactly 0.0
  with a working control; `fit` sees only the masked training window through the same
  `harness.py:144` line every detector uses; the fallback scale is unreachable; and
  `RollingStd.threshold_from` **is** `Detector.threshold_from`, the same function object
  the GRU's calibration calls. The catches are sustained, not stray: excluding
  footprint-1 events, 0 of 25 are single-step and the alarm covers a median 95% of the
  event.
- **D38 -- `gru-quantile` catches a strict subset of the floor's events on Mission 1.**
  Both 27, only-GRU 0, only-`rstd` 7, neither 12 on the gate set; 26/0/8/8 on `m1-ss5`.
  There is not one event on either set, in any taxonomy cell, that the flying detector
  catches and the corrected floor misses. The seven it misses are short and sharp --
  footprints 1, 1, 1, 1, 24, 28, 54 -- where its own 105-span EWMA smooths the excursion
  away. The per-event argument for the forecaster is withdrawn on these two sets; its
  case rests on the quality of the same catches, which is what D3's gate metric measures.
- **D39 -- the headline cell is not the contextual class, and the contextual class is
  caught by nothing.** Of the 32 headline-cell events, 3 are truly contextual by the
  0.1/99.9 training envelope and 6 by hard min/max, against 8 to 16 predicted. The other
  29 breach at least one channel's own envelope. The three genuinely contextual events
  are caught by neither detector, at reaches of 0.11 to 0.61 against a threshold of 1. On
  the evidence available this project has no measured instance of catching a contextual
  anomaly. Stated carefully because an envelope is not a limit: breaching one does not
  establish that a limit would have tripped, only the converse.
- **The state-adaptation hypothesis is dead and backwards.** The GRU's score does not
  collapse inside a sustained event -- median half-life 2,805 and 6,284 steps, and it
  never halves in a third of them -- while `rstd`'s halves in about its own 120-step
  window, 114 and 112. Whatever causes the GRU's misses, it is not that it learns the
  anomaly. No structural remedy was scoped, because 22.3 made scoping conditional on the
  hypothesis holding and it did not.
- **Two predictions were refuted in the direction the work item was commissioned to
  avoid**, which 22.4 said in advance to watch for. And L5's own definition was defective:
  it did not condition on footprint, so nine footprint-1 events counted as stray ticks
  when a one-step crossing on a one-step event is a perfect catch. Recorded Refuted
  anyway, because a definition that needed fixing after seeing the data is what
  pre-registration exists to expose.

### Added - work item 9.6 follow-up, 2026-09-02

- **`docs/MODELS.md` 22.11**, the four answers the audit's own artifact held and 22.6 to
  22.10 did not read: the four sets by id, fold, cell and footprint for both channel sets;
  the fact that every only-`rstd` event is in fold 1, and that six of the seven are the
  same six section 14 recorded as the GRU's fold-1 losses against the LSTM; `m1-ss5`'s 31
  headline-cell events classified, replicating D39 at 3/31 on the same three event ids;
  and the correction of the work item's brief, whose "twelve GRU-missed events" are in
  fact `lstm-quantile` misses of which `gru-quantile` recovered seven. Zero bucket
  operations.
- **D40 -- "truly contextual" is defined relative to a watched channel set.** Hard
  min/max contextual goes from 6/32 on twelve channels to 18/31 on six while the 0.1/99.9
  count stays at 3, verified as a strict superset relation with no violations. The
  mechanism is monotonicity. D39's headline and all five of its consequences stand,
  better supported than before.
- **`tests/test_documents_are_current.py`**, three checks: no new live occurrence of a
  claim `Objective.md` 1.1 has retired, every tracked document ASCII, and a test count a
  live document states matching what pytest collects. The stated counts were 493 and 473
  against 508 collected; both are corrected.

### Held - not done, and not quietly

- **The restatement of the central claim** in every document that carries it. Nine
  sentences still assert a ratio D37 falsified and D38 reversed, and they are pinned by
  the new test rather than fixed. Held pending review (`docs/STATUS.md` section 7).
- **Two gaps in the audit's instrument**, named in 22.10 and priced in 22.11: GRU run
  lengths were never recorded, and `lead_of` cannot separate a crossing at onset from an
  alarm already running. One bundle load together, 15 Class B and 1 Class A.

## [0.6.1] - 2026-09-01 - Phase 2 work item 9.5: the floor was wrong

A D8 correctness fix and the re-score it forced. Zero new capability; one published
claim falsified.

### Fixed - work item 9.5, 2026-09-01 - `baselines._rolling`

- **`_rolling` accumulated its prefix sums in float32 and lost the statistic it computed.**
  `baselines.py:40` now promotes to float64 before squaring and `np.cumsum` accumulates with
  `dtype=np.float64`. Against the flight rule the gap closes from **7.6584e+00 to
  1.8284e-08** and the spurious exact zeros go from 3,975 to none.
  `tests/test_rolling_precision.py` pins it against `numpy.nanstd` in five regimes, and the
  three work-item-9 tests that pinned the size of the divergence are inverted, as their own
  docstrings instructed.
- **The floor moved a long way.** Re-scored at **41 Class B and 3 Class A**, old artifacts
  preserved, no forecaster row recomputed and no run naming any other detector:

  ```
    m1-g8.9.10  rstd  F0.5  0.250 -> 0.676    MVGS  3/32 -> 25/32   lead -1,512 -> +0.0
    m1-ss5      rstd  F0.5  undefined -> 0.663   MVGS  0/31 -> 25/31
    m2-ss1      rstd  nominal-step FA  17.30% -> 0.003%    rare  84/424 -> 22/424
    m1-g3       rstd  F0.5  0.029 -> 0.351    MVGS  3/10 -> 8/10
  ```

- **The pre-registered falsification fired, and this is the finding.** `docs/MODELS.md` 21.4
  named 22/32 -- `gru-quantile`'s headline cell -- as the line past which the project's
  central claim would be "in serious question". The corrected floor reached **25/32**. A
  per-channel statistic finds twenty-five of the thirty-two cross-channel events, not three.
  The sentence this repository has quoted since work item 4 -- "a per-channel statistic
  finds three; a forecaster over the channel set finds twenty-eight" -- is about arithmetic,
  and is corrected everywhere it appears with the old figure beside it.
  **(!) CORRECTED 2026-09-02: that last clause was not true when it was written.** The
  sweep corrected every scorecard *table* and left the *prose claim* standing in nine
  sentences, including `Objective.md` 1.1's KEPT block, which is the passage
  `docs/INDEX.md` sends every reader to first. The restatement is held pending review at
  `docs/STATUS.md` section 7 on D39; `tests/test_documents_are_current.py` now pins the
  nine so the set cannot grow while it waits.
- **What survives.** On the gate metric D3 fixed before any of this was measured -- event-wise
  F0.5, never bare recall -- `gru-quantile` still clears the corrected floor **0.804 to
  0.676**, reaching comparable recall at **a third of the alarm rate** with precision 0.885
  against 0.661. D28's architecture gate never involved `rstd`. D25 and D29 are untouched,
  and D29's evidence is cleaner: on Mission 2 the corrected floor alarms on 0.003% of nominal
  time rather than a sixth, so the adoption number is now a comparison between two working
  detectors instead of one working detector and a broken one.
- **W1 was wrong and backwards.** It predicted the corrected threshold would fall; it rose
  from 3.34 to 13.67, because the scale divisor is the standard deviation of the spread
  series itself and correcting the numerator shrank the denominator more. Recorded Wrong.
- **A limit the fix does not remove.** `sqrt(S2/n - (S1/n)^2)` loses accuracy as the square
  of `|mean|/sigma` at any precision -- negligible at the ratios ESA-ADB's min-max scaling
  produces, total at 1e8, silently zero at 1e9. `baseline_reference.py` and therefore
  `flight/src/Baseline.cpp` share it exactly. The boundary is pinned by test and reported;
  making the form unconditionally stable is an algorithm change that would move the flight
  golden vectors, and was not taken here.
- Tests 493 -> 505.

## [0.6.0] - 2026-09-01 - Phase 2 work item 9 (tag wi9)

The F' component and the Level 1 safe-failure mode. Reviewed and checkpointed 2026-09-01.
Zero bucket operations throughout.

### Added - work item 9, 2026-09-01 - the F' component and Level 1

- **`Sentinel::Monitor` builds in F' v4.3.0's own Ref deployment**, which is work item 9's
  definition of done. Ref moved out of the framework root to `TestDeploymentsProject/Ref` at
  v4.3.0 and the move is not in the release notes, so the brief's target had to be found
  before it could be hit (`docs/MODELS.md` 20.2 correction 3). Two proofs: this project's own
  `fprime/SentinelRef` deployment, committed and rebuildable from a fresh clone, and F's Ref
  via `scripts/fprime_ref_patch.sh` -- 2,428,064 bytes against stock Ref's 2,352,560, carrying
  234 Sentinel symbols. Nothing is copied: `fprime/` is an F' library, so Ref consumes it the
  way a mission would, with one `library_locations` line.
- **Level 1 works, and was watched working.** All **11/11** refusal codes degrade to the
  statistical baseline with the code named in the event; **0/11** fail the topology; the
  component served 200 ticks after a refusal in test. Run for six seconds with no model file,
  the deployment emits `DegradedToBaseline: NO_MODEL_FILE` and carries on. Objective.md
  decision 10's Level 1 is resolved; Levels 2 and 3 stay open.
- **The loader has 11 refusal codes, not 16.** The 16 is the number of load *cases* in
  `flight/test/RefusalTests.cpp` -- 15 refusing, 1 accepting. `CHANGELOG.md`, `docs/STATUS.md`
  and `docs/MODELS.md` 19.8's prediction F7 were all loose the same way.
- **The Level 1 baseline transcribes the rule and not the implementation, deliberately** (D37).
  `baselines._rolling` runs `np.cumsum` on a float32 array and differences the result to
  recover a second moment, which is catastrophic cancellation: three independent
  implementations agree to 1.8e-08 and disagree with it by **7.6584e+00** on a true sigma of
  3.0, and on this project's own fixture it produces **3,975** exact zeros against float64's
  **1,123**. `flight/src/Baseline.cpp` matches `src/sentinel_models/baseline_reference.py`
  **exactly** -- 0.000e+00 over four tiers and 1,600 steps, flags exact. The harness repair is
  scoped as work item 9.5.
- **D32 to D37**: a passive component on a synchronous `Svc.Sched`; direct port wiring rather
  than a telemetry-path tap, which resolves Objective.md decision 3 and declines the tap on
  evidence; Level 1's constants as PrmDb-style parameters rather than model-file fields; a
  guarded types shim, amending D31 consequence 2; the whole-file read with the chunked reader
  deferred; and the `_rolling` finding.
- **The types shim swap, proven both ways.** One file differs between the freestanding build
  and the F' build, and it is the shim; `sizeof(Detector)` is still exactly 312,112 bytes under
  F' types. D31 consequence 2's literal wording is not achievable -- `Fw/FPrimeBasicTypes.hpp`
  needs a generated config header the Makefile build has no way to produce -- so the shim
  selects rather than replaces, and three tests hold the claim in place.
- **`FW_HAS_F64` does not exist in F' v4.3.0.** `docs/MODEL_FILE.md` 9 and the old `Types.hpp`
  both said F' treats F64 as switchable; `F64` is unconditional at `Fw/Types/BasicTypes.h:86`
  and the macro appears exactly once in the whole framework, in the documentation table this
  project read. The shim asserts the property instead. Both documents amended.
- **`clang-tidy` ran for the first time**: 170 findings, 16 fixed, 154 excluded with a written
  reason each, 0 remaining across three configurations -- ours, the framework's root config and
  the framework's release config for flight code. Two findings earned the exercise: an
  out-of-bounds access and a division by zero that the loader makes unreachable at load time
  but that a radiation bit-flip in RAM could reach afterwards. D31 consequence 5 was wrong
  about where clang-tidy comes from; it is Homebrew's llvm, not F'.
- **Footprint**: `sizeof(Sentinel::Monitor)` is **623,152 bytes** against 624,528 predicted
  before the component existed -- 1,376 B under. Of that, 312,112 is the `Detector`, 302,048
  the model-file buffer and 8,216 the `Baseline`.
- **Predictions C1 to C10 are re-tabulated in `docs/MODELS.md` 20.9; eight held and two were
  wrong.** C6 predicted fewer than 50 lint findings dominated by `readability-*` and got 170
  dominated by `misc-include-cleaner`; C9 predicted a 10-to-20-minute first F' build and got
  **12.4 seconds**, because F' builds only the modules the topology references.
- **Thirteen corrections to the work item's brief** are recorded in `docs/MODELS.md` 20.2 and
  four more things F' settled once code was being written in 20.10 -- among them that a
  component with parameters is required to carry command ports, and that a library's modules
  must be namespaced, which moved the component to `fprime/Sentinel/Monitor`.
- Tests 473 -> 493. `docs/FPRIME.md` records the toolchain and `scripts/fprime_setup.sh`
  rebuilds it from nothing.

## [0.5.0] - 2026-09-01 - Phase 2 work item 8 (tag wi8)

Phase 2's first work item: the flight inference core and the frozen model file. Reviewed
and checkpointed 2026-09-01. Zero bucket operations throughout.

### Added - work item 8, 2026-09-01 - the C++ inference core and the frozen `model.bin`

- **The format is frozen at version 1** (D30, `docs/MODEL_FILE.md`, normative). Plain
  little-endian float32 in `reference.Weights.arrays()` order with **both bias vectors
  unsummed**; a 64-byte self-protecting header naming the architecture and the gate order
  rather than leaving them to be inferred; a channel map; and a **separately-CRC'd
  parameter block** carrying the normalisation constants, the threshold, the EWMA span,
  `baseline_only` and the tier. A recalibration in orbit overwrites a fixed-size block and
  two header words, and never touches the 278.0 KiB of weights.
- **Objective.md 14.10's "quantized, self-describing FlatBuffer, TFLite-Micro compatible"
  is superseded and marked so, never deleted.** Nothing here consumes TFLite; a FlatBuffer
  parser is templated, allocating third-party code F' CPP-25 and CPP-1 exclude; and a fixed
  layout with a CRC is byte-inspectable by a review board. Quantization goes with it: the
  tolerance against `reference.py` is 1e-5 and int8 loses far more. Every requirement 14.10
  stated is met. Objective.md 14.2 is resolved.
- **F' pinned at v4.3.0** (D31), which resolves Objective.md 14.4. Reading F's own
  statement of its C/C++ rules corrected three this project had from memory: F' states no
  no-recursion rule (that is Power of Ten 1 and the JPL C standard, which F' cites at
  CPP-27); its no-heap rule is CPP-1, not Power of Ten 3; and CPP-3 forbids bare `float`
  and `double` outright, which was recorded nowhere and changes every declaration.
- `flight/`: the GRU forward pass and the frozen decision layer (D25) transcribed from
  `src/sentinel_models/reference.py`. Freestanding C++14 behind a types shim work item 9
  swaps for `Fw/FPrimeBasicTypes.hpp`. No exceptions, no RTTI, no STL, no allocation
  anywhere, no recursion, every loop bounded by a header field already checked.
  `sizeof(Detector)` is **312,112 bytes** against 312,642 predicted before the code existed.
- **The core matches the reference at 1.8e-07** worst case across seven weight sets --
  three seeded tiers and the four cached production fits -- over 504 steps spanning a chunk
  boundary and a reset, with the **crossing flag exact on every step**. The tolerance was
  1e-5, so the margin is roughly fifty-fold, and it sits where `docs/MODELS.md` 2 already
  measured the NumPy reference against torch (1.2e-07).
- `src/sentinel_export/` stops being a placeholder: `format.py`, `writer.py`, `reader.py`,
  standard library and numpy only, as its docstring has always promised. The reader returns
  a `Status` whose values are shared with the C++ `LoadStatus`, so one test asserts both
  sides refuse the same bytes for the same reason. Sixteen refusal cases; seven files
  round-trip Python to C++ to Python byte-identically.
- Golden vectors under `flight/test/vectors/`, committed and regenerable: deleting them all
  and rebuilding reproduces fourteen files byte-identically. **No vector uses real
  telemetry and none can** -- there is none on local disk -- so every input is the seeded
  fixture or a seeded generator, and the cached production weights supply the fourth tier.
- Determinism: the same 400-tick digest, `0xD66576B4`, twice in one process and again in a
  fresh one. Guaranteed by `-ffp-contract=off` and the absence of `-ffast-math`, both
  pre-registered. The build is **silent** at `-Wall -Wextra -Wpedantic -Wconversion
  -Wshadow -Werror`; `clang-tidy` is deferred to work item 9 with the F' toolchain.
- Tests 406 -> 473. `docs/MODELS.md` 19 is the pre-registration, committed before a line of
  C++, with its PREDICTED table re-tabulated in 19.8; all eight predictions held.

### Planned - Phase 2, and after

- After work item 9: the in-orbit threshold recalibration path, exercised end to end on the
  F' Ref. See docs/PHASE2.md and docs/STATUS.md section 7.
- Post-gate: injected-fault sensitivity study - controlled drifts and decouplings injected into
  real ESA-ADB telemetry, for a detection sensitivity curve and lead-time measurement. Never a
  headline number; see Objective.md section 13.

### Open decisions, and those resolved

| # | Decision | Deadline |
|---|---|---|
| 1 | Architecture selection - LSTM vs GRU vs TCN | **Resolved 2026-08-29: the GRU (D28)** |
| 2 | Model-file format freeze. The FlatBuffer/TFLite-Micro container above is **superseded by D30, not deleted**; the requirement it carried - normalisation constants and thresholds stored separately as PrmDb-style parameters (Objective.md 14.10) - stands and is met | **Resolved 2026-09-01: plain little-endian float32, version 1 (D30)**, specified in `docs/MODEL_FILE.md` |
| 3 | Channel-ingestion mechanism - telemetry-path tap vs direct port wiring | Early Phase 2 |
| 4 | Target F' version pin | **Resolved 2026-09-01: v4.3.0 (D31)** |
| 5 | Harness base - build on TimeEval or standalone | **Resolved 2026-08-25: standalone** (0.3.0) |
| 6 | R2 ingest sizing for 11.6 GB | **Resolved 2026-08-24: 11.53 GB in 234 objects** (0.2.0) |
| 7 | Second independent scoring set | **Resolved in practice and spent 2026-08-29**: Mission 2 the adoption number, Mission 1 group 3 the recall exam (RESULTS.md 6k) |
| 9 | SatNOGS as subsystem-prior corpus | Post-gate |
| 10 | Tiered capability architecture - Level 1 / 2 / 3, one loader, one file format. Level 1 is the loader's mandatory safe failure mode | Before Phase 2 |

Decision 8, normalisation policy, is **resolved**: identity. See Objective.md 14.

## [0.4.0] - 2026-08-29 - Phase 1 closed (tag wi7)

Phase 1 complete: the detection mathematics proven on the harness, the architecture gate passed, and the held-back transfer sets scored once. Work items 4 to 7; tags wi4 to wi7 and their Releases.

### Added - work item 4, 2026-08-25 to 2026-08-28 - telemanom reproduced with a multivariate LSTM

- `src/sentinel_models/lstm.py`, `reference.py`, `telemanom.py`, `detectors.py`: telemanom's
  detection method driven by one multivariate 2x80 LSTM over the channel set (91,640 parameters,
  358.0 KiB), trained in PyTorch and scored through a plain-NumPy reference held to 1e-5
  (4.1e-08 measured) - the Phase 2 C++ blueprint. Every deviation from the published
  configuration in MODELS.md section 1's ledger, ten rows.
- The floor cleared and the thesis held: 28/32 headline-cell events on `m1-g8.9.10` against
  `rstd`'s 3/32 (RESULTS.md 2). Fitting moved to a rented GPU, twelve fits in 15.9 minutes,
  certified by the equivalence assertion rather than a matching environment (D15, D16).
- D17: telemanom's published `min_delta = 3e-4` had silently disabled training for every fit in
  the project - one epoch kept, up to 8.8x better weights discarded. Replaced by a relative
  `min_improvement`; the forecast improved about fortyfold and the published threshold collapsed
  from 182 to 3,548 alarm ranges (RESULTS.md 6a). Thresholds belong to the model, not the method.
- The selection criterion measured over 5,684,580 reference windows: 92.6% chose the range
  minimum (D18, THRESHOLD.md). OS-CFAR pre-registered, run, retested and refuted (D20, D22).
  `lstm-whitened` built and measured - 2/48 rare-event false alarms at 21/32 (RESULTS.md 6d, D24).
- Lead time measured from a moment the detector could reach: the reported +26 was the batching
  latency counted backwards, the honest median is 0.0 (D21, RESULTS.md 6f); Objective.md 1.1
  retires the early-warning claim and keeps the cross-channel one.
- D25: the decision layer frozen as `lstm-quantile` - F0.5 0.838, recall 26/46, 21/32, 2/48,
  nominal-step 0.002%, honest lead +0.0 on `m1-g8.9.10` (RESULTS.md 6g), identical across every
  architecture at the gate. Telecommands wired as model inputs (`lstm-commanded`, D6, D7); the
  ablation on post-fix weights not yet run, so every figure remains telemanom-minus-commands.

### Added - work item 5, 2026-08-28

- `Hyper.cell`: the recurrent cell as a field of the LSTM's configuration, emitted only when
  not the default so no banked LSTM weight or published fingerprint moved (D26, pinned by test).
  One torch module builds `nn.LSTM` or `nn.GRU`; `train()` is shared verbatim.
- `reference.gru_cell` / `gru_layer`: the GRU tick beside the LSTM's, `GRU_GATES`, a one-vector
  state, and the cell of a weight file derived from its arrays and verified against the file's
  `cell` field. The third recurrent bias sits inside the reset product and cannot be folded --
  MODELS.md section 3 amended for the file format.
- Detectors `gru-telemanom`, `gru-quantile` (the gate arm) and `gru-smoke`.
- `scripts/fit_folds.py --determinism-check`; report written before the ledger; held-back sets
  refused. `scripts/head_to_head.py` records each event's reach beside the booleans.
- MODELS.md section 14: the `gru-quantile` pre-registration and its outcome. RESULTS.md 6h.

### Added - work item 6, 2026-08-29

- `Hyper.cell = "tcn"` with a TCN-only `kernel` field, emitted only for a TCN; `hidden` reused as
  the width of each residual block. `TelemanomTCN` (Bai et al. 2018 blocks, no weight norm) built
  by `build_model`, the one place the architecture is chosen; `train()` unchanged.
- `reference.ConvWeights`, `causal_conv1d`, `tcn_block`: the TCN blueprint, stateless by
  contract -- `forward` refuses a state and returns none. Receptive field 253, 91,670 parameters
  at the flown shape. Weight files carry `cell = "tcn"` with their own keys.
- Detectors `tcn-telemanom`, `tcn-quantile` (the gate arm) and `tcn-smoke`. D27.
- MODELS.md section 16: the `tcn-quantile` pre-registration and its outcome. RESULTS.md 6j: the
  TCN beside both cells -- MVGS 9/32 on the gate set, a floor above both incumbents' on every
  fold, one stalled fit. Three rows exist; the gate is a decision.
- MODELS.md section 17 and `scripts/combination_scope.py`: the LSTM+GRU combination scoped on
  cached weights, not built -- not nested; an OR reaches 25/32 at 2/48 on the gate set; a combined
  score recovers no gate-set solo event. Post-gate, its own decision if ever.

### Added - work item 7, 2026-08-29 - the gate and the held-back sets; Phase 1 closed

- D28: the architecture gate selects the GRU on Objective.md section 8's criteria; the LSTM stays
  the published baseline, the TCN rows the stateless answer. `lstm-gru-or`, the union as a
  detector, so the closure could score it through the one tested path.
- The held-back sets scored once, on a GO, with MODELS.md section 18's predictions committed
  first. `m2-ss1`: 4/424, 4/424, 6/424 rare-event false alarms and 0 nominal-step alarms for the
  LSTM, GRU and TCN; the floors 84/424 and 122/424. `m1-g3`: fold 0 clean; folds 1-2 a
  calibration collapse -- the noise floor fixed on the past sat under 87% of a later window.
- D29: `gru-quantile` flies alone; the union is not adopted (its cost failed on Mission 2, its
  edge evaporated on m1-g3); the calibration's transfer is what Phase 2 inherits. `docs/PHASE2.md`.

### Phase 1 work items, in order - all complete

- Reproduce telemanom's detection method with a multivariate LSTM forecaster (done).
- Train and score GRU (done, 2026-08-28; two stop-and-report rules fired, see RESULTS.md 6h).
- Train and score TCN (done, 2026-08-29; see RESULTS.md 6j).
- Pass the architecture selection gate, and score `m2-ss1` across LSTM, GRU, TCN, `rstd` and
  `mavg` together so the adoption number on an independent spacecraft is a comparison rather
  than a lone figure (done, 2026-08-29; D28, D29, RESULTS.md 6k). Phase 1 closed.

## [0.3.0] - 2026-08-25

The evaluation harness. Phase 1 work item 3 complete: the referee exists, it has
been checked against its own extremes, and the trivial baselines are scored.

### Added

- `src/sentinel_eval/` - the harness. `catalog` (manifest-only key resolution,
  typed dataclasses), `read` (streaming, checksum-verified, sharded-first),
  `labels` (events, taxonomy, the four categories), `grid` (zero-order hold with
  a staleness guard), `splits` (forward chaining, contamination reporting),
  `bundle` (fetch once, hold in memory, subset without re-reading), `metrics/`
  (event-wise F0.5, VUS-PR, false alarms, quarantined diagnostics), `harness`,
  `scorecard`, `tasks`, `ops`, `synthetic` and a CLI.
- `src/sentinel_models/` - the players. Trivial baselines plus the registry that
  work items 4-6 extend. The harness never imports a model; `tests/test_layering.py`
  enforces the direction.
- `src/sentinel_export/` - Phase 2 placeholder for the `model.bin` writer.
- `scripts/check_no_list.py` - the source-level LIST/glob ban, which the brief
  believed already existed. It did not.
- `docs/HARNESS.md`, `docs/RESULTS.md`.
- 136 tests, all offline against a generated fixture at zero R2 operations.

### Fixed

- `r2.fetch_ledger` caught bare `Exception` and returned a fresh ledger, so a
  transient failure reported the month's spend as zero. Only a genuinely absent
  ledger now starts fresh.
- The operations tripwire (1,000 per run) and monthly ceiling (50,000) were
  constants with no enforcement anywhere. Both now raise, at the point of
  spending, through the existing per-HTTP-attempt hook.
- The ops ledger was written with `new_ledger()`, erasing the month's history on
  every run. It is now read-modify-write.

### Decided

- **Normalisation is identity** (Objective.md 14.8). Cross-group spanning is
  acceptable; per-channel rescaling is refused because it erases the amplitude
  ratios ESA preserved within each group.
- **The gate number is event-wise F0.5**, never bare recall. Recall alone is
  satisfiable by carpet-bombing, which is how the trivial baseline first appeared
  to score 29/31.
- **`m1-g8.9.10` is the primary recall set**, promoted post-hoc on footprint
  evidence; `m1-ss5` is demoted, retained, and reported alongside it in every
  result. Partial runs are barred from RESULTS.md.
- **Standalone metrics, not TimeEval** (Objective.md 14.5): TimeEval requires
  Python <3.13 against this project's 3.14, pins `dask==2022.12.1` and needs
  Docker.

## [0.2.0] - 2026-08-24

ESA-ADB ingested to Cloudflare R2. Phase 1 work item 2 complete; the evaluation
harness (item 3) can now be built against a stable, manifest-addressed dataset.

### Added

- `src/sentinel_data/` - the ingest toolkit. `zenodo.py` (resumable source
  download, MD5-verified), `esa_adb.py` (nested-zip reader), `transcode.py`
  (parquet with a hard 90 MiB object ceiling and time-based sharding),
  `r2.py` (client with per-HTTP-attempt operation accounting), `manifest.py`,
  `docs_gen.py`, and a `spike / download / transcode / prepare / upload` CLI.
- `scripts/roundtrip_check.py` - proves manifest -> key -> object -> DataFrame.
- `docs/DATA.md` and `docs/manifest.snapshot.json` - regenerated from the
  manifest on every ingest, so they cannot drift from the bucket.
- `.env.example`, `requirements.txt`.

### Data

- 224 channels across 3 missions (76 / 100 / 48; 58 / 47 / 24 target),
  ~2.30 billion points, 11.53 GB as zstd parquet in 234 objects.
- 821 telecommand files merged to one object per mission. 681 of Mission1's 698
  carry executions; the other 17 are declared but never executed.
- Annotations in 4 objects, with `labels` pre-joined to `anomaly_types` so the
  harness reads one object instead of two on every run.
- 4 channels exceeded the 90 MiB ceiling and were split into time-ordered
  shards, recorded in the manifest under `shards`.

### Verified

- Every object checked by size, ETag against the locally computed MD5, and a
  SHA-256 carried in object metadata. The manifest is published only after all
  234 objects verify, so its presence guarantees the dataset it describes.
- 236 Class A and 238 Class B operations, 0.5% of the 50,000/month ceiling.
- Nothing retained locally: source archives and parquet are deleted per mission
  once verified, and the scratch directory is removed and asserted gone.

### Decided

- Timestamps stay nanosecond and values keep their native dtype. The dataset
  documents its anonymisation as numerically lossless, so the archive does not
  downcast. Some channels are categorical and are stored as strings.
- Storage is 11.53 GB, about 1.5 GB beyond R2's 10 GB free tier (~$0.02/month).
  The brief's "possibly inside the free tier" does not hold: the source pickles
  are already deflate-compressed and float32 values resist zstd.
- Source verification uses MD5. Zenodo publishes no SHA-256 for this record.

## [0.1.0] - 2026-08-24

Repository stood up, documentation-first. No code yet, by design: Phase 1 proves the detection
in Python before a line of flight C++ is written.

### Added

- `Objective.md` - the living objective document and single source of truth. Covers the problem
  and the 41% contextual-anomaly evidence, prior art (telemanom, OPS-SAT, and the empty lane for
  a reusable F' block), the generic-code / mission-data architecture split, the learning
  formulation, the detection discipline (persistence filter, trend projection, explanation
  layer), the LSTM vs GRU vs TCN selection gate, the data stack with ESA-ADB as primary, the
  cold-start analysis and its five fixes, five permanent safety rules, the four-phase roadmap,
  and the open decisions.
- `README.md` - project summary, status and pointers.
- `CHANGELOG.md` - this file.
- `.gitignore` - datasets, trained artifacts and Python build output excluded.

### Decided

- ESA-ADB is the primary evaluation set. SMAP/MSL demoted to legacy comparability only: it is
  publicly discredited (Wu & Keogh, IEEE TKDE 2023) and its channels are not synchronised with
  each other, so it physically cannot demonstrate the cross-channel claim.
- Metrics: event-wise F0.5 / VUS-PR. Point-adjusted F1 is avoided as it inflates results.
- Datasets are never committed to the repository. Code and docs only.

[Unreleased]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi9.5...dev
[0.6.1]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi9...wi9.5
[0.6.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi8...wi9
[0.5.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi7...wi8
[0.4.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi3...wi7
[0.3.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi2...wi3
[0.2.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi1...wi2
[0.1.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/releases/tag/wi1
