# Project status and roadmap

> **Paths outside this branch resolve on `dev`.** `master` carries the component and the
> evidence it works, and nothing else (`docs/DECISIONS.md` D67). A citation here into
> `src/`, `scripts/`, `tests/`, `docs/MODELS.md` or `third_party/` points into the
> development branch at the commit this snapshot was taken from.
> `scripts/check_references.py --master` is what keeps that true rather than hoped.

**2026-09-09. Phase 2 in progress. The objective is re-framed** (`docs/DECISIONS.md`
D57): Sentinel warns before a limit trips using **every kind of context the telemetry
carries**, and "cross-channel means sensor-to-sensor" is retired as the sole thesis.
A ten-minute read. Every number is read from a named artifact under `runs/`, or from a
test that pins it.

**(!) THIS PAGE IS SILENT ABOUT THE RETRAINER, AND THAT IS A GAP RATHER THAN A
JUDGEMENT.** It was written on 2026-09-09, six days before D70 registered OxCaml as a
candidate, and the whole Phase 5 track -- the retraining engine, the hub crossing, the
sanity gate, D82's bounds-checking -- arrived after it and never reached this file.
**`master:docs/STATUS.md` is the current status document for that work**: roadmap row E,
and the four paragraphs after it naming what is unenforceable, what nothing onboard
scores, and what has never run on flight hardware. The record is `docs/DECISIONS.md`
D70, D73, D74, D76, D77, D78, D82 and `docs/MODELS.md` 47 to 76. Noted here in 2026-09-22
rather than back-filled, because a status page rewritten to cover a track it did not
watch would be a record pretending to be a status.

**(!) This document was rewritten on 2026-09-09 and is a status, not a record.** The
previous version had grown to an eighty-item changelog of its own. Nothing is lost:
`CHANGELOG.md` carries every work item version by version, `docs/MODELS.md` carries every
pre-registration beside its outcome, `docs/DECISIONS.md` carries D1 to D86 (was D85, was D84, was D83, was D82, was D78, was D75, was D63) with superseded
entries marked and never deleted, and `docs/NARRATIVE.md` carries what happened in order
with the errors in it. This page says where the project **is**.

---

## 1. Goal

A reusable F' (F Prime) flight-software component that **warns before a limit trips**,
using every kind of context the telemetry provides. It warns; it never commands
(`Objective.md` section 11 rule 3).

```
  (a) a channel's own history      TESTED.  10 of 38 in-range contextual
                                            sequences on NASA SMAP/MSL, at
                                            0.68% of nominal steps
  (b) commands -> telemetry        BARELY TESTED. NASA's arrays carry commands
                                            on the telemetry's own clock -- 24
                                            one-hot columns on SMAP, 54 on MSL.
                                            One encoding measured (D49), which
                                            bounded itself to that encoding
  (c) other sensors -> telemetry   UNMEASURED on any public data. The F' Ref
                                            physics testbed is the venue
```

**Single-context or multi-context is not the distinction. Earlier than a limit check,
with minimal false alarms, is.** The model is frozen in flight; the threshold is
recalibrated in orbit without retraining; a shadow model may retrain in flight under
human approval, which is Phase 5 and is scoped in `docs/PHASE5.md` and built by nobody.

The detection method is Hundman et al., KDD 2018 (JPL's telemanom). **The reusable
flight packaging is this project's contribution.**

## 2. Why it matters

- Onboard protection today is a per-channel limit check - a thermostat. Nothing in an F'
  deployment watches relationships between channels at all (`Objective.md` 2.1, 3).
- **39 of 43** labelled contextual anomalies on SMAP/MSL stay **at or within** their
  channel's own historical range (D46; **23 of the 39 touch a rail exactly**, D46.1).
  A limit check cannot see them by construction.
- The method was published in 2018 and no reusable flight component followed.

**No wall-clock claim is made.** "Hours before anything breaks" was struck from
`Objective.md` sections 0 and 1 on 2026-09-09. The break-to-limit-trip lead is Phase 3's
measurement, on a real clock against real dictionary limits, and it is unmeasured.

## 3. How it works

1. **Normalise: identity.** Per-channel rescaling would erase the amplitude ratios
   between related channels the method exists to watch (D2).
2. **Forecast.** A GRU - two layers of 80, 250-step lookback, ten-step-ahead head -
   predicts the watched channels. Multivariate over a channel set on ESA-ADB; per channel
   on SMAP/MSL, whose 81 streams are unsynchronised.
3. **Residual.** Per channel, the absolute difference between forecast and reality.
4. **Smooth and reduce.** An EWMA (span 105) per channel, then the maximum across
   channels.
5. **Threshold.** Either a label-free static quantile (D25) **or** the published dynamic
   threshold. **The component ships both and a mission selects one**, because which rule
   is needed is a property of the telemetry regime and not of the method (D48, D49).
   A threshold is a measured noise floor, never an alarm budget (`docs/HARNESS.md` 1).

Training runs on the ground in PyTorch. Flight inference is C++ transcribed from the
NumPy reference in `src/sentinel_models/reference.py` - static memory, no exceptions.

## 4. Where we are

**Phase 1 CLOSED 2026-08-29** (tag `wi7`; architecture GRU, D28; transfer validated, D29).
**Phase 2 in progress**: the flight core and `model.bin` are done (tag `wi8`), the F'
component and Level 1 are done (tag `wi9`), and work items 9.5 to 9.14 are the science
that followed.

**NASA SMAP/MSL leads, because it is where the claim is measured.** telemanom's own
accounting, `TP/(TP+FP)` with true positives deduplicated per matched event. MSL first: it
is the one population that matches the paper's exactly at 36, all six D17 stalls being
SMAP's.

| Arm | What it is | MSL recall | Total recall | precision | FP | artifact |
|---|---|---|---|---|---|---|
| stage 4 | `gru-telemanom`, published dynamic threshold, swept to 0.6838% nominal | - | 47/100 | - | - | `runs/smap-msl/_forensics/2026-09-03T212818Z-stage4-ndt.json` |
| `A0` | `1a+1b`, the faithful base | **16/36** | 57/98 | 62/110 | - | `runs/smap-msl/_forensics/2026-09-08T182416Z-wi910-port.json` |
| `F` | the complete port, the reference ceiling | 3/36 | 34/98 (34.7%) | **34/38 (89.5%)** | **4** | same artifact |
| `T` | `F` plus the published training configuration | 3/36 | 46/104 (44.2%) | 46/55 (83.6%) | 9 | `runs/smap-msl/_forensics/2026-09-08T201450Z-wi910-port.json` |
| `R` | `T` plus the residual rung | 3/36 | 46/104 (44.2%) | 46/55 (83.6%) | 9 | `runs/smap-msl/_forensics/2026-09-08T211109Z-wi910-port.json` |
| `H` | `gru-zscore`, the probabilistic head | 5/36 | 47/102 | 47/256 (18.4%) | 209 | `runs/smap-msl/_forensics/2026-09-08T220104Z-wi910-port.json` |
| `G` | `R` with the two absolute filters made dimensionless | 3/36 | 48/104 (46.2%) | 48/70 (68.6%) | 22 | `runs/smap-msl/_forensics/2026-09-09T043011Z-wi910-port.json` |
| `C1` | `T` with the commands WITHHELD (D60) | 4/36 | 54/104 (51.9%) | 54/58 (93.1%) | 4 | `runs/smap-msl/_forensics/2026-09-09T054400Z-wi910-port.json` |
| `W` | 34.8's label-free forecaster winner, own cut | 3/36 | 56/104 (53.8%) | 56/61 (91.8%) | 5 | `runs/smap-msl/_forensics/2026-09-09T224554Z-wi910-port.json` |

**(!) FROZEN 2026-09-09 (D62): the pipeline is stage 4's configuration.** Nine arms have now
been measured against stage 4's **10/38 at 0.6820% (was 0.6838%)** and none beats it. At or below that
rate: `WS`, the label-free forecaster winner, reaches **8/38**; `C1S` reaches 6/38; the
3-seed ensemble and the isolation forest reach 0; and **four arms - a transition-aware
floor, per-channel calibration, both together, and gradient-boosted trees - have no
operating point at or below 0.6820% (was 0.6838%) at all**, which is D48 arriving for the fifth through
eighth arms. The full ladder is `docs/MODELS.md` 35.7.
| **paper** | Hundman et al., Table 2 | **25/36** | 84/105 (80.0%) | 84/96 (87.5%) | **12** | - |

**(!) SUPERSEDED 2026-09-10 by D65. The paragraph above stands as the record of what nine
arms measured; its freeze does not.** A tenth arm -- the smoothed residual fused with the
**first derivative of the raw value**, both standardised against a trailing window only --
reaches **EVAL 17 of 19 against the frozen arm's 4**, at the frozen arm's own **0.6820%**,
on the channel-disjoint split committed before any sweep. It is a **strict superset**: it
loses nothing the frozen arm caught. **n = 19, UNDERPOWERED (`docs/HARNESS.md` 1).**
Reproduced identically on two independent reads (`docs/MODELS.md` 38.15). The all-38 figure of 30 is
**CONTAMINATED** -- it includes the 19 TUNE events the cut was selected on -- and is never
the headline. **D65 adopts nothing**: it re-opens the decision layer and does not name a
flight configuration.

**The headline number is stage 4's `10 of 38`**: in-range contextual sequences caught, at
0.6820% (was 0.6838%) of nominal steps, by `gru-telemanom` scored per channel, univariate, without
commands (D46, D48, `docs/MODELS.md` 26.18). **It is 26%**, on one dataset, with no floor
available to compare against at that alarm rate - `rstd` at 5,000x its calibrated
threshold still alarms on 15.17% of nominal steps. **Seven model variants have since been
measured and none has replaced it.**

**(!) MSL has barely moved since the port was built.** `F`, `T`, `R` and `G` all catch
**3 of 36** where the paper catches 25 with 2 false alarms; `C1` catches **4**, the first
movement of any kind, and **one event is not a finding**. Eight training changes moved it
by zero (28.7); the residual rung moved it by zero (29.4); relaxing the two absolute
candidate filters moved it by zero (30.4). With 29.4's scale hypothesis also refuted,
**pruning at `p = 0.13` is the last named candidate**, and it was named in advance.

**(!) The command context is measured, and it is negative** (D60, `docs/MODELS.md` 32.7).
Withholding the commands from the published training configuration **gains eight events,
removes five false alarms and lowers the alarm rate at the same time** -- `C1` at 54/104
and 93.1% precision against Arm `T`'s 46/104 and 83.6%. That is the second measurement in
the same direction, on a second training configuration and a second model family, so
**D49 did not need the bound it gave itself.** `Objective.md` 1.1's context (b) is no
longer *barely tested*; at this encoding, on this data, it costs.

**ESA-ADB, one line.** `gru-quantile` leads the gate metric on `m1-g8.9.10` at F0.5
**0.804** against the corrected floor's 0.676, at a third of its alarm rate, and transfers
to an unseen spacecraft at **4/424** rare-event false alarms and **0** nominal-step alarms
in 4,155,841 (D29). **And a calibrated per-channel range check ties or beats it at a
matched rate on both sets, and the forecaster never speaks first across 53 caught events**
(D44). ESA-ADB's labelled faults are gross; the full record is `docs/RESULTS.md` 6l, 6m.

**Flight.** `flight/`'s C++ core matches the NumPy reference at **1.8e-07** worst case over
seven weight sets (D30). `Sentinel::Monitor` builds in this project's deployment and in F'
v4.3.0's own Ref, and all **12/12** loader refusal codes degrade to the Level 1 statistical
baseline with the code named in the event, **0/12** failing the topology (D32-D37).

**Operations.** 238 Class A and 5,740 Class B for 2026-09, of 50,000 each, read from the
last artifact and never transcribed.

## 5. What we did, and why

- [x] Repository stood up documentation-first - decisions recorded before code, D1 to D86 (was D85, was D84, was D83, was D82, was D78, was D75, was D63),
      superseded entries marked and never deleted.
- [x] Ingested ESA-ADB (3 real ESA missions, 11.53 GB, 234 checksummed objects) and
      SMAP/MSL (162 arrays, verified 82/82 against the canonical labels before upload).
- [x] Built the evaluation harness **before** any model - an honest referee before any
      player. It caught five defects in our own measurement design first.
- [x] Reproduced the published LSTM baseline, then corrected it: the published `min_delta`
      had kept every fit at its first epoch (D17), and the published dynamic threshold
      degenerated under the better forecaster (D18).
- [x] Trained the GRU, built the TCN under its own pre-registration, and passed the
      architecture gate on evidence (D26, D27, D28).
- [x] Ran the held-back transfer exam once, and spent it (D29).
- [x] Built the flight inference core, froze `model.bin` at version 1, and built the F'
      component with its Level 1 safe-failure mode (D30-D37).
- [x] Corrected `baselines._rolling`, which accumulated in float32 and lost the statistic -
      and it **falsified this project's own central claim** (D37).
- [x] Vendored telemanom's source and read it in full, correcting five of our own recorded
      readings (D53), then ported the detection stack completely (D54).
- [x] Measured the published training configuration (Arm T), the residual rung (Arm R) and
      a probabilistic head (`gru-zscore`, Arm H) - D55, D56.
- [x] Surveyed the public benchmarks and found the first independent replication of a
      finding of ours (`docs/RESEARCH.md` Part IV).
- [x] **Re-framed the objective on that evidence** - D57, 2026-09-09.

## 6. What we learned

- **Faithful reproduction produced the bug.** A published constant silently disabled
  training and every fit kept its first epoch (D17).
- **Absolute constants in data units do not transfer (D55).** Three instances in three
  stages of one method - and the sharpest is that **the same constant was measured
  correct** on the other dataset's scale. That separates *the constant is wrong* from
  *the constant does not travel*, and only the second dataset could show it. The toolkit
  ships dimensionless equivalents.
- **A defect in the floor is a defect in every comparison drawn against it** (D37). The
  project's most-quoted sentence turned out to be about arithmetic.
- **The threshold is a noise floor, not a dial** (`docs/HARNESS.md` 1), and a static one
  describes a **stationary** regime only (D48).
- **There is a dataset ceiling, and it was found by elimination.** Six model variants have
  been scored on SMAP/MSL and none beats 10/38. The gap against Table 2 is not the scoring
  rule, the commands, pruning's rung, cross-window tracking, the aggregation, the window
  regime or the training configuration - each measured and closed.
- **A design claim written so it can fail, failing (H4).** `gru-zscore`'s sigma was
  predicted not to be approximately constant; measured, its coefficient of variation is
  **0.0504** at the median and above 0.25 on 11 of 79 channels. **H5 held on 79 of 79**, so
  the likelihood objective trained: the model could have learned a varying sigma and did
  not. Not an optimisation failure - a finding about a single univariate channel.
- **The ceiling is neither the forecaster nor the decision layer (D62).** A 32-configuration
  label-free grid moved held-out forecast error by 26.5% and its winner scores 8/38 where
  stage 4 scores 10. Four decision-layer variants cannot reach a flyable alarm rate at all.
  **A ceiling established by elimination is worth more than one asserted**, and this one cost
  nine arms.
- **A better forecaster can be a worse detector.** The 3-seed ensemble's residual is smooth
  enough that a train-calibrated quantile finally transfers - and it catches 2 events at zero
  false alarms. D18's shape, one more time.
- **The context that was supposed to help was hurting (D60).** Command conditioning is
  negative on this data under both training configurations and both model families tested.
  It was measured once, bounded carefully, and the bound turned out not to be needed -
  which is an argument for bounding a finding rather than against it.
- **A dimensionless constant is not a strictly safer absolute one (D58).** It is
  channel-relative -- looser where residuals are small, **tighter where they are large** -
  so substituting one changes *which* channels a filter binds on. Caught only because G4
  was written **per channel** and as a stop: pooled, the same arm reads as a clean
  relaxation.
- **A run that produces a documented figure lands its script in the same commit as the
  figure** (`docs/NARRATIVE.md` 11). Learned by losing a study script and watching two
  gates become unreproducible.
- **Write the artifact before the ledger.** The expensive thing is the result; the ledger
  is bookkeeping that can be retried (commit `75cc846`).

## 7. What's next, in order

**A. Re-frame and record - DONE 2026-09-09.** *Done when* `Objective.md`, `docs/STATUS.md`,
`README.md` and `docs/INDEX.md` carry the re-framed claim, D57 is recorded, `CHANGELOG.md`
and `docs/NARRATIVE.md` are current, and every script that produced a live figure is cited.

**B0. Work item 9.13, dimensionless guards - DONE 2026-09-09** (`docs/MODELS.md` 30.4,
D58), 165 Class B, weight store +0. **G1 refuted**: MSL moves by zero events. **G2 and G3
held. G4 refuted at 79/81 and its stop fired** - the replacement tightens on two channels,
so a dimensionless form is not uniformly a relaxation. Not adopted; the multiplier stays
unswept.

**B. Work item 9.15, the command-context arms - DONE 2026-09-09** (`docs/MODELS.md` 32.7,
D60), 165 Class B, weight store +153, 33.3 min at 4 workers. **K1 refuted**: withholding
the commands is worth +8 events, -5 false alarms and a lower alarm rate at once, so the
command context is negative under the published training too. **K3 refuted and its stop
fired**: sigma's variation doubled with commands attached and stayed far below the bar, so
C2 stopped and K4 was not adjudicated. **K5 held.** **(!) K2 has no verdict** - the
`eps_mult` sweep 32.3 pre-registered for C1 was never implemented, so the matched-rate
in-range contextual figure does not exist and no per-event comparison is drawn. That
comparison is owed.

**B1. Work item 9.16, the dial and the owed comparison - DONE 2026-09-09**
(`docs/MODELS.md` 33.7, D61). **M3 held** so the dial is usable; **M1 and M2 refuted** at
the measured point - swept to 0.4465%, C1 catches **6 of stage 4's own 38** against stage
4's **10/38**, so **stage 4's 10/38 stands unreplaced and the front page does not change**.
**K2 is discharged and HELD.** C1's large advantage over Arm T lives at 1.88% nominal and
does not survive being brought near a flyable rate. **The grid refinement ran the same
day** (33.8): resolved at 0.01 across the crossing, 55 points, with the caught set retained
at every one. **M1 and M2 are refuted robustly** - the best count at or under the target is
**6 against stage 4's 10**, and even given a rate 10% **louder** than stage 4's, C1 reaches
only 8. There is no operating point near 0.6820% (was 0.6838%) where C1 competes. Also corrected
here: C1 carried a second unintended lever (T-g, the published target length), **worth zero
events** - both numbers kept.

**B2. Arm 2, label-free forecaster selection - DONE 2026-09-09** (`docs/MODELS.md` 34.8,
D61). 512 fits over 32 configurations on a seeded 16-channel subsample, winner chosen only
on held-out nominal validation error. **The winner is the published configuration with the
cell swapped** - `gru`, l_s 250, hidden 80, epochs 35. **P3 held on one axis, P4 refuted,
P5 refuted**: more training makes held-out error consistently worse, because the published
`restore_best=False` keeps the last weights rather than the best.

**B3. Arms S1-S4 - DONE 2026-09-09** (`docs/MODELS.md` 35.7, D62). A transition-aware floor,
per-channel calibration, both, a 3-seed ensemble, gradient-boosted trees and an
isolation-forest control. **None beats stage 4. Four cannot reach its alarm rate at all.**

**B4. Per-event forensics on all 38 - DONE 2026-09-09** (`docs/MODELS.md` 36), 55 Class B,
weight store unmoved. **10 caught, 18 lost in the decision layer, 10 invisible in the
residual.** Of the 28 misses, **15 die in pruning and 13 below threshold**; within the 18
recoverable, **13 die in pruning**. An oracle per-channel threshold at the frozen arm's own
quiet rate would reach **28 of 38**. And **all ten "invisible" events show z > 3 in the
first derivative or in the disagreement across the ten predicted horizons**, neither of
which reaches the decision layer today. **The next work is the alarm rule, not the
network.** **Seven arms and a stride item are now registered and not run** (`docs/MODELS.md` 37 and 38):
pruning demoted to a registered negative result, a causal normalisation, persistence by
run length, CUSUM, an EVT peaks-over-threshold threshold (the priority arm), adaptive
conformal inference, and the union of them -- on a channel-disjoint 19/19 split committed
before any sweep, with the union budget **fitted jointly rather than allocated**.
**D62's freeze stands until one of them beats stage 4 at a matched rate.**

**THE PIPELINE IS FROZEN (D62)** on stage 4's configuration: `gru-telemanom`, per channel,
univariate, no commands, published dynamic threshold swept to 0.6820% (was 0.6838%), **10 of 38**. **The
C++ port must carry the dynamic threshold**, which `flight/` does not yet have - it
transcribes D25's static quantile alone. S1 and S2 are not adopted.

**(!) THE FREEZE IS SUPERSEDED 2026-09-10 (D65), and the sentence above is kept.** Arm 2 --
residual plus first derivative, trailing-standardised -- reaches **EVAL 17/19 against 4** at
the same 0.6820%. **Nothing is adopted and no flight configuration is named.** What the C++
port must carry grows by one stream: the dynamic threshold **and** the first derivative with
its trailing standardisation. Pruning (P1.1) and CUSUM (P4b.1) close as clean negatives;
arms 4a, 6 and 7 and prediction P2.3 are **NOT ADJUDICATED** and no number of theirs is
reported.

**Roadmap item 1, the C++ port, is PRE-REGISTERED and not yet written**
(`docs/MODELS.md` 39, 2026-09-10). `model.bin` stays at **`format_version` 1** -- the
dynamic rule's constants become `constexpr` and the selectable mode remains a version-2
decision after Phase 3. Three departures are registered with their costs: the window
**already trails**, so what is at stake is emission timing and **guard cells are not a free
way to remove it** (10.7 measured 38/46 -> 34/46 on `m1-g8.9.10` and the reverse on
`m1-ss5`); **backward dilation cannot be emitted** and is dropped, its price registered as
**N6 and unadjudicated**, because measuring it needs a read that has not been taken; and the
footprint grows **312,112 -> 581,488 B**, which is a new numbered prediction beside 19.8's
F1 rather than an edit to it. **(!) THE PORT IS WRITTEN AND GREEN, 2026-09-11** (`docs/MODELS.md` 39.13). `TrailingWindow`,
`DynamicThreshold` and `DerivativeStream`, held to the reference at **1e-5 with the emission
flag exact on every step** (N1 held); determinism bit-identical across processes with 2,170
samples of carried state (N4 held). **`sizeof(Detector)` is 603,032 B against N3's predicted
581,488 -- N3 FAILED at +3.70% and the account is itemised, not the band moved.**
**(!) It was 603,024 when 39.13 measured it**, and `99fffd2` then added
`U32 m_peakChannel` so rule 4 could name a channel; padding took it to 603,032 and the
footprint check's 2% band was wide enough to hide 8 B. The band is now an equality
(`flight/test/Footprint.cpp`), and 39.13's figure stays as the record it is (39.13.5).
**Three decision layers run and one emits**: `crossing()` and `emitted()` are still D25's
static quantile, because D65 names no flight configuration and promoting either new rule
would make the port the adoption decision. **On the second read (2026-09-11), with the cut floating as the frozen arm's does: neither
departure costs recall on this data.** **N6** -- forward-only dilation -- reaches **23 of 38
at a matched 0.6820%, EVAL 11/19 against the frozen arm's 4**, a strict superset. Its band
**FAILED in the opposite direction to its intent**, having asked what the departure would
cost, **and the +13 is confounded with the dial**, which moved 0.5506 -> 0.5009 inside the
band 38.15 says re-admits pruned steps -- so it is not a clean measurement of the lever.
**N5** -- guard cells -- **remains NOT ADJUDICATED**: its cut pins at 1.0 and the rate lands
14% below target, so no matched-rate number of it is reported; at that quieter rate it
catches 16 of 38 against 10, which is dominance rather than a matched comparison.
**(!) THE FLIGHT CONFIGURATION IS ADOPTED, 2026-09-11 (D68), and the sentence above is
kept.** `emitted` now follows **`max(z_residual, z_derivative)` against one calibrated cut**
-- exactly what D65 measured twice -- on a **`param_version` 2** PARAMS block. The byte
layout does not change, so **`format_version` stays 1 and D30's freeze is untouched**; what
changes is which statistic the threshold cuts, and **both readers now refuse a
`param_version` they do not know** with `BAD_PARAM_VERSION`. Measured on tier `p1`: the same
bytes with the same cut cross **339 times as version 2 and 0 as version 1**. A version-1 file
still gets D25's rule, so every committed `.vec` vector still passes. The dynamic threshold
is carried and reported and drives nothing. **One arm, one dataset, n = 19 per half,
UNDERPOWERED, and no early-warning claim.** A third read -- both dilations at one fixed cut
-- is **owed and not taken**; it refines N6 and blocks nothing.

**(!) C IS BUILT, RUN AND ADJUDICATED, 2026-09-14** (`docs/MODELS.md` 42.9). The toolkit
fitted a model on the testbed's own healthy telemetry -- 8 channels, `param_version` 2, cut
**20.062910** derived as the 0.999 quantile of its pooled nominal statistic -- and ten seeded
runs and ten healthy controls were scored by a harness that **links `Sentinel::Detector`
itself**. **All seven predictions HELD**: positive lead on **10 of 10**, median lead **9,774.5
timesteps** and **T4 fraction 0.871**, false alarms **0.1608%** against the toolkit's own
held-out prediction of 0.1830%, worst-case tick **0.033%** of the 1 Hz period.
**(!) The first warning is not the detection, and the healthy control is what shows it**: the
naive reading gives a lead of +16,525 from a warning at tick 2,700 that the **healthy run makes
too**. Ground truth by construction removes it -- same seed, same plant, one scalar differs --
leaving **10 fault-attributable warnings, the first at tick 8,100**, 100 ticks after injection.
**(!) The warning time is in TIMESTEPS; the harness's 0.131 s is compute time, not a warning
time.** At the deployment's 1 Hz the median is 2 h 43 min, and at another rate it is a different
number of seconds and the same number of ticks.
**(!) Ten runs are not ten independent systems.** The first limit crossing lands at 19,224 or
19,225 in **all ten** -- a two-tick spread -- so the effective `n` is close to one and 10 of 10
is about reproducibility, not power. One fault mode, one model, one testbed. **No early-warning
claim about spacecraft telemetry follows**: 37.7a's 0 of 10 and 45.6's *less late, not early*
both stand beside it. **The GDS recording is owed.**

**C. Work item 11, the F' Ref physics testbed** (Phase 3, pulled forward). Coupled
current/heat/temperature/voltage, 8-12 channels, real dictionary limits, real clock, faults
seeded **in the physics** and in-limits throughout for the contextual family, ground truth
by construction. *Done when* a model gate runs on it at a matched rate reporting in-limits
catch rate, **time-to-limit-trip** and manoeuvre false alarms - and the live GDS recording
exists.
**(!) PRE-REGISTERED AND NOT YET BUILT, 2026-09-14** (`docs/MODELS.md` 42). Seven
predictions, four departures, zero bucket operations. Two of the departures were found in the
reading rather than anticipated: **F' evaluates telemetry limits on the ground, not onboard**,
so a warning time subtracted across two clocks is not reported at all and the testbed evaluates
the dictionary's own limit values against the onboard time base; and **40.3a's data floor sets
the length of every run** -- 250 + 2,100 of warm-up plus 4,200 of calibration is **6,550 ticks,
which at `SentinelRef`'s 1 Hz base clock is 1 h 49 min per healthy run** and 36 hours for a
false-alarm figure over twenty. The registered way out raises the base clock and keeps the
clock real; a **simulated time source is refused**, because a real clock is the one thing this
testbed exists to provide.

**D. Work item 10, in-orbit threshold recalibration.** File uplink, human-approved reload,
on `model.bin` **version 1**: PARAMS is separately CRC'd and carries its own
`param_version`, so no format change is needed. `docs/PHASE2.md` 5b states what it must
survive. *Done when* it is exercised end to end on the Ref. **A selectable dynamic-threshold
mode is a separate later item and a `format_version` 2 decision**, taken after Phase 3 shows
which rule a mission needs (D30, `docs/MODEL_FILE.md` 11).

**E. The ESA regression of `gru-zscore`** (H3), deferred by `docs/MODELS.md` 31.6 and now
moot for adoption since H4 fired. *Done when* it is priced against the multivariate variant
with a one-fold timing smoke first.

**F. Work item 12, the toolkit.** One command: mission telemetry in, `model.bin` out.
*Done when* it runs on the synthetic fixture at zero cloud operations, is dimensionless
throughout (D55), prints the pre-launch sanity report at the end of every calibration
(**reported, never targeted**), offers the three tiers, and a newcomer with no ML background
produces a `model.bin` in under an hour. **(!) PRE-REGISTERED AND NOT YET WRITTEN,
2026-09-11** (`docs/MODELS.md` 40). The cut is **derived** -- the `(1 - q)` quantile of the
mission's own pooled nominal fused statistic -- and the rate curve is **reported** around it
and selected at no value, because `docs/HARNESS.md` 1 struck the target-rate framing. Two
silent defaults are registered as stop conditions: `param_version` defaults to **1**, so a
fused z-score cut written without setting it is a version-1 file carrying a version-2
statistic; and `Hyper.cell` defaults to `lstm` where the adopted forecaster is the GRU.
**(!) The headline is univariate and the toolkit's default file is multivariate** (40.3): the
17 of 19 was measured one model per channel on SMAP/MSL and is **never quoted beside a
multivariate `model.bin`**. **No full-mission fit is started** until a single-channel wall
clock is measured and the options are costed (40.13).
**(!) TIERS 1 AND 2 ARE BUILT AND GREEN, 2026-09-11** (`docs/MODELS.md` 40.14).
`src/sentinel_toolkit/`, invoked as `PYTHONPATH=src python -m sentinel_toolkit`. On the
generated fixture's longest healthy run -- 18,000 timesteps by 7 channels, zero cloud
operations -- it trains, calibrates label-free, writes a `param_version` 2 `model.bin` and
reads it back through the flight-mirroring loader. **Seven of the eight predictions held**;
**T1 is NOT ADJUDICATED** and is reported as unrun rather than as the half that passed: the
round trip holds byte-identically, and the 1e-5 accuracy comparison needs a vector tier
generated from the toolkit's own model, which was not built. **T6 held at 0.75x.**
40.7's moves are proven: `to_spec` and the trailing statistic left `scripts/` for `src/`,
and **every committed golden and fused vector regenerates byte-identically** afterwards.
**(!) 40.3a is a departure the pre-registration did not know about**: the trailing span the
statistic standardises against is **compile-time in the flight component and is not a field
in the model file**, so neither a mission nor the toolkit may choose it -- which sets a
floor of 250 + 2,100 of warm-up plus 4,200 of calibration before any figure can be reported.
**(!) TIER 3 RAN 2026-09-11 AND THE COMPUTE GATE IS DISCHARGED** (`docs/MODELS.md`
40.14.1): ESA-ADB Mission 1's twelve-channel gate set, after a smoke whose projection
matched exactly, for **15 Class B and 1 Class A**. **73.5 s and 2.97 GiB peak** -- no
provisioning and nothing to parallelise, since more threads measured slower on a single
multivariate fit. Cut **11.875282**, sanity rate **230 / 283,563 = 0.0811%** against a
calibration half of 0.1002%, **ratio 0.81x**, so **T6 holds on real telemetry** and not
only on a fixture. **(!) The healthy window is 7.7% of the archive** -- 1,138,952 of
14,728,316 timesteps -- which is the finding: a mission archive is not a training set, and
every cost estimate made against the grid was conservative for that reason. The labels were
read and they chose the **window**, not the cut; `fit_model` has no parameter for a label.

**G. Work item 13, Phase 5**, scoped in `docs/PHASE5.md`. *Done when* the toolkit exists,
Phase 3 has measured something, and the Armadillo static-allocation claim is discharged with
a citation - it is marked UNVERIFIED and stays marked until then.

**H. Housekeeping.** Licence (still "not yet selected"), pre-publication sweep, public
release. *Done before* the repository goes public. The BSD clause-3 obligation travels with
it: no document may present this project as endorsed by Caltech or JPL.

**I. The customer branch, D69, 2026-09-11.** `master` carries **customer documents** rather
than `dev`'s own prose: `README.md` and `docs/STATUS.md` replaced, `Objective.md` and
`docs/RESULTS.md` replaced by a design and an evidence document, `docs/DECISIONS.md` and
`docs/NARRATIVE.md` removed from the branch and kept whole on `dev`. **Nothing is deleted
from `dev`.** The reason is measured rather than aesthetic: of four figures `master` states
about `dev`, **two were stale** -- each correct on the day it was written -- and **a third
went stale one commit later** when `docs/MODELS.md` 40 landed (D69.1). **DONE 2026-09-11.**
`master` carries four customer documents -- `README.md`, a design document, an evidence
document and a status document -- and `Objective.md`, `docs/DECISIONS.md`,
`docs/NARRATIVE.md` and `docs/RESULTS.md` are off the branch and kept whole here. **96
files to 94, 2.96 MiB to 2.41.** `flight/` and `fprime/` were refreshed in the same commit,
because until then the public branch carried the lint target that could not fail.
`tests/test_master_documents_are_current.py` re-derives every figure `master` states from
the source that produces it -- the footprint from the assertion, the refusal count from the
enum, the F' coverage from that test's own loop bound, the headline from D65's table -- and
checks the citations in `master`'s documents, which **nothing had ever read**: `--master`
mode resolves `dev`'s documents under the curated branch's rules and never opens the
branch. **The debt register is empty**: all four stale figures died with the old
`README.md`, and the guard is now proved by a probe rather than by a live defect.
**(!) `main` IS RETIRED, 2026-09-14** (D69.2), in the order D69 set: the GitHub default moved
to `master`, the live prose was rewritten, `origin/main` was deleted and then local `main`.
`git rev-list master..main` was **0** before it went and all nine tags were on `dev`, so no
commit object and no Release was lost. Three of D69's nine named sites turned out to be dated
`CHANGELOG.md` entries and were **kept with a rider rather than edited**, which leaves seven
live sites rewritten. **External `/blob/main/` links will 404**: none exists inside either
tree and none outside can be grepped for.

**J. Three more pre-registrations, all written 2026-09-14 and none run.** Zero bucket
operations between them.

- **The Raspberry-Pi envelope** (`docs/MODELS.md` 41), which `Objective.md` 12 carries as
  Phase 4. Nine predictions against a **named board**, with the rule that a different board
  re-derives the absolute bands **as a rider before it is switched on**. Two dimensionless
  predictions are written to survive that change. **(!) Measured while writing it: a fresh
  checkout runs 2 of the 7 golden tiers**, because `g3.bin` and `g4_*` are gitignored and
  `GoldenVectors.cpp` skips an absent tier silently -- so the ARM proof would have passed on
  3 and 7 channels and reported success. R2 therefore asserts the tier count. **Nothing exists
  at 16 channels**, the compile-time maximum, and the timing fixture for it is generated on the
  target rather than committed, because tracked content is **7.5016 MiB** (2026-09-21, after 76) and already inside
  39's N8 middle band -- **but D77 re-derived N8 as a vectors-only figure**, so the band that
  applies to a new tier is the **vectors'**: they are **2.0924 MiB**, inside N8's HOLD band
  with **5,146,037 B** under its 7 MiB stop. The binding limit on the tree as a whole is now
  D64's 8 MiB cap, **522,581 B** away, and it is prose rather than vectors that is spending it -- it was 768,620 B on 2026-09-20 and five sections of record have cost the difference.
- **The stride reduction** (`docs/MODELS.md` 43), which 37.7a recorded and declined to
  register. Its cost half is **arithmetic once 41's R6 lands** -- `mean(S) = m(1 + (R-1)/S)` --
  and its structural half is derivable from A6's committed table. **(!) The lever is confounded
  with the statistic by construction**: `SOLVE_WINDOW = ERROR_WINDOW + STRIDE`, so a stride
  change moves the window, `MAX_SEQUENCES` and therefore the footprint. **(!) AND S5 IS
  WITHDRAWN BEFORE ITS READ WAS SPENT, 2026-09-14** (43.8). The adopted rule **emits per tick**
  -- `Detector.cpp` compares the fused score to the cut every tick -- so the segment-end latency
  A6 measured belongs to the **frozen** arm, and the flight equivalent, `dynamicEmitted()`, is
  exposed and **never consumed**. There is no latency in the flown path for a smaller stride to
  remove, and the only route left runs through `z_residual`, which D65.3 measured as deciding
  nothing. **165 Class B and 1 Class A not spent.** S1-S4 stand; the stride is a **compute**
  parameter and no longer a latency one.
- **The derivative-only ablation** (`docs/MODELS.md` 44). **(!) RUN 2026-09-14** (44.7, D65.3),
  165 Class B and 1 Class A after a smoke whose projection matched exactly, weight store
  unmoved. Both gates pass: the frozen arm returns 0.6820% and 10/38, the fused arm returns
  D65's own 17/19. **Derivative-only reaches EVAL 17/19 against 17/19 -- identical -- at the
  same 0.6820%**, on a cut that falls from 5.288128 to 4.431455, and it is a **strict superset
  on all 38** whose two extra events are both on TUNE. **(!) `z_residual` reaches the cut on
  none of the 30 events the fused rule catches**: median 0.080, maximum 4.652 against a cut of
  5.288, negative on 15 of 30 -- **though the 30 span only 25 distinct onsets** (45.6.2), so
  the effective n is lower. **The residual term is not weak, it is inert**, and D65's
  finding on this data **is the first derivative**. Nothing is adopted; D68 stands. It frees no
  memory, and it is one arm, one dataset, n = 19 per half, UNDERPOWERED.

**(!) The flown rule's lead time is measured, and it is less late rather than early**
(`docs/MODELS.md` 45.6, 2026-09-14). 165 Class B and 1 Class A after a smoke, both projections
exact, weight store unmoved, both gates passing. On the **9** events 37.7a also caught, the
flown rule's median lead is **-2.0 against the frozen arm's -68.0, better on every one of the
nine** -- and on the five where both still fire late, it is less late every time, which is the
segment-end latency it does not carry. **LD1 HELD.**
**(!) LD2 FAILED and the failure is not a finding of early warning.** The raw median is
**+141.5 with 18 of 30 positive**, and its band required that a positive median be checked
twice. It was: at 0.6820%, **P(an alarm somewhere in a 200-tick lookback) = 74.6%**, so chance
alone gives about 22 of 30 and the rule manages 18. **The positive median measures the nearest
routine alarm in the lookback, not a precursor** -- the upper quartile is +186 against a cap of
+200. **The +141.5 is not quoted as a lead anywhere.** And the 30 events span only 25 onsets,
with four channels crossing at one timestep, so the effective n is far below 30. **No
early-warning claim: that stays with work item 11.**

**The four attribution sites are corrected on both branches** (45.1.1).

**(!) And 45.6.1's refutation rests on a computed null, which 46 registers as measurable.**
The 74.6% assumes alarms are independent across nominal ticks; they run in streaks, so the true
figure is **lower** -- which would make 18 of 30 look **less** like chance, not more. **45.6.1
is therefore the conservative reading and could be wrong in the interesting direction.** 46
scans matched decoy onsets drawn from nominal regions, seeded and fixed before the answer, so
the null is measured. **165 Class B and 1 Class A. Not taken -- it blocks nothing, because the
+141.5 is quoted nowhere.** If it overturns 45.6.1, its own band sends the result to a second
dataset before a sentence of it is written.

**What 45.1 found, kept because it is why 45 exists.** Four live sites stated 37.7a's median
**-63.5**, 0 of 10 positive, as the project's lead-time measurement without saying it was
measured on the **frozen** arm, which emits at a segment's end. All four *disclaimed* early
warning, so nothing overclaimed; the defect was the other shape -- **a measured negative
reported where the truth was unmeasured**. All four now carry the attribution (45.1.1), two of
them on the customer branch.

**D68.1 is withdrawn before its read was spent, 2026-09-14.** Dilation lives in
`DynamicThreshold` and reaches nothing the flown rule consumes: `eps()` has no readers,
`emitted()` is read only by `dynamicEmitted()`, which nothing calls, and `zResidual` -- the one
consumed output -- is the trailing window's z, which dilation does not touch. **165 Class B and
1 Class A not spent.** The question remains a real one about telemanom's published rule and is
no longer a debt of this component's.

**Owed and registered, not argued (D68.3).** A `param_version` 3 rule dropping the residual
term is arguable after 44.7 and is not argued: one arm, one dataset, n = 19 per half. It needs
the same ablation on a second dataset and its own pre-registration before it is proposed.
`format_version` would stay 1 either way; D30 is untouched.

**Carried open.** D5, the tier ladder, is the only OPEN decision in the register. D14's
weight-cache key is still positional. D21's `error_buffer` effect on alarm width is
undecided. `Objective.md` 13 item 8, the injected-fault sensitivity study on real telemetry,
is unbuilt. Arm H's weights are not persisted, so that arm is reproducible only from its
seed.

## 8. Where things live

- **Code**: `src/sentinel_data` (ingest; the `r2.py` hook), `src/sentinel_eval` (the
  referee - never imports a model), `src/sentinel_models` (the players, the NumPy reference,
  the Level 1 baseline reference), `src/sentinel_export` (the `model.bin` writer and
  reader), `flight/` (the C++ core, Level 1, and their golden vectors), `fprime/` (the F'
  library: the component, a deployment, `settings.ini`; the framework checkout and tool venv
  are gitignored and rebuilt by `scripts/fprime_setup.sh`), `scripts/`, `tests/` (**931 tests**, zero R2 operations).
- **Evidence**: `third_party/telemanom/`, the published source pinned at commit
  `2e6c5b6c`, vendored as evidence and never a dependency (D53). `docs/TELEMANOM_EXCERPTS.md`
  indexes every citation into it.
- **Data**: R2 bucket `fprime-sentinel-data`, manifest-addressed reads only, never LIST;
  50,000 Class A and 50,000 Class B per calendar month, hard; tripwire 1,000 per run; every
  operation in the ledger (`docs/DATA.md`). **No telemetry on local disk, ever.**
- **Outputs**: `runs/`, gitignored; every number in every document is read from an artifact
  there and cited by path.
- **Documents**: `docs/INDEX.md` is the map, one sentence each.
- **History**: **two branches.** All work lands on `dev` (tags `wi1`-`wi9`) and it is
  **never rewritten**; `master` takes a snapshot when something is done. Nothing is ever
  force-pushed.
  **`master` is the public branch and the repository default** (D67, D67.1, D69): **116
  files, 2.5562 MiB**, its own four customer documents, and `flight/` and `fprime/`
  byte-identical to `dev`. Every figure it states about this repository is re-derived by
  `tests/test_master_documents_are_current.py`, which compares **trees rather than walking
  ancestry** -- the two branches **share no commit**.
  **`main` was retired 2026-09-14** (D69.2): an earlier snapshot branch, every commit of it
  reachable from `master`, all nine tags on `dev`, and the external-link cost stated in that
  rider rather than dismissed.
  The repository is **private**, so no `LICENSE` is required yet, and **licence selection is
  the one remaining item** before it can go public.
  *(This bullet was rewritten in one pass on 2026-09-14. It had been amended three times and
  carried both 96 files and 94, and both "exists locally" and "pushed"; the superseded detail
  is in D67, D69 and `CHANGELOG.md` 0.6.41 to 0.6.43, which are the records.)*

## 9. Verify

```bash
.venv/bin/python -m pytest -q                                    # 931 tests
.venv/bin/python scripts/check_no_list.py                        # 124 files, no LIST, no glob
.venv/bin/python scripts/check_references.py                     # every citation resolves;
                                                                 # citations into an absent
                                                                 # fprime/lib/ SKIP (D79)
.venv/bin/python scripts/check_references.py --master            # D67's curated branch
PYTHONPATH=src .venv/bin/python -m sentinel_eval selftest         # 8/8, oracle 1.0
make -C flight test                                              # the C++ core and Level 1
make -C flight lint                                              # clang-tidy, three configs.
                                                                 # Without the F' checkout this
                                                                 # is PARTIAL and exits non-zero
                                                                 # (D79); LINT_ALLOW_PARTIAL=1
                                                                 # accepts a partial run

# the F' half, after one-time setup with scripts/fprime_setup.sh
cd fprime && . fprime-venv/bin/activate
cd Sentinel/Monitor && fprime-util check                         # the component
cd ../../SentinelRef && fprime-util build                        # the deployment
bash scripts/fprime_ref_patch.sh                                 # and F's own Ref

# one scoring run on the offline fixture, zero R2 operations
PYTHONPATH=src .venv/bin/python -m sentinel_eval run synthetic \
    --detector gru-smoke --detector rstd --no-sweep
```

Everything above costs zero bucket operations; a Mission-1 scoring task costs 15 Class B and
1 Class A, and a SMAP/MSL study 165 Class B and 1 Class A.
