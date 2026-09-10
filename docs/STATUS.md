# Project status and roadmap

**2026-09-09. Phase 2 in progress. The objective is re-framed** (`docs/DECISIONS.md`
D57): Sentinel warns before a limit trips using **every kind of context the telemetry
carries**, and "cross-channel means sensor-to-sensor" is retired as the sole thesis.
A ten-minute read. Every number is read from a named artifact under `runs/`, or from a
test that pins it.

**(!) This document was rewritten on 2026-09-09 and is a status, not a record.** The
previous version had grown to an eighty-item changelog of its own. Nothing is lost:
`CHANGELOG.md` carries every work item version by version, `docs/MODELS.md` carries every
pre-registration beside its outcome, `docs/DECISIONS.md` carries D1 to D57 with superseded
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
- **39 of 43** labelled contextual anomalies on SMAP/MSL stay entirely inside their
  channel's own historical range (D46). A limit check cannot see them by construction.
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
been measured against stage 4's **10/38 at 0.6838%** and none beats it. At or below that
rate: `WS`, the label-free forecaster winner, reaches **8/38**; `C1S` reaches 6/38; the
3-seed ensemble and the isolation forest reach 0; and **four arms - a transition-aware
floor, per-channel calibration, both together, and gradient-boosted trees - have no
operating point at or below 0.6838% at all**, which is D48 arriving for the fifth through
eighth arms. The full ladder is `docs/MODELS.md` 35.7.
| **paper** | Hundman et al., Table 2 | **25/36** | 84/105 (80.0%) | 84/96 (87.5%) | **12** | - |

**The headline number is stage 4's `10 of 38`**: in-range contextual sequences caught, at
0.6838% of nominal steps, by `gru-telemanom` scored per channel, univariate, without
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
v4.3.0's own Ref, and all **11/11** loader refusal codes degrade to the Level 1 statistical
baseline with the code named in the event, **0/11** failing the topology (D32-D37).

**Operations.** 214 Class A and 4,415 Class B for 2026-09, of 50,000 each, read from the
last artifact and never transcribed.

## 5. What we did, and why

- [x] Repository stood up documentation-first - decisions recorded before code, D1 to D57,
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
only 8. There is no operating point near 0.6838% where C1 competes. Also corrected
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
network.** **Three arms are now registered and not run** (`docs/MODELS.md` 37):
pruning swept, a richer causal detection statistic, and both, on a channel-disjoint 19/19 split
committed before any sweep. **D62's freeze stands until one of them beats stage 4 at a
matched rate.**

**THE PIPELINE IS FROZEN (D62)** on stage 4's configuration: `gru-telemanom`, per channel,
univariate, no commands, published dynamic threshold swept to 0.6838%, **10 of 38**. **The
C++ port must carry the dynamic threshold**, which `flight/` does not yet have - it
transcribes D25's static quantile alone. S1 and S2 are not adopted.

**C. Work item 11, the F' Ref physics testbed** (Phase 3, pulled forward). Coupled
current/heat/temperature/voltage, 8-12 channels, real dictionary limits, real clock, faults
seeded **in the physics** and in-limits throughout for the contextual family, ground truth
by construction. *Done when* a model gate runs on it at a matched rate reporting in-limits
catch rate, **time-to-limit-trip** and manoeuvre false alarms - and the live GDS recording
exists.

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
produces a `model.bin` in under an hour.

**G. Work item 13, Phase 5**, scoped in `docs/PHASE5.md`. *Done when* the toolkit exists,
Phase 3 has measured something, and the Armadillo static-allocation claim is discharged with
a citation - it is marked UNVERIFIED and stays marked until then.

**H. Housekeeping.** Licence (still "not yet selected"), pre-publication sweep, public
release. *Done before* the repository goes public. The BSD clause-3 obligation travels with
it: no document may present this project as endorsed by Caltech or JPL.

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
  are gitignored and rebuilt by `scripts/fprime_setup.sh`), `scripts/`, `tests/` (**605
  tests**, zero R2 operations).
- **Evidence**: `third_party/telemanom/`, the published source pinned at commit
  `2e6c5b6c`, vendored as evidence and never a dependency (D53). `docs/TELEMANOM_EXCERPTS.md`
  indexes every citation into it.
- **Data**: R2 bucket `fprime-sentinel-data`, manifest-addressed reads only, never LIST;
  50,000 Class A and 50,000 Class B per calendar month, hard; tripwire 1,000 per run; every
  operation in the ledger (`docs/DATA.md`). **No telemetry on local disk, ever.**
- **Outputs**: `runs/`, gitignored; every number in every document is read from an artifact
  there and cited by path.
- **Documents**: `docs/INDEX.md` is the map, one sentence each.
- **History**: all work lands on `dev` (tags `wi1`-`wi9`); `main` advances only by one
  snapshot commit per approved checkpoint; neither branch is ever rewritten.

## 9. Verify

```bash
.venv/bin/python -m pytest -q                                    # 606 tests
.venv/bin/python scripts/check_no_list.py                        # 75 files, no LIST, no glob
PYTHONPATH=src .venv/bin/python -m sentinel_eval selftest         # 8/8, oracle 1.0
make -C flight test                                              # the C++ core and Level 1
make -C flight lint                                              # clang-tidy, three configs

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
