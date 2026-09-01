# Decisions

`Objective.md` section 14 records **what** was decided. This file records **why**,
what else was considered, and what evidence settled it. That reasoning is the
part that decays fastest and is the hardest to reconstruct: a decision with its
alternatives stripped out reads as an arbitrary preference a year later, and
gets re-litigated by whoever inherits it.

**Superseded entries are marked superseded. Nothing is deleted.** A decision that
turned out wrong is more informative than one that was always right, and a reader
who can see only the surviving choices has to take our judgement on trust.

Format: DECISION / DATE / CONTEXT / ALTERNATIVES / EVIDENCE / CONSEQUENCE /
STATUS. Updated in the same commit as the decision it records
(`docs/HARNESS.md`).

---

## D1. ESA-ADB is the primary dataset; SMAP/MSL is demoted to legacy

**DATE** 2026-08-24 | **STATUS** resolved

**CONTEXT.** telemanom -- the reference method -- was developed and published on
SMAP/MSL, so the obvious move was to evaluate on it and compare directly to
published numbers.

**ALTERNATIVES.** SMAP/MSL as primary, with ESA-ADB as a secondary set. CATS, the
synthetic stress test. OPSSAT-AD.

**EVIDENCE.** Two findings, and the second is the one that decided it.

1. Wu & Keogh (IEEE TKDE 2023) show SMAP/MSL suffers triviality, unrealistic
   anomaly density, mislabelled ground truth and run-to-failure bias -- and that
   a moving average and a standard deviation reach state of the art on it. *If a
   two-line script wins your benchmark, your benchmark is not measuring
   sophistication.*
2. Per the PATH dataset paper (Kaufmann et al.), SMAP/MSL's channels **are not
   synchronised with each other**. "82 channels" is 82 unsynchronised univariate
   streams.

**CONSEQUENCE.** The second point is fatal rather than inconvenient: cross-channel
detection is this project's entire claim, and there are no genuine cross-channel
relationships in that data to find. We could not have proven the core claim on
it -- not because the method is weak, but because the data physically cannot
demonstrate it. SMAP/MSL is reported for continuity only, flagged, never
headlined. External comparability to published ESA-ADB numbers is explicitly out
of scope for Phase 1 (Objective.md 12).

---

## D2. Normalisation is identity

**DATE** 2026-08-24 | **STATUS** resolved

**CONTEXT.** Nearly every anomaly-detection pipeline standardises its inputs, and
doing so was the default assumption when the loader was designed.

**ALTERNATIVES.** Per-channel z-score. Per-channel min-max. Per-channel robust
scaling. Cross-group rescaling.

**EVIDENCE.** ESA already min-max scaled ESA-ADB to [0,1] **within each channel
group** -- one shared range across a group's channels -- so the amplitude ratios
between related channels survive in the data as shipped.

**CONSEQUENCE.** Per-channel rescaling is refused, and the refusal is executable
rather than advisory. The fact that channel A normally moves ten times more than
its group-mate B is *the* information the cross-channel claim rests on; a routine
z-score erases it, and no model can recover it because the original scale is no
longer present anywhere. Our method would have looked weak when we had broken the
data ourselves. Cross-group spanning is accepted: those offsets are fixed,
invertible and uninformative, and a model absorbs them in its first layer.
Enforced at a chokepoint (`sentinel_eval/normalisation.py`) and by
`tests/test_no_per_channel_scaler.py`.

---

## D3. The gate metric is event-wise F0.5, never bare recall

**DATE** 2026-08-25 | **STATUS** resolved

**CONTEXT.** The pre-registered prediction for the trivial baselines was framed on
headline-cell recall. `mavg` then scored 29/31 on it, which read as the benchmark
being broken.

**ALTERNATIVES.** Recall on the headline taxonomy cell. Point-adjusted F1. Plain
F1. VUS-PR alone.

**EVIDENCE.** `mavg` bought that 29/31 with **5,535 alarms for 42 events**, at
24.9% precision, alarming on 39 of 48 commanded manoeuvres. On F0.5 it scored
0.028 against `rstd`'s 0.250. The prediction had been framed on a quantity that
carpet-bombing satisfies.

**CONSEQUENCE.** F0.5 leads every block; recall and precision appear beneath it as
its components; no recall is ever rendered without the precision it must be read
against. Point-adjusted F1 is quarantined to `--diagnostics` because one lucky
sample promotes an entire segment. Enforced by `tests/test_gate_metric.py`.
**Superseded in part by D9**: F0.5 is necessary and is no longer sufficient.

---

## D4. `m1-g8.9.10` is the primary recall set, promoted post-hoc

**DATE** 2026-08-25 | **STATUS** resolved

**CONTEXT.** `m1-ss5` was chosen as primary on point-anomaly coverage -- it is the
only channel set in ESA-ADB carrying all eleven point anomalies. The baseline run
then produced results that inverted the project's own thesis.

**ALTERNATIVES.** Keep `m1-ss5`. Use all 76 Mission1 channels. Use group 3.

**EVIDENCE.** Event footprint, measured after the fact:

```
  set                nch  MVGS  <1 cell   median      p75
  ss5 = group 8        6    38       18       30    3,056   <-- outlier
  groups 8+9+10       12    40        0    1,951    8,828
  group 3              8    14        0    3,594    5,776
  ALL mission1        76    47        0    1,749   11,331
```

Group 8 is the **only** channel set in Mission1 where headline-cell events
register as sub-grid-cell spikes. Groups 9 and 10 see the *same events* as
multi-hour subsequences. The original selection was made on point-anomaly
coverage, which was correct as far as it went; event footprint was the gap.

**CONSEQUENCE.** We changed the evaluation set after seeing a result we did not
like. However sound the reasoning, that is externally indistinguishable from
cherry-picking, and **the claim that the choice was made a priori is permanently
forfeit**. The enforcement is structural rather than promised: `m1-ss5` is
retained and reported beside the primary in every result, a `--only` run is
stamped `"partial": true` and barred from `docs/RESULTS.md`, and one twelve-channel
load scores both so the pair differs in channels alone.

---

## D5. Tiered capability architecture -- Level 1 / 2 / 3

**DATE** 2026-08-25 | **STATUS** OPEN, resolve before Phase 2

**CONTEXT.** F' spans JPL flagships with months of flatsat data and CubeSats with
a weekend of it (Objective.md 10). One model file cannot serve both.

**ALTERNATIVES.** Require sufficient mission data as a precondition. Ship a single
generic model. Ship nothing until data exists.

**EVIDENCE.** Level 1 is settled by a different argument than the one that
prompted the tiering: it is the **loader's safe failure mode**. A corrupt file, a
version mismatch, a failed CRC or a radiation bit-flip must degrade to a
statistical baseline with an event and an active-tier telemetry channel, never
fail the topology -- which is required regardless of how much data a mission has.

Level 2 rests on transfer learning, and the evidence is encouraging rather than
conclusive; see D6.

**CONSEQUENCE.** One C++ loader, one model file format, three tiers. Level 1
mandatory. Level 2 the intended shipped default once it exists, and it cannot be
built before the Phase 1 architecture gate because you cannot pretrain without
knowing what to pretrain. Full detail in Objective.md 14.10.

---

## D6. Command conditioning -- telecommands become model inputs

**DATE** 2026-08-26 | **STATUS** open, being implemented

**CONTEXT.** The decision grid could not fix the false-alarm rate: the best of 24
cells still alarmed on 15 of 48 commanded manoeuvres. The cause turned out not to
be the decision layer at all.

**ALTERNATIVES.** More thresholding. Wider channel agreement. Longer persistence.
Accepting the rate as a property of the data.

**EVIDENCE.** Hundman et al. feed the LSTM "prior telemetry values for a given
channel **and encoded command information sent to the spacecraft**", and their
Figure 3 shows that encoding letting the model predict a commanded event so it is
not flagged. We fed telemetry only. Our false alarms are on commanded manoeuvres,
resets and calibrations -- so we withheld the one input that makes them
predictable, then measured the detector's failure to predict them. ESA-ADB
requirement R7 asks precisely this of algorithms, and **every published baseline
scores zero on it**.

**(!) THE MAGNITUDE IS UNPROVEN, AND THIS CAVEAT TRAVELS WITH THE DECISION.**
No ESA-ADB paper has published a commands-on / commands-off ablation. In ESA's
own baselines, adding telecommands coincided with **worse** precision: Mission1
Telemanom-ESA-Pruned fell from corrected event-wise F0.5 **0.786 to 0.061**,
which the paper attributes to more target channels meaning more chances of false
detection. Those baselines cannot exploit commands and ours is built to, which is
a reason to expect a different result and **not** evidence that we will get one.
If commands do not help, that is a finding, and it points at richer command
features rather than at more thresholding.

**CONSEQUENCE.** A harness scope change (authorised, additive) to supply
telecommands through `Bundle` and `Context`, and a controlled ablation with
identical architecture, hyperparameters, folds and seeds. Until that ablation
exists, every figure in this repository is **telemanom-minus-commands** and is not
comparable to published telemanom false-alarm numbers.

---

## D7. Eleven priority-3 telecommands on Mission1, not fifteen

**DATE** 2026-08-26 | **STATUS** resolved

**CONTEXT.** ESA grades telecommands into priority levels 0-3 and feeds only the
priority-3 commands to its own Telemanom-ESA and DC-VAE-ESA baselines. The
briefing figure was "the 15 priority-3 commands", and 15 would have been used as
the input width had it not been checked.

**ALTERNATIVES.** Take the figure as given. Select by subsystem match. Use all
821.

**EVIDENCE.** Read directly from `annotations/telecommands.parquet` -- columns
`mission`, `Telecommand`, `Priority`, preserved verbatim by the ingest
(`prepare.py:235-238` uses a bare `pd.read_csv` and drops nothing):

```
                P0    P1    P2    P3
  mission1     345   323    19    11
  mission2       -     -   119     4
```

**Fifteen is the dataset-wide count. Mission1 holds 11.** `m1-g8.9.10` is a
Mission1 set.

**CONSEQUENCE.** Eleven is the number Layer 1 starts from, and 15 would have been
wrong on every result derived from it. Two related facts recorded with it: the key
is the **(mission, Telecommand) pair**, not the name alone -- 821 rows over 698
distinct names because both missions reuse `telecommand_N` -- and 17 of Mission1's
698 telecommands are declared but never executed, so any of the 11 falling in that
set is dropped before modelling. A dead input column would dilute the command
features with a constant zero and quietly weaken the ablation.

Recorded as its own entry because it is exactly the kind of correction that gets
forgotten and reappears in a paper.

---

## D8. The harness is closed to scope changes, never to correctness fixes

**DATE** 2026-08-25 | **STATUS** resolved

**CONTEXT.** "The harness is finished" was the working rule. A correctness defect
was then found in `sentinel_eval` and the rule appeared to forbid touching it.

**ALTERNATIVES.** Fix it quietly. Work around it in the model layer and leave the
defect. Treat the harness as immutable.

**EVIDENCE.** `Bundle.subset` rebuilt its truth from the labels alone and never
read `self.valid`, so unobserved timesteps were counted as scorable nominal time
on every subset -- the free specificity `docs/HARNESS.md` section 1 explicitly
refuses for gaps and invalid segments. A referee that computes the wrong number is
not a fixed reference point; it is a wrong one, and freezing it does not make its
output true.

**CONSEQUENCE.** Closed to scope changes -- no new features, no new metrics, no
model work bleeding into referee work. Never closed to correctness fixes, with two
obligations: escalate before making one, because the person who finds a defect is
usually the person it is blocking and "fix it and move on" reads later as editing
the referee to suit the player; and record **both** numbers when a fix moves a
published one. The latter earned its keep immediately -- the subset fix moved 40
timesteps and zero event-wise counts, and that is only demonstrable because both
tables were kept.

---

## D9. Lead time is a gate metric; a negative one disqualifies

**DATE** 2026-08-26 | **STATUS** resolved

**CONTEXT.** Every metric the harness computed answered *did you catch it*. None
answered *how long before*, despite early warning being the project's headline
claim (Objective.md 2 and 4).

**ALTERNATIVES.** Leave it to Phase 3, where time-to-limit is validated on the F'
Ref deployment with a real clock. Report it as a supporting figure. Do not measure
it.

**EVIDENCE.** The metric was added, and its **first application overturned which
detector looked better**. Before it:

```
  lstm-quantile    F0.5 0.421   rare-event FA  1/48    <- better on both headlines
  lstm-telemanom   F0.5 0.269   rare-event FA 22/48
```

After it:

```
  lstm-telemanom     +26 timesteps    5 of 37 late,  alarms 204 wide
  mavg                 0              fires at the event boundary
  lstm-quantile     -122              ALL SIX catches late, alarms 1,151 wide
  rstd            -1,512              the floor -- 2 of its 3 catches late
```

**CONSEQUENCE.** A configuration whose median lead time is negative is **not a
candidate** for the flight configuration, whatever its other numbers -- a rule,
not a preference. `rstd` remains the F0.5 floor for comparability and is
disqualified on timeliness; it was never a flight candidate and that was
unknowable before the metric existed. `lstm-quantile` goes the same way despite
winning both prior headlines. **`lstm-telemanom` is currently the only surviving
candidate**, and its 26-timestep head start is the budget every subsequent layer
spends from. Several precision mechanisms buy their gains by waiting: the
persistence sweep moved lead time from +26 to -37 as N rose, for almost no change
in F0.5.

---

## D10. Top-k by running insertion, not `np.partition`

**DATE** 2026-08-25 | **STATUS** resolved

**CONTEXT.** Channel agreement needs the k-th largest per-channel score at every
timestep.

**ALTERNATIVES.** `np.partition` along the channel axis. Full sort. Materialise
the `(T, C)` score matrix and index it.

**EVIDENCE.** `np.partition` copies its input. On the largest training window that
is `11,046,237 x 12` float32 -- **530 MB**, allocated beside an 843 MB resident
bundle.

**CONSEQUENCE.** A running insertion over columns keeps the top few in `(depth, T)`
and never materialises the full matrix, at 132 MB for depth 3. Same result,
bounded memory, and it answers every value of k from one pass -- which is what
makes a decision-layer grid cost one scoring pass instead of twenty-four.

---

## D11. Command impulses are stored; the decay trace is derived

**DATE** 2026-08-26 | **STATUS** open, being implemented

**CONTEXT.** A bare command impulse is unlikely to be enough for the model: the
response transient outlasts the command by many timesteps, so a short
exponentially decaying "recently commanded" feature is wanted alongside it.

**ALTERNATIVES.** Store both features. Store the decay only. Store a sparse event
list and rasterise per window.

**EVIDENCE.** Over 14.7M timesteps and 11 commands, `uint8` impulses cost **162
MB**; a float32 decay trace alongside costs **648 MB**. The decay is a one-line
recursive filter over the impulses.

**CONSEQUENCE.** Impulses are stored and the decay is computed per window inside
the model. **A decay trace costs four times the memory of the thing it derives
from**, which is the whole argument. Recorded because the reverse -- precomputing
a derived feature because it is convenient -- is the obvious thing to do and is
wrong at this scale.

---

## D12. Weights persist under `runs/`; Rule 1 restated

**DATE** 2026-08-25 | **STATUS** resolved

**CONTEXT.** Rule 1 read "the Mac is a pipe, not a store", which appeared to
forbid a trained model surviving the process that produced it. Every run
therefore refitted: 45 minutes per channel set, repeated for every experiment that
changed nothing about the model.

**ALTERNATIVES.** Keep refitting. Cache in memory only. Commit weights to the
repository.

**EVIDENCE.** Rule 1 exists to stop this machine becoming a **data** store. 358
KiB of learned parameters is not telemetry, is not parquet and is not a cached
dataset; it is an output, the same category as the scorecards already written
under `runs/`, which `tests/test_no_local_persistence.py` already exempted.

**CONSEQUENCE.** Weights persist under `runs/_weights`, gitignored, keyed by a
content digest over hyperparameters, channel set, fold window and a strided sample
of the data -- so a stale checkpoint cannot be found rather than being guarded
against. `--no-cache` refuses them and refits, and no result enters
`docs/RESULTS.md` until reproduced that way. The `runs/` exemption was widened and
the dataset ban **tightened** in the same change: parquet, pickles and archives
are now refused everywhere in the tree including `runs/`, which the previous test
skipped wholesale.

---

## D13. The quantile-threshold branch is closed, on structural grounds

**DATE** 2026-08-26 | **STATUS** resolved

**CONTEXT.** `lstm-quantile` held the best event-wise F0.5 in the project (0.421)
and the best rare-event false-alarm rate (1/48), on the *same weights* as
`lstm-telemanom`. It had never been swept across channel agreement. Before
building four further layers on the dynamic-threshold branch, it was worth
establishing that the branch was the right one -- a grid that could only redirect
what came after it.

**ALTERNATIVES.** Proceed on the dynamic threshold without checking. Adopt the
quantile branch on its F0.5 and adoption numbers. Carry both forward.

**EVIDENCE.** Twenty-four cells, persistence 1/5/20/60 against agreement 1/2/3,
both channel sets. **No cell achieves positive median lead time.** The best is
-100 timesteps; the 75th percentile is negative in every cell, so three quarters
of catches are late in every configuration. Recall is *identical* at 6/46 in all
24 cells, as is headline-cell recall at 6/32 and point recall at 0/11 -- agreement
cannot change a verdict when the detector fires 21 times in eleven million
timesteps. Persistence makes it monotonically worse, -122 to -181.

**The mechanism, which is why this closes rather than merely loses.** The two
rules respond to different quantities. A nonparametric dynamic threshold measures
error against the *local* recent error scale, so a rising error crosses it at the
**onset** of divergence. A global quantile measures error against the 99.9th
percentile of years of training scores, so the error must grow to an absolute
**magnitude** before it crosses -- and growing takes time. Neither channel
agreement nor persistence touches this: both make a detector fire *less*, and this
detector's defect is firing *late*.

**CONSEQUENCE.** The branch is closed and will not be reopened by better tuning,
because tuning is not what is wrong with it. The programme continues on the
dynamic threshold -- the only rule that has produced a positive lead time. The
quantile variant is retained as a reported diagnostic, because the contrast
between two decision rules over identical weights is the clearest evidence the
project has that the residuals carry the signal and the thresholding decides what
is done with it.

Recorded as a decision rather than an experiment because the finding is
structural: it says something about what a global threshold *is*, not about what
this one scored.

---

**(!) REVISITED 2026-08-27. THE STRUCTURAL CLAIM DOES NOT HOLD. The branch stays
closed on its measured numbers; the reason given for closing it was wrong.**

D13 says a global threshold must fire late *by construction* -- it responds to
absolute magnitude, and growing to a magnitude takes time. That generalisation is
contradicted by direct measurement.

`scripts/oscfar_curve.py` on a clean calibration pool produces cells in which the
**floor** -- a global quantile of nominal residuals -- decides almost every
window, and they fire **early**:

```
  set          rule                binding   median lead   recall    headline cell
  m1-ss5       independent 0.10%     0.011        +28.0     24/42          18/31
  m1-ss5       joint       0.01%     0.010        +24.5     20/42          18/31
  m1-g8.9.10   joint       0.01%     0.041        +29.0     21/46          18/32
```

A binding rate of 0.011 means the local term wins in one window in ninety; the
threshold is global in all but name. **It leads by +28 and catches 18 of 31
headline-cell events.** Seventeen of the eighteen cells on the clean curve have a
positive median lead, +24.0 to +32.5.

**And the original evidence was confounded, which is the part worth keeping.**
`lstm-quantile` runs through `telemanom.top_columns`, which applies **no
`error_buffer` dilation and no pruning** -- verified, zero calls. Every
telemanom-path detector has each exceedance widened by +/-99 timesteps before
alarm ranges are formed, and lead time is measured from a range's start (D21). So
the -122 against `lstm-telemanom`'s +26 was **never a like-for-like comparison**:
one arm had up to 99 timesteps of dilation and the other had none. The gap
between them is not 148 timesteps of structure; an unmeasured part of it is a
post-processing asymmetry.

**What survives and what does not.**

* **Survives:** the measured numbers. `lstm-quantile` scored median lead -122 on
  its own terms, best cell -100 across 24 grid points, recall identical at 6/46
  throughout. Those are unchanged and the branch remains closed on them.
* **Does not survive:** *a global threshold fires late because it needs
  magnitude*. A global threshold fitted on genuinely nominal residuals, given the
  same alarm shaping as everything else, fires early. What made `lstm-quantile`
  late was its own configuration, not the class it belongs to.
* **Not the cause:** the calibration-pool contamination. The harness masks
  `train_scores[usable]` before `threshold_from` (`harness.py:157`), so
  `lstm-quantile`'s threshold was always fitted on genuinely nominal scores. This
  correction is independent of that bug.

**(!) SUPERSEDED BY MEASUREMENT, 2026-08-28. The conclusion does not stand
either.** Everything above was measured on **pre-fix weights** -- the one-epoch
models produced by the `min_delta` defect (D17). `lstm-quantile` had never been
scored on properly trained weights until now. Re-measured, on post-fix weights
and the honest lead-time reading:

```
  m1-g8.9.10   F0.5 0.838   recall 26/46   MVGS 21/32   rare-FA 2/48   lead  +0.0
  m1-ss5       F0.5 0.885   recall 27/42   MVGS 21/31   rare-FA 2/48   lead  +0.0
```

Against the branch's recorded 0.421, 6/46, 6/32 and **-122**. The recall that
"agreement cannot change when the detector fires 21 times in eleven million
timesteps" is now 26 of 46, and the lateness the branch was closed for is gone:
`lstm-quantile` neither batches nor dilates, so its honest reading equals its
reported one.

**The general obligation, which is the part worth carrying forward.** This branch
was closed in good faith on numbers that were correct for the weights they were
measured on, and those weights were later found to be defective. **A
disqualification made on buggy weights is not a disqualification.** Every closed
branch inherits the obligations of every later correctness fix, and nothing in
this repository re-opened D13 when D17 landed -- it took a separate investigation
stumbling into it. `docs/NARRATIVE.md` carries the lesson.

**CONSEQUENCE.** D13's conclusion stands and its reasoning is withdrawn. Nothing
built on the sentence *a global rule cannot warn early* may continue to rest on
it -- including D20, whose whole design was a local term added because a global
one was assumed structurally late. That assumption is now measured and false, and
D20's two-term form was solving a problem that may not exist.

Recorded rather than quietly amended because the sentence was quoted forward into
two later designs. **A structural claim is exactly the kind that gets reused
without being rechecked**, which is why one built on a single detector's
configuration is worse than no claim at all.

---

## D14. The weight-cache key must be named and versioned, not positional

**DATE** 2026-08-26 | **STATUS** open, implement after Layer 1 lands

**CONTEXT.** Adding telecommands to the model meant the weight cache had to
distinguish a fit that saw commands from one that did not -- otherwise a
detector would be served weights trained on inputs it does not have. An element
was appended to the key tuple, `None` when no commands were involved.

Launching the Layer 1 ablation, the control arm -- which takes no commands, and
whose new key element is therefore `None` -- **refitted all six folds anyway**.
An eight-element tuple hashes differently from the seven-element one that banked
the existing weights, `None` included. Every previously cached fit became
unreachable at once. About 45 minutes, and it will happen again: work items 5 and
6 add a GRU and a TCN, and both will want fields of their own.

**ALTERNATIVES.** Leave it positional and accept a full refit whenever the key
changes. Re-key immediately. Version the key.

**EVIDENCE.** Correctness was never at risk: fits are seeded and deterministic, so
the refits produced bit-identical weights. What was lost was only time -- but the
distinction that matters is sharper than that:

> **The content digest protects against serving *wrong* weights. It does not
> protect against throwing away *right* ones.**

Those are different guarantees and only the first was designed for. The store was
built so that a stale checkpoint *cannot be found*; nothing in it ensures a valid
checkpoint *remains* findable.

**The deeper cause is worth separating from the incident.** A `None` moved the
hash, which means **the key's shape is load-bearing, not just its values**. That
is a fragile property in a way varying values are not: a value changing is the
mechanism working as intended, while a shape changing invalidates entries for
callers the change does not even apply to. A positional tuple makes every future
field addition a silent, total cache invalidation, and gives no signal that it
happened -- the run simply takes longer.

**CONSEQUENCE.** The key becomes a **named, versioned structure** -- explicit
fields plus a schema version -- so that adding an optional field with a null value
leaves the hash unmoved, and a genuine incompatibility is declared by bumping the
version rather than discovered by a slow run. Deferred until Layer 1 lands, and
deliberately: re-keying now would orphan the twelve fits that run is banking and
make Layers 2 to 5 pay the same cost a second time. Done afterwards it costs one
refit.

Recorded because the failure is invisible by construction. Nothing errors, nothing
warns, and the only symptom is a run that takes longer than it should -- which is
easy to attribute to the machine.

**(!) 2026-08-28, work item 5.** The GRU's field was added the way this entry
prescribes and not the way the incident above happened: `Hyper.cell` is emitted
into the dict, the key and the fingerprint **only when it is not the default**,
so the twelve banked LSTM fits and every published fingerprint are unmoved --
pinned by `tests/test_lstm_detector.py` against the literal pre-change tuple,
digest and fingerprints (D26). The full named, versioned key this entry calls
for is still outstanding; the status stays open.

---

## D15. Fitting is portable because the arithmetic is asserted, not because the environments match

**DATE** 2026-08-26 | **STATUS** resolved

**CONTEXT.** Fitting is the only expensive part of a run and the only part a GPU
helps with: scoring goes through `sentinel_models.reference`, the plain-NumPy
forward pass Phase 2's C++ is transcribed from, which no GPU touches. So the work
splits -- fit on a rented GPU, score on the Mac -- and weights travel between two
machines that agree on nothing.

**ALTERNATIVES.** Fit and score in the same place. Pin an identical environment on
both. Containerise.

**EVIDENCE.** The fitting box ran Ubuntu 24.04, Python 3.12.3, torch 2.6.0+cu124
on an NVIDIA A40-8Q. The scoring box runs macOS, Python 3.14.6, torch 2.13.0 on an
Apple M5. **Nothing matches.** `tests/test_reference_equivalence.py` passed on
both, holding the torch model and the NumPy reference to 1e-5.

Measured on the same fold, 35 epochs:

```
  M5, 4 threads          18.9 s/epoch    11.0 min
  A40, fused cuDNN        0.9 s/epoch     0.5 min     21x
  A40, deterministic      1.0 s/epoch     0.6 min     19x
```

Twelve fits in **15.9 minutes** against ~5.5 hours locally. And determinism was
checked as a property rather than trusted as a flag: two fits, same seed, same
box, **bit-identical weights on both paths**, so the deterministic path costs 11%
and no publication rule needed amending.

**CONSEQUENCE.** A weight file is certified by an **equivalence assertion, not an
environment lockfile**. That is what makes fitting portable at all, and work items
5 and 6 inherit it for the GRU and the TCN without re-deriving the argument.

Two caveats travel with it. Bit-identical means **on the same box**: a GPU and a
laptop produce different weights from the same seed because reduction order
follows the hardware, so "reproduced from cold" means on equivalent fitting
hardware and provenance records the device. And nothing persists on a rented box
-- data streams from R2, weights come back, the machine is destroyed.

---

## D16. A cache that fails safe still has to be verified

**DATE** 2026-08-26 | **STATUS** resolved

**CONTEXT.** `Weights` gained an `n_exogenous` field when telecommands became
model inputs. `_save_weights` was never updated to write it.

**EVIDENCE.** Every commanded model failed to reconstruct on load. The failure was
**invisible**, by design: `_load_weights` catches any exception and returns a
cache miss, so the consequence was not a wrong answer but a silent refit --
correct numbers, hours of wasted fitting, and no signal that anything was wrong.

It was caught by the verification step before a pod was released, which is the
only reason it did not cost a second GPU session. An hour of fitting had already
been thrown away by then.

**CONSEQUENCE.**

> **Failing safe is not working.** A cache that degrades to a silent recomputation
> converts a defect into a cost, and a cost with no signal attached is
> indistinguishable from the machine being slow -- which is exactly how the
> positional-key defect in D14 hid, and how this one did.

`n_exogenous` is written, and inferred exactly from the weight shapes for files
that predate the field, so nothing already banked was orphaned. A test asserts a
commanded model survives the round trip.

**And the general rule, which is what the pod procedure exists to enforce:
"downloaded" is not "verified".** Every file is loaded through the production
loader and one is scored end to end before a machine that cannot be recovered is
released. This defect is the argument for that step; without it the twelve fits
would have been declared safe and the Mac would have quietly refitted all of them.

---

## D17. Dimensionless constants transfer across scales, but not across distribution shapes

**DATE** 2026-08-26 | **RECORDED** 2026-08-27, two commits late -- see the note
at the end | **STATUS** resolved, with the stated mechanism **measured and refuted**
-- see MEASURED below

**(!) READ THE STATUS.** The mechanism below is the reasoning that was acted on
when `z` was moved onto `Config`. It was **not** a measured finding when this
entry was written, and it has since been measured. **The conclusion survives and
the mechanism does not**: the residual did not become near-Gaussian, it became
very much more heavy-tailed, and a dimensionless constant failed to transfer for
the opposite of the stated reason. The original wording is kept below exactly as
it was acted on; MEASURED at the end is what the data says.

**CONTEXT.** Two published constants failed in the same week, in the same way,
and the second failure showed the first lesson had been drawn too narrowly.

The first was `min_delta`. telemanom applies early stopping as
`current < best_loss - min_delta` with a published `min_delta` of 3e-4.
Validation MSE on ESA-ADB is about 1e-4, so after the first epoch the bar became
`current < 1.7e-4 - 3e-4 = -1.3e-4`. That is negative, and no mean-squared error
can be negative, so no epoch after the first ever registered as an improvement.
`best_epoch` stayed 0, `restore_best` restored epoch 1, and patience fired at
epoch 11 -- for every fit in this project, across three work items. Median 3.1x
better weights discarded, worst case 8.8x.

The rule drawn from it was: **absolute constants in data units do not transfer;
dimensionless ones do.** The replacement is a fraction of the standing best
rather than a smaller constant, because picking another absolute number by eye
reproduces the failure with different digits.

The second was `z`. telemanom's published sweep is `np.arange(2.5, 12, 0.5)`,
and it was audited under that rule and passed: `z` counts standard deviations of
the smoothed error, so it is dimensionless and immune to rescaling. Fixing
`min_delta` then improved the forecast about fortyfold, and that same audited,
scale-free range produced **3,548 alarm ranges where it had produced 182**.

**ALTERNATIVES.** Replace the rule -- treat the audit as worthless. Keep the
rule and call `z` a special case. Pick a new `z` by eye. Amend the rule.

**EVIDENCE.** Detection improved on every other axis at the same time, which is
what rules out a worse model as the cause: 38 of 46 events caught against 37,
headline cell held at 28/32, lead time unchanged at +26 with fewer landing late.
Only precision collapsed. A worse model catches fewer events; this one catches
more, at the same lead time, firing twenty times as often.

**(!) Two figures the original reasoning did not include, added on 2026-08-27
because they are read from the same artifacts and they qualify it.** The
adoption number moved the wrong way too -- rare-event false alarms 22/48 to
30/48 on `m1-g8.9.10` and 17/48 to 33/48 on `m1-ss5` -- and event-wise F0.5 fell
from 0.269 to 0.026 and from 0.664 to 0.035. "Only precision collapsed" is true
and reads as narrower than it is.

**CONSEQUENCE.** The audit's rule is amended rather than replaced. It was right
about units and incomplete about distributions:

> **A dimensionless constant is immune to rescaling. It is not immune to a
> change in the shape of the thing it indexes.** With a poor forecast the
> residual is dominated by model bias and 2.5 sigma is a real excursion; with a
> good one it is dominated by irreducible noise and 2.5 sigma sits on the floor.

`z` therefore moves from a module constant onto `Config`, where it can be fitted
per model, with the silence fallback tied to the sweep's ceiling so that raising
the range raises the fallback with it. **The floor is fitted, not transcribed.**

If the mechanism holds it is the failure mode that returns every time the model
improves, so work items 5 and 6 will meet it with the GRU and the TCN, and it is
a property of the method rather than of this fit.

**The Phase 2 consequence, recorded against Objective.md decision 2.** Storing
thresholds separately from weights began as an argument from flexibility; it is
now a measured requirement. A threshold that suits one model does not suit a
better one, so it is a fitted quantity belonging to the model it was measured
against. A mission uplinking an improved `model.bin` must uplink its thresholds
with it, and must be able to recalibrate them without retraining.

**NOTE -- how this entry came to be written late, which is itself the record.**
`docs/HARNESS.md` is unconditional that this file is updated **in the same
commit as the decision it records**. It was not. Commit `1aeff2d` made the
`min_delta` decision and did not touch this file; commit `0648fe7` made the `z`
decision, stated in its own message that "Corrected form recorded in
DECISIONS.md", and did not touch this file either. In between, five places
began citing an entry that did not exist -- `Objective.md` section 14.10,
`src/sentinel_models/telemanom.py`, `src/sentinel_models/lstm.py`,
`tests/test_early_stopping.py` and `scripts/threshold_sweep.py` -- and the
handover brief for the next work item cited it as well.

Nothing errored. The rule that was supposed to catch this is a convention, and a
convention has no test. Recorded plainly rather than backfilled silently,
because a commit message asserting that a document was updated when it was not
is a worse defect than the missing document: it is a claim in the audit trail
that the audit trail does not support.

---

**MEASURED, 2026-08-27.** `scripts/threshold_diagnostics.py`, both channel sets,
all three folds, both generations of weights, **5,684,580 reference windows**,
every one of them verified byte-identical to the live
`telemanom.dynamic_threshold`. Artifact
`runs/m1-g8.9.10/_threshold/2026-08-27T*-diagnostics.json`. No labelled anomaly
informs any figure below; alarms are counted, never scored.

**The stated mechanism is wrong.** It said that with a good forecast the residual
is "dominated by irreducible noise" -- near-Gaussian, nothing left for an
outlier-finder to find. Measured, the residual moved the other way. Excess
kurtosis of the signed residual, pre-fix to post-fix:

```
  m1-g8.9.10   fold 0     63.6 ->     71.7        m1-ss5   fold 0    143.9 ->    159.1
               fold 1    145.2 ->    176.3                 fold 1    117.0 ->    153.2
               fold 2     27.3 ->  6,754.6                 fold 2    148.1 ->  7,348.8
```

A trained forecaster predicts the bulk almost perfectly and leaves a small number
of large excursions it cannot predict. That is **more** tailed, not less.

**The conclusion survives, by a different route, and it is the local moments that
matter rather than the global distribution.** `eps = mu + z*sigma` is computed
per 2,170-sample window, so what governs it is the within-window scale, and that
is what collapsed:

```
                     within-window sigma       fraction of windows with
                     (median) pre -> post      (max-mu)/sigma >= 2.5
  m1-g8.9.10 fold 0  6.95e-4 -> 1.19e-3 (1.7x)   0.224 -> 0.083     <- control
             fold 1  6.59e-4 -> 2.11e-4 (0.32x)  0.216 -> 0.729
             fold 2  9.27e-4 -> 2.06e-4 (0.22x)  0.073 -> 0.803
  m1-ss5     fold 0  1.66e-3 -> 6.10e-4 (0.37x)  0.010 -> 0.252
             fold 1  1.34e-3 -> 1.76e-4 (0.13x)  0.056 -> 0.921
             fold 2  1.07e-3 -> 1.73e-4 (0.16x)  0.143 -> 0.953
```

The chain, end to end: the local scale falls three- to eightfold; the window
*maximum* does not fall with it, because the tail got heavier; so
`(max - mu)/sigma` **rises**, from a median of 2.04 to 2.94 on `m1-g8.9.10` fold
2 and 2.24 to 3.28 on `m1-ss5` fold 2; so a floor of 2.5 that used to be out of
reach in 93% of windows is now cleared in 80% of them; and every window that
clears it contributes an alarm at least 199 timesteps wide, because
`error_buffer = 100` dilates a single exceeded sample by +/-99. Fifty alarm
ranges become 1,505.

**Fold 0 is the control and it behaves as the mechanism predicts.** Its forecast
improved 1.6x rather than 40x, its within-window sigma went *up*, its reachable
fraction went *down* -- 0.224 to 0.083 -- and its alarm count barely moved, 42 to
73. The two arms are in the same run.

**So a dimensionless constant is not immune to a change in distribution shape,
which is what this entry claimed, and the reason is not the one it gave.** `z`
counts standard deviations, and a standard deviation is a poor summary of a
distribution whose kurtosis is in the thousands. What transfers across scales
does not transfer across a change in the relationship between a distribution's
scale and its extremes. That is the corrected form.

**A competing explanation was tested and refuted, and it is recorded because it
was the leading candidate.** `dynamic_threshold` omits two admissibility
conditions published telemanom applies -- `len(E_seq) <= 5` and
`len(i_anom) < len(e_s) * 0.5` (`docs/MODELS.md` deviation 8). The second is a
50% coverage cap and looked like exactly the guard against a selector flagging
huge swathes. **Measured, neither condition binds on a single one of the
5,684,580 windows**, and the alarm count under published telemanom's rule is
identical in every fold of both channel sets. The reason is upstream: EWMA at
span 105 makes exceedances contiguous, so they merge into one or two sequences
covering a few hundred of 2,170 samples, far under both limits. The omission is
a real defect in the reproduction and it is **not** the cause of this. Reported
before any change was proposed, which is why it is a finding rather than a fix
that would have moved every published number for nothing.

---

## D18. The selection criterion is degenerate on these residuals, and no threshold is chosen

**DATE** 2026-08-27 | **STATUS** OPEN -- reported, deliberately unresolved

**CONTEXT.** Work item 4's threshold investigation was asked for three
diagnostics and told to report them together **before** proposing any change,
and not to pick a threshold by looking at scored results. It found the criterion
is not selecting anything. The obvious next move is therefore to choose a rule,
and this entry exists to record that the choice was not made and why.

**EVIDENCE.** `scripts/threshold_diagnostics.py`, 5,684,580 reference windows,
both channel sets, all three folds, both generations of weights, every window
verified byte-identical to the live `telemanom.dynamic_threshold`. Alarms
counted, never scored; no labelled anomaly informs any of it.

```
  windows examined                                    5,684,580
  windows that selected a z                           2,081,285   (36.6%)
    of those, z = 2.5, the range minimum              1,927,583   (92.6%)
  windows with more than one candidate -- a choice     1,049,475   (50.4% of firing)
    of those, z = 2.5, the range minimum                895,773   (85.4%)
```

The criterion is **monotone decreasing in z** in the large majority of sampled
windows, so its argmax is the boundary of the candidate range by construction.
Its median normalised value falls 1.000, 0.843, 0.676, 0.504, 0.350 across
z = 2.5, 3.0, 3.5, 4.0, 4.5. There is a gradient -- the best candidate beats the
next by 20-45% -- and it points monotonically downhill, at the floor.

> **On these residuals the nonparametric dynamic threshold reduces to
> `eps = mu + 2.5*sigma`.** A fixed multiplier on the local scale, with the
> selection decorative and `z_floor` doing all the work. That is what made the
> `z_floor` sweep look so effective, and it is why the sweep was measuring the
> wrong thing: raising the floor raises the only quantity that was ever live.

**Why it is monotone, decomposed.** Medians over sampled windows that had more
than one candidate, `m1-g8.9.10` post-fix fold 2:

```
      z       n  n_above  n_seq  covered   denominator   numerator      score
    2.5   3,330       53      1      289           290      0.1517   4.54e-04
    3.0   3,330       30      1      237           238      0.1021   3.99e-04
    3.5   2,664       21      1      220           221      0.0863   3.82e-04
    4.0   1,998       15      1      213           214      0.0742   3.48e-04
    4.5   1,332       10      1      207           208      0.0576   2.83e-04
```

**The denominator is a constant in disguise.** `len(E_seq)` is **1** at every
candidate, so the squared anti-fragmentation term contributes 1 and is inert.
`covered` falls only from 289 to 207 across the range and asymptotes at 199,
because `error_buffer = 100` dilates a single exceeded sample by +/-99 and
merging collapses the rest. The numerator meanwhile falls 2.6-fold, from 0.152
to 0.058, because removing fewer points changes the moments less.

> A ratio whose denominator cannot move and whose numerator falls with z has its
> maximum at the smallest z on offer. **The criterion is arithmetically forced to
> prefer the bottom of its range**, and no residual distribution can rescue it.
> `error_buffer` is a parameter of the smoothing stage, not of the threshold, and
> it is what disables the selection.

Two things follow. The `|E_seq|^2` term -- the part telemanom's paper describes
as stopping the sweep buying a statistical improvement with a shower of
fragmented detections -- **never activates here**, because the same buffer that
pins the denominator also merges every exceedance into one sequence. And the
apparent uptick above z = 5.5 in the full table is a selection effect, not a
peak: only windows with a large `reach` survive into those columns and there are
113 to 292 of them against 3,330, so the population differs column to column.

**ALTERNATIVES considered and refused.**

*Adopt `z_floor = 8.0` from the sweep.* Its cell is the best in the grid --
F0.5 0.794, precision 22/22, rare-event false alarms 0/48, median lead +35.5
with no late detections. It is also chosen by reading a table of scores against
46 labelled anomalies. A mission has no failures to fit to (Objective.md 6.1),
so this is a number no adopter could ever obtain, and `docs/MODELS.md` section 7
already refused the same move once under the name "oracle threshold sweep".

*Replace the criterion.* Premature. It is degenerate here for a reason that is
now measured, and the reason implicates stages upstream of it -- see below. A
replacement chosen before those are examined would be fitted to a defect one
layer down.

*Scale `z` by a robust dispersion rather than the standard deviation.* The most
promising direction and still a proposal, not a decision. It is derivable from
residual properties alone, which is the test the brief sets: the failure is that
`sigma` collapses in quiet windows while the extremes do not, and a median
absolute deviation or an interquartile range has the same problem in a different
proportion. It needs measuring before it is adopted, on the same windows, and
that is the next piece of work rather than this one.

**CONSEQUENCE. No threshold value is chosen, and the three stages that the
measurement implicates are flagged rather than retuned.** `docs/HARNESS.md` is
explicit that a stage needing revisiting is flagged, not adjusted inside a run
measuring something else.

| Stage | Why the measurement implicates it | Status |
|---|---|---|
| `error_buffer = 100` | One exceeded sample becomes a 199-timestep alarm. It sets the floor on what a single crossing costs, and it is also why published telemanom's 50% coverage guard cannot bind here | **Flagged, untouched** |
| EWMA span 105 | Makes exceedances contiguous, which is what collapses `len(E_seq)` and `covered` and renders both published guards inert. Inherited, never examined | **Flagged, untouched** |
| `z_floor = 2.5` | Selected in 92.6% of firing windows. It is not a floor on a range, it is the operating threshold | **Flagged, untouched** |
| k-of-n agreement | Tuned when there were 182 alarm ranges. There are 3,548 | **Flagged, stale, untouched** |
| pruning `p = 0.13` | Reads the same residual ladder the threshold does. Not measured here | **Flagged, unmeasured** |

**And one thing that is settled.** The competing hypothesis -- that our omission
of published telemanom's two admissibility conditions caused this -- is refuted:
neither binds on any of the 5,684,580 windows and the alarm count under
published telemanom's rule is identical everywhere. The omission stays in the
deviation ledger as a real defect (`docs/MODELS.md` deviation 8) and is not the
cause. Restoring it would move no number, which is a good reason to restore it
for faithfulness and no reason at all to expect it to help.

---

## D19. A run has one provenance, not one per record

**DATE** 2026-08-27 | **STATUS** resolved

**CONTEXT.** `RunRecord.git_commit` was `field(default_factory=git_commit)` over
a plain function, so it re-ran for every record a run constructed -- one per
channel set. Found while reading the pre-fix artifacts to build
`docs/RESULTS.md` section 6a.

**ALTERNATIVES.** Leave it and note the discrepancy in the document. Stamp the
commit in `cli.py` and pass it down. Cache the resolution for the process.

**EVIDENCE.** `runs/m1-g8.9.10/lstm-telemanom/2026-08-26T212610Z-1f8b6fd6.json`
stamps `1bf6710` on the `m1-g8.9.10` record and `5bab55e` on the `m1-ss5` record.
One invocation, one bundle, one set of weights, two answers to *which code
produced this*. The run spanned a commit because a paired run takes long enough
to.

**CONSEQUENCE.** `git_commit` resolves once per process (`lru_cache(maxsize=1)`).
A correctness fix, escalated and authorised before being made, per
`docs/HARNESS.md`.

**No number moves, and that is worth stating rather than assuming.** The field
is provenance, not a measurement: no metric reads it, no scorecard derives from
it, and the obligation to record both numbers when a fix moves a published one
therefore has nothing to record. What was broken is the ability to trust a
stamp, which is the only thing a stamp is for.

Existing artifacts are **not rewritten**. The pre-fix artifact keeps its two
commits and `docs/RESULTS.md` section 6a says so, because an artifact edited
after the fact is worth less than one that carries its own defect visibly. The
artifact path is the unambiguous identifier and is what the documents cite.

`tests/test_run_provenance.py` reproduces the defect directly -- HEAD moving
between the two records of a pair -- rather than asserting the cache exists,
because a test that pins the mechanism instead of the behaviour passes for the
wrong reasons later.

---

## D20. A local scale under a global floor: neither threshold is safe alone

**DATE** 2026-08-27 | **STATUS** open, pre-registered and not yet run

**CONTEXT.** Two thresholding rules have been measured on identical weights, and
each fails in the opposite direction. That symmetry is the design.

**THE TWO FAILURES, and they are the same fact seen twice.**

| | responds to | fails by | measured |
|---|---|---|---|
| Nonparametric dynamic threshold | the **local** recent error scale | the local scale collapsing onto a noise floor when the forecast improves | 50 alarm ranges to 1,505 (D17, D18) |
| Global quantile | the **99.9th percentile of years of training scores** | needing absolute magnitude, so it fires only after divergence has grown | median lead **-122**, all six catches late (D13) |

D13 closed the quantile branch on structure and the wording was careful:
*a rising error crosses a local threshold as soon as it departs from the recent
norm; to cross a global one it must grow to an absolute size, and growing takes
time.* That is exactly what makes a global rule unusable **as a threshold** and
exactly what makes it sound **as a floor**. The local term still triggers at
onset whenever the local scale is healthy; the floor only decides what happens
when it is not.

> **Local for timeliness, global for the fact that the local scale can vanish.**
> The 26-timestep head start is bought by locality and this keeps it. What the
> floor removes is the regime in which locality is meaningless -- a window whose
> entire content is smaller than anything the model was ever wrong by on nominal
> data.

**ALTERNATIVES.** Keep `mu + z*sigma` and fit `z` honestly per model. Replace
the criterion but keep the family. Adopt a global quantile. Adopt a local order
statistic alone.

**EVIDENCE.** Fitting `z` honestly is refused on measurement, not on taste:
`sigma` fell three- to eightfold between weight generations while the window
maximum did not, at excess kurtosis reaching 6,754, so **every** member of the
`mu + k*sigma` family inherits the defect whatever `k` is and however it was
chosen. An order statistic is unchanged by tail weight, which is what removes
that. A local order statistic **alone** is refused for the reason stated in the
proposal's own objection below.

**CONSEQUENCE.** The proposed form, per segment, from its trailing reference
window `R` and the nominal residual pool `N` of the fitting window under the same
weights:

```
  eps = max( alpha * Q_p(R) ,        local  -- onset sensitivity
             beta  * Q_p(N) )        floor  -- bounds the collapse
```

`mu` and `sigma` appear nowhere. `alpha` and `beta` are fitted on nominal
residuals against a **measured noise floor** -- never against a detection score.

**(!) The wording here was "an alarm budget in timesteps -- what a mission's
operators can act on", and it is struck** (`docs/HARNESS.md`, and Objective.md
10.2 fix 3a). There is no alarm budget. The detector fires when the relationship
breaks and is silent otherwise, and the count belongs to the spacecraft rather
than to us. What is calibrated is the line between sensor hum and a real break,
read from nominal residuals. The distinction the entry was reaching for was
label-free versus label-fitted, and that half stands; the operator-capacity
justification does not. `p` is bounded below the measured contamination rate: the median
window carries 53 exceedances in 2,170 samples, 2.4%, so a rank near 0.75 sits
far under any plausible breakdown point, and that bound is read from residuals
rather than transcribed.

**(!) THE OBJECTION TO THIS DESIGN, RECORDED BEFORE IT IS TESTED AND NOT AS A
CAVEAT AFTERWARDS.** An order statistic is immune to tail **weight**, not to
scale **collapse**. `Q_p` of a uniformly tiny window is tiny. So the honest claim
is narrower than *order statistics fix it*:

> **What fixes it is calibrating the multiplier against nominal residuals instead
> of transcribing a constant.** The order statistic is what makes that
> calibration robust to a tail that moves; the floor is what carries the rest.

If the floor does all the work and the local term never binds, this has collapsed
back to `lstm-quantile` and **is reported as such** rather than defended. That
condition is written into the pre-registration with a number attached, so the
result is read against an expectation rather than a hope.

**What this is not.** Not a replacement for `lstm-telemanom`, which stays as the
published-comparison baseline the Phase 1 gate is written against. An addition, in
`sentinel_models` only, so the harness never learns about it and the layering
thesis holds. `error_buffer` is **outside** it -- D21.

---

**OUTCOME, 2026-08-27. Refuted as run; retested as intended; still open.**
`docs/MODELS.md` 10.7 and 10.8.6 carry the numbers.

*As run*, the two terms were calibrated independently and combined with `max()`,
which admits far less than either term alone -- **0.00066 of a 0.001 budget**.
The floor dominated, and `lstm-oscfar` reproduced `lstm-quantile` exactly at a
median lead of **-196.5**. That is my arithmetic, not the design.

*As intended*, with both terms fitted so the maximum admits the target, the rule
**gets worse**: at a 1% budget on `m1-g8.9.10` the correct fit spends **2,405
alarm ranges** where the broken one spent **126**, for 40/46 recall against
37/46. **The under-admission had been acting as an unintended out-of-sample
margin**, and removing the error removed the margin. That is a real finding about
the two-term form and it is not favourable to it.

*And the floor turns out to be the part that was wrong.* Run bare, the local
order statistic alone reaches 21/46 recall, **19/32 headline cell**, **85 alarm
ranges** and **+26.0** median lead at a 0.1% budget -- better than
`lstm-telemanom` on every one of those axes. The floor was not load-bearing; it
was the part that broke the design.

**D20 therefore does not close.** The two-term form it proposes is measured and
unfavourable, and what replaces it -- a local order statistic with no floor -- is
a *different* design that needs its own decision, its own pre-registration, and
`k`-of-`n` re-derived rather than inherited (**D23**). Nothing is decided until
that proposal exists.

---

## D21. `error_buffer` reaches into two stages it does not belong to

**DATE** 2026-08-27 | **STATUS** OPEN -- logged now so it is not lost, decided
after the threshold is frozen

**CONTEXT.** `error_buffer = 100` is a parameter of telemanom's post-smoothing
step: it dilates every exceeded timestep by +/-99 and merges the results. It has
now been measured reaching into two stages that are not its own, and one of those
reaches into a **headline number**.

**FIRST REACH -- into the threshold's selection criterion. Now moot, and
recorded because of how it was found.** `covered` is the length of the buffered,
merged sequences, and it is the criterion's denominator. Measured, `len(E_seq)`
is 1 at every candidate and `covered` falls only from 289 to 207 across the whole
range, asymptoting at 199 because one exceeded sample already costs it. **The
denominator cannot move**, the numerator decays with `z`, and the criterion is
arithmetically forced to the bottom of its range (D18).

D20 dissolves this rather than fixing it: an order-statistic threshold has no
selection criterion, so there is no denominator to pin. **That is why
`error_buffer` is outside D20's proposal** -- the coupling disappears as a side
effect of removing the stage that reached, and changing both at once would
confound them.

**SECOND REACH -- into lead time, and this one is live.**

> **(!) A +/-99 dilation moves an alarm range's start up to 99 timesteps earlier
> than the exceedance that caused it. Lead time is measured as
> `event_start - start of the earliest alarm range overlapping the event`. So up
> to 99 timesteps of every lead-time figure this project reports may be the
> buffer rather than the detector.**

The reported median is **+26** (`docs/RESULTS.md` section 3), and 99 is nearly
four times it. This is not a claim that the lead time is an artifact -- it is
**not measured**, and it might be that the exceedances genuinely precede the
event and the buffer contributes nothing. It is a claim that a headline number
has an unexamined term of a size that could dominate it, and that this was
noticed while measuring something else.

It matters more than an ordinary open item because lead time is not a supporting
figure here. It is a **gate metric with a disqualifying rule** (D9): a negative
median disqualifies a configuration whatever its F0.5, and that rule has already
disqualified `rstd`, `mavg` and `lstm-quantile`. A metric that decides
eligibility should not have an unmeasured additive term.

**How to measure it, and it is cheap.** Lead time recomputed against the
**undilated** exceedance rather than the buffered range, on the same cached
weights and the same bundle load. The difference is the buffer's contribution,
per detector, per fold, per channel set. It changes no published number -- it
adds a second reading of one -- and both are then reported side by side.

**MEASURED 2026-08-28, and the second reach is worse than the caveat allowed
for.** `scripts/event_forensics.py`, per caught event, both channel sets, cached
weights. Lead time taken twice: from the dilated alarm range's start, as this
project reports it, and from the **first actual threshold crossing inside that
range**.

```
  m1-g8.9.10  n=38        median    p25     p75    positive
    as reported (dilated)  +26.0  +15.2   +46.8      34/38
    from the crossing       +0.0   -6.0    +0.0       3/38
    inflation              +27.5    +16     +40   max  +63

  m1-ss5      n=37
    as reported (dilated)  +27.0  +16.0   +47.0      35/37
    from the crossing       +0.0   -3.0    +0.0       1/37
    inflation              +32.0    +17     +47   max  +63
```

> **The reported +26 is the buffer. Measured from the moment the detector
> actually crosses its threshold, the median lead is ZERO, and the detector warns
> in advance in 3 of 38 events rather than 34 of 38.**

In **31 of 38** cases on the gate set and **34 of 37** on the subset, the entire
positive lead is the dilation: the crossing is at or after the event start and
only the backward widening puts the range's start before it.

**Why this is a correctness question and not a presentational one.** The
dilation widens a range *backwards from a crossing that has already happened*. A
flight component emits when it detects, not retroactively, so no operator
receives anything at the dilated start. Crediting lead time from there credits
warning that was never given.

**(!) The mechanism, corrected 2026-08-28 after implementing the fix.** This
entry first said the dilation reaches back `+/-99` and therefore that up to 99
timesteps of the figure could be buffer. **The backward reach is bounded by the
stride, not by `error_buffer`**: `channel_ratios` clips each dilated sequence to
the judged segment (`lo = max(lo, offset)`), so a range cannot start before its
own 70-step batch does. The measured maximum inflation is **+63**, inside one
batch, which is consistent and was the tell.

That makes the quantity being credited **the batching latency itself**: a range is
dated from the start of the batch in which a crossing occurred, while the
detector can only speak at that batch's end. The median inflation of +27.5 is
about a third of a batch, which is what you would expect from crossings landing
anywhere within one. The measured numbers are unchanged; only the explanation
was wrong, and it was wrong in a way that overstated the mechanism while
understating how ordinary it is.

**What it invalidates.** Lead time is a gate metric with a disqualifying rule
(D9), and it has been compared across detectors that do not all receive the
dilation. `lstm-quantile` and the trivial baselines run through paths with **no
`error_buffer`** (D13's revision), so `+26` against `mavg`'s `0` and
`lstm-quantile`'s `-122` was never like-for-like -- one side carried a median
`+27.5` the other did not. Every lead-time figure in `docs/RESULTS.md` inherits
this, including the ones that disqualified `rstd`, `mavg` and `lstm-quantile`.

**RESOLVED 2026-08-28 for the second reach.** `sentinel_eval` gained an
optional emission point, additive and absent unless a detector provides one, and
the honest reading is in `docs/RESULTS.md` 6f. Medians of **-43.0** and **-41.5**
on the gate set against +26.0 and +29.0, with 22 of 23 and 18 of 18 detections
negative -- **and fifteen of thirty-eight events on the gate set have no emission
overlapping them at all**, caught by the backward widening rather than by the
detector. No detection count moved; only the reading.

**CONSEQUENCE.** `error_buffer` is still held at 100 and its first reach -- into
the alarm width itself -- is still undecided, so this entry stays open on that.
What is settled is the metric: every detector is now measured from where it could
speak, those that do not dilate falling through unchanged, so the comparison is
finally like-for-like. **D9's disqualifications were made on the old reading and
should be revisited on the honest one before the architecture gate.** **Also carried into work items 5 and 6**: `error_buffer` is why the
persistence filter measured as subsumed on this detector (Objective.md 7.1), so
the GRU and the TCN inherit all three couplings unexamined unless this is settled
first.

---

## D22. A pre-registered falsification condition that would have passed the failure it was written for

**DATE** 2026-08-27 | **STATUS** resolved

**CONTEXT.** `docs/MODELS.md` 10.3 pre-registered the condition that would
falsify the two-term threshold: *if the local term binds in fewer than 10% of
segments, the floor is doing all the work and this has collapsed back to
`lstm-quantile`.* It carried a number deliberately, so the result would be read
against an expectation rather than a hope. That was the right instinct applied to
the wrong quantity.

**EVIDENCE.** The `independent @ 0.1%` cell **binds at 0.127**, above the
threshold. It is also the cell that reproduces `lstm-quantile` exactly -- recall
6/46, headline cell 6/32, point 0/11, all three identical to the branch closed in
D13, at a median lead of -196.5.

> **The condition passes the configuration it was written to catch.**

**ALTERNATIVES.** Lower the threshold. Drop mechanism conditions and rely on
outcomes. Keep both and require both.

**CONSEQUENCE.** The defect is not the number 0.10, and lowering it would be
fitting the condition to the result it failed to catch. The defect is that
**binding rate measures whether the local term wins somewhere, not whether it
wins where it matters.** A rule winning 13% of segments scattered through quiet
history is a global rule with decoration, and the statistic cannot tell the two
apart.

The general form, which is what makes this worth an entry rather than a
correction:

> **A condition on the mechanism is only worth pre-registering if it is *harder*
> to satisfy than the outcome it stands in for.** This one was easier, so it
> could only ever have added false reassurance. What actually falsified the
> design was the outcome -- recall, headline cell and point recall identical to a
> closed branch.

Mechanism conditions are still worth writing, because an outcome condition tells
you *that* something failed and not *why*. But they are recorded **alongside** an
outcome condition and never instead of one, and where they disagree the outcome
wins. Applied to the next proposal: the falsifier is *reproduces `lstm-quantile`
on recall, headline cell and point recall*, with the binding rate reported as a
diagnostic that explains rather than decides.

This is the most consequential methodological finding of the run, and it is
recorded because it is the kind that is invisible when the design succeeds. Had
the corrected calibration worked, the condition would have been cited as evidence
it survived, and nobody would have checked what it measured.

---

## D23. The decision layer is channel-blind; the cross-channel claim rests on the forecaster alone

**DATE** 2026-08-27 | **STATUS** OPEN -- structural, resolve before the architecture gate

**CONTEXT.** Objective.md 2.4 defines the target class as one where every channel
is individually legal while the combination is wrong, and `docs/RESULTS.md`
reports 28 of 32 headline-cell events caught against `rstd`'s 3. Where in the
pipeline that claim is actually expressed had never been written down.

**EVIDENCE.** Every stage between the forecast and the alarm, inspected:
smoothing, thresholding, sequence-finding, pruning and persistence all take **one
channel at a time**. `top_ratios` / `top_columns` -- the `k`-of-`n` reduction --
is the only stage that takes the `(T, C)` matrix and combines across it. Full
table in `docs/MODELS.md` section 11.

And what it tests is **co-occurrence, not relationship**: `top[k-1] >= 1` means
`k` channels are simultaneously over their own individual thresholds. That is not
the claim *the relationship between them broke*.

**CONSEQUENCE.**

> **The forecaster is multivariate. The decision layer is twelve univariate
> detectors and a vote.** The cross-channel claim rests entirely on the
> forecaster, and every headline-cell number is the forecaster's, filtered by a
> rule that cannot see what the forecaster learned.

Two things follow and neither is decided here. **The false-alarm problem lives in
that gap**: a commanded manoeuvre makes several channels individually surprising
at once, which is precisely what `k`-of-`n` rewards, and it is nominal -- so the
one stage that recovers cross-channel structure recovers the wrong kind. And
`k`-of-`n` was tuned against 182 alarm ranges when there are now 3,548 (D18), so
the only structure-aware stage is stale by a factor of twenty.

**Any threshold proposal that goes forward re-derives `k`-of-`n` rather than
inheriting it**, and reports the trade explicitly on the new residuals -- alarm
count, headline-cell recall and lead time, per fold, both channel sets. If
agreement recovers precision without costing headline-cell recall, that is a
better answer than any threshold, because it expresses the thesis instead of
working around it.

Recorded as open because the alternative -- a decision stage that tests
relationships rather than magnitudes -- is a design question and not a tuning
knob, and it should not be answered inside a run measuring something else. Scoped
in `docs/MODELS.md` section 12; stated in Objective.md 4.4, because a document
that describes a capability the code does not have is worse than one that admits
the gap.

**Two things the scoping established that change the shape of this entry.**

**The information is discarded, not absent.** A relationship *is* visible in the
residual vector -- it is a direction in `R^C` that nominal data does not visit --
and what destroys it is `np.abs()` at `detectors.py:410` taking the sign off,
then twelve marginal thresholds that cannot express a covariance. Both channels
rising together and one rising while the other falls become the same number, and
that difference is the whole of the relationship. So the cheapest shape -- a
nominal residual covariance, **144 floats at C=12** against 91,640 model
parameters -- is a fixed matrix multiply with no state, and it down-weights
exactly the commanded manoeuvres that `k`-of-`n` rewards.

**And the explanation layer does not exist.** Objective.md 7 has promised "a
learned map of which channel pairs move together, including time-lagged" since
day one, and 4.2 shows its output as a named *pair*. What is implemented is
`last_attribution`, the index of the largest-error channel, and it is not written
to any artifact. **Naming a channel is not naming a relationship**, and
Objective.md 11 rule 4 -- every warning explainable -- rests on that difference.
It was assumed built because the objective says so, which is the same failure as
D17's missing entry in a different register.

---

## D24. The decision layer freezes as `lstm-whitened`, and the eleven are the price

**DATE** 2026-08-28 | **STATUS** resolved -- frozen for the architecture gate

**CONTEXT.** Four hypotheses for the events `lstm-whitened` loses were measured
and refuted, and a fifth -- a locally-referenced threshold -- was pre-registered
and run. It did not recover them. The pre-registered rule for that outcome was
that the rule as it stands freezes and the loss is recorded as measured, which is
what this entry does.

**EVIDENCE.** `docs/MODELS.md` 13.5. Against `lstm-telemanom` on `m1-g8.9.10`:

```
                        rare-FA     MVGS     precision   nominal-step FA
  lstm-telemanom          30/48    28/32    75/3,548             4.972%
  lstm-whitened            2/48    21/32        41/43             0.028%
  lstm-whitened-local      0/48    20/32        26/26             0.024%
```

The local reference made recall **worse** and the rule quieter, with its
calibration correct to four significant figures on every fold -- a clean negative
rather than a broken one.

**ALTERNATIVES.** Keep `lstm-telemanom`. Adopt the local reference. Loosen the
global threshold. Combine the two rules.

**EVIDENCE against each.** `lstm-telemanom` alarms on 30 of 48 commanded
manoeuvres and 4.97% of nominal timesteps; Objective.md 11 rule 2 says a detector
doing that is muted within a week. The local reference is measured and worse.
Loosening needs roughly a fourfold cut -- the lost events peak at a median 0.240
of the threshold -- which takes the 2/48 with it. Combining was scoped and
abandoned when the forensics showed **no lost event is single-channel** and
`lstm-whitened` catches nothing `lstm-telemanom` misses: there is no second view
to combine, only a stricter and a looser rule.

**CONSEQUENCE. `lstm-whitened` is the decision layer**, with the global
reference and pruning disabled, and it is **identical across LSTM, GRU and TCN**
at the architecture gate -- Objective.md 10.2's argument, that changing two
things at once makes the comparison unattributable.

> **The price is stated rather than absorbed: seven headline-cell events on the
> gate set and eleven anomalies overall, given up to take commanded-manoeuvre
> false alarms from 30/48 to 2/48 and nominal-step alarms from 4.97% to 0.028%.**

That trade is a judgement about what the component is for, and it is recorded as
one. The case for it is Objective.md 11 rule 2 -- a detector that alarms at most
manoeuvres is muted, and a muted detector catches nothing at all. The case
against it is that seven of the events given up are exactly the class the project
exists to catch, and nobody should pretend otherwise.

**(!) SUSPENDED 2026-08-28, before the architecture gate.** This entry froze
`lstm-whitened` without ever comparing it against `lstm-quantile` on post-fix
weights, because D13 had closed that branch -- on evidence now superseded. The
head-to-head is in `docs/RESULTS.md`; on `m1-g8.9.10` the two are the same
detector by every measure that matters (**headline cell identical at 21/32**,
recall differing by one event, rare-event rate identical at 2/48), and on
`m1-ss5` `lstm-quantile` is strictly better. Its nominal-step alarm rate is
**fourteen times lower** and its honest lead is **+0.0 against -41.5**.

**The decision layer is not frozen until that is settled**, and this entry does
not bind until then.

**What is not claimed.** Not that the eleven are unrecoverable -- four hypotheses
are refuted and the fifth failed, which is not the same as exhausting the space.
Not that whitening is the best relationship test, only that it is a measured
improvement on twelve channels read one at a time. And **not that the lead-time
figures above are comparable**: D21 measured the reported +26 as almost entirely
`error_buffer` dilation, and until that metric is honest the +29.0 here means
only *no worse than the arm it is compared with*.

---

## D25. The decision layer is `lstm-quantile`. Supersedes D24

**DATE** 2026-08-28 | **STATUS** resolved -- frozen for the architecture gate

**CONTEXT.** D24 froze `lstm-whitened` without ever comparing it against
`lstm-quantile` on post-fix weights, because D13 had closed that branch -- on
one-epoch models, evidence now superseded. The comparison D24 should have had is
below.

**EVIDENCE. Every axis, both sets, post-fix weights, honest lead-time reading.**

| `m1-g8.9.10` | F0.5 | recall | MVGS | precision | rare-FA | nominal-step FA | honest lead |
|---|---|---|---|---|---|---|---|
| `lstm-whitened` | 0.848 | 27/46 | **21/32** | 41/43 | 2/48 | 0.028% | **-41.5** |
| `lstm-quantile` | 0.838 | 26/46 | **21/32** | 40/42 | 2/48 | **0.002%** | **+0.0** |

| `m1-ss5` | F0.5 | recall | MVGS | precision | rare-FA | nominal-step FA | honest lead |
|---|---|---|---|---|---|---|---|
| `lstm-whitened` | 0.853 | 23/42 | 18/31 | 118/119 | 2/48 | 0.319% | **-51.0** |
| `lstm-quantile` | **0.885** | **27/42** | **21/31** | 127/130 | 2/48 | 0.293% | **+0.0** |

**And the detections are NESTED on both sets, in opposite directions -- there is
no crossing anywhere.**

```
  m1-g8.9.10   both 26   only whitened 1 (id_132)          only quantile 0
  m1-ss5       both 23   only whitened 0                   only quantile 4
                                                (id_149, id_165, id_186, id_187)
```

Every one of those five events has a **footprint of 1**. The two rules are
**ordered, not complementary**: on each set one strictly contains the other, so
an OR-combination equals the superset and buys nothing a single rule does not
already have. That closes the combination question without needing it scoped.

**ALTERNATIVES.** Keep `lstm-whitened` per D24. Combine the two. Keep
`lstm-telemanom`.

**Against each.** `lstm-whitened` is worse on `m1-ss5` by four events and three
headline-cell events, has a nominal-step alarm rate fourteen times higher on the
gate set, and its honest lead is **-41.5 against +0.0**. Combining is pointless
under nesting. `lstm-telemanom` alarms on 30 of 48 commanded manoeuvres.

**And the flight argument, which decides what the numbers leave close.**

| | `lstm-quantile` | `lstm-whitened` |
|---|---|---|
| per cycle | C EWMA updates, a max, one compare | C EWMA updates, C normalisations, **C^2 = 144 MACs**, a sqrt, one compare |
| `model.bin` | ~2 floats | **169 floats** (mean, scale, C x C precision, threshold) |
| failure mode | a wrong threshold is obvious | **a corrupted or ill-conditioned precision matrix yields plausible garbage** -- measured at condition 4.3e8 during development |

Both are fixed-time with no allocation and trivially bounded WCET. `lstm-quantile`
is simpler on every axis and its decision -- *this channel exceeded its threshold*
-- is directly auditable where a quadratic form is not.

**CONSEQUENCE. `lstm-quantile` is the decision layer**, identical across LSTM,
GRU and TCN at the architecture gate. `lstm-whitened` is **retained, not
deleted**: it is the measured proof that a relationship stage adds nothing at
this threshold on this data, which is a result in its own right and the evidence
behind `docs/NARRATIVE.md`'s entry on the mechanism this project preferred.

**(!) What freezing this does NOT fix.** `lstm-quantile`'s honest median lead is
**+0.0** -- it fires *at* the labelled event boundary, not before it. The metric
is no longer negative; that is not the same as warning early. Objective.md 1.1
stands unchanged: the retired claim stays retired, and the break-to-limit-trip
lead remains a Phase 3 measurement.

---

## D26. The GRU is a field of the LSTM's `Hyper`, and the LSTM's hash does not move

**DATE** 2026-08-28 | **STATUS** resolved

**CONTEXT.** Work item 5 adds the GRU as the second player at the architecture
gate. `docs/MODELS.md` section 1 is unconditional -- items 5 and 6 "have to
differ from this by architecture alone or the gate compares data pipelines
instead of architectures" -- and D25 froze the decision layer as identical
across LSTM, GRU and TCN. So the only permitted difference is the recurrent
cell, and the question is where that one difference lives.

Two things constrained the answer. The weight cache is keyed by a positional
tuple built from `Hyper.as_dict_key()` plus data digests (D14, still open), and
twelve production LSTM fits are banked under it; a GRU fitted with the same
`Hyper` on the same fold would otherwise hash to the **same file** and be served
LSTM weights. And nothing in the trainer, the reference or the weight file said
which cell a set of arrays was.

**ALTERNATIVES.** A separate `gru.py` with its own module and trainer. A ninth
positional element in the key. D14's named, versioned key, implemented now.
Leaving the cell out of the key and trusting the detector class.

**EVIDENCE against each.** A second trainer duplicates the fit loop, so every
shared line -- the seeding, the sampler, the relative stopping rule, the guard
that raises on `best_epoch == 0` -- becomes a place for the two cells to drift,
and the gate would then compare training loops as well as cells; it also
closes an import cycle with `lstm.py`, and a module that imports `DEVICE` by
value freezes `"cpu"` while `scripts/fit_folds.py` rebinds it to `"cuda"` at
run time. A ninth positional element is the D14 incident again: every banked
fit orphaned, silently. Implementing D14 now orphans the same twelve fits for a
reason unrelated to the GRU, and would put two changes in one comparison.
Trusting the class collides in the store: measured, `Hyper()` and a hypothetical
`Hyper(cell="gru")` without the field produce the same digest, and a GRU
detector would then load a 4-gate file and score the wrong architecture with
no error -- `reference.forward` runs whatever arrays it is handed.

**What was built.**

* `Hyper.cell`, default `"lstm"`, refused unless it names a known cell.
  `as_dict()` emits it **only when it is not the default** -- D14's rule, an
  optional field at its null value leaves the hash unmoved. `tests/
  test_lstm_detector.py` pins the LSTM's fourteen keys, its literal key tuple,
  a cache digest (`cc91392d...`) and the fingerprints of the banked artifacts
  (`6b9ebb0d`, `2717441a`, `9d5cca78`); the test passed before the change and
  after it, and every one of the 57 files under `runs/_weights/` loads through
  the new loader with its cell inferred.
* One torch module, `TelemanomRNN`, whose only cell-specific line is which of
  `nn.LSTM` / `nn.GRU` it builds; `TelemanomLSTM` and `TelemanomGRU` are the two
  by name. `train()` is untouched apart from constructing it, so the GRU is
  fitted by the same seeds, sampler, loss, optimiser, stopping rule and guard
  by construction.
* `reference.py` gains `gru_cell` and `gru_layer` beside the LSTM's, and the
  cell of a `Weights` is **derived from its arrays' gate count** -- 4H rows is
  an LSTM, 3H a GRU -- never stored as a claim. The weight file writes `cell`
  anyway, and the loader verifies it against the arrays, inferring it for files
  that predate the field; a file that declares one cell and holds the other is
  refused. The state is `(h, c)` for an LSTM and `(h,)` for a GRU, one tuple
  per layer; a GRU does not carry a dead cell vector, because the flight state
  is exactly what the reference lists.
* Detectors declare their cell on the class and refuse a `Hyper` of the other
  one, so a GRU detector handed the fixture-scale `Hyper(...)` the tests pass
  around cannot silently become an LSTM. `gru-telemanom`, `gru-quantile` (the
  gate arm) and `gru-smoke` are registered beside their LSTM twins.

**Measured.** 71,160 parameters at the flown configuration, 278.0 KiB float32,
against the LSTM's 91,640 -- 22.35% fewer overall, exactly 25% fewer in the
recurrent layers, the head being shared. torch-vs-NumPy agreement at 12
channels, 2x80, 250 steps: **1.19e-07** for the GRU (4.1e-08 for the LSTM),
unchanged on a 4,096-step chunk; `tests/test_reference_equivalence.py` holds
both cells at 1e-5 through the same parametrised tests.

**A format consequence, recorded in docs/MODELS.md section 3.** The GRU's third
recurrent bias `b_hn` sits *inside* the reset product,
`n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`, so it cannot be pre-summed with
`b_in` the way an LSTM's two bias vectors can. The model file must store both
bias vectors unsummed for a GRU, and a loader written from the LSTM habit is
wrong on exactly that gate. Two tests pin it: folding the biases diverges in
general and agrees once the reset gate is saturated open.

**CONSEQUENCE.** The GRU is a value of one field, and the LSTM is that field's
default. D14 remains open and is annotated. Nothing in `sentinel_eval` changed.

---

## D27. The TCN is the third value of the same field, and its state is nothing

**DATE** 2026-08-29 | **STATUS** resolved

**CONTEXT.** Work item 6 adds the temporal convolutional network, the third
player Objective.md section 8 names. The rule is D26's: the architecture is
the only permitted difference, so it must travel through the same `Hyper`,
the same `train`, the same detector and the same decision layer. A TCN is not
a recurrent cell, and it has no state, so two of D26's arrangements had to be
extended rather than reused.

**ALTERNATIVES.** A separate module and trainer for the TCN. A second field
for the architecture beside `cell`. Carrying a dummy state through `forward`
so the three players share one signature. A receptive field of exactly 250.

**EVIDENCE against each.** A separate trainer is the drift D26 refused: every
shared line becomes a place for the players to diverge. A second field is a
second thing to keep out of the LSTM's hash; one field with a third value,
emitted only when it is not the default, keeps the twelve banked LSTM fits and
every fingerprint unmoved -- the identity pin from D26 passes unchanged. A
dummy state is a lie in the flight blueprint: Objective.md 8 lists *no state
to corrupt* as the TCN's reason to exist, and a `(h,)` of zeros carried for
symmetry would be exactly the kind of thing a C++ port reads as a requirement.
A receptive field of exactly 250 is not a standard shape -- `1 + 2(k-1)(2^L-1)`
gives 249 at k=5, L=5 and 253 at k=3, L=6 -- and the brief asks for at least
250; 253 is the first standard shape past it.

**What was built.**

* `Hyper.cell = "tcn"`, with one TCN-only field, `kernel`, emitted by
  `as_dict` only for a TCN. `hidden` is reused as the width of each residual
  block, so the flown shape is `hidden=(50,)*6, kernel=3`: dilations 1 to 32,
  receptive field **253**, **91,670 parameters** against the LSTM's 91,640 --
  the closest the shape allows (49 gives 88,222; 51 gives 95,184). The GRU was
  not size-matched (71,160), and Objective.md 8's criterion 2 is read with
  that in mind.
* `TelemanomTCN`: Bai, Kolter and Koltun's block -- two causal dilated
  convolutions, ReLU and dropout after each, a 1x1 shortcut where the width
  changes, a final ReLU -- with the recurrent module's head, final dropout and
  `forward` contract. **No weight normalisation**: it is an optimisation aid
  folded into plain weights at export, so the file format never carries it,
  and leaving it out keeps the count exact. `build_model` is the one place the
  architecture is chosen; `train` is otherwise untouched, so the TCN is fitted
  by the same seeds, sampler, loss, optimiser, stopping rule and guard.
* `reference.ConvWeights`, `causal_conv1d`, `tcn_block`: the blueprint. A
  kernel tap `j` reads the input `(k-1-j)*dilation` steps back; the
  `(k-1)*dilation` steps before the sequence are zero -- **the ring buffer at
  boot**. `forward` **refuses a state** for these weights and returns an empty
  one: the third contract, beside `(h, c)` and `(h,)`. The harness's chunked
  scoring already warms every chunk with `window = 250` real steps and carries
  nothing across chunks, which is exactly a 252-input ring buffer after a cold
  start; the three steps a 253-field reaches past that prefix are zero-filled,
  as they are in every 250-step training window, so training and scoring see
  the same arithmetic.
* The weight file writes `cell = "tcn"` with its own keys (`b{i}_w1 ... `,
  `dilations`), verified on load; the two cells' files and the loader's path
  for them are byte-for-byte what they were. `tcn-telemanom`, `tcn-quantile`
  (the gate arm) and `tcn-smoke` are registered.

**Measured before any fit.** torch-vs-NumPy at the flown shape **4.8e-07**
(6.0e-07 on a 4,096-step chunk); the field pinned by perturbation -- 252 steps
back moves the forecast, 253 does not; a chunk warmed by 252 real steps equals
the uncut stream to 1e-5; the future cannot reach the past.

**CONSEQUENCE.** Three players, one field, one trainer, one detector, one
decision layer. What differs is the block that reads the history and the
shape of what it carries between ticks: two vectors, one, none.

---

## D28. The architecture gate selects the GRU; the LSTM stays the baseline, the TCN the answer to "why not stateless"

**DATE** 2026-08-29 | **STATUS** resolved -- the gate. Not reopened by the
held-back transfer result (D29), by construction.

**CONTEXT.** Objective.md section 8 set five criteria in priority order before
any model existed; D3 and D9 refined what "detection performance" is measured
by. Three players now hold rows on the same referee, the same data, the same
folds and the same frozen decision layer (D25): `docs/RESULTS.md` 6h (GRU), 6i
(the LSTM reseed), 6j (TCN), with the combination scoped in `docs/MODELS.md`
17. The architecture is decided here, on Mission 1 evidence, **before** the
held-back sets are scored -- so that the transfer scoring stays an exam and
does not become another tuning set.

**EVIDENCE, criterion by criterion.** Artifacts:
`runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json`,
`runs/m1-g8.9.10/gru-quantile/2026-08-28T222635Z-6d146f5d.json`,
`runs/m1-g8.9.10/tcn-quantile/2026-08-29T162030Z-c48bd47d.json`,
`runs/m1-g8.9.10/_forensics/2026-08-28T223610Z-head-to-head.json`,
`runs/m1-g8.9.10/_forensics/2026-08-28T232538Z-reseed-lstm-quantile-seed1.json`,
`runs/m1-g8.9.10/_forensics/2026-08-29T170340Z-combination-scope.json`, and
the fit reports under `runs/_weights_pod/`.

| criterion | LSTM | GRU | TCN | reading |
|---|---|---|---|---|
| **1** detection, as section 8 wrote it: contextual recall first, rare-event FA second -- `m1-g8.9.10` / `m1-ss5` | 26/45, 2/48 / **27/41, 2/48** | **27/45, 1/48** / 26/41, 3/48 | 9/45, 3/48 / 16/41, 3/48 | LSTM and GRU one event apart on each axis, in opposite directions on the two sets: **a tie at the resolution 1/n the harness itself refuses to read as a finding** (`docs/HARNESS.md` 1). The TCN is out |
| 1, as D3 reads it: event-wise F0.5 | **0.838** / **0.885** | 0.804 / 0.593 | 0.411 / 0.649 | **The LSTM leads on both sets.** On the gate set the gap is one alarm range's worth of precision (40/42 against 139/157); on `m1-ss5` it is 0.29, from precision 127/130 against 101/172 |
| 1, as D9 reads it: honest lead | +0.0 | +0.0 | -4.0 | nothing is disqualified |
| headline cell, `m1-g8.9.10` | 21/32 | **22/32** | 9/32 | one event, resolution 1/32 |
| **2** parameters | 91,640 | **71,160** | 91,670 | GRU 22.35% smaller, 25% in the recurrence |
| **3** inference cost, MAC per tick from the section-3 shapes | 90,240 | **70,080** | 90,900 | GRU 0.78x |
| **4** C++ implementation and verification | four gates, two state vectors; the two bias vectors may be summed | **three gates, one state vector**; `b_hn` sits inside the reset product and is the one place a loader written from the LSTM is wrong (`docs/MODELS.md` 3, tested two-sided) | no state, one tap-order convention; the simplest of the three -- and out on criterion 1 | GRU over LSTM |
| **5** determinism and restart | bit-identical on a box (D15); restores two vectors | bit-identical (D26); **restores one** | bit-identical (D27); restores nothing | GRU over LSTM |
| trainability -- a recorded gate consideration since `docs/RESULTS.md` 6i | **stalled** on fold 0, seed 0: 14 epochs, best 3, 1.691e-4; the reseed reached 1.986e-5 and was still 4.3x the GRU | **no stall on any fold of either set**, six fits, all to or near the cap | **stalled** on `m1-ss5` fold 2: 13 epochs, best 2, 6.059e-5 | GRU alone |

**The sentence a reader is owed.** On the gate metric D3 named, event-wise
F0.5, the LSTM is ahead on both sets, and **a reader weighing that metric
alone would pick the LSTM.** The selection does not rest there. Criterion 1
as Objective.md section 8 wrote it is a tie inside the resolution the harness refuses to
call a finding; Objective.md section 8 wrote its own outcome for that case -- *"LSTM
retained as the published-comparison baseline, with GRU or TCN selected for
flight if it matches"* -- and the GRU matches; and on every criterion below
the first, and on the one criterion the gate learned it needed after it was
written, the GRU is the smaller, cheaper, simpler, single-state cell that
did not stall.

**The price, stated in the entry and not footnoted.** Six events lost on fold
1 for seven gained on fold 0 (`_forensics/2026-08-28T223610Z-head-to-head.json`)
-- the fold-1 noise floor rose 1.81x with the validation MSE unchanged to 5%,
the floor being a tail statistic and the loss a mean. Precision 139/157
against 40/42 on the gate set and 101/172 against 127/130 on `m1-ss5`. F0.5
0.593 on `m1-ss5`. Those numbers travel with the decision.

**ALTERNATIVES.** Keep the LSTM. Select the TCN. Defer the gate to the
held-back transfer result.

**Against each.** The LSTM's F0.5 lead on the gate set is one alarm range of
precision, and the LSTM is the cell that stalled on the data-poor fold under
the published protocol and, reseeded, still landed 4.3x short. The TCN
catches nine of forty-six on the gate set and its floor sat above both cells'
on every fold; criteria 4 and 5 do not outrank criterion 1. Deferring would
let the exam set choose the architecture, which is the one use of a held-back
set that destroys it (`docs/MODELS.md` 6).

**CONSEQUENCE.**

1. **`gru-quantile` is the flight architecture.** The GRU reference
   (`reference.gru_cell`, held at 1e-5) is the Phase 2 blueprint.
2. **The LSTM is retained** as the published-reproduction baseline -- the
   telemanom lineage row, and half of the union below. Its published row
   stays the seed-0 fit; the reseed (`docs/RESULTS.md` 6i) is evidence on
   trainability, not a row, and the question of refitting the row is closed
   with this entry.
3. **The TCN rows are retained** as the measured answer to "why not
   stateless": strict-subset catches, a forecast 1.3-9.9x worse than the
   GRU's, one stalled fit -- at the published protocol and the size-matched,
   lookback-matched shape, which is all that was tested.
4. **The union `lstm-quantile OR gru-quantile` is a post-gate deployment
   configuration, not a gate contestant.** Measured (`_forensics/2026-08-29T170340Z-combination-scope.json`):
   33/46 and **25/32 at 2/48** on `m1-g8.9.10`, 34/42 and 25/31 at 3/48 on
   `m1-ss5`. The gate question is *which architecture*; the union is a
   configuration built from two answers. Its flight cost is on the record --
   1.78x the multiplies per tick, 636 KiB of weights, two thresholds, two
   reference paths, two equivalence tests: a doubled verification surface.
   **Its adoption is a separate decision, D29, taken after the held-back
   transfer result exists and nowhere else.** That result does not reopen
   this entry.

Objective.md 14.1 (row 1 of the open-decisions table) is resolved by this entry.

---

## D29. `gru-quantile` flies alone; the union is not adopted; the calibration's transfer is the finding

**DATE** 2026-08-29 | **STATUS** resolved -- Phase 1 closed on this entry.
D28 is not reopened.

**CONTEXT.** The held-back sets were scored once, on a GO, with the commands
and predictions of `docs/MODELS.md` 18 committed before the seal broke:
`m2-ss1` (Mission 2, 424 rare events test-side, recall disabled) and `m1-g3`
(Mission 1 group 3, 11 anomalies and 13 rare events test-side, three folds).
Section 18.6 set the rule for this entry in advance: the union is the
recommended configuration if its edge survives (R3) and its cost holds (T4);
otherwise `gru-quantile` alone.

**EVIDENCE.** `runs/m2-ss1/<detector>/2026-08-29T204415Z-*.json`,
`runs/m1-g3/<detector>/2026-08-29T223625Z-*.json`; `docs/RESULTS.md` 6k;
`docs/MODELS.md` 18.8.

*The union.* On `m2-ss1` its rare-event alarms were 8/424 -- the LSTM's 4 and
the GRU's 4, **disjoint** -- against a pre-registered fail line of 1.5x the
larger member: **T4 failed.** On `m1-g3` its headline-cell recall was 9/10,
the LSTM's 9/10: **R3 evaporated**, at the GRU's rare-FA of 10/13. Either
alone selects the GRU alone; both did. The Mission-1 economy -- the GRU's one
rare alarm sitting inside the LSTM's two -- did not transfer.

*The recipe, on an independent spacecraft.* `lstm-quantile` 4/424, `gru-quantile`
4/424, `tcn-quantile` 6/424 rare-event false alarms; **0 nominal-step alarms in
4,155,841** for all three; the per-channel floors 84/424 with 17.3% of nominal
time (`rstd`) and 122/424 (`mavg`). T1, T2, T6 held; T3 held on the ordering.
**(!) The two floor figures are corrected 2026-09-01 (D37): `rstd` is 22/424 with
0.003% of nominal time, not 84/424 with 17.3%, and `mavg` is 208/424. The
forecaster figures are unmoved and were not recomputed. This entry's conclusion
is unchanged -- 4/424 against 22/424, and 0 nominal-step alarms against 126 --
and its evidence is cleaner, because the floor it is measured against now
works.**
That is the number that goes in front of a mission.

*The recipe, on a later period of the same spacecraft.* `m1-g3` fold 0 held as
the gate set did -- 0 nominal-step alarms, rare-FA 1/3. **Folds 1 and 2 did
not**: thresholds of 0.0109 and 0.0111, calibrated on the first 7.36M steps,
sat under **86.7% of the fold-1 test window's nominal residual** for both
cells (3,095,318 of 3,570,272 steps), and under 4.4% / 0.7% of fold 2's.
Pooled nominal-step FA 29.9% and 28.7%, rare-FA 8/13 and 10/13. Six fits, no
stall, validation MSE in the gate set's range: the forecaster fitted; the
frozen rule's premise did not hold.

**ALTERNATIVES.** Adopt the union anyway on Mission 1's +4/32. Recalibrate on
`m1-g3`'s later window and re-score. Reopen D28 on the `m1-g3` numbers.

**Against each.** The union's Mission-1 edge was the pre-registered claim
under test and it did not survive either exam; adopting it on the evidence
it failed is the tuning-to-the-test the held-back sets exist to prevent.
Recalibrating and re-scoring is the one thing `docs/MODELS.md` 6 forbids in
as many words -- a disappointing held-back result is the finding, not a
problem to fix -- and the value of the sealed data was spent in one
transaction. D28 was decided before the seal broke precisely so that this
result could not choose the architecture; and the `m1-g3` collapse is not an
architecture result -- both cells collapsed identically, by the same
mechanism, on the same window.

**CONSEQUENCE.**

1. **`gru-quantile` is the recommended deployment configuration, alone.** The
   union is not adopted; it stays measured (`docs/MODELS.md` 17, `docs/RESULTS.md`
   6k) as a configuration to revisit only if a future decision layer makes
   the two cells' alarms overlap again.
2. **The transfer evidence is split, and both halves are reported.** On an
   independent spacecraft the recipe -- fit on a third of the history,
   calibrate on your own nominal residual, no labels, no tuning -- alarms on
   one rare event in a hundred and on no nominal step. On a later period of a
   second subsystem of Mission 1, the same recipe's floor was under 87% of the
   window. The first is the adoption number; the second is the limit of a
   floor fixed in the past.
3. **What Phase 2 inherits from the second half.** D25's rule stays the Phase 1
   decision layer; its premise -- one global quantile of the fitting window is
   the noise floor of what follows -- is now measured to fail across a regime
   change. Objective.md 10.2 fix 4 and 14.10 already require thresholds to be
   recalibrated in orbit and stored outside the weights; `m1-g3` fold 1 is the
   measured case, and how the flight component recalibrates without learning
   a degradation as normal (Objective.md 11 rule 1) is Phase 2's first
   decision on the decision layer. Hypothesis, unmeasured, recorded as the
   first thing to look at: the residual's scale in group 3 changes after the
   seed window; a local threshold would have followed it and would have fired
   constantly on the gate set (D17, D18) -- the same asymmetry from the other
   side.
4. **Phase 1 is closed.** Its gate -- match or beat a telemanom baseline
   reproduced on this harness, plus evidence-based architecture selection --
   is passed on the gate set (`docs/RESULTS.md` 6e-6h, D28), and the
   transfer experiment the design protected since day one is spent and
   written. No held-back set remains.

---

## D30. The `model.bin` format is frozen: plain little-endian float32, gate order named, parameters in their own CRC'd block. Supersedes Objective.md 14.10's format implication

**DATE** 2026-09-01 | **STATUS** resolved -- the freeze. Objective.md 14.2 is
resolved by this entry

**CONTEXT.** Objective.md 14.2 requires the model-file format frozen **before Phase 2
starts**, because it is the contract between the Python toolkit and the C++ loader.
Two committed statements of that format disagree, and the disagreement had to be
settled before any flight code could be written. Objective.md 14.10 specifies "a
quantized, self-describing FlatBuffer, TFLite-Micro compatible", with a header
carrying "model type, version, channel count and ordering, window length, quantization
parameters, a mandatory `baseline_only` flag, and a CRC over the weights";
`CHANGELOG.md` repeats it in the open-decisions table. `docs/PHASE2.md` 3 and
`docs/STATUS.md` 7, both written at the close of Phase 1, say plain float32 arrays
with the gate order named in the header. Objective.md declares itself the living
document the others defer to, so precedence alone does not settle it.

**ALTERNATIVES.** The FlatBuffer as Objective.md 14.10 writes it. A quantized plain
format. The plain float32 format `docs/PHASE2.md` describes.

**Against the first two, three reasons, in the order they bind.**

1. **Nothing here consumes TFLite.** The inference core is a hand-written
   transcription of `reference.py`, checked against it at 1e-5 -- that is D15's
   premise and the whole purpose of `tests/test_reference_equivalence.py`. A
   TFLite-Micro-compatible container is compatibility with an interpreter this
   project does not have and, by Objective.md 4.3 step 2, will not fly.
2. **A FlatBuffer parser is templated, allocating third-party code the flight rules
   exclude.** F' CPP-25 forbids the STL and templates beyond the simple, CPP-1
   forbids allocation after initialisation, and CPP-7 caps template complexity.
   Vendoring a generated parser to read ten arrays inverts the cost.
3. **A fixed layout with a CRC is byte-inspectable by a review board.** A reviewer can
   read `docs/MODEL_FILE.md` 3 and a hexdump side by side and check the file by hand.
   That is worth more here than self-description, because the thing being reviewed is
   a neural network on a spacecraft.

**Against quantization specifically.** The acceptance tolerance against `reference.py`
is 1e-5 (`tests/test_reference_equivalence.py:36`), and the GRU's measured headroom is
1.2e-07. Int8 quantization loses two to three orders of magnitude more than that, so
a quantized file cannot meet the tolerance every result in this project was measured
against. `docs/MODELS.md` 3 never endorsed quantization: it says only that at 278.0
KiB "quantization is about flash budget and load time rather than feasibility" -- and
at 278.0 KiB neither is pressing.

**EVIDENCE.** 71,160 parameters, 284,640 bytes as float32 (`docs/MODELS.md` 3,
`tests/test_reference_equivalence.py:282`). A complete file at the flown shape is
285,136 bytes: a 64-byte header, a 240-byte channel map, the weights, and a 192-byte
parameter block. The whole file fits a standard F' file uplink without special
handling.

*And the measurement that shapes the parameter block.* Thresholds are
fitting-procedure-specific, not merely model-specific: two calibrations of one rule on
one set of residuals produced 126 alarm ranges and 2,405 -- nineteenfold, from how the
constants were fitted (Objective.md 14.10, `docs/RESULTS.md` 6b). A threshold that
suited a one-epoch model produced 3,548 alarm ranges on a trained one (D17). And D29
measured the case that decides the design: on `m1-g3` folds 1 and 2 a floor calibrated
on the first 7.36M steps sat under 86.7% of a later window's nominal residual, on the
same spacecraft. Thresholds must be recalibrable in orbit without retraining.

**CONSEQUENCE.**

1. **The format is frozen at version 1 and specified byte for byte in
   `docs/MODEL_FILE.md`**, which is normative and owned by `src/sentinel_export/` as
   that package's docstring has said since it was created. Little-endian float32, no
   quantization, no generated parser, no third-party dependency.
2. **The gate order is named in the header, not inferred from a slice index.**
   `gate_order_id = 1` means `reference.GRU_GATES == ("reset", "update", "new")`.
   `arch_id = 1` means the GRU. Both are verified against the payload on load, never
   merely read -- the D16 rule `detectors._load_weights` already applies to `cell`.
3. **Both bias vectors are stored unsummed.** `b_hn` sits inside the reset product and
   cannot be folded into `b_in` (`docs/MODELS.md` 3, amended by D26). The LSTM's may
   be summed and the GRU's may not, and the format does not offer the choice.
4. **The parameter block carries its own CRC, separate from the weights' CRC.** This
   is the entry's operative decision. Normalisation constants, the threshold, the EWMA
   span, `baseline_only` and the tier live there, outside the weight block, at a fixed
   size, so an in-orbit recalibration is a fixed-length overwrite plus an eight-byte
   header patch and the 278.0 KiB of weights are never touched or re-verified. That is
   what Objective.md 10.2 fix 4, Objective.md 14.10 and D29 consequence 3 require, made
   executable.
5. **The threshold is stored as F64 and compared as F64.** `harness.py:169-172`
   compares `combined.astype(np.float64) >= threshold` against a `np.quantile` result;
   storing an F32 would round the cut and could flip a crossing at the boundary. The
   comparison is `>=`, not `>`.
6. **`baseline_only` is in the header of the parameter block, as Objective.md 14.10
   requires.** It is the Level 1 switch (D5) and work item 9 wires it to the
   active-tier telemetry channel.
7. **Objective.md 14.10's format implication is superseded by this entry and marked
   superseded, not deleted**, together with its restatement in `CHANGELOG.md`.
   Everything else in 14.10 stands: the three tiers, Level 1 as the loader's mandatory
   safe failure mode, the `baseline_only` flag, the CRC, and constants stored outside
   the weights. Only the container -- FlatBuffer, quantized, TFLite-Micro -- is struck.

---

## D31. F' is pinned at v4.3.0, and the work item 8 core is freestanding C++14 behind a types shim

**DATE** 2026-09-01 | **STATUS** resolved. Objective.md 14.4 is resolved by this entry

**CONTEXT.** Objective.md 14.4 lists the target F' version pin as an early-Phase-2
decision, with the reason "everything downstream depends on it". Work item 8 forced
it early for a second reason: the work item requires the flight rules cited from F's
own coding standard, and a rule cannot be cited without a version.

**EVIDENCE.** The current F' release is **v4.3.0**, published 2026-08-20. Its
authoritative statement of the C/C++ design rules is
`.github/skills/fprime-cpp-design/SKILL.md`, which describes itself as "the **single
source of truth** for the C/C++ design rules F Prime flight software is held to" and
carries the numbered rules CPP-1 to CPP-34. Reading it corrected three rules this
project had stated from memory, all recorded in `docs/MODELS.md` 19.3: F' states no
no-recursion rule (that is Power of Ten rule 1 and the JPL C standard, which F' cites
at CPP-27); F's own no-heap rule is CPP-1 rather than Power of Ten rule 3; and F'
forbids bare `float` and `double` outright at CPP-3, which the project had not
recorded anywhere and which changes every declaration in the core.

**ALTERNATIVES.** Build work item 8 against a vendored F' checkout. Defer the pin to
work item 9. Pin an older release.

**Against each.** Building against F' now would pull work item 9's integration into
work item 8, which the work item scopes out, and would require a build system this
machine does not have -- `cmake` is not installed. Deferring the pin leaves the flight
rules uncitable, which is the one thing work item 8 was asked to get right. An older
release would pin to rules that have since been restated; v4.3.0 is current and its
rule set is the one a reviewer will hold this code to.

**CONSEQUENCE.**

1. **F' is pinned at v4.3.0** for Phase 2. `docs/MODELS.md` 19.3 tabulates the rules
   this core obeys, each cited to its CPP number.
2. **The work item 8 core is freestanding C++14**, depending only on `<cstdint>`,
   `<cstring>`, `<cmath>` and the three `<algorithm>` and `<limits>` helpers CPP-25
   explicitly permits. `flight/include/sentinel/Types.hpp` defines `F32`, `F64`, `U8`,
   `U16` and `U32` with F's exact names and meanings; work item 9 replaces that one
   header with `Fw/FPrimeBasicTypes.hpp` and no line of the core changes.
3. **CRC is carried locally and matches F' exactly.** `flight/` holds its own
   CRC-32/IEEE 802.3 table; `Utils/Hash/Crc32/Crc32.hpp` at v4.3.0 implements the same
   polynomial, so work item 9 may substitute `Utils::Hash` with no change in value.
4. **`flight/CMakeLists.txt` is written now and used later.** It carries the identical
   flag set to the Makefile so the F' build cannot be quietly laxer than the one work
   item 8 verified against; a test asserts the two agree rather than trusting them to.
5. **`clang-tidy` is deferred to work item 9**, with the F' toolchain. It is not
   available on the development machine. `flight/.clang-tidy` and a `make lint` target
   are written now and run the moment a toolchain has it.

---

## D32. `Sentinel::Monitor` is a passive component driven by a synchronous `Svc.Sched`

**DATE** 2026-09-01 | **STATUS** resolved

**CONTEXT.** Nothing in the repository had ever stated whether the F' component is
passive, queued or active: a search of `Objective.md`, `README.md`, `CHANGELOG.md` and
every document under `docs/` for those words in the F' sense returns nothing, and
`schedIn` appears nowhere. The choice had to be made against F's own guidance rather
than from the project's memory.

**EVIDENCE.** `nasa/fprime` v4.3.0
`docs/user-manual/framework/component-and-port-selection.md` classifies work as cyclic,
event-driven or background, and says of the first: "Passive components are the natural
choice for cyclic work as we intend them to execute in the context of the invoking rate
group. A good starting model for cyclic work is to have a passive component with a
`sync` port of type `Svc.Sched` that performs the repeating work for that component each
cycle." It also says why not asynchronous: "Asynchronous invocations are not used because
this work would be run outside the rate group cycling context and thus slips and failure
to reach deadlines would be hidden in another thread." Sentinel's flight loop is exactly
this -- one `step()` per rate-group tick, fixed compute, no waiting on anything.

The same document names the condition that would change the answer: "This simple
`passive` pattern breaks down when the cyclic component needs to accept some Event-Driven
work (e.g. it processes some commands)", for which it prescribes `queued` with the queue
dispatched inside the `Svc.Sched` handler, and gives `Svc.Health` as the example.

**ALTERNATIVES.** Queued now, to leave room for work item 10's reload command. Active
with an `async` `Svc.Sched`, F's "Cyclic Notification Pattern".

**Against each.** Queued now buys nothing: work item 9 has no asynchronous port and no
command, and the same document closes with "You should prefer the simpler models above
wherever possible." A queue with nothing to dispatch is a place for a fault to hide.
Active is wrong on its face -- it hides rate-group slip in another thread, which is the
one thing the cyclic section says not to do, and it would put a neural network forward
pass on its own thread with no deadline visible to the topology.

**CONSEQUENCE.**

1. **The component is `passive`, with `sync input port schedIn: Svc.Sched`.** All work
   happens in the tick, in the caller's context, so a slip is visible as a slip.
2. **Work item 10 promotes it to `queued`**, because its reload path is an `async`
   command. That is a one-word change in the FPP plus a queue dispatch at the top of the
   `schedIn` handler, and it is recorded here so work item 10 does not have to
   rediscover it. The reload command is named in the FPP as a comment and in the SDD;
   no code implements it.
3. **Both input ports are `sync`, and the topology assumption is documented and tested**:
   the sample producer sits on the same rate group at a lower port index, so one thread
   runs both. A mission wiring a producer on another thread makes `channelsIn` `guarded`,
   which F' describes as "synchronous with an internal mutex to protect data".

---

## D33. Channel ingestion is direct port wiring with an FPP-modelled vector, not a telemetry-path tap. Resolves Objective.md decision 3

**DATE** 2026-09-01 | **STATUS** resolved. Objective.md decision 3 is resolved by this entry

**CONTEXT.** Objective.md decision 3 -- "Channel-ingestion mechanism: tapping the
telemetry path vs. direct port wiring" -- was deferred to early Phase 2 with the note
"This is a work item 9 question: the inference core takes a vector of channel values and
does not care how they arrive." It is now due.

**EVIDENCE.** The tap is the more attractive design on its face, because it is what
`Objective.md` 4.3 step 5 promises -- "Nothing downstream needs to know Sentinel exists"
-- and because `docs/MODEL_FILE.md` 4 already stores "`id  U32  the F' telemetry channel
id (FwChanIdType)`" for every channel, which only a tap would need. F' supports it:
`Svc::TlmPacketizer` carries `sync input port TlmRecv: Fw.Tlm` and is the framework's own
example of a rate-group-driven telemetry consumer.

**It is nevertheless not implementable against this file format.** `Fw.Tlm` is declared
in `Fw/Tlm/Tlm.fpp` as carrying `ref val: TlmBuffer` -- "Buffer containing serialized
telemetry value". Deserialising a `TlmBuffer` requires knowing the channel's declared
type, and `docs/MODEL_FILE.md` 4's CHANNELS record carries `id` and `name` and **no type
tag**. A tap would therefore have to assume every watched channel is serialized F32, and
be silently wrong on a mission whose battery voltage is a `U16`. Adding a type tag is a
`format_version` bump, and D30 froze version 1.

**ALTERNATIVES.** The tap, with an assumed F32 encoding. The tap, with a
`format_version = 2` carrying a per-channel type tag. Direct port wiring.

**Against the first two.** An assumed encoding fails silently, which is the exact failure
mode `docs/MODELS.md` 3 rejects for gate order -- "a model that runs, converges to
nothing useful, and fails silently". A version bump to buy an ingestion convenience
reopens a format frozen eight days earlier for reasons that have not changed.

**CONSEQUENCE.**

1. **Ingestion is a `sync input port channelsIn: Sentinel.ChannelSample`**, where
   `port ChannelSample(ref values: ChannelVector, valid: bool)` and
   `array ChannelVector = [16] F32`, sized against `Config::MAX_CHANNELS` by
   `static_assert`. FPP-modelled, per CPP-23.
2. **The mission supplies a small adapter.** F' names the shape: the "Passive Adapter
   Pattern", "a passive component as an adapter that is called synchronously as part of
   the primary port call ... you need to connect two components with incompatible port
   types and need to reconcile those types". This is a real adoption cost and
   `Objective.md` 4.3 step 5's "nothing downstream needs to know Sentinel exists" is
   thereby weakened, not met: a mission wires one adapter. Recorded rather than glossed.
3. **The CHANNELS block's `id` field keeps its meaning and its use.** It is what the
   warning event reports, so an operator sees a channel id they already know, and it is
   what a future tap would need if a later format version carries a type tag.
4. **The tap stays measured and declined, not deleted**, as `lstm-whitened` did at D25.
   If a `format_version = 2` ever carries per-channel types, this entry is the place to
   revisit.

---

## D34. Level 1's baseline constants are F' parameters with defaults, not `model.bin` fields

**DATE** 2026-09-01 | **STATUS** resolved

**CONTEXT.** Level 1 is "the loader's safe failure mode" (D5, Objective.md 14.10): on a
corrupt file, a failed CRC or a version mismatch, the component runs a statistical
baseline instead of the network. The baseline is the `rstd` rule, and the rule is not
parameter-free: `RollingStd` carries a fitted per-channel scale (`baselines.py:105`) and
is compared against a threshold calibrated on its own statistic. `model.bin` version 1
carries neither. Its PARAMS block holds the *model's* threshold, whose committed values
run 0.2737089991569519 to 0.951245903968811, where `rstd`'s calibrated cuts on
`m1-g8.9.10` are 3.339457480522701, 4.386269506802676 and 4.380987492380889. They are
different statistics on different scales and one cannot stand in for the other.

**EVIDENCE.** The decisive argument is not storage, it is availability. The refusal that
Level 1 exists to survive is `BAD_HEADER_CRC` -- and `docs/MODEL_FILE.md` 3 requires the
header CRC be "verified *before* any length field is used", precisely because nothing in
a file with a bad header can be trusted to bound a read. Baseline constants stored inside
that file are unreadable in exactly the case they are needed.

`Objective.md` already names the mechanism that does work. 14.10: "Normalisation
constants and detection thresholds are stored separately, as small PrmDb-style
parameters". And `Objective.md:1012-1018` records the precedent: "F's own equivalents are
PrmDb - a generic component loading a mission file ... We are following an established
framework pattern, not inventing one."

**ALTERNATIVES.** Extend PARAMS and bump to `format_version = 2`. Compile-time constants
in the component's configuration. F' parameters with FPP defaults.

**Against the first two.** The version bump reopens D30's freeze and still leaves the
constants unreadable on a bad header. Compile-time constants are always available, which
is their merit, but a threshold baked into flight code has no provenance and cannot be
recalibrated in orbit -- which D29 consequence 3 and Objective.md 10.2 fix 4 both forbid,
and which two calibrations of one rule differing nineteenfold (`docs/RESULTS.md` 6b)
makes concrete.

**CONSEQUENCE.**

1. **`param BASELINE_SCALE: ChannelVector` and `param BASELINE_THRESHOLD: F64`**, both
   with FPP defaults. The default is the compile-time fallback, so Level 1 has constants
   with no parameter database present; PrmDb supplies mission values; and work item 10's
   uplink path reaches them by the same mechanism it will use for the model's own
   threshold.
2. **`model.bin`'s format does not change.** D30 stands, version 1 stands, and
   `docs/MODEL_FILE.md` needs no revision for this entry.
3. **The baseline is armed independently of the model.** The component can be in
   BASELINE mode with no model file present at all, which is what "never fail the
   topology" requires.

---

## D35. The types shim is guarded, not replaced

**DATE** 2026-09-01 | **STATUS** resolved. Amends D31 consequence 2

**CONTEXT.** D31 consequence 2 says "work item 9 replaces that one header with
`Fw/FPrimeBasicTypes.hpp` and no line of the core changes". Work item 9's brief adds a
requirement D31 did not anticipate: "The freestanding make target must STILL build and
pass all WI8 tests after the swap -- both worlds, one core."

**EVIDENCE.** The two cannot both hold with an unconditional include.
`Fw/FPrimeBasicTypes.hpp` at v4.3.0 includes `Fw/FPrimeBasicTypes.h`,
`Fw/Types/BasicTypes.hpp` and `config/FppConstantsAc.hpp`. The last is generated by the
F' build. It does not exist for `make -C flight test`, and cannot be made to exist
without giving the freestanding build an F' dependency, which is the whole thing D31
consequence 2 was protecting.

A second fact, found while making the change: `Types.hpp`'s `SizeType` and `I32` are
**declared and never used** anywhere in `flight/`. The surface being swapped is `U8`,
`U16`, `U32`, `U64`, `F32` and `F64` only.

**ALTERNATIVES.** Unconditional include and abandon the freestanding build. Vendor a stub
`FppConstantsAc.hpp`. A guarded shim.

**Against the first two.** Abandoning the freestanding build gives up the property work
item 8 was built around -- that the arithmetic can be verified without a framework, which
is D15's premise. A vendored stub is a second copy of a generated file, which drifts.

**CONSEQUENCE.**

1. **`Types.hpp` selects on `SENTINEL_FPRIME_TYPES`**, defined by the F' build only. F's
   names are aliased into `namespace Sentinel` so no other file changes.
2. **The zero-change claim holds where it matters and is restated precisely**: exactly
   one file differs between the two worlds, and it is the shim. Every other `.cpp` and
   `.hpp` under `flight/` is byte-identical in both builds, and a test asserts it.
3. **The shim asserts the float properties the dtype map needs**, so a platform that
   cannot provide them fails to compile instead of silently becoming a different
   detector: `sizeof(F32) == 4`, `sizeof(F64) == 8`, and `is_iec559` on both.

   **(!) AMENDED 2026-09-01, hours after this entry was written, while writing the
   shim.** This consequence originally read "The shim carries a hard `FW_HAS_F64`
   check", on the strength of `docs/MODEL_FILE.md` 9. **`FW_HAS_F64` does not exist in
   F' v4.3.0.** `F64` is defined unconditionally at `Fw/Types/BasicTypes.h:86`, and was
   at v4.2.2 too; the macro appears exactly once in the framework, in the documentation
   table work item 8 read at `docs/reference/numerical-types.md:35`. Writing
   `#if !FW_HAS_F64` would have expanded to `#if !0` and failed every build. The
   requirement is real and the mechanism was not, so the property is asserted directly.
   `docs/MODEL_FILE.md` 9 is amended in the same commit, and this is correction 13 in
   `docs/MODELS.md` 20.2. Recorded rather than quietly rewritten, because a decision
   that turns out to rest on a false premise is worth more visible than invisible.
4. **D31 consequence 2 is amended, not deleted.** Its intent -- one header, no core
   change -- is met; its literal wording is not achievable and this entry records why.

---

## D36. The model file is read whole, through `Os::File`, into a component-owned buffer

**DATE** 2026-09-01 | **STATUS** resolved

**CONTEXT.** `docs/MODEL_FILE.md` 8 says: "Work item 8's reader takes the whole buffer
from its caller because work item 8 has no filesystem -- `Os::File` and the chunked feed
are work item 9's, and the check order below is written so that a chunked reader can be
dropped in without changing which failure is reported first." Work item 9 therefore has
to decide whether to build the chunked feed now.

**EVIDENCE.** The chunked reader saves memory and costs verification. The whole-buffer
path costs 302,048 bytes of statically allocated storage -- 64 header, 320 channel map,
301,440 weight arena at `MAX_PARAMETERS = 75,360`, and 224 of parameter block -- against
a component that already carries a 312,112-byte `Detector`. The chunked path would save
most of that and would require re-establishing, by test, that all 16 load cases still
report the same status in the same order, which `flight/test/RefusalTests.cpp` currently
proves against a resident buffer. That re-verification is the larger part of the risk in
this work item, and it buys memory on a machine where memory is not yet the constraint --
Phase 4 is where the board is chosen and the envelope measured.

**ALTERNATIVES.** Build the chunked reader now. Have the topology own the buffer. Have
the component own it.

**Against the first two.** Building it now trades a measured, tested load path for an
untested one in the same work item that first puts the core under a framework, and
work item 9's brief does not ask for it. A topology-owned buffer couples every
deployment to Sentinel's internals and complicates the unit test for no gain.

**CONSEQUENCE.**

1. **The component owns the buffer and reads with `Os::File`.** `Detector::load(data,
   length)` is called unchanged, so the shim-swap proof stays clean and no core file is
   touched by this decision.
2. **The chunked reader is deferred, and the property that permits it is preserved.**
   The check order in `ModelFile.cpp` is not altered by this work item, so
   `docs/MODEL_FILE.md` 8's sentence remains true and the reader remains droppable-in.
3. **The cost is pre-registered, not discovered.** `docs/MODELS.md` 20.5 predicts
   `sizeof(Sentinel::Monitor)` at 624,528 bytes, of which 302,048 is this decision.

---

## D37. `baselines._rolling` computes its prefix sums in float32 and loses the statistic. The flight baseline transcribes the rule, not the implementation

**DATE** 2026-09-01 | **STATUS** resolved. The repair ran at work item 9.5 and
falsified a published claim; see the closing block of this entry

**CONTEXT.** Work item 9 must transcribe the `rstd` statistical baseline into flight C++
under work item 8's golden-vector discipline: seeded vectors, Python-versus-C++ match at
1e-5, committed. Before writing the transcription the Python source was measured against
independent implementations, as D15 requires -- portability is an assertion, not an
environment.

**EVIDENCE.** `baselines.py:44` reads
`cumulative = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])`.
The bundle's values are float32 (`bundle.py:169`) and normalisation is identity (D2), so
`np.cumsum` runs at **float32** and the promotion to float64 happens at the surrounding
`np.concatenate`, after the precision is already gone. `trailing_sum` then differences two
large float32 prefix sums to recover a small second moment -- catastrophic cancellation --
and the variance floor `np.maximum(second - mean * mean, 0.0)` at `baselines.py:55`
clamps the negative result to zero.

Measured on `N(1000, 3)`, float32, T = 8,000, C = 12, post-warm-up rows only:

```
  _rolling(float64 input)   vs  exact windowed recompute      1.8284e-08
  exact windowed recompute  vs  numpy.nanstd                  3.5489e-10
  _rolling(float32 input)   vs  all three                     7.6584e+00
```

True sigma is 3.0. At t = 5000, `_rolling` returns `[6.65, 0.00, 3.90, 0.00]` against a
true `[2.91, 3.28, 2.96, 2.82]`: two channels of four clamped to exactly zero. Three
implementations agree with each other to 1.8e-08 and disagree with the one in the tree by
7.66.

**It is not a stress-case artifact.** On this project's own offline fixture -- 39,992
steps, 7 channels, per-channel `|mean|/std` between 2.08 and 5.91 -- the float32 and
float64 forms differ by up to **4.9252e-03** on spreads of order 0.02 to 0.24, and the
float32 form yields **3,975/279,104** exact zeros against float64's **1,123/279,104**.
2,852 steps report zero spread because of arithmetic. `_rolling` also serves `mavg`,
whose exposure is two orders smaller: **2.2078e-04** on a residual scale of
**3.8092e-02**.

**ALTERNATIVES.** Transcribe `_rolling`'s float32 behaviour into flight. Fix
`baselines.py` inside work item 9. Transcribe the rule and scope the fix separately.

**Against the first two.** Transcribing the defect is impossible and undesirable: a
whole-array float32 prefix sum cannot be reproduced by a fixed-memory ring buffer, and
flying a statistic known to be wrong would violate Objective.md 11 rule 5's premise that
the flight code computes what it says it computes. Fixing `baselines.py` inside work item
9 moves published Phase 1 figures -- `docs/RESULTS.md` 2's floor of 0.250 and the 3/32 in
`docs/MODELS.md` 4's OBSERVED verdict -- and requires bucket operations, against a work
item whose budget is zero.

**CONSEQUENCE.**

1. **The flight Level 1 baseline implements the rule, in F64, pinned against
   `numpy.nanstd`.** It does not reproduce `_rolling`'s float32 accumulation. **The
   divergence is deliberate and this entry is its record**: the flight component and the
   harness's `rstd` are knowingly different numbers until the harness is fixed.
2. **`src/sentinel_models/baselines.py` is not touched by work item 9.**
3. **A golden-vector tier exists specifically to catch a regression**: tier B3
   (`docs/MODELS.md` 20.7) is the offset regime that exposes the cancellation, so an
   implementation that reintroduces float32 accumulation fails a test rather than
   passing quietly.
4. **The repair is scoped, dated and sequenced.** `docs/MODELS.md` 21 is the plan: a D8
   correctness fix to `_rolling`, a pinning test against `numpy.nanstd`, one re-score
   covering `m1-g8.9.10` and `m1-ss5` together, every affected figure republished as
   `new (was old, D37)` with old artifacts preserved, and a falsification stated in
   advance. It runs **after work item 9 and before work item 10**, because the figure at
   stake is the denominator of the project's headline comparison.
5. **Whether the spent held-back sets are re-scored is escalated, not decided.**
   `docs/MODELS.md` 21.5 argues both sides and recommends a targeted `rstd`/`mavg`-only
   re-score of `m2-ss1` and `m1-g3`; the decision belongs to the checkpoint review, and
   this entry will record it when it is made.
6. **D8's exemption is what makes the repair legitimate.** `docs/HARNESS.md`: the harness
   is closed to scope changes, never to correctness fixes. This is a correctness fix.

**(!) RESOLVED 2026-09-01, work item 9.5, and the outcome falsified a published claim.**
`baselines._rolling` now promotes to float64 before squaring and accumulates with
`dtype=np.float64`; the gap against the flight rule closes from 7.6584e+00 to 1.8284e-08
and the spurious zeros disappear. Re-scored at a cost of 41 Class B and 3 Class A:

```
  m1-g8.9.10  rstd   F0.5   0.250 -> 0.676     MVGS   3/32 -> 25/32
                     recall  3/46 -> 34/46     lead  -1,512 -> +0.0
  m1-ss5      rstd   F0.5    undefined -> 0.663    MVGS  0/31 -> 25/31
  m2-ss1      rstd   nominal-step FA  17.30% -> 0.003%   rare  84/424 -> 22/424
  m1-g3       rstd   F0.5   0.029 -> 0.351     MVGS  3/10 -> 8/10
```

**The falsification pre-registered in `docs/MODELS.md` 21.4 fired.** It named 22/32 -- the
GRU's headline cell -- as the line past which "the project's central claim ... is in
serious question and this document says so in those words". The corrected floor reached
**25/32**. A per-channel statistic finds twenty-five of the thirty-two cross-channel
events, not three; the three was an artifact of this defect, and every restatement of it
is corrected in place with the old figure beside it.

**What that does and does not overturn.** The gate metric was fixed as event-wise F0.5,
never bare recall, by D3 and before any of this was measured: on it `gru-quantile` still
clears the corrected floor 0.804 to 0.676, reaching comparable recall at a third of the
alarm rate with precision 0.885 against 0.661. D28's architecture gate never involved
`rstd`; D25 and D29 are untouched, and D29's evidence is cleaner, because the floor it is
compared against on Mission 2 now works. What is overturned is the ratio -- three against
twenty-eight -- that this repository had used as its headline, and W1's reasoning about
which way the threshold would move, which was exactly backwards and is recorded as Wrong
in 21.9.

**And a limit the fix does not remove.** `sqrt(S2/n - (S1/n)^2)` loses accuracy as the
square of `|mean|/sigma` at any precision: negligible at the ratios ESA-ADB's min-max
scaling produces, total at 1e8, and silently zero at 1e9. `baseline_reference.py` and
therefore `flight/src/Baseline.cpp` share it exactly.
`tests/test_rolling_precision.py` pins the boundary. Making the form unconditionally
stable is an algorithm change that would move the flight golden vectors, and is reported
rather than taken.

---

## D38. `gru-quantile` catches a strict subset of the corrected floor's events on Mission 1

**DATE** 2026-09-01 | **STATUS** resolved as a finding. It changes no selection and
retires one argument

**CONTEXT.** Work item 9.5 corrected `baselines._rolling` and the floor rose from
F0.5 0.250 to 0.676 and from 3/32 to 25/32 headline-cell events (D37). Work item 9.6
was commissioned to audit that result before the restatement was allowed to stand,
and pre-registered fourteen predictions in `docs/MODELS.md` 22 before computing
anything. L9 named the sharpest one: **if the forecaster catches no event the floor
misses, the thesis has no per-event evidence on this set.**

**EVIDENCE.** One bundle load, 15 Class B, cached weights, nothing refitted. The audit
reproduces every published scorecard count exactly -- 27/46, 34/46, 22/32, 25/32 --
before reporting anything new.

```
  m1-g8.9.10   both 27   only-GRU 0   only-rstd 7   neither 12
  m1-ss5       both 26   only-GRU 0   only-rstd 8   neither  8
```

**Not one event, on either set, in any taxonomy cell, is caught by `gru-quantile`
and missed by the corrected `rstd`.** The seven the floor catches and the
forecaster misses have footprints of 1, 1, 1, 1, 24, 28 and 54 steps, with `rstd`
reaches of 5.2 to 6.4 against the GRU's 0.55 to 0.62.

**And the floor is not leaking**, which is the first thing a result like this must
survive. L1 to L4 all hold: perturbing a future sample changes rows before `t` by
exactly 0.0 with a working control; `fit` sees only the masked training window
through the same `harness.py:144` line every detector uses; the fallback scale is
unreachable; and `RollingStd.threshold_from` **is** `Detector.threshold_from`, the
same function object the GRU's calibration calls. The catches are sustained rather
than stray: excluding events whose footprint is 1, **0 of 25** are single-step and
the alarm covers a median 95% of the footprint.

**ALTERNATIVES.** Treat the nesting as an artifact of the corrected floor.
Re-open the architecture gate. Record it as a finding and change nothing.

**Against the first two.** The nesting is not an artifact: the audit reproduces the
harness's own counts and the four named leaks are all refuted. Re-opening the gate
would be answering the wrong question -- D28 compared LSTM, GRU and TCN against each
other on event-wise F0.5 and never involved `rstd`, and nothing in this audit moves
any forecaster figure or their ordering.

**CONSEQUENCE.**

1. **`gru-quantile` remains the selected architecture and the flying detector.**
   D28, D25 and D29 are untouched. On D3's gate metric -- event-wise F0.5, fixed
   before any of this -- it clears the corrected floor 0.804 to 0.676, with
   precision 139/157 against 84/127 and 0.00066 alarms per thousand nominal steps
   against 0.0024.
2. **The per-event argument for the forecaster is withdrawn on Mission 1.** It
   catches nothing the floor does not. Its case rests entirely on the *quality* of
   the same catches -- fewer alarms, higher precision -- which is a real and
   measured difference and is what the gate metric was chosen to capture. Every
   statement in this repository that the forecaster sees events a per-channel
   statistic cannot is withdrawn for these two sets.
3. **What the two detectors are is now measured rather than assumed**: ordered, not
   complementary, in exactly the sense D25 established for `lstm-whitened` and
   `lstm-quantile`. A union buys nothing, and none is proposed.
4. **This does not generalise beyond Mission 1 without measurement.** `m2-ss1` and
   `m1-g3` carry no per-event overlap analysis, and none is run here.

---

## D39. The headline cell is not the contextual class, and the contextual class is caught by nothing

**DATE** 2026-09-01 | **STATUS** resolved as a finding. It is the more serious of the two

**CONTEXT.** Objective.md 2.3 motivates this entire project with Hundman's finding
that 41% of real spacecraft anomalies are contextual -- every channel inside its
limits while the combination or the trajectory is wrong -- and states that a
per-channel limit check cannot see them. The project has treated the 32
`Multivariate/Global/Subsequence` events on `m1-g8.9.10` as that class since work
item 4. **It was never measured.** `docs/MODELS.md` 22 called it Claim B and
pre-registered L13 and L14 to test it.

**EVIDENCE.** Per event, whether any watched channel leaves its own training
envelope at any step inside the event span. The envelope is the 0.1/99.9 quantile
pair over the fold's fitting window with `train_mask` applied -- fitting data only,
no test sample and no test label -- the construction `scripts/envelope_proxy.py`
already uses. Definition fixed in 22.2 before the numbers existed.

```
  truly contextual, no channel outside its own envelope :  3/32   (6/32 by hard min/max)
  at least one channel leaves its envelope              : 29/32
```

Predicted 8 to 16. Measured 3.

**And neither detector catches any of the three.**

```
  id_121  footprint 257   GRU reach 0.110   rstd reach 0.544   channels breaching 0
  id_153  footprint  10   GRU reach 0.178   rstd reach 0.608   channels breaching 0
  id_157  footprint  51   GRU reach 0.238   rstd reach 0.572   channels breaching 0
```

Neither comes within half its threshold. `rstd` scores 0/3; `gru-quantile` scores
0/3. L14's 86-point gap between contextual and breaching recall is real, and it is
not evidence for the forecaster, because the forecaster's gap runs 76 points the
same way.

**A related fact, from cached forensics at zero cost.** None of the 32 events is
single-channel to the watched view: 21 touch all 12 channels and the fewest touches
5. So the events are genuinely multivariate; what they are not is *contextual*.
Multi-channel and contextual are different claims and only the second is the
project's.

**Stated carefully, because an envelope is not a limit.** A real RED or YELLOW limit
sits *outside* a channel's historical envelope, so breaching the envelope does not
establish that a limit would have tripped; the implication runs only the other way.
What is established is that **29 of 32 are not in the class that is provably
invisible to limit checking**, and that the three which are, nothing here catches.

**CONSEQUENCE.**

1. **Objective.md 2.3's motivating claim is not tested by `m1-g8.9.10`'s headline
   cell.** The cell is 29/32 envelope-breaching. Whatever this project has
   demonstrated on this set, it is not that it catches anomalies a limit check
   cannot see.
2. **The three events that are provably in that class are missed by both
   detectors**, at reaches of 0.11 to 0.61. On the evidence available, this project
   has **no** measured instance of catching a contextual anomaly.
3. **No detector, threshold or decision layer changes on this finding.** It is a
   statement about what the evidence supports, not about what the component should
   do, and acting on 3 events would be acting on n=3.
4. **The honest version of the project's claim, for every document that states
   it**: this is a reusable flight component that packages a published detection
   method, measured against a floor on a set whose headline cell is largely
   limit-visible. The early-warning claim was already retired (Objective.md 1.1);
   the contextual-class claim is now in the same position and is retired by this
   entry until a set exists that tests it.
5. **What would settle it** is a scoring set whose events are selected for the
   contextual property rather than assumed to have it. ESA-ADB may not contain one
   at n >= 20; `docs/HARNESS.md`'s own rule would stamp anything smaller
   UNDERPOWERED. That is a Phase 3 question and is not answered here.
