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
residuals against an **alarm budget in timesteps** -- what a mission's operators
can act on (`docs/RESEARCH.md`, ISA-18.2 and EEMUA 191) -- never against a
detection score. `p` is bounded below the measured contamination rate: the median
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

**CONSEQUENCE.** Held at 100 while D20's threshold is measured, so the two do not
confound. Decided afterwards, with the threshold frozen, against both reaches at
once. **Also carried into work items 5 and 6**: `error_buffer` is why the
persistence filter measured as subsumed on this detector (Objective.md 7.1), so
the GRU and the TCN inherit all three couplings unexamined unless this is settled
first.
