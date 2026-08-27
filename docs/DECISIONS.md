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
