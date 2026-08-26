# The evaluation harness

The referee, built before the players. It loads ESA-ADB labels, streams telemetry
from R2, and scores **any** candidate detector on identical terms, so that
LSTM vs GRU vs TCN becomes a table of numbers rather than an argument.

**The single most important thing in this document is section 7**, written before
the first baseline ran.

---

## 1. What it measures

Every recall, precision and false-alarm figure is reported as **k/n** with the
resolution `1/n` beside it, and anything with `n < 20` is stamped UNDERPOWERED.
A recall of 0.36 over eleven events takes twelve possible values; printing only
the rate invites a reader to treat a one-event difference as a finding.

| Metric | Definition | Status |
|---|---|---|
| **Event-wise F0.5** | From event-wise precision and recall, weighting precision twice | **GATE** |
| **Rare-event false alarms** | Fraction of rare nominal events the detector alarmed on | **ADOPTION** |
| **Lead time** | Timesteps between the first attributable alarm and the labelled event start | **GATE** |
| Contextual / headline-cell / point recall | Recall over `Multivariate`, over `Multivariate/Global/Subsequence`, over `Length == Point` | components |
| VUS-PR | Threshold-free, buffer-integrated | required |
| Alarms per 1,000 nominal **timesteps** | Never per hour -- see section 4 | supporting |
| Point-adjusted F1 | **Quarantined.** `--diagnostics` only, never in RESULTS.md | diagnostic |

**Bare recall is never reported alone, anywhere.** Recall is satisfiable by
carpet-bombing: the trivial baseline reached 29/31 headline-cell recall by firing
5,535 alarms for 42 events. A detector that fires constantly detects everything
and is worth nothing. So F0.5 leads every block, recall and precision appear
beneath it as its components, and the per-cell recalls sit under a caption naming
the precision they must be read against. `tests/test_gate_metric.py` enforces it.

### Lead time is a gate metric, and a negative one disqualifies

**A configuration whose median lead time is negative is not a candidate for the
flight configuration, whatever its other numbers.** A rule, not a preference.

Objective.md 2 states the purpose: detection happens *earlier*, and the
capability works through ground-contact gaps. A detector that alarms 122
timesteps after an event began is an accurate historian, not an early-warning
system, and the component being built is not a historian.

The rule exists because the numbers came close to hiding it. `lstm-quantile`
holds the best event-wise F0.5 in the project **and** the best false-alarm rate --
0.421 and 1/48 against `lstm-telemanom`'s 0.269 and 22/48 -- and every one of its
six detections lands *after* the event started, on alarm ranges five times wider.
On F0.5 and adoption alone it is the better detector. On what the component is
for, it has failed.

So lead time joins F0.5 and the rare-event rate as a headline number. It appears
on every scorecard, in every `docs/RESULTS.md` row, and as a primary column in
every sweep rather than a footnote. Work items 5 and 6 report it for the GRU and
the TCN.

**Several precision mechanisms buy their gains by waiting longer**, and waiting
is the one thing this project cannot spend freely. The persistence sweep already
showed the shape: lead time fell almost one timestep per unit of N, from +26 at
N=1 to -37 at N=60, while F0.5 moved barely at all. Every layer from here reports
its lead-time cost beside its precision gain, and a layer that buys precision
with lateness justifies the trade rather than being scored on F0.5 alone.

The **rare-event false-alarm rate** appears on every scorecard, every run,
without exception. It is what actually condemned the baseline -- 39/48 (0.812) --
and it is what decides whether a mission would fly this at all: a detector
alarming at 81% of commanded manoeuvres is muted within a week in orbit, and a
muted detector is worse than none (Objective.md 11, rule 2).

### No number enters a document that was not read from an artifact

**A standing rule.** If a value is not to hand, the cell is left empty and said to
be empty. A plausible number is not a placeholder; it is a claim, and it is
indistinguishable from a measured one once it is written down.

The rule exists because it was broken. Filling the lead-time column of
`docs/RESULTS.md`, two baseline figures were written from expectation rather than
read from the run artifacts -- `rstd` as `+2` and `mavg` as `+18`. The measured
values were `-1,512` and `0`. `rstd`'s was wrong by three orders of magnitude and
in the opposite direction, and had it stood it would have concealed the single
most consequential finding in that document: that the floor the whole work item
was measured against is not an early-warning detector.

It was caught by reading the artifacts, corrected before the commit, and reported
rather than buried. That is the discipline working. But it happened *inside the
document arguing that measurement beats assumption*, which is why the rule is
written here rather than left as a lesson someone is assumed to have learnt.

**Contextual anomaly** is operationalised as `Dimensionality == "Multivariate"`:
an event manifesting across more than one channel at once. Objective.md 2.4
defines it as every channel being individually legal while the combination is
wrong; the ESA-ADB encoding is the nearest thing the labels carry. That
equivalence is an interpretation and every number derived from it inherits it.

**Four categories, not two.** `Anomaly` is the positive class. `Rare Event` --
manoeuvres, resets, calibrations -- is **nominal**, and firing on one is a false
alarm. `Communication Gap` and `Invalid Segment` are **neither**: there is no
measurement to be right or wrong about, so they are masked out of every metric
rather than counted as nominal. Counting a detector "correctly quiet" across 401
gaps and 156 invalid segments would hand it free specificity.

## 2. Tasks

| Task | Role | What it carries |
|---|---|---|
| `m1-g8.9.10` | **GATE -- primary recall** | 12 channels, groups 8+9+10, one spacecraft. 46 anomalies, 32 headline cell, **11/11 point** |
| `m1-ss5` | point-anomaly coverage, fast iteration | the 6 channels of group 8. Demoted, retained, always reported |
| `m1-ss5-cv3` | point-anomaly coverage, folded | same channels, forward-chaining |
| `m2-ss1` | **GATE -- adoption** | mission2, 12 channels, ~600 rare events. **Recall disabled at the task level** |
| `synthetic` | fixture | generated data, zero R2 operations |

### The primary set was promoted post-hoc. Say so.

> **m1-g8.9.10 was promoted to primary after the baseline run, on footprint
> evidence measured post-hoc. The prior primary m1-ss5 is retained and reported
> alongside it in every result.**

We changed the evaluation set after seeing a result we did not like. However
sound the reasoning, that is externally indistinguishable from cherry-picking,
and the claim that the choice was made a priori is permanently forfeit. A
reviewer who spots this unaided discounts everything we publish; one who reads us
stating it does not.

**Reporting both sets forever is the enforcement.** You cannot cherry-pick if you
never discard anything. `m1-ss5` is a strict subset of `m1-g8.9.10`, so one
twelve-channel load scores both -- 15 Class B rather than the 25 two runs would
cost -- and both share the primary's grid and split, so the pair differs in
channels alone. A `--only` run is stamped `"partial": true` and is barred from
`docs/RESULTS.md`, which makes single-set publication structurally impossible
rather than merely discouraged.

### Two regimes, not a winner and a loser

| Set | Regime | Expectation |
|---|---|---|
| `m1-ss5` | fast, spiky anomalies | a per-channel moving average handles these |
| `m1-g8.9.10` | slow, drifting anomalies | where a cross-channel forecaster should win |

A real spacecraft produces both. If our forecaster loses to a moving average on
the spiky regime, that is an **applicability boundary** worth knowing and
publishing, not a failure to bury. The moving average has no answer at all to the
slow case -- which is the 41% that limit checking misses, and the reason the
project exists.

### The detector is regime-agnostic

Nothing above says we are building a slow-drift detector that group 8 broke. The
model forecasts the next values and compares against reality, and **any**
divergence grows the error -- fast or slow, it does not distinguish. The
persistence filter (Objective.md 7) suppresses single-sample noise; it does not
bias toward slow events, and a genuine fast event lasting more than N cycles is
caught normally.

Group 8's problem is that it contains almost no slow-drift events, so it **cannot
demonstrate** the advantage over per-channel methods. That is a property of the
test set, not a limitation of the detector. Stated plainly here because otherwise
the promotion of `m1-g8.9.10` reads as tuning the test to the model rather than
choosing a set that can show the difference at all.

### The limitation, stated plainly

**Recall rests on one spacecraft.** There is no second viable recall set in
ESA-ADB. `m2-ss1` deduplicates from a naive 49 anomalies to **18**, of which one
to three fall test-side; Mission3 has 8 anomalies in total and only 4 of its 48
channels are numeric, with every anomaly-bearing group entirely categorical. So
the roles are split -- Mission1 carries recall, Mission2 carries the adoption
number -- and this caveat is written into every scorecard and every `RESULTS.md`
row, because results tables get screenshotted and travel without the document
that qualifies them.

## 3. Splits

Strictly forward in time: a model is never fitted on data that follows what it is
scored on. Training is **normal-only** (Objective.md 6.1) -- annotated anomalies
are removed from the fitting window and the amount removed is reported.

Where the boundary goes is not a detail. Measured on `m1-ss5`:

```
  split   boundary    | train yrs  train rare |  test anom  MVGS  point  rare
  --------------------------------------------------------------------------
  25/75   2003-07-02  |    3.5        15      |     42       31     11     48
  50/50   2007-01-01  |    7.0        27      |     29       21      8     36
```

25/75 costs 3.5 training years -- still ~3.7M timesteps -- and buys 45% more
anomalies and **all 11 point anomalies instead of 8**. Forward chaining reaches
the same denominators while each fold trains on more history than the last, which
yields a data-sufficiency curve for free.

A split leaving fewer than five scorable anomalies raises `SplitTooThin` rather
than reporting a confident, meaningless number. `mission1/subsystem_3` at a 75%
boundary leaves *zero*.

## 4. Facts that would silently corrupt results

**Timesteps, never hours.** ESA-ADB timestamps are anonymised mission time,
scaled by an undisclosed factor greater than 1 and shifted to start 2000-01-01.
They are not UTC and no elapsed duration derived from them is a real duration.
The in-memory column is named `t_anon` so there is no `.timestamp` to reach for
by habit. Time-to-limit estimation is a Phase 3 deliverable on the F' Ref
deployment, where the clock and the dictionary limits are real.

**Zero-order hold, with a staleness guard.** ESA prescribes ZOH and forbids
linear or Fourier interpolation. A plain forward fill holds a value across a
multi-day outage and produces a flat line indistinguishable from a healthy quiet
subsystem; the grid therefore refuses any point whose backing sample is older
than `max_hold` (default ten steps) and folds it into the unscorable mask.

**Normalisation is identity** -- Objective.md 14.8, RESOLVED. ESA min-max scaled
within each channel group, so amplitude ratios between related channels survive.
Cross-group spanning is fine: those offsets are fixed, invertible and
uninformative, and any model absorbs them. **Per-channel rescaling is not**: it
erases the fact that one channel normally moves ten times more than its
group-mate, and no model can recover it. Enforced by a chokepoint and by
`tests/test_no_per_channel_scaler.py`.

**VUS-PR buffer widths come from the 75th percentile, not the median.**

> **CLOSED, and no scored result was ever computed under the defect.** The
> collapse was found and fixed during harness development, in commit `1f00a43`,
> which is the same commit the first baseline run records as its own provenance --
> so the fix predates every number this project has published. That run's
> artifacts record the buffers it actually used: `[0, 2149, 4298, 6446, 8595]` on
> `m1-g8.9.10` and `[0, 616, 1233, 1850, 2466]` on `m1-ss5`, neither degenerate.
> Re-measuring `m1-ss5` after the `Bundle.subset` correction moved `rstd`'s VUS-PR
> by 4e-6 relative, which is consistent with forty timesteps leaving the scorable
> mask and inconsistent with a metric that had been degenerate. Recorded this
> plainly because the paragraph below describes the defect in the past tense, and
> a document that makes a closed issue look open costs somebody a day.

Event footprints are extremely skewed -- on `m1-ss5` the median is 1 timestep
while the mean is 2,909 and the maximum 81,717 -- so a median-derived sweep
collapsed to `[0, 1]` and VUS silently stopped integrating any volume, becoming
plain AUC-PR on that set and not on the other. The metric itself is sound: checked against
detectors of graded quality it is monotone in signal quality and separates
carpet-bombing (0.388) from silence (0.245) from sniping (1.000). Only the
schedule was wrong, and `MIN_BUFFER_STEPS` now floors it so it cannot collapse
again.

**Sharded channels carry an empty parent checksum.** Four real channels have
`key: null` *and* `sha256: ""`; the digests live per shard. Integrity is verified
per object, and the reader was written against `mission1/channel_74` first.

## 5. Operations

| Run | Class B | Class A |
|---|---|---|
| Any Mission1 task, **any number of detectors, all folds** | 10 | 1 (ledger) |
| `m2-ss1` (12 channels) | 16 | 1 |
| Development and the whole test suite | **0** | **0** |

Nothing is cached locally, so a run's cost is its reads. `bundle.load` fetches
once and every detector and fold scores the resident arrays -- three detectors
cost ten operations, not thirty. All development runs against a generated fixture
that never touches R2.

### Rule 1, stated precisely

> **No telemetry, no parquet, no cached datasets on local disk. Model weights and
> run artifacts under `runs/` are outputs, not data, and may persist. Gitignored,
> never committed.**

The earlier wording was "the Mac is a pipe, not a store", which is the right
instinct and the wrong rule: it reads as forbidding a trained model to survive
the process that produced it. Rule 1 exists to stop this machine becoming a data
store. A 358 KiB file of learned parameters is not telemetry, is not parquet, and
is not a cached dataset -- it is an output, the same category as the scorecards
already written under `runs/`.

The failure mode that would have mattered is a stale checkpoint being scored by
accident. It is closed by construction rather than by care: the weight cache key
is a content digest over the hyperparameters, the channel set, the fold window
and a strided sample of the data itself, so weights fitted on anything else
cannot match.

**Any published result must still be reproducible from cold.** `--no-cache`
refuses every cached weight and refits, and nothing enters `docs/RESULTS.md`
until it has been verified that way. Cache for iteration; verify from scratch.

`tests/test_no_local_persistence.py` widens the `runs/` exemption to cover
weights and **tightens** the dataset ban at the same time: parquet, pickles and
archives are now refused everywhere in the tree, `runs/` included, where before
that directory was skipped entirely.

Enforcement is at the point of spending: the per-run tripwire raises at 1,000 and
a human may acknowledge it; the monthly ceiling raises at 50,000 and has no
override. The ledger is read-modify-write, never reset.

## 6. Layout

```
src/sentinel_eval/     the referee   -- never imports a model
src/sentinel_models/   the players   -- items 4-6 land here
src/sentinel_export/   Phase 2 model.bin writer (placeholder)
```

The dependency runs one way, enforced by `tests/test_layering.py`, so LSTM, GRU
and TCN are added without touching a line of scoring code. `cli.py` is the only
module that knows about both.

One deviation from the agreed tree: the scoring loop lives in `harness.py` rather
than inside `cli.py`, so that tests can call it directly and the CLI stays a
composition root.

## 5a. What "the harness is closed" means

The harness is **closed to scope changes**. No new features, no model work
bleeding into referee work, no metric added because a result would look better
with it. That closure is what makes the architecture gate a table of numbers:
LSTM, GRU and TCN are scored by code that stopped moving before any of them
existed.

**It is never closed to correctness fixes.** A referee that computes the wrong
number is not a fixed reference point, it is a wrong one, and freezing it does
not make its output true. The distinction is the whole of the rule:

```
  scope change      a new capability, a new metric, a new option        REFUSED
  correctness fix   the existing contract, computed wrongly             REQUIRED
```

Two obligations come with a correctness fix, and neither is optional.

**Escalate before making it.** The person who finds the defect is usually the
person whose work it is blocking, which is exactly the position in which "fix it
and move on" looks reasonable and reads, later, as editing the referee to suit
the player. So it is reported, and someone else decides.

**Anything that moves a published number gets both numbers recorded.** Not the
corrected figure with the old one deleted -- both, side by side, with the defect
and the fix described. Same discipline as section 7 below, and for the same
reason: if the numbers land within noise, that is *evidence* the earlier results
were not distorted, and you only have that evidence if you kept them. A reviewer
who finds a correctness fix in the git history with only the corrected numbers
visible has to wonder what else was quietly cleaned up.

### The register of authorised additions

A scope change that is authorised is still a scope change, so each one is
recorded here with its date and its reason. An addition nobody can see is an
addition nobody re-examines.

| Date | Addition | Reason | Conditions met |
|---|---|---|---|
| 2026-08-25 | **Lead-time metric** -- timesteps between first attributable alarm and labelled event start | Objective.md 4's headline claim is early warning, and no metric had ever measured *how early*. It is also a genuine architecture discriminator where 46 events cannot separate LSTM, GRU and TCN on recall alone | Additive only; absent from a scorecard unless computed, so existing artifacts are byte-identical; tests carry hand-computed values |
| 2026-08-25 | **`m1-g3` task** -- Mission1 group 3, 8 channels | A held-back recall set nominated before decision-layer tuning began. `m1-ss5` cannot serve: it is a strict subset of the primary carrying the same events | Additive only; no existing task definition changed |
| 2026-08-25 | **`--no-cache` flag** | Weight persistence is now permitted for iteration, so a published result needs a way to be reproduced from cold | Additive only; default behaviour unchanged |

### The first case, recorded here

`Bundle.subset` rebuilt its ground truth from the labels alone and never read
`self.valid`, so the unobserved-timestep fold-in that `bundle.load` performs was
lost on every subset. `m1-ss5` is scored as a subset of the twelve-channel load,
so ~1,622 unobserved timesteps were counted as scorable nominal time -- the free
specificity section 1 explicitly refuses for gaps and invalid segments.

It survived because **the trivial baselines are NaN-tolerant by construction**
(`nan_to_num`, `nanstd`), so nothing had ever asked the mask to be right. The
first detector that could not tolerate a NaN found it immediately. Both sets of
numbers are in `docs/RESULTS.md`.

## 6a. How channel sets are grouped, and why

This is the defence against the most dangerous failure mode in the whole method:
**learning a relationship that is coincidence rather than physics.** Across 14
years and billions of samples, unrelated channels will correlate by chance. A
model that learns such a pair raises a false alarm when it "breaks" -- damaging
precisely the adoption number in section 1. Four defences, in order of
importance.

**1. Grouping is engineering, not statistics.** We do not hand a model 224
channels and let it hunt for whatever correlates. Sets are drawn from ESA's own
`Subsystem` and `Group` columns in `channels.parquet` -- labels written by the
engineers who operated these spacecraft. The model can only model relationships
inside a physically-grouped set, so it cannot invent a link between a reaction
wheel and a radio: they are never in the same model.

**2. A deploying mission has this more directly.** Mission engineers designed the
spacecraft and know the battery, charger and thermistor are one system. That is
domain knowledge, not statistical discovery. We use ESA's groups as a proxy only
because ESA-ADB is anonymised -- no channel names, no units -- so physical
selection is impossible on it. A benchmark limitation, not a product one.

**3. Volume attenuates coincidence.** A spurious correlation over 100 samples is
common. Holding across thousands of orbits, eclipse cycles and seasons is much
less likely. Not a guarantee -- a reason training spans years rather than days.

**4. Every warning names its cause, and a human vets it.** Output is
"BattTemp and ChargeCurrent decoupled at 14:32", not an opaque score, so an
operator evaluates the claim in seconds. And Sentinel commands nothing
(Objective.md 11, rules 3 and 4), so a spurious warning costs attention, never a
spacecraft. This failure mode is exactly what those two rules exist to contain.

## 6b. Benchmark selection is not deployment selection

A real mission cannot do what we just did. It has **no labelled anomalies** --
its spacecraft has not failed yet -- so there is no footprint distribution to
measure and the method used in section 7 is simply unavailable to it.

A mission selects on:

1. **Physics.** Power and thermal first: current to temperature holds in vacuum,
   attitude dynamics do not (Objective.md 10.2, fix 2).
2. **Data sufficiency.** The training toolkit grades pre-launch data and disables
   subsystems it cannot cover (fix 1).
3. **Criticality.** Which subsystem failing ends the mission.

None of these is "which channels make our numbers look good" -- there are no
numbers to look at. Two different processes, forced by two different constraints.
Stated here so nobody reads our benchmark method as the recommended deployment
method.

## 7. The pre-registration, and how it failed

A pre-registration that can be quietly replaced is not a pre-registration. The
audit trail is the asset, so the original prediction is preserved below with what
actually happened, rather than tidied away and rewritten.

### PREDICTED (before the first run)

Wu & Keogh (IEEE TKDE 2023) showed a moving average with a standard-deviation
threshold achieves state-of-the-art on SMAP/MSL, which is much of why
Objective.md 9.2 demoted that benchmark. The registered prediction was:

| Metric | Prediction |
|---|---|
| **Headline-cell / contextual recall** | **Near-floor** -- blind to cross-channel structure by construction |
| Point recall | May legitimately be decent -- spikes are a moving average's home turf |
| Rare-event false alarms | Expect bad -- a commanded manoeuvre looks like a spike |

Stop-and-report trigger: the baseline scores well on headline-cell recall.

### OBSERVED

The trigger fired. `mavg` scored **29/31 (0.935)** on `m1-ss5` and **25/32
(0.781)** on `m1-g8.9.10`. Four checks located the cause:

* a random detector reached only 10/31 at **twice** mavg's alarm budget, so
  existence-recall was not degenerate and the detection was real;
* raw label spans confirmed the rasterisation was correct;
* `selftest` and 123 tests confirmed the harness arithmetic;
* the surrounding numbers were dismal -- precision 0.249, rare-event false alarms
  39/48, VUS-PR at the base rate.

### ROOT CAUSE -- two errors, both ours, both in the measurement design

1. **The prediction was framed on the wrong quantity.** Recall alone is
   satisfiable by carpet-bombing; mavg fired 5,535 alarms for 42 events. The
   discriminating number is F0.5, which put mavg at 0.028 against `rstd`'s 0.250.
2. **`m1-ss5` selects for sub-grid-cell event footprint.** Group 8 is the only
   channel set in mission1 where headline-cell events register as spikes -- 18 of
   38, against 0 for every other group. Groups 9 and 10 see the *same events* as
   multi-hour subsequences.

```
  set                nch  MVGS  <1 cell   median      p75
  ss5 = group 8        6    38       18       30    3,056   <-- outlier
  groups 8+9+10       12    40        0    1,951    8,828
  group 3              8    14        0    3,594    5,776
  ALL mission1        76    47        0    1,749   11,331
```

The original selection was made on point-anomaly coverage, which was correct as
far as it went. Event footprint was the gap.

### CORRECTED prediction, on F0.5

| Metric | Prediction |
|---|---|
| **Event-wise F0.5 on `m1-g8.9.10`** | **Low.** A per-channel method buys recall only with alarms, and F0.5 weights precision twice |
| Point recall | Still expected decent -- unchanged, and it was: 9/11 |
| Rare-event false alarms | Still expected bad |
| `m1-ss5` versus `m1-g8.9.10` | mavg should look *better* on `m1-ss5`, because that set is the spiky regime it suits |

Stop-and-report trigger, restated: **a trivial baseline achieving competitive
event-wise F0.5 on `m1-g8.9.10`.** Recall alone no longer triggers anything,
because recall alone no longer means anything.

### What this cost, and what it bought

One day and 40 Class B operations, against three trained models scored on a
metric that could not distinguish them. The harness was sound throughout -- oracle
1.000, silent 0, 123 tests green -- and so was the data. Both errors were in the
measurement design, which is the good outcome and the reason to pre-register.

## 8. Using it

```bash
python -m sentinel_eval list-tasks
python -m sentinel_eval selftest                       # 0 operations
python -m sentinel_eval describe m1-ss5
python -m sentinel_eval run m1-ss5-cv3 --detector mavg --detector rstd
```

`selftest` is the harness checking its own arithmetic: a detector handed the
answers must score 1.0 on every metric, and one that never fires must score 0
recall with *undefined* precision rather than zero. A referee never checked
against its own extremes is untested machinery.
