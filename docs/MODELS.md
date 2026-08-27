# The models

The players, scored by the referee in `docs/HARNESS.md`. Work items 4, 5 and 6
add an LSTM, a GRU and a TCN here; the harness never changes, so the
architecture gate is a table of numbers rather than an argument.

**Read `docs/HARNESS.md` first**, in particular section 7, which records a
prediction that failed and why. This document follows the same discipline.

---

## 1. What work item 4 built, and what it did not

telemanom (Hundman et al., KDD 2018) is the reference method Objective.md
section 3 names. What is implemented here is **telemanom's detection method with
a multivariate forecaster**: the LSTM forecast, the nonparametric dynamic
threshold and the pruning step, driven by one model over the whole channel set
rather than one univariate model per channel.

That distinction is not a footnote and the Phase 1 gate wording in Objective.md
section 12 was amended to carry it. A per-channel model cannot express
cross-channel structure, which is the project's entire claim (Objective.md 2.4),
and items 5 and 6 have to differ from this by architecture alone or the gate
compares data pipelines instead of architectures.

### (!) These are telemanom-minus-commands, not telemanom

**Correction, recorded before the ledger because it qualifies every number below.**

Hundman et al. feed the LSTM two things: "prior telemetry values for a given channel **and
encoded command information sent to the spacecraft**", where the module a command was issued to
and whether it was sent or received are one-hot encoded into each timestep. Their Figure 3 shows
that encoding letting the model predict a commanded event so that it is *not* flagged.

We fed telemetry only. The 821 telecommand series ingested in work item 2 have never been wired
in, and deviation 2 below recorded them as "a future item". **That was wrong: it is the missing
third of the method.**

The consequence is specific and it lands on the number that matters most. Our false alarms are on
*rare nominal events* -- commanded manoeuvres, resets, calibrations -- and we withheld the one
input that makes them predictable, then measured the detector's failure to predict them.

> **22/48 is not telemanom's rare-event false-alarm rate. It is
> telemanom-minus-commands.** Every comparison drawn in this repository against published
> telemanom figures carries that caveat until the commands-on / commands-off ablation has been
> run and reported.

The ablation is the next piece of work. Until it exists, no figure here should be read as
evidence about the published method's false-alarm behaviour -- only about ours without a third of
its inputs.

### The deviation ledger

Every departure from telemanom's published configuration, with its reason. A
claim of faithfulness that cannot be audited is worth nothing.

| # | Published telemanom | Here | Why |
|---|---|---|---|
| 1 | One univariate LSTM per channel | **One multivariate LSTM over the channel set**, C in / C out | Objective.md 4.3: "given the recent history of all watched channels". **The largest deviation** -- the section 12 gate names it |
| 2 | Telecommand signals as one-hot model inputs | **Absent -- and this is a defect, not a deviation** | Originally recorded as "a future item" because `Bundle` hands a detector the channel matrix only. That reasoning was about our plumbing, not about the method: commands are a third of telemanom's input and their absence is what our worst number measures. Being corrected -- see the note above the ledger |
| 3 | 35 epochs over a channel's whole training set (~2-8k steps) | Up to 35 epochs, sequence budget **scaled to the fold**, one sequence per 180 usable steps | Folds hold 3.6M to 10.8M usable steps. See section 3 -- a fixed budget would destroy the data-sufficiency curve |
| 4 | Random 20% validation split | **Chronological last 20%** of usable steps | `docs/HARNESS.md` section 3 is unconditional: never fit on data that follows what is scored, and early stopping is a fitting decision |
| 5 | Keras `EarlyStopping`, which stops but does not restore | **Restores the best-validating weights** | Strictly the better estimator. Recorded rather than assumed |
| 6 | Error window centred on the errors it judges | **Trailing**: the window ends at the segment it judges rather than extending past it | See section 1.1. telemanom's windows extend forward, so a timestep is scored partly from errors that had not happened yet. `rstd`, the number we have to beat, is strictly trailing. **Corrected 2026-08-27**: this row said "the 2,100 errors *before* a 70-step segment". The window is `e_s[seg_lo - 2100 : seg_hi]` -- 2,170 samples, the judged segment **included**. It must be, since the segment's sequences are found in the same array, but it means a segment contributes ~3.2% of its own threshold and a sustained event raises the bar it has to clear. Measured as a variant in section 10 |
| 7 | Inference over l_s=250 windows from a zero state | **Chunked-parallel streaming**, each chunk warmed by a 250-step zero-initialised prefix | Identical arithmetic. A warmed chunk sees exactly the history telemanom's own windowed inference sees |
| 8 | `find_epsilon` accepts a candidate only if `score >= max_score` **and** `len(E_seq) <= 5` **and** `len(i_anom) < len(e_s) * 0.5` | **Neither condition is present.** `dynamic_threshold` takes the argmax of the criterion and nothing else | **A defect, not a deviation, and the third of them.** Verified against `khundman/telemanom`, `telemanom/errors.py`. The second condition is a 50% coverage cap: published telemanom rejects outright any candidate that would flag more than half the reference window. Both are invisible while the forecast is poor and few candidates exceed anything, and both bear directly on the failure now under investigation. Found while auditing the criterion for work item 4; measured before anything is changed |
| 9 | `min_delta = 3e-4`, applied as `current < best_loss - min_delta` | **`min_improvement = 0.001`**, applied as `current < best * (1 - min_improvement)` | The published value is absolute and in the units of the loss. Validation MSE on ESA-ADB is ~1e-4, so the bar went negative and **no epoch after the first ever qualified, for every fit in three work items**. A ratio has no units. `docs/DECISIONS.md` D17 |
| 10 | `z` swept over the module constant `np.arange(2.5, 12, 0.5)` | **The same range, moved onto `Config`** as `z_floor` / `z_ceiling` / `z_step`, fittable per model | The range is unchanged and remains the default; what changed is that it is no longer a constant of the method. A threshold measured against one model does not suit a better one, so it belongs to the model it was fitted against. `docs/DECISIONS.md` D17, and the Phase 2 consequence in Objective.md 14.10 |

### 1.1 One deviation was withdrawn, and it would have been fatal

The first version of the detection stack stepped the 2,100-error window forward a
whole window at a time instead of telemanom's 70. It looked like an obvious
economy: thirtyfold overlap is ~52,600 windows per channel per fold on a 3.68M-step
ESA-ADB fold, against 1,753 without it. The ledger entry said *coarser, and if
anything more conservative*.

Measured, it was neither.

```
  injected anomaly     non-overlapping     overlapping
  100 timesteps            detected         detected
  500 timesteps            MISSED           detected
  2,100 timesteps          MISSED           detected
  8,828 timesteps          MISSED           detected
```

A threshold of the form `mu + z*sigma`, computed over the same errors it is
judging, cannot see an anomaly that fills its own window: the anomaly raises both
moments until it sits below its own threshold. `m1-g8.9.10`'s headline-cell events
have a **median footprint of 1,951 timesteps and a 75th percentile of 8,828**
(docs/HARNESS.md section 7). The economy would have blinded the detector to
essentially the whole primary recall set -- and the result would have read as a
weak model rather than a broken thresholder, which is exactly the failure this
project's documents keep warning about.

The measured cost of doing it properly was about 112 microseconds per window, or
some sixteen minutes across a whole run, and the sweep skips the `z` values that
cannot produce an exceedance -- free, since they have none by construction -- which
brought it well under that. The estimate that justified the deviation was simply
wrong.

**What replaced it is not telemanom's window either, and that is deliberate.** The
overlap is restored, but the reference window *trails*: each threshold is chosen
from the 2,100 errors before a 70-step segment and applied to that segment alone.
telemanom's windows extend forward, so a timestep there is scored partly from
errors that had not happened yet. `sentinel_models.baselines` states this
repository's position in as many words -- *a detector that peeks at future samples
is not something that can fly, and the harness should not measure one that does* --
and `rstd`, the number we have to beat, is strictly trailing.

The mechanism that matters survives: the reference window at an onset is still
mostly nominal, which is why long events remain visible. What it costs is a fixed
70-step batching latency, measured and bounded by test, which **delays** detection
rather than improving it, and which is how the flight component would have to run
anyway.

**Kept exactly:** `l_s = 250`, layers `[80, 80]`, dropout 0.3 after *each* LSTM,
MSE loss, Adam at 1e-3, `l_p = 10` predictions aggregated by mean, `epochs = 35`,
`patience = 10`, `batch_size = 70`, `smoothing_perc = 0.05`
(EWMA window 105), `window_size = 30` giving `h = 2100`, `error_buffer = 100`,
pruning at `p = 0.13`, and the z sweep `arange(2.5, 12, 0.5)`.

**(!) `min_delta = 3e-4` was on that list and should not have been.** It is
deviation 9 now, and it was a *defect* for as long as it sat here: the value was
transcribed faithfully and it silently disabled training for every model in the
project. A list headed "kept exactly" is the last place anyone looks for a bug,
which is most of why it survived three work items. Corrected rather than
quietly removed, because the correction is the more useful record.

---

## 2. Trained in torch, scored in NumPy

The forecaster is fitted with PyTorch autograd and **scored through our own
plain-NumPy forward pass** in `src/sentinel_models/reference.py`. Two reasons,
and the second is the larger one.

**Gradients.** A hand-written LSTM backward pass that is subtly wrong produces a
*plausible* bad number, which is the failure mode `sentinel_eval.errors` exists
to prevent elsewhere in this repository. Autograd removes that risk entirely.

**Phase 2 has to verify against something.** Objective.md 4.3 step 2 is
categorical -- no interpreter goes to space -- so the flight component is C++
reading a file of numbers. If both training and scoring ran through PyTorch's
fused LSTM kernel, the C++ port would have no reference to check itself against
and every discrepancy would be guesswork. The NumPy path is a line-by-line
blueprint instead: `lstm_cell` is exactly the arithmetic of one rate-group tick.

The two are held together by `tests/test_reference_equivalence.py`, a **hard
assertion at 1e-5 in float32**, covering the production configuration, a trained
model, gate ordering, both bias vectors, and the state-carrying that chunked
scoring depends on.

### Measured divergence

| Case | max abs difference |
|---|---|
| 2x80, 12 channels, 250-step window | **4.1e-08** |
| 2x80, 12 channels, 4,096-step chunk | **5.2e-08** |

It does not grow with sequence length, so the 1e-5 tolerance carries about two
orders of magnitude of headroom. That margin is deliberate: it means a failure of
this test is a structural defect -- a swapped gate, a dropped bias, a transposed
weight -- and never float32 noise on a bad day.

---

## 3. What this establishes for Objective.md decision 14.2

The model-file format must freeze before Phase 2. Extracting the weights turned
several parts of that decision from theory into measurement.

**Size.** The full telemanom configuration over 12 channels is **91,640
parameters, 358.0 KiB as float32**:

```
  layer 0   w_ih (320, 12)   w_hh (320, 80)   b_ih (320)   b_hh (320)
  layer 1   w_ih (320, 80)   w_hh (320, 80)   b_ih (320)   b_hh (320)
  head      w (120, 80)      b (120)
```

Small enough that quantization is about flash budget and load time rather than
feasibility, and small enough to uplink over a standard F' file service without
special handling.

**Gate order must be in the format, not in the reader's memory.** The four gates
are stacked along axis 0 as `input, forget, cell, output` -- PyTorch's order,
named in `reference.GATES` so a C++ loader author never has to infer it from a
slice index. A loader that guesses wrong produces a model that runs, converges to
nothing useful, and fails silently.

**Both bias vectors are real.** PyTorch carries `b_ih` and `b_hh` separately and
adds both. Mathematically one summed vector suffices and the file format should
store one, but that is a *decision* the format has to state, because a loader
that keeps only `b_ih` is wrong and nothing about the arithmetic announces it.
A test asserts the difference is detectable.

**Dropout has no line in the file.** It is identity at inference, exports no
parameters, and a flight component that dropped a random 30% of hidden units per
cycle would violate Objective.md 11 rule 5 outright.

**Normalisation constants and thresholds stay outside the weights**, per
Objective.md 14.10 -- as PrmDb-style parameters, so a mission recalibrates in
orbit without retraining. Phase 1 normalisation is identity (Objective.md 14.8),
so the constant is trivial here; the *slot* still has to exist, because a mission
whose data is not ESA-preprocessed will need it.

---

## 4. The pre-registration

`docs/HARNESS.md` section 7 records a prediction that failed, and it is preserved
there rather than tidied away because the audit trail is the asset. The same
discipline applies here. **Written and committed before the first run against
ESA-ADB**, so it cannot be quietly replaced by whatever happened.

The floor is `rstd` on `m1-g8.9.10`: **event-wise F0.5 = 0.250**, from recall
3/46 and precision 6/7.

### PREDICTED

| Metric | Prediction | Reasoning |
|---|---|---|
| **Event-wise F0.5 on `m1-g8.9.10`** | **Clears 0.250** | The dynamic threshold plus pruning buy recall without buying alarms, which is precisely what F0.5 rewards. On the fixture the same stack scored 0.370 against `mavg`'s 0.088 |
| **Headline-cell (MVGS) recall** | **Above `rstd`'s 3/32** | The cross-channel class the method exists for. A multivariate forecaster should see events that a per-channel statistic cannot |
| **Rare-event false alarms** | **Low, near `rstd`'s 1/48** | Rare nominal events are *in* the training data by Objective.md 6.2. This is a direct test of that claim, and the one number that decides adoption |
| **`m1-ss5`** | **May lose to `mavg`'s 0.135** | The spiky regime suits a per-channel method. An applicability boundary, published rather than buried -- docs/HARNESS.md section 2 |
| **`lstm-quantile` vs `lstm-telemanom`** | telemanom's dynamic threshold should **win** | Otherwise the paper's central contribution is not earning its place on this data |

### Stop-and-report triggers

1. **F0.5 on `m1-g8.9.10` below 0.250.** Report before touching GRU or TCN.
   Failing to beat a two-line rolling standard deviation means either the
   reproduction is wrong or something in the harness still is, and both are worth
   finding before two more architectures are built on top.
2. **`lstm-quantile` beating `lstm-telemanom` substantially.** That is a finding
   about telemanom, not a reason to keep the better number quietly.
3. **Rare-event false alarms above `mavg`'s 10/48.** A detector that alarms at one
   commanded manoeuvre in five is muted within a week in orbit, whatever its
   recall (Objective.md 11 rule 2).

### What is deliberately not predicted

Point recall. `m1-g8.9.10` carries eleven point anomalies, so recall over them
takes twelve possible values and cannot separate two detectors. It is reported as
a coverage check and is stamped UNDERPOWERED wherever it appears.

### OBSERVED

**A second pre-registration exists**, for the threshold rule that replaces the
one this section's run used: section 10. This one is preserved unchanged.

Filled in beside the prediction above, never in place of it. **Both generations
are given**, because the run that tested the prediction was made with weights
that were trained for one epoch (`docs/DECISIONS.md` D17) and the corrected
weights answer several of these differently. `docs/RESULTS.md` section 6a is the
side-by-side record; the artifacts are named there.

| Prediction | Pre-fix | Post-fix | Verdict |
|---|---|---|---|
| **F0.5 on `m1-g8.9.10` clears `rstd`'s 0.250** | **0.269** | **0.026** | **Held, then lost.** It cleared the floor by 0.019 with a one-epoch model and fell an order of magnitude below it once the model was trained. The floor is a two-line rolling standard deviation |
| **Headline-cell recall above `rstd`'s 3/32** | 28/32 | 28/32 | **Held, decisively, and it is the project's thesis.** A per-channel statistic finds three; a forecaster over the channel set finds twenty-eight, at 0.231 and 0.021 precision respectively |
| **Rare-event false alarms low, near `rstd`'s 1/48** | 22/48 | 30/48 | **Wrong, and wrong by a lot.** Trigger 3 fired: the prediction's own stop-and-report threshold was `mavg`'s 10/48 |
| **`m1-ss5` may lose to `mavg`'s 0.135** | 0.664 | 0.035 | **Wrong pre-fix, right post-fix**, for a reason the prediction did not contain: not the spiky regime suiting a per-channel method, but the threshold collapsing |
| **`lstm-quantile` loses to `lstm-telemanom`** | 0.421 vs 0.269 | not re-run | **Wrong on F0.5, right on the thing that matters.** Trigger 2 fired and was reported: the quantile branch wins F0.5 and is structurally late, and is closed on that ground (`docs/DECISIONS.md` D13) |

**Two of three stop-and-report triggers fired, and both were reported rather
than absorbed.** Trigger 1 -- F0.5 below 0.250 -- did not fire on the run the
prediction was written for and **does fire now**, at 0.026. That is what work
item 4's threshold investigation exists to explain, and it is recorded here
rather than in a commit message because a pre-registration that only records
the triggers that fired conveniently is not one.

**What the prediction got wrong about itself.** Every row above was framed on
the assumption that a better forecaster would produce better detection. The
post-fix column is the counter-example: the forecast improved about fortyfold
and every precision-bearing number got worse, because the decision rule
downstream was calibrated against residuals that no longer exist.

---

## 5. The metric set determines the conclusion

**Lead time did not exist three runs ago. Its first application overturned which
detector looked better.** That is worth recording on its own, because it is the
clearest evidence this project has produced that a conclusion is a property of
what was measured rather than of what is true.

Before it existed, the ranking read:

```
  lstm-quantile    F0.5 0.421   rare-event FA  1/48     <- the better detector
  lstm-telemanom   F0.5 0.269   rare-event FA 22/48
```

`lstm-quantile` wins the gate metric and wins the adoption metric. On the numbers
available at the time there was no argument to be had. Then one measurement was
added -- how many timesteps before the labelled event start the first attributable
alarm fired -- and:

```
  lstm-quantile    median lead  -122 steps   ALL SIX catches late, alarms 5x wider
  lstm-telemanom   median lead   +26 steps   5 of 37 late, alarms 204 wide
```

The detector that won on both headline numbers is **structurally late**. It is
not detecting early and cheaply; it is detecting broadly and after the fact. For
a component whose stated purpose (Objective.md 2) is warning *before* a limit
trips and covering ground-contact gaps, that is a failure at the thing being
built, and no F0.5 compensates for it.

Nothing about the model changed. Nothing about the data changed. The conclusion
changed because the measurement set changed, and it changed **one run after** the
metric was added -- which is roughly the shortest possible interval between
building an instrument and having it overturn something.

The general lesson, and the reason it is recorded here rather than in a commit
message: **a metric set that omits a property of interest will confidently rank
detectors on the properties it does measure.** It will not report that something
is missing. `docs/NARRATIVE.md` collects the other four instances of this in the
project so far.

Lead time is consequently promoted to a gate metric with a disqualifying rule --
`docs/HARNESS.md` section 1.

---

## 6. Held-back sets, nominated before any tuning began

Items 1 and 2 of the decision-layer work -- the persistence filter and the
k-of-n channel agreement -- tune a detector while looking at 46 anomalies that
have now been examined repeatedly. That is exactly the condition under which
tuning-to-the-test-set happens, and it happens without anyone intending it.

So two sets are nominated **here, in writing, before the first tuning run**.
Nominated afterwards they would prove nothing.

| Set | Guards against | Why it is clean |
|---|---|---|
| **`m2-ss1`** | tuning the false-alarm rate to *these* 48 rare events | Independent spacecraft, ~600 rare nominal events, recall disabled by design, and **never scored by anything** -- held back by construction rather than by decision. It targets precisely the number items 1 and 2 are tuning |
| **`m1-g3`** | tuning recall to *these* 46 anomalies | Mission1 group 3, 8 channels, 14 headline-cell events, median footprint 3,594 timesteps and **zero sub-grid-cell events** -- the opposite of the group 8 problem that forced the primary set's promotion. Untouched: no result has ever been computed on it |

`m1-ss5` was considered and **rejected as a held-back set**. It is a strict
subset of `m1-g8.9.10` and carries the same events, so freezing it would
demonstrate nothing about generalisation; and it has already been inspected in
detail. It stays a reported set, not a held-back one.

### The condition that makes this worth anything

**Both sets are run once, at the end, with every setting already frozen. If the
result disappoints, we do not go back and re-tune.**

Tuning against a held-back set converts it into another training set and
destroys the only clean evidence the project has. There is no version of "we
adjusted it slightly after seeing the held-back number" that preserves the
guarantee.

**A disappointing held-back result is the finding, not a problem to fix.** It
would mean the tuning fitted 46 events rather than building a better detector --
which is worth knowing, and worth publishing, and is the entire reason for
nominating a set in advance. The trivial baseline already showed once how
convincing a bad detector looks when nobody set the test up beforehand
(docs/HARNESS.md section 7).

---

## 7. Two diagnostics considered and declined

Recorded so the absence is a decision rather than an oversight. Both would have
produced numbers; neither would have produced evidence.

### Oracle threshold sweep -- declined

The harness can report the best F0.5 any threshold could have reached
(`harness._sweep_best_f_beta`, behind `--no-sweep`). It is tempting as a
diagnostic: a high ceiling with a low honest score says the forecaster ranks well
and the threshold is wrong.

**It is fitting to the labels.** A real spacecraft chooses its operating point
from training data before it has ever seen an anomaly (`Detector.threshold_from`,
and Objective.md 6.1 -- there are no failure examples to tune against). A number
obtained by trying two hundred thresholds and keeping the one the answers liked
is not a number that mission could ever obtain, and it would sit in an artifact
next to figures that are.

The persistence sweep and the k-of-n sweep answer the same diagnostic question --
*is the forecaster the problem, or the decision rule?* -- using only choices a
mission could actually make in advance. So the ceiling is not worth the risk of
being read as a result.

### Forward-looking NDT variant -- declined

**This reverses a proposal I made when the first results landed.** Having found
that `lstm-quantile` beat `lstm-telemanom`, I suggested running published
telemanom's forward-extending error window as a diagnostic, to establish how much
of the gap my causal window was responsible for. That was the wrong instinct.

Published telemanom's error window extends forward, so a rare nominal event
inflates its own window's mean and variance, raises its own threshold, and
suppresses its own alarm. Ours is strictly trailing, deliberately
(`docs/MODELS.md` section 1.1). Running the non-causal variant would produce a
number from a method **we would never fly** -- a spacecraft cannot see the future
-- and the number's existence invites its misuse, because nothing about the digits
records which configuration produced them.

**The confound is therefore documented rather than measured**, and it is stated
plainly here so that no comparison is made in ignorance of it:

> Our rare-event false-alarm rate is **not directly comparable to published
> telemanom figures.** The published method uses a forward-extending error window
> that we deliberately do not, and that difference may account for some or all of
> the gap. We have not measured how much, and will not.

An unmeasured, clearly-stated confound is honest. A measured number from a method
we would never fly is not worth having.

---

## 8. What the LSTM found that the baselines could not

The first gate run died seventy-five minutes in, at its last stage, because a
sequence sampled from supposedly usable training data contained a NaN.

The mask was wrong, and not in this package. `sentinel_eval.bundle.load` folds
unobserved timesteps into `truth.unscorable`, and `splits.train_mask` removes
them, so *usable* implies *observed*. `Bundle.subset` rebuilt its truth from
`labels.truth` alone and never read `self.valid`, so that fold-in was lost --
and `m1-ss5` is scored as a subset of the twelve-channel load. On the six group-8
channels that is **1,622 unobserved timesteps inside an eleven-million-step
training window, marked usable**.

It surfaced on fold 2 rather than fold 0 because sampling is random: 59,771
sequences drawn from eleven million steps will find 1,622 bad ones; 20,000 drawn
from 3.7 million often will not.

**The reason it had never surfaced at all is worth stating plainly.** The trivial
baselines are NaN-tolerant *by construction* -- `nan_to_num`, `nanstd`, a rolling
mean that simply skips what is missing. Not defensively, but because that is what
those one-liners are. So in three work items nothing had ever asked the mask to
be correct, and the defect sat in a referee that had 136 passing tests.

A forecaster cannot be tolerant in that way. One NaN entering a recurrent state
makes every state after it NaN, which is why the sampler checks rather than
hopes. **The LSTM found a harness bug that the baselines were structurally
incapable of finding** -- and it found it on the first run.

That is an argument for having built both, and for the order they were built in.
The baselines were scored first so the harness had been used in anger before any
neural network depended on it (docs/RESULTS.md section 1), and that was right;
but a detector that tolerates anything also validates nothing. The floor and the
candidate test different properties of the referee, and a project that only ever
ran one of them would still be carrying this.

### The fix in this package stays after the harness is fixed

`ForecastDetector.fit` intersects `usable` with what is actually finite. Once
`Bundle.subset` reads `self.valid`, that intersection becomes redundant -- and it
should remain anyway.

It stands on its own terms: **usable can only mean what the model is able to
learn from**, and a one-dimensional mask cannot express which of twelve channels
went missing. The harness's mask answers "is this timestep scorable", which is a
question about the timeline; a forecaster needs "can I read every input at this
timestep", which is a question about the matrix. Those are different questions
that happen to have the same answer when nothing is missing.

Belt and braces on a failure that would otherwise be silent: a NaN in the
recurrence does not raise, it propagates, and every score after it is NaN. The
cost of keeping the check is one boolean AND per fit.

## 9. Dependency note: torch vendors fsspec

`torch==2.13.0` pulls in `fsspec` transitively, along with filelock, sympy,
networkx, jinja2, typing_extensions and setuptools. fsspec expands glob patterns
into paginated LIST operations silently, which is precisely what docs/DATA.md
section 4 forbids, and it is now *importable* in this environment where before it
would have raised.

Nothing changed about the control: `scripts/check_no_list.py` bans `fsspec` and
`s3fs` at source level across `src/` and `scripts/`, and `tests/test_no_list.py`
fails the suite on any violation. Recorded here so that seeing fsspec in
`pip list` does not read as a regression.

---

## 10. The second pre-registration: `lstm-oscfar`

**Written and committed before the first run**, like section 4 and for the same
reason: a pre-registration that can be quietly replaced is not one. Section 4's
failed and is preserved with what happened beside it. This one will be treated
the same way.

The design and its justification are `docs/DECISIONS.md` D20. The measurement
that forced it is `docs/THRESHOLD.md`; D17 and D18 are what it settled.

### 10.1 What is being tested

A new detector, `lstm-oscfar`, added beside `lstm-telemanom` and **not replacing
it** -- the published-comparison baseline the Phase 1 gate is written against
does not move. Same forecaster, same cached weights, same folds, same bundle.
**The decision rule is the only thing that differs**, which is what made the
`lstm-telemanom` / `lstm-quantile` pair informative and is the only way this pair
can be.

Per segment, from its trailing reference window `R` and the nominal residual pool
`N` -- the fitting window scored under the same weights:

```
  eps = max( alpha * Q_p(R) ,        local  -- onset sensitivity
             beta  * Q_p(N) )        floor  -- bounds the collapse
```

`mu` and `sigma` appear nowhere, because measurement rules out the whole
`mu + k*sigma` family and not merely one value of `k` (D20).

### 10.2 What is fitted, and how it stays label-free

| Quantity | Fitted from | Why a mission can do this |
|---|---|---|
| `p`, the quantile rank | bounded below the measured contamination rate -- 53 exceedances in 2,170 samples, 2.4% -- so a rank near 0.75 is far under the breakdown point | read from its own residuals |
| `alpha`, `beta` | the **nominal alarm budget**, on the fitting window's residuals | it states what its operators can act on (`docs/RESEARCH.md`: ISA-18.2, EEMUA 191). It has no failures to fit to |

**(!) SUPERSEDED 2026-08-27, and the paragraph below is preserved because it was
acted on.** It argued that pinning the budget to
`DEFAULT_THRESHOLD_QUANTILE = 0.999` stops it being a free parameter. That is
half right and the wrong half is load-bearing:

> **0.1% is label-free and it is not derived.** No anomaly label is used and a
> mission that has never failed can compute *fire on 0.1% of my nominal
> telemetry* -- that part stands. But nobody derived 0.1%. **Relabelling an
> arbitrary threshold as an arbitrary budget does not remove the arbitrary
> constant**, which is the failure this whole work item exists to correct.

The budget is not ours to pick. It is an **operational input** -- how many alarms
a mission's operators can act on -- and a two-person university CubeSat team
answers differently from an ESOC or JPL control room. Neither answer is a
property of the detector. So section 10.8 reports **a curve across admission
rates**, never a point, and recommends no operating point. That is
`docs/RESEARCH.md`'s precision@k and EEMUA 191 pattern: a mission states what it
can absorb and reads its point off the curve.

*Superseded text, kept:* **The budget is not a number invented for this run.** It
is the operating point this project already uses and has used since the first
baseline: `sentinel_eval.detector.DEFAULT_THRESHOLD_QUANTILE = 0.999`, the
label-free 99.9th percentile of training scores that every trivial baseline and
`lstm-quantile` are scored at. Applying the same convention here rather than
choosing a fresh target is what stops the budget becoming a free parameter fitted
by eye. Concretely, per channel, on the fitting window's residuals under the same
weights:

```
  floor  =  Q_0.999( e_s )                        the existing convention, unchanged
  alpha  =  Q_0.999( e_s / Q_p(R) )               the same convention, applied to
                                                  the LOCALLY NORMALISED error
  eps(t) =  max( alpha * Q_p(R(t)) , floor )
```

So the two multipliers are one idea used twice, and every constant in the rule
traces to a quantile already committed to this repository. `alpha` is estimated
from a sample of nominal segments rather than the whole fitting window, because
running the full local rule over 10.8M steps to fit one scalar is the cost the
NDT short-circuit exists to avoid.

**No labelled anomaly enters the fit.** The fitting window is normal-only by
construction -- `splits.train_mask` removes annotated anomalies -- and the target
is an alarm rate, not a detection score. This is the test `docs/HARNESS.md`
section 6b sets and the one `docs/MODELS.md` section 7 refused the oracle sweep
for failing.

**The alarm count is therefore not a prediction. It is set by the budget.** What
is genuinely predicted is what survives at that budget, which is what section
10.4 is about.

### 10.3 (!) The objection to this proposal, before its first run

An order statistic is immune to tail **weight**. It is not immune to scale
**collapse**: `Q_p` of a uniformly tiny window is tiny. So the claim is narrower
than *order statistics fix it*:

> **What fixes it is calibrating the multiplier against nominal residuals instead
> of transcribing a constant.** The order statistic makes that calibration robust
> to a tail that moves; the floor carries the rest.

**FALSIFICATION CONDITION, stated with a number so the result is read against an
expectation rather than a hope.** If the local term binds in **fewer than 10%**
of scored windows, the floor is doing all the work, this has collapsed back to
`lstm-quantile`, and **it is reported as such rather than defended.**

### 10.4 PREDICTED

Per fold, never pooled. **Fold 0 is the control** -- its forecast improved 1.6x
where folds 1 and 2 improved ~40x, its within-window `sigma` went *up* while
theirs fell three- to eightfold, and its alarm count barely moved. If the new
rule behaves the same on fold 0 as on 1 and 2, the mechanism in D17 is not what
is being fixed.

| # | Prediction | Reasoning |
|---|---|---|
| **P1** | The **local term binds in 60-90%** of scored windows on folds 1 and 2, and **more often on fold 0** | The model is fitted on `N`, so in-sample residuals are smaller than out-of-sample: `Q_p(R)` should typically exceed `Q_p(N)`. Fold 0's residuals are ~6x larger in level than folds 1 and 2 |
| **P2** | **Fold 0 changes least on every axis.** Its alarm count stays within ~2x of the current 73 (`m1-g8.9.10`) and 26 (`m1-ss5`), where folds 1 and 2 fall by roughly an order of magnitude from 1,970 / 1,505 and 692 / 757 | The control did not suffer the collapse, so it has little to recover |
| **P3** | **Headline-cell recall holds at or above 25/32** on `m1-g8.9.10` at a budget near the pre-fix alarm rate | The NDT reached 28/32 at 182 ranges before the training fix and 28/32 at 3,548 after, so the events are visible at both budgets; the question is whether a fitted floor keeps them |
| **P4** | **Median lead time stays positive on both sets**, and falls -- predicted in the **+10 to +30** band against the current +26 | A floor is a magnitude condition and magnitude takes time to reach (D13). This predicts the cost is real and bounded, not absent |
| **P5** | **Rare-event false alarms improve on 30/48 and 33/48**, and do **not** reach `mavg`'s 10/48 | The false alarms are on commanded manoeuvres, and no thresholding change makes a commanded event predictable. That needs the command inputs (D6) |
| **P6** | The **guard-cell variant** (10.5) changes fewer than 5% of windows' `eps` by more than 1% overall, and its effect concentrates on windows overlapping long events, in the direction of **raising** recall | 70 of 2,170 is 3.2% of a quantile's support in a nominal window; under a sustained event the segment is event energy raising the bar the event must clear |

**What is deliberately not predicted.** Point recall -- eleven events take twelve
values and cannot separate two detectors. VUS-PR -- the score transform differs
between the two rules, so a threshold-free ranking metric is not comparing like
with like and is reported as a diagnostic.

### 10.5 The run, and the variants in it

One bundle load, both channel sets, all three folds, **cached weights and no
refit**, held-back sets untouched. Four arms scored against the same residuals:

| Arm | What it isolates |
|---|---|
| `lstm-telemanom` | the control. Unchanged, reproduces the published numbers |
| `lstm-oscfar` | the proposed rule |
| `lstm-telemanom`, guard-cell variant | the reference window excluding the segment it judges |
| `lstm-oscfar`, guard-cell variant | the same, on the proposed rule |

**The guard-cell variant is measured, not assumed.** Deviation 6 in the ledger
was corrected on 2026-08-27: the reference window is `e_s[seg_lo - 2100 : seg_hi]`
-- 2,170 samples, the judged segment **included** -- so a segment contributes
about 3.2% of its own threshold and a sustained event raises the bar it has to
clear. That is the exact CFAR mechanism `docs/RESEARCH.md` records, now confirmed
in our own code rather than cited from radar practice. It is near-free in the
same pass and better known than carried forward.

**`error_buffer` is held at 100 throughout** (`docs/DECISIONS.md` D21). It is
decided afterwards with the threshold frozen, because changing a smoothing
parameter and a threshold rule together would confound them.

### 10.6 Stop-and-report triggers

1. **Median lead time negative on either channel set.** Report before anything
   else. D9 disqualifies such a configuration whatever its F0.5, and +26 is the
   entire budget this project has to spend.
2. **The local term binds in under 10% of windows.** Section 10.3's falsification
   condition. Report as a collapse to `lstm-quantile`, not as a result.
3. **Headline-cell recall below 20/32** on `m1-g8.9.10`. Half the events the
   project exists to catch, given up to buy precision, is a trade that needs
   authorising rather than reporting.
4. **Fold 0 moving as much as folds 1 and 2.** The control behaving like the
   treatment means the mechanism in D17 is not what is being addressed, and the
   result would be uninterpretable whichever way the numbers fell.
5. **Median lead time below +15 on folds 1 or 2, on either channel set** -- even
   though positive, and even though D9 only disqualifies a negative one.

**Why trigger 5 exists and why +15.** D9 draws one line, at zero. Between "still
positive" and "too expensive" there was nothing, so a configuration landing at
+8 would have its acceptability judged **while looking at the result**, which is
the one thing this project refuses everywhere else. The number is therefore set
here, before the run, and it is read from an artifact rather than chosen:

> **+15 is the current 25th percentile.**
> `runs/m1-g8.9.10/lstm-telemanom/2026-08-27T012552Z-8f48b731.json` records
> `p25 = 15.25` on `m1-g8.9.10` and `15.25` on `m1-ss5`, against medians of
> `+26.0` and `+26.5`.

A median that has fallen to where the first quartile used to be means **the
typical detection is now as late as the worst quarter used to be**. That is a
change in kind rather than a cost worth absorbing quietly, and it is reported
before anything is decided. Per-fold post-fix medians are `+18 / +22 / +36` on
`m1-g8.9.10`, so folds 1 and 2 have `+7` and `+21` of room; fold 0 sits near the
line already and is reported rather than triggered on, because it is the control.

### 10.6a (!) How to read a favourable lead-time result from this run

**A good lead-time number from this run is not clean, and must not be reported as
though it were.** `error_buffer = 100` widens every exceedance by +/-99 before
alarm ranges are formed, and lead time is measured from a range's start -- so up
to 99 timesteps of any figure here may be the buffer rather than the detector,
against a budget of +26 (`docs/DECISIONS.md` D21, `docs/HARNESS.md` section 1).
The dilation is nearly four times the whole number being defended.

This cuts both ways and that is the point. A result clearing +15 does not
establish that the rule detects early; it establishes that it clears +15 **under
the same unmeasured term as every figure it is being compared against**. The
comparison is sound; the absolute claim is not, until D21's measurement lands.
Every lead-time figure from this run carries that sentence.

### 10.7 OBSERVED

*Not yet run. Filled in after the run and beside the predictions above, never in
place of them.*

---

## 10.8 The retest: what ran was not what was designed

**Written before the retest runs.** Section 10.7 records the first result and is
not edited. This section says what is being retested and why the first attempt
does not settle the design.

### 10.8.1 The first run failed on an arithmetic error of mine, not on the design

`lstm-oscfar` reproduced `lstm-quantile` exactly -- **recall 6/46, headline cell
6/32, point 0/11, identical in all three** -- at median lead **-196.5**, worse
than the -122 that closed that branch in `docs/DECISIONS.md` D13. Triggers 1 and
3 fired.

The mechanism is understood and it is not a property of order statistics:

> **The two terms were calibrated independently, each to admit 0.1%, and then
> combined with `max()`. The maximum of two thresholds each admitting `r` admits
> far less than `r`.** Measured on nominal residuals, the independent fit admits
> **0.00066 of a 0.001 budget** and **0.000056 of a 0.0001 budget** -- a third to
> a half short. The rule was far quieter than its budget asked for, the floor
> dominated every window, and a dominant global term is the global rule.

So **the design has not been tested.** D20 is refuted *as run* and retested *as
intended*; both outcomes stay in the record, and if the corrected form also fails
D20 closes on two pieces of evidence rather than one.

### 10.8.2 Three arms

| Arm | What it is | Why |
|---|---|---|
| `independent` | the fit that ran, preserved | the corrected form is measured against it, not in place of it |
| `joint` | both terms fitted so **the maximum** admits the target | the design as intended. The honest retest |
| `local_only` | the local order statistic **with no floor** | **it has never operated.** The floor dominated every window of the first run, so there is not one measurement of the local term |

### 10.8.3 PREDICTED, and `local_only` is a new arm so it is pre-registered like one

| # | Prediction | Reasoning |
|---|---|---|
| **P7** | `joint` binds above **10%** at every admission rate, clearing 10.3's falsification threshold, and fires materially more than the 7 ranges the independent fit produced | it admits what it was asked to; the independent fit did not |
| **P8** | `local_only` **alarms far more than `joint` and less than the NDT's 3,548** on `m1-g8.9.10` at 0.1% -- predicted in the **100 to 1,500** band -- with **positive** median lead and headline-cell recall **between 15/32 and 28/32** | it is a local scale rule like the NDT and inherits its onset sensitivity, but its multiplier is *fitted on this model's own nominal residuals* rather than transcribed, which is the thing D17 says was actually broken |
| **P9** | `local_only` **degrades toward the NDT's failure** as the model improves: worse on folds 1 and 2 than on fold 0, relative to its own budget | 10.3's objection -- an order statistic is immune to tail weight, not to scale collapse -- predicts the local term suffers the same collapse, only more slowly |
| **P10** | On every arm, alarm ranges and recall rise **monotonically** with the admission rate across 0.01% / 0.1% / 1% | an internal consistency check on the calibration itself. Non-monotonicity means the fit is not doing what it claims, whatever the detection numbers say |

**Not predicted:** which admission rate is right. That is the mission's input,
not ours, and no operating point is recommended from these numbers. Selecting the
best-scoring cell would be the oracle sweep section 7 already refused, and it
stays refused.

### 10.8.4 What is reported

Every arm, at every admission rate, **per fold, per channel set**, `k/n`
throughout: **binding rate** first, then recall, headline-cell recall, point
recall, precision, rare-event false alarms, alarm ranges, and **lead time**.
`lstm-telemanom` is the control and is unchanged.

### 10.8.5 Stop and report

1. **`joint` also collapses to the global rule.** That closes D20, and the next
   step is a decision rather than another variant.
2. **`local_only` holds up without a floor.** That is a *different design* from
   the one proposed and needs its own decision before being pursued.
3. **Median lead negative on any arm at any rate.** D9 disqualifies it whatever
   else it scores.
4. **Non-monotonic response to the admission rate.** The calibration is not doing
   what it claims and nothing downstream of it can be read.
