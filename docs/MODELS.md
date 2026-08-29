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
blueprint instead: `lstm_cell` is exactly the arithmetic of one rate-group tick,
and since work item 5 `gru_cell` is the same for the other candidate.

The two are held together by `tests/test_reference_equivalence.py`, a **hard
assertion at 1e-5 in float32**, covering the production configuration, a trained
model, gate ordering, both bias vectors, and the state-carrying that chunked
scoring depends on.

### Measured divergence

| Case | max abs difference |
|---|---|
| LSTM 2x80, 12 channels, 250-step window | **4.1e-08** |
| LSTM 2x80, 12 channels, 4,096-step chunk | **5.2e-08** |
| GRU 2x80, 12 channels, 250-step window (2026-08-28) | **1.2e-07** |
| GRU 2x80, 12 channels, 4,096-step chunk | **1.2e-07** |

It does not grow with sequence length, so the 1e-5 tolerance carries about two
orders of magnitude of headroom for either cell. That margin is deliberate: it means a failure of
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

**(!) AMENDED 2026-08-28, by the GRU.** The sentence above -- one summed vector
suffices -- is true of the LSTM and **false of the GRU**. The GRU's third gate
is `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`: its recurrent bias sits
inside the reset product and cannot be folded into `b_in`. So the format must
store **both bias vectors unsummed** whenever the cell is a GRU, and the loader
must add `b_hh` to the recurrent product before the reset multiplication. That
is the mistake a C++ author with the LSTM in their hands is most likely to make,
and `tests/test_reference_equivalence.py` pins it two-sided: folding the biases
diverges in general and agrees exactly once the reset gate is saturated open.
The original sentence is kept because it records the LSTM decision correctly.

### The GRU, measured the same way (work item 5, 2026-08-28)

The same configuration with the other cell is **71,160 parameters, 278.0 KiB
as float32** -- 22.35% below the LSTM overall, and exactly 25% in the recurrent
layers, the head being identical:

```
  layer 0   w_ih (240, 12)   w_hh (240, 80)   b_ih (240)   b_hh (240)
  layer 1   w_ih (240, 80)   w_hh (240, 80)   b_ih (240)   b_hh (240)
  head      w (120, 80)      b (120)
```

Three gates stacked along axis 0 as `reset, update, new` -- PyTorch's order,
named `reference.GRU_GATES`. **One state vector**, `h`, where the LSTM carries
`h` and `c`; the reference lists a GRU layer's state as `(h,)` and refuses an
`(h, c)` handed to it, because the flight state is exactly what is listed and a
dead vector carried for symmetry would be a place for a fault to hide. **A file
says which cell it is by its arrays**: 4H rows per gate block is an LSTM, 3H a
GRU; the file also writes `cell`, and the loader verifies the claim against the
arrays rather than trusting it (`docs/DECISIONS.md` D26).

Objective.md section 8 estimated "~25% fewer parameters". Measured: 25% in the
recurrent layers, 22.35% of the whole model.

### The TCN, measured the same way (work item 6, 2026-08-29)

Six residual blocks of 50 channels, kernel 3, dilations 1 to 32 -- receptive
field **253**, the first standard shape past `l_s = 250` -- is **91,670
parameters, 358.1 KiB as float32** against the LSTM's 91,640 (87,410 on six
channels):

```
  block 0   w1 (50, 12, 3)   b1 (50)   w2 (50, 50, 3)   b2 (50)   res_w (50, 12, 1)   res_b (50)
  blocks 1-5   w1 (50, 50, 3)   b1 (50)   w2 (50, 50, 3)   b2 (50)   -- no shortcut weights: the width does not change
  head      w (120, 50)   b (120)
```

**Kernels are `(out, in, k)` as PyTorch stores them, and tap `j` reads the
input `(k - 1 - j) * dilation` steps back.** A loader that reverses the taps
produces a model that runs and is wrong; the reference names the convention and
a perturbation test pins the field at exactly 253. **No weight normalisation**
and no batch normalisation: nothing in the file but kernels, biases and the
head. **No state**: the file carries no state shape at all, and the flight
component holds the last 252 inputs in a ring buffer, zero-filled at boot --
which is what the causal zero padding in every convolution *is*
(`docs/DECISIONS.md` D27).

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
| `alpha`, `beta` | the **nominal noise floor**, on the fitting window's residuals | it measures what its own normal looks like. It has no failures to fit to |

**(!) SUPERSEDED FRAMING, 2026-08-28.** This table read *the nominal alarm
budget -- what its operators can act on*. **There is no alarm budget**
(`docs/HARNESS.md`). The detector fires when the relationship breaks and is
silent otherwise; the count is reality's, not ours. The quantity being measured
is the **noise floor** -- the line between sensor hum and a real break -- and it
is read from nominal residuals, never chosen so the numbers look right. The
sweeps below stay as a sensitivity measurement, reported at every value and
selected at none; what is struck is the claim that a mission picks its value from
operator capacity.

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

### 10.7 OBSERVED -- the first run

Beside the predictions, never in place of them. Artifacts
`runs/m1-g8.9.10/{lstm-telemanom,lstm-oscfar,lstm-telemanom-guarded,lstm-oscfar-guarded}/2026-08-27T185132Z-*.json`.

**The control reproduced exactly.** `lstm-telemanom` scored 38/46, 75/3,548,
28/32, +26.0 -- identical to the published figures, so `guard_segment=False`
moved nothing and every comparison in the run rests on that exactness.

| arm, `m1-g8.9.10` | F0.5 | lead | recall | precision | MVGS | rare-FA |
|---|---|---|---|---|---|---|
| `lstm-telemanom` (control) | 0.026 | +26.0 | 38/46 | 75/3,548 | 28/32 | 30/48 |
| `lstm-oscfar` | 0.429 | **-196.5** | **6/46** | 7/7 | **6/32** | 0/48 |
| `lstm-telemanom-guarded` | 0.022 | +21.0 | 34/46 | 78/4,507 | 27/32 | 32/48 |
| `lstm-oscfar-guarded` | 0.429 | **-196.5** | 6/46 | 7/7 | 6/32 | 0/48 |

**P1 to P5: refuted.** `lstm-oscfar` reproduced `lstm-quantile` -- **recall 6/46,
headline cell 6/32, point 0/11, identical in all three** -- at a median lead of
-196.5, worse than the -122 that closed that branch (`docs/DECISIONS.md` D13).
Stop-and-report triggers 1 and 3 fired.

**The cause was an arithmetic error in the calibration, not the design.** The two
terms were fitted independently, each to admit the target rate, then combined
with `max()`. The maximum of two thresholds each admitting `r` admits far less
than `r`: measured, **0.00066 of a 0.001 budget** and **0.000056 of a 0.0001
budget**. The rule was quieter than its budget asked, the floor dominated, and a
dominant global term is the global rule. Section 10.8 is the retest.

**P6: refuted, on direction and magnitude.** Predicted a small effect (<5% of
windows) on long events, **raising** recall. Measured on `m1-g8.9.10`: alarm
ranges **3,548 to 4,507 (+27%)**, recall **38/46 to 34/46**, lead **+26.0 to
+21.0**, and fold 0 collapsing from 12/15 at +18.0 to **8/15 at +2.0**.

**(!) And the sign reverses between channel sets, which is the interesting
part.** On `m1-ss5` guard cells *helped*: 38/42 to 39/42, MVGS 29/31 to 30/31.
Twelve channels lose, six channels gain. **A change conditional on something we
have not identified** -- event footprint, channel count or fold composition are
all candidates and none is established. Stated as an open question rather than
averaged away. Guard cells are neither adopted nor rejected; the docstring
correction stands regardless, because the window *is* 2,170 samples with the
segment included whatever its measured effect.

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

### 10.8.6 OBSERVED -- the retest

Artifacts `runs/m1-g8.9.10/_curve/2026-08-27T2011*Z-{independent,joint,local_only}.json`.
Nine cells, three calibrations across three admission rates, both channel sets,
all folds, cached weights, nothing fitted. 45 Class B.

**P10 -- monotonicity: HELD, everywhere.** Alarm ranges and recall rise with the
budget on all six set-mode pairs, so the calibration does what it claims and the
rest is readable. On `m1-g8.9.10`: `independent` 2/9/126 ranges,
`joint` 3/198/2,405, `local_only` 10/85/2,359.

**P7 -- joint binds above 10% at every rate: REFUTED.** Pooled binding on
`m1-g8.9.10` is **0.073** at 0.01%, 0.157 at 0.1%, 0.316 at 1%; on `m1-ss5`
**0.009**, **0.075**, 0.167. It collapses at the tight end of the curve.

**P8 -- `local_only` at 0.1%: two of three.** 85 alarm ranges against a predicted
100-1,500 (just outside), MVGS **19/32** inside the predicted 15-32, median lead
**+26.0** and positive as predicted.

**P9 -- `local_only` degrades as the model improves: SUPPORTED, and it is this
document's own objection confirmed.** At 1% on `m1-g8.9.10`, per fold: fold 0
(the control, 1.6x improvement) spends **405 alarm ranges for 14/15**; fold 1
(40x) spends **1,338 for 13/15**. The improved fold needs **3.3x the control's
alarms for the same recall**. An order statistic is immune to tail weight and not
to scale collapse, exactly as 10.3 said, and the effect is measurable rather than
theoretical.

**(!) Correcting the arithmetic error made the detector worse.**

| `m1-g8.9.10` at a 1% budget | F0.5 | lead | recall | precision | MVGS | rare-FA |
|---|---|---|---|---|---|---|
| `independent` -- the **broken** fit | **0.449** | +26.0 | 37/46 | 51/**126** | 27/32 | **14/48** |
| `joint` -- the **correct** fit | 0.033 | +24.0 | 40/46 | 64/**2,405** | 29/32 | 22/48 |

Same stated budget, **nineteen times the alarms**. The under-admission was acting
as an unintended out-of-sample margin and fixing the arithmetic removed it. The
`independent` cell is the highest event-wise F0.5 this project has recorded *with
positive lead time* -- 0.449 against the previous 0.269 -- at the same +26, with
rare-event false alarms nearly halved. **It is reported and not recommended**: it
works partly by an error whose out-of-sample margin is unmeasured, and selecting
a cell for its score is the oracle sweep section 7 refuses.

**Trigger 2 fired: `local_only` holds up without a floor.** At 0.1% on
`m1-g8.9.10` it reaches 21/46 recall, **19/32 MVGS**, **85 alarm ranges**, median
lead **+26.0**, rare-event false alarms **8/48** -- better than `lstm-telemanom`
on every one of those axes (38/46 at 3,548 ranges and 30/48). The floor was not
load-bearing. It was the part that broke the design. Per 10.8.5 that is a
different design from the one proposed and **nothing is decided here**.

**Trigger 3 fired twice.** Median lead negative at `independent` 0.1%
(**-47.0**, n=6) and `local_only` 0.01% (**-120.0**, n=7). Small n, and D9 is a
rule rather than a preference. Every other cell lands +24.0 to +34.5, all under
10.6a's caveat.

### 10.8.7 (!) The falsification condition was badly designed, and that is the finding

Section 10.3 said: *if the local term binds in fewer than 10% of segments, the
floor is doing all the work and this has collapsed back to `lstm-quantile`.*

> **The `independent @ 0.1%` cell binds at 0.127 -- above the threshold -- and it
> is the cell that reproduces `lstm-quantile` exactly.** The condition would have
> **passed** the design that demonstrably failed.

A binding rate above 10% says the local term wins *somewhere*. It says nothing
about whether it wins where it matters, and a rule that wins 13% of segments
uniformly scattered through quiet history is a global rule with decoration. The
condition measured the wrong quantity and it was pre-registered, which is exactly
the position in which a wrong condition does damage: it would have been cited as
evidence the design survived.

What actually falsified it was the **outcome** -- recall, headline cell and point
recall identical to the closed branch. A condition on the mechanism is only worth
having if it is harder to satisfy than the outcome it stands in for, and this one
was easier. Recorded in `docs/DECISIONS.md` D22.

---

## 11. (!) What tests cross-channel structure, and what does not

**Established by inspection of every stage between the forecast and the alarm,
2026-08-27.** It matters because Objective.md 2.4 is the project's whole claim
and this is where it is and is not expressed.

**The forecaster is multivariate and the decision layer is not.** One LSTM
predicts every channel from every channel, so a broken relationship raises the
residual on whichever channels became unpredictable -- that is why 28 of 32
headline-cell events are caught (`docs/RESULTS.md`). Everything after that point
sees **twelve independent error series**:

| Stage | Input | Sees more than one channel? |
|---|---|---|
| `ewma` smoothing | `(T, C)` | **No.** Vectorised over columns, independent per column |
| `dynamic_threshold` / order statistic | one column | **No** |
| `sequences_at`, `_buffered` | one column | **No** |
| `prune` | one column | **No** |
| `channel_ratios` | one column | **No** |
| **`top_ratios` / `top_columns`** | `(T, C)` | **Yes -- the only one** |
| `apply_persistence` | the combined 1-D mask | **No** |

> **`k`-of-`n` channel agreement is the only stage downstream of the forecaster
> that looks at more than one channel at a time.** And it tests **co-occurrence,
> not relationship**: `top[k-1] >= 1` means *k channels are simultaneously over
> their own individual thresholds*, which is not the same claim as *the
> relationship between them broke*.

**So the decision layer asks "did errors get large", where the thesis is "did the
relationship break".** The two overlap and they are not the same, and the
false-alarm problem lives in the gap: a commanded manoeuvre makes several
channels individually surprising at once, which is exactly what `k`-of-`n`
rewards, and it is nominal.

**Two consequences, stated rather than left to be inferred.**

1. **The cross-channel claim currently rests on the forecaster alone.** The
   decision layer neither strengthens nor tests it. Every headline-cell number
   this project reports is the forecaster's, filtered by a channel-blind rule.
2. **`k`-of-`n` was tuned against 182 alarm ranges and there are now 3,548**
   (`docs/DECISIONS.md` D18). It is the one stage that partially recovers the
   structure and its setting is stale by a factor of twenty.

**Therefore any threshold proposal that goes forward must re-derive `k`-of-`n`
rather than inherit it**, and report what requiring 2 or 3 channels to agree
costs and buys **on the new residuals**: alarm count, headline-cell recall, and
lead time, per fold, both channel sets. If agreement recovers precision without
costing headline-cell recall, that is a better answer than any threshold, because
it is the one that expresses the thesis rather than working around it.

---

## 12. Scoping a relationship-testing stage

**A scope, not a design, and nothing here is decided.** Prompted by section 11
and `docs/DECISIONS.md` D23: the decision layer cannot see what the forecaster
learned, and the one stage that looks across channels tests the wrong thing.

### 12.1 One premise worth correcting first

The natural reading is that a relationship test cannot come from residuals,
because residuals are per-channel by construction. **That is not quite right, and
the difference decides how much work this is.**

The residual *vector* `r_t` in R^C is jointly informative even when every
component is unremarkable on its own. A broken relationship is a **direction** in
residual space that nominal data does not visit -- one channel up while its
group-mate is flat -- and that direction exists in the residuals we already
compute. What destroys it is not the residual; it is what happens next:

| Where | What is lost |
|---|---|
| `detectors.py:410`, `np.abs(filled - forecast)` | **the sign.** Both channels rising together and one rising while the other falls become the same number, and that difference *is* the relationship |
| per-channel smoothing and thresholding | **the joint distribution.** Twelve marginals cannot express a covariance |
| `top_ratios` | **the pattern.** It counts how many channels exceeded, not which, nor in what combination |

So the information is discarded by the decision layer, not absent from the
forecaster. That is the cheaper diagnosis and it should be stated before anything
is built.

### 12.2 What the forecaster already provides

- **Signed residuals**, at no extra forecast cost. The harness already scores the
  fitting window, so nominal residual statistics are free where they are computed
  today -- `scripts/threshold_diagnostics.py` already keeps the signed residual.
- **A normal-only fitting window** (`splits.train_mask`), which is the sample any
  nominal covariance or pair map would be estimated from. Label-free.
- **Nothing else.** No covariance, no pair map, no lag structure, no persisted
  attribution.

### 12.3 Three shapes, cheapest first

**A. Whitened residual (Mahalanobis).** Estimate the nominal residual covariance
`S` (C x C) on the fitting window; score `r_t' S^-1 r_t`. One number per timestep
that is a genuine relationship test: a residual consistent with normal
co-variation scores low **even when large**, and one orthogonal to it scores high
**even when small**.

- *Adds:* a C x C matrix. **144 floats at C=12** -- against 91,640 model
  parameters, it is rounding error in `model.bin`, and in flight it is a fixed
  matrix multiply with no state.
- *Why it targets the actual failure:* a commanded manoeuvre moves channels
  together in a learned pattern, so its residual lies **along** a high-variance
  direction of `S` and is down-weighted -- exactly where k-of-n rewards it.
- *Cost:* keeping the sign, and a new detector. Both additive.
- *Limit:* it tests **linear** co-variation and it names nothing. It would
  improve detection without satisfying section 11 rule 4.

**B. Lagged pair map -- the explanation layer section 7 promises.** For each pair
and each lag, the nominal relationship; at runtime, compare a windowed estimate
against it and name the pair whose agreement broke.

- *Adds:* a `C x C x L` map, and a windowed correlation per cycle. At C=12 and a
  modest lag set this is still small, but it is **real new work** and the runtime
  cost is no longer trivial.
- *This is the only shape that produces `BattTemp / ChargeCurrent decoupled`.*
  It is what makes a warning auditable rather than a score.

**C. Direction-only test.** Keep the sign and ask whether the residual direction
falls in a nominal cone. Cheapest, weakest, and mostly a degenerate case of A.

### 12.4 Phase 1, or the architecture gate

**Recorded as a recommendation with its reasoning, for someone else to decide.**

**A looks like a Phase 1 item.** It is small, label-free, additive, and it aims
directly at the number that decides adoption. It changes no existing metric and
adds one detector beside `lstm-telemanom`.

**B looks like an architecture-gate-or-later item.** It is a product capability
rather than detection mathematics, its runtime cost needs the Phase 4 envelope,
and its output format is a Phase 2 decision.

**And one constraint binds both, from Objective.md 10.2's own argument.** The
Phase 1 gate asks *which forecaster architecture*. Adding a decision stage while
comparing LSTM, GRU and TCN makes the comparison unattributable -- the same
reason Phase 1 stays single-instance. So whatever is adopted must be **fixed
before the gate and held identical across all three**, or deferred past it
entirely. It cannot land in the middle.

**A competing answer already exists and is unmeasured.** `lstm-commanded` (D6)
attacks the same false-alarm number by giving the model the input that makes
commanded events predictable, and its ablation has never been run against
post-fix weights. Whether a relationship stage is needed *in addition* to command
conditioning is not established, and building both before measuring either would
repeat the mistake this section exists to prevent.

---

## 13. Pre-registration: `lstm-whitened-local`

**Written and committed before the run.** Section 10's first pre-registration
failed and is preserved; this one is treated the same way.

### 13.1 What is being tested, and why this and not a looser threshold

`lstm-whitened` misses eleven anomalies `lstm-telemanom` catches, and four
explanations are measured and refuted (`docs/RESULTS.md` 6d). What separates lost
from kept is that the lost events are **weak** -- median peak `||z||` of 3.76
against 16.41 -- and they peak at a **median 0.240 of the threshold**. Loosening
globally would need roughly a fourfold reduction and would take the 2/48 with it.

The remaining hypothesis is about *where* the threshold comes from, not how high
it is. `lstm-telemanom` compares each channel against a **local** window of 2,100
errors and so adapts to a quiet stretch; `lstm-whitened` compares a joint score
against **one global quantile** of the whole nominal pool. Excellent precision
(39/40), poor sensitivity to small excursions -- which is `docs/DECISIONS.md`
D13's asymmetry arriving from the other side.

`lstm-whitened-local` keeps the relationship test unchanged and replaces only the
reference: the whitened length `d` is compared against a **trailing local
quantile of `d` itself**, with the multiplier calibrated on nominal residuals to
the same noise floor. Nothing else moves -- same covariance, same smoothing, same
`error_buffer`, pruning disabled as in section 6d.

**And it is the design most likely to reintroduce the failure this work item
began with.** A locally-referenced threshold is exactly what collapsed onto the
noise floor when the forecaster improved (D17, D18). That is why the collapse
signature below is defined before the run rather than after it.

### 13.2 PREDICTED

The eleven, ordered by how close they came -- `reach` is the event's peak whitened
length as a fraction of the threshold, read from
`runs/m1-g8.9.10/_forensics/2026-08-28T034312Z-discriminator.json`:

```
  id_93   0.647   id_107  0.637   id_110  0.632   id_90   0.602
  id_12   0.602   id_109  0.567   id_114  0.485   id_138  0.367
  id_157  0.300   id_20   0.240   id_89   0.136
```

| # | Prediction | Reasoning |
|---|---|---|
| **P11** | **Between 4 and 7 of the seven highest-reach events return** -- `id_93`, `id_107`, `id_110`, `id_90`, `id_12`, `id_109`, `id_114`. Headline cell moves from **21/32 to between 23/32 and 25/32** | a local reference lowers the bar only where the neighbourhood is quiet, so events already at half the global threshold are the ones it can reach |
| **P12** | **`id_89` (0.136) and `id_20` (0.240) do not return.** A local reference cannot close a sevenfold gap | |
| **P13** | **All seven high-reach events are on fold 0**, so the recovery concentrates there and folds 1 and 2 move little | read from the table above, not assumed. Worth stating because if recovery appears on folds 1 and 2 instead, the mechanism is not the one proposed |
| **P14** | **Rare-event false alarms stay at or below 4/48** on both sets, against 2/48 now | the noise floor is recalibrated to the same target, so the rate should hold |

### 13.3 The collapse signature, and it is harder than the outcome

`docs/DECISIONS.md` D22 records a pre-registered condition that would have
**passed** the failure it was written for, because it was easier to satisfy than
the outcome it stood in for. This one is written to be harder.

**Outcome condition (what success looks like):** headline cell >= 23/32 and
rare-event false alarms <= 4/48 on `m1-g8.9.10`.

**Collapse signature -- ANY of these means the noise-floor failure has recurred,
whatever the detection numbers say:**

1. **Measured nominal admission on the fitting window exceeds 2x its target.**
   The rule is calibrated to admit `r` of nominal timesteps; if it actually
   admits more than `2r` on that same nominal pool, the locality is not
   controlled.
2. **Alarm ranges exceed 400 on `m1-g8.9.10`**, ten times the 40 the global rule
   produces.
3. **Nominal-step false alarms exceed 0.5%** on the test window, against 0.026%
   for the global rule.

**Why condition 1 is strictly harder than the outcome.** It is measured on
nominal data, before any event is scored, and it **can fail while the outcome
passes** -- a rule firing constantly can still miss the 48 rare events and report
a healthy 2/48. That is not hypothetical: the run that produced 3,548 alarm
ranges reported a rare-event rate of 30/48 while its nominal-step rate was
**4.97%**, so the rate that mattered was visible in the nominal data and not in
the adoption number. A mechanism condition earns its place only if it can catch
something the outcome misses, and this one can.

### 13.4 The run

Two arms, cached weights, both channel sets, all folds, no refit, `error_buffer`
held at 100:

| Arm | Why |
|---|---|
| `lstm-whitened` | the global rule at `pruning_p = 0`, giving authoritative scored numbers for the default adopted in section 6d rather than paying for a separate pass |
| `lstm-whitened-local` | the local reference |

### 13.5 OBSERVED

Beside the predictions, never in place of them. Artifacts
`runs/m1-g8.9.10/lstm-whitened{,-local}/2026-08-28T0500*Z-*.json`.

| `m1-g8.9.10` | rare-FA | **MVGS** | recall | precision | F0.5 | lead | nominal-step FA |
|---|---|---|---|---|---|---|---|
| `lstm-telemanom` | 30/48 | **28/32** | 38/46 | 75/3,548 | 0.026 | +26.0 | 4.972% |
| `lstm-whitened` (global) | 2/48 | **21/32** | 27/46 | 41/43 | 0.848 | +29.0 | 0.028% |
| `lstm-whitened-local` | **0/48** | **20/32** | 25/46 | 26/26 | 0.856 | +29.0 | 0.024% |

| `m1-ss5` | rare-FA | **MVGS** | recall | precision | F0.5 | lead |
|---|---|---|---|---|---|---|
| `lstm-telemanom` | 33/48 | **29/31** | 38/42 | 42/1,475 | 0.035 | +26.5 |
| `lstm-whitened` (global) | 2/48 | **18/31** | 23/42 | 118/119 | 0.853 | +27.0 |
| `lstm-whitened-local` | **0/48** | **17/31** | 21/42 | 22/22 | 0.833 | +29.0 |

**P11 REFUTED, and in the wrong direction.** Predicted headline cell of 23/32 to
25/32; measured **20/32**, which is *below* the 21/32 it started from. The local
reference did not recover a single one of the eleven. It made the rule **quieter**
-- 26 alarm ranges against 43 -- rather than more sensitive, which is the opposite
of what a locally-adaptive threshold was supposed to do.

**P12 held trivially** -- `id_89` and `id_20` did not return, but neither did
anything else, so it tested nothing.

**P13 is moot.** There was no recovery to concentrate anywhere.

**P14 held.** Rare-event false alarms 0/48 on both sets, against a ceiling of
4/48.

**The collapse signature did not fire, and that is informative rather than
merely reassuring.** Measured nominal admission was 0.0010040, 0.0010033,
0.0010008 on `m1-g8.9.10` and 0.0010042, 0.0010015, 0.0010020 on `m1-ss5`,
against a target of 0.001 -- correct to four significant figures on every fold.
**The rule is calibrated exactly as intended and simply does not help.** That is
a cleaner negative than a broken one: nothing here is a bug to fix.

**Why it went the wrong way -- a hypothesis, not a finding.** The local reference
is a trailing quantile over a window that **includes the segment being judged**,
which is the same guard-cell violation deviation 6 records for telemanom. A
sustained event raises its own reference and therefore the bar it must clear, and
the fitted multipliers are large -- 6.2 to 12.0 -- so the bar moves with it. That
would explain a rule that gets quieter exactly where an event is. **It is not
measured**, and it is the obvious first thing to check if this direction is ever
reopened.

**Also settled here, with authoritative scored numbers.** The `lstm-whitened`
column is at `pruning_p = 0` and confirms section 6d's free win: `m1-ss5`
headline cell **18/31** against 15/31 at `p = 0.13`, recall 23/42 against 18/42,
rare-event rate unmoved at 2/48. `m1-g8.9.10` is unchanged at 21/32 as measured
before. Precision on the gate set reads 41/43 rather than 39/40, which is the
three extra alarm ranges pruning had been removing.

---

## 14. Pre-registration: `gru-quantile` (work item 5)

**Written and committed before any fit.** Sections 4, 10 and 13 each recorded
a prediction and what happened to it; this one is treated the same way, and
its OBSERVED section is filled in beside the predictions, never over them.

### 14.1 What is being tested, and what is not

The architecture gate asks one question: **which forecaster**. The decision
layer is frozen as `lstm-quantile` (`docs/DECISIONS.md` D25) and is identical
here: per-channel EWMA of the absolute residual, the maximum across channels,
one label-free 99.9th percentile of the anomaly-masked fitting window's scores.
`gru-quantile` is that rule on a GRU forecast. **The cell is the only
variable** -- `Hyper(cell="gru")`, every other value shared with the LSTM by
construction (D26): 12 channels, `l_s = 250`, two layers of 80, dropout 0.3
after each layer, MSE, Adam 1e-3, batch 70, up to 35 epochs, patience 10,
relative `min_improvement = 0.001`, seed `0 + fold`, the same sampler and the
same guard against a fit that keeps its first epoch.

**The hypothesis under test, stated as one.** `lstm-quantile` misses twelve
events on `m1-g8.9.10` that `lstm-telemanom` catches, and eleven on `m1-ss5`
(section 14.2). Five hypotheses about the decision layer have been measured
and did not recover them (`docs/RESULTS.md` 6d, section 13, D24, D25). The
forecaster has not been varied. The route by which a different cell could
reach them is **a lower residual noise floor**: the threshold is a quantile of
nominal residuals, so a forecaster whose nominal residual is smaller sets a
lower bar, and an event whose peak sat below the LSTM's bar may clear the
GRU's. That is a hypothesis, not a conclusion on record: D24 says in as many
words that the twelve are not claimed unrecoverable, and section 4 records the
counter-example in which a fortyfold better forecast made detection worse.
What this run measures is whether the mechanism operates at all.

**What does not move.** The harness, the folds, the split, the bundle, the
decision layer, `error_buffer` (D21 first reach still open, and irrelevant to
the gate arm, which has no dilation), `k`-of-`n` (D23, resolved before the
gate and not here), and the held-back sets `m2-ss1` and `m1-g3`.

### 14.2 The events, and their reach under the frozen rule

Read from `runs/m1-g8.9.10/_forensics/2026-08-28T022324Z-events.json`,
`.../2026-08-28T184844Z-head-to-head.json`, and the reach run below. **Reach**
is an event's peak score inside its span as a fraction of the fold's
threshold: 1.0 is the bar; a miss sits below it. Under `lstm-quantile` that
score is the maximum across channels of the smoothed absolute residual, so a
reach is in the units the frozen rule actually cuts on -- the whitened reaches
in section 13.2 are a different quantity and are not reused here.

Artifact `runs/m1-g8.9.10/_forensics/2026-08-28T205347Z-head-to-head.json`
(`lstm-telemanom` vs `lstm-quantile`, cached weights, weight store unchanged,
15 Class B). Ordered by reach. `T` is `lstm-telemanom`'s reach under its own
dynamic threshold, for the record; it catches all of these.

**`m1-g8.9.10`** -- `lstm-quantile` catches 26/46, all of them inside `lstm-telemanom`'s 38; kept events reach a median 1.40 (min 1.09). The 12 it misses that `lstm-telemanom` catches:

```
  fold  event    cell                               footprint   reach      T
  1     id_132   Multivariate/Global/Point                1   0.995   1.66
  1     id_138   Multivariate/Global/Subsequence       1787   0.480   1.28
  2     id_157   Multivariate/Global/Subsequence         51   0.307   1.18
  1     id_20    Multivariate/Local/Subsequence       16227   0.209   1.56
  0     id_110   Multivariate/Global/Point                1   0.181   1.09
  0     id_109   Multivariate/Global/Point                1   0.172   1.10
  0     id_90    Multivariate/Global/Subsequence         27   0.168   1.81
  0     id_12    Multivariate/Global/Subsequence      11937   0.168   1.81
  0     id_107   Multivariate/Global/Subsequence         32   0.158   1.69
  0     id_93    Multivariate/Global/Subsequence         63   0.150   1.58
  0     id_114   Multivariate/Global/Point                1   0.147   1.17
  0     id_89    Multivariate/Global/Subsequence       8995   0.111   1.31
```

**`m1-ss5`** -- `lstm-quantile` catches 27/42, all of them inside `lstm-telemanom`'s 38; kept events reach a median 1.36 (min 1.05). The 11 it misses that `lstm-telemanom` catches:

```
  fold  event    cell                               footprint   reach      T
  1     id_138   Multivariate/Global/Subsequence       1786   0.321   1.39
  2     id_157   Multivariate/Global/Subsequence         51   0.236   1.14
  0     id_109   Multivariate/Global/Point                1   0.208   1.17
  1     id_121   Multivariate/Global/Subsequence          1   0.204   1.00
  0     id_90    Multivariate/Global/Subsequence          1   0.202   2.20
  0     id_12    Multivariate/Global/Subsequence          1   0.202   2.20
  0     id_93    Multivariate/Global/Subsequence          1   0.200   1.73
  0     id_110   Multivariate/Global/Point                1   0.195   1.23
  0     id_114   Multivariate/Global/Point                1   0.195   1.39
  0     id_107   Multivariate/Global/Subsequence          1   0.175   2.12
  0     id_89    Multivariate/Global/Subsequence       4117   0.139   1.34
```

**What the table says before any GRU exists.** One event, `id_132` -- a
footprint-1 point anomaly on fold 1 -- sits at **0.995 of the bar**: a threshold
half a percent lower catches it, and that is inside the run-to-run noise of any
refit. Its return would say nothing about the mechanism in either direction,
and it is scored that way below. **Every other missed event is at 0.48 or
less** on the gate set and 0.32 or less on `m1-ss5`; the eight fold-0 events
sit between 0.11 and 0.18. For those to return, the GRU's floor would have to
fall to **less than half** the LSTM's -- on fold 0, to less than a fifth -- while
P17 predicts it moves by a quarter at most. That is the gap the hypothesis of
14.1 has to close, stated in the rule's own units before the fit.


### 14.3 PREDICTED

Anchors, all from `runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json`
and `runs/_weights_pod/fit_report.json`. Fold 0 is the control, as in section
10.4: it is the data-poor fold, its LSTM forecast is ~40x worse than folds 1
and 2, and its noise floor is ~10x theirs.

| fold | LSTM val-MSE, `m1-g8.9.10` / `m1-ss5` | LSTM noise floor (threshold), `m1-g8.9.10` / `m1-ss5` | LSTM epochs (best), `m1-g8.9.10` |
|---|---|---|---|
| 0 | 1.691e-4 / 3.441e-5 | 0.1468 / 0.1020 | 14 (3) |
| 1 | 3.552e-6 / 5.205e-6 | 0.01563 / 0.01243 | 35 (28) |
| 2 | 4.089e-6 / 8.096e-6 | 0.01178 / 0.01545 | 32 (21) |

| # | Prediction | Reasoning |
|---|---|---|
| **P15** | **71,160 parameters, 278.0 KiB**, against the LSTM's 91,640 -- 22.35% fewer overall, 25% fewer in the recurrent layers | Arithmetic, and already measured by `Weights.n_parameters` (section 3). Recorded so that the number the gate's criterion 2 reads was stated before the fit |
| **P16** | **Validation MSE within 0.7x-1.4x of the LSTM's on every fold, both sets** | Objective.md 8's working hypothesis: parity at this scale. Fold 0 is the fold most likely to leave the band, in either direction -- less capacity on the least data |
| **P17** | **Noise floor within 0.8x-1.25x of the LSTM's threshold on every fold, both sets**, and **fold 0's floor stays at least 5x folds 1 and 2** on `m1-g8.9.10` | If P16 holds the nominal residual is the same size and the quantile of it moves little. The fold-0 gap is set by the data, not the cell |
| **P18** | On `m1-g8.9.10`: **MVGS 19/32 to 23/32** (21/32 now), **rare-event FA at most 4/48** on both sets (2/48 now), **nominal-step FA at most 0.006%** (0.002% now), **pooled honest median lead +0.0** on both sets | A quantile rule on a residual of the same size fires at the same events. The crossing is the emission on this path, and a magnitude rule fires at the boundary, not before it |
| **P19** | **None of the eleven whitened-lost events returns, on either set.** `id_132` (reach 0.995) is a coin toss and is predicted neither way; its return alone is **not** a recovery of the weak events and is not scored as one. If any of the eleven does return, it is the **highest-reach** of them (`id_138`, 0.48, fold 1) or one of the fold-0 eight, and only on a fold whose floor fell by the factor its reach requires | An event returns only if its reach rises to 1. With P17's floor moving by at most 25%, only events above 0.8 can return and there is exactly one, at 0.995. The eleven need the floor to halve at least, and on fold 0 to fall fivefold; a return there without that fall is a mechanism other than the noise floor and is reported as such |
| **P20** | On `m1-ss5`: **MVGS 19/31 to 23/31** (21/31 now), recall **24/42 to 30/42** (27/42 now) | Same reasoning as P18 on the six-channel view |

**Deliberately not predicted.** Point recall -- eleven events, twelve values,
UNDERPOWERED wherever it appears. VUS-PR. And `gru-telemanom`'s numbers: it is
scored in the same run at no extra cost as the reproduction reading beside
`lstm-telemanom`, but it runs through telemanom's dynamic threshold and
`error_buffer`, both of which D17, D18 and D21 have shown to be properties of
the decision rule rather than the forecaster. It is reported, and it is not
the gate.

### 14.4 Outcome condition, and a mechanism condition that is harder

D22: a mechanism condition earns its place only if it can fail while the
outcome passes. Section 13.3's condition -- nominal admission on the fitting
window -- is **tautological for this rule**: the threshold *is* the 99.9th
percentile of those scores, so the fitting-window admission is 0.1% by
construction and tests nothing. A different condition is needed.

**Outcome condition (parity):** MVGS >= 21/32 and rare-event FA <= 2/48 on
`m1-g8.9.10`.

**Mechanism condition, strictly harder, ANY of which fails the run whatever
the detection numbers say:**

1. **Nominal-step false alarms on the *test* window exceed 3x the LSTM's on
   any fold of either set** (floored at 0.001% of that fold's nominal steps,
   because fold 0 of `m1-g8.9.10` is at 0/3,569,953). Measured on nominal
   timesteps, independently of every labelled event, and it can fail while the
   outcome passes: a rule can hold 21/32 and 2/48 while firing three times as
   often on healthy telemetry, and 48 rare events cannot see that where ten
   million nominal steps can. A GRU that "recovers" an event this way has
   bought it with alarms, not sensitivity.
2. **A recovered event whose fold's floor did not fall by what its reach
   required.** An event at reach `r < 1` under the LSTM needs the GRU's
   threshold at or below `r` times the LSTM's on that fold, all else equal. If
   it returns on a fold where the floor fell less than that -- or not at all --
   the noise floor is not what recovered it and the hypothesis of 14.1 is not
   what the run confirmed: reported as a recovery by an unidentified mechanism
   (a residual that grew under the event rather than a floor that fell), never
   as support for the stated one. `id_132` at 0.995 is exempt from this test
   by construction and from P19 by declaration.
3. **`best_epoch == 0` on any fold.** The guard raises; a raise is the stop.

### 14.5 The run

| Step | What | Cost |
|---|---|---|
| Fit, on a rented GPU | `scripts/fit_folds.py --task m1-g8.9.10 --detector gru-quantile --device cuda` -- six fits, two sets by three folds, after `tests/test_reference_equivalence.py` passes on that box (D15) and after two same-seed fits of `m1-ss5` fold 0 prove bit-identical (D15 measured that for the LSTM only) | ~16 operations, 1 Class A |
| Verify, on the Mac | every file through the production loader, `cell == "gru"`, `best_epoch > 0`; torch-vs-NumPy on the real weights at 1e-5; then the scoring run below must show a cache hit on fold 0 before the pod is destroyed (D16) | 0 |
| Score, on the Mac | `python -m sentinel_eval run m1-g8.9.10 --detector gru-quantile --detector gru-telemanom --no-sweep` -- one bundle load, both sets, all folds, both arms; weight store must gain no file | 15 Class B, 1 Class A |
| Per event | `scripts/head_to_head.py --a lstm-quantile --b gru-quantile`, with reach | 15 Class B, 1 Class A |

Held-back sets untouched. Nothing refitted for the LSTM.

### 14.6 Stop and report

1. **`tests/test_reference_equivalence.py` cannot reach 1e-5 on the fitting
   box.** No fit is banked.
2. **Any of the eleven returns** (`id_132` reported, not stopped on). That is
   the headline, reported before anything else is written, with 14.4's second
   condition read alongside it.
3. **GRU MVGS at or above 21/32 on `m1-g8.9.10`.** Report and wait before
   anything further -- the two cells would then be separated by criteria 2 to
   5 of Objective.md 8, which is a decision and not a measurement.
4. **Any fold's validation MSE above 3x the LSTM's.** A fitting problem to
   diagnose, not a cell result to record.
5. **More than 200 operations in any single run.** A script shaped wrongly;
   the tripwire is 1,000 and this plan expects at most 16.

### 14.7 On the record for later, not built

A two-forecaster *agreement* ensemble -- LSTM and GRU both exceeding their own
floors -- as a noise-reduction route to whatever weak events remain. Nothing on
record anywhere in this repository considers it; it is evaluable only once GRU
and TCN rows exist, and it cannot land mid-comparison (section 12.4). Logged
so it is not lost, and not touched.

### 14.8 OBSERVED

Beside the predictions, never in place of them. Artifacts
`runs/m1-g8.9.10/gru-quantile/2026-08-28T222635Z-6d146f5d.json`,
`runs/m1-g8.9.10/gru-telemanom/2026-08-28T222635Z-a428c1d6.json`,
`runs/_weights_pod/gru-2026-08-28/fit_report.json`, and the head-to-head named
in 14.8.2. The full tables are `docs/RESULTS.md` 6h.

**The run.** Six fits on an A40-2Q slice in 6.6 minutes, every fold with
`best_epoch > 0`, the `m1-ss5` fold-0 refit **bit-identical**; every file
loaded on the M5, `cell == "gru"` in field and arrays, torch-vs-NumPy at most
3.0e-07 on the real weights; scored with refits refused, weight store
unchanged. 16 operations per run.

#### 14.8.1 The predictions

| # | Predicted | Observed | Verdict |
|---|---|---|---|
| **P15** | 71,160 parameters | 71,160 (12 channels), 64,860 (6) | **Held** |
| **P16** | val-MSE 0.7x-1.4x the LSTM's, every fold | `m1-g8.9.10` **0.03x** / 0.95x / 0.85x; `m1-ss5` **0.14x** / 0.99x / 0.76x | **Refuted on fold 0 of both sets, in the other direction**; held on folds 1-2 |
| **P17** | floor 0.8x-1.25x, every fold; fold 0 at least 5x folds 1-2 | `m1-g8.9.10` **0.09x** / **1.81x** / 1.24x; `m1-ss5` **0.09x** / **1.38x** / 0.66x. Fold 0 is now the *lowest* floor on the gate set | **Refuted on five of six**. The floor moved by an order of magnitude on fold 0 and rose on fold 1 with the MSE unchanged |
| **P18** | MVGS 19-23/32; rare-FA <= 4/48 both sets; nominal-step <= 0.006%; honest lead +0.0 both sets | **22/32**; **1/48** and **3/48**; **0.001%**; **+0.0** and **+0.0** | **Held**, on all four |
| **P19** | none of the eleven returns; `id_132` predicted neither way; a return only on a fold whose floor fell by what the reach required | see 14.8.2 | see 14.8.2 |
| **P20** | `m1-ss5` MVGS 19-23/31, recall 24-30/42 | 21/31, 26/42 | **Held** |

**Not predicted, reported:** point recall 5/11 and 5/11; VUS-PR 0.374 and
0.591 (LSTM 0.343 and 0.604); `gru-telemanom` in `docs/RESULTS.md` 6h.4, where
D17's collapse returns at 6,206 alarm ranges and 39/48.

**Why P16 and P17 failed, as far as the artifacts say.** The LSTM's fold-0 fit
had stopped at epoch 14 with its best at epoch 3; the GRU's ran to the cap with
its best at 34 and reached the MSE the LSTM reaches on folds 1 and 2. So the
GRU did not have a better cell on fold 0 so much as a fit that did not stall,
and whether the LSTM's stall is the cell or the seed is **unmeasured** -- one
fold-0 refit of the LSTM with another seed would say, and it has not been run.
On fold 1 the MSEs agree to 5% and the floor is 1.8x higher: the 99.9th
percentile of the nominal residual is a tail statistic and the loss is a mean,
and P17's reasoning -- same MSE, same floor -- assumed they move together. They
do not, and D17's MEASURED section had already recorded kurtosis in the
thousands on this residual. That is the corrected form of the hypothesis in
14.1: **a lower noise floor recovers weak events, and the noise floor is not
the validation loss.**

#### 14.8.2 The per-event line

Artifact `runs/m1-g8.9.10/_forensics/2026-08-28T223610Z-head-to-head.json`
(`lstm-quantile` vs `gru-quantile`, cached weights, weight store unchanged at
69, 15 Class B). Reach is the peak score in the event span over the fold's
threshold; `needed` is the LSTM's threshold over the GRU's, the factor an
event at LSTM reach `r` required the floor to fall by (`1/r`) against the
factor it fell (14.4, condition 2).

**`m1-g8.9.10`** -- both 20, only `lstm-quantile` 6, only `gru-quantile` 7,
neither 13. **The seven the GRU recovers are seven of the eleven**, every one
on fold 0, every one by the floor:

```
  fold  event    cell                               fp      LSTM reach -> GRU   floor fell   needed
  0     id_107   Multivariate/Global/Subsequence      32     0.158 -> 1.286      11.1x        6.3x
  0     id_109   Multivariate/Global/Point             1     0.172 -> 1.271      11.1x        5.8x
  0     id_110   Multivariate/Global/Point             1     0.181 -> 1.233      11.1x        5.5x
  0     id_114   Multivariate/Global/Point             1     0.147 -> 1.275      11.1x        6.8x
  0     id_12    Multivariate/Global/Subsequence   11937     0.168 -> 1.269      11.1x        5.9x
  0     id_90    Multivariate/Global/Subsequence      27     0.168 -> 1.269      11.1x        5.9x
  0     id_93    Multivariate/Global/Subsequence      63     0.150 -> 1.249      11.1x        6.7x
```

Four of the seven are headline-cell events; that is 21/32 to 22/32 net of the
losses below. The eleven's other four did not return: `id_89` (fold 0, 0.111 ->
0.393 -- the floor fell 11x but the event's own peak fell with it), `id_138`
(0.480 -> 0.181), `id_20` (0.209 -> 0.130) and `id_157` (0.307 -> 0.238), the
last three on folds whose floor rose. `id_132`, the coin toss at 0.995, landed
at 0.553 and is scored neither way, as declared.

**The six the GRU loses are all on fold 1, and all by the floor too**, in the
other direction: `id_122, id_124, id_129, id_130, id_140, id_142`, at LSTM
reach 1.087-1.207, land at 0.596-0.616 -- the same peaks divided by a threshold
1.81x higher. Two are headline-cell.

**`m1-ss5`** -- both 19, only `lstm-quantile` 8, only `gru-quantile` 7,
neither 8. **The same seven return**, on fold 0, at reach 1.873-1.917 against
0.175-0.208, the floor having fallen 11.6x where 4.8x-5.7x was needed. Eight
are lost on fold 1: the six above plus `id_132` (1.269 -> 0.914) and `id_145`
(2.055 -> 0.884), against a floor 1.38x higher. `id_89` (0.139 -> 0.470),
`id_121`, `id_138` and `id_157` stay missed.

**P19 refuted, and mechanism condition 2 satisfied on every recovery.** The
prediction was *none of the eleven returns*; seven did, twice. The condition
was that a recovery happens only on a fold whose floor fell by what the reach
required; every recovered event is on fold 0 and the floor fell by 11x against
a requirement of 4.8x-6.8x, so **the recoveries are the stated mechanism** --
a lower noise floor -- and not something else. What the pre-registration got
wrong was not the mechanism but its size: it predicted the floor would move by
a quarter and it moved by an order of magnitude on the one fold where the
LSTM's fit had stalled, and rose on the fold where it had not. **Trigger 2
fired.**

#### 14.8.3 The conditions

**Outcome condition (parity): PASSED.** MVGS 22/32 >= 21/32, rare-FA 1/48 <= 2/48.

**Mechanism condition 1 -- nominal-step alarms on the test window <= 3x the
LSTM's, floored at 0.001% of the fold's nominal steps: FAILED on three of six.**

```
  m1-g8.9.10   fold 0    79 vs     0   of 3,569,953   cap  36    FAIL
               fold 1     0 vs    69   of 3,515,149   cap 207    pass
               fold 2    63 vs   145   of 3,590,386   cap 435    pass
  m1-ss5       fold 0   958 vs   121   of 3,644,798   cap 363    FAIL  (7.9x)
               fold 1   118 vs 31,607  of 3,608,842                pass
               fold 2   460 vs   101   of 3,622,049   cap 303    FAIL  (4.6x)
```

The condition was written to be able to fail while the outcome passed, and it
did exactly that: pooled rare-FA improved to 1/48 while fold 0 of both sets
fired more often on healthy telemetry than the LSTM, and `m1-ss5` fold 2 too.
The recoveries on fold 0 are real detections (14.8.2) and they were bought
partly with alarms the adoption number cannot see. Both are reported.

**Mechanism condition 2 -- a returned event's fold floor fell by what its reach
required:** adjudicated per event in 14.8.2.

**Mechanism condition 3 -- `best_epoch == 0`:** did not fire on any fold.

#### 14.8.4 Stop and report

Trigger 1 did not fire (equivalence 3.0e-07). Trigger 4 did not fire (no fold
worse than 3x; fold 0 was 36x *better*). Trigger 5 did not fire (16 operations).
**Trigger 3 fired**: MVGS 22/32 is at or above the LSTM's 21/32. **Trigger 2**
is adjudicated in 14.8.2. Everything from here is reported and nothing is
decided: which cell goes forward is Objective.md 8's five criteria, read
together with the TCN row that does not exist yet, and that is a decision for
the record and not for this section.

---

## 15. Pre-registration: the LSTM's fold 0, reseeded

**Written and committed before the fit.** Section 14's headline is seven weak
events recovered on fold 0, and 14.8.1 says why that is two readings and not
one. This is the one fit that separates them.

### 15.1 The question

On fold 0 the LSTM's banked fit (seed 0, fitted on an A40) stopped at epoch 14
with its best at epoch 3, validation MSE 1.691e-4 -- forty times worse than its
own folds 1 and 2 -- and a noise floor of 0.14681. The GRU's fit on the same
fold ran to the cap, best at 34, 4.628e-6, floor 0.01324. Every recovered
event was recovered by that floor (14.8.2). **Was fold 0's transformation the
cell, or that one optimisation path?**

One fit answers it: `lstm-quantile` with `Hyper(seed=1)` -- the seed is the only
field changed; window, layers, dropout, loss, optimiser, batch, epoch cap,
patience, relative stopping rule, sampler and validation pinning are the
published protocol, and the fold, the data and the frozen decision layer are
identical. Scored on fold 0 through `harness._score_fold`, the referee's own
path, on both channel sets from one bundle load.

**One confound, stated.** The reseed is fitted on the M5 (CPU, four threads);
the seed-0 fit was made on an A40. D15 measured that the same seed on different
hardware gives different weights, so this run changes the optimisation path
twice over. That is the question being asked -- *was it that path* -- and it
does not need to isolate the seed from the box to answer it. It cannot say
which of the two mattered, and does not claim to.

**Cost, and why local.** A fold-0 LSTM fit is 20,064 sequences per epoch at
18.9 s/epoch on the M5 (D15): at most 11 minutes if it runs to the cap, plus
about four minutes of NumPy scoring per set. 15 Class B, 1 Class A. A pod's
setup alone cost 25 minutes on 2026-08-28.

### 15.2 Anchors

Fold 0, from `runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json`,
`.../gru-quantile/2026-08-28T222635Z-6d146f5d.json`, both fit reports, and
`_forensics/2026-08-28T223610Z-head-to-head.json`.

| fold 0 | val-MSE | epochs (best) | floor | recall | MVGS | precision | rare-FA | nominal-step | honest lead |
|---|---|---|---|---|---|---|---|---|---|
| `m1-g8.9.10` LSTM seed 0 | 1.691e-4 | 14 (3) | 0.14681 | 4/15 | 4/11 | 7/8 | 1/12 | 0 / 3,569,953 | -77.0 (n=4) |
| `m1-g8.9.10` GRU | 4.628e-6 | 35 (34) | 0.01324 | 11/15 | 8/11 | 13/31 | 1/12 | 79 | +0.0 (n=11) |
| `m1-ss5` LSTM seed 0 | 3.441e-5 | 22 (11) | 0.10195 | 4/13 | 4/10 | 13/15 | 1/12 | 121 / 3,644,798 | -41.0 (n=4) |
| `m1-ss5` GRU | 4.968e-6 | 35 (32) | 0.00878 | 11/13 | 8/10 | 19/89 | 2/12 | 958 | +0.0 (n=11) |

The seven (`id_107, id_109, id_110, id_114, id_12, id_90, id_93`), fold 0,
`m1-g8.9.10`: raw peak score (reach times threshold) **0.0216-0.0265 under the
stalled LSTM**, 0.0163-0.0170 under the GRU. The stalled LSTM's peaks already
exceed the GRU's floor. `id_89` peaks at 0.0163 under the LSTM and 0.0052 under
the GRU (reach 0.393) and is not expected back under either.

### 15.3 PREDICTED

| # | Prediction | Reasoning |
|---|---|---|
| **P21** | **The reseed does not stall**: `epochs_run >= 25`, `best_epoch >= 15`, validation MSE **<= 1.0e-5** on `m1-g8.9.10` fold 0 | The GRU fitted this fold to 4.6e-6; the LSTM fits folds 1 and 2, whose data contain fold 0's, to 3.5e-6 and 4.1e-6. A best epoch at 3 with patience exhausted at 14 is the signature of one poor optimisation path, not of a data limit |
| **P22** | Noise floor **0.010-0.020** (0.75x-1.5x the GRU's 0.01324) | On this fold the floor followed the fit for the GRU; P21 says the fit will be there |
| **P23** | **At least five of the seven** are caught by the reseeded LSTM on fold 0 at reach >= 1; `id_89` stays missed | Their peaks under a fitted model land near the GRU's 0.016-0.017 and clear a floor near 0.013. If the floor is there and the events are not, the peaks fell with the fit -- see the mechanism condition |
| **P24** | Fold-0 rare-event false alarms **<= 2/12**; nominal-step alarms **20-300** (from 0) | A fitted floor admits 0.1% in-sample by construction and more out of sample; the GRU's 79 is the anchor |
| **P25** | **The verdict rule, set now.** P21-P23 all hold: fold 0's transformation was the fit, and the GRU's fold-0 advantage over the *banked* LSTM is not a property of the cell. The reseed stalls (val-MSE > 5e-5 with `best_epoch <= 5`): it is the cell, and the LSTM stalls on this fold where the GRU does not. Anything between -- fitted but the floor did not follow, or the floor there and fewer than three of the seven -- is reported without a verdict | Written before the number so the number is read against it |

**Not predicted.** Fold-0 F0.5 and precision (n < 20 both). Lead time. `m1-ss5`
is fitted and reported in the same run at no extra operations, and reads the
same rules; the verdict is taken on the gate set.

### 15.4 Mechanism condition, harder than the outcome

The outcome is P23. The mechanism claimed is *the floor follows the fit*, and it
is read from the training curve and the threshold **before** any event is
looked at:

1. `best_epoch > 0` (the guard raises otherwise).
2. **The floor and the MSE move together or the mechanism has failed**: val-MSE
   <= 1e-5 with a floor above 0.05, or a floor below 0.02 with val-MSE above
   5e-5, fails this condition whatever P23 says -- it would be section 14's
   fold-1 finding again, the floor and the loss moving independently.
3. **Nominal-step alarms on the fold-0 test window <= 3x the GRU's 79** (237).
   Above that, the recoveries were bought with alarms and the comparison with
   the GRU's fold 0 is not like for like.

### 15.5 Stop and report

1. The reseed stalls (P25's second branch). Report; the cell is implicated and
   the gate reads differently.
2. The reseed catches **more than the seven** the GRU recovered on fold 0.
3. More than 50 operations in the run.

### 15.6 What this run does not do

It moves no published row. `lstm-quantile`'s record is the seed-0 fit and stays
as scored. If the fit is the cause, whether the LSTM's banked fold-0 weights
should be refitted for the gate is a **decision** -- it changes the LSTM's row --
and it is recorded as one before anything is refitted, not done here. Held-back
sets untouched.

### 15.7 The run

`scripts/reseed_fold.py --task m1-g8.9.10 --detector lstm-quantile --seed 1 --fold 0 --both`
-- one bundle load, fold 0 of both sets, the harness's `_score_fold`, per-event
reach, artifact `runs/m1-g8.9.10/_forensics/<stamp>-reseed-lstm-quantile-seed1.json`
written before the ledger commit. The weight store must gain exactly one file
per set.

### 15.8 OBSERVED

Beside the predictions. Artifact
`runs/m1-g8.9.10/_forensics/2026-08-28T232538Z-reseed-lstm-quantile-seed1.json`
(`device: cpu`, M5, four threads; weight store 69 -> 71, one new fit per set;
15 Class B, 1 Class A; fingerprint `fe33b0f3`). Banked rows untouched.

| fold 0 | val-MSE | epochs (best) | floor | recall | MVGS | precision | rare-FA | nominal-step | honest lead |
|---|---|---|---|---|---|---|---|---|---|
| `m1-g8.9.10` LSTM seed 0 (banked) | 1.691e-4 | 14 (3) | 0.14681 | 4/15 | 4/11 | 7/8 | 1/12 | 0 | -77.0 (n=4) |
| **`m1-g8.9.10` LSTM seed 1** | **1.986e-5** | **35 (34)** | **0.02220** | **4/15** | 4/11 | 7/18 | 2/12 | **0** | -4.0 (n=4) |
| `m1-g8.9.10` GRU | 4.628e-6 | 35 (34) | 0.01324 | 11/15 | 8/11 | 13/31 | 1/12 | 79 | +0.0 (n=11) |
| `m1-ss5` LSTM seed 0 (banked) | 3.441e-5 | 22 (11) | 0.10195 | 4/13 | 4/10 | 13/15 | 1/12 | 121 | -41.0 (n=4) |
| **`m1-ss5` LSTM seed 1** | **3.462e-5** | **22 (11)** | **0.10172** | **4/13** | 4/10 | 13/15 | 1/12 | 121 | -41.0 (n=4) |
| `m1-ss5` GRU | 4.968e-6 | 35 (32) | 0.00878 | 11/13 | 8/10 | 19/89 | 2/12 | 958 | +0.0 (n=11) |

**The seven, under the reseeded LSTM** -- reach on `m1-g8.9.10` / `m1-ss5`:
`id_109` 0.933 / 0.207, `id_90` 0.921 / 0.203, `id_12` 0.921 / 0.203, `id_93`
0.917 / 0.200, `id_114` 0.890 / 0.196, `id_110` 0.870 / 0.194, `id_107` 0.796 /
0.173. **None caught on either set.** The four caught on each set are the four
the banked fit catches (`id_10, id_101, id_102, id_103`, reach 8.1-8.2 and
1.77). `id_89` 0.607 / 0.140.

| # | Predicted | Observed | Verdict |
|---|---|---|---|
| **P21** | no stall: >= 25 epochs, best >= 15, val-MSE <= 1.0e-5 | 35 epochs, best 34, **1.986e-5** | **Refuted on the number, held on the shape.** It ran to the cap and was still improving; it reached 2x the band's edge and 4.3x the GRU |
| **P22** | floor 0.010-0.020 | **0.02220** | **Refuted, narrowly** -- 11% above the band, 1.68x the GRU's |
| **P23** | >= 5 of the seven caught | **0 of 7**, at reach 0.80-0.93 | **Refuted.** Every one of them is within 7-20% of the bar |
| **P24** | rare-FA <= 2/12; nominal-step 20-300 | 2/12; **0** | Held on rare-FA; **refuted on nominal-step** -- a floor of 0.022 admitted nothing on 3.57M nominal steps where the GRU's 0.013 admitted 79 |
| **P25** | verdict rule | fitted but short of the GRU; floor followed; none of the seven | **The pre-registered "between" branch: no verdict.** |

**Mechanism condition (15.4): held on all three.** `best_epoch` 34 and 11; the
floor moved with the MSE on both sets (gate: MSE 8.5x better, floor 6.6x
lower; `m1-ss5`: both unchanged to 1%); nominal-step alarms 0 and 121, under
237. **Stop and report (15.5): none fired** -- no stall by the rule's
definition, not more than seven, 16 operations.

**What the run establishes, without the verdict it was built to give.**

1. **The banked fold-0 fit's stall was that path.** Seed 1 did not stop at
   epoch 14; it reached 1.99e-5, 8.5x better, and its floor fell 6.6x.
2. **And a second path still lands well short of the GRU on this fold.** The
   validation history is the mechanism: `4.2e-04, 1.9e-04, 2.4e-04, 1.9e-04, 2.9e-04, 2.6e-04, 2.1e-04, 2.0e-04, 3.3e-04, 2.0e-04, 2.7e-04, 3.6e-04, 2.4e-04, 1.8e-04, 2.0e-04, 2.1e-04, 1.8e-04, 2.0e-04, 3.3e-04, 1.8e-04, 1.6e-04, 1.8e-04, 1.2e-04, 3.9e-05, 2.4e-05, 2.6e-05, 4.1e-05, 2.2e-05, 3.9e-05, 2.5e-05, 2.2e-05, 4.1e-05, 2.4e-05, 2.1e-05, 2.0e-05`. A plateau at
   1.2e-04-3.6e-04 from epoch 6 to epoch 23, then a fall to 2.4e-5 at epoch
   24. Seed 0's best was at epoch 3 and the published patience of 10 ended it
   at 14, *on* that plateau; seed 1's small improvements along the plateau kept
   resetting patience until the fall. On `m1-ss5` neither seed left the plateau:
   `2.7e-04, 3.4e-04, 3.7e-05, 1.7e-04, 1.5e-04, 9.2e-05, 6.4e-05, 6.0e-05, 1.6e-04, 6.2e-05, 3.5e-05, 3.5e-05, 4.9e-05, 7.3e-05, 4.4e-05, 3.7e-05, 3.5e-05, 5.7e-05, 3.5e-05, 3.5e-05, 3.6e-05, 3.5e-05` -- stopped at 22 with the best at 11, twice, to three
   significant figures. **The LSTM's fold-0 optimisation is plateau-prone under
   the published protocol and the GRU's, on the same data, was not**: 4.6e-6
   and 5.0e-6 with the best epoch at the cap.
3. **The seven are a floor away, not a cell away.** Under the escaped LSTM path
   they sit at 0.80-0.93 of a floor of 0.0222; the GRU's floor is 0.0132. Their
   raw peaks under the reseed (reach times floor, 0.0177-0.0207) are above the
   GRU's floor and below the LSTM's.

**What it does not establish, stated so it is not read in.** Not that the cell
is the cause: two LSTM paths and one GRU path is not a distribution, and the
hardware differed (15.1). Not that a longer patience or a higher epoch cap
would carry the LSTM to the GRU's level -- both are published constants
(`patience = 10`, `epochs = 35`, section 1) and changing them is a protocol
change to both cells, a decision rather than a run. Not that `m1-ss5`'s
plateau would ever break. And nothing here moves a published row: the gate row
stays the seed-0 fit, and whether it should be the seed-1 fit -- which would
change `lstm-quantile`'s fold 0 from 4/15 at 0.147 to 4/15 at 0.022 and its
honest lead from -77.0 to -4.0, and nothing else -- is the decision this run
was meant to inform, recorded as open.


---

## 16. Pre-registration: `tcn-quantile` (work item 6)

**Written and committed before any fit.** Sections 14 and 15 are the template
and the record this is read against.

### 16.1 What is being tested, and the shape

The architecture, and nothing else. `tcn-quantile` is the frozen decision
layer (D25) on a temporal convolutional forecast: `Hyper(cell="tcn",
hidden=(50,)*6, kernel=3)`, every other field telemanom's and shared with both
cells -- 12 channels, 250-step training windows, `l_p = 10`, MSE, Adam 1e-3,
batch 70, up to 35 epochs, patience 10, `min_improvement = 0.001`, seed
`0 + fold`, the same sampler, validation pinning and guard, fitted by the same
`train` (D27).

**The shape, and why.** Bai, Kolter and Koltun's standard form: six residual
blocks of two causal dilated convolutions each, kernel 3, dilations 1, 2, 4, 8,
16, 32, ReLU and dropout 0.3 after each convolution, a 1x1 shortcut on the
first block, a final dropout before the same head. Receptive field
`1 + 2(k-1)(2^L - 1) = 253` -- the first standard shape at or past `l_s = 250`
(k=5, L=5 gives 249). Width 50 gives **91,670 parameters** against the LSTM's
91,640; 49 and 51 give 88,222 and 95,184. No weight normalisation (folded at
export; the file never carries it). PyTorch's default initialisation. **The
TCN is size-matched to the LSTM; the GRU was not** (71,160), and criterion 2
of Objective.md 8 is read with that in mind.

**The stateless contract, designed rather than inherited.** The forecast at
`t` is a function of inputs `t-252 .. t` and nothing else. `reference.forward`
refuses a state for `ConvWeights` and returns an empty one; the flight
component holds the last 252 inputs in a ring buffer that is zero-filled at
boot, and the causal zero padding in every convolution is that buffer. The
harness's chunked scoring warms every chunk with `window = 250` real steps and
carries nothing across chunks -- so the three steps a 253-field reaches past
the prefix are zero-filled at every chunk start, exactly as they are in every
250-step training window. Training and scoring see the same arithmetic; a
chunk warmed by 252 real steps reproduces the uncut stream to 1e-5 by test.

**Measured before the fit.** torch-vs-NumPy 4.8e-07 at the flown shape,
6.0e-07 on a 4,096-step chunk; the field pinned at exactly 253 by
perturbation; the future cannot reach the past; the LSTM's and GRU's
fingerprints and keys unmoved.

**What does not move.** The harness, folds, bundle, decision layer,
`error_buffer` (D21, irrelevant to the gate arm), `k`-of-`n` (D23), the
held-back sets. `tcn-telemanom` is scored in the same run from the same
weights as the non-gate reproduction reading.

### 16.2 Anchors

From `runs/m1-g8.9.10/lstm-quantile/2026-08-28T171349Z-2717441a.json`,
`.../gru-quantile/2026-08-28T222635Z-6d146f5d.json`, both fit reports,
`_forensics/2026-08-28T205347Z-head-to-head.json` (LSTM reaches),
`_forensics/2026-08-28T223610Z-head-to-head.json` (GRU reaches) and
`_forensics/2026-08-28T232538Z-reseed-lstm-quantile-seed1.json`.

| fold | LSTM val-MSE / floor | LSTM seed 1 (fold 0 only) | GRU val-MSE / floor | LSTM recall, MVGS | GRU recall, MVGS |
|---|---|---|---|---|---|
| `m1-g8.9.10` 0 | 1.691e-4 / 0.14681 | 1.986e-5 / 0.02220 | 4.628e-6 / 0.01324 | 4/15, 4/11 | 11/15, 8/11 |
| `m1-g8.9.10` 1 | 3.552e-6 / 0.01563 | -- | 3.378e-6 / 0.02827 | 10/15, 7/9 | 4/15, 4/9 |
| `m1-g8.9.10` 2 | 4.089e-6 / 0.01178 | -- | 3.468e-6 / 0.01455 | 12/16, 10/12 | 12/16, 10/12 |
| `m1-ss5` 0 | 3.441e-5 / 0.10195 | 3.462e-5 / 0.10172 | 4.968e-6 / 0.00878 | 4/13, 4/10 | 11/13, 8/10 |
| `m1-ss5` 1 | 5.205e-6 / 0.01243 | -- | 5.169e-6 / 0.01710 | 11/14, 7/9 | 3/14, 3/9 |
| `m1-ss5` 2 | 8.096e-6 / 0.01545 | -- | 6.120e-6 / 0.01022 | 12/15, 10/12 | 12/15, 10/12 |

Pooled: LSTM 26/46, 21/32, 2/48, 0.002%; GRU 27/46, 22/32, 1/48, 0.001%;
honest lead +0.0 both.

**The twelve `lstm-quantile` misses on `m1-g8.9.10`, and what the reach
arithmetic says a TCN needs.** Raw peak = reach x floor.

```
  fold  event    LSTM reach  GRU reach   LSTM-s1 reach   raw peak under LSTM-s0 / LSTM-s1 / GRU
  0     id_107   0.158       1.286       0.796           0.0232 / 0.0177 / 0.0170     the seven: back if the
  0     id_109   0.172       1.271       0.933           0.0253 / 0.0207 / 0.0168     fold-0 floor lands at
  0     id_110   0.181       1.233       0.870           0.0265 / 0.0193 / 0.0163     or below ~0.017-0.020
  0     id_114   0.147       1.275       0.890           0.0216 / 0.0198 / 0.0169
  0     id_12    0.168       1.269       0.921           0.0247 / 0.0204 / 0.0168
  0     id_90    0.168       1.269       0.921           0.0247 / 0.0204 / 0.0168
  0     id_93    0.150       1.249       0.917           0.0221 / 0.0204 / 0.0165
  0     id_89    0.111       0.393       0.607           0.0163 / 0.0135 / 0.0052     needs a floor below ~0.005-0.013
  1     id_132   0.995       0.553       --              0.0156 / -- / 0.0156         the coin toss, at the LSTM's floor
  1     id_138   0.480       0.181       --              0.0075 / -- / 0.0051         needs a fold-1 floor below ~0.006
  1     id_20    0.209       0.130       --              0.0033 / -- / 0.0037         needs a fold-1 floor below ~0.0035
  2     id_157   0.307       0.238       --              0.0036 / -- / 0.0035         needs a fold-2 floor below ~0.0036
```

**And the six the GRU lost on fold 1** (`id_122, id_124, id_129, id_130,
id_140, id_142`): LSTM reach 1.087-1.207 at floor 0.01563 -- raw peaks
0.0170-0.0189 -- caught by any fold-1 floor at or below ~0.017.

### 16.3 PREDICTED

| # | Prediction | Reasoning |
|---|---|---|
| **P26** | **91,670 parameters** (87,410 on six channels), receptive field 253 | Measured before the fit (16.1); recorded so criterion 2 was stated first |
| **P27** | **Training shape: no plateau.** On fold 0 of both sets `best_epoch >= 20` and validation MSE **<= 1.0e-5**; on every fold `best_epoch > 5` | Residual convolutional stacks have no recurrent state to saturate and their gradients reach every input tap at the same depth; the LSTM's fold-0 plateau (section 15) is a recurrent phenomenon the GRU did not show either. Trainability is now a recorded gate consideration (`docs/RESULTS.md` 6i) and this is the prediction that tests it |
| **P28** | **Validation MSE within 0.5x-2.0x of the GRU's on every fold, both sets** | Objective.md 8's hypothesis that TCNs equal recurrent models at this scale, with a wider band than P16's because a 253-step window and a recurrent memory are different instruments and the direction is not known |
| **P29** | **Noise floor: fold 0 within 0.7x-1.5x of the GRU's** (0.01324 / 0.00878); **folds 1 and 2 within 0.7x-1.5x of the LSTM's** (0.01563, 0.01178 / 0.01243, 0.01545) | Fold 0's floor followed the fit for both a GRU and a reseeded LSTM; on folds 1 and 2 the GRU's rise was a tail property of that cell's residual, and there is no reason on record to expect the TCN's tail to be the heavier one -- so the LSTM's floors are the anchor there |
| **P30** | `m1-g8.9.10`: **MVGS 21/32 to 25/32**, rare-event FA **<= 4/48** both sets, nominal-step FA **<= 0.006%**, pooled honest lead **+0.0** both sets. `m1-ss5` MVGS 19/31 to 24/31 | If P29 holds, the seven return (fold 0) and the GRU's six fold-1 losses do not recur: 21 + 4 headline from the seven = 25 at the top; losses elsewhere take it down |
| **P31** | **Per event, by the arithmetic in 16.2**: **five to seven of the seven return** on both sets (P29's fold-0 floor at or below their ~0.017-0.020 peaks); **`id_89`, `id_138`, `id_20`, `id_157` do not**; **`id_132`** predicted neither way; **at least four of the GRU's six fold-1 losses are caught** (a fold-1 floor near the LSTM's clears their 0.017-0.019 peaks) | Every clause is a reach against a predicted floor; each can fail on its own and says which floor moved |
| **P32** | **The floor follows the MSE across folds**: on every fold where the TCN's floor differs from the GRU's by more than 1.5x, its validation MSE differs in the same direction | Section 14 showed the two can decouple (fold 1). This is the claim that, for this architecture, they do not -- and it is the one most likely to be wrong |

**Not predicted.** Point recall, VUS-PR, `tcn-telemanom`'s numbers (reported
beside `lstm-telemanom` and `gru-telemanom`, not the gate).

### 16.4 Outcome condition, and mechanism conditions that are harder

**Outcome condition (parity with the better incumbent on each axis):** MVGS
>= 22/32 and rare-event FA <= 1/48 on `m1-g8.9.10`.

**Mechanism conditions -- read from the fit reports and the thresholds before
any event is scored; ANY failing is reported whatever the detection numbers
say:**

1. **No plateau**: no fold with `best_epoch <= 5` after ten or more epochs, on
   either set. A stall is P27's failure and the trainability finding of 6i
   recurring; it can coexist with a fine pooled row (the LSTM's did).
2. **The floor tracks the fit** (P32 as a condition): any fold whose floor is
   more than 1.5x from the GRU's must have its validation MSE on the same side
   of the GRU's. Failing it means a recovery or a loss on that fold is a tail
   effect, not a forecast effect, and 14.8.2's reading applies.
3. **Nominal-step alarms on the test window <= 3x the higher of the two
   incumbents' on every fold** (floored at 0.001% of the fold's nominal
   steps). Recoveries bought with alarms the 48 rare events cannot see are
   reported as such.
4. **Every recovered event is recovered by the floor**: its raw peak under the
   TCN divided by the TCN's floor is at least 1 on a fold whose floor is below
   the event's raw peak under the LSTM -- the same test 14.4's condition 2
   made, which every GRU recovery passed.

### 16.5 The run

| Step | What | Cost |
|---|---|---|
| Fit, rented GPU | `scripts/fit_folds.py --task m1-g8.9.10 --detector tcn-quantile --device cuda --threads 1 --determinism-check`, after `pod_setup.sh` and a green suite on the box; the `m1-ss5` fold-0 refit must be bit-identical (cuDNN convolutions under `use_deterministic_algorithms(True)`, not measured before) | ~16 operations |
| Verify, M5 | every file through the loader, `cell == "tcn"`, `best_epoch > 0`, torch-vs-NumPy on the real weights at 1e-5; the scoring run's fold 0 must hit the cache with refits refused before the pod is destroyed | 0 |
| Score, M5 | `run m1-g8.9.10 --detector tcn-quantile --detector tcn-telemanom --no-sweep`, one bundle load, both sets, all folds | 15 Class B, 1 Class A |
| Per event | `head_to_head.py --a lstm-quantile --b tcn-quantile` and `--a gru-quantile --b tcn-quantile` | 2 x (15 B, 1 A) |

Held-back sets untouched. Nothing refitted for the incumbents.

### 16.6 Stop and report

1. `tests/test_reference_equivalence.py` short of 1e-5 on the fitting box: no
   fit is banked.
2. **Any of the twelve returns** -- the headline, reported before anything
   else, with 16.4's condition 4 read beside it. (P31 predicts this will fire.)
3. **TCN MVGS at or above either incumbent's** -- 21/32 or 22/32 -- report and
   wait.
4. A fold with `best_epoch <= 5` after ten epochs: the trainability finding,
   reported as such.
5. Any fold's validation MSE above 3x the worse incumbent's.
6. More than 50 operations in any run.

### 16.7 OBSERVED

Beside the predictions. Artifacts
`runs/m1-g8.9.10/tcn-quantile/2026-08-29T162030Z-c48bd47d.json`,
`runs/m1-g8.9.10/tcn-telemanom/2026-08-29T162030Z-4c35b17a.json`,
`runs/_weights_pod/tcn-2026-08-29/fit_report.json`, and the two head-to-heads
in 16.7.2. Tables: `docs/RESULTS.md` 6j.

**The run.** Six fits on an A40-2Q slice in 3.5 minutes of GPU; the first
attempt crashed in the determinism *comparison* after both `m1-ss5` fold-0
fits had run (my code compared recurrent gates by attribute; fixed, commit
`f258762`, and 15 Class B from that attempt's bundle load are unrecorded
because the crash preceded the ledger commit); the rerun found the four
banked fits in the cache, refitted `m1-ss5` fold 0 once more **bit-identical**
under cuDNN convolutions, and fitted the last two. Every file loaded on the
M5, `cell == "tcn"`, receptive field 253, torch-vs-NumPy at most 9.5e-07;
scored with refits refused, weight store unchanged at 77.

#### 16.7.1 The predictions

| # | Predicted | Observed | Verdict |
|---|---|---|---|
| **P26** | 91,670 / 87,410 parameters, field 253 | as predicted | **Held** |
| **P27** | no plateau: fold 0 `best_epoch >= 20` and val-MSE <= 1e-5 both sets; `best_epoch > 5` everywhere | fold 0: 35 (33) at **1.972e-5** and 35 (33) at 7.501e-6; **`m1-ss5` fold 2: 13 epochs, best 2, 6.059e-5** | **Refuted twice** -- the gate-set fold 0 missed the band by 2x and one fold stalled outright |
| **P28** | val-MSE 0.5x-2x the GRU's, every fold | 4.26x / 1.66x / 3.91x; 1.51x / 1.25x / **9.90x** | **Refuted on four of six** |
| **P29** | fold-0 floor 0.7x-1.5x the GRU's; folds 1-2 0.7x-1.5x the LSTM's | fold 0 **1.63x** / **1.62x** the GRU's; fold 1 **2.49x** / 1.53x the LSTM's; fold 2 **2.18x** / **3.39x** the LSTM's | **Refuted on five of six** -- above the band every time it left it |
| **P30** | `m1-g8.9.10` MVGS 21-25/32; rare-FA <= 4/48; nominal-step <= 0.006%; honest lead +0.0 both sets; `m1-ss5` MVGS 19-24/31 | **9/32**; 3/48 and 3/48; 0.00002%; **-4.0** and -0.5; **13/31** | **Refuted on MVGS and lead**; held on rare-FA and nominal-step, which a high floor makes easy |
| **P31** | five to seven of the seven back, both sets; `id_89`, `id_138`, `id_20`, `id_157` not; `id_132` neither way; >= 4 of the GRU's six fold-1 losses caught | see 16.7.2 | see 16.7.2 |
| **P32** | the floor follows the MSE across folds | MSE above the GRU's on all six folds and the floor above the GRU's on all six; on the gate set 4.3x/1.7x/3.9x against 1.6x/1.4x/1.8x | **Held on every fold** -- the one written as most likely to be wrong |

**Not predicted, reported:** point recall 0/11 and 3/11; VUS-PR 0.367 and
0.607 (the second the highest of the three); `tcn-telemanom` in
`docs/RESULTS.md` 6j.4, where the TCN's residuals carry **30/32** under the
local rule at 4,085 ranges and 29/48.

**Why, as far as the artifacts say.** The TCN forecast this telemetry worse
than the GRU on every fold and its floor rose with the error, on every fold,
in proportion: for this architecture the tail quantile and the mean square
did not decouple. So the recoveries the GRU bought with a floor of 0.013 on
fold 0 were not available at 0.0215, and the six fold-1 events the LSTM catches
at 0.0156 were not available at 0.039. The size-matched, lookback-matched TCN
is a worse forecaster of this data than either cell at the published
protocol, and the row says so. Whether a larger or differently shaped
convolutional stack would not be is untested and is not claimed either way.

#### 16.7.2 The per-event line

Artifacts `runs/m1-g8.9.10/_forensics/2026-08-29T162907Z-head-to-head.json`
(`lstm-quantile` vs `tcn-quantile`) and `.../2026-08-29T162813Z-head-to-head.json`
(`gru-quantile` vs `tcn-quantile`); cached weights, weight store unchanged at
77, 15 Class B each. Reach is the peak score in the event span over the fold's
threshold.

**`m1-g8.9.10`: the TCN's nine are a strict subset of both incumbents' --
only-TCN 0 against both.** None of the twelve returns:

```
  fold  event    LSTM reach   GRU reach   TCN reach
  0     id_109   0.172        1.271       0.951      the seven, at 0.82-0.95 of a floor of 0.0215:
  0     id_90    0.168        1.269       0.940      the reseeded LSTM's 0.80-0.93 at 0.0222, again.
  0     id_12    0.168        1.269       0.940      Their raw peaks under the TCN, 0.018-0.020,
  0     id_93    0.150        1.249       0.927      are the GRU's 0.016-0.017 and the reseed's
  0     id_110   0.181        1.233       0.908      0.018-0.021; only the GRU's floor got under them
  0     id_114   0.147        1.275       0.868
  0     id_107   0.158        1.286       0.823
  0     id_89    0.111        0.393       0.556
  1     id_132   0.995        0.553       0.438
  1     id_138   0.480        0.181       0.291
  1     id_20    0.209        0.130       0.122
  2     id_157   0.307        0.238       0.258
```

Against the LSTM the TCN loses seventeen: the GRU's six on fold 1
(`id_122, id_124, id_129, id_130, id_140, id_142`, LSTM reach 1.09-1.21, TCN
0.43-0.52 -- a floor 2.5x higher) and **eleven on fold 2** (`id_149, id_150,
id_160, id_165, id_172, id_176, id_177, id_183, id_184, id_186, id_187`, LSTM
reach 1.38-1.48, TCN 0.67-0.85 -- a floor 2.2x higher). Against the GRU it
loses eighteen: the seven and the same eleven. **Trigger 2 did not fire on
the gate set.**

**`m1-ss5`: the seven return -- the same seven, at reach 1.17-1.23 against a
floor of 0.0142** (`id_107` 1.234, `id_109` 1.217, `id_12` 1.215, `id_90`
1.215, `id_93` 1.193, `id_110` 1.174, `id_114` 1.174; LSTM 0.17-0.21). That
floor is below their raw peaks under the LSTM (0.018-0.021), so 16.4's
condition 4 holds: **a recovery by the floor, on the one fold-and-set where
the TCN's floor got under them**, and the same events the GRU recovered
(14.8.2) -- nothing the GRU did not already reach. `id_89` (0.546),
`id_138`, `id_157` and `id_132` (0.870) do not return.

Two more only-TCN catches on `m1-ss5`, both fold 1, both **not by the
floor**: `id_142` (GRU 0.986 -> 1.022, the bar's edge) and `id_145` (GRU
0.884 -> **3.491**, footprint 6,300). The TCN's floor on that fold is above
the LSTM's, so these are residuals that grew under the event -- a forecast
effect, reported as such and not as support for the mechanism. And on
`m1-ss5` fold 2 the stalled fit loses all twelve the incumbents catch
(`id_149 ... id_187`, TCN reach 0.34-0.39 at a floor of 0.052).

**P31, adjudicated.** *Five to seven of the seven return on both sets*:
**refuted on the gate set (0 of 7), held on `m1-ss5` (7 of 7)**. *`id_89`,
`id_138`, `id_20`, `id_157` do not return*: **held**. *`id_132`*: neither way,
missed. *At least four of the GRU's six fold-1 losses caught*: **refuted** --
none on the gate set, one (`id_142`) on `m1-ss5`. **Trigger 2: fired on
`m1-ss5` by the letter -- seven of the eleven -- and they are the GRU's seven,
recovered by the same mechanism on the same fold; not fired on the gate set,
where the twelve are defined.**

#### 16.7.3 The conditions

**Outcome condition (parity): FAILED** -- MVGS 9/32 against 22/32, rare-FA
3/48 against 1/48.

**Mechanism condition 1, no plateau: FAILED** on `m1-ss5` fold 2 (best epoch
2, stopped at 13). **Condition 2, the floor tracks the fit: HELD** on every
fold. **Condition 3, nominal-step alarms within 3x the higher incumbent's:
HELD** on every fold (2 / 1 / 1 and 788 / 31,473 / 0 against caps of 237 /
207 / 435 and 2,874 / 94,821 / 1,380). **Condition 4, recovery by the
floor:** read per event in 16.7.2.

#### 16.7.4 Stop and report

Trigger 1 did not fire (9.5e-07). **Trigger 4 fired** (`m1-ss5` fold 2) and
**trigger 5 fired** (gate fold 2 at 3.3x the LSTM's MSE; `m1-ss5` fold 2 at
7.5x); both were reported from the fit reports before any scoring. Trigger 3
did not fire. Trigger 2 is adjudicated in 16.7.2. Trigger 6 did not fire (16
operations per run). Nothing is decided here.


---

## 17. Scoping the two-forecaster combination (post-gate; scope, not build)

**Written and committed before the numbers are computed.** Section 14.7 banked
this question with two preconditions: all three rows exist, and the gate
leaves events on the table. Both hold (sections 6h, 6i, 6j of
`docs/RESULTS.md`). This is a scoping measurement on cached weights and
banked artifacts, run after the architecture gate and never inside it. It
lands, if it ever lands, as its own decision after Phase 1 closes. Nothing
here is a detector; no registry entry, no arm, no published row moves.
Held-back sets untouched.

### 17.1 The nesting check, first

The whitened precedent (D25): two rules that are nested close the combination
question by inspection, because an OR equals the superset. Read from
`_forensics/2026-08-28T223610Z-head-to-head.json`:

```
  m1-g8.9.10   both 20   only lstm-quantile 6   only gru-quantile 7   neither 13   NOT NESTED
  m1-ss5       both 19   only lstm-quantile 8   only gru-quantile 7   neither  8   NOT NESTED
```

Six and seven, eight and seven: the two cells catch different events. The
question is open, and the rest of this section is what it costs.

### 17.2 What is computed, and from what

One local pass, cached weights, one bundle load, both sets, all folds, both
detectors scored through the harness's own path. For every timestep the two
**normalised** scores `r_L = score_L / threshold_L` and `r_G = score_G /
threshold_G` -- reach as a series, 1.0 the bar each rule already uses. Three
combination rules, each a one-line function of the two series and nothing
else, each scored by the referee's own metric code on its alarm mask:

| rule | mask | what it is |
|---|---|---|
| **OR** | `max(r_L, r_G) >= 1` | both alarm streams pass through: the plain union, and its whole false-alarm exposure |
| **AND** | `min(r_L, r_G) >= 1` | agreement: both over their own bars at the same moment |
| **MEAN** | `(r_L + r_G) / 2 >= 1` | a combined score: one can carry the other part of the way |

Per anomaly: caught by each rule; each model's peak reach in the span; and
**the other model's reach at the moment of this model's peak** -- the
concurrence reading, which the banked per-model peaks cannot give because two
peaks need not coincide. Per rare nominal event: the same two reaches and
which rules alarm on it. Per fold: recall, headline-cell recall, rare-event
FA, nominal-step FA and alarm-range count for every rule, `k/n`.

**Not a mission-available quantity, said plainly.** The three rules reuse
each model's own calibrated bar and add no fitted constant, so a mission
could compute them; what a mission could not do is *choose among them* on
these 46 events, and this section chooses nothing.

### 17.3 Flight cost, stated before the recall (facts, not predictions)

From the array shapes in section 3, per rate-group tick, multiply-accumulates
of the recurrence and head:

```
                     LSTM      GRU       both      ratio to LSTM alone
  MAC / tick        90,240   70,080   160,320     1.78x
  parameters        91,640   71,160   162,800   1.78x
  float32 payload   358.0 KiB  278.0 KiB  636.0 KiB
  carried state     2 x 80 x 2 = 320 floats   2 x 80 = 160   480
  EWMA state        12         12        24 (one per channel per model)
  thresholds        1          1         2, plus the rule's own constant (1.0 -- none fitted)
```

`model.bin` carries two weight sets, two gate orders, two state shapes and two
calibrated thresholds; the verification surface is two reference paths (both
exist, both held at 1e-5), two equivalence tests, and the combination rule.
Objective.md 8's criteria 2 to 5 are all worse than either cell alone by
construction; whatever recall is gained is bought against that.

### 17.4 PREDICTED

From the banked head-to-head and scorecards
(`lstm-quantile/2026-08-28T171349Z-2717441a.json`,
`gru-quantile/2026-08-28T222635Z-6d146f5d.json`):

| # | Prediction | Reasoning |
|---|---|---|
| **E1** | **OR: recall 33/46, MVGS 25/32 on `m1-g8.9.10`; 34/42, 25/31 on `m1-ss5`** -- exactly the union of the banked catches | By construction of the masks; a departure means a scoring difference from the banked runs and is reported as a defect |
| **E2** | **OR rare-event FA: 2/48 or 3/48 on the gate set** (LSTM 2, GRU 1; the GRU's one is likely among the LSTM's two), **3/48 to 5/48 on `m1-ss5`**; **OR nominal-step: at most the sum**, 356 of 10,675,488 on the gate set (0.0033%) and 33,365 on `m1-ss5`, and above the larger of the two on every fold where both fire | Alarm sets union; the exact overlap is what the pass measures |
| **E3** | **AND: recall 20/46 on the gate set, 19/42 on `m1-ss5` -- the intersection, and no solo event**; rare-event FA at most the smaller incumbent's (1/48; 2/48) | Agreement can only lose events; it buys precision |
| **E4** | **MEAN on the gate set catches none of the thirteen solo events.** The GRU's seven sit at 0.15-0.18 under the LSTM (mean at most 0.73); the LSTM's six at 0.60-0.62 under the GRU (mean at most 0.91). **On `m1-ss5` MEAN catches most of the fifteen solo events** -- the banked peak-reach means are all above 1 -- **but fewer than fifteen once concurrence is read at the same timestep** | The peak-reach mean is an upper bound: the two peaks need not coincide |
| **E5** | **Concurrence, gate set: at the LSTM's peak on the GRU's seven, the LSTM's own reach is below 0.25; at the GRU's peak on the LSTM's six, the GRU's own reach is below 0.7.** Each solo catch is invisible or half-visible to the other model at the moment it matters | If concurrence is higher than the banked peaks say, a combined score has something to use; if lower, it has less |

**The verdict rule, set now.** If E4 and E5 hold on the gate set, the
combination is **an OR and nothing cleverer** -- its recall is the union and
its cost is the union's false alarms, and a combined score has no concurrence
to exploit on the events that matter. If MEAN catches five or more of the
thirteen gate-set solo events while its rare-event FA stays at or below the
LSTM's 2/48, a combined score is real and is scoped further. Anything between
is reported without a verdict.

### 17.5 Stop and report

1. E1 fails -- the pass does not reproduce the banked catches. A scoring
   defect; nothing downstream is readable.
2. More than 50 operations.
3. Anything that would touch a published row, the frozen layer or the
   held-back sets.

### 17.6 OBSERVED

*Filled in after the pass, beside the predictions.*
