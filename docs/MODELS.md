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

## 3. What this establishes for Objective.md 14.2 (the model-file format)

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

**(!) CORRECTED 2026-09-01 (D37, section 21).** That floor was measured with a
defect: `baselines._rolling` accumulated its prefix sums in float32 and lost the
statistic. Re-scored, the floor is **F0.5 = 0.676**, from recall **34/46** and
precision 84/127. The predictions below were made against 0.250 and 3/32 and are
left exactly as they were written, because that is what a pre-registration is;
the OBSERVED table restates the verdicts that the correction changes.

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
| **F0.5 on `m1-g8.9.10` clears `rstd`'s 0.250** | **0.269** | **0.026** | **Held, then lost -- and on the corrected floor it never held.** It cleared 0.250 by 0.019 with a one-epoch model and fell an order of magnitude below it once the model was trained. **(!) CORRECTED 2026-09-01 (D37): the floor is 0.676, so 0.269 never cleared it.** The floor is a two-line rolling standard deviation, and it is a better one than this project measured |
| **Headline-cell recall above `rstd`'s 3/32** | 28/32 | 28/32 | **Held against the number as it stood, and the number was wrong.** The verdict here read: "Held, decisively, and it is the project's thesis. A per-channel statistic finds three; a forecaster over the channel set finds twenty-eight, at 0.231 and 0.021 precision respectively." **(!) CORRECTED 2026-09-01 (D37): a per-channel statistic finds twenty-five of thirty-two, not three.** The 3/32 was an artifact of `_rolling`'s float32 accumulation. The forecaster's 28/32 is unchanged and was not recomputed, but it is no longer evidence that a per-channel statistic cannot see the cell, because it can. What survives is stated in `docs/RESULTS.md` 1a: on the gate metric this project declared in advance, the forecaster still wins, and it wins by reaching comparable recall at a third of the alarm rate |
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

### 10.8 The retest: what ran was not what was designed

**Written before the retest runs.** Section 10.7 records the first result and is
not edited. This section says what is being retested and why the first attempt
does not settle the design.

#### 10.8.1 The first run failed on an arithmetic error of mine, not on the design

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

#### 10.8.2 Three arms

| Arm | What it is | Why |
|---|---|---|
| `independent` | the fit that ran, preserved | the corrected form is measured against it, not in place of it |
| `joint` | both terms fitted so **the maximum** admits the target | the design as intended. The honest retest |
| `local_only` | the local order statistic **with no floor** | **it has never operated.** The floor dominated every window of the first run, so there is not one measurement of the local term |

#### 10.8.3 PREDICTED, and `local_only` is a new arm so it is pre-registered like one

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

#### 10.8.4 What is reported

Every arm, at every admission rate, **per fold, per channel set**, `k/n`
throughout: **binding rate** first, then recall, headline-cell recall, point
recall, precision, rare-event false alarms, alarm ranges, and **lead time**.
`lstm-telemanom` is the control and is unchanged.

#### 10.8.5 Stop and report

1. **`joint` also collapses to the global rule.** That closes D20, and the next
   step is a decision rather than another variant.
2. **`local_only` holds up without a floor.** That is a *different design* from
   the one proposed and needs its own decision before being pursued.
3. **Median lead negative on any arm at any rate.** D9 disqualifies it whatever
   else it scores.
4. **Non-monotonic response to the admission rate.** The calibration is not doing
   what it claims and nothing downstream of it can be read.

#### 10.8.6 OBSERVED -- the retest

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

#### 10.8.7 (!) The falsification condition was badly designed, and that is the finding

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
residual on whichever channels became unpredictable -- that is why **22 of 32**
headline-cell events are caught (was "28 of 32", `lstm-telemanom`'s figure;
D37, D38, `docs/RESULTS.md` 6l). The mechanism is unchanged; the number is
the flying detector's. Everything after that point
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
  improve detection without satisfying Objective.md section 11 rule 4.

**B. Lagged pair map -- the explanation layer Objective.md section 7 promises.** For each pair
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
`error_buffer`, pruning disabled as in `docs/RESULTS.md` 6d.

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
| `lstm-whitened` | the global rule at `pruning_p = 0`, giving authoritative scored numbers for the default adopted in `docs/RESULTS.md` 6d rather than paying for a separate pass |
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
column is at `pruning_p = 0` and confirms `docs/RESULTS.md` 6d's free win: `m1-ss5`
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

Artifact `runs/m1-g8.9.10/_forensics/2026-08-29T170340Z-combination-scope.json`;
cached weights, weight store unchanged at 77, 15 Class B, 1 Class A. Every
rule is the referee's own metric code on a mask; `k/n` throughout; all
per-fold denominators under 20.

**`m1-g8.9.10`**

| rule | recall | MVGS | rare-FA | nominal-step FA | alarm ranges |
|---|---|---|---|---|---|
| `lstm-quantile` | 26/46 | 21/32 | 2/48 | 214 / 10,675,488 (0.0020%) | 42 |
| `gru-quantile` | 27/46 | 22/32 | 1/48 | 142 (0.0013%) | 157 |
| **OR** | **33/46** | **25/32** | **2/48** | **293 (0.0027%)** | 60 |
| AND | 20/46 | 18/32 | 1/48 | 63 (0.0006%) | 139 |
| MEAN | 20/46 | 18/32 | 2/48 | 129 (0.0012%) | 49 |

**`m1-ss5`**

| rule | recall | MVGS | rare-FA | nominal-step FA | alarm ranges |
|---|---|---|---|---|---|
| `lstm-quantile` | 27/42 | 21/31 | 2/48 | 31,829 / 10,875,689 (0.293%) | 130 |
| `gru-quantile` | 26/42 | 21/31 | 3/48 | 1,536 (0.014%) | 172 |
| **OR** | **34/42** | **25/31** | 3/48 | 33,025 (0.304%) | 162 |
| AND | 19/42 | 17/31 | 2/48 | 340 (0.003%) | 140 |
| **MEAN** | **34/42** | **25/31** | **2/48** | 25,983 (0.239%) | 148 |

#### 17.6.1 The per-event union (question 1)

**Gate set: OR catches thirteen events neither incumbent catches alone** --
the GRU's seven on fold 0 (`id_107, id_109, id_110, id_114, id_12, id_90,
id_93`; four headline-cell) and the LSTM's six on fold 1 (`id_122, id_124,
id_129, id_130, id_140, id_142`; two headline-cell) -- for 33/46 and 25/32.
**The honest cost of the plain OR: rare-event FA 2/48, unchanged** -- the
GRU's one rare alarm (`id_1`, footprint 20,190, both models over their bars:
LSTM 1.38, GRU 3.21) is one of the LSTM's two; the other (`id_34`, LSTM 4.07,
GRU 0.92) is the LSTM's alone -- and **nominal-step alarms 293 against 214
and 142**, 0.0027%: more than either, less than their sum by the 63 they
share. Sixty alarm ranges against 42 and 157.

**`m1-ss5`: OR catches fifteen solo events** (the same seven, plus the
LSTM's eight on fold 1 including `id_132` and `id_145`) for 34/42 and 25/31,
at **rare-FA 3/48** -- the GRU's third (`id_17`, GRU 1.22, LSTM 0.30) comes
through -- and 33,025 nominal steps, 0.30%, almost all of it the LSTM's fold
1.

#### 17.6.2 The agreement reading (question 2)

Concurrence read at the moment of each solo catch's own peak:

```
  m1-g8.9.10   the GRU's seven (fold 0):  GRU peak 1.23-1.29   LSTM at that moment 0.15-0.18
               the LSTM's six (fold 1):   LSTM peak 1.09-1.21  GRU at that moment  0.60-0.62
  m1-ss5       the GRU's seven (fold 0):  GRU peak 1.87-1.92   LSTM at that moment 0.17-0.21
               the LSTM's eight (fold 1): LSTM peak 1.27-2.05  GRU at that moment  0.88-0.99
```

The peaks coincide with the banked per-model peaks in every case -- these are
footprint-1 and short events, and both models peak on the same step -- so the
banked reaches were already the concurrence.

**On the gate set, agreement is a fantasy.** At the moment the GRU catches
the seven, the LSTM is at a sixth of its own bar; at the moment the LSTM
catches the six, the GRU is at three fifths of its. **MEAN catches none of
the thirteen** (20/46, 18/32 -- the intersection with two rare alarms), and
AND is the intersection by construction. No combined score built from these
two series recovers a solo event on the gate set without lowering a bar,
which is the loosening D24 already priced at the rare-event rate.

**On `m1-ss5`, agreement is real, and it is the better rule there.** The GRU
is at 0.88-0.99 of its bar on every one of the LSTM's eight, and its own
peaks on the seven are 1.9, so **MEAN catches all fifteen solo events -- 34/42,
25/31 -- at 2/48 and 0.239% nominal, better than both incumbents on every
axis of that set**. The six-channel view is a different regime
(`docs/HARNESS.md` section 2): the same events register as spikes both
models see at once. It does not transfer to the twelve-channel gate set, on
which the project's claim rests.

#### 17.6.3 Flight cost beside the recall gained (question 3)

Section 17.3: 1.78x the multiplies per tick, 636 KiB of weights, 480 floats
of state in two shapes, two thresholds, two reference paths and two
equivalence tests, plus a combination rule with its own constant. For that,
on the gate set: **+7 events and +4 headline-cell (21/32 to 25/32) at the
same 2/48 and 79 more nominal-step alarms in ten million** -- as an OR, and
only as an OR.

#### 17.6.4 The expectations

| # | Predicted | Observed | Verdict |
|---|---|---|---|
| **E1** | OR 33/46, 25/32; 34/42, 25/31 | exactly | **Held** -- the pass reproduces the banked catches |
| **E2** | OR rare-FA 2-3/48 gate, 3-5/48 `m1-ss5`; nominal at most the sum and above the larger | 2/48 and 3/48; 293 (sum 356, larger 214) and 33,025 (sum 33,365, larger 31,829) | **Held** |
| **E3** | AND = intersection, no solo event; rare-FA at most the smaller | 20/46, 19/42; 1/48, 2/48 | **Held** |
| **E4** | MEAN catches none of the thirteen on the gate set; on `m1-ss5` most but fewer than fifteen | none; **all fifteen** | **Held on the gate set, refuted on `m1-ss5`** -- the peaks coincide, so the upper bound was the value |
| **E5** | LSTM below 0.25 at the GRU's seven; GRU below 0.7 at the LSTM's six | 0.15-0.18; 0.60-0.62 | **Held** |

**Verdict, by the rule set in 17.4: on the gate set the combination is an OR
and nothing cleverer.** Its recall is the union and its cost is the union's
false alarms -- which, measured, is the LSTM's own rare-event rate and a
third more nominal-step alarms -- and a combined score has no concurrence to
exploit on the events that matter. `m1-ss5`'s MEAN result is recorded as
what it is: a real agreement effect in the regime where both models see the
same spike, and not the gate set's.

**What this does not decide.** Whether +4 headline-cell events on one
spacecraft are worth 1.78x the flight compute and a second reference path is
Objective.md 8's criteria 2 to 5 against criterion 1, and it is a decision
for after Phase 1 closes, taken on the held-back transfer numbers that do not
yet exist. Nothing is built. Stop rules: none fired (E1 held; 16 operations).

---

## 18. The Phase 1 closure: the held-back sets, scored once

**Proposed and committed before any seal breaks; executed only on an explicit
GO, in so many words, after the predictions in 18.4 have been reviewed.**
Nothing in this section has loaded, listed, described or consulted `m2-ss1`
or `m1-g3`. The architecture is decided (D28) on Mission 1 evidence before
this runs, so that what follows is an exam and not a tuning set.

### 18.1 What runs, exactly once, settings as they stand today

```
python -m sentinel_eval --verbose run m2-ss1 --detector rstd --detector mavg \
    --detector lstm-quantile --detector gru-quantile --detector tcn-quantile \
    --detector lstm-gru-or --no-sweep

python -m sentinel_eval --verbose run m1-g3 --detector rstd --detector mavg \
    --detector lstm-quantile --detector gru-quantile --detector lstm-gru-or --no-sweep
```

- `m2-ss1`: all five detectors Objective.md section 13 item 7 names, plus
  the union -- "so the adoption number on an independent spacecraft is a
  comparison and not a lone figure". The TCN's row here is the
  independent-spacecraft verdict on the stateless question, one cheap fold.
- `m1-g3`: the two cells, the union, and the two free floor rows. No TCN
  on the recall exam: its Mission-1 rows closed that question (6j).
- One bundle load per set; `paired_with` returns each id alone, `partial`
  is False, the artifacts are not barred. `--no-sweep`: the oracle stays
  refused (section 7). **No `describe`** -- it loads a bundle and has no
  held-back guard. No second run, no variant, no second attempt.
- The union is listed after its members so its fits are cache hits inside
  the process; it is the OR mask of section 17 and nothing cleverer
  (`detectors.UnionQuantile`, pinned on the fixture by
  `tests/test_union_detector.py`).

### 18.2 What is fitted, and how -- the recipe as a mission would run it

Every setting is the frozen one: the published telemanom protocol for each
cell (section 1), `Hyper(seed=0)` plus the fold index, the frozen decision
layer (D25) with its label-free 99.9th percentile of each detector's own
anomaly-masked fitting window, persistence 1, agreement 1.

| set | channels | grid | split | fits per trained detector | calibration |
|---|---|---|---|---|---|
| `m2-ss1` | 12: `channel_9`-`channel_20`, Mission 2 `subsystem_1`, groups 5/8/9/10 | ~6.13M steps at **18 s** (timesteps, never hours; the grid period differs from Mission 1's 30 s) | `chronological(fraction=0.30)`: one fold, train the first ~1.84M steps, test the last ~4.29M | 1 each (LSTM, GRU, TCN) | each detector's own fitting window |
| `m1-g3` | 8: `channel_12, 13, 19, 20, 27, 28, 36, 37`, Mission 1 `subsystem_6` group 3 | 14,728,316 steps at 30 s | `forward_chaining(seed 0.25, 3 folds)`: train 3.68M / 7.36M / 11.05M | 3 each (LSTM, GRU) | per fold, as on the gate set |

`m2-ss1` has recall disabled at the task level: its scorecard carries the
rare-event false-alarm rate, the nominal-step rate, alarms per 1,000 nominal
timesteps and the training blocks; F0.5, lead time and VUS-PR are absent by
design (`harness._pool`). `m1-g3` carries the full scorecard. `check()`
refuses `m1-g3` after the load if fewer than five anomalies fall test-side
(`splits.MIN_TEST_ANOMALY_EVENTS`); that refusal would be the finding.

**Every headline-cell figure on `m1-g3` is UNDERPOWERED by the project's own
rule**: 14 events, `n < 20`. The powered exam is `m2-ss1`'s adoption number,
~600 rare events. The plan says so before the numbers exist.

### 18.3 Cost, and why local

| run | Class B | Class A | wall clock on the M5 |
|---|---|---|---|
| `m2-ss1` | 16 (manifest, labels, 12 channels, ledger) | 1 | ~35-45 min: three fits on ~1.84M steps (~5 min each), scoring ~6.1M steps for five arms, the union reading cached errors |
| `m1-g3` | 11 (manifest, labels, 8 channels, ledger) | 1 | ~2.5-3 h: six fits (LSTM ~11 / 25 / 35 min, GRU similar), then scoring -- overnight |
| **total** | **27** | **2** | |

No pod, no credential exposure, no flag lifted on any held-back guard: the
fits happen inside the scoring run, through the one tested path. `cli.py`
reads the ledger at run start and commits it at run end -- before the
artifact is written, a known ordering (harness territory, D8); the weights
are on disk inside `fit`, so a ledger failure could cost the scorecard but
never a fit. Operations are reported from the artifact.

### 18.4 PREDICTED -- committed before the run, reviewed before GO

Anchors, all Mission 1: rare-event FA `rstd` 1/48, `mavg` 10/48, LSTM 2/48,
GRU 1/48, TCN 3/48, union 2/48; nominal-step FA LSTM 0.002%, GRU 0.001%,
TCN 0.00002%, union 0.0027%, `mavg` 1.796 alarms per 1,000 nominal steps;
headline-cell recall LSTM 21/32, GRU 22/32, union 25/32; honest lead +0.0 for
both cells; the stall record (6i: LSTM fold 0; 6j: TCN `m1-ss5` fold 2);
`docs/RESULTS.md` 2, 6h, 6i, 6j and section 17.6.

**`m2-ss1` -- the adoption number on a spacecraft nothing here was tuned on**

| # | Prediction | Holds | Fails | No verdict |
|---|---|---|---|---|
| **T1** | Each trained detector's rare-event FA rate transfers: LSTM 2-8%, GRU 1-6%, TCN 1-6% | rate <= 12.5% (3x the Mission-1 rate, floored at 2/48) | rate > 20.8% -- `mavg`'s Mission-1 rate, the "muted within a week" line (Objective.md 11 rule 2; section 4 trigger 3) | 12.5% to 20.8% |
| **T2** | Nominal-step FA: LSTM, GRU, TCN each <= 0.05% | <= 0.05% (25x the Mission-1 rates) | > 0.5% -- the collapse level of 6d/6e | between |
| **T3** | The floor rows keep their Mission-1 order: `rstd` rare-FA <= 5%, `mavg` 15-35% and worse than every forecaster | `mavg` worse than all three cells on rare-FA | `mavg` at or better than any cell -- the Mission-1 ordering did not transfer, reported first | -- |
| **T4** | The union's cost transfers: rare-FA of `lstm-gru-or` <= 1.25x max(LSTM, GRU), and nominal-step <= LSTM + GRU | both | rare-FA > 1.5x max(LSTM, GRU) | between. (Its recall edge cannot be read here; that is R3) |
| **T5** | Trainability on new data: no fit with `best_epoch <= 5` after 10 or more epochs. The training window, ~1.84M steps, is half of Mission-1 fold 0's, the regime where the LSTM plateaued: GRU and TCN predicted clean; LSTM predicted to land >= 3x the GRU's validation MSE without a hard stall | no stall in any of the three | a stall in the GRU | an LSTM or TCN stall -- **reported as the finding, never refit** |
| **T6** | The stateless question on an independent spacecraft: TCN rare-FA and nominal-step within 3x the GRU's | both | either above 10x | between 3x and 10x |

**`m1-g3` -- the recall exam, underpowered by construction (14 headline-cell events)**

| # | Prediction | Holds | Fails | No verdict |
|---|---|---|---|---|
| **R1** | Headline-cell recall transfers: LSTM and GRU each 8-11/14 (Mission-1 rates 0.66-0.69) | each >= 8/14 | either < 6/14 | 6-7/14 |
| **R2** | Rare-event FA on `m1-g3`'s rare events (n unknown until scored): both cells <= 12.5% | <= 12.5% | > 20.8% | between |
| **R3** | **The union's edge -- the specific claim under test (+4/32 on the gate set).** Union headline-cell >= max(LSTM, GRU) + 1 at rare-FA <= max(LSTM, GRU) + 1 event | survives | union headline-cell = max(LSTM, GRU): evaporates | anything else. **At n = 14 the resolution is one event in fourteen; the verdict is written UNDERPOWERED** |
| **R4** | Honest lead, median: LSTM and GRU >= -10 | >= -10 | < -50, the D9 zone | between |
| **R5** | Event-wise F0.5: LSTM >= 0.6, GRU >= 0.5 (Mission-1 0.838, 0.804) | both | either < 0.3 | between |
| **R6** | Trainability: GRU no stall on any fold; LSTM fold 0 (3.68M steps, the stall regime) is not predicted either way | GRU clean | a GRU stall | an LSTM stall -- reported, never refit |

**Deliberately not predicted.** Point recall. VUS-PR. Anything about a spike
regime on `m1-g3`: it has zero sub-grid-cell events by its nomination.

### 18.5 The standing rule, verbatim (section 6)

> **Both sets are run once, at the end, with every setting already frozen. If the
> result disappoints, we do not go back and re-tune.**
>
> Tuning against a held-back set converts it into another training set and
> destroys the only clean evidence the project has. There is no version of "we
> adjusted it slightly after seeing the held-back number" that preserves the
> guarantee.
>
> **A disappointing held-back result is the finding, not a problem to fix.** It
> would mean the tuning fitted 46 events rather than building a better detector --
> which is worth knowing, and worth publishing, and is the entire reason for
> nominating a set in advance.

Restated for this run: one run per set; no adjusted threshold; no refit of a
stalled fold; no second seed; no "just checking one thing"; no `describe`. A
`SplitTooThin` refusal on `m1-g3` is the finding too. The value of the sealed
data is spent in one transaction and cannot be refunded.

### 18.6 What the result decides, and the one thing it does not

- **It does not reopen D28.** The architecture was decided on Mission 1
  evidence before the seal broke, precisely so that this stays an exam.
- **It decides the recommended deployment configuration -- D29.** If the
  union's edge survives (R3) and its cost holds (T4), the union is the
  recommended configuration at its stated cost (1.78x the multiplies, 636
  KiB, two thresholds, two reference paths). If the edge evaporates or the
  cost fails, `gru-quantile` flies alone. A no-verdict outcome is written as
  one, and D29 says which evidence is missing and that R3 was underpowered.
- **It is the transfer evidence for the whole recipe** -- the number that
  goes in front of a mission when they ask how we know it works on a
  spacecraft we never saw. T1 is that number.

### 18.7 After GO

`m2-ss1` first, then `m1-g3` overnight. Then, in order: `docs/RESULTS.md`
6k with every row of both sets, per fold and pooled, `k/n`, UNDERPOWERED
stamped, a provenance row (git commit, device M5, torch 2.13.0, seed 0,
operations read from the artifacts); 18.8 OBSERVED with every prediction
beside its outcome; D29 in the same commit as the decision; `docs/NARRATIVE.md`
10; CHANGELOG; Objective.md section 13 item 7 and the Phase 1 gate line; the
`wi7` tag and Release, "Phase 1 closed"; and the Phase 2 inheritance note --
both reference blueprints (LSTM, GRU) at 1e-5, the format facts (gate
orders, the GRU's unsummed bias vectors, one state vector against two,
thresholds outside the weights per Objective.md 14.10), the frozen decision
layer and its calibration recipe, the union's status per D29, and the
decisions still open (D14, D21's first reach, D23).

### 18.8 OBSERVED

Beside the predictions, once. `m2-ss1` scored 2026-08-29 20:25-20:44Z, `m1-g3`
20:45-22:36Z; each set loaded once, through `python -m sentinel_eval run`,
with the commands of 18.1 verbatim; no `describe`, no second attempt, nothing
refitted. Artifacts `runs/m2-ss1/<detector>/2026-08-29T204415Z-*.json` and
`runs/m1-g3/<detector>/2026-08-29T223625Z-*.json`; git `0c5fbfb`; operations read
from the artifacts: `m2-ss1` 15 Class B, 2 Class A; `m1-g3` 11 Class B, 2 Class A.

#### 18.8.1 `m2-ss1` -- what the exam set actually held

The Mission-2 bundle: 12 channels, **6,129,598 steps at 18 s**; the split put
1,838,879 steps train-side (1,835,997 usable; 2,882 anomaly steps and 179
rare events removed) and 4,290,719 test-side, of which **424 rare nominal
events** and 3 anomalies (1 headline-cell) -- so the adoption denominator is
424, not the ~600 dataset-wide figure the nomination quoted, and every rare-FA
figure below is `k/424`, resolution 0.24%.

| detector | rare-event FA | nominal-step FA | alarm ranges on nominal time | threshold | training |
|---|---|---|---|---|---|
| `rstd` | 84/424 (19.81%) | **718,831 / 4,155,841 (17.30%)** | 73 | 3.690 | -- |
| `mavg` | **122/424 (28.77%)** | 0 | 0 | 25.14 | -- |
| `lstm-quantile` | **4/424 (0.94%)** | 0 | 0 | 0.2100 | 35 (29), 6.218e-5 |
| `gru-quantile` | **4/424 (0.94%)** | 0 | 0 | 0.4368 | 19 (8), 7.161e-5 |
| `tcn-quantile` | 6/424 (1.42%) | 0 | 0 | 0.3085 | 26 (15), 6.889e-5 |
| `lstm-gru-or` | **8/424 (1.89%)** | 0 | 0 | 1.0 | both members' |

| # | Predicted | Observed | Verdict |
|---|---|---|---|
| **T1** | each cell's rare-FA transfers: LSTM 2-8%, GRU 1-6%, TCN 1-6%; holds <= 12.5% | **0.94%, 0.94%, 1.42%** | **Held**, all three -- and below every predicted band. Four rare events in 424 is one in a hundred |
| **T2** | nominal-step FA <= 0.05% for each cell | **0 of 4,155,841** for all three | **Held** -- not one nominal step alarmed, on a spacecraft none of them was tuned on |
| **T3** | floor rows keep order: `rstd` <= 5%, `mavg` 15-35% and worst | `mavg` 28.77%, worst; **`rstd` 19.81%**, and **17.30% of nominal steps** | **Held on the ordering, refuted on `rstd`**: the Mission-1 floor is a carpet-bomber on Mission 2 -- its quantile threshold, taken on a training window a third the length, admits a sixth of all nominal time |
| **T4** | the union's cost transfers: rare-FA <= 1.25x max(LSTM, GRU); fails > 1.5x | **8/424 = 4 + 4**: the two cells' rare alarms are **disjoint**, and the union pays the full sum; 2.0x max | **FAILED.** On Mission 1 the GRU's one rare alarm was inside the LSTM's two; here nothing overlaps. Nominal-step 0 <= 0 + 0 held, trivially |
| **T5** | no stall; GRU and TCN clean; LSTM >= 3x the GRU's MSE | LSTM 35 (29) at 6.218e-5; GRU 19 (8) at 7.161e-5; TCN 26 (15) at 6.889e-5 | **Held on the rule** (no `best_epoch <= 5`); **the sub-prediction refuted the other way** -- the LSTM fitted best of the three and the GRU stopped earliest, at 19. The three land within 15% of each other |
| **T6** | TCN within 3x the GRU on both FA axes | 6 against 4 (1.5x); 0 against 0 | **Held** |

**What `m2-ss1` says.** The recipe transfers: three architectures fitted on a
third of an unseen spacecraft's history, calibrated on their own nominal
residuals with no label and no tuning, alarm on **one rare event in a
hundred and on no nominal step at all**. The per-channel floors do not
transfer -- `rstd` alarms on a sixth of nominal time -- which is the
cross-channel claim of Objective.md 1.1 on an independent spacecraft. And
the union's Mission-1 economy -- the GRU's rare alarm hiding inside the
LSTM's -- was a Mission-1 coincidence: on Mission 2 the two cells' four rare
alarms each are four different events, and the OR costs their sum.

**Not a finding, said so it is not read as one.** The thresholds are 10-30x
the Mission-1 floors (0.21-0.44 against 0.012-0.028): Mission 2's groups are
scaled differently and its grid is 18 s, and a floor is a quantile of that
spacecraft's own residual. Comparing floors across missions compares
telemetry scales, not detectors.

#### 18.8.2 `m1-g3` -- the recall exam

The Mission-1 group-3 bundle: 8 channels, 14,728,319 steps at 30 s. Forward
chaining put **11 anomalies test-side (10 headline-cell) and 13 rare events**
-- folds 0 / 1 / 2 hold 6 / 4 / 1 anomalies, 6 / 4 / 0 headline, 3 / 8 / 2
rare; four of the nomination's fourteen headline events fell into the seed
window. Every figure is `n < 20`: UNDERPOWERED throughout. `check()`'s floor
of five was cleared (11) and nothing was refused.

| detector | F0.5 | recall | MVGS | precision | rare-FA | nominal-step FA | honest lead |
|---|---|---|---|---|---|---|---|
| `rstd` | 0.029 | 3/11 | 3/10 | 12/499 | 0/13 | 332,245 / 10,881,882 (3.05%) | -45.0 (n=3) |
| `mavg` | 0.021 | 5/11 | 5/10 | 58/3,367 | 0/13 | 270,203 (2.48%) | -1,272.0 (n=5) |
| `lstm-quantile` | 0.679 | 10/11 | 9/10 | 6,002/9,406 | **8/13** | **3,256,485 (29.93%)** | -86.5 (n=10) |
| `gru-quantile` | 0.835 | 9/11 | 8/10 | 4,819/5,740 | **10/13** | **3,122,203 (28.69%)** | -26.0 (n=9) |
| `lstm-gru-or` | 0.657 | 10/11 | 9/10 | 6,042/9,835 | 10/13 | 3,274,525 (30.09%) | -37.5 (n=10) |

Per fold, the two cells and the union:

```
  fold  detector        threshold   recall  MVGS  precision      rare  nominal-step FA           ranges  honest lead   training
  0     lstm-quantile   0.01149     5/6     5/6   682/691        1/3   0 / 3,643,618             0       -89.0 (n=5)   27 (16)  8.992e-6
  0     gru-quantile    0.00908     4/6     4/6   703/716        1/3   0                         0       -49.5 (n=4)   24 (13)  8.539e-6
  0     lstm-gru-or     1.0         5/6     5/6   707/718        1/3   0                         0       -50.0 (n=5)
  1     lstm-quantile   0.01093     4/4     4/4   5,313/5,340    7/8   3,095,318 / 3,570,272 (86.7%)  19   +129,270 (n=4)  20 (9)   9.909e-6
  1     gru-quantile    0.01114     4/4     4/4   4,111/4,135    7/8   3,095,200 (86.7%)         14      +129,270 (n=4)  23 (12)  8.949e-6
  1     lstm-gru-or     1.0         4/4     4/4   5,324/5,351    7/8   3,095,335 (86.7%)         19      +129,270 (n=4)
  2     lstm-quantile   0.00833     1/1     0/0   7/3,375        0/2   161,167 / 3,667,992 (4.39%)  3,368   -148.0 (n=1)  35 (28)  1.207e-5
  2     gru-quantile    0.00764     1/1     0/0   5/889          2/2   27,003 (0.74%)            833     -269.0 (n=1)  28 (17)  9.782e-6
  2     lstm-gru-or     1.0         1/1     0/0   11/3,766       2/2   179,190 (4.89%)           3,704   -148.0 (n=1)
```

**Fold 0 transfers as Mission 1's gate set did**: no nominal-step alarm in
3.64M, one rare event in three, five of six and four of six anomalies, floors
of 0.011 and 0.009 in line with the gate set's folds 1 and 2. **Fold 1 is a
collapse of the calibration, not of the forecaster.** Both cells' thresholds
-- 0.0109 and 0.0111, calibrated on the first 7.36M steps -- sit below
**87% of the test window's nominal residual**: the alarm is essentially the
whole window, cut into 14-19 ranges by the few steps that dip under, and
everything inside it counts as caught -- 4/4 recall, 7/8 rare events, and a
"lead time" of +129,270 steps that is the distance from the window's start
to the first event. Fold 2 is the same in miniature: 4.4% and 0.7% of nominal
time, 3,368 and 833 alarm ranges. The forecasts themselves fitted normally
(validation MSE 8.5e-6 to 1.2e-5, no stall); what did not hold is the
premise of the frozen rule -- that a quantile of the fitting window's
residual is the noise floor of what follows. On this subsystem, after the
first quarter of its history, it is not: the residual's scale moved, and a
global cut fixed in the past cannot follow it. **The recall figures above are
bought the way `mavg` bought 29/31 on the gate set (`docs/HARNESS.md` 1) and
are not findings**; the nominal-step rate is.

| # | Predicted | Observed | Verdict |
|---|---|---|---|
| **R1** | headline-cell recall 8-11/14 each; holds >= 8/14 | 9/10 and 8/10 | **Held on the letter, void in substance**: bought with alarms on 29-30% of nominal time. Bare recall is never a finding here (D3) |
| **R2** | rare-FA <= 12.5%; fails > 20.8% | **8/13 (61.5%) and 10/13 (76.9%)** | **FAILED**, far outside the band; fold 1 alone is 7/8 for both |
| **R3** | the union's edge: MVGS >= max + 1 at rare-FA <= max + 1 | union 9/10 = the LSTM's 9/10; rare 10/13 = the GRU's | **Evaporated** -- +0 over the better member, at the worse member's cost. UNDERPOWERED at n = 10, and the number is not the reason it is void |
| **R4** | honest lead median >= -10; fails < -50 | LSTM **-86.5**, GRU -26.0 | **LSTM failed, GRU no verdict** -- and fold 1's +129,270 is the window-long alarm, not a warning |
| **R5** | F0.5 LSTM >= 0.6, GRU >= 0.5 | 0.679, 0.835 | **Held on the letter, void in substance** -- fold 1's 5,313/5,340 "precision" counts pieces of one window-long alarm that overlap long events |
| **R6** | GRU no stall; LSTM fold 0 not predicted | no stall on any of the six fits (best epochs 16, 9, 28; 13, 12, 17) | **Held** |

**The one prediction the pre-registration did not make, and should have.**
Every band in 18.4 assumed the calibration would hold and asked how much
recall or how many rare events would move. Nothing asked whether the noise
floor measured on the first 7.36M steps would still be the noise floor of the
next 3.68M. `m2-ss1` -- a different spacecraft -- said yes (0 of 4.16M nominal
steps); `m1-g3` fold 1 -- the same spacecraft, a later period, a different
subsystem -- said no, by 87% of the window. That is the finding, it is not
re-tuned, and section 18.6's D29 is written on it.

**A hypothesis, not a measurement, recorded as the first thing to look at.**
Group 3's residual scale appears to change after the seed window (a mode or
configuration change in `subsystem_6` mid-history); telemanom's local
threshold, which adapts to a 2,100-sample window, is the rule this project
closed on the gate set for firing constantly when the forecaster improved
(D17, D18). The two failures are the same asymmetry from opposite sides: a
local floor follows the noise and cannot see a sustained break; a global
floor sees the break and cannot follow the noise. Objective.md 10.2 fix 4 and
14.10 already say thresholds are recalibrated in orbit and stored outside
the weights; `m1-g3` fold 1 is the measured case for why.

**The console rendered the union's training block as "0 epochs, best -1"**:
the scorecard printer expects one report and the union carries one per
member; the artifacts carry both members' blocks correctly
(`tests/test_union_detector.py`). Cosmetic; recorded.

---

## 19. Pre-registration: the C++ inference core and the `model.bin` freeze (work item 8)

**Written and committed before a line of C++ exists.** The format is frozen by
`docs/DECISIONS.md` D30 and specified byte for byte in `docs/MODEL_FILE.md`, which is
normative; this section is the pre-registration the flight work is judged against -
the flight rules with their citations, the predicted footprint, and the acceptance
tolerances. Objective.md 14.2 requires the freeze **before Phase 2 starts**. Zero
bucket operations: every input is the seeded fixture or a seeded generator, and the
only trained weights used are already cached under `runs/_weights`.

### 19.1 What is being built, and what is not

Built: the GRU forward pass and the frozen decision layer (D25) transcribed from
`src/sentinel_models/reference.py` into freestanding C++14 under `flight/`; the
`model.bin` writer and reader in `src/sentinel_export/`; and golden vectors pinning
every stage.

Not built, and deliberately: the F' component, its ports and events (work item 9);
the recalibration uplink path (work item 10); any inference library (D30); any change
to Python training code or to the frozen decision layer. The held-back sets are spent
and are not touched.

### 19.2 The file, in one paragraph

Four blocks - a 64-byte self-protecting header, a channel map, the float32 weights in
`reference.Weights.arrays()` order with **both bias vectors unsummed**, and a
separately-CRC'd parameter block carrying the threshold, the EWMA span, the
normalisation constants and `baseline_only`. Three CRCs, all CRC-32/IEEE 802.3. The
parameter block having its own CRC is what lets a mission recalibrate in orbit without
touching 278.0 KiB of weights, which is the requirement D29 consequence 3 and
Objective.md 14.10 both state. Full layout: `docs/MODEL_FILE.md`.

### 19.3 The flight rules, as F' itself writes them

The authority is `.github/skills/fprime-cpp-design/SKILL.md` at `nasa/fprime`
**v4.3.0** (released 2026-08-20), which calls itself "the **single source of truth**
for the C/C++ design rules F Prime flight software is held to". The version pin is
D31, which resolves Objective.md 14.4.

| rule | as F' states it | how this core obeys it |
|---|---|---|
| **CPP-1** | no `new`/`delete`/`malloc`/`free` after init; "pre-sized arrays sized at compile or init time" allowed | every buffer is a fixed member sized from `constexpr` maxima in `Config.hpp`. No allocator at all |
| **CPP-25** | no exceptions, no RTTI, no STL, no `std::string`; `std::min`/`max`/`numeric_limits`/`<cstdint>` permitted | `-fno-exceptions -fno-rtti`; the loader returns a status and cannot throw by construction |
| **CPP-3** | fixed-size types only; bare `int`, `float`, `double` forbidden outside external APIs | `F32`/`F64`/`U8`/`U16`/`U32` throughout, via `Types.hpp` |
| **CPP-5** | C++14; C++17 and C++20 features are not portable to every toolchain | `-std=c++14` |
| **CPP-34** | "All loops must have a provable upper bound"; prefer `for` | every loop is counted against a header field already bounded by `TOO_LARGE` |
| **CPP-9 / CPP-10** | no C-style or function-style casts; `reinterpret_cast` needs justification | `memcpy` for every field read out of the byte stream; `static_cast` elsewhere |
| **CPP-19** | every variable explicitly initialised | member-initialiser lists and brace initialisation |
| **CPP-32** | every fallible return value checked, or `(void)`-cast with a reason | the reader's status is checked at every call site |
| **CPP-8 / CPP-30** | typed `constexpr` over `#define`; no bare numeric literals for configuration | `Config.hpp` names every bound, with its derivation |

**Three corrections to the work item's own statement of the rules**, recorded because
a rule cited to the wrong authority does not survive review:

1. **F' states no no-recursion rule.** It states CPP-34 (bounded loops) and cites the
   **JPL C Coding Standard** (CPP-27) and the F' style wiki (CPP-26). Recursion is
   forbidden here anyway - it is **Power of Ten rule 1** and a JPL C standard rule -
   but it is cited to those and not to F'.
2. **"No dynamic allocation after initialization" is Power of Ten rule 3** and the
   citation is correct as far as it goes, but F's own equivalent is **CPP-1**, and F'
   cites the JPL C standard rather than the Power of Ten. Both are recorded.
   `Fw::MemAllocator` is the F'-idiomatic init-time allocator
   (`docs/user-manual/framework/memory-management/memory-allocation.md`: "Flight
   Software coding standards forbid dynamic memory allocation outside of system
   initialization") and it is **not used**, because our shapes are known at compile
   time and a fixed array is simpler than an allocator.
3. **"Buffers sized at compile time from the model header" cannot be done as
   written** - a header is read at runtime. Buffers are sized from compile-time
   maxima and the header is **validated against them**, refusing with `TOO_LARGE`.

**Not a library.** `mlpack` and Armadillo are not used, in flight or anywhere near
it: both throw, both allocate on the heap, and neither supports a bare-metal target.
The transcription is hand-written and checked against `reference.py`, which is the
whole point of D15 and of `tests/test_reference_equivalence.py`.

### 19.4 PREDICTED

Committed before the core is written. Every figure is arithmetic from the array
shapes in section 3, not an estimate.

| # | Prediction | Reasoning |
|---|---|---|
| **F1** | The declared members of `Sentinel::Detector` total **312,642 bytes** at the compile-time maxima (`C=16, L=2, H=80, l_p=10`), and `sizeof` exceeds that by under 64 bytes of alignment padding | 301,440 B of weight arrays plus 11,202 B of state, ring, scratch and parameters. The full derivation is the table below |
| **F2** | Of that, the flown shape uses **293,874 bytes = 287.0 KiB**, of which **284,640 B (278.0 KiB) is weights** | 71,160 parameters x 4, matching section 3 and the frozen assertion at `tests/test_reference_equivalence.py:282` |
| **F3** | The maxima cost **18,768 B (6.4%)** over the flown shape | `C=16` against 12 and `n_in=16` against 12 in `w_ih` and the head |
| **F4** | Compute is **70,080 multiply-accumulates per tick** | Reproduces D28 criterion 3 exactly, independently derived here from the shapes |
| **F5** | The C++ matches `reference.py` at **<= 1e-5** on hidden state and forecast at every tier, and **exactly** on the crossing flag, including across a chunk boundary and after a reset | The same `TOLERANCE` that holds the reference to torch (`tests/test_reference_equivalence.py:36`), where the GRU measures **1.2e-07** at the flown shape and does not grow with sequence length |
| **F6** | A file at the flown shape is **285,136 bytes** and round-trips Python -> C++ -> Python byte-identical | 64 + 240 + 284,640 + 192 |
| **F7** | The reader refuses a bad magic, a bad version, a bad header CRC, a bad static CRC, a bad param CRC, a truncation and an oversize, each with its own status code and **no exception** | `-fno-exceptions` makes throwing impossible; the codes are `docs/MODEL_FILE.md` 8 |
| **F8** | Patching the parameter block leaves `static_crc32` unchanged and the weights unread | `docs/MODEL_FILE.md` 6.1 - the in-orbit recalibration path, demonstrated at the format level |

**F1 and F2, derived.**

```
                              maxima      flown
  weight arrays              301,440    284,640
  channel map (id + name16)      320        240
  parameter block                230        198
  hidden state (L x H)           640        640
  prediction ring (l_p x l_p*C) 6,400      4,800
  EWMA numerator (C x F64)       128         96
  EWMA denominator                 8          8
  forecast / residual / EWMA     192        144
  gate scratch (proj + recur)  1,920      1,920
  layer in / out                 640        640
  input vector                    64         48
  head output                    640        480
  counters, status, flags         20         20
  -----------------------------------------------
  TOTAL                      312,642    293,874
                            305.3 KiB  287.0 KiB
```

### 19.5 The named risk, stated before it is measured

`np.tanh` and `np.exp` against libm's `tanhf` and `expf`, and NumPy's blocked BLAS
accumulation against a counted F32 loop over 80 terms, each differ at about one unit
in the last place, and a recurrence compounds them over 250 steps and two layers. The
reference-to-torch measurement - 1.2e-07 at the flown shape, "not growing with
sequence length" (section 2) - says the margin against 1e-5 is roughly two orders of
magnitude wide. It is not a proof.

Accumulation is in **F32**, matching the reference's dtype exactly. **If 1e-5 is not
reached, that is reported as a finding and not tuned away**; the F64-accumulate
variant is reported beside it as evidence of where the difference came from, not
silently substituted for it.

**The dtype map is not uniform and must not be made uniform.** The decision layer is
F64 by construction: `windows.aggregate_predictions` accumulates in float64 and casts
to float32, `telemanom.ewma` computes entirely in float64 and returns float32, and the
threshold comparison at `harness.py:169-172` is float64. A core that ran the decision
layer in F32 throughout would be a different detector. `-Wconversion -Werror` makes
every narrowing explicit so the compiler holds that map rather than the author's care.
It also makes **F64 a hard requirement of the flight target**, which F' treats as a
switchable platform feature (`FW_HAS_F64`); Phase 4's board selection inherits it.

### 19.6 The golden vectors

Four tiers. Each records, per step: the hidden state per layer, the head output, the
forecast, the residual, the EWMA, the max across channels, the score and the crossing
flag - plus a chunk-boundary case and a post-reset case.

| tier | shape | weights | input | committed |
|---|---|---|---|---|
| **G1** | 3 channels, (4, 4), `l_p` 2 | seeded | seeded | the `model.bin` itself, 1,276 bytes, **and** its vectors - a byte-frozen format regression, small enough to audit by hand |
| **G2** | 7 channels, (24, 24), `l_p` 5 | seeded | **the synthetic fixture**, `synthetic.build(seed=0, n=8000)` | vectors |
| **G3** | 12 channels, (80, 80), `l_p` 10 | seeded | seeded | vectors only; the 278.0 KiB of weights is regenerated from the seed and never committed |
| **G4** | 12 channels, production | the four cached fits under `runs/_weights` | seeded | nothing - local only, skipped when absent, reported in the work-item report |

Seeded weights use PyTorch's own GRU initialiser, `U(-k, k)` with `k = 1/sqrt(H)`, by
the recipe fixed in `docs/MODEL_FILE.md` 10, so a committed vector regenerates on a
fresh clone. `tests/test_golden_vectors.py` regenerates every vector and asserts it is
unchanged, so they cannot drift silently.

**No golden vector uses real telemetry, and none can.** "No telemetry on local disk"
(README, provenance rules); `runs/` holds weights and scorecards only. The cached
*weights* are local and G4 uses them; every *input* is synthetic. The work item's own
wording assumed otherwise and is corrected here.

**Which production weights.** The scorecard fingerprint `6d146f5d` is a SHA-256 of the
detector's name and parameters (`sentinel_eval/detector.py:96-100`, recomputed and
confirmed), not of any weight file, and the weight-cache key is a digest that includes
a strided sample of the telemetry (`detectors._digest`), which is not on disk. So a
cached file cannot be mapped back to a named fold. G4 therefore uses **all four**
cached fits at the flown shape and names each by its store filename and its embedded
training report.

### 19.7 Stop and report

1. The C++ cannot reach 1e-5 against `reference.py` on any tier, or a crossing flag
   ever differs. That is a finding about the transcription or about float32, and it is
   reported, not tuned away.
2. A vector needs anything beyond the cached weights and the fixture.
3. The format needs a field the documents did not anticipate. `baseline_only`
   (Objective.md 14.10) was already one the work item's own brief had omitted; it is
   in the format and this trigger covers the next one.
4. Anything would touch `main`, the frozen decision layer, the spent held-back sets,
   or Python training code.
5. Any bucket operation at all. This work item's budget is zero.

### 19.8 OBSERVED

**2026-09-01.** Everything below is printed by `make -C flight test` and by
`.venv/bin/python -m pytest -q`. Zero bucket operations.

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **F1** | 312,642 B of declared members; `sizeof` within 64 B of it | **`sizeof(Detector)` = 312,112 B = 304.8 KiB** | **Held, 530 B under (0.17%)** -- see below |
| **F2** | the flown shape uses 284,640 B of weights = 278.0 KiB | 284,640 B; `MAX_PARAMETERS` = 75,360 float32 = 301,440 B of arena | **Held, exactly** |
| **F3** | the maxima cost 18,768 B over the flown shape, of which 16,800 B is the weight arena | weight arena headroom **16,800 B (5.9%)** | **Held, exactly**, on the half a single object can report |
| **F4** | 70,080 multiply-accumulates per tick | 70,080, re-derived from the array shapes | **Held** |
| **F5** | <= 1e-5 on state and forecast, **exact** on the crossing flag | **worst stage difference 1.788e-07** over seven tiers and 504 steps; crossing and emitted flags **exact on every step of every tier** | **Held, with two orders of margin** |
| **F6** | 285,136 bytes, round-tripping Python -> C++ -> Python byte-identical | 285,136 B; **7 of 7 files re-emitted byte-identically**, four of them at the flown shape | **Held** |
| **F7** | a status code and no exception for each refusal | **16 load cases**, each returning its own code; `-fno-exceptions` makes throwing impossible | **Held** |
| **F8** | patching the parameter block leaves `static_crc32` and the weights untouched | the only bytes that move are `param_crc32`, `header_crc32` and the block itself | **Held** |

**F1, and where the 530 bytes went.** The pre-registration budgeted 640 B of
ping-pong "layer in / out" buffers. The implementation does not need them: layer
`i`'s output **is** its hidden state, so `Gru::step` writes into `m_hidden[i]` and
layer `i+1` reads it, and no intermediate buffer exists. That saves 640 B; 110 B come
back as struct padding and the `LayerView` offsets the prediction did not itemise.
Net 530 B under. Nothing was resized to meet the prediction.

**F5, stage by stage.** The largest difference on any tier, at any step, in any
stage:

```
  tier   channels  shape        worst |diff|   where
  g1      3        (4, 4)        5.960e-08    hidden / head / forecast / residual / EWMA
  g2      7        (24, 24)      1.192e-07    residual, step 1
  g3     12        (80, 80)      8.941e-08    head output, step 16
  g4_0   12        production    1.788e-07    head output, step 15
  g4_1   12        production    1.192e-07    head output, step 11
  g4_2   12        production    1.192e-07    head output, step 15
  g4_3   12        production    1.192e-07    residual, step 62
```

That is the float32 resolution the reference itself carries: `docs/MODELS.md` 2
measured the NumPy reference against torch at 1.2e-07 for the GRU at the flown shape,
and the C++ sits in the same place. **The named risk in 19.5 did not materialise.**
`tanhf`/`expf` against `np.tanh`/`np.exp`, and a counted F32 loop against NumPy's
blocked accumulation, cost between 2.98e-08 and 1.79e-07 over 72 steps, two layers
and a chunk boundary -- roughly fifty times inside the tolerance. Accumulation stayed
in F32 as pre-registered; the F64 variant was never needed and was not built.

**The G4 tier, named.** The four cached fits at the flown shape are
`runs/_weights/37f6c44412c194ddfe477941dd9ce47f.npz` (best epoch 8, validation MSE
7.161e-05), `4e336c4e32cd08fecfb385c4977cb3bc.npz` (26, 3.378e-06),
`5fcd163f4b4bca44d5d2729e90819610.npz` (18, 3.468e-06) and
`6ab5ebd98331bd8b5a53152bf7104f32.npz` (34, 4.628e-06). All four are 12 channels, two
layers of 80, `l_p = 10`, `window` 250, `n_exogenous` 0.

**Determinism.** The output digest over 400 ticks -- a CRC over the raw bit patterns
of every hidden state, head output, forecast, residual, EWMA, score and flag -- is
`0xD66576B4` twice in one process and again in a fresh process. Bit patterns and not
values, because the claim is about the encoding.

**The EWMA denominator recursion.** `Ewma.hpp` replaces the reference's
`(1 - decay^(t+1)) / alpha` with `D[t] = 1 + decay * D[t-1]`, which removes a `pow`
and an unbounded counter from every tick. Measured against the closed form over
8,000 steps: **maximum relative deviation 3.218e-15**, converging to `1/alpha` = 53.0.
Reported rather than asserted to be zero, as 19.5 said it would be.

**The build.** Silent at `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`,
with `-std=c++14 -fno-exceptions -fno-rtti -ffp-contract=off -O2`. Not one warning at
any level in any file. `flight/CMakeLists.txt` carries the identical flag set and
refuses `-ffast-math`, `-Ofast` and `-funsafe-math-optimizations` with a fatal error;
`tests/test_flight_build.py` asserts the two lists agree rather than trusting them to.

**Lint: deferred, and why.** `clang-tidy` is not on this machine -- not on `PATH`, not
in the Xcode toolchain, no Homebrew LLVM. `flight/.clang-tidy` carries the check set
with a reason beside each exclusion, and `make -C flight lint` runs it when present
and says so plainly when not. It runs at work item 9 with the F' toolchain (D31).

**Stop-and-report triggers: none fired.** The C++ reached 1e-5 with two orders to
spare, no crossing flag differed, no vector needed anything beyond the cached weights
and the fixture, the format needed no field beyond `baseline_only` (which
Objective.md 14.10 had already required and the work item's brief had omitted),
nothing touched `main`, the frozen decision layer, the held-back sets or Python
training code, and the run cost zero bucket operations.

**Verification.** 473 tests pass (406 before this work item, plus 30 on the file
format, 27 on the golden vectors and 10 driving the C++ suite from `pytest`);
`scripts/check_no_list.py` clean on 66 files; `sentinel_eval selftest` 8/8;
`make -C flight test` green on footprint, 16 refusals, determinism twice, seven
golden-vector tiers and seven byte-identical round trips.

---

## 20. Pre-registration: the F' component and the Level 1 safe-failure mode (work item 9)

**Written and committed before a line of component code exists.** Work item 8 left a
core that refuses a bad file and then goes quiet; this work item builds the thing that
keeps serving the topology after that refusal. `docs/MODEL_FILE.md` is still normative
on the file; `docs/DECISIONS.md` D32 to D37 carry the decisions this work item forces;
F' is pinned at v4.3.0 (D31). Objective.md 14.10's Level 1 -- "the loader's safe failure
mode ... never fail the topology" -- and `Objective.md` decision 3, the channel-ingestion
mechanism, are both discharged here. Zero bucket operations: every input is a seeded
generator or the offline fixture.

### 20.1 What is being built, and what is not

Built: an FPP-modelled F' component wrapping the work item 8 core, with a rate-group
`schedIn`, a channel-vector input, five telemetry channels, five events, and the Level 1
degrade-to-baseline path; the statistical baseline transcribed into `flight/` under the
work item 8 golden-vector discipline; the types shim swapped for F's own; a deployment
that builds and a `TestDeploymentsProject/Ref` instantiation that builds; `clang-tidy`
run for the first time.

Not built, and deliberately: the recalibration uplink path and its human approval (work
item 10) -- the reload command is named in the FPP as a comment and in the SDD, and no
code implements it; the chunked `Os::File` reader (`docs/MODEL_FILE.md` 8 anticipates it,
D36 defers it, and the check order is left untouched so it stays droppable-in); the
`_rolling` correctness fix (scoped as section 21, not executed); any change to Python
training, to the frozen decision layer, or to the spent held-back sets; any inference
library.

### 20.2 Thirteen corrections to the work item's brief

The brief asks to be verified against the repository before anything is written, and the
record says every previous coder found errors by doing so. Thirteen, each a fact read
from a named file. The last was found while writing the code rather than while reading,
and is recorded here beside the others rather than in a commit message. Where the brief and a document disagree, the document wins -- including F's
own.

| # | The brief, or a document, says | Verified |
|---|---|---|
| 1 | "16 refusal status codes", "each load refusal (the 16 codes)" | **11 refusal codes.** `flight/include/sentinel/Status.hpp:14-27` carries 12 enumerators, one of them `OK`. The 16 is the count of load *cases* in `flight/test/RefusalTests.cpp` -- 15 refusing, 1 accepting. `CHANGELOG.md:55`, `docs/STATUS.md:132` and section 19.8's prediction F7 are loose the same way. This work item emits one event carrying the code, over 11 codes |
| 2 | "read its exact form from `src/sentinel_eval`" | `rstd` is `RollingStd`, **`src/sentinel_models/baselines.py:86-110`**, kernel `_rolling` at `:34-55`, registered at `src/sentinel_models/registry.py:38`. `sentinel_eval` supplies the reduction (`detector.py:143-155`), the 99.9th-percentile recipe (`detector.py:127-132`) and the `>=` comparison (`harness.py:169-172`) |
| 3 | "done when it builds in an F' Ref deployment" | **`Ref/` does not exist at the v4.3.0 tag.** It is `TestDeploymentsProject/Ref/`. Present at v4.2.2, absent at v4.3.0, and absent from the v4.3.0 breaking-change notes. `docs/user-manual/design-patterns/rate-group.md:88` now links Ref through a pinned older commit. Both a project-owned deployment and the relocated Ref are built (D33 note, section 20.8) |
| 4 | "docs/MODELS.md section 20" | There was no section 20. The file ended at 19.8. This section creates it |
| 5 | "the baseline: the rstd rule as the harness defines it" | The rule needs **two fitted quantities `model.bin` version 1 does not carry**: a per-channel scale (`baselines.py:105`) and its own threshold. `rstd`'s calibrated cuts on `m1-g8.9.10` are 3.339457480522701, 4.386269506802676 and 4.380987492380889; the model path's committed tiers sit at 0.2737089991569519 to 0.951245903968811. Different statistics on different scales. Resolved by D34 |
| 6 | D31: "work item 9 replaces that one header ... and no line of the core changes" | Not literally achievable beside "the freestanding make target must STILL build". `Fw/FPrimeBasicTypes.hpp` includes `config/FppConstantsAc.hpp`, a generated F' build artifact that does not exist in the Makefile build. The shim is **guarded** (D35). Shim-only still holds: no other file changes |
| 7 | `Types.hpp` defines the shim | `SizeType` and `I32` are declared and **never used** anywhere in `flight/`. The swap is narrower than D31 implies |
| 8 | "the WARNING event carrying the peaking channel's id/name" | **The index does not exist.** `flight/src/Detector.cpp:100-106` reduces into a local and discards it. `Objective.md:417` already records that `last_attribution` is not persisted. Taken by a component-side argmax over `smoothed()` using the identical strict-`>` rule, so ties keep the lowest index as the core does. Zero core change |
| 9 | Section 19.3's nine CPP rules | A subset. v4.3.0's `.github/skills/fprime-cpp-design/SKILL.md` carries CPP-1 to CPP-34. **CPP-4, CPP-21, CPP-22, CPP-23, CPP-24 and CPP-28** bear directly on a component and are absent from the 19.3 table. Section 20.4 extends it |
| 10 | D31: "`clang-tidy` is deferred to work item 9, with the F' toolchain" | **F' v4.3.0 ships no `clang-tidy`.** It came from `brew install llvm`, which is what `flight/.clang-tidy` itself says. D31's premise was wrong about where the tool comes from, not about when it runs. F' does ship its own `.clang-tidy` and `release.clang-tidy` at its repository root, and the core is run against those too |
| 11 | `Objective.md` decision 3: telemetry-path tap vs direct port wiring | **The tap is not implementable against this file format.** `Fw.Tlm` carries a serialized `Fw::TlmBuffer`; `docs/MODEL_FILE.md` 4's CHANNELS record carries `id` and `name` and **no type tag**, so a tap cannot deserialise a value without knowing its declared type. Resolved by D33 |
| 12 | Work item 8's flag set is the bar | `TestDeploymentsProject/Ref/CMakeLists.txt` adds `-Wsign-conversion -Wold-style-cast -Woverloaded-virtual -Wnon-virtual-dtor -Wformat-security -Wundef` on top of it. The component is held to the union |
| 13 | `docs/MODEL_FILE.md` 9 and `Types.hpp`: F' "treats `F64` as a configurable platform type that a platform may switch off (`FW_HAS_F64`)" | **The macro does not exist in F's code.** `F64` is unconditional at `Fw/Types/BasicTypes.h:86` at v4.3.0 and at v4.2.2. `FW_HAS_F64` occurs exactly once in the framework, in the documentation table this project read, `docs/reference/numerical-types.md:35`. The real macro is `FW_HAS_64_BIT` and it guards `U64`/`I64`. Found while writing the shim, which now asserts `sizeof(F64) == 8` and `is_iec559` instead. Both documents amended |

Two smaller ones, fixed in the same commit: `docs/INDEX.md` was stale in four places
(D1-D29, "0.1.0 to 0.4.0", `wi1`-`wi7`, and the pre-registration list); and
`Objective.md:294-298`'s two named events both describe capabilities `Objective.md:283-289`
says do not exist, so this work item builds the channel-naming event only, exactly as
`CHANGELOG.md:13` already scoped it.

### 20.3 The component

`Sentinel::Monitor`, a **passive** component (D32). The core already owns
`Sentinel::LoadStatus`, so the ground-facing FPP enum is named `ModelLoadStatus` to
avoid the collision.

```
  array ChannelVector = [16] F32          # static_assert'd against Config::MAX_CHANNELS
  port  ChannelSample(ref values: ChannelVector, valid: bool)
  enum  Mode: U8            { MODEL = 0, BASELINE = 1 }
  enum  ModelLoadStatus: U8 { OK = 0 ... BAD_NORM_POLICY = 11 }
```

| Port | Kind | Type | Why |
|---|---|---|---|
| `schedIn` | `sync input` | `Svc.Sched` | the rate-group tick. F's `component-and-port-selection.md` names a passive component with a `sync` `Svc.Sched` input as the model for cyclic work |
| `channelsIn` | `sync input` | `Sentinel.ChannelSample` | D33. Latched; consumed once per tick |
| `tlmOut`, `eventOut`, `textEventOut`, `timeGetOut`, `prmGetOut`, `prmSetOut` | special | | telemetry, events, time, parameters |

| Telemetry | Type | Meaning |
|---|---|---|
| `Score` | `F32` | the max-across-channels statistic this tick |
| `Threshold` | `F64` | the active cut, the model's or the baseline's |
| `ActiveMode` | `Sentinel.Mode` | D5's active-tier channel; D30 consequence 6 wires `baseline_only` here |
| `TicksSinceWarmup` | `U32` | saturating, 0 while warming |
| `LoadStatus` | `Sentinel.ModelLoadStatus` | the last load result |

| Event | Severity | Carries |
|---|---|---|
| `ModelLoaded` | activity high | provenance, tier, `n_channels`, `n_parameters` |
| `ModelRefused` | warning high | the code, and the byte count read |
| `DegradedToBaseline` | warning high | the reason: a refusal code, or `BASELINE_ONLY_SET` |
| `CrossChannelWarning` | warning high | channel id, channel name, score, threshold; throttled |
| `WarmupComplete` | activity low | ticks |

Parameters, per D34: `BASELINE_SCALE: ChannelVector` and `BASELINE_THRESHOLD: F64`, both
with FPP defaults so Level 1 has constants even with no parameter database. The model
file path arrives through `configure()` in topology setup, which is the shape v4.3.0's
own breaking-change note gives for `FileHandling::prmDb.configure("PrmDb.dat")`.

**The assumption on the sample, stated so it can be tested.** Upstream delivers one
vector of `n_channels` F32 values per tick, from a producer on the same rate group at a
lower port index -- so both `sync` ports run on one thread and no mutex is needed, which
is the F' cyclic model. A mission wiring a producer on another thread makes `channelsIn`
`guarded`. If no sample arrived since the last tick the tick runs with `valid = false`,
which scores negative infinity and cannot alarm.

### 20.4 The flight rules, extended for a component

Section 19.3's table stands and is not repeated. Six rules it does not carry, each read
from `.github/skills/fprime-cpp-design/SKILL.md` at v4.3.0:

| rule | as F' states it | how this component obeys it |
|---|---|---|
| **CPP-4** | `FW_ASSERT` is for programmer-controllable invariants; never on "any off-device data crossing a hub / bridge / driver boundary" | `model.bin` is uplinked. **No assert anywhere on its contents.** Every refusal returns a code and emits an event, which is what the work item 8 loader already does |
| **CPP-21** | no C-style arrays in interfaces; pair array and length | every ground-facing interface is an FPP type. The core's `step(const F32*, bool)` and `load(const U8*, U32)` stay as they are and are recorded here as a knowing exception at an internal boundary, not an oversight |
| **CPP-22** | prefer `Fw/DataStructures` containers | the baseline's ring is a fixed member array sized from `Config`, as CPP-1 permits; no bespoke container escapes the class |
| **CPP-23** | commands, events, parameters and telemetry are declared in `.fpp` and the autocoded types used directly | `Mode` and `ModelLoadStatus` are FPP enums, `ChannelVector` an FPP array. Nothing is hand-serialized |
| **CPP-24** | prefer `Fw::String` over `char*` | the channel name in `CrossChannelWarning` and the provenance in `ModelLoaded` are `Fw::String`. `statusName()` returns a literal and is used only at the boundary |
| **CPP-28** | prefer configurable `Fw*` types where range varies by project | `FwChanIdType` for the channel id, `FwSizeType` and `FwIndexType` for sizes and indices in component code. The core keeps its fixed-size types, which CPP-3 requires and CPP-28 does not override |

### 20.5 PREDICTED

Committed before the component is written. Every figure is arithmetic from shapes
already measured, not an estimate.

| # | Prediction | Reasoning |
|---|---|---|
| **C1** | `sizeof(Sentinel::Monitor)` is **624,528 B**, a delta of **+312,416 B** over work item 8's 312,112 B | the derivation below |
| **C2** | The shim swap touches **1/1 files** and `make -C flight test` still passes all five suites | D35's guard keeps the freestanding path byte-identical |
| **C3** | The C++ baseline matches `baseline_reference.py` to **<= 1e-5**, and the worst case is **<= 1e-9** | the baseline path is F64 end to end with no narrowing, unlike the model path's three F32 narrowings that produced 1.788e-07 |
| **C4** | **11/11** refusal codes degrade to BASELINE and emit; **0/11** fail the topology | Objective.md 14.10 |
| **C5** | The component is passive: **0** threads, **0** queues, **2** sync input ports | D32 |
| **C6** | `clang-tidy` produces **at least 1 and fewer than 50** findings across the three configurations, the largest class being `readability-*` | 1,798 lines never linted, `WarningsAsErrors: '*'` |
| **C7** | Both deployments build at the union flag set of correction 12 | |
| **C8** | BASELINE mode costs **1,920** F64 multiply-adds per tick against MODEL mode's **70,080**, i.e. **2.7%** | 120 x 16 window recompute; 70,080 is 19.8's F4 |
| **C9** | The first full F' build completes in **10 to 20 minutes** on this machine | 10 cores, 16 GiB, measured in 20.9 |
| **C10** | A degraded component starts emitting **2,230 ticks earlier** than a healthy one -- baseline warm-up 120, model warm-up 2,350 | `baselines.py:96-98`, `docs/MODEL_FILE.md:185`. Stated because it is a real operational consequence of degrading, not a defect |

The footprint derivation, in bytes:

```
  Detector, measured at work item 8                              312,112
  model.bin buffer, 64 + 20*16 + 4*75,360 + 96 + 8*16            302,048
  Baseline: ring 120*16*4 = 7,680; finite mask 120*2 = 240;
            scale 16*8 = 128; scratch 2*16*8 = 256; counters 16    8,320
  component base, ports, latch, mode, name and provenance          2,048
  ----------------------------------------------------------------------
                                                                 624,528
```

### 20.6 The named risk -- and the one that fired before any code was written

**The named risk.** `clang-tidy` has never run on this core. `flight/.clang-tidy` sets
`WarningsAsErrors: '*'` over `bugprone-*`, `cert-*`, `clang-analyzer-*`, `misc-*`,
`performance-*` and `readability-*`, and F' ships two stricter configurations of its
own. If the finding count is large, findings are reported and triaged in the open;
**a check is not switched off to make a build quiet.** An exclusion added here needs the
same written reason the existing four carry.

**The risk that fired first, measured 2026-09-01 before the pre-registration was
committed.** The Python source the baseline is to be transcribed from does not compute
the rule it states. `baselines.py:44` runs `np.cumsum` on a float32 array and promotes
to float64 only at the surrounding `np.concatenate`, so the prefix sums that
`trailing_sum` differences are accumulated in float32. Differencing two large float32
prefix sums to recover a small second moment is catastrophic cancellation, and the
variance floor at `baselines.py:55` then clamps the negative result to zero.

Measured, three independent implementations against the one in the tree, on
`N(1000, 3)` float32, T = 8,000, C = 12, post-warm-up rows only:

```
  _rolling(float64 input)  vs  exact windowed recompute      1.8284e-08
  exact windowed recompute vs  numpy nanstd                  3.5489e-10
  _rolling(float32 input)  vs  all three                     7.6584e+00
```

True sigma is 3.0. At t = 5000 `_rolling` returns `[6.65, 0.00, 3.90, 0.00]` where the
answer is `[2.91, 3.28, 2.96, 2.82]`: two channels of four clamped to exactly zero.

It is not confined to a stress case. On this project's own offline fixture -- 39,992
steps, 7 channels, per-channel `|mean|/std` between 2.08 and 5.91, which is a mild
regime -- the float32 and float64 forms differ by up to **4.9252e-03** on spreads of
order 0.02 to 0.24, and the float32 form produces **3,975/279,104** exact zeros against
float64's **1,123/279,104**. That is 2,852 steps where the floor detector reports zero
spread because of arithmetic rather than data.

**Consequence for this work item, and it is deliberate.** The flight baseline
transcribes **the rule**, in F64, pinned against `numpy.nanstd`; it does **not**
reproduce `_rolling`'s float32 accumulation, which no fixed-memory implementation could
do in any case. So the flight Level 1 baseline and the harness's `rstd` are knowingly
different numbers, and this paragraph is the record of why. `baselines.py` is not
touched here: correcting it moves published Phase 1 figures and costs bucket operations,
and both belong to their own work item. That work item is scoped in section 21 and D37.

### 20.7 The golden vectors

The work item 8 discipline, applied to the baseline: seeded inputs, a committed vector
per tier, a manifest recording what each tier is, and a C++ binary held to the Python at
1e-5 with the crossing flag exact.

| tier | shape | input | committed |
|---|---|---|---|
| B1 | 3 channels, window 120 | seeded `default_rng(11)`, unit normal | yes |
| B2 | 12 channels at the flown width | seeded `default_rng(12)`, unit normal, 2% NaN | yes |
| B3 | 12 channels, offset 1000 sigma 3 | seeded `default_rng(13)` -- the regime that exposes the cancellation | yes |
| B4 | 7 channels | the offline synthetic fixture, `synthetic.build(seed=0)` | yes |

B3 exists specifically so a future implementation that reintroduces float32 accumulation
fails a test rather than passing quietly.

### 20.8 Stop and report

1. The shim swap requires a change to any core file other than `Types.hpp`.
2. The C++ baseline cannot match `baseline_reference.py` at 1e-5.
3. F' v4.3.0's documents contradict this section -- F's own documents win.
4. `FW_HAS_F64` is unavailable on the build platform.
5. The F' unit-test or deployment build cannot be held to correction 12's union flag set.
6. Anything would touch `main`, work item 10's scope, Python training, the frozen
   decision layer or the spent held-back sets.
7. Any bucket operation at all. This work item's budget is zero.

### 20.9 OBSERVED

**2026-09-01.** Everything below is printed by `make -C flight test`,
`make -C flight lint`, `fprime-util check`, `scripts/fprime_ref_patch.sh` and
`.venv/bin/python -m pytest -q`. Zero bucket operations.

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **C1** | `sizeof(Monitor)` 624,528 B, delta +312,416 | **623,152 B, delta +311,040** | **Held**, 1,376 B under (0.22%) |
| **C2** | the shim swap touches 1/1 files | **1/1**, and `sizeof(Detector)` is still exactly 312,112 under F' types | **Held** |
| **C3** | baseline matches the reference at <= 1e-5, expected <= 1e-9 | **0.000e+00** over four tiers and 1,600 steps, flags exact | **Held**, with nothing left to spare because there is no difference |
| **C4** | 11/11 refusal codes degrade, 0/11 fail the topology | **11/11 and 0/11** | **Held** |
| **C5** | passive: 0 threads, 0 queues, 2 sync input ports | **0, 0, 2** -- and three command ports F' requires that were not predicted | **Held, with a correction** |
| **C6** | clang-tidy: 1 to 50 findings, largest class `readability-*` | **170 findings, largest class `misc-include-cleaner` at 87** | **Wrong**, on the count and on the class |
| **C7** | both deployments build at the union flag set | both build; the union is real and includes `-Wold-style-cast -Wdouble-promotion -Wsign-conversion` | **Held** |
| **C8** | BASELINE costs 1,920 F64 MACs against MODEL's 70,080, 2.7% | unchanged; it is arithmetic from the window and the channel count | **Held** |
| **C9** | first full F' build 10 to 20 minutes | **12.4 s** -- generate 6.4 s, build 6.0 s | **Wrong**, by two orders of magnitude |
| **C10** | a degraded component emits 2,230 ticks earlier | 120 against 2,350, and the unit test pins the gate at both ends | **Held** |

**C1, and where the 1,376 bytes went.** 312,112 `Detector` + 302,048 model-file
buffer + 8,216 `Baseline` = 622,376, leaving 776 bytes of component state where
2,048 was budgeted. The `Baseline` came in 104 B under its predicted 8,320
because the ring's finite-mask estimate was generous. Nothing came in over.

**C6, and what the two findings that mattered were.** The whole exercise earned
its keep on two lines, both from `clang-analyzer`, the path-sensitive checks:
`Detector.cpp:68` out-of-bounds access past `m_hidden`, and `Detector.cpp:118`
division by zero. Both are unreachable at load time -- `ModelFile.cpp:109`
refuses the shapes that would cause them. But the loader validates once and the
`Model` struct then lives in RAM for the mission, and D5's own motivation for
Level 1 is "a corrupt file, a version mismatch, a failed CRC or **a radiation
bit-flip**". A flipped bit in `nLayers` after load walks off the array. The
shape is now re-checked in `step`, three compares against 70,080
multiply-accumulates, and a corrupted shape scores negative infinity so it reads
on the ground as a dead channel rather than a frozen one. Sixteen findings were
fixed, 154 excluded, and every exclusion carries a written reason in
`flight/.clang-tidy`. All three configurations -- ours, the framework's root and
the framework's release config -- now report **0** on the core, the tests and the
component.

**C9, and why the reasoning was wrong.** The prediction assumed a framework build
means thousands of translation units. F' builds only the modules the topology
references: 400 objects and 113 static libraries, in 21 CPU-seconds. The error
was in the model of the build system, not in the arithmetic.

### 20.10 What F' decided that reading its documents had not

Four things the framework settled once code was being written rather than read.
They are recorded here rather than in 20.2, which was committed before any of
this existed.

14. **Parameters force command ports.** F' refuses a component that declares
    parameter specifiers and no command receive port, because the parameter
    protocol *is* the autocoded `PARAM_SET` and `PARAM_SAVE` commands. So D34's
    choice of parameters obliges the component to carry `cmdIn`, `cmdRegOut` and
    `cmdResponseOut` while still having no command of its own. **This does not
    disturb D32**: the generated base class holds an `Os::Mutex m_paramLock` for
    parameter sets and gets, so a passive component's parameters are already
    guarded against the command dispatcher's thread, and the concurrency argument
    that would have forced `queued` does not arise.
15. **A library's modules must be namespaced.**
    `docs/how-to/develop/develop-fprime-libraries.md`: "Placing container
    directories directly at the root of the repository is *strongly* forbidden."
    Making `fprime/` an F' library so Ref could consume it moved the component
    from `fprime/Monitor` to `fprime/Sentinel/Monitor` and its module name from
    `Monitor` to `Sentinel_Monitor`. A library shipping a module called `Monitor`
    at its root collides with the next library that has one.
16. **A deployment can only reference modules already defined**, so the
    component's `add_fprime_subdirectory` must precede the deployment's. The
    `fprime-util new` wizard appends each at the end of the project's
    `CMakeLists.txt`, which produces the wrong order; F' then says so precisely.
17. **A telemetry packet set is exhaustive.** Ref downlinks through one, and FPP
    requires every channel of every instance to be packetized or explicitly
    omitted -- so adding Sentinel to Ref means saying what happens to its five
    channels. They went into a packet rather than the omit list.

### 20.11 The build proof

| Artifact | Result |
|---|---|
| `fprime/SentinelRef`, this project's own deployment | builds; **1,956,600 B**; instantiated on the 1 Hz rate group |
| F's own `TestDeploymentsProject/Ref`, via `scripts/fprime_ref_patch.sh` | builds; **2,428,064 B** against stock Ref's 2,352,560; **234** Sentinel symbols including the rate-group handler |
| The component's F' unit tests | **10/10**, covering all eleven refusal codes |
| `make -C flight test` | five suites, eleven vector tiers, seven byte-identical round trips |
| `make -C flight lint` | three configurations, **0** findings |
| `pytest -q` | **493** tests, up from 473 |

**Level 1 was watched working, not only tested.** Running `SentinelRef` for six
seconds with no model file present:

```
  WARNING_HI: (SentinelRef.sentinelMonitor) DegradedToBaseline :
      Sentinel degraded to the Level 1 baseline: NO_MODEL_FILE (2)
                                                (NOT_LOADED (12))
```

and PrmDb, the version events and the rate groups all ran afterwards. The two
`PrmIdNotFound` warnings for `0x20000000` and `0x20000001` beside it are
`BASELINE_SCALE` and `BASELINE_THRESHOLD` falling back to their FPP defaults,
which is D34 working: Level 1 runs with no parameter database at all.

### 20.12 Stop-and-report triggers: one fired, and it was not on the list

Of 20.8's seven, **trigger 3 fired** -- "F' v4.3.0's documents contradict this
section" -- four times, and each is recorded as a correction rather than
absorbed: Ref's location (3), where `clang-tidy` comes from (10), `FW_HAS_F64`
naming a macro F's code does not define (13), and the four items in 20.10. None
of the other six fired: no core file other than the shim changed for the swap,
the baseline matched exactly, the union flag set held, nothing touched `main`,
work item 10's scope, Python training, the frozen decision layer or the spent
held-back sets, and the work item cost zero bucket operations.

**And one fired that this section did not list, from the work item's own brief:**
"the baseline transcription cannot match its Python source at 1e-5". It cannot,
and the reason is that the source is wrong -- 20.6 and D37. It was reported
before the transcription was written, the decision to transcribe the rule rather
than the implementation was taken at the checkpoint rather than by this work
item, and the repair is scoped in section 21 to run before work item 10.

**Verification.** 493 tests pass (473 before this work item, plus 14 on the
baseline reference, 4 on the shim and the F' bridge, 1 on the toolchain
exemption and 1 driving the component's F' suite); `scripts/check_no_list.py`
clean on 68 files; `sentinel_eval selftest` 8/8; `make -C flight test` green on
footprint, refusals, determinism twice, seven golden-vector tiers, four baseline
tiers and seven byte-identical round trips; `make -C flight lint` clean on three
configurations; `fprime-util check` 10/10.

---

## 21. Scoping the `_rolling` correctness fix (work item 9.5; scope, not build)

**Written and committed before anything is changed or re-scored.** Section 20.6 measured
a defect in `baselines._rolling`, the kernel behind both `rstd` and `mavg`. This section
scopes the repair. Nothing here is executed by work item 9: the flight baseline in work
item 9 transcribes the correct rule and says so, and this section is the plan for making
the harness agree. It runs **after work item 9 and before work item 10**, because the
figure at stake is the denominator of the project's headline comparison.

### 21.1 Why it cannot wait

`docs/RESULTS.md` 2 states the floor as "`rstd` on `m1-g8.9.10`: event-wise
F0.5 = 0.250", headline-cell recall **3/32**, and section 20.6 shows the statistic behind
that number is computed wrongly. Section 4's own OBSERVED table
(`docs/MODELS.md:324`) reads "**Headline-cell recall above `rstd`'s 3/32** | 28/32 | ...
**Held, decisively, and it is the project's thesis. A per-channel statistic finds three;
a forecaster over the channel set finds twenty-eight**". A thesis stated as a ratio
against a baseline is only as good as the baseline. If the floor was under-measured by
arithmetic, the comparison has to be restated.

### 21.2 What changes, and what is preserved

1. **`baselines._rolling` accumulates in float64.** A one-line change at
   `baselines.py:44`, promoting before `np.cumsum` rather than after. This is a **D8
   correctness fix**, which `docs/HARNESS.md` explicitly exempts from the
   closed-to-scope-changes rule: "closed to scope changes, never to correctness fixes".
2. **A pinning test lands with it**, asserting `_rolling(x, W, "std")` matches
   `numpy.nanstd` over the trailing window to a stated tolerance, on the B3 regime of
   section 20.7 -- the offset case that exposes the cancellation. The defect cannot
   return silently.
3. **Every old artifact under `runs/` is preserved**, and every re-scored figure is
   published as `new (was old, D37)`. Nothing is overwritten and no number is deleted.
   This is the same discipline sections 4 and 6a-6g already use for corrections.

### 21.3 The re-score, and what it costs

`m1-ss5` is a strict subset of `m1-g8.9.10`, and `docs/HARNESS.md` 4 records that "one
twelve-channel load scores both -- 15 Class B rather than the 25 two runs would cost".
So the primary re-score is **one run, not two**:

```
  python -m sentinel_eval --verbose run m1-g8.9.10 \
      --detector rstd --detector mavg --no-sweep
```

| Run | Class B | Class A |
|---|---|---|
| `m1-g8.9.10` + `m1-ss5`, both baselines, all folds | 10 to 15 | 1 |

`docs/HARNESS.md` 5's table says 10 for any Mission1 task; `docs/HARNESS.md` 4 and
`docs/STATUS.md:204` say 15. The discrepancy is noted here and resolved by reading the
ledger on the day, not by choosing the convenient one. Either way it is negligible
against the 50,000-per-class monthly ceiling and the 1,000 tripwire.

Both baselines are re-scored, not just `rstd`, because `_rolling` feeds both. Their
exposure differs by two orders of magnitude and the scope should say so: on the offline
fixture the `mavg` residual moves by **2.2078e-04** against a residual scale of
**3.8092e-02** (0.6%), where the `rstd` spread moves by up to **4.9252e-03** on spreads
of 0.02 to 0.24, and gains 2,852 spurious exact zeros.

### 21.4 PREDICTED, before the re-score

| # | Prediction | Reasoning |
|---|---|---|
| **W1** | The corrected `rstd` threshold on `m1-g8.9.10` fold 0 is **below** 3.339457480522701 | the corrupted statistic has a fatter upper tail -- spurious 6.65 where the truth is 2.91 -- so its 99.9th percentile sits high. Correcting it should lower the floor |
| **W2** | Corrected `rstd` headline-cell recall is **>= 3/32** | a lower floor fires more often, and the defect's spurious zeros could only suppress |
| **W3** | Corrected `rstd` nominal-step false alarms rise **above 0.000%** | same mechanism |
| **W4** | The F0.5 floor **moves from 0.250**, by **less than 0.15** | direction not predicted: recall up and precision down pull F0.5 opposite ways, and F0.5 weights precision twice |
| **W5** | `mavg` moves by less than `rstd` on every metric | 21.3's measured exposure |
| **W6** | `gru-quantile`, `lstm-quantile` and `tcn-quantile` **do not move at all** | none of them calls `_rolling`. The flying path is `detectors.py` and `reference.py`. This is checked by re-running nothing and diffing the code path, not assumed |

**The falsification, stated plainly.** If corrected `rstd` reaches or exceeds the GRU's
headline-cell **22/32**, the project's central claim -- that a per-channel statistic
cannot see the cross-channel class -- is in serious question and this document says so in
those words. If it merely improves on 3/32, section 4's OBSERVED verdict is restated with
both numbers and the thesis is re-argued at the corrected margin, not quietly left at the
old one.

### 21.5 The held-back sets: escalated, not decided here

`docs/RESULTS.md:1463` records that two held-back sets exist: **`m2-ss1`** and
**`m1-g3`**. Both carry `rstd` rows computed with the defect -- `m2-ss1` at
**84/424 (0.198)** rare-event false alarms and **718,831/4,155,841 (17.30%)** of nominal
steps (`docs/RESULTS.md:1314`), `m1-g3` at **3/11** headline-cell and
**332,245/10,881,882 (3.05%)** (`docs/RESULTS.md:1344`). Whether those rows are
re-scored is **not decided in this section**. Both cases:

**For re-scoring (rstd and mavg rows only).** D8 exempts correctness fixes by name, and
this is one. The held-back discipline exists to stop a *candidate* being tuned against a
set; `rstd` is a baseline, its only parameter is a default `window = 120` that no result
selected, and re-running it chooses nothing. A targeted run
(`--detector rstd --detector mavg`) produces a new artifact containing only those rows,
so the flying detectors' numbers are never recomputed and cannot be re-read. And the cost
of leaving it is real: `m2-ss1`'s `rstd` row is what "a floor fixed in the past collapsed
on a later period of the same spacecraft" is argued from, and that sentence should not
rest on an artifact of float32 accumulation. 16 Class B and 1 Class A for `m2-ss1`.

**Against re-scoring.** The task definition's own `headline` field reads "HELD BACK --
recall. Run once, at the end, settings frozen", and `tasks.py:163-166` adds "If the
result disappoints we do not re-tune: tuning against a held-back set converts it into
another training set." A rule that admits a well-argued exception is a rule that ends at
the first well-argued exception, and this would be the first. The defect is in a
baseline, and **D29's adoption decision does not move**: `gru-quantile`'s 4/424 and its
0 of 4,155,841 nominal-step alarms come from a path that never calls `_rolling`, so
nothing that was decided on those sets is in doubt. An annotation is complete and costs
nothing -- "computed with `_rolling`'s float32 accumulation; see D37; not re-scored,
because the held-back sets are scored once" -- and the project already preserves wrong
numbers rather than re-running them (section 4's OBSERVED, D22).

**Recommendation, for the decision to accept or overrule.** Re-score `m2-ss1` and
`m1-g3` for `rstd` and `mavg` only, in a run that names no other detector, and preserve
the original artifacts beside the new ones. The seal protects the candidate from the
set; it was never a promise that a baseline's arithmetic error would be published
uncorrected. But the argument against is not weak, and the decision is not this
document's to make.

### 21.6 Documents that carry an `rstd` figure

Every one is updated from the new artifact, old figure kept in parentheses with the
D-reference: `README.md:40` and `:57`; `docs/STATUS.md:56`; `docs/RESULTS.md:15`, `:21`,
`:35`, `:38`, `:93`, `:104`, `:123`, `:144`, `:505`, `:569`, `:623`, `:1314`, `:1344`,
`:1366`; `docs/MODELS.md:279`, `:287`, `:323`, `:324`, `:325`;
`docs/PHASE1_REPORT.md:140`; `CHANGELOG.md:104`. Line numbers are as of this commit and
are a starting list, not a substitute for grepping on the day.

### 21.7 Stop and report

1. The corrected `rstd` reaches the GRU's headline cell -- 21.4's falsification.
2. Any flying detector's number moves. Nothing in the flying path calls `_rolling`; if
   one moves, the fix reached further than this section says and the work item stops.
3. The re-score would cost more than 50 Class B operations in total.
4. The held-back question in 21.5 has not been decided.

### 21.8 OBSERVED

**2026-09-01.** Three runs, **41 Class B and 3 Class A** in total, all recorded in
the ledger. Old artifacts preserved beside the new ones; no forecaster row was
recomputed, and no run named a detector other than `rstd` and `mavg`.

**The fix.** `baselines.py:40` promotes to float64 before the values are squared
and `np.cumsum` accumulates with `dtype=np.float64`. Against the flight rule the
gap closes from **7.6584e+00 to 1.8284e-08**, and the spurious exact zeros go
from 3,975 to none. `tests/test_rolling_precision.py` pins it against
`numpy.nanstd` in five regimes.

### 21.9 PREDICTED against MEASURED

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **W1** | the corrected fold-0 threshold is **below** 3.339457480522701 | **13.666420229522133** -- and folds 1 and 2 likewise rose, to 13.44 and 11.95 | **Wrong**, and the reasoning was backwards |
| **W2** | corrected headline-cell recall **>= 3/32** | **25/32** | **Held**, and far past the falsification line |
| **W3** | corrected nominal-step false alarms rise **above 0.000%** | **0.024%** (2,563 / 10,675,488) | **Held** |
| **W4** | F0.5 moves from 0.250, by **less than 0.15** | **0.676**, a move of **0.426** | **Wrong on the magnitude**, right that it moved |
| **W5** | `mavg` moves **less than** `rstd` on every metric | `mavg` F0.5 0.028 -> 0.170 (+0.142) against `rstd`'s +0.426; but `mavg`'s rare-event false alarms moved 10/48 -> 15/48 where `rstd`'s moved 1/48 -> 3/48, and on `m2-ss1` `mavg` got **worse** (122/424 -> 208/424) where `rstd` got much better | **Mostly held, not on every metric** |
| **W6** | no forecaster moves at all | none was recomputed; nothing in any forecaster's path calls `_rolling` | **Held by construction** |

**W1 is worth dwelling on, because the reasoning was exactly inverted.** The
prediction argued that a corrupted statistic has a fatter upper tail, so its
99.9th percentile sits high and correcting it would lower the floor. What
actually happened: the scale divisor is `nanstd` of the spread series itself
(`baselines.py:105`), and the corrupted spread series was far *noisier*, so its
standard deviation was far *larger*, so the normalised scores were far *smaller*.
Correcting the numerator shrank the denominator more. The threshold rose from
3.34 to 13.67 and the detector still fires eleven times more often. The
prediction reasoned about one half of a ratio and forgot the other.

### 21.10 The falsification fired

`docs/MODELS.md` 21.4 stated it in advance: "**If corrected `rstd` reaches or
exceeds the GRU's headline-cell 22/32, the project's central claim -- that a
per-channel statistic cannot see the cross-channel class -- is in serious
question and this document says so in those words.**"

It reached **25/32**. So, in those words: **the project's central claim as it was
written is falsified.** A per-channel statistic finds twenty-five of the
thirty-two `Multivariate/Global/Subsequence` events on the gate set. The "three"
that this repository has quoted since work item 4 was an artifact of arithmetic.

`m1-g8.9.10`, corrected floor beside the detector that flies:

| | F0.5 | recall | MVGS | precision | rare-event FA | alarms / 1k nominal |
|---|---|---|---|---|---|---|
| `rstd`, corrected | 0.676 | 34/46 | **25/32** | 84/127 (0.661) | 3/48 | 0.0024 |
| `gru-quantile` | **0.804** | 27/46 | 22/32 | **139/157 (0.885)** | **1/48** | **0.00066** |

**What survives, and why it is not a consolation prize.** D3 fixed the gate metric
as event-wise F0.5, *never bare recall*, before any of this was measured, and
`docs/HARNESS.md` 1 requires recall to be read against precision and never alone.
On that metric the forecaster wins by 0.128, and it wins in the way the argument
always said it should: comparable recall at **a third of the alarm rate**, with
precision 0.885 against 0.661. The corrected floor buys its events by alarming
more often. That is a real difference and it is the one the project pre-committed
to measuring.

**What does not survive** is the *ratio* -- three against twenty-eight -- that has
been the rhetorical centre of this repository. It was never the gate criterion,
and it was wrong.

**What is untouched.** D28's architecture gate compared LSTM, GRU and TCN and
never involved `rstd`. D25's decision layer, D29's transfer result and adoption
number, and every forecaster figure in this repository stand exactly as measured.

### 21.11 The held-back sets, and what the re-score found there

Re-scored on the recommendation accepted at the work item 9 checkpoint: `rstd`
and `mavg` only, in runs that named no other detector, with the original
artifacts preserved.

**`m2-ss1`, the transfer set.** This is the largest correction in the document,
and it runs the other way:

| | rare-event FA | nominal-step FA |
|---|---|---|
| `rstd`, was | 84/424 | 718,831 / 4,155,841 (**17.30%**) |
| `rstd`, corrected | **22/424** | **126 / 4,155,841 (0.003%)** |
| `gru-quantile` | 4/424 | 0 / 4,155,841 |

`docs/PHASE1_REPORT.md` said the per-channel floor "alarmed on a sixth of nominal
time" on an independent spacecraft. It did not. It alarmed on three thousandths
of a percent. **D29's conclusion is unchanged and its evidence is now cleaner**:
the forecaster still wins the adoption number 4/424 against 22/424 and 0
nominal-step alarms against 126. The comparison is now between two working
detectors instead of one working detector and a broken one.

**`m1-g3`, the held-back recall set.** The corrected floor matches the flying
detector's headline cell exactly:

| | F0.5 | recall | MVGS | precision | alarms / 1k |
|---|---|---|---|---|---|
| `rstd`, corrected | 0.351 (was 0.029) | 8/11 (was 3/11) | **8/10** (was 3/10) | 173/557 | 0.035 |
| `gru-quantile` | **0.835** | 9/11 | **8/10** | 4,819/5,740 | 0.078 |

Same 8/10, at half the alarm rate, at a third of the precision. The gate metric
separates them by 0.484.

### 21.12 Stop-and-report triggers

Trigger 1 fired -- "the corrected `rstd` reaches the GRU's headline cell" -- and
is the subject of 21.10. Trigger 2 did not: no forecaster moved, because none was
recomputed and nothing in their path calls `_rolling`. Trigger 3 did not: 41
Class B against a budget of 50. Trigger 4 did not: the held-back question was
decided at the work item 9 checkpoint before this work item ran.

**And a finding that was not on the list.** The float64 fix is necessary and it
is not sufficient in general. `sqrt(S2/n - (S1/n)^2)` has a regime limit at any
precision: its relative error grows as the **square** of `|mean|/sigma`.

```
  |mean|/sigma        relative error, float64, after the fix
             6            1.9e-13
         2.8e+03            4.6e-08     a 28 V bus with 10 mV of noise
           1e+06            6.7e-03
           1e+08            1.0         the statistic is gone
           1e+09            0.0         the variance floor clamps, silently
```

ESA-ADB is min-max scaled within channel groups, so its ratios run about 2 to 6
and the fix is decisive for everything this project scores. But the last row is
the same silent-zero failure the fix removed, at a higher threshold, and
`src/sentinel_models/baseline_reference.py` -- and therefore
`flight/src/Baseline.cpp` -- shares the limit exactly, because it transcribes the
same formula. `tests/test_rolling_precision.py` pins the boundary. Making the
form unconditionally stable is an algorithm change that would move the flight
golden vectors; it is reported here, not taken.

**Verification.** 505 tests pass; `scripts/check_no_list.py` clean; `sentinel_eval
selftest` 8/8; `make -C flight test` green with the Level 1 baseline still exact
at 0.000e+00; `make -C flight lint` clean on three configurations.

---

## 22. Pre-registration: is the corrected floor real? (work item 9.6)

**Written and committed before a single figure is computed.** Work item 9.5
corrected `baselines._rolling` and the floor moved from F0.5 0.250 to 0.676 and
from 3/32 to 25/32 headline-cell events, which falsified this project's central
claim as it was written (D37, section 21.10). Before that restatement is allowed
to stand, the corrected floor is audited the way a hostile reviewer would audit
it: **a result that overturns a thesis has to survive more scrutiny than the
thesis did, not less.**

Nothing here changes a detector, a threshold, a decision layer or a weight.
Nothing is retrained. `main` is not touched. The restatement already committed at
`b41f93b` stands as the record; whether it is the *final* reading depends on what
this section finds.

**Budget: one bundle load, 15 Class B and 1 Class A.** Everything below is
computed from that single load plus the cached weights under `runs/_weights/`,
in one script, because four scripts would be four loads.

### 22.1 The four questions, and why each is asked

1. **Is the corrected `rstd` leaking?** A floor that suddenly catches eleven
   times more events is exactly what a look-ahead bug looks like. Four specific
   leaks are checked, not a general impression.
2. **Which events does each detector catch?** The scorecard says how many, never
   which. **The only-GRU set is the thesis's remaining evidence** and it is the
   first thing the report states.
3. **Why does the GRU miss events a two-line statistic catches?** The
   state-adaptation hypothesis: a forecaster that carries state across ticks
   learns a sustained anomaly and stops being surprised by it, while a trailing
   variance statistic does not.
4. **What are the 32 headline-cell events actually made of?** The project has
   asserted since Objective.md 2.3 that this cell is contextual -- every channel
   inside its own limits while the combination is wrong. That has never been
   measured on this data. It is Claim B, and it was skipped.

### 22.2 Definitions fixed in advance

Fixed here so they cannot be chosen after seeing the answer.

- **Training envelope**: per channel, the **0.1 / 99.9 quantile pair** over the
  fold's fitting window with `train_mask` applied -- the construction
  `scripts/envelope_proxy.py:95-108` already uses, built from fitting data only,
  no test sample and no test label. The hard min/max is reported beside it and
  **where the two disagree that disagreement is the finding**.
- **Truly contextual event**: one where **no** watched channel leaves its own
  training envelope at any step inside the event span. This is the conservative
  direction: a real limit sits *outside* the historical envelope, so an event
  that never leaves the envelope is invisible to any limit check *a fortiori*.
- **Per-channel variance signature**: at least one channel whose trailing-120
  rolling standard deviation, inside the event span, exceeds that channel's own
  99.9th percentile of the same quantity over the fitting window.
- **Sustained crossing**: a run of consecutive steps at or above threshold inside
  the event span. Reported as count of runs and the median run length; a
  **stray-tick catch** is one whose longest run is a single step.
- **Residual half-life**: steps from the event's first scorable step to the first
  step at which the detector's per-step score has fallen to half its peak value
  within the event span. Undefined if it never halves; that is a reported outcome,
  not a missing value.

### 22.3 PREDICTED

Committed before the script runs.

**Leakage. The prediction is that there is none, and each is checked separately.**

| # | Prediction |
|---|---|
| **L1** | `_rolling`'s window is trailing and right-inclusive: no sample at `t' > t` enters the statistic at `t`. Checked by index arithmetic **and** by perturbation -- change a future sample, assert the score at `t` is bit-identical |
| **L2** | `RollingStd.fit` sees `values[train_lo:train_hi]` with `train_mask` applied and nothing else -- the same window and the same mask the GRU's calibration uses, through the same `harness.py:142-157` call |
| **L3** | `score`'s fallback scale, which would compute `nanstd` over the *scored* window, is never reached under the harness because `fit` always precedes it |
| **L4** | the threshold is `np.quantile(train_scores[usable], 0.999)` through `Detector.threshold_from` -- byte-for-byte the same machinery as the GRU's, not a parallel implementation |
| **L5** | the caught events are **sustained**, not stray ticks: median longest run inside a caught event **>= 100 steps**, and **<= 2 of 34** caught events are stray-tick catches |
| **L6** | the lead jump to +0.0 is a concentration, not an average: **>= 80%** of caught events have a per-event lead of exactly 0 |

**Overlap on `m1-g8.9.10`. `rstd` catches 34/46, `gru-quantile` 27/46.**

| # | Prediction |
|---|---|
| **L7** | all events: both **25**, only-GRU **2**, only-rstd **9**, neither **10**. Only-GRU is predicted in **[0, 6]** |
| **L8** | headline cell only: both **20**, only-GRU **2**, only-rstd **5**, neither **5** |
| **L9** | **only-GRU >= 1 on the headline cell.** If it is **0**, the forecaster catches a strict subset of what a two-line statistic catches and the thesis has no per-event evidence left on this set. This is the sharpest single number in the work item |
| **L10** | for **only-GRU** events, **no** channel shows a per-channel variance signature -- that is why `rstd` misses them. For **only-rstd** events, at least one does |

**State adaptation.**

| # | Prediction |
|---|---|
| **L11** | for events with footprint **> 1,000**, the GRU's per-step score halves within **105 steps** (the EWMA span) of event onset in **>= 60%** of cases, while the event is still running |
| **L12** | `rstd`'s score does **not** decay comparably on the same events: its median half-life is **> 2x** the GRU's, or undefined |

**Claim B, the label composition of the headline cell.**

| # | Prediction |
|---|---|
| **L13** | of the 32 `Multivariate/Global/Subsequence` events, **8 to 16** are truly contextual by the envelope test. Central estimate **12**, from Hundman's 41% of 32 |
| **L14** | `rstd`'s recall on the truly-contextual subset is **at least 20 percentage points below** its recall on the envelope-breaching subset. **This is the test that decides how much of the thesis survives**: if `rstd`'s 25/32 is concentrated in envelope-breaching events, the forecaster's value is precisely the contextual class and the claim restates rather than dies |

### 22.4 The named risk

**That this work item is motivated reasoning with a budget.** It was commissioned
after a result nobody wanted, and it is looking for reasons the result might not
count. The protections are that every definition in 22.2 is fixed before the
numbers exist, every prediction in 22.3 is numeric and falsifiable, and **L9 and
L14 are both stated in the direction that would hurt**: L9 names the outcome that
would leave the thesis with no per-event evidence at all, and L14 names the
threshold below which the restatement stands as written. A leak found is a
finding; a leak *not* found is also a finding, and the more likely one.

### 22.5 Stop and report

1. Any of L1 to L4 fails -- `rstd` is leaking and the 9.5 numbers are void.
2. `only-GRU == 0` on the headline cell (L9 refuted).
3. Any figure belonging to a forecaster moves. Nothing here recomputes one.
4. More than 20 Class B operations.
5. Anything would touch `main`, a detector, a threshold or a weight.

### 22.6 OBSERVED

**2026-09-01.** One bundle load, **15 Class B and 1 Class A** against the budget of
20. Weight store unchanged at 86 files -- nothing was fitted, nothing retrained,
no threshold moved. Artifact
`runs/m1-g8.9.10/_forensics/2026-09-01T230303Z-floor-audit.json`.

**The audit reproduces the published scorecards exactly** before it says anything
new: GRU 27/46 and 22/32, `rstd` 34/46 and 25/32, all four matching
`docs/RESULTS.md` to the event. An audit that could not reproduce the numbers it
is auditing would not be worth reading.

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **L1** | no future sample reaches the statistic at `t` | perturbing samples at t=200, 350, 500 changes rows before `t` by **0.0** exactly; the control confirms the perturbed row itself moved by 90.1 | **Held** |
| **L2** | `fit` sees only the masked training window, same path as the GRU | `harness.py:144` is one line, `detector.fit(values[train_lo:train_hi], usable, context)`, called identically for every detector | **Held** |
| **L3** | the fallback scale is unreachable | `fit` at 144 always precedes `score` at 154 | **Held** |
| **L4** | the threshold is the same machinery, not a parallel one | `RollingStd.threshold_from is Detector.threshold_from` is **True** -- the same function object | **Held** |
| **L5** | sustained not stray: median longest run >= 100, stray-tick catches <= 2/34 | median longest run **54**, naive stray-tick **9/34** | **Refuted as written, and the definition was the defect** -- see below |
| **L6** | lead exactly 0 in >= 80% of caught events | **79%** (27/34) on the gate set, **94%** (32/34) on `m1-ss5` | **Refuted by one point** on the gate set, held on the subset |
| **L7** | all events: both 25, only-GRU 2, only-rstd 9, neither 10 | both **27**, only-GRU **0**, only-rstd **7**, neither **12** | **Refuted** |
| **L8** | headline: both 20, only-GRU 2, only-rstd 5, neither 5 | both **22**, only-GRU **0**, only-rstd **3**, neither **7** | **Refuted** |
| **L9** | only-GRU >= 1 on the headline cell | **0**, on both sets, and **0 on all events too** | **REFUTED. Stop-and-report trigger 2 fired** |
| **L10** | only-GRU events show no variance signature; only-rstd events do | only-rstd **7/7** and **8/8** show one; the only-GRU side is **vacuous, there are no such events** | **Held on the half that exists** |
| **L11** | GRU score halves within 105 steps in >= 60% of events with footprint > 1,000 | **2/15 (13%)** on the gate set, **0/10 (0%)** on `m1-ss5`; median GRU half-life **2,805** and **6,284** steps | **Refuted. The state-adaptation hypothesis is dead** |
| **L12** | `rstd` decays more slowly than the GRU, half-life > 2x | **the opposite**: `rstd` median half-life **114** and **112** steps against the GRU's **2,805** and **6,284** | **Refuted in the reverse direction** |
| **L13** | 8 to 16 of 32 headline-cell events are truly contextual | **3/32** by the 0.1/99.9 envelope, **6/32** by hard min/max | **Refuted, and by a long way** |
| **L14** | `rstd`'s recall on contextual is >= 20 points below its recall on breaching | **0% against 86%**, a gap of **86 points** | **Held, overwhelmingly** |

**L5, and a defect in this section's own definitions.** 22.2 defined a stray-tick
catch as one whose longest run is a single step, and did not condition on the
event's footprint. Nine of the 34 catches on the gate set are single-step -- and
every one of them is an event whose **footprint is 1**, where a one-step crossing
is not a stray tick but a perfect catch. Excluding footprint-1 events:
**0 of 25** catches are stray ticks on the gate set and **0 of 8** on `m1-ss5`,
median longest run **54** steps, and the alarm covers a median **95%** of the
event's footprint. The claim L5 was testing -- that the catches are real and
sustained -- holds decisively. The prediction as written is still recorded
Refuted, because a definition that needed fixing after seeing the data is
exactly what pre-registration exists to expose.

### 22.7 L9: the flying detector catches a strict subset of the floor's events

**There is not one event on either Mission 1 set that `gru-quantile` catches and
the corrected `rstd` misses.** Not in the headline cell, not anywhere.

```
  m1-g8.9.10   both 27   only-GRU 0   only-rstd 7   neither 12
  m1-ss5       both 26   only-GRU 0   only-rstd 8   neither  8
```

This is the same nesting D25 found between `lstm-whitened` and `lstm-quantile`,
and the same consequence follows: **two detectors whose catches are nested are
ordered, not complementary.** Here the two-line rolling standard deviation is the
superset and the forecaster is the subset.

The seven events `rstd` catches and the GRU misses on the gate set are short and
sharp -- footprints of 1, 1, 1, 1, 24, 28 and 54 steps -- with `rstd` reaches of
5.2 to 6.4 against the GRU's 0.55 to 0.62. The GRU's EWMA over a 105-span smooths
a one-step excursion below its threshold; a 120-step variance window does not.

**What this leaves of the thesis on this data: nothing at the per-event level.**
The forecaster's remaining defence is entirely in the *quality* of the same
catches -- precision 139/157 against 84/127, and 0.00066 alarms per thousand
nominal steps against 0.0024 -- which is a real difference and is the one D3's
gate metric measures. It is not the difference the project has been claiming.

### 22.8 L13 and L14: the headline cell is not the contextual class

Objective.md 2.3's argument rests on Hundman's finding that 41% of real
spacecraft anomalies are contextual -- every channel inside its limits while the
combination is wrong -- and this project has treated the 32
`Multivariate/Global/Subsequence` events as that class. **Measured, they are
not.**

```
  truly contextual, no channel outside its own 0.1/99.9 training envelope :  3/32
  at least one channel leaves its envelope                                : 29/32
```

By hard min/max the contextual count is 6/32, and 22.2 said in advance that where
the two readings disagree the disagreement is the finding: it is, and both
readings are far below the 8-to-16 predicted.

**And the three that are genuinely contextual are caught by nothing.**

| event | footprint | GRU | `rstd` | GRU reach | `rstd` reach | channels breaching |
|---|---|---|---|---|---|---|
| `id_121` | 257 | no | no | 0.110 | 0.544 | 0 |
| `id_153` | 10 | no | no | 0.178 | 0.608 | 0 |
| `id_157` | 51 | no | no | 0.238 | 0.572 | 0 |

Neither detector comes within half its threshold. `rstd` scores 0/3 and the
**GRU also scores 0/3**. L14's 86-point gap is real and it is not evidence for
the forecaster, because the forecaster's gap is 76 points in the same direction.

**Stated carefully, because the envelope is not a limit.** A real RED or YELLOW
limit sits *outside* a channel's historical envelope, so "breaches its envelope"
does **not** mean "would have tripped a limit"; the implication runs only the
other way, and only for the three. What can be said is that **29 of 32 events are
not in the class that is provably invisible to limit checking**, and that the
three which are, this project's detector does not catch.

### 22.9 L11 and L12: the state-adaptation hypothesis is dead, backwards

The hypothesis was that a forecaster carrying state across ticks learns a
sustained anomaly and stops being surprised by it, which would explain its
misses. The opposite is true.

```
  median half-life, events with footprint > 1,000
                        gate set     m1-ss5
    gru-quantile          2,805       6,284      and never halves in 5/15, 7/10
    rstd                    114         112
```

The GRU's score does **not** decay inside a sustained event -- it stays elevated
for thousands of steps. It is `rstd` that decays fast, in about its own 120-step
window, which is what a trailing variance statistic does once the new variance
becomes the window's normal. So the GRU's misses have some other cause, and this
work item did not find it. The seven only-`rstd` events point at the EWMA
smoothing short excursions rather than at state adaptation, but that is a
hypothesis for another work item and is not tested here.

### 22.10 Triggers, and what this does not touch

**Trigger 2 fired** -- `only-GRU == 0` on the headline cell -- and it is the
subject of 22.7 and D38. Trigger 1 did not: L1 to L4 all hold and the corrected
`rstd` is not leaking, so work item 9.5's numbers stand as measured. Trigger 3
did not: no forecaster figure was recomputed, and the audit reproduces every
published one exactly. Trigger 4 did not: 15 Class B against 20. Trigger 5 did
not: `main` is untouched, no detector, threshold or weight moved, the weight store
is unchanged at 86 files.

**A limitation of this audit, named rather than left to be found.** L6 records
where the first in-span alarm step falls; it cannot separate "crossed at onset"
from "an alarm was already running and overlapped". Arithmetic bounds it -- 2,563
nominal alarm steps over 34 caught events caps a universal pre-existing run at 75
steps on average -- but it is not settled per event, and settling it needs a
second bundle load this section's budget does not allow. Separately, run lengths
were instrumented for `rstd` and **not** for the GRU, so 22.6's L5 finding has no
forecaster comparison. Both are gaps in the instrument, not in the data.

### 22.11 The rows 22.6 to 22.10 did not read

**Added 2026-09-02. No prediction is attached to this subsection, and that
is deliberate.** Every figure below is derived from the artifact work item
9.6 already committed --
`runs/m1-g8.9.10/_forensics/2026-09-01T230303Z-floor-audit.json` -- and
those figures were seen before this subsection was written. A
pre-registration written after its own answers are known is theatre, and
22.4 exists to prevent exactly that, so this is filed as a **derivation from
a committed artifact** rather than as a new experiment. It spends **zero
bucket operations**: no bundle is loaded, no weight is read, nothing is
refitted, and the weight store stays at 86 files.

It exists because the work item's brief asked for four things that 22.6 to
22.10 reported only as counts, or did not report at all.

#### The four sets, by id

The brief asked for the four sets "with ids, folds, footprints" and to lead
the report with the only-GRU set. 22.7 and D38 give the counts and seven
footprints; the ids have never been published. **Reach** is the peak score
inside the event span divided by that detector's own calibrated threshold,
so 1.00 is the crossing and anything below it is a miss with its distance
stated.

**`m1-g8.9.10`** -- 46 scorable `Anomaly` events over folds 0, 1 and 2; both
**27**, only-GRU **0**, only-`rstd` **7**, neither **12**.

**only-GRU: 0 events. The set is empty.** There is no event on this set that
`gru-quantile` catches and the corrected `rstd` misses, in any taxonomy
cell. That is D38 stated as a list instead of a count, and the list has
nothing in it.

**only-`rstd`: 7 events.**

| event | fold | cell | footprint | GRU reach | `rstd` reach |
|---|---|---|---|---|---|
| `id_142` | 1 | MVGS | 54 | 0.62 | 6.39 |
| `id_122` | 1 | MVGS | 28 | 0.60 | 6.34 |
| `id_124` | 1 | MVGS | 24 | 0.60 | 6.02 |
| `id_129` | 1 | MVGP | 1 | 0.60 | 6.01 |
| `id_130` | 1 | MVGP | 1 | 0.61 | 5.24 |
| `id_132` | 1 | MVGP | 1 | 0.55 | 6.43 |
| `id_140` | 1 | MVGP | 1 | 0.61 | 6.01 |

**neither: 12 events.**

| event | fold | cell | footprint | GRU reach | `rstd` reach |
|---|---|---|---|---|---|
| `id_89` | 0 | MVGS | 8,995 | 0.39 | 0.89 |
| `id_13` | 0 | MVGS | 5,459 | 0.29 | 0.52 |
| `id_96` | 0 | MVGS | 4,537 | 0.27 | 0.64 |
| `id_91` | 0 | MVLS | 55 | 0.22 | 0.53 |
| `id_20` | 1 | MVLS | 16,227 | 0.13 | 0.60 |
| `id_138` | 1 | MVGS | 1,787 | 0.18 | 0.61 |
| `id_121` | 1 | MVGS | 257 | 0.11 | 0.54 |
| `id_118` | 1 | MVGP | 1 | 0.08 | 0.53 |
| `id_11` | 2 | MVLS | 1,375 | 0.21 | 0.68 |
| `id_157` | 2 | MVGS | 51 | 0.24 | 0.57 |
| `id_153` | 2 | MVGS | 10 | 0.18 | 0.61 |
| `id_42` | 2 | UVLP | 1 | 0.17 | 0.61 |

**both: 27 events** -- `id_12`, `id_10`, `id_101`, `id_103`, `id_102`,
`id_93`, `id_107`, `id_90`, `id_109`, `id_110`, `id_114`, `id_134`,
`id_116`, `id_145`, `id_18`, `id_183`, `id_177`, `id_160`, `id_176`,
`id_150`, `id_149`, `id_184`, `id_166`, `id_187`, `id_186`, `id_165`,
`id_172`.

**`m1-ss5`** -- 42 scorable `Anomaly` events over folds 0, 1 and 2; both
**26**, only-GRU **0**, only-`rstd` **8**, neither **8**.

**only-GRU: 0 events. The set is empty.** There is no event on this set that
`gru-quantile` catches and the corrected `rstd` misses, in any taxonomy
cell. That is D38 stated as a list instead of a count, and the list has
nothing in it.

**only-`rstd`: 8 events.**

| event | fold | cell | footprint | GRU reach | `rstd` reach |
|---|---|---|---|---|---|
| `id_145` | 1 | MVGS | 6,300 | 0.88 | 2.40 |
| `id_122` | 1 | MVGS | 1 | 0.99 | 6.06 |
| `id_124` | 1 | MVGS | 1 | 0.99 | 3.91 |
| `id_129` | 1 | MVGP | 1 | 0.99 | 3.90 |
| `id_130` | 1 | MVGP | 1 | 0.99 | 5.33 |
| `id_132` | 1 | MVGP | 1 | 0.91 | 3.92 |
| `id_140` | 1 | MVGP | 1 | 0.99 | 3.90 |
| `id_142` | 1 | MVGS | 1 | 0.99 | 3.89 |

**neither: 8 events.**

| event | fold | cell | footprint | GRU reach | `rstd` reach |
|---|---|---|---|---|---|
| `id_89` | 0 | MVGS | 4,117 | 0.47 | 0.90 |
| `id_96` | 0 | MVGS | 2,574 | 0.32 | 0.64 |
| `id_138` | 1 | MVGS | 1,786 | 0.20 | 0.62 |
| `id_118` | 1 | MVGP | 1 | 0.14 | 0.54 |
| `id_121` | 1 | MVGS | 1 | 0.15 | 0.39 |
| `id_157` | 2 | MVGS | 51 | 0.37 | 0.56 |
| `id_153` | 2 | MVGS | 10 | 0.26 | 0.60 |
| `id_42` | 2 | UVLP | 1 | 0.24 | 0.61 |

**both: 26 events** -- `id_101`, `id_10`, `id_102`, `id_103`, `id_90`,
`id_12`, `id_93`, `id_107`, `id_109`, `id_110`, `id_114`, `id_134`,
`id_116`, `id_18`, `id_149`, `id_150`, `id_160`, `id_165`, `id_166`,
`id_172`, `id_176`, `id_177`, `id_183`, `id_184`, `id_186`, `id_187`.

#### Every only-`rstd` event is in fold 1

Stated as an observation, with no mechanism claimed. Of the 7 only-`rstd`
events on `m1-g8.9.10`, **all 7 are in fold 1**; of the 8 on `m1-ss5`, **all
8 are in fold 1**. The gate set's folds carry 15, 15 and 16 scorable events,
so the events were available in the other two folds and the entire nesting
gap between the two detectors still lands in one of them.

This is consistent with what section 14 already recorded from a different
direction: `docs/MODELS.md` :1514-1517 names six fold-1 events the GRU
**lost** when it replaced the LSTM -- `id_122`, `id_124`, `id_129`,
`id_130`, `id_140`, `id_142` -- and six of the seven here are those same
events. The seventh, `id_132`, is the one section 14 declined to score
either way at a reach of 0.553. So D38's only-`rstd` set is not a new
population; it is the fold-1 loss D26 already paid for, now visible against
a floor that was not broken.

What it is not is an explanation. Nothing here says why fold 1, and no
mechanism is proposed.

#### Claim B on `m1-ss5`: the 31 the brief asked for and 22.8 did not report

The brief asked to "classify the 32 (and `m1-ss5`'s 31)". 22.8 and D39
report the gate set alone. The artifact carries `envelope_breach_quantile`,
`envelope_breach_minmax` and `channels_breaching` populated for every
`m1-ss5` row as well, because `floor_audit.py`'s per-event loop runs
identically for each member task, so the second half of the question was
answerable all along.

```
                                              m1-g8.9.10      m1-ss5
  channels watched                                    12           6
  MVGS events                                         32          31
  truly contextual, 0.1/99.9 envelope                  3           3
  at least one channel leaves the envelope            29          28
  contextual by hard min/max                           6          18
```

**`m1-ss5` replicates D39 exactly on the strict reading, and on the same
three events.** The contextual set is `id_121`, `id_153`, `id_157` on both
channel sets -- the identical ids, found on a six-channel view and a
twelve-channel one. D39's most serious finding was measured once; it now has
an independent corroboration, and this subsection is where it is recorded.

**And neither detector catches any of them on `m1-ss5` either**:
`gru-quantile` **0/3**, `rstd` **0/3**, against 21/28 and 25/28 on the
envelope-breaching remainder. The two catch counts inside the contextual
class are zero on both sets, which is the honest size of the claim and is
what D39 consequence 2 already says.

#### The loose reading is view-dependent; the strict one is not

22.2 fixed the hard min/max as a second reading and said in advance that
"where the two disagree that disagreement is the finding". It is, and it is
larger than 22.8 saw, because 22.8 only ever looked at one channel set.

```
  contextual by 0.1/99.9 envelope    12 channels   3/32     6 channels   3/31
  contextual by hard min/max         12 channels   6/32     6 channels  18/31
  of the min/max set, caught by either detector     0                12
```

Halving the watched channels leaves the strict count at 3 and moves the
loose count from 6/32 to 18/31. The reason is mechanical and it is not a
defect: an event is classified contextual when **no watched channel** leaves
its envelope, so removing channels removes chances to breach and can only
move events into the contextual class, never out of it. The strict reading
did not move because the 0.1/99.9 envelope is tight enough that the events
breaching it breach on channels present in both views.

**The consequence is a definition, not a number.** "Truly contextual" is not
a property of an event. It is a property of an event *relative to a watched
channel set*, and a figure quoted without naming that set is
underdetermined. On the six-channel view 12 of the 18 min/max-contextual
events are caught by at least one detector, where on the twelve-channel view
none of the 6 are -- the same detectors, the same threshold recipe, a
different set of channels to be contextual with respect to. Recorded as
**D40**.

This qualifies D39's second reading and leaves its first intact. D39's
headline figure is the 0.1/99.9 one, it is the conservative direction, and
it is now measured at 3/32 and 3/31 on two views. **Nothing in D39's
consequences changes.**

#### The brief's "twelve GRU-missed events" is a mis-citation

The work item's brief asked to cross the overlap sets with "the twelve
GRU-missed events on record (`docs/MODELS.md` 14 and 16)". **The twelve on
record are not GRU-missed.** Section 14 defines them at :1297-1313 as the
twelve events `lstm-telemanom` catches and **`lstm-quantile`** misses -- the
set work item 5 was commissioned to attack. Section 14.8.2 then records at
:1497-1505 that `gru-quantile` **recovered 7 of them**. Crossing the audit
against "the twelve GRU-missed events" would cross it against a set that
does not exist.

Two questions do exist in its place, and both are answerable from the
committed artifact.

**Of the historical twelve, which does the corrected `rstd` catch?** 8 of
the 12 that are scorable on the gate set.

| event | in the twelve | GRU | `rstd` | GRU reach | `rstd` reach |
|---|---|---|---|---|---|
| `id_12` | recovered by GRU | yes | yes | 1.27 | 6.04 |
| `id_20` | still GRU-missed | no | no | 0.13 | 0.60 |
| `id_89` | still GRU-missed | no | no | 0.39 | 0.89 |
| `id_90` | recovered by GRU | yes | yes | 1.27 | 6.02 |
| `id_93` | recovered by GRU | yes | yes | 1.25 | 6.48 |
| `id_107` | recovered by GRU | yes | yes | 1.29 | 6.48 |
| `id_109` | recovered by GRU | yes | yes | 1.27 | 5.47 |
| `id_110` | recovered by GRU | yes | yes | 1.23 | 5.94 |
| `id_114` | recovered by GRU | yes | yes | 1.28 | 5.39 |
| `id_132` | still GRU-missed | no | yes | 0.55 | 6.43 |
| `id_138` | still GRU-missed | no | no | 0.18 | 0.61 |
| `id_157` | still GRU-missed | no | no | 0.24 | 0.57 |

So of the 5 of the twelve the GRU still misses -- `id_20`, `id_89`,
`id_132`, `id_138`, `id_157` -- the corrected floor catches **1**, `id_132`,
and misses the other **4**, which sit in the `neither` set above. The floor
does not rescue the events the architecture gate could not.

**And `gru-quantile`'s actual miss list on this set is 19, not 12** --
`id_11`, `id_118`, `id_121`, `id_122`, `id_124`, `id_129`, `id_13`,
`id_130`, `id_132`, `id_138`, `id_140`, `id_142`, `id_153`, `id_157`,
`id_20`, `id_42`, `id_89`, `id_91`, `id_96`. Of those 19 the corrected
`rstd` catches 7 and misses 12. The coincidence that the audit's `neither`
set also numbers 12 is a coincidence and nothing more; it shares 4 members
with the historical twelve.

#### What is still open, and what it would cost

22.10 named two gaps in the instrument and this subsection closes neither,
because both need data the artifact does not carry. They are restated here
with a price so a later work item can decide rather than rediscover.

**GRU run lengths were never instrumented.** `runs_of` is called once, on
`rstd`'s alarm mask, at `scripts/floor_audit.py:260`; there is no
`gru_runs`, `gru_longest_run` or `gru_median_run` field. So 22.6's L5
finding -- that the catches are sustained rather than stray -- has no
forecaster comparison, and the sustained-catch claim is established for the
floor alone.

**L6 cannot separate "crossed at onset" from "an alarm was already
running".** `lead_of` at `:184-188` returns the negated index of the first
in-span alarm step and never inspects the mask before the span, so it is
structurally incapable of a positive value and carries no information about
alarm state at onset. 22.10 bounds it arithmetically at 75 steps on average
over 34 caught events and leaves it unsettled per event.

**Both are one bundle load, together: 15 Class B and 1 Class A**, both
channel sets, all folds, from cached weights, by extending
`scripts/floor_audit.py` with three fields and re-running it. Nothing else
in this subsection costs anything. The month stands at 3 Class A and 56
Class B against a ceiling of 50,000 each.

Not attempted here, and named so it is not mistaken for an omission: why
fold 1. The concentration is reported above as an observation and no
mechanism is proposed for it.


## 23. Pre-registration: making the forecaster's cross-channel advantage measurable (work item 9.7; scope, not build)

**Written and committed before a single figure is computed, and nothing in this
section is run.** Work item 9.5 falsified the ratio the project quoted (D37) and
work item 9.6 found worse: on both Mission 1 sets the flying detector catches a
**strict subset** of a two-line rolling standard deviation's events (D38), and
the headline cell is **3/32 contextual** with all three caught by nothing (D39).

What survives is a claim about **quality** -- `gru-quantile` clears the corrected
floor 0.804 to 0.676 on D3's gate metric, at precision 139/157 against 84/127
and a third of the alarm rate. That is real, it is measured, and it is not the
claim this project has been making. The claim it has been making is that a
forecaster over the channel set sees relationship breaks a per-channel statistic
cannot, and **there is now no per-event evidence for it on this data.**

This section scopes the four studies that would settle it, in the order of what
they cost. Two are measurements this repository can make; two are designs that
would have to be built first and are costed rather than started.

Nothing here changes a detector, a threshold, a weight, the frozen decision
layer or `main`. Nothing is retrained. **No study below runs until it is
approved separately**, and the operation cost of each is stated before its
predictions rather than after.

### 23.1 The four studies, and why each is asked

1. **A -- the matched-operating-point comparison.** D38 withdrew the per-event
   argument and left the quality argument standing. The quality argument has
   never been tested at a matched operating point: the two detectors are
   compared at *their own* thresholds, and a detector that alarms more will
   catch more. Raise the floor's threshold until it is as quiet as the
   forecaster and the comparison becomes like-for-like for the first time.
2. **B -- the reduction study.** D23 has been OPEN since 2026-08-27 and says
   the decision layer is twelve univariate detectors and a vote. The forecaster
   is multivariate and everything after it is not. `k`-of-`n` "was tuned against
   182 alarm ranges when there are now 3,548", and D23 requires that "any
   threshold proposal that goes forward re-derives `k`-of-`n` rather than
   inheriting it". Nothing has.
3. **C -- leave-one-out cross-prediction.** Every design in this project
   predicts each channel from *all* channels including itself, so a channel can
   be forecast from its own history and the residual need not measure a
   relationship at all. Predicting each channel from the *others only* makes the
   residual a relationship measurement by construction. This is the thesis as
   architecture rather than as hope.
4. **D -- the injected-fault study.** Objective.md 13 item 8, never run. D39
   consequence 5 says what would settle Claim B is a scoring set whose events
   are selected for the contextual property rather than assumed to have it, and
   that ESA-ADB may not contain one at `n >= 20`. A synthesised set does, by
   construction, at a sample size that is not underpowered.

### 23.2 Three corrections to the framing, stated before the predictions

The work item was framed with three assumptions. Two are wrong and one bounds
what a result could mean, so they are recorded here rather than discovered in
23.9.

**1. A and B are not free, and nothing cached can make them free.** Nothing
under `runs/` holds a per-step or per-channel score.
`scripts/floor_audit.py:149` computes the per-channel matrix and `:154` returns
it unread; `runs/m1-g8.9.10/_curve/` and `_grid/` are 25 to 27 August, from
before `gru-quantile` existed and before `_rolling` was corrected, so they
describe neither detector in this comparison. **A and B share one bundle load:
15 Class B and 1 Class A**, both channel sets, all folds, every arm -- because
`_TOPS` (`src/sentinel_models/detectors.py:414-421`) is keyed on everything
upstream of the reduction and on nothing downstream, and the weights are already
cached, so every arm after the first is a reduction and a threshold comparison.
The month stands at 3 Class A and 56 Class B against a ceiling of 50,000 each.

**2. `k`-of-`n` at `k = 2` and `k = 3` needs no new code.** `MAX_AGREEMENT = 3`
(`detectors.py:78`), `agreement` is already a constructor parameter validated at
`:259-263`, the reduction is already `combined = top[self.agreement - 1]`
(`:370`, `:392`), and `scripts/decision_grid.py` already sweeps it against
persistence. Only the L2 norm and the sum are new code, because `_tops` keeps
the top `MAX_AGREEMENT` ratios and discards the other nine at `C = 12`; an
aggregate over all channels needs the full matrix retained.

**3. L2 and sum cannot test the relationship claim, and a win by either would
not be evidence for it.** `np.abs(filled[lo:hi] - forecast)` at
**`detectors.py:465`** removes the sign before any reduction sees the residual
vector. D23 states the consequence exactly: *"Both channels rising together and one rising while the other falls
become the same number, and that difference is the whole of the relationship."*
L2 and sum aggregate magnitude across channels; they are a softer `k`-of-`n`,
and what they test is co-occurrence. The only reduction that could express a
relationship is one over the **signed** residual vector -- D23's nominal
residual covariance, 144 floats at `C = 12` against the GRU's 71,160 parameters
-- and it is not in B, because it is a detector change and this section builds
nothing.

**A citation correction, found while writing this.** Both
`docs/DECISIONS.md` D23 (at `:1313`) and `docs/MODELS.md` section 11 (at
`:1022`) locate that `np.abs` at `detectors.py:410`. It was there when D23 was
written on 2026-08-27 and it is not there now -- the file has grown and the call
is at `:465`. Neither entry's finding changes and neither is edited; the pointer
is corrected here, and it is worth noting that a line-number citation is a fact
with no test behind it, which is the same class of rot
`tests/test_documents_are_current.py` was written for and does not yet cover.

And the relationship layer has been measured once already. `lstm-whitened` was
built, scored and declined: D25 found the plain quantile matched it event for
event at a **fourteen times lower** nominal-step alarm rate
(`docs/DECISIONS.md` D25, `docs/RESULTS.md` 6d). **B's predictions are written
against that record.** A reduction that beats `max` here would be the first
cross-channel decision stage in this project that ever has.

### 23.3 Study A -- the matched-operating-point comparison

**What is measured.** Both detectors' thresholds are swept, and the full
recall-versus-alarm-rate curve is reported for each, on both channel sets, per
fold and pooled, every recall as `k/n` with `n < 20` stamped UNDERPOWERED. Two
points on those curves are named in advance:

```
  matched-quiet    rstd's threshold raised until its nominal-step alarm rate
                   equals gru-quantile's 0.001%   (rstd is at 0.024% today)
  matched-loud     gru-quantile's threshold lowered until its nominal-step
                   alarm rate equals rstd's 0.024%
```

At each, the four overlap sets are recomputed by id, exactly as 22.11 reports
them at the calibrated thresholds.

**The threshold is swept, never chosen.** `scripts/oscfar_curve.py` established
the precedent and states the rule: what is reported is the curve, and no
operating point is recommended from it. Choosing the best-scoring cell is the
oracle sweep `docs/MODELS.md` section 7 refuses, and it stays refused.
**Nothing in A moves the flying detector's threshold**, which is D25's frozen
99.9th percentile of the anomaly-masked fitting window and is not a dial.

### 23.4 Study B -- the reduction study, D23 re-derived

**What is measured.** Under the frozen calibration recipe -- the same 99.9th
quantile, the same anomaly-masked fitting window, identity normalisation (D2) --
the max-across-channels reduction is replaced by four alternatives, each scored
as a **new named arm** so that `gru-quantile` and the frozen layer are untouched,
the way `lstm-whitened` and `lstm-oscfar` were added:

```
  gru-l2     L2 norm across channels          new code: the full residual matrix
  gru-sum    sum across channels              new code: the full residual matrix
  gru-k2     agreement k=2                    no new code (detectors.py:370)
  gru-k3     agreement k=3                    no new code (detectors.py:370)
```

Per arm: F0.5, event recall, headline-cell recall, rare-event false alarms,
nominal-step false alarms and lead, both channel sets, per fold and pooled. Per
event: what happens to the three contextual ids `id_121`, `id_153`, `id_157`; to
the seven fold-1 events the floor catches and the forecaster misses; and to each
of `gru-quantile`'s 19 misses on the gate set.

**What B cannot do**, restated here so no result is over-read: all four arms are
downstream of `np.abs()` at `detectors.py:410` and none of them can see a sign.
A gain by any of them is a gain in **co-occurrence sensitivity**, not evidence
that a relationship test works. 23.2's third correction states why, and D25's
measured verdict on the one relationship layer this project did build is the
record B is written against.

### 23.5 Study C -- leave-one-out cross-prediction (scoped, not built)

Every forecaster in this project predicts channel `c` from all `C` channels
including `c` itself, so nothing forces the residual on `c` to carry information
about any other channel. A channel with strong autocorrelation is forecast from
its own history, its residual measures its own predictability, and the
cross-channel claim rides on an architecture that never had to express it.
**Leave-one-out predicts `c` from the other `C-1` only**, which makes the
residual a relationship measurement by construction rather than by hope.

**Delivered as a costed design, not as code.** The scoping states: parameters
and multiply-accumulates per tick against `gru-quantile`'s 71,160 and the Phase
4 envelope; whether the cheapest workable form is `C` separate models, one
shared trunk with a masked input, or one model with a zeroed diagonal; training
time and whether the existing fold and weight-cache machinery carries it
unchanged (D14); how it meets the frozen decision layer, which is channel-blind
and would be unchanged by it; whether `model.bin` version 1 can carry it without
a format change (D30, `docs/MODEL_FILE.md`); and the pre-registration it would
itself need before a single fit. **No fit is run and no weight is written.**

### 23.6 Study D -- the injected-fault study (design only)

Objective.md 13 item 8, never run, and the only instrument that can settle Claim
B at a sample size that is not underpowered. D39 consequence 5 says what is
needed is a scoring set whose events are selected for the contextual property
rather than assumed to have it, and that ESA-ADB may not contain one at
`n >= 20`. This manufactures one instead of hunting for it.

**The construction.** Relationship breaks synthesised on real nominal telemetry
from the fitting windows, with **every channel held inside its own 0.1/99.9
training envelope at every injected step** -- the same envelope 22.2 fixed and
D39 measured against, so an injected event is contextual by construction and by
the same definition, and D40 requires the watched channel set to be named with
it. Three families, each a break in a relationship rather than an excursion:

```
  coupling offset    a channel's response to its driver shifted in amplitude
  response delay     a channel's lag to its driver lengthened
  decoupling         a channel's response to its driver removed
```

**`n >= 50`**, which is the first sample size in this project that clears
`docs/HARNESS.md` section 1's UNDERPOWERED rule for this class. Scored blind by
`gru-quantile`, the corrected `rstd`, and every arm from B.

**The design states**, before anything is built: which nominal windows and which
folds, so no injection lands in a window a detector was fitted on; how injection
reaches the values without touching the frozen decision layer, any threshold or
any weight; how blindness is enforced, given that whoever writes the injector
knows the answers; how an injected event that no detector could ever cross on is
distinguished from one both simply miss; and the exact operation cost.

### 23.7 Definitions fixed in advance

- **Matched operating point**: equal **nominal-step false-alarm rate**, the
  column both detectors already report, computed on the same fold's nominal
  steps. Not equal alarm count, not equal precision.
- **Recovered event**: an event a reduction arm catches that `gru-quantile` at
  its frozen threshold does not, on the same fold and the same channel set.
- **Truly contextual**: unchanged from 22.2, and per D40 always stated with the
  channel set it is contextual with respect to.
- **Sign-blind**: a reduction whose value is unchanged when the sign of any
  channel's residual is flipped. All four arms in B are sign-blind; the test is
  one line and it is asserted rather than assumed.
- **Detectable injection**: an injected event whose peak reach under *some*
  detector in the study exceeds 1.0. An injection detectable by nothing is a
  property of the injection and is reported as such, not as a miss.

### 23.8 PREDICTED

Committed before any study runs. Every one is numeric or a yes/no a document can
be held to, and each names what would refute it.

**Study A. The quality argument, tested like for like for the first time.**

| # | Prediction | Refuted by |
|---|---|---|
| **M1** | at **matched-quiet**, `rstd`'s headline-cell recall falls to **10 to 20 of 32**, central estimate 15, and below `gru-quantile`'s 22/32 | `rstd` reaching **>= 22/32** while as quiet as the forecaster. **This is the sharpest number in A.** It would mean the floor matches the flying detector at the flying detector's own operating point, and the quality defence D38 consequence 2 left standing would be gone too -- the last argument for the forecaster on this data. The document would say so in those words |
| **M2** | at **matched-loud**, `gru-quantile`'s headline-cell recall reaches **25 to 30 of 32**, at or above the corrected floor's 25/32 | `gru-quantile` below **25/32** while as loud as the floor: the floor would then lead at both ends of the curve and the ordering would not be an operating-point artifact in the forecaster's favour |
| **M3** | at **at least one** of the two matched points, only-GRU **>= 1** on the gate set -- there exists an operating point at which the forecaster sees something the floor does not | only-GRU **0** at both matched points **and** across the swept curve. D38 would then hold not at one threshold but everywhere, and the nesting would be a property of the two detectors rather than of their calibration. Stated deliberately in the direction that hurts |
| **M4** | swept down to the alarm rate at which each detector's event-wise precision first falls below **0.5**, `rstd` catches **>= 1** of the three contextual events and `gru-quantile` catches **0** | either half. The GRU catching one would be this project's first measured contextual catch, at a stated cost in precision |

**Study B. D23 re-derived, against the record of the one relationship layer that
was built.**

| # | Prediction | Refuted by |
|---|---|---|
| **M5** | `gru-k2` and `gru-k3` **lose** headline-cell recall against `k = 1` on both channel sets, despite recalibration on the reduced statistic | either gaining. The reduction is strictly harder at a fixed threshold and the recalibration is predicted not to compensate |
| **M6** | **at least one** of the four arms recovers **>= 1** of `gru-quantile`'s 19 gate-set misses at a nominal-step false-alarm rate no worse than `gru-quantile`'s | none does. D23's open question would then be answerable **no** on this data with the reductions available, and D23 could close as measured rather than stay open |
| **M7** | **no** arm catches any of `id_121`, `id_153`, `id_157` | any arm catching one. It would be the first measured contextual catch in this project, and 23.2's third correction would be wrong about what a sign-blind reduction can see |
| **M8** | **none** of the seven fold-1 only-`rstd` events is recovered by any arm, because they are lost in the per-channel EWMA at span 105 **upstream** of the reduction, not in the reduction | any arm recovering one, which would locate the loss in the reduction after all and make the EWMA explanation in 22.7 wrong |
| **M9** | **no** arm beats `gru-quantile`'s **0.804** F0.5 on the gate set | one does. D25 would be re-opened for scoping -- not changed, and not on this evidence alone -- because the frozen decision layer would not be the best of the reductions available to it |

**Studies C and D. Design claims, checkable without running anything.**

| # | Prediction | Refuted by |
|---|---|---|
| **M10** | the cheapest workable leave-one-out form costs **<= 2x** `gru-quantile`'s 71,160 parameters, not the naive `C` x | the cheapest form found costing more than 2x, which would make C a Phase 4 envelope question before it is a Phase 3 accuracy question |
| **M11** | `model.bin` version 1 **cannot** carry a leave-one-out model without a format change (D30) | it can, which would be the better answer and would make C cheaper to adopt than it looks |
| **M12** | a set of **>= 50** contextual injections can be constructed that stays inside every channel's 0.1/99.9 envelope **and** is detectable by at least one detector in the study | the two constraints proving incompatible -- no injection that stays inside the envelope produces a residual any detector could cross on. **That would itself be the finding**, and a large one: it would mean the contextual class as this project defines it is not reachable by residual thresholding at all, which is a statement about the method and not about the data |

**Deliberately not predicted.** Which of the four B arms wins, if any -- the
prediction is about whether *any* does, and naming a favourite would invite
reading the winner as confirmation. The lead time of any A or B configuration
beyond reporting it. Anything about `m2-ss1` or `m1-g3`, which are spent (D29)
and are not consulted. Why every only-`rstd` event is in fold 1, which 22.11
records as an observation with no mechanism and which no study here tests.

### 23.9 Cost, and the order of spending

```
  A + B     one bundle load, both channel sets, all folds, every arm
            15 Class B, 1 Class A          cached weights, nothing refitted
  C         zero operations                a costed design, no fit
  D         zero operations                a design, no injection built
```

A and B are one script and one load, for the reason 21.3 gives: four scripts
would be four loads. The month stands at **3 Class A and 56 Class B** against a
ceiling of 50,000 each and a per-run tripwire of 1,000.

**C and D are written first**, before A and B are approved to spend anything,
because a design that has to be honest about its own cost is cheapest to
abandon on paper.

### 23.10 The named risk

**That this work item exists to find the forecaster a win.** Work item 9.6 ended
with the project's central claim withdrawn on two counts, and a study
commissioned immediately afterwards to look for cross-channel advantage is
looking for a specific answer. That is the same risk 22.4 named, and it did not
prevent 9.6 from returning two results against the direction it was commissioned
in.

The protections are the same and they are stated in the same place: every
definition in 23.7 is fixed before any number exists; M1, M3 and M6 are stated
in the direction that would hurt, and M6 names the outcome that would let D23
**close as answered no** rather than stay open as a hope; 23.2 bounds in advance
what a B result could mean, so a co-occurrence gain cannot later be read as a
relationship result; and M12's refutation is written as a finding rather than a
failure.

**A negative result here is publishable and this section says so now.** If A
shows the floor matching the forecaster at matched quiet, if B shows no
reduction recovering anything, and if C costs more than the envelope allows,
then the honest reading is that on this data the forecaster's advantage is
precision at a fixed operating point and nothing more -- and `docs/RESULTS.md`
and Objective.md would say that, in those words.

### 23.11 Stop and report

1. **M1 refuted** -- the floor matches the forecaster at matched quiet. Report
   before anything else in this section is written up.
2. **M3 refuted** -- only-GRU is 0 across the whole swept curve. D38 would
   generalise from a threshold to a detector.
3. **M12 refuted** -- a contextual injection detectable by nothing.
4. Any figure belonging to a **frozen** artifact moves: `gru-quantile`'s
   scorecard, the decision layer, a weight, a held-back score.
5. More than **20 Class B** in any run, or the per-run tripwire at 1,000.
6. Any arm in B is proposed for adoption. **B measures; it does not select.**
   Adoption is a separate decision with its own pre-registration.

### 23.12 What this decides, and what it may not

**Decides.** Whether the forecaster has any measurable cross-channel advantage
over the corrected floor at a matched operating point (A). Whether D23 closes as
answered or stays open (B). Whether leave-one-out is affordable enough to be
worth a pre-registration of its own (C). Whether Claim B can be tested at
`n >= 50` at all, and what it would cost (D).

**May not.** Move any threshold, retrain anything, alter the frozen decision
layer, re-score a held-back set, adopt any arm, or touch `main`. Re-open D28,
which compared LSTM, GRU and TCN against each other and never involved `rstd`.
Restate the central claim -- that is held at `docs/STATUS.md` section 7 on D39
and is not this section's to settle. And it may not treat a co-occurrence gain
in B as evidence for the relationship claim; 23.2 fixed that reading in advance.

### 23.13 OBSERVED

**2026-09-02. Study A, the half of it that costs nothing, and it is the finding.**
Studies B, C and D have not run. **Zero bucket operations**: everything below is
derived from `runs/m1-g8.9.10/_forensics/2026-09-01T230303Z-floor-audit.json`,
which stores each event's **reach** -- the peak score inside the event span
divided by that detector's own calibrated threshold.

**Why a reach is a threshold sweep.** An event is caught at threshold multiplier
`m` exactly when `reach >= m`, so the cached reaches give the complete
recall-versus-threshold curve for both detectors on both sets without loading
anything. The identity was checked before it was used: `reach >= 1` reproduces
the published catch flag on **176 of 176** (event, detector) pairs, with no null
reaches, on both sets.

**What it cannot give** is the alarm rate at any multiplier, because the
nominal-step score distribution is not cached. So this is matched **recall**,
not the matched **false-alarm rate** 23.7 defined and M1 to M3 were written
against. Those still need one bundle load.

#### D38's nesting is a property of the two thresholds, not of the two detectors

At the frozen operating point the audit reports -- the GRU catching 27 and the
floor 34 -- only-GRU is 0, and that reproduces exactly. Hold the two to the
**same number of catches** and it stops being true.

```
  m1-g8.9.10                                    only-GRU
    frozen, asymmetric  (GRU 27, rstd 34)           0     <- D38, reproduced
    matched recall      (both at 27)                6
    matched recall      (both at 34)                0
    worst case          (both at 15)                7

  m1-ss5
    frozen, asymmetric  (GRU 26, rstd 34)           0     <- D38, reproduced
    matched recall      (both at 26)                2
    matched recall      (both at 34)                0
    worst case          (both at 15)                6
```

Only-GRU is **> 0 at 39 of 46** matched-recall levels on the gate set and **31
of 42** on `m1-ss5`. The two detectors are **not rank-identical**: they order
the same events differently, and the nesting D38 found appears only where the
floor is allowed to catch seven more events than the forecaster.

**The six the forecaster sees at matched recall on the gate set**, with the rank
each detector gives them:

| event | fold | cell | footprint | GRU reach | `rstd` reach | GRU rank | `rstd` rank |
|---|---|---|---|---|---|---|---|
| `id_114` | 0 | MVGP | 1 | 1.28 | 5.39 | 10 | 32 |
| `id_109` | 0 | MVGP | 1 | 1.27 | 5.47 | 11 | 31 |
| `id_110` | 0 | MVGP | 1 | 1.23 | 5.94 | 15 | 30 |
| `id_177` | 2 | MVGS | 64 | 1.14 | 5.97 | 22 | 28 |
| `id_172` | 2 | MVGP | 1 | 1.12 | 5.94 | 24 | 29 |
| `id_145` | 1 | MVGS | 10,717 | 1.06 | 2.64 | 27 | 34 |

Two are headline-cell. On `m1-ss5` the pair is `id_90` and `id_12`, both MVGS,
GRU ranks 8 and 9 against the floor's 32 and 33.

**What this does and does not change.** It does **not** overturn D38: at the
thresholds both detectors actually fly with, the flying detector still catches
nothing the floor misses, and that is the configuration in the repository. What
it removes is the reading D38 invited -- *"ordered, not complementary"* as a
statement about the two detectors. They are ordered **at one pair of operating
points**. Ranked against each other at equal catch counts, the forecaster
promotes six events the floor buries thirty ranks down, and five of the six are
short: four have a footprint of 1 and one of 64.

**And it does not yet vindicate the forecaster**, because matched recall is not
matched cost. Catching the same number of events says nothing about how many
alarms each spent doing it, and the floor alarms 3.71x more per nominal step on
the gate set at the frozen point (`docs/RESULTS.md` 6l). **M1, M2 and M3 remain
open as written.** What is now known is that their answer is not foreclosed:
before this, D38 made only-GRU look structurally zero, and it is not.

#### Predictions adjudicated so far

| # | Status |
|---|---|
| **M3** | **Open, and materially more likely to hold.** M3 asks for only-GRU >= 1 at a matched *false-alarm* point. On the matched *recall* axis it is 6 and 2, at 39 of 46 and 31 of 42 levels. Not the pre-registered test, and not scored |
| **M1, M2** | Open. Both are defined on the false-alarm axis and need the load |
| **M4** | Open. Partially answerable here: the three contextual events rank **38th, 42nd and 45th** of 46 by GRU reach and **39th, 42nd and 43rd** by `rstd` reach, so both would have to run near the bottom of their own event ordering to reach them. The precision axis needs the load |
| **M5 to M9** | Not started. Study B needs the per-channel residual matrix, which nothing caches |
| **M10 to M12** | See 23.14; C and D are costed, not run |

**Nothing was retrained, no threshold moved, the weight store is unchanged at 86
files, and no artifact under `runs/` was written.** This subsection is arithmetic
over a committed artifact.

### 23.14 Studies C and D, costed. Nothing is built

**2026-09-02, zero bucket operations.** 23.5 and 23.6 said what C and D are.
This says what they cost, which is the deliverable, and it is written before
either is approved because a design honest about its own cost is cheapest to
abandon on paper.

#### C -- leave-one-out cross-prediction

Parameter counts are derived from the architecture, not quoted: two GRU layers of
80, horizon 10, `C` channels. The model is checked against a real artifact --
at `C = 6` it gives **64,860**, and `runs/_weights/055d4af3db3f92eefeb06f4c78e37142.npz`
holds exactly 64,860 learned parameters. At `C = 12` it gives **71,160**, which
is `gru-quantile`'s published figure.

```
  baseline gru-quantile, C = 12      params  71,160     MAC/tick  70,080
```

| design | params | x base | MAC/tick | x base | fits / files |
|---|---|---|---|---|---|
| **(a)** `C` independent models, each `C-1 -> 1` | 744,120 | 10.46x | 732,480 | 10.5x | 12 fits, 12 files |
| **(b)** one shared trunk, target channel masked, `C` passes | **71,160** | **1.00x** | 840,960 | 12.0x | 1 fit, 1 file |
| **(c)** `C` input projections, shared recurrence, `C` passes | 99,960 | 1.40x | 732,480 | 10.5x | 1 fit, 1 file |

**M10 holds as written and was the wrong quantity to have predicted.** It named
`<= 2x` on **parameters**, and design (b) is 1.00x. But no leave-one-out design
avoids `C` recurrent passes: the exclusion has to hold per target channel, and a
single shared recurrent state mixes every channel at layer 0, so isolation
requires `C` states however the weights are arranged. **Every form costs about
`C` times the arithmetic**, 10.5x to 12.0x here. That is a Phase 4 envelope
question (Objective.md 12) before it is a Phase 3 accuracy question, and 23.8
should have predicted MACs. Recorded as a defect in the prediction, not in the
result.

**M11 holds, and for a sharper reason than "the shapes do not fit".** For design
(b) the shapes fit **exactly**: `l0_w_ih` is `(3*80, n_inputs = 12)` either way,
so a leave-one-out model of that shape is byte-compatible with a version-1
`model.bin`, and a version-1 reader would load it and **run it once instead of
twelve times**, silently computing the wrong residual. Compatibility is the
hazard here, not the obstacle. `docs/MODEL_FILE.md` already closes it: the
header carries `reserved0` and `reserved1`, and the standing rule is that a
**non-zero reserved field is refused, not tolerated** -- so declaring
leave-one-out in one of them makes every existing reader refuse the file, which
is a format change by construction and the safe outcome. `arch_id` reserves 2
and 3 for LSTM and TCN and could take a fourth value on the same argument.

**What C would still need before a line is written**: its own pre-registration;
a decision on whether the frozen decision layer is re-derived with it (D23, D25),
since a relationship-measuring residual reaching a channel-blind reduction
throws away what it was built to produce; and a training-time estimate, which is
not derivable from parameter counts and is the one number this scoping does not
have.

#### D -- the injected-fault study

**Operations: none of its own.** The injections are built in memory from nominal
windows of the fitting data, and the scoring uses the same resident arrays as A
and B. Run in one script with them, D adds **0 Class B and 0 Class A** to the
15 and 1 that A and B already cost.

**(!) A constraint that changes the design, and it is Rule 1.** An injected set
is real telemetry with values altered, so writing one to disk creates a **cached
dataset derived from telemetry on local disk**, which `docs/HARNESS.md` section
5 Rule 1 forbids and `tests/test_no_local_persistence.py` enforces -- and the
ban was tightened at work item 9 to cover `runs/` too. **So the injected set is
never stored as values.** What is stored under `runs/` is the **recipe**: fold,
window indices, the driver and target channel ids, the family, its parameters
and the seed -- metadata that regenerates the set exactly, in process, at no
bucket cost. That also makes D reproducible from cold, which storing arrays
would not.

**Blindness.** Whoever writes the injector knows every answer, so blindness
cannot be a matter of care. The recipe is generated and committed under its own
hash **before** any detector is run against it, and the scoring script reads
only the recipe and emits per-event catch flags; the mapping from event id to
family and parameters is not read by the scorer. That is checkable after the
fact from the two artifacts, which is the point.

**Detectability, and the honest failure mode.** 23.7 defines a detectable
injection as one whose peak reach under *some* detector in the study exceeds
1.0. An injection detectable by nothing is a property of the injection, and
23.8's M12 already names the outcome where the envelope constraint and
detectability prove incompatible as **a finding, not a failure**: it would mean
the contextual class as this project defines it is not reachable by residual
thresholding at all. On the evidence from 23.13 that is a live possibility --
the three real contextual events rank 38th, 42nd and 45th of 46 by GRU reach --
and D is the instrument that would settle it at `n >= 50` instead of `n = 3`.

**What D would still need**: the driver/target pairs, which require a
correlation study on nominal data that is itself one pass over the same load;
and a decision on whether injections land in test windows only, which they must,
since a fold's fitting window sets that fold's threshold.

### 23.15 OBSERVED -- studies A and B

**2026-09-02. One bundle load, 1 Class A and 15 Class B**, both channel sets,
all folds, all five reductions, cached weights, weight store unchanged at 86
files. Artifact `runs/m1-g8.9.10/_forensics/2026-09-02T193623Z-reduction-and-curve.json`.
Studies C and D were costed at 23.14 and remain unbuilt.

**The gate first.** The `max` arm is not a reimplementation of `gru-quantile`;
it is the same reduction of the same forecast, so it must reproduce the
published scorecard before anything else here is believed. All six checks pass:

```
  m1-g8.9.10   max  27/46 and 22/32   rstd  34/46 and 25/32   denominators 46, 32
  m1-ss5       max  26/42 and 21/31   rstd  34/42 and 25/31   denominators 42, 31
```

#### Study A: at a matched alarm rate the nesting reverses

D38 compared the two detectors at their own calibrated thresholds, where the
floor alarms on 18x more nominal steps than the forecaster on the gate set.
Held to the same nominal-step rate, the comparison inverts.

```
  m1-g8.9.10                     recall    MVGS   nominal-step rate
    frozen      gru-quantile      27/46   22/32       0.0013%
    frozen      rstd              34/46   25/32       0.0240%
    matched-quiet  rstd  x7.435    7/46    7/32       0.0009%
    matched-loud   gru   x0.346   35/46   26/32       0.0174%

  m1-ss5
    frozen      gru-quantile      26/42   21/31       0.0141%
    frozen      rstd              34/42   25/31       0.0408%
    matched-quiet  rstd  x6.083   11/42   11/31       0.0078%
    matched-loud   gru   x0.936   32/42   24/31       0.0169%
```

**Held to the forecaster's alarm rate, the floor finds 7 of 32 headline-cell
events, not 25.** Its 25/32 was bought with eighteen times the alarm rate. And
allowed the floor's alarm rate, the forecaster reaches **26/32** against the
floor's 25/32 on the gate set.

**The four sets at each operating point, and M3 is settled:**

```
  m1-g8.9.10          both  only-GRU  only-rstd  neither
    frozen              27       0         7        12     <- D38
    matched-quiet        7      20         0        19
    matched-loud        34       1         0        11

  m1-ss5
    frozen              26       0         8         8     <- D38
    matched-quiet       11      15         0        16
    matched-loud        32       0         2         8
```

**At every matched operating point on both sets, only-`rstd` is 0 or 2 and
only-GRU is 0 to 20.** D38's strict-subset result is an artifact of the two
detectors sitting at very different alarm rates, and at a matched rate the
containment runs the other way on the gate set. This is what 23.13 could only
suggest from the recall axis; it is now measured on the axis M1 to M3 were
written against.

#### Study B: every reduction is a strict subset of `max`, and none recovers anything

```
  m1-g8.9.10   arm    recall   MVGS   nominal steps flagged
                max    27/46   22/32       142
                l2      9/46    9/32         2
                sum     9/46    9/32         0
                k2      8/46    8/32         0
                k3      8/46    8/32        12

  m1-ss5        max    26/42   21/31     1,536
                l2     26/42   21/31     1,124
                sum     7/42    7/31       945
                k2      7/42    7/31       826
                k3      7/42    7/31       833
```

**Not one arm catches a single event `max` misses, on either set.** Every arm's
catch set is a strict subset of the flying reduction's, and none of the four
recovers any of `gru-quantile`'s 19 gate-set misses, any of the seven fold-1
events the floor catches, or any of the three contextual events -- and all four
run at a *lower* nominal-step rate than `max`, so the failure is not a threshold
handicap.

**The one thing a reduction improves is precision, on one set.** On `m1-ss5`
`l2` matches `max` event for event -- 26/42 and 21/31, the same events -- at
**1,124 nominal steps flagged against 1,536**, 27% fewer. On the gate set the
same arm collapses to 9/46. The mechanism is scale: aggregating across twelve
channels raises the calibrated floor faster than it raises an anomaly's peak,
because the anomalies are concentrated in a few channels while the noise is
spread across all of them. **That is D23's premise inverted** -- `max` wins
because cross-channel evidence is sparse, not because the decision layer was
never asked to look.

#### PREDICTED against MEASURED

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **M1** | at matched-quiet `rstd`'s MVGS falls to **10-20 of 32**, below the GRU's 22/32 | **7/32**, and **11/31** on `m1-ss5` | **Refuted on the band, held on the claim.** The direction is right and far past it; the band was too generous to the floor |
| **M2** | at matched-loud the GRU reaches **25-30 of 32**, at or above `rstd`'s 25/32 | **26/32** on the gate set; **24/31** on `m1-ss5`, below `rstd`'s 25/31 | **Held on the gate set, refuted on `m1-ss5`** |
| **M3** | only-GRU **>= 1** at at least one matched point | **20** and **1** on the gate set, **15** and **0** on `m1-ss5` | **Held, decisively.** The sharpest number in A |
| **M4** | swept to precision 0.5, `rstd` catches >= 1 contextual and the GRU 0 | **Not measured.** The sweep recorded nominal-step rate, not event-wise precision | **Unresolved -- an instrument gap, see below.** At both matched points neither detector catches any of the three |
| **M5** | `k2` and `k3` lose headline-cell recall against `k=1` on both sets | 8/32 and 8/32 against 22/32; 7/31 and 7/31 against 21/31 | **Held** |
| **M6** | at least one arm recovers **>= 1** of the 19 gate-set misses at no worse nominal rate | **0**, by all four arms, on both sets, at strictly lower nominal rates | **REFUTED. This is the finding, and it closes D23 as answered** |
| **M7** | no arm catches `id_121`, `id_153`, `id_157` | none does, on either set | **Held** |
| **M8** | no arm recovers any of the seven fold-1 only-`rstd` events | none does | **Held.** The loss is upstream of the reduction, in the per-channel EWMA, as 22.7 argued |
| **M9** | no arm beats `gru-quantile`'s 0.804 F0.5 on the gate set | **Not measured** | **Unresolved -- an instrument gap, see below** |

#### Two gaps in this instrument, named rather than left to be found

**M4 and M9 both need event-wise precision and this script did not record it.**
The sweep stores, per multiplier, the events caught and the nominal steps
flagged -- enough for a recall-versus-alarm-rate curve, which is what 23.3
specified, and not enough for F0.5, which needs alarm *ranges* classified as
true or false. `alarm_ranges` was never accumulated. That is a defect in the
script, not in the design: 23.8 asked for both quantities and only one was
instrumented.

Closing it costs **one further bundle load, 15 Class B and 1 Class A**, by
adding `mask_to_ranges` and `eventwise.score` to the same loop. It is not spent
here. Nothing in the results above depends on it: M1, M2, M3, M5, M6, M7 and M8
are all settled on quantities that were recorded.

**What this does not touch.** No detector, threshold, weight or decision layer
moved; no arm is proposed for adoption, and 23.11 trigger 6 makes that a
separate decision; `m2-ss1` and `m1-g3` were not loaded; `main` is untouched.

## 24. Pre-registration: cross-channel detection before the per-channel check (work item 9.8)

**Written and committed before a single figure is computed.** Work items 9.5 to 9.7
took the project's headline apart. The floor was corrected and rose (D37). The flying
detector was found to catch a strict subset of it (D38) -- and that turned out to be an
artifact of the two sitting at very different alarm rates: held to the forecaster's
rate the floor finds **7 of 32** headline-cell events rather than 25, and the
forecaster catches **20** the floor misses (D41). No cross-channel aggregation recovers
anything `max` misses, so D23 closed as answered no (D42). Through all of it the
contextual claim stayed dead: **3/32** by the 0.1/99.9 band, caught by nothing (D39),
with the count depending on the watched channel set (D40).

**The headline is held** (`docs/STATUS.md` section 7). This section is the evidence it
will be written from, and it is written before the evidence exists.

Four parts, one bundle load, cached weights, nothing refitted. Nothing here changes a
detector, a threshold, a weight, the frozen decision layer or `main`.

### 24.1 The four parts, and why each is asked

1. **Contextual, redefined as the literature means it.** D39 measured the contextual
   class with the **0.1/99.9 quantile band** of the fitting window. That band is this
   project's own noise envelope; it is not a limit, and no limit check ever held it. A
   real RED or YELLOW limit sits **outside** a channel's historical range, so the
   tightest bar any limit could possibly hold is the training-window **min/max**. Under
   that reading the class is larger, and D39 measured the wrong quantity for the claim
   it was testing.
2. **Lead against a per-channel range check, per event, at a matched alarm rate.** The
   project's purpose is warning *before* a limit trips, and Objective.md 1.1 retired
   the only lead figure it had. ESA-ADB carries no dictionary limits, so the honest
   proxy is: widen a per-channel envelope until it is exactly as noisy as we are, and
   measure how many timesteps earlier we speak.
3. **The amplitude mechanism.** If the residual crosses only when the raw channel is
   already extreme, then the forecaster is an expensive magnitude detector and the
   cross-channel story is decoration. This is the test that could show that.
4. **Per-channel noise floors.** D23 named channel-blindness in two forms. Study B
   closed the first -- aggregation across channels (D42). The second is untested: one
   loud channel sets a **single global bar** for eleven quiet ones, so a quiet channel
   must become as surprising as the loudest to be heard at all.

### 24.2 Definitions fixed in advance

Fixed here so they cannot be chosen after the answer is visible.

- **Contextual (this section's primary reading)**: an event where **no watched channel
  leaves its own training-window min/max** at any step inside the event span, the
  envelope built from fitting data only with `train_mask` applied. This is the
  construction `scripts/envelope_proxy.py:95-110` already uses and 22.2 already
  reported beside the quantile pair. **It is a lower bound on the in-limits class**: a
  real limit is wider than the historical range, so an event that never leaves min/max
  cannot have tripped one, and the implication runs only that way.
- **Per D40, every contextual count names the channel set it is contextual with respect
  to.** That rule is not repealed by the change of definition; it applies to the
  min/max reading exactly as it applied to the quantile one, and more sharply, since
  the min/max count is the view-dependent one.
- **Widened per-channel envelope (part 2)**: per channel, the training-window min/max
  centre-scaled by a single factor `w` shared across channels, `w` swept until the
  envelope's **nominal-step alarm rate** equals `gru-quantile`'s on the same fold. The
  alarm is "any watched channel outside its widened envelope", the same
  `first_break` rule as `scripts/envelope_proxy.py:113-119`.
- **Lead against that check**: for a caught event, the first in-span step at which any
  channel leaves the widened envelope, **minus** our honest emission step -- positive
  means we spoke first. Both in test-window coordinates, both in **timesteps, never
  hours** (`docs/HARNESS.md` section 4). Our emission is the harness's own re-dated
  alarm start (`harness.py:208-224`); in quantile mode `last_emission` is `None` and
  the crossing **is** the emission, so for `gru-quantile` this is the first crossing.
- **Raw z and residual z (part 3)**: at the first crossing inside an event, the raw
  channel's value minus its anomaly-masked fitting-window mean over that window's
  standard deviation; beside it, the smoothed residual on the same channel over the
  same window's residual standard deviation. The channel is the one attaining the
  maximum smoothed residual at that step.
- **The per-channel bar (part 4)**: per channel, the **0.999 quantile of that channel's
  own smoothed residual** over the anomaly-masked fitting window -- **D25's recipe
  applied per column**, not a new recipe. Each column is divided by its own bar, the
  maximum is taken across channels, and the cut is 1.0.
  `Detector.threshold_from` (`src/sentinel_eval/detector.py:127-132`) cannot produce
  this: its `np.isfinite` mask flattens a `(T, C)` matrix and it returns a single
  scalar pooled over every channel and step. `np.nanquantile(..., 0.999, axis=0)` is
  used explicitly and the difference is stated because it is the whole point of part 4.
- **Matched operating point**: equal **nominal-step false-alarm rate**, computed on the
  same fold's clean nominal steps. Unchanged from 23.7. Not equal alarm count, not
  equal precision.

**(!) Part 4 is not a per-channel scaler and does not breach D2.**
`src/sentinel_eval/normalisation.py` refuses `per_channel_zscore` and its relatives and
`tests/test_no_per_channel_scaler.py` enforces it, because rescaling **input values**
per channel erases the amplitude ratios between related channels -- the information the
cross-channel claim rests on. Part 4 normalises **residuals**, after the multivariate
forecast has already used those ratios. `telemanom.py:362-437` and `oscfar.py:279-348`
both already do this. Identity normalisation on inputs is untouched.

**(!) Part 4's arm is louder than `gru-quantile` by construction, and the comparison is
made at a matched rate only.** "Any channel over its own 99.9th percentile" is roughly
`C` chances to exceed a 99.9th percentile where `gru-quantile` has one. Reporting its
recall beside the forecaster's at the natural cut would repeat exactly the error D41
was written to correct. **Every comparison against `gru-quantile` is stated at the
multiplier where their nominal-step rates are equal. The cut-at-1.0 figure is reported
for completeness, is labelled informational, and no verdict, no prediction and no
sentence of any headline rests on it. N7 and N8 are adjudicated at the matched rate
only.**

### 24.3 What is already published, and is therefore not predicted

A prediction written after its own answer is theatre; 22.4 said so and 22.11 was filed
as a derivation rather than an experiment for exactly this reason. Three quantities
this section uses are **already in print** and no `N` is spent on them:

```
  min/max-contextual, 12-channel view   6/32     docs/MODELS.md 22.11, D40
  min/max-contextual,  6-channel view  18/31     docs/MODELS.md 22.11, D40
  caught by either detector, frozen      0  and  12
```

What is new in part 1 is the **ids**, and the catch counts at the **matched-rate**
operating points D41 established rather than at the frozen thresholds. Those are
predicted below.

The operating points themselves are also already measured (23.15) and are inputs here,
not results:

```
  m1-g8.9.10   matched-quiet  rstd x7.435       matched-loud  gru x0.346
  m1-ss5       matched-quiet  rstd x6.083       matched-loud  gru x0.936
```

**And the nearest prior art for part 4 is on record.** `lstm-oscfar`'s bare local order
statistic -- a per-channel calibrated threshold -- reached 21/46 recall, **19/32
headline cell**, 85 alarm ranges and a **+26.0** median lead at a 0.1% budget
(`docs/DECISIONS.md` D20's OUTCOME block). That is LSTM-era, pre-correction, and on a
different rule; it settles nothing. It is quoted because N7 should be written against
the best available prior rather than against a blank page.

### 24.4 PREDICTED

Committed before the script runs. `N2`, `N4` and `N7` are stated in the direction that
would hurt.

**Part 1 -- contextual under the min/max reading.**

| # | Prediction | Refuted by |
|---|---|---|
| **N1** | the gate set's **6** min/max-contextual ids are a **subset** of `m1-ss5`'s **18** | any gate-set id absent from `m1-ss5`'s. D40 established that removing watched channels can only move events **into** the class, so a violation would mean the two envelopes are not nested and D40's monotonicity argument is wrong |
| **N2** | at the matched-rate operating points, `gru-quantile` catches **>= 1** of the gate set's 6, central estimate **2** | **0**. This is the sharpest number in the work item. A catch here is the project's **first measured detection of an event no limit check could have seen**; zero leaves D39 consequence 2 standing exactly as written -- no measured instance, on the corrected definition as well as the old one |
| **N3** | on `m1-ss5`, over the 18, `gru-quantile`'s catch count at matched rate **exceeds** the corrected `rstd`'s | the floor equalling or beating it, which would put the one set where the contextual class is large in the floor's hands |

**Part 2 -- lead against a per-channel range check at a matched alarm rate.**

| # | Prediction | Refuted by |
|---|---|---|
| **N4** | the per-event lead has a **positive median** on **both** sets | a median **<= 0** on either. That would say a range check as noisy as we are sees these events no later than we do, and the early-warning argument loses its last proxy on this data -- with Objective.md 1.1's retirement of "+26" already standing, there would be nothing left to replace it with before Phase 3 |
| **N5** | **>= 60%** of caught events have their first envelope breach **after** our emission | below **50%** |

**Part 3 -- the amplitude mechanism.**

| # | Prediction | Refuted by |
|---|---|---|
| **N6** | at the first crossing, the raw channel's z stays inside **3 sigma** for **>= 60%** of caught events while the residual's z exceeds it | below **40%**. The forecaster would then be crossing mostly when the raw value is already extreme, and the cross-channel account of what it does would be a description of a magnitude detector |

**Part 4 -- per-channel noise floors, adjudicated at the matched rate only.**

| # | Prediction | Refuted by |
|---|---|---|
| **N7** | at a matched nominal-step rate, the per-channel bar recovers **>= 3** of `gru-quantile`'s **19** gate-set misses | **0**, which closes D23's second form the way D42 closed its first: neither aggregating across channels nor giving each channel its own bar recovers anything, and the channel-blind decision layer costs no measured event on this data |
| **N8** | it recovers **0** of the three 0.1/99.9-contextual events (`id_121`, `id_153`, `id_157`) at the matched rate | any of the three, which would be a first and would require D39 to be re-read a second time |

**Deliberately not predicted.** Which channel part 3's crossings land on. Whether part
4's arm beats `gru-quantile` on F0.5 -- M9 is being closed in the same run and naming a
favourite here would contaminate it. The size of part 2's lead beyond its sign and
median. Anything about `m2-ss1` or `m1-g3`, which are spent (D29) and not loaded.

### 24.5 The named risk

**That this work item is the third attempt to find the forecaster a win.** 9.6 withdrew
the per-event argument, 9.7 restored it at a matched rate, and this one redefines the
contextual class in a direction that makes the class larger. A reader is entitled to
ask whether the definition moved because the answer was inconvenient.

The defence is that the direction was **stated before it was measured and is stated
against interest**: 22.2 fixed both readings in advance, 22.11 published the min/max
figures beside the quantile ones, and D40 recorded that the min/max reading is the
**unstable** one -- 6/32 against 18/31 on two views of the same data. Adopting it makes
the class larger *and* the count more fragile, and 24.2 keeps D40's naming rule rather
than dropping it. N2 then names zero as the outcome that leaves D39 untouched.

The second protection is that **nothing here can rescue the claim by itself**. Part 1
can only change the size of the class; it takes N2 to put a single event in it, and N2
is one number with no band to hide in.

### 24.6 Cost

**One bundle load: 15 Class B and 1 Class A**, both channel sets, all folds, every arm,
from cached weights. Parts 2, 3 and 4 each need per-channel data that nothing caches;
part 1 needs no new data at all and is a join of two committed artifacts. The month
stands at **4 Class A and 71 Class B** of 50,000 each, tripwire 1,000 per run.

`scripts/reduction_and_curve.py` is extended rather than replaced, so this is the same
single load Studies A and B used. **M4 and M9, left unresolved at 23.15 because
event-wise F0.5 was never instrumented along the sweep, are closed in the same run at
no additional cost.**

**Every new code path is exercised on the offline synthetic fixture, at zero
operations, before anything is spent.**

### 24.7 Stop and report

1. **The reproduction gate fails.** The `max` arm and `rstd` must reproduce 27/46,
   22/32, 34/46, 25/32 on the gate set and 26/42, 21/31, 34/42, 25/31 on `m1-ss5`
   before **any** new figure in this section is read. A failure means the extension
   broke something and its new numbers are worthless until that is found. Report the
   failure and nothing else.
2. **N2 refuted** -- zero contextual catches under the corrected definition too.
3. **N4 refuted** -- the range check is not later than we are.
4. Any figure belonging to a frozen artifact moves: `gru-quantile`'s scorecard, the
   decision layer, a weight, a held-back score.
5. More than **20 Class B** in the run, or the per-run tripwire at 1,000.
6. Any part-4 comparison is about to be stated at the natural cut rather than at the
   matched rate.

### 24.8 What this decides, and what it may not

**Decides.** Whether the contextual class is larger under the definition the literature
means, and whether anything catches an event inside it (parts 1, N2, N3). Whether this
project has a measurable lead over a per-channel range check at its own alarm rate
(part 2, N4, N5). Whether the forecaster's crossings are cross-channel or amplitude
(part 3, N6). Whether per-channel calibration recovers anything the global bar misses,
which is D23's second form (part 4, N7, N8). And M4 and M9, carried from 9.7.

**May not.** Move any threshold, retrain anything, alter the frozen decision layer
(D25), adopt part 4's arm, re-score a held-back set, or touch `main`. Edit D39 or D40 --
a new entry re-reads them and both stand as written. Or write the headline: that is
held at `docs/STATUS.md` section 7, it is written once from Studies A and B and this
section together, and it is written for approval rather than committed.

### 24.9 OBSERVED

**2026-09-03. One bundle load, 1 Class A and 15 Class B**, both channel sets, all
folds, cached weights, weight store unchanged at 86 files. Artifact
`runs/m1-g8.9.10/_forensics/2026-09-03T172705Z-reduction-and-curve.json`.

**The gate passed first.** All six reproduction checks: `max` 27/46 and 22/32,
`rstd` 34/46 and 25/32 on the gate set; 26/42, 21/31 and 34/42, 25/31 on
`m1-ss5`; denominators 46/32 and 42/31. Only then was anything new read.

**(!) A defect in this section's own instrument, found by its own cross-check and
fixed before any figure was reported.** Part 1's min/max flag is the `w = 1.0`
case of part 2's envelope, which is why it was computed there and cross-checked
against the floor audit. The first run disagreed on **2** gate-set and **14**
`m1-ss5` events, and every one of them had an envelope reach of exactly
**1.000**: the script compared `>=` where the published definition is strict.
An event that *touches* a channel's historical extreme has not *left* it, and
`scripts/envelope_proxy.py:115`, `scripts/floor_audit.py` and D39, D40 and
22.11 all use `(x < lo) | (x > hi)`. ESA-ADB is min-max scaled per group, so
values sitting exactly on a training extreme are common rather than rare. Under
the strict rule the counts reproduce the floor audit **exactly, 11/11 and 24/24
over all events and 6/32 and 18/31 over the headline cell**, and the script now
carries a `STRICT` rule with the reason. **No re-run was needed**: an exact
float equality occurs only at `w = 1.0`, every other multiplier on the grid is
unaffected, and no matched operating point is at 1.0.

#### Part 1 -- contextual under the definition a limit check would hold

```
  min/max-contextual, MVGS      m1-g8.9.10  6/32      m1-ss5  18/31
    gate set ids   id_121, id_13, id_153, id_157, id_89, id_96
```

The gate set's six are the three the 0.1/99.9 band already found (`id_121`,
`id_153`, `id_157`) plus `id_13`, `id_89` and `id_96` -- the three large
fold-0 events in the `neither` set of 22.11. Caught, at each operating point:

```
  m1-g8.9.10, over the 6      gru-quantile   rstd    neither
    frozen thresholds              0/6        0/6      6/6
    matched-quiet                  0/6        0/6      6/6
    matched-loud                   1/6        0/6      5/6

  m1-ss5, over the 18
    frozen thresholds              9/18      12/18     6/18
    matched-quiet                  9/18       1/18     9/18
    matched-loud                  11/18      12/18     6/18
```

**The one catch on the gate set is `id_89`**, footprint 8,995, at the
matched-loud point -- the forecaster loosened to the floor's alarm rate, thirteen
times its own. At its own operating point it catches none of the six.

#### Part 2 -- lead against a per-channel range check, and it is the finding

The envelope is built from the fitting window only, with `train_mask` applied --
the same window and mask `gru-quantile`'s own calibration uses -- then widened
by a single factor until its nominal-step rate matches the forecaster's.

```
  m1-g8.9.10                    recall    MVGS    nominal      F0.5
    gru-quantile, frozen         27/46   22/32   0.0013%      0.804
    range check, matched w=1.106 34/46   25/32   0.0000%      0.934

  m1-ss5
    gru-quantile, frozen         26/42   21/31   0.0141%      0.593
    range check, matched w=0.672 34/42   25/31   0.0002%      0.543
```

**At an equal or lower alarm rate the range check catches more than the
forecaster on both sets**, and on the gate set its event-wise F0.5 is 0.934
against 0.804. The overlap at the matched point is **both 27, only-GRU 0,
only-envelope 7, neither 12** on the gate set and **26 / 0 / 8 / 8** on
`m1-ss5`: the forecaster catches nothing the range check misses.

**And the lead, which is what part 2 exists to measure:**

```
  timesteps from our first crossing to the first channel leaving the envelope
    m1-g8.9.10   n=27   median +0.0   mean -105.1   we fire first  0/27  (0%)
    m1-ss5       n=26   median +0.0   mean   -0.3   we fire first  0/26  (0%)
```

**Not once in 53 caught events does the forecaster speak before the range
check.** On 39 of them the two fire at the same step; on 14 the range check is
earlier, by up to 2,638 timesteps.

**A limitation, the same one 22.10 named.** Both figures are the first in-span
step at or above threshold, so neither can distinguish "crossed at onset" from
"an alarm was already running". The *comparison* is like for like -- both
detectors are measured the same way -- but neither can be said to have warned
before the labelled event began.

#### Part 3 -- the amplitude mechanism

At the first crossing, the raw value of the channel attaining the maximum
residual, against that channel's own anomaly-masked fitting-window mean and
standard deviation:

```
                     raw |z| at our first crossing
  m1-g8.9.10   n=27   min 6.60    median 103.84   max 26,602
  m1-ss5       n=26   min 17.72   median 102.94   max 117.58
    below 3 sigma: 0 of 53        below 10 sigma: 1 of 53
```

**There is no crossing, on either set, at which the raw channel is still inside
3 sigma.** The median is a hundred standard deviations outside its training
distribution. The forecaster is not crossing on subtle relationship breaks; it
crosses once the channel that raised it is already far outside anything the
fitting window contained.

**Stated with its own caveat**: ESA-ADB is min-max scaled per group, so a
channel's fitting-window standard deviation can be very small next to an
excursion spanning the scaled range, and a "3 sigma" bar is low for data at this
scale. The bar was fixed in 24.2 before the numbers existed and is not moved
now. The direction is not in doubt at 0 of 53 with a median of 103.

#### Part 4 -- per-channel noise floors, at the matched rate

```
  m1-g8.9.10                        recall    MVGS    nominal     F0.5
    gru-quantile, frozen             27/46   22/32   0.0013%     0.804
    per-channel bar, matched w=1.651 25/46   22/32   0.0012%     0.805
    per-channel bar, cut at 1.0      28/46   23/32   0.2678%     0.044   [informational]

  m1-ss5
    gru-quantile, frozen             26/42   21/31   0.0141%     0.593
    per-channel bar, matched w=1.494 29/42   22/31   0.0117%     0.700
    per-channel bar, cut at 1.0      34/42   25/31   0.2510%     0.460   [informational]
```

24.2 fixed the natural cut as informational and it is: at 1.0 the arm is 206x
louder than the flying detector on the gate set, and its F0.5 collapses to 0.044.
**At a matched rate it is a tie on the gate set** -- 22/32 either way, F0.5 0.805
against 0.804 -- **and better on `m1-ss5`**, 22/31 against 21/31 at F0.5 0.700
against 0.593. It recovers **0** of the forecaster's 19 gate-set misses and
**3** of its 16 on `m1-ss5` (`id_130`, `id_132`, `id_142`).

#### PREDICTED against MEASURED

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **N1** | the gate set's 6 min/max-contextual ids are a subset of `m1-ss5`'s 18 | 5 of the 6 are; **`id_13` is not scorable on `m1-ss5` at all** -- it is the one MVGS event the subset view drops | **Refuted as written, held on the 5 common events.** The prediction assumed a common event set and the two sets differ by one |
| **N2** | at a matched-rate point, `gru-quantile` catches **>= 1** of the gate set's 6; central estimate 2 | **1**, `id_89`, and only at matched-loud -- the forecaster run at the floor's alarm rate, 13x its own. **0** at its own operating point and 0 at matched-quiet | **Held, at the bottom of its band and on a loosened detector** |
| **N3** | on `m1-ss5`, over the 18, the forecaster's matched-rate catch count exceeds the floor's | matched-quiet **9 against 1**; matched-loud **11 against 12** | **Held at matched-quiet, refuted at matched-loud.** The prediction named "the matched points" without saying which, and they disagree |
| **N4** | the per-event lead has a **positive median** on both sets | median **+0.0** on both, mean **-105.1** and **-0.3** | **REFUTED. Stop-and-report trigger 3 fired** |
| **N5** | **>= 60%** of caught events breach the envelope after our emission | **0 of 53**, on both sets | **REFUTED, at the floor of the range** |
| **N6** | raw \|z\| inside 3 sigma for **>= 60%** of crossings while the residual exceeds it | **0 of 53**; median raw \|z\| **103.84** and **102.94** | **REFUTED, at the floor of the range** |
| **N7** | the per-channel bar recovers **>= 3** of the 19 gate-set misses at a matched rate | **0** on the gate set; 3 on `m1-ss5` | **Refuted on the set it was stated for** |
| **N8** | it recovers **0** of `id_121`, `id_153`, `id_157` at a matched rate | **0**, on both sets | **Held** |

**Two of the three predictions written in the direction that would hurt fired.**
N4 and N7 are refuted; N2 held, by one event, on a detector loosened to thirteen
times its own alarm rate.

#### M4 and M9, carried from work item 9.7 and now closed

**M4 -- refuted.** It predicted that swept down to the alarm rate at which
event-wise precision first falls below 0.5, `rstd` catches at least one of the
three 0.1/99.9-contextual events and `gru-quantile` catches none. Measured:
**neither catches any**, on either set. `max` reaches precision 0.393 at
`m = 0.588` with 33/46 recall and `rstd` 0.223 at `m = 0.718` with 35/46, and the
contextual three are absent at both.

**M9 -- held on the set it was stated for.** No Study B arm beats
`gru-quantile`'s 0.804 F0.5 on the gate set, and it is closed **analytically**
rather than by measurement: event-wise F0.5 at perfect precision is
`1.25R / (0.25 + R)`, so `l2` and `sum` at 9/46 are bounded by **0.549** and
`k2`, `k3` at 8/46 by **0.513**. None can reach 0.804 however precise. On
`m1-ss5` `l2`'s bound is 0.890 against the forecaster's 0.593, so it *could*
beat it; that arm was not swept and its F0.5 there is **not computed**. Part 4's
per-channel arm, which did not exist when M9 was written, ties the gate set at
0.805 and beats `m1-ss5` at 0.700.

#### What this section did not measure

`l2`'s event-wise F0.5 on `m1-ss5` -- the four Study B arms were not in the swept
set, so only the analytic bound is available there and it does not settle it.
Whether either detector fires before an event begins: 24.9's part 2 note and
22.10 name the same instrument gap, and it needs the alarm state before the span,
which nothing here records.

#### Triggers

**Trigger 3 fired** -- N4 refuted -- and it is the subject of D43. Trigger 1 did
not: the reproduction gate passed all six checks. Trigger 2 did not: N2 held.
Trigger 4 did not: no frozen figure moved and the weight store is unchanged at 86
files. Trigger 5 did not: 15 Class B against 20, month at 6 Class A and 86 Class
B of 50,000 each. Trigger 6 did not: every part-4 comparison above is stated at
the matched rate and the natural cut is labelled informational on both sets.

### 24.10 Scoped, not run

- **The matched-rate curve as a mission-selectable operating point** for the toolkit's
  pre-launch sanity report (`docs/STATUS.md` section 7). A mission picks an alarm rate
  it can absorb and reads the recall that comes with it; the curve is documented before
  launch and **never tuned on results**, which is `scripts/oscfar_curve.py`'s standing
  rule and section 7's refusal of the oracle sweep.
- **The fold-1 floor rise as the target for forecaster improvement.** Every only-`rstd`
  event on both sets is in fold 1 (22.11), and six of the seven are the events the GRU
  lost to a floor 1.81x higher (section 14 at :1517-1520). Training window, seed, and
  the in-flight fine-tuning tiers of Objective.md 14.10 are the levers; none is pulled
  here.
- **Study D, the injected relationship breaks** -- `>= 50` events, every channel kept
  inside its envelope, scored blind -- costed at 23.14 and deferred until after work
  item 10.

## 25. Pre-registration: are the forecaster's nominal-period alarms precursors, or noise? (work item 9.9 study 2)

**Written and committed before a single figure is computed.** D44 measured that a
per-channel range check, widened until it is exactly as noisy as the forecaster,
catches more than the forecaster on both Mission 1 sets and is never later --
0 of 53 caught events where we speak first. What that leaves untested is the
forecaster's **other** output: the alarms it raises in nominal periods, which
every scorecard in this repository counts as false.

If those alarms cluster in the window **before** a later labelled anomaly, they
are early warning that the event-wise metrics have been scoring as error. If they
do not, they are noise, and D44's reading stands with nothing behind it.

**A null result is the finding**, and this section says so before the test runs.

Nothing here changes a detector, a threshold, a weight, the frozen decision layer
or `main`. One bundle load, cached weights, nothing refitted.

### 25.1 Why this is asked now

Every alarm outside a labelled span is currently a false alarm by definition, and
that definition has never been checked. `docs/HARNESS.md` section 1 counts them
against precision and the adoption number, which is correct for a detector that is
wrong -- and exactly backwards for one that is early. The two are
indistinguishable on the metrics as they stand, and the only way to tell them
apart is whether they land where a precursor would.

The population is restricted to **in-range nominal time**, where every watched
channel is inside its own training min/max, because that is the region a limit
check cannot see (D43). An alarm there is the only kind that could be worth
anything the range check of D44 does not already deliver.

### 25.2 Definitions fixed in advance

- **Eligible region**: test-window steps that are `scorable`, not `anomaly`, not
  `rare_event`, **and** with every watched channel strictly inside its own
  training-window min/max -- D43's envelope at `w = 1.0` under the strict rule
  24.9 established. This is nominal time that is also invisible to a range check.
- **Alarm start**: the first step of a maximal run of `gru-quantile`'s alarm mask
  at its frozen threshold, counted only where that first step is in the eligible
  region. Runs, not steps, so one claim to an operator counts once.
- **Pre-anomaly window**: the `W` steps immediately before a labelled `Anomaly`
  event's start, intersected with the eligible region. Overlapping windows are
  merged, so an alarm is counted once however many events it precedes.
- **Group match**: the alarm's attributed channel -- the one attaining the maximum
  smoothed residual at that step -- shares an ESA `Group` with at least one channel
  the event touches (`docs/HARNESS.md` 6a, `Event.channels`).
- **W**: **primary 10,000 timesteps.** Sensitivity at **1,000** and **100,000**,
  both reported, neither able to become the primary after the fact.
- **Null 1, primary -- circular shift.** The fold's alarm indicator is rotated by a
  uniformly random offset on the torus of its test window, 10,000 times. This
  preserves the alarm process's own temporal structure -- its rate, its
  autocorrelation, its clustering -- and randomises only its phase against the
  anomalies. It is the null that cannot be beaten by "alarms are bursty".
- **Null 2, secondary -- uniform placement.** The same number of alarm starts
  placed uniformly at random in the eligible region, 10,000 draws. Simpler,
  and wrong in the specific way null 1 is designed to survive; reported beside it
  because a disagreement between the two is itself informative.
- **Statistic**: the count of eligible alarm starts falling inside a group-matched
  pre-anomaly window.
- **Effect**: the rate ratio, observed divided by the null mean. **p**: one-sided
  for clustering, `(1 + #{null >= observed}) / (1 + N)`.
- Both channel sets, per fold and pooled, every count as `k/n`, `n < 20` stamped
  UNDERPOWERED.

### 25.3 PREDICTED

| # | Prediction | Refuted by |
|---|---|---|
| **S1** | the eligible region is **>= 50%** of nominal test time on the gate set -- in-range nominal time is the common case, not a corner | below 50%, which would mean a range check sees most of nominal time too and the population this study needs is thin |
| **S2** | `gru-quantile` produces **>= 20** eligible alarm starts per set | below 20, in which case the set is stamped UNDERPOWERED and no p-value from it is quoted as evidence either way |
| **S3** | **primary.** At `W = 10,000` on the gate set, under the circular-shift null, the rate ratio is **> 1** with **p < 0.05** | **p >= 0.05.** This is the study. A null result says the forecaster's nominal-period alarms are not precursors, the metrics have been right to count them as false, and D44's reading stands with nothing behind it. It is reported in those words |
| **S4** | the two nulls agree in **direction** | disagreement, which would locate the effect in the alarm process's own burstiness rather than in its placement relative to anomalies -- and would make null 2's p-value the misleading one |
| **S5** | the effect is present at `W = 1,000` as well as `W = 10,000` | an effect only at `W = 100,000`, which at 100k steps against a ~3.5M-step window is a third of nominal time and would indicate a wide-window artifact rather than a precursor |
| **S6** | `m1-ss5` agrees in direction with the gate set | opposite signs, which would leave the result set-dependent in the way D41 and 6l already caught once |

**Deliberately not predicted.** The size of the effect if there is one. Which
group or which events carry it. Anything about `m2-ss1` or `m1-g3`, which are
spent (D29) and are not loaded.

### 25.4 The named risk

**That this is the fourth attempt to find the forecaster a win**, after 9.6
withdrew the per-event argument, 9.7 restored it at a matched rate, and 9.8 took
it away again. A permutation test with three window lengths, two nulls and two
channel sets offers twelve ways to produce a small p-value.

The protections are that the primary is fixed here -- **`W = 10,000`, circular
shift, gate set, one-sided** -- and everything else is declared sensitivity before
any of it exists; that the structure-preserving null is the primary rather than the
one easier to beat; and that **S3 names the null result as the finding** rather
than as a failure.

**And the honest limit, stated now.** A precursor effect cannot distinguish "the
anomaly began before its label was written" from "an independent early signal".
The labels are hindsight annotations by operations engineers (Objective.md 1.1),
and an onset earlier than the annotation is the most likely explanation for any
clustering this finds. That would still be useful -- it would mean the detector
speaks before the label -- but it is **not** the same as predicting an event that
has not started, and no wording will be allowed to blur the two.

### 25.5 Cost

**One bundle load: 15 Class B and 1 Class A**, both channel sets, all folds, from
cached weights. The month stands at **6 Class A and 86 Class B** of 50,000 each,
tripwire 1,000 per run. The permutation is arithmetic on resident arrays and costs
nothing. Every new code path runs on the offline fixture first, at zero operations.

### 25.6 Stop and report

1. **S3 refuted** -- no clustering. Report it as the finding, in those words.
2. **S2 refuted on both sets** -- too few eligible alarms to test anything. Report
   the power failure rather than a p-value.
3. **S4 refuted** -- the two nulls disagree.
4. Any frozen figure moves: `gru-quantile`'s scorecard, a threshold, a weight.
5. More than 20 Class B, or the per-run tripwire.

### 25.7 What this decides, and what it may not

**Decides** whether the forecaster's nominal-period alarms in the region a range
check cannot see are placed like precursors or like noise, and therefore whether
the event-wise metrics have been mis-scoring them.

**May not** move a threshold, retrain anything, alter the frozen decision layer,
re-score a held-back set, change how the harness counts a false alarm, or touch
`main`. Re-open D44, which measured catches and lead and is not addressed by this.
Or write the headline, which is held at `docs/STATUS.md` section 7.

### 25.8 OBSERVED

**2026-09-03. One bundle load, 1 Class A and 15 Class B**, both channel sets, all
folds, cached weights, nothing refitted during the run. Artifact
`runs/m1-g8.9.10/_forensics/2026-09-03T185207Z-precursor.json`.

**Stop-and-report trigger 2 fired: S2 is refuted on both sets, and the reason is
the finding.** There is no precursor population to test, because the forecaster
raises **no independent alarm in in-range nominal time at all**.

```
                        nominal steps   eligible (in-range)   eligible alarm STARTS
  m1-g8.9.10             10,675,488     10,675,463 (100.00%)          0
  m1-ss5                 10,875,689     10,875,683 (100.00%)          6
```

**All 157 of `gru-quantile`'s alarm ranges on the gate set begin inside a
labelled anomaly or rare-event span.** Not one begins in quiet, in-range nominal
time. Its 142 nominal-flagged steps (24.9) are therefore **overhang** -- the tails
of alarms that started inside a labelled span and ran past its end -- and not
independent alarms that could have been early warnings of anything.

On `m1-ss5` six independent starts exist. One of them falls in a 100,000-step
pre-anomaly window against a null mean of 32, a rate ratio of **0.03x** -- far
*fewer* than chance rather than more. At **n = 6** that is UNDERPOWERED by
`docs/HARNESS.md` section 1's own rule and **no p-value from it is quoted as
evidence either way**, exactly as S2 said in advance.

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **S1** | the eligible region is >= 50% of nominal test time | **100.00%** on both sets -- only 25 and 6 nominal steps in ~10.7M are out of range | **Held, overwhelmingly** |
| **S2** | >= 20 eligible alarm starts per set | **0** and **6** | **REFUTED on both sets. Trigger 2 fired** |
| **S3** | primary: rate ratio > 1 at W=10,000 with p < 0.05 under the circular shift | **not evaluable.** The observed statistic is 0 of 0 alarms on the gate set | **No verdict.** S2's power condition was not met and 25.3 said in advance that no p-value would be quoted from an underpowered set |
| **S4** | the two nulls agree in direction | they do -- both null means track each other at every W, and both give p = 1.0000 | **Held, vacuously.** With an observed statistic of 0 the agreement carries no information |
| **S5** | the effect is present at W=1,000 as well as W=10,000 | no effect at any W | **No verdict**, for S3's reason |
| **S6** | `m1-ss5` agrees in direction with the gate set | both are at or below chance | **Held, vacuously** |

#### What this actually establishes, and what it does not

**Establishes.** The forecaster's false-alarm count is not made of spurious alarms
in quiet time. Every alarm range it raises on the gate set starts inside something
the labels already mark, so the precision figure is being spent on **where alarms
end**, not on where they begin. That is a different defect from the one the
metrics have implied, and a smaller one.

**Does not establish** that the forecaster has no early-warning capability. The
test looked for alarms that *begin* in nominal time and there are none; it cannot
speak about the ones that begin inside a labelled span, and D44 already measured
that those do not precede a rate-matched range check. It also cannot say **where**
those 157 ranges begin among anomaly and rare-event spans -- the script recorded
eligibility, not the complement -- and settling that needs another bundle load
this section's budget does not allow.

**And the null is reported as the finding**, which is what 25.3's S3 committed to
in advance. What was not anticipated is that the study would fail on power rather
than on effect: 25.3 predicted at least 20 eligible alarm starts and there were
zero. That is a prediction refuted, not a study that could not be run.

#### The instrument, and one bookkeeping fact

The permutation was verified before it was spent: on planted precursors it returns
p = 0.009 under the circular shift and 0.0001 under uniform placement at a rate
ratio of 3.98, and on randomly placed alarms of the same count it returns 0.72 and
0.80. It has power and it is calibrated; the null here is the data's, not the
test's.

**The weight store is 89 files, not the 86 quoted since work item 9.6.** The three
additions are dated 2026-09-03 11:41 and are **synthetic-fixture weights** written
by this section's offline dry runs, which pass `--allow-fit` because the fixture
has no cache. The in-run invariant held -- the count was identical before and
after the Mission 1 run -- so nothing was refitted on Mission 1 data. Recorded
because a quoted invariant that quietly changes is worse than one that moves for a
stated reason.

## 26. Pre-registration: is the labelled contextual class actually in range? (work item 9.9 study 1, stage 1)

**Written and committed before a single figure is computed, and after the data was
ingested but before it was read back.**

**(!) This is not a test of the cross-channel thesis, and no result from it may be
quoted as one.** Objective.md 9.2 demoted SMAP/MSL for a reason that has not
changed: its 81 channels are **unsynchronised univariate streams**, so there are no
genuine cross-channel relationships in them to find, and the project's core claim
could not be proven on it even in principle. What this section tests is narrower and
answerable: **whether a forecaster with command context beats a calibrated range
check on anomalies that are genuinely inside their own historical range** -- the
"invisible to a limit check" claim, not the cross-sensor one -- and D6, which has
been open since work item 4 because every figure in this project is
telemanom-minus-commands.

**Stage 1, this section, is the visibility diagnostic only.** Of the 43 sequences
the dataset labels `contextual`, how many actually stay inside their own channel's
training min/max? That count is the **honest denominator** for anything stage 2
could claim, and it is the direct answer to Wu & Keogh's triviality critique on this
benchmark. **Stage 2 -- fitting `gru-quantile` with commands, `rstd` and the range
check -- runs only on approval, and only if a population survives.**

### 26.1 Definitions fixed in advance

- **The telemetry channel is column 0.** SMAP arrays are 25 columns and MSL 55; all
  but the first are one-hot command indicators. The envelope is built on column 0
  alone, because a one-hot indicator has no meaningful range and including it would
  make every event trivially "in range" on 24 or 54 constant columns.
- **Training envelope**: `min` and `max` of column 0 over that channel's **`train`
  array**, which is the split the paper defines and this section honours
  (`docs/DATA.md`). No test sample enters it.
- **In range**: an anomaly sequence where **no** step of column 0 inside `[lo, hi]`
  leaves that envelope, **strictly** -- touching an extreme is not leaving it, the
  rule 24.9 fixed after the strictness defect it caught.
- **The population**: the 105 sequences of `labeled_anomalies.csv`, 43 `contextual`
  and 62 `point`, verified against the arrays at ingest. `P-2`'s duplicate row is
  **counted once** and the defect is recorded (`docs/DATA.md`).
- Reported per spacecraft and pooled, as `k/n`, `n < 20` stamped UNDERPOWERED.

### 26.2 PREDICTED

The letter is **V**; C, D, E, F, G, L, M, N, P, R, S, T and W are taken.

| # | Prediction | Refuted by |
|---|---|---|
| **V1** | of the **43** labelled `contextual` sequences, **15 to 30** stay strictly inside their channel's training min/max; central estimate **22** | a count outside that band. Low would mean the label does not mean what the paper says it means, which is Wu & Keogh's charge; high would mean ESA-ADB is the unusual set, not this one |
| **V2** | of the **62** labelled `point` sequences, **<= 10** stay in range | more than 10. A point anomaly is an extreme value by definition, so this is the diagnostic's own sanity check -- if point anomalies read as in-range, the instrument is wrong, not the labels |
| **V3** | the in-range **fraction** among `contextual` exceeds that among `point` by **>= 30 percentage points** | a gap below 30, which would say the two labels do not separate on this diagnostic at all and neither class can be trusted to mean what it says |
| **V4** | SMAP and MSL agree in **direction** on V3 | opposite signs, which would make the pooled figure an average of two different datasets |
| **V5** | **the decision gate.** The in-range `contextual` population is **>= 20**, the sample size `docs/HARNESS.md` section 1 requires before a recall is a comparison rather than a coverage check | below 20. **Stage 2 would then be underpowered by construction and does not run** -- and that, not a detector result, is the finding: the benchmark with the largest labelled contextual class in public spacecraft data would not carry enough of it to test the claim |

**Deliberately not predicted.** Which channels carry the in-range events. Anything
about detector performance -- no detector runs in stage 1. Whether the command
columns help, which is stage 2's question and D6's.

### 26.3 The named risk

**That the definition of "in range" was chosen on ESA-ADB and is now being applied
to a dataset where it may flatter us.** The mitigation is that it is D43's
definition unchanged, strict, on the training split the paper itself defines, and
that **V2 is a falsifiable sanity check in the opposite direction**: if the
diagnostic cannot tell point anomalies from contextual ones, it is broken and V3
says so.

And the standing risk, stated for the fourth time: this is another dataset visited
after a bad result. What protects against it is that **26.2's first paragraph
forbids quoting any of this as cross-channel evidence**, and that V5 can close
stage 2 before it starts.

### 26.4 Cost

Stage 1 reads the 81 `train` and 81 `test` arrays back through
`_manifest/smap_msl.json`: **162 Class B and 1 Class A**, against a month at 172
Class A and 104 Class B of 50,000 each, per-run tripwire 1,000. The ingest itself
cost **164 Class A** and is already recorded in the ledger. Stage 2 is not costed
here and does not run without approval.

### 26.5 Stop and report

1. **V5 refuted** -- fewer than 20 in-range contextual sequences. Stage 2 does not
   run and the reason is reported.
2. **V2 refuted** -- the diagnostic cannot separate point from contextual. The
   instrument is wrong; report that and nothing else.
3. Any ESA-ADB figure, task, weight or manifest moves. Nothing here touches them.
4. More than 200 Class B in the run, or the per-run tripwire at 1,000.

### 26.6 OBSERVED

Reserved. Nothing has run.

### 26.6 OBSERVED

**2026-09-03. 1 Class A and 165 Class B**, manifest-addressed reads through
`_manifest/smap_msl.json`, no LIST. Artifact `runs/smap-msl/_forensics/2026-09-03T192723Z-visibility.json`.
The month stands at 173 Class A and 269 Class B of 50,000 each.

**105 label rows become 104 sequences**, because `P-2`'s duplicate row is counted
once as 26.1 fixed in advance. 43 `contextual`, 61 `point`.

```
  in range -- no step of column 0 outside the channel's training min/max, strict

    contextual   39/43   (90.7%)
    point        11/61   (18.0%)
    gap                  +72.7 percentage points

    SMAP   contextual 26/26 (100.0%)   point  8/42   gap +81.0 pp
    MSL    contextual 13/17  (76.5%)   point  3/19   gap +60.7 pp
```

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **V1** | 15 to 30 of the 43 contextual sequences are in range, central 22 | **39** | **REFUTED, and in the direction that helps.** The labels mean *more* than predicted, not less. On SMAP every one of the 26 is in range |
| **V2** | <= 10 of the point sequences are in range | **11 of 61** | **Refuted by one.** The denominator is 61 rather than 62 because `P-2`'s duplicate is counted once. The sanity check the prediction was for is intact: 82% of point anomalies breach |
| **V3** | the in-range fraction among contextual exceeds point by >= 30 points | **+72.7** | **Held, overwhelmingly** |
| **V4** | SMAP and MSL agree in direction | **+81.0** and **+60.7** | **Held** |
| **V5** | **the gate**: in-range contextual >= 20 | **39** | **Held.** Stage 2 is not underpowered by construction and the population survives |

#### What this establishes

**The labelled contextual class on SMAP/MSL is real by our own diagnostic**, and it
is the population ESA-ADB does not have. Against D43's 6/32 (18.8%) on the gate
set's headline cell, SMAP/MSL's labelled contextual class is **39/43 (90.7%)**
in range -- a population **six and a half times larger in absolute terms** and five
times denser.

**And Wu & Keogh's triviality critique is answered precisely rather than in
general.** Their charge is that trivial one-liners reach state of the art on this
benchmark. On the **point** class this diagnostic agrees with them: 82% breach
their channel's historical range and a range check would see them without a model.
On the **contextual** class it does not: 91% stay inside, so those events are
**not** reachable by a range check at any width that respects the training range.
The benchmark is trivial in the part Wu & Keogh measured and not trivial in the
part this project needs. Both halves are stated because only reporting the second
would be the selective reading their paper is about.

#### What it does not establish

**Nothing about detection, and nothing about the cross-channel claim.** No detector
ran. 26's first paragraph stands: SMAP and MSL are 81 unsynchronised univariate
streams (Objective.md 9.2) and no figure here may be quoted as cross-channel
evidence. What stage 1 has produced is a **denominator**, not a result.

**And V1's refutation is recorded as a prediction wrong in the comfortable
direction**, which is the kind that needs saying loudest. The band 15 to 30 was
set expecting the labels to hold up badly; they held up well. That is a mis-set
prior about the data, and it is left exactly as written.

#### Stage 2, unblocked but not run

V5's gate is cleared at n = 39, past `docs/HARNESS.md` section 1's threshold of 20.
Stage 2 -- `gru-quantile` **with the command columns as exogenous inputs**, the D6
path that has been open since work item 4, against `rstd` and a calibrated range
check, on the paper's own train/test split, scored at a matched nominal rate on the
39 and on all 104 -- **is warranted and does not run without approval.** Its
falsification will be stated against the **measured** matched-rate multiplier and
not "by construction": on `m1-ss5` that multiplier was **0.672**, tighter than the
training range, and a range check tightened below the training range does catch
in-range events.

### 26.7 Stage 2, pre-registered. What runs, and what it is not

**Written and committed before a single detector is fitted.** V5's gate cleared at
n = 39, so stage 2 is warranted. **26's first paragraph still governs**: SMAP and
MSL are 81 unsynchronised univariate streams (Objective.md 9.2), nothing here is
cross-channel evidence, and no figure below may be quoted as such. What is being
tested is the **in-limits** claim -- can a forecaster see anomalies that never leave
their channel's historical range, which a range check cannot -- and **D6**, open
since work item 4 because every figure in this project is telemanom-minus-commands.

**Three detectors, per channel, on the paper's own split.** `train.npy` fits and
calibrates; `test.npy` is scored. That is the split Hundman et al. defined and it is
what makes this comparable to their numbers; it is **not** comparable to this
project's forward-chained ESA-ADB folds and no figure is placed beside one.

```
  gru-quantile+cmd   telemetry column 0, command columns 1..n as EXOGENOUS inputs
  gru-quantile       the same, commands withheld        <- the pair IS D6's ablation
  rstd               trailing standard deviation, window 120, on column 0
  range check        per-channel training min/max, widened by a swept multiplier
```

**The commanded pair differ in that alone** -- identical architecture,
hyperparameters, seed and detection stack -- which is what makes it an ablation
rather than a comparison, in `TelemanomCommanded`'s own words
(`detectors.py:838-851`). No `gru-commanded` is added to the registry: the flag is
set on the instance, as work item 9.7's reduction arms were computed without adding
detectors.

### 26.8 Definitions fixed in advance

- **Calibration recipe, unchanged**: threshold is the **99.9th percentile** of the
  reduced score over the **fitting** window, `Detector.threshold_from`. The paper's
  `train` split carries no labelled anomalies, so the anomaly mask is all-true and
  the recipe applies unaltered.
- **Nominal step**: a test step not inside any labelled anomaly span, of either
  class. **Matched operating point**: equal nominal-step false-alarm rate, pooled
  across all 81 channels. Unchanged from 23.7 and 24.2.
- **The range check's multiplier `w`** is swept until its pooled nominal-step rate
  equals the commanded forecaster's. **`w` is measured, not assumed**, and every
  claim about the range check is stated against the measured value.
- **Populations, fixed by stage 1 and not re-derived**: **39** in-range contextual,
  **4** out-of-range contextual, **11** in-range point, **104** sequences in total
  after `P-2`'s duplicate is counted once. All four are reported; none is the
  headline on its own.
- **Every catch carries the Wu & Keogh diagnostic beside it** -- whether that
  sequence leaves its channel's training min/max, and by how many steps -- so a
  catch on a trivially-visible event can never be read as a catch on a hard one.
- **`D-12`** has 312 training rows, fewer than 100 usable windows at window 250.
  It is fitted anyway and **flagged**; it carries one out-of-range point sequence
  and none of the 39, so it cannot move the headline population. Recorded here so
  the exclusion cannot be made later.
- `k/n` throughout; `n < 20` stamped UNDERPOWERED. The 4 and the 11 are below that
  line by construction and are reported as coverage checks, never as comparisons.

### 26.9 PREDICTED

| # | Prediction | Refuted by |
|---|---|---|
| **V6** | at a matched nominal rate, `gru-quantile+cmd` catches **15 to 32** of the **39** in-range contextual sequences; central estimate **24** | a count outside the band. This is the number the whole two-stage study exists to produce: the first measurement of whether this method sees events a range check provably cannot |
| **V7** | the range check's matched multiplier is **`w < 1.0`** -- the envelope must tighten *inside* the training range to be as quiet as the forecaster | **`w >= 1.0`**, in which case the range check catches **0 of 39 by construction** and the comparison is **uninformative, not a win**. It is reported in those words and V8 is withdrawn rather than counted |
| **V8** | conditional on `w < 1.0`: the forecaster's catch count on the 39 exceeds the range check's by **>= 10** sequences | a margin **<= 0**. The forecaster would then have no advantage on the one population built to favour it, and the in-limits claim would be finished on this data as well as on ESA-ADB |
| **V9** | `gru-quantile+cmd`'s event-wise F0.5 over all **104** at its own operating point is **0.40 to 0.75** | outside the band. Hundman et al. report F1 near 0.71 on this benchmark; a figure far below would mean the recipe does not transfer, and far above would need explaining |
| **V10** | **D6, finally tested.** The commanded arm catches **more** of the 39 than the uncommanded arm | equal or fewer, which answers D6 **no** on this data: commands do not help, and the project's telemanom-minus-commands figures lose nothing by the omission |
| **V11** | both detectors catch a **higher fraction** of the 4 out-of-range contextual than of the 39 in-range | a lower fraction, which would invert the expected difficulty ordering and put the diagnostic itself in question. `n = 4`, a coverage check |
| **V12** | at the matched `w`, the range check catches **<= 2** of the 11 in-range point sequences | more than 2, which would say the tightened envelope is finding in-range events generally rather than the contextual ones specifically. `n = 11`, a coverage check |

**Deliberately not predicted.** Which channels carry the catches. `rstd`'s figures,
reported as the floor and not predicted. Anything about lead time -- the paper's
split gives no comparable emission anchor and D44 already settled lead on ESA-ADB.

### 26.10 The named risk

**That a dataset was added after a bad result and will now be reported selectively.**
The protections are that the populations were fixed by stage 1 **before** any
detector existed; that **all four** are reported, including the 4 and the 11 that can
only complicate the story; that **V7 names the condition under which the range-check
comparison is worthless** and requires it be called uninformative rather than a win;
and that 26.7 forbids quoting any of it as cross-channel evidence.

**And the honest limit.** 39 sequences across 26 channels, each a separate univariate
model, is not a demonstration that this method works. It is one measurement, on a
benchmark this project has documented as discredited in the part Wu & Keogh measured
(D46), of a claim ESA-ADB could not test at all.

### 26.11 Cost

**Measured before committing to it.** One GRU fit on a median channel takes **8.8
seconds**; all 81 train series together are 196,321 steps, **6% of one ESA-ADB
fold**. So **162 fits -- both arms -- is about 24 minutes on this machine**, and no
rented GPU is warranted. `rstd` and the range check need no fit.

Reads are 162 arrays plus the manifest and labels: **165 Class B and 1 Class A**,
against a month at 173 Class A and 269 Class B of 50,000 each, per-run tripwire
1,000. Weights are written to `runs/_weights` and the store is expected to grow by
**162 files**, which is the one place this study legitimately moves an invariant.

### 26.12 Stop and report

1. **V7 refuted** -- `w >= 1.0`. Report the range-check comparison as uninformative
   and withdraw V8 rather than scoring it.
2. **V8 refuted** -- the forecaster has no margin on the 39.
3. **V10 refuted** -- commands do not help, and D6 is answered no.
4. Any ESA-ADB figure, task, weight, manifest or decision moves. Nothing here
   touches them; the weight store growing by 162 SMAP/MSL entries is expected and
   is not that.
5. More than 200 Class B, or the per-run tripwire at 1,000.

### 26.13 OBSERVED -- stage 2, and it does not produce a detector comparison

**2026-09-03. Two runs; the first discarded for the defect D47 records, the second
after it.** Artifact `runs/smap-msl/_forensics/2026-09-03T205255Z-stage2.json`.
1 Class A and 165 Class B each. **The weight store grew by 0 on the second run**,
which confirms 26.14's claim that `error_window` lives in `config` and not in the
cache key: the fits were reused rather than repeated.

**The populations shrank before any detector ran.** D17's first-epoch guard refused
4 channels outright -- `E-3`, `G-1`, `D-11` on a stall and `D-12` for holding no
complete sequence -- and a channel enters only if **both** forecaster arms fit, so
the ablation stays paired. 77 of 81 channels scored, and the populations became
**38** in-range contextual, **4** out-of-range contextual, **9** in-range point,
**100** sequences. Every figure below is against those denominators.

#### The finding: no train-calibrated threshold transfers on this data

```
  arm         multiplier   nominal-step rate    in-range contextual
    gru+cmd        1.000            10.4682%              11/38
    gru            1.106             8.1890%              11/38
    rstd          50.000            15.5654%               0/38
    range          1.106            10.2814%               0/38
```

**`rstd` at fifty times its calibrated threshold -- the top of the sweep -- still
alarms on 15.57% of nominal steps.** On ESA-ADB it runs at 0.024%. A trailing
standard deviation cannot be made quiet on this data by any multiplier the grid
contains, and neither can the forecaster: its own 99.9th-percentile threshold admits
**10.47%** of nominal time.

That is not a property of any one detector. It is the **train-to-test regime shift**
on SMAP/MSL: the residual scale on the test split is so much larger than on the train
split that a threshold calibrated on the first does not describe the second, for any
arm. **D48** records it.

**So stage 2 has no usable operating point, and the comparison it was built to make
cannot be made.** Reporting 11/38 as "the forecaster's catch count at a matched rate"
would be reporting a number taken at 10% nominal alarms, which no mission would fly
and which V6 was not asking for.

| # | Prediction | Outcome |
|---|---|---|
| **V6** | forecaster catches 15-32 of the 39 at a matched rate | **No verdict.** The matched rate is 10.47% nominal. The measured 11/38 is not the quantity V6 asked for and is not reported as it |
| **V7** | the range check's matched multiplier is **w < 1.0** | **REFUTED. w = 1.106** |
| **V8** | forecaster beats the range check by >= 10 on the 39 | **Withdrawn**, by 26.9's own rule, because V7 was refuted. The range check's 0/38 is 0 **by construction** at w >= 1 and is uninformative, not a win. Recorded as the pre-registration required, not as a result |
| **V9** | F0.5 over all 104 is 0.40 to 0.75 | **No verdict.** Not computed, and it would describe a 10%-alarm operating point if it had been |
| **V10** | **D6**: the commanded arm catches more of the 39 | **No verdict.** Both arms catch 11/38. They are not identical -- 4/9 against 2/9 on in-range point, 54 against 56 overall, and different nominal rates -- so the command flag demonstrably takes effect, but D6 is not answered at an operating point this broken |
| **V11** | both catch a higher fraction of the 4 out-of-range than of the 38 | **Consistent, not scored.** 2/4 against 11/38 for the forecaster and 2/4 against 0/38 for the range check, at n = 4 and at an unusable anchor |
| **V12** | the range check catches <= 2 of the in-range point sequences | **Held trivially.** 0/9, and 0 by construction at w >= 1 -- the prediction was right for a reason that carries no information |

**What survived stage 2 and is worth keeping**: the D6 wiring is correct, the
populations are fixed and reproducible, D17's guard fired honestly on four channels
rather than letting them through, and D47's correction did what 26.14 predicted it
would. **What did not**: any statement about whether this method sees in-range
anomalies. That needs an operating point, and stage 3 is pre-registered to obtain one
the way flight would.

### 26.14 Amendment to 26.8: `error_window` is proportional, not absolute

**Written 2026-09-03, after a run that was discarded and before the run that
replaced it.** The predictions V6 to V12 are **unchanged**, the populations are
unchanged, and the falsification is still stated against the **measured** matched
multiplier. What changes is one constant that was wrong for this data.

**The defect.** `ForecastDetector.warmup_steps` is `hyper.window +
config.error_window` = 250 + 2100 = **2,350 steps**. That is sized for ESA-ADB,
where one fold's test window is ~3.5 million steps and 2,350 is negligible. The
median SMAP/MSL training series is **2,690 steps**. Measured on the discarded run:

```
  channels whose ENTIRE test array sits inside the warm-up   16 / 81
  channels with <500 non-warm-up TRAINING steps                48 / 81
  pooled nominal-step alarm rate                            12.5852%
  the same rate on a healthy channel (A-4), verified directly   0.14%
```

So 16 channels' events were uncatchable by construction while still counted in the
denominator, and 48 channels' 99.9th-percentile thresholds were computed mostly
from warm-up-region scores that mean nothing. **The discarded run's figures --
9/38, 7/38, `w = 1.000` -- are artefacts of that and are not reported as results.**

**The correction.** `error_window` becomes `SMOOTHING_PERC * len(series)`, which is
telemanom's own proportional definition (`telemanom.py:84`, 0.05), rather than the
absolute `ERROR_WINDOW_BATCH * ERROR_WINDOW_COUNT` = 2100 fixed for ESA-ADB. On a
2,690-step series that is 134, and the warm-up falls from 2,350 to **384**.

**It is applied to SMAP/MSL only, and ESA-ADB keeps 2,100.** The same formula on a
3.5-million-step fold would give 175,000, which is not a correction but a different
detector. This is a **per-dataset choice, not a universal fix**, and saying otherwise
would overclaim it. No ESA-ADB figure moves and no ESA-ADB code path is touched.

**Weights are not invalidated.** The cache key is over `hyper`; `error_window` lives
in `config`, so the 145 fits already on disk are reused and the re-run costs reads
rather than another 24 minutes of fitting.

Recorded as **D47**, in D17's family: an absolute constant that should have been
relative, silently disabling the thing it configures.

## 26.15 Pre-registration: stage 3, the commissioning-window operating point

**Written and committed before a single figure is computed.** (Numbered 26.15 rather
than 26.10, which is 26's named risk and is not overwritten.)

D48 established that no threshold calibrated on SMAP/MSL's **train** split describes
its **test** split -- `rstd` at fifty times its threshold still alarms on 15.57% of
nominal steps. So stage 2 had no operating point and made no comparison. Stage 3
obtains one **the way flight would**.

**The commissioning window.** Each channel's threshold is recalibrated on the
**leading nominal prefix of its own test stream**: steps `[0, first labelled anomaly)`,
which are labelled nominal and contain no anomaly of either class. This is exactly
the in-orbit recalibration path work item 10 implements -- a threshold uplinked after
a commissioning period on the vehicle's own data (D29's consequence, Objective.md
14.10) -- and D48 is the measured case for why it exists.

**This is not tuning on the test set, and the reasons are structural rather than
promised.** The window contains **no labelled anomaly**, so no label of the kind
being scored enters it. It is applied **identically to all four arms**, so no arm is
advantaged. Its length rule is **fixed below, before the numbers exist**. And nothing
is swept to maximise a result: the matched-rate comparison is the same instrument
23.3 and 24.2 already use. What it *is* is a change of calibration window, from a
split that D48 showed does not transfer to one that flight would actually use, and
26.16 will report it as such.

### 26.15.1 Definitions fixed in advance

- **Commissioning window**: `[0, first_anomaly_start)` on the test array, minus the
  channel's warm-up (250 + `SMOOTHING_PERC * len(train)`, D47). What remains are the
  **scorable commissioning steps**.
- **The length rule, primary**: a channel is scored only if it has **>= 500 scorable
  commissioning steps**. Measured now, before any detector runs, so the exclusions are
  a stated fact and not a later choice:

```
  min scorable    channels    of the 39    of the 4 out-of-range    of the 11 in-range point
        250            75           37                      3                            8
        500            72           33                      3                            8       <- primary
       1000            57           26                      0                            6
```

  **The primary rule keeps 33 of the 39**, comfortably past `docs/HARNESS.md` section
  1's `n >= 20`. 250 and 1000 are **pre-declared sensitivity** and neither can become
  the primary afterwards.
- **(!) A limitation of the estimator, stated now.** A 99.9th percentile needs about
  1,000 samples to be an interior quantile. At 500 it **is the maximum of the
  commissioning window**. That is still a defensible flight rule -- "the largest score
  seen during commissioning" -- but it is not the same statistic D25 computes over
  millions of ESA-ADB steps, and no figure here is placed beside one.
- **Everything else is unchanged**: the same four arms, the same per-channel scoring,
  the paper's own split for fitting, matched on pooled nominal-step false-alarm rate,
  `k/n` throughout, the Wu & Keogh diagnostic beside every catch, and the 4 and the 11
  reported separately.

### 26.15.2 PREDICTED

| # | Prediction | Refuted by |
|---|---|---|
| **V13** | with commissioning-window calibration, `gru-quantile+cmd`'s nominal-step alarm rate falls **below 1%**, from stage 2's 10.47% | **>= 1%**. The recalibration would then not have fixed what D48 diagnosed, and stage 3 has no operating point either -- in which case SMAP/MSL cannot be scored by this decision layer at all, and that is the finding |
| **V14** | `rstd` reaches the matched rate **within the sweep**, rather than saturating at the grid maximum as it did in stage 2 | saturation again, which would say the regime shift is not what the commissioning window fixes |
| **V15** | at a matched rate, `gru-quantile+cmd` catches **12 to 28** of the surviving in-range contextual sequences; central estimate **19** | outside the band. **This is the number the whole three-stage study exists to produce** |
| **V16** | the range check's measured matched multiplier is **`w < 1.0`** | **`w >= 1.0`**, in which case it catches the in-range population **0 by construction**, the comparison is **uninformative rather than a win**, and V17 is **withdrawn** rather than scored -- the same rule 26.9 applied to V8, and it fired there |
| **V17** | conditional on `w < 1.0`: the forecaster's catch count on the in-range population exceeds the range check's by **>= 8** | a margin **<= 0**. The forecaster would have no advantage on the one population built to favour it, on the one dataset that carries enough of it, and the in-limits claim would be finished on both datasets |
| **V18** | **D6, at last answered**: the commanded arm catches **more** of the in-range contextual than the uncommanded arm | equal or fewer, which answers D6 **no**: commands do not help, and every telemanom-minus-commands figure in this project loses nothing by the omission |

**Deliberately not predicted.** Which channels carry the catches. `rstd`'s catch
count, reported as the floor. Lead time, which the paper's split gives no comparable
anchor for and which D44 settled on ESA-ADB.

### 26.15.3 The named risk

**That the calibration window was changed after a result nobody wanted.** It was --
D48 is that result, and 26.15 exists because of it. What makes it defensible rather
than convenient is that the window contains **no labelled anomaly**, that it is
applied **identically to every arm** including the two that could beat the
forecaster, that its length rule and exclusions are **fixed above before any figure
exists**, and that **V13 can close stage 3 the same way D48 closed stage 2**. If the
recalibration does not produce a usable operating point, that is reported and the
study ends unmeasured for the second time.

**And the standing one**: this is the third stage on a dataset added after a bad
result. 26's first paragraph still governs -- **none of it is cross-channel
evidence**, Objective.md 9.2 stands, and SMAP/MSL remains 81 unsynchronised
univariate streams.

### 26.15.4 Cost and stop-and-report

**One read from cached weights: 165 Class B and 1 Class A.** The month stands at 176
Class A and 606 Class B of 50,000 each, per-run tripwire 1,000. No fit is repeated;
D47 already demonstrated the weight store grows by 0 on a re-run.

Stop and report if: **V13 is refuted** and there is again no operating point; **V16 is
refuted**, in which case V17 is withdrawn rather than scored; any ESA-ADB figure,
task, weight or manifest moves; or more than 200 Class B.

### 26.16 OBSERVED -- stage 3, and it closes the study unmeasured

**2026-09-03. 1 Class A and 165 Class B**, cached weights, weight store **+0**.
Artifact `runs/smap-msl/_forensics/2026-09-03T210913Z-stage3.json`.

**V13 is refuted, in the wrong direction, and 26.15.4 makes that a stop.**

```
  gru-quantile+cmd nominal-step alarm rate
    stage 2, train-split calibration            10.4682%
    stage 3, commissioning-window calibration   32.5104%     <- V13 predicted < 1%
```

**Recalibrating the way flight would made it three times worse**, and the cause is
the limitation 26.15.1 fixed in advance: *"A 99.9th percentile needs about 1,000
samples to be an interior quantile. At 500 it **is** the maximum of the commissioning
window."* A short window's maximum is far below a long stream's, so the threshold
falls and the alarm rate rises. The estimator note was the binding constraint and it
was written before the run.

```
  arm         mult   nominal    in-range ctx   out-of-range   in-range point      all
    gru+cmd  1.000  32.5104%          20/33            2/3            7/8       66/92
    gru      1.000  32.0076%          21/33            2/3            8/8       76/92
    rstd     1.000  27.9373%          13/33            1/3            8/8       67/92
    range    1.000  12.5883%           5/33            2/3            7/8       60/92
```

**These are not results and are not reported as any.** At a third of nominal time
alarming, 20/33 means the detector is on for a third of the mission and happened to
be on during 20 events. Nine channels were excluded -- four by D17's guard, five by
the commissioning-length rule (`C-1`, `C-2`, `T-12`, `T-13`, `D-16`) -- leaving
33 of the 39, as 26.15.1 said it would.

| # | Prediction | Outcome |
|---|---|---|
| **V13** | nominal rate falls below 1% | **REFUTED. 32.51%**, worse than stage 2's 10.47% |
| **V14** | `rstd` reaches the matched rate without saturating | **Held vacuously.** Every arm sits at multiplier 1.000 because the target rate is 32.5% and trivially met. It carries no information |
| **V15** | forecaster catches 12-28 of the surviving in-range contextual | **No verdict.** 20/33 is inside the band and means nothing at this alarm rate |
| **V16** | the range check's matched multiplier is `w < 1.0` | **No verdict, and a reporting correction.** `w = 1.000`, but under commissioning recalibration the range check's *base* threshold is no longer the training min/max, so 26.9's "0 by construction" argument **does not apply** -- which is why it catches 5/33 here. The script printed the stage 2 message on the first pass; it is corrected and the message is not quoted |
| **V17** | conditional on `w < 1.0`: forecaster beats the range check by >= 8 | **Withdrawn**, V16 having no verdict |
| **V18** | **D6**: the commanded arm catches more | **No verdict, and it leans the other way.** 20/33 commanded against **21/33** uncommanded, and 66/92 against 76/92 overall. At a 32% alarm rate that is noise, not an answer, but nothing here supports commands helping |

#### What the three stages establish, together

**On SMAP/MSL this decision layer cannot be calibrated at all, by either route.**
The train split does not transfer (D48). The commissioning window is too short for
the percentile to be a percentile, so it degenerates to a short-window maximum and
alarms three times more. Both failures have **one root**: the 99.9th-percentile rule
needs a long, representative calibration window, and this dataset provides neither.

That is a real and bounded finding, and it is the third time this project has
measured the same thing from a different angle -- D18 (telemanom's dynamic threshold
degenerates under a good forecaster), D29 (`m1-g3`: a floor calibrated on early data
sat under 86.7% of a later window), and now D48 with stage 3's failed remedy.

**What was never measured, and is now recorded as unmeasured**: whether a forecaster
sees in-range anomalies a limit check cannot. Stage 1 built the population -- 39 of
43, six and a half times ESA-ADB's (D46) -- and stages 2 and 3 could not obtain an
operating point to score it at. **The question this study existed to answer is still
open, and the reason is calibration rather than detection.**

**D6 is still open.** The ablation is correctly wired and ran twice, and both times at
an alarm rate that makes the comparison meaningless. Nothing here answers it, and
26.16 does not pretend to.

## 26.17 Pre-registration: stage 4, Route 1 -- the published dynamic threshold

**Written and committed before a single figure is computed.**

D48 established that no *static* threshold calibrated on SMAP/MSL's train split
transfers to its test split, and 26.16 that a commissioning window short enough to
exist is too short for a 99.9th percentile to be a percentile. Both failures are
about a **fixed** cut derived from one window and applied to another.

**telemanom's own threshold does not do that.** Its nonparametric dynamic threshold
recomputes the cut from a trailing window of the stream being scored, so there is no
train-to-test transfer to fail. `ForecastDetector.threshold_from` returns **1.0** in
NDT mode (`detectors.py:401-412`) because the operating point lives inside the score.
That is Route 1, and it is the published method rather than a variant of ours.

**It is also the arm this project reproduced and rejected on ESA-ADB** (D18): the
dynamic rule degenerated under a good forecaster, 92.6% of windows selecting the
range minimum. On stationary folds the static quantile won. Stage 4 asks the same
question on the regime where the static quantile is the one that fails, so **a result
either way is about the regime and not about the rule**, exactly as D48 says.

**26's first paragraph still governs.** None of this is cross-channel evidence;
Objective.md 9.2 stands.

### 26.17.1 Definitions fixed in advance

- **The arm**: `gru-telemanom` -- `GRUForecastDetector`, NDT mode -- **with the
  command columns as exogenous inputs**, per channel, on the paper's own split. The
  uncommanded arm is scored beside it at no extra cost, because mode changes only the
  detection stack and both reuse the **weights already fitted in stage 2**. No fit is
  repeated.
- **`error_window` scales to the stream the NDT thresholds**, which is the test
  series: `SMOOTHING_PERC * len(test)` (D47's rule, applied to the scored stream
  rather than the fitting one, because that is the window the dynamic threshold
  actually reads). Stated because it differs from stages 2 and 3, which scaled to
  `train`.
- **(!) The gate, and it is checked before any comparison.** The pooled nominal-step
  alarm rate of `gru-telemanom+cmd` must be **under 1%**. Above it, stage 4 Route 1
  has no operating point either, **no comparison is reported**, and Route B is the
  fallback (26.17.4). This is the same gate stages 2 and 3 failed, applied first
  rather than discovered last.
- **The sweep is widened to 5,000.** In stage 2 `rstd` saturated at the grid maximum
  of 50 and still alarmed on 15.57%; a grid that cannot reach the target cannot
  produce a matched comparison. Fixed here, before the numbers.
- **Everything else is unchanged**: matched on pooled nominal-step false-alarm rate,
  the same populations from stage 1, `k/n` throughout, the Wu & Keogh diagnostic
  beside every catch, the 4 and the 11 reported separately, and channels excluded only
  by D17's guard.

### 26.17.2 PREDICTED

| # | Prediction | Refuted by |
|---|---|---|
| **V19** | **the gate.** `gru-telemanom+cmd`'s pooled nominal-step alarm rate is **under 1%** | **>= 1%.** Route 1 then has no operating point, no comparison is reported, and the study moves to Route B. Stages 2 and 3 both failed here and this is the third attempt |
| **V20** | at a matched rate, `gru-telemanom+cmd` catches **12 to 28** of the surviving in-range contextual sequences; central **19** | outside the band. **This is the number the whole four-stage study exists to produce**, and it is the same band V15 carried, unchanged |
| **V21** | `rstd` reaches the matched rate **within the widened sweep** rather than saturating | saturation again at 5,000, which would say no multiplier makes the floor quiet on this data and the matched comparison is unobtainable for it |
| **V22** | the range check's measured matched multiplier is **`w < 1.0`** relative to the training min/max | **`w >= 1.0`**, in which case it catches the in-range population **0 by construction**, the comparison is **uninformative rather than a win**, and V23 is **withdrawn** rather than scored -- the rule that already fired on V8 and V16 |
| **V23** | conditional on `w < 1.0`: `gru-telemanom+cmd` beats the range check by **>= 8** on the in-range population | a margin **<= 0**, which would finish the in-limits claim on both datasets |
| **V24** | **D6**: the commanded NDT arm catches **more** in-range contextual than the uncommanded one | equal or fewer, answering D6 **no** on the one dataset that carries commands |

**Deliberately not predicted.** Whether the NDT degenerates the way D18 found on
ESA-ADB -- that is worth measuring and naming a direction for it would prejudge the
regime question stage 4 exists to ask. `rstd`'s catch count, reported as the floor.

### 26.17.3 The named risk

**Fourth attempt, same dataset, after three failures.** What limits it: **V19 is a
gate checked first**, so a failed run produces no comparison rather than a weak one;
the band in V20 is **unchanged from V15**, so it cannot be widened to fit; V22 carries
forward the rule that has already fired twice; and 26.17's opening states that a
result either way is about the regime, which stops a Route 1 success being read as
vindication of the dynamic rule that ESA-ADB rejected.

### 26.17.4 Route B, scoped and not run

If V19's gate fails, the fallback is a **robust small-sample threshold**: the
commissioning window's **median + k * MAD** rather than its 99.9th percentile.
The median and MAD are stable at `n = 500` where a 99.9th percentile is just the
maximum, which is precisely why 26.16 failed. It would need `k` fixed in advance,
the same commissioning window and exclusions, and its own pre-registration. **It is
not run here and no `k` is chosen now**, because choosing one after seeing Route 1's
numbers is the thing this section is arranged to prevent.

### 26.17.5 Cost and stop-and-report

**One read, no fits: 165 Class B and 1 Class A.** Weights are reused from stage 2 --
mode changes the detection stack, not the cache key -- so the weight store is expected
to grow by **0**, as it did on both re-runs. Month stands at 177 Class A and 771
Class B of 50,000 each.

Stop and report if: **V19's gate fails** (report the rate and nothing else, then Route
B); **V22 is refuted** (withdraw V23 rather than score it); the weight store moves; or
more than 200 Class B.

### 26.18 OBSERVED

Reserved. Nothing has run.
