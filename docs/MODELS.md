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

### The deviation ledger

Every departure from telemanom's published configuration, with its reason. A
claim of faithfulness that cannot be audited is worth nothing.

| # | Published telemanom | Here | Why |
|---|---|---|---|
| 1 | One univariate LSTM per channel | **One multivariate LSTM over the channel set**, C in / C out | Objective.md 4.3: "given the recent history of all watched channels". **The largest deviation** -- the section 12 gate names it |
| 2 | Telecommand signals as one-hot model inputs | **Absent** | `Bundle` hands a detector the channel matrix only. Adding exogenous inputs means changing the referee, which is closed. A future item |
| 3 | 35 epochs over a channel's whole training set (~2-8k steps) | Up to 35 epochs, sequence budget **scaled to the fold**, one sequence per 180 usable steps | Folds hold 3.6M to 10.8M usable steps. See section 3 -- a fixed budget would destroy the data-sufficiency curve |
| 4 | Random 20% validation split | **Chronological last 20%** of usable steps | `docs/HARNESS.md` section 3 is unconditional: never fit on data that follows what is scored, and early stopping is a fitting decision |
| 5 | Keras `EarlyStopping`, which stops but does not restore | **Restores the best-validating weights** | Strictly the better estimator. Recorded rather than assumed |
| 6 | Error window centred on the errors it judges | **Trailing**: threshold chosen from the 2,100 errors *before* a 70-step segment, applied to that segment | See section 1.1. telemanom's windows extend forward, so a timestep is scored partly from errors that had not happened yet. `rstd`, the number we have to beat, is strictly trailing |
| 7 | Inference over l_s=250 windows from a zero state | **Chunked-parallel streaming**, each chunk warmed by a 250-step zero-initialised prefix | Identical arithmetic. A warmed chunk sees exactly the history telemanom's own windowed inference sees |

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
`patience = 10`, `min_delta = 3e-4`, `batch_size = 70`, `smoothing_perc = 0.05`
(EWMA window 105), `window_size = 30` giving `h = 2100`, `error_buffer = 100`,
and pruning at `p = 0.13`.

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

*Not yet run. This section is filled in after the run and beside the prediction
above, never in place of it.*

---

## 5. What the LSTM found that the baselines could not

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

## 6. Dependency note: torch vendors fsspec

`torch==2.13.0` pulls in `fsspec` transitively, along with filelock, sympy,
networkx, jinja2, typing_extensions and setuptools. fsspec expands glob patterns
into paginated LIST operations silently, which is precisely what docs/DATA.md
section 4 forbids, and it is now *importable* in this environment where before it
would have raised.

Nothing changed about the control: `scripts/check_no_list.py` bans `fsspec` and
`s3fs` at source level across `src/` and `scripts/`, and `tests/test_no_list.py`
fails the suite on any violation. Recorded here so that seeing fsspec in
`pip list` does not read as a regression.
