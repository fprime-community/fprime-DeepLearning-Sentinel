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
| 6 | Error windows of h=2100 stepping by 70 (30x overlap) | **Non-overlapping** windows of h=2100, z-sweep vectorised across all of them | Stride 70 over a 3.68M-step fold is ~52,600 windows per channel per fold. Thresholds adapt every 2,100 steps rather than every 70 -- coarser, and if anything more conservative |
| 7 | Inference over l_s=250 windows from a zero state | **Chunked-parallel streaming**, each chunk warmed by a 250-step zero-initialised prefix | Identical arithmetic. A warmed chunk sees exactly the history telemanom's own windowed inference sees |

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

## 4. Dependency note: torch vendors fsspec

`torch==2.13.0` pulls in `fsspec` transitively, along with filelock, sympy,
networkx, jinja2, typing_extensions and setuptools. fsspec expands glob patterns
into paginated LIST operations silently, which is precisely what docs/DATA.md
section 4 forbids, and it is now *importable* in this environment where before it
would have raised.

Nothing changed about the control: `scripts/check_no_list.py` bans `fsspec` and
`s3fs` at source level across `src/` and `scripts/`, and `tests/test_no_list.py`
fails the suite on any violation. Recorded here so that seeing fsspec in
`pip list` does not read as a regression.
