# What Phase 2 inherits

**Written at the close of Phase 1, 2026-08-29.** The C++ flight component is
built against the artefacts below and nothing else. Every number here was
read from a run artifact under `runs/` or from a test that pins it; the
documents that carry the reasoning are named beside each item.

## 1. The architecture: the GRU (D28)

`gru-quantile` is the flight architecture -- Objective.md section 8's five
criteria adjudicated in `docs/DECISIONS.md` D28. Two layers of 80, 12 inputs,
`l_p = 10`: **71,160 parameters, 278.0 KiB float32**. Selected on criteria 2
to 5 and trainability with criterion 1 a tie at the harness's resolution; the
LSTM leads on event-wise F0.5 and D28 says so. The LSTM (91,640 parameters)
remains the published-reproduction baseline and the TCN rows the answer to
"why not stateless".

## 2. The blueprints: `src/sentinel_models/reference.py`

Plain NumPy, no framework, held to torch at **1e-5** by
`tests/test_reference_equivalence.py` -- measured 4.1e-08 (LSTM), 1.2e-07
(GRU), 4.8e-07 (TCN) at the flown shapes, not growing with sequence length.
The C++ is a transcription of these functions and is verified against them
(D15: portability is an assertion, not an environment):

| player | per-tick function | state carried | gate / tap order |
|---|---|---|---|
| **GRU (flies)** | `gru_cell` | **one vector**, `h` (80 floats per layer) | `GRU_GATES = (reset, update, new)`, stacked along axis 0 of `w_ih (3H, in)`, `w_hh (3H, H)` |
| LSTM (baseline) | `lstm_cell` | two, `(h, c)` | `GATES = (input, forget, cell, output)`, `(4H, ...)` |
| TCN (retained) | `tcn_block` / `causal_conv1d` | **none**; a ring buffer of the last 252 inputs, zero at boot | kernels `(out, in, k)`, tap `j` reads `(k-1-j)*dilation` steps back |

**The one place a loader written from the LSTM is wrong on the GRU** (D26,
`docs/MODELS.md` section 3): the third gate is
`n = tanh(W_in x + b_in + r * (W_hn h + b_hn))` -- `b_hn` sits **inside** the
reset product and cannot be folded into `b_in`. The file format stores **both
bias vectors unsummed**; the LSTM's may be summed and the GRU's may not.
Pinned two-sided by test: folding diverges in general and agrees exactly when
the reset gate is saturated open.

Chunked scoring and the flight loop are the same arithmetic: a 250-step
warmed chunk from a zero state is what telemanom's own inference sees, and
the state carried across chunks is the state a rate-group tick carries
(`reference.forward`, `test_state_carries_across_calls`).

## 3. The model file: what it must carry (Objective.md 14.2, 14.10)

- Weights as plain float32 arrays with the gate order named in the header, not
  inferred from a slice index; `cell`, channel count and order, `window`
  (250), `n_predictions` (10), `n_exogenous` (0 until D6's command inputs are
  measured). The weight files under `runs/_weights/` are the reference
  layout (`detectors._save_weights`): `l{i}_{w_ih,w_hh,b_ih,b_hh}`, `head_w`,
  `head_b`, and `cell` written **and verified against the arrays on load**.
- **Normalisation constants and detection thresholds outside the weights**,
  as PrmDb-style parameters, replaceable without retraining. Measured
  necessity: a threshold that suited a one-epoch model produced 3,548 alarm
  ranges on a trained one (D17); two calibrations of one rule differed
  nineteenfold (RESULTS.md 6b). Thresholds are fitted quantities belonging
  to the model they were measured against, and to the spacecraft: Mission 2's
  floors are 10-30x Mission 1's (RESULTS.md 6k).
- Dropout has no line in the file. No weight normalisation, no batch
  normalisation, no state shape for the TCN.
- Level 1 of the tiered architecture (Objective.md 14.10) -- the loader's
  mandatory safe failure mode on a corrupt file, CRC or version mismatch --
  is unbuilt and is Phase 2's first obligation.

## 4. The decision layer: `lstm-quantile`'s rule, frozen (D25)

Per channel, the EWMA (span 105, telemanom's `smoothing_perc = 0.05` over a
2,100-sample window) of the absolute residual `|actual - forecast|`, where the
forecast at `t` is the mean of the up-to-ten predictions made at `t-1 ... t-10`
(`windows.aggregate_predictions`); the **maximum across channels**; one
alarm when it exceeds a **single global threshold calibrated as the 99.9th
percentile of that statistic over the mission's own anomaly-masked nominal
fitting window** (`harness.py:157`, `Detector.threshold_from`). No persistence
filter beyond N=1, no `error_buffer` dilation on this path (the crossing is
the emission, D21), no k-of-n. There is no alarm budget and no dial
(`docs/HARNESS.md` section 1): the threshold is the measured noise floor.
Per cycle in flight: C EWMA updates, a max, one compare (D25).

Its record, `m1-g8.9.10` / `m2-ss1`: rare-event FA 1/48 / 4/424, nominal-step
0.001% / 0, honest lead +0.0 (RESULTS.md 6h, 6k). **What it does not do:
warn early.** Median honest lead is 0.0 -- it fires at the labelled boundary.
Objective.md 1.1's retired claim stays retired; the break-to-limit-trip lead
is Phase 3's measurement on a real clock against real limits.

## 5. The union configuration: status per D29

**Not adopted (D29).** `gru-quantile` flies alone. On `m2-ss1` the union's
rare alarms were its members' 4 + 4 with no overlap (8/424), failing the
pre-registered cost band; on `m1-g3` its headline-cell edge over the LSTM was
+0. The union stays measured (RESULTS.md 6k, MODELS.md 17) as the
configuration to revisit only if a future decision layer makes the two cells'
alarms overlap again; nothing in the flight component is built for it.

**And the finding Phase 2 inherits ahead of the union question**: on `m1-g3`
folds 1 and 2 the floor calibrated on the first 7.36M steps sat under 87% of
a later window's nominal residual. The frozen rule's premise -- one global
quantile of the fitting window is the noise floor of everything after --
held on an independent spacecraft and failed on a later period of the same
one. Thresholds must be recalibrable in orbit without retraining (Objective.md
10.2 fix 4, 14.10), and the loader must treat a threshold as a parameter with
a provenance, not a constant.

## 6. Decisions carried open into Phase 2

- **D14** -- the weight-cache key is positional; fields for the GRU and TCN
  were added the way D14 prescribes (emitted only when non-default) and the
  named, versioned key is still outstanding.
- **D21, first reach** -- `error_buffer = 100` is held and its effect on alarm
  width is undecided; irrelevant to the flying path, which does not dilate.
- **D23** -- the decision layer is channel-blind; k-of-n was never re-derived.
  `lstm-whitened` (RESULTS.md 6d) is the measured proof that a relationship
  test added nothing at this threshold on this data.
- **D6** -- command conditioning: every figure in this project is
  telemanom-minus-commands; the ablation on post-fix weights has not run.
- Objective.md 13 item 8, the injected-fault sensitivity study, and the
  Phase 3 lead-time measurement.
