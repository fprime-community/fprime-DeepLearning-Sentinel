# Design

> **Paths outside this branch resolve on `dev`** at commit **`28f7d27`** (`docs/DECISIONS.md`
> D69, on `dev`). The guards that keep these figures true run on `dev`, not here.

What the component does, the rule it flies today, and the five constraints that are
permanent. **Current state only.** The reasoning, the alternatives and the measurements that
settled each choice are on `dev` in `docs/DECISIONS.md` and `docs/MODELS.md`.

## 1. The chain, end to end

```
  telemetry in            one rate-group tick, nChannels wide, each value valid or not
      |
  normalise               IDENTITY. The constants exist in the file and are 0.0 and 1.0.
                          Per-channel rescaling is refused: it erases the amplitude
                          ratios between related channels that the method watches
      |
  forecast                one GRU over the channel set -- two layers of 80, a 250-step
                          lookback, a ten-step-ahead head. The forecast for a timestep is
                          the mean of the up-to-ten predictions made before it
      |
  residual                per channel, |forecast - reality|
      |
  smooth                  EWMA, span 105, bias-corrected, F64
      |
  standardise             each of two streams against its own TRAILING 2,100 samples:
                            z_residual    the smoothed residual
                            z_derivative  |x[t] - x[t-1]|, with x[-1] := x[0]
      |
  decide                  emitted = max(z_residual, z_derivative) >= threshold
                          one calibrated cut, read from the model file
      |
  warn                    an F' event naming the channel. Nothing else.
```

**Carried, reported, and driving nothing:** telemanom's dynamic threshold, solved once per
70-tick segment over the trailing 2,170 window, with forward-only dilation. It is computed
and published on telemetry so a mission can compare, and it does not gate the warning.

**Why forward-only.** Backward dilation marks timesteps that have already been emitted, and
a warn-only component has no un-emit. The choice rests on that, not on a measurement --
`docs/EVIDENCE.md` says what the measurement did and did not settle.

## 2. The rule is versioned, and the version is checked

The decision rule is a property of the **parameter block**, not of the code:

```
  param_version   what `threshold` cuts
  1               the EWMA of the absolute residual, max across channels
  2               the fused max(z_residual, z_derivative), max across channels
```

A version-1 file gets the version-1 rule, unchanged, so every vector committed before
2026-09-11 still passes. A version-2 file gets the fused rule.

**(!) A reader refuses a `param_version` it does not know**, with `BAD_PARAM_VERSION`. This
is normative and it is not decoration. The two cuts are **different statistics on different
scales** -- a version-1 cut is a percentile of a smoothed error in data units, a version-2
cut is a z-score -- so a reader that accepted any value would apply one generation's cut to
the other's statistic in silence. Measured on the committed tier `p1`: **the same bytes with
the same cut cross 339 times as version 2 and 0 times as version 1**, over 3,200 steps.

The byte layout is identical between the two, so `format_version` stays **1**.
`docs/MODEL_FILE.md` is normative for all of it; where it and any implementation disagree,
**the document is right and the implementation is a defect**.

## 3. The five permanent constraints

Each has a stated reason. **None is a v1 limitation.**

| # | Rule | Reason |
|---|---|---|
| **1** | **Model frozen in flight.** Retraining is explicit and human-approved | A slowly degrading spacecraft must never be able to teach the detector that degradation is normal -- that is precisely the condition it exists to catch |
| **2** | **Silent until validated.** No output until sufficient history backs a warning | A detector that cries wolf gets ignored, and an ignored detector is worse than none |
| **3** | **Warn-only. Commands nothing** | Detection and response are separated. Response belongs to existing fault protection and to humans. Nobody has to trust a neural network with control authority |
| **4** | **Every warning explainable.** Named channels, named relationship breaks | Operators must evaluate claims, not trust scores |
| **5** | **Deterministic onboard code.** Fixed memory, fixed compute per cycle, same inputs -> same outputs | This is what flight review boards can approve |

**No online learning. Ever.**

**Rule 1 governs the weights, not the threshold.** A threshold is a parameter with a
provenance and is recalibrated in orbit under human approval; the parameter block is
separately CRC'd and separately replaceable so that this needs no format change.

**Rule 3 is machine-checkable, and here is its exact form.** `Monitor.fpp` declares **no
`async`, `sync` or `guarded` command of its own**. It *does* declare `command recv`,
`command reg` and `command resp` ports, because F' autocodes `PARAM_SET` and `PARAM_SAVE`
for any component with parameters, and the file says so in situ. The distinction survives a
reviewer opening the file, which is why it is written this way rather than as "zero
commands".

## 4. Failing safe, and exactly what is proven

The loader validates a `model.bin` once and the struct then lives in RAM for the mission, so
`Detector::step` re-checks the shape it is about to trust -- a radiation bit-flip is the
reason Level 1 exists and the reason that re-check exists.

```
  Status.hpp            OK plus 12 refusal codes
  flight/test/          18 load cases, exercising all 12 plus the accept path
  fprime/.../test/ut/   11 of the 12 proven to degrade to Level 1 with the code named
                        in the event, without failing the topology
```

**(!) The twelfth is not in that loop.** `BAD_PARAM_VERSION` was added on 2026-09-11 with
the version-2 rule, and the F' component's unit test still iterates the eleven that existed
before it. The C++ refusal suite covers it; the topology-level degradation test does not
yet. **Stated rather than rounded up to twelve.**

## 5. The tier ladder

The component ships as three tiers, all sharing one loader and one file format. The file
carries the tier and a `baseline_only` switch.

```
  Level 1   statistical cross-channel baseline, zero mission data
            MANDATORY, and not primarily about data availability: it is the loader's
            safe failure mode. A corrupt file, a version mismatch, a failed CRC or a
            bit-flip degrades here with an event and an active-tier telemetry channel,
            and never fails the topology. Its constants are PrmDb-style parameters
            rather than model-file fields, so they are readable when the file is not
  Level 2   small pretrained model, fine-tuned on limited data
            DOES NOT EXIST, and cannot be built before the architecture gate: you
            cannot pretrain without knowing which architecture to pretrain
  Level 3   full mission-specific training
            What a released toolkit would produce. The toolkit is unreleased
```

## 6. Flight rules, and what enforces each

| Rule | Here |
|---|---|
| No `new`/`delete`/`malloc`/`free` after init | Every buffer is a fixed member sized from `constexpr` maxima in `Config.hpp`. **No allocator at all** |
| No exceptions, no RTTI, no STL containers | `-fno-exceptions -fno-rtti`; the loader returns a status and **cannot throw by construction**. `std::min`, `std::max`, `numeric_limits` and `<cstdint>` are permitted and used |
| Fixed-size types only | `F32`/`F64`/`U8`/`U16`/`U32` throughout, via `Types.hpp` |
| Every loop has a provable upper bound | Each is counted against a header field already bounded by `TOO_LARGE` |
| No C-style casts | `memcpy` for every field read out of the byte stream, `static_cast` elsewhere |
| Every fallible return checked | The reader's status is checked at every call site |
| Typed `constexpr` over `#define`, no bare configuration literals | `Config.hpp` names every bound, with its derivation |
| Determinism | `-ffp-contract=off`; bit-identical output digests in-process and across processes, with 2,170 samples of carried state per channel |

Compiled C++14 at `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2`, and linted
against `flight/.clang-tidy`; F's own two configurations run as well when the F' checkout is
present, which a clone of this branch alone does not have (`README.md`).

## 7. Size, and what it is made of

`sizeof(Detector)` is **603,032 B (588.9 KiB)**, asserted **exactly** by
`flight/test/Footprint.cpp` -- a compile-time constant gets an equality, so moving it is a
decision somebody makes on purpose. The pre-registered prediction for this object was
581,488 B and it **failed at +3.70%**; the account of the difference is in
`docs/EVIDENCE.md` rather than hidden.

```
  sizeof(Detector)            603,032 B   = 588.9 KiB
  sizeof(Model)               302,088 B
    of which weight arena     301,440 B   (75,360 float32)
  carried state and scratch   300,944 B
  flown model weights         284,640 B   = 278.0 KiB
  maxima headroom              16,800 B   (5.9%)
```

Compile-time maxima: **16 channels** (12 flown), 2 layers, hidden 80, 10 predictions.
`warmup_steps` is **2,350** = a 250-step window plus the 2,100-sample error window, so the
format's warm-up outlasts everything the fused statistic's own windows need.

**Whether this fits a small single-board computer is not measured.**
`docs/PI_ENVELOPE.md` is reserved and deliberately empty; a figure in it that is not a
measurement would be a defect.
