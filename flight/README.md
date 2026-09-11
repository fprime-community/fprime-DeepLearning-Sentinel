# `flight/` - the C++ inference core

The deterministic, static-memory transcription of `src/sentinel_models/reference.py`'s
GRU forward pass and the frozen decision layer (D25), plus the `model.bin` reader.

**Warn-only.** The core produces a score and a flag. It commands nothing, writes
nothing and has no side effects (Objective.md 11 rule 3).

Not here, deliberately: the F' component, its ports and events (work item 9); the
recalibration uplink path (work item 10); any inference library (D30).

## Build and verify

```bash
make -C flight test      # footprint, refusals, determinism x2, golden vectors, round trip
make -C flight lint      # clang-tidy when present; says so and skips when not
```

`make test` builds at `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`, so
a single warning fails it. `tests/test_flight_build.py` runs the same target from
`pytest`, so the verification trio still covers the C++.

## Layout

```
  include/sentinel/
    Types.hpp       F' primitive names. Work item 9 swaps this one header for
                    Fw/FPrimeBasicTypes.hpp and nothing else changes (D31)
    Config.hpp      compile-time bounds; every buffer is sized from these
    Status.hpp      LoadStatus, shared by value with sentinel_export.Status
    Crc32.hpp       CRC-32/IEEE 802.3, the same one F' Utils::Hash computes
    ModelFile.hpp   the reader. docs/MODEL_FILE.md is normative
    Gru.hpp         the forward pass, transcribed from reference.gru_cell
    Ewma.hpp        span 105, bias-corrected, F64 in and F32 out
    Baseline.hpp    Level 1, the statistical baseline the loader degrades to
    TrailingWindow.hpp  a trailing window per channel with F64 moments, the
                    primitive both new streams are built on (MODELS 39 stage 1)
    DynamicThreshold.hpp  telemanom's nonparametric threshold, streamed and
                    solved once per 70-tick segment (D62, MODELS 39 stage 2)
    DerivativeStream.hpp  the first derivative, trailing-standardised. Fused
                    with the residual it reached EVAL 17 of 19 where the frozen
                    arm reached 4 (D65, MODELS 39 stage 3)
    Detector.hpp    the whole per-tick pipeline
  src/              one .cpp per header
  test/             the test binaries and the committed golden vectors
```

**(!) Three decision layers run; one emits.** `Detector::crossing` and
`Detector::emitted` are D25's static quantile and nothing else. The dynamic
threshold and the derivative stream are computed every tick and reported through
`dynamicEmitted()`, `fused()` and `fusedScore()` -- **D65 re-opens the decision
layer and names no flight configuration**, so promoting either to the warning
would make this port the adoption decision. Which one the F' component raises an
event on is the component's to wire and the owner's to decide.

## The flight rules this obeys

F' is pinned at v4.3.0 (D31). Its own statement of the C/C++ design rules is
`.github/skills/fprime-cpp-design/SKILL.md` at that tag. `docs/MODELS.md` 19.3
tabulates each rule with its CPP number and how this core meets it, and records
three rules the work item's brief had stated inaccurately.

The short version: no exceptions, no RTTI, no STL, no allocation anywhere, no
recursion, every loop bounded by a header field the loader has already checked,
`F32`/`F64`/`U32` rather than bare `float`/`double`/`int`, `memcpy` rather than
`reinterpret_cast`, and every fallible return value checked.

## What is not here, and where it went instead

- **The measurement that would price one of the two departures.** Backward
  dilation is dropped (`docs/MODELS.md` 39.5) because it would mark timesteps
  already emitted, and what that costs on the 38 in-range contextual events is
  registered as **N6 and is not yet measured** -- it needs one bucket read, which
  has not been taken. No figure for it exists anywhere and none should be quoted.
- **The guard-cell choice.** 39.4: the window includes the 70 steps it judges, so
  a segment is decided at its end. Guard cells would remove that latency and
  `docs/MODELS.md` 10.7 measured them costing recall 38/46 to 34/46 on one channel
  set while *helping* on another. **Neither is adopted**; N5 measures it.

## The three things easiest to get wrong

1. **`b_hn` sits inside the reset product** and cannot be folded into `b_in`
   (`docs/MODELS.md` 3, D26). `Gru::step` forms the recurrent product once with the
   whole of `b_hh` and multiplies its third block by the reset gate.
2. **The state update is ATen's `(h - n) * z + n`**, not the textbook
   `(1 - z) n + z h`. Equal in exact arithmetic, not in float32, and the reference
   uses ATen's.
3. **The dtype map is not uniform.** The GRU and the head are F32; the prediction
   aggregation and the EWMA are F64 with an F32 result; the threshold comparison is
   F64. A core that ran the decision layer in F32 throughout would be a different
   detector. `-Wconversion -Werror` keeps every narrowing explicit.
