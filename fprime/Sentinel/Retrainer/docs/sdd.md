# Sentinel::Retrainer -- the retrainer, exported opt-in

**Exported OPT-IN, OFF by default** (`docs/DECISIONS.md` D84). `fprime/library.cmake`
registers this component only when `SENTINEL_WITH_RETRAINER` is ON, so a mission that does
nothing inherits no OCaml runtime. Its status is D83's, exactly: **the chosen retraining
implementation, host-verified, not flight-qualified.** E1 and E3 passed; E2 returned **NO
VERDICT**; E4 has never run, because it needs flight hardware.

**The case against, beside it** (D83, D84): no flight heritage; no qualified compiler; no
certification precedent for a garbage-collected runtime in flight; x86-64 and arm64 Linux and
arm64 macOS only -- no 32-bit ARM, no musl, no documented cross-compile recipe; no stability
promise in its own documentation; Rust's footprint is smaller, with Ferrocene qualified and
OPS-SAT heritage -- and **no Rust comparison has been built here**.

**What it trains on.** Its only input is `schedIn`. The window its cycle trains on is a
deterministic drive the component fills itself -- **not telemetry** -- and its step budget
is a unit test's value. What a mission adopts is the pipeline: the shape, the cycle, the
candidate file and the separate process. **No candidate may be swapped in operationally**;
nothing onboard scores one, and a human's `RELOAD_MODEL` is the only way a model changes.

**Where it lived.** `fprime/SentinelRef/Retrainer/` until D84, with the FPP module `Retrain`
and the UT executable `SentinelRef_Retrainer_ut_exe`, which is what `docs/MODELS.md` 47 to 73
cite. D84 consequence 5 maps those names to these.

## 1. What it is for

E1 proves a chain and nothing else: **compile, link, OCaml runtime startup, the C
boundary, F' integration**. It carries **no machine learning**, and the payload is
arithmetic whose answer can be checked by hand -- feed `1.0 .. 10.0`, read back count 10,
sum 55, mean 5.5, all three exact in F64 -- so a transcription error cannot hide inside a
tolerance. `docs/MODELS.md` 47.9 registers it as predictions X1 to X3.

## 2. The two boundaries, which are not the same boundary

`docs/MODELS.md` 47.6 separates them, because conflating them produces a design F'
forbids.

| | Boundary A, the hub | Boundary B, the C boundary |
|---|---|---|
| Between | two OS processes | C++ and OCaml, **inside** one process |
| Crosses | serialized values only | fixed-size scalars, status codes, `Bigarray` |
| Forbidden | **pointers, `Fw::Buffer` included** -- an address is meaningless on the far side | any OCaml value persisting in C++ across a call |

**This component sits entirely on boundary B.** It does not use `Svc::GenericHub` and does
not cross a hub. The separate-process arrangement D70 consequence 2 requires is what E2
builds and measures; E1 proves the process can host the runtime at all.

**`Bigarray` is the telemetry carrier at boundary B and nothing else is**, because its
buffer lives outside the OCaml heap and is therefore never moved by the collector. The
stubs wrap caller-owned memory with `CAML_BA_EXTERNAL`, so OCaml neither owns nor frees
it.

## 3. The rules it is written to

Quoted at `docs/PHASE5.md` 3 from `.github/skills/fprime-cpp-design/SKILL.md` at
`nasa/fprime` v4.3.0.

- **CPP-1, no allocation after init.** The accumulator is sized once at
  `RETRAINER_CAPACITY`; the sample and export buffers are fixed-size members. A feed that
  would exceed capacity is **refused with a status code**, never reallocated.
  **(!) This says nothing about the OCaml runtime's own heap**, which is exactly what E2
  and E3 exist to measure and is **not** claimed here.
- **CPP-3, fixed-size numerical types**, at the header and throughout.
- **CPP-21, no C-style arrays in interfaces.** Every array in `sentinel_retrainer.h` is
  paired with its length.
- **CPP-25, no exceptions.** `retrainer.ml` catches every OCaml exception in `guard` and
  returns a status code. This deployment builds `-fno-exceptions`, so a crossing would be
  undefined behaviour; it is made **structurally impossible** rather than merely avoided.
- **`Objective.md` 11 rule 5, fixed compute per cycle.** One feed of a fixed block, one
  step, one export. No convergence criterion and no variable iteration count.
- **`Objective.md` 11 rule 3 in spirit.** It declares no command and no parameter, so it
  also carries no command ports -- which is additionally why the `-Wshadow` collision
  `PowerSim.fpp` records cannot arise here.

## 4. Degrade rather than die

A runtime that does not start is **not a transient fault**. `boot()` raises
`RuntimeUnavailable` once, leaves the component inert, and **does not retry**; `schedIn`
then returns immediately and silently. The topology goes on running. This is the same
posture `Sentinel::Monitor` takes on a refused model file (D5, `Objective.md` 14.10), and
for the same reason: apparatus that fails a deployment teaches nothing about the thing
being measured.

## 5. What it does not prove

- **Not that OxCaml allocates nothing.** That is E3, and `[@zero_alloc strict]` appears
  nowhere in E1.
- **Not that the detector is isolated from the retrainer's collector.** That is E2, which
  returned **NO VERDICT**, and C2 is still unverified. D83 chose OxCaml with that
  precondition unmet and says so; it is not converted into a pass by being chosen.
- **Not that any of this builds for a flight target.** That is E4.
- **Not that the arithmetic is useful.** It is a sum and a mean, chosen because both are
  exact in F64.

## 6. The mission's shape (D84, `docs/MODELS.md` 77)

Until D84 the cycle was fixed at `Config.hpp`'s maxima -- 16 inputs, 75,360 parameters -- and
no candidate it built could replace an 8-channel model (72.4). D83.1 route 1 is now taken:
`scripts/oxcaml_shape.sh --channels C --predictions P` generates `deep_f32.ml` at the
mission's shape by substituting three lines, runs every gate against the generated source --
`[@zero_alloc strict]` with zero `assume`, a deliberate allocation rejected, an out-of-range
read raising, HO1 on the cycle's own weights, and (with `EX1=1`) the exhaustive gradient
check -- and builds the object this component links, beside the generated
`sentinel_cycle_shape.h` that `Retrainer.hpp` reads its extents from. The F' build selects it
with `SENTINEL_RETRAINER_CHANNELS` and `SENTINEL_RETRAINER_PREDICTIONS` and stops, naming the
command, if it is absent or disagrees. Hidden `[80, 80]` and window 250 are fixed.

## 7. Where it may be switched on

`retrainer_platform.cmake` refuses any system, processor or cross-compile OxCaml does not
support, with one message naming the supported platforms. **This project has built it only
on arm64 macOS**; Linux is OxCaml's claim, not a result here.

