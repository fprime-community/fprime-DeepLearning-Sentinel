# Phase 5: in-flight retraining, under human approval

**Scoped 2026-09-09 (`docs/DECISIONS.md` D57, Objective.md section 12). Nothing
here is built.** This document records a design and the questions it has not
answered, so that the design can be argued with before any code exists. It is
written at the same point in the process as `docs/PHASE2.md` was: at the boundary,
describing what a later phase would inherit.

**Read `Objective.md` section 11 first.** Rule 1 says the model is frozen in
flight and retraining is explicit and human-approved, because *a slowly degrading
spacecraft must never be able to teach the detector that degradation is normal -
that is precisely the condition it exists to catch*. **Nothing in this document
weakens that rule, and a design that did would be refused rather than argued for.**

---

## 1. The problem this exists for

`docs/PHASE2.md` 5b and D29 established that a detector's calibration does not
survive the mission. A floor fitted on Mission 1's first 7.36M steps sat under
**86.7%** of a later window's nominal residual on the same spacecraft, and on
SMAP/MSL no train-calibrated threshold transfers for any arm at all (D48).

Work item 10 answers half of that: **the threshold** is recalibrated in orbit
without retraining, because `model.bin`'s PARAMS block is separately CRC'd and
carries its own `param_version` (`docs/MODEL_FILE.md` 6.1).

**It does not answer the other half.** A threshold recalibration cannot fix a
*forecaster* that has drifted away from the spacecraft - a solar array that has
aged, a thermal path that has changed, a duty cycle nobody flew before launch.
The model is the part that encodes what normal looks like, and after enough years
the pre-launch training data no longer describes the vehicle.

**The honest statement of the gap:** everything this project has built assumes the
mission's pre-launch telemetry is representative for the mission's life. For a
two-year smallsat that is close to true. For a flagship it is not.

---

## 2. The design

```
  FLYING MODEL          frozen, the model.bin that was uplinked.
                        Never modified in flight. Rule 1.
                              |
                              | telemetry, every cycle
                              v
  SHADOW MODEL          retrains onboard on recent HEALTHY telemetry,
                        in statically allocated memory, using
                        mlpack/ensmallen. It scores nothing operational
                        and raises no event.
                              |
                              | both models' sanity reports, downlinked
                              v
  GROUND                compares shadow against flying using the
                        pre-launch sanity report run in orbit -- how
                        often each fired on held-out healthy data.
                        A human reads it.
                              |
                              | an explicit, human-approved command
                              v
  SWAP                  the shadow becomes the flying model. The
                        PREVIOUS model is retained for rollback.
                        The threshold recalibrates per work item 10.
```

**Five properties, each of which is the answer to an objection.**

1. **The flying model never changes.** The shadow is a separate set of weights in
   separate memory, and no path exists from the shadow to an emitted event.
   Objective.md 11 rule 1 holds literally, not in spirit.
2. **A human commands the swap.** Not a threshold, not a schedule, not an
   autonomous criterion. The ground decides, from evidence it can read.
3. **The evidence is the instrument this project already has.** The pre-launch
   sanity report (`docs/HARNESS.md` section 1) reports how often a detector fired
   on held-out healthy data. Run in orbit on both models, it is a comparison a
   reviewer can evaluate. **It is a sanity report, not a target**: nobody tunes
   anything on the strength of it, and a shadow model that fires less is not
   thereby better.
4. **The previous model is kept.** A swap that turns out wrong is undone by a
   second command, not by an uplink pass.
5. **Training data is healthy data, by exclusion.** Objective.md 6.1 - a
   spacecraft has no failure examples, so the model learns normality. The onboard
   selection of "recent healthy telemetry" is the hardest unsolved part of this
   design and is question 4 below.

---

## 3. The flight rules this has to satisfy

Read from F' v4.3.0's own statement of them,
`fprime/lib/fprime/.github/skills/fprime-cpp-design/SKILL.md` at commit
`7d8f579f159d2f7c2d4984d92828575e37f87fa6` (D31), quoted rather than recalled.

**CPP-1, no dynamic memory after initialization** (`SKILL.md:41-50`):

> Allocate all dynamic storage during component initialization or boot. After
> init, no `new` / `delete` / `malloc` / `free`, and no implicit allocation
> through STL containers, `std::string`, or exception machinery (see CPP-25).
>
> Allowed: pre-sized arrays sized at compile or init time, `Fw::Buffer`
> references to memory owned elsewhere (CPP-2), object pools allocated once at
> boot.

**This is the binding constraint on the whole design.** A training loop allocates:
gradients, optimiser state, minibatches, temporaries. Every one of those has to be
sized at init and never grow.

**CPP-25, no exceptions, RTTI, STL, `std::string`** (`SKILL.md:279-290`):

> F Prime flight code is compiled with exceptions and RTTI disabled
> (`-fno-exceptions` and the no-RTTI flag) and deliberately avoids the STL.
>
> - `try` / `catch` / `throw`: forbidden.
> - `dynamic_cast`, `typeid` (require RTTI): forbidden.
> - `std::vector`, `std::map`, `std::list`, etc.: forbidden -- use
>   `Fw/DataStructures` (CPP-22).
> - `std::string`: forbidden -- use `Fw::String` (CPP-24).

**CPP-3, always use fixed-size numerical types** (`SKILL.md:113-127`): `F32` and
`F64` rather than `float` and `double`, which are "forbidden in F Prime code
except at external boundaries".

**And Objective.md 11 rule 5**, deterministic onboard code: fixed memory, fixed
compute per cycle, same inputs give same outputs. A training step that takes a
variable number of iterations is not obviously compatible with a rate group, and
question 5 below is about that.

---

## 4. Why mlpack, and the claim this rests on

`flight/` is a hand-written transcription of `src/sentinel_models/reference.py`
checked against it at 1e-5, and that was the right choice for **inference**: a few
hundred lines of arithmetic, no allocation, auditable by a review board. Writing a
*training* loop the same way - forward, backward, an optimiser, a learning-rate
schedule - is a much larger surface, and it is the kind of code where a library
that many people have used is worth more than a transcription only we have read.

mlpack is C++, header-oriented, and builds on Armadillo for its linear algebra,
with ensmallen for the optimisers.

### (!) The load-bearing claim is UNVERIFIED

> **Armadillo can be made to run on statically allocated memory - fixed-size
> matrix types with no allocation after initialisation.**

**This claim has not been verified and nothing in this document may be quoted as
though it has.** It reached this project as a spoken finding, with no source
attached. `docs/HARNESS.md` section 1 states the rule it falls under - *no number
enters a document that was not read from an artifact* - and `docs/NARRATIVE.md` 11
records what it cost the last time a reading was carried without its source: five
recorded readings turned out to be wrong and one of them cost four pre-registered
rungs of correct, careful, correctly-stopped work.

**What would discharge it:** the exact Armadillo mechanism named, cited by file and
version - the fixed-size matrix template and its guarantees, whether it covers the
expression templates mlpack's layers actually instantiate, and whether ensmallen's
optimisers allocate. Until then this section says only that *if* the mechanism
exists as described, mlpack is the candidate; and if it does not, **the design
falls back to a hand-written training loop under the same discipline as
`flight/`**, which is more work and is not blocked on anybody.

**Everything in section 2 downstream of this claim is conditional on it.** The
shadow-model architecture, the ground comparison and the human-approved swap are
not - they are library-independent - but the assertion that retraining can happen
onboard *at all* under CPP-1 is exactly what this claim is carrying.

---

## 5. The open questions, stated as open

1. **The exception-free path.** mlpack and Armadillo report errors by throwing.
   CPP-25 compiles with `-fno-exceptions`, under which a `throw` terminates. What
   is the supported configuration - a build flag, an error-callback hook, or a
   vendored subset - and does it cover the code paths a training loop reaches?
2. **A fixed-memory recurrent cell.** The models here are GRUs. Whether mlpack's
   recurrent layers can be instantiated at fixed size, with fixed-size gradient
   buffers and no allocation in the backward pass, is unknown and is the question
   that decides whether the library is usable at all for this.
3. **OpenBLAS determinism.** Armadillo delegates to a BLAS. Objective.md 11 rule 5
   requires same inputs to give same outputs, and threaded BLAS reductions are not
   generally bitwise reproducible. Single-threaded, a pinned build, or a
   reference BLAS - each is a different cost, and the answer has to be measured
   the way `flight/`'s determinism already is (`docs/MODEL_FILE.md` 9).
4. **What counts as "recent healthy telemetry", decided onboard.** This is the
   hardest question and it is the one closest to the rule. Training the shadow on
   a window that silently contains the degradation is exactly how rule 1's failure
   mode arrives by another route. Candidates: only windows the *flying* detector
   was quiet through, only windows the ground has cleared, or a
   data-sufficiency-graded window under Objective.md 10.2. **None is chosen here,
   and the choice is a pre-registration of its own.**
5. **Bounded compute per cycle.** A training step has to fit inside a rate group's
   budget or run in a background task with a hard cap, and either way the answer
   has to be a fixed number of operations rather than a convergence criterion.
6. **Where the shadow's weights live.** `model.bin` version 1 has no room for two
   models, and D30 froze it. A second file, a second slot, or a `format_version` 2
   are three different answers with three different review costs.
7. **Whether a Gaussian or otherwise larger head is ever representable.** D56's
   arm already hit this: `Weights` refused a doubled head, because **a Gaussian
   head is not representable in `model.bin` version 1**. Any architecture Phase 5
   retrains has to be one the format can carry.

---

## 6. What would have to be true before this is built

- **The toolkit exists** (Objective.md 13 item 12). Onboard retraining is the same
  training recipe as the ground one; building it twice, in two languages, before
  the ground one is settled would be building the second one against a moving
  target.
- **Phase 3 has measured something.** The physics testbed is where a shadow model
  can be shown to beat a stale flying one on data with ground truth, which is the
  only honest way to demonstrate the swap is worth offering to a human.
- **The claim in section 4 is discharged**, one way or the other.

**Until all three, this document is the deliverable.** It is recorded now because
the design was worked out now, and `docs/HARNESS.md` 5a's rule applies to
documentation as much as to decisions: written later is written wrong, because the
alternatives get forgotten first.
