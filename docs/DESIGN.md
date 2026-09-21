# Design

> **Paths outside this branch resolve on `dev`** at commit **`7791419`** (`docs/DECISIONS.md`
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

**(!) And this branch ships a retrainer, so the two sentences are reconciled here rather
than left to a reader to notice.** *Online* learning means the flying model updates itself
from the telemetry it is judging. That is what rule 1 forbids and it stays forbidden. What
`fprime/SentinelRetrain/` does is different in every respect that matters: it trains a
**shadow** model, in a **separate OS process**, from a **frozen snapshot** of healthy
telemetry; the flying model is never written to; and a swap is **offered**, never taken --
it requires a pre-launch sanity report and an explicit ground command. **Rule 1's own words
are the ones that make both true: retraining is explicit and human-approved.** Section 9
says what is built, what is proven, and what is not.

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
  fprime/.../test/ut/   12 of the 12 proven to degrade to Level 1 with the code named
                        in the event, without failing the topology
```

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

## 8. Integration: adopting this component into your deployment

**Four edits to your project, and none of them copies a source file.**
`scripts/fprime_ref_patch.sh` performs exactly these against F' v4.3.0's own `Ref`
deployment, unmodified, and then proves the result by symbol rather than by string -- so
this recipe is executed and checked rather than described.

**There is no manifest and nothing to generate.** `fprime/library.cmake` is the only file
an adopting project needs from here, and it already exists.

```
  1  settings.ini      add one line:  library_locations: <path to this repo's fprime/>
                       fprime_ref_patch.sh:46-60

  2  Top/instances.fpp add one instance:
                         instance sentinelMonitor: Sentinel.Monitor base id 0x20000000
                       fprime_ref_patch.sh:76

  3  Top/topology.fpp  add the instance to the topology, and one rate-group connection:
                         rateGroup1Comp.RateGroupMemberOut[8] -> sentinelMonitor.schedIn
                       fprime_ref_patch.sh:86, :89

  4  Top/*Packets.fppi one telemetry packet, if your deployment packetizes:
                         packet Sentinel id 100 group 2 { Score, Threshold,
                           ActiveMode, TicksSinceWarmup, LoadStatus }
                       fprime_ref_patch.sh:104-110
```

Then `fprime-util generate && fprime-util build` (`fprime_ref_patch.sh:116-123`).

### What you still have to write, and it is about forty lines

**The component does not read your telemetry database.** `Sentinel/Monitor/docs/sdd.md` and
`Monitor.fpp:20-25` give the reason: `Fw.Tlm` carries a serialized `TlmBuffer`, and
`model.bin`'s CHANNELS record has no per-channel type tag to deserialize it with. So a
mission converts its own typed telemetry into one `Sentinel.ChannelVector` per tick, using
F's **Passive Adapter Pattern**.

**`fprime/SentinelRef/ExampleAdapter/` is that adapter, at its smallest useful size.** It
holds the latest value of each channel and emits the vector on the rate-group tick; the
conversion is the only thing it does. **Copy the directory, rename it, and replace its
`valueIn` port with your own types.** It is example code and is deliberately not exported:
`fprime/library.cmake` exports `Sentinel/Monitor` and nothing else, so adopting Sentinel
does not drag this -- or the testbed, or the OxCaml experiment -- into your build.

You wire exactly two input ports:

```
  sentinelMonitor.schedIn      <- your rate group          Monitor.fpp:110
  sentinelMonitor.channelsIn   <- your adapter's output    Monitor.fpp:116
```

Both are `sync`. Put `channelsIn` at a **lower** port index than `schedIn` on the same rate
group, so the vector for this cycle lands before the detector steps; a producer on another
thread makes that port `guarded` instead (`Monitor.fpp:112-115`).

**(!) The component is `queued`, so its instance declaration needs a queue size.** That is one
extra line and no extra thread:

```
  instance sentinelMonitor: Sentinel.Monitor base id 0x20000000 \
    queue size 10
```

The queue exists for one thing, the reload command below, and it is **drained at the top of
the rate-group tick** rather than on its own thread -- so the cyclic work still runs in the
context of the rate group that ticks it and a slip is still visible as a slip. The drain is
bounded by the queue size, so a burst of commands cannot make one tick unbounded.

**The component declares one command, `RELOAD_MODEL`, and it takes the path of the file to
load.** The path is bounded by F's `FW_CMD_STRING_MAX_SIZE`, which is **40** characters in the
default configuration, so a model file has to live somewhere short -- beside the binary rather
than at a long absolute path.

**Uplink the new file beside the running one, not over it.** A candidate that fails to load
would otherwise have destroyed the file it was offered to replace; with a separate path the
component restores the previous model and says so, and rolling back is a second `RELOAD_MODEL`
naming the old file. **A reload that changes the channel count is refused**, because the
channel source is wired by your topology and has not changed -- a file declaring a different
width is a model for a different subsystem, and it would otherwise load without complaint.

### The model file, and an example you can load today

The component loads a `model.bin` at init and refuses it on any of twelve grounds rather
than flying a file it cannot verify. `docs/MODEL_FILE.md` specifies the format completely
enough to write one independently.

**Two loadable example files are committed in this branch:**
`flight/test/vectors/p1.bin` (3 channels) and `flight/test/vectors/p2.bin` (1 channel),
both at `param_version` 2, both produced by the flight configuration and both re-emitted
byte-identically by the round-trip test. They are named as test vectors because that is
what they are for -- but they are real model files, and loading one is the fastest way to
see the component arm.

### (!) Two things that will not work from this branch alone

**The F' framework itself is not here.** `fprime/lib/` is gitignored, so the F' documents
this file and the SDDs cite -- `docs/user-manual/design-patterns/hub-pattern.md`,
`docs/user-manual/framework/component-and-port-selection.md` -- resolve only once you have
your own F' v4.3.0 checkout. `docs/FPRIME.md` pins the version and the commit.

**`scripts/`, `src/` and `tests/` are not on this branch** (see the omissions table in
`README.md`). `scripts/fprime_ref_patch.sh` is cited above because it is the executable
form of this recipe; it resolves on `dev` at the commit named at the top of this file.

## 9. Onboard retraining, and the language it is being tried in

**Status: an experiment on the branch, adopted for nothing.** Nothing in section 1's chain
depends on any of it, `fprime/library.cmake` exports the Monitor and not this, and a mission
adopting Sentinel inherits none of it.

### Why a retrainer at all
The flying model is frozen so that a degrading spacecraft cannot teach the detector that
degradation is normal. That is right, and after five years it describes a spacecraft that no
longer exists. A shadow model retrained on recent healthy telemetry is the only way to fix
the second problem without reintroducing the first -- provided the flying model is never
touched and no swap happens without a human.

### Why a separate process, and it is a requirement rather than a preference
`SentinelRetrain` is its own deployment with its own `main`. The reason is that the
retrainer is written in OxCaml, whose runtime is garbage-collected: OCaml 5's minor
collector is stop-the-world across domains, so a retrainer sharing a process with the
detector would stall the detector at a collection barrier **even if the retrainer's own code
allocated nothing**. `SentinelRef` carries no OCaml runtime at all, and that is asserted by
symbol on every test run rather than assumed.

### Why OxCaml, and everything against it in the same paragraph
Training is thousands of lines of array arithmetic, which is where memory-management
mistakes live. Flight rules forbid heap allocation after init; in C++ that is checked by a
human reading code, and OxCaml's `[@zero_alloc strict]` makes it **a compile error,
transitively across callees**. The strategic question is whether flight software can have a
safe high-level language at all, and this is a contained component with nothing to lose.

**The case against, which is stronger on every row that has been measured by anybody:**
OxCaml has **no flight heritage** and **no qualified compiler**; it targets 64-bit Linux and
arm64 macOS **only**; its own documentation promises no stability; there is **no
certification precedent for a garbage-collected runtime** in flight software; **Rust's
footprint is smaller**, it has a qualified toolchain in Ferrocene and actual spaceflight
heritage on OPS-SAT; and **no Rust comparison has been built here**, so on that comparison
this project is reasoning rather than measuring. Two further findings belong in this list
because they were discovered rather than anticipated: a `Bigarray` enum-conversion warning
that the framework's own flag set rejects, and -- **the one a reader should weigh most** --
**the runtime's thread affinity is a deployment constraint that unit-test evidence cannot
show.** OCaml 5 grants the domain lock to whichever thread calls `caml_startup`, and an F'
active rate group ticks from a different one; every rung of the evidence before a real
deployment ran single-threaded and could not have seen it.

### What is proven
The arithmetic of a training step -- forward pass, a 250-step backward pass, two GRU layers
and the output head, and the optimiser -- holds at float32 under `[@zero_alloc strict]` with
zero `assume` annotations, and its gradients were checked at **every one of 75,360 indices**,
giving a **margin of 3.26x over the tolerance model**. The cycle runs through real F' ports.
The retrainer's weights have a file format this branch's own loader reads. The OCaml runtime
starts inside a real F' deployment binary and runs a cycle per tick. Two deployments now talk
across a `Svc::GenericHub`, carrying **serialized values only**, and a full cycle crosses
from the retrainer's process into the detector's and on to the ground.

**The crossing is not lossy, and that is now measured rather than owed.** A passive tap inside
the deployment counts what the hub emits, because the ground is downstream of the telemetry
database and the decoder and cannot see the crossing at all. Over `Drv.Udp` **every tick
delivered every channel and the event**; over a stream transport **none did**, because the hub
dispatches on an exact size match and drops anything else silently, so two messages that
coalesce into one read are both lost. The substitution to a datagram transport was registered
before the number that forced it.

**And the retrainer's process now writes a candidate model file that this branch's own loader
accepts.** The retraining process builds it as the flying file with new weights -- copying the
bytes, overwriting the weights payload and patching the two CRCs, touching no shape field and
leaving `format_version` at 1 -- and `Detector::load` returns `OK` on the result. One flipped
weights byte still returns `BAD_STATIC_CRC`. The file's own static CRC is reported in an event
that travels by a different path from the file, so the ground checks the bytes it received
against a number that did not arrive with them.

### What is not
**The pre-launch sanity report is not a usable gate yet, and that is measured rather than
suspected.** A criterion was built, tested against a model that had nothing to learn, and
**certified it** -- so it was rebuilt from a measured noise floor instead of a chosen margin.
The rebuilt second term works and discriminates strongly. The first term was blocked on a
design decision nobody had taken -- whether the onboard retrainer reproduces the flying model's
random seed -- because that decision selected between two noise floors whose margins differ by
a factor of nearly five.

**That decision has since been taken: the shadow is initialised from the flying model's
weights and fine-tuned, so there is no random initialisation on the flight path and no seed to
reproduce.** The smaller floor is the operative one and the margin follows from a factor fixed
before any floor was measured. **The trade-off is real and is stated rather than discovered
later: a warm-started shadow tracks drift, it does not re-learn.** It begins at the flying
model's answer, so a systematic error in the flown weights -- a channel it never modelled well,
a regime absent from the original training set -- is inherited by every shadow this design will
produce. It buys adaptation and gives up correction, and a mission whose problem is the second
one is not served by it.

**Taking that decision removed the blocker and did not make the gate usable, and the arm that
was owed to show it has now been run.** It found something worse than a missing measurement.

**(!) A warm-started shadow is better than the flying model whether or not anything has
drifted.** Six shadows were warm-started from the flying model's weights and retrained, one on
each segment of healthy data -- and **every one of them beat the flying model by a wide margin,
including the one retrained on the flying model's own training data.** No drift. No new data.
No later segment. Just more training.

**So the gate's first term -- "the shadow is measurably better" -- was satisfied by the
retraining itself.** It could not distinguish *the spacecraft changed* from *the shadow ran for
longer*, and no choice of threshold repairs a term that passes unconditionally.

**That term has since been replaced, and the criterion now discriminates.** The candidate is no
longer measured against the flying model. It is measured against a **control**: a second shadow,
warm-started from the same weights, trained at the same step budget, on **the flying model's own
data**. Control and candidate differ in exactly one thing -- the data they were retrained on --
so the training advantage cancels by construction and what is left is whatever the new data did.

```
                       against the flying model     against the control
  nothing has drifted          much better              slightly worse
  something has drifted        much better              much better
```

Measured: with nothing to learn, the candidate comes out **2.6217% worse than its control**,
on the wrong side of zero. With a drift to learn it comes out **42.8489% better than its
control**, against a margin derived from how far two controls differ from each other. **The old
comparison could not tell those two cases apart.** This is the first version of this criterion
that has been near zero when nothing happened and large when something did.

**The cost is real and is stated rather than discovered.** Certification now needs **two trained
models, not one**. Either the retraining process trains the control as well -- doubling the
training work per cycle, and the fixed step budget was not derived for that -- or the ground
reproduces the control, which means keeping the flying model's original training data somewhere
the ground can reach it. **Neither route is chosen here**, and the choice belongs to whoever
costs the downlink and the onboard budget together.

**`Objective.md` section 12's gate is still not enforceable, and that claim is not made.** The
criterion discriminates **on one synthetic fixture**, at a drift magnitude that is a sensor
doubling its output, and nothing about a mission follows from that. The way it could still fail
was named before the arm ran and did not occur here: the control trains on older data than the
candidate, so on telemetry with real seasonal structure the candidate could beat it from
recency alone. That is the first thing to check on real data. No shadow model may be swapped
in.

**The candidate reaches the ground and a reload can be commanded.** The file downlink carries
it, a single command loads an approved model into the running detector, and the model it
replaces is restored if the new one is refused -- which is what `Objective.md` section 12 means
by keeping the previous model for rollback, in the form that costs no extra memory.

**(!) And running that path found something reading it did not.** A candidate built for a
different subsystem -- a valid file, every checksum correct, no refusal code -- **loaded into
the reference deployment without complaint**, because the loader lets the file's channel count
win over the topology's. That is right when the component starts, where the file is the only
thing that knows its own shape. It is wrong on a commanded reload, where the channel source is
already wired and has not changed, and it would have left the detector scoring whatever the
unwired slots of the vector held. The commanded path now refuses a width change, restores the
previous model and names both widths.

**And the candidate the retrainer writes is not a replacement for the model this deployment
flies.** The training cycle is fixed at the configuration maxima and the reference deployment's
model is narrower, so the two weight blocks are different sizes and the writer refuses the
mismatch -- correctly, because writing one architecture's numbers into another's container
produces a file that loads and means nothing. **The retraining engine can produce a candidate;
it cannot yet produce a candidate for this mission.**

**Also not proven:** that any of this runs on flight hardware
(**E5 is HOST-VERIFIED PENDING TARGET**); that the separate process actually isolates the
detector's timing (**C2**, unverified -- this host's power management downclocks an idle core
and the confound exceeds the effect); and that the toolchain builds for the flight target at
all (**C4**, unverified). **No timing figure from any of this work should be quoted.**

**Nothing onboard scores a candidate.** The metrics that cross with it are the cycle's own --
the file's length and CRC, the step count, the loss, the sample count -- and none of them is
either part of the sanity criterion. **Both parts are computed on the ground from the
downlinked candidate.** Onboard held-out scoring does not exist and is owed.

The full record is on `dev`: `docs/DECISIONS.md` D70, D73, D74 with its riders, D76 and D77,
and `docs/MODELS.md` sections 47 through 72.
