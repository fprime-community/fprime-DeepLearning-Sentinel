# fprime-DeepLearning-Sentinel

**A reusable flight-software component for NASA's F' (F Prime) framework, which warns of
spacecraft anomalies that never cross a limit line.**

> **This branch is the product.** Paths outside it resolve on `dev` at commit **`6fa33bf`**
> (D69). The guards that keep every figure here true run on `dev`, not here.

[The problem](#the-problem) |
[The result](#the-result) |
[How it works](#how-it-works) |
[What is in this branch](#what-is-in-this-branch) |
[Quick start](#quick-start) |
[Use it in your F' project](#use-it-in-your-f-project) |
[The retrainer](#the-retrainer) |
[Status](#status) |
[Documents](#documents) |
[Not on this branch](#not-on-this-branch) |
[Branches](#branches) |
[Licence](#licence) |
[Citation](#citation)

## The problem

Onboard fault protection checks each telemetry channel against a high and a low limit. That
catches loud failures and misses quiet ones. A radiator that should swing -20 C to +40 C
each orbit but sits flat at +15 C reports a legal number while the sensor is dead. It is not
the value, it is the value in context, and a limit check has no notion of context.

## The result

### Detection, on NASA's SMAP/MSL telemetry

Channel-disjoint split fixed before tuning; held-out (EVAL) half:

| Arm | Caught | Alarm rate |
|---|---|---|
| telemanom's published rule (frozen) | 4 of 19 | 0.6820% |
| Sentinel (residual fused with the first derivative) | **17 of 19** | 0.6820% |

**n = 19 per half; the harness's own n<20 rule (`docs/HARNESS.md` 1, on `dev`) stamps this
UNDERPOWERED.** Alarm rate is matched, so recall is the only thing varying.

**The population is anomalies that stay at or within their channel's training range**, so a
limit check sees none of them -- a premise of how the population was selected, not a measured
score, and it rests on real flight limits sitting outside the historical range.

**"At or within", not "strictly inside": 23 of the 39 sit exactly on a training-range rail.**
The two halves sum to 38, not 39, because one event is excluded on a training stall: of the
labelled contextual anomalies in the dataset, **39** stay at or within the envelope and
**38** are scored. `docs/EVIDENCE.md` 2 is that arithmetic.

The method is JPL's (Hundman et al., KDD 2018, *telemanom*). What this adds is the flight
packaging, the evidence base, and one finding: telemanom's published false-alarm filter
deletes true detections on this population, and the first derivative recovers them.

### Warning time, on a physics testbed

No dataset this project holds can produce a warning time, so one was built: a simulated
coupled power and thermal plant with declared limits, on a real 1 Hz clock
(`fprime/SentinelRef/PowerSim/`).

| Measure | Value |
|---|---|
| median fault-attributable lead | **9,774.5 ticks** before the first limit trip |
| false alarms | **0.1608%** of warmed healthy ticks |
| worst tick | **326 us**, against this deployment's 1 Hz period |

**Three caveats, all mandatory:**

- **In ticks, never hours or minutes.** The plant's time constants are chosen, so its
  seconds are not a mission's.
- **Only the fault-attributable lead counts.** The naive lead -- first warning to first limit
  crossing -- is **16,525 ticks and it is false**, because the healthy control warns at the
  same tick with no fault present. A warning is the fault's only when it is absent from the
  healthy run on the same seed.
- **Ten runs are not ten systems.** They share a plant, a fault mode, a rate and an injection
  tick, so the effective **n is close to 1**.

`docs/EVIDENCE.md` is both arguments with their caveats.

## How it works

```text
  healthy telemetry  ->  ground toolkit  ->  model.bin  ->  Monitor, on the spacecraft
                                                                |
                                                    warning event naming the channel
                                                                |
                                                                v
                                                              ground

  retrain loop:  SentinelRetrain (own process)  ->  candidate model.bin
                        ->  downlink  ->  A HUMAN APPROVES  ->  uplink
                        ->  RELOAD_MODEL command  ->  Monitor
```

- A GRU forecaster predicts each watched channel every cycle. The rule is the maximum of two
  signals, each standardised against that channel's own trailing window: the smoothed
  prediction residual, and the channel's first derivative.
- **The derivative is what decides.** Over **both** halves together this arm catches
  **30 of 38** -- which is **not** the headline, because it includes the events the cut was
  chosen on -- and across all thirty the residual alone reached the cut on none of them
  (`docs/EVIDENCE.md`).
- **The threshold is derived from the mission's own data.** No target alarm rate is an input
  anywhere in the toolkit.
- **It warns only** -- no output ports of any kind. It *receives* one command,
  `RELOAD_MODEL`, which loads a model file a human has approved, and issues none.

## What is in this branch

```text
  flight/                     The C++14 inference core. No allocation after init, no
                              exceptions, no RTTI, no STL containers, -Werror
  fprime/Sentinel/Monitor/    The F' component: Monitor.fpp, its SDD, its unit tests
  fprime/Sentinel/Retrainer/  The retrainer component, exported OPT-IN and OFF by
                              default (D84). Host-verified, not flight-qualified
  fprime/SentinelRef/         Reference deployment, topology, and the physics testbed.
                              Apparatus, not product. Also ExampleAdapter/, the adapter
                              a mission copies
  fprime/SentinelRetrain/     The retrainer's own deployment and OS process, and the
                              only place an OCaml runtime exists. The pattern a mission
                              copies when it switches the retrainer on
  oxcaml/                     The retrainer's OxCaml sources and its C stubs
  src/                        The ground toolkit and its import closure. Reaches no bucket
                              and reads no credential (D80)
  scripts/                    The build scripts this branch's commands call
  docs/                       Design, evidence, status, the normative file format
  requirements-toolkit.txt    numpy, pyarrow, torch. Nothing else
```

## Quick start

**Every block starts from the repository root.**

The flight core -- a C++ toolchain and nothing else:

```bash
make -C flight test
LINT_ALLOW_PARTIAL=1 make -C flight lint    # PARTIAL until the F' checkout is built
```

The ground toolkit -- your healthy telemetry in, a `model.bin` out. **`--telemetry` takes a
`.npy` array, 2-D and finite, one row per timestep and one column per channel, at most 16
channels, and every row must be healthy.**

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-toolkit.txt
PYTHONPATH=src .venv/bin/python -m sentinel_toolkit selftest
```

Then, **with your own `.npy`** in place of `healthy.npy` -- these two need data this
repository does not ship:

```bash
PYTHONPATH=src .venv/bin/python -m sentinel_toolkit fit --telemetry healthy.npy --out model.bin
PYTHONPATH=src .venv/bin/python -m sentinel_toolkit verify --model model.bin
```

The F' half -- the framework checkout, then the detector's deployment:

```bash
scripts/fprime_setup.sh                     # F' v4.3.0 into the gitignored fprime/lib/
(cd fprime && source fprime-venv/bin/activate \
   && fprime-util generate -f && fprime-util build -p ./SentinelRef)
```

The retrainer -- **optional, and off unless you switch it on** (see
[OPTIONAL: the retrainer](#optional-the-retrainer)). **This assumes the toolkit and F' blocks
above have already been run.** `oxcaml_shape.sh` generates the training cycle at
`SentinelRef`'s shape -- 8 channels, 10 predictions -- runs its gates against the generated
source, and builds the object the deployment links. The `generate -f` is **not optional**: the
option is read at generate time, and without it `SentinelRetrain` skips itself.
`oxcaml_setup.sh` compiles a compiler: measured at **373.52 s** and **2.7 GiB** on disk on the
development host, well inside its pre-registered ceiling.

```bash
scripts/oxcaml_setup.sh
bash scripts/oxcaml_shape.sh --channels 8 --predictions 10
(cd fprime && source fprime-venv/bin/activate \
   && fprime-util generate -f -DSENTINEL_WITH_RETRAINER=ON -DSENTINEL_RETRAINER_CHANNELS=8 \
   && fprime-util build -p ./SentinelRetrain)
```

What `make -C flight test` proves, on this branch:

| Check | What it shows |
|---|---|
| footprint | `sizeof(Detector)` checked by a test at run time, exactly 603,032 B |
| refusals | 18 load cases, exercising all 12 refusal codes plus the accept path |
| determinism | bit-identical within a process and across two |
| vectors | every tier inside the tolerance the tests pin -- but of the **3** golden tiers the committed manifest describes, only **2** ship a model file here; that tier and the production-shaped ones regenerate from weights committed nowhere, and skip |

## Use it in your F' project

`scripts/fprime_ref_patch.sh` performs exactly these four edits against F' v4.3.0's own
`Ref` deployment, builds it, asserts `Sentinel::Monitor::schedIn_handler` is in the binary,
and reverts. Nothing is copied.

| # | File | What the script adds |
|---|---|---|
| 1 | `settings.ini` | `library_locations: ./path/to/fprime-DeepLearning-Sentinel/fprime` |
| 2 | `Top/instances.fpp` | `instance sentinelMonitor: Sentinel.Monitor base id 0x20000000 queue size 10` |
| 3 | `Top/topology.fpp` | `instance sentinelMonitor` in the instance list, and `rateGroup1Comp.RateGroupMemberOut[8] -> sentinelMonitor.schedIn` |
| 4 | `Top/RefPackets.fppi` | `packet Sentinel id 100 group 2 { ... }` carrying the component's five telemetry channels |

**Two more steps that the Ref proof does not need and your mission does.**

First, in `configureTopology()`, give the component a model and a channel count:

```cpp
sentinelMonitor.configure("SentinelModel.bin", YOUR_CHANNEL_COUNT);
(void)sentinelMonitor.loadModel();
```

Second, write the one piece that is yours: an **adapter** converting your typed telemetry
into one `Sentinel.ChannelVector` per tick -- the `F32` array of channel values -- and
emitting it on a `Sentinel.ChannelSample` port, which carries that array and a validity
flag. Put it on a `sync` port at a lower rate-group index than the detector's.

Copy `fprime/SentinelRef/ExampleAdapter/`: **176 lines across three files**, of which **65**
are the implementation and **46** are neither blank nor a comment -- which is the "about
forty lines" the other documents mean. `fprime/README.md` and `docs/DESIGN.md` 8 carry the
recipe in full.

- **By default, a mission inherits the Monitor and the inference core and nothing else.**
  `fprime/library.cmake` exports `Sentinel/Monitor` and `sentinel_core` -- not the testbed,
  and no OCaml runtime. The retrainer is exported too, but only if you switch it on: the
  next section.
- **Make the model with the ground toolkit**, from your own healthy telemetry:
  `PYTHONPATH=src .venv/bin/python -m sentinel_toolkit fit --telemetry healthy.npy --out SentinelModel.bin`.
  `--telemetry` takes a **`.npy` array, 2-D and finite, one row per timestep and one column
  per channel**, at most **16** channels -- the maximum the component is compiled for -- and
  every row must be healthy, because the model learns what healthy looks like.
- **Minimum data: 6,550 contiguous healthy timesteps.** Fewer is refused, and a refusal is
  the correct output.
- Two loadable examples sit at `flight/test/vectors/p1.bin` and `p2.bin`. A missing or
  refused model is not fatal: the component degrades to its Level 1 statistical baseline,
  names the refusal code in an event, and keeps ticking.

### OPTIONAL: the retrainer

**Off unless you switch it on, and a mission that does not switch it on sees no change** --
no OxCaml toolchain, no OCaml runtime, and the same Monitor code (`docs/DECISIONS.md` D84, on
`dev`). **Status: the chosen retraining implementation, host-verified, not flight-qualified**
(D83). Read the case against, at the end of this section, before you switch it on.

**What you get, and what you do not.** A background process, in **its own deployment**, that
retrains a candidate model at **your** channel count and writes it as a model file your
detector can load. **Since D85 it trains on the detector's own telemetry**: a tap beside the
Monitor copies every vector over the hub to it, and it trains only on recent healthy windows,
never within a guard band of a detector crossing. What you adopt is the pipeline: the tap, the
shape, the training cycle, the candidate file, the separate process and the ground gate. **No
candidate may be swapped in** (D85.1): on a simulated plant that did not change, the ground
gate certified retrained candidates anyway, so the gate is not usable yet, and a human's
`RELOAD_MODEL` is the only way a model changes.

**1. The setting**, in your project's `settings.ini`:

```ini
[fprime]
library_locations: ./path/to/fprime-DeepLearning-Sentinel/fprime
default_cmake_options: SENTINEL_WITH_RETRAINER=ON
    SENTINEL_RETRAINER_CHANNELS=8
    SENTINEL_RETRAINER_PREDICTIONS=10
```

Use your own channel count, 1 to 16, and prediction count, 1 to 10. Hidden `[80, 80]` and
window 250 are fixed. `-DSENTINEL_WITH_RETRAINER=ON` on `fprime-util generate` does the same.

**2. The toolchain and your shape**, from this repository's root. `oxcaml_shape.sh` generates
the training cycle at your shape and **re-runs its gates against the generated source** --
`[@zero_alloc strict]` with zero `assume`, a deliberate allocation rejected, an out-of-range
read raising, and a candidate that `flight/`'s reader loads. `EX1=1` in front of it adds the
exhaustive gradient check, which is ten local processes for several minutes.

```bash
scripts/oxcaml_setup.sh
bash scripts/oxcaml_shape.sh --channels 8 --predictions 10
```

**3. A second deployment.** The retrainer never runs in the detector's process: OCaml 5's
collector stops the world across domains. Add `fprime/SentinelRetrain/` to your project with
`add_fprime_subdirectory(...)` after the library, or copy it; point its `configureShadow` at
your own flown model (`fprime/SentinelRetrain/Top/instances.fpp`).

**4. The hub link**, so its events and telemetry reach your ground system: a
`Svc.GenericHub` over `Drv.Udp` on each side -- `fprime/SentinelRef/Top/instances.fpp:85-143`
and `topology.fpp:75-105` are the detector's end, `fprime/SentinelRetrain/Top/topology.fpp:63-90`
the retrainer's. Over a stream transport the hub dropped every tick (`docs/DESIGN.md` 9).

**The proof, against F' v4.3.0's own `Ref`**: the setting, `SentinelRetrain` as a second
deployment, Ref asserted to carry the Monitor and **no** OCaml runtime, the second deployment
the runtime, and a standalone candidate that `flight/`'s reader loads. It reverts Ref after.

```bash
bash scripts/fprime_ref_retrainer.sh --channels 8 --predictions 10
```

**Where it can be switched on.** OxCaml supports x86-64 and arm64 Linux and arm64 macOS only;
switched on anywhere else -- or cross-compiling -- the build stops with one message naming
those. **This project has built it on arm64 macOS only.**

**The case against, stronger on every row measured by anybody.** No flight heritage; no
qualified compiler; no certification precedent for a garbage-collected runtime in flight;
64-bit Linux and arm64 macOS only, with no 32-bit ARM, no musl and no documented
cross-compile recipe; no stability promise in its own documentation. **Rust's footprint is
smaller**, with a qualified toolchain in Ferrocene and OPS-SAT heritage -- and **no Rust
comparison has been built here**, so on that row this project is reasoning rather than
measuring.

## The retrainer

**Optional, and off by default** ([above](#optional-the-retrainer)). A frozen model goes stale over a ten-year mission.
`SentinelRetrain` trains a candidate in **its own OS process**, downlinks it, and **a human
approves it** before `RELOAD_MODEL` changes anything. Nothing the detector does depends on
it.

**Why OxCaml, and it is now the chosen implementation rather than an experiment** (D83) --
**host-verified, not flight-qualified.** Training is thousands of lines of array
arithmetic, where memory bugs live, and flight rules forbid allocation after init. Mark a
function `[@zero_alloc strict]` and the build fails if anything in its call tree allocates:
the rule becomes a compile error rather than a review item. **56 functions carry it, with
zero `assume` escape hatches, and since D82 every array access in the retrainer is
bounds-checked as well** -- the two hold together, which was measured rather than assumed.

**The case against, stronger on every row measured by anybody.** No flight heritage, no
qualified compiler, no certification precedent for a garbage-collected runtime in flight;
64-bit Linux and arm64 macOS only; no stability promise in its own documentation. **Rust's
footprint is smaller**, with a qualified toolchain in Ferrocene and OPS-SAT heritage -- and
**no Rust comparison has been built here**, so on that row this project is reasoning rather
than measuring. Found rather than anticipated: the runtime's thread affinity is a deployment
constraint unit-test evidence cannot show.

**Its limits, all three:**

- The gate that would let a retrained model be *offered* for a swap is **not a usable gate
  yet**: one synthetic fixture, one seed, one drift shape (`docs/DESIGN.md` 9).
- **Host-verified only, never run on flight hardware.** The capability register marks the
  engine **HOST-VERIFIED PENDING TARGET** (E5), and two claims unverified: whether the
  retrainer's process disturbs the detector's timing (C2), and whether the toolchain builds
  for a flight target at all (C4).
- **It trains on the detector's own telemetry since D85, one step per admitted tick.** Run
  end to end through both deployments on the host (D85.2), its candidate was byte-identical to
  the host harness's and the ground gate refused it; the swap steps were printed, not sent.
  Until D84 it
  trained only at the compile-time maxima, **75,360 parameters**, and no candidate fitted
  `SentinelRef`'s narrower model. It is now generated at the mission's shape, so its candidate
  for `SentinelRef` has that model's shape and `RELOAD_MODEL` accepts it on the host -- which
  proves the shapes and the format meet, and says nothing about whether the candidate is any
  good.

**Building it from this branch has been run, not argued**: every block in this README that
needs no data of your own was run verbatim, in order, from a fresh clone of this branch on
2026-09-25 -- the opt-in blocks included (`docs/datasets/REPRODUCING.md`).

**The detector's own binary carries no OCaml runtime.** That is asserted by reading the
symbol table **whenever a `SentinelRef` binary has been built**; on a tree without one --
which is a fresh clone, and this machine today -- **the check skips rather than passes**, and
says so.

## Status

**Verified**

- The C++ core against its committed vectors, bit-deterministic within a process and across
  two. This runs on a fresh clone.
- The F' component built and its unit tests passed **when last run**, before the build trees
  were deleted on 2026-09-21. They need the F' toolchain and skip without it.
- For the ground toolkit, **3 of its 3 acceptance-ladder rungs have run** on `dev` -- a
  generated fixture, a single channel, and a real twelve-channel mission. **One mission, one
  split.**
- The retraining chain end to end, on a development host: candidate written, downlinked,
  approved, reloaded.

**Not yet verified**

- The retrainer on flight hardware.
- The sanity gate on data with seasonal structure.
- A drift magnitude that is not a sensor doubling its output.

**Owed**

- One Raspberry Pi session, which covers four things at once: **E4**, whether the retrainer
  builds and runs on the target; **Stage 41**, the flight envelope -- timing, memory and
  determinism on that board; and **C2** and **C4** above.
- Measurements on real mission data, for the gate and for a realistic drift.
- A licence.

**No early-warning claim is made about spacecraft telemetry** -- not "warns N minutes
before", not any wall-clock figure. The only warning time measured is the testbed's above:
a simulated plant and a fault this project designed, in ticks, fault-attributable only,
transferring to no mission.

## Documents

| Document | What it holds |
|---|---|
| `docs/DESIGN.md` | What it does, the rule it flies, the five safety rules, onboard retraining |
| `docs/EVIDENCE.md` | The result, the split, the alarm rate, the reproduction, the caveats |
| `docs/STATUS.md` | Where it is and what is next |
| `docs/FPRIME.md` | The F' v4.3.0 toolchain this is built against |
| `docs/MODEL_FILE.md` | **Normative** for the loader; where it and an implementation disagree, the document is right |
| `docs/OMISSIONS.md` | **Every path on `dev` that is not here, and why.** Every omission is named |
| `docs/PI_ENVELOPE.md` | Small-hardware envelope -- reserved, deliberately empty |
| `docs/datasets/` | The data, its licences, and the caveats that would corrupt a result |
| `fprime/README.md`, the SDDs | The layout, the adoption recipe, each component's design |

**Read in this order:** this file, `docs/EVIDENCE.md`, `docs/DESIGN.md`, `docs/STATUS.md`.

## Not on this branch

**Every omission is named**, in `docs/OMISSIONS.md` -- every path `dev` has and this branch
does not, what it is, and why. A curated branch that quietly drops things is worse than an
uncurated one, because a reader cannot tell what they are not seeing.

## Branches

`dev` carries the complete history with tags `wi1`-`wi9`; `master` is this branch and the
default. **Neither is ever rewritten and nothing is force-pushed.** **They share no commit**
(different root commits, `git merge-base` empty), which is why the guard on `dev` compares
trees, not ancestry. **`main` was retired 2026-09-14** -- every commit it held is reachable
from here, and a `/blob/main/...` link held elsewhere will 404. **Some commits on `dev` carry
a personal email address in their authorship; no commit on this branch does.** Removing it
from `dev` would invalidate every tag and Release for a cosmetic gain, so it is disclosed
rather than rewritten.

## Licence

**Not yet selected.** Intended for community release to the F' ecosystem. One obligation
binds regardless -- telemanom is BSD 3-Clause (Caltech/JPL 2018), third clause verbatim:

> *"Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory,
> nor the names of its contributors may be used to endorse or promote products derived from
> this software without specific prior written permission."*

**No document here presents this project as endorsed by, affiliated with, or produced by
Caltech or JPL.** Naming telemanom's authorship and citing the paper is description, not
endorsement. ESA-ADB carries a separate requirement, **CC BY 3.0 IGO**
(`docs/datasets/ESA_ADB.md`).

## Citation

- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. *Detecting
  Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
  arXiv:1802.04431. BSD 3-Clause.
- Kotowski, K. et al. *European Space Agency Benchmark for Anomaly Detection in Satellite
  Telemetry.* arXiv:2406.17826. Zenodo `15237121` v2, DOI 10.5281/zenodo.15237121,
  CC BY 3.0 IGO.
