# fprime-sentinel -- the F' library, and the two deployments that exercise it

**This directory is an F' library and a project that consumes it.** If you know
`fprime-community/fprime-sensors-reference`, you already know the shape; the map from its
vocabulary to this one is in the table below.

**What you adopt is one component.** `library.cmake:31` exports `Sentinel/Monitor` and nothing
else -- not the testbed, not the OxCaml retraining experiment, not the example code. It also
brings the plain-CMake inference core `sentinel_core` (`library.cmake:26`), which is stated
here rather than left to be found: you get it whether or not you knew it existed, and it is
what actually does the arithmetic.

## The layout, in `fprime-sensors-reference`'s terms

| here | that project calls it |
|---|---|
| `Sentinel/` | the library -- what a consumer adds to `library_locations` |
| `SentinelRef/` | `ReferenceDeployment/` -- one working topology |
| `SentinelRef/{ExampleAdapter,ExampleSource,PowerSim,Retrainer,HubCounter}/` | `Components/` |
| `CMakeLists.txt`, `library.cmake`, `settings.ini` | the same three files at its root |
| `lib/` | `lib/` -- the framework checkout |

**One real departure, and it is named rather than smoothed over.** That project vendors F' as a
git submodule. Here `lib/` is **gitignored** and rebuilt by `scripts/fprime_setup.sh` (on
`dev`), at the same pin -- **F' v4.3.0**, `docs/DECISIONS.md` D31. A fresh clone of this
repository has no framework in it until that script has run, and `settings.ini` says so in its
own header.

## Adopting the component: four edits

This is not a sketch. It is the executable form of `scripts/fprime_ref_patch.sh` (on `dev`),
which performs exactly these four edits against **F' v4.3.0's own `Ref` deployment** in a clean
checkout, builds it, asserts the symbol `Sentinel::Monitor::schedIn_handler` is in the binary,
and reverts. Nothing is copied.

**1. Point your project at the library** -- `settings.ini`:

```
[fprime]
library_locations: ./path/to/fprime-sentinel/fprime
```

**2. Instance the component** -- your `Top/instances.fpp`:

```
  instance sentinelMonitor: Sentinel.Monitor base id 0x20000000 \
    queue size 10
```

**(!) The queue size is required and the reason is one command.** The component is `queued`
(`docs/DECISIONS.md` D32 consequence 2) because it accepts `RELOAD_MODEL`, and the queue is
**drained at the top of the rate-group tick** rather than on a thread of its own -- so the
cyclic work still runs in the context of the rate group that ticks it, and a slip is still
visible as a slip. The drain is bounded, so a burst of commands cannot make one tick unbounded.

**3. Wire two input ports** -- your `Top/topology.fpp`:

```
  rateGroup.RateGroupMemberOut[n]     -> yourAdapter.schedIn
  rateGroup.RateGroupMemberOut[n + 1] -> sentinelMonitor.schedIn
  yourAdapter.channelOut              -> sentinelMonitor.channelsIn
```

Both of the component's inputs are `sync`. Put `channelsIn` at a **lower** port index than
`schedIn` on the same rate group, so the vector for this cycle lands before the detector steps.
A producer on another thread makes `channelsIn` `guarded` instead.

**4. Give it a model and a channel count**, in `configureTopology()`:

```
  sentinelMonitor.configure("SentinelModel.bin", YOUR_CHANNEL_COUNT);
  (void)sentinelMonitor.loadModel();
```

A missing or refused file is not fatal: the component degrades to its Level 1 statistical
baseline, names the refusal code in an event, and keeps ticking.

## The adoption chain, worked

**The component does not read your telemetry database**, and `Sentinel/Monitor/docs/sdd.md`
gives the reason: `Fw.Tlm` carries a serialized buffer and the model file has no per-channel
type tag to deserialize it with. So a mission converts its own typed telemetry into one
`Sentinel.ChannelVector` per tick, using F's Passive Adapter Pattern.

Two directories here are that chain, end to end, and both are **example code you copy rather
than link**:

```
  ExampleSource/     a deterministic synthetic feed -- NOT a sensor, and not a model
                     of one. It stands where your telemetry path stands.
  ExampleAdapter/    the conversion, at its smallest useful size. Last value wins;
                     emits one vector per tick. About forty lines.
```

**(!) Neither is instanced in `SentinelRef`, and that is deliberate.** This deployment already
feeds `sentinelMonitor.channelsIn` from `PowerSim/`, the physics testbed, and `channelsIn` is a
single `sync` port -- a second producer on it would make the testbed's recorded run
non-deterministic. So the two example components are registered, built and readable, and
nothing wires them. Build either by its own path; `fprime-util build -p ./SentinelRef` does
**not** build a registered-but-uninstanced module and exits 0 having built nothing, which reads
exactly like success.

## The two deployments

| | what it is |
|---|---|
| `SentinelRef/` | **the testbed.** The detector, a coupled power/thermal simulation with declared limits, and the ground link. **No OCaml runtime, asserted by symbol on every test run.** |
| `SentinelRetrain/` | **the retrainer's.** Its own OS process, and all of the OCaml runtime. It exists because a garbage collector must not be in the detector's process (`docs/DECISIONS.md` D70 consequence 2). |

Each has its own README with its instances, its base ids and what it proves.

## Which model file to load

**Two loadable examples are committed**, and they are real model files rather than fixtures
that resemble them: `../flight/test/vectors/p1.bin` (**3 channels**) and `p2.bin`
(**1 channel**), both at `param_version` 2, both re-emitted byte-identically by the round-trip
test. Loading one is the fastest way to see the component arm.

**(!) They are not what `SentinelRef` runs.** That deployment configures itself for the
testbed's **8** channels, so `p1.bin` in its place is refused `BAD_SHAPE` -- correctly, because
the weights only make sense at their own shape. A model file and a topology have to agree about
the channel count, and the loader is the thing that checks. Conflating the two is a reader's
first failed build.

## Building

```
source fprime-venv/bin/activate      # created by scripts/fprime_setup.sh, on dev
fprime-util generate -f
fprime-util build -p ./SentinelRef
fprime-util build -p ./SentinelRetrain
```

The OxCaml toolchain is needed only for `SentinelRetrain`. `Retrainer/` skips itself with a
message when the switch is absent, so a clone without it still builds the detector's
deployment, and `make -C ../flight test` and `lint` are untouched.

**(!) After changing any `.fpp`, run `fprime-util generate -f` before `fprime-util build`.** An
incremental reconfigure can leave the dictionary generator's import list empty, and F' v4.3.0's
`fpp_to_dict_wrapper.py` then fails with a `TypeError` about a bool. It is framework code, it
is gitignored here, and a full generate clears it.

## What is not here

**The framework itself.** `lib/` is gitignored, so every F' document this directory cites
resolves only once you have your own v4.3.0 checkout.

**The ground toolkit**, which trains a model from a mission's healthy telemetry and derives its
threshold without labels. It is `src/sentinel_toolkit/` on the `dev` branch, along with
`scripts/`, `tests/` and the research record. The customer branch carries the component and the
evidence it works, and nothing else -- `docs/DECISIONS.md` D69.

**Any claim that this has flown.** It has not. It has never run on flight hardware, and the
retraining experiment in `SentinelRetrain/` is an experiment: not adopted, not exported, and on
no path the detector takes.
