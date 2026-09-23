# SentinelRetrain -- the retrainer's own process

**This deployment carries the OCaml runtime, and `SentinelRef` carries none of it.** That is
the whole reason it exists as a second deployment rather than four more instances in the
first one.

`docs/DECISIONS.md` D70 consequence 2: *"The retrainer is a separate OS process. That is a
requirement of this decision, not an implementation detail left to whoever builds it, and its
reason is the stop-the-world minor collector above."* OCaml 5's minor collector stops the
world across domains, so a retrainer sharing a process with the detector would stall the
detector at a collection barrier **even if the retrainer's own code allocated nothing**.

`docs/MODELS.md` 47.15, in its 47.15b rider, named this exact shape as the only thing that
could discharge C5 --
*"a **minimal retrainer deployment** -- its own `Top/`, a topology carrying the `Retrainer`
and a rate group and nothing else"* -- and `docs/MODELS.md` 65 is where it was built and C5 was
discharged.

## What is in it, and what is deliberately not

```
  retrainer          Retrain.Retrainer             0x30000000   passive
  rateGroup_1Hz      Svc.ActiveRateGroup           0x30001000   active
  chronoTime         Svc.ChronoTime                0x30010000   passive
  rateGroupDriver    Svc.RateGroupDriver           0x30011000   passive
  timer              Svc.LinuxTimer                0x30012000   passive
  textLogger         Svc.PassiveTextLogger         0x30013000   passive
  hub                Svc.GenericHub                0x30014000   passive
  hubAdapter         Drv.ByteStreamBufferAdapter   0x30015000   passive
  hubComm            Drv.Udp                       0x30016000   passive
  hubBufferManager   Svc.BufferManager             0x30017000   passive
```

Ten instances and **no subtopologies at all** -- no command dispatcher, no event manager, no
telemetry database, no CCSDS stack, no file handling. A deployment whose point is that it
carries nothing else has to be checked against that claim, not just described by it.

The last four are the hub (`docs/MODELS.md` 70), which carries this process's events and
telemetry to `SentinelRef` over `Drv.Udp`. **`Drv.Udp` and not `Drv.TcpClient`**, and 65.4
registered that route before the number that forced it: over TCP the crossing delivered
**0 of 256** ticks complete, over UDP **256 of 256** (71.3).

Base ids are `0xDSSCCxxx` with `D = 3`. `D = 2` is not available: `SentinelRef` already places
`sentinelMonitor` at `0x20000000` and `powerSim` at `0x21000000`, and the two dictionaries are
merged for the ground system, where a duplicated id is refused outright
(`Top/instances.fpp:9-13`).

The component itself lives at `fprime/SentinelRef/Retrainer/` and is registered by that
deployment (`fprime/SentinelRef/CMakeLists.txt:27`). This deployment only instances it. The
registration order in `fprime/CMakeLists.txt:29` and `:35` is load-bearing: an F' module may
be registered once, so `SentinelRef` must be added before this deployment's topology names
`Retrain.Retrainer`.

## (!) The OCaml runtime starts on the ticking thread, and moving it breaks the process

`Retrainer.cpp`'s `schedIn_handler` boots the runtime lazily on the first tick -- the handler
opens at `:187`, the reasoning is the comment at `:192-200` and the call is `:201-203` -- and
that comment says why:

> *"`caml_startup` grants the lock to its CALLER; an `ActiveRateGroup` runs this handler on
> its own task, so a runtime started in the topology's `configureTopology()` aborts the
> process with 'Fatal error: no domain lock held' on the first call in. Deferring the boot
> to the first tick makes the thread that owns the domain the same one that uses it, for the
> life of the process."*

`SentinelRetrainTopology.cpp` carries the mirrored note saying the runtime is deliberately
**not** started there. This was found by running the deployment (`docs/MODELS.md` 65); every
rung from 48 to 62 ran single-threaded in a unit-test process and could not have seen it.
`docs/DECISIONS.md` D70 consequence 9 is unchanged -- exactly one domain, and `Domain.spawn`
is never called. What moves is only *which* thread holds it.

## Building and running it

```
source fprime/fprime-venv/bin/activate
cd fprime
fprime-util generate -f
fprime-util build -p ./SentinelRetrain
```

This deployment **needs the OxCaml switch**, unlike `SentinelRef`. Without it the `Retrainer`
module skips itself and the deployment has nothing to instance.

**The OxCaml sources and the scripts that build the switch are on both branches** since
D80: `oxcaml/retrainer/` with `scripts/oxcaml_setup.sh` to build the switch and
`scripts/oxcaml_s61.sh` to build the object this deployment links. **The retrainer is still
an experiment rather than a released feature** -- it has never run on flight hardware,
its sanity gate is not a usable gate yet, and it trains at the compile-time maxima rather
than at `SentinelRef`'s flown shape. `docs/DESIGN.md`, on `master`, states the case for it
and the case against it in the same passage.

```
./SentinelRetrain -a 127.0.0.1 -p 0        # run the cycle with no hub
```

`-p 0` or no `-p` runs the cycle standalone; a non-zero port connects to a hub server.

## What it proves, and what it does not

**Proved** (`docs/MODELS.md` 65.6, 70, 71, 72): the binary builds through `fprime-util`, boots
the OCaml runtime inside a real F' deployment, runs one training cycle per tick, carries that
cycle across a hub into `SentinelRef`'s process, and **writes a candidate model file that
`flight/`'s own reader loads** (72.8, HO1). **0** OCaml symbols in `SentinelRef`'s binary and
**3,019** in this one, checked by symbol on every test run
(`tests/test_detector_binary_has_no_ocaml_runtime.py`).

**(!) The candidate is not a replacement for `SentinelRef`'s model.** The training cycle is
fixed at `Config.hpp`'s maxima -- 16 inputs, 75,360 parameters -- and `SentinelModel.bin` is an
8-channel model with 66,960 weights, so the shapes do not meet. `docs/MODELS.md` 72.4 measured
that and 72.9 carries it as owed.

**Not proved.** C2 -- that the separate process actually isolates the detector's timing -- is
UNVERIFIED and deferred to hardware: this host downclocks an idle core and the confound
exceeds the effect (`docs/MODELS.md` 62.6). C4 -- that the toolchain builds for the flight
target -- is UNVERIFIED and deferred to E4. **No timing figure should be quoted from this
deployment on this host.**
