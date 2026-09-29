# SentinelRetrain -- the retrainer's own process

**This deployment carries the OCaml runtime, and `SentinelRef` carries none of it.** That is
the whole reason it exists as a second deployment rather than four more instances in the
first one -- and it is the pattern a mission copies when it switches the retrainer on
(`docs/DECISIONS.md` D84: exported **opt-in**, OFF by default).

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
  retrainer          Sentinel.Retrainer            0x30000000   passive
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

The component itself lives at `fprime/Sentinel/Retrainer/` and is registered by the library
when `SENTINEL_WITH_RETRAINER` is ON (`fprime/library.cmake`, D84). This deployment only
instances it, and skips itself when the option is OFF. The order in `fprime/CMakeLists.txt` is
load-bearing: an F' module may be registered once, so the library is included before this
deployment's topology names `Sentinel.Retrainer`. Until D84 the component lived at
`fprime/SentinelRef/Retrainer/` as FPP module `Retrain`, which is what the records cite.

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
scripts/oxcaml_setup.sh
bash scripts/oxcaml_shape.sh --channels 8 --predictions 10
source fprime/fprime-venv/bin/activate
cd fprime
fprime-util generate -f -DSENTINEL_WITH_RETRAINER=ON -DSENTINEL_RETRAINER_CHANNELS=8
fprime-util build -p ./SentinelRetrain
```

This deployment **needs the OxCaml switch and the generated shape**, unlike `SentinelRef`.
`oxcaml_shape.sh` generates the cycle at 8 channels and 10 predictions -- `SentinelRef`'s shape
-- runs its gates against the generated source and builds the object this deployment links.
With the option OFF this deployment skips itself; ON without the shape, the build stops and
names the command.

**The OxCaml sources and the scripts that build the switch are on both branches** since
D80: `oxcaml/retrainer/` with `scripts/oxcaml_setup.sh` to build the switch and
`scripts/oxcaml_shape.sh` to build the object at a mission's shape. **OxCaml is the chosen
retraining implementation, host-verified, not flight-qualified** (D83). Chosen is not
qualified: it has never run on flight hardware, and its ground gate is not a usable gate yet --
it certified candidates on a plant that did not change, so no candidate may be swapped in
(D85.1). Since D85 it trains on the detector's own telemetry. Since D84 it is generated at the
mission's shape, so a candidate for `SentinelRef`'s 8 channels is what it builds; that makes
the shapes meet, not the candidate good. **The case against, beside it:** no flight heritage,
no qualified compiler, no certification precedent for a garbage-collected runtime in flight,
x86-64 and arm64 Linux and arm64 macOS only, no stability promise, and Rust's footprint is
smaller with Ferrocene qualified -- and **no Rust comparison has been built here**.

```
./SentinelRetrain -a 127.0.0.1 -p 0        # run the cycle with no hub
```

`-p 0` or no `-p` runs the cycle standalone; a non-zero port connects to a hub server. `-t US`
shortens the base tick for a host run and `-E` logs LC2's replica emit ticks (both apparatus;
`scripts/d85_e1.sh` uses them, `docs/MODELS.md` 78.11).

## What it proves, and what it does not

**Proved** (`docs/MODELS.md` 65.6, 70, 71, 72): the binary builds through `fprime-util`, boots
the OCaml runtime inside a real F' deployment, runs one training cycle per tick, carries that
cycle across a hub into `SentinelRef`'s process, and **writes a candidate model file that
`flight/`'s own reader loads** (72.8, HO1). **0** OCaml symbols in `SentinelRef`'s binary and
**3,149** in this one (3,082 until D85 added the window gate), checked by symbol on every test run
(`tests/test_detector_binary_has_no_ocaml_runtime.py`).

**(!) The shapes meet since D84, and that is all that follows.** Until D84 the training cycle
was fixed at `Config.hpp`'s maxima -- 16 inputs, 75,360 parameters -- against `SentinelModel.bin`'s
8 channels and 66,960 weights, so no candidate could be loaded in its place (`docs/MODELS.md`
72.4). Generated at 8 channels and 10 predictions, the cycle's candidate has that file's shape,
and `RELOAD_MODEL` accepts it on the host (`docs/MODELS.md` 77). **Since D85 it trains on the
detector's own telemetry, the ground gate judges it, and no candidate may be swapped in
(D85.1).** Run end to end through both deployments (`docs/MODELS.md` 78.12, D85.2), its
candidate is byte-identical to the host harness's, and the gate refused it.

**Not proved.** C2 -- that the separate process actually isolates the detector's timing -- is
UNVERIFIED and deferred to hardware: this host downclocks an idle core and the confound
exceeds the effect (`docs/MODELS.md` 62.6). C4 -- that the toolchain builds for the flight
target -- is UNVERIFIED and deferred to E4. **No timing figure should be quoted from this
deployment on this host.**
