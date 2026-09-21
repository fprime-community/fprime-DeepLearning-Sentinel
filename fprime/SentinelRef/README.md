# SentinelRef -- the reference deployment

**Apparatus, not product.** This deployment exists so the Sentinel component can be watched
doing its job against a simulated subsystem with declared limits. A mission adopting Sentinel
consumes `fprime/library.cmake`, which exports `Sentinel/Monitor` and nothing else
(`fprime/library.cmake:31`) -- not this deployment, not the testbed, and not the retrainer.

## (!) There are two deployments, and the second one is not optional

| Deployment | What runs in it | OCaml runtime |
|---|---|---|
| `SentinelRef` | the detector (`Sentinel.Monitor`), the physics testbed, the GDS link | **none, and it is asserted** |
| `SentinelRetrain` | the retrainer (`Retrain.Retrainer`) and a rate group, and nothing else | all of it |

**The split is a requirement, not a preference.** `docs/DECISIONS.md` D70 consequence 2:
*"The retrainer is a separate OS process. That is a requirement of this decision, not an
implementation detail left to whoever builds it, and its reason is the stop-the-world minor
collector above."* OCaml 5's minor collector stops the world across domains, so a retrainer
sharing a process with the detector would stall the detector at a collection barrier **even
if the retrainer's own code allocated nothing**.

That separation is asserted rather than trusted. `Retrainer/CMakeLists.txt:46` links the
OxCaml object **PUBLIC**, so anything that depends on the `Retrainer` module transitively
links a whole OCaml runtime, and `SentinelRef` is clean only because its topology never names
`Retrainer`. `tests/test_detector_binary_has_no_ocaml_runtime.py` counts the symbols in both
binaries on every test run, with a positive control that proves the check can see a runtime
when one is there.

## What is in this directory

| Path | What it is |
|---|---|
| `Top/` | the topology: 16 instances, the four subtopology imports, and the base-id convention |
| `PowerSim/` | the physics testbed -- coupled current, heat, temperature and voltage with declared limits, wired to the component's input port (`docs/MODELS.md` 42) |
| `Retrainer/` | **the OxCaml retrainer component.** It lives here and is registered here (`CMakeLists.txt:27`), but it is **instanced only by `SentinelRetrain`** |
| `ExampleAdapter/` | the forty-line Passive Adapter Pattern example a mission copies to wire its own channels in (`ExampleAdapter/ChannelAdapter.cpp`). The pattern itself is written up on `master` in its `docs/DESIGN.md` section 8 |
| `HubCounter/` | the tap that counts the hub crossing where the hub emits, so a delivery rate is not taken from the ground (`docs/MODELS.md` 70.6, 71). FPP module `HubTap` |
| `Main.cpp` | the deployment entry point |

`PowerSim/`, `Retrainer/`, `ExampleAdapter/` and `HubCounter/` are registered in this
deployment's own `CMakeLists.txt` rather than in `fprime/library.cmake`, and the reason is the
same for all four: that file exports what a mission adopting Sentinel consumes, and a simulated
battery, an OCaml runtime, example code and a test instrument are none of them.

## The instances, and the base-id convention

Base ids follow `0xDSSCCxxx` -- deployment digit, subtopology digits, component digits, then
room for the component's own events, commands and telemetry (`Top/instances.fpp:7-13`). This
deployment is `D = 1`; `SentinelRetrain` is `D = 3`, because `D = 2` is already spent on this
deployment's own `sentinelMonitor` and `powerSim` and the dictionary merge refuses a
collision.

```
  rateGroup_1Hz      Svc.ActiveRateGroup           0x10001000   active
  rateGroup_0_5Hz    Svc.ActiveRateGroup           0x10002000   active
  rateGroup_0_25Hz   Svc.ActiveRateGroup           0x10003000   active
  cmdSeq             Svc.CmdSequencer              0x10004000   active
  chronoTime         Svc.ChronoTime                0x10010000   passive
  rateGroupDriver    Svc.RateGroupDriver           0x10011000   passive
  systemResources    Svc.SystemResources           0x10012000   passive
  timer              Svc.LinuxTimer                0x10013000   passive
  comDriver          Drv.TcpClient                 0x10014000   passive
  hub                Svc.GenericHub                0x10015000   passive
  hubAdapter         Drv.ByteStreamBufferAdapter   0x10016000   passive
  hubServer          Drv.Udp                       0x10017000   passive
  hubBufferManager   Svc.BufferManager             0x10018000   passive
  hubCounter         HubTap.HubCounter             0x10019000   passive
  sentinelMonitor    Sentinel.Monitor              0x20000000   passive
  powerSim           Testbed.PowerSim              0x21000000   passive
```

The five hub instances carry `SentinelRetrain`'s events and telemetry into this process
(`docs/MODELS.md` 70, 71). **`Drv.Udp` and not `Drv.TcpServer`**: `GenericHub.cpp:129-130`
dispatches on an exact size match and `:249-252` drops anything else silently, so a stream
transport that coalesces two messages into one read loses both. Measured, over 256 ticks:
**TCP 0 complete, UDP 256** (71.3). `hubCounter` sits between the hub and `CdhCore` and counts
what the hub emits, because the ground is downstream of `Svc.TlmChan` and cannot measure the
crossing (70.6).

`powerSim` is **passive on purpose** and `sentinelMonitor` is **queued** with both its input
ports still `sync`, which comes to the same thing for the cyclic work: each runs in the context
of the rate group that ticks it, so a slip shows up as a slip rather than being absorbed by a
queue, and the plant advances in lockstep with the detector watching it -- which is what lets a
warning instant and a limit instant be compared at all.

`sentinelMonitor`'s queue exists for one thing: **`RELOAD_MODEL`, the only command the
component has ever declared** (`docs/MODELS.md` 73, D32 consequence 2). It is drained at the
top of the tick, bounded by the queue size, so a burst of commands cannot make one tick
unbounded. The component still **issues** no command and has no commanding port of any kind --
`Objective.md` 11 rule 3 is about what it issues -- and receiving one is rule 1's other half,
that retraining is explicit and human-approved.

Four F' core subtopologies are imported and supply everything else: `CdhCore` (command
dispatch, events, text logging, fatal handling), `ComCcsds` (the CCSDS uplink and downlink),
`FileHandling` (file uplink, file downlink, file manager, parameter database) and
`DataProducts`.

## Building it

```
source fprime/fprime-venv/bin/activate
cd fprime
fprime-util generate -f
fprime-util build -p ./SentinelRef
```

**(!) `fprime-util build -p <deployment>` does not build a registered-but-uninstanced
module.** It exits 0 having built nothing, which reads exactly like success. `Retrainer/` is
registered by this deployment but instanced by the other one, so building it through
`SentinelRef` builds nothing at all -- build it by its own path, or build `SentinelRetrain`.

The OxCaml toolchain is not needed here. `Retrainer/` skips itself with a message when the
switch is absent, so a clone without it still builds this deployment and `make -C flight test`
and `make -C flight lint` are untouched.

## Running it with the F' GDS

```
cd fprime/SentinelRef
fprime-gds                 # ground system and the app together
fprime-gds --no-app        # ground system only
```

The binary can also be run on its own from `build-artifacts/<platform>/bin/`:

```
./SentinelRef -a 127.0.0.1 -p 50000
```

## Where the reasoning is

The decisions are on `dev` in `docs/DECISIONS.md` -- D32 for the component's passive shape
and its refusal behaviour, D70 for why OxCaml is a candidate and what is registered against
it, D73 for the fixed step budget. The pre-registrations and what they measured are in
`docs/MODELS.md`: 42 for the testbed, 47 for the OxCaml gateway, 65 for the second deployment.
