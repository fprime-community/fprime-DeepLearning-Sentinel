# Sentinel::Monitor

A reusable F' component that watches the relationships between telemetry
channels and warns before a limit trips. It warns; it never commands.

The detection method is Hundman et al., KDD 2018 (JPL's telemanom). The
arithmetic lives in `flight/`, a freestanding C++14 core verified against a NumPy
reference at 1.788e-07 (`docs/MODELS.md` 19.8); this component is the F' wrapper
around it.

## 1. Requirements this discharges

| Requirement | Where it comes from |
|---|---|
| An F' component with ports, telemetry and a warning event naming the channel | `docs/STATUS.md` 7 |
| Level 1: a corrupt file, CRC or version mismatch degrades to the statistical baseline with an event, never failing the topology | D5, Objective.md 14.10 |
| Warn-only; commands nothing | Objective.md 11 rule 3 |
| Silent until validated | Objective.md 11 rule 2 |
| Deterministic: fixed memory, fixed compute per cycle, same inputs give same outputs | Objective.md 11 rule 5 |
| Every warning explainable: named channel and score | Objective.md 11 rule 4 |

## 2. Design

**Passive, driven by a synchronous `Svc.Sched`** (D32). F's own
`component-and-port-selection.md` names this the model for cyclic work: the tick
runs in the rate group's context, so a slip is visible as a slip rather than
hidden on another thread. Work item 10's reload command makes the component
`queued`, which is a one-word FPP change plus a queue dispatch at the top of the
`schedIn` handler.

**Channel values arrive by direct port wiring** (D33), as one
`ChannelVector` of `F32` per tick, in the model's channel order. The alternative
-- tapping the telemetry path with an `Fw.Tlm` input, as `Svc::TlmPacketizer`
does -- was declined on evidence: `Fw.Tlm` carries a serialized `Fw::TlmBuffer`,
and `model.bin`'s CHANNELS record carries an id and a name and no type tag, so a
tap cannot deserialize a value without knowing its declared type. A mission
supplies a small adapter, which is F's own Passive Adapter Pattern.

**The Level 1 baseline's constants are parameters, not model-file fields**
(D34). `BASELINE_SCALE` and `BASELINE_THRESHOLD` are PrmDb-style parameters with
FPP defaults, because the refusal Level 1 exists to survive is `BAD_HEADER_CRC`,
and nothing in a file with a bad header can be trusted. With no parameter
database at all the FPP defaults apply and the baseline still runs.

**The model file is read whole** (D36), into a 302,048-byte member buffer sized
from the compile-time maxima, so the read is bounded before the header is
trusted. The chunked reader `docs/MODEL_FILE.md` 8 anticipates is deferred, and
the loader's check order is untouched so it stays droppable-in.

## 3. Ports and commands

| Port | Kind | Type |
|---|---|---|
| `schedIn` | `sync input` | `Svc.Sched` |
| `channelsIn` | `sync input` | `Sentinel.ChannelSample` |
| `cmdIn`, `cmdRegOut`, `cmdResponseOut` | command | the autocoded `PARAM_SET` and `PARAM_SAVE` protocol, and since work item 10 `RELOAD_MODEL` below |
| `timeCaller`, `eventOut`, `textEventOut`, `tlmOut`, `prmGetOut`, `prmSetOut` | special | |

| Command | Kind | Opcode | Argument | What it does |
|---|---|---|---|---|
| `RELOAD_MODEL` | `async` | `0x10` | `modelPath: string size 40` | Loads the named model file, refusing and rolling back to the previous one on any loader refusal or a channel-width mismatch |

**It issues no command and has no commanding port of any kind. Since work item 10
it *receives* one** -- `RELOAD_MODEL`, which loads a model file a human has
approved. `Objective.md` 11 rule 3 is about what a component *issues*; receiving
one is rule 1's other half, that retraining is explicit and human-approved. The
component therefore declares exactly one command and sends none.

**Why the command is `async`, and why that makes the component `queued`.** An
`async` command is queued rather than run in the caller's thread, and F' requires
a `queued` or `active` component to hold one. Section 2 takes the `queued` route:
the queue is drained at the top of `schedIn_handler`, bounded by
`DISPATCH_DEPTH`, so the cyclic work still runs on the rate group's thread and a
burst of commands cannot make one tick unbounded.

**The opcode is `0x10`, not `0x0`.** The two parameters above autocode four
commands -- a SET and a SAVE each -- and FPP assigns those from 0, so `0x0`
through `0x3` are already spent. `0x10` leaves a mission room to add its own
below it.

**Topology assumption, stated so it can be checked.** The producer of
`channelsIn` runs on the same rate group at a lower port index, so both `sync`
ports execute on one thread and no mutex is needed. A mission wiring a producer
on another thread makes `channelsIn` `guarded`. Parameter access is already
guarded by the framework: the generated base class holds an `Os::Mutex` for
parameter sets and gets.

## 4. Telemetry and events

| Channel | Type | Meaning |
|---|---|---|
| `Score` | `F32` | the maximum across channels this tick |
| `Threshold` | `F64` | the active cut |
| `ActiveMode` | `Mode` | MODEL or BASELINE -- the active-tier channel D5 requires |
| `TicksSinceWarmup` | `U32` | 0 while warming |
| `LoadStatus` | `ModelLoadStatus` | the loader's verdict |

| Event | Severity |
|---|---|
| `ModelLoaded` | activity high |
| `ModelRefused` | warning high |
| `DegradedToBaseline` | warning high |
| `CrossChannelWarning` | warning high, throttled at 10 |
| `WarmupComplete` | activity high |
| `WarningThrottleCleared` | activity low |

`ModelRefused` carries the refusal code as an argument rather than being eleven
near-identical events, which would be eleven places for the text to drift. The
eleven codes are `flight/include/sentinel/Status.hpp`'s, mirrored into the FPP
enum `ModelLoadStatus` and pinned to it by `static_assert`.

## 5. Level 1, precisely

```
  load the file
    |
    +-- no file, or unreadable      -> BASELINE, DegradedToBaseline(NO_MODEL_FILE)
    +-- loader refuses (11 codes)   -> BASELINE, ModelRefused(code)
    |                                            + DegradedToBaseline(MODEL_REFUSED)
    +-- loads, baseline_only = 1    -> BASELINE, ModelLoaded
    |                                            + DegradedToBaseline(BASELINE_ONLY_SET)
    +-- loads, baseline_only = 0    -> MODEL,    ModelLoaded
```

In every branch the component returns, ticks, and emits telemetry. It never
asserts on file contents -- CPP-4 forbids `FW_ASSERT` on off-device data and
`model.bin` is uplinked -- never throws, and never fails the topology.

**The baseline is what runs before anything is loaded**, not a fallback entered
later, so there is no window in which the component is neither.

**A degraded component speaks sooner than a healthy one.** The baseline's warm-up
is its 120-sample window; the model's is 2,350 (`window` 250 plus `error_window`
2,100). So Level 1 begins warning 2,230 ticks earlier. That is a property of the
two detectors, not a defect, and it is recorded because an operator seeing
warnings sooner after a degradation should know why.

## 6. What this component does not do

- **It does not warn early.** The frozen decision layer's measured median lead is
  +0.0: it fires at the labelled event boundary (D25, `docs/PHASE2.md` 4). The
  break-to-limit-trip lead is Phase 3's measurement on a real clock.
- **It does not name a relationship.** Objective.md 4.2 shows an event naming a
  decoupled *pair*; no pair map, lag map or covariance exists anywhere in the
  project (Objective.md 4.4). `CrossChannelWarning` names the single channel
  whose smoothed residual took the maximum, which is what exists.
- **It does not reload.** Work item 10 owns the uplink path and the human
  approval around it. No code here implements it.
- **It does not learn.** The model is frozen in flight (Objective.md 11 rule 1).

## 7. The baseline diverges from the harness, deliberately

The Level 1 baseline transcribes the `rstd` rule, in F64, pinned against
`numpy.nanstd`. It does **not** reproduce `baselines._rolling`, which accumulates
its prefix sums in float32 and loses the statistic -- 7.6584e+00 of error on a
true sigma of 3.0. So the flight baseline and the harness's `rstd` are knowingly
different numbers until the harness is repaired. D37 records why;
`docs/MODELS.md` 21 scopes the repair.
