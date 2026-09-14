# PowerSim -- the physics testbed

A coupled power and thermal subsystem, simulated, deterministic and seeded, with
limits declared in its own dictionary. It exists so that a **warning time** can be
measured against a **real limit on a real clock**, which no dataset this project
holds can provide: SMAP/MSL ships no limit definitions and ESA-ADB's clock is
anonymised. Pre-registered at `docs/MODELS.md` 42, on `dev`.

**This is apparatus, not product.** `fprime/library.cmake` exports
`Sentinel/Monitor` and nothing here, so a mission adopting the component does not
inherit a simulated battery.

## What it publishes

Eight coupled channels, in the model's channel order: `SolarInput`,
`ChargeCurrent`, `LoadCurrent`, `BusVoltage`, `CellTemp`, `RadiatorTemp`,
`HeaterDuty`, `StateOfCharge`. Each carries RED and YELLOW limits in
`PowerSim.fpp`.

## The fault touches one scalar

`RESISTANCE_RISE` ramps the cell's internal resistance and **nothing else**.
Every other channel moves because the physics couples it: ohmic heating rises with
`I^2 R`, the cell warms against a radiator, the bus sags under load. That is what
makes the anomaly **contextual by construction** rather than injected into a
channel, and it is why no annotation is needed to say when it began.

## (!) The solar array's low limits sit below zero, and a healthy run is what proved it

`SolarInput` was first declared with a **yellow low of 5 W**. It fired on **tick 0
of every run, healthy ones included**.

The array reads **0 W through eclipse, which is 35% of every orbit**. Low solar
power is not a fault -- **it is night**. Whether a low reading is anomalous depends
on the relationship between that channel and the orbit phase, and no fixed
threshold on the channel alone can hold that distinction: the same 0 W is correct
in shadow and catastrophic in sunlight.

**This is the project's own thesis arriving in its own instrument.**
`Objective.md` 2 argues that a per-channel limit check cannot see an anomaly whose
wrongness lies in a combination rather than in a value. The first channel this
testbed tried to limit-check turned out to be one that **cannot be limit-checked**,
and it took a run to see it -- not a review, not an estimate. The limits are now
set where a **sensor bias** would be (`red -5.0, yellow -2.0`) and nowhere near the
operating floor, because that is the only thing a limit on this channel can
honestly detect.

The measurement, for the record: on the seeded run, seed 1, fault injected at tick
**8,000**, the crossings are

```
  channel       first YELLOW    first RED
  SolarInput      tick 0  (with the 5 W yellow low; never, once corrected)
  CellTemp           19,225       20,140
  BusVoltage         20,700       21,757
  every other         never        never
```

## (!) Yellow is the bar, not red

`CellTemp` crosses **yellow at 19,225 and red at 20,140** -- a ground system with
yellow alarms sees this fault **915 ticks before the red trip**. A warning scored
against the red trip alone would credit Sentinel with beating a limit check it had
**not** beaten. Every lead figure this testbed produces is measured against the
**first crossing of any colour**, with the red trip reported beside it. Recorded at
`docs/MODELS.md` 42.8.1.

## Determinism, and why it is a requirement

A warning time is the gap between two instants in one run. If the run is not
reproducible the gap cannot be re-measured. `PowerPlant` uses a counter-based
noise mix rather than a stateful generator, so the sample at tick `t` does not
depend on how many samples were drawn before it, and a run is reproducible from
its seed and its parameters alone.

## The limits are evaluated twice, and the second copy is guarded

F' checks telemetry limits on the **ground**, from the dictionary
(`docs/reference/system-functional/telemetry-chan.md:32`), and exposes **no limit
constant to C++**. A limit trip stamped on the ground and a warning stamped
onboard sit on different clocks, and subtracting one from the other measures the
downlink rather than the detector. So `PowerSim.cpp` evaluates the same values
itself, against the onboard time base, and `PowerPlant.hpp`'s `PLANT_RED` is the
second copy that makes it possible.

**An unchecked second copy is what `docs/MODELS.md` 42.3 departure 1 forbids.**
`tests/test_powersim_limits.py` re-derives the table from `PowerSim.fpp` on every
run, fails in both directions, names the channel and both values, and is proved by
a probe.

## A component may not live in the deployment's own namespace

Declared in `module SentinelRef`, this component's autocoded
`cmdResponseOut_out(..., U32 cmdSeq, ...)` shadows the `cmdSeq` **instance** of
`Svc.CmdSequencer`, and F' builds at `-Wshadow -Werror`. It is `module Testbed`.
`Sentinel.Monitor` never hit this because it has its own module.

## What it does not do

- **It does not command anything.** Like the component it feeds, it is warn-only
  by interface; its three command ports are F's autocoded `PARAM_SET` and
  `PARAM_SAVE` plumbing and it declares no command of its own.
- **It is not a spacecraft.** Its time constants are chosen, so its seconds are
  not a mission's seconds. Figures are reported in **timesteps**, and the
  transferable quantity is the ratio of warning time to fault-onset-to-limit time.
- **It does not make an early-warning claim about spacecraft telemetry.** It
  measures sensitivity to a fault this project designed. `docs/MODELS.md` 37.7a's
  **0 of 10 positive leads** on SMAP/MSL stands beside every figure it produces.
