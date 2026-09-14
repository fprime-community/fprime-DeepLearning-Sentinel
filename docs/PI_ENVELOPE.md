# The Raspberry Pi envelope

> **(!) RESERVED AND EMPTY. There are no numbers on this page, and there will be none
> until something is measured.** The shape is fixed here so that a measurement has
> somewhere to land and so that nobody fills the gap with an estimate in the meantime.
> A figure in this document that is not a measurement is a defect.

## Why this document exists before its contents

`Objective.md` section 12 carries the Raspberry-Pi envelope as **Phase 4**, and the curated
branch's status document carries it as a roadmap row: whether the flight core runs inside a
representative small-computer budget, and with how much margin.
It is the first question anyone asks about a neural forecaster on a spacecraft, and the
temptation is to answer it from the numbers that already exist -- `sizeof(Detector)`, the
multiply-accumulate count, the flown weight size -- multiplied by a guess.

**That would be an estimate wearing a measurement's clothes**, and this project has a
record of what that costs: `docs/NARRATIVE.md` 6.9 records a figure written into a planning
mock-up and later read back as data. So the page exists, the shape is agreed, and it stays
empty.

**(!) This page cited `docs/STATUS.md` section 7 until 2026-09-14**, which carries no such
item on either branch. The roadmap moved and the reference did not -- the same drift this
document exists to refuse, arriving in the document itself.

## What is already known, and it is not an envelope

These are measured, and they are **inputs** to an envelope rather than one:

- **Static memory.** `sizeof(Detector)` is measured by `flight/test/Footprint.cpp` on every
  build and asserted against a pre-registered prediction. **No allocation happens after
  initialisation** (F' CPP-1), so the static figure is the whole figure.
- **Arithmetic per tick.** The forward pass's multiply-accumulate count is derived from the
  array shapes and re-derived by the same test. The decision layer adds a bounded amount on
  top, re-solved once per stride rather than once per tick.
- **Determinism.** `flight/test/DeterminismTest.cpp` shows the core is bit-identical across
  runs and across processes, which is what makes a timing measurement meaningful at all.

**None of that is a wall-clock number on real hardware**, and the difference between an
operation count and a millisecond is the entire point of this document.

## The shape a measurement must take

When it is taken, it fills in exactly this and nothing more:

| | |
|---|---|
| Hardware | The board, its revision, its clock, its RAM, and whether a heatsink or fan was fitted |
| Toolchain | Compiler and version, the exact flags, and whether they match `flight/Makefile`'s |
| Build | The commit, and `sizeof(Detector)` as that build reports it |
| Tick cost | Wall-clock per tick: median, and the **worst case over the run**, because a flight rate group is bounded by the worst case and not by the median |
| Stride cost | The tick on which the threshold re-solves is the expensive one. Reported **separately**, not averaged into the others |
| Memory | Resident set, against the static figure the build predicts |
| Thermal | Whether the number changes after the board has been running long enough to throttle |
| Margin | Against a stated rate-group period, with the period stated |

**Every figure in timesteps and milliseconds, never in "fast enough".**

## The rules this measurement inherits

- **Pre-register before running.** A numbered prediction with HOLD / NO VERDICT / FAIL
  bands, committed before the board is switched on, in `docs/MODELS.md` -- the same as every
  other measurement here.
- **The worst case is the result.** A median tick cost on a system with a rate group is a
  number that cannot be used.
- **Report the losers.** If it does not fit, that is the finding and it is published as one.
- **No estimate is published here**, and if the measurement cannot be taken, this page says
  so and stays empty.

## The measurement is pre-registered, and this page is still empty

**`docs/MODELS.md` 41, 2026-09-14.** Nine numbered predictions with HOLD / NO VERDICT / FAIL
bands, a stated falsification and five stop conditions, **written before a board was in hand**
and naming the board they are registered against. It covers the eight rows above and adds
training on the target, which `docs/reorg_plan.json`'s definition of done requires and the
table did not name.

**Scheduled, not blocked.** A board is to be obtained. **If a different one arrives, 41's
absolute bands are re-derived and committed as a rider before it is switched on**, never
after -- a band written once the hardware is known is not a pre-registration.

**Nothing on this page changes until the measurement is taken.** The rule at the top still
holds: a figure here that is not a measurement is a defect.
