# Status

> **Paths outside this branch resolve on `dev`** at commit **`01e9492`** (`docs/DECISIONS.md`
> D69, on `dev`). The guards that keep these figures true run on `dev`, not here.

**Current state and what is next. No history** -- the chronology, the retractions and the
reasoning are on `dev` in `docs/NARRATIVE.md` and `docs/DECISIONS.md`.

## 1. In one line

**The flight half is built, green and adopted. The ground half is not written, and without
it no mission can use this.**

## 2. What is done

| | State |
|---|---|
| **The C++ inference core** | Built and green. GRU forward pass, telemanom's dynamic threshold, the trailing-standardised derivative stream, the `model.bin` reader. Held to the NumPy reference at **1e-05** on every stream with the emission flag exact; determinism bit-identical in-process and across processes, over a run that wraps the window |
| **The decision rule** | **Adopted 2026-09-11.** `max(z_residual, z_derivative)` against one calibrated cut, on a `param_version` 2 parameter block. A version-1 file still gets the earlier rule, so every previously committed vector still passes |
| **The file format** | `format_version` **1, frozen**. `docs/MODEL_FILE.md` is normative. Both readers refuse an unknown `param_version` |
| **The F' component** | `Sentinel::Monitor` builds in F' v4.3.0's own Ref. Warn-only by interface. Its unit tests pass, covering **12 of the 12** refusal codes degrading to Level 1 without failing the topology |
| **Level 1, the safe failure mode** | Built and wired to the active-tier telemetry channel |
| **The evidence** | `docs/EVIDENCE.md`. One arm, one dataset, UNDERPOWERED, and the caveats are stated with the number |

## 3. (!) What is not done, and it is the blocking one

**The ground training toolkit is not released, and this branch does not carry it.**
**3 of its 3 acceptance-ladder rungs have run on `dev`** (`docs/MODELS.md` 40.14, on `dev`,
2026-09-11): a generated fixture, a single channel, and **a real twelve-channel mission**.
It trains, calibrates label-free, writes a `param_version` 2 `model.bin` and reads it back
through the same loader this branch ships.

On the real mission it took **73.5 s and 2.97 GiB** -- so the compute gate it was held
behind is discharged on measurement rather than waived -- and its held-out alarm rate came
out at **0.0811% against a calibration half of 0.1002%, a ratio of 0.81x**, which is the
first evidence outside a fixture that its operating point survives a split of a mission's
own healthy data. **One mission and one split.** Seven of its eight pre-registered
predictions held; the accuracy prediction is **NOT ADJUDICATED** and is reported as unrun
rather than as the half that passed.

**(!) One number from that run is worth carrying off this page.** The mission archive is
14,728,316 timesteps and the longest contiguous healthy run in it is **1,138,952 -- 7.7%**.
An archive is not a training set, and a mission planning to use this should size its
expectations against the healthy fraction rather than the total.

**None of that changes what you can do with this branch.** A mission holding it has a
component that can run a `model.bin` and no way to produce one, except by implementing
`docs/MODEL_FILE.md` independently -- which the document is complete enough to support and
which is the honest answer available today.

What the pre-registration commits to, so it can be held to it:

- One command: a mission's healthy telemetry in, a `model.bin` and a **label-free
  calibration report** out.
- **The cut is derived, and the curve is reported.** No target alarm rate is an input and no
  operating point is selected off a curve. The threshold is a noise floor measured on the
  mission's own nominal data; the report states the rate that cut produces on **held-out
  healthy** data, reported and never targeted, with the rate-against-threshold curve around
  it as a sensitivity measurement, selected at no value.
- **Dimensionless throughout.** A constant in the units of the data is a property of the
  dataset it was fitted on; every one the toolkit ships has a dimensionless equivalent, and
  any absolute form is a per-mission override with its provenance attached.
- Refusals that name their reason and exit non-zero: history too short, non-finite values,
  too many channels, a calibration window too short for the estimator.
- Verified by round trip through this branch's own loader before the file is called written.
- **No full-mission training run is started** until a single-channel wall clock is measured
  and the options are costed.

## 4. What is next, in order

| | Item | Done when |
|---|---|---|
| **A** | **The ground toolkit** | It runs on a synthetic fixture at zero cloud operations, prints the calibration report, offers the tier ladder, and a newcomer with no ML background produces a `model.bin` in under an hour |
| **B** | **The F' Ref physics testbed** | Coupled current, heat, temperature and voltage; 8-12 channels; real dictionary limits; a real clock; faults seeded **in the physics** and in-limits throughout. A model gate runs on it at a matched rate reporting in-limits catch rate, **time-to-limit-trip**, and manoeuvre false alarms |
| **C** | **In-orbit threshold recalibration** | File uplink and human-approved reload, exercised end to end on the Ref. The format already permits it: the parameter block is separately CRC'd and separately replaceable, so no format change is needed |
| **D** | **The hardware envelope** | `docs/PI_ENVELOPE.md` carries a measurement instead of a reservation |
| **E** | **In-flight retraining of a shadow model** under human approval | The toolkit exists, C has measured something, and the shadow is measurably better before any swap is offered |

## 5. Open, and named

- **A float64 reference** for the dynamic threshold; without it the port's N2 prediction
  cannot be judged and is recorded as NO VERDICT.
- **Levels 2 and 3 of the tier ladder.** Level 2 cannot be built before the architecture
  gate; Level 3 waits on the toolkit.
- **Time-to-limit-trip has never been measured**, because the data this project holds has no
  dictionary limits and an anonymised clock. It is item B's, on a real clock.
- Three improvement arms unrun and three predictions unadjudicated, named in
  `docs/EVIDENCE.md` section 9.

## 6. Release

**The licence is not yet selected** and the repository is private until it is. Intended for
community release to the F' ecosystem.

**This branch is the GitHub default**, as of 2026-09-14. The earlier snapshot branch
`main` was retired the same day: every commit it held is reachable from here and all nine
tags are on `dev`, so nothing was lost, but **a `/blob/main/...` link held outside this
repository will now 404**. No such link exists inside it, and one held elsewhere cannot be
searched for, so that is stated rather than dismissed.

One obligation binds whichever licence is chosen: **no document may present this project as
endorsed by, affiliated with or produced by Caltech or the Jet Propulsion Laboratory.** The
reference method is theirs and its source is BSD 3-Clause; clause 3 is quoted in full in
`README.md`.

## 7. Verify

```bash
make -C flight test      # the core against its committed vectors
make -C flight lint      # clang-tidy at -Werror
```

`README.md` says exactly what that green output covers on this branch and what it does not.
