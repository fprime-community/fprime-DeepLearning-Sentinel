# Status

> **Paths outside this branch resolve on `dev`** at commit **`9bc7205`** (`docs/DECISIONS.md`
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

## 3. (!) What is not done

**The ground training toolkit is on this branch since D80** -- `src/sentinel_toolkit/` with
its import closure, and `requirements-toolkit.txt`. **What is not done is the evidence
behind it, not its availability.**
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
| **E** | **In-flight retraining of a shadow model** under human approval | The toolkit exists, C has measured something, and the shadow is measurably better before any swap is offered. **The chosen retraining implementation, host-verified and not flight-qualified** (D83) -- `docs/DESIGN.md` 9. Exported from the library **opt-in and OFF by default** (D84), at a mission's own channel count. The engine runs in its own process and a cycle crosses to the detector; the sanity report that would let a swap be *offered* is **not yet a usable gate**, and that is the blocking item |

## 5. Open, and named

- **A float64 reference** for the dynamic threshold; without it the port's N2 prediction
  cannot be judged and is recorded as NO VERDICT.
- **Levels 2 and 3 of the tier ladder.** Level 2 cannot be built before the architecture
  gate; Level 3 waits on the toolkit.
- **Time-to-limit-trip has never been measured**, because the data this project holds has no
  dictionary limits and an anonymised clock. It is item B's, on a real clock.
- Three improvement arms unrun and three predictions unadjudicated, named in
  `docs/EVIDENCE.md` section 9.
- **The pre-launch sanity gate is unenforceable.** Its first term used to compare the candidate
  against the flying model, and that was measuring the retraining rather than the drift -- a
  warm-started shadow beat the flying model whether or not anything had changed. **The term has
  been replaced with a comparison against a control shadow trained on the flying model's own
  data, and the criterion now discriminates**: near zero when nothing has drifted, large when
  something has. **It discriminates on one synthetic fixture**, which is not a mission.
  `docs/DESIGN.md` 9.
- **Certification now needs two trained models, not one**, and which of the two routes the flown
  design takes -- the retrainer trains the control, or the ground reproduces it -- is not
  chosen. Neither is free.
- **The two terms of the criterion now compare against different baselines**, the first against
  the control and the second against the flying model. They disagree on the stationary case,
  which is refused either way; one decision is owed rather than an accident.
- **A drift magnitude that is operationally realistic**, rather than a sensor doubling its
  output. Unchanged.
- **Nothing onboard scores a candidate model.** The retraining process can write one, this
  branch's loader accepts it, the file downlink carries it to the ground and a command reloads
  an approved one -- but both parts of the sanity criterion are computed on the ground from
  that downlinked file. Onboard held-out scoring does not exist.
- **The file uplink half of that path has not been run.** The mechanism is in the reference
  deployment and the ground tool exists; the commanded reload was exercised against a file
  already on the host, because both processes run there by design. It is not built, rather
  than not working.
- **Whether refusing any width change on a commanded reload is the right rule.** It is what
  stops a model for a different subsystem being accepted, and it also means a mission that
  genuinely re-wires its channel set between reloads has no way to say so.
- **The candidate has the reference deployment's shape, and nothing more is known of it.**
  Since D84 the training cycle is generated at the mission's channel count, and at
  `SentinelRef`'s 8 channels `RELOAD_MODEL` accepts its candidate on the host. It is trained
  for one step on a deterministic drive, not telemetry; whether any candidate is better is
  exactly what the unusable gate above would have to say. `docs/DESIGN.md` 9.
- **The retraining engine is host-verified and has never run on flight hardware.** Whether
  its toolchain even builds for the target is unverified, and so is whether its separate
  process isolates the detector's timing. **No timing figure from that work should be
  quoted**; this host's power management downclocks an idle core and the confound exceeds
  the effect.

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
make -C flight lint      # clang-tidy at -Werror, three configs. PARTIAL and
                         # non-zero if any is missing; LINT_ALLOW_PARTIAL=1
                         # accepts that, which a clone of this branch needs
```

`README.md` says exactly what that green output covers on this branch and what it does not.
