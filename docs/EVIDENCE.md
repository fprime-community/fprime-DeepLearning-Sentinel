# Evidence

> **Paths outside this branch resolve on `dev`** at commit **`664de7e`** (`docs/DECISIONS.md`
> D69, on `dev`). Every figure below names the `dev` decision or section it comes from and
> the run artifact it was read from. **Artifacts under `runs/` are cited by path and are not
> in git on any branch** -- the path is the identifier.

**No number in this document was written without an artifact to read it from.** Where a
figure cannot be checked from this branch, that is said.

## 1. The claim, and its caveats in the same breath

**Sentinel catches anomalies that stay inside a channel's limits, and the decision layer is
what makes the difference.** On NASA's SMAP/MSL telemetry, with a channel-disjoint split
fixed before any tuning:

```
  arm                                         alarm rate    TUNE     EVAL
  telemanom's published rule (frozen)           0.6820%     6/19     4/19
  residual fused with the first derivative      0.6820%    13/19    17/19
```

**(!) The component computes both terms, and the derivative is what decides.** On all thirty
events this arm catches -- **30 events across only 25 distinct onsets**, so the effective count
is lower -- `z_residual` reached the cut on **none of them**: median **0.080**
and a maximum of 4.652 against a cut of **5.288**, negative on 15 of the 30 -- and the
derivative alone reaches the same **EVAL 17 of 19** at the same alarm rate
(`docs/DECISIONS.md` D65.3, on `dev`, measured 2026-09-14). **The flown rule is unchanged and
still computes both**; what changes is that this finding, on this data, is the first
derivative.

**The headline is EVAL's 17 of 19**, at the frozen rule's own alarm rate. It is the clean
number: the cut was selected on TUNE, and EVAL is the half it was never selected on.
Source `docs/DECISIONS.md` D65 and `docs/MODELS.md` 38.15; artifacts
`runs/smap-msl/_forensics/2026-09-10T173015Z-arms.json` and
`runs/smap-msl/_forensics/2026-09-10T182234Z-arms2.json`, producer
`scripts/decision_layer_arms.py`.

**The caveats are not footnotes.**

- **UNDERPOWERED.** n = 19 per half. The harness's own rule stamps any denominator under 20,
  and this one is stamped wherever it is quoted, including here.
- **One arm, one dataset.** Not a survey. Not a benchmark result.
- **No floor to compare against at that alarm rate.** On this data no per-channel statistic
  reaches a flyable rate at all: a rolling standard deviation at **5,000x** its calibrated
  threshold still alarms on **15.17%** of nominal steps, and a range check on **5.84%**,
  against the forecaster's 0.68% (`docs/MODELS.md` 26.18). So "better than the floor" is not
  a claim this data can support in either direction -- there is no floor here to beat.
- **The measurement is univariate and the shipped file is not.** Every figure above was
  measured with **one model per channel**. The file format and the flight core are
  multivariate -- one model over up to 16 channels. Both shapes are legal and the univariate
  case is the multivariate one at one channel, but **these numbers are not a prediction
  about a multivariate model** and are not quoted beside one.
- **The population is confounded with envelope width.** See section 3.
- **No early-warning claim.** See section 7.

**(!) The all-38 figure, labelled.** The same arm scores **30 of 38** across both halves.
That number is **CONTAMINATED** -- it includes the 19 TUNE events the cut was selected on --
and it exists only so the comparison against the published rule's 10 of 38 can be made at
all. It is never the headline.

## 2. What the 38 are, and why that number

The population is **38 labelled contextual anomalies across 25 channels** of SMAP/MSL that
stay **at or within** their channel's training minimum and maximum.

```
  43   labelled contextual anomalies in the dataset
  39   of them stay at or within the channel's training envelope
                                                  (docs/DECISIONS.md D46, D46.1,
       artifact runs/smap-msl/_forensics/2026-09-03T192723Z-visibility.json)
   1   G-1[4770-4890] excluded: its channel is one of four the frozen arm's own
       artifact lists as excluded, on a training stall
  38   scored
```

**(!) "At or within", and the difference is 23 of the 39.** The in-range test is
**inclusive**: touching a channel's historical extreme does not count as leaving it.
Measured 2026-09-14 (D46.1): **23 of the 39 touch a rail exactly** -- margin `0.0`, not
floating-point noise -- and only **16 sit strictly inside**. **18 of those 23 sit in an
envelope of exactly `[-1, +1]`**, which is the test-split pre-scaling of section 3 showing
through: a map fitted on the test split pins that split's extremes at the rails by
construction.

**It moves no figure** -- 39, 38 and every recall number are unchanged -- and the claim
that a limit check cannot see these still holds, **on the premise that a real RED or YELLOW
limit sits outside a channel's historical operating range** (D43). That premise is
load-bearing rather than incidental: on the arithmetic alone, an inclusive limit placed
**at** the historical extreme would fire on 23 of these 39.

**Four sequences fail the test, and one is not a failure.** All four are MSL; all 26 SMAP
contextual sequences are in range. **`C-2[1540-1575]`'s training split is constant at
`-1.0`** -- a **zero-width envelope** -- so every test value above `-1.0` counts as outside
by construction and 33 of its 36 steps do. That is a degenerate channel rather than an
anomaly leaving its range. The other three: `M-2[1110-2250]` 622/1141 steps outside,
`M-5[1250-1550]` 35/301, `M-1[1110-2250]` 27/1141.

**And the point/contextual split is telemanom's own field, not derived here.**
`labeled_anomalies.csv` ships a `class` column, documented at
`third_party/telemanom/README.md:104-109`; this project fetches it verbatim and checksums
it, and never recomputes it.

For contrast, **11 of 61** point anomalies stay in range. That gap is the reason this
population is the one the claim rests on.

## 3. (!) The confound, stated because it changes what may be read off the number

The in-range class is **confounded with channel envelope width**
(`docs/DECISIONS.md` D66, artifact
`runs/smap-msl/_forensics/2026-09-10T041723Z-probe.json`). Contextual anomalies are labelled
on channels whose signal is rail-to-rail across both splits; point anomalies on channels
whose signal is narrow-band. That is a property of the dataset's composition, not of its
preparation, and it is **not correctable**.

**So the 39-of-43 arithmetic stands and the inference from it is bounded.** "These anomalies
are in range" is measured and correct. "Contextual anomalies are the population a limit
check cannot see, and point anomalies are not" may **not** be read off it as a property of
the anomalies alone.

**One candidate explanation was investigated and refuted.** The published arrays are
pre-scaled by one affine map per channel, fitted on the test split and applied to both
arrays. An in-range determination is **invariant** under an affine map applied identically
to both, so the arithmetic is immune to it rather than merely unaffected.

## 4. What the flight core computes, and how closely

The component flies `max(z_residual, z_derivative) >= threshold` on a `param_version` 2
parameter block (`docs/DECISIONS.md` D68). **This is exactly the arm measured above**, and
adopting it was taken as a decision in its own right rather than as a side effect of writing
the port.

The C++ is held to a NumPy reference, and these are the numbers `make -C flight test` prints
**on this branch**:

| Stream | Worst absolute difference | Tolerance |
|---|---|---|
| Level 1 baseline, 4 tiers | 0.000e+00 | 1e-05 |
| Trailing window, 3 tiers | 3.738e-10 | 1e-05 |
| Derivative stream, 2 tiers | 2.899e-07 | 1e-05 |
| Dynamic threshold `eps`, 2 tiers | 5.072e-06 | 1e-05 |
| **Fused score, tier `p1`** | **3.098e-06**, and **85 of 85** emissions exact | 1e-05 |
| GRU forward pass, tiers `g1` and `g2` | 1.192e-07 | 1e-05 |

Tier `p1` is the end-to-end case: a real `model.bin` at `param_version` 2, loaded into a
`Detector` and stepped **3,200 times**.

**The version check is load-bearing, not decoration.** On `p1`, the same bytes with the same
cut value **cross 339 times as version 2 and 0 times as version 1**, differing on 339 of
3,200 steps. A reader that accepted an unknown generation would apply one rule's cut to the
other's statistic in silence.

## 5. The port's four pre-registered predictions, reported with their outcomes

Registered in `docs/MODELS.md` 39.9 before the C++ existed; outcomes at 39.13.

| # | Prediction | Outcome |
|---|---|---|
| **N1** | The C++ matches the reference to 1e-5 on every continuous output, and **exactly** on the emission flag, over every tier | **HELD** |
| **N4** | Determinism survives 2,170 samples of carried state: bit-identical in one process and across two | **HELD** |
| **N3** | `sizeof(Detector)` = 581,488 B +/- 64 | **FAILED** at **603,032 B, +3.70%**, outside its own 1% band |
| **N2** | The new path lands where the existing one landed, at 1e-6 | **NO VERDICT.** The reference is float32, so the contract cannot be judged; a float64 reference is registered as owed and not built |

**N3 failed and the band was not moved.** The account, itemised: **+8,960 B** because the
rings are 2,170 deep, not 2,100 -- the window the threshold solves over is 2,100 of history
plus the 70 it judges; **+12,168 B** for the pruning ladder's scratch, which the prediction
did not itemise at all; **+408 B** for the second moment-accumulator set that lets one ring
serve both the threshold's 2,170 contents and the fused statistic's 2,100 moments.

**And the figure moved 8 bytes after it was adjudicated.** A later commit added a `U32`
member so a warning could name its peak channel, and padding took the object from 603,024 to
**603,032**. The verdict is unchanged -- +3.7043% against +3.7057%, both reported as +3.70%
-- but nothing caught the drift, because the check was a 2% band on a compile-time constant.
It is an **equality** now. Recorded at `docs/MODELS.md` 39.13.5 on `dev`.

## 5a. The physics testbed, and the first warning time measured against a real limit

**Numbered 5a so sections 6 to 10 keep the numbers other documents cite**, the way
`docs/HARNESS.md` 5a does on `dev`.

No dataset this project holds can produce a warning time: SMAP/MSL ships no limit definitions
and ESA-ADB's clock is anonymised. So a subsystem was simulated instead -- a coupled power and
thermal plant, eight channels, limits declared in its own F' dictionary, and a seeded
degradation that ramps **one scalar**, the cell's internal resistance, and lets the physics
carry it everywhere else. The ground toolkit fitted a model on **that plant's own healthy
telemetry**, on a seed none of the scored runs use. Ten seeded runs, ten healthy controls.
Source `docs/MODELS.md` 42 and 42.9 on `dev`; component `fprime/SentinelRef/PowerSim/`.

### (!) The method is the result: the null is measured away, not computed

**The naive reading of a seeded run is wrong, and the healthy control is what shows it.** Seed
1 says *first warning at tick 2,700*, against a first limit crossing at 19,225 -- a lead of
**+16,525**. But the healthy control, **same seed, same plant, no fault, warns at tick 2,700
as well**. That alarm is the plant's, not the fault's.

Because the two runs differ in exactly one scalar, a warning present in the seeded run and
absent in the healthy run **at the same tick** is the fault's, and nothing else can have caused
it. That is ground truth by construction, and it replaces a computed null with a measured one:

```
  seed 1, 22,000 ticks. One column is about 323 ticks.

  healthy control   ........#..###.##..##.#...##.#.#....###.#..###.##...#.##..#..###..#.
  seeded run        ........#..###.##..##.#..###.###.##.###.##.###.##.###.###.##.###.##.
  ATTRIBUTABLE      .........................#....#..##......#........##....#..#.....#..
                                            ^                                  ^  ^
                                            fault 8,000            yellow 19,225  red 20,140

  40 warnings in the seeded run.  30 in the healthy run.  30 shared, so not the fault's.
  10 attributable, the first at tick 8,100 -- one hundred ticks after injection.
```

**The attributable row is empty for the first 8,000 ticks, and that is the argument.** Every
figure below is measured from the attributable warning. **Quoted without that method, the
number is 16,525 and it is false.**

### What was measured

| | |
|---|---|
| **Lead, seed 1** | **11,125 timesteps** before the first limit crossing of any colour |
| **Lead, median of ten** | **9,774.5 timesteps**; min 5,305, max 11,125. Positive on **10 of 10** |
| **As a fraction** of the fault-onset-to-limit interval | **0.871** median -- the dimensionless figure, and the only one that transfers off this plant |
| **False alarms** | **316 / 196,500 warmed healthy ticks = 0.1608%**, against the **0.1830%** the toolkit's own pre-launch report predicted for data it had not calibrated on |
| **Compute** | per tick **median 11 us, worst 326 us**, against this deployment's 1 Hz period -- **0.033% of the budget**, worst case, and that figure includes the trace write |

### (!) Read these in the same breath as the numbers

- **Timesteps, not seconds.** At this deployment's 1 Hz rate group the median lead is 2 h 43
  min. At another rate it is a different number of seconds and **the same number of ticks**.
  The plant's time constants are chosen, so its seconds are not a mission's.
- **Ten runs are not ten independent systems.** They share a plant, a fault mode, a rate and an
  injection tick and differ only in noise. The first limit crossing lands at **19,224 or 19,225
  in all ten** -- a two-tick spread -- so the **effective `n` is close to one**, and 10 of 10 is
  a statement about reproducibility far more than about power.
- **One fault mode, one model, one testbed**, and it is **a fault this project designed**. A
  method that catches the degradation someone thought to write is not thereby a method that
  catches degradations nobody wrote.
- **This is not an early-warning claim about spacecraft telemetry.** On real telemetry the flown
  rule is **less late than the rule it replaced, and not early** (`docs/MODELS.md` 45.6 on
  `dev`). Section 7 stands unchanged.
- **The live GDS recording is not produced** and is owed.

### (!) The first channel the testbed tried to limit-check was one that cannot be

`SolarInput` was first declared with a **yellow low of 5 W**. It fired on **tick 0 of every
run, healthy ones included**. The array reads **0 W through eclipse, which is 35% of every
orbit**: low solar power is not a fault, **it is night**. Whether a low reading is anomalous
depends on the relationship between that channel and the orbit phase, and the same 0 W is
correct in shadow and catastrophic in sunlight.

Its limits now sit where a **sensor bias** would be, `red -5.0, yellow -2.0`, because that is
the only thing a fixed threshold on that channel can honestly detect. **This project's own
argument, demonstrated in its own instrument, and found by a run rather than a review.**

### And the bar is yellow, not red

`CellTemp` crosses **yellow at 19,225 and red at 20,140**. A ground system with yellow alarms
sees this fault **915 ticks before the red trip**, so a lead scored against the red trip alone
would credit the component with beating a limit check it had **not** beaten. Every figure above
is measured against the **first crossing of any colour**.

## 6. What you can check from this branch, and what you cannot

**Can, with a C++ toolchain and nothing else:** every figure in section 4, the footprint in
section 5, the 18 load cases, determinism, and a byte-identical round trip of the committed
model files. `make -C flight test`.

**Cannot, here:** every figure in sections 1 to 3. They come from scoring runs against
SMAP/MSL and ESA-ADB through the Python harness in `src/`, which is not on this branch, over
data staged in a private bucket. The artifacts are cited by path so that a reader on `dev`
can find them; **a reader here is taking them on the citation.** That is the honest position
and it is not dressed up.

`docs/datasets/REPRODUCING.md` says exactly what `dev` lets you recompute with no dataset
and no credential.

## 7. What is not claimed

- **No early-warning claim about spacecraft telemetry, in any form.** Two lead measurements
  exist on real telemetry and neither is one. **0 of 10 positive leads**
  (`docs/MODELS.md` 37.7a) was measured on the **frozen decision layer**, which emits at the
  end of a 70-tick segment; the rule this branch ships crosses **per tick**, and its own lead
  was measured separately at `docs/MODELS.md` 45.6 on `dev`, which found it **less late than
  the rule it replaced and not early** -- the one positive median it produced failed to beat
  its own null. **Section 5a's testbed lead is a different claim**: a simulated plant and a
  fault this project designed, transferring to no mission. Lead, where reported at all, is in
  **timesteps**, never in hours, and a figure of "+26 timesteps" that once appeared in this
  project's own documents was **retired** -- it was dated from the start of an alarm range that
  had been widened backwards from a crossing that had already happened.
- **No point-adjusted F1, ever**, on the grounds Kim et al. (AAAI 2022) give.
- **No claim that the component beats a limit check on a mission's own data.** It has never
  been run on a mission.
- **No hardware envelope claim.** `docs/PI_ENVELOPE.md` is reserved and empty.

## 8. Closed negatives, reported because they cost something to establish

- **Pruning -- telemanom's published false-alarm filter -- is a clean negative on this
  population.** No value of `p` catches more than **10 of 38** at or under the matched
  0.6820%. The published paper's own section 4.3 states that `p` was tuned against labels;
  a mission has none. This is the finding the project's contribution rests on.
- **CUSUM on the standardised residual: 0 of 38** at the matched rate, with the reference
  slack fixed a priori and dimensionless.
- Both from `docs/DECISIONS.md` D65 and the artifacts in section 1.

## 9. Owed, and named rather than left implicit

- **A third read separating the dilation lever from the rate dial.** Dropping backward
  dilation measured **+13 events** at a matched rate -- but the cut moved from 0.5506 to
  0.5009 inside a band where a sub-1.0 dial re-admits pruned steps, so part of that is the
  lever and part is the dial, and no read so far divides them. **The forward-only choice
  does not rest on that number**; it rests on the fact that backward dilation would mark
  timesteps already emitted and a warn-only component has no un-emit.
- **Guard cells**: not adjudicated. Its cut pins and the rate lands 14% below target, so no
  matched-rate figure of it is reported in either direction.
- **Three improvement arms unrun**, and three predictions not adjudicated. Named in
  `docs/DECISIONS.md` D65 and D68.
- **The twelfth refusal code** is not yet in the F' component's degrade-to-Level-1 test
  (`docs/DESIGN.md` section 4).
- **A float64 reference** for the dynamic threshold, without which N2 cannot be judged.
- **The ground toolkit is not released on this branch**, and without it a mission cannot use
  any of this. It is written: **3 of its 3 acceptance-ladder rungs have run** on `dev`, the
  last on a real twelve-channel mission. What is owed here is the release, not the build.

## 10. Sources

Every decision cited above is on `dev` at `664de7e` in `docs/DECISIONS.md`: **D46** the
in-range population, **D65** the decision-layer finding, **D66** the envelope confound,
**D68** the flight configuration, **D69** this branch's curation. The pre-registrations and
their outcomes are `docs/MODELS.md` 26.18, 36, 37, 38 and 39.

Datasets, their licences and the caveats that would corrupt a result: `docs/datasets/`.
