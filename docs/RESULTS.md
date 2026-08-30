# Results

Every number the harness has produced, with the floor it had to clear and the
corrections it has been through. Nothing is overwritten here: a figure that has
been superseded stays, beside what replaced it and the reason.

**Read `docs/HARNESS.md` first**, in particular section 7, which records a
prediction that failed. `docs/MODELS.md` carries the deviation ledger and the
caveat that qualifies every LSTM figure below.

---

## 1. The floor, and what cleared it

**The floor was `rstd` on `m1-g8.9.10`: event-wise F0.5 = 0.250.** A two-line
rolling standard deviation. Work items 4, 5 and 6 must beat it.

**Work item 4 cleared it**, and the margin is not the interesting part:

```
  rstd            F0.5 0.250    headline-cell recall   3/32
  lstm-telemanom  F0.5 0.269    headline-cell recall  28/32
  lstm-quantile   F0.5 0.421    headline-cell recall   6/32
```

### (!) The floor was never a flight candidate, and only one detector survives

Measuring lead time -- which nothing in this project did until work item 4 --
changed which detectors are eligible at all:

```
  lstm-telemanom     +26 timesteps    the only detector that warns in advance
  mavg                 0              fires at the event boundary
  lstm-quantile     -122              alarms after the event began
  rstd            -1,512              the floor
```

**`rstd` remains the F0.5 floor and is retained for comparability.** Every result
in this repository is measured against 0.250 and that does not change. But over
the three events it catches it alarms a median of **1,512 timesteps late**, so by
the rule in `docs/HARNESS.md` section 1 it is **disqualified on timeliness and was
never a candidate for the flight configuration**. That was unknowable before the
metric existed, which is the point.

`lstm-quantile` is disqualified on the same ground at -122, despite holding the
best F0.5 and the best false-alarm rate in this document.

**The consequence, stated plainly: `lstm-telemanom` is currently the only
surviving candidate.** Everything from here is about fixing its false-alarm rate
**without spending its 26-timestep head start.** That head start is the budget,
it is small, and several of the mechanisms that buy precision buy it by waiting --
the persistence sweep already moved lead time from +26 to -37 as N rose from 1 to
60, for almost no gain in F0.5. Every layer from here reports its lead-time cost
beside its precision gain.

The cross-channel claim is what was actually tested. `m1-g8.9.10` carries 32
`Multivariate/Global/Subsequence` events -- the taxonomy cell Objective.md 2.4's
argument lives in -- and a per-channel statistic finds three of them. A
forecaster over the whole channel set finds twenty-eight. That is the
project's thesis measured rather than asserted.

**What did not clear anything is the false-alarm rate.** `lstm-telemanom` alarms
on 22 of 48 commanded manoeuvres. Objective.md 11 rule 2 is explicit that a
detector doing that is muted within a week in orbit, whatever its recall.
`docs/MODELS.md` records why: **we reproduced telemanom without its command
inputs**, which is the one input that makes commanded events predictable.

## 2. Every detector, both sets

Both channel sets are reported for every detector, always. `m1-ss5` was the
primary until the baseline run and was demoted on footprint evidence measured
**post-hoc**; it is retained rather than discarded, because you cannot
cherry-pick if you never discard anything. See `docs/HARNESS.md` section 2.

All figures at the **default operating point** -- persistence 1, channel
agreement 1. Section 4 sweeps both.

> **(!) EVERY LSTM ROW IN THIS SECTION IS PRE-TRAINING-FIX.** A defect in early
> stopping meant every model in this project trained for one epoch
> (`docs/DECISIONS.md` D17). It was corrected on 2026-08-26 and the forecaster
> improved about fortyfold, and the detection stack downstream of it collapsed:
> event-wise F0.5 fell from **0.269 to 0.026** here and from **0.664 to 0.035**
> on `m1-ss5`. **Section 6a carries both sets of numbers side by side and is the
> current state.** These tables are kept, not replaced, for the reason section 6
> gives.

**GATE -- `m1-g8.9.10`** (12 channels, groups 8+9+10, the primary recall set)

| Detector | **F0.5** | **lead** | recall | precision | MVGS | contextual | point | **rare-event FA** | alarms/1k | VUS-PR |
|---|---|---|---|---|---|---|---|---|---|---|
| `lstm-quantile` | **0.421** | **-122 (!)** | 6/46 (0.130) | 20/21 (0.952) | 6/32 (0.188) | 6/45 (0.133) | 0/11 (0.000) | **1/48 (0.021)** | 0.000 | 0.259 |
| `lstm-telemanom` | **0.269** | **+26** | 37/46 (0.804) | 42/182 (0.231) | 28/32 (0.875) | 37/45 (0.822) | 9/11 (0.818) | **22/48 (0.458)** | 0.015 | 0.078 |
| `rstd` | 0.250 | **-1,512 (!)** | 3/46 (0.065) | 6/7 (0.857) | 3/32 (0.094) | 3/45 (0.067) | 0/11 (0.000) | **1/48 (0.021)** | 0.000 | 0.030 |
| `mavg` | 0.028 | 0 | 34/46 (0.739) | 452/19909 (0.023) | 25/32 (0.781) | 34/45 (0.756) | 9/11 (0.818) | **10/48 (0.208)** | 1.796 | 0.226 |
| `quiet` | undefined | n/a | 0/46 (0.000) | -/0 | 0/32 (0.000) | 0/45 (0.000) | 0/11 (0.000) | **0/48 (0.000)** | 0.000 | 0.030 |

**(!) Two detectors are disqualified by the lead-time rule** (`docs/HARNESS.md`
section 1), and one of them is the floor.

`lstm-quantile` holds the best F0.5 and the best false-alarm rate in the table and
alarms a median of **122 timesteps after** the event began -- all six of its
catches late.

`rstd`, **the floor this whole work item was measured against**, is worse:
a median of **1,512 timesteps late** over the three events it caught. It cleared
F0.5 = 0.250 while being, in the terms Objective.md 2 sets out, not an
early-warning detector at all. That does not retract the floor -- it was and
remains the number to beat on F0.5 -- but it does mean the floor was never a
candidate for flight, and nobody could have known that before the metric existed.

`mavg` sits at exactly 0: it fires at the event boundary, neither early nor late,
which is what a per-channel threshold on a spike does.

**Only `lstm-telemanom` warns before the event**, at +26.

**`m1-ss5`** (6 channels, group 8; demoted, point-anomaly coverage)

| Detector | **F0.5** | recall | precision | MVGS | contextual | point | **rare-event FA** | alarms/1k | VUS-PR |
|---|---|---|---|---|---|---|---|---|---|
| `lstm-telemanom` | **0.664** | 36/42 (0.857) | 39/62 (0.629) | 27/31 (0.871) | 36/41 (0.878) | 9/11 (0.818) | **17/48 (0.354)** | 0.005 | 0.184 |
| `lstm-quantile` | 0.452 | 6/42 (0.143) | 76/77 (0.987) | 6/31 (0.194) | 6/41 (0.146) | 0/11 (0.000) | **1/48 (0.021)** | 0.000 | 0.458 |
| `mavg` | 0.135 | 32/42 (0.762) | 399/3573 (0.112) | 23/31 (0.742) | 32/41 (0.780) | 9/11 (0.818) | **8/48 (0.167)** | 0.240 | 0.139 |
| `rstd` | undefined | 0/42 (0.000) | 0/98 (0.000) | 0/31 (0.000) | 0/41 (0.000) | 0/11 (0.000) | **1/48 (0.021)** | 0.009 | 0.009 |
| `quiet` | undefined | 0/42 (0.000) | -/0 | 0/31 (0.000) | 0/41 (0.000) | 0/11 (0.000) | **0/48 (0.000)** | 0.000 | 0.013 |

`n < 20` denominators -- point recall throughout, and `rstd`'s precision --
cannot distinguish detectors: over 11 events, recall takes 12 values. Treat
them as coverage checks, not as comparisons.

## 3. Lead time -- how early, not just whether

Every metric above answers *did you catch it*. None answers *how long before*,
and early warning is the project's headline claim (Objective.md 4). Measured
here for the first time, in **timesteps** -- ESA-ADB time is anonymised and
scaled, so no figure on this data may be expressed in hours.

**GATE -- `m1-g8.9.10`**

| Detector | median | p25 | p75 | detected late | credited alarm width |
|---|---|---|---|---|---|
| `lstm-telemanom` | **+26** | +16 | +46 | 5/37 | 204 |
| `mavg` | 0 | - | - | 6/34 | - |
| `lstm-quantile` | **-122** | -124 | -42 | **6/6** | 1,151 |
| `rstd` | **-1,512** | - | - | 2/3 | - |
| `quiet` | n/a | - | - | 0/0 | - |

**This reverses a ranking, and it is the reason the metric was added.**
`lstm-quantile` has the better F0.5 and by far the better false-alarm rate --
and **every one of its catches is after the event has already begun**, on alarm
ranges five times wider. It is not detecting early; it is detecting broadly and
late. `lstm-telemanom` warns a median of 26 timesteps ahead, on tight alarms.

For a component whose purpose is warning hours before a limit trips, a detector
that fires after the fact has failed at the thing being built, whatever its
F0.5. Neither number is discarded; both are reported, because they measure
different properties and the gate has to weigh them.

By taxonomy cell, `lstm-telemanom`: **+41** median on
`Multivariate/Global/Point` (n=9), **+21** on `Multivariate/Global/Subsequence`
(n=28).

**(!) WHAT THIS METRIC IS MEASURED AGAINST, and it qualifies every figure in this
section whatever its sign.** Lead time here is the gap to the **labelled event
start** -- a hindsight annotation written by an operations engineer after the
fact. It is **not** a limit trip. A positive figure would mean *the detector
spoke before the annotation begins*, not *before the spacecraft was in danger*,
and only the second is the product claim.

The break-to-limit-trip lead **cannot be computed on this data at all**: ESA-ADB
carries no dictionary limits and its timestamps are anonymised and scaled
(`docs/HARNESS.md` section 4). It is a Phase 3 measurement on the F' Ref
deployment, on a real clock. **No wall-clock or "hours" figure may be derived
from anything in this document** (Objective.md 1.1).

What this project claims from Phase 1 is not earliness. It is that Sentinel is the
**first and only observer of cross-channel relationship breaks** -- 28 of 32
headline-cell events against a per-channel statistic's 3.

**(!) MEASURED 2026-08-28: the term is the whole of the number.** From the first
actual threshold crossing, `lstm-telemanom`'s median lead is **+0.0** on both
sets against the **+26.0** and **+27.0** reported here, and it leads in **3 of
38** events rather than 34 of 38. Median inflation **+27.5** and **+32.0**, max
+63; in 31 of 38 cases the buffer is the entire lead. The rows below are retained
and are **not** a measure of early warning. `docs/DECISIONS.md` D21.

**(!) The original note, 2026-08-27, kept:** `error_buffer = 100` widens each exceedance by +/-99 timesteps
before alarm ranges are formed, and lead time is measured from a range's start.
So an alarm can begin up to **99 timesteps** before the exceedance that caused
it, against a reported median of **+26**. Not measured, not claimed to be an
artifact, and not removable from these numbers retrospectively -- recorded here
because a table gets screenshotted and travels without the document that
qualifies it. `docs/DECISIONS.md` D21; `docs/HARNESS.md` section 1.

## 4. The decision layer, swept as a grid

Two mechanisms the project designed were off while the figures in section 2 were
measured: the persistence filter at N=1, and channel combination at the maximum
over channels. Swept together, because both suppress alarms and one can make the
other redundant. **Every cell is reported. No cell is selected**: choosing a
configuration and publishing only its numbers would hide the trade the grid
exists to show.

**`m1-g8.9.10`**, N = persistence, k = channels required to agree:

| N | k | **F0.5** | recall | precision | MVGS | point | **rare-FA** | lead med |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.269 | 37/46 | 42/182 (0.231) | 28/32 | 9/11 | 22/48 | +26 |
| 1 | 2 | 0.504 | 24/46 | 27/54 (0.500) | 18/32 | 6/11 | 17/48 | +24 |
| 1 | 3 | **0.534** | 21/46 | 24/43 (0.558) | 17/32 | 4/11 | 15/48 | +22 |
| 5 | 1 | 0.271 | 37/46 | 42/181 (0.232) | 28/32 | 9/11 | 22/48 | +22 |
| 5 | 3 | **0.534** | 21/46 | 24/43 (0.558) | 17/32 | 4/11 | 15/48 | +18 |
| 20 | 1 | 0.254 | 35/46 | 39/179 (0.218) | 28/32 | 7/11 | 21/48 | +10 |
| 20 | 3 | 0.503 | 20/46 | 22/42 (0.524) | 17/32 | 3/11 | 15/48 | +5 |
| 60 | 1 | 0.216 | 28/46 | 33/177 (0.186) | 25/32 | 3/11 | 21/48 | -32 |
| 60 | 3 | 0.450 | 17/46 | 20/42 (0.476) | 16/32 | 1/11 | 15/48 | -37 |

`m1-ss5` inverts: its best cell is **k=1 at 0.664**, and agreement *hurts* it
(0.511 at k=3), because group 8's events are spikes on fewer channels. The two
sets are two regimes and the grid is not reconciled across them.

**What the grid says.**

- **k is the lever; N is not.** k=1 to k=2 nearly doubles F0.5. N=1 to N=5 moves
  it by 0.002.
- **Persistence is subsumed here.** N=5 removed *one* alarm range out of 182,
  because telemanom's `error_buffer` already widens every exceedance by +/-99
  timesteps, so no blips survive to be filtered. Recorded in Objective.md 7.1;
  the mechanism is retained, and may be live again for the GRU and TCN.
- **The trade k buys, stated plainly.** Precision 0.231 to 0.558 and rare-event
  false alarms 22/48 to 15/48, paid for with **headline-cell recall 28/32 to
  17/32** and point recall 9/11 to 4/11. Half the events the project exists to
  catch, given up to buy precision.
- **Lead time falls almost exactly 1:1 with N** (+26, +22, +10, -37 across
  N=1/5/20/60), which is the persistence filter working as designed and makes
  its cost visible rather than inferred.
- **No cell fixes the adoption number.** The best in the whole grid is 15/48
  (0.312), still worse than `mavg`'s 10/48. **Decision-layer tuning did not
  solve the false-alarm problem**, which is what sent the work to command
  conditioning.

Full grid, including every cell omitted above:
`runs/m1-g8.9.10/_grid/2026-08-25T230914Z.json`.

## 5. The quantile branch, closed on timeliness

`lstm-quantile` held the best F0.5 in the project and the best false-alarm rate,
and had never been swept across channel agreement. Before building four more
layers on the dynamic-threshold branch, the question was whether **any** k
restores positive lead time.

**Twenty-four cells. None does.**

| N | k | F0.5 | recall | precision | **lead med** | **lead p75** | rare-FA |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.421 | 6/46 | 20/21 (0.952) | **-122** | -42 | 1/48 |
| 1 | 3 | 0.420 | 6/46 | 17/18 (0.944) | **-100** | -36 | 1/48 |
| 20 | 1 | 0.421 | 6/46 | 20/21 (0.952) | **-141** | -61 | 1/48 |
| 60 | 3 | 0.420 | 6/46 | 16/17 (0.941) | **-160** | -94 | 1/48 |

The best cell in the entire grid is **-100**. The 75th percentile is negative
everywhere, so this is not a median concealing a healthy tail: three quarters of
all catches, in every configuration, land after the event began.

**k does almost nothing here.** F0.5 spans 0.397 to 0.421 across the whole grid,
and recall is *identical* at 6/46 in all 24 cells -- as is headline-cell recall at
6/32 and point recall at 0/11. Requiring more channels to agree cannot change a
verdict when the detector fires 21 times in eleven million timesteps.

**N makes it monotonically worse**, -122 to -181, the same shape the dynamic-threshold
branch showed.

### Why -- and the reason this closes the branch rather than losing to it

The two thresholding rules differ in what they respond to, and that difference is
definitional rather than incidental:

| | responds to | fires on |
|---|---|---|
| **Nonparametric dynamic threshold** | error relative to the **local** recent error scale | the **onset** of divergence |
| **Global quantile** | error relative to the **99.9th percentile of years of training scores** | the **magnitude** of divergence |

A rising prediction error crosses a *local* threshold as soon as it departs from
the recent norm. To cross a *global* one it must grow to an absolute size, and
growing takes time. **That is why one warns early and the other warns late, and no
amount of channel agreement or persistence changes it** -- both of those make a
detector fire less, and this detector's problem is that it fires too late, not
too often.

The same mechanism explains the rest of the row: precision 20/21 and false alarms
1/48 are excellent because the detector almost never fires, and recall 6/46 is the
price. It is an accurate, quiet, late detector -- a good historian.

**The branch is closed on structural grounds**, which is a cleaner result than
losing on a number: it will not be reopened by better tuning, because tuning is
not what is wrong with it. The programme continues on the dynamic threshold, which
is the only branch that has ever produced a positive lead time.

Full grid: `runs/m1-g8.9.10/_grid/2026-08-26T184418Z-quantile.json`.

## 6. The correction: pre-fix and post-fix

`Bundle.subset` rebuilt its ground truth from the labels alone and never read
`self.valid`, so the unobserved fold-in `bundle.load` performs was lost on every
subset -- and `m1-ss5` is scored as a subset. Unobserved timesteps were
therefore counted as **scorable nominal time**, the free specificity
`docs/HARNESS.md` section 1 explicitly refuses for gaps and invalid segments.

It survived three work items because the trivial baselines are NaN-tolerant by
construction and nothing had ever asked the mask to be right. The first detector
that could not tolerate a NaN found it immediately.

**What the fix moved:** 40 timesteps on `m1-ss5`, from scorable to unscorable.

**What it changed in the results: nothing that is reported as a count.**

| | |
|---|---|
| Metric comparisons | 90 (10 set-detector pairs x 9 metrics) |
| Event-wise counts that moved | **0** -- F0.5, recall, precision, MVGS, contextual, point, rare-event FA all identical on both sets for all five detectors |
| Figures that moved | **6**, all on `m1-ss5`, all in the sixth or seventh significant figure: four `alarms/1k` (denominator lost 40 clean-nominal steps) and two `VUS-PR` (40 steps left the scorable mask) |

```
  m1-ss5  rstd   alarms/1k   0.008918973516  ->  0.008918975156
  m1-ss5  rstd   VUS-PR      0.009266952993  ->  0.009266917267
  m1-ss5  quiet  VUS-PR      0.013301995965  ->  0.013301137415
```

**Both sets of numbers are kept because that is the only way the conclusion
exists.** The tables in section 2 are post-fix. Had they simply replaced the
originals, nobody could tell whether the correction mattered -- and it is
precisely because both were retained that we can say the earlier results were
not distorted. A reviewer finding a correctness fix in the history with only
corrected numbers visible has to wonder what else was quietly cleaned up.

Pre-fix artifacts remain under `runs/m1-g8.9.10/*/2026-08-25T*.json`.

## 6a. The second correction: the training fix, pre and post

`docs/HARNESS.md` requires that a fix moving a published number has **both**
numbers recorded. This is that record for the early-stopping fix, and it is a
far larger movement than section 6's.

**What the defect was.** telemanom applies early stopping as
`current < best_loss - min_delta` with a published `min_delta` of 3e-4.
Validation MSE on ESA-ADB is about 1e-4, so the bar went negative and no epoch
after the first ever qualified. Every fit in this project stopped at epoch 11
having kept epoch 1. `docs/DECISIONS.md` D17.

**What it moved.** Same harness, same data, same folds, same decision layer at
persistence 1 and agreement 1. The only difference is the stopping rule.

**GATE -- `m1-g8.9.10`**

| | **F0.5** | **lead** | recall | precision | MVGS | contextual | point | **rare-event FA** | nominal-step FA | alarms/1k | VUS-PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pre-fix | **0.269** | +26 | 37/46 (0.804) | 42/182 (0.231) | 28/32 (0.875) | 37/45 (0.822) | 9/11 (0.818) | **22/48 (0.458)** | 21,780/10,675,488 (0.0020) | 0.015 | 0.078 |
| post-fix | **0.026** | +26 | 38/46 (0.826) | 75/3,548 (0.021) | 28/32 (0.875) | 38/45 (0.844) | 9/11 (0.818) | **30/48 (0.625)** | 530,769/10,675,488 (0.0497) | 0.320 | 0.068 |

**`m1-ss5`** (6 channels, group 8; demoted, always reported)

| | **F0.5** | **lead** | recall | precision | MVGS | contextual | point | **rare-event FA** | nominal-step FA | alarms/1k | VUS-PR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pre-fix | **0.664** | +28 | 36/42 (0.857) | 39/62 (0.629) | 27/31 (0.871) | 36/41 (0.878) | 9/11 (0.818) | **17/48 (0.354)** | 8,066/10,875,689 (0.0007) | 0.005 | 0.184 |
| post-fix | **0.035** | +26.5 | 38/42 (0.905) | 42/1,475 (0.028) | 29/31 (0.935) | 38/41 (0.927) | 9/11 (0.818) | **33/48 (0.688)** | 217,618/10,875,689 (0.0200) | 0.133 | 0.074 |

Recall components are read against the precision beside them, which is the
whole of what this table says: **detection improved on every axis and precision
collapsed.** Recall 37/46 to 38/46, headline cell held at 28/32, lead time
unchanged at +26 with one fewer catch landing late. A worse model catches fewer
events. This one catches more, at the same lead time, firing twenty times as
often, so the model is not what got worse.

Lead time, both sets, is unchanged within the resolution of 38 events: median
+26 to +26 and +28 to +26.5, p75 +46 to +46.75 and +47.25 to +46.75, with 5 of
37 late becoming 4 of 38 and 3 of 36 becoming 2 of 38.

**The adoption number moved the wrong way and that is the serious half.** Rare-
event false alarms 22/48 to 30/48 and 17/48 to 33/48. `Objective.md` 11 rule 2
is explicit that a detector alarming at most commanded manoeuvres is muted
within a week in orbit whatever its recall.

### Per fold, which is where the mechanism shows

The forecast did not improve equally across folds, and the alarm count tracked
it. Validation MSE is read from each cached fit's embedded training report;
alarm ranges are the denominators of the per-fold event precision.

```
  m1-g8.9.10   val MSE pre -> post          alarm ranges pre -> post   recall
    fold 0     2.745e-4 -> 1.691e-4  (1.6x)      42 ->    73  (1.7x)   13/15 -> 12/15
    fold 1     1.447e-4 -> 3.552e-6 (40.7x)      90 -> 1,970 (21.9x)   12/15 -> 13/15
    fold 2     1.599e-4 -> 4.089e-6 (39.1x)      50 -> 1,505 (30.1x)   12/16 -> 13/16

  m1-ss5
    fold 0     1.709e-4 -> 3.441e-5  (5.0x)      20 ->    26  (1.3x)   13/13 -> 12/13
    fold 1     9.309e-5 -> 5.205e-6 (17.9x)      20 ->   692 (34.6x)   11/14 -> 13/14
    fold 2     9.972e-5 -> 8.096e-6 (12.3x)      22 ->   757 (34.4x)   12/15 -> 13/15
```

**Fold 0 barely improved and barely exploded, on both channel sets.** That is a
control arm and a treatment arm inside the same run, and it is why the pooled
figures above must not be read on their own: pooling averages the control into
the treatment. It was sitting in the artifacts unread until work item 4's
threshold investigation went looking.

**Artifacts.** Pre-fix `runs/m1-g8.9.10/lstm-telemanom/2026-08-26T212610Z-1f8b6fd6.json`,
post-fix `runs/m1-g8.9.10/lstm-telemanom/2026-08-27T012552Z-8f48b731.json`.
Both carry all three folds and both channel sets.

**(!) A provenance defect noticed while reading them.** `RunRecord.git_commit`
is a `default_factory`, evaluated when each record is constructed rather than
once at run start, so the pre-fix artifact records `1bf6710` for `m1-g8.9.10`
and `5bab55e` for `m1-ss5` -- one run, two commits, because it was in flight
while the repository moved. The artifact path is the unambiguous identifier and
is what this document cites. Reported, not fixed: it is a harness correctness
question and `docs/HARNESS.md` requires escalation before a fix.

**The cause, measured.** `docs/DECISIONS.md` D17 and D18, from
`runs/m1-g8.9.10/_threshold/2026-08-27T*-diagnostics.json`. In one line: the
forecast improved, the **within-window** standard deviation of the smoothed
error fell three- to eightfold, the window maximum did not fall with it because
the residual's tail got *heavier* rather than lighter, so `(max - mu)/sigma`
rose and a `z` floor of 2.5 that used to be out of reach in 93% of windows is
now cleared in 80% of them. Each crossing costs at least 199 alarm timesteps
because `error_buffer = 100` dilates one exceeded sample by +/-99.

**No threshold has been chosen and none will be chosen from this table.** D18
records why: the best cell of the `z` sweep reaches F0.5 0.794 with zero
rare-event false alarms, and it is selected by reading scores against 46
labelled anomalies, which no adopting mission can do.

## 6b. The order-statistic arms, as a curve across admission rates

> **(!) TWO READINGS, BOTH KEPT. The tables below are the CONTAMINATED run and
> are retained, not corrected in place.**
>
> Every multiplier in them was fitted on the fitting window's residuals, which
> the code and this document both called normal-only. `splits.train_mask` removes
> annotated anomalies before the **fit**; it is not applied to the **scoring
> call** the calibration reads (`harness._score_fold`). So the floor was set
> partly by the anomalies it exists to sit above. `docs/NARRATIVE.md` section 6.
>
> **Section 6c is the clean re-run and is the reading to use.** Both are here
> because the difference is the finding: at tight budgets the contaminated
> numbers were almost entirely artifact, and one conclusion drawn from them --
> and one published decision built on that -- did not survive.

**GATE -- `m1-g8.9.10`**

| calibration | rate | **F0.5** | **lead** | recall | precision | MVGS | point | **rare-FA** |
|---|---|---|---|---|---|---|---|---|
| `independent` | 0.01% | 0.185 | +25.5 | 2/46 | 2/2 | 2/32 | 0/11 | 0/48 |
| `independent` | 0.10% | 0.411 | **-47.0 (!)** | 6/46 | 8/9 | 6/32 | 0/11 | 1/48 |
| `independent` | 1.00% | **0.449** | **+26.0** | 37/46 | 51/126 | 27/32 | 9/11 | **14/48** |
| `joint` | 0.01% | 0.172 | +25.5 | 2/46 | 2/3 | 2/32 | 0/11 | 1/48 |
| `joint` | 0.10% | 0.119 | +27.5 | 18/46 | 20/198 | 16/32 | 2/11 | 5/48 |
| `joint` | 1.00% | 0.033 | +24.0 | 40/46 | 64/2,405 | 29/32 | 9/11 | 22/48 |
| `local_only` | 0.01% | 0.454 | **-120.0 (!)** | 7/46 | 9/10 | 7/32 | 0/11 | 1/48 |
| `local_only` | 0.10% | 0.317 | **+26.0** | 21/46 | 25/85 | 19/32 | 2/11 | **8/48** |
| `local_only` | 1.00% | 0.063 | +26.0 | 40/46 | 120/2,359 | 30/32 | 9/11 | 28/48 |

**`m1-ss5`**

| calibration | rate | **F0.5** | **lead** | recall | precision | MVGS | **rare-FA** |
|---|---|---|---|---|---|---|---|
| `independent` | 0.01% | 0.200 | +25.5 | 2/42 | 2/2 | 2/31 | 0/48 |
| `independent` | 0.10% | 0.200 | +25.5 | 2/42 | 2/2 | 2/31 | 0/48 |
| `independent` | 1.00% | 0.377 | +28.0 | 36/42 | 41/124 | 27/31 | 18/48 |
| `joint` | 0.01% | 0.200 | +25.5 | 2/42 | 2/2 | 2/31 | 0/48 |
| `joint` | 0.10% | 0.096 | +34.5 | 14/42 | 14/172 | 12/31 | 5/48 |
| `joint` | 1.00% | 0.055 | +26.5 | 38/42 | 56/1,262 | 29/31 | 33/48 |
| `local_only` | 0.01% | 0.200 | +25.5 | 2/42 | 2/2 | 2/31 | 0/48 |
| `local_only` | 0.10% | 0.315 | +27.0 | 19/42 | 19/65 | 17/31 | 4/48 |
| `local_only` | 1.00% | 0.065 | +26.5 | 38/42 | 47/884 | 29/31 | 33/48 |

**(!) Two cells are disqualified on lead time** (`docs/HARNESS.md` section 1):
`independent` at 0.10% (-47.0, n=6) and `local_only` at 0.01% (-120.0, n=7).
Small denominators, and D9 is a rule rather than a preference. Every lead figure
here carries the open caveat in `docs/DECISIONS.md` D21 -- `error_buffer`'s
+/-99 dilation is nearly four times the +26 being defended, so these are
comparable to one another and not clean in absolute terms.

**Binding rate**, the share of segments whose threshold came from the local term
rather than the floor, pooled over folds:

```
                     0.01%   0.10%   1.00%
  m1-g8.9.10  ind    0.252   0.127   0.187
              joint  0.073   0.157   0.316
  m1-ss5      ind    0.019   0.082   0.068
              joint  0.009   0.075   0.167
  local_only  -- 1.000 everywhere, by construction: there is no floor
```

### What the curve says

**Correcting an arithmetic error made the detector worse.** At a 1% budget on the
gate set the *broken* independent fit spends **126** alarm ranges and the
*correct* joint fit spends **2,405**, for 37/46 against 40/46 recall. The
independent fit admits only about two thirds of the budget it is given, and that
shortfall had been acting as an unintended out-of-sample margin.

**`independent` at 1% is the highest event-wise F0.5 this document records with a
positive median lead time** -- 0.449, against 0.269 for pre-fix `lstm-telemanom`
and 0.250 for the `rstd` floor, at the same +26 and with rare-event false alarms
nearly halved, 14/48 against 22/48. **It is reported and not recommended.** It
works partly through an error whose out-of-sample margin is unmeasured, and no
cell here is selected.

**The floor was the part that was wrong.** `local_only` at 0.1% beats
`lstm-telemanom` on every axis at once -- 19/32 headline cell against 28/32 at
**85 alarm ranges against 3,548**, +26.0 lead, 8/48 rare events against 30/48.
Run bare, the local order statistic holds up. That is a different design from the
one proposed and `docs/DECISIONS.md` D20 stays open until it has its own
pre-registration.

**Per fold, and fold 0 remains the control.** `local_only` at 1% spends 405 alarm
ranges for 14/15 on fold 0, and **1,338 for 13/15 on fold 1** -- the 40x-improved
fold needs 3.3x the alarms for the same recall. The local scale degrades as the
forecast improves, which is the limitation `docs/MODELS.md` 10.3 predicted for
this design in advance.

## 6c. The same curve on a clean calibration pool

Same nine cells, same weights, same folds, the calibration pool masked to
genuinely nominal timesteps. Artifacts
`runs/m1-g8.9.10/_curve/2026-08-27T2318*Z-clean-*.json`.

**P10 monotonicity holds on every set-mode pair**, so the calibration does what
it claims and the rest is readable.

**GATE -- `m1-g8.9.10`**, contaminated -> clean:

| calibration | rate | alarm ranges | recall | MVGS | **F0.5** | **lead** |
|---|---|---|---|---|---|---|
| `independent` | 0.01% | 2 -> 11 | 2/46 -> 9/46 | 2/32 -> 9/32 | 0.185 -> 0.549 | +25.5 -> **-50.0 (!)** |
| `independent` | 0.10% | 9 -> 30 | 6/46 -> 24/46 | 6/32 -> 19/32 | 0.411 -> **0.806** | -47.0 -> **+27.5** |
| `independent` | 1.00% | 126 -> 259 | 37/46 -> 37/46 | 27/32 -> 27/32 | **0.449 -> 0.232** | +26.0 -> +26.0 |
| `joint` | 0.01% | 3 -> 27 | 2/46 -> 21/46 | 2/32 -> 18/32 | 0.172 -> **0.788** | +25.5 -> +29.0 |
| `joint` | 0.10% | 198 -> 401 | 18/46 -> 37/46 | 16/32 -> 27/32 | 0.119 -> 0.150 | +27.5 -> +26.0 |
| `joint` | 1.00% | 2,405 -> 2,700 | 40/46 -> 40/46 | 29/32 -> 29/32 | 0.033 -> 0.030 | +24.0 -> +24.0 |
| `local_only` | 0.01% | 10 -> 43 | 7/46 -> 20/46 | 7/32 -> 16/32 | 0.454 -> 0.651 | -120.0 -> **+24.0** |
| `local_only` | 0.10% | 85 -> 393 | 21/46 -> 36/46 | 19/32 -> 26/32 | 0.317 -> 0.253 | +26.0 -> +27.5 |
| `local_only` | 1.00% | 2,359 -> 2,678 | 40/46 -> 40/46 | 30/32 -> 30/32 | 0.063 -> 0.054 | +26.0 -> +26.0 |

**`m1-ss5`**, contaminated -> clean:

| calibration | rate | alarm ranges | recall | MVGS | **F0.5** | **lead** |
|---|---|---|---|---|---|---|
| `independent` | 0.01% | 2 -> 12 | 2/42 -> 12/42 | 2/31 -> 10/31 | 0.200 -> 0.667 | +25.5 -> +32.5 |
| `independent` | 0.10% | 2 -> 35 | 2/42 -> 24/42 | 2/31 -> 18/31 | 0.200 -> **0.741** | +25.5 -> +28.0 |
| `independent` | 1.00% | 124 -> 307 | 36/42 -> 38/42 | 27/31 -> 29/31 | 0.377 -> 0.176 | +28.0 -> +26.5 |
| `joint` | 0.01% | 2 -> 23 | 2/42 -> 20/42 | 2/31 -> 18/31 | 0.200 -> **0.820** | +25.5 -> +24.5 |
| `joint` | 0.10% | 172 -> 236 | 14/42 -> 36/42 | 12/31 -> 27/31 | 0.096 -> 0.207 | +34.5 -> +28.0 |
| `joint` | 1.00% | 1,262 -> 1,334 | 38/42 -> 38/42 | 29/31 -> 29/31 | 0.055 -> 0.055 | +26.5 -> +26.5 |
| `local_only` | 0.01% | 2 -> 29 | 2/42 -> 26/42 | 2/31 -> 20/31 | 0.200 -> **0.868** | +25.5 -> +31.0 |
| `local_only` | 0.10% | 65 -> 198 | 19/42 -> 36/42 | 17/31 -> 27/31 | 0.315 -> 0.238 | +27.0 -> +28.0 |
| `local_only` | 1.00% | 884 -> 953 | 38/42 -> 38/42 | 29/31 -> 29/31 | 0.065 -> 0.062 | +26.5 -> +26.5 |

### What the correction moved

**The tight end of the curve was almost entirely artifact.** At 0.01% every arm
had been reporting 2 events caught; clean, they catch 9 to 26. The contaminated
floor sat above the events, so the detector was silent and every conclusion drawn
from that region was about the bug.

**One highlighted result did not survive.** `independent @ 1%` was reported as
the highest event-wise F0.5 in this document with positive lead time, at 0.449.
Clean it is **0.232** -- below the `rstd` floor of 0.250. It is retained above
and withdrawn here.

**One conclusion did survive, weakened.** *Correcting the calibration arithmetic
made the detector noisier* holds: at 1% the joint fit still spends far more alarm
ranges than the independent one, 2,700 against 259. The ratio falls from 19x to
10.4x.

**And one published decision did not.** `docs/DECISIONS.md` **D13** closed the
quantile branch on the structural claim that *a global threshold must fire late*.
Clean, cells whose binding rate is 0.010 to 0.041 -- global in all but name --
fire at **+24.0 to +29.0** and catch 18/32 and 18/31 headline-cell events.
Seventeen of eighteen cells have a positive median lead. **The branch stays
closed on its measured numbers and its stated reason is withdrawn.** D13 carries
the full revision, including the confound in its original evidence:
`lstm-quantile` receives no `error_buffer` dilation where every telemanom-path
detector does, so the -122 against +26 was never like-for-like.

## 6d. `lstm-whitened` -- a decision layer that tests the relationship

`docs/DECISIONS.md` D23 established that every stage downstream of the forecaster
reads one channel at a time, and that `k`-of-`n` -- the only stage that does not
-- tests **co-occurrence rather than relationship**, which is precisely what a
commanded manoeuvre produces. This scores the whitened length of the **signed**
residual vector against its nominal covariance instead: one number per timestep,
low for a residual consistent with normal co-variation however large, high for
one orthogonal to it however small.

Same forecaster, same cached weights, same folds, same `error_buffer`, same
pruning. **The decision rule is the only difference.** Artifact
`runs/m1-g8.9.10/lstm-whitened/2026-08-27T225817Z-88b4df0d.json`.

**GATE -- `m1-g8.9.10`**

| | **rare-event FA** | **MVGS** | recall | precision | **F0.5** | **lead** |
|---|---|---|---|---|---|---|
| `lstm-telemanom` | 30/48 (0.625) | **28/32** | 38/46 | 75/3,548 | 0.026 | +26.0 |
| `lstm-whitened` | **2/48 (0.042)** | 21/32 | 27/46 | **39/40** | **0.861** | **+29.0** |

**`m1-ss5`**

| | **rare-event FA** | **MVGS** | recall | precision | **F0.5** | **lead** |
|---|---|---|---|---|---|---|
| `lstm-telemanom` | 33/48 (0.688) | **29/31** | 38/42 | 42/1,475 | 0.035 | +26.5 |
| `lstm-whitened` | **2/48 (0.042)** | 15/31 | 18/42 | 112/113 | 0.785 | +21.0 |

**The trade, stated rather than absorbed.** The adoption number falls by an order
of magnitude -- 30/48 to 2/48 and 33/48 to 2/48, with alarm ranges 3,548 to 40
and 1,475 to 113 at 39/40 and 112/113 precision, and lead time held or improved
on the gate set. **It is paid for in the class the project exists to catch:
headline-cell recall 28/32 to 21/32, and 29/31 to 15/31.** Seven events on the
gate set; fourteen on the subset, nearly half.

Event-wise F0.5 of **0.861** is the highest this document records by a wide
margin, against the `rstd` floor of 0.250, and it clears the lead-time rule at
+29.0. **That is not sufficient to adopt it** -- `docs/HARNESS.md` section 1 is
explicit that recall and precision are read together, and giving up seven
cross-channel events is a decision about what the component is for rather than a
number to optimise.

**(!) The 21/32 is not what it looked like, and neither is the 15/31. Measured
per event 2026-08-28** (`runs/m1-g8.9.10/_forensics/2026-08-28T022324Z-events.json`):

* **No lost event is single-channel.** Every one touches 5 to 12 of the channels
  in view -- `0/11` and `0/20` single-channel. The hypothesis that `lstm-whitened`
  is structurally blind to single-channel faults, and therefore needs a second
  detection path beside it, is **refuted**. There is no such path to build.
* **`lstm-whitened` catches nothing `lstm-telemanom` misses.** `whitened-only` is
  **0** on both sets, so its detections are a strict subset. It is a filter on
  telemanom's detections, not a different view of the data.
* **Neither channel count nor footprint separates lost from kept.** On
  `m1-g8.9.10`, lost events touch a median of 10 channels against 12 kept, and
  have a median footprint of 51 against 42 -- lost events are not shorter. On
  `m1-ss5` both are identical at 6 channels and a footprint of 1. **Why these
  particular events are lost is not established by this measurement.**

**On six channels versus twelve.** The loss rate is 11/38 on twelve channels and
20/38 on six, so the rule does perform better on the wider set. But the mechanism
usually offered for that -- low-channel-count events being the ones lost -- is
**not** what the data shows: within each set, lost and kept events have the same
channel counts, and on `m1-ss5` every event touches all six, so the comparison
cannot be made there at all. The `m1-ss5` figure is also confounded by a defect
already on record: group 8 renders these events as sub-grid-cell spikes
(`docs/HARNESS.md` section 7), and 17 of its 20 losses have a footprint of 1.
**The aggregate holds and the explanation does not**, and that distinction should
travel with the number.

**Four hypotheses for the losses, measured and refuted, 2026-08-28.**
Artifacts `runs/m1-g8.9.10/_forensics/2026-08-28T03*-{pruning,discriminator}.json`.

| Hypothesis | Verdict |
|---|---|
| single-channel events the joint test cannot see | **refuted** -- every lost event touches 5 to 12 channels; `0/11` and `0/20` |
| lower channel count | **refuted** -- lost and kept have the same counts within each set |
| shorter footprint | **refuted** -- lost median 51 against kept 42 on the gate set; identical on the subset |
| pruning discarding real detections | **refuted on the gate set** -- disabling it leaves 27/46 and 21/32 unchanged. **Real on the subset**: recall 18/42 to 23/42 and MVGS 15/31 to **18/31**, for six extra alarm ranges and **no** false-alarm cost. **Adopted 2026-08-28** -- see below |

**Pruning is now disabled for this rule** (`whiten.Config.pruning_p = 0.0`), and
it is a re-derivation rather than a tuning: telemanom's ladder was built for
per-channel absolute errors and this rule feeds it one joint whitened length.
Free or better on **both** sets, so it is a single global choice and not a per-set
dial. Intermediate values were not swept -- the claim is that a filter built for
another quantity does not apply, not that 0.0 is optimal. The tables in this
section are at `p = 0.13` and authoritative scored numbers for the new default
arrive with section 6e.
| in-pattern excursions the whitening divides out | **refuted** -- see below |

**What does separate them: the lost events are simply weak, in every view.**

```
  m1-g8.9.10   n    peak ||z||   peak whitened   ratio   reach (peak/threshold)
    kept       27        16.41           38.21   2.429                    1.933
    lost       19         3.76            5.85   1.449                    0.240
  m1-ss5
    kept       22        16.62           29.75   1.144                    1.867
    lost       20         3.41            6.42   1.257                    0.380
```

The in-pattern hypothesis predicted a **large** standardised length with a **low**
ratio -- a big excursion travelling along a learned direction, divided out by the
whitening. The opposite is measured: the lost events are **4.4 times smaller in
`||z||` before any whitening is applied**. Their residual is small in the raw
view too, so nothing is being divided out. `lstm-whitened` is not blind to them
in principle; it is **less sensitive** than `lstm-telemanom` and they fall below
it.

**And they are not marginal.** Median reach is **0.240** -- the lost events peak
at a quarter of the threshold. Recovering them by loosening would mean roughly a
fourfold reduction, which is not a tweak and would take the 2/48 with it.

**The likely mechanism, stated as a hypothesis and not measured.**
`lstm-telemanom` thresholds each channel against a **local** window of 2,170
errors, so it adapts to a quiet stretch and fires on a small excursion inside
one. `lstm-whitened` thresholds a joint score against **one global quantile** of
the whole nominal pool. That is the same asymmetry `docs/DECISIONS.md` D13 was
written about, and it predicts exactly what is seen: excellent precision (39/40)
and poor sensitivity to small events. Whether a *locally* referenced whitened
length recovers the eleven without reintroducing the collapse is the obvious next
question and is **not** answered here.

**(!) An open question that should be answered before this is adopted or
rejected.** The six-channel set loses **far more** recall than the twelve-channel
one, 14 events against 7. That is the opposite of what a relationship test should
do -- fewer channels means less relational structure for a covariance to exploit
-- so this may be measuring how much joint structure the rule has to work with
rather than how good the rule is. Untested either way.

**The whitening's own health is now asserted rather than assumed.** Median
whitened length 1.71 to 2.63 against `sqrt(12) = 3.46`, condition numbers 83 to
129. The first attempt produced 25.7 to 36.3 and condition numbers to 4.3e8 and
**scored anyway, silently**; `whiten.Accumulator.finish` now refuses both.

## 6e. The frozen decision layer

`docs/DECISIONS.md` **D24**. `lstm-whitened` -- the whitened-residual relationship
test, global reference, pruning disabled -- is the decision layer, held identical
across LSTM, GRU and TCN at the architecture gate. Artifacts
`runs/m1-g8.9.10/lstm-whitened/2026-08-28T050007Z-e51a6453.json` and
`runs/m1-g8.9.10/lstm-whitened-local/2026-08-28T050015Z-29053248.json`.

**GATE -- `m1-g8.9.10`**

| Detector | **rare-event FA** | **MVGS** | recall | precision | nominal-step FA | F0.5 | lead |
|---|---|---|---|---|---|---|---|
| `lstm-telemanom` | 30/48 (0.625) | **28/32** | 38/46 | 75/3,548 | 4.972% | 0.026 | +26.0 |
| **`lstm-whitened`** | **2/48 (0.042)** | **21/32** | 27/46 | 41/43 | **0.028%** | 0.848 | +29.0 |
| `lstm-whitened-local` | 0/48 | 20/32 | 25/46 | 26/26 | 0.024% | 0.856 | +29.0 |

**`m1-ss5`**

| Detector | **rare-event FA** | **MVGS** | recall | precision | F0.5 | lead |
|---|---|---|---|---|---|---|
| `lstm-telemanom` | 33/48 (0.688) | **29/31** | 38/42 | 42/1,475 | 0.035 | +26.5 |
| **`lstm-whitened`** | **2/48 (0.042)** | **18/31** | 23/42 | 118/119 | 0.853 | +27.0 |
| `lstm-whitened-local` | 0/48 | 17/31 | 21/42 | 22/22 | 0.833 | +29.0 |

**The trade, stated and not absorbed.** Commanded-manoeuvre false alarms fall
from **30/48 to 2/48** and nominal-step alarms from **4.972% to 0.028%** -- a
factor of 178 -- and alarm ranges from 3,548 to 43 at 41/43 precision. It is paid
for in **seven headline-cell events on the gate set**, 28/32 to 21/32, and eleven
anomalies overall.

Objective.md 11 rule 2 is the case for it: a detector alarming at 30 of 48
commanded manoeuvres is muted within a week in orbit, and a muted detector
catches nothing at all. The case against it is that the seven given up are
exactly the class the project exists to catch. **Both are true and the decision is
recorded as a judgement, not as an optimisation.**

**(!) The lead-time column is not comparable in absolute terms.** `docs/DECISIONS.md`
D21 measured the reported +26 as almost entirely `error_buffer` dilation: from
the first actual threshold crossing the median is **+0.0**, and the detector
leads in 3 of 38 events rather than 34 of 38. `+29.0` here means *no worse than
the arm beside it*, nothing more, until that metric is made honest.

## 6f. Lead time, measured from a moment the detector could reach

`docs/DECISIONS.md` D21, closed. Both readings, neither deleted. The honest one
re-dates each alarm range to the batch boundary at which the detector could
actually emit, keeping the range; the detected set stays the detector's.
Artifacts `runs/m1-g8.9.10/lstm-{telemanom,whitened}/2026-08-28T162*Z-*.json`.

**GATE -- `m1-g8.9.10`**

| Detector | reading | median | p25 | p75 | n | negative |
|---|---|---|---|---|---|---|
| `lstm-telemanom` | as reported | **+26.0** | +15.2 | +46.8 | 38 | 4 |
| `lstm-telemanom` | **honest** | **-43.0** | -54.5 | -21.5 | **23** | **22** |
| `lstm-whitened` | as reported | **+29.0** | +17.0 | +46.5 | 27 | 1 |
| `lstm-whitened` | **honest** | **-41.5** | -50.5 | -20.8 | **18** | **18** |

**`m1-ss5`**

| Detector | reading | median | n | negative |
|---|---|---|---|---|
| `lstm-telemanom` | as reported | +26.5 | 38 | 2 |
| `lstm-telemanom` | **honest** | **-53.0** | **11** | 10 |
| `lstm-whitened` | as reported | +27.0 | 23 | 1 |
| `lstm-whitened` | **honest** | **-51.0** | **8** | 8 |

### The two things this says, and the second is worse than the first

**Every detector warns late.** Medians of **-43.0** and **-41.5** on the gate set
against the +26.0 and +29.0 reported, and **22 of 23** and **18 of 18**
individual detections negative. The reported figures were dated from the start of
the batch in which a crossing occurred; the detector can only speak at that
batch's end.

**And the denominator falls, which is not a measurement artifact.** `n` goes 38 to
**23** on the gate set and 38 to **11** on the subset. Those events have **no
emission overlapping them at all** -- the detector's first word came after the
event had ended, and the only reason the alarm range touched the event was that
`error_buffer` widened it backwards into the past. **Fifteen of 38 events on the
gate set were caught by the dilation and not by the detector.** Their lead is not
-43; it is worse than that and the median over the surviving 23 understates it.

**What this does not do.** It does not change any detection count: recall stays
38/46 and 27/46, and `event_recall`, precision, MVGS and the rare-event rate are
untouched. Only the lead-time reading moves, and both readings are kept.

**What it does do.** Objective.md 2's claim is warning *before* an event.
Measured from a moment that exists, **this project does not currently warn
early** -- on either channel set, with either decision layer. The +26 was the
batching latency counted backwards.

**And it makes the metric usable again.** Every detector is now treated
identically: those that do not dilate fall through to their alarm mask unchanged,
because for them a crossing is the emission. `lstm-quantile`'s -122 and `mavg`'s
0 were never comparable with `lstm-telemanom`'s +26; under this reading they are.
D9's disqualifications should be revisited on the honest figures before the
architecture gate, and that is **not** done here.

## 6g. The frozen decision layer, settled: `lstm-quantile`

`docs/DECISIONS.md` **D25**, superseding D24. Both detectors on post-fix weights,
honest lead-time reading, both channel sets. Artifacts
`runs/m1-g8.9.10/lstm-{whitened,quantile}/2026-08-28T1*Z-*.json` and
`runs/m1-g8.9.10/_forensics/2026-08-28T184844Z-head-to-head.json`.

| `m1-g8.9.10` | F0.5 | recall | MVGS | precision | rare-FA | nom-step FA | honest lead |
|---|---|---|---|---|---|---|---|
| `lstm-whitened` | 0.848 | 27/46 | **21/32** | 41/43 | 2/48 | 0.028% | **-41.5** |
| **`lstm-quantile`** | 0.838 | 26/46 | **21/32** | 40/42 | 2/48 | **0.002%** | **+0.0** |

| `m1-ss5` | F0.5 | recall | MVGS | precision | rare-FA | nom-step FA | honest lead |
|---|---|---|---|---|---|---|---|
| `lstm-whitened` | 0.853 | 23/42 | 18/31 | 118/119 | 2/48 | 0.319% | **-51.0** |
| **`lstm-quantile`** | **0.885** | **27/42** | **21/31** | 127/130 | 2/48 | 0.293% | **+0.0** |

**The detections are nested on both sets, in opposite directions, with no
crossing.** On `m1-g8.9.10` quantile is a strict subset of whitened, differing by
one event (`id_132`); on `m1-ss5` whitened is a strict subset of quantile,
differing by four (`id_149`, `id_165`, `id_186`, `id_187`). **All five have a
footprint of 1.** The rules are ordered rather than complementary, so an
OR-combination equals the superset and buys nothing.

**(!) And the finding that outranks the choice.** `lstm-quantile` is **also
channel-blind** -- a maximum over per-channel smoothed errors against one global
threshold -- and it reaches the **same 2/48** the relationship test does.

> **The false-alarm gain came from having a high global threshold, not from
> testing the relationship. Whitening bought nothing measurable that a global
> quantile did not.**

`docs/DECISIONS.md` D23's observation -- the decision layer cannot see
relationships -- is unchanged and still true. The causal claim built on it, that
the false-alarm problem lives in that gap, is refuted. The false alarms came from
telemanom's **local** threshold adapting downward until it fired constantly, and
any sufficiently high global cut removes them.

**What this does not fix.** `lstm-quantile`'s honest median lead is **+0.0**: it
fires *at* the labelled event boundary, not before it. The metric is no longer
negative, which is not the same as warning early. Objective.md 1.1 stands.

## 6h. Work item 5: the GRU beside the LSTM

`docs/MODELS.md` section 14 (pre-registered 2026-08-28, commit `4407988`,
before any fit) and `docs/DECISIONS.md` D26. **The cell is the only variable**:
`Hyper(cell="gru")`, every other value the LSTM's, the frozen `lstm-quantile`
decision layer (D25) unchanged. Fitted on a rented A40 slice (six fits, 6.6
minutes, the `m1-ss5` fold-0 refit bit-identical), scored on the M5 through the
NumPy reference with refits refused. Artifacts
`runs/m1-g8.9.10/gru-quantile/2026-08-28T222635Z-6d146f5d.json`,
`runs/m1-g8.9.10/gru-telemanom/2026-08-28T222635Z-a428c1d6.json`,
`runs/_weights_pod/gru-2026-08-28/fit_report.json`; per event,
`runs/m1-g8.9.10/_forensics/2026-08-28T2*Z-head-to-head.json` (section 6h.3).
15 Class B, 1 Class A per run.

### 6h.1 The gate row

**GATE -- `m1-g8.9.10`**

| Detector | **F0.5** | recall | **MVGS** | precision | **rare-FA** | nom-step FA | point | VUS-PR | **honest lead** |
|---|---|---|---|---|---|---|---|---|---|
| `lstm-quantile` | **0.838** | 26/46 | 21/32 | **40/42** | 2/48 | 0.002% | 5/11 | 0.343 | +0.0 (n=26) |
| **`gru-quantile`** | 0.804 | **27/46** | **22/32** | 139/157 | **1/48** | **0.001%** | 5/11 | **0.374** | +0.0 (n=27) |

**`m1-ss5`**

| Detector | **F0.5** | recall | **MVGS** | precision | **rare-FA** | nom-step FA | point | VUS-PR | **honest lead** |
|---|---|---|---|---|---|---|---|---|---|
| `lstm-quantile` | **0.885** | **27/42** | 21/31 | **127/130** | **2/48** | 0.293% | 6/11 | **0.604** | +0.0 (n=27) |
| **`gru-quantile`** | 0.593 | 26/42 | 21/31 | 101/172 | 3/48 | **0.014%** | 5/11 | 0.591 | +0.0 (n=26) |

Point recall is over eleven events and cannot distinguish detectors; treat it
as a coverage check. Honest lead is the crossing itself on this path (no
dilation), median over caught events; 13 of 27 and 6 of 26 are negative.

**Pooled, the two cells are a wash on the gate set** -- one more event, one
more headline-cell event, one fewer rare-event false alarm, half the
nominal-step rate, and a lower F0.5 because precision fell from 40/42 to
139/157 alarm ranges. **Per fold they are not a wash at all**, and the pooled
row hides two opposite movements.

### 6h.2 Per fold: the floor fell on fold 0 and rose on fold 1

Threshold is the fold's noise floor -- the 99.9th percentile of the
anomaly-masked fitting window's scores -- and it is what the pre-registration
(P17) said would move by a quarter at most.

**`m1-g8.9.10`**, GRU against LSTM:

| fold | val-MSE GRU / LSTM | threshold GRU / LSTM | recall | MVGS | precision | rare-FA | nom-step FA | honest lead | GRU epochs (best) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.628e-6 / 1.691e-4 (**0.03x**) | 0.01324 / 0.14681 (**0.09x**) | **11/15** vs 4/15 | **8/11** vs 4/11 | 13/31 vs 7/8 | 1/12 vs 1/12 | 79 vs 0 of 3,569,953 | +0.0 (n=11) vs -77.0 (n=4) | 35 (34) vs 14 (3) |
| 1 | 3.378e-6 / 3.552e-6 (0.95x) | 0.02827 / 0.01563 (**1.81x**) | 4/15 vs **10/15** | 4/9 vs **7/9** | 114/114 vs 21/22 | 0/21 vs 1/21 | 0 vs 69 of 3,515,149 | -123.0 (n=4) vs +0.0 (n=10) | 35 (26) vs 35 (28) |
| 2 | 3.468e-6 / 4.089e-6 (0.85x) | 0.01455 / 0.01178 (1.24x) | 12/16 vs 12/16 | 10/12 vs 10/12 | 12/12 vs 12/12 | 0/15 vs 0/15 | 63 vs 145 of 3,590,386 | +0.0 (n=12) vs +0.0 (n=12) | 29 (18) vs 32 (21) |

**`m1-ss5`**:

| fold | val-MSE GRU / LSTM | threshold GRU / LSTM | recall | MVGS | precision | rare-FA | nom-step FA | honest lead | GRU epochs (best) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4.968e-6 / 3.441e-5 (**0.14x**) | 0.00878 / 0.10195 (**0.09x**) | **11/13** vs 4/13 | **8/10** vs 4/10 | 19/89 vs 13/15 | 2/12 vs 1/12 | 958 vs 121 of 3,644,798 | +0.0 (n=11) vs -41.0 (n=4) | 35 (32) vs 22 (11) |
| 1 | 5.169e-6 / 5.205e-6 (0.99x) | 0.01710 / 0.01243 (**1.38x**) | 3/14 vs **11/14** | 3/9 vs **7/9** | 70/71 vs 102/102 | 1/21 vs 1/21 | 118 vs 31,607 of 3,608,842 | -1.0 (n=3) vs +0.0 (n=11) | 35 (27) vs 33 (22) |
| 2 | 6.120e-6 / 8.096e-6 (0.76x) | 0.01022 / 0.01545 (0.66x) | 12/15 vs 12/15 | 10/12 vs 10/12 | 12/12 vs 12/13 | 0/15 vs 0/15 | 460 vs 101 of 3,622,049 | +0.0 (n=12) vs +0.0 (n=12) | 24 (13) vs 35 (33) |

Every fold denominator is under 20 and UNDERPOWERED; the per-fold rows are
where the mechanism shows, not where a ranking is read.

**Fold 0.** The LSTM's fold-0 fit stopped at epoch 14 with the best at 3 and a
validation MSE 40x worse than its other folds; the GRU ran to the 35-epoch cap
with its best at 34 and reached the same MSE as folds 1 and 2. Its noise floor
fell **elevenfold**, and seven events came back on each set -- recall 4/15 to
11/15 and 4/13 to 11/13 -- at the price of 31 alarm ranges where the LSTM had 8,
and 79 nominal-step alarms where the LSTM had none. Honest lead on those events
is +0.0 against the LSTM's -77.0.

**Fold 1.** Validation MSE is the same to 5%, and the **noise floor is 1.8x
higher**. Six events lost on the gate set, eight on `m1-ss5`, and the four still
caught are caught late (-123.0). The 99.9th percentile of the GRU's nominal
residual is heavier-tailed on this fold than the LSTM's while its mean square is
not: **the noise floor is a tail statistic and the validation loss is a mean,
and the two moved independently.** D17's MEASURED section said the same about
the LSTM's own residual -- excess kurtosis in the thousands -- and the gate has
now measured it between two cells.

**Fold 2.** Identical detections. The floor 1.24x higher and 0.66x lower on
the two sets, and nothing that mattered moved.

### 6h.3 Per event -- the headline

Artifact `runs/m1-g8.9.10/_forensics/2026-08-28T223610Z-head-to-head.json`
(`lstm-quantile` vs `gru-quantile`, cached weights, weight store unchanged at
69, 15 Class B). Reach is the peak score in the event span over the fold's
threshold; `needed` is the LSTM's threshold over the GRU's, the factor an
event at LSTM reach `r` required the floor to fall by (`1/r`) against the
factor it fell (14.4, condition 2).

**`m1-g8.9.10`** -- both 20, only `lstm-quantile` 6, only `gru-quantile` 7,
neither 13. **The seven the GRU recovers are seven of the eleven**, every one
on fold 0, every one by the floor:

```
  fold  event    cell                               fp      LSTM reach -> GRU   floor fell   needed
  0     id_107   Multivariate/Global/Subsequence      32     0.158 -> 1.286      11.1x        6.3x
  0     id_109   Multivariate/Global/Point             1     0.172 -> 1.271      11.1x        5.8x
  0     id_110   Multivariate/Global/Point             1     0.181 -> 1.233      11.1x        5.5x
  0     id_114   Multivariate/Global/Point             1     0.147 -> 1.275      11.1x        6.8x
  0     id_12    Multivariate/Global/Subsequence   11937     0.168 -> 1.269      11.1x        5.9x
  0     id_90    Multivariate/Global/Subsequence      27     0.168 -> 1.269      11.1x        5.9x
  0     id_93    Multivariate/Global/Subsequence      63     0.150 -> 1.249      11.1x        6.7x
```

Four of the seven are headline-cell events; that is 21/32 to 22/32 net of the
losses below. The eleven's other four did not return: `id_89` (fold 0, 0.111 ->
0.393 -- the floor fell 11x but the event's own peak fell with it), `id_138`
(0.480 -> 0.181), `id_20` (0.209 -> 0.130) and `id_157` (0.307 -> 0.238), the
last three on folds whose floor rose. `id_132`, the coin toss at 0.995, landed
at 0.553 and is scored neither way, as declared.

**The six the GRU loses are all on fold 1, and all by the floor too**, in the
other direction: `id_122, id_124, id_129, id_130, id_140, id_142`, at LSTM
reach 1.087-1.207, land at 0.596-0.616 -- the same peaks divided by a threshold
1.81x higher. Two are headline-cell.

**`m1-ss5`** -- both 19, only `lstm-quantile` 8, only `gru-quantile` 7,
neither 8. **The same seven return**, on fold 0, at reach 1.873-1.917 against
0.175-0.208, the floor having fallen 11.6x where 4.8x-5.7x was needed. Eight
are lost on fold 1: the six above plus `id_132` (1.269 -> 0.914) and `id_145`
(2.055 -> 0.884), against a floor 1.38x higher. `id_89` (0.139 -> 0.470),
`id_121`, `id_138` and `id_157` stay missed.

**P19 refuted, and mechanism condition 2 satisfied on every recovery.** The
prediction was *none of the eleven returns*; seven did, twice. The condition
was that a recovery happens only on a fold whose floor fell by what the reach
required; every recovered event is on fold 0 and the floor fell by 11x against
a requirement of 4.8x-6.8x, so **the recoveries are the stated mechanism** --
a lower noise floor -- and not something else. What the pre-registration got
wrong was not the mechanism but its size: it predicted the floor would move by
a quarter and it moved by an order of magnitude on the one fold where the
LSTM's fit had stalled, and rose on the fold where it had not. **Trigger 2
fired.**

### 6h.4 `gru-telemanom`, reported and not the gate

telemanom's dynamic threshold and `error_buffer` on the GRU forecast, beside
`lstm-telemanom` (section 6e). Same weights as `gru-quantile`, no extra fit.

| `m1-g8.9.10` | F0.5 | recall | MVGS | precision | rare-FA | nom-step FA | lead as reported | honest lead |
|---|---|---|---|---|---|---|---|---|
| `lstm-telemanom` | 0.026 | 38/46 | 28/32 | 75/3,548 | 30/48 | 4.972% | +26.0 (n=38) | -43.0 (n=23) |
| `gru-telemanom` | 0.019 | 40/46 | **29/32** | 97/6,206 | **39/48** | **8.782%** | +26.0 (n=40) | -43.0 (n=24) |

| `m1-ss5` | F0.5 | recall | MVGS | precision | rare-FA | nom-step FA | lead as reported | honest lead |
|---|---|---|---|---|---|---|---|---|
| `lstm-telemanom` | 0.035 | 38/42 | 29/31 | 42/1,475 | 33/48 | -- | +26.5 (n=38) | -53.0 (n=11) |
| `gru-telemanom` | 0.019 | 39/42 | **30/31** | 43/2,801 | **40/48** | 3.987% | +26.0 (n=39) | -53.5 (n=10) |

D17 said the collapse "returns every time the model improves"; it has. A
better forecast under `mu + 2.5 sigma` on a 2,170-sample window produces
**6,206 alarm ranges** and alarms on 39 of 48 commanded manoeuvres. Recorded as
the reproduction reading; nothing about the gate is read from it.

### 6h.5 The pre-registration, adjudicated

`docs/MODELS.md` 14.8 carries every prediction beside its outcome. In brief:
P15 (71,160 parameters) held; **P16 and P17 refuted on fold 0 of both sets**
-- the forecast improved 36x and 7x and the floor fell 11x, against a predicted
band of 0.7x-1.4x and 0.8x-1.25x -- and held on folds 1 and 2 for the MSE
while the floor left the band upward on fold 1; **P18 held** on all four axes
(MVGS 22/32, rare-FA 1/48 and 3/48, nominal-step 0.001%, honest lead +0.0);
P19, the per-event prediction, is adjudicated in 6h.3; P20 held. The outcome
condition (parity: MVGS >= 21/32, rare-FA <= 2/48) **passed**. The mechanism
condition **failed** where it was designed to be able to: nominal-step alarms
on the test window exceeded 3x the LSTM's on fold 0 of both sets (79 against 0;
958 against 121) and on `m1-ss5` fold 2 (460 against 101). The fold-0
recoveries were bought partly with alarms that 48 rare events cannot see and
3.6 million nominal steps can.

**Two stop-and-report rules fired**: the GRU's headline-cell recall meets the
LSTM's (22/32 against 21/32), and events the pre-registration predicted would
not return did (6h.3). Both are reported here and nothing further is decided.

### 6h.6 What is not claimed

Not that the GRU is the better cell: pooled F0.5 is lower on both sets, and
on `m1-ss5` by 0.3. Not that the fold-0 recovery is the cell rather than the
fit -- the LSTM's fold-0 fit stopped at epoch 14 with its best at 3, and a
refit of the LSTM on fold 0 with a different seed has not been run; that is the
obvious control and it is one pod fit. Not that lead time improved: +0.0 is
the boundary, on both cells. And every figure remains telemanom-minus-commands.

## 6i. The LSTM's fold 0, reseeded: not the stall, and still not the GRU

`docs/MODELS.md` section 15, pre-registered (commit `664b1c6`) before the fit.
`lstm-quantile` with `Hyper(seed=1)` and nothing else changed, fold 0 of both
sets, fitted on the M5 and scored through `harness._score_fold`. Artifact
`runs/m1-g8.9.10/_forensics/2026-08-28T232538Z-reseed-lstm-quantile-seed1.json`.
15 Class B, 1 Class A. **No published row moves**; the gate row remains the
seed-0 fit.

| fold 0 | val-MSE | epochs (best) | floor | recall | MVGS | precision | rare-FA | nominal-step | honest lead |
|---|---|---|---|---|---|---|---|---|---|
| `m1-g8.9.10` LSTM seed 0 (the row) | 1.691e-4 | 14 (3) | 0.14681 | 4/15 | 4/11 | 7/8 | 1/12 | 0 / 3,569,953 | -77.0 (n=4) |
| `m1-g8.9.10` LSTM seed 1 | 1.986e-5 | 35 (34) | 0.02220 | 4/15 | 4/11 | 7/18 | 2/12 | 0 | -4.0 (n=4) |
| `m1-g8.9.10` GRU | 4.628e-6 | 35 (34) | 0.01324 | 11/15 | 8/11 | 13/31 | 1/12 | 79 | +0.0 (n=11) |
| `m1-ss5` LSTM seed 0 (the row) | 3.441e-5 | 22 (11) | 0.10195 | 4/13 | 4/10 | 13/15 | 1/12 | 121 / 3,644,798 | -41.0 (n=4) |
| `m1-ss5` LSTM seed 1 | 3.462e-5 | 22 (11) | 0.10172 | 4/13 | 4/10 | 13/15 | 1/12 | 121 | -41.0 (n=4) |
| `m1-ss5` GRU | 4.968e-6 | 35 (32) | 0.00878 | 11/13 | 8/10 | 19/89 | 2/12 | 958 | +0.0 (n=11) |

All fold-0 denominators are under 20, UNDERPOWERED.

**The seven weak events the GRU recovered, under the reseeded LSTM: none
caught, at reach 0.80-0.93 on the gate set** (`id_109` 0.933, `id_90` 0.921,
`id_12` 0.921, `id_93` 0.917, `id_114` 0.890, `id_110` 0.870, `id_107` 0.796)
and 0.17-0.21 on `m1-ss5`. The banked fit had them at 0.15-0.18.

**Three things the run settles.** The banked fold-0 stall (best epoch 3,
stopped at 14) was that one optimisation path: a second seed ran to the cap and
landed 8.5x better with the floor 6.6x lower. A second LSTM path is still 4.3x
worse than the GRU in MSE and 1.7x higher in floor on this fold, and on `m1-ss5`
the second path reproduced the first to three significant figures -- stopped
at epoch 22, best at 11, floor 0.1017 against 0.1020. And the seven sit just
under the escaped path's bar: a floor away, not a cell away.

**Why, from the training curves.** The gate-set history is a plateau at
1.2e-04-3.6e-04 from epoch 6 to 23 and a fall to 2.4e-5 at epoch 24. The
published `patience = 10` ended seed 0 at epoch 14 on that plateau; seed 1's
small improvements kept resetting it until the fall. `m1-ss5` never left its
plateau on either seed. **The LSTM's fold-0 optimisation is plateau-prone under
the published protocol; the GRU's on the same data was not** (best epoch at the
cap, 4.6e-6 and 5.0e-6).

**The pre-registered verdict rule returned no verdict** (MODELS.md 15.8, P25's
middle branch): the reseed neither stalled nor reached the GRU. What follows
from it is a decision and is recorded as open: whether the gate row's fold 0
should be the seed-1 fit (same detections, floor 0.022, honest lead -4.0
against -77.0), and whether `patience` and the epoch cap -- published constants
that bind both cells -- are the right protocol on the data-poor fold. Neither
is done here. Trigger 3 from section 6h stands as noted.

## 6j. Work item 6: the TCN beside both cells

`docs/MODELS.md` section 16 (pre-registered 2026-08-29, commit `3f6e6fa`,
before any fit) and `docs/DECISIONS.md` D27. **The architecture is the only
variable**: `Hyper(cell="tcn", hidden=(50,)*6, kernel=3)` -- six residual
blocks of causal dilated convolutions, receptive field 253, **91,670
parameters** against the LSTM's 91,640 -- every other field the LSTM's, the
frozen `lstm-quantile` decision layer (D25) unchanged, stateless by contract.
Fitted on a rented A40 slice (six fits, 3.5 minutes of GPU, the `m1-ss5`
fold-0 refit bit-identical under cuDNN), scored on the M5 through the NumPy
reference with refits refused. Artifacts
`runs/m1-g8.9.10/tcn-quantile/2026-08-29T162030Z-c48bd47d.json`,
`runs/m1-g8.9.10/tcn-telemanom/2026-08-29T162030Z-4c35b17a.json`,
`runs/_weights_pod/tcn-2026-08-29/fit_report.json`; per event, section 6j.3.
15 Class B, 1 Class A per run.

### 6j.1 The gate row

**GATE -- `m1-g8.9.10`**

| Detector | **F0.5** | recall | **MVGS** | precision | **rare-FA** | nom-step FA | point | VUS-PR | **honest lead** |
|---|---|---|---|---|---|---|---|---|---|
| `lstm-quantile` | **0.838** | 26/46 | 21/32 | **40/42** | 2/48 | 0.002% | 5/11 | 0.343 | +0.0 (n=26) |
| `gru-quantile` | 0.804 | **27/46** | **22/32** | 139/157 | **1/48** | 0.001% | 5/11 | **0.374** | +0.0 (n=27) |
| **`tcn-quantile`** | 0.411 | 9/46 | **9/32** | 21/37 | 3/48 | **0.00002%** | 0/11 | 0.367 | **-4.0** (n=9) |

**`m1-ss5`**

| Detector | **F0.5** | recall | **MVGS** | precision | **rare-FA** | nom-step FA | point | VUS-PR | **honest lead** |
|---|---|---|---|---|---|---|---|---|---|
| `lstm-quantile` | **0.885** | **27/42** | **21/31** | **127/130** | **2/48** | 0.293% | 6/11 | 0.604 | +0.0 (n=27) |
| `gru-quantile` | 0.593 | 26/42 | **21/31** | 101/172 | 3/48 | **0.014%** | 5/11 | 0.591 | +0.0 (n=26) |
| **`tcn-quantile`** | 0.649 | 16/42 | 13/31 | 108/137 | 3/48 | 0.297% | 3/11 | **0.607** | -0.5 (n=16) |

Point recall is over eleven events; treat it as a coverage check. Honest lead
is the crossing itself on this path; 9 of 9 and 8 of 16 are negative.

**The TCN catches nine of forty-six on the gate set, against twenty-six and
twenty-seven.** It is quieter than either cell -- two nominal-step alarms in
ten million, 21/37 alarm ranges -- and it is quiet because its noise floor is
higher than both incumbents' on every fold but the LSTM's stalled fold 0. The
gate reads it as the third of three; trigger 3 did not fire.

### 6j.2 Per fold: a higher floor everywhere, and one stalled fit

**`m1-g8.9.10`**, TCN against GRU and LSTM (threshold is the fold's noise floor):

| fold | val-MSE TCN / GRU / LSTM | threshold TCN / GRU / LSTM | recall T / G / L | MVGS T / G / L | precision T | rare-FA T | nom-step T / G / L | honest lead T | TCN epochs (best) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.972e-5 / 4.628e-6 / 1.691e-4 | **0.02152** / 0.01324 / 0.14681 | 4/15 / 11/15 / 4/15 | 4/11 / 8/11 / 4/11 | 7/22 | 2/12 | 0 / 79 / 0 | -4.0 (n=4) | 35 (33) |
| 1 | 5.620e-6 / 3.378e-6 / 3.552e-6 | **0.03899** / 0.02827 / 0.01563 | 4/15 / 4/15 / 10/15 | 4/9 / 4/9 / 7/9 | 13/14 | 1/21 | 1 / 0 / 69 | -24.5 (n=4) | 35 (34) |
| 2 | 1.356e-5 / 3.468e-6 / 4.089e-6 | **0.02564** / 0.01455 / 0.01178 | **1/16** / 12/16 / 12/16 | 1/12 / 10/12 / 10/12 | 1/1 | 0/15 | 1 / 63 / 145 | -1.0 (n=1) | 26 (15) |

**`m1-ss5`**:

| fold | val-MSE TCN / GRU / LSTM | threshold TCN / GRU / LSTM | recall T / G / L | MVGS T / G / L | precision T | rare-FA T | nom-step T / G / L | honest lead T | TCN epochs (best) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.501e-6 / 4.968e-6 / 3.441e-5 | 0.01420 / 0.00878 / 0.10195 | **11/13** / 11/13 / 4/13 | **8/10** / 8/10 / 4/10 | 19/48 | 2/12 | 788 / 958 / 121 | +0.0 (n=11) | 35 (33) |
| 1 | 6.437e-6 / 5.169e-6 / 5.205e-6 | 0.01908 / 0.01710 / 0.01243 | 5/14 / 3/14 / 11/14 | 5/9 / 3/9 / 7/9 | 89/89 | 1/21 | 31,473 / 118 / 31,607 | -1.0 (n=5) | 35 (28) |
| 2 | **6.059e-5** / 6.120e-6 / 8.096e-6 | **0.05238** / 0.01022 / 0.01545 | **0/15** / 12/15 / 12/15 | 0/12 / 10/12 / 10/12 | 0/0 | 0/15 | 0 / 460 / 101 | -- (n=0) | **13 (2)** |

Every fold denominator is under 20 and UNDERPOWERED.

**The floor followed the fit, and the fit was worse.** On the gate set the
TCN's validation MSE is 4.3x, 1.7x and 3.9x the GRU's and its floor 1.6x,
1.4x and 1.8x -- higher on every fold, in the same direction as the MSE on
every fold (P32 held). The forecast is worse and the residual's tail is
heavier with it, and a floor that is a tail quantile rises. That is the
opposite of section 6h's fold-1 finding for the GRU, where MSE and floor
decoupled; for this architecture they moved together.

**One fit stalled.** `m1-ss5` fold 2 stopped at epoch 13 with its best at 2,
validation MSE 6.06e-5 -- 9.9x the GRU's and 7.5x the LSTM's -- after a
history of 1.7e-4, 7.0e-5, 6.1e-5 and then eleven epochs between 6.5e-5 and
9.2e-5. Its floor of 0.0524 catches nothing: 0/15. The pre-registration's
P27 -- *no plateau, the convolutional stack trains clean* -- is refuted on
that fold, and section 6i's trainability finding now has an instance on the
TCN as well as the LSTM. The other five fits ran to or near the cap without
one.

**Fold 0 of the gate set is the reseeded LSTM again.** 1.97e-5 against seed
1's 1.99e-5, a floor of 0.0215 against 0.0222, 4/15 against 4/15: two
architectures, two seeds, one number. The GRU's 4.6e-6 and 0.0132 on the same
fold remain the only fit that got below the seven's peaks.

### 6j.3 Per event

Artifacts `runs/m1-g8.9.10/_forensics/2026-08-29T162907Z-head-to-head.json`
(`lstm-quantile` vs `tcn-quantile`) and `.../2026-08-29T162813Z-head-to-head.json`
(`gru-quantile` vs `tcn-quantile`); cached weights, weight store unchanged at
77, 15 Class B each. Reach is the peak score in the event span over the fold's
threshold.

**`m1-g8.9.10`: the TCN's nine are a strict subset of both incumbents' --
only-TCN 0 against both.** None of the twelve returns:

```
  fold  event    LSTM reach   GRU reach   TCN reach
  0     id_109   0.172        1.271       0.951      the seven, at 0.82-0.95 of a floor of 0.0215:
  0     id_90    0.168        1.269       0.940      the reseeded LSTM's 0.80-0.93 at 0.0222, again.
  0     id_12    0.168        1.269       0.940      Their raw peaks under the TCN, 0.018-0.020,
  0     id_93    0.150        1.249       0.927      are the GRU's 0.016-0.017 and the reseed's
  0     id_110   0.181        1.233       0.908      0.018-0.021; only the GRU's floor got under them
  0     id_114   0.147        1.275       0.868
  0     id_107   0.158        1.286       0.823
  0     id_89    0.111        0.393       0.556
  1     id_132   0.995        0.553       0.438
  1     id_138   0.480        0.181       0.291
  1     id_20    0.209        0.130       0.122
  2     id_157   0.307        0.238       0.258
```

Against the LSTM the TCN loses seventeen: the GRU's six on fold 1
(`id_122, id_124, id_129, id_130, id_140, id_142`, LSTM reach 1.09-1.21, TCN
0.43-0.52 -- a floor 2.5x higher) and **eleven on fold 2** (`id_149, id_150,
id_160, id_165, id_172, id_176, id_177, id_183, id_184, id_186, id_187`, LSTM
reach 1.38-1.48, TCN 0.67-0.85 -- a floor 2.2x higher). Against the GRU it
loses eighteen: the seven and the same eleven. **Trigger 2 did not fire on
the gate set.**

**`m1-ss5`: the seven return -- the same seven, at reach 1.17-1.23 against a
floor of 0.0142** (`id_107` 1.234, `id_109` 1.217, `id_12` 1.215, `id_90`
1.215, `id_93` 1.193, `id_110` 1.174, `id_114` 1.174; LSTM 0.17-0.21). That
floor is below their raw peaks under the LSTM (0.018-0.021), so 16.4's
condition 4 holds: **a recovery by the floor, on the one fold-and-set where
the TCN's floor got under them**, and the same events the GRU recovered
(14.8.2) -- nothing the GRU did not already reach. `id_89` (0.546),
`id_138`, `id_157` and `id_132` (0.870) do not return.

Two more only-TCN catches on `m1-ss5`, both fold 1, both **not by the
floor**: `id_142` (GRU 0.986 -> 1.022, the bar's edge) and `id_145` (GRU
0.884 -> **3.491**, footprint 6,300). The TCN's floor on that fold is above
the LSTM's, so these are residuals that grew under the event -- a forecast
effect, reported as such and not as support for the mechanism. And on
`m1-ss5` fold 2 the stalled fit loses all twelve the incumbents catch
(`id_149 ... id_187`, TCN reach 0.34-0.39 at a floor of 0.052).

**P31, adjudicated.** *Five to seven of the seven return on both sets*:
**refuted on the gate set (0 of 7), held on `m1-ss5` (7 of 7)**. *`id_89`,
`id_138`, `id_20`, `id_157` do not return*: **held**. *`id_132`*: neither way,
missed. *At least four of the GRU's six fold-1 losses caught*: **refuted** --
none on the gate set, one (`id_142`) on `m1-ss5`. **Trigger 2: fired on
`m1-ss5` by the letter -- seven of the eleven -- and they are the GRU's seven,
recovered by the same mechanism on the same fold; not fired on the gate set,
where the twelve are defined.**

### 6j.4 `tcn-telemanom`, reported and not the gate

| set | Detector | F0.5 | recall | MVGS | precision | rare-FA | nom-step FA | lead as reported | honest lead |
|---|---|---|---|---|---|---|---|---|---|
| `m1-g8.9.10` | `lstm-telemanom` | 0.026 | 38/46 | 28/32 | 75/3,548 | 30/48 | 4.972% | +26.0 (n=38) | -43.0 (n=23) |
| | `gru-telemanom` | 0.019 | 40/46 | 29/32 | 97/6,206 | 39/48 | 8.782% | +26.0 (n=40) | -43.0 (n=24) |
| | `tcn-telemanom` | 0.026 | **41/46** | **30/32** | 85/4,085 | 29/48 | 5.559% | +26.0 (n=41) | -43.0 (n=28) |
| `m1-ss5` | `lstm-telemanom` | 0.035 | 38/42 | 29/31 | 42/1,475 | 33/48 | -- | +26.5 (n=38) | -53.0 (n=11) |
| | `gru-telemanom` | 0.019 | 39/42 | 30/31 | 43/2,801 | 40/48 | 3.987% | +26.0 (n=39) | -53.5 (n=10) |
| | `tcn-telemanom` | 0.059 | 37/42 | 28/31 | 41/852 | 31/48 | 1.096% | +29.0 (n=37) | -53.0 (n=11) |

Under telemanom's local threshold the TCN's residuals carry **30 of 32**
headline-cell events -- the most of the three -- at 4,085 alarm ranges and
29/48. The signal is in the residual; the frozen global quantile is what does
not reach it on this architecture. Recorded as the reproduction reading;
nothing about the gate is read from it.

### 6j.5 The pre-registration, adjudicated

`docs/MODELS.md` 16.7 carries every prediction beside its outcome. In brief:
P26 held (91,670); **P27 refuted** (one stalled fold, and fold 0 at 1.97e-5
against a predicted <= 1e-5); **P28 refuted** on four of six folds (the MSE
1.5x to 9.9x the GRU's, three of them past 2x); **P29 refuted** on five of six
(floors above the band on every fold but `m1-ss5` fold 1); **P30 refuted** --
MVGS 9/32 against a predicted 21-25; P31 adjudicated in 6j.3; **P32 held on
all six folds** -- the one prediction written as most likely to be wrong. The
outcome condition (parity with the better incumbent) **failed**. Mechanism
conditions: no-plateau **failed** (`m1-ss5` fold 2); floor-tracks-fit
**held**; nominal-step **held**; recovery-by-floor is read in 6j.3.

**Stop rules 4 and 5 fired on the training reports and were reported before
scoring; trigger 3 did not fire; trigger 2 is read in 6j.3.** Nothing is
decided here. The three rows now exist and the gate is a decision for the
record.

### 6j.6 What is not claimed

Not that a TCN cannot forecast this telemetry: under telemanom's local rule
its residuals carry 30 of 32 headline events. Not that the shape is the only
shape -- 253 steps and 50 channels were chosen to match the LSTM's lookback
and size, and a wider or deeper stack was not tried, by design. Not that the
stall on `m1-ss5` fold 2 is the architecture rather than the seed -- section
6i's lesson applies and one refit would say. And every figure remains
telemanom-minus-commands.

## 6k. Phase 1 closure: the held-back sets, scored once

`docs/MODELS.md` section 18 (pre-registered, commit `0c5fbfb`) and section
18.8; `docs/DECISIONS.md` D28 (the gate, decided before the seal broke) and
D29 (the deployment configuration, decided on these numbers). Each set was
loaded once, through `python -m sentinel_eval run`, with the section-18.1
commands verbatim: no `describe`, no sweep, no second attempt, nothing
refitted. Settings as frozen: the published protocol per cell, seed 0 + fold,
the D25 decision layer with each detector's own label-free 99.9th percentile
of its own anomaly-masked fitting window. Artifacts
`runs/m2-ss1/<detector>/2026-08-29T204415Z-*.json` (git `0c5fbfb`, 15 Class B,
2 Class A) and `runs/m1-g3/<detector>/2026-08-29T223625Z-*.json` (11 Class B, 2 Class A).

### 6k.1 `m2-ss1` -- the adoption number on a spacecraft nothing here was tuned on

Mission 2, `subsystem_1` groups 5/8/9/10, 12 channels (`channel_9`-`channel_20`),
**6,129,598 steps at 18 s** (timesteps, never hours). One chronological fold:
train the first 1,838,879 steps (1,835,997 usable after 2,882 anomaly steps
and 179 rare events are removed), test the last 4,290,719, holding **424 rare
nominal events** and 3 anomalies. Recall is disabled on this task by design;
the scorecard is the adoption number and the nominal-step rate.

| Detector | **rare-event FA** | nominal-step FA | alarm ranges on nominal time | threshold | training (epochs, best, val-MSE) |
|---|---|---|---|---|---|
| `rstd` | 84/424 (0.198) | **718,831 / 4,155,841 (17.30%)** | 73 | 3.690 | -- |
| `mavg` | **122/424 (0.288)** | 0 / 4,155,841 | 0 | 25.14 | -- |
| `lstm-quantile` | **4/424 (0.009)** | **0** | 0 | 0.2100 | 35, 29, 6.218e-5 |
| `gru-quantile` | **4/424 (0.009)** | **0** | 0 | 0.4368 | 19, 8, 7.161e-5 |
| `tcn-quantile` | 6/424 (0.014) | **0** | 0 | 0.3085 | 26, 15, 6.889e-5 |
| `lstm-gru-or` | 8/424 (0.019) | **0** | 0 | 1.0 | its members' |

Resolution 1/424 = 0.24%. The three forecasters alarm on **one rare event in
a hundred and on no nominal timestep**; the per-channel floors do not
transfer -- `rstd` alarms on a sixth of nominal time and a fifth of the rare
events, `mavg` on more than a quarter. **The union's 8 is exactly 4 + 4: the
two cells' rare alarms on Mission 2 are disjoint**, where on Mission 1 the
GRU's one sat inside the LSTM's two. The thresholds are 10-30x the Mission-1
floors because Mission 2's groups are scaled differently and its grid period
is 18 s; a floor is a quantile of that spacecraft's own residual, and the
comparison that matters is within the row, not across missions. No fit
stalled; the GRU stopped earliest (19 epochs, best 8) and the LSTM fitted
best -- the reverse of Mission 1's fold 0.

### 6k.2 `m1-g3` -- the recall exam, underpowered by construction

Mission 1 `subsystem_6` group 3, 8 channels (`channel_12, 13, 19, 20, 27, 28,
36, 37`), 14,728,319 steps at 30 s, forward chaining at 25% seed in three
folds. Test-side: **11 anomalies, 10 headline-cell, 13 rare events** (folds
6 / 4 / 1, 6 / 4 / 0, 3 / 8 / 2) -- four of the nomination's fourteen headline
events fell in the seed window. **Every denominator is under 20; every figure
here is UNDERPOWERED.**

| Detector | **F0.5** | recall | **MVGS** | precision | **rare-FA** | **nominal-step FA** | honest lead |
|---|---|---|---|---|---|---|---|
| `rstd` | 0.029 | 3/11 | 3/10 | 12/499 | 0/13 | 332,245 / 10,881,882 (3.05%) | -45.0 (n=3) |
| `mavg` | 0.021 | 5/11 | 5/10 | 58/3,367 | 0/13 | 270,203 (2.48%) | -1,272.0 (n=5) |
| `lstm-quantile` | 0.679 | 10/11 | 9/10 | 6,002/9,406 | **8/13** | **3,256,485 (29.93%)** | -86.5 (n=10) |
| `gru-quantile` | 0.835 | 9/11 | 8/10 | 4,819/5,740 | **10/13** | **3,122,203 (28.69%)** | -26.0 (n=9) |
| `lstm-gru-or` | 0.657 | 10/11 | 9/10 | 6,042/9,835 | 10/13 | 3,274,525 (30.09%) | -37.5 (n=10) |

Per fold (`docs/MODELS.md` 18.8.2 carries the full block): **fold 0 is
clean** -- 0 nominal-step alarms in 3,643,618, rare-FA 1/3, recall 5/6 (LSTM)
and 4/6 (GRU), floors 0.0115 and 0.0091, no stall. **Fold 1 is a calibration
collapse**: floors of 0.0109 and 0.0111, calibrated on the first 7.36M steps,
sit below **86.7% of the test window's nominal residual** (3,095,318 of
3,570,272 steps alarmed by both cells), so the alarm is the window, cut into
14-19 ranges, and 4/4 recall, 7/8 rare events and a "+129,270-step lead" are
what a window-long alarm scores. Fold 2 repeats it at 4.4% and 0.7% of
nominal time. Six fits, no stall (best epochs 16, 9, 28 and 13, 12, 17;
validation MSE 8.5e-6 to 1.2e-5). **The recall and F0.5 columns for the
cells are bought the way `mavg` bought 29/31 on the gate set (section 1,
`docs/HARNESS.md` 1) and are not findings; the nominal-step column is.**

### 6k.3 The pre-registration, adjudicated

`docs/MODELS.md` 18.8 carries every prediction beside its band. On `m2-ss1`:
**T1, T2, T6 held; T3 held on the ordering and refuted on `rstd`; T5 held on
the rule and refuted on its sub-prediction (the LSTM fitted best); T4 --
the union's cost -- FAILED** (2.0x the larger member, against a fail line of
1.5x). On `m1-g3`: **R1 and R5 held on the letter and are void in substance (recall
bought with 29-30% of nominal time alarmed); R2 FAILED (8/13 and 10/13); R3 --
the union's edge -- EVAPORATED (+0 over the LSTM at the GRU's cost); R4 failed
for the LSTM (-86.5) and returned no verdict for the GRU (-26.0); R6 held (no
stall).** The pre-registration asked how far recall and rare alarms would
move and did not ask whether the floor measured on the first half of the
history would still be the floor of the second; on `m2-ss1` it was, on `m1-g3`
fold 1 it was not. Not re-tuned; the finding.

### 6k.4 What the closure decides -- D29

`docs/DECISIONS.md` **D29. `gru-quantile` flies alone; the union is not
adopted; and the calibration's transfer is the finding Phase 2 inherits.**

By the rule set in 18.6 before the seal broke: the union's cost failed T4 on
`m2-ss1` (its 8/424 is the members' 4 + 4, disjoint) and its edge evaporated
on R3 (+0). Either alone selects `gru-quantile` alone; both did.

The transfer evidence, in one place. On **an independent spacecraft** the
recipe held: 4/424 rare events, 0 nominal steps in 4.16M, for both cells and
the TCN, with the per-channel floors failing (17% and 29%). On **a second
subsystem of the same spacecraft**, one fold in three held the same way and
two did not: the noise floor calibrated on the past was under 87% of a
later window. The forecaster is not what failed -- six clean fits, MSE in
the gate set's range -- the frozen rule's premise is: *one global quantile of
the fitting window's residual is the noise floor of everything after it.*
That premise is what D25 froze, what Objective.md 14.10 already says must be
replaceable in orbit, and what the C++ component must be designed to
recalibrate. Recorded, not reopened: D28 stands, D25 stands as the Phase 1
layer, and the recalibration question is Phase 2's first decision on the
decision layer.

## 7. What these numbers say

**The forecaster works and the decision rule does not.** `lstm-quantile` and
`lstm-telemanom` are the **same weights**, from the same cache, differing only in
how a threshold is chosen: 6/46 recall at 0.952 precision against 37/46 at 0.231.
One model, two decision rules, opposite failure modes. That is as direct a
demonstration as this data can give that the residuals carry the signal and the
thresholding is what is throwing it away.

**The trivial baseline still did not win, and still not for the reason
predicted.** `mavg` genuinely detects -- 25/32 headline-cell -- but buys it with
19,909 alarms at 2.3% precision. F0.5 and the adoption number both see through
it.

**The two sets remain two regimes.** `lstm-telemanom` scores 0.664 on `m1-ss5`
against 0.269 on `m1-g8.9.10`, and prefers opposite values of k on each. Group 8
renders these events as spikes; the primary set renders the same events at their
real multi-hour extent.

## 8. Provenance

| | |
|---|---|
| Command | `python -m sentinel_eval --verbose run m1-g8.9.10 --detector rstd --detector mavg --detector quiet --detector lstm-telemanom --detector lstm-quantile --no-sweep` |
| Git commit | `4d63cee` |
| Harness version | 0.1.0 |
| Dataset | `esa-adb` v1, manifest schema 1.1, generated `2026-08-24T21:31:02Z` |
| Channels | `m1-g8.9.10`: channel_41-52 (groups 8, 9, 10). `m1-ss5`: channel_41-46 (group 8) |
| Grid | 14,728,316 timesteps of 30s, zero-order hold, normalisation **identity** |
| Split | forward chaining, seed 25%, 3 folds -- both sets, so the pair differs in channels alone |
| Persistence / agreement | 1 / 1 for section 2; swept in section 4 |
| Thresholds | `lstm-telemanom`: telemanom's nonparametric dynamic threshold, folded into the score so a fixed 1.0 reproduces it. Everything else: label-free 99.9th percentile of each fold's training scores. No oracle sweep |
| LSTM | 2x80 LSTM, l_s=250, l_p=10, dropout 0.3, Adam 1e-3; seed 0, `torch.set_num_threads(4)` pinned; trained in PyTorch, **scored through the plain-NumPy reference** at 4.1e-08 agreement |
| **(!) correction** | This row read "early stopping at ~19 of 35 epochs". **It was never true.** Every cached fit behind the section 2 tables records `epochs_run = 11` and `best_epoch = 0` -- the defect in D17. The figure was written from expectation rather than read from a training report, which is the rule `docs/HARNESS.md` states and which nothing was reading at the time. Corrected here rather than deleted |
| Seeds | 0; the trivial baselines are deterministic and use none |
| **Operations measured** | **15 Class B, 1 Class A** for both sets and all five detectors |
| **TCN (section 6j)** | `Hyper(cell="tcn", hidden=(50,)*6, kernel=3)`, receptive field 253, 91,670 parameters, dropout 0.3, Adam 1e-3, seed 0+fold; fitted on an NVIDIA A40-2Q slice (Ubuntu 24.04, Python 3.12.3, torch 2.13.0+cu126, 1 thread, `CUBLAS_WORKSPACE_CONFIG=:4096:8`), `m1-ss5` fold-0 refit bit-identical under cuDNN; scored on the M5 (NumPy reference, refits refused); git `f258762` |
| **Closure, `m2-ss1` (section 6k)** | `python -m sentinel_eval --verbose run m2-ss1 --detector rstd --detector mavg --detector lstm-quantile --detector gru-quantile --detector tcn-quantile --detector lstm-gru-or --no-sweep`; git `0c5fbfb`; M5, torch 2.13.0, seed 0, four threads; 2026-08-29 20:25-20:44Z; artifacts `runs/m2-ss1/{rstd/2026-08-29T204415Z-70632603, mavg/2026-08-29T204415Z-81cac82b, lstm-quantile/2026-08-29T204415Z-2717441a, gru-quantile/2026-08-29T204415Z-6d146f5d, tcn-quantile/2026-08-29T204415Z-c48bd47d, lstm-gru-or/2026-08-29T204415Z-a9e0d056}.json`; **15 Class B, 2 Class A** |
| **Closure, `m1-g3` (section 6k)** | `python -m sentinel_eval --verbose run m1-g3 --detector rstd --detector mavg --detector lstm-quantile --detector gru-quantile --detector lstm-gru-or --no-sweep`; git `0c5fbfb`; M5, torch 2.13.0, seed 0 + fold, four threads; 2026-08-29 20:45-22:36Z; artifacts `runs/m1-g3/{rstd/2026-08-29T223625Z-70632603, mavg/2026-08-29T223625Z-81cac82b, lstm-quantile/2026-08-29T223625Z-2717441a, gru-quantile/2026-08-29T223625Z-6d146f5d, lstm-gru-or/2026-08-29T223625Z-a9e0d056}.json`; **11 Class B, 2 Class A** |
| **GRU (section 6h)** | `Hyper(cell="gru")`, 2x80, l_s=250, l_p=10, dropout 0.3, Adam 1e-3, seed 0+fold; fitted on an NVIDIA A40-2Q slice (Ubuntu 24.04, Python 3.12.3, torch 2.13.0+cu126, 1 thread -- the six reports record `torch_threads: 4`, the module default, because the field was read at class definition; corrected in `lstm.train` afterwards and immaterial to CUDA arithmetic), `m1-ss5` fold-0 refit bit-identical; scored on the M5 (torch 2.13.0, NumPy reference) with `detectors.train` replaced by a function that raises, so a cache miss would have failed rather than refitted; git `aeade73` |

`quiet` is not a candidate. It is the harness checking its own floor: a detector
that never fires must score zero recall with *undefined* precision, never zero.

**(!) Every LSTM figure here is telemanom-minus-commands.** telemanom feeds its
model encoded command information alongside telemetry; we fed telemetry only.
The rare-event false-alarm figures in particular are not comparable to published
telemanom numbers. See `docs/MODELS.md`.

## 9. Limitation -- applies to every row above

**Recall rests on Mission1 alone.** ESA-ADB holds no second viable recall set:
Mission2 deduplicates to 18 anomalies with 1-3 test-side, and Mission3 has 8
anomalies with only 4 of its 48 channels numeric. The roles are therefore split --
Mission1 carries recall, Mission2 (`m2-ss1`) carries the adoption number -- and
this limitation is stated on every result rather than in a document alongside
them, because results tables get screenshotted and travel without their context.

`m1-g8.9.10` was promoted to primary **after** the baseline run, on evidence
measured post-hoc. The claim that the choice was made a priori is forfeit; both
sets are reported permanently in consequence.

**Two held-back sets exist and are not in this document.** `m2-ss1` and `m1-g3`
were nominated before decision-layer tuning began (`docs/MODELS.md` section 6)
and have never been scored. They are run once, at the end, with settings frozen.
