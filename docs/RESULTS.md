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
were nominated before decision-layer tuning began (`docs/MODELS.md` section 5)
and have never been scored. They are run once, at the end, with settings frozen.
