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

## 5. The correction: pre-fix and post-fix

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

## 6. What these numbers say

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

## 7. Provenance

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
| LSTM | 2x80 LSTM, l_s=250, l_p=10, dropout 0.3, Adam 1e-3, early stopping at ~19 of 35 epochs; seed 0, `torch.set_num_threads(4)` pinned; trained in PyTorch, **scored through the plain-NumPy reference** at 4.1e-08 agreement |
| Seeds | 0; the trivial baselines are deterministic and use none |
| **Operations measured** | **15 Class B, 1 Class A** for both sets and all five detectors |

`quiet` is not a candidate. It is the harness checking its own floor: a detector
that never fires must score zero recall with *undefined* precision, never zero.

**(!) Every LSTM figure here is telemanom-minus-commands.** telemanom feeds its
model encoded command information alongside telemetry; we fed telemetry only.
The rare-event false-alarm figures in particular are not comparable to published
telemanom numbers. See `docs/MODELS.md`.

## 8. Limitation -- applies to every row above

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
