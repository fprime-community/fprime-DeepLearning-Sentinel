# Results

The floor that every trained model must clear. Nothing here is a claim about
Sentinel's detection performance -- these are trivial baselines, scored first so
that the harness had been used in anger before any neural network depended on it.

**Read `docs/HARNESS.md` first**, in particular section 7, which records a
prediction that failed and why.

---

## 1. The floor

**`rstd` on `m1-g8.9.10`: event-wise F0.5 = 0.250.** Best of anything tested.
Work items 4, 5 and 6 must clear it on the primary set.

Nothing else comes close, and the reason matters: `mavg` reaches nearly ten times
the recall and still scores 0.028, because it buys that recall with 19,909 alarms
at 2.3% precision. Recall alone is not a result.

## 2. Every detector, both sets

Both channel sets are reported for every detector, always. `m1-ss5` was the
primary until the baseline run and was demoted on footprint evidence measured
**post-hoc**; it is retained and reported rather than discarded, because you
cannot cherry-pick if you never discard anything. See `docs/HARNESS.md` section 2.

**GATE -- `m1-g8.9.10`** (12 channels, groups 8+9+10, the primary recall set)

| Detector | **F0.5** | recall | precision | MVGS | contextual | point | **rare-event FA** | alarms/1k | VUS-PR |
|---|---|---|---|---|---|---|---|---|---|
| `rstd` | **0.250** | 3/46 (0.065) | 6/7 (0.857) | 3/32 (0.094) | 3/45 (0.067) | 0/11 (0.000) | **1/48 (0.021)** | 0.000 | 0.030 |
| `mavg` | 0.028 | 34/46 (0.739) | 452/19909 (0.023) | 25/32 (0.781) | 34/45 (0.756) | 9/11 (0.818) | **10/48 (0.208)** | 1.796 | 0.226 |
| `quiet` | undefined | 0/46 (0.000) | -/0 | 0/32 (0.000) | 0/45 (0.000) | 0/11 (0.000) | **0/48 (0.000)** | 0.000 | 0.030 |

**`m1-ss5`** (6 channels, group 8; demoted, point-anomaly coverage)

| Detector | **F0.5** | recall | precision | MVGS | contextual | point | **rare-event FA** | alarms/1k | VUS-PR |
|---|---|---|---|---|---|---|---|---|---|
| `mavg` | **0.135** | 32/42 (0.762) | 399/3573 (0.112) | 23/31 (0.742) | 32/41 (0.780) | 9/11 (0.818) | **8/48 (0.167)** | 0.240 | 0.139 |
| `quiet` | undefined | 0/42 (0.000) | -/0 | 0/31 (0.000) | 0/41 (0.000) | 0/11 (0.000) | **0/48 (0.000)** | 0.000 | 0.013 |
| `rstd` | undefined | 0/42 (0.000) | 0/98 (0.000) | 0/31 (0.000) | 0/41 (0.000) | 0/11 (0.000) | **1/48 (0.021)** | 0.009 | 0.009 |

`n < 20` denominators -- point recall throughout, and `rstd`'s precision -- cannot
distinguish detectors: over 11 events, recall takes 12 values. Treat them as
coverage checks, not as comparisons.

## 3. What these numbers say

**The trivial baseline did not win, but not for the reason predicted.** `mavg`
genuinely detects: a random detector reached only 10/31 headline-cell events at
*twice* its alarm budget. What it cannot do is detect cheaply -- 19,909 alarms and
one commanded manoeuvre in five falsely flagged. F0.5 and the adoption number
both see through it; bare recall did not, which is why bare recall is no longer
reported alone anywhere.

**The two sets behave as two regimes**, as expected. `mavg` scores nearly five
times better on `m1-ss5` (0.135) than on `m1-g8.9.10` (0.028), because group 8
renders these events as spikes -- 18 of its 38 headline-cell events are shorter
than one grid cell, the only group in mission1 where that happens. A per-channel
method suits the fast regime. It has no answer to the slow one.

**`rstd` scores below silence on `m1-ss5`** -- VUS-PR 0.009 against `quiet`'s
0.013 -- which is not a bug. The headline-cell anomalies there involve channels
freezing, and a frozen channel has *lower* rolling variance than nominal data, so
rolling-standard-deviation ranks the anomaly as less anomalous than normal
operation. Measured directly on the fixture: 0.0543 inside freeze events against
0.0557 on nominal data. It is anti-correlated with the thing it is looking for.

## 4. Provenance

Every figure above comes from a single run, and this section is what makes it
reproducible.

| | |
|---|---|
| Command | `python -m sentinel_eval run m1-g8.9.10 --detector rstd --detector mavg --detector quiet --no-sweep` |
| Git commit | `1f00a43` |
| Harness version | 0.1.0 |
| Dataset | `esa-adb` v1, manifest schema 1.1, generated `2026-08-24T21:31:02Z` |
| Channels | `m1-g8.9.10`: channel_41-52 (groups 8, 9, 10). `m1-ss5`: channel_41-46 (group 8) |
| Grid | 14,728,316 timesteps of 30s, zero-order hold, normalisation **identity** |
| Split | forward chaining, seed 25%, 3 folds -- both sets, so the pair differs in channels alone |
| Persistence filter | 1 (off) |
| Thresholds | label-free, 99.9th percentile of each fold's training scores. No oracle sweep |
| VUS-PR buffers | `m1-g8.9.10` [0, 2149, 4298, 6446, 8595]; `m1-ss5` [0, 616, 1233, 1850, 2466] |
| Seeds | none used -- every detector here is deterministic |
| **Operations measured** | **15 Class B, 1 Class A** for both sets and all three detectors |

`quiet` is not a candidate. It is the harness checking its own floor: a detector
that never fires must score zero recall with *undefined* precision, never zero.

## 5. Limitation -- applies to every row above

**Recall rests on Mission1 alone.** ESA-ADB holds no second viable recall set:
Mission2 deduplicates to 18 anomalies with 1-3 test-side, and Mission3 has 8
anomalies with only 4 of its 48 channels numeric. The roles are therefore split --
Mission1 carries recall, Mission2 (`m2-ss1`) carries the adoption number -- and
this limitation is stated on every result rather than in a document alongside
them, because results tables get screenshotted and travel without their context.

`m1-g8.9.10` was promoted to primary **after** the baseline run, on evidence
measured post-hoc. The claim that the choice was made a priori is forfeit; both
sets are reported permanently in consequence.
