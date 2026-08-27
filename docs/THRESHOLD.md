# The threshold-selector investigation

Work item 4. Three diagnostics on telemanom's nonparametric dynamic threshold,
reported together, with no threshold chosen and no source changed.

**Read `docs/DECISIONS.md` D17 and D18 for the decisions this produced**, and
`docs/RESULTS.md` section 6a for the numbers it explains. This file is the
account of the measurement itself: what was asked, what was measured, and what
it refuted.

Artifacts: `runs/m1-g8.9.10/_threshold/2026-08-27T172004Z-diagnostics.json` and
`-windows.npz`. Script: `scripts/threshold_diagnostics.py`.

---

## 1. The question

Correcting the early-stopping defect (`docs/DECISIONS.md` D17) improved the
forecast about fortyfold. Detection improved on every axis and precision
collapsed:

```
  lead time       +26  ->  +26      held
  events caught    37  ->  38       improved
  MVGS recall   28/32  ->  28/32    held
  event F0.5    0.269  ->  0.026    collapsed
  rare-event FA 22/48  ->  30/48    got worse
  ALARM RANGES    182  ->  3,548    twenty times more
```

A worse model catches fewer events. This one catches more, at the same lead
time, firing twenty times as often, so the model is not what got worse.

A `z` sweep was run first and it was the wrong instrument. telemanom's `z` is
not a constant: it is selected per window by maximising

```
  ( d_mu/mu + d_sigma/sigma ) / ( |E_seq|^2 + |e_a| )
```

over a candidate range, and that self-selection is the entire meaning of
*nonparametric dynamic thresholding*. Sweeping `z_floor` sweeps the **lower
bound of the candidate range** -- it does not tune a threshold, it takes options
away from a selector whose answers we did not like. So the question is not *what
z should we pick*, it is *why is the selection criterion choosing badly*.

## 2. How it was measured

| | |
|---|---|
| Windows examined | **5,684,580** reference windows |
| Coverage | both channel sets, all three folds, both weight generations |
| Fitting | **none.** Weights reused from `runs/_weights`; the script refuses to train and aborts if the store changes |
| Fidelity | every window verified **byte-identical** to the live `telemanom.dynamic_threshold` |
| Labels | **none.** Alarms are counted, never scored |
| Source changed | **none.** The guarded variant lives in the script |
| Operations | 15 Class B, 1 Class A -- month 212 / 20 of 50,000 |
| Wall clock | about twelve minutes, one process, no pod |

**The fidelity assertion is not decoration.** The script re-implements the
candidate loop in order to read what `dynamic_threshold` discards, so with the
guards disabled its selected `eps` must equal the live function's, exactly, on
every window it touches. Any mismatch aborts. Without it the counterfactual
would be a statement about a function nobody scores.

**The selected `z` is recovered without touching the source.**
`dynamic_threshold` returns `eps` and throws `z` away; recomputing `mu` and
`sigma` from the same window gives `z = (eps - mu) / sigma` exactly.

## 3. M0 -- the guard counterfactual

Published telemanom refuses a candidate unless **both** hold
(`khundman/telemanom`, `telemanom/errors.py`, `find_epsilon`):

```python
if score >= max_score and len(E_seq) <= 5 and \
        len(i_anom) < (len(e_s) * 0.5):
```

`src/sentinel_models/telemanom.py:249` is `if score >= best_score:` and nothing
else. The second condition is a **50% coverage cap**: a candidate flagging more
than half the reference window is rejected outright. This was the leading
hypothesis -- it is exactly the guard against a selector drifting to the bottom
of its range and flagging huge swathes.

> **It binds on zero of the 5,684,580 windows.** The alarm-range count under
> published telemanom's own rule is identical in every fold of both channel
> sets: 73, 1,970, 1,505 on `m1-g8.9.10` and 26, 692, 757 on `m1-ss5`.

The reason is upstream. EWMA at span 105 makes exceedances contiguous, so they
merge into one or two sequences covering a few hundred samples of 2,170 -- far
under both limits. Neither guard can trip at this buffer-to-window ratio.

**The omission is a real defect** -- the third from a reproduction this
repository calls faithful, after `min_delta` and the command inputs, now
deviation 8 in `docs/MODELS.md`. **It is not the cause of anything here**, and
restoring it would move no number.

## 4. D1 -- what the criterion is choosing

Nineteen candidates are on offer: `2.5` to `11.5` in steps of `0.5`.

```
  windows examined                                    5,684,580
  windows that selected a z                           2,081,285   (36.6%)
    of those, z = 2.5, the range minimum              1,927,583   (92.6%)
  windows with more than one candidate -- a choice    1,049,475   (50.4% of firing)
    of those, z = 2.5, the range minimum                895,773   (85.4%)
```

Pooled over both channel sets, both generations, all folds:

```
    z      windows      share
   2.5   1,927,583   92.6150%
   3.0     143,490    6.8943%
   3.5       9,512    0.4570%
   4.0         437    0.0210%
   4.5 - 11.5    250    0.0120%   <- fifteen candidates, together
```

**Three of nineteen candidates absorb 99.97% of the selections.** A range whose
bottom rung is chosen almost always is not a range being searched; it is a
constant being applied.

## 5. D2 -- is there anything to choose between

Evaluated across the whole candidate range on a stratified sample of windows,
the criterion is **monotone decreasing in z** in the large majority. Its median
value, normalised per window, `post-fix` fold 1:

```
    z          2.5     3.0     3.5     4.0     4.5     5.0     5.5     6.0
    m1-g8.9.10   1.000   0.843   0.676   0.504   0.350   0.282   0.250   0.220
    m1-ss5       1.000   0.825   0.653   0.494   0.347   0.286   0.283   0.231
```

There *is* a gradient -- the winner beats the runner-up by 20 to 45% -- and it
points steadily downhill, at the floor.

### Why, decomposed

Medians over multi-candidate sampled windows, `m1-g8.9.10` post-fix fold 2:

```
      z       n  n_above  n_seq  covered   denominator   numerator      score
    2.5   3,330       53      1      289           290      0.1517   4.54e-04
    3.0   3,330       30      1      237           238      0.1021   3.99e-04
    3.5   2,664       21      1      220           221      0.0863   3.82e-04
    4.0   1,998       15      1      213           214      0.0742   3.48e-04
    4.5   1,332       10      1      207           208      0.0576   2.83e-04
```

**The denominator is a constant in disguise.** `len(E_seq)` is **1** at every
candidate, so the squared anti-fragmentation term contributes 1 and never
activates -- the very term the paper describes as stopping the sweep buying a
statistical improvement with a shower of fragmented detections. `covered` falls
only from 289 to 207 and asymptotes at 199, because `error_buffer = 100` dilates
a single exceeded sample by +/-99 and merging collapses the rest. The numerator
meanwhile falls 2.6-fold, because removing fewer points changes the moments
less.

> A ratio whose denominator cannot move and whose numerator decays with z has
> its maximum at the smallest z on offer. **The criterion is arithmetically
> forced to prefer the bottom of its range**, and no residual distribution can
> rescue it. `error_buffer` is a parameter of the smoothing stage, not of the
> threshold.

The apparent uptick above `z = 5.5` in the full table is a selection effect, not
a peak: only windows with a large reach survive into those columns, and there
are 113 to 292 of them against 3,330.

> **On these residuals the nonparametric dynamic threshold reduces to
> `eps = mu + 2.5*sigma`** -- a fixed multiplier on the local scale, with the
> selection decorative and `z_floor` doing all the work.

Which is why sweeping `z_floor` appeared to work so well, and why that
appearance was misleading rather than informative.

## 6. D3 -- did the residual change shape

Same weights, same data, same folds; only the stopping rule differs. This is a
controlled comparison, not a before-and-after.

**The standing hypothesis was that the residual became near-Gaussian** --
irreducible noise, nothing left for an outlier-finder to find. Excess kurtosis
of the signed residual says the opposite:

```
  m1-g8.9.10   fold 0     63.6 ->     71.7        m1-ss5   fold 0    143.9 ->    159.1
               fold 1    145.2 ->    176.3                 fold 1    117.0 ->    153.2
               fold 2     27.3 ->  6,754.6                 fold 2    148.1 ->  7,348.8
```

A trained forecaster predicts the bulk almost perfectly and leaves a small
number of large excursions it cannot predict. That is **more** tailed, not less.

### What actually broke is local, not global

`eps = mu + z*sigma` is computed per 2,170-sample window, so the within-window
scale governs it, and that is what collapsed:

```
                     within-window sigma       fraction of windows with
                     (median) pre -> post      (max-mu)/sigma >= 2.5     alarm ranges
  m1-g8.9.10 fold 0  6.95e-4 -> 1.19e-3 (1.7x)   0.224 -> 0.083          42 ->    73   <- control
             fold 1  6.59e-4 -> 2.11e-4 (0.32x)  0.216 -> 0.729          90 -> 1,970
             fold 2  9.27e-4 -> 2.06e-4 (0.22x)  0.073 -> 0.803          50 -> 1,505
  m1-ss5     fold 0  1.66e-3 -> 6.10e-4 (0.37x)  0.010 -> 0.252          20 ->    26   <- control
             fold 1  1.34e-3 -> 1.76e-4 (0.13x)  0.056 -> 0.921          20 ->   692
             fold 2  1.07e-3 -> 1.73e-4 (0.16x)  0.143 -> 0.953          22 ->   757
```

### The chain, end to end

1. The forecast improves, so the **within-window standard deviation** of the
   smoothed error falls three- to eightfold.
2. The window **maximum does not fall with it**, because the tail got heavier.
3. So `(max - mu)/sigma` **rises** -- median 2.04 to 2.94 on `m1-g8.9.10` fold
   2, 2.24 to 3.28 on `m1-ss5` fold 2.
4. A floor of `2.5` out of reach in **93%** of that fold's windows is now
   cleared in **80%** of them.
5. Every crossing costs at least **199 alarm timesteps**, because
   `error_buffer = 100` dilates one exceeded sample by +/-99.

Fifty alarm ranges become 1,505.

### Fold 0 is the control, and it was already on disk

The forecast did not improve equally across folds. Fold 0 improved 1.6x against
folds 1 and 2 at ~40x. Its within-window sigma went **up**, its reachable
fraction went **down**, and its alarm count barely moved. A control arm and a
treatment arm inside the same run, on both channel sets, unread until something
went looking for them.

## 7. The conclusion

**Degenerate, not mis-ranged.** The criterion is not choosing badly; it is not
choosing. Its argmax is the boundary of the candidate range by construction,
because a stage upstream of it -- the `+/-99` error buffer -- pins the
denominator it is supposed to trade against.

The corrected general rule, recorded as `docs/DECISIONS.md` D17: **a
dimensionless constant is immune to rescaling and not immune to a change in the
relationship between a distribution's scale and its extremes.** A standard
deviation is a poor summary of a distribution whose kurtosis is in the
thousands. That was the conclusion the original entry reached; the mechanism it
gave for it was wrong.

## 8. What was not done

**No threshold value is chosen.** The `z_floor` sweep's best cell is genuinely
tempting -- F0.5 **0.794**, precision **22/22**, rare-event false alarms
**0/48**, median lead time **+35.5** with not one late detection -- and it is
selected by reading a table of scores against 46 labelled anomalies. A
spacecraft that has never failed has no such table (Objective.md 6.1), so it is
a number no adopting mission could ever obtain. `docs/MODELS.md` section 7
refused the same move once already, under the name *oracle threshold sweep*.

**Five stages are flagged and none is retuned.** `docs/HARNESS.md` is explicit
that a stage needing revisiting is flagged, not adjusted inside a run measuring
something else. Detail in `docs/DECISIONS.md` D18:

| Stage | Why the measurement implicates it |
|---|---|
| `error_buffer = 100` | One exceeded sample becomes a 199-timestep alarm, and it is what pins the criterion's denominator |
| EWMA span 105 | Makes exceedances contiguous, collapsing both quantities the published guards test. Inherited, never examined |
| `z_floor = 2.5` | Selected in 92.6% of firing windows. Not a floor on a range -- the operating threshold |
| k-of-n agreement | Tuned when there were 182 alarm ranges. There are 3,548 |
| pruning `p = 0.13` | Reads the same residual ladder the threshold does. Not measured here |

## 9. What this cost, and what it refuted

One script, no refitting, no source changed, 15 Class B operations, twelve
minutes. **Two hypotheses refuted**, one of them the one this investigation was
most confident in and had put first.

That is the same shape as the five entries in `docs/NARRATIVE.md` section 2. The
difference worth recording is that the measurement was designed so that it could
say *no*: the guard counterfactual would have been just as easy to run in a form
that could only confirm, and the fold-0 control was sitting in the artifacts
already, waiting to be read as a control rather than as an outlier.
