# What happened, in order

A short account of how this project got where it is. It exists because the most
useful thing here is not the numbers -- it is the sequence of mistakes that were
caught, and what caught them.

Updated at each work-item boundary (`docs/HARNESS.md`).

---

## 1. The referee was built before the players, deliberately

The obvious first move on an anomaly-detection project is to train a model. We
built the scoring harness instead, and scored trivial baselines on it before any
neural network existed.

The reasoning was defensive: build models first and every result becomes an
argument about splits, metrics and what "caught it" means. Build the referee
first and LSTM against GRU against TCN is a table of numbers.

That was the intent. What actually happened is that **the referee spent its first
week catching our errors rather than the models'**, and it caught five of them
before a single model was trained.

## 2. The five things the referee caught

**A metric that rewarded carpet-bombing.** The pre-registered prediction was that
a moving average would score near-floor on cross-channel recall. It scored
**29/31**. For an afternoon that read as the benchmark being broken. It was not:
`mavg` bought that recall with **5,535 alarms for 42 events** at 24.9% precision,
alarming on 39 of 48 commanded manoeuvres. The prediction had been framed on a
quantity that carpet-bombing satisfies. On F0.5 the same detector scored 0.028
against `rstd`'s 0.250. The failure was in our measurement design, not in the data
or the harness -- which is the good outcome, and the reason to pre-register.

**A channel set that inverted the project's own thesis.** `m1-ss5` was chosen as
primary because it carries all eleven point anomalies. Measuring event footprint
afterwards showed group 8 is the *only* channel set in Mission1 where
headline-cell events register as sub-grid-cell spikes -- 18 of 38, against 0
everywhere else. Groups 9 and 10 see the same events at their real multi-hour
extent. We had selected the one set that renders slow cross-channel events as
fast single-channel ones, on a project whose entire claim is slow cross-channel
events.

**A VUS buffer sweep that silently collapsed.** Event footprints are extremely
skewed -- median 1 timestep, mean 2,909, maximum 81,717 -- so a median-derived
buffer schedule collapsed to `[0, 1]` and VUS quietly stopped integrating any
volume, becoming plain AUC-PR on one set and not the other. Nothing raised. The
number kept being produced and kept looking reasonable.

**A mask defect the baselines were structurally incapable of surfacing.**
`Bundle.subset` rebuilt its ground truth from the labels alone and never read
`self.valid`, so unobserved timesteps counted as scorable nominal time on every
subset. It sat inside a harness with 136 passing tests for three work items,
because the trivial baselines are NaN-tolerant *by construction* -- `nan_to_num`,
`nanstd` -- and nothing had ever asked the mask to be right. The first detector
that could not tolerate a NaN found it on its first run, seventy-five minutes in.

**An optimisation that would have blinded the detector to the primary set.**
Telemanom slides its error window forward 70 timesteps at a time; stepping a whole
window instead looked like an obvious economy and was written into the plan as
"coarser, and if anything more conservative". Measured against injected anomalies
of 100, 500, 2,100 and 8,828 timesteps, it detected the first and missed the other
three: a threshold derived from the moments of the window it is judging cannot see
an anomaly that fills that window. The primary set's headline events have a median
footprint of 1,951 timesteps. The economy would have blinded us to nearly all of
them, and the result would have read as a weak model rather than a broken
thresholder.

## 3. A pre-registered prediction that was wrong, and kept

`docs/HARNESS.md` section 7 records a prediction made before the first baseline
ran, the trigger that fired, the four checks that located the cause, and the two
errors -- both ours, both in the measurement design. It is preserved with what
actually happened beside it rather than tidied away and rewritten.

A pre-registration that can be quietly replaced is not a pre-registration. The
audit trail is the asset.

## 4. Then we found the missing third of the reference method

Work item 4 reproduced telemanom and cleared the floor. The cross-channel claim
came out well: 28 of 32 headline-cell events against `rstd`'s 3. The false-alarm
rate did not: 22 of 48 commanded manoeuvres.

A 24-cell decision-layer grid could not fix it -- the best cell still alarmed on
15 of 48. The cause was not in the decision layer at all. **telemanom's LSTM takes
telemetry *and encoded command information*; we had fed it telemetry only.** The
821 telecommand series ingested in work item 2 had never been wired in, and had
been recorded in the deviation ledger as "a future item".

Our false alarms are on commanded events. We withheld the one input that makes
them predictable, then measured the detector's failure to predict them.

## 5. So we went to six other fields

Every published ESA-ADB baseline scores zero on requirement R7 -- learn to
distinguish rare nominal events so they are not alarmed after the first
occurrence. The benchmark's authors state that commanded manoeuvres "cannot be
distinguished from actual anomalies without expert knowledge". We hold that
expert knowledge as a data channel we already own.

The false-alarm programme borrows from process-control alarm management
(ISA-18.2, EEMUA 191), radar detection (CFAR guard cells, order statistics),
fraud detection (cascades, alert budgets), avionics (EGPWS mode-based
suppression), intensive-care monitoring (alarm delays), and extreme-value theory.
`docs/RESEARCH.md` carries the sources.

---

## 6. The measurements that changed conclusions

This is the through-line, and it is the most transferable thing the project has
produced. Seven times now, a cheap measurement has changed a decision that
looked settled without it -- and the seventh is the one that refuted a
hypothesis rather than replacing one, which is the harder thing for a
measurement to be able to do.

**Reporting `k/n` instead of bare rates showed the samples were far smaller than
the percentages implied.** "Recall 0.36" over eleven events takes twelve possible
values. Printing the rate alone invites a reader to treat a one-event difference
as a finding, and we were about to.

**Measuring event footprint showed the chosen channel set inverted the thesis.**
The selection was made on point-anomaly coverage, which was correct as far as it
went. Nobody had measured how *long* the events were.

**Measuring the non-overlapping error window before shipping it showed it would
have blinded the detector to the primary recall set.** The justification for the
economy was a cost estimate that turned out to be wrong by an order of magnitude:
112 microseconds per window, not the prohibitive figure assumed.

**Keeping pre-fix and post-fix numbers proved a correction was immaterial.** The
`Bundle.subset` defect moved 40 timesteps and **zero event-wise counts**; six
figures shifted in the sixth or seventh significant digit. That the earlier
results were not distorted is a *finding*, and it would not exist had the old
table been overwritten. A reviewer seeing only corrected numbers has to take our
word for it.

**Measuring lead time overturned which detector looked better, one run after the
metric was added.** `lstm-quantile` won on F0.5 (0.421 against 0.269) and on
false alarms (1/48 against 22/48) and had no argument against it. Then: median
lead **-122 timesteps**, all six catches after the event began, on alarms five
times wider. And `rstd` -- the floor the entire work item was measured against --
came in at **-1,512**. The floor was never an early-warning detector, and that was
unknowable before the metric existed.

**A hypothesis stated confidently in the handover brief was wrong, and so was
the one that replaced it.** The brief for the threshold investigation named a
cause: the residual distribution had moved from heavy-tailed model bias to
near-Gaussian irreducible noise, so a criterion built to find outliers in
structured error had nothing left to find. Auditing the criterion against
published telemanom then turned up a better candidate -- two admissibility
conditions we do not have, one of them a 50% coverage cap that rejects outright
any threshold flagging more than half its reference window. That looked like the
whole answer, and it was put first in the plan.

**It binds on zero of 5,684,580 windows.** The alarm count under published
telemanom's own rule is identical in every fold of both channel sets. And the
residual had not become Gaussian: excess kurtosis of the signed residual went
27.3 to **6,754.6** on one fold. Both hypotheses failed, and what the
measurement found instead was a **cross-stage coupling** neither of them
contained -- `error_buffer`, a smoothing parameter, pinning the denominator the
threshold's selection criterion is supposed to trade against. `docs/THRESHOLD.md`
carries the measurement; `docs/DECISIONS.md` D17 and D18 carry what it settled.

**The part worth teaching is that the test was built so it could say no.** The
guard counterfactual would have been just as easy to run in a form that could
only confirm -- take a sample of windows where the selector chose the floor, show
that the coverage cap would have rejected that choice, and stop. Instead it
evaluated **every** candidate in **every** window on the real path, recorded the
guard flags whether or not they fired, and reported the binding rate as a
measurement rather than as an illustration. The confirming-only version costs
the same and cannot produce a zero.

The same design decision runs through the rest of it. The selected `z` was
recovered as `(eps - mu)/sigma` from the live function's own output rather than
recomputed, so the distribution is the scored one and not a reconstruction of
it. Every window asserted that this file's unguarded selection reproduced
`telemanom.dynamic_threshold` exactly, so a mismatch aborted the run rather than
producing a plausible number about a function nobody scores. And fold 0 -- which
improved 1.6x where the others improved 40x -- was read as a **control arm**
rather than as an outlier, which is the only reason "better forecast causes the
alarm explosion" is an association in the data rather than a story about it.

Two smaller things from the same investigation, recorded because they are the
same discipline applied to smaller stakes. The handover brief listed four
metrics that moved and omitted that the **adoption number moved the wrong way on
both channel sets** -- rare-event false alarms 22/48 to 30/48 and 17/48 to 33/48
-- which is the figure Objective.md 11 rule 2 says decides whether a mission
flies this at all, and which should have led. And a set of illustrative figures
written into an options mock-up while planning the work were read back as though
they were data. They were not measured, they were placeholders showing a table's
shape, and they were corrected in the open before anything was built on them.
**An illustrative figure is still a figure**, which is exactly what the rule
below says.

**And once, the failure mode occurred inside the document arguing against it.**
Filling the lead-time column of `docs/RESULTS.md`, two baseline figures were
written from expectation rather than read from the run artifacts: `rstd` as `+2`
and `mavg` as `+18`. The measured values were `-1,512` and `0`. `rstd`'s was wrong
by three orders of magnitude *and in the opposite direction*, and had it stood it
would have concealed the most consequential finding in that document. It was
caught by going back to the artifacts, corrected before the commit, and reported
rather than buried -- the same discipline, applied to itself. `docs/HARNESS.md`
now carries the rule as a standing one: **no number enters a document that was not
read from an artifact.** If a value is not to hand, the cell stays empty and says
so.

### What it costs, and what it bought

Each of these was nearly free. A denominator beside a rate. A histogram of event
lengths. Four injected anomalies at different durations. Keeping a table instead
of replacing it. One subtraction per caught event. Two extra boolean columns in
a table that was being computed anyway.

Each changed a decision.

> **What we measure determines what we conclude.** A metric set that omits a
> property of interest will confidently rank detectors on the properties it does
> measure, and it will not report that something is missing. Adding a cheap
> measurement is almost always worth it -- and the ones worth adding are, by
> definition, the ones nobody has thought to ask for yet.

---

## 7. Then training started working, and everything downstream broke

A defect in early stopping had been disabling training for the whole project.
telemanom's published `min_delta` is 3e-4, applied as
`current < best_loss - min_delta`, and it is **absolute, in the units of the
loss**. Validation MSE on ESA-ADB is about 1e-4, so the bar went negative, no
epoch after the first ever qualified, and every fit in three work items stopped
at epoch 11 having kept epoch 1. Median 3.1x better weights discarded, worst
case 8.8x.

It announced itself in no way at all. No error, no warning. The signature --
eleven epochs with `best_epoch = 0`, every time -- was sitting in the training
reports, and **nothing was reading the training reports**. They are on the
scorecard now, and a fit that keeps its first epoch raises rather than warns,
because a line of output nobody looks at is indistinguishable from silence.

Fixing it improved the forecast about fortyfold. And:

```
  lead time       +26  ->  +26      held
  events caught    37  ->  38       improved
  MVGS recall   28/32  ->  28/32    held
  event F0.5    0.269  ->  0.026    collapsed
  rare-event FA 22/48  ->  30/48    got worse
  ALARM RANGES    182  ->  3,548    twenty times more
```

A worse model catches fewer events. This one catches more, at the same lead
time, firing twenty times as often -- so the model was not what got worse.

### The instinct was to sweep the threshold, and that was the wrong instrument

A `z` sweep was run and produced a clean curve with a seductive answer:
`z_floor = 8.0` gives F0.5 0.794, precision 22/22, zero rare-event false alarms
and a median lead time of +35.5 with not one late detection.

But telemanom's `z` **is not a constant**. It is selected per window by
maximising `(dmu/mu + dsigma/sigma) / (|E_seq|^2 + |e_a|)` over a candidate
range, and that self-selection is the entire meaning of *nonparametric dynamic
thresholding*. Sweeping `z_floor` sweeps the **lower bound of the candidate
range**. It does not tune a threshold; it takes options away from a selector
whose answers we did not like. And picking the winning cell would be fitting to
46 labelled anomalies, which is precisely what a real spacecraft -- with no
failures to fit to -- cannot do.

So the question was not *what `z` should we pick*. It was *why is the selection
criterion choosing badly*.

### What the measurement found, and what it refuted

`scripts/threshold_diagnostics.py`, both channel sets, all three folds, both
generations of weights, **5,684,580 reference windows**, every one verified
byte-identical to the live `telemanom.dynamic_threshold` -- because a
re-implementation that is merely similar describes a function nobody scores.

**The leading hypothesis was wrong, and it was ours.** Auditing the criterion
against published telemanom turned up two admissibility conditions we do not
have: `len(E_seq) <= 5` and `len(i_anom) < len(e_s) * 0.5`. The second is a 50%
coverage cap, and it looked like exactly the guard against a selector flagging
huge swathes of a window -- the behaviour under investigation. **Neither binds
on a single one of the 5.68 million windows.** The alarm count under published
telemanom's own rule is identical, in every fold, on both channel sets. The
reason is a stage upstream: EWMA at span 105 makes exceedances contiguous, so
they merge into one or two sequences covering a few hundred samples of 2,170,
far under both limits. The omission is a real defect in the reproduction and it
is not the cause of anything here.

**The brief's hypothesis was wrong too, and it is the more interesting one.** It
predicted the residual had moved from heavy-tailed model bias to near-Gaussian
irreducible noise, leaving an outlier-finder with nothing to find. Measured, the
residual moved the *other way*: excess kurtosis of the signed residual rose from
27.3 to **6,754.6** on one fold and 148.1 to **7,348.8** on another. A trained
forecaster predicts the bulk almost perfectly and leaves a few large excursions
it cannot predict.

**What actually happened is local, not global.** `eps = mu + z*sigma` is computed
per 2,170-sample window, so the within-window scale is what governs it, and that
collapsed three- to eightfold. The window *maximum* did not fall with it,
because the tail got heavier. So `(max - mu)/sigma` **rose** -- 2.04 to 2.94 on
one fold -- and a floor of 2.5 that used to be out of reach in 93% of windows is
now cleared in 80% of them. Every window that clears it contributes an alarm at
least 199 timesteps wide, because `error_buffer = 100` dilates one exceeded
sample by +/-99. Fifty alarm ranges become 1,505.

**And the criterion is not choosing.** Of the 2,081,285 windows that selected
anything, **92.6% selected the range minimum**. Restricting to the 1,049,475
windows that had more than one candidate -- a real choice -- **85.4% still chose
the minimum**, and the criterion is monotone decreasing in `z` in the large
majority of sampled windows. On these residuals the nonparametric dynamic
threshold reduces to `eps = mu + 2.5*sigma`: a fixed multiplier on the local
scale, with the selection decorative and `z_floor` doing all the work. Which is
why sweeping `z_floor` appeared to work so well, and why that appearance was
misleading rather than informative.

**Fold 0 is the control, and it was sitting in the artifacts unread.** The
forecast did not improve equally across folds: fold 0 improved 1.6x against
folds 1 and 2 at ~40x. Its within-window sigma went *up*, its reachable fraction
went *down*, and its alarm count barely moved -- 42 to 73, against 90 to 1,970
and 50 to 1,505. A control arm and a treatment arm inside the same run, on both
channel sets, and nobody had looked.

### What it cost, and the pattern it repeats

One script, no refitting, no source changed, 15 Class B operations, about twelve
minutes. Two hypotheses refuted, one of them the one this coder was most
confident in and had put first in the plan.

That is the fourth entry in section 6's list, and it is the same shape as the
other three. The difference is that this time the measurement was designed to be
able to say *no*: the guard counterfactual would have been just as easy to run
in a form that could only confirm, and the fold-0 control was already on disk
waiting for somebody to read it as a control rather than as an outlier.
