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
produced. Twelve times now, a cheap measurement has changed a decision that
looked settled without it. The eleventh took the project's headline claim -- warning
before the event -- from +26 timesteps to zero. Two of the last three refuted a hypothesis rather
than replacing one, which is the harder thing for a measurement to be able to
do -- and one of them refuted the *instrument*, which is harder still.

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

**The project's own preferred mechanism, refuted by its own scoreboard.** The
decision layer was found to be channel-blind -- twelve univariate detectors and a
vote, with the one cross-channel stage testing co-occurrence rather than
relationship -- and the conclusion drawn was that *the false-alarm problem lives
in that gap*. A relationship test was designed, built and measured, and it did
take rare-event false alarms from 30/48 to 2/48.

Then a detector that had been closed on superseded evidence was re-measured on
the same weights, and reached **the same 2/48** -- while being **also
channel-blind**, a plain maximum over per-channel smoothed errors against one
global threshold.

> **The false-alarm gain came from having a high global threshold, not from
> testing the relationship. Whitening bought nothing measurable that a global
> quantile did not.**

The observation that the layer cannot see relationships is unchanged and still
true. What is refuted is the *causal* claim built on it. The false alarms came
from telemanom's **local** threshold adapting downward until it fired constantly,
and any sufficiently high global cut removes them -- with or without the
covariance, the sign, or the joint direction.

That is the most expensive kind of wrong to be: a mechanism that is real, elegant,
matches the project's thesis, and is not what was happening. It survived because
it was measured against the detector it was designed to beat and never against
the simpler thing sitting in the repository with a closed-branch label on it.

**Every closed branch inherits the obligations of every later correctness fix.**
`lstm-quantile` was disqualified on one-epoch weights; when the training defect
was fixed, nothing re-opened it. **A disqualification made on buggy weights is
not a disqualification**, and the re-check costs one run.

**Four explanations for one gap, each plausible, each refuted by a cheap
measurement.** `lstm-whitened` loses eleven anomalies that `lstm-telemanom`
catches, and the obvious candidates were tested rather than argued: single-channel
events a joint test cannot see (every lost event touches five to twelve channels);
lower channel count (identical between lost and kept); shorter footprint (lost are
*longer*, 51 against 42); and pruning discarding real detections -- which turned
out to be **true on six channels and false on twelve**, recovering three
headline-cell events for free on the subset and nothing at all on the gate set.

The fourth was mine and the most interesting if it had held: that whitening
divides out an anomaly which moves channels together in the learned pattern, the
exact arithmetic that suppresses a commanded manoeuvre. It predicted a large
standardised residual with a low whitened ratio. **Measured, the lost events are
4.4 times smaller in the raw standardised length before any whitening happens.**
Nothing is being divided out. They are weak events, and they peak at a quarter of
the threshold.

What that leaves is unglamorous and worth more than the hypothesis was: the two
rules differ in **sensitivity**, not in what they can see, and the likely reason
is that one thresholds locally and the other globally -- which is the same
asymmetry D13 was written about, arriving from the other side.

The pattern across all four is the one this section keeps recording. Each
hypothesis was cheap to test, each would have been comfortable to assume, and the
one that survived is the one nobody proposed.

**And measured properly, the claim does not survive.** The honest reading -- each
alarm re-dated to the batch boundary where the detector could actually speak --
puts the median lead at **-43.0** against the reported +26.0, with 22 of 23
detections negative. Worse, **fifteen of the thirty-eight events on the gate set
have no emission overlapping them at all**: the detector's first word came after
the event had ended, and the alarm touched the event only because the buffer
widened it backwards into the past.

Two implementation errors of mine on the way, both caught by reading the output
rather than by the tests, which is its own signal. The first described the
dilation as reaching back 99 steps when it is clipped to the 70-step batch -- the
measured maximum of +63 was the tell and I did not read it. The second used the
emission mask *as* the alarm mask, which silently changed which events counted as
detected and would have reported a lead time and a recall change as one number.

**The project does not currently warn early.** Not on either channel set, not
with either decision layer. The +26 was the batching latency counted backwards.

**The project's headline claim measured itself from a start it never reached.**
Objective.md section 2 promises warning *before* an event, and every lead-time
figure this repository publishes is measured from the start of an alarm range.
telemanom's `error_buffer` widens each range by +/-99 timesteps **backwards from
a crossing that has already happened**. Measured from the crossing itself,
`lstm-telemanom`'s median lead is **+0.0** rather than +26.0, and it warns in
advance in **3 of 38** events rather than 34 of 38. In 31 of 38 the entire
positive lead is the widening.

A flight component emits when it detects, not retroactively, so nothing reaches
an operator at the dilated start. The metric was crediting warning that was never
given -- and the raw figure is *still* optimistic, because it ignores the fixed
70-step batching latency the detector also carries.

**It is a gate metric with a disqualifying rule, and it had been compared across
detectors that do not all receive the dilation.** `lstm-quantile` and the trivial
baselines run through paths with no `error_buffer` at all, so +26 against
`mavg`'s 0 and `lstm-quantile`'s -122 was never like-for-like: one side carried a
median +27.5 that the other did not. Two detectors were disqualified on that
comparison.

The check cost one subtraction per caught event -- the first threshold crossing
instead of the range start -- on data already in memory. It had never been made
because the number it produced looked like the number the project wanted.

**Twenty-five minutes of finished analysis thrown away by a bookkeeping write,
because the bookkeeping came first.** Every script in `scripts/` committed the
operations ledger and *then* wrote its artifact -- the house pattern, copied from
the first one. A run completed its whole analysis, held it in memory, and died
on the ledger `PutObject` with `RequestTimeTooSkewed`: the laptop's clock had
drifted past the signing tolerance mid-run, most likely by suspending. Measured
afterwards the skew was **+1 second**, so it was transient and unreproducible.
The result was gone anyway.

Nothing about it was subtle once seen. **The artifact is the expensive,
unrepeatable thing and the ledger is bookkeeping that can be retried**, so the
order was exactly backwards in all five scripts. Artifact first now, and the
ledger commit wrapped so a failure is reported loudly rather than taking the
result with it -- the operations were genuinely spent, and a ledger that does not
record them understates real usage against a hard ceiling.

The transferable part is not the ordering. It is that **the failure had nothing
to do with the work**: no data problem, no modelling error, no bad number. A
correct result was destroyed by an unrelated network write on the way out, and
the design made that possible. Worth asking of anything long-running: *if the
last line fails, what have I lost?*

**A structural claim that was really a configuration, quoted forward into two
designs before anyone rechecked it.** `docs/DECISIONS.md` D13 closed the quantile
branch on a generalisation rather than on its numbers: *a global threshold
responds to absolute magnitude, so it must fire late.* Clean measurement
contradicts it. Cells whose binding rate is 0.010 to 0.041 -- a threshold that is
global in all but name -- fire at **+24.0 to +29.0** and catch 18 of 32
headline-cell events, and seventeen of eighteen cells on the curve lead
positively.

The original evidence was also confounded, and the confound is checkable in four
lines of code: `lstm-quantile` runs through `top_columns`, which applies **no
`error_buffer` dilation and no pruning**, while every telemanom-path detector has
each exceedance widened by +/-99 timesteps before alarm ranges are formed. Lead
time is measured from a range's start. So the -122 against +26 was never
like-for-like -- one arm had up to 99 timesteps of dilation and the other had
none.

**The branch stays closed on its measured numbers and the reason is withdrawn.**
What makes it worth an entry is what the sentence had been doing since: D20's
entire two-term design existed because a global term was assumed structurally
late and therefore needed a local one beside it. **That design was solving a
problem that may not exist.** A structural claim is exactly the kind that gets
reused without rechecking, which is why one resting on a single detector's
configuration is worse than no claim at all.

**A threshold calibrated on "nominal" data that was not nominal, and it
contaminated two separate results before anything caught it.**

Two detection rules in a row fit their operating point on the fitting window's
residuals, and both documents and both docstrings said the same thing: *the
fitting window is normal-only by construction, `splits.train_mask` removes
annotated anomalies*. That sentence is true of **fitting** and false of the
**scoring call** the calibration hooks. From `harness._score_fold`:

```
  usable = train_mask(fold, truth)[train_lo:train_hi]
  detector.fit(bundle.values[train_lo:train_hi], usable, context)     # mask applied
  train_raw = detector.score(bundle.values[train_lo:train_hi], ...)   # mask NOT applied
```

So a 99.9th-percentile threshold meant to admit one nominal timestep in a
thousand was set partly by the anomalies inside the fitting window -- the largest
excursions in the sample, sitting at exactly the end of the distribution the
quantile reads. **The threshold was pushed up by the events it exists to catch.**

It surfaced on the whitened-residual stage, which went silent: 0/48 rare-event
false alarms and headline-cell recall of 2/32 against the reproduced telemanom's
28/32, firing ten times in eleven million timesteps. Reported as a *broken
measurement* rather than as that design failing, because it does not test that
design.

**The reason it matters is what it did to a result already recorded.**
`lstm-oscfar`'s floor was fitted on the same contaminated pool, so the "the floor
dominates every window" conclusion in `docs/RESULTS.md` section 6b -- which was
read as a finding about the two-term design -- is partly an artifact of this bug.
That section is marked provisional until the curve is re-run on a clean pool. Not
to reopen a closed branch, but because **a conclusion that may not survive a fix
should not sit in the record as though it had.**

Same family as the rest of section 6: a number that was not what the document
said it was. What is different is the blast radius. A wrong constant produces one
wrong result; a wrong *calibration pool* silently rescales every threshold fitted
from it, and it had been quoted in two documents and two designs before anything
looked. The check that would have caught it existed and was not run -- for a
whitened twelve-dimensional residual the length should sit near `sqrt(12) = 3.5`,
and the fitted thresholds were **25.7 to 36.3**. That arithmetic takes a minute
and it is now an assertion in the code, so a broken whitening cannot score
silently again.

**A pre-registered falsification condition would have passed the failure it was
written to catch, and that is the most transferable thing this run produced.**
The two-term threshold design carried a condition with a number attached --
*if the local term binds in fewer than 10% of segments, the floor is doing all
the work and this has collapsed back to the branch we closed*. Deliberately
numeric, so the result would be read against an expectation rather than a hope.

The cell that reproduced the closed branch exactly -- recall 6/46, headline cell
6/32, point 0/11, all three identical -- **binds at 0.127**. Above the threshold.
The condition passes it.

The defect is not the number, and lowering it would be fitting the condition to
the result it missed. **Binding rate measures whether the local term wins
somewhere, not whether it wins where it matters**, and a rule winning 13% of
segments scattered through quiet history is a global rule with decoration. The
general form is worth more than the instance:

> **A condition on the mechanism is only worth pre-registering if it is harder to
> satisfy than the outcome it stands in for.** This one was easier, so it could
> only ever have added false reassurance.

What actually falsified the design was the outcome. And the reason to record this
rather than quietly fix it is that **the defect is invisible when the design
succeeds**: had the corrected calibration worked, the condition would have been
cited as evidence it survived, and nobody would have gone back to ask what it
measured. `docs/DECISIONS.md` D22.

**Then correcting an arithmetic error made the detector worse.** The two terms
had been calibrated independently, each to admit the target rate, and combined
with `max()` -- which admits far less than either, measured at 0.00066 of a
0.001 budget. Fixing it so the combined rule admits what it is asked spent
**2,405 alarm ranges where the broken version spent 126**, at the same budget on
the same residuals. The error had been acting as an unintended out-of-sample
margin. A correct calibration is still the right thing to have; what it revealed
is that the two-term form was leaning on a mistake.

**And the part of the design that was wrong was the part added for safety.** The
floor was there to bound the collapse. Run bare, the local order statistic alone
beats the reproduced telemanom on every axis at once -- 19/32 headline cell at
**85 alarm ranges against 3,548**, +26 lead, 8/48 rare-event false alarms against
30/48. The floor was not load-bearing. It was the thing breaking it.

**Two other predictions from the same programme were refuted, and one of them
reversed sign between channel sets.** Guard cells -- excluding the judged segment
from its own reference window, the CFAR arrangement -- were predicted to have a
small effect concentrated on long events, raising recall. On twelve channels they
raised alarms 27% and *lost* recall, 38/46 to 34/46, with the control fold
collapsing from 12/15 at +18 to 8/15 at +2. On six channels they *helped*. A
change conditional on something not yet identified, stated as an open question
rather than averaged into a mean.

**And a figure written into a planning mock-up was read back as data.** Numbers
put into an options preview to show a table's shape were taken for measurements.
They were placeholders, they were corrected in the open before anything rested on
them, and the lesson is the smallest and most repeatable one here: **an
illustrative figure is still a figure.**

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

## 8. Work item 5: the second player, and what the first one's control fold turned out to be

**2026-08-28.** The GRU was added as a value of one field, `Hyper.cell`, so the
LSTM and the GRU share every line of the trainer and differ by the cell alone;
the NumPy reference gained the GRU's three-gate tick beside the LSTM's four,
and a test that a C++ author folding the third recurrent bias into the input
bias cannot pass. The LSTM's cache key and every published fingerprint were
pinned by test before the field existed and did not move (D26).

**The pre-registration was anchored on a measurement, not a hope.** Under the
frozen rule, the twelve events `lstm-quantile` misses sat at 0.11 to 0.48 of
the threshold, except `id_132` at 0.995 -- a coin toss, predicted neither way.
For the eleven to return the floor had to at least halve, fivefold on fold 0
where eight of them sat, against a predicted movement of a quarter.

**It fell elevenfold on fold 0, and rose 1.8x on fold 1.** Seven events came
back on the control fold of each set, and six were lost on fold 1 with the
validation MSE unchanged to 5%. Two predictions refuted in opposite directions
inside one run, and the correction they force is the same one D17 recorded for
the LSTM's own residual: **the noise floor is a tail statistic and the loss is
a mean, and they move independently.**

**And "fold 0 is the control" was a statement about a fit, not a fold.** The
LSTM's fold-0 fit had stopped at epoch 14 with its best at epoch 3 and a
validation MSE 40x worse than its other folds; section 7 read that as the
data-poor fold behaving as a data-poor fold should. The GRU's fold-0 fit ran
to the cap with its best at 34 and reached the same MSE as folds 1 and 2.
Whether the LSTM stalled because of the cell or because of the seed is one pod
fit away and unmeasured; what is measured is that the fold nobody expected to
move is where nearly everything moved.

**The mechanism condition did what it was written to do.** It failed on fold 0
of both sets while the outcome condition passed: the recovered events came with
79 nominal-step alarms where the LSTM had none, and 958 against 121 on the
subset -- alarms 48 rare events cannot see and 3.6 million nominal steps can.
D22's lesson, applied: a condition that can only confirm is worth nothing, and
this one could say no.

**Two lost afternoons, both procedural, both recorded.** A first pod was
destroyed by hand during setup and had cost 25 minutes of installing; the
default `torch` wheel on the second was built for CUDA 13 and the driver was
12.4, so the fit could not start until the cu126 build replaced it. Neither
touched a number. Both are why the setup is now one script.

**The two stop rules fired -- headline-cell recall meets the LSTM's, and
events predicted not to return did -- and the work stops here**, with the rows
written and the decision unmade. The obvious next measurement is not the TCN;
it is the LSTM's fold 0 refitted, because until that is run the recovery can
be read as a better cell or as a luckier fit and the documents must not choose.

**Addendum, the same evening: the reseed.** The one fit that could separate
the cell from the path was pre-registered and run before anything else. The
banked stall was the path -- a second seed ran to the cap and its floor fell
6.6x -- and it was still 4.3x short of the GRU on the same fold, with the seven
weak events at 0.80-0.93 of its bar and none caught; on `m1-ss5` the second
seed reproduced the first to three figures. The training curves say why: a
plateau at ~4e-04 that the published patience of 10 cuts one path off on and
lets another escape at epoch 24. The verdict rule set before the number
returned no verdict, and that is what was written. What is decided next -- the
row, the protocol, or the TCN -- is a decision and not a run.

## 9. Work item 6: the third player, and the floor that followed the fit

**2026-08-29.** The TCN arrived the way the GRU did -- a third value of the
same field, one trainer, one detector, the frozen decision layer -- with a
contract the two cells do not have: no state at all. Six residual blocks of
causal dilated convolutions, receptive field 253, size-matched to the LSTM at
91,670 parameters. The reference gained a forward pass that refuses a state
and returns none, pinned by perturbation to see exactly 253 steps.

**The pre-registration was the most specific yet, and the most wrong.** Seven
predictions anchored on both incumbents and the reseed; five refuted, most of
them in the same direction. The TCN forecast worse than the GRU on every fold
-- 1.3x to 9.9x -- and its noise floor rose with the error on every fold, in
proportion. The one prediction written as most likely to fail, that the floor
would follow the fit, was the one that held everywhere. The gate row is 9 of
46 events, 9 of 32 headline-cell, against 26 and 27, and it is quiet for the
wrong reason: a bar too high to reach.

**And one fit stalled, on a fold where neither cell had.** `m1-ss5` fold 2
stopped at epoch 13 with its best at 2; section 6i's trainability finding,
recorded for the LSTM, now has a convolutional instance. Whether it is the
architecture or the seed is one refit away and unclaimed.

**A crash of my own, recorded.** The determinism check compared recurrent
gates by attribute and raised on the TCN after both of its fits had run; ten
minutes, a second bundle load whose fifteen operations went unrecorded, and a
one-line fix. The check itself did its job on the rerun: bit-identical under
cuDNN.

**Three rows now exist.** The gate is a decision with all its inputs on the
table, and the documents stop where the measurements stop.

**Addendum: the combination, scoped and not built.** With three rows on the
table, the two-forecaster question banked in section 14 was measured on
cached weights in one pass, expectations first. Not nested -- six and seven
events apart -- so the whitened precedent did not close it. An OR of the two
cells reaches 33/46 and 25/32 on the gate set at the LSTM's own 2/48 and a
third more nominal-step alarms; a mean of the two normalised scores recovers
none of the thirteen solo catches there, because at the moment one model
catches an event the other is at a sixth or three fifths of its bar. On the
six-channel subset the same mean recovers all fifteen and beats both cells,
which says what that regime is and nothing about the gate set. Recorded as
an OR and nothing cleverer, priced at 1.78x the multiplies and two reference
paths, and left for a decision after Phase 1 closes.
