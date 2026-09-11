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
came out well: 28 of 32 headline-cell events against `rstd`'s 3. **(!) Both
numbers were wrong, and it took until work items 9.5 and 9.6 to find out: the 3
was float32 accumulation and is 25 (D37), the 28 belongs to a detector that was
disqualified on false alarms, and the detector that flies catches a strict
subset of the corrected floor's events (D38). This paragraph is the record of
what was believed, not of what is true.** The false-alarm
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

### 6.1 Two confident hypotheses about the threshold criterion, both wrong

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

### 6.2 The project's own preferred mechanism, refuted by its own scoreboard

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

### 6.3 The early-warning claim, measured from a start it never reached

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

### 6.4 A gate metric compared across two different populations

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

### 6.5 A structural claim that was really a configuration

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

### 6.6 A threshold calibrated on nominal data that was not nominal

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

### 6.7 A falsification condition that would have passed the failure it was written to catch

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

### 6.8 Correcting an arithmetic error made the detector worse

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

### 6.9 A figure from a planning mock-up, read back as data

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
stopped at epoch 13 with its best at 2; `docs/RESULTS.md` 6i's trainability finding,
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
table, the two-forecaster question banked in `docs/MODELS.md` 14.7 was measured on
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

## 10. Phase 1 closes: the exam, and what it examined

**2026-08-29.** The two held-back sets nominated before any decision-layer
tuning began were scored once each, on a GO, with the commands and twelve
predictions committed first (`docs/MODELS.md` 18). Nothing was described,
swept, refitted or run twice.

**On a spacecraft nothing here had ever seen, the recipe held.** Fit on a
third of Mission 2's history, calibrate on your own nominal residual with no
label and no tuning: four rare events alarmed in 424, and not one nominal
timestep in four million, for the LSTM, the GRU and the TCN alike. The
per-channel floors did not hold -- `rstd` alarmed on a sixth of nominal time
-- which is the cross-channel claim of Objective.md 1.1 answered on
independent data. That is the adoption number.

**On a later period of the same spacecraft, the floor did not hold.** On
`m1-g3` fold 1 the threshold calibrated on the first half of the history sat
under 87% of the next window's nominal residual, for both cells identically;
the recall it "scored" is a window-long alarm cut into pieces, and the
harness's own rule says a recall bought that way is not a finding. The
pre-registration had asked how far recall and rare alarms would move; it had
not asked whether the noise floor would still be the noise floor. Nothing was
re-tuned. The finding went into D29 and into what Phase 2 inherits: a
threshold is a parameter with a provenance, recalibrated in orbit, not a
constant fixed in the past.

**The union's edge did not survive either exam.** Its Mission-1 economy -- the
GRU's rare alarm hiding inside the LSTM's -- was a coincidence: on Mission 2
the two cells' four rare alarms each were eight different events. `gru-quantile`
flies alone (D29).

**Thirteen measurements changed a conclusion in this project; the fourteenth
changed the question.** Every earlier one asked which detector, which
threshold, which cell. This one asked whether a floor measured yesterday is
still the floor today, and on one subsystem in three folds the answer was no
twice. That is the question the flight component is built around.

## 11. Phase 2, work item 9.9: four rungs against a mechanism that was not there

**2026-09-08.** This section is written at a work-item boundary, which is when
`docs/HARNESS.md` 5a says it gets written, and it records two mistakes rather
than one result.

### The study code was never committed

Between 2026-09-03 and 2026-09-04 this project ran stage 5 and four rungs against
telemanom's published Table 2, and published five results from them: D50, D51,
D52 and `docs/MODELS.md` 26.19 to 26.28. **Nine commits landed in that window and
not one of them touched a file under `scripts/`.** The artifacts are real and
under `runs/smap-msl/_forensics/`; the code that produced them was written
somewhere outside the repository and is gone.

So every figure in that ladder is sourced to a named artifact, which is the rule,
and **none of it is reproducible from the repository**, which nobody had written
a rule about. `74/98` and `98/221` could be quoted and could not be re-derived.

The rule this produces, and it is narrow enough to keep: **a run that produces a
documented figure lands its script in the same commit as the figure.** Not the
next commit, not a tidy-up later. The pre-registration commit may precede the
code; the OBSERVED commit may not.

### And the source was read without being kept

`docs/MODELS.md` 26.21.1 quotes `errors.py:363-371`. 26.23.1 quotes
`errors.py:365-371`. 26.25.2 quotes `process_batches`. Those readings were done
properly -- the project's rule is that a divergence is read from the source and
located, never guessed, and it was followed. **No copy of `errors.py` was kept**,
so the quotations were assertions with line numbers attached, and neither the
line numbers nor the code around them could be checked by anyone who was not
there.

On 2026-09-08 the source was vendored (`third_party/telemanom/`, D53) and read in
full. Five recorded readings turned out to be wrong, and one of them had cost
four pre-registered rungs:

- telemanom clips each window to its newest `batch_size` (`errors.py:355-359`),
  so it is **causal** after its opening window, and its cross-window accumulator
  unions **disjoint** batches.
- **There is no union over roughly thirty overlapping verdicts.** Rungs 1c, 1c-i
  and 1c-ii were built to reproduce a mechanism the source does not contain, and
  the "flyable reproduction" is more permissive than the published algorithm
  rather than faithful to it. The arm that was already there, `1a+1b`, is the
  faithful one.
- The published precision denominator is `matched events + unmatched ranges`
  (`detector.py:117-136`), not ranges over ranges, so every precision figure in
  the ladder is the generous statistic.
- `~91` was `80.0 / 87.5`, arithmetic that appears in no document. The paper's
  own number is **twelve false positives**.

**What went wrong is not "the source was not read".** It was read. What was
missing is that the reading left nothing behind, so it could not be audited, and
a wrong reading survived four rungs of careful, pre-registered, correctly-stopped
work. Every stop condition in 26.21 to 26.28 fired or held exactly as designed.
Z4 caught a two-lever arm. Q4 passed 75 of 75. X3 held on 1,721 of 1,721 calls.
**The discipline worked perfectly on top of a premise nobody could check**, which
is the most expensive kind of correct.

### What it cost, and what it bought

Five reads at 165 Class B each, four pre-registrations, and a decision entry that
is now superseded at its premise. What it bought is a precise target: the
reproduction is chasing **12 false positives against our 81**, on a population
that matches the paper's exactly on MSL, and five named mechanisms in the source
that this stack does not implement. That is a better question than the one the
ladder started with, and it was only reachable by getting the first one wrong in
public.

### The two bugs the smoke caught, and why they are worth a paragraph

Work item 9.10 spent 7 Class B on a two-channel smoke before its 165-Class-B
read, and the smoke's only stated purpose was to prove the weight-cache key still
hit. Two defects were caught before a single operation was spent on the real run.

**The fit context was dropping the channel id.** The new script built its
`Context` with a literal placeholder where `scripts/smap_stage2.py` passes the
channel. `context.channels` is the second element of the weight-cache key
(`src/sentinel_models/detectors.py:421-424`), so every one of the **318 banked
fits** would have missed, and the run would have silently refitted 152 models
instead of reading them -- **D14's failure exactly**, which cost 45 minutes the
first time and is still open as a decision. It was caught by reading the new
`ctx_for` against the old one rather than by running anything.

**The buffer dilation was quadratic.** Transcribing `errors.py:291-298` literally
-- concatenate `i + arange(1, 100)` and `i - arange(1, 100)` for every exceeded
index -- is correct and unusable: over twelve arms it is hours. Dilating the
contiguous runs instead gives the identical set in linear time, which is the same
argument `src/sentinel_models/telemanom.py:208-224` already makes for the same
operation.

**Neither would have announced itself.** The first produces correct numbers after
an unnecessary hour of fitting and a weight store that has silently doubled; the
second just looks like a slow script. The smoke was priced at 6 Class B in the
pre-registration for the first of them, and it paid for itself twice.

### A compute plan that was measured and still wrong

Work item 9.11's fits projected to 111 minutes serial, over the hour the
pre-registration set as the threshold for asking, so the options were put with
numbers: one published fit was measured at **508 MB peak RSS** and four workers
were sized at about **2.8 GB against 4.3 GB free**. The parallel path was
validated first against three already-banked channels, and it hit the cache with
the weight store unmoved -- proof that fanning the fits out changed neither the
key nor the weights.

**Then the operating system killed the run for low memory, after the fits and
before the artifact.**

Two costs were left out of an otherwise honest measurement. Each **spawned worker
imports torch independently**, several hundred megabytes before it fits anything,
which a single-process measurement cannot see. And the **parent had already grown
past its own baseline** by the time the fits started: it was holding 81 channels'
train and test arrays and both cells' smoothed-error caches. The per-fit figure
was right; the total was assembled from it as though the fit were the only thing
in memory.

**What survived is the part that was expensive.** Weights are written as each fit
completes, so 77 of 78 were on disk and the second run refitted one channel and
read the other 77 from cache. **What did not survive is the bookkeeping**: 165
Class B were spent and never reached the ledger, because the process died before
`ops.commit`. That is commit `75cc846` from the other side -- it moved the
artifact ahead of the ledger so a ledger failure could not take the result, and
here the result and the ledger were both lost while the weights, which nobody had
thought of as the durable thing, came through.

**The transferable part is not "measure memory".** It was measured. It is that a
per-unit measurement multiplied by a worker count is an estimate of one term in a
sum, and the other terms -- the runtime each worker re-imports, the parent's own
growth since the baseline -- are the ones that decide whether it fits.

## 12. Work items 9.12 and 9.14: two hypotheses of mine, measured and refuted

**2026-09-08, written at the work-item boundary** where `docs/HARNESS.md` 5a says
it gets written. Section 11 ended with the reproduction finally reaching a
precise target: twelve false positives, on a population that matches the paper's
exactly on MSL. Two arms ran against it. Neither closed the gap, and both refuted
a hypothesis this project had written down first, which is the part worth
keeping.

### The residual rung changed nothing, and its own author's story was wrong

Arm R was the last transcription difference left. telemanom computes its residual
over the supervised region only and then replaces the first `l_s` smoothed
samples with the mean of the first `2*l_s` (`errors.py:48-64`), where ours ran
the EWMA across the full test array including the padded warm-up. Our `e_s`
entered the scored region carrying smoothing state from padded residuals, and
`mean_e_s` and `sd_e_s` set every epsilon. It looked like the kind of thing that
would matter.

**It equals Arm T in every cell.** 46/104, 83.6% precision, 9 false positives,
the same nominal rate to four decimal places. It moves individual events between
forensic classes and moves no verdict.

The interesting refutation is the other one. `docs/MODELS.md` 28.8 had argued --
and it was mine -- that a better forecaster produces **smaller** residuals, so the
published absolute floors would filter them out and that is what silences MSL.
It is D18 one layer down, and it reads well.

**Measured, the opposite is true.** `max(e_s)` on MSL is **0.7323** at the median
against a `sd(values)` of 0.5721, and **0 of 27** channels fall below the 0.05
floor. The residuals are not small. They are large, and `errors.py:339` is
nowhere near binding on any channel.

So the forecaster is simply doing badly on MSL. That is a different problem from
the one 28.8 described and a considerably more ordinary one, and the wrong
hypothesis is kept in full with its reasoning, because a principle that was
arrived at by three correct guesses and one wrong one is better supported than
one that was never tested.

**R2 is worth a paragraph for the opposite reason.** Its refutation clause said
five or more `offset` events would reopen the coordinate question, and thirteen
fired. The stop was discharged on evidence rather than waived: a frame error puts
every one of them at plus or minus 250, and the measured distances scatter from
-503 to +427 with **0 of 13** in that band, on both spacecraft, with the same
class firing at the same rate on SMAP where detection works. The defect was in my
classifier, which trips whenever a peak forty samples from a narrow label falls
outside the span. A stop that fires and is then discharged by measurement is the
mechanism working, not being bypassed.

### The detector that predicts its own uncertainty predicted a constant

By then the reproduction had answered what it was asked. The gap is not the
scoring rule, the commands, pruning's rung, cross-window tracking, the
aggregation, the window regime or the training configuration -- each measured and
closed. **It is the residual itself**, and D55 had just said why every transcribed
constant failed to travel.

`gru-zscore` was the answer to that: the same GRU with a head emitting `mu` and
`log sigma^2`, a Gaussian likelihood on nominal data, and `z = (x - mu) / sigma`
under D25's unchanged threshold. **`z` is dimensionless by construction**, so D55
is satisfied structurally rather than by choosing better constants. It stops
chasing Table 2 and changes the detector instead of the transcription.

**H4 was written as a stop precisely so this could fail, and it failed.** The
per-channel `sigma` was predicted not to be approximately constant -- coefficient
of variation above 0.25 on more than half of channels. Measured: **0.0504 at the
median, above 0.25 on 11 of 79.** 31.5's own sentence, written before the run,
describes the outcome exactly: the head "learned a global scale, `z` is
`|x - mu|` divided by a constant, and the arm is the old detector with extra
parameters."

MSL came in at 5/36 against `A0`'s 16/36 and the paper's 25/36, with 27 false
alarms against 2. Both refuted.

**And H5 is why this is a finding rather than a bug report.** Held-out nominal NLL
improved on **79 of 79** channels, median best epoch 34 of 35. The likelihood
objective trained. This is not an optimisation failure and it is not a plumbing
failure: the model could have learned a varying sigma and did not. A single
univariate channel gives the likelihood no reason to vary it with state, which is
named as the next question and deliberately not registered as an arm yet.

### Three defects, and two of them never reached a read

Two were caught by the smoke. The relative early-stopping rule raised the bar on
a **negative** loss, which is nonsense the moment an objective can go below zero
and which no MSE arm could ever have exposed; it is fixed sign-safely, is
identical for every non-negative loss, and is pinned by a test. And `Weights`
refused a doubled head, which is a real finding rather than an inconvenience: **a
Gaussian head is not representable in `model.bin` version 1** (D30), so this
detector could not fly as it stands even if it had worked.

The third survived to the artifact. **The weight store grew by 0, not the
pre-registered +81**, because `build_zscore` fits through `lstm.train` directly
while the cache lives in `ForecastDetector.fit`. Arm H's weights are therefore
never persisted and the arm is reproducible only from its seed. That is the
weaker half of `docs/NARRATIVE.md` 11's rule arriving again from a different
direction: the script is committed and the figure is sourced, and the fits behind
it are not on disk.

### What the two arms bought

Nothing on the leaderboard. Stage 4's 10/38 still stands unreplaced, and the
fourth arm in a row failed D48's way -- `gru-zscore` alarms on 19.41% of nominal
time at D25's threshold, and swept it saturates at the grid maximum still holding
1.18% with 0 of 39 in-range contextual. A constant divisor cannot repair a
distribution shift.

What they bought is the end of a line of inquiry, established rather than
assumed. Six model variants have now been measured on this data and none beats
10/38. The reproduction is closed as the route to better recall, and the
remaining contexts -- commands, and a testbed with real coupled physics -- are
where the question goes next.

## 13. Work items 9.13 and 9.15: two stops, and the context that was costing

**2026-09-09**, at the work-item boundary. Two arms ran, both had a structural
prediction written as a stop, and **both stops fired**. Neither run was wasted,
and the reason is the same in both cases: the stop is what made the negative
result legible.

### The guards were not what silenced MSL, and a "relaxation" was not one

Work item 9.13 replaced telemanom's two candidate filters, absolute in the units
of the data, with `mean(e_s) + 1*sd(e_s)`. D55 had established the principle and
30.1 fixed the multiplier at 1.0 in advance, with a sentence saying that needing
another value would be a finding about the form rather than a parameter to search.

**MSL moved by zero events.** Not one, and not one false alarm either. With the
residual rung already inert and the scale hypothesis already refuted, that closes
the third of four candidates and leaves pruning at `p = 0.13`, which 29.4 and
30.2 had both named in advance.

**The more interesting refutation was the structural one.** G4 predicted that the
dimensionless arm's alarm set would be a superset of the absolute arm's **on every
channel**, because every replacement was assumed to be a relaxation. It held on 79
of 81. On `D-2` and `G-6` the new arm **lost** alarms the old one had -- on `G-6`
the whole-window bail-out fired twice under the absolute floor and sixty-four
times under the dimensionless one.

The mechanism is D55 turning round on itself. A noise-relative floor is not
uniformly looser than an absolute one: it is looser where the residuals are small
and **tighter where they are large**, and 29.4 had already measured that this data
has both. So substituting a dimensionless constant changes **which** channels a
filter binds on, not merely how hard.

**And in aggregate the arm still looked like a clean relaxation** -- bail-outs
1,836 to 996, alarm rate 2.05% to 3.80%. A pooled statistic would have said
*relaxation*, and stopped there, and been wrong about two channels nobody would
have gone looking for. G4 was written per channel because a structural claim is
cheap to write and expensive to skip. That is now the second time (26.26 was the
first).

### Then the commands, which turned out to be the thing that was hurting

Work item 9.15 was the one D57 made matter. The re-framing lists three contexts;
commands-to-telemetry was the only one this dataset could test and it had been
tested once, by D49, which found commands making the detector **worse** and then
carefully bounded itself to the configuration it had measured.

The two conditions outside that bound were now buildable. So D59 re-opened D6
narrowly, on exactly those two and nothing else.

**The counterfactual was the arm that did not exist.** Arm T already carries the
commands -- telemanom's `X` is all columns, telemetry and the one-hot indicators,
while `y` is column 0 alone -- so the work was to build the arm **without** them
and hold all eight other published differences identical.

**It gained eight events, removed five false alarms, and lowered the alarm rate at
the same time.** 54/104 against 46/104, 93.1% precision against 83.6%, 4 false
positives against 9, 1.8815% nominal against 2.0450%. An arm that is better on
every axis at once is not a rate artifact; the quieter one is also the one that
catches more.

K1's refutation clause had said, before the run, what this would mean: it would
put D49's finding **outside its own bound**. It did. D49 was right and did not
need the caution it gave itself on that axis -- which is an argument for bounding
a finding, not against it. The bound cost one run to discharge and would have cost
a false general claim to skip.

**The second arm stopped on its own design claim, for the second time.** C2 gave
`gru-zscore` the commands, testing the reason 31.9 had named for sigma's collapse:
a single univariate channel gives the likelihood no reason to vary sigma with
state, and commands are state. Sigma's coefficient of variation **doubled**, from
0.0504 to 0.1145 at the median, and cleared the 0.25 bar on 16 of 78 channels
against a prediction of more than half. Directionally right, numerically
insufficient -- the same shape as rung 1b. The likelihood trained on 78 of 78, so
once again the head could have learned what it was asked to learn and did not.

### The instrument nearly cost the stop, and a smoke caught it

`zscore_diagnostics` collected only the old cell, so C2's `sigma_cv` would never
have reached the artifact -- and K3, the stop the pre-registration adjudicates
**first**, would have been unadjudicable from the record afterwards.

That is 27.8's defect precisely: S1 and S2 could not be settled there because the
instrument did not retain what the check needed. It was caught by running a
three-channel smoke and reading its artifact, not by thinking carefully about the
code, which is the same way the weight-cache defect was caught in 27.6. **The
smoke has now paid for itself three times**, and its cost each time is nine
operations.

A second defect was mine and smaller: the peak-RSS meter guessed its unit from the
magnitude of `ru_maxrss`, which is bytes on macOS and kibibytes on Linux. Guessing
a unit from a magnitude is how a plausible number becomes a wrong one, and it was
replaced before it produced a figure anybody read.

### What was left undone, and named as such

**K2 has no verdict.** The pre-registration said each arm would also be swept to a
rate matched to stage 4's 0.6838%, and for C1 that sweep was never implemented --
it was registered as a `port` arm, which has no dial. So the one figure that would
license comparing C1's in-range contextual count against stage 4's 10/38 does not
exist, and 20/39 sits in the artifact next to Arm T's 12/39 with both above the
alarm rate at which this project permits a per-event number to be quoted at all.

The honest thing is that the comparison is **owed**, not that it is nearly there.
It is recorded in 32.7, in D60's consequence 6 and in the status checklist, and it
is not folded into anything else as a consolation.

### What the two together bought

Four candidates for MSL's silence, three now closed. One context measured and
found to cost rather than pay. One principle -- ship dimensionless constants --
qualified by measurement into something more useful than it was: dimensionless is
not automatically safer, it is differently distributed, and a mission whose
channels differ in residual scale will see a filter move between them.

And a best arm nobody was looking for. C1, which is the published reproduction
with an input **removed**, reaches 93.1% precision against the paper's 87.5% on
four false positives against a scaled eleven. Its recall is still 51.9% against
80.0%, so it settles nothing about parity. It is the highest this project has
reached on the population that matches the paper's, and it got there by taking
something away.

## 14. Work items 9.16 to 9.18: nine arms, a ceiling, and a pre-registration that ignored its own record

**2026-09-09**, at the boundary where the improvement ladder ends. The question was whether
anything could beat stage 4's 10 of 38 at a flyable alarm rate. Nine arms answered no, and
the way they failed is more useful than the answer.

### The dial, and an advantage that lived at the wrong alarm rate

C1 -- the published reproduction with the command inputs removed -- was the best arm this
project had produced: 54 of 104 at 93.1% precision, beating the paper's own precision on a
third of its false positives. Arm 1 gave the port a dial and swept it down to stage 4's
rate.

**It collapsed.** Six of stage 4's 38 against stage 4's 10. The first grid was too coarse to
see the crossing and the refinement made the answer worse rather than better: even given a
rate **louder** than stage 4's, C1 reaches only 8. **C1's advantage was a precision result
all along, and precision at 1.88% nominal is not a product.**

### Choosing a forecaster without looking at a label

Arm 2 was the one worth doing properly. Thirty-two configurations over cell, lookback,
width and epochs, chosen on **held-out nominal forecast error alone** -- no label, no
recall, no alarm rate. The rule mattered more than the result: with 104 labelled sequences,
selecting on recall is selecting the model that best fits the labels, and this project has
spent five work items avoiding exactly that.

The winner is **the published configuration with the cell swapped**. Lookback 250, width 80,
epochs 35 -- three of four axes landed on telemanom's own values, out of a grid that offered
alternatives for every one. Six days of transcription had been chasing a configuration that
was very nearly right.

**And a prediction I took from my own measurement was wrong.** The cost surface, run on
synthetic arrays, showed 35 and 70 epochs costing identical wall clock at every corner. Equal
cost means equal epochs run, which means equal weights -- so I registered P5, that the epochs
axis was inert, and kept the axis in the grid anyway on the argument that dropping half a
registered grid on a timing measurement is the edit this discipline exists to prevent.

On real telemetry 15 of 16 pairs differ, by up to 20%, and **more training is consistently
worse**. The mechanism is in the published configuration itself: `restore_best=False` keeps
the last weights rather than the best, so a longer run simply travels further past the
minimum. The published 35 is a budget doing real work. **Keeping the axis cost an hour and
bought a finding the shortcut would have destroyed.**

### The arm where the pre-registration was the defect

Then four decision-layer variants: a transition-aware floor, per-channel calibration, both
together, and a tree forecaster.

**None of them can reach a flyable alarm rate at all.** Not at a coarse grid, and not with
the dial extended to `1e-8` -- below one over the length of every training split, which
saturates the threshold at the training maximum. The quietest points are 2.01%, 13.55%,
21.49% and 5.17%, against a target of 0.68%.

The reason has been on the record since 2026-09-03. **D48: on SMAP/MSL no train-calibrated
threshold transfers, for any arm.** The S-family is a static train-calibrated quantile --
D25's recipe, split by regime or by channel -- and D48 exists precisely to bound D25 to
stationary regimes on this dataset.

**The pre-registration cites D25 and never mentions D48.** That is not a coding error caught
by a smoke; it is an arm designed against a route already measured not to work, by someone
who had written the measurement up six days earlier. The arms ran, cost two reads, and
returned a result that was already in the file.

The transferable part is narrow and unflattering: **a pre-registration is not only a
statement of what will be measured, it is a claim that the thing is measurable**, and
checking that claim means reading the entries that bound the ones being cited.

### The ensemble that improved the forecast and silenced the detector

The sharpest negative came from the most conventional idea. Three seeds, forecasts averaged
before the residual -- the standard way to buy a better forecast.

It worked. The residual came out smooth enough that a train-calibrated quantile **finally
transfers**, where nothing else in the S-family could reach the target at all. And the
detector caught **2 events in 104**, at zero false alarms and 100% precision.

**A better forecaster and a silent detector.** That is D18 again, from the other end: a
threshold rule fitted to one residual scale degrades when the residual gets better, and this
time the improvement was ours rather than the published method's.

### Trees, which lost both halves of their own split

The last arm changed model class entirely: histogram-binned gradient-boosted trees on
label-free lag features, with an isolation forest as a control, judged by a step-likeness
score fixed in advance and cut at its median so no threshold could be searched.

Trees lose the steppy half 7 to 27 and the smooth half 10 to 29, **while running more than
twice as loud**. No hybrid is worth building, and the fixed cut is what makes that
conclusion cheap rather than arguable. The isolation forest reached 1 of 104, which at least
confirms the premise the whole project rests on: this data's anomalies are not detectable
without forecasting.

**One number from the losing arm is worth keeping.** Trees fit in 2.23 seconds per channel
against the GRU's 43.3 -- about twenty times cheaper, at less memory. For a mission that has
to train on the ground with no ML staff, that ratio is a real argument, and it belongs to the
toolkit even though the arm lost.

### The same instrument defect, for the fourth time

27.8 could not adjudicate S1 and S2 because alarm sets were not retained. 32.7 nearly lost
K3 because `sigma_cv` never reached the artifact. 33.7 could not settle M1 because the sweep
kept a count instead of a caught set. And this work item printed `nominal 0.0000% caught 0`
for every arm that found no qualifying operating point -- **a placeholder rendered as a
measurement**, sitting beside arm rows showing 2% to 21% -- and retained neither the
per-channel fit cost nor the step-likeness score its own pre-registration required.

Four times, the same shape: **a value computed and not retained is a value that does not
exist.** Each time it cost a read to learn again.

### What the ladder bought

Not a better number. Nine arms across the forecaster, the decision layer, the model class
and the inputs, and the best of them reaches 8 of 38 where stage 4 reaches 10.

What it bought is that **10 of 38 is now a ceiling rather than a starting point**. It was
measured by elimination, against arms chosen in advance and adjudicated whether they held or
not, and every negative is on the record with the reasoning that produced it. The pipeline
freezes there.

And it settles where the remaining question lives. Every arm here varied a model or a rule on
81 unsynchronised univariate streams. The claim D57 re-framed has one context left
unmeasured, no public dataset can measure it, and the F' Ref physics testbed is the only
venue that can.
