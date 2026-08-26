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
produced. Six times now, a cheap measurement has changed a decision that looked
settled without it.

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
of replacing it. One subtraction per caught event.

Each changed a decision.

> **What we measure determines what we conclude.** A metric set that omits a
> property of interest will confidently rank detectors on the properties it does
> measure, and it will not report that something is missing. Adding a cheap
> measurement is almost always worth it -- and the ones worth adding are, by
> definition, the ones nobody has thought to ask for yet.
