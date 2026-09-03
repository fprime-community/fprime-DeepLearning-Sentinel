# Phase 1, start to finish

*Written at the close of Phase 1, 2026-08-29, for a reader with no prior context. Every claim
names the decision entry (`docs/DECISIONS.md`, D1-D29) or the result section (`docs/RESULTS.md`)
it rests on, and every number was read from a committed run artifact.*

## 1. The problem

Onboard fault protection on a spacecraft today is a per-channel limit check: `if voltage < 3.2,
alarm`. Hundman et al. (KDD 2018), studying real, expert-confirmed anomalies on the SMAP satellite
and the Curiosity rover, found that **41% of them were contextual** - every individual channel
stayed inside its limits while the combination or the trajectory was wrong. No limit can be tuned
to catch that class, because the wrongness is not in any single value. The same paper published
the method that does catch it: forecast the telemetry with an LSTM and alarm on sustained
divergence. ESA's OPS-SAT showed deep-learning inference runs acceptably on CubeSat-class
hardware. What did not exist was a reusable flight component in F', NASA/JPL's open-source flight
framework. That gap is the project (Objective.md sections 2, 3).

## 2. The method, in plain language

A small neural forecaster is trained on the ground, on the mission's own nominal telemetry: given
the last 250 readings of every watched channel, predict the next ten. In flight the component
runs that forecast every cycle and compares it with what arrived. The per-channel error is
smoothed, the largest across channels is taken, and one threshold is applied - not a chosen alarm
budget, but the measured noise floor: the 99.9th percentile of that same statistic over the
mission's own nominal history (D25). Cross that floor and a standard F' event is raised. The
component warns and never commands; the model is frozen in flight; nothing learns online
(Objective.md section 11).

Two design choices carry most of the weight. **Normalisation is identity** (D2): ESA-ADB is
scaled within channel groups, and rescaling each channel separately would erase the amplitude
ratios between related channels - exactly the relationships the method watches. And **the
forecaster is multivariate**: one model predicts every channel from every channel, so a broken
relationship raises the error on the channel that became unpredictable. Published telemanom
trains one model per channel; that cannot express cross-channel structure, and the Phase 1 gate
was amended to say so (Objective.md section 12; `docs/MODELS.md` section 1).

## 3. The journey, as it happened

### The referee was built before the players

The first thing built was not a model but the scoring harness (`src/sentinel_eval/`), with a
rule that it never imports a model (`tests/test_layering.py`), so that LSTM against GRU against
TCN would be a table of numbers rather than an argument. It spent its first week catching the
project's own mistakes: a pre-registered prediction that a moving average would score near the
floor on cross-channel recall failed - it scored 29/31 - because it bought that recall with 5,535
alarms for 42 events. The gate metric became event-wise F0.5, and bare recall is never reported
alone anywhere (D3, `docs/HARNESS.md` section 7). Four more defects followed, all in the
measurement design, all recorded (`docs/NARRATIVE.md` section 2). The floor was set: a rolling
standard deviation at F0.5 0.250, 3 of 32 headline-cell events (`docs/RESULTS.md` section 1).
**(!) Corrected 2026-09-01: those two figures were wrong. The floor is F0.5 0.676 and 25 of
32 (D37). What follows is the Phase 1 story as it was lived, against the floor as it was
then believed; `docs/RESULTS.md` 1a states what the correction does to the conclusion.**

### The reproduction, and the constant that had disabled training

Work item 4 reproduced telemanom's detection method with one multivariate LSTM (91,640
parameters), trained in PyTorch and scored through a plain-NumPy forward pass held to torch at
1e-5 - the blueprint the C++ will be transcribed from (`docs/MODELS.md` sections 2-3). The thesis
appeared to hold at once: **28 of 32 headline-cell events against the floor's 3** (`docs/RESULTS.md`
section 2) -- **(!) and neither figure survived. The floor's 3 was an arithmetic defect and is 25
(D37); the 28 is `lstm-telemanom`'s and the detector that flies scores 22/32; and the flying
detector catches a strict subset of the corrected floor's events (D38). Section 6l is the
corrected comparison.** The false-alarm rate did not: 22 of 48 commanded manoeuvres alarmed on, and a third of the
published method's input - the telecommands - had never been wired in (D6).

Then a published constant turned out to have disabled training. telemanom's `min_delta = 3e-4`
is an absolute quantity in the units of the loss; on ESA-ADB the validation loss is about 1e-4,
so after the first epoch the bar was negative and no epoch could ever clear it. Every fit in the
project had kept its first epoch, discarding weights up to 8.8x better. Replaced by a relative
`min_improvement`, the forecast improved about **fortyfold** - and every precision number got
worse, because the published threshold, transcribed faithfully, went from 182 alarm ranges to
**3,548** on identical data (D17, `docs/RESULTS.md` section 6a). A threshold that suits one model
does not suit a better one; thresholds belong to the model, not the method, and must travel
outside the weights (Objective.md 14.10).

### The threshold, measured rather than tuned

The obvious response - sweep the threshold until the numbers look right - was refused as fitting
to the labels (`docs/MODELS.md` section 7). Instead the selection criterion was measured over
**5,684,580** reference windows: 92.6% of the windows that selected anything selected the range
minimum, so the "nonparametric dynamic threshold" had reduced to a fixed multiplier on the local
scale (D18, `docs/THRESHOLD.md`). An order-statistic rule was pre-registered, run, found to have
been miscalibrated by an arithmetic error, retested as designed, and refuted on the retest -
with the pre-registered falsification condition itself found to have been too easy to pass (D20,
D22). A whitened relationship test was built and measured: 2 of 48 rare events alarmed on, 21 of
32 headline events caught (`docs/RESULTS.md` section 6d, D24). And then a plain global quantile of
the nominal residual - the rule that had been closed early on one-epoch weights - was re-measured
on trained weights and matched the whitened rule event for event, at a fourteen-times-lower
nominal-step alarm rate and a simpler flight implementation. It was frozen as the decision layer
for the architecture gate: `lstm-quantile`, F0.5 0.838, 26/46, 21/32, 2/48, 0.002% of nominal
steps (D25, `docs/RESULTS.md` section 6g).

### Lead time, measured honestly

Every lead-time figure the project had published - a median of +26 timesteps - was dated from the
start of an alarm range that `error_buffer` had widened backwards from a crossing that had already
happened. Measured from the crossing, the median lead is **0.0**, and fifteen of thirty-eight
detections on the gate set had no emission overlapping the event at all (D21, `docs/RESULTS.md`
section 6f). The claim "+26 timesteps of early warning" was retired in Objective.md section 1.1
and the surviving claim stated in its place: the first and only observer of cross-channel
breaks in an F' deployment -- which is a statement about what an F' deployment contains, and
stands. **(!) The catch comparison that used to be offered as its evidence is withdrawn
(D37, D38); see `docs/RESULTS.md` 6l and work item 9.7.**

**(!) And the claim was restated on 2026-09-03, after Phase 1 closed.** Sentinel catches
anomalies a limit check can never see: on SMAP/MSL, 39 of 43 labelled contextual anomalies stay
entirely in range, no per-channel statistic reaches a flyable alarm rate there, and the
forecaster under a **dynamic** threshold operates at 0.68% and catches 10 of 38 (D46, D48,
`docs/MODELS.md` 26.18). That result is `gru-telemanom` per channel and univariate, **not** the
`gru-quantile` configuration this report describes, and it post-dates every measurement below. The break-to-limit lead remains a Phase 3 measurement on a real clock.

### Three architectures, one variable

The GRU arrived as a single field of the LSTM's configuration - the cell - with every other line
of the trainer shared, so the two differed by the cell alone; the LSTM's cache keys and published
fingerprints were pinned unmoved by test (D26). Pre-registered before the fit, the GRU recovered
**seven of the eleven weak events** the frozen layer had missed - all on fold 0, all because its
noise floor there fell elevenfold - and lost six on fold 1, where its floor rose 1.8x with the
validation loss unchanged: the noise floor is a tail statistic and the loss is a mean, and they
move independently (`docs/RESULTS.md` section 6h). A reseeded LSTM fit on fold 0 then showed the
LSTM's stall there was one optimisation path - and still left it short of the GRU (6i):
trainability became a recorded gate consideration. The TCN, size-matched to the LSTM at 91,670
parameters with a 253-step receptive field and no state at all, forecast worse on every fold and
its floor rose with the error on every fold; it caught 9 of 46, a strict subset of both cells,
and one of its six fits stalled (D27, section 6j).

**The gate (D28)** selected the GRU on Objective.md section 8's criteria: criterion 1 a tie at the
harness's own resolution - one event apart each way on the two channel sets - with the LSTM ahead
on event-wise F0.5 and the entry saying so; criteria 2 to 5 and trainability to the GRU: 71,160
parameters, 0.78x the multiplies per cycle, one state vector, no stall on any fold. The LSTM stays
the published-reproduction baseline; the TCN rows stay as the measured answer to "why not
stateless".

### The combination, scoped and declined

With three rows on the table, the union of the two cells' alarm streams was scoped on cached
weights, expectations first. Not nested - six and seven events apart - so the question was open.
A plain OR reached 25 of 32 headline events at the LSTM's own 2/48; a combined score recovered
none of the thirteen solo catches, because at the moment one model catches an event the other is
at a sixth or three fifths of its bar. An OR and nothing cleverer, priced at 1.78x the compute
and two reference paths (`docs/MODELS.md` section 17).

### The exam

Two sets had been nominated before any decision-layer tuning began and refused by every analysis
script since: `m2-ss1`, a second spacecraft with 424 rare nominal events test-side, and `m1-g3`,
an untouched subsystem of the first. They were scored once each, on a GO, with twelve predictions
committed first, under a rule written in advance: whatever they said would be the finding, and
nothing would be re-tuned (`docs/MODELS.md` section 18).

**On the independent spacecraft the recipe held.** Fit on a third of an unseen mission's history,
calibrated on its own nominal residual with no label and no tuning, the LSTM, the GRU and the TCN
alarmed on **4, 4 and 6 of 424 rare events and on none of 4,155,841 nominal timesteps**. The
per-channel floors did not, on the numbers available at the time: `rstd` was recorded
as alarming on a sixth of nominal time. **(!) Corrected 2026-09-01 (D37): that 17.30%
was an artifact of `baselines._rolling`'s float32 accumulation. Re-scored, `rstd` alarms
on 126 of 4,155,841 nominal steps -- 0.003%, not a sixth -- and on 22 of 424 rare events
rather than 84. The forecasters still win the adoption number decisively, 4/424 against
22/424 and 0 nominal-step alarms against 126, so D29 stands; the margin it is quoted
against does not.** The union's Mission-1
economy did not transfer - its eight rare alarms were the members' four and four, disjoint - and
its recall edge evaporated on `m1-g3`; `gru-quantile` flies alone (D29).

**On a later period of the same spacecraft, the floor did not hold.** On `m1-g3` fold 0 the recipe
transferred cleanly; on folds 1 and 2 the threshold calibrated on the first half of the history
sat under **87%** of a later window's nominal residual, for both cells identically. The recall it
"scored" there is a window-long alarm cut into pieces, and the harness's own rule says a recall
bought that way is not a finding. The forecaster had fitted normally; the premise of the frozen
rule - that a quantile of yesterday's residual is today's noise floor - had not held across a
regime change (`docs/RESULTS.md` section 6k). It was not re-tuned. It went into D29 as the finding
Phase 2 inherits first.

## 4. What Phase 2 inherits

`docs/PHASE2.md` carries it in full. In brief: the GRU is the architecture and
`src/sentinel_models/reference.py` is its blueprint, held to torch at 1e-5; the model file stores
plain float32 arrays with the gate order named in the header and **both bias vectors unsummed**
(a GRU's third recurrent bias sits inside the reset product and cannot be folded - the one place a
loader written from the LSTM is wrong); normalisation constants and thresholds live outside the
weights as parameters with a provenance, replaceable without retraining; the decision layer is
D25's - per-channel EWMA, max across channels, one calibrated quantile - and its calibration must
be repeatable in orbit without teaching the detector that a degradation is normal; the union is
not adopted. Open decisions carried: D14 (the weight-cache key is still positional), D21's first
reach (`error_buffer`'s effect on alarm width), D23 (the decision layer is channel-blind), and D6
(the command-conditioning ablation has never run on trained weights - every figure here is
telemanom-minus-commands).

Fourteen measurements changed a conclusion in this phase. The last one changed the question.
