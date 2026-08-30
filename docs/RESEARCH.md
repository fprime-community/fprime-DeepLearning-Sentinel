# The evidence base

External sources that changed a design choice here, with what each says, whether
we read it directly or at second hand, and what we took from it.

**Provenance is marked per source.** *Primary* means the paper or standard was
read. *Secondary* means the finding reached us through a summary and the figure
has not been checked against the original. That distinction matters more than it
usually does, because several numbers below are load-bearing.

**Where the evidence is thin, it says so.** A source that supports a decision
weakly is worth more when its weakness is recorded than when it is cited as
though it settled the matter.

Updated whenever external evidence changes a design choice (`docs/HARNESS.md`).

---

## Part I -- the method, the benchmark and the data

### telemanom -- Hundman et al., KDD 2018

**Primary.** The reference method: an LSTM forecasts telemetry, prediction error
is smoothed, and a nonparametric dynamic threshold plus a pruning step turn the
error into alarms. Evaluated on real expert-confirmed SMAP and Curiosity
anomalies.

*Taken:* the whole detection architecture, and the configuration transcribed in
`docs/MODELS.md` -- `l_s=250`, layers `[80, 80]`, dropout 0.3, `l_p=10`, `p=0.13`,
the z sweep, the error buffer.

*Also taken, late:* the paper states the model's inputs are "prior telemetry
values for a given channel **and encoded command information sent to the
spacecraft**", one-hot encoded by module and by sent-or-received, and its Figure 3
shows that encoding letting the model predict a commanded event so it is not
flagged. **We missed this on the first reading and reproduced the method without
it** -- see `docs/DECISIONS.md` D6. The paper also notes that "no information
about the nature of the command itself is passed to the models", flagging richer
command features as the obvious next step.

*Source of the 41% figure* in Objective.md 2.3: contextual anomalies as a share
of real expert-confirmed anomalies.

### Wu & Keogh, IEEE TKDE 2023 -- "Current time series anomaly detection benchmarks are flawed"

**Secondary.** Identifies triviality, unrealistic anomaly density, mislabelled
ground truth and run-to-failure bias in the standard benchmarks, and shows that
one-line methods reach state of the art on them.

*Taken:* the demotion of SMAP/MSL (D1), the decision to score trivial baselines
*first* so the harness is used in anger before any model depends on it, and the
choice of `mavg` and `rstd` as those baselines.

### PATH dataset paper -- Kaufmann et al.

**Secondary.** States that SMAP/MSL's channels, while technically multivariate,
**are not synchronised with each other**.

*Taken:* the decisive half of D1. This is the load-bearing claim behind demoting
SMAP/MSL and it reached us second-hand; **it has not been verified against the
raw data.** If it is wrong, the demotion's second argument falls and only Wu &
Keogh's remains. Worth checking before anything is published on it.

### ESA-ADB -- the benchmark and its requirements

**Primary** for the dataset and its documentation; **secondary** for the baseline
result tables.

*Requirement R7* asks algorithms to "learn to distinguish rare nominal events, so
they are not alarmed after the first occurrence". **Every published baseline
scores zero on it.** The authors state commanded manoeuvres "should not be
alarmed to SOEs" but "cannot be distinguished from actual anomalies without
expert knowledge".

*Taken:* the framing of the false-alarm programme as work at the benchmark's
frontier rather than as tuning; the choice of corrected event-wise F0.5 as the
headline, on the same reasoning Objective.md 11 rule 2 gives -- a high false
positive rate disqualifies an algorithm even with perfect scores elsewhere; and
the priority grading of telecommands, of which ESA feeds only priority 3 to its
own baselines.

**(!) The telecommand result that qualifies our whole Layer 1.** In ESA's own
baselines, adding telecommands coincided with **worse** precision: Mission1
Telemanom-ESA-Pruned fell from corrected event-wise F0.5 **0.786** (lightweight,
no telecommands) to **0.061** (full set), which the paper attributes to more
target channels meaning more chances of false detection.

> **This is the thinnest and most consequential evidence in this document.** It is
> secondary, the two configurations differ in more than telecommands, and the
> attribution is the authors' rather than a controlled result. It is recorded
> because it is the only published evidence bearing on whether command
> conditioning helps, and it points the wrong way. **No ESA-ADB paper has
> published a commands-on / commands-off ablation.** Ours will be the first, and
> until it exists the magnitude of the effect is unproven for spacecraft
> telemetry -- in either direction.

---

## Part II -- F' portability and the tiered architecture

### NASA cFS table services

**Secondary.** cFS applications ship default configuration tables so a mission has
working behaviour on day one and overrides them later.

*Taken:* the precedent for D5's tiering. We are following an established framework
pattern rather than inventing one, which matters for review-board acceptance more
than it matters technically.

### F' PrmDb, and the `FPRIME_ENABLE_TEXT_LOGGERS` pattern

**Primary** (framework documentation). PrmDb is a generic component loading a
mission-specific parameter file; the text-logger flag is a capability that can be
compiled out.

*Taken:* the architecture of D5 and the `model.bin` split in Objective.md 4.1 --
generic code, mission data. Also the decision that normalisation constants and
detection thresholds live outside the weights as PrmDb-style parameters, so a
mission can recalibrate in orbit without retraining.

### Baireddy et al., "Spacecraft Time-Series Anomaly Detection Using Transfer Learning" -- CVPR Workshops 2021 (AI4Space)

**Secondary.** Purdue and Lockheed Martin Space. Pretrained on unlabelled Mars
Reconnaissance Orbiter telemetry, fine-tuned on SMAP/MSL at roughly half the
training time. Reports **recall 0.774 / precision 0.765** transferred against
**recall 0.835 / precision 0.794** from scratch.

*Taken:* the evidence for Level 2 in D5.

**The commonly quoted "0.769 vs 0.814 F1" pair is derived, not published** -- the
paper reports precision and recall, and `F1 = 2PR/(P+R)` gives those values, so
transfer retained about 94% of from-scratch F1. Recorded with the arithmetic
visible so nobody quotes it back as a figure the paper printed.

*Where it is thin:* transfer learning is well established for **forecasting** and
much less so for **detection** specifically. And ESA-ADB is anonymised, so we can
build a generic model of telemetry *dynamics* but not a generic *battery* model.
Whether generic dynamics transfer sufficiently is unproven -- Objective.md 14.10
carries the open risk.

### Tiny Time Mixers, and small-model CubeSat deployment

**Secondary.** Cited in Objective.md 14.10's line of argument that a
flight-sized model is feasible. **We have not verified the 59 KB deployment
figure against a primary source.** It supports a claim we are not currently
relying on -- our own measured model is 358 KiB of float32 (`docs/MODELS.md`
section 3) -- so nothing turns on it today.

---

## Part III -- false-alarm reduction, borrowed from six fields

Every source here addresses the same problem in a different domain: a detector
sensitive enough to be useful produces more alarms than a human can act on.

### ISA-18.2 and EEMUA 191 -- process-control alarm management

**Secondary.** State-based alarming -- suppressing alarms that are expected given
the plant's current mode -- delivers **66-92% nuisance reductions** industrially.
Both converge on roughly **one alarm per operator per ten minutes** in normal
operation, with about 12/hour the maximum manageable and above 30/hour "seriously
deficient"; an alarm flood is more than 10 alarms in any 10-minute period.

ISA-18.2 also **reclassifies suppressed alarms as recorded events** rather than
discarding them.

*Taken:* Layer 2's commanded-event-window masking, and its hard rule that every
suppressed detection is logged and counted on the scorecard.

**(!) NOT taken, and the correction is 2026-08-28: the alarm budget.** These
standards set rates a human operator can absorb, and an earlier reading of them
here justified calibrating our threshold to a target admission rate. **That was
the wrong borrowing.** A process plant's alarm rate is a design variable an
engineer sets; a spacecraft's relationship breaks are not. Sentinel fires when a
learned relationship breaks and is silent otherwise, and the count is the
spacecraft's (`docs/HARNESS.md`; Objective.md 10.2 fix 3a). What survives is the
weaker and still useful part: **a rate computed on healthy data is a sanity check
on the calibration**, and ISA-18.2's rule that a suppressed detection is recorded
rather than discarded. The rates are quoted in wall-clock and **our data cannot be**: ESA-ADB
timestamps are anonymised and scaled, so we state budgets in timesteps and note
that a mission converts using its own cycle rate.

### CFAR detection -- radar, ~50 years of practice

**Secondary.** Constant False Alarm Rate detection solves a problem structurally
identical to ours. **Guard cells** exclude samples adjacent to the point under
test from the noise estimate, so target energy spilling into neighbours cannot
inflate the threshold and mask the target. **OS-CFAR** estimates scale from a high
quantile rather than a mean, for robustness to interfering targets. **GO-CFAR**
handles clutter edges, where a reference window straddles two noise regimes and
produces either a false-alarm burst or masking.

*Taken:* all of Layer 4. The mapping is direct: our smeared event edges are target
energy in the reference cells, our rare nominal spikes are interfering targets,
and commanded transitions are clutter edges. The multiplier chosen analytically
from a target false-alarm probability is a strict improvement over an arbitrary
quantile.

### Siffer et al., KDD 2017 -- "Anomaly Detection in Streams with Extreme Value Theory"

**Secondary.** SPOT sets thresholds by Peaks-Over-Threshold with a Generalized
Pareto tail; the main parameter is a risk `q`, an explicit false-positive
regulator, typically 1e-3 to 1e-5.

*Taken:* the **static** variant only. SPOT refits online and Objective.md 11 rule
1 forbids that outright -- a slowly degrading spacecraft must never teach the
detector that degradation is normal. We fit the tail offline and freeze it, losing
drift adaptation and gaining determinism; drift is handled by ground
recalibration and model v2 (Objective.md 10.2 fix 4).

### EGPWS and TCAS -- certified avionics

**Secondary.** EGPWS Mode 4 alerts are conditioned on landing configuration (gear,
flaps), and Mode 2B is "desensitized to permit normal landing approach manoeuvres
close to terrain". The visual display stays live when audio is inhibited.

*Taken:* the argument that mode-based suppression is **accepted practice in
certified life-critical systems**, not a shortcut -- which is the form the
argument has to take for a flight review board. And the never-silently-drop rule:
inhibiting a channel is not the same as discarding the detection.

### Fraud detection -- cascades and alert budgets

**Secondary.** Handles base rates of 0.1-1.8%, comparable to ESA-ADB's 1.19%
anomaly density, with a cheap high-recall triage stage feeding a costlier
high-precision adjudication stage. Operating points are chosen from human review
capacity, and results reported as precision@k.

*Taken:* Layer 3's two-stage cascade. The cascade is what makes stage 2
flight-realisable: pure fixed logic, no learning, no allocation, bounded compute.

**Not taken: precision@k and budget-normalised reporting** (2026-08-28). Choosing
an operating point from human review capacity is exactly the framing struck in
`docs/HARNESS.md` -- a fraud team can decide how many cases it will look at, and a
spacecraft cannot decide how many of its relationships will break.

### Gorges et al. 2009 -- ICU alarm delays

**Secondary.** In intensive-care monitoring, a **14-second** alarm delay removed
**50%** of alarms and a **19-second** delay removed **67%**.

*Taken:* the shape of the persistence-filter trade, and a caution. Delay is the
cheapest false-alarm reduction available in any monitoring domain -- and this
project's headline claim is *early* warning, so delay is the one currency we
cannot spend freely. Our own measurement bears that out: lead time fell almost
one timestep per unit of persistence N, from +26 at N=1 to -37 at N=60, for
almost no gain in F0.5 (`docs/DECISIONS.md` D9). The ICU result is why the trade
was expected; our own numbers are why it was refused.
