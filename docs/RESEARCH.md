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

**(!) Read at first hand 2026-09-10, section 4.3, and it bears directly on `p`.** The
paper's own ablation: *"pruning only decreases overall recall by 4.8 percentage points
(84.8% to 80.0%) while increasing overall precision by 38.6 percentage points (48.9% to
87.5%)"* -- the arithmetic of the two Table 2 rows already recorded at `docs/MODELS.md`
26.19. The sentence that follows is the load-bearing one and was not on record here:

> *"The `p` parameter is an important lever for controlling precision and recall, and an
> appropriate value can be inferred **when labels are available**. In our setting,
> reasonable results were achieved with `0.05 < p < 0.20`."*

**The method's own author states that `p` is tuned against labels**, which a deploying
mission does not have (`docs/HARNESS.md` 6b) -- the shippability objection this project
had reached by inference. And `docs/MODELS.md` 37.7a measured that **every one of the 13
pruned-and-recoverable events would be retained at a `p` inside that published band**
(max 0.1137, median 0.0834). They are not lost to an exotic setting; they are lost to one
value the author calls reasonable while another equally reasonable value keeps them.

**(!) Vendored and pinned, 2026-09-08 (D53).** The source now lives at
`third_party/telemanom/`, commit `2e6c5b6c3558e7835601519b7bdef37c649bdbdc`,
because five readings of it had been published from line numbers with no copy
kept, and all five were wrong. It is evidence and never a dependency.
**Its licence is BSD 3-Clause** (Caltech/JPL 2018), not the Apache-2.0
`docs/DATA.md` recorded until today; clause 3 forbids using Caltech's or JPL's
name to endorse anything derived from it, which binds on publication.
*Primary* from here on, in the strong sense: `docs/MODELS.md` 26.29 cites files
in this repository.

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

*Taken:* the **static** variant only. We fit the tail offline and freeze it, losing
drift adaptation and gaining determinism; drift is handled by ground
recalibration and model v2 (Objective.md 10.2 fix 4).

**(!) The REASON given here was wrong, and it is corrected rather than deleted
(D63, 2026-09-10).** This paragraph used to begin: *"the **static** variant only.
SPOT refits online and Objective.md 11 rule 1 forbids that outright -- a slowly
degrading spacecraft must never teach the detector that degradation is normal."*
That reading extends rule 1 from the model to the threshold, and **the frozen
pipeline does not survive it**: D62 froze stage 4 on telemanom's published
*dynamic* threshold, which recomputes its cut every `stride` steps from a trailing
window of the stream being scored (`src/sentinel_models/telemanom.py:397-411`;
`docs/MODELS.md` 26.17). Rule 1 governs the model and its weights; a noise floor
measured from nominal data is not online learning within its meaning. **The
preference for the static variant may still be right on other grounds -- it is
deterministic, and it is what `flight/` can carry today -- but it is no longer
supported by this reason**, and an EVT arm that adapts from nominal data is
registrable rather than refused at the door.

**(!) NOW PRIMARY, read at first hand 2026-09-10.** The entry above was written
without reading the paper, and it was the basis on which a whole method was set
aside. The paper has since been read from the open mirror
`www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p1067.pdf`
(cite as DOI `10.1145/3097983.3098144`; the mirror is recorded because that is
what was read). What follows is transcribed from it, not recalled.

**How it was read, and why that is stated.** `WebFetch` returns undecoded streams
for this PDF, and the machine has no PDF text extractor. The document was decoded
with a stdlib-only reader that resolves **every glyph through the `/ToUnicode` CMap
of the font that was active when it was drawn**, per page and per Form XObject.
That matters here specifically: a first attempt used one global ligature map and
**silently rendered `sigma` as the "fi" ligature**, because code `0x1b` is `sigma`
in the maths fonts (`rtxmi`, `rtxmi7`, `LinLibertineI7`) and the `fi` ligature in
the text fonts (`LinLibertineT`, `TB`, `TI`). The font-aware decode was verified on
exactly that point before anything was read from it.

**Residual ambiguity, named rather than inferred.** 81 glyph instances of 46,628
(0.17%) have no CMap entry. In the two formulas transcribed below **every one of
them is from `txexs`**, the TX extension font, which carries only large delimiters
and big operators: `0x20`/`0x21` big brackets, `0x12`/`0x13` big parentheses,
`0xd5` the summation sign, `0x10`/`0x11` interval brackets. **No variable is
missing from either formula.**

*Taken, and this is the mechanism `docs/MODELS.md` 38's Arm 5 is built on:*

```
  Theorem 3.1 (Pickands-Balkema-de Haan). F in D_gamma iff a function sigma
  exists, for all x with 1 + gamma*x > 0, such that
      Fbar(t + sigma(t)x) / Fbar(t)  ->  (1 + gamma*x)^(-1/gamma)   as t -> tau
  so the excesses X - t over a high threshold t follow a Generalized Pareto
  Distribution with parameters (gamma, sigma); the location mu is null here.

  Equation 1, the quantile:
      z_q  ~=  t + (sigma_hat / gamma_hat) * [ (q*n / N_t)^(-gamma_hat) - 1 ]
  where t is a "high" threshold, q the desired probability, n the total number of
  observations, and N_t the number of peaks, i.e. of X_i with X_i > t.

  Algorithm 1 (POT), verbatim:
      1: procedure POT(X_1 ... X_n, q)
      2:   t     <- SetInitialThreshold(X_1 ... X_n)
      3:   Y_t   <- { X_i - t | X_i > t }
      4:   gamma_hat, sigma_hat <- Grimshaw(Y_t)
      5:   z_q   <- CalcThreshold(q, gamma_hat, sigma_hat, n, N_t, t)
      6:   return z_q, t

  The Grimshaw reduction (3.4.2): with l(gamma, sigma) = log L(gamma, sigma), any
  solution of grad l = 0 has x* = gamma*/sigma* solving the SCALAR equation
      u(x) v(x) = 1,   u(x) = (1/N_t) sum_i 1 / (1 + x Y_i)
                       v(x) = 1 + (1/N_t) sum_i log(1 + x Y_i)
  and then gamma* = v(x*) - 1, sigma* = gamma* / x*. Roots are only candidates:
  all of them are found, their likelihoods computed, and the best tuple kept.
  1 + x*Y_i must be strictly positive, so the search runs on (-1/Y_M, +inf) with
  Y_M = max Y_i.

  The initial threshold t (4.3.3): "in practice its value is not paramount except
  that it must be 'high' enough" -- higher t means a better GPD fit (low bias) but
  a sparser peaks set (high variance); the one hard condition is t < z_q. "In
  practice we set t to a high empirical quantile (98%)."

  DSPOT (4.2.2): SPOT is run not on X_i but on X'_i = X_i - M_i, where
  M_i = (1/d) * sum_{k=1..d} X*_{i-k} over "the last d NORMAL observations".
  Strictly trailing, and anomalies are excluded from the local model. It adds one
  parameter, the window d.
```

**(!) Verified against a second source, as the transcription rule requires.**
Equation 1 was cross-checked against an independent statement of the POT return
level, `z_m = u + (sigma/xi) * [ (m * zeta_u)^xi - 1 ]` with `zeta_u` the
exceedance fraction and `m` the return period. Substituting `u = t`,
`zeta_u = N_t/n` and `m = 1/q` gives `t + (sigma/gamma) * [ ((1/q)(N_t/n))^gamma - 1 ]`,
and `(q*n/N_t)^(-gamma) = ((1/q)(N_t/n))^gamma`, so the two are algebraically
identical. The decode and the independent statement agree.

**(!) Two sentences that bear directly on D63, and they are the arm's flight case
rather than a footnote.** First, on the initialisation (4.2): *"The POT primitive
may be seen as a training step but this is partly wrong because the initial batch
X_1 ... X_n is not labeled and is not considered as a ground truth in our
algorithm. The initialization is more a calibration step."* Second, on the update
(4.2.1): *"The anomalies are not taken into account for the model update."*
**SPOT withholds what it has flagged from its own update**, which is exactly the
boundary D63 draws -- the detector's notion of normal is never updated from data
it has not treated as normal. And 4.2.1 records that *"it is possible to do it
off-line at fixed time interval"* rather than per sample, which maps onto this
project's `stride` structure rather than requiring a per-tick refit.

**What is NOT in the paper, and must not be attributed to it.** There is **no
statement that the peaks set may be bounded to a fixed size**. The paper says only
that it stores "only the peaks", so "it requires low memory". Any bound is this
project's own engineering decision under F' CPP-1, and `docs/MODELS.md` 38 registers
it as such and measures what it costs.

*Where it is thin:* the experiments (section 5) and the complexity analysis have not
been read; nothing is claimed here about SPOT's measured performance on any dataset.

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

---

## Part IV -- the public-benchmark survey

**Added 2026-09-08.** The question this project has been unable to answer from
its own data: does a public dataset exist that can test the **cross-channel**
claim -- every channel inside its historical envelope while the relationship
between them is broken? Objective.md 9.2 demoted SMAP/MSL for the second half of
that question and D46 measured the first half on it. This part records what the
wider corpus offers.

**(!) Provenance is marked harder here than anywhere else in this document**,
because the survey is a secondary compilation and one of its sources is
load-bearing enough to be checked line by line.

### Pinet et al. 2026 -- the load-bearing source, verified

**Primary for the abstract; secondary for everything else.** Fetched from
`arxiv.org/abs/2606.02670` on 2026-09-08 and checked field by field:

```
  title      Anomalies in Multivariate Time Series Benchmarks Are Mostly Univariate
  authors    Marc Pinet, Julien Cumin, Samuel Berlemont, Dominique Vaufreydaz
  dates      submitted 1 Jun 2026 (v1); last revised 15 Jun 2026 (v4)
  comments   Accepted at the 12th International Workshop on Mining and Learning
             from Time Series (MiLeTS), co-located with KDD 2026
```

The abstract states, of **eight** widely used public benchmarks, that the
diagnostic "shows that **no cross-channel rupture occurs without an accompanying
univariate deviation across a range of reasonable thresholds**"; that "on six of
the eight benchmarks, at least half of the labeled anomaly segments deviate
univariately on 89% to 100% of their timesteps, reaching 100% on three of these
datasets"; that a channel-independent against channel-dependent comparison of a
recent state-of-the-art detector "further confirms that CD modeling brings no
measurable gain"; and it concludes that "current MTSAD benchmarks are unsuitable
for validating cross-channel modeling capabilities".

**(!) What is NOT verified, and is therefore attributed to the survey rather than
to the paper.** The abstract page does not name the eight benchmarks, does not
give a segment count, and does not carry the per-threshold figures. The survey's
statements -- that the eight are GECCO, MSL, SMAP, PSM, SMD, SWAN-SF, SWaT and
WADI; that there are 373 long segments; that the strictly-cross-channel count is
exactly zero at default thresholds and at most 34 of 373 at extreme ones -- come
from the body, **which has not been read here**. They are recorded as the
survey's and are not quoted as the paper's.

### (!) It does not say what our headline says, and the difference is the diagnostic

**Pinet's test and D46's test are different tests, and reading one for the other
would make this section look like a refutation of our own claim.**

```
  Pinet et al.   does the channel deviate from its NORMAL HISTORY -- a
                 distributional test, z-score and correlation based
  D46 (ours)     does the channel leave its TRAINING MIN/MAX -- the envelope a
                 limit check actually holds
```

A sample can sit inside the min/max envelope a limit check enforces and still be
a large statistical deviation from history. **The two findings are compatible**:
39 of 43 SMAP/MSL contextual sequences never leave their channel's range (D46)
*and* the labelled anomalies of these benchmarks are accompanied by univariate
deviation (Pinet). The first is about what a **limit check** can see; the second
is about what a **univariate statistical detector** can see. Our own D48 measured
the second directly and agrees with it: no per-channel statistic reaches a
flyable alarm rate on SMAP/MSL, at any multiplier.

### What it corroborates, and it is the first outside confirmation of either

**"CD modeling brings no measurable gain" on real benchmarks is an independent
replication of D42 and D23.** D42 measured that no cross-channel reduction
recovers anything `max` misses on ESA-ADB, and D23 closed as answered no. Those
were single-project findings on one benchmark; an outside group reaches the same
place on eight. That is worth more than any figure in this section.

### What it bounds

1. **Objective.md 9.2 is strengthened rather than merely restated.** SMAP/MSL was
   demoted on the PATH paper's claim that its channels are unsynchronised -- which
   `docs/RESEARCH.md` Part I still flags as secondary and unverified. The
   cross-channel case against the public corpus no longer rests on that one claim.
2. **Phase 3's physics testbed is not a convenience; it is the only venue.** If no
   public benchmark carries strictly cross-channel segments, the cross-channel and
   early-warning claims cannot be earned on one, and the coupled
   current-heat-temperature subsystem in `SentinelRef` is where they have to be
   measured or abandoned.
3. **It does not touch the headline's first two clauses**, which rest on our own
   artifacts: 39 of 43 in range (D46), and `rstd` at 5,000x still alarming on
   15.17% of nominal steps (D48).

### The rest of the survey -- secondary, compiled, not independently checked

**Recorded as a research direction, not as evidence.** Only the three sources
above were fetched and checked; everything in this subsection is the survey's own
compilation and **no figure from it may be quoted in another document until it is
verified at its primary source.**

- **ESA-ADB** -- Kotowski et al. **Verified**: `arxiv.org/abs/2406.17826`,
  "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry",
  **eleven** authors (was "twelve"; re-verified at `arxiv.org` 2026-09-14, and
  `docs/datasets/ESA_ADB.md` names all eleven in order), submitted 25 Jun 2024,
  revised 17 Aug 2025, comments "87 pages, 24 figures, 19 tables". **The survey's
  "DMLR 2026" is not on that page and is not verified.** The survey's structural
  point is the one that matters here and it matches our own reading: the
  Univariate/Multivariate, Local/Global and Point/Subsequence attributes are
  inferred from **how many channels are annotated and over what span**, not from
  in-range status -- which is exactly why
  `docs/HARNESS.md` section 1 records that operationalising "contextual" as
  `Dimensionality == Multivariate` is an interpretation, and why D43 had to define
  the class by training min/max instead.
- **Wu and Keogh** -- **Verified**: `arxiv.org/abs/2009.13807`, "Current Time
  Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of
  Progress", Renjie Wu and Eamonn J. Keogh, 29 Sep 2020, comments "Full paper
  accepted by IEEE TKDE, extended abstract accepted by IEEE ICDE 2022". Part I's
  entry gave TKDE 2023 without an identifier; the identifier is now on record.
- **SWaT and WADI** (iTrust, SUTD) -- real water-treatment and distribution
  plants, expert-designed physical attacks, a common 1 Hz clock, genuine physical
  coupling, access by data agreement. The survey ranks these first for this
  project's purpose and notes that Pinet classify every long segment in them as
  **BOTH** rather than cross-channel-only, and that one SWaT segment is roughly
  65% of anomalous time and single-feature-detectable. Counts of channels and
  attacks disagree between secondary sources; the survey says to use iTrust's
  primary documentation, and this document has not.
- **SKAB** -- an open pump rig, correlated sensors on a common clock, physically
  induced faults. Cheap corroboration, real coupling, small.
- **Tennessee Eastman**, **CATS**, **C-MAPSS** -- simulated. CATS is already in
  Objective.md 9.3 flagged as synthetic.
- **OPS-SAT-AD** -- univariate per fragment by construction, so it cannot test
  this class. Consistent with Objective.md 9.3's note.
- **Exathlon**, **SMD**, **PSM** -- IT and server telemetry, not physical systems.
- **Non-public routes** the survey names for real spacecraft telemetry with
  expert cross-channel labels: ESA/ESOC ARTS through the ESA-ADB authors, JPL,
  JAXA, CNES, DLR, and CubeSat operators including SatNOGS -- which Objective.md
  decision 9 already carries as an open item.

**The reporting standard worth adopting**, and the one concrete thing this
section changes: Pinet's per-segment diagnostic --
`UNIVARIATE / CROSS-CHANNEL / BOTH / UNDETECTED` -- as an **acceptance test for
any dataset this project claims tests cross-channel modelling**. Our stage-1
visibility diagnostic already does the min/max half per segment
(`scripts/smap_visibility.py`); the correlation half is not built, and no dataset
should be adopted for the cross-channel claim until it is.

**Caveat carried from the survey and worth repeating**: near-binary channels
defeat a z-score diagnostic, which is a live concern for SMAP/MSL specifically,
and absence across eight benchmarks is strong evidence rather than proof.

---

## Part V -- evaluation metrics, and the decision layer's prior art

**Added 2026-09-10.** Sources read for the decision-layer arms registered in
`docs/MODELS.md` 38 and for the metric authorised in `docs/HARNESS.md` 5a.

### Kim et al., AAAI 2022 -- "Towards a Rigorous Evaluation of Time-series Anomaly Detection"

**Primary for the abstract**, fetched `arxiv.org/abs/2109.05257` on 2026-09-10; body not
read. Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, Sungroh Yoon.

The paper *"theoretically and experimentally reveal[s] that the PA protocol has a great
possibility of overestimating the detection performance; that is, **even a random anomaly
score can easily turn into a state-of-the-art TAD method**"*, and that *"the comparison of
TAD methods after applying the PA protocol can lead to misguided rankings."*

*Taken:* nothing new -- point-adjusted F1 was already quarantined on Wu & Keogh's
authority and Objective.md 9.5. **What this supplies is the primary citation that
quarantine did not have**, and a sharper statement of the failure: not merely inflation but
inflation that survives a random input. `docs/HARNESS.md` 1's table now carries it.

### Huet et al., KDD 2022 -- "Local Evaluation of Time Series Anomaly Detection Algorithms"

**Primary for the abstract**, fetched `arxiv.org/abs/2206.13167` on 2026-09-10; body not
read. Alexis Huet, Jose Manuel Navarro, Dario Rossi.

The paper first *"highlight[s] the limitations of the classical precision/recall, as well
as the main issues of the recent event-based metrics -- for instance, we show that **an
adversary algorithm can reach high precision and recall on almost any dataset under weak
assumption**"*, then proposes *"a theoretically grounded, robust, **parameter-free** and
interpretable extension to precision/recall metrics, based on the concept of 'affiliation'
between the ground truth and the prediction sets"*, which *"leverage[s] measures of
duration between ground truth and predictions"* and yields a *"normalized
precision/recall, quantifying how much a given set of results is better than a random
baseline prediction."*

*Taken:* the authorisation of affiliation precision/recall as a reported metric
(`docs/HARNESS.md` 5a, 2026-09-10), and the disclosure in `docs/HARNESS.md` 1 that **the
range-based pair this project already reports belongs to the family the adversary result
covers**. Nothing is withdrawn; the existing metrics are no longer reported alone.

*Where it is thin:* the adversary construction is in the body and has not been read here,
so the *strength* of the attack against this project's specific configuration -- fixed
`alpha = 0.0`, flat bias -- is not established. The disclosure states the finding and
attributes it; it does not claim to have reproduced it.

### Gibbs & Candes, NeurIPS 34 (2021) -- "Adaptive Conformal Inference Under Distribution Shift"

**Primary**, read at `ar5iv` on 2026-09-10 including the update rule and Proposition 4.1.

The method updates a miscoverage level online, `alpha_{t+1} := alpha_t + gamma * (alpha -
err_t)`, where `err_t` is the indicator that the observation fell outside the prediction
set at `alpha_t`. Proposition 4.1 gives, **with probability one and for all `T`**,
`|(1/T) sum_t err_t - alpha| <= (max{alpha_1, 1 - alpha_1} + gamma) / (T * gamma)`, so
long-run coverage holds **with no distributional assumption on the data-generating
process**. The authors use `gamma = 0.005` and caution that *"large fluctuations in
alpha_t may be undesirable as it allows the method to oscillate between outputting small
conservative and large anti-conservative prediction sets."*

*Taken:* the mechanism for `docs/MODELS.md` 38's Arm 6. It is registered against D29 --
a threshold calibrated on Mission 1's first 7.36M steps sat under **86.7%** of *fold 1's*
nominal residual -- and against D48, no train-calibrated threshold transferring on
SMAP/MSL at all.

*Where it is thin, and it is stated because the arm must not overstate it:* D29's finding
is that **the frozen rule's premise failed, not the forecaster** -- *"the forecaster
fitted; the frozen rule's premise did not hold"* -- and fold 2's collapse is 4.4% and 0.7%,
not 86.7%. ACI addresses the class of failure D29 exposed; **it has not been shown to
address D29's instance**, and Arm 6 is written to test that rather than to assume it. The
guarantee is also a *long-run average* over `T`, which is a different object from the
per-window behaviour a mission cares about.

### Basseville and Nikiforov, "Detection of Abrupt Changes: Theory and Application", Prentice-Hall 1993 -- the read source for CUSUM

**Primary**, read at first hand 2026-09-10 from the authors' own freely-available copy at
`people.irisa.fr/Michele.Basseville/kniga/kniga.pdf`. Chapter 2, section 2.2, pages 35-41.
Decoded with the same stdlib font-aware reader used for Siffer; this document is a
1992-era PDF with standard encodings and its mathematics decoded cleanly.

*Taken, and it is the mechanism `docs/MODELS.md` 38's Arm 4b is built on:*

```
  The intuition (2.2.1, p.36-37). The log-likelihood ratio S_k drifts negative before
  a change and positive after it, so the informative quantity is the difference
  between S_k and its running minimum:
      g_k = S_k - m_k >= h                                          (2.2.1)
      S_k = sum_{i=1..k} s_i,   s_i = log[ p_1(y_i) / p_0(y_i) ]    (2.2.2)
      m_k = min_{1<=j<=k} S_j
      t_a = min{ k : g_k >= h } = min{ k : S_k >= m_k + h }         (2.2.3, 2.2.4)
  B&N note this "is nothing but a comparison between the cumulative sum S_k and an
  ADAPTIVE threshold m_k + h", modified on-line.

  The recursive form (2.2.2, p.38-39), which is what makes it flight-shaped:
      g_k = g_{k-1} + log[p_1(y_k)/p_0(y_k)]  if that is > 0, else 0    (2.2.8)
      g_0 = 0
  compacted as
      g_k = (g_{k-1} + s_k)^+,  where (x)^+ = sup(0, x)                (2.2.9)
      t_a = min{ k : g_k >= h }                                        (2.2.10)
  and B&N state this form "is equivalent to the other form that we presented in
  (2.2.4)". One scalar of state, one add, one max, one compare per sample.
```

**Why this source rather than Page 1954.** Page's original has no open full text. B&N
carry the algorithm, its derivation as a **repeated sequential probability ratio test**
(2.2.2), the off-line statistical derivation (2.2.3), the two-sided form (2.2.5), and the
average-run-length theory -- so one read source discharges what four paywalled ones were
wanted for.

**(!) One caution taken from it, and it changes Arm 4b's design.** B&N introduce the
**average run length function** as the tool that "concentrates the information about
both these performance indexes" -- mean delay and mean time between false alarms -- and
state plainly that **"the computation of this function is difficult for most of the
practically relevant change detection problems"**, which is why they give numerical
algorithms for it. So the map from the decision interval `h` to an alarm-rate budget is
**not closed-form in general**. Arm 4b therefore calibrates `h` by bisection on nominal
data to hit the budget, the way `src/sentinel_models/oscfar.py`'s `_fit_jointly` already
calibrates its multipliers, rather than inverting an ARL formula.

*Where it is thin:* only chapter 2 section 2.2 has been read. The ARL numerical algorithms,
the two-sided form's details and the optimality results are cited as framing, not
transcribed.

### Page 1954; Lorden 1971; Moustakides 1986; Pollak -- quickest change detection

**CITED BUT NOT READ, and the distinction is deliberate.** Page, *"Continuous inspection
schemes"*, Biometrika 41(1-2):100-115, DOI `10.1093/biomet/41.1-2.100`, is the origin of
CUSUM and has no open full text. **Nothing in this repository is implemented from it.**
The mechanism is taken from Basseville and Nikiforov above, which is read; Page is cited
as the origin, as B&N themselves cite it. Lorden, Moustakides and Pollak are cited as
*framing* only -- the formal statement that the decision layer is minimising detection
delay subject to a false-alarm constraint -- and no mechanism is taken from any of them.
This is the same treatment `docs/DECISIONS.md` D53 established for a published method:
the claim is only as strong as the source actually opened.

### Nelson 1984; Western Electric 1956; Hawkins 1987; Quesenberry 1991 -- SPC runs rules and self-starting charts

**Not read, and deliberately not chased.** They would supply runs rules and a
self-starting parameterisation for Arm 4. That arm is **predicted inert** on this stack
(`Objective.md` 7.1 measured persistence as subsumed: N=5 removed one alarm range of 182,
because `error_buffer` already dilates every exceedance by +/-99), so the sources are not
worth obtaining ahead of the measurement. **The slot is registered in `docs/MODELS.md` 38
with the sources it would need**, so that a later reader knows the gap is deliberate
rather than overlooked.
