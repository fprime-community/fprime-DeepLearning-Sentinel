# fprime-sentinel - Project Objective

**Status:** Phase 1 (Python, pre-flight-code)
**Document type:** Living. Fix the objective here; other docs defer to this one.
**Audience:** Anyone joining the project. No prior context assumed.

---

## 0. The project in one paragraph

fprime-sentinel is a reusable F' flight component that lets a spacecraft analyse its own
telemetry onboard. A small neural forecaster is trained on the ground from the mission's own
pre-launch test data, exported as a plain file of numbers, and uplinked. In flight, deterministic
C++ compares reality against the forecast every cycle. Sustained divergence - the early signature
of a developing fault, visible long before any limit trips - is downlinked as a standard F' event
with a named cause and a time-to-limit estimate. The detection method is JPL's published
reference approach; onboard feasibility was demonstrated in orbit by ESA. **The reusable F'
packaging is this project's original contribution.**

---

## 1. The goal

> Build a plug-in module for NASA/JPL's F' flight software that lets a spacecraft spot its own
> problems hours before anything breaks - and make it a standard building block any mission can
> drop in.

Not a research paper. Not a mission-specific script. **A block for the box.**

**Definition of done:** an F' component that a JPL flagship, a commercial smallsat, and a
university CubeSat can each adopt without ML expertise, using telemetry they already generate.

---

## 2. The problem

### 2.1 What spacecraft do today

An F' deployment streams telemetry continuously - voltages, currents, temperatures, wheel
speeds, queue depths. Onboard protection is a **per-channel limit check**:

```
    if (BattVoltage < 3.2) { alarm(); }
```

That is a thermostat. It is the entire onboard intelligence in the fault-awareness space.

### 2.2 Three costs of that architecture

| Cost | Consequence |
|---|---|
| **Late detection** | Alarms fire only *after* a value has already crossed a danger threshold |
| **Wasted downlink** | Thousands of readings shipped in which nothing is happening |
| **Blind between passes** | A degradation starting after loss-of-signal is found hours later |

### 2.3 The evidence that limit checking is structurally insufficient

Hundman et al. (KDD 2018), studying real expert-confirmed anomalies from the **SMAP satellite**
and the **Curiosity rover**, found that **41% were contextual**: every individual channel stayed
inside its limits while the *combination* or the *trajectory* was wrong.

No amount of limit tuning catches that class. The wrongness is not in any single value.

### 2.4 What a contextual anomaly looks like

```
  NORMAL - the learned relationship holds
  ----------------------------------------------------------------

  ChargeCurrent  _.:=#=:.__.:=#=:._         current rises...
  BattTemp       __.:=#=:.__.:=#=:.         ...temp follows ~5 min later
                    +-- consistent lag, every orbit, for years --+


  ANOMALY - both values legal, relationship broken
  ----------------------------------------------------------------

  ChargeCurrent  _.:=#=:.__.:=#=:._     2.1 A   OK inside limits
  BattTemp       ...................     18 C    OK inside limits
                                  ^
                    temperature did not follow

  LIMIT CHECKER SAYS:  all green, nothing to report
  SENTINEL SAYS:       "BattTemp / ChargeCurrent decoupled at 14:32"
```

Sensor failure, disconnected cell, failed thermal path - something is genuinely wrong, and
nothing alarms.

---

## 3. Prior art - the method flies, but not in F'

Three independently verifiable findings anchor credibility.

**1. The method is published and validated by JPL.**
Hundman et al.'s *telemanom* trains an LSTM to forecast telemetry; anomalies are detected as
sustained forecast-vs-reality divergence, thresholded nonparametrically. Evaluated on real SMAP
and Curiosity anomalies. Same paper as the 41% figure. **We are implementing the field's
reference method, not inventing one.**

**2. Onboard ML anomaly detection has flown.**
ESA's **OPS-SAT** CubeSat ran ML anomaly detection on its own telemetry in orbit, including deep
models - proving this class of inference runs acceptably on CubeSat-class hardware. Feasibility
is demonstrated, not speculative.

**3. No reusable F' implementation exists.**
The F' framework ships per-channel limit checking and a component-liveness ping. Nothing
learned, nothing cross-channel, nothing predictive. No community library provides it. OPS-SAT's
work was mission-specific experimental code; telemanom is a ground-side research codebase.

```
             proven method   flown onboard   reusable F' block
  telemanom       OK              NO               NO
  OPS-SAT         OK              OK               NO
  fprime-sentinel OK              OK (target)      OK  < the gap
```

**The lane is empty. That gap is the project.**

---

## 4. System architecture

### 4.1 The fundamental split: code vs. data

Everything about this project follows from one decision.

```
  +--------------------------------------------------------------+
  |  GENERIC CODE - identical on every mission, written once     |
  +--------------------------------------------------------------+
  |   * Python training toolkit         (stays on Earth)         |
  |   * C++ Sentinel flight component   (flies)                  |
  +--------------------------------------------------------------+
                              |
                              |  reads
                              v
  +--------------------------------------------------------------+
  |  MISSION DATA - different every time, generated by the       |
  |                 mission running our toolkit                  |
  +--------------------------------------------------------------+
  |   model.bin  >  weights, normalisation constants,            |
  |                 thresholds, persistence settings,            |
  |                 channel map                                  |
  +--------------------------------------------------------------+
```

**We ship the code. The mission generates the data.** The component never changes - not one
line. What changes is a file of numbers it reads at startup.

### 4.2 End-to-end flow

```
   +------------------- ON THE GROUND (Python) --------------------+
   |                                                               |
   |   flatsat runs -+                                             |
   |   day-in-life  -+-->  nominal telemetry  -->  TRAIN           |
   |   sim campaigns-+     (normal only)           forecaster      |
   |                                                   |           |
   |                                                   v           |
   |                                              EXPORT           |
   |                                            model.bin          |
   +---------------------------------------------------+-----------+
                                                       |
                             F' file uplink  ----------+  human-approved
                             (or installed at launch)  |
                                                       v
   +------------------- ONBOARD (C++) -----------------------------+
   |                                                               |
   |   rate group tick                                             |
   |        |                                                      |
   |        v                                                      |
   |   read watched channels --> forward pass --> forecast         |
   |                                                  |            |
   |                              reality ------------+            |
   |                                                  v            |
   |                                          prediction error     |
   |                                                  |            |
   |                        +-------------------------+            |
   |                        v                         v            |
   |                persistence filter        error statistics     |
   |                  (N cycles)                      |            |
   |                        |                         v            |
   |                        +-------------->  trend projection     |
   |                        |                  (time-to-limit)     |
   |                        v                                      |
   |                explanation layer                              |
   |              (which pair decoupled)                           |
   |                        |                                      |
   |                        v                                      |
   |            F' EVENT + TELEMETRY CHANNELS                      |
   +---------------------------------------------------+-----------+
                                                       |
                                    standard F' downlink
                                                       v
                        Yamcs / F' GDS / anything - UNMODIFIED
```

### 4.3 The five steps in words

**1 - Learn on Earth.** Feed the toolkit pre-launch telemetry. It trains a small sequence model
to answer, at every timestep: *given the recent history of all watched channels, what should the
next values be?*

**2 - Export a file of numbers.** Weights as plain arrays, normalisation constants, thresholds,
channel map. **No Python, no ML framework, no interpreter goes to space.** Flight review boards
will not approve an interpreter on a spacecraft. They will approve a C++ program reading a data
file - because F's own parameter database already works exactly that way.

**3 - Uplink.** Standard F' file-uplink service, or baked in at launch. Updating later means:
uplink a new file, command a reload. **Human-approved every time.**

**4 - Predict vs. reality.** Each rate-group cycle, run the forward pass, compare, update error
statistics. Cost is a fixed sequence of matrix multiplies and elementwise nonlinearities over
~thousands of parameters - trivial on Linux-class flight processors (Ingenuity flew F' on a
Snapdragon-class part). Envelope measured explicitly in Phase 4.

**5 - Warn, never act.** Output is a standard F' event:

```
  EVT_SENTINEL_DECOUPLING   WARNING_HI
      "BattTemp / ChargeCurrent decoupled at 14:32"

  EVT_SENTINEL_TREND        WARNING_HI
      "BattVoltage predicted to cross RED_LO in ~4h 12m"
```

Nothing downstream needs to know Sentinel exists.

---

## 5. "But how is it LEGO if it's retrained per mission?"

The most common objection. The answer is that the brick and its contents are different objects.

| | Who makes it | Same for everyone? |
|---|---|---|
| C++ flight component | Us, once | [x] **Identical everywhere** |
| Python training toolkit | Us, once | [x] **Identical everywhere** |
| `model.bin` | Mission runs our toolkit | [-] Different every time |

**This is exactly how F' already works:**

| F' component | Generic code | Mission-specific data |
|---|---|---|
| Limit checker | [x] | your limits, from your dictionary |
| Parameter database | [x] | your parameter file |
| Command dispatcher | [x] | your command list |
| **Sentinel** | [x] | **your `model.bin`** |

We are not bending the building-block philosophy. We are obeying it.

### Why it *must* be per-mission

**"Normal" is not universal.** A Mars rover's normal is nothing like a CubeSat's - different
batteries, orbits, thermal environments, duty cycles. A model trained on SMAP would be *actively
wrong* on your spacecraft and would alarm constantly on healthy hardware.

**A shared model is not a better product. It is a broken one.**

### What adoption actually costs a mission

1. Point our training script at telemetry they already have
2. Run it -> get `model.bin`
3. Uplink it, or bake it in

**No ML expertise. No neural-network knowledge. No labelled failures.**

---

## 6. What kind of learning this is

Three labels, all correct, increasing in precision:

| Label | Meaning |
|---|---|
| **Self-supervised** | Nobody hand-labels anything. The next timestep *is* the answer; we hide it and ask the model to guess. |
| **Semi-supervised / one-class** | Trained on normal data only. The term to use in a paper. |
| **Forecasting-based anomaly detection** | The actual technique: predict -> compare -> flag sustained divergence. |

### 6.1 Why normal-only training is mandatory, not preferred

**Your spacecraft has never failed.** You have zero failure examples. Any method requiring
*"here are 500 examples of the battery failing"* is impossible - that data does not exist and
will not exist until it is too late.

So we invert the problem: **learn normal perfectly; treat everything else as suspicious.**
We never have to know in advance what the failure will look like.

Supervised classifiers are excluded by this fact, not by preference.

### 6.2 "Rare nominal events" are normal data

A frequent point of confusion, resolved here:

```
  +-- TRAINING SET ----------------------------+   +-- EXCLUDED --+
  |                                            |   |              |
  |   routine telemetry        (common normal) |   |  anomalies   |
  |   manoeuvres, resets,      (RARE normal)   |   |              |
  |   calibrations                             |   |              |
  +--------------------------------------------+   +--------------+
```

**Rare != abnormal.** Manoeuvres, resets and calibrations are nominal - a human deliberately
caused them - and they belong in training. The reason they matter is that they are *rare*, so a
naive model barely sees them and panics when they occur.

Having them **labelled** lets us do both halves:
- put them in **training** so the model learns "a thruster burn looks like this, it's fine"
- put them in **test** to *prove* we do not false-alarm on them

We exclude only anomalies.

---

## 7. Detection: prediction error plus discipline

A healthy spacecraft matches its forecast; error stays small. A developing fault pulls reality
away from the learned trajectory and error grows - **while every raw value is still comfortably
inside its limits.** That gap in time is the early warning.

```
  value
    |                                          RED LIMIT
    | - - - - - - - - - - - - - - - - - - - - - - - - -
    |                                              /
    |   forecast ............................     /
    |   reality  ~~~~~~~\                        /
    |                    \______________________/
    |                    ^                       ^
    |              divergence begins        limit trips
    |              SENTINEL WARNS           legacy alarm
    |                    |<---- ~4 hours ---->|
    +---------------------------------------------------- time
```

Three mechanisms turn a raw error signal into an operator-grade warning:

| Mechanism | What it does | Why |
|---|---|---|
| **Persistence filter** | Signal must survive N consecutive cycles | Suppresses blips; a single weird sample is ignored. **Measured as subsumed on the LSTM - see below. Retained.** |
| **Trend projection** | Rolls forecast forward against dictionary limits | Turns "something's off" into "crosses RED_LO in ~4h" |
| **Explanation layer** | Learned map of which channel pairs move together, including time-lagged | Names the break, so the warning is **auditable, not a score** |

The explanation layer is what makes warnings actionable. An operator can pull up the two named
channels and judge the claim in seconds. `anomaly: 0.87` is unactionable and gets ignored.

### 7.1 The persistence filter is subsumed on the LSTM, and is kept anyway

Measured on `m1-g8.9.10`, sweeping N over 1, 5, 20 and 60 against the k-of-n channel agreement:

```
      N       F0.5    alarm ranges   MVGS recall   lead time (median)
      1      0.269         182          28/32           +26 steps
      5      0.271         181          28/32           +22 steps
     20      0.254         179          28/32           +10 steps
     60      0.216         177          25/32           -32 steps
```

**N=5 removed one alarm range out of 182.** The reason is that telemanom's detection stack already
widens every threshold exceedance by +/-99 timesteps (its `error_buffer`), so by the time a signal
reaches the persistence filter there are no single-sample blips left to suppress. The mechanism
this section describes is real; on this detector something upstream already performs it.

Past N=20 the filter costs rather than buys: recall falls, and lead time falls roughly one
timestep per unit of N, which is the filter working exactly as designed and is why early warning
must be measured rather than assumed.

**The mechanism is not removed.** It is architecture, not a tuning knob, and it may well be live
again for the GRU and the TCN in work items 5 and 6 if their detection stacks carry no equivalent
widening. What is recorded here is a measurement on one detector, not a repeal.

---

## 8. Model architecture: LSTM vs GRU vs TCN

All three answer one question: **how does the model remember recent history?**

### Approach A - carry a running summary

A small internal state, updated by each new reading.
*Analogy: a nurse who never re-reads the chart but keeps a running impression.*

```
  reading1 --> +-------+ --> +-------+ --> +-------+ --> forecast
               | state |     | state |     | state |
  reading2 ---->       +----->       +----->       |
               +-------+     +-------+     +-------+
```

### Approach B - re-read a fixed window

No carried state. Keep the last N readings and look at all of them fresh, every cycle.
*Analogy: a nurse who remembers nothing but re-reads the last 30 pages each visit.*

```
  +----------- history ring buffer (N steps) -----------+
  | ################################################## |--> forecast
  +-----------------------------------------------------+
         dilated causal convolutions across the window
```

### The three candidates

| | Memory | Pros | Cons |
|---|---|---|---|
| **LSTM** | Running summary, 4 gates, 2 state vectors | JPL's published choice -> directly comparable numbers; small runtime footprint (no history buffer) | Most parameters per unit capacity; **nastiest C++** - 4 gate computations, 2 states to manage |
| **GRU** | Running summary, 3 gates, 1 state vector | ~25% fewer parameters; literature shows parity with LSTM at this scale; **much simpler C++ - fewer places for a fault to hide**; smaller model file | Not the published reference |
| **TCN** | None - fixed window | **Identical compute every cycle** (clean WCET analysis); **no state to corrupt**; **restarts perfectly clean**; easiest to verify in bare C++ | Needs history ring buffer (small at our scale); **hard blindness beyond the window** |

**Working hypothesis:** a GRU matches LSTM detection performance on our benchmarks. Recent
time-series literature frequently shows TCNs equalling or beating recurrent models, so the TCN
is a serious candidate, not a curiosity.

### Selection gate - five criteria, in priority order

```
  1. Detection performance   < contextual recall FIRST, false-alarm rate second
  2. Parameter count
  3. Inference cost per cycle
  4. C++ implementation & verification simplicity
  5. Determinism / restart behaviour
```

We do not argue about this. All three train on identical data and score on identical labels.

**Outcome worth flagging to stakeholders in advance:** LSTM retained as the published-comparison
baseline, with GRU or TCN selected for flight if it matches. That result is itself a presentable
finding - *"the reference method's accuracy at a fraction of the flight complexity."*

---

## 9. Data

### 9.1 Primary - ESA-ADB (ESA Anomaly Detection Benchmark)

**This is our primary evaluation set.** Built over 18 months by Airbus Defence and Space,
KP Labs, and ESA's European Space Operations Centre. Labels written by spacecraft operations
engineers who actually flew these missions. Zenodo, CC BY 3.0 IGO.

| Property | Value |
|---|---|
| Missions | 3 real ESA missions (anonymised) |
| Scale (our 3-mission ingest) | 224 channels, 821 control signals, 1430 annotated events: **157 anomalies**, **716 rare nominal events**, 401 communication gaps, 156 invalid segments |
| Benchmark subset (2 missions) | 176 channels, ~1.55 billion data points, 17.5 mission-years, 844 annotated events of which **690** are rare nominal events |
| Anomaly density | ~1.80% (Mission1), ~0.57% (Mission2) - realistic |
| Taxonomy | 54 event classes by dimensionality / locality / length |
| Size | ~11.6 GB |

Note that **1430 counts every annotated event across all four categories**, not
anomalies. Anomalies number **157 dataset-wide** - 51 on the group-8 channel set,
46 on the Phase 1 primary set. Reading 1430 as an anomaly count overstates the
evidence available by roughly nine to one.

**Five reasons it is transformative for this project specifically:**

1. **Genuinely multivariate and synchronised** - real cross-channel relationships exist, so our
   correlation layer can be *proven*. Its R3 design requirement explicitly rewards modelling
   inter-channel dependencies. It was built to test exactly our capability.
2. **The right taxonomy** - most anomalies are *"multivariate global subsequence"*, precisely our
   target class. We can report recall on that specific class, which is the number that proves
   the thesis.
3. **Realistic density** - real spacecraft are almost always fine. A benchmark reflecting that
   punishes false alarms honestly.
4. **690 labelled rare nominal events** in the 2-mission benchmark subset; **716** across our
   full 3-mission ingest. Both figures are correct in their own scope - do not reconcile them by
   changing one. Manoeuvres, resets, calibrations. See section 6.2. This is
   the single most valuable feature: it lets us measure and fix the exact behaviour (crying wolf
   at routine operations) that would otherwise kill adoption within a week of flight.
5. **The top published baseline is our own method** - *Telemanom-ESA-Pruned*, a pruned variant of
   the LSTM prediction-error approach we are implementing. Architecture validated on this data,
   and a specific number to beat.

**Known trade-off:** anonymised - channel names, units and absolute timelines hidden, so no
physics-informed feature engineering. **Nearly harmless for us, arguably a feature:** our method
never uses channel semantics, and anonymised data forces exactly the mission-agnostic generality
we claim. If an experiment ever needs full metadata, CATS is the fallback.

### 9.2 Why we demoted SMAP/MSL (telemanom)

Formerly the anchor. Now **legacy comparability only** - reported for continuity with prior work,
flagged, never headlined.

**Reason 1 - publicly discredited.** Wu & Keogh (IEEE TKDE 2023) identify four flaws:
triviality, unrealistic anomaly density, mislabelled ground truth, run-to-failure bias - and
show that trivial one-liners (a moving average and a standard deviation) achieve
state-of-the-art on it. *If a two-line script wins your benchmark, your benchmark is not
measuring sophistication.*

**Reason 2 - fatal for us specifically.** Per the PATH dataset paper (Kaufmann et al.), while
SMAP/MSL are technically multivariate, **the channels are not synchronised with each other.**
"82 channels" is effectively 82 unsynchronised univariate streams.

> Cross-channel detection is our entire selling point. There are no genuine cross-channel
> relationships in that data to find. **We could not have proven our core claim on it** - not
> because the method is weak, but because the data physically cannot demonstrate it.

### 9.3 The full data stack

| Dataset | Role | Notes |
|---|---|---|
| **ESA-ADB** | **Primary** | Real, multivariate, synchronised, operator-labelled |
| **OPSSAT-AD** | Real-data secondary | 9 channels, ~2,100 fragments, Zenodo 10.5281/zenodo.12588359. **Univariate per segment** - tests single-channel behaviour only, *not* cross-channel |
| **CATS** | Synthetic stress test | 17 vars, 5M timestamps, 200 injected anomalies, full root-cause metadata. Zenodo 10.5281/zenodo.8338435. **Mark clearly as synthetic** |
| **SMAP/MSL** | Legacy comparability | Report with flags; never a headline claim |
| ALFA / DASHlink | Cross-domain robustness | UAV / aviation. Second-tier |

### 9.4 The search is closed

A targeted sweep of 2023-2026 JAXA, CNES, DLR, Chinese-mission and KOMPSAT releases found **no
new real, multivariate, publicly downloadable, expert-labelled spacecraft anomaly dataset beyond
ESA-ADB and OPSSAT-AD.**

| Candidate | Fails on |
|---|---|
| Mars Express | No anomaly labels (forecasting dataset) |
| LASP / WebTCAD | No ground-truth labels |
| CATS | Synthetic |
| KOMPSAT / Chinese / JAXA | Private, non-public |
| NASA Shuttle Valve | Univariate, trace-level labels |
| New "CubeSat" sets (CuCD-ID, AegisSat) | Cybersecurity/intrusion on digital twins, not operational anomalies |

**This is a result, not a dead end.** The question "are we missing a better dataset?" is closed
with evidence, and "we exhaustively surveyed the public landscape" is defensible in review.

### 9.5 Metrics and storage

**Metrics:** adopt **event-wise F0.5 / VUS-PR**. **Avoid point-adjusted F1** - known to inflate
results. Consider building the harness on **TimeEval**, which ESA-ADB's official harness
(`kplabs-pl/ESA-ADB`) already uses.

**Storage policy:** datasets live on the project's Cloudflare R2 bucket, converted to parquet,
with SHA-256 checksums and a manifest recording provenance. Streamed into memory for
experiments. **Never committed to the repository** - code and docs only, with a gitignored local
`data/` staging path.

(!) **Sizing note:** ESA-ADB is ~11.6 GB vs telemanom's 272 MB - roughly **40x more data** through
the ingest. Not a blocker, but the ingest, parquet conversion and streaming approach must be
sized for it **before** the harness is written against it.

---

## 10. The cold-start problem - and how we solve it

**The honest statement of the problem.** Our design assumes a mission has enough pre-launch
telemetry to train on. Sometimes true, sometimes not - and the "not" cases are often the ones
using F' most.

### 10.1 Three stacked problems

**(!) Volume varies wildly.**

| Mission type | Realistic pre-launch telemetry |
|---|---|
| JPL flagship | Months of flatsat + sim. Plenty. |
| Commercial smallsat | Weeks. Fine. |
| University CubeSat | Days. Sometimes a weekend before shipping. |

F' spans all three. We cannot design for the best case.

**(!) Ground data != flight data.** A flatsat has no vacuum, no orbital thermal cycling, no real
solar illumination, no microgravity, and wall power instead of a battery under real load. Learned
relationships may genuinely differ in orbit. That is physics, not a defect in the method.

**(!) Launch is the worst possible moment.** LEOP is when *nothing* is normal - deployments,
detumble, first sun acquisition, first eclipse. Every value is transient and unprecedented.
**Human operators do not know what is normal either; that is what commissioning is for.** A
detector claiming full confidence during LEOP is lying.

### 10.2 Five fixes - all adopted

**[x] 1. Data-sufficiency report - the key move.**
The training toolkit does not just produce a model. It **grades the mission's data before
launch** and refuses to overstate coverage:

```
  -- SENTINEL DATA SUFFICIENCY REPORT -----------------------------
   Power    : 340,182 samples,  47 eclipse cycles   SUFFICIENT   OK
   Thermal  :  12,004 samples,   0 full cycles      INSUFFICIENT NO
              > thermal channels DISABLED in model.bin
   AOCS     :  88,410 samples,  12 slews            MARGINAL     !!
              > wide thresholds, low-confidence tier
  -----------------------------------------------------------------
```

A mission finds out **on the ground, not in orbit**. No mission ever gets a false sense of
security. This alone converts an open risk into a managed, documented one.

**[x] 2. Watch few channels, not all.**
Model **8-12 channels in one subsystem** - power and thermal first. Their relationships are the
most physical, most universal and most stable between bench and orbit (current->temperature holds
in vacuum; attitude dynamics do not). Fewer channels means far less data needed *and* the
strongest relationships.

**This is a per-instance limit, not a per-spacecraft one.** Sentinel is generic code loading a
mission-specific model file, so an F' topology may instantiate it once per subsystem -
`Sentinel(power)`, `Sentinel(thermal)`, `Sentinel(aocs)` - each with its own `model.bin` and its
own channel set. Coverage scales by adding instances, not by widening one model. This keeps each
model small enough to train on limited pre-launch data, restricts learned relationships to
channels that are physically related, keeps the explanation layer tractable, and lets the
data-sufficiency report enable or disable subsystems independently. It also avoids the failure
mode of a single wide model: across many channels some pairs correlate by chance rather than
physics, and a model that learns those raises false alarms when they break. Instances also fail
and restart independently, where one wide model is a single point of failure.

*(Phase 1 note. The Phase 1 evaluation set is `m1-g8.9.10`: 12 channels spanning **two**
subsystems - group 8 is subsystem_5, groups 9 and 10 are subsystem_6. Spanning normalisation
groups is permitted under the resolved policy in section 14.8: cross-group scale offsets are
fixed, invertible and uninformative, and any model absorbs them; it is per-channel rescaling that
destroys information. The set was chosen because it is the only one carrying both all 11 point
anomalies and the real multi-hour extent of the headline-cell events. **Phase 1 stays
single-instance** - the gate asks which architecture, not how many channels, and changing both at
once makes the result unattributable. See docs/HARNESS.md.)*

*(Scope note. Phase 1's channel selection is a benchmark artifact and does not describe how an
adopting mission chooses. A mission has no labelled anomalies, so it selects on physics, on the
data-sufficiency report from fix 1, and on criticality - never on measured detection performance,
because there is none to measure. See docs/HARNESS.md, section 6b.)*

**[x] 3. Train on simulator + flatsat together.**
Nearly every mission builds a sim or digital twin, and F' has strong sim tooling.

```
   SIMULATOR                      FLATSAT
   + eclipses          OK          + real sensor noise      OK
   + thermal cycling   OK          + real component quirks  OK
   + orbital dynamics  OK          + true bus behaviour     OK
   + real hw noise     NO          + orbital conditions     NO
            +--------------+--------------+
                  COMBINED - gaps covered
```

Neither alone is sufficient; both together is usually plenty for a model this small.

**[x] 4. Confidence tiers - honest from day one.**
Sentinel ships with **wide, conservative thresholds** and reports its own confidence as
telemetry.

| Phase | Behaviour |
|---|---|
| **LEOP** | Watching. Learning nothing. Only flags gross decoupling. |
| **Commissioning** | Ground compares predictions vs. reality; thresholds tightened |
| **Nominal ops** | Model v2 uplinked, trained on **real flight data**. Full sensitivity. |

**Model v2 is planned, not a failure.** It uses the same file-uplink path already built - and
after v2 the model is trained on actual orbit data, which is better than any ground-only model
could ever be.

**[x] 5. Subsystem priors - the long game.**
Battery/current/temperature physics does not care whose logo is on the bus. We train a generic
**power-subsystem base model** on public data; each mission **fine-tunes** it on their small
dataset. A CubeSat with days of data starts from something that already understands batteries.
Standard transfer learning; directly rescues the university case.

*This is **Level 2** of the tiered capability architecture - section 14.10, decision 10, where
the open risk is recorded: ESA-ADB is anonymised, so we can build a generic model of telemetry
dynamics but not a generic battery model, and whether that transfers is unproven.*

### 10.3 The honest claim

> **Not:** "full anomaly detection from second one."
>
> **Yes:** "conservative, self-aware monitoring from second one - covering the subsystems the
> mission proved it had data for - sharpening to full sensitivity after commissioning."

The first version does not survive a review board. The second does.

### 10.4 The compounding advantage

Every mission that flies Sentinel generates real, labelled flight telemetry. Over time that is a
growing corpus of genuine spacecraft normal, feeding better priors for the next adopter.

**The cold-start problem shrinks with every mission that adopts.**

---

## 11. Permanent safety and operational rules

Five rules constrain the design permanently. Each has a stated reason. **None is a v1
limitation.**

| # | Rule | Reason |
|---|---|---|
| **1** | **Model frozen in flight.** Retraining is explicit and human-approved. | A slowly degrading spacecraft must never be able to teach the detector that degradation is normal - that is precisely the condition it exists to catch. |
| **2** | **Silent until validated.** No output until sufficient history backs a warning. | A detector that cries wolf gets ignored, and an ignored detector is worse than none. |
| **3** | **Warn-only. Commands nothing.** | Detection and response are separated. Response belongs to existing fault protection and to humans. Nobody has to trust a neural network with control authority. |
| **4** | **Every warning explainable.** Named channels, named relationship breaks, time-to-limit. | Operators must evaluate claims, not trust scores. |
| **5** | **Deterministic onboard code.** Fixed memory, fixed compute per cycle, same inputs -> same outputs. | This is what flight review boards can approve. |

**No online learning. Ever.**

---

## 12. Roadmap - four phases, four gates

A phase ends when its gate is **demonstrably** passed. A gate that cannot be passed **stops
work** while the design is revisited. **Nothing gets coded around.**

```
  +------------------------------------------------------------------+
  |  PHASE 1  <-- WE ARE HERE                          Python        |
  |  Prove the mathematics                                           |
  |  GATE: match or beat a telemanom baseline reproduced on this     |
  |        harness, plus evidence-based architecture selection.      |
  |        "Reproduced" is qualified: telemanom's detection method   |
  |        - LSTM forecast, nonparametric dynamic thresholding,      |
  |        pruning - with a MULTIVARIATE forecaster over the whole   |
  |        channel set, not one univariate model per channel, which  |
  |        cannot express the cross-channel structure this project   |
  |        exists to detect. Every deviation is listed in            |
  |        docs/MODELS.md. External comparability to published       |
  |        ESA-ADB numbers is out of scope for Phase 1.              |
  |                                                                  |
  |  Time-to-limit estimation is validated in Phase 3 on the F'      |
  |  Ref deployment. ESA-ADB's anonymised timestamps make it         |
  |  structurally impossible in Phase 1.                             |
  +------------------------------------------------------------------+
  |  PHASE 2                                            C++          |
  |  The flight component - FPP model, static-memory inference,      |
  |  model-file loader, event/channel outputs, unit tests            |
  |  GATE: tests green, flight-rule compliance clean                 |
  +------------------------------------------------------------------+
  |  PHASE 3                                            C++          |
  |  Integration + demo - wired into the F' Ref deployment;          |
  |  faults injected as slow drifts and decouplings that stay        |
  |  inside limits throughout                                        |
  |  GATE: limit alarms SILENT while Sentinel emits early warnings   |
  |        with time-to-limit - captured as a screen recording       |
  +------------------------------------------------------------------+
  |  PHASE 4                                            C++          |
  |  Hardware envelope - same deployment on Raspberry Pi /           |
  |  Snapdragon-class hardware; CPU, memory, timing measured         |
  |  GATE: comfortable margins documented                            |
  +------------------------------------------------------------------+
                              v
     Adoption path: flatsat testing > first flight on sponsoring
     organisation's mission > community release to the F' ecosystem
```

**Phase 3's recording is the project's single most persuasive artifact:** a spacecraft where
every limit light stays green while Sentinel says *"this crosses its red limit in four hours."*

### Why no C++ yet

Flight-grade C++ is slow and expensive - FPP models, static memory, no dynamic allocation, no
exceptions, bounded loops, full unit tests, flight-rule compliance. If we build all that and
*then* discover the detection is weak or the architecture wrong, months are gone.

**Prove it in Python where changing your mind is free. Then build it once, correctly.**

---

## 13. Current work

**Phase 1. The immediate task is the evaluation harness - not a model.**

It loads the labels, streams telemetry from R2, and scores **any** candidate detector on:

- recall over point anomalies
- **recall over contextual anomalies** <- the class that matters
- false-alarm rate, including on rare nominal events

### Why the harness comes first

> **Build the referee before the players.**

Build models first and every result is a debate about splits, metrics and what "caught it" means.
Build the referee first and every subsequent experiment is instantly and identically measurable.
LSTM vs GRU vs TCN becomes a **table of numbers, not an argument.**

### Work items, in order

```
  1. OK  Repository stood up, documentation-first
  2. OK  Ingest ESA-ADB > R2 (parquet, SHA-256, manifest)   234 objects, 11.53 GB
  3. OK  Evaluation harness built, baselines scored, floor recorded
  4. --  Reproduce telemanom's detection method with a       <-- current task
         multivariate LSTM forecaster
  5. --  Train + score GRU
  6. --  Train + score TCN
  7. --  Pass the architecture selection gate. Scores m2-ss1 as well,
         across LSTM, GRU, TCN, rstd and mavg together, so the adoption
         number on an independent spacecraft is a comparison and not a
         lone figure
  8. --  Post-gate: injected-fault sensitivity study
```

**Item 8, recorded now so it is not lost.** Controlled drifts and decouplings injected into real
ESA-ADB telemetry. **Not for headline numbers** - scoring on faults we designed only tests
whether the detector finds what we thought of. It buys two things the 46 real events cannot: a
**detection sensitivity curve** (how slow a drift can be before we lose it) and **lead-time
measurement**, both of which discriminate between LSTM, GRU and TCN where the real events may not.
It complements Phase 3's F' Ref fault injection (section 12) and CATS (section 9.3) rather than
replacing either.

---

## 14. Open decisions to track

| # | Decision | Deadline | Why it matters |
|---|---|---|---|
| 1 | **Architecture selection** - LSTM vs GRU vs TCN | End of Phase 1 | Criteria in section 8 |
| 2 | **Model-file format freeze** | **Before Phase 2 starts** | It is the contract between the Python toolkit and the C++ loader. The format implication is recorded in 14.10; empirical findings from the work item 4 weight extraction are in `docs/MODELS.md` |
| 3 | **Channel-ingestion mechanism** - tapping the telemetry path vs. direct port wiring | Early Phase 2 | Resolve against the pinned F' version |
| 4 | **Target F' version pin** | Early Phase 2 | Everything downstream depends on it |
| 5 | **Harness base** - build on TimeEval or standalone | Now | TimeEval gives ESA-ADB-comparable metrics for free |
| 6 | **R2 ingest sizing** for 11.6 GB | Before item 3 | 40x the previous data volume |
| 7 | **Second independent scoring set** | Before item 7 | **Open.** No second viable *recall* set exists in ESA-ADB: Mission2 dedupes to 18 anomalies with 1-3 test-side, and Mission3 has 8 anomalies with 4 of 48 channels numeric. Resolved in practice by splitting the roles - Mission1 carries recall, Mission2 carries the adoption number - with the single-spacecraft limitation stated on every result |
| 8 | **Normalisation policy** | Was: before the loader | **RESOLVED - identity.** ESA min-max scaled within each channel group, so amplitude ratios between related channels survive. Cross-group spanning is acceptable: those offsets are fixed, invertible and uninformative, and a model absorbs them. Per-channel rescaling is refused, because it erases the ratios and no model can recover them. Enforced at a chokepoint and by `tests/test_no_per_channel_scaler.py` |
| 9 | **SatNOGS as subsystem-prior corpus** | Post-gate | Open. Feeds section 10.2 fix 5 - a generic power-subsystem base model that each mission fine-tunes on its own small dataset |
| 10 | **Tiered capability architecture** - Level 1 / 2 / 3 | **Before Phase 2** | **OPEN.** One C++ loader, one file format, three capability tiers. Level 1 is the loader's mandatory safe failure mode, not a data-availability fallback. Detail in 14.10 |

### 14.10 Tiered capability architecture (decision 10, OPEN)

The component ships as three tiers, all sharing **one C++ loader and one model file format**.

```
  Level 1   statistical cross-channel baseline, zero mission data
  Level 2   small pretrained model, fine-tuned on limited data
  Level 3   full mission-specific training
```

**Level 1 is mandatory, and it is not primarily about data availability.** It is the loader's
safe failure mode. A corrupt file, a version mismatch, a failed CRC or a radiation bit-flip must
degrade to Level 1 with an event and an active-tier telemetry channel - **never fail the
topology**. That requirement holds regardless of how much data a mission has, which is why
Level 1 is not optional for well-provisioned missions.

**Level 2 is the intended shipped default once it exists.** A mission then gets a working
detector out of the box and fine-tunes it, rather than training from scratch. It is section 10.2
fix 5 - subsystem priors - made concrete. It **cannot be built before the Phase 1 architecture
gate**: you cannot pretrain without knowing which architecture to pretrain.

**(!) Open risk on Level 2, flagged for investigation.** ESA-ADB is anonymised - no channel
names, no units - so we cannot identify which subsystem is power and which is thermal. We can
therefore build a generic model of telemetry *dynamics*, but **not a generic battery model**.
Whether generic dynamics transfer sufficiently is unproven and must be tested.

Prior evidence is encouraging but not conclusive. Baireddy et al., *Spacecraft Time-Series
Anomaly Detection Using Transfer Learning* (CVPR Workshops 2021, AI4Space; Purdue and Lockheed
Martin Space) pretrained on unlabelled Mars Reconnaissance Orbiter telemetry and fine-tuned on
SMAP/MSL at roughly half the training time, reporting:

```
                     recall   precision
  transferred        0.774      0.765
  trained from       0.835      0.794
  scratch
```

**The often-quoted 0.769-vs-0.814 F1 pair is derived, not published.** The paper reports
precision and recall; `F1 = 2PR/(P+R)` gives 0.769 transferred against 0.814 from scratch, so
transfer retained about 94% of from-scratch F1. The arithmetic is shown here so nobody quotes
those two numbers back as figures the paper printed.

**Zero-shot detection is the weakest link.** Transfer learning works well for *forecasting*;
anomaly *detection* specifically is less established. An inaccurate Level 2 is worse than no
Level 2 at all - by section 11 rule 2, a detector that cries wolf gets ignored, and an ignored
detector is worse than none. **Ship Level 2 only when measured.**

**Format implication for decision 2**, which must freeze before Phase 2: a quantized,
self-describing FlatBuffer, TFLite-Micro compatible. The header carries model type, version,
channel count and ordering, window length, quantization parameters, a mandatory `baseline_only`
flag, and a CRC over the weights. **Normalisation constants and detection thresholds are stored
separately**, as small PrmDb-style parameters, and are never baked into the weights - so a
mission can recalibrate in orbit without retraining.

**Precedent, worth recording.** This mirrors NASA cFS, where apps ship default configuration
tables so a mission has working behaviour on day one and overrides them later. F's own
equivalents are PrmDb - a generic component loading a mission file - and the
`FPRIME_ENABLE_TEXT_LOGGERS` pattern, a capability that can be compiled out. We are following an
established framework pattern, not inventing one.

---

## 15. The hallway version

> We're building a plug-in module for NASA's F' flight software. It learns what a spacecraft's
> normal telemetry looks like from pre-launch tests, then runs onboard in C++ watching for the
> relationships *between* channels breaking down - which is how 41% of real anomalies show up,
> invisible to limit checks. It warns hours early with a named cause and a time-to-limit, and
> never commands anything. JPL proved the method, ESA proved it flies, nobody's made it
> reusable. That's us. Right now we're in Python, proving the detection works against ESA's
> benchmark before writing a line of flight code.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **F' / F Prime** | NASA/JPL's open-source flight software framework. Flew on Ingenuity. |
| **Channel** | One telemetry measurement, e.g. `BattVoltage` |
| **Contextual anomaly** | Every channel legal, but the combination or trajectory is wrong |
| **Cross-channel analysis** | Watching whether channels move together as they normally do |
| **Rare nominal event** | Manoeuvre, reset, calibration - abnormal-looking but **normal** |
| **LEOP** | Launch and Early Orbit Phase - when nothing is normal yet |
| **FPP** | F Prime Prime, F's modelling language for components |
| **WCET** | Worst-Case Execution Time - what flight reviewers demand bounds on |
| **`model.bin`** | The mission-specific file of numbers Sentinel loads at startup |
| **Telemanom** | JPL's published LSTM anomaly-detection system; our reference method |
| **ESA-ADB** | ESA Anomaly Detection Benchmark - our primary dataset |