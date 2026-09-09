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
of a developing fault, visible before any limit trips - is downlinked as a standard F' event
with a named cause and a time-to-limit estimate. **The forecast is conditioned on every kind of
context the telemetry provides**: a channel's own history, the commands the spacecraft was sent,
and the other sensors watching the same subsystem. The detection method is JPL's published
reference approach; onboard feasibility was demonstrated in orbit by ESA. **The reusable F'
packaging is this project's original contribution.**

---

## 1. The goal

> Build a plug-in module for NASA/JPL's F' flight software that lets a spacecraft spot its own
> problems before anything breaks - and make it a standard building block any mission can
> drop in.

Not a research paper. Not a mission-specific script. **A block for the box.**

**Definition of done:** an F' component that a JPL flagship, a commercial smallsat, and a
university CubeSat can each adopt without ML expertise, using telemetry they already generate.

**(!) RE-FRAMED 2026-09-09 (`docs/DECISIONS.md` D57).** Sections 0 and 1 used to say
"visible long before any limit trips" and "hours before anything breaks". The wall-clock
words are struck here and everywhere: no such figure has been measured, and section 1.1
has said so since 2026-08-28. What replaces them is not weaker, it is testable -
**earlier than a limit check, with minimal false alarms** - and the lead in real time is
Phase 3's measurement on a real clock against real dictionary limits. Section 0 also
gains the sentence about context, which is the substance of D57: the claim is early
warning from whatever context the telemetry carries, and the number of contexts is not
the distinction.

---

### 1.1 (!) What is claimed, and what is retired -- 2026-08-28

**Read this before any figure in this document is quoted.**

**(!) RE-FRAMED 2026-09-09, and this block governs everything below it**
(`docs/DECISIONS.md` D57). Nothing below is deleted or corrected; what changes is
which claim the evidence is offered for.

**THE CLAIM. Sentinel warns before a limit trips, using every kind of context the
telemetry provides.** Three kinds, and the project has measured them to very
different depths:

```
  (a) a channel's own history      TESTED.  SMAP/MSL, 10 of 38 in-range
                                            contextual sequences at 0.68%
                                            nominal (D46, D48, MODELS 26.18)
  (b) commands -> telemetry        TESTED, AND NEGATIVE. NASA's per-channel
                                            arrays carry the commands on the same
                                            clock: 24 one-hot columns on SMAP, 54
                                            on MSL, column 0 the telemetry.
                                            Conditioning on them COSTS -- twice,
                                            on two training configurations and
                                            two model families (D49, D60). At
                                            this encoding, on this data, the
                                            context is there and using it makes
                                            the detector worse. A richer encoding
                                            is the one lever left (MODELS 32.6)
  (c) other sensors -> telemetry   UNMEASURED on any public data. ESA-ADB's
                                            labelled faults are gross (D44); the
                                            F' Ref physics testbed is the venue
```

**Single-context or multi-context is not the distinction. Earlier than a limit
check, with minimal false alarms, is.**

**(!) Amended 2026-09-09 by D60.** Context (b) was listed as *barely tested* when
this block was written on the same day. It has since been tested and the answer is
negative: withholding the commands from the published training configuration gains
eight events, removes five false alarms and lowers the alarm rate at once
(`docs/MODELS.md` 32.7). **A context being present in the data is not the same as
it being usable**, and this project now has that measured rather than assumed. It
does not weaken the claim above -- it narrows which contexts currently pay. Warn-only, never commands (section 11
rule 3). The model is frozen in flight (rule 1); the threshold is recalibrated in
orbit without retraining (14.10); a shadow model may retrain in flight under human
approval, which is Phase 5 and is scoped in `docs/PHASE5.md` and built by nobody
yet.

**RETIRED: "cross-channel means sensor-to-sensor, and that is the whole selling
point."** This document said from 2026-08-24 that detecting broken relationships
*between sensors* was the thesis, and every design choice below was made to serve
it - the identity normalisation (D2), the multivariate forecaster, the channel
grouping, the dataset search at 9.4. Three findings retire it as **the sole
thesis**:

- **D42.** No cross-channel reduction available to the decision layer recovers an
  event `max` misses on ESA-ADB - L2, sum, `k = 2` and `k = 3` are all strict
  subsets at lower alarm rates. D23 closes as answered no.
- **D44.** On ESA-ADB's stationary folds a calibrated per-channel range check is
  sufficient and better at a matched rate, and the forecaster never speaks first
  across 53 caught events.
- **`docs/RESEARCH.md` Part IV.** Pinet et al. (arXiv:2606.02670, MiLeTS at KDD
  2026) find across eight public benchmarks that "no cross-channel rupture occurs
  without an accompanying univariate deviation across a range of reasonable
  thresholds", and that channel-dependent modelling brings no measurable gain.
  **This is the first independent replication of a finding of ours** (D42, D23),
  and it removes the thesis' only public venue.

**(!) The retirement is conditional, and the conditions travel with it.**
`docs/RESEARCH.md` states the finding as *if* no public benchmark carries strictly
cross-channel segments; it names **SWaT, WADI and SKAB as unchecked candidates**;
it records that **absence across eight benchmarks is strong evidence rather than
proof**; and it notes that a near-binary channel defeats a z-score diagnostic,
which is a live concern for SMAP/MSL specifically. Pinet test deviation from
*normal history* while D46 tests leaving the *training min/max* a limit check
actually holds - compatible questions, not the same one. **None of these four
qualifications may be dropped when the retirement is quoted.**

**What is NOT retired.** The capability itself. The multivariate forecaster is
built, matches its NumPy reference at 1.8e-07 in C++ (D30) and runs in a live F'
deployment (D32-D37); D46's in-range contextual population is real at 39 of 43;
and nothing else in an F' deployment watches relationships between channels at
all. **What changed is that sensor-to-sensor is one of three contexts rather than
the whole claim, and the one venue that can test it is Phase 3's testbed.**

**Also retired with it: the wall-clock framing in sections 0 and 1.** See the
rider there.

---

**The 2026-08-28 and 2026-09-03 record follows, unchanged.**

**RETIRED: "+26 timesteps of early warning."** Every lead-time figure this
project published was dated from the start of an alarm range that
`error_buffer` had widened **backwards from a crossing that had already
happened**. Measured from the batch boundary at which the detector can actually
emit, the median lead is **-43.0** on the gate set, and **fifteen of thirty-eight
detected events have no emission overlapping them at all** -- the alarm touched
them only through the backward widening. `docs/RESULTS.md` 6f,
`docs/DECISIONS.md` D21.

**And a second qualification that stands whatever the number is.** Lead time here
is measured against the **labelled event start**, which is a hindsight annotation
written by an operations engineer after the fact. It is *not* a limit trip. Even
a positive figure would say *we spoke before the annotation begins*, not *we
spoke before the spacecraft was in danger*. The two are different quantities and
only the second is the product claim.

**KEPT, and narrower than it was:** nothing else in an F' deployment watches
the relationships between channels at all. That is a statement about the F'
ecosystem, verified in section 3, and nothing in D37, D38 or D39 touches it.

**WITHDRAWN, and it was the evidence offered for the sentence above.** This
used to continue: "On `m1-g8.9.10` a forecaster over the channel set finds 28 of
32 headline-cell events where a per-channel statistic finds 3." Every figure in
it is superseded. The 3 was `baselines._rolling` accumulating in float32 and is
**25** (D37). The 28 of 32 is `lstm-telemanom`'s, a detector disqualified at 22
of 48 commanded manoeuvres; the detector that flies scores **22/32**. And the
flying detector catches a **strict subset** of the corrected floor's events on
both Mission 1 sets (D38) -- which D41 later showed is a property of the two
thresholds and not of the two detectors, and D44 later made moot: at a matched
alarm rate a per-channel range check beats the forecaster on both sets and the
forecaster never speaks first.

**(!) RESTATED 2026-09-03, and this is the claim.**

**Sentinel catches anomalies a limit check can never see.** On NASA's SMAP/MSL
telemetry, **39 of 43** labelled contextual anomalies stay entirely inside their
channel's historical range; **no per-channel statistic reaches a flyable alarm rate
there** -- a rolling standard deviation at 5,000x its calibrated threshold still
alarms on 15.17% of nominal steps; and the forecaster **under a dynamic threshold**
operates at **0.68%** and catches **10 of 38** -- the baseline now being improved.

**(!) Read the configuration with the claim, never apart from it.** That 10 of 38 is
`gru-telemanom` -- the **published dynamic threshold**, not D25's frozen static
quantile -- scored **per channel, univariate, without command conditioning**, on
SMAP/MSL's 81 unsynchronised streams. **It is not the configuration that flies
today**, which is `gru-quantile`: multivariate, static quantile. The component ships
both rules and a mission selects one, because which rule is needed is a property of
the telemetry regime and not of the method (D48, D49, `docs/MODELS.md` 26.18).
**And it is 26%**: ten of thirty-eight, on one dataset, with no floor available to
compare against at that alarm rate.

**Subordinate, and the reason the component ships both:** on ESA-ADB -- stationary
folds, gross faults -- a **calibrated per-channel range check is sufficient and
better**, catching 34/46 and 25/32 at an equal or lower alarm rate against the
forecaster's 27/46 and 22/32, and the forecaster never speaks first across 53 caught
events (D44). The full record is unchanged in `docs/RESULTS.md` 6l and 6m and in
D43 to D49.

**What it replaced**, kept for the record: the Phase 1 headline comparison was
withdrawn on 2026-09-02 pending re-measurement (D37, D38), and the corrected
baseline beside the flying detector is in `docs/RESULTS.md` 6l -- the forecaster
leading on `m1-g8.9.10` (F0.5 0.804 against 0.676) and trailing on `m1-ss5` (0.593
against 0.663). Work items 9.7, 9.8 and 9.9 are the re-measurement.

**The break-to-limit-trip lead is a Phase 3 deliverable and is unmeasured.**
ESA-ADB carries no dictionary limits and its timestamps are anonymised and
scaled, so the quantity cannot be computed on this data at all (section 9.1,
`docs/HARNESS.md` section 4). It is measured on the F' Ref deployment, on a real
clock, against real limits.

> **No wall-clock claim -- no "hours", no "~4h" -- may be made from Phase 1
> evidence.** Every such figure in this document is a **design target for Phase
> 3**, not a result, and is marked as one below.

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

### 4.4 (!) What is built, and what section 4.2 describes but does not exist

**Added 2026-08-27, because otherwise this document describes a system we do not
have.** Measured by inspecting every stage between the forecast and the alarm
(docs/MODELS.md section 11, docs/DECISIONS.md D23).

**The forecaster is multivariate. The decision layer is not.** One model predicts
every channel from every channel, which is why **22 of 32** headline-cell events
are caught (was "28 of 32", which was `lstm-telemanom`'s; D37, D38).
The mechanism below is unchanged by the correction; only the number is. Everything after that point sees C independent error series: smoothing,
thresholding, sequence-finding, pruning and persistence each take **one channel
at a time**. The k-of-n channel agreement is the only stage that looks at more
than one, and what it tests is **co-occurrence, not relationship** -- k channels
simultaneously over their own individual thresholds is not the claim that the
relationship between them broke.

> **The cross-channel claim currently rests entirely on the forecaster. The
> decision layer is blind to what the forecaster learned.**

That is not a neutral gap. **A commanded manoeuvre makes several channels
individually surprising at once, which is exactly what k-of-n rewards, and it is
nominal** -- so the one stage meant to recover cross-channel structure recovers
the kind that generates false alarms. The adoption number and the thesis are
failing at the same place, for the same reason.

**And the explanation layer in section 7 does not exist.** Section 7 promises a
"learned map of which channel pairs move together, including time-lagged", and
section 4.2 shows its output -- `BattTemp / ChargeCurrent decoupled at 14:32`, a
named *pair*. What is implemented is `last_attribution`: the index of the single
channel with the largest error at each timestep, which is not written to any
artifact. No pair map, no lag map, no covariance exists anywhere in `src/`.
Naming a channel is not naming a relationship, and section 11 rule 4 -- every
warning explainable -- rests on the difference.

Neither is a defect in what was built; both are things that were assumed built
because this document says so. What a relationship-testing stage would require is
scoped in docs/MODELS.md section 12. **Nothing is decided.**

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
    |                    |<-- Phase 3 target -->|
    +---------------------------------------------------- time
```

Three mechanisms turn a raw error signal into an operator-grade warning:

| Mechanism | What it does | Why |
|---|---|---|
| **Persistence filter** | Signal must survive N consecutive cycles | Suppresses blips; a single weird sample is ignored. **Measured as subsumed on the LSTM - see below. Retained.** |
| **Trend projection** | Rolls forecast forward against dictionary limits | Turns "something's off" into "crosses RED_LO in ~4h". **(!) NOT BUILT and not measurable on ESA-ADB** -- no dictionary limits, anonymised clock. Phase 3 |
| **Explanation layer** | Learned map of which channel pairs move together, including time-lagged | Names the break, so the warning is **auditable, not a score**. **(!) NOT BUILT -- see 4.4.** What exists is the index of the single largest-error channel, which is not persisted. Naming a channel is not naming a relationship |

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
| **GRU** | Running summary, 3 gates, 1 state vector | ~25% fewer parameters (**measured 2026-08-28: 71,160 against 91,640 -- 25% in the recurrent layers, 22.35% overall**, docs/MODELS.md section 3); literature shows parity with LSTM at this scale; **much simpler C++ - fewer places for a fault to hide**; smaller model file | Not the published reference |
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
| Size | ~11.6 GB (11.53 GB measured at ingest, CHANGELOG 0.2.0) |

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

**(!) RE-READ 2026-09-09 (`docs/DECISIONS.md` D57). Both reasons stand and the
conclusion drawn from them does not.** The blockquote above is correct about what
SMAP/MSL cannot do and wrong about what follows from it.

**What stands.** Reason 1 stands: Wu & Keogh's critique is confirmed for the point
class - 11 of 61 point anomalies are in range against 39 of 43 contextual (D46) -
and the P-2 duplicate found at ingest is a live instance of their mislabelled-ground-truth
flaw. Reason 2 stands as recorded, with the caveat `docs/RESEARCH.md` already carries:
the unsynchronised-channels claim comes from the PATH paper **second-hand and has not
been verified against the raw arrays.**

**What does not.** "We could not have proven our core claim on it" was true of the
sensor-to-sensor claim, which D57 retires as the sole thesis. SMAP/MSL turns out to
support two other claims this project cares about more:

1. **The in-limits claim.** 39 of 43 labelled contextual sequences stay strictly inside
   their channel's training min/max - six and a half times ESA-ADB's population and five
   times denser (D46). It is the population ESA-ADB does not have, and it is the only
   place the in-limits claim has been measured at all: 10 of 38 at 0.68% nominal.
2. **The command-context claim.** Its per-channel arrays carry the commands **on the
   same clock as the telemetry**, which no other dataset here does: 24 one-hot columns
   on SMAP and 54 on MSL, column 0 being the telemetry. That is context (b) of 1.1, and
   D49 measured exactly one encoding of it before bounding itself to that encoding.

**So the demotion is narrowed rather than reversed.** SMAP/MSL remains legacy
comparability for anything cross-channel, and is **primary** for the in-limits and
command-context questions, which is what every result since 2026-09-03 has used it for.
Its limitations are reported on every figure drawn from it.

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

(!) **Sizing note:** ESA-ADB is ~11.6 GB (11.53 GB measured at ingest) vs telemanom's 272 MB - roughly **40x more data** through
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

### 10.2 Five fixes - all adopted - and a sixth (3a) added 2026-08-28

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

**[x] 3a. No alarm budget, ever. The threshold is a measured noise floor.**
*(Added 2026-08-28.)* The toolkit never asks a mission what alarm rate it wants,
and nothing in the workflow turns a sensitivity dial. **Sentinel fires when a
learned relationship breaks and is silent otherwise; zero alarms one month and
five the next are both correct outcomes.** The count belongs to the spacecraft.

The threshold exists for one reason: *the relationship broke* is a measurement
with sensor noise on it, so a line between hum and break has to be drawn. That
line is **calibrated** from the mission's own nominal pre-launch residuals -
what does this spacecraft's normal noise look like - and never **fitted** to make
a result acceptable. A target rate would teach the detector to under-report a
spacecraft that is genuinely degrading, which is section 11 rule 1 arriving by a
different route.

What a mission does get is a **sanity report, not a target**: after calibration,
how often the detector fired on held-out healthy data. A sane number validates
the calibration; an absurd one means it is broken and needs investigating, not
turning down. It sits beside the data-sufficiency report in fix 1 and is read the
same way - a statement about whether the instrument is working, not a knob.

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
  |  PHASE 1  CLOSED 2026-08-29 (D28, D29; docs/PHASE2.md)  Python   |
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

**(!) AMENDED 2026-09-09 (`docs/DECISIONS.md` D57): Phase 3 is pulled forward, and a
Phase 5 is added.**

**Phase 3 is no longer only a demo. It is the benchmark.** `docs/RESEARCH.md` Part IV
establishes that if no public benchmark carries strictly cross-channel segments, then the
coupled-physics subsystem in `SentinelRef` is **the only venue** where the
sensor-to-sensor claim can be earned or abandoned - and it is already the only venue
where the break-to-limit-trip lead can be computed at all, because ESA-ADB's clock is
anonymised and scaled (section 9.1) and SMAP/MSL carries no dictionary limits. So its
gate gains a measurement beside the recording: **a model gate on seeded, reproducible,
in-limits physics faults at a matched alarm rate, reporting in-limits catch rate,
time-to-limit-trip and manoeuvre false alarms.** It is scheduled ahead of the remaining
Phase 2 toolkit work rather than after it.

**And the recording's own line changes.** "This crosses its red limit in four hours" is
the shape of the demo, not a figure: the hours are whatever the testbed measures, and
until it measures them no number goes in that sentence (section 1.1).

```
  +------------------------------------------------------------------+
  |  PHASE 5                                            C++          |
  |  In-flight retraining under human approval - the flying model    |
  |  stays frozen (section 11 rule 1) and a SHADOW model retrains    |
  |  onboard on recent healthy telemetry, in statically allocated    |
  |  memory, with ground comparing the two and a human-approved      |
  |  command swapping them. Previous model kept for rollback.        |
  |  GATE: no heap allocation after init, no exceptions, and the     |
  |        shadow measurably better on the pre-launch sanity report  |
  |        before any swap is offered                                |
  |  Scoped in docs/PHASE5.md. Built after the toolkit, not before.  |
  +------------------------------------------------------------------+
```

**Phase 5 does not weaken rule 1 and is not online learning.** Nothing the shadow model
learns can reach the detector without a human command, the frozen model keeps flying
throughout, and the rollback path is part of the design rather than a recovery plan.

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
  4. OK  Reproduce telemanom's detection method with a
         multivariate LSTM forecaster
  5. OK  Train + score GRU   (2026-08-28; docs/RESULTS.md 6h -- two stop-and-
         report rules fired and the next step is a decision, not the TCN)
  6. OK  Train + score TCN   (2026-08-29; docs/RESULTS.md 6j -- MVGS 9/32, the
         floor above both cells' on every fold; the gate is now a decision)
  7. OK  Pass the architecture selection gate (D28: the GRU, 2026-08-29).
         m2-ss1 scored once across LSTM, GRU, TCN, rstd and mavg, and the
         union: 4/424, 4/424, 6/424, 84/424, 122/424, 8/424 rare-event
         false alarms, 0 nominal-step alarms for every forecaster -- the
         adoption number on an independent spacecraft is a comparison and
         not a lone figure. m1-g3 scored once: fold 0 clean, folds 1-2 a
         calibration collapse (87% of one window alarmed) -- the finding
         Phase 2 inherits (D29, docs/RESULTS.md 6k). PHASE 1 CLOSED
  8. --  Post-gate: injected-fault sensitivity study
```

**(!) EXTENDED 2026-09-09 (`docs/DECISIONS.md` D57).** Items 1 to 8 are Phase 1's and
are closed at item 7 (tag `wi7`). Phase 2's items were tracked in `docs/STATUS.md` rather
than here, which is how this ladder went a month out of date. They are named here so the
document a newcomer reads first is the one that lists them.

```
   9. OK  The F' component, and Level 1 safe failure   (tag wi9; D32-D37)
  9.5 OK  The _rolling correctness fix, and it falsified a published claim (D37)
  9.6 OK  Auditing the corrected floor                 (D38, D39, D40)
  9.7 OK  The comparison made like for like            (D41, D42; D23 closed)
  9.8 OK  The per-channel range check                  (D43, D44)
  9.9 OK  SMAP/MSL, stages 1 to 5                      (D45-D50)
 9.10 OK  The faithful telemanom port                  (D53, D54)
 9.11 OK  The published training configuration, Arm T  (D55)
 9.12 OK  The residual rung, Arm R                     (D55)
 9.13 --  Dimensionless guards. PRE-REGISTERED AND NOT RUN (docs/MODELS.md 30)
 9.14 OK  gru-zscore, and H4's stop fired              (D56)

  10.  --  WI10  In-orbit threshold recalibration: file uplink, human-approved
                 reload, exercised end to end on the Ref. On model.bin version 1;
                 PARAMS already carries a separate CRC and its own param_version,
                 so no format change is needed. docs/PHASE2.md 5b states what it
                 must survive, measured rather than assumed. A SELECTABLE
                 dynamic-threshold mode is a SEPARATE later item and a
                 format_version 2 decision, taken after Phase 3 shows which rule
                 a mission needs -- not folded in here (D30, docs/MODEL_FILE.md 11)
  11.  --  WI11  The F' Ref physics testbed: coupled current/heat/temperature/
                 voltage, 8-12 channels, real dictionary limits, real clock,
                 faults seeded IN THE PHYSICS and in-limits throughout for the
                 contextual family. Ground truth by construction. Then a model
                 gate on it at a matched rate. This is Phase 3, pulled forward
  12.  --  WI12  The toolkit: one command, mission telemetry in, model.bin out,
                 with the guards that exist, DIMENSIONLESS EVERYWHERE (D55), the
                 pre-launch sanity report, the tier ladder (14.10) and a no-ML
                 quickstart
  13.  --  WI13  Phase 5, in-flight retraining of a shadow model under human
                 approval. Recorded in docs/PHASE5.md; built after WI12
```

**Item 8 is still open and is now partly superseded in venue.** The injected-fault
sensitivity study was scoped against real ESA-ADB telemetry; WI11's testbed provides
seeded, reproducible, ground-truth-by-construction faults with a real clock, which is
what item 8 wanted the curve for. The ESA-ADB injection remains worth doing for the
different reason it was proposed - sensitivity on *real* telemetry - and neither replaces
the other.

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
| 1 | **Architecture selection** - LSTM vs GRU vs TCN | Was: end of Phase 1 | **RESOLVED 2026-08-29 - the GRU** (docs/DECISIONS.md D28), on section 8's criteria: criterion 1 a tie at the harness's resolution, criteria 2-5 and trainability to the GRU; the LSTM retained as the published baseline, the TCN rows retained as the stateless answer. The LSTM leads on event-wise F0.5, and D28 says so |
| 2 | **Model-file format freeze** | Was: before Phase 2 starts | **RESOLVED 2026-09-01 - plain little-endian float32, version 1** (docs/DECISIONS.md D30), specified byte for byte in `docs/MODEL_FILE.md`. Gate order and architecture named in the header, both bias vectors unsummed, and the normalisation constants and threshold in a **separately-CRC'd parameter block** so a mission recalibrates in orbit without touching the weights. 14.10's FlatBuffer/quantized container is superseded and marked so, never deleted; every requirement it stated is met |
| 3 | **Channel-ingestion mechanism** - tapping the telemetry path vs. direct port wiring | Early Phase 2 | **RESOLVED 2026-09-01 - direct port wiring** (docs/DECISIONS.md D33). The tap is the more attractive design and is not implementable against this file format: `Fw.Tlm` carries a serialized `Fw::TlmBuffer`, and `docs/MODEL_FILE.md` 4's CHANNELS record carries an id and a name and **no type tag**, so a tap cannot deserialize a value without knowing its declared type. Adding one is a `format_version` bump and D30 froze version 1. A mission supplies a small adapter -- F's own Passive Adapter Pattern -- and section 4.3 step 5's "nothing downstream needs to know Sentinel exists" is thereby weakened rather than met, which D33 records rather than glosses |
| 4 | **Target F' version pin** | Was: early Phase 2 | **RESOLVED 2026-09-01 - v4.3.0**, published 2026-08-20 (docs/DECISIONS.md D31). Forced early by work item 8: the flight rules cannot be cited without a version. F's own statement of them is `.github/skills/fprime-cpp-design/SKILL.md` at that tag, and reading it corrected three rules this project had stated from memory (docs/MODELS.md 19.3) |
| 5 | **Harness base** - build on TimeEval or standalone | Was: now | **RESOLVED 2026-08-25 - standalone** (CHANGELOG 0.3.0, Decided). TimeEval would have given ESA-ADB-comparable metrics for free; external comparability is out of Phase 1's scope (section 12) |
| 6 | **R2 ingest sizing** for 11.6 GB | Was: before item 3 | **RESOLVED 2026-08-24** - 11.53 GB as zstd parquet in 234 objects under a 90 MiB ceiling, four channels sharded (CHANGELOG 0.2.0). 40x the previous data volume |
| 7 | **Second independent scoring set** | Was: before item 7 | **RESOLVED in practice, and spent 2026-08-29** (docs/RESULTS.md 6k): Mission 2 carried the adoption number (4/424 rare events, 0 nominal-step alarms) and Mission 1 group 3 the recall exam. No second viable *recall* set exists in ESA-ADB: Mission2 dedupes to 18 anomalies with 1-3 test-side, and Mission3 has 8 anomalies with 4 of 48 channels numeric. Resolved in practice by splitting the roles - Mission1 carries recall, Mission2 carries the adoption number - with the single-spacecraft limitation stated on every result |
| 8 | **Normalisation policy** | Was: before the loader | **RESOLVED - identity.** ESA min-max scaled within each channel group, so amplitude ratios between related channels survive. Cross-group spanning is acceptable: those offsets are fixed, invertible and uninformative, and a model absorbs them. Per-channel rescaling is refused, because it erases the ratios and no model can recover them. Enforced at a chokepoint and by `tests/test_no_per_channel_scaler.py` |
| 9 | **SatNOGS as subsystem-prior corpus** | Post-gate | Open. Feeds section 10.2 fix 5 - a generic power-subsystem base model that each mission fine-tunes on its own small dataset |
| 10 | **Tiered capability architecture** - Level 1 / 2 / 3 | **Before Phase 2** | **OPEN for Levels 2 and 3; LEVEL 1 RESOLVED 2026-09-01.** One C++ loader, one file format, three capability tiers; Level 1 is the loader's mandatory safe failure mode, not a data-availability fallback. The file format carries the `baseline_only` flag and the tier, and the loader refuses a bad magic, version or CRC with a status code and no exception (D30). Work item 9 built the rest: all **11/11** refusal codes degrade to the statistical baseline with the code named in the event, **0/11** fail the topology, `baseline_only` is wired to the active-tier telemetry channel, and the baseline's constants are PrmDb-style parameters rather than model-file fields so they are readable when the file is not (D34). Watched working in a live deployment (docs/MODELS.md 20.11). Levels 2 and 3 remain open. Detail in 14.10 |

Rows of this table are cited elsewhere as `Objective.md 14.N` - decision N of the table; only 14.10 has a subsection of its own, below.

### 14.10 Tiered capability architecture (decision 10; Level 1 RESOLVED, Levels 2 and 3 OPEN)

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

**BUILT 2026-09-01, work item 9.** `Sentinel::Monitor` runs the statistical baseline whenever
the model cannot be run: all 11 refusal codes, `baseline_only` set, or no model file at all.
The radiation bit-flip in the sentence above turned out to matter twice: once as the reason
Level 1 exists, and once as the reason `Detector::step` re-checks the shape it is about to
trust, because the loader validates a `model.bin` once and the struct then lives in RAM for the
mission (docs/MODELS.md 20.9, found by clang-analyzer). The baseline transcribes the `rstd`
rule and **not** `baselines._rolling`, which computes it wrongly -- D37, and the repair is
scoped as docs/MODELS.md 21.

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

**(!) A second, sharper measurement, 2026-08-27.** Two calibrations of the *same*
threshold rule, given the *same* stated alarm budget on the *same* residuals,
produced **126 alarm ranges and 2,405**. Nineteenfold, from how the constants
were fitted rather than from any change to the model, the data or the budget.
Thresholds are not merely model-specific: they are *fitting-procedure*-specific,
and two defensible procedures disagree by more than an order of magnitude. **A
transcribed constant cannot express that and a format that bakes thresholds into
the weights cannot carry it.** They must travel in `model.bin` as fitted values
with their fitting procedure recorded beside them, and be replaceable without
retraining. This is what decision 2's format freeze needs to know.
See docs/RESULTS.md section 6b.

**(!) Empirical justification for storing thresholds separately, 2026-08-27.** The
recommendation below -- thresholds as PrmDb-style parameters rather than baked into the
weights -- began as an argument from flexibility. It is now a measured requirement.
Correcting a training defect improved the forecast about fortyfold, and the transcribed
detection threshold, unchanged, went from 182 alarm ranges to 3,548 on identical data.
**A threshold that suits one model does not suit a better one**, so it cannot be a
constant of the method: it is a fitted quantity belonging to the model it was measured
against. A mission uplinking an improved `model.bin` must uplink its thresholds with it,
and must be able to recalibrate them without retraining. See docs/DECISIONS.md, the
amendment on dimensionless constants.

**Format implication for decision 2**, which must freeze before Phase 2: a quantized,
self-describing FlatBuffer, TFLite-Micro compatible. The header carries model type, version,
channel count and ordering, window length, quantization parameters, a mandatory `baseline_only`
flag, and a CRC over the weights. **Normalisation constants and detection thresholds are stored
separately**, as small PrmDb-style parameters, and are never baked into the weights - so a
mission can recalibrate in orbit without retraining.

**(!) THE CONTAINER IS SUPERSEDED BY D30, 2026-09-01. The requirements are not.** The
paragraph above is kept because everything in it except the container still binds. What
is struck is *quantized, self-describing FlatBuffer, TFLite-Micro compatible*, for three
reasons recorded in `docs/DECISIONS.md` D30: nothing in this project consumes TFLite --
the inference core is a hand-written transcription of `reference.py` checked against it
at 1e-5, and section 4.3 step 2 refuses an interpreter outright; a FlatBuffer parser is
templated, allocating third-party code the flight rules exclude (F' CPP-25, CPP-1); and a
fixed layout with a CRC is byte-inspectable by a review board, which is worth more here
than self-description. Quantization is struck with it: the acceptance tolerance against
`reference.py` is 1e-5 and int8 loses far more than that.

**What replaces it**, frozen at version 1 and specified byte for byte in
`docs/MODEL_FILE.md`: plain little-endian float32 in `reference.Weights.arrays()` order
with **both bias vectors unsummed**; a 64-byte self-protecting header naming the
architecture and the gate order rather than leaving them to be inferred; a channel map;
and a **separately-CRC'd parameter block** carrying the normalisation constants, the
threshold, the EWMA span, `baseline_only` and the tier -- so a recalibration in orbit
overwrites a fixed-size block and never touches the 278.0 KiB of weights. Every
requirement this section stated -- constants outside the weights, the `baseline_only`
flag, the CRC, recalibration without retraining -- is met; only the container changed.

**(!) AMENDED 2026-09-09 by D55: every constant this tier ladder ships must be
dimensionless, or carry its provenance.** The section above argues that *thresholds* are
fitted quantities belonging to the model and the spacecraft. D55 generalises it from three
measured instances in three stages of one method: D17's `min_delta` disabled training
entirely on ESA-ADB's ~1e-4 loss; **the same constant was then measured correct** on
SMAP/MSL's (-1,1)-scaled data (`docs/MODELS.md` 28.7), which is what separates *the
constant is wrong* from *the constant does not travel*; and telemanom's candidate filters
at `errors.py:339` and `:343` are absolute in the units of the data. A fourth instance was
predicted and **refuted**, and is kept with its reasoning.

**So a constant expressed in the units of the data or the loss is a property of the
dataset it was fitted on.** Level 2's pretrained model and Level 3's per-mission training
both inherit this: every such constant ships a dimensionless equivalent - a fraction, a
quantile, or a multiple of the series' own dispersion - and the absolute form is available
only as a per-mission override with its provenance attached. `docs/MODELS.md` 30 tests one
implementation of the principle and is registered against the guards rather than adopted
from D55, because a principle being right does not make a particular replacement right.

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
> invisible to limit checks. It aims to warn early with a named cause and a time-to-limit -- both Phase 3 targets, neither measured -- and
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