# The datasets this project does not use, and why

> **Paths outside this branch resolve on `dev`.** `master` carries the component and the
> evidence it works, and nothing else (`docs/DECISIONS.md` D67). A citation here into
> `src/`, `scripts/`, `tests/`, `docs/MODELS.md` or `third_party/` points into the
> development branch at the commit this snapshot was taken from.
> `scripts/check_references.py --master` is what keeps that true rather than hoped.

**Twelve candidates were examined and none was adopted.** The reasons are recorded here so
a reader can disagree with them individually rather than take the search on trust -- and so
the next person does not re-run a survey that has already been run. `docs/RESEARCH.md`
Part IV is the full treatment; this is the same content for a reader who does not have it.

> **(!) `docs/RESEARCH.md`'s standing rule applies to every figure on this page: no number
> from the compiled survey may be quoted until it has been verified at source.** Where a
> count below is marked unverified, it is unverified, and the primary documentation is the
> place to settle it.

## What a dataset would have to be

The core claim is **early warning from whatever context the telemetry carries** (D57),
measured on anomalies that **never cross a limit line**. That needs, together:

1. **real** telemetry, not simulated;
2. channels on a **common clock** with genuine physical coupling -- unsynchronised streams
   cannot demonstrate a cross-channel claim (D1);
3. labels by people who **operated the system**, not by an injection script;
4. a population of anomalies that **stay inside the operating envelope**, because that is
   the population a limit check cannot see.

**No public dataset has all four.** ESA-ADB has 1-3 and not 4; SMAP/MSL has 3 and 4 and
not 2. That gap is why the F' Ref physics testbed exists on the roadmap: it is the only
instrument that can produce all four, plus a real clock.

## The twelve, with the reason each was declined

| Dataset | What it is | Why not |
|---|---|---|
| **SWaT** (iTrust, SUTD) | Real water-treatment plant, expert-designed physical attacks, common 1 Hz clock, genuine coupling | **The strongest candidate and not closed.** Access is by data agreement. Pinet classify every long segment as **BOTH** rather than cross-channel-only, and one segment is roughly **65% of anomalous time and single-feature-detectable**. Channel and attack counts **disagree between secondary sources** and are **unverified** here |
| **WADI** (iTrust, SUTD) | Water distribution, same programme | As SWaT, same agreement, same unverified counts |
| **SKAB** | Open pump rig, correlated sensors on a common clock, physically induced faults | Real coupling and cheap corroboration, but **small**. Worth revisiting as a second opinion, not as a primary set |
| **Tennessee Eastman** | Chemical-process simulation | **Simulated.** A model's anomalies are the model's |
| **CATS** | Synthetic multivariate with full root-cause metadata | **Simulated**, and already flagged as such in `Objective.md` 9.3 |
| **C-MAPSS** | Turbofan degradation simulation | **Simulated**, and framed as remaining-useful-life rather than anomaly detection |
| **OPS-SAT-AD** | Real ESA CubeSat telemetry | **Univariate per fragment by construction**, so it cannot test this class at all |
| **Exathlon** | Spark cluster traces | **IT telemetry, not a physical system** |
| **SMD** | Server machine dataset | Same |
| **PSM** | Pooled server metrics | Same |
| **GECCO** | Water-quality challenge data | Named in the survey's eight; not a spacecraft-like coupled system |
| **SWAN-SF** | Solar-flare prediction benchmark | A forecasting benchmark on a different problem shape |

## The non-public routes, named rather than pretended away

The survey names these for real spacecraft telemetry with expert cross-channel labels:
**ESA/ESOC ARTS** through the ESA-ADB authors, **JPL**, **JAXA**, **CNES**, **DLR**, and
CubeSat operators including **SatNOGS**. `Objective.md` decision 9 carries this as an open
item. **None has been approached**, and nothing in this repository depends on one being.

## (!) The acceptance test any future dataset must pass first

The one concrete thing the survey changed. **Pinet's per-segment diagnostic --
`UNIVARIATE / CROSS-CHANNEL / BOTH / UNDETECTED` -- is the acceptance test for any dataset
this project claims tests cross-channel modelling.**

`scripts/smap_visibility.py` already does the **min/max half** per segment. **The
correlation half is not built**, and **no dataset should be adopted for the cross-channel
claim until it is.** D66 adds a second requirement beside it: an **envelope-width report**,
because on SMAP/MSL the in-range contextual class turned out to sit on systematically wider
channels than the point class, and a dataset can look like it has the population this
project needs when what it has is a channel-composition artifact.
