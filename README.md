# fprime-DeepLearning-Sentinel

A reusable F' (F Prime) flight component that lets a spacecraft analyse its own telemetry
onboard.

A small neural forecaster is trained on the ground from the mission's own pre-launch test data,
exported as a plain file of numbers, and uplinked. In flight, deterministic C++ compares reality
against the forecast every cycle. Sustained divergence - the early signature of a developing
fault, visible long before any limit trips - is downlinked as a standard F' event with a named
cause and a time-to-limit estimate.

The detection method is JPL's published reference approach (telemanom); onboard feasibility was
demonstrated in orbit by ESA's OPS-SAT. **The reusable F' packaging is this project's original
contribution.**

## How to review this repository

The commit history is complete and unrewritten; the milestone view is the
[Releases page](https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/releases),
one release per Phase 1 work item (`wi1` to `wi5`), each a short summary with the
headline numbers and links into the documents. Reading order:

1. [Objective.md section 1.1](Objective.md) - what is claimed and what has been
   retired, before any figure elsewhere is quoted. Then the rest of the objective.
2. [docs/RESULTS.md](docs/RESULTS.md) - the scorecard record: every detector, both
   channel sets, `k/n` throughout, both numbers wherever a correction moved one.
3. [docs/NARRATIVE.md](docs/NARRATIVE.md) - what happened in order, mistakes included,
   and the measurements that changed conclusions.
4. [docs/DECISIONS.md](docs/DECISIONS.md) - D1 to D26: why, what else was considered,
   what evidence settled it. Superseded entries are marked, never deleted.

`docs/HARNESS.md` is the referee's contract and `docs/MODELS.md` the players'
pre-registrations, each preserved beside what actually happened. No number in any
document was written without an artifact under `runs/` to read it from.

## Why

Onboard fault protection today is per-channel limit checking. Hundman et al. (KDD 2018) found
that **41% of real, expert-confirmed anomalies on SMAP and Curiosity were contextual**: every
individual channel stayed inside its limits while the combination or the trajectory was wrong.
No amount of limit tuning catches that class.

## Status

**Phase 1 - Python, pre-flight-code.** Proving the mathematics before any flight code is
written. The harness is built and closed, telemanom's method is reproduced with a multivariate
LSTM and the decision layer is frozen (`docs/DECISIONS.md` D25), and the GRU is scored beside
it on identical terms (`docs/RESULTS.md` section 6h, 2026-08-28). Two stop-and-report rules
fired on the GRU row; a reseeded LSTM fit on fold 0 (`docs/RESULTS.md` section 6i) settled
that the LSTM's fold-0 stall was one optimisation path and still left it short of the GRU, and
returned no verdict by its own pre-registered rule. The next step is a decision, not a run.
See [docs/HARNESS.md](docs/HARNESS.md) for what is measured and
[docs/MODELS.md](docs/MODELS.md) for the pre-registrations.

| Phase | Scope | Gate |
|---|---|---|
| **1 (current)** | Python - prove the mathematics | Match/beat a telemanom baseline reproduced on our own harness - telemanom's detection method with a multivariate forecaster, not one univariate model per channel; every deviation listed in docs/MODELS.md. Plus evidence-based architecture selection. External comparability out of scope |
| 2 | C++ - the flight component | Tests green, flight-rule compliance clean |
| 3 | C++ - integration + demo in the F' Ref deployment | Limit alarms silent while Sentinel warns early |
| 4 | C++ - hardware envelope | Comfortable margins documented |

## Permanent rules

1. Model frozen in flight. Retraining is explicit and human-approved. No online learning, ever.
2. Silent until validated.
3. Warn-only. Commands nothing.
4. Every warning explainable - named channels, named relationship breaks, time-to-limit.
5. Deterministic onboard code - fixed memory, fixed compute per cycle.

## Documentation

[**Objective.md**](Objective.md) is the living objective document and the single source of truth
for this project: the problem, the prior art, the architecture, the data strategy, the cold-start
analysis, the roadmap and the open decisions. Other documents defer to it.

[CHANGELOG.md](CHANGELOG.md) records the version history.

| Document | What it holds |
|---|---|
| [docs/HARNESS.md](docs/HARNESS.md) | the referee: what is measured, on what, and the rules that govern changing it |
| [docs/MODELS.md](docs/MODELS.md) | the players: the deviation ledger, the pre-registration, and what the reproduction is missing |
| [docs/RESULTS.md](docs/RESULTS.md) | every number, with the corrections it has been through |
| [docs/DECISIONS.md](docs/DECISIONS.md) | why each decision was taken, what else was considered, what settled it |
| [docs/RESEARCH.md](docs/RESEARCH.md) | the external evidence base, with thin evidence marked as thin |
| [docs/NARRATIVE.md](docs/NARRATIVE.md) | what happened in order, errors included |
| [docs/DATA.md](docs/DATA.md) | where the data is and how to read it |
| [docs/THRESHOLD.md](docs/THRESHOLD.md) | work item 4: what telemanom's z-selection criterion is actually choosing, measured over 5.68M windows |

## Data

Datasets are never committed here. The primary evaluation set is **ESA-ADB** (~11.6 GB, CC BY 3.0
IGO, Zenodo), with OPSSAT-AD and CATS as secondary sets and SMAP/MSL retained for legacy
comparability only. Telemetry is staged to the project's Cloudflare R2 bucket as parquet with
SHA-256 checksums and a provenance manifest. See Objective.md, section 9.

## Licence

Not yet selected. Intended for community release to the F' ecosystem.
