# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0 the version
tracks documentation and Phase 1 research milestones rather than a released flight component.

## [Unreleased]

### Planned - Phase 1 work items, in order

- Ingest ESA-ADB to R2 (parquet, SHA-256, manifest). Note: ~11.6 GB, roughly 40x the previous
  data volume - size the ingest for it before the harness is written against it.
- Build the evaluation harness (current task).
- Reproduce the LSTM baseline.
- Train and score GRU.
- Train and score TCN.
- Pass the architecture selection gate.

### Open decisions

| # | Decision | Deadline |
|---|---|---|
| 1 | Architecture selection - LSTM vs GRU vs TCN | End of Phase 1 |
| 2 | Model-file format freeze | Before Phase 2 starts |
| 3 | Channel-ingestion mechanism - telemetry-path tap vs direct port wiring | Early Phase 2 |
| 4 | Target F' version pin | Early Phase 2 |
| 5 | Harness base - build on TimeEval or standalone | Now |
| 6 | R2 ingest sizing for 11.6 GB | Before the ingest |

## [0.1.0] - 2026-08-24

Repository stood up, documentation-first. No code yet, by design: Phase 1 proves the detection
in Python before a line of flight C++ is written.

### Added

- `Objective.md` - the living objective document and single source of truth. Covers the problem
  and the 41% contextual-anomaly evidence, prior art (telemanom, OPS-SAT, and the empty lane for
  a reusable F' block), the generic-code / mission-data architecture split, the learning
  formulation, the detection discipline (persistence filter, trend projection, explanation
  layer), the LSTM vs GRU vs TCN selection gate, the data stack with ESA-ADB as primary, the
  cold-start analysis and its five fixes, five permanent safety rules, the four-phase roadmap,
  and the open decisions.
- `README.md` - project summary, status and pointers.
- `CHANGELOG.md` - this file.
- `.gitignore` - datasets, trained artifacts and Python build output excluded.

### Decided

- ESA-ADB is the primary evaluation set. SMAP/MSL demoted to legacy comparability only: it is
  publicly discredited (Wu & Keogh, IEEE TKDE 2023) and its channels are not synchronised with
  each other, so it physically cannot demonstrate the cross-channel claim.
- Metrics: event-wise F0.5 / VUS-PR. Point-adjusted F1 is avoided as it inflates results.
- Datasets are never committed to the repository. Code and docs only.

[Unreleased]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/releases/tag/v0.1.0
