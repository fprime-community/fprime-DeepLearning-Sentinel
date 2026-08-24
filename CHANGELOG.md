# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0 the version
tracks documentation and Phase 1 research milestones rather than a released flight component.

## [Unreleased]

### Planned - Phase 1 work items, in order

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

## [0.2.0] - 2026-08-24

ESA-ADB ingested to Cloudflare R2. Phase 1 work item 2 complete; the evaluation
harness (item 3) can now be built against a stable, manifest-addressed dataset.

### Added

- `src/sentinel_data/` - the ingest toolkit. `zenodo.py` (resumable source
  download, MD5-verified), `esa_adb.py` (nested-zip reader), `transcode.py`
  (parquet with a hard 90 MiB object ceiling and time-based sharding),
  `r2.py` (client with per-HTTP-attempt operation accounting), `manifest.py`,
  `docs_gen.py`, and a `spike / download / transcode / prepare / upload` CLI.
- `scripts/roundtrip_check.py` - proves manifest -> key -> object -> DataFrame.
- `docs/DATA.md` and `docs/manifest.snapshot.json` - regenerated from the
  manifest on every ingest, so they cannot drift from the bucket.
- `.env.example`, `requirements.txt`.

### Data

- 224 channels across 3 missions (76 / 100 / 48; 58 / 47 / 24 target),
  ~2.30 billion points, 11.53 GB as zstd parquet in 234 objects.
- 821 telecommand files merged to one object per mission. 681 of Mission1's 698
  carry executions; the other 17 are declared but never executed.
- Annotations in 4 objects, with `labels` pre-joined to `anomaly_types` so the
  harness reads one object instead of two on every run.
- 4 channels exceeded the 90 MiB ceiling and were split into time-ordered
  shards, recorded in the manifest under `shards`.

### Verified

- Every object checked by size, ETag against the locally computed MD5, and a
  SHA-256 carried in object metadata. The manifest is published only after all
  234 objects verify, so its presence guarantees the dataset it describes.
- 236 Class A and 238 Class B operations, 0.5% of the 50,000/month ceiling.
- Nothing retained locally: source archives and parquet are deleted per mission
  once verified, and the scratch directory is removed and asserted gone.

### Decided

- Timestamps stay nanosecond and values keep their native dtype. The dataset
  documents its anonymisation as numerically lossless, so the archive does not
  downcast. Some channels are categorical and are stored as strings.
- Storage is 11.53 GB, about 1.5 GB beyond R2's 10 GB free tier (~$0.02/month).
  The brief's "possibly inside the free tier" does not hold: the source pickles
  are already deflate-compressed and float32 values resist zstd.
- Source verification uses MD5. Zenodo publishes no SHA-256 for this record.

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

[Unreleased]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/releases/tag/v0.1.0
