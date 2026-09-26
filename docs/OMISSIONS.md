# What is not on this branch, and why

> `master` carries the component, the toolkit and the evidence they work. This document
> lists everything `dev` has that this branch does not. Paths outside this branch resolve
> on `dev` at the commit named at the top of `README.md`.

**Every omission is named.** A curated branch that quietly drops things is worse than an
uncurated one, because a reader cannot tell what they are not seeing.
`tests/test_master_omissions_are_all_named.py` (on `dev`) compares the two trees and fails
if a tracked path on `dev` is absent here and not accounted for below.

| Absent | What it is | Why |
|---|---|---|
| `src/` -- **in part** | **The ground toolkit and its import closure are HERE since D80**: `src/sentinel_toolkit/`, `src/sentinel_export/`, and the eight `src/sentinel_models/` and twelve `src/sentinel_eval/` modules they need. Absent: the scoring harness (`harness.py`, `scorecard.py`, `splits.py`, `ops.py`, `metrics/`), the detector registry and the baselines, and **all of `src/sentinel_data/`** | The absent half is the research apparatus. `src/sentinel_data/` is the R2 ingest package, and **D80 keeps it off this branch deliberately**: it holds the cloud client and the credential reader, and the toolkit reaches neither |
| `scripts/` -- **in part** | **Ten build scripts are HERE**: `fprime_setup.sh` and `fprime_ref_patch.sh`, which this branch's own documents tell you to run, and `oxcaml_setup.sh`, `oxcaml_e1.sh`, `oxcaml_s61.sh` and `oxcaml_s72.sh`, which build the retrainer's object (D80); and, since D84, `oxcaml_shape.sh` and `oxcaml_shape.py`, which generate and gate it at a mission's shape, `s72_flying_file.py`, which writes the flying file HO1 needs, and `fprime_ref_retrainer.sh`, the opt-in proven in F's own Ref. Absent: the guards, the vector generators, and every study that produced a figure | The absent ones are development apparatus, not product |
| `tests/` | The Python suite, which runs against `src/` | Runs against `src/`, which is not here |
| `docs/MODELS.md` | Every pre-registration beside its outcome | The research record. Cited from here, resolves on `dev` |
| `docs/DECISIONS.md` | Every decision with its alternatives and the evidence that settled it | The research record. **D69 removed it from this branch and it is kept whole on `dev`** -- `docs/EVIDENCE.md` cites the active entries by number |
| `docs/NARRATIVE.md` | What happened in order, mistakes included | Same. The retractions bearing on the result are in `docs/EVIDENCE.md` |
| `docs/HARNESS.md`, `docs/DATA.md`, `docs/RESEARCH.md`, `docs/THRESHOLD.md`, `docs/PHASE2.md`, `docs/PHASE5.md`, `docs/PHASE1_REPORT.md`, `docs/TELEMANOM_EXCERPTS.md`, `docs/REORG_PLAN.md`, `docs/INDEX.md` | The internal documents | Same |
| `docs/RESULTS.md` | Every scored result, every sweep, and the tables the headline is read from | The research record. `docs/EVIDENCE.md` carries the result of record and its caveats |
| `Objective.md` | What this project is for, its permanent rules and its four phase gates | The research record. `docs/DESIGN.md` states the rules that bind the component |
| `CHANGELOG.md` | Version by version | Development history; `dev` has it |
| `third_party/telemanom/` | The published source, vendored byte-identical at `2e6c5b6c` | Evidence for the research record. **This branch therefore does not redistribute it**, so BSD clauses 1 and 2 do not bind here -- clause 3 does, and `README.md`'s Licence section carries it verbatim |
| `docs/manifest.snapshot.json`, `docs/reorg_plan.json` | Machine-readable companions to the research record: a reference snapshot of the data manifest, and the data behind `docs/REORG_PLAN.md` | The research record |
| `.env.example`, `conftest.py`, `requirements.txt` | A credential template, pytest's collection scope, and the repository-wide dependency list | Development apparatus, not product. **`requirements.txt` is deliberately not here**: it installs `boto3`, `botocore`, `requests` and `python-dotenv` for the ingest apparatus, and a dependency list that installed a cloud client would contradict the `src/` row. `requirements-toolkit.txt` is this branch's list -- `numpy`, `pyarrow`, `torch` |
