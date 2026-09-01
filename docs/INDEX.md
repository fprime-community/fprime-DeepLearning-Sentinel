# The documents, one sentence each

Read in the order the README gives: Objective.md section 1.1, RESULTS, NARRATIVE, DECISIONS.
Then whatever the question needs.

| Document | What it contains | Read it when |
|---|---|---|
| [Objective.md](../Objective.md) | The living objective: the problem, prior art, architecture, the LSTM/GRU/TCN gate, the data stack, cold start, the five permanent rules, the roadmap, the open decisions - and section 1.1, what is claimed and what is retired | First, and whenever a figure is quoted |
| [docs/STATUS.md](STATUS.md) | The one-page status and roadmap: the goal, the headline table, what was done and learned, what happens next, where everything lives | First, beside the README, for a ten-minute assessment |
| [docs/PHASE1_REPORT.md](PHASE1_REPORT.md) | The self-contained account of Phase 1 for a newcomer: the problem, the method, the journey as it happened, what Phase 2 inherits | When you want the whole story in one sitting |
| [docs/RESULTS.md](RESULTS.md) | The scorecard record: every detector, both channel sets, `k/n` throughout, both numbers wherever a correction moved one; sections 6a-6k are the corrections and the three architectures; 6k the held-back sets | For any number |
| [docs/NARRATIVE.md](NARRATIVE.md) | What happened in order, errors included, and the measurements that changed conclusions | To understand why the rules exist |
| [docs/DECISIONS.md](DECISIONS.md) | D1 to D37: each decision with its context, alternatives, evidence and consequence; superseded entries marked, never deleted | Before changing anything |
| [docs/HARNESS.md](HARNESS.md) | The referee's contract: metrics, tasks, splits, the facts that would silently corrupt results, operations, the closed-to-scope-changes rule, and its own failed pre-registration (section 7). Its section 5a follows section 6, by history | Before scoring anything |
| [docs/MODELS.md](MODELS.md) | The players: the deviation ledger against published telemanom, the trained-in-torch/scored-in-NumPy contract, every pre-registration (sections 4, 10, 13, 14, 15, 16, 18, 19, 20) preserved beside its outcome, the combination scoped (17) and the `_rolling` correctness fix scoped (21), the flight core and format pre-registered (19), the F' component and Level 1 pre-registered (20) | Before fitting anything |
| [docs/THRESHOLD.md](THRESHOLD.md) | Work item 4's measurement of telemanom's z-selection criterion over 5.68M windows (its D1-D3 are diagnostics, not decisions) | When a threshold question comes back |
| [docs/DATA.md](DATA.md) | Generated from the manifest on every ingest: bucket layout, missions, operating rules, the caveats (anonymised time, per-group scaling, zero-order hold) | Before reading the bucket |
| [docs/RESEARCH.md](RESEARCH.md) | The external evidence base, primary and secondary marked, thin evidence flagged | When an external claim is cited |
| [docs/MODEL_FILE.md](MODEL_FILE.md) | The normative `model.bin` specification, frozen at version 1 by D30: the byte layout, the gate order named rather than inferred, both bias vectors unsummed, the separately-CRC'd parameter block, the refusal codes and the determinism flags. Owned by `src/sentinel_export/`; the C++ loader is written against it | Before touching the file format, the writer or the loader |
| [docs/FPRIME.md](FPRIME.md) | The F' toolchain work item 9 was built against, pinned at v4.3.0 (D31): one command to rebuild it, every version as measured, what is gitignored and why it lives inside the repository, the toolchain proof, and where `Ref` moved to at v4.3.0 | Before building anything under `fprime/` |
| [docs/PHASE2.md](PHASE2.md) | What the C++ phase inherits: the GRU blueprint, the format facts, the frozen decision layer and its calibration, the union's status, the open decisions carried; section 7 is what work item 9 settled | At the start of Phase 2 |
| [CHANGELOG.md](../CHANGELOG.md) | Versions 0.1.0 to 0.5.0 (Phase 1 closed at tag wi7; the flight core and the frozen model file at tag wi8), work item by work item, and the open decisions | For what changed when |
| [docs/manifest.snapshot.json](manifest.snapshot.json) | A generated copy of the bucket manifest for reading by humans; **no code reads it** (`check_no_list` rule 4) | To see what the bucket holds without spending an operation |

Branches: `main` is one commit per checkpoint; the full history is on `dev` with tags `wi1`-`wi8`
and their Releases. Artifacts live under `runs/`, gitignored, and are cited by path.
