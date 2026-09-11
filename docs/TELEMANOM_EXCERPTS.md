# telemanom, indexed: where every citation in this repository resolves

**Written 2026-09-09.** This document is a **navigation index into
`third_party/telemanom/`**, which is vendored in this repository at a pinned
commit and stays there. It quotes nothing that is not already in that tree, it
replaces nothing, and no citation anywhere in this repository points at this file
instead of at the source.

**Why an index and not an excerpt file.** A handover proposed replacing the
vendored tree with a document containing the cited lines verbatim. That was
withdrawn on the evidence: **78 distinct locations are cited across 13 files** (was 64 across 9, counted by hand), and
the cited ranges of `errors.py` alone cover most of it, so an honest excerpt file
would be the source with prose around it and a worse provenance. More
importantly, `docs/MODELS.md` 26.29 opens *"Every line number below is a line in
this repository. Nothing here is recalled"* -- and `docs/NARRATIVE.md` 11 records
that the last time a reading of this source was kept without the source, five
recorded readings turned out to be wrong and one cost four pre-registered rungs.
**The index is additive. The source is the evidence.**

---

## 1. Provenance

```
  vendored at   third_party/telemanom/
  upstream      https://github.com/khundman/telemanom
  commit        2e6c5b6c3558e7835601519b7bdef37c649bdbdc
  dated         2025-01-17 07:42:45 -0600
  retrieved     2026-09-08, shallow clone, source only
  paper         Hundman, Constantinou, Laporte, Colwell, Soderstrom,
                "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
                Dynamic Thresholding", KDD 2018, arXiv:1802.04431
  decision      docs/DECISIONS.md D53
```

**If it is ever re-pinned, the commit SHA changes in the same commit as the files,
and every quotation whose line numbers move is re-checked.** The line numbers in
this index are line numbers at the SHA above and nowhere else.
`tests/test_layering.py::test_the_vendored_source_is_present_and_pinned` asserts
the SHA and the licence string are both still in `PROVENANCE.md`, so this index
cannot silently outlive the tree it points into.

## 2. Licence: BSD 3-Clause, and one clause that binds on release

`third_party/telemanom/LICENSE.txt`, unmodified:

> Copyright (c) 2018, California Institute of Technology ("Caltech").
> U.S. Government sponsorship acknowledged.

Three clauses, of which **clause 3 is an obligation on this project and not only
on its code**: it forbids using the name of Caltech or of the Jet Propulsion
Laboratory to endorse or promote anything derived from this software.

> **No document in this repository may present this project as endorsed by,
> affiliated with, or produced by Caltech or JPL.** Naming telemanom's authorship
> and citing the paper is description, not endorsement, and remains correct.

**This is release housekeeping, not a footnote** (Objective.md 13, housekeeping).
`docs/DATA.md` recorded the upstream as Apache-2.0 until 2026-09-08 and that was
wrong; the stored R2 manifest object `_manifest/smap_msl.json` **still carries the
wrong string**, because rewriting a pinned manifest costs one Class A and has not
been approved. Recorded rather than quietly fixed.

## 3. It is evidence, never a dependency

- Nothing under `src/`, `scripts/`, `tests/` or `flight/` imports it, asserted per
  file by `tests/test_layering.py::test_nothing_imports_the_vendored_source`.
- Nothing in it is ever executed here.
- `scripts/check_no_list.py` scans `src/` and `scripts/` only, so this subtree is
  outside it by construction -- correct, since the code never runs, and stated in
  `PROVENANCE.md` rather than left implicit.
- `LICENSE.txt` is `.txt` and therefore outside the ASCII test's `.md` filter. It
  contains three non-ASCII bullet characters. It is upstream's file and is not
  edited to suit this project's typography.

## 4. The mechanisms, by name

The nine locations that carry a finding, each re-resolved against the vendored
files on 2026-09-09 and each at the line it is cited at. **What each contains is
described here; the code is read in the source.**

| Location | Mechanism | What it settled |
|---|---|---|
| `errors.py:355-359` | **the batch clip.** Window 0 ignores its opening history; every later window judges only its newest `batch_size` | **telemanom is causal after its opening window**, so its accumulator unions *disjoint* batches and there is no union over ~30 overlapping verdicts anywhere in it. D52 superseded at its premise; `1a+1b` is the faithful arm and `1c-ii` a departure (D53) |
| `detector.py:117-136` | **`evaluate_sequences`' accounting.** True positives deduplicated per matched *true* anomaly at `:130-132`; false positives counted per *predicted* range at `:135-136` | The published precision denominator is a **mixed unit**, not ranges over ranges, so every precision figure in the rung ladder was the generous statistic (D53) |
| `modeling.py:113`, called at `:172` | **`aggregate_predictions(..., method='first')`**, called with no method | **Published telemanom forecasts from the single one-step-ahead prediction** where this project averages ten. `windows.py`'s docstring asserted the opposite from work item 4 until 2026-09-08. The largest single training difference, T-a (`docs/MODELS.md` 28.1) |
| `errors.py:342-343` | **the magnitude conjunct**: a candidate needs `e_s >= epsilon` *and* `e_s > 0.05 * inter_range` | Ladder rung L1. Bound on **23 indices in 83,780 window-passes**, settling L1 as a property of the data rather than a transcription defect (`docs/MODELS.md` 28.7) |
| `errors.py:337-340` | **the whole-window bail-out**: a window whose errors are small relative to its values returns with no candidates at all | Ladder rung L2 |
| `errors.py:314-315` | **the two `find_epsilon` guards**: at most 5 sequences, and fewer than half the window's indices flagged | Ladder rung L3, and 28.8's candidate for what silences MSL -- which 29.4 then refuted from the other side |
| `errors.py:132-148` | **the inverse pass**: `find_epsilon`, `compare_to_epsilon` and `prune_anoms` run a second time on a mirrored error series, and the two index sets are **unioned** at `:147-148` | Ladder rung L4. Pinned by `tests/test_smap_rungs_port.py::test_the_inverse_pass_only_ever_adds` |
| `errors.py:84-93` | **`adjust_window_size`**: shrinks the window until the series admits at least one, and raises if `batch_size` exceeds the test length | Ladder rung L5, the published window regime |
| `errors.py:48-64` | **the residual rung**: the raw residual over the supervised region only, then the first `l_s` smoothed samples replaced by the mean of the first `2*l_s` | Arm R (`docs/MODELS.md` 29). **Measured inert**: Arm R equals Arm T in every cell |

**Two constants that are the whole of D55's third instance**: `errors.py:339`
requires `max(e_s) > 0.05` and `:343` requires `e_s > 0.05 * inter_range`, both
absolute or semi-absolute in the units of the data. `docs/MODELS.md` 30 registers
their dimensionless replacements and has not run.

**And the published hyperparameters** are `config.yaml`, transcribed into
`src/sentinel_models/lstm.py:90-96` with each marked *published* and each checked
against the file: `l_s: 250`, `layers: [80,80]`, `dropout: 0.3`,
`n_predictions: 10`, `batch_size: 70`, `epochs: 35`, `patience: 10`,
`min_delta: 0.0003`, `validation_split: 0.2`, `lstm_batch_size: 64`, `p: 0.13`,
`error_buffer: 100`, `window_size: 30`, `smoothing_perc: 0.05`. **`batch_size: 70`
is the error-window batch and `lstm_batch_size: 64` is the trainer's** -- two
published constants this project conflated until T-b (`docs/MODELS.md` 28.1).

## 5. The complete index

Every location cited anywhere in this repository, and what cites it. **Generated
from the tree on 2026-09-10 by `scripts/telemanom_citations.py`**, not
hand-maintained: it is regenerated rather than edited, and a citation added
without a regeneration will simply be missing from this table rather than wrong
in it.

**(!) The generator is committed as of 2026-09-10, and until then it was not.**
The sentence above claimed this table was generated while nothing in the tree
could regenerate it, which is how it came to be missing fifteen locations --
among them `errors.py:62`, `errors.py:70`, `errors.py:337-339` and the whole
`403`-`418` pruning ladder that `docs/MODELS.md` 37.2 reads line by line. The
script also checks every cited range against the length of the file it names:
**no citation overruns its file**. One rule in it is worth knowing before adding
a citation: **a bare `detector.py:NNN` is NOT indexed**, because this repository
has a `detector.py` of its own and every bare occurrence in the tree resolves to
that one. Cite the vendored file as `telemanom/detector.py:NNN` if you mean it.

| Location in `third_party/telemanom/` | Cited by |
|---|---|
| `telemanom/channel.py:55` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/channel.py:55-67` | `docs/MODELS.md` |
| `telemanom/channel.py:62` | `docs/MODELS.md`, `src/sentinel_models/lstm.py` |
| `telemanom/channel.py:63-67` | `docs/DECISIONS.md`, `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/channel.py:69-82` | `CHANGELOG.md`, `docs/DECISIONS.md`, `docs/MODELS.md`, `scripts/decision_layer_probe.py` |
| `telemanom/detector.py:117-136` | `docs/MODELS.md` |
| `telemanom/errors.py:40-42` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:48-49` | `docs/MODELS.md` |
| `telemanom/errors.py:48-64` | `CHANGELOG.md`, `docs/MODELS.md`, `docs/NARRATIVE.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:51-52` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:51-59` | `docs/MODELS.md` |
| `telemanom/errors.py:58` | `src/sentinel_models/telemanom.py` |
| `telemanom/errors.py:62` | `CHANGELOG.md`, `docs/MODELS.md`, `docs/REORG_PLAN.md` |
| `telemanom/errors.py:62-64` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:70` | `docs/DECISIONS.md`, `docs/MODELS.md` |
| `telemanom/errors.py:84-93` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:111-168` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:123-130` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:132-136` | `docs/MODELS.md` |
| `telemanom/errors.py:132-142` | `docs/MODELS.md` |
| `telemanom/errors.py:132-148` | `docs/MODELS.md`, `scripts/smap_rungs.py`, `tests/test_smap_rungs_port.py` |
| `telemanom/errors.py:147-148` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:152` | `docs/MODELS.md` |
| `telemanom/errors.py:157-160` | `docs/MODELS.md` |
| `telemanom/errors.py:159-160` | `docs/MODELS.md` |
| `telemanom/errors.py:159-166` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:165` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:165-166` | `docs/MODELS.md` |
| `telemanom/errors.py:241` | `scripts/smap_rungs.py`, `src/sentinel_models/telemanom.py` |
| `telemanom/errors.py:249-250` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:252` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:252-253` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:262-267` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:269-322` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:285` | `docs/MODELS.md`, `scripts/smap_rungs.py`, `src/sentinel_models/telemanom.py` |
| `telemanom/errors.py:286` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:288` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:290` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:291` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:291-298` | `CHANGELOG.md`, `docs/NARRATIVE.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:302-304` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:304` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:310-311` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:314-315` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:315` | `tests/test_smap_rungs_port.py` |
| `telemanom/errors.py:324-384` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:337-339` | `CHANGELOG.md`, `docs/DECISIONS.md`, `docs/MODELS.md`, `docs/REORG_PLAN.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:337-340` | `docs/MODELS.md`, `scripts/smap_rungs.py`, `tests/test_smap_rungs_port.py` |
| `telemanom/errors.py:339` | `CHANGELOG.md`, `Objective.md`, `docs/DECISIONS.md`, `docs/MODELS.md`, `docs/NARRATIVE.md` |
| `telemanom/errors.py:342-343` | `CHANGELOG.md`, `docs/MODELS.md`, `scripts/smap_rungs.py`, `tests/test_smap_rungs_port.py` |
| `telemanom/errors.py:343` | `tests/test_smap_rungs_port.py` |
| `telemanom/errors.py:347` | `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:347-352` | `src/sentinel_models/telemanom.py` |
| `telemanom/errors.py:347-353` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:355-359` | `CHANGELOG.md`, `docs/DECISIONS.md`, `docs/MODELS.md`, `docs/NARRATIVE.md`, `scripts/smap_rungs.py`, `tests/test_smap_rungs_port.py` |
| `telemanom/errors.py:359` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:363-371` | `docs/MODELS.md`, `docs/NARRATIVE.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:365-371` | `docs/MODELS.md`, `docs/NARRATIVE.md` |
| `telemanom/errors.py:367-369` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:374-375` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:386-418` | `src/sentinel_models/telemanom.py` |
| `telemanom/errors.py:386-435` | `CHANGELOG.md`, `docs/MODELS.md`, `scripts/smap_rungs.py` |
| `telemanom/errors.py:403` | `docs/MODELS.md` |
| `telemanom/errors.py:403-418` | `docs/MODELS.md` |
| `telemanom/errors.py:404` | `docs/MODELS.md` |
| `telemanom/errors.py:404-405` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:405` | `docs/MODELS.md` |
| `telemanom/errors.py:408-414` | `docs/MODELS.md` |
| `telemanom/errors.py:411` | `scripts/smap_rungs.py` |
| `telemanom/errors.py:411-412` | `docs/MODELS.md` |
| `telemanom/errors.py:414` | `CHANGELOG.md` |
| `telemanom/errors.py:417-418` | `docs/MODELS.md` |
| `telemanom/errors.py:427-435` | `scripts/smap_rungs.py` |
| `telemanom/modeling.py:72-75` | `docs/MODELS.md` |
| `telemanom/modeling.py:83,88` | `docs/MODELS.md` |
| `telemanom/modeling.py:99` | `docs/MODELS.md`, `src/sentinel_models/lstm.py` |
| `telemanom/modeling.py:113` | `CHANGELOG.md`, `docs/MODELS.md`, `src/sentinel_models/windows.py` |
| `telemanom/modeling.py:133-136` | `src/sentinel_models/windows.py`, `tests/test_published_training.py` |

**Not cited anywhere, and vendored anyway**: `telemanom/helpers.py`,
`telemanom/plotting.py`, `telemanom/__init__.py`, `README.md`,
`requirements.txt`. They are in the tree because it is a copy of a commit rather
than a selection from one, and a selection is the thing this project decided not
to keep.

**Not vendored, each for a reason** (`third_party/telemanom/PROVENANCE.md`):
`labeled_anomalies.csv` is data and lives in R2 with its sha256 pinned;
`result-viewer.ipynb` is 2.0 MB and would consume half the 4 MiB repository
budget `tests/test_no_local_persistence.py` enforces; `results/` and `data/` are
outputs and data; `Dockerfile` and `example.py` are packaging rather than method.
