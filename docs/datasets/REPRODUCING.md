# Reproducing this without the bucket

> **Paths outside this branch resolve on `dev`.** `master` carries the component and the
> evidence it works, and nothing else (`docs/DECISIONS.md` D67). A citation here into
> `src/`, `scripts/`, `tests/`, `docs/MODELS.md` or `third_party/` points into the
> development branch at the commit this snapshot was taken from.
> `scripts/check_references.py --master` is what keeps that true rather than hoped.

**No dataset, no credential, no cloud account, and no bucket.** A surprising amount of this
repository checks itself, and this page is the honest boundary between what you can verify
and what you have to take on citation. Everything in the first list was **run to produce
this page**, on 2026-09-11.

## What you can recompute, at zero cost

```
git clone <this repository>
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

| Command | What it proves | Measured |
|---|---|---|
| `.venv/bin/python -m pytest -q` | The whole suite, **zero R2 operations by construction** -- `tests/test_ops_guard.py` fails if a test would spend one | **With the F' and OxCaml build trees: 799 passed, 3 skipped**, measured 2026-09-22. Without them, **794 passed, 8 skipped** -- each skip names the script that would satisfy it. **There is no state in which nothing skips, and the entry here claimed "795 passed" once they are built.** Three tests skip BECAUSE the checkout is present and say so: `test_flight_lint_reports_partial_honestly.py` (a full run is expected once clang-tidy has all three configs) and two in `test_references_resolve.py` (citations into `fprime/lib/` are checked rather than announced) |
| `.venv/bin/python scripts/check_no_list.py` | No source file can LIST the bucket, glob it, or read the manifest snapshot | **117 files clean** |
| `.venv/bin/python scripts/check_references.py` | Every section citation, repository path, `file:line` range, relative link and pytest node id resolves | 775 / 742 / 512 / 39 / 4, all resolving, plus **1 citation into the absent `fprime/lib/` announced as SKIPPED** rather than broken (D79) |
| `PYTHONPATH=src .venv/bin/python -m sentinel_eval selftest` | The referee scores a known-answer oracle correctly and a silent detector at zero | **8/8, oracle 1.0** |
| `make -C flight test` | The C++ core against its committed golden vectors | green: 7 categories, worst eps 5.072e-06 against a 1e-05 tolerance |
| `make -C flight lint` | clang-tidy, three configurations, at `-Werror` | `clean (3 of 3)` with the F' checkout present; **`PARTIAL (1 of 3)` and non-zero** without it, which is a fresh clone's normal state -- `LINT_ALLOW_PARTIAL=1` accepts it (D79) |
| `PYTHONPATH=src .venv/bin/python -m sentinel_eval run synthetic --detector gru-smoke --detector rstd --no-sweep` | A **complete scoring run** -- fit, score, every metric, `k/n` throughout -- on a generated fixture | writes `runs/synthetic/...`, **zero operations** |

**The golden vectors are in the repository**, which is the part people do not expect. A
fresh clone gets `g1` and `g2` complete (vector **and** model file), `g3`'s vector,
the four baseline tiers `b1`-`b4`, the three trailing tiers `t1`-`t3`, the two
threshold tiers `d1`-`d2`, the two derivative tiers `f1`-`f2`, and **`p1` and `p2`, D68's
flight configuration end to end** -- a real `model.bin` at `param_version` 2 with 3,200
steps of expected output. Every generator that made them is committed too, and
`tests/test_golden_vectors.py` regenerates each from its seed and compares **byte for
byte**.

**And all of `third_party/telemanom/`**, byte-identical at commit `2e6c5b6c`, with
**78 citations into it from 13 files** -- every one resolvable by opening the file at the
line. `docs/TELEMANOM_EXCERPTS.md` indexes them and `scripts/telemanom_citations.py`
regenerates that index, so a claim about the published method can be checked against the
published method without leaving the clone.

**And `docs/manifest.snapshot.json`**, a generated copy of the bucket manifest for reading
by humans. **No code reads it** -- `check_no_list.py` rule 4 refuses any that would -- so
it cannot become a second source of truth.

## (!) The retrainer builds from this branch alone, and that is now executed rather than read

**This was the largest unproven claim `master` made.** That `master` alone can build the
retraining engine was established by **reading** `fprime/library.cmake`, the per-module
`CMakeLists.txt` files and the two setup scripts -- never by running them from a clean
clone. D80 moved `oxcaml/` here; nothing had since checked that what arrived was
sufficient.

**Executed 2026-09-23, from a fresh `git clone --branch master --single-branch`:**

```
  scripts/fprime_setup.sh     rc=0    F' v4.3.0 checkout + tool virtualenv
  scripts/oxcaml_setup.sh     rc=0    switch 5.2.0+ox
  scripts/oxcaml_s61.sh       rc=0    cycle_complete.o, the object the deployment links
```

238 tracked files, and **nothing outside the clone was needed**.

**(!) And the object is code-identical to the one built in the development tree**, which
is the part worth checking rather than assuming:

```
  symbols        4,816  =  4,816          __text     395,632  =  395,632
  OCaml symbols  3,083  =  3,083          __cstring   13,940  =   13,940
```

The two files differ by **6,152 B** in total size and in nothing else. Both are built
with `-g`, both embed their own build path in the debug info, and the clone's path is
**83 characters longer**. The code sections are identical; the difference is DWARF.

**Cost, measured:** the OxCaml switch dominates at **2.7 GiB**, with F's checkout at
70 MB and the tool virtualenv at 354 MB. All three are gitignored and rebuilt by the
scripts, which is why none of them is committed.

**What this does NOT establish.** That any of it builds for a **flight target** -- that
is C4, and it is still unverified. That it runs on flight hardware -- E4, never run. The
clone was built on the same host and the same architecture as the development tree, so
this proves sufficiency of the branch, not portability of the toolchain.

## (!) What you cannot recompute, and it is most of the numbers

**Any figure scored on ESA-ADB or SMAP/MSL.** The telemetry is not here and never will be
(`Objective.md` 11): no dataset, no parquet, no cached array is committed, and
`tests/test_no_local_persistence.py` refuses those formats **everywhere** including
`runs/`, gitignored files included. That is a deliberate trade -- the alternative is a
repository that carries gigabytes of somebody else's data under somebody else's licence.

What stands in its place:

- **Every figure is cited to an artifact by path.** `runs/` is gitignored, so those paths
  resolve on the machine that produced them and not in your clone. They are named so the
  claim is falsifiable by someone with the data, not so it is checkable by everyone.
- **Every measurement has a pre-registration committed before it ran**, with numbered
  predictions and HOLD / NO VERDICT / FAIL bands. `docs/MODELS.md` carries all of them
  **beside their outcomes, losers included** -- which is the part that is checkable here.
- **The operations ledger.** Every bucket read is counted and the month-to-date total is
  printed into each artifact and quoted in `docs/STATUS.md`.

## Getting the data yourself

`ESA_ADB.md` and `SMAP_MSL.md` carry the DOIs, the record identifiers, the licences and
the caveats. **Both licences are now verified at their sources**: ESA-ADB's **CC BY 3.0
IGO** was read at the Zenodo record on 2026-09-11 (record `15237121`, version v2, published
2025-04-17), and SMAP/MSL's **BSD 3-Clause** by reading `third_party/telemanom/LICENSE.txt`
directly. SMAP/MSL's **clause 3 obligation** binds anything published from it and is quoted
verbatim in `SMAP_MSL.md`.

`scripts/ingest_smap_msl.py` is the ingest that produced this project's copy, and it
verifies every array against the canonical labels before uploading a byte -- **82 of 82 rows
agreed, 0 mismatches**. It is the thing to read if you want to stage the data the same way.
