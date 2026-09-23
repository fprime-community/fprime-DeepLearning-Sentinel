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
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
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

## `master` builds from scratch, and every Quick start block was run verbatim to prove it

**This was the largest unproven claim `master` made**: that the branch alone can build
the retraining engine was established by **reading** the CMake files and the setup
scripts, never by executing them. It is executed now.

**2026-09-23, fresh `git clone --branch master --single-branch` into an ordinary user
directory, 238 tracked files, every Quick start block run VERBATIM and in order:**

```
  make -C flight test                                   rc=0
  LINT_ALLOW_PARTIAL=1 make -C flight lint               rc=0   PARTIAL, as documented
  python3 -m venv .venv && pip install -r requirements-toolkit.txt   rc=0
  sentinel_toolkit selftest                              rc=0   8/8
  scripts/fprime_setup.sh                                rc=0   F' v4.3.0
  fprime-util generate -f && build -p ./SentinelRef      rc=0
  scripts/oxcaml_setup.sh                                rc=0   switch 5.2.0+ox
  bash scripts/oxcaml_s61.sh                             rc=0   cycle_complete.o
  fprime-util generate -f && build -p ./SentinelRetrain  rc=0
```

**Both deployments built and linked**, which is what the earlier attempt could not show:

```
  SentinelRef       2,092,392 B      0 OCaml symbols   -- D70 c.2 holds
  SentinelRetrain   1,409,760 B  3,082 OCaml symbols
  Sentinel_Monitor_ut_exe          15 of 15 tests pass
  SentinelRef_Retrainer_ut_exe      1 of 1 test passes
```

The symbol counts were taken against **the clone's own binaries**, not the development
tree's -- and doing so is what found the recorded 3,019 to be a pre-D82 figure
(`docs/MODELS.md` 72.11).

**(!) TWO README DEFECTS WERE FOUND BY DOING THIS, AND BOTH WOULD HAVE STOPPED A
STRANGER.** The Quick start said `python -m venv`, and there is no `python` on macOS or
most current Linux -- a first-time reader failed on their first command of the second
block. And the retrainer block built `SentinelRetrain` without regenerating the build
cache, so the deployment -- registered only if the OxCaml object exists at generate time
-- was silently absent and `fprime-util build` printed `ninja: no work to do` and
**exited 0 with no binary**. Both are fixed, both are guarded by
`tests/test_documented_commands_are_runnable.py`, and the run above is from a clone made
after the fixes, so every block worked **first time**.

**What this does NOT establish.** That any of it builds for a **flight target** (C4,
unverified) or runs on flight hardware (E4, never run). Same host, same architecture: it
proves the branch is sufficient, not that the toolchain is portable.

**(!) And one location does not work, for reasons outside this repository.** A clone
under `/tmp` fails: F' v4.3.0's `lib/fprime/cmake/settings/ini.cmake:37-39` compares two
path strings with `STREQUAL`, and macOS's `/tmp` is a symlink to `private/tmp`, so the
same file reaches CMake under two spellings. Clearing that one reveals a second, deeper
failure -- `Platform/Darwin.cmake:8` calls `FIND_PACKAGE(Threads)` inside a platform
file, which re-enters through every nested `try_compile` until CMake's recursion limit.
Both are upstream, D31 pins v4.3.0, and neither is patched here. **Clone anywhere
ordinary and it does not arise**: `$HOME` works, and so does a deliberately symlinked
path under `$HOME`, so the trigger is the two spellings reaching CMake by different
routes rather than symlinks as such.

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
