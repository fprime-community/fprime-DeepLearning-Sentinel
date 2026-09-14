# `scripts/`

Thirty-six files: thirty-three Python and three shell. Every one of them either
**produces a figure some document cites**, **generates something committed**, or
**guards something**. Nothing here is a library -- `src/` is where importable code
lives, and a script that grew a function worth importing should move it there.

**Read this before running anything that touches the bucket.** The operations
ceiling is 50,000 Class A and 50,000 Class B per calendar month with a tripwire at
1,000 per run (`docs/DATA.md` 4, `src/sentinel_data/config.py:95-96`), every
operation is counted through the before-send hook in `src/sentinel_data/r2.py`,
and the rule is **state the cost before spending it and smoke first**. The smokes
have caught a real defect every single time, including a projection that was one
operation light -- `docs/MODELS.md` 39.13.4.

## The conventions every script here follows

- **Root is `Path(__file__).resolve().parents[1]`.** Twenty-eight of the
  thirty-three Python files compute it that way, which is why `scripts/` is flat:
  a subdirectory would move `parents[1]` under every one of them.
- **The artifact is written before the ledger is committed** (commit `75cc846`),
  and the ledger commit sits in its own `try`/`except`, so a ledger failure
  reports loudly instead of destroying a result that was paid for.
- **The weight store is asserted unmoved** on any run that must not fit:
  `before == after` on `runs/_weights`, printed and stored in the artifact.
- **Pre-registration precedes measurement.** A script that scores an arm names the
  `docs/MODELS.md` section that registered it, in its docstring, and that section
  was committed first.

## Guards -- these fail the build

| Script | What it refuses |
|---|---|
| `check_no_list.py` | LIST-class access to R2 at the source level: LIST calls, `s3fs`/`fsspec` imports, `.glob()` in a module that imports the R2 client, and any read of `docs/manifest.snapshot.json`. Pinned by `tests/test_no_list.py` |
| `check_references.py` | A citation that does not resolve -- sections, repository paths, `file:line` ranges, relative links and pytest node ids. Carries D67's `--master` mode, where an off-branch path resolves on `dev` rather than breaking. Pinned by `tests/test_references_resolve.py` |
| `telemanom_citations.py` | Drift between `docs/TELEMANOM_EXCERPTS.md` section 5 and the citations actually in the tree. Regenerates the index; pinned by `tests/test_telemanom_index.py` |

## Generators -- these write something committed

| Script | What it generates | Held to |
|---|---|---|
| `make_golden_vectors.py` | `flight/test/vectors/g*.vec` and `g*.bin` -- the C++ core against `src/sentinel_models/reference.py` | 1e-5, exact on the flags (`docs/MODELS.md` 19.8 F5: worst 1.788e-07) |
| `make_baseline_vectors.py` | `b1`-`b4.bvec` -- Level 1 against `baseline_reference.py` | 1e-5. Tier `b3` is the N(1000, 3) regime D37's float32 defect destroyed |
| `make_trailing_vectors.py` | `t1`-`t3.tvec` -- `TrailingWindow` against `trailing_stats` | 1e-5 (worst 3.738e-10) |
| `make_threshold_vectors.py` | `d1`-`d2.dvec` -- `DynamicThreshold` against `flight_reference` | 1e-5 on `eps`, **exact** on the emission flag |
| `make_derivative_vectors.py` | `f1`-`f2.fvec` -- `DerivativeStream` against `decision_layer_arms.py:224` | 1e-5 (worst 2.899e-07) |
| `make_fused_vectors.py` | `p1.pvec` and `p1.bin` -- **D68's flight configuration end to end**, a real `param_version` 2 model loaded and stepped | 1e-5 on the score, **exact** on both flags |

## Studies -- these produce figures documents cite

Each names its pre-registration. **A figure with no artifact behind it is not a
figure this project reports.**

| Script | Registered at | What it measured |
|---|---|---|
| `smap_visibility.py` | `docs/MODELS.md` 26.1-26.6 | Is the labelled contextual class actually in range? D46's 39 of 43 |
| `smap_stage2.py` | 26.7-26.12 | Stage 2 of the SMAP/MSL ladder |
| `smap_rungs.py` | 27 | The faithful port of published telemanom, rung by rung (D54) |
| `smap_forensics_38.py` | 36 | Why the frozen arm misses 28 of the 38. **Diagnosis only** |
| `decision_layer_probe.py` | 37 | The pruning ladder, the alarm rate beside every figure, lead time (A1, A5, A6) |
| `decision_layer_arms.py` | 38 | Seven arms and a stride. Arm 2 reached **EVAL 17 of 19** (D65) |
| `decision_layer_departures.py` | 39, N5 and N6 | What the port's two departures cost (39.13) |
| `decision_grid.py` | 10 | The decision layer swept as a grid rather than chosen |
| `floor_audit.py` | 22 | Work item 9.6: is the corrected floor real? |
| `reduction_and_curve.py` | 23, 24 | Work items 9.7 and 9.8 |
| `precursor_test.py` | 25 | Are the forecaster's nominal-period alarms precursors? |
| `oscfar_curve.py` | 10.8 | The admission-rate curve for the order-statistic rule |
| `threshold_diagnostics.py` | `docs/THRESHOLD.md` | What telemanom's z-selection is actually choosing, over 5.68M windows |
| `threshold_sweep.py` | -- | The threshold swept against the residuals of the model it will run on |
| `whitened_pruning.py` | -- | Does pruning explain what `lstm-whitened` loses? |
| `whitened_discriminator.py` | -- | What separates the events it loses from the ones it keeps |
| `head_to_head.py` | -- | Which events each of two detectors catches, and which only one |
| `event_forensics.py` | -- | Which events each decision rule catches, event by event |
| `envelope_proxy.py` | -- | Would a limit check have seen it at all? A proxy that errs against us |
| `combination_scope.py` | 17 | The two-forecaster combination, scoped on cached weights |

## Operations and fitting

| Script | Cost | Note |
|---|---|---|
| `ingest_smap_msl.py` | Class A, one ingest | Verifies every array against the canonical labels **before** a byte is uploaded: 82 of 82 agreed, 0 mismatches, and a single failure would have aborted it |
| `bench_forecaster.py` | zero | Measures what a run will cost **before** it is spent |
| `roundtrip_check.py` | a few Class B | manifest -> key -> object -> DataFrame, proven rather than assumed |
| `fit_folds.py` | zero bucket, GPU hours | Fits and banks weights. No scoring |
| `reseed_fold.py` | zero bucket | Refits one fold on another seed |

## Shell

| Script | What it does |
|---|---|
| `fprime_setup.sh` | Rebuilds the work item 9 F' toolchain from nothing, pinned at v4.3.0 (D31). The checkout is gitignored; this is how it comes back |
| `fprime_ref_patch.sh` | Builds Sentinel inside F's own reference deployment |
| `pod_setup.sh` | One-shot setup for a rented GPU box. **No secret is persisted by it** |

## What is deliberately absent

- **No script deletes anything.** A script behind any documented figure is
  ARCHIVE-and-cite at worst (`Objective.md` 11's never-delete rule).
- **No script writes to `master`**, or pushes. (`main` was retired 2026-09-14, D69.2.)
- **No script reads `docs/manifest.snapshot.json`** -- it is a generated copy for
  humans, and `check_no_list.py` rule 4 refuses any code that reads it.
