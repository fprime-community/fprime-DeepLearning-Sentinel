# telemanom, vendored as evidence

**This is not a dependency. It is evidence.** Nothing under `src/`, `scripts/`,
`tests/` or `flight/` imports it, and nothing in it is ever executed here. It is
committed so that every claim this project makes about a divergence from the
published method cites a file in this repository rather than a memory of one.

The reason it exists: `docs/MODELS.md` 26.21 to 26.28 located four divergences by
quoting `errors.py` at specific line numbers, and no copy of `errors.py` was ever
kept. A later reader could not check the quotations, and a later author had to
re-derive them. `docs/NARRATIVE.md` records what that cost.

## Provenance

```
  upstream     https://github.com/khundman/telemanom
  commit       2e6c5b6c3558e7835601519b7bdef37c649bdbdc
  dated        2025-01-17 07:42:45 -0600
  subject      Merge pull request #88 from vc1492a/fix/data_source_documentation
  retrieved    2026-09-08, shallow clone, source only
  paper        Hundman, Constantinou, Laporte, Colwell, Soderstrom,
               "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
               Dynamic Thresholding", KDD 2018, arXiv:1802.04431
```

**(!) The licence is BSD 3-Clause, not Apache-2.0.** `LICENSE.txt` is a
three-clause BSD notice, "Copyright (c) 2018, California Institute of Technology
(Caltech). U.S. Government sponsorship acknowledged." `docs/DATA.md` records the
upstream as Apache-2.0 and that is wrong; the correction is carried in
`docs/DECISIONS.md` D53 and in `docs/DATA.md`'s SMAP/MSL block, which is
hand-written prose, and in `scripts/ingest_smap_msl.py:57` so any future ingest
is right.

**(!) Corrected here 2026-09-09.** This paragraph used to say the correction was
carried in `src/sentinel_data/docs_gen.py`, "which generates that line". It does
not: the generator has no telemanom licence line at all, and its only licence
interpolation is ESA-ADB's at `docs_gen.py:175`. The stored R2 manifest object
`_manifest/smap_msl.json` also still carries the Apache-2.0 string, as
`docs/DATA.md` records; rewriting it costs one Class A and has not been done.

**Two obligations follow, and both bind on public release.** The copyright notice
and the disclaimer are retained here verbatim, which satisfies clause 1. Clause 3
forbids using the name of Caltech or of the Jet Propulsion Laboratory to endorse
or promote anything derived from this software: **no document in this repository
may present this project as endorsed by, affiliated with, or produced by JPL.**
Naming telemanom's authorship and citing the paper is description, not
endorsement, and remains correct.

## What is vendored, and what deliberately is not

```
  telemanom/*.py    channel, detector, errors, helpers, modeling, plotting
  config.yaml       the published hyperparameters, as shipped
  requirements.txt  the library versions the published results were produced on
  README.md         upstream, unmodified
  LICENSE.txt       upstream, unmodified
```

Not vendored, and each for a reason:

```
  labeled_anomalies.csv   data. It lives in R2 under smap-msl/v1/ with its
                          sha256 pinned in _manifest/smap_msl.json (Rule 1)
  result-viewer.ipynb     2.0 MB of notebook, no evidentiary value, and it
                          would consume half the 4 MiB repository budget
                          tests/test_no_local_persistence.py enforces
  results/, data/         outputs and data, neither of which may be committed
  Dockerfile, example.py  packaging, not method
```

Every file here is byte-identical to the pinned commit. **If it is ever re-pinned,
the commit SHA above changes in the same commit as the files, and any quotation
whose line numbers move is re-checked.**

## What the guards do and do not cover

- `scripts/check_no_list.py` scans `src/` and `scripts/` only, so this subtree is
  outside it by construction. That is correct: the code is never executed, and a
  LIST call in a file nothing runs is not a risk. Stated here rather than left
  implicit.
- `tests/test_documents_are_current.py` scans every tracked `.md`, so
  `README.md` above is ASCII-checked and claim-checked like any other. It passes:
  measured **0** non-ASCII lines.
- `LICENSE.txt` is `.txt` and therefore outside that test's filter. It contains
  three non-ASCII lines, which are the bullet characters in the clause list. It
  is upstream's file and is not edited to suit our typography.
- `tests/test_layering.py` carries the rule that nothing under `src/` or
  `scripts/` may import this package.

## Where it is read

`docs/MODELS.md` 26.29 and 26.30 record what was read, with file and line,
against `src/sentinel_models/telemanom.py` line by line. (Corrected 2026-09-11
from `26.31`, which has never existed: the readings are 26.29.1 to 26.29.5 and
26.30.1 to 26.30.4. Found by `scripts/check_references.py` on its first run --
the one dangling section citation in the tree, and the reason the guard exists.) `docs/RESEARCH.md`'s telemanom
entry records that the source is now pinned here rather than recalled.
