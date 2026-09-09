# The F' toolchain, and how to rebuild it

**F' is pinned at v4.3.0** (`docs/DECISIONS.md` D31), commit
`7d8f579f159d2f7c2d4984d92828575e37f87fa6`, released 2026-08-20. This document
records the toolchain work item 9 was built and verified against, and how to
reproduce it from nothing. It is a development-machine record: nothing here is
spacecraft-specific, and no part of it flies.

## 1. One command

```bash
bash scripts/fprime_setup.sh
```

Idempotent, and it prints every version it found or installed. It clones the
framework, creates the tool virtualenv from **F's own `requirements.txt`**, and
reports whether `clang-tidy` is present.

To use the toolchain afterwards:

```bash
. fprime/fprime-venv/bin/activate
cd fprime
```

## 2. What it creates, and why none of it is committed

| Path | What | Size |
|---|---|---|
| `fprime/lib/fprime/` | the `nasa/fprime` v4.3.0 checkout, shallow, with submodules | 69 MB |
| `fprime/fprime-venv/` | the tool virtualenv | 354 MB |
| `fprime/build-fprime-automatic-*/`, `fprime/build-artifacts/` | build caches | varies |

All four are gitignored. Only `fprime/`'s own sources -- the component, the
deployment and their build files -- are tracked.

**Why it lives inside the repository.** Everything for this project stays inside
this directory. That collides with `tests/test_no_local_persistence.py`, which
enforces Rule 1 by asserting the **tracked** set stays under 4 MiB, so the two
subtrees above are exempted there with a stated reason and a second test asserts
the exemption stays narrow. The checkout was scanned before the exemption was
written and holds **0/0** files with a dataset or array suffix. `conftest.py`
separately keeps them out of pytest collection, because the vendored googletest
ships twelve pytest-shaped files of its own that fail to import.

**Submodules matter.** The clone is `--recurse-submodules`: googletest is one,
and F's unit-test build needs it.

## 3. Versions, as measured 2026-09-01

```
  fprime               v4.3.0 (7d8f579f159d2f7c2d4984d92828575e37f87fa6)
  python               3.14.6
  cmake                3.26.0
  ninja                1.11.1.git.kitware.jobserver-1
  fpp                  v3.3.0
  fprime-tools         4.3.0
  fprime-gds           4.3.0
  fprime-fpp           3.3.0
  fprime-fpy           0.5.1
  fprime-visual        1.0.2
  fprime-fpl-layout    1.0.4
  fprime-fpl-write-pic 1.0.4
  clang++              Apple clang version 21.0.0 (clang-2100.1.1.101)
  clang-tidy           Homebrew LLVM version 23.1.0
  host                 macOS 26.6.1, arm64, 10 cores, 16 GiB
```

**There is no `fprime/requirements.txt` and there deliberately is not one.** The
setup script installs from `lib/fprime/requirements.txt`, which is F's own pin.
A second copy in this repository would drift from it silently, which is the
failure D16 exists to prevent.

**`cmake` comes from the virtualenv, not from Homebrew.** F' pins `cmake==3.26.0`
and `ninja` as pip packages, so activating `fprime-venv` puts the versions F'
tests against ahead of anything else on `PATH`. This machine also has Homebrew's
`cmake` 4.4.3, which is **not** what the build uses, and the difference is not
cosmetic: CMake 4 removed compatibility with `cmake_minimum_required` below 3.5
and warns below 3.10.

**`clang-tidy` is not shipped by F'.** `docs/DECISIONS.md` D31 consequence 5 said
it was deferred "to work item 9, with the F' toolchain", which was right about
the timing and wrong about the source. It comes from `brew install llvm`, which
is what `flight/.clang-tidy` said all along. Recorded as correction 10 in
`docs/MODELS.md` 20.2.

## 4. The toolchain proof

F's own reference deployment, at its v4.3.0 location, built from this toolchain
before any Sentinel code was written:

```bash
cd fprime/lib/fprime/TestDeploymentsProject/Ref
fprime-util generate
fprime-util build -j 8
```

| Step | Wall clock |
|---|---|
| `fprime-util generate` | 6.4 s |
| `fprime-util build -j 8` | 6.0 s |
| **total** | **12.4 s** |

Producing `build-artifacts/Darwin/Ref/bin/Ref`, 2,352,560 bytes, from 400 objects
and 113 static libraries.

**This falsified a pre-registered prediction.** `docs/MODELS.md` 20.5 C9 predicted
10 to 20 minutes. The reasoning was that a framework build means thousands of
translation units; in fact F' builds only the modules the topology references --
400, not thousands -- and clang on this host compiles them in 21 CPU-seconds. The
prediction is recorded as **Wrong** in 20.9 rather than revised, which is what
section 19.5's rule requires.

## 5. Where `Ref` went

`Ref/` is **not at the root of `nasa/fprime` at v4.3.0**. It is
`TestDeploymentsProject/Ref/`, a project of its own with its own `settings.ini`
that consumes the framework through `find_package(FPrime)`. It was present at
v4.2.2 and absent at v4.3.0, and the move is not in the v4.3.0 breaking-change
notes; `docs/user-manual/design-patterns/rate-group.md` still links Ref's
topology through a pinned older commit. Recorded as correction 3 in
`docs/MODELS.md` 20.2, and it is why "done when it builds in an F' Ref
deployment" needed a reading rather than a path.
