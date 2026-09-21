# fprime-DeepLearning-Sentinel

> **This branch is the product. Paths outside it resolve on `dev`.**
> It carries the F' flight component and the evidence that it works, and nothing else
> (`docs/DECISIONS.md` D69, on `dev`). A citation into `src/`, `scripts/`, `tests/`,
> `docs/MODELS.md` or `third_party/` points into `dev` at commit **`ba06258`**.
> **The guards that keep these figures true run on `dev`, not here** -- they are
> `tests/test_master_documents_are_current.py`, which re-derives every figure this
> branch states about `dev`, and `scripts/check_references.py --master`. You cannot run
> them from this branch, and that is stated rather than implied.

**Sentinel** is a reusable NASA F' (F Prime) flight-software component that warns of
spacecraft anomalies **that never cross a limit line**. A GRU forecaster, trained on a
mission's own healthy telemetry, predicts each watched channel every cycle. The prediction
residual and the channel's first derivative, each standardised against that channel's own
trailing window, drive a warning event naming the channel.

**It warns only.** `Monitor.fpp` declares no command of its own.

The detection method is JPL's -- Hundman et al., KDD 2018, *telemanom*. What this project
adds is the flight packaging, the evidence base, and one finding: telemanom's published
false-alarm filter deletes true detections on the in-range population, and the first
derivative recovers them. `docs/EVIDENCE.md` is that argument with its numbers and its
caveats.

## (!) What this is not, and what you cannot do with it yet

- **The ground training toolkit is not released, and a mission cannot deploy this without
  it.** The component *runs* a model; it does not *produce* one. Turning healthy telemetry
  into a `model.bin` is the toolkit's job, the toolkit is `src/`, and `src/` is not on this
  branch. **3 of its 3 acceptance-ladder rungs have run on `dev`**, the last of them on a real
  mission (`docs/STATUS.md`), so it exists and is not yours yet. **What is here is the flight half
  of a two-half product.** `docs/MODEL_FILE.md` specifies the file completely enough to
  write one independently, which is the honest answer available today.
- **No early-warning claim is made.** Not "warns N minutes before", not "~4 hours", not any
  wall-clock figure. On real telemetry the rule this branch ships is **less late than the rule
  it replaced, and not early** (`docs/MODELS.md` 45.6); the frozen layer it replaced measured
  **0 of 10 positive leads** (37.7a). A warning time **has** been measured against a real limit
  on a **simulated** plant -- `docs/EVIDENCE.md` section 5a -- and that is a claim about a
  testbed and a fault this project designed, not about a mission. Every figure here is in
  **timesteps**, never in hours.
- **The evidence is one arm on one dataset, UNDERPOWERED.** n = 19 per half. Read
  `docs/EVIDENCE.md` before quoting anything from it.

## Build and verify

**Everything on this branch builds and verifies with a C++ toolchain and nothing else.**
No Python, no credentials, no dataset, no network.

```bash
make -C flight test      # the core against its committed vectors
make -C flight lint      # clang-tidy at -Werror
```

**What the green output proves here, measured on this branch:**

```
  footprint             sizeof(Detector) asserted exactly, 603,032 B
  refusals              18 load cases, exercising all 12 refusal codes plus the
                        accept path; CRC check value 0xCBF43926
  determinism           bit-identical in-process and across two processes, over
                        a 6,400-tick run that wraps the window twice
  golden vectors        2 tiers: g1 at 3 channels, g2 at 7
  baseline vectors      4 tiers, max |diff| 0.000e+00
  trailing window       3 tiers, worst 3.738e-10
  dynamic threshold     2 tiers, worst eps 5.072e-06
  derivative stream     2 tiers, worst 2.899e-07
  flight configuration  2 tiers, both 3,200 steps: p1 at 3 channels and p2 at 1,
                        fused score matched to 3.098e-06, 85 of 85 emissions
                        exact on each
  round trip            2 committed model.bin files re-emitted byte-identically
```

Tolerance is **1e-05** throughout.

**(!) And what it does not prove here.** The forward pass has **seven** golden tiers and
**two** run on this branch. The 12-channel tier `g3` and the four production-shaped `g4_*`
tiers need weight files that are **deliberately committed nowhere** -- they regenerate
exactly from a seed, and the generator lives in `scripts/`, which is not on this branch.
`g3.vec` is here and its input is not, so that tier **skips silently**. You should know
that rather than read seven where two ran. Everything the C++ port and D68 added does run
here at full width.

**`make -C flight lint` can fail, and could not until 2026-09-11.** The recipe ran
clang-tidy, checked no exit status, and printed `lint: clean` unconditionally. It was
reporting nine errors at the time, three of them in flight code. All nine are fixed and the
recipe now carries `set -e`. Recorded because a gate you cannot watch fail is a gate you
should not trust.

**(!) On a fresh clone, lint runs one configuration, not three.** The target runs
`flight/.clang-tidy` always, and F's own two configurations **only if the F' checkout is
present** -- `fprime/lib/fprime/` is gitignored and rebuilt by a script that is not on this
branch, so a clone of this branch alone prints `framework checkout absent; skipping its two
configs`. `docs/FPRIME.md` rebuilds the checkout with one command, and then all three run.

**To build the F' component** you need the F' v4.3.0 toolchain; `docs/FPRIME.md` pins it and
rebuilds it from nothing with one command.

## (!) `fprime/SentinelRef/Retrainer/` is an experiment, and nothing here depends on it

This branch gained an F' component on 2026-09-15 that calls into a library written in
**OxCaml**, Jane Street's branch of OCaml. It is named here rather than left to be found,
because a curated branch that quietly acquires a garbage-collected runtime would be worse
than an uncurated one.

**What it is.** The first of four experiments asking whether a language whose compiler can
*prove* a code path performs no allocation is a candidate for onboard model retraining. It
computes a sum and a mean and contains **no machine learning**. It exists to prove a chain --
compile, link, runtime startup, the C boundary, F' integration -- and it proved it.

**What it is not.** Not adopted, not flown, not on any path the detector takes, and not
exported: `fprime/library.cmake` exports `Sentinel/Monitor` and nothing else, so a mission
adopting Sentinel does not inherit an OCaml runtime. The experiment that would decide whether
the approach survives at all has **not been run**.

**And it does not change what you can build here.** `make -C flight test` and
`make -C flight lint` are untouched and need no new tool. The module **skips itself with a
message** if the OxCaml toolchain is absent, which on a fresh clone of this branch it always
is, so the F' build is unaffected. Every claim about it, and the decision to try it at all,
lives on `dev` in `docs/MODELS.md` 47 and `docs/DECISIONS.md` D70.

## What is here

| Path | What it holds |
|---|---|
| `flight/` | **The C++ inference core.** GRU forward pass, telemanom's dynamic threshold, the trailing-standardised derivative stream, the `model.bin` reader, and the committed vectors. C++14, no exceptions, no RTTI, no STL containers, **no allocation after init**, `-Werror` |
| `fprime/Sentinel/Monitor/` | **The F' component.** `Monitor.fpp`, its SDD and its unit tests |
| `fprime/SentinelRef/` | The reference deployment that instantiates it, the topology, and **the physics testbed**: a simulated coupled power and thermal subsystem with declared limits, wired to the component's input port. Apparatus, not product -- `library.cmake` exports the Monitor and not this. **It also carries `Retrainer/`, an experiment and not a feature** -- see the note below -- and `ExampleAdapter/`, the forty-line Passive Adapter Pattern example a mission copies to wire its own channels in (`docs/DESIGN.md` 8) |
| `docs/DESIGN.md` | What the component does, the rule it flies, and the five permanent safety rules |
| `docs/EVIDENCE.md` | The result, the split, the alarm rate, the reproduction, and the caveats |
| `docs/STATUS.md` | Where it is and what is next |
| `docs/MODEL_FILE.md` | **Normative** for the loader. Where it and any implementation disagree, the document is right and the implementation is a defect |
| `docs/FPRIME.md` | The F' v4.3.0 toolchain this is built against |
| `docs/PI_ENVELOPE.md` | Whether it runs on small hardware -- **reserved, and deliberately empty** |
| `docs/datasets/` | What the data is, its licences, and the caveats that would corrupt a result |

**Read in this order:** this file, then `docs/EVIDENCE.md`, then `docs/DESIGN.md`, then
`docs/STATUS.md`. **To adopt it into your own
deployment, `docs/DESIGN.md` 8 is the recipe** -- four edits, no copied sources, and two committed `model.bin`
files at `flight/test/vectors/p1.bin` and `p2.bin` you can load today. About fifteen minutes to the point where you can decide whether to keep
reading.

## (!) What is not on this branch, and why

**Every omission is named.** A curated branch that quietly drops things is worse than an
uncurated one, because a reader cannot tell what they are not seeing.

| Absent | What it is | Why |
|---|---|---|
| `src/` | The ground toolkit: ingest, referee, models, the `model.bin` writer, and the toolkit itself, whose **3 of its 3 acceptance-ladder rungs have run** on `dev` | **Not released.** See the warning above |
| `scripts/` | The guards, the vector generators, every study that produced a figure | Development apparatus, not product |
| `tests/` | The Python suite, which runs against `src/` | Runs against `src/`, which is not here |
| `docs/MODELS.md` | Every pre-registration beside its outcome | The research record. Cited from here, resolves on `dev` |
| `docs/DECISIONS.md` | Every decision with its alternatives and the evidence that settled it | The research record. **D69 removed it from this branch and it is kept whole on `dev`** -- `docs/EVIDENCE.md` cites the active entries by number |
| `docs/NARRATIVE.md` | What happened in order, mistakes included | Same. The retractions bearing on the result are in `docs/EVIDENCE.md` |
| `docs/HARNESS.md`, `docs/DATA.md`, `docs/RESEARCH.md`, `docs/THRESHOLD.md`, `docs/PHASE2.md`, `docs/PHASE5.md`, `docs/PHASE1_REPORT.md`, `docs/TELEMANOM_EXCERPTS.md`, `docs/REORG_PLAN.md`, `docs/INDEX.md` | The internal documents | Same |
| `docs/RESULTS.md` | Every scored result, every sweep, and the tables the headline is read from | The research record. `docs/EVIDENCE.md` carries the result of record and its caveats |
| `Objective.md` | What this project is for, its permanent rules and its four phase gates | The research record. `docs/DESIGN.md` states the rules that bind the component |
| `CHANGELOG.md` | Version by version | Development history; `dev` has it |
| `third_party/telemanom/` | The published source, vendored byte-identical at `2e6c5b6c` | Evidence for the research record. **This branch therefore does not redistribute it**, so BSD clauses 1 and 2 do not bind here -- clause 3 does, and is below |

## Branches

- **`dev`** carries the complete development history, decision by decision, with tags
  `wi1`-`wi9` and their Releases. All work lands there and **it is never rewritten**.
- **`master`** is this branch and **the repository default**: the component and the
  evidence it works, taking a snapshot when something is done. A visitor arriving at this
  repository lands here.
- **(!) `main` was retired on 2026-09-14.** It was an earlier snapshot branch. Every commit
  it held is reachable from this one -- `git rev-list master..main` was 0 before it went
  -- and all nine tags were and remain on `dev`, so no commit and no Release was lost.
  **A `/blob/main/...` link held outside this repository will now 404.** No such link
  exists inside it; one held elsewhere cannot be searched for, so that risk is stated rather
  than dismissed.
- **Nothing is ever force-pushed, and no branch is ever rewritten.**

**`dev` and this branch share no commit.** Different root commits, and `git merge-base`
between them is empty. That is why the guard on `dev` compares trees rather than walking
ancestry, and why every figure above names the `dev` commit it was derived from.

## Licence

**Not yet selected.** Intended for community release to the F' ecosystem. The repository is
private until then.

**(!) One obligation binds regardless of which licence is chosen.** The reference method is
JPL's -- Hundman et al., KDD 2018 -- and its source is BSD 3-Clause (Caltech/JPL 2018),
whose third clause reads, verbatim:

> *"Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory,
> nor the names of its contributors may be used to endorse or promote products derived from
> this software without specific prior written permission."*

**No document in this repository presents this project as endorsed by, affiliated with, or
produced by Caltech or the Jet Propulsion Laboratory.** Naming telemanom's authorship and
citing the paper is description, not endorsement.

ESA-ADB carries a separate attribution requirement -- **CC BY 3.0 IGO**, verified at the
Zenodo record on 2026-09-11 (record `15237121`, version v2, published 2025-04-17).
`docs/datasets/ESA_ADB.md` carries it.

**Author email.** Some commits on `dev` carry a personal email address in their authorship.
**No history is rewritten to remove it**, on this branch or any other: rewriting authorship
would invalidate every existing tag, Release and commit citation for a cosmetic gain. It is
disclosed here instead.

## Citation

- Hundman, K., Constantinou, V., Laporte, C., Colwell, I., Soderstrom, T. *Detecting
  Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding.* KDD 2018.
  arXiv:1802.04431. BSD 3-Clause.
- Kotowski, K. et al. *European Space Agency Benchmark for Anomaly Detection in Satellite
  Telemetry.* arXiv:2406.17826. Zenodo `15237121` v2, DOI 10.5281/zenodo.15237121,
  CC BY 3.0 IGO.
