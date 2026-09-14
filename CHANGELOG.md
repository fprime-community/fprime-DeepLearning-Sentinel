# Changelog

All notable changes to this project are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0 the version
tracks documentation and Phase 1 research milestones rather than a released flight component.

## [Unreleased]

### Planned - work item 10

- The in-orbit threshold recalibration path: file uplink, human-approved reload. The hook is
  noted in the component's FPP and SDD and no code implements it. It makes the component
  `queued` (D32 consequence 2).

## [0.6.47] - 2026-09-14 - The residual half of the flown rule decides nothing, on any of the 30

**`docs/MODELS.md` 44 run** (44.7, D65.3). **165 Class B and 1 Class A** after a **9 Class B**
smoke whose projection matched actual exactly, both times. Cached weights, **weight store
1,313 -> 1,313**, 77 of 81 channels, 67.5 s. Month after: **236 Class A and 5,566 Class B**.

### Both reproduction gates pass, and everything below rests on them
The frozen arm returns **0.6820% and 10 of 38**; the fused arm returns D65's own **13/19,
17/19, 30/38**. The restatement the ablation is measured against **is** what D65 measured.

```
  arm                          cut       rate       TUNE    EVAL    all 38
  frozen (stage 4)          0.550599   0.6820%      6/19    4/19    10/38
  fused: max(z_r, z_d)      5.288128   0.6820%     13/19   17/19    30/38
  derivative only           4.431455   0.6820%     15/19   17/19    32/38
```

### (!) z_residual reaches the cut on none of the thirty
At the step the fused rule fires: median **0.080**, maximum **4.652** against a cut of
**5.288**, **negative on 15 of 30**. `z_derivative` over the same thirty has a minimum of
5.403. **The residual term is not weak, it is inert** -- the `max()` takes the derivative every
time it fires, and on half the events the residual sits below its own trailing mean at the
moment the component warns. D65.1's two traces were typical after all.

### EVAL is identical, and the +2 is on TUNE
17 against 17. Derivative-only is a **strict superset on all 38**; its two extra events,
`P-1[3539]` and `T-8[870]`, are **both TUNE channels**. So the honest statement is *the
residual contributes nothing*, not *removing it helps* -- a distinction only 37.8's split
makes available.

### All five predictions held, and AB4 more strongly than its band asked
AB1 held at zero events apart; AB2 reported the dial (-0.856673) rather than absorbing it;
AB3 named the relation; AB4 asked for fewer than 2 of 19 and got **0 of 30**; AB5 held because
it was written to -- **nothing is adopted and D68 stands**.

### What it does not do
It frees no memory: `zResidual` reads the dynamic threshold's own window, which is carried
regardless, and 44.2 said so before the read. It is **one arm, one dataset, n = 19 per half,
UNDERPOWERED**. A `param_version` 3 rule dropping the residual term is **arguable after this
and is not argued here**.

## [0.6.46] - 2026-09-14 - The customer branch said the toolkit was unwritten, and the guard could not see it

**Two customer-facing statements on `master` claimed the ground toolkit is less finished than
it is**, both contradicted by **master's own** status document, which already said three of
three acceptance-ladder rungs have run. `docs/EVIDENCE.md` section 9 said *"pre-registered and
unwritten"*; `README.md`'s absent-paths table said *"at tiers 1 and 2"*.

**This is D69's drift mechanism arriving on the branch D69 created.** The guard built in answer
to D69.1 re-derives every figure from the source that produces it, and `_toolkit_rungs_built`
reads the `LADDER` tuple in `src/sentinel_toolkit/selftest.py`. It covered `README`'s and
`STATUS`'s claims and **not** these two, because neither stated a number for it to check.

### Both rewritten, and two `Figure` rows added so they cannot drift again
`tests/test_master_documents_are_current.py` grows `toolkit_rungs/EVIDENCE` and
`toolkit_rungs/README_absent`. **The guard's own comment had predicted this**: *"tiers 1 and 2
would have stopped matching the moment tier 3 ran, and a guard that stops matching is a guard
that has gone blind"* -- the exact phrasing it warned about, left standing one table away.
Both rows survive the existing mutation probe, which corrupts each figure in turn and requires
the guard to report that one and only that one.

### The shared documents move on both branches together
`docs/datasets/SMAP_MSL.md` carried the `UNDERPOWERED (D3)` miscitation twice and is
byte-identical across branches, so both were corrected in the same pass.
`docs/PI_ENVELOPE.md` follows `dev` too, and **still carries no figures**.

### And the snapshot commit moves
`master`'s four customer documents now name `dev` at the commit their citations resolve at,
which the guard checks against every shared path.

## [0.6.45] - 2026-09-14 - Four pre-registrations, none run, zero bucket operations

**`docs/MODELS.md` 41 to 44, written before anything is measured**, in the form every section
from 19 onward uses: numbered predictions with HOLD / NO VERDICT / FAIL bands, a stated
falsification, reporting rules and stop conditions. **Nothing is adopted, nothing is built and
no read is taken.** The operations ledger is unmoved.

### 41, the Raspberry-Pi envelope (`Objective.md` 12, Phase 4)
Nine predictions against a **named board**, with the rule that a different board re-derives the
absolute bands **as a rider before it is switched on**. Two are dimensionless so they survive
that change. `docs/PI_ENVELOPE.md` gains a pointer and **still carries no figures**.

- **(!) A fresh checkout runs 2 of the 7 golden tiers.** `g3.bin` and `g4_*` are gitignored with
  a stated reason and `GoldenVectors.cpp` skips an absent tier **silently**, so the ARM
  cross-platform proof would have run 3 and 7 channels and reported success. 39.7 already wrote
  the rule this trips. R2 now asserts the **tier count**, not only the tolerance.
- **Nothing exists at 16 channels**, the compile-time maximum a mission sizing a rate group
  needs. The fixture is generated on the target and not committed.
- **Tracked content is 6.07 MiB, inside 39's N8 middle band** before this adds anything. N8's
  row is a record and is not edited; the movement is reported beside it.
- The margin denominator is **1,000 ms**, read from the topology rather than assumed.

### 42, the F' Ref physics testbed (work item 11)
Seven predictions. `SentinelRef` already has the component instanced, passive, on a 1 Hz rate
group behind a real clock; **the gap is one unconnected port**, and the topology says so.

- **(!) F' evaluates telemetry limits on the ground, not onboard.** So a limit trip and a
  warning have different time bases, and **a warning time subtracted across two clocks is not
  reported at all**. The testbed evaluates the dictionary's own limit values against the
  onboard base, and the ground-to-onboard difference is reported as what it is.
- **(!) 40.3a's floor sets the run length**: 6,550 ticks, **1 h 49 min per healthy run at 1 Hz**,
  36 hours for a false-alarm figure over twenty. Raising the base clock is registered; a
  **simulated time source is refused**.
- This is the section that would make an early-warning claim permissible, **within its own
  bounds and never about spacecraft telemetry**, with 37.7a's 0 of 10 quoted beside it.

### 43, the stride reduction
37.7a recorded this and declined to register it. Registered now, in two halves so the
zero-operation claim is honest rather than deferred.

- The cost side is **arithmetic once 41's R6 lands**: `mean(S) = m(1 + (R-1)/S)`, worked at
  R6's three band boundaries.
- **(!) The lever is confounded with the statistic by construction.**
  `SOLVE_WINDOW = ERROR_WINDOW + STRIDE`, so a stride change moves the window, `MAX_SEQUENCES`
  and the footprint -- and therefore N3's equality and N8's band, both records. The separation
  is committed in advance, which is 39.13.3's lesson applied before rather than after.
- **The ceiling is stated**: even at stride 1 the median lead on this population cannot better
  about **+5.5 timesteps**, on n = 10. **Removing structural latency is not warning early.**
- **S5 needs 165 Class B and 1 Class A. Not taken.**

### 44, the derivative-only ablation
The adopted rule with `z_residual` removed, matched **0.6820%**, same 19/19 split.

- **It frees no memory, and saying otherwise would be wrong**: `zResidual` reads the dynamic
  threshold's own window, which 39.8 and D68 keep carried regardless. What it would settle is
  whether D65's headline is the fusion or **the derivative**.
- **Two-sided bands**, because 39's N6 was *"a one-sided worry written as a two-sided band"* and
  measured +13 in the direction it was not looking.
- D65.1's two traces raise the question and say in as many words that two events are not a
  measurement. **Needs 165 Class B and 1 Class A. Not taken.**

### Owed and deliberately unregistered
**D68.1** -- both dilations at one fixed cut -- and the repair of **arms 4a, 6 and 7** stay owed
and unregistered. Bundling them into 44's read would make one approval cover four questions.

## [0.6.44] - 2026-09-14 - "At or within", not "inside": D46.1's correction is carried out on `dev`

**D46.1 measured that 23 of the 39 touch a rail exactly and said the sites were a separate
edit. This is that edit.** `master` was migrated on the same day; every remaining live site
was on `dev`. Zero bucket operations. **No figure moves** -- 39, 38, 23, 16 and every recall
number are exactly as D46.1 left them.

### Seven live sites rewritten
`Objective.md` 1.1 and 9, `README.md` twice, `docs/STATUS.md` 2, `docs/RESULTS.md` 3 and
`docs/PHASE1_REPORT.md` 3. **Ten more are records and are quoted rather than corrected**,
named in D46.2.

### (!) The list of eleven the sweep was handed was wrong in six ways
Three entries were not sites at all, two were offset by a line, two carry the **inclusive**
wording and are right as they stand, and **two live sites were missing from it**. Every line
was read before it was touched, and the triage is D46.2's.

### The premise is stated once, where a reader meets the claim
`Objective.md` 1.1 now says that D43's premise -- a real RED or YELLOW limit sits **outside**
a channel's historical operating range -- is **load-bearing rather than incidental**.

### The retired-claim guard is not widened, and the measurement is reported instead
`CLAIMS` matches none of these phrases. Widening it would have put **ten record sites** into
the register to protect seven live corrections, which is the failure
`tests/test_documents_are_current.py` argues against at `:66-77`. Reported, not made.

### Four stale figures and citations repaired in the same pass
- **`docs/STATUS.md` 9's `check_no_list` figure was 87 and the live count is 102**, read from
  the run rather than transcribed. Nothing re-derives it, unlike the test count.
- **`docs/PI_ENVELOPE.md` cited `docs/STATUS.md` section 7**, which carries no such item on
  either branch. It is `Objective.md` section 12, Phase 4. The correction is kept as a note,
  because it is the drift that document exists to refuse arriving in the document itself.
- **The ESA-ADB author count was "twelve" in `docs/RESEARCH.md` and eleven in
  `docs/datasets/ESA_ADB.md`**, which enumerates them. **Re-verified at `arxiv.org` 2026-09-14:
  eleven**, and every other field of that record matches.
- **The UNDERPOWERED stamp cites D3, which is the F0.5 gate metric.** The `n < 20` rule is
  `docs/HARNESS.md` 1. Corrected at the two live sites, quoted at the record sites, and
  recorded as D65.2.

### And both branches move for the shared document
`docs/datasets/SMAP_MSL.md` carries the D3 miscitation twice and is byte-identical on both
branches. Corrected on both **on the owner's word**, in the same pass, which is what D69's
convention requires of a document neither branch owns alone.

## [0.6.43] - 2026-09-14 - `main` is retired and `master` is the front door

**D69 consequence 8's costing, executed on the owner's word** (D69.2). In its stated
order: the GitHub default moved to `master`, the live prose was rewritten, `origin/main`
was deleted, then local `main`.

### Safety, re-measured immediately before the deletion
- **`git rev-list --count master..main` was 0**, and the same for `origin/main`: every
  commit `main` held is reachable from `master`, so **no commit object was lost**.
- **All nine tags `wi1`-`wi9` were on `dev`**, none on `main`, so **no Release was
  orphaned**.

### Seven live sites rewritten, three kept as records
`dev`'s `README.md`, `docs/INDEX.md`, `docs/STATUS.md` (twice) and `scripts/README.md`;
`master`'s `README.md` and `docs/STATUS.md`. **`CHANGELOG.md` 0.6.41 and 0.6.42 are
dated entries and are kept with a rider rather than edited** -- they record what was
true on 2026-09-11, including that the default then stayed on `main`. D67, this file's
earlier entries, `docs/MODELS.md`'s scope fences and `docs/REORG_PLAN.md`'s dated
snapshot are likewise untouched.

### (!) The residual risk, stated rather than dismissed
**External `/blob/main/` links now 404.** Zero such URLs exist inside either tree, which
is measured; **a grep cannot see a link in somebody else's document**, and nothing here
bounds that number. Tags and Releases resolve into `dev` and are unaffected, and a
visitor now lands on `master`.

## [0.6.42] - 2026-09-11 - The ESA-ADB licence is verified at source, and both branches are pushed

**The reorganisation is complete.** Zero bucket operations.

### The licence, settled
**CC BY 3.0 IGO, read at the Zenodo record on 2026-09-11** -- record `15237121`, **version
v2, published 2025-04-17**, Rights field read directly. **The repository was right and the
earlier brief's "CC-BY 4.0" is confirmed wrong.**

**It was carried as `verified: "repository-only"` for two days across three failed attempts**
-- `zenodo.org` unreachable with a TLS certificate error on the record URL, the DOI and the
DOI's redirect target, while `arxiv.org` answered from the same environment. **Five internal
copies agreeing with each other is not verification**, and the rule that an unverifiable
licence is marked unverified rather than stated is what kept the claim honest until somebody
could open the page.

Updated where it was live: `docs/datasets/ESA_ADB.md`, `docs/datasets/REPRODUCING.md`,
`docs/INDEX.md`. **Kept with riders rather than edited**: `docs/REORG_PLAN.md`'s dated
paragraph and `CHANGELOG.md` 0.6.39, both of which record what was and was not known at the
time.

### Pushed
**`master` and `dev` are pushed to `origin`.** No force-push, no history rewrite, and the
nine tags stay where they are. The repository is **private**, so no `LICENSE` is required
until public release.

**The GitHub default branch stays on `main`.** **Licence selection is the one remaining item
before it moves**, and it is a separate decision.

## [0.6.41] - 2026-09-11 - Reorganisation tranche 5: the public branch exists, locally

**`docs/REORG_PLAN.md` tranche 5.** Zero bucket operations. **Nothing is pushed**, `master`
has **no remote tracking branch**, and the GitHub default stays on `main` until a licence is
selected.

### The branch
- **`git branch master main`**, so every existing link, Release and citation into `main`
  keeps resolving. **`dev` is untouched**, the nine tags stay where they are, and **no
  history is rewritten and nothing is force-pushed.**
- **96 of 270 tracked files, 2.96 MiB**: `flight/` with its committed vectors, the Monitor
  component and the SentinelRef deployment with the three build files they need, `README.md`,
  `Objective.md`, `STATUS`, `RESULTS`, `DECISIONS`, `NARRATIVE`, `MODEL_FILE`, `FPRIME`,
  `PI_ENVELOPE` and `docs/datasets/`. Plus **`.gitignore`**, which D67's lists name neither
  way: it is infrastructure rather than content, and it is what keeps telemetry out of a
  repository whose first permanent rule is that none is ever committed.
- **The public `README.md`** carries the branch paragraph, the omissions table, the
  dev-resolution convention pinned at `99348fa`, **BSD clause 3 verbatim**, the author-email
  disclosure, and a repository map and run instructions rewritten to be true of that branch.

### What the checklist found, and all three were real
- **(!) Four dead links.** `README.md` linked `docs/PHASE2.md`, `docs/PHASE5.md`,
  `docs/PHASE1_REPORT.md` and `docs/INDEX.md`, none on the branch, so a public reader
  clicking any of them would have got a 404. Converted to named references. **Twelve
  relative links on `master`, all resolving.**
- **(!) The green build covered less than it looked like.** Cloned fresh and run with nothing
  else, `master` runs **two of the seven** forward-pass golden tiers: `g3`'s and the four
  `g4_*` weight files are **deliberately uncommitted anywhere** -- 285 KB that regenerates
  from a seed, gitignored by policy with a test asserting it stays that way -- and the
  generator is in `scripts/`, which is not on the branch. So `g3.vec` ships with no input and
  **skips silently**. The policy is not overridden; the README states the gap, because a
  reader should not read seven tiers where two ran.
- **Everything the port and D68 added does run there**, at full width: the four baseline
  tiers, three trailing, two threshold, two derivative, **and `p1`, the flight configuration
  end to end** -- a real `param_version` 2 `model.bin` stepped 3,200 times, fused score to
  **3.098e-06**, both flags exact. Plus 18 refusal cases, determinism twice in-process and
  twice across processes, and the footprint against its prediction. `make -C flight lint`
  clean at `-Werror`.

### `must_never_reach_public`, all eight categories, against what `master` actually carries
**Credentials**: none tracked, no pattern anywhere. **Host and pod addresses**: none; the two
regex hits are `127.0.0.1` in `SentinelRef/README.md` -- the one recorded exception -- and a
clang version string. **Telemetry in any form**: no parquet, pickle, archive, array or CSV
tracked. **Non-redistributable material**: `third_party/` is not on the branch, so nothing is
redistributed and BSD clauses 1 and 2 do not bind there. **Caltech/JPL endorsement**: the four
matches are the obligation being *stated*, not claimed. **Assistant attribution**: none, in
files or in commit messages. **Personal names, emails, meetings, reviewers**: none; every
`master` commit is `GalacticDroid448` at the noreply address. **Retired claims restated as
live**: five sites, and all five are exactly what `dev`'s own guard pins as records -- D9's
measured lead table, D9's paragraph, a blockquote, `docs/NARRATIVE.md`'s narrative and
`docs/RESULTS.md` section 1's table.

## [0.6.40] - 2026-09-11 - D67.1: the honesty record and the reserved envelope go on master

**A rider, not an edit.** D67's two lists are unchanged; this adds to them. Zero bucket
operations.

- **`docs/NARRATIVE.md` and `docs/PI_ENVELOPE.md` go on `master`.** Tranche 4's public
  reading order put both in its nine steps while D67's list carried neither, so the order as
  planned cited two documents the curated branch would not have had. **That was flagged at
  `README.md` rather than papered over**, and this is the answer.
- **Why both, and it is the same reason:** a sceptical reader needs them first. The
  narrative is what happened in order, mistakes included -- section 6 alone carries nine
  sub-sections of results this project retracted, two of them its own confident hypotheses --
  and a reader deciding whether to trust a number is better served by that than by any
  surviving one. The Pi envelope is the first question anyone asks about a neural forecaster
  on a spacecraft, and **its answer today is "reserved, and deliberately empty"**: a public
  branch that omits it looks like one that has not been asked.
- **Nothing else changes.** `src/`, `scripts/`, `tests/`, `docs/MODELS.md`, the remaining
  `docs/` internals, `docs/INDEX.md`, `CHANGELOG.md` and `third_party/` stay off `master`.

## [0.6.39] - 2026-09-11 - Reorganisation tranche 4: the datasets, made obtainable without the bucket

**`docs/REORG_PLAN.md` tranche 4, complete.** Zero bucket operations, no fit, weight store
**1,313 -> 1,313**. 683 tests. Tracked content **5.82 MiB**, against N8's 6 MiB band.

### Written
- **`docs/datasets/ESA_ADB.md`** -- the primary dataset for a reader who will never have the
  bucket: the paper, the DOIs, the three missions and their channel counts, and the four
  caveats that silently corrupt a result, including the one that is the reason **no figure
  here is ever quoted in hours** (timestamps are anonymised mission time scaled by an
  undisclosed factor).
- **`docs/datasets/SMAP_MSL.md`** -- the legacy set that carries the headline number anyway,
  with **D66 reproduced in full** as the entry itself asks. The `P-2` labelling defect (81
  unique channels, not the 82 rows its label file implies), the test-split pre-scaling, and
  **Wu and Keogh's objection accepted rather than argued with** -- with the narrow thing that
  survives it stated narrowly.
- **`docs/datasets/OTHERS.md`** -- twelve datasets examined and declined, one reason each,
  the non-public routes named rather than pretended away, and the acceptance test any future
  dataset must pass first.
- **`docs/datasets/REPRODUCING.md`** -- what anyone can recompute with **no dataset, no
  credential and no bucket**, and plainly what they cannot. **Every command in its first
  table was run to produce the page**, including a complete scoring run on the generated
  fixture at zero operations.
- **`docs/PI_ENVELOPE.md`** -- reserved, the shape of a measurement fixed, **and no numbers**.
  It says why it is empty and names the failure it is guarding against:
  `docs/NARRATIVE.md` 6.9 records a figure written into a planning mock-up and later read
  back as data.
- **The public reading order** in `README.md`: nine steps, with a **claim-to-evidence table**
  whose last row is *"It warns early -- no such claim is made"*.

### (!) The ESA-ADB licence is still NOT verified, on a second attempt
The plan requires a re-fetch at write time and it was attempted: **`zenodo.org` is
unreachable from this environment** -- TLS *"unable to get issuer certificate"* -- on the
record URL, the DOI, **and** the DOI's redirect target, on **2026-09-11** as on 2026-09-09.
`arxiv.org` answers from the same environment, so this is Zenodo-specific rather than a
missing network.

**The licence stays marked `verified: "repository-only"` and the failure is stated on the
page rather than hidden.** An earlier brief said CC-BY 4.0, this repository says CC BY 3.0
IGO, and **neither is asserted beyond what was checked**. It must be re-checked at the record
before publication.

**What the attempt did verify**, at `arxiv.org` and field by field: the paper's title, its
eleven authors in order, submitted 25 June 2024, revised 17 August 2025, *"87 pages, 24
figures, 19 tables"*. And SMAP/MSL's **BSD 3-Clause was verified by reading
`third_party/telemanom/LICENSE.txt` directly**, with clause 3 quoted verbatim from line 11
because it binds this project and not only its code.

### (!) Two steps of the planned reading order are not on the public branch
`docs/NARRATIVE.md` and `docs/PI_ENVELOPE.md` are in the reading order and **not in D67's
on-`master` list**. Rather than silently amend either, the README says so where a reader
meets it and flags it as something **D67 may want to revisit**: step 6 is this project's
honesty record, and step 9 is the first question anyone asks about a neural forecaster on a
spacecraft.

### And the excerpts index moved, which is the guard working
`docs/datasets/SMAP_MSL.md` cites into the vendored source -- `channel.py:69-82`,
`errors.py:70`, `LICENSE.txt` -- so the citing-file count went **13 to 14** and
`tests/test_telemanom_index.py` failed until the index was regenerated. **A new document
that cites the published method should move that index, and a guard that did not notice
would be the defect.**

### The debt list is empty
`check_references.PLANNED` carried the five documents cited before they existed. **Tranche 4
wrote all five**, so it is emptied and they are checked like everything else -- which is why
the list is empty rather than deleted.

## [0.6.38] - 2026-09-11 - Reorganisation tranche 3: the link-check guard, and one citation that never resolved

**`docs/REORG_PLAN.md` tranche 3, complete.** Zero bucket operations, no fit, weight store
**1,313 -> 1,313**. 683 tests, up from 666. Tracked content **5.78 MiB**, against N8's 6 MiB
band.

### The guard, and what it found
- **`scripts/check_references.py`** with **`tests/test_references_resolve.py`** beside it, the
  shape `check_no_list.py` already uses. It checks five kinds of reference -- **458 section
  citations, 481 repository paths, 319 `file:line` ranges, 30 relative links, 4 pytest node
  ids** -- and it exists to **hold** the clean result `docs/REORG_PLAN.md` 3 audited by hand,
  not to discover one.
- **It found one thing the hand audit had not.** `third_party/telemanom/PROVENANCE.md:92`
  cited `docs/MODELS.md` **26.31**, which has never existed -- the readings are 26.29.1 to
  26.29.5 and 26.30.1 to 26.30.4. Corrected in the same commit, in a file this project wrote.
- **And D67's `--master` mode**, which the branch work needs: `master` carries the component
  and the evidence and nothing else, so **367 citations resolve on `dev` rather than on
  `master`** -- reported as dev-resolving, and a break only if absent from **both**. The
  dev-resolution note is now at the top of all seven on-`master` documents.
- **Three conventions it had to be taught**, each measured before it was granted: `Objective.md`
  **14.N** is a table row; upstream `nasa/fprime` docs namespace-collide with this repository's
  `docs/`; and a bare `detector.py:NNN` at `scripts/smap_rungs.py:1263` means the **vendored**
  file. A fourth is debt rather than convention: `PLANNED` lists the five tranche-4 documents
  cited today and not yet written, and a test fails if one of them quietly appears.

### Legibility, additive throughout
- **`scripts/README.md`**: all 36 scripts, grouped by what they are for -- guards, generators,
  studies, operations, shell -- each study naming the `docs/MODELS.md` section that registered
  it, each generator naming what it is held to.
- **Tables of contents** for `docs/MODELS.md` (**353 entries**) and `docs/DECISIONS.md`
  (**68**), generated by `scripts/make_tocs.py` and pinned by `tests/test_tocs_are_current.py`.
  **In document order, and where that departs from numeric order it says so**: 33.6 follows
  33.8, 34.7 follows 34.8, and 26.6 appears twice. **Nothing is moved or renumbered** -- the
  numbering is cited from hundreds of places and a table of contents is not a reason to break
  one.
- **`docs/MODELS.md` 35.7 demoted from `##` to `###`**, which every other OBSERVED block
  already was. It read as a top-level section in any generated index.
- **`docs/NARRATIVE.md` section 6 gains nine sub-headings** over its 365 unbroken lines.
  **No sentence is reworded.**

### Two false positives the guard found in itself, and both are recorded
- **It read its own test's prose as a citation.** `tests/test_tocs_are_current.py` says
  "`docs/DECISIONS.md` 68 entries", which the section pattern took as a citation to a section
  68. `docs/DECISIONS.md` has **zero** numbered headings -- every entry is `## D42.` -- so a
  bare number after its name can never be a section citation there, and the checker now says
  so. **A guard that cries wolf is one people switch off.**
- **And it checked itself.** Its own docstring shows a pytest node id and a dataset path as
  worked examples of the forms it matches, and its test quotes the one dangling citation it
  caught. (This entry originally quoted one of those examples verbatim and the guard flagged
  **it** -- a third instance of the same shape, fixed by rewording the prose rather than by
  exempting a third file. The guard was right both times.) Both are now named as self-referential -- the same exemption,
  for the same reason, that `scripts/telemanom_citations.py` takes and that
  `tests/test_documents_are_current.py` has always taken when it skips itself for quoting
  every pattern it bans.

### The comment sweep, largest-risk first
- **(!) The model path computed a maximum and never an argmax**, so it named no channel --
  while `Baseline`, the **degraded** Level 1 path, has named one from the start, and
  `baseline_reference.py:119` and the F' component both describe an argmax the primary path
  was not computing. The degraded path explaining itself better than the healthy one is
  backwards, and `Objective.md` 11 rule 4 requires a named channel. `Detector::peakChannel()`
  now exists, ties to the lowest index as `Baseline`'s scan does.
- **`telemanom.py` carried zero citations into the source it transcribes**, where
  `scripts/smap_rungs.py` carries 72. Six sites now cite it by file and line, read at first
  hand -- including the sharp one: **telemanom has two asymmetric index-set forms**, and
  `errors.py:347-352` is the buffering this implements while `:359` is the batch clip 26.29.1
  records and is **not** implemented here. Reading the two as one is how a reproduction goes
  wrong quietly.
- **`flight/src/Detector.cpp`'s forecast comment** now cites `windows.py:227-234` and records
  what it had not: **the mean is this project's choice, not published telemanom's**, which
  uses the single one-step-ahead prediction (`modeling.py:113`, called at `:172`). The largest
  single training difference, 28.1 T-a, inherited by the flight core.
- **`Z_LIMIT` and `window_ratios` are marked unused** rather than deleted -- both are code
  rather than a documented figure, and both are what a reader comparing this module against
  `errors.py` will look for.

## [0.6.37] - 2026-09-11 - D68: the flight configuration is adopted, on a version-2 PARAMS block

**The first entry since D62 to name a flight configuration.** Zero bucket operations -- it
rests on D65's two reads and the port's own vectors, and spends nothing new. 666 tests, up
from 655.

### Adopted
- **`emitted` is `max(z_residual, z_derivative) >= threshold`** -- exactly what D65 measured
  twice at **EVAL 17 of 19 against the frozen arm's 4**, a strict superset, with the
  horizon-disagreement stream dropped because P2.4 was refuted in the informative direction.
- **Forward-only dilation**, on 39.5's reason -- backward dilation marks timesteps already
  emitted and there is no un-emit -- and **not** on N6's number, which 39.13 records as
  confounded with the dial.
- **The dynamic threshold is carried and reported and drives nothing**, which meets D62's
  requirement: `flight/` no longer transcribes D25's static quantile alone.
- **The static quantile still drives `baseline_only`**, D5's Level 1 path, unchanged.

### (!) The format question, answered without touching D30's freeze
The fused threshold is a **different statistic on a different scale**: a version-1 cut is a
99.9th percentile of a smoothed error in data units, a version-2 cut is a z-score. **The byte
layout does not change** -- no field added, moved or resized -- so `format_version` stays
**1** and `docs/MODEL_FILE.md` 11's "adding a field is a `format_version` bump" is not
engaged. What moves is **`param_version`, 1 -> 2**, which is exactly the independence 11
already defines. **D68 is the first use of it and is the reason it exists.**

**And the field is now checked, which it was not.** It was read and never validated on either
side, so a reader would have applied one generation's cut to the other's statistic in
silence. Both readers now refuse an unknown generation with **`BAD_PARAM_VERSION`**, a new
status mirrored by value in C++ and Python, covered by `RefusalTests` on both the refusal and
the acceptance and by `tests/test_param_version.py` on both implementations.

### Measured
- Tier **`p1`**, the end-to-end evidence: a real `model.bin` at `param_version` 2 loaded into
  a `Detector` and stepped 3,200 times. **85 emissions matched exactly**, fused score within
  **3.098e-06**.
- **The version is load-bearing, not decoration**: the same bytes with the same cut value
  **cross 339 times as version 2 and 0 times as version 1**.
- **Every committed `.vec` vector still passes** and all seven models round-trip
  byte-identically, because a version-1 file still gets D25's rule.

### A defect the end-to-end tier caught, which nothing watching flags could
`Detector`'s fused-maximum scan started at **0.0**, so a statistic that is a **z-score** --
negative whenever a channel is quieter than its own trailing window, which is most of the
time -- was clamped to zero over every quiet stretch, and channel 0 was named as the peak
when nothing was peaking. **It never moved a flag**, because the cut sits far above zero, so
no test that only compared flags could see it. The scan now seeds from channel 0 as
`Detector::step`'s own `largest = m_smoothed[0]` always has. `DynamicThreshold` got the same
treatment pre-emptively.

### Registered as owed, not taken
- **D68.1, the third read**: both dilations at **one fixed cut**, so the lever moves and the
  dial does not. It refines N6 and **blocks nothing** -- D68 does not rest on N6's number.
  165 Class B and 1 Class A, after a smoke.

### What this does not establish
One arm, **one dataset**, **n = 19 per half, UNDERPOWERED (D3)**, no floor to compare against
at that alarm rate, and **no early-warning claim**: 37.7a measured 0 of 10 positive leads.
The cut in tier `p1` is a **test fixture and not a flight constant**; a mission calibrates its
own on its own nominal data.

## [0.6.36] - 2026-09-11 - The second read: neither departure costs recall, and N6's band was mis-specified

**`docs/MODELS.md` 39.13 second read.** **One further 165 Class B and 1 Class A**, cached
weights, **weight store 1,313 -> 1,313**, after a smoke at 9 Class B whose **projected cost
matched actual exactly**. Artifact
`runs/smap-msl/_forensics/2026-09-11T013152Z-departures2.json`. Month: **227 Class A and
5,321 Class B of 50,000**. Both gates pass again.

### Measured, with the cut floating as the frozen arm's own 0.5506 does
```
  arm                          cut       rate       TUNE    EVAL    all 38
  frozen (stage 4)          0.5506     0.6820%      6/19    4/19    10/38   matched
  N6: forward-only          0.5009     0.6820%     12/19   11/19    23/38   MATCHED
  N5: guard cells           1.0000     0.5848%     10/19    6/19    16/38   14% below
```
**Both are strict supersets of the frozen arm on EVAL.** **n = 19 per half,
UNDERPOWERED (D3).**

### (!) N6 fails its band in the opposite direction to its intent
The band asked what dropping backward dilation would **cost** -- 0 to 1 event HOLD, 4 or more
FAIL, with FAIL re-opening 39.5's choice. The measurement is **+13**, so the band fires while
the concern it was written to detect does not materialise. **A one-sided worry written as a
two-sided band**, and the mis-specification is the finding. 39.5's choice was made because a
warn-only component cannot emit retroactively, and nothing here re-opens it.

The mechanism is D65's P2.4 in another guise: backward dilation spends 99 steps of alarm
budget **before** every crossing, where the evidence has not happened yet, so it cannot buy a
detection and does count against the rate. Remove it and the rate-matched cut falls from
0.5506 to 0.5009.

### (!) And the +13 is confounded with the dial, which 38.15 already diagnosed
Arm 1's finding applies here unchanged: *a multiplier below 1.0 re-admits pruned steps,
because `channel_ratios` maps a suppressed step to `raw/(1+raw)`... the dial is not
rate-matching here, it is changing the character of the decision.* The frozen arm sits at
**0.5506** and N6 at **0.5009**, both inside that band, so part of the +13 is the lever and
part is the dial and **this read cannot say how it divides**. N6's number is reported as
measured and **is not a clean measurement of what backward dilation costs**. Separating them
needs both dilations scored at **one fixed cut** -- a third read, not taken.

### (!) N5 remains NOT ADJUDICATED
Its cut pins at **1.0000** and the rate lands at **0.5848%, 14% below target**, so no
matched-rate number of it is reported -- 38.15's rule for Arm 6, an arm that cannot be
brought to the budget is unrun rather than losing. What stands without matching: at a cut of
exactly 1.0 **no pruned step is re-admitted**, which is the legal-dial regime 38.15 asked
for, and there guard cells catch **16 of 38 at 0.5848% against 10 of 38 at 0.6820%** --
**more events at a quieter rate, which is dominance and needs no matched rate.** It is the
opposite of 10.7's ESA-ADB result, where guard cells cost 38/46 -> 34/46; **the sign has now
reversed on a third dataset and the reason is still not established.**

### Registered and not built
- **39.14: a float64 Python reference for the threshold**, so N2's contract can be judged.
  The residual measures the **reference's** float32 resolution, not the transcription's, so
  the question N2 asked cannot be answered against the present reference at all. Deferred
  deliberately; `telemanom.py`'s arithmetic is **not** changed to make a test tidier.

### Unchanged
**Nothing is adopted.** The port emits on D25's static quantile; D65 names no flight
configuration; **N3's failure stands as recorded and its band is not moved.**

## [0.6.35] - 2026-09-11 - N5 and N6: one read spent, and neither departure got its price

**`docs/MODELS.md` 39.13, the port's OBSERVED entry.** **One read of 165 Class B and 1 Class
A**, cached weights, **weight store 1,313 -> 1,313** (asserted), after two smokes. Artifact
`runs/smap-msl/_forensics/2026-09-11T011756Z-departures.json`; producer
`scripts/decision_layer_departures.py`, in this commit. Month: **225 Class A and 5,147 Class
B of 50,000**.

### Both gates passed, and everything else rests on them
- **The reproduction gate.** The frozen arm rebuilt on 77 of 81 channels returns **0.6820%
  and 10 of 38**, TUNE 6/19 and EVAL 4/19 -- identical to 38.15 on both reads.
- **The reference gate, which is new.** `flight_reference.ratios` with the departure switched
  off is `telemanom.channel_ratios` **exactly** -- `np.array_equal` on all 77 channels' full
  score arrays. The restatement the C++ is held to is the published rule.

### (!) N5 and N6 are NOT ADJUDICATED, and the defect is mine
Both variants were scored with the cut **floored at 1.0** and neither reached the matched
rate: forward-only landed at **0.3787%**, 44% off target; guard cells at **0.5848%**, 14%
off. **Neither figure is reported as a result** -- 38.15's rule for arms 4a, 6 and 7 applies
unchanged, and *a recall figure without its alarm rate is not a result*. **The 16 of 38 in
particular must not be quoted, in either direction.**

The floor was inherited from Arm 1's repair, where it was right and here is wrong. Arm 1's
**lever was pruning `p`**, and a dial below 1.0 re-admits deleted steps, so flooring it
separated lever from dial. Here the levers are dilation and guard cells, and **the frozen
arm's own operating point sits at 0.5506** -- below 1.0. Flooring the variants at 1.0 while
the baseline runs at 0.5506 compares two kinds of operating point rather than matching a
rate. At a cut of exactly 1.0 only surviving-sequence steps alarm, so forward-only's 0.3787%
is its smaller alarm extent by construction and not a measurement of recall.

**The repair is one line** -- let the cut float as the frozen arm's does -- and it needs a
**second read of 165 Class B and 1 Class A, which is not taken here** and is brought to the
owner with its cost. Until then 39.4's and 39.5's choices stand on their stated reasons and
not on a number.

### The first smoke diverged by one operation, which is what smokes are for
Projected **6** Class B for two channels, actual **7**. The run was right and the arithmetic
was wrong: `ops.load` fetches the ledger before anything else and the formula omitted it. The
**165** this project has quoted all along is `1 ledger + 1 manifest + 1 labels + 162 arrays`,
so the ledger was always in the number and never in the projection. Corrected, re-smoked:
**projected 9, actual 9**; the full read then went **projected 165, actual 165**.

### The port's other verdicts
**N1 held** (1e-5 throughout, emission flag exact), **N4 held** (bit-identical determinism
with 2,170 samples of carried state), **N7 held**, **N8 held** at 5.53 MiB. **N2 no verdict**
with its cause identified. **N3 failed** at +3.70%, itemised at 39.13.2.

## [0.6.34] - 2026-09-10 - Port stages 3 and 4: the derivative stream, wired and reported, and N3 missed

**`docs/MODELS.md` 39 stages 3 and 4 of 4. The port is complete as registered.**
Zero bucket operations, no fit, weight store **1,313 -> 1,313**. 654 tests, up from 649.
Tracked content **5.53 MiB**, against N8's 6 MiB band and 7 MiB stop.

### Built
- **`Sentinel::DerivativeStream`** -- `dx[t] = |x[t] - x[t-1]|` with `x[-1] := x[0]`,
  standardised against a trailing 2,100 window. D65's second stream, and the one that
  carried the finding.
- **Both streams wired into `Detector`**, with `dynamicEmitted()`, `fused()`,
  `fusedScore()` and `fusedChannel()` beside the existing accessors. `fused` is
  `max(z_residual, z_derivative)` -- **the ablation**, because P2.4 was refuted in the
  informative direction and the horizon-disagreement stream spends more budget than it
  returns.
- **`Detector.hpp`'s file comment replaced** with what the unit now implements: four
  numbered layers, and which one emits.

### (!) Three decision layers run; one emits, and that is a decision
`crossing()` and `emitted()` are **still D25's static quantile**. D62 required the port to
carry the dynamic threshold and it now does -- it runs every tick, is held to the reference
at 1e-5 and its emission flag is exact -- but **D65 re-opens the decision layer and names no
flight configuration**, so promoting either new rule to the warning would make this port the
adoption decision. It would also move every committed `.vec` vector, which 39.3 said would
not happen. **The core computes all three; which one the F' component raises an event on is
the component's to wire and the owner's to decide.** `flight/README.md` says so where a
reader will meet it.

### Measured
```
  f1    3 ch x 2300 steps   z_derivative max |diff| 2.720e-07
  f2    6 ch x 2300 steps   z_derivative max |diff| 2.899e-07
```
**N1 holds** on the derivative as it did on the threshold. The existing `.vec` tiers still
pass unchanged and all seven models still round-trip byte-identically, which is what says the
forward path is bit-for-bit what it was.

### (!) N3 is MISSED, and the account is itemised rather than the band moved
`sizeof(Detector)` is **603,024 B (588.9 KiB)** against N3's **581,488 +/- 64** --
**+21,536 B, +3.70%**, outside its own 1% band. Itemised the way 19.8 F1's 530 B was:

```
   +8,960   the rings are SOLVE_WINDOW deep, not ERROR_WINDOW: 2,170 is 2,100 of
            history plus the 70 the threshold judges. 39.6 sized both at 2,100.
  +12,168   the pruning ladder's scratch -- m_seq, m_kept, m_peaks, m_order at
            MAX_SEQUENCES = 1,085, plus eps and the latest sample per channel.
            39.6 did not itemise it at all. Shared across channels, so once.
     +408   the second moment-accumulator set, so one ring serves the threshold's
            2,170 contents and arm 2's 2,100 moments rather than two rings doing it.
```

**19's F1 is a record and is not edited.** It measured an object that no longer exists; the
port's detector is **1.93x** it. Both `flight/test/Footprint.cpp` and
`tests/test_flight_build.py` now print F1 as history and assert against the measurement, with
the same 2% latitude F1 was given.

### N4 holds, and it was the one that could not have a NO VERDICT
Determinism is bit-identical in-process and across two process invocations **with 2,170
samples of carried state per channel on two rings**. A detector that size which did not
reproduce could not fly (Objective.md 11 rule 5).

### Still owed, and named rather than left implicit
**N5** (the guard-cell choice) and **N6** (the price of dropping backward dilation) are
**unmeasured**. Both need the one approved read, which has not been taken. No figure for
either exists and none should be quoted.

## [0.6.33] - 2026-09-10 - Port stage 2: the dynamic threshold, and a correction to 39.6's own arithmetic

**`docs/MODELS.md` 39 stage 2 of 4.** Zero bucket operations, no fit, weight store
**1,313 -> 1,313**. 649 tests, up from 632. Tracked content **5.24 MiB**, against N8's
6 MiB band and 7 MiB stop.

### Built
- **`Sentinel::DynamicThreshold`** -- telemanom's nonparametric threshold, streamed. It
  solves once per 70-tick segment over the trailing window, sweeping 19 `z` candidates,
  skipping any above the window's own reach, scoring
  `(dmu + dsigma) / (sequences^2 + covered)` and taking ties to the larger `z`. When
  nothing qualifies it returns `mu + 12*sigma` and reports nothing: **silence, not a guess.**
- **`src/sentinel_models/flight_reference.py`** -- the NumPy statement of the streamed rule,
  the same move `baseline_reference.py` made and for the same reason: the core transcribes
  **the rule**, not a harness implementation, and adding a forward-only option to
  `telemanom.py` would be a scope change to the module every published figure rests on.

### (!) 39.6's footprint arithmetic was wrong, and the code is right instead
The solve window is **`error_window + stride` = 2,170**, not 2,100:
`telemanom.py:397-400` takes `e_s[seg_lo - error_window : seg_hi]`, which is 2,100 of
history **plus the 70 being judged** -- and 39.4's entire guard-cell question is about those
70. 39.6 nonetheless sized both rings at 2,100, so its L1 figure is **8,960 bytes light**.
`Config::SOLVE_WINDOW` is added and says so; `TrailingWindow` now carries a logical span
inside a ring sized for the larger user, so one class serves the threshold's 2,170 and the
derivative's 2,100. **N3's band will be missed and the miss is reported rather than the band
being moved.**

### Measured
Against `flight_reference`, forward-only dilation, tiers of 3 and 12 channels x 2,800 steps:

```
  d1    3 ch   eps max |diff| 1.064e-06    emissions 17 of 17, exact
  d2   12 ch   eps max |diff| 5.072e-06    emissions 40 of 40, exact
```

**N1 HOLDS**: within 1e-5 on `eps`, and the emission flag is exact on every step of both
tiers. **N2 gets NO VERDICT** -- its band asked for 1e-6 and the worst is 5.072e-06 -- and
**the cause is identified rather than left open**: the reference means and standard-deviates
a **float32** array where the core accumulates in F64, and the gap scales with `z` exactly as
observed (6.9e-07 at z=2.5, 2.9e-06 at z=11.5 on a representative window). **The core is the
more accurate of the two**; the residual is the reference's own float32 resolution, which is
what 19.8 F5 said of the forward pass.

### The faithfulness claim, and it is tested rather than asserted
Two of the three streaming departures are real (`39.5`), so the third must be nothing: with
**backward dilation restored**, `flight_reference.emissions` reproduces
`telemanom.channel_ratios`' own emission array **byte-identically** across four
seed-and-length combinations. Whatever the streamed form changes, it is not the arithmetic.
`solve_window` also delegates to the published sweep outright in that mode, so there is one
sweep and not two.

### One thing the first run got wrong, and it was the test
The eps comparison ran **after** `step`, and on a segment's last tick `step` solves and
replaces the threshold -- so it compared the next segment's threshold against this segment's
expectation. Every difference it reported sat on a boundary tick, which is what said so.
`eps_held[t]` is the threshold **in force** at `t`; the comparison now reads it before the
tick, and the test carries the reason.

## [0.6.32] - 2026-09-10 - Port stage 1: the trailing window, and its drift measured rather than assumed

**`docs/MODELS.md` 39 stage 1 of 4.** Zero bucket operations, no fit, weight store
**1,313 -> 1,313**. 632 tests, up from 623.

### Built
- **`Sentinel::TrailingWindow`** -- a trailing right-inclusive window per channel with its
  mean and standard deviation carried in F64. Both new streams need it and they need
  different halves: the dynamic threshold needs the window's **contents** (it re-solves over
  19 `z` candidates, recomputes both moments with candidates removed, and finds and prunes
  runs inside it), the derivative stream needs only the **moments**. One class, two
  instances, which is layout **L1** as 39.6 registered it.
- **Channel-major**, unlike `Baseline`'s slot-major ring: the threshold walks one channel's
  whole window 19 times per re-solve and never walks a slot across channels.

### (!) It carries its moments where `Baseline` recomputes them, and the drift was measured
`Baseline.cpp` recomputes its 120-sample window every tick and says why -- *"a running
add/subtract drifts without bound over a mission"*. At 2,100 x 16, twice over, that would
cost **67,200 operations a tick** against the model path's own 70,080, and 39's **N7**
registers the derivative at O(1). So the drift was measured on D37's own regime before the
choice was made: **1,000,000 ticks, window 2,100, N(1000, 3) float32**, running against an
exact recompute --

```
  worst relative drift, mean   0.000e+00
  worst relative drift, sd     4.246e-10
```

The mean's is exactly zero because every sample leaves by the same subtraction that admitted
it. **Stated as a measurement over 1e6 ticks and not as a proof for all time.**

### Measured against the reference
`scripts/decision_layer_arms.py:53-58`'s `trailing_stats`, which differences float64 prefix
sums over the whole series where the core carries a window -- the same quantity by different
means, which is what makes the comparison worth drawing.

```
  t1    3 ch x 2300 steps   max |diff| 6.661e-16
  t2    6 ch x 2300 steps   max |diff| 4.263e-14
  t3   12 ch x 2300 steps   max |diff| 3.738e-10   <- D37's N(1000, 3) regime
```

**Worst 3.738e-10 against N1's 1e-5**, and it lands within 12% of the 4.246e-10 the Python
drift measurement predicted, which is the two arriving at the same answer independently.
**N2's band is met with four orders to spare.** Every tier runs 200 steps past the window,
so the ring wraps and the subtraction is exercised rather than assumed.

### Two things found while building it
- **`TestSupport::readFile` fills a 512 KiB buffer and tier T3 is 552,064 B.** Reading it
  whole would have truncated it **silently** -- a short read looks like a short file -- and
  the comparison would then have run against the wrong bytes. `TrailingVectors` streams a
  record at a time instead; a record is at most 320 bytes, and 39.7's larger decision-layer
  vectors will not hit the same wall.
- **T2 was cut from 12 channels to 6.** T3 already covers the flown width, and a second tier
  at the same width buys coverage nobody has and 276 KB of tracked content somebody pays
  for. Tracked content is **4.73 MiB**, against N8's 6 MiB band and 7 MiB stop.

## [0.6.31] - 2026-09-10 - The C++ port, pre-registered before a line of it exists

**Roadmap item 1, registered and not built.** `docs/MODELS.md` 39. Zero bucket operations,
no fit, weight store unmoved. **No C++ is written until this is reviewed.**

### Registered
- **`model.bin` stays at `format_version` 1** (39.3). The dynamic rule's constants become
  `constexpr` in `Config.hpp`; PARAMS already carries `ewma_span` 105 and `warmup_steps`
  2,350, so the format anticipated the trailing window before there was one. The selectable
  mode remains a version-2 decision after Phase 3, unchanged from D62 c.2. **D30 untouched.**
  19.7's stop condition 3 is discharged rather than tripped.
- **Eight numbered predictions, N1 to N8**, with bands and a stated falsification: the 1e-5
  accuracy contract, the footprint, determinism, the two departure costs, the derivative's
  per-tick cost, and the vector budget.
- **The derivative is computed and reported, never in the emitted flag** (39.8). D65 adopts
  nothing, so wiring the fused rule into the warning would make the port the adoption
  decision.

### Two things the reading corrected, and both were in the plan this section came from
- **The window already trails.** The plan said the C++ window must end at `t-1` because the
  published one looks ahead. It does not -- `telemanom.py`'s docstring records that this
  repository already replaced telemanom's forward-extending window. What is actually at
  stake is **emission timing**: the window includes the 70 steps it judges, so a segment is
  decided at its end. **And guard cells are not a free way to remove that latency**: 10.7
  measured recall **38/46 -> 34/46** and alarm ranges **+27%** on `m1-g8.9.10`, and the
  *reverse* on `m1-ss5`, with the sign reversing for reasons not established. Neither is
  adopted; **N5** measures the choice on the 38.
- **Backward dilation cannot be emitted.** `_buffered` widens every run by `+/-99` and
  merges; the backward half marks timesteps already emitted, and there is no un-emit.
  Dropped, with **N6** registering its price -- and **N6 cannot be measured at zero
  operations**, because it needs the smoothed error and no telemetry is on local disk. It is
  carried unadjudicated and the read is brought to the owner with its cost.

### Measured while writing it
- **The footprint.** Both trailing windows are unavoidable: `dynamic_threshold` needs the
  window materialised, and the derivative's `z` needs the departing sample. Three layouts
  costed against the measured **312,112 B**: L1 two rings at 16 channels, **581,488 B**;
  L2 at 12 channels, **495,520 B**, which buys its saving from capability and would collapse
  the maxima onto the flown shape; L3 sharing `Baseline`'s ring, **573,872 B**, for 1.3%.
  **L1 registered, L2 the named fallback, L3 not recommended.** 19's F1 is a record and is
  not edited; N3 sits beside it.
- **The vector format will not scale.** `g3.vec` is 94,960 B for 80 steps -- about 1,187
  B/step, dominated by the hidden-state trace -- and the dynamic rule needs ~3,000 steps, so
  ~3.5 MB per tier. A leaner decision-layer format is registered at ~250 B/step, and **N8
  stops the work if tracked content would pass 7 MiB**. Read back, that is a second reason
  D64 was right, and not the one given at the time.
- **`GoldenVectors.cpp:83` silently skips a tier whose files are absent**, and only `g1`,
  `g2` and `b1`-`b4` are tracked. Under D67 `flight/test/` is the evidence on `master`, so
  the new tiers are committed or they are not counted.

### Recorded
- **39.1: the second handover brief, adjudicated in eighteen rows**, in the form 37.1 uses.
  Neither handover brief was ever committed -- `SENTINEL_PM_HANDOVER.md` has never existed
  in the tree or in any commit's tree.

### A defect of mine, found and repaired in the same work
- **The 0.6.30 regeneration overwrote `docs/TELEMANOM_EXCERPTS.md` section 4.** That section
  is nine hand-written findings -- the batch clip, `evaluate_sequences`' mixed precision
  denominator, `aggregate_predictions(method='first')` and six more -- under a
  `| Location | Mechanism | What it settled |` header. The generator anchored on the prefix
  `| Location`, matched section 4 first, and replaced analysis with an index. **Restored from
  `8476ad7` and verified row by row.** The generator now anchors on section 5's exact and
  unique header, excludes itself from the scan, and `tests/test_telemanom_index.py` pins all
  of it: the generated table equals the generator's output, section 4's three named rows are
  present, and both anchors are unique. **The guard exists because the mistake happened, not
  in case it might.**
- **And that guard then counted itself.** Its assertions quote row text -- `errors.py`,
  `detector.py` and `modeling.py` line references -- so once it was tracked the scan read
  them as citations and the index described itself, at 13 citing files instead of 12. The
  generator now names its self-referential files explicitly, which is the same exemption
  `tests/test_documents_are_current.py` already takes when it skips itself for quoting every
  pattern it bans.

## [0.6.30] - 2026-09-10 - Reorganisation tranches 1 and 2: the guards, and the figures they did not catch

**Recorded as done in the handover; verified as not done, then done.** Zero bucket
operations, no fit, weight store unmoved. 617 tests, up from 608.

### Guards repaired (tranche 1)
- **The test-count guard matched only an unwrapped count.** A count broken across a line --
  which is how `docs/STATUS.md` once carried it -- read as no count at all, so the guard
  passed while its subject was wrong. It now flattens the document first. Proven against the
  three forms it was blind to: wrapped, emphasised, and both. It fired on its first real use
  in this very tranche, at 616 against a collected 617.
- **The retired-claim guard's `lead` pattern was widened** to `26-timestep` and
  `26 timesteps ahead`. Four new sites: two live-prose ones in `docs/RESULTS.md` corrected in
  house form, two inside decision entries pinned as records.
- **The `ratio` pattern was NOT widened to the bare `28/32`, and that is a measurement.** It
  matches **30** sites, every one of them `lstm-telemanom`'s standing measured headline-cell
  recall -- one of them inside a pre-registered band. What D37 and D38 retired is the
  comparison against a per-channel statistic's 3, which the `three` pattern carries. Recorded
  where the change would have gone.
- **`docs_gen.write_data_md` refuses instead of overwriting.** It ended in a bare
  `path.write_text`, so the next ingest would have deleted `docs/DATA.md`'s hand-added prose
  in silence. A digest footer records the generated text; delimited hand-authored regions are
  carried across verbatim and excluded from the digest. `docs/DATA.md` is armed.
  `tests/test_docs_gen_refuses.py` pins eight behaviours.
- **The `_rolling` float32 defect is described in the past tense** in
  `baseline_reference.py`, `flight/include/sentinel/Baseline.hpp` and
  `flight/src/Baseline.cpp`. Re-measured: **3.5e-07** against `numpy.nanstd` with **zero**
  spurious exact zeros, against the 7.66 of error recorded when those comments were written.
  The flight core is unchanged and needed no change.

### Stale figures and dangling citations (tranche 2)
- **`docs/TELEMANOM_EXCERPTS.md` section 5 is generated, and now there is a generator.** The
  section claimed to be generated from the tree while nothing in the tree could regenerate
  it. `scripts/telemanom_citations.py` is committed with it. The index was missing **15**
  locations -- `errors.py:62`, `:70`, `:337-339` and the whole `403`-`418` pruning ladder
  `docs/MODELS.md` 37.2 reads line by line. **75 distinct locations across 12 files**
  (was 64 across 9). No cited range overruns the file it names.
- One rule in that generator is worth knowing: **a bare `detector.py:NNN` is not indexed**,
  because this repository has a `detector.py` of its own and every bare occurrence resolves
  to it. Indexing it bare would have added a dozen false rows.
- `docs/MODELS.md` 35.7 cited `runs/.../2026-09-09T2138*`, which matches no artifact; the
  two real timestamps are named. **A path correction, not a figure correction.**
- `README.md` said Releases `wi1` to `wi7`; **nine tags exist**.
- `CHANGELOG.md`'s `[Unreleased]` link compared against tag `wi9.5`, **which was never
  created**. Repointed at `wi9`, with a note that versions 0.6.1 onward are untagged and no
  link is invented for them.
- `docs/RESULTS.md`'s "warning **hours** before a limit trips" is retired as a unit and
  kept: both datasets' timestamps are anonymised or resampled, so every lead figure is in
  **timesteps**, and an hours figure can come only from the unbuilt physics testbed.

## [0.6.29] - 2026-09-10 - D67: `master` is curated, and every omission is stated

**A release-policy decision recorded before the branch exists, so tranche 5 inherits a
decision rather than taking one. No branch is created and nothing is pushed.** Zero bucket
operations.

### Decided
- **D67.** `master` carries the customer-facing C++ component and the evidence it works, and
  nothing else. `dev` remains complete and is never rewritten. Of **227** tracked files,
  **68** are the component and its evidence (**1.02 MiB**) and **159** are the apparatus that
  produced it.
- **On `master`:** `flight/` including `flight/test/`, `fprime/Sentinel/Monitor/`,
  `fprime/SentinelRef/`, the three top-level `fprime/` build files, `README.md`,
  `Objective.md`, `docs/STATUS.md`, `docs/RESULTS.md`, `docs/DECISIONS.md`,
  `docs/datasets/`, `docs/MODEL_FILE.md`, `docs/FPRIME.md`, and `LICENSE` once selected.
- **Not on `master`:** `src/`, `scripts/`, `tests/`, `docs/MODELS.md` and the remaining
  `docs/` internals, `docs/INDEX.md`, `CHANGELOG.md`, `third_party/`.
- **Every omission is stated; none is silent.** The public `README.md` names what is absent
  and why, and states that **the ground training toolkit is unreleased, so a mission cannot
  deploy this without it**.
- **Any off-branch path resolves on `dev` at a named commit**, stated once per document --
  the convention reaches about **380** references, of which `scripts/` and `runs/` are 116 --
  and enforced by a **`master` mode** in `tests/test_references_resolve.py`.
- **BSD clause 3 travels regardless.** `master` does not redistribute the telemanom source,
  so clauses 1 and 2 do not bind there; the non-endorsement obligation does, and goes on the
  public `README.md`.

### Superseded and kept
- `docs/REORG_PLAN.md` section 7 and `docs/reorg_plan.json`'s `branch_proposal`, which
  proposed a `master` whose tree is identical to `dev`'s head. The prose gains a rider
  pointing at D67 and is not otherwise edited; it is a dated research snapshot.

### Unchanged
- No history is rewritten and nothing is force-pushed, ever. Tags `wi1`-`wi9` and their
  Releases stay on `dev`. The author-email disclosure note stays. **Nothing is pushed until
  the licence is selected and the owner says so** -- and it is still "not yet selected".

## [0.6.28] - 2026-09-10 - D66: the in-range contextual class is confounded with envelope width

**A qualification, not a correction. D46's arithmetic stands and nothing is edited.** Zero
bucket operations; no fit; weight store unmoved. Written from
`runs/smap-msl/_forensics/2026-09-03T192723Z-visibility.json`, `...2026-09-10T041723Z-probe.json`
field `extrema`, and `third_party/telemanom/` read directly.

### Decided
- **D66.** All 39 in-range contextual sequences sit on channels whose training envelope spans
  at least **1.9 of the 2.0** available, and **30 of the 39 span the full `[-1, 1]` exactly**;
  channels carrying point anomalies have a median span of **0.274**. The 72.7-point gap over
  the point class **may not be read as a property of the anomalies alone**.
- **The published pre-scaling is REFUTED as the mechanism.** The vendored source has no
  scaler at all (`telemanom/channel.py:69-82`); `S-1`'s train `[-0.400000, +1.000000]` against
  test `[-1.000000, +1.000000]` settles that there is **one affine map per channel, fitted on
  the test split and applied to both arrays**. D46's range determination is **immune** to it:
  an affine map applied identically to both arrays cannot change whether a test value leaves
  the training range.
- **D48 is not touched, and it is the load-bearing half of the claim.** `rstd` at 5,000 times
  its calibrated threshold still alarms on **15.17%** of nominal steps against the
  forecaster's **0.6820%**, and that measurement does not depend on the envelope definition.

### Named rather than omitted
- The only within-channel test available is **n = 9. UNDERPOWERED (D3)**, and no conclusion
  is drawn from it.
- What would settle it does not exist in SMAP/MSL.

## [0.6.27] - 2026-09-10 - D65: the freeze is superseded, and the derivative is the finding

**D62's freeze is superseded and nothing is adopted.** Zero bucket operations; no fit; weight
store **1,313 -> 1,313**. The entry is written from the two reads already recorded at
`docs/MODELS.md` 38.15 -- artifacts `runs/smap-msl/_forensics/2026-09-10T173015Z-arms.json`
and `...T182234Z-arms2.json` -- and adds no measurement of its own.

### Decided
- **D65.** Arm 2, the smoothed residual fused with the first derivative of the raw value,
  both standardised against a trailing window only, reaches **EVAL 17 of 19 against the
  frozen arm's 4** at the frozen arm's own **0.6820%**, a **strict superset**, reproduced
  identically on two independent reads. **n = 19, UNDERPOWERED (D3).** The all-38 figure of
  30 is **CONTAMINATED** and is labelled so wherever it appears.
- **D62 is superseded and kept.** Its finding -- nine arms failed to beat stage 4 -- stands
  as the record of what was measured then; its freeze does not survive a tenth arm.
- **Nothing is adopted.** D65 re-opens the decision layer and names no flight configuration.

### Named rather than omitted
- **P2.3 is NOT ADJUDICATED.** Neither artifact carries the per-event causal `z`, so the
  prediction cannot be settled from what was measured. **36.4's second rider is therefore
  not triggered and stays owed.**
- Arms 4a, 6 and 7 remain unadjudicated and the union unmeasured, as 38.15 already records.
- The affiliation metric authorised under `docs/HARNESS.md` 5a is **not yet built** and has
  been applied to none of these figures.

### Corrected in the live documents, records left alone
- `docs/STATUS.md`: five live-prose sites gave the frozen arm's rate as **0.6838%**, which is
  the **commanded** arm's rate at 6/38 (`docs/MODELS.md` 26.18). Corrected to
  **0.6820% (was 0.6838%)** in the house form. Section 4's stage-4 table row is a
  measured-at-the-time record and is **not** touched.
- `docs/STATUS.md` operations line: 214 Class A / 4,415 Class B -> **222 / 4,966**, read from
  the last artifact.
- `docs/STATUS.md` verify block: `check_no_list` states **78 files** (was 75), which is what
  the script reports today.

## [0.6.26] - 2026-09-10 - Second read: pruning and CUSUM close as negatives; three arms still will not run

**Work item 9.21, second read. Both approved reads are now spent.** D62's freeze is NOT
lifted here; a draft entry that would supersede it is prepared and **not committed**.
**One read of 165 Class B and 1 Class A** after two 11 Class B smokes whose projected cost
again matched actual exactly; cached weights, **weight store 1,313 -> 1,313**. Artifact
`runs/smap-msl/_forensics/2026-09-10T182234Z-arms2.json`.

### Repaired, and the repairs settled two arms
- **Arm 1's dial is floored at 1.0 so it can only tighten.** The first read's 20/38 and
  21/38 at `p = 0.16` and `p = 0.20` came from dials of 0.5099 and 0.5045 -- **below 1.0,
  therefore re-admitting steps pruning had deleted**. With a legal dial they collapse to
  10/38 and 8/38, and at `p = 0.13` the arm is **identical to the frozen arm event for
  event**. **P1.1 fails; pruning is closed as a lever**, and the first read's diagnosis is
  now a measurement.
- **Arm 4b's reference slack is `CUSUM_K = 3.0`**, so the in-control drift is negative as
  B&N's derivation requires. It **matches the rate exactly and catches 0 of 38**.
  **P4b.1 fails.**

### Confirmed on an independent run
- **Arm 2 reproduces exactly**: 0.6820%, TUNE 13/19, **EVAL 17/19**, and the EVAL event
  set is **identical** to the first read's. Arm 5 likewise at EVAL 10/19, bounded and
  unbounded still identical in every cell.

### Still not adjudicated
- **Arms 4a, 6 and 7.** Arm 4a's rescored run-length statistic still leaves no cut that
  admits the budget; Arm 6's ACI, given the 1,000-sample buffer floor `docs/PHASE2.md` 5b
  argues for, returns 17.88% and pins `alpha_0` at its bound; **Arm 7 inherits both** and
  never descends to the budget. **No number of theirs is reported.** The union is
  **unmeasured** and no heterogeneity claim is made.
- Repairing them needs a third read, which is not taken. Month: **222 Class A and
  4,966 Class B of 50,000**.

## [0.6.25] - 2026-09-10 - The derivative stream reaches 17 of 19 at the frozen arm's own rate

**Work item 9.21, first read. Four arms adjudicated, three NOT, Arm 7 not run. D62's
freeze is NOT lifted.** `docs/MODELS.md` 38.15. **One read of 165 Class B and 1 Class A**
after three smokes whose projected cost matched actual exactly at every size; cached
weights, **weight store 1,313 -> 1,313**. Artifact
`runs/smap-msl/_forensics/2026-09-10T173015Z-arms.json`; producer
`scripts/decision_layer_arms.py`, in this commit.

### Measured
- **The reproduction gate passes**: the frozen arm rebuilt independently returns
  **0.6820% and 10 of 38**.
- **P2.1 holds decisively.** Residual plus first derivative, both standardised on a
  **trailing** window, reaches **EVAL 17 of 19** against the frozen arm's 4 -- +13 where
  the band asked +3 -- at the same 0.6820%, and it is a **strict superset**: it loses
  nothing the frozen arm caught.
- **P2.2 holds.** Four of EVAL's five invisible-in-residual events are caught. **36.4's
  finding is reached causally**, which 37.4 had explicitly said was unproven.
- **P2.4 refuted in the informative direction.** Dropping horizon disagreement is
  **better** -- 30/38 against 25/38 fused -- consistent with 37.4's 10-of-10 for the
  derivative against 4-of-10 for the disagreement.
- **P5.1 holds** (Arm 5 POT, EVAL 10/19) and **P5.5 holds at its strongest**: the bounded
  peaks set is **identical to the unbounded fit in every cell**, so the price of
  flight-legality is **zero events** on this data.

### Not adjudicated, and reported as such
- **Arm 1** is not the clean negative result it was registered as: a multiplier below 1.0
  **re-admits pruned steps**, because `channel_ratios` maps a suppressed step to
  `raw/(1+raw)`. So tightening `p` and lowering the dial trades pruning for a different
  rule rather than isolating the lever, and **38.3's separation of lever from dial does
  not hold for this score shape**. To be re-registered with a dial that cannot cross 1.0.
- **Arms 4a, 4b and 6 did not run cleanly** -- a scoring defect, a reference slack that
  left the CUSUM with positive in-control drift, and a trailing buffer too short to
  support a 0.68% quantile. **None of their numbers is reported as a result**, because a
  defective implementation is an unrun arm and not a losing one.

### Unchanged
- **D62's freeze stands.** The 30/38 figure is scored on all 38 including the 19 the
  parameters were selected on and is **reported as contaminated**, as 38.10 requires. The
  clean number is EVAL's 17 of 19. Nothing is adopted.

## [0.6.24] - 2026-09-10 - The size cap is raised to 8 MiB; the format refusals are untouched

**Infrastructure only. Zero R2 operations, no arm ran, D62's freeze stands.** Landed as
its own commit before the scoring runs so a size failure cannot interrupt one.

### Changed
- **`tests/test_no_local_persistence.py`'s tracked-size assertion: 4 MiB -> 8 MiB** (D64).
  Tracked content was **3.67 MiB, 91.8% of the cap**, and the five largest files are all
  prose; `docs/REORG_PLAN.md`'s tranches 3 and 4 are entirely additive and would not fit.
- Its docstring is rewritten to claim what the assertion actually does: **a coarse backstop
  against bulk, not the guard against data.** It also replaces the stale "784 KB was the
  agreed size", which had contradicted its own 4 MiB assertion.

### Unchanged, deliberately
- **The three format refusals, which are the part that matters.** Parquet, pickles and
  archives stay refused **everywhere including `runs/`**; `.npy`/`.npz` stay permitted only
  under `runs/`; the counted set and the gitignored set are still asserted disjoint.
  **A byte count cannot distinguish a dataset from a document; a suffix can**, and every
  real instance this project has reasoned about was caught by kind, not by size.
- **Rule 1 itself.** Nothing about what may be committed changes; only how many bytes of
  prose may be.

## [0.6.23] - 2026-09-10 - Seven arms registered from sources read at first hand, and none of them runs

**Work item 9.21. NOTHING RAN and nothing is adjudicated.** D62's freeze stands, weight
store untouched at 1,313, **zero R2 operations**. `docs/MODELS.md` 38.

### Added
- **`docs/MODELS.md` 38**, pre-registering Arm 1 (pruning, demoted to a registered
  negative result), Arm 2 (causal normalisation plus the derivative and
  horizon-disagreement streams), **Arm 4a** (persistence by run length), **Arm 4b**
  (CUSUM), **Arm 5** (peaks-over-threshold with an EVT tail -- the priority arm),
  Arm 6 (adaptive conformal inference), Arm 7 (the union) and **the stride as its own
  item**. Twenty-eight numbered predictions with hold/fail/no-verdict bands and a stated
  falsification per arm, all on 37.8's committed 19/19 split, all UNDERPOWERED and stamped.
- **Arm 4 split into 4a and 4b**, because a run-length rule and a CUSUM accumulator are two
  mechanisms with two levers and the standing rule is one lever per arm.

### Read at first hand
- **Siffer et al., KDD 2017** (DOI 10.1145/3097983.3098144), now **Primary**. `WebFetch`
  cannot decode the PDF and the machine has no PDF tooling, so it was decoded with a
  stdlib font-aware reader that resolves every glyph through the `/ToUnicode` CMap of the
  font active when it was drawn. **This mattered:** a first attempt with one global
  ligature map **silently rendered `sigma` as the "fi" ligature**, because code `0x1b` is
  `sigma` in the maths fonts and `fi` in the text fonts. The map was verified on exactly
  that point before anything was read from it. Residual ambiguity is **81 glyphs of 46,628
  (0.17%), named and not inferred** -- and in both transcribed formulas every one is from
  `txexs`, the extension font of large delimiters, so **no variable is missing**.
  Recovered: Theorem 3.1, equation 1, Algorithm 1, the Grimshaw reduction, the choice of
  the initial threshold, and DSPOT's trailing detrender.
- **Equation 1 was cross-checked against an independent statement of the POT return level**
  and agrees algebraically. The decode and the second source do not disagree.
- **Basseville and Nikiforov 1993**, chapter 2 section 2.2, pp. 35-41 -- the read source
  for CUSUM: the intuition (2.2.1-2.2.4) and the recursive form
  `g_k = (g_{k-1} + s_k)^+`, `g_0 = 0` (2.2.9), alarm at `g_k >= h` (2.2.10). One read
  source discharges what four paywalled ones were wanted for.
- **Hundman et al. section 4.3** -- already recorded in 0.6.22 and now load-bearing for
  Arm 1's registration.

### Recorded
- **(!) A correction to the commissioning brief, and it is an attribution.** The brief said
  Siffer's paper sanctions bounding the peaks set to a fixed size. **It does not** -- it
  says only that storing "only the peaks" needs little memory. **The bound is this
  project's own departure**, registered as an engineering decision required by
  `Objective.md` 11 rule 5 and F' CPP-1, and **measured rather than assumed**: P5.5 reports
  the bounded variant against the unbounded one at the matched rate, so the price of
  flight-legality is a number.
- **Two sentences of Siffer's carry Arm 5's flight case under D63.** 4.2: the initial batch
  *"is not labeled and is not considered as a ground truth ... The initialization is more a
  calibration step."* 4.2.1: **"The anomalies are not taken into account for the model
  update."** SPOT withholds what it has flagged from its own peaks set -- the D63 boundary
  satisfied by construction in the published method. And 4.2.1's sanction of **batched
  offline updates** maps onto this project's `stride`, so the flight design is grounded in
  the source rather than in convenience.
- **B&N's own caution changes Arm 4b's design:** the average run length *"is difficult to
  compute for most of the practically relevant change detection problems"*, so `h` is
  **bisected on nominal data** rather than inverted from an ARL formula.
- **The union budget is fitted jointly, not allocated** -- `oscfar.py:80-84` already
  records this project making the other half of that error: *"it is wrong, and it is kept
  because it is what ran"*.
- **Page 1954 is cited but unread**, with Basseville and Nikiforov as the read source.
  Lorden, Moustakides and Pollak are framing only. Nelson, Western Electric, Hawkins and
  Quesenberry stay **deferred with their slot registered**, deliberately not chased for an
  arm predicted inert.

## [0.6.22] - 2026-09-10 - Governance before the arms: rule 1 adjudicated, and a metric authorised

**Work item 9.21, governance half. NO ARM IS REGISTERED YET and nothing ran.** D62's
freeze stands, the weight store is untouched, **zero R2 operations**. These four changes
alter the rules the next pre-registration is written under, so they land before it rather
than alongside it.

### Added
- **D63 -- rule 1 governs the model and its weights, not the alarm threshold.** The
  conflict was real and unreconciled: `docs/RESEARCH.md` rejected SPOT because *"SPOT
  refits online and Objective.md 11 rule 1 forbids that outright"*, while **the frozen
  pipeline does the same thing** -- D62 froze stage 4 on telemanom's published *dynamic*
  threshold, which recomputes its cut every `stride` steps from a trailing window of the
  stream being scored (`telemanom.py:397-411`; `error_window = 0.05 * len(test)`,
  `stride = 70` under `proportional_config`, D47). On the strict reading, the pipeline this
  project selected could not fly under its own permanent rule. Adopted: the threshold is a
  **measured noise floor** (`docs/HARNESS.md` 1), and one measured from nominal data is not
  online learning. The boundary is stated so it can be applied: what rule 1 forbids is any
  path by which the detector's notion of normal is updated from data it has not been told
  is normal -- which makes an anomaly entering the trailing window unlabelled a
  **measurement question per arm**, not a licence. `Objective.md` 11 is not edited; its
  text already says "model".
- **`docs/RESEARCH.md` Part V** -- Kim et al. (AAAI 2022), Huet et al. (KDD 2022) and
  Gibbs & Candes (NeurIPS 2021), each read at first hand and each marked for what was
  actually read. Plus the two source sets deliberately **not** obtained, with the reason.

### Changed
- **`docs/HARNESS.md` 5a gains its fifth authorised addition and second-ever metric**:
  affiliation precision/recall (Huet et al., arXiv:2206.13167), escalated under D8 rather
  than added. Additive only; no existing metric changed; the gate stays event-wise F0.5
  (D3, D9); existing artifacts stay byte-identical.
- **`docs/HARNESS.md` 1 now discloses that a metric this project reports is gameable.**
  Huet et al. show *"an adversary algorithm can reach high precision and recall on almost
  any dataset under weak assumption"* under the recent event-based metrics -- the family
  the range-based pair in `metrics/eventwise.py` belongs to. **Nothing is withdrawn**; it
  is no longer reported alone. Two further facts a reader needs are recorded with it: the
  pair is fixed at `alpha = 0.0` and `bias = "flat"`, and `eventwise.score` does not plumb
  `alpha` or `cardinality` through, so they are locked at their defaults in **every**
  scored run this project has published.
- **Point-adjusted F1's quarantine gains its primary citation** -- Kim et al.: *"even a
  random anomaly score can easily turn into a state-of-the-art TAD method"*. It had been
  quarantined on Wu & Keogh and Objective.md 9.5 alone.
- **`docs/RESEARCH.md`'s Siffer entry is corrected in the house form, old text kept.** Its
  conclusion -- prefer the static variant -- may still be right; its stated *reason* was
  wrong. It also now says plainly that the entry is **Secondary**, that a whole method was
  set aside on an unread reading, and that nothing further is claimed about SPOT's
  mechanism until the paper is read at first hand.

### Recorded
- **(!) Hundman's own paper says `p` is tuned against labels.** Read at first hand,
  section 4.3: *"The `p` parameter is an important lever ... an appropriate value can be
  inferred **when labels are available**. In our setting, reasonable results were achieved
  with `0.05 < p < 0.20`."* A deploying mission has no labels (`docs/HARNESS.md` 6b). And
  37.7a measured that **all 13 pruned-and-recoverable events would be retained at a `p`
  inside that published band** -- max 0.1137, median 0.0834. They are lost to one value the
  author calls reasonable while another equally reasonable value keeps them.
- The 38.6 / 4.8 figures are the arithmetic of the two Table 2 rows already at
  `docs/MODELS.md` 26.19; they had never been stated in points anywhere here.

## [0.6.21] - 2026-09-10 - A1, A5 and A6: the pruning curve, and every caught event is late

**Work item 9.20, diagnosis half. NO ARM RAN and nothing is adjudicated; D62's freeze
stands.** `docs/MODELS.md` 37.7a. **One 7 Class B smoke, then one read of 165 Class B and
1 Class A**, cached weights, **weight store 1,313 -> 1,313**, 54.4 s. Artifact
`runs/smap-msl/_forensics/2026-09-10T041723Z-probe.json`; producer
`scripts/decision_layer_probe.py`, landed in this commit.

### Added
- **`scripts/decision_layer_probe.py`**, which records the pruning **ladder** per
  deciding window -- the quantity the forensic could not answer A1 from -- plus emission
  timesteps and the pooled nominal rate over all 81 channels.
- **The reproduction gate passes exactly.** At `p = 0.13` the probe returns 10 caught at
  0.6820% and the caught set is **event for event identical** to 36's.

### Measured
- **A1.** All 13 pruned-and-recoverable events are deleted at `p = 0.13` **by wide
  margins**: the most nearly retained needs `p <= 0.1137`, the median `0.0834`, and three
  need below `0.05`. Recovering all 13 needs `p <= 0.0058`, which is pruning switched off.
- **A5.** Swept at the frozen multiplier, **no `p` catches more than 10 at or below
  0.6820%**. Loosening to 0.10 buys +3 events for a **31% relative rise** in the alarm
  rate; to 0.08, +9 for +67%; `p = 0` reaches 25 of 39 at **8.06%**, twelve times the
  flyable rate. **This is the raw trade-off at a fixed multiplier and NOT the matched-rate
  comparison Arm 1 registers** -- the ladder depends on `eps`, so re-solving the
  multiplier cannot be done from cached ladders. Arm 1 still has to run, against a prior
  that is now quantified and unfavourable.
- **A6, and it is the sharpest new fact. Not one of the ten caught events is detected
  before its labelled onset**: median **-63.5 timesteps**, 0 of 10 positive. The lateness
  is substantially **structural** -- the detector emits at the end of the stride segment
  carrying the crossing and `stride = 70`, so seven of ten sit inside one stride and the
  median is nine-tenths of one. Reducing the stride costs compute, not detection.
  Recorded, not registered as an arm. No wall-clock figure is derived.
- **Every figure above is UNDERPOWERED (D3)**: n = 13, n = 10.

### Corrected
- **(!) 37.5's account of the scaling confound was wrong about the mechanism, and the
  old text is kept beside the correction.** It implied the published `(-1,1)` pre-scaling
  created the confound. Settled from source and from the arrays:
  `third_party/telemanom/telemanom/channel.py:69-82` `load_data()` calls `np.load()` and
  **there is no scaler anywhere in the package**; and `S-1` has train
  `[-0.400000, +1.000000]` against test `[-1.000000, +1.000000]`, so a separately-fitted
  train scaler is ruled out. There is **one affine map per channel, fitted on the test
  split and applied to both arrays.** **D46's arithmetic is therefore IMMUNE, not merely
  unchanged**: a min/max range test is invariant under an affine map applied identically
  to both arrays. **The confound survives and its mechanism is different** -- it is which
  channels carry which class, not how the data was scaled.
- **(!) `0.6838%` corrected to `0.6820%` in `docs/REORG_PLAN.md` and
  `docs/reorg_plan.json`**, wherever it stood as the frozen arm's rate, in the house form
  with the old figure kept. It had reached the public reading order's claim-to-evidence
  table, which is where it would have become public material.

## [0.6.20] - 2026-09-09 - The decision layer: three arms registered, none run

**Work item 9.20. Research and pre-registration only. NOTHING RAN and no arm is
adjudicated.** `docs/MODELS.md` 37. **Zero R2 operations**, weight store untouched at
1,313. **D62's freeze stands** and stays standing until an arm beats stage 4's 10 of 38 at
a matched rate.

### Added
- **`docs/MODELS.md` 37**, the pre-registration: the deepened diagnosis (37.1-37.7), the
  committed channel-disjoint split (37.8), and three arms with numbered predictions,
  hold/fail/no-verdict bands and stated falsifications (37.9-37.13). 37.14 is the OBSERVED
  heading, reserved and empty so an outcome lands beside its prediction.
- **The split, committed before any sweep**: 25 channels carrying the 38 in-range
  contextual events, ordered by a stated rule and assigned to two channel-disjoint halves
  of **19 events each**. **Both halves are UNDERPOWERED (D3, n < 20)** and are stamped so
  wherever quoted; 38 events cannot produce two powered halves and the pre-registration
  says so rather than discovering it later.
- **The pruning mechanism, read from `third_party/telemanom/telemanom/errors.py:386-435`
  and stated as a rule**: a candidate at rank `r` survives if and only if
  `max(drop_i : i >= r) >= p`, because the removal list is RESET at `errors.py:414`
  whenever a normalised drop reaches `p`. So what survives is the trailing run of the
  ladder after the last such drop, and the ladder's last rung is `non_anom_max`, the
  channel's largest nominal error. **An in-range contextual event fails it because it is
  in range**: its error is modest, it lands close to the nominal maximum, and pruning asks
  whether an error stands out rather than whether it is large.

### Recorded
- **(!) The frozen arm's own nominal alarm rate is 0.6820%, not 0.6838%.** 26.18's table:
  `gru` at multiplier 0.551 gives 0.6820% and **10/38**; `gru+cmd` at 1.000 gives 0.6838%
  and 6/38. The two are the matched pair V24 compared, and 26.18 calls them "the same
  rate" -- but every arm in 37 matches at **0.6820%** and reports 0.6838% beside it.
- **(!) 36.4's z-scores are not causal, and 36.4 does not say so.** The statistics are
  (`dx[t] = |x[t] - x[t-1]|`; the ten predictions OF `t` are each made from a window
  ending at or before `t-1`), but the normalisation is against `nom`, every non-event
  timestep of the **whole test array** (`scripts/smap_forensics_38.py:191-196, 236-239`).
  36.1 stamps the oracle "not implementable" for the same reason; 36.4 should carry the
  same rider. Arm 2 may not use those figures as its commissioned number and R2.4 tests a
  re-derived causal statistic instead.
- **(!) The derivative carries ten of ten; the horizon disagreement carries four.** 36.4's
  "or both" is exact and the work is being done by the derivative. R2.3 is written on it.
- **(!) A confound in the in-range contextual population, measured for the first time.**
  `third_party/telemanom/README.md:94` records that SMAP/MSL is pre-scaled to `(-1,1)`
  **by the min/max of the TEST set** -- which this repository had not recorded. Measured
  consequence: channels carrying contextual events have a training envelope spanning a
  median **2.000 of the 2.0 available** (30 of 39 span `[-1,1]` exactly), against
  **0.173** for channels carrying point anomalies. **D46's arithmetic is unchanged at
  39/43**, but its 72.7-point gap over the point class is confounded with envelope width
  and may not be read as a property of the anomalies alone. The only within-channel
  comparison available is 4 channels and 9 events -- UNDERPOWERED, and no conclusion is
  drawn. **D48 is untouched**: `rstd` at 5,000x still alarming on 15.17% of nominal steps
  does not depend on the envelope definition.
- **(!) The denominator is 38 rather than D46's 39 because one channel would not train.**
  Stage 4's `excluded_channels` lists `G-1` with "training kept its FIRST epoch after
  running 11" -- **D17's stall signature** -- and `G-1[4770-4890]` is the only in-range
  contextual event on the four excluded channels. The 38 is a **scorable** population, not
  the labelled one.
- **Label provenance, read from the paper.** Hundman et al. section 4.1 "Setup": anomalies
  come from **Incident Surprise Anomaly reports**, and "all telemetry channels discussed in
  an individual ISA were reviewed ... and specific anomalous time ranges were manually
  labeled." `third_party/telemanom/README.md:117-124` gives 105 sequences from **47 unique
  ISAs** across 82 channels, 43 of them contextual. Destined for
  `docs/datasets/SMAP_MSL.md`.
- **Six places the commissioning brief disagreed with this repository**, in 37.1, the
  repository winning in each -- including that only **2 of the 5** below-threshold events
  are "just under" it (3.17% and 4.73% short; the other three are 39.71%, 46.81% and
  63.30% short), and that **2 of the 10** "invisible" events had in fact cleared their
  dynamic threshold and were pruned.

### Owed
- **A1, A5 and A6 are specified and NOT RUN** (37.7): the smallest `p` that retains each of
  the 13, the pooled alarm rate beside every recall figure, and lead time in timesteps for
  each caught event. None is reconstructible from a committed artifact -- the forensic
  keeps each event's peak and threshold but **not the pruning ladder**, and retains **no
  alarm timestep**. Cost stated before spending: **165 Class B and 1 Class A**, one bundle
  load, no fits, after a **7 Class B** smoke. Month stands at 214 Class A and 4,415 Class B
  of 50,000 each.

### Changed
- `docs/STATUS.md` 7's B4 paragraph, which said "Nothing is registered against this yet",
  now names the three registered arms and restates that D62's freeze stands.
- `docs/MODELS.md` 36.6 gains a rider pointing at 37. **Its original sentences are kept
  verbatim**; a record corrected in hindsight is not a record.

## [0.6.19] - 2026-09-09 - Per-event forensics: 18 of 28 misses are the alarm rule's fault

Work item 9.19, diagnosis only. One read at 55 Class B and 1 Class A, cached weights,
weight store unmoved. `docs/MODELS.md` 36, `scripts/smap_forensics_38.py`.

### The classification, rule fixed before any event was read

```
   caught                    10 / 38
   lost in the decision      18 / 38     an ORACLE per-channel threshold at the frozen
     layer                              arm's OWN quiet rate would catch these
   invisible in the          10 / 38
     residual
```

**An oracle decision layer would reach 28 of 38 on residuals that already exist**, against
the frozen arm's 10. That is a ceiling, not a proposal: the oracle is per channel and
label-free in construction but not implementable.

### Which stage kills them

```
                              below-threshold   pruning
   lost in the decision layer        5            13
   invisible in the residual         8             2
```

**Pruning is the single largest killer** -- 13 of the 18 recoverable events clear their own
dynamic threshold and are then discarded by `prune` at `p = 0.13`. D50 named pruning as the
first lever six days ago and 29.4 named it again as the last standing candidate; this is the
first per-event evidence for it. **Warm-up killed nothing.** The magnitude conjunct,
whole-window bail-out, coverage cap and sequence cap are mechanisms of the port and **not of
the frozen path**, so they could not have killed anything here -- reported as absent rather
than omitted.

### (!) The ten "invisible" events are not invisible in the data

**All ten show z > 3 in the first derivative of the raw value, in the disagreement across
the ten predicted horizons, or both.** `E-13[5600]` has a residual peak *below* its
channel's nominal mean -- the forecaster predicts it better than it predicts normal data --
and its derivative is 4.17 sigma out; `F-3[5600]`'s derivative is three orders of magnitude
out. **The residual as constructed discards signal that is present in the input**, which is a
statement about what the decision layer is fed rather than about the network's capacity.

### Step-likeness is closed as an explanatory variable

7/10 caught, 9/18 decision-layer, 6/10 invisible, cut at the median across the 26 channels.
No signal -- consistent with 35.7, where trees lost both halves of the same split.

### Nothing registered

D62's freeze stands. This is diagnosis; no arm exists against it and none will until one is
pre-registered.

---

## [0.6.18] - 2026-09-09 - The ladder closes: nine arms, none beats stage 4, and the pipeline freezes

Work items 9.17 and 9.18. Four reads at 165 Class B and 1 Class A each plus two smokes;
weight store 1070 -> 1313. `docs/MODELS.md` 34.8, 35.7, D61, D62.

### The ladder, at each arm's own operating point

```
   arm    what                                nominal%   recall     irc on stage 4's 38
   stage4 gru-telemanom, swept   (the target)  0.6820    47/100           10/38
   WS     34.8's label-free winner, swept      0.6828    30/104            8/38
   C1S    C1, the command-free port            0.6021    31/104            6/38
   S3     3-seed ensemble                      0.0000     2/104            0/39
   S4C    isolation forest, the control        0.1195     1/104            0/39
   S1 transition floor / S2 per-channel / S12 / S4 trees -- NO operating point at target
```

### Measured

- **P1 and P2 refuted.** The label-free forecaster winner reaches **8/38**, below stage 4's
  10, and 30/104 against C1S's 31.
- **S1a, S1b, S2a, S12a have NO VERDICT**, and that is the structural finding: four arms have
  **no operating point at or below 0.6838% at all**. The dial was extended to `1e-8`, which
  saturates at the training maximum, and the quietest points are 2.01%, 13.55%, 21.49% and
  5.17%. **This is D48 arriving for the fifth through eighth arms**, and 35.1/35.2 cite D25
  without reckoning with the entry that bounds it -- a defect in the pre-registration, not the
  code, and recorded as mine.
- **S3a refuted, and it is the sharpest negative.** Averaging three seeds' predictions
  smooths the residual enough that a train-calibrated quantile **finally transfers** -- and
  the result is 2 events at zero false alarms. **A better forecaster and a silent detector**,
  which is D18's shape again.
- **S4a refuted, S4b held, and together they kill the hybrid.** Trees lose the steppy half
  7/54 to 27/54 **and** the smooth half 10/50 to 29/50, while running louder at 5.17% against
  2.22%. No hybrid is built and no cut is searched for.
- **S4c held.** The isolation-forest control reaches 1/104 against trees' 17 and the GRU's 56.

### Added - a result for the toolkit even though the arm loses

Trees fit in **2.23 s median per channel against the GRU's 43.3 s mean** -- about 20x
cheaper, at 432 MB against 642 MB peak. For a mission training on the ground without ML
staff that ratio matters and is recorded for work item 12.

### Added - D62, the freeze

**The pipeline freezes on stage 4's configuration**: `gru-telemanom`, per channel,
univariate, no commands, under telemanom's published nonparametric dynamic threshold swept to
0.6838%, catching **10 of 38**. **The C++ port must carry the dynamic threshold** -- it is the
only rule measured to reach a flyable rate under regime shift, and `flight/` transcribes
D25's static quantile alone. **S1 and S2 are not adopted.**

### (!) Two instrument defects, both mine, and the fourth of a kind

The first S-run printed `nominal 0.0000% caught 0` for every arm that found no qualifying
operating point, because the print read `best[1] if best else 0` -- **a placeholder rendered
as a measurement**, beside arm rows showing 2% to 21%. And it retained neither per-channel fit
cost nor step-likeness though the builders computed both, so 35.4's own reporting requirement
could not be met from the artifact. **Fourth time**: 27.8's alarm ranges, 32.7's `sigma_cv`,
33.7's caught sets, now these. A value computed and not retained is a value that does not
exist, and it cost an extra read to learn again.

### Dependency

`scikit-learn==1.9.0` pinned. `lightgbm` was installed and removed: its wheel needs
`libomp.dylib`, absent here, and supplying it means a Homebrew system install outside the
project for a ground-side study. `HistGradientBoostingRegressor` is the same algorithm class
with no native dependency.

---

## [0.6.17] - 2026-09-09 - The dial: C1's advantage does not survive a flyable alarm rate

Work item 9.16, pre-registered the same day. **Two** reads at 165 Class B and 1 Class A
each, no fits, weight store 558 -> 558 on both -- the second to resolve a crossing the
first grid could not see. `docs/MODELS.md` 33.7 and 33.8, D61.

### Added - D61, C1 as the ladder's base

C1 is the base every improvement arm reports against, and the ladder is bounded at three
arms in advance so that one more idea is a new pre-registration rather than a continuation.
**The ladder is label-free where it selects**: arm 2 chooses a forecaster only on held-out
nominal validation error, because selecting on recall would be choosing the model that best
fits 104 labelled sequences. A base is not an endorsement; C1 flies nowhere.

### Measured

- **M3 HELD**, zero inversions across 24 grid points, so "matched rate" is well defined for
  this arm -- which is what M3 existed to establish before any comparison was drawn.
- **M1 and M2 REFUTED at the measured point.** Swept to 0.4465%, C1 catches **6 of stage
  4's own 38** in-range contextual sequences against stage 4's **10/38**. **Stage 4's 10/38
  stands unreplaced and the front page does not change.**
- **K2, owed since 32.7, is discharged and HELD.** It predicted C1's in-range contextual
  count below Arm T's 12/39 at a matched rate; it is 6. It held for a reason 32.4 did not
  anticipate: not because the commands were load-bearing, but because **C1's whole advantage
  lives at an alarm rate no mission would fly.**
- **M4 refuted at three.** MSL contributes 3 catches against a predicted 2 or fewer. The
  ratio is the interesting part: SMAP halves as the rate falls fourfold, 50/68 to 25/68,
  while MSL moves 4 to 3.

### The crossing, resolved the same day (33.8)

A second read at 165 Class B, no fits, re-adjudicated M1 and M2 on a two-stage grid -- the
24 coarse points kept so the curves compare, **[1.80, 2.10] resolved at 0.01**, and the
**caught set retained at every point** rather than a count of it, so any population
restriction is computable from the artifact forever. **M1 and M2 refuted robustly**: the
best in-range contextual count at or under the target is **6 against stage 4's 10**, and
even at a rate 10% **louder** than stage 4's, C1 reaches only 8. To reach 16 it needs twice
stage 4's alarm rate. **C1's advantage is a precision result and not an early-warning one.**

### (!) Why a second read was needed -- the grid could not see the crossing

The selection rule takes the best point at or below the target, and this grid's neighbours
straddle it -- **0.7578% and 0.4465% with nothing between** -- so C1 is compared **35%
quieter** than the arm it is measured against. At 1.865 it catches 34 total events against
28, so six sit in the gap, and the curve **did not retain the contextual breakdown per
multiplier**, so whether enough of them are contextual to flip the verdict cannot be
settled from the artifact. Whether C1 is behind at exactly 0.6838% is **unresolved**. A
refinement is owed and named, not run, and not folded into arm 2. Second time in two work
items that a curve was kept at lower resolution than the question needed.

### (!) A second lever in C1, found while building the dial, and worth nothing

C1 was built to differ from Arm T in exactly one lever and differed in two: the commands,
which was the point, and `tail`, which carries T-g's published target length. The dispatch
selected it **by arm name**, so Arm T carried it and C1 did not, and 32.2's own enumeration
of held-identical differences omits T-g while the sentence "nothing else changed" carried
it implicitly.

Corrected in the same run at no extra cost, and **worth exactly zero**: 54/104, 93.1%
precision and 4 false positives are identical, and only the nominal rate moves, 1.8815% to
1.8777%. 33.2.1 predicted "probably nothing" before measuring and said that was a
prediction rather than a reason to skip it. Both numbers kept; the condition is now keyed
by cell so a new published arm cannot be forgotten the way C1 was.

---

## [0.6.16] - 2026-09-09 - The commands were what was hurting, and K3's stop fires

Work item 9.15, pre-registered the same day. One read at 165 Class B and 1 Class A, two
3-channel smokes at 9 each, 33.3 min at 4 workers, weight store 405 -> 558.
`docs/MODELS.md` 32.7, D60.

### Measured - K3 first, as 32.4 requires

```
   arm                    channels   sigma CV median   above 0.25
   H   (no commands)          79          0.0504         11 / 79
   C2  (with commands)        78          0.1145         16 / 78
   K3 needs                                              more than half
```

- **K3 REFUTED and the pre-registered stop fires.** Commands more than doubled sigma's
  variation and did not come close to the bar. **Nothing after that line in C2 is reported
  as a result**, so K4 is not adjudicated. The likelihood trained on **78 of 78** fitted
  channels: the head could have learned a state-varying sigma from the commands and did not.
- 31.9's first candidate is closed. Two remain, both named before any of this ran.

### Measured - K1, refuted in the direction that matters

```
   arm   set     recall            precision (TP/(TP+FP))    FP    nominal
   T     Total   46/104  44.2%     46/55    83.6%             9    2.0450%
   C1    Total   54/104  51.9%     54/58    93.1%             4    1.8815%
   paper Total   84/105  80.0%     84/96    87.5%            12
```

- **K1 predicted C1 below 46/104; it is 54/104.** Withholding the commands **gains eight
  events, removes five false alarms and lowers the alarm rate at the same time**. C1
  dominates Arm T on every axis, so the comparison is not rate-confounded.
- K1's own refutation clause, written before the run, said this would put D49's finding
  **outside its own bound**. It does: commands fail to help under the published training
  too, and **D49 did not need the qualification it gave itself** on that axis.
- **C1 is the best arm on this population**: 93.1% precision against the paper's 87.5%, on
  **4 false positives against a scaled target of 11**, at 51.9% recall against 80.0%.
- **(!) MSL moves for the first time since the port was built, by one event** (3/36 to
  4/36), with its false alarm falling 2 to 1. **One event is not a finding** and is not
  reported as one; the pair is recorded because it moves together.
- **K5 HELD**: C1 fitted the same 81 channels and scored the same 104 sequences as Arm T,
  so narrowing the input did not change which channels train.

### (!) K2 has no verdict, and it is a defect in the instrument

32.3 pre-registered a sweep to a rate matched to stage 4's 0.6838% "for C1 through
`Mech.eps_mult`", and **that sweep was never implemented** -- C1 was registered as a `port`
arm, which has no dial. So the matched-rate figure K2 asks for does not exist. C1 catches
20 of 39 in-range contextual at its own 1.8815% and Arm T catches 12 of 39 at 2.0450%;
**both are above the ~1% ceiling 27.3 sets for quoting a per-event number**, so neither is
compared to the other or to stage 4's 10/38. The comparison is owed and unpaid.

### (!) Two instrument defects, one of which nearly cost the stop

- **`zscore_diagnostics` collected only the old cell**, so C2's `sigma_cv` would never have
  reached the artifact and **K3 would have been unadjudicable from the record** -- 27.8's
  defect exactly. Caught by the smoke, fixed before the read, and the second smoke exists
  to confirm it.
- The peak-RSS meter guessed its unit from the magnitude of `ru_maxrss`, bytes on macOS and
  kibibytes on Linux. Replaced with a platform test before it produced a number anybody read.

### The compute plan, measured against itself

The 3-channel smoke projected 143 min serial and 2.4 GiB at four workers; the 81-channel
run measured **101 min serial, 2.7 GiB, and 33.3 min wall clock**. The smoke over-estimated
time by 42% and under-estimated peak memory by 10%, **and the memory-gate arithmetic was
right this time because the term that killed 28.7 was measured rather than omitted.**

### Weight store

+153, not the pre-registered +162, and the difference is accounted for rather than
excused: six fits were already banked by the smokes and **three C2 channels stalled** --
`D-11` and `F-3` on D17's first-epoch guard, `D-12` with no complete sequence in its
training split. 162 - 6 - 3 = 153.

---

## [0.6.15] - 2026-09-09 - Dimensionless guards: MSL does not move, and G4's structural stop fires

Work item 9.13, pre-registered on 2026-09-08 and run today. One read at 165 Class B and
1 Class A, preceded by a 2-channel smoke at 7 Class B and 1 Class A. Cached weights,
weight store **399 -> 399**. `docs/MODELS.md` 30.4, D58.

### Added - Arm G, in the study script only

- Telemanom's two candidate filters replaced by `mean(e_s) + 1*sd(e_s)`, **the multiplier
  fixed in advance at 1.0 and not swept**. The coverage cap, the sequence cap and
  `sd_e_s > 0.05 * sd_values` are left exactly as published, because they are already
  scale-free and changing one would be a second lever.
- `src/sentinel_models/telemanom.py` is untouched (D8); Arm R is re-scored in the same
  load as its own control and **reproduces 29.4 in every cell**, which is the run's
  internal gate.

### Measured

```
   arm   set     recall            precision (TP/(TP+FP))    FP    nominal
   R     MSL      3/36    8.3%      3/5     60.0%             2
   G     MSL      3/36    8.3%      3/5     60.0%             2
   R     SMAP    43/68   63.2%     43/50    86.0%             7
   G     SMAP    45/68   66.2%     45/65    69.2%            20
   R     Total   46/104  44.2%     46/55    83.6%             9    2.0450%
   G     Total   48/104  46.2%     48/70    68.6%            22    3.7958%
```

- **G1 REFUTED. MSL moves by exactly zero events and zero false alarms**, against a
  prediction of recall past 15/36. Every point of movement is SMAP's -- the third
  consecutive arm of which that is true.
- **G2 HELD** at 2 MSL false positives, the paper's own figure. **G3 HELD** at 45/68 SMAP,
  above the 43/68 floor. So the arm works; what it does is not what MSL needed.
- **G4 REFUTED at 79 of 81 channels, and the pre-registered stop fires.** On `D-2` and
  `G-6` the dimensionless arm **loses** 140 and 350 steps the absolute arm had. On `G-6`
  the whole-window bail-out fires 2 times under the absolute floor and **64** under the
  dimensionless one.

### Added - D58, and it qualifies D55 rather than withdrawing it

- **A dimensionless filter is not uniformly looser than an absolute one.** It is
  channel-relative: looser where residuals are small, **tighter where they are large** --
  and 29.4 already measured that this data has both, `max(e_s)` on MSL being 0.7323 at the
  median. Replacing an absolute constant changes **which** channels a filter binds on.
- **In aggregate the arm is still a relaxation**: bail-outs 1,836 -> 996 of 13,995 windows,
  the magnitude conjunct 4 -> 0, the alarm rate 2.0450% -> 3.7958%. **A pooled statistic
  would have reported a relaxation and concealed the two channels.** G4 was written per
  channel, as a stop, before any number existed, and it is the only reason they are on the
  record.
- The implementation is **not adopted**. 30.2 pre-registered that adoption was decided on
  G1-G4 rather than inherited from D55, and it decided against. The multiplier stays
  unswept: needing another value is a finding about the form, and a swept multiplier is the
  alarm budget arriving in a new costume.

### Consequence - the last named candidate

With the residual rung inert (29.4), the scale hypothesis refuted (R4) and now the guards
moving nothing, **pruning at `p = 0.13` is what is left** for MSL's silence. It was named
in 29.4 and again in 30.2's own refutation clause, both before this ran.

### (!) An ambiguity in the pre-registration, recorded rather than absorbed

`errors.py:337-339` is a three-term test and 30.1's table named two of them. The middle
term, `max(e_s) > 0.05 * inter_range`, is not listed; it was **left as published** on the
literal reading, because a pre-registration is a literal instrument and moving an unlisted
lever is what rung 1c's stop fired for. 30.4 states which test was run, since G4's
adjudication depends on it.

---

## [0.6.14] - 2026-09-08 - `gru-zscore`: the head learned a constant, and its own stop fired

Work item 9.13 pre-registered and not run; work item 9.14 pre-registered and run at 165
Class B and 1 Class A. `docs/MODELS.md` 30, 31, D55, D56.

### Added - D55, from three measured instances and one refuted hypothesis

- **Absolute constants in data units do not transfer.** D17's `min_delta` disabling
  training on ESA-ADB's ~1e-4 loss; T4 measuring **the same constant correct** on
  (-1,1)-scaled SMAP/MSL; and the candidate filters at `errors.py:339` and `:343`. Plus
  one hypothesis measured and refuted, kept with its reasoning.
- **The toolkit ships dimensionless equivalents**; an absolute form is a per-mission
  override with its provenance attached. A requirement on the toolkit, not a finding
  about telemanom, whose constants are correct on telemanom's data.

### Added - work item 9.13, dimensionless guards: registered, not run

- `docs/MODELS.md` 30 replaces each absolute filter with `mean(e_s) + 1*sd(e_s)`, the
  multiplier **fixed in advance and not swept**. The coverage and sequence caps are left
  alone because they are already scale-free. G1 MSL past 15/36, G2 MSL false positives 6
  or fewer, G3 SMAP not below 43/68, G4 a superset on every channel **and a stop**.
- One read, 165 Class B, **no fits**. It has not run: this is the only section in
  `docs/MODELS.md` carrying a pre-registration and no OBSERVED.

### Added - work item 9.14, and it stops chasing Table 2

- **`gru-zscore`**: the same GRU with a head emitting `mu` and `log sigma^2` per channel,
  a Gaussian NLL loss on nominal data, and the statistic `z = (x - mu) / sigma` under
  D25's unchanged label-free threshold. **`z` is dimensionless by construction, so D55 is
  satisfied structurally** rather than by choosing better constants.
- The reproduction answered what it was asked. The gap is **not** the scoring rule, the
  commands, pruning's rung, cross-window tracking, the aggregation, the window regime or
  the training configuration -- each measured and closed. **It is the residual itself.**

### Measured - H4's stop fired, and the design claim is the thing that failed

Artifact `runs/smap-msl/_forensics/2026-09-08T220104Z-wi910-port.json`. 81 channels
attempted, **79 fitted**, 102 sequences, MSL 36 -- like-for-like with the paper.

```
   arm    set     recall            precision (TP/(TP+FP))    FP
   A0     MSL    16/36   44.4%      16/42    38.1%            26
   T      MSL     3/36    8.3%       3/5     60.0%             2
   H      MSL     5/36   13.9%       5/32    15.6%            27
   paper  MSL    25/36   69.4%      25/27    92.6%             2
```

- **H4 REFUTED and the pre-registered stop fired.** Sigma's coefficient of variation is
  **0.0504** at the median and above 0.25 on **11 of 79** channels, against a prediction
  of more than half. In 31.5's own words, written before the run: the head "learned a
  global scale, `z` is `|x - mu|` divided by a constant, and the arm is the old detector
  with extra parameters."
- **H1 REFUTED** at 5/36 against `A0`'s 16/36. **H2 REFUTED** at 27 false alarms against
  the paper's 2. **H3 not run**, as 31.6 pre-registered, and now moot for adoption.
- **H5 HELD on 79 of 79 channels**, median best epoch 34 of 35. The likelihood objective
  trained. **This is not an optimisation failure**: the model could have learned a varying
  sigma and did not.
- **D48 for the fourth arm running.** At D25's label-free threshold `gru-zscore` alarms on
  **19.41%** of nominal time; swept, it saturates at the grid maximum and is still 1.18%
  with 0 of 39 in-range contextual. **Stage 4's 10/38 stands unreplaced.**

### (!) Three defects in this arm, recorded rather than tidied

- **The weight store grew by 0, not the pre-registered +81.** `build_zscore` fits through
  `lstm.train` directly and the cache lives in `ForecastDetector.fit`, so Arm H's weights
  are never persisted and the arm is reproducible only from its seed.
- The relative early-stopping rule raised the bar on a **negative** loss. Fixed
  sign-safely, identical for every non-negative loss, pinned by test.
- **A Gaussian head is not representable in `model.bin` version 1** (D30): `Weights`
  refused a doubled head. Both were caught by the smoke before the read.

### Named, not registered

- Why sigma stayed constant. A single univariate channel gives the likelihood no reason to
  vary it with state; the multivariate channel set, a variance term the loss cannot
  trivially satisfy, and capacity are each one arm.

---

## [0.6.13] - 2026-09-08 - The published training buys eight events, and the residual rung buys none

Work items 9.11 and 9.12. Three reads at 165 Class B (one discarded), plus smokes at 9
Class B. `docs/MODELS.md` 28, 29, D55.

### Added - work item 9.11, Arm T: the published training configuration

Eight differences read line by line from `third_party/telemanom/` `modeling.py`,
`channel.py` and `config.yaml`, applied together because reproducing a configuration one
constant at a time would take eight reads. The largest is **T-a**: telemanom's
`aggregate_predictions` defaults to `method='first'` (`modeling.py:113`) and is called
with no method at `:172`, so its forecast is the **single one-step-ahead prediction** where
ours averages ten. **Ours smooths the residual tenfold.**

### Measured - T4 held with zero stalls, and it changed the population

Artifact `runs/smap-msl/_forensics/2026-09-08T201450Z-wi910-port.json`. Weight store
**321 -> 399**, the pre-registered growth.

- **Zero of 81 channels kept their first epoch** under the published absolute
  `min_delta = 0.0003`, against a prediction of 20 or fewer; our relative rule stalls six
  on the LSTM and four on the GRU. **Arm T is the first arm to score the whole
  104-sequence population**, the paper's 105 less the P-2 duplicate.
- **D17's replacement is necessary on ESA-ADB and unnecessary here** -- the
  dimensionless-constants argument holding in both directions on the same code, and a
  stronger statement than D17 could make alone. It is instance 2 of D55.

```
   arm     set      recall              precision (TP/(TP+FP))     FP
   F       Total   34/98    34.7%      34/38     89.5%              4
   T       MSL      3/36     8.3%       3/5      60.0%              2
   T       SMAP    43/68    63.2%      43/50     86.0%              7
   T       Total   46/104   44.2%      46/55     83.6%              9
   paper   MSL     25/36    69.4%      25/27     92.6%              2
   paper   Total   84/105   80.0%      84/96     87.5%             12
```

- **T1 REFUTED**: 46/104 (44.2%) against a 55-85 band. Restricted to Arm F's own 75
  channels the two share a population exactly, and **T is 42/98 against F's 34**.
- **The published training buys eight events and costs four false alarms** -- +8.2 points
  of recall for -5.5 of precision, at 83.6% precision against the paper's 87.5%.
- T2 and T5 held; S1 and S2 are now adjudicated at 75/75 on every rung. The magnitude
  conjunct bound on **23 indices in 83,780 window-passes**, settling L1 as a property of
  the data rather than a transcription defect.

### (!) MSL did not move at all

Every point of the gain is SMAP's. **MSL is 3/36 before and after.** On the one population
that matches the paper's exactly, the paper catches **25 of 36 with 2 false alarms** and
this reproduction catches **3 with 2**, and eight training changes moved it by zero events.

### Added - work item 9.12, Arm R: the residual rung

Arm T plus telemanom's residual over the supervised region only, with the first `l_s`
smoothed samples replaced by the mean of the first `2*l_s` (`errors.py:48-64`). Artifact
`runs/smap-msl/_forensics/2026-09-08T211109Z-wi910-port.json`, 165 Class B, weight store
**+0**.

- **Arm R equals Arm T in every cell.** R1 refuted: the residual rung is not the cause.
- **R2 refuted and its stop discharged on evidence.** The 13 `offset` events scatter from
  -503 to +427 with **0 of 13** within 250 +/- 40, so a frame error is ruled out and the
  classifier was loose. The same class fires at the same rate on SMAP, where detection
  works.
- **R4 refuted, and it refutes 28.8's own hypothesis, which was ours.** 28.8 argued a
  better forecaster produces smaller residuals that the absolute floors then filter out.
  Measured, the opposite: `max(e_s)` on MSL is **0.7323** at the median against a
  `sd(values)` of 0.5721, and **0 of 27** channels fall below the 0.05 floor. **The
  residuals are large, not small -- the forecaster is doing badly on MSL**, which is a
  different and more ordinary problem. R5 held at 0 events moved on SMAP.
- On MSL **no named guard fires** in the published window regime -- coverage 0, sequence
  cap 0, magnitude 0 -- so what silences 23 of 27 channels is downstream, in pruning at
  `p = 0.13`. An instrument flaw is recorded with it: the fallback counter double-counts
  the inverse pass, so 56% is nearer 12%.

### Process - a compute plan that was measured and still wrong

One published fit was measured at **508 MB peak RSS** and four workers sized at about
2.8 GB against 4.3 GB free. **The operating system killed the run for low memory, after
the fits and before the artifact.** Two costs were omitted from an otherwise honest
measurement: each spawned worker imports torch independently, and the parent had already
grown past its own baseline holding 81 channels of arrays. 77 of 78 fits survived on disk
because weights are written as each fit lands; **165 Class B were spent and never reached
the ledger**, which is commit `75cc846` from the other side. Corrected on 2026-09-08.

---

## [0.6.12] - 2026-09-08 - The public-benchmark survey, and the first outside confirmation of a finding here

Zero operations. `docs/RESEARCH.md` Part IV, verified citation by citation.

### Added

- **Pinet et al., arXiv:2606.02670, MiLeTS at KDD 2026**, *"Anomalies in Multivariate Time
  Series Benchmarks Are Mostly Univariate"*. Across eight widely used public benchmarks
  their diagnostic "shows that **no cross-channel rupture occurs without an accompanying
  univariate deviation across a range of reasonable thresholds**", and a
  channel-independent against channel-dependent comparison of a recent state-of-the-art
  detector "further confirms that CD modeling brings no measurable gain".
- **This is an independent replication of D42 and D23, the first this project has.** D42
  measured that no cross-channel reduction recovers anything `max` misses on ESA-ADB and
  D23 closed as answered no. Those were single-project findings on one benchmark; an
  outside group reaches the same place on eight.

### (!) What it does not say, recorded because it is the diagnostic

- **It does not say what our headline says.** Pinet test deviation from **normal history**;
  D46 tests leaving the **training min/max** a limit check actually holds. The two findings
  are compatible and the section says so rather than borrowing their authority.
- What could not be verified is marked as the survey's rather than quoted as the paper's:
  the identity of the eight benchmarks, the 373 long segments, and the
  strictly-cross-channel counts.
- **Two caveats travel with it**: a near-binary channel defeats a z-score diagnostic, which
  is a live concern for SMAP/MSL specifically, and absence across eight benchmarks is
  strong evidence rather than proof. SWaT, WADI and SKAB are named as unchecked candidates.

### Consequence

- **Phase 3's physics testbed is the only venue for the cross-channel claim**, not a
  convenience. If no public benchmark carries strictly cross-channel segments, the
  cross-channel and early-warning claims cannot be earned on one.

---

## [0.6.11] - 2026-09-08 - The faithful port: it reproduces the paper's precision and not its recall

Work item 9.10. One read at 165 Class B and 1 Class A, preceded by a 2-channel smoke at
7 Class B and 1 Class A. Cached weights, **weight store +0** on both. `docs/MODELS.md` 27,
D54.

### Added - twelve arms, one bundle load

`scripts/smap_rungs.py`, committed **with** the figures it produces (`docs/NARRATIVE.md`
11's rule). A0 the faithful `1a+1b` base, A1 to A5 the gates, L1 to L5 the ladder adding
one telemanom mechanism at a time with each citing the vendored source by line, F the
complete port as the reference ceiling and FG the same on the GRU.
`src/sentinel_models/telemanom.py` is not touched (D8).

### Measured - G1 refuted, and the way it failed is the result

Artifact `runs/smap-msl/_forensics/2026-09-08T182416Z-wi910-port.json`.

```
   arm  exercises                            measured            recorded
   A1   src/sentinel_models/telemanom.py     44/98,  45/59       44/98,  45/59     EXACT
   A3   stage 4's swept arm                  0.551, 0.6820%,     0.551, 0.6820%,
                                             47/100, 10/38       47/100, 10/38     EXACT
   A0   the lost rung script (1a+1b)         57/98,  62/110      59/98,  65/146    REFUTED
   A2   the lost rung script (1c-ii)         74/98, 100/237      74/98,  98/221    recall exact,
                                                                                   ranges REFUTED
```

- **The gate splits exactly along the committed/uncommitted line, and that was not
  predicted.** Both gates exercising code in this repository reproduce to the digit, which
  clears the bundle load, the weights, the scorable masks, the event populations, the P-2
  dedup, the D17 stall set and both scoring rules. **Both gates depending on the study
  script that was never committed do not.**
- **`docs/NARRATIVE.md` 11's finding arriving as a measurement rather than an argument.**
  The divergence is located and is a defect in neither implementation: the lost script
  built `1a+1b` on `channel_ratios`, this port on telemanom's own loop, and they differ in
  series-level singleton dropping and in `channel_ratios`' 315-step opening suppression.
  **Two faithful readings of the same prose differ by 2 events and 36 ranges.**
- Populations confirmed at 37 of 98 in-range contextual on the LSTM and 38 of 100 on the
  GRU, exactly as 26.30.3 predicted.

### Added - D54, the withheld arms released at zero new operations

```
   arm   set     recall            precision (TP/(TP+FP))     FP   target FP
   F     MSL     3/36    8.3%      3/4     75.0%               1        2
   F     SMAP   31/62   50.0%     31/34    91.2%               3        9
   F     Total  34/98   34.7%     34/38    89.5%               4       11
   paper MSL    25/36   69.4%     25/27    92.6%               2
   paper Total  84/105  80.0%     84/96    87.5%              12
```

- **The faithful port reproduces the paper's precision and does not reproduce its recall**:
  **89.5% against 87.5%**, on **4 false positives against a scaled target of 11**, at
  **34.7% recall against 80.0%**. **That is the inverse of D50**, where precision was 11.2
  points short and recall 35.1.
- On MSL, the one exactly like-for-like population, **the paper catches 25 of 36 with 2
  false alarms and this reproduction catches 3 with 1.**
- **The committed port supersedes the lost script.** `1a+1b` is now 57/98 and 62/110
  against the recorded 59/98 and 65/146; `1c-ii` is 74/98 and 100/237 against 74/98 and
  98/221. Both kept everywhere. From here the port **is** `1a+1b` and `1c-ii`.
- **D51 consequence 2 is discharged and the forecaster rungs open** -- the candidate count
  now matches, 38 predicted units against 96 and 4 false positives against 12.
- **N1 refuted**: FG's nominal rate is 1.5639% against stage 4's 0.6838%, so by 27.3's rule
  no comparison is drawn and **stage 4's 10/38 stands unreplaced.** The port has no dial.
- S3 held at 3.0-7.2 windows per index proportional against 31.0 published. **L1 is settled
  as a property of the data**, not a transcription defect: `tests/test_smap_rungs_port.py`
  builds windows where each mechanism must fire and shows it does.

### Process - two defects the smoke caught before a single operation was spent

- **The fit context was dropping the channel id.** `context.channels` is the second element
  of the weight-cache key (`detectors.py:421-424`), so every one of the 318 banked fits
  would have missed and the run would have silently refitted 152 models -- **D14's failure
  exactly**. Caught by reading the new `ctx_for` against the old one rather than by running
  anything.
- **The buffer dilation was quadratic.** Transcribing `errors.py:291-298` literally is
  correct and unusable; dilating the contiguous runs gives the identical set in linear
  time. Neither defect would have announced itself.

---

## [0.6.10] - 2026-09-08 - telemanom is vendored and read: it is causal, and four rungs chased a mechanism that is not in it

Zero operations. `docs/MODELS.md` 26.29, 26.30, D53.

### Added - the source, pinned as evidence and never a dependency

- **`third_party/telemanom/`** at commit `2e6c5b6c3558e7835601519b7bdef37c649bdbdc`, source
  only, 84 KB. Sections 26.21 to 26.28 located four divergences by quoting `errors.py` at
  specific line numbers and **no copy of that file was ever kept**, so the quotations could
  not be checked and 26.28's closing question could not be answered at all.
- Nothing under `src/` or `scripts/` imports it, pinned by `tests/test_layering.py`, and it
  is never executed. `PROVENANCE.md` records the commit, the licence, what is vendored,
  what is not, and which guards do and do not cover it.

### Corrected - reading it in full corrects five recorded readings

- **It clips each window to its newest `batch_size`** (`errors.py:355-359`), so **telemanom
  is causal** after its opening window and its cross-window accumulator unions **disjoint**
  batches. 26.25.1's "every index is judged using data ahead of it" is true of window 0
  alone.
- **There is no union over roughly thirty overlapping verdicts anywhere in it.** Rungs 1c,
  1c-i and 1c-ii were built to reproduce a mechanism the source does not contain. **`1a+1b`
  is the faithful arm and `1c-ii` is a departure** that is more permissive than the
  published algorithm. **D52 is superseded at its premise and kept in full.**
- **The precision denominator is a mixed unit** (`detector.py:117-136`, `:167-173`): true
  positives deduplicated per matched event, false positives counted per predicted range.
  So `1c-ii` is **74/197 = 37.6%**, not 98/221 = 44.3%. Both numbers kept everywhere.
- **The target is 12 false positives, not "~91 candidate ranges".** `~91` was `80.0/87.5`,
  arithmetic that appears in no document. On MSL the denominators match the paper's exactly
  at 36 and the gap is **2 against 58**.
- **The licence is BSD 3-Clause** (Caltech/JPL 2018), **not the Apache-2.0** `docs/DATA.md`
  recorded. `scripts/ingest_smap_msl.py` is corrected so any future ingest is right; **the
  stored manifest object `_manifest/smap_msl.json` still carries the wrong string** and is
  left alone, because rewriting a pinned manifest costs 1 Class A and is not a change to
  make without approval. Recorded rather than quietly fixed.

### Added - five mechanisms in the source this reproduction does not implement

A magnitude conjunct (`errors.py:342-343`), a whole-window bail-out (`:337-340`), the two
`find_epsilon` guards (`:314-315`), an inverse pass (`:132-148`) and `adjust_window_size`
(`:84-93`). Work item 9.10 is pre-registered against them.

### (!) What went wrong is not that the source was not read

It **was** read, and the project's rule that a divergence is located rather than guessed was
followed. What was missing is that the reading left nothing behind, so a wrong premise
survived four rungs of careful, pre-registered, correctly-stopped work. Every stop condition
in 26.21 to 26.28 fired or held exactly as designed. **The discipline worked perfectly on
top of a premise nobody could check**, which is the most expensive kind of correct.

---

## [0.6.9] - 2026-09-03 - Stage 5: the reproduction gap is the pruning step

One read at 1 Class A and 165 Class B, plus a discarded run whose 75 fits are reused.

### Corrected before running - the hypothesis was untestable

- Stage 5 was commissioned to score two ways, "Hundman's overlap-after-pruning rule and
  our event-wise rule", on the hypothesis that the gap is the scoring rule. **There is
  only one rule.** `eventwise.detected` is "fires anywhere inside the span" -- any
  overlap, Hundman's recall rule; precision is predicted ranges overlapping a truth
  range over all predicted ranges, his precision rule; and `PRUNING_P = 0.13` is his
  published value, in our detection stack. Checked in the source, before spending.

### Added - D50, and the paper's Table 2 retrieved rather than recalled

- **Precision reproduces within 11.2 points** (76.3% against 87.5%); **recall is 35.1
  short** (44.9% against 80.0%), over 98 sequences.
- **The paper's own ablation reproduces in direction on both datasets and both
  metrics**, so our detection stack has its internal structure.
- **And it localises the gap.** Pruning costs the paper **7.3** points of SMAP recall
  and **0.0** on MSL; it costs this reproduction **32.3** and **22.2**. Without pruning
  we are 11.3 points short of the paper; with it, 35.1. **Pruning triples the gap.**
- The available excuse is ruled out rather than used: the gap is **not** the missing
  command inputs, because D49 measured commands making the detector worse and the
  ablation locates the difference in a stage commands do not touch.
- **It re-reads D18.** A second published component behaves differently in our hands
  than in the authors'. Neither is evidence the method is wrong; both are evidence that
  a reproduction is a measurement with its own error bars, and one is now quantified.

### Process

- A first stage 5 run fitted 75 channels and then failed in the scoring code -- `Event`
  takes `segments`, not `channels`, which is a derived property. The fits were cached,
  so the corrected re-run took 57 seconds and cost one read. My error.

## [0.6.8] - 2026-09-03 - Stage 4: the dynamic threshold works, and D6 is answered

Two reads at 1 Class A and 165 Class B each; the first discarded for a defect of mine.
Cached weights, weight store +0 -- no fit repeated, because mode changes the detection
stack and not the cache key.

### Added - the gate passes

- **The published dynamic threshold reaches 0.6838% of nominal steps**, where both
  static routes failed (D48, 26.16). It recomputes the cut from a trailing window of
  the stream being scored, so there is no train-to-test transfer to fail.
- **The forecaster catches 10 of 38 in-range contextual sequences at that rate** - the
  project's **first measurement of the in-limits claim**, on events that by stage 1's
  diagnostic never leave their channel's historical range. It is **below** the 12-28
  predicted (V20 refuted), and 10 of 38 is 26%: not nothing, not a vindication.
- **No comparison against the floor is available.** `rstd` at **5,000 times** its
  calibrated threshold still alarms on 15.17%, and the range check on 5.84%. The sweep
  was widened to 5,000 so this could be tested rather than assumed, and both saturate.
  **This is the exact inverse of ESA-ADB**, where a rate-matched range check beat the
  forecaster and was never later (D44). The two datasets answer oppositely, and both
  answers are about the regime rather than about detection.

### Added - D49, closing D6 after it was open since work item 4

- **Conditioning on commands makes the detector worse.** At matched rates - 0.6838%
  against 0.6820% - the commanded arm catches **6/38** in-range contextual against
  **10/38** uncommanded, and **33/100** against **47/100** overall. The arms differ in
  that alone: same architecture, hyperparameters, seed, weights and detection stack.
- It agrees with the external evidence `docs/RESEARCH.md` flagged as the thinnest in
  the project - ESA's baselines lost precision with telecommands too. Agreement does
  not make either strong, and D49 bounds the finding to this encoding and this dataset.

### Process

- **The first stage 4 run did not widen the sweep.** `--sweep-max 5000` was accepted
  and never used, because the patch meant to apply it targeted an anchor line that does
  not exist in this script and failed silently. The defect announced itself in the
  shape of the result -- both static arms pinned at exactly 50 -- and was caught by
  checking the printed multiplier against the flag. The second read is my error.

## [0.6.7] - 2026-09-03 - Stages 2 and 3: the population is real and cannot be scored

Two runs at 1 Class A and 165 Class B each, plus one discarded. Both close without a
detector comparison, and the reason is calibration rather than detection.

### Added - D47, an absolute constant that should have been relative

- **`error_window` is 2100, fixed for ESA-ADB where a fold is ~3.5M steps.** The
  median SMAP/MSL training series is 2,690, so the warm-up (250 + 2100) consumed whole
  channels: 16 of 81 test arrays sat entirely inside it and 48 of 81 had their
  threshold computed from warm-up scores. Corrected on SMAP/MSL only to telemanom's
  own proportional definition, `SMOOTHING_PERC * len(series)`; ESA-ADB keeps 2,100,
  because the same formula on a 3.5M-step fold gives 175,000. **This is D17's class of
  defect** - a constant that should have been relative, disabling what it configures
  without erroring.

### Added - D48, and the bound it puts on D25

- **No train-calibrated threshold transfers on SMAP/MSL, for any arm.** `rstd` at
  **fifty times** its calibrated threshold still alarms on **15.57%** of nominal
  steps, where it runs at 0.024% on ESA-ADB; the forecaster's own threshold admits
  10.47%. Every calibrated arm fails at once, which rules out a detector-specific
  cause: the test-split residual scale is far larger than the train-split scale.
- **It is a replication, not a new result.** D29 found it on ESA-ADB - a floor
  calibrated on early data sat under 86.7% of a later window's nominal residual on
  `m1-g3`. SMAP/MSL is the same finding on independent data and in a sharper form.
  **D25 is bounded, not withdrawn**: the frozen static quantile describes a stationary
  regime. Every ESA-ADB figure stands.
- It also explains telemanom's dynamic threshold, which this project reproduced and
  rejected on ESA-ADB (D18). Both results are about the data, not the rule.

### Stage 3 - the remedy, pre-registered, and refuted

- **Recalibrating on a commissioning window made it worse: 10.47% -> 32.51%.** The
  cause is the estimator limitation 26.15.1 fixed **in advance**: a 99.9th percentile
  needs ~1,000 samples to be interior, so at 500 it is the commissioning maximum, and
  a short window's maximum is far below a long stream's. **V13 refuted**, which
  26.15.4 had named a stop.
- Both routes fail for one reason: the percentile rule needs a long, representative
  calibration window and this dataset provides neither.

### Not measured, and recorded as unmeasured

- **Whether a forecaster sees in-range anomalies a limit check cannot.** Stage 1 built
  the population - 39 of 43, six and a half times ESA-ADB's (D46) - and no operating
  point could be obtained to score it at. **D6 is still open** for the same reason:
  the ablation is correctly wired, ran twice, and both times at an alarm rate that
  makes the comparison meaningless.

### Process

- A commit went in with `tests/test_no_list.py` failing, because a `check_no_list`
  failure was piped through `tail -2` and its closing line misread as a pass. Fixed in
  the following commit rather than rewritten (`c429683`).

## [0.6.6] - 2026-09-03 - Work item 9.9 study 1 stage 1: a contextual population that is real

SMAP/MSL ingested and the visibility diagnostic run. 164 Class A to upload, 165 Class B
to read back; the month stands at 173 Class A and 269 Class B of 50,000 each.

### Added - the dataset, 2026-09-03

- **`smap-msl/v1/` in R2**, 162 arrays and the canonical `labeled_anomalies.csv`, under
  its own `_manifest/smap_msl.json`. `_manifest/manifest.json` is untouched: the ingest
  writes only under `smap-msl/v1/` and its own manifest, because `manifest.py`
  hard-codes the single key `esa-adb` and `Catalog.load` reads it. Recorded in
  `docs/HARNESS.md` 5a's register of authorised additions, additive-only.
- **Every array verified against the canonical labels before upload** -- fetched from
  `khundman/telemanom` directly rather than from the redistribution -- 82 of 82 rows,
  0 mismatches, with the script written to abort before the first put.
- **A labelling defect recorded, not deduplicated silently.** `P-2` appears twice with
  conflicting spans, so the dataset has 81 unique channels rather than the 82 its own
  label file implies. Carried in the manifest's `labelling_defects` field.

### Added - D46, the finding

- **The labelled contextual class is genuinely in range.** 39 of 43 contextual
  sequences stay strictly inside their channel's training min/max, against 11 of 61
  point -- a **72.7-point gap**, in the direction the labels claim, on both spacecraft
  (SMAP 26/26, MSL 13/17). Against D43's 6/32 on ESA-ADB's headline cell this is a
  population six and a half times larger and five times denser.
- **Wu & Keogh's triviality critique, answered precisely.** Confirmed for the point
  class -- 82% breach their range and a one-liner finds them. Refuted for the
  contextual class -- 91% do not. Both halves recorded, because reporting only the
  second would be the selective reading their paper is about.
- **Objective.md 9.2 stands and is better specified.** SMAP/MSL is still unusable for
  the cross-channel claim and is now demonstrably usable for the in-limits claim.
  No figure here may be quoted as cross-channel evidence.

### Predictions - one wrong in the comfortable direction

- **V1 refuted**: 15 to 30 in-range contextual predicted, **39** measured. The band was
  set expecting the labels not to hold up; they held up. Recorded loudest for that
  reason. **V2 refuted by one** (11 of 61 against a predicted 10). **V3, V4, V5 held**,
  V5 being the gate at n >= 20 that decides whether stage 2 is worth running.

### Held - stage 2 does not run without approval

- `gru-quantile` with the command columns as exogenous inputs (D6, open since work item
  4), against `rstd` and a calibrated range check, on the paper's own split, at a
  matched nominal rate. Its falsification will be stated against the measured
  matched-rate multiplier, not "by construction" -- on `m1-ss5` that multiplier was
  0.672, tighter than the training range.

## [0.6.5] - 2026-09-03 - Work item 9.9 study 2: the precursor test came back empty

Pre-registered at `docs/MODELS.md` 25 before a figure existed, then run in one bundle
load: 1 Class A and 15 Class B, cached weights, nothing refitted during the run.

### Added - the precursor test, 2026-09-03

- **D45 -- every alarm the forecaster raises begins inside a labelled span.** All 157
  of `gru-quantile`'s alarm ranges on `m1-g8.9.10` start inside a labelled anomaly or
  rare-event span; **none** begins in quiet, in-range nominal time. Its 142
  nominal-flagged steps are the tails of alarms that started inside a labelled span and
  ran past its end. So its false-alarm count is spent on where alarms *end*, not where
  they begin -- a different defect from the one the metrics imply, and a smaller one.
- **The study failed on power, not on effect, and that is a prediction refuted.** S2
  predicted at least 20 eligible alarm starts per set and measured **0** and **6**. The
  primary test could not run; at n=6 the subset is UNDERPOWERED and no p-value is
  quoted from it, exactly as 25.3 said in advance.
- **The permutation was verified before it was spent**: on planted precursors it
  returns p = 0.009 under the circular-shift null at a rate ratio of 3.98, and 0.72 on
  randomly placed alarms of the same count. The null is the data's, not the test's.
- **No precursor population means the early-warning argument gains nothing here.** D44
  stands: no measured lead over a rate-matched range check, and now no nominal-period
  alarms that could have been early either.

### Known gaps

- **Where those 157 ranges begin** -- the split between anomaly and rare-event spans --
  is not instrumented. One bundle load. Named rather than left to be found.
- **The weight store is 89 files, not the 86 quoted since work item 9.6.** The three
  additions are synthetic-fixture weights written by this section's offline dry runs,
  which pass `--allow-fit` because the fixture has no cache. The in-run invariant held:
  the count was identical before and after the Mission 1 run, so nothing was refitted on
  Mission 1 data. Recorded because a quoted invariant that quietly changes is worse than
  one that moves for a stated reason.

## [0.6.4] - 2026-09-03 - Phase 2 work item 9.8: the yardstick the forecaster does not clear

Pre-registered at `docs/MODELS.md` 24 before a figure existed, then run in one bundle
load: 1 Class A and 15 Class B, cached weights, nothing refitted, weight store unchanged
at 86 files. All six reproduction checks passed before anything new was read.

### Added - work item 9.8, 2026-09-03

- **D44 -- at a matched alarm rate a per-channel range check beats the forecaster on
  both sets, and the forecaster never speaks first.** The envelope is the fitting
  window's own per-channel min/max under `train_mask`, widened until its nominal-step
  rate matches. On `m1-g8.9.10` it catches **34/46 and 25/32 at 0.0000% nominal, F0.5
  0.934**, against `gru-quantile`'s 27/46, 22/32, 0.0013% and 0.804; on `m1-ss5`
  34/42 and 25/31. The catches are nested against the forecaster -- only-GRU **0** at
  the matched point on both sets. And in **0 of 53** caught events does the forecaster
  fire first: median lead +0.0, mean -105.1 and -0.3, with the range check earlier on
  14 of them by up to 2,638 timesteps. This is the most serious result the project has
  produced, and it retires the last proxy the early-warning argument had on Phase 1
  evidence.
- **The mechanism is amplitude.** At every one of those 53 first crossings the channel
  that raised the alarm is already outside 3 sigma of its own anomaly-masked
  fitting-window distribution -- median 103.8 and 102.9, minimum 6.6, none below 3.
  Stated with its caveat: ESA-ADB is min-max scaled per group, so 3 sigma is a low bar
  at this scale. The bar was fixed before the numbers existed and was not moved.
- **D43 -- contextual is the training-window min/max, not the 0.1/99.9 band.** A real
  limit sits outside a channel's historical range, so the tightest bar any limit could
  hold is min/max, and D39 measured a quantity tighter than the claim it was testing.
  The class is **6/32** and **18/31**, reproducing 22.11 and D40 exactly from a
  different script. Over the gate set's six the flying detector catches **none** at its
  own operating point and **one** -- `id_89` -- only when loosened to the floor's alarm
  rate. D39 consequence 2 survives the change of definition; D39 and D40 are re-read and
  neither is edited.
- **`docs/RESULTS.md` 6m**, the per-event tables, and `scripts/reduction_and_curve.py`
  extended with the per-channel bar, the range check, the amplitude z-scores and
  event-wise F0.5 along the sweep.

### Fixed - a defect this section's own cross-check caught

- **The envelope breach was computed with `>=` where the published definition is
  strict.** An event that *touches* a channel's historical extreme has not *left* it,
  and ESA-ADB's per-group min-max scaling makes exact boundary values common: the first
  run disagreed with the floor audit on 2 gate-set and 14 `m1-ss5` events, every one at
  a reach of exactly 1.000. Under the strict rule the counts reproduce the floor audit
  exactly, 11/11 and 24/24. No re-run was needed -- the disagreement can only occur at
  `w = 1.0`, and no matched operating point sits there. The script now carries a
  `STRICT` rule with the reason.

### Closed - carried from work item 9.7

- **M4 refuted**: swept to the alarm rate where event-wise precision first falls below
  0.5, neither detector catches any of the three 0.1/99.9-contextual events on either
  set. **M9 held on the gate set**, closed analytically: event-wise F0.5 at perfect
  precision is `1.25R/(0.25+R)`, so no Study B arm above 9/46 recall can reach 0.804.
  `l2`'s F0.5 on `m1-ss5` is not computed and is named as the one gap.

### Predictions - two of the three written against interest fired

- **Refuted**: N4 (positive median lead -- measured +0.0), N5 (60% breaching after our
  emission -- measured 0 of 53), N6 (raw z inside 3 sigma -- measured 0 of 53), N7 (3
  misses recovered by the per-channel bar -- measured 0 on the gate set), N1 as written.
- **Held**: N2 by one event on a loosened detector, N8, N3 at one of its two matched
  points, M9.

### Held - still not written

- **The headline.** It is written once, from work items 9.7 and 9.8 together, for
  approval. Nothing is written to a claim site meanwhile, and `main` stays held.

## [0.6.3] - 2026-09-02 - Phase 2 work item 9.7: the comparison made like for like

The false headline withdrawn, and the two studies that could be run, run. One bundle
load for both, 1 Class A and 15 Class B, cached weights, nothing refitted, weight
store unchanged at 86 files.

### Changed - the headline comparison is withdrawn, 2026-09-02

- **Ten sites asserted "a forecaster finds 28 of 32 headline-cell events where a
  per-channel statistic finds 3" as live.** All ten are withdrawn, each keeping the
  sentence in quotation. Three things were wrong with it: the 3 was float32
  accumulation and is 25 (D37); the 28 of 32 is `lstm-telemanom`'s, a detector
  disqualified at 22 of 48 commanded manoeuvres, while the detector that flies scores
  22/32; and D38 measured the flying detector's catches as a strict subset of the
  floor's. What replaces them is a holding note, **not a new claim**.
- **What is kept, separately**: nothing else in an F' deployment watches the
  relationships between channels. That is a claim about the ecosystem, verified in
  Objective.md 3, and no correction touches it.
- **`docs/RESULTS.md` 6l** is new -- the corrected floor beside the flying detector,
  which no section carried, because 6h to 6k compare the architectures against each
  other and the correction came later.

### Fixed - a one-sided figure, found while writing 6l

- **The gate-metric result is not the same on the two channel sets and only one had
  ever been quoted.** On `m1-g8.9.10` the forecaster leads 0.804 to 0.676; on
  `m1-ss5` the corrected floor leads **0.663 to 0.593**. The alarm-rate advantage is
  3.71x on the gate set -- the "third of the alarm rate" every document repeated --
  and 1.40x on `m1-ss5`. Both were inside the same artifacts, each carrying both sets.
  D38 consequence 1 gets a rider; the selection does not move, since D28 never
  involved `rstd`.

### Added - work item 9.7 studies A and B, 2026-09-02

- **D41 -- at a matched alarm rate the nesting reverses.** D38 compared the two at
  their own thresholds, where the floor alarms on eighteen times more nominal steps.
  Held to the forecaster's rate the floor finds **7 of 32** headline-cell events, not
  25; allowed the floor's rate the forecaster reaches **26/32** against 25/32. The
  four sets follow: at matched-quiet on the gate set, only-GRU **20** and only-`rstd`
  **0**. At every matched point on both sets only-`rstd` is 0 or 2. **D38's strict
  subset is a calibration artifact**, and its counts remain exactly right for the
  frozen configuration.
- **D42 -- D23 closes as answered no.** `k`-of-`n` re-derived at k=2 and k=3, plus L2
  and sum, under the frozen calibration recipe, all five from one forecast pass so
  the `max` arm **is** `gru-quantile` and reproduces its scorecard. Every arm's catch
  set is a strict subset of `max`'s on both sets; not one recovers any of the 19
  gate-set misses, the seven fold-1 events or the three contextual events, and every
  arm runs at a *lower* nominal rate, so it is not a threshold handicap. The one
  improvement is `l2` on `m1-ss5`: the same 26/42 and 21/31 at 1,124 nominal steps
  against 1,536, 27% fewer -- and the same arm collapses to 9/46 on the gate set. The
  mechanism inverts D23's premise: `max` wins because cross-channel evidence is
  sparse. All four arms are sign-blind, so a signed reduction remains untested.
- **Neither result rehabilitates the contextual claim.** D39 and D40 stand: the
  headline cell is 3/32 contextual and nothing catches the three, at any operating
  point or under any reduction.
- **`docs/MODELS.md` 23.14** costs studies C and D without building either. Every
  leave-one-out form costs about C times the arithmetic, whatever the parameter
  count; a version-1 `model.bin` would load design (b) and silently run it once
  instead of twelve times, which the reserved-field refusal rule closes. And Rule 1
  forbids storing an injected set as values, so D stores a regenerable recipe.

### Known gaps - named, not left to be found

- **M4 and M9 are unresolved.** The sweep recorded events caught and nominal steps
  flagged, not alarm ranges classified, so event-wise F0.5 along the curve was never
  computed. A defect in `scripts/reduction_and_curve.py`, not in the design: 23.8
  asked for both and one was instrumented. One further bundle load closes it.
- **The new headline is not written.** No replacement claim is stated; the measured
  state is in `docs/RESULTS.md` 6l and `docs/MODELS.md` 23.15, and `main` is held.

## [0.6.2] - 2026-09-01 - Phase 2 work item 9.6: auditing the corrected floor

The audit of a result that overturned a thesis, pre-registered at `e31533d` before a
figure was computed. It found the floor clean and two things worse than the correction
had been. Zero new capability; one bundle load, 15 Class B and 1 Class A against a
budget of 20; cached weights, nothing refitted, the weight store unchanged at 86 files.

### Added - work item 9.6, 2026-09-01

- **`docs/MODELS.md` 22**, fourteen predictions L1 to L14 with their falsifications, and
  `scripts/floor_audit.py`. Artifact
  `runs/m1-g8.9.10/_forensics/2026-09-01T230303Z-floor-audit.json`. The audit reproduces
  every published scorecard count exactly -- 27/46, 34/46, 22/32, 25/32 -- before it says
  anything new.
- **The corrected floor is not leaking, and work item 9.5's numbers stand as measured.**
  L1 to L4 all hold: perturbing a future sample changes rows before `t` by exactly 0.0
  with a working control; `fit` sees only the masked training window through the same
  `harness.py:144` line every detector uses; the fallback scale is unreachable; and
  `RollingStd.threshold_from` **is** `Detector.threshold_from`, the same function object
  the GRU's calibration calls. The catches are sustained, not stray: excluding
  footprint-1 events, 0 of 25 are single-step and the alarm covers a median 95% of the
  event.
- **D38 -- `gru-quantile` catches a strict subset of the floor's events on Mission 1.**
  Both 27, only-GRU 0, only-`rstd` 7, neither 12 on the gate set; 26/0/8/8 on `m1-ss5`.
  There is not one event on either set, in any taxonomy cell, that the flying detector
  catches and the corrected floor misses. The seven it misses are short and sharp --
  footprints 1, 1, 1, 1, 24, 28, 54 -- where its own 105-span EWMA smooths the excursion
  away. The per-event argument for the forecaster is withdrawn on these two sets; its
  case rests on the quality of the same catches, which is what D3's gate metric measures.
- **D39 -- the headline cell is not the contextual class, and the contextual class is
  caught by nothing.** Of the 32 headline-cell events, 3 are truly contextual by the
  0.1/99.9 training envelope and 6 by hard min/max, against 8 to 16 predicted. The other
  29 breach at least one channel's own envelope. The three genuinely contextual events
  are caught by neither detector, at reaches of 0.11 to 0.61 against a threshold of 1. On
  the evidence available this project has no measured instance of catching a contextual
  anomaly. Stated carefully because an envelope is not a limit: breaching one does not
  establish that a limit would have tripped, only the converse.
- **The state-adaptation hypothesis is dead and backwards.** The GRU's score does not
  collapse inside a sustained event -- median half-life 2,805 and 6,284 steps, and it
  never halves in a third of them -- while `rstd`'s halves in about its own 120-step
  window, 114 and 112. Whatever causes the GRU's misses, it is not that it learns the
  anomaly. No structural remedy was scoped, because 22.3 made scoping conditional on the
  hypothesis holding and it did not.
- **Two predictions were refuted in the direction the work item was commissioned to
  avoid**, which 22.4 said in advance to watch for. And L5's own definition was defective:
  it did not condition on footprint, so nine footprint-1 events counted as stray ticks
  when a one-step crossing on a one-step event is a perfect catch. Recorded Refuted
  anyway, because a definition that needed fixing after seeing the data is what
  pre-registration exists to expose.

### Added - work item 9.6 follow-up, 2026-09-02

- **`docs/MODELS.md` 22.11**, the four answers the audit's own artifact held and 22.6 to
  22.10 did not read: the four sets by id, fold, cell and footprint for both channel sets;
  the fact that every only-`rstd` event is in fold 1, and that six of the seven are the
  same six section 14 recorded as the GRU's fold-1 losses against the LSTM; `m1-ss5`'s 31
  headline-cell events classified, replicating D39 at 3/31 on the same three event ids;
  and the correction of the work item's brief, whose "twelve GRU-missed events" are in
  fact `lstm-quantile` misses of which `gru-quantile` recovered seven. Zero bucket
  operations.
- **D40 -- "truly contextual" is defined relative to a watched channel set.** Hard
  min/max contextual goes from 6/32 on twelve channels to 18/31 on six while the 0.1/99.9
  count stays at 3, verified as a strict superset relation with no violations. The
  mechanism is monotonicity. D39's headline and all five of its consequences stand,
  better supported than before.
- **`tests/test_documents_are_current.py`**, three checks: no new live occurrence of a
  claim `Objective.md` 1.1 has retired, every tracked document ASCII, and a test count a
  live document states matching what pytest collects. The stated counts were 493 and 473
  against 508 collected; both are corrected.

### Held - not done, and not quietly

- **The restatement of the central claim** in every document that carries it. Nine
  sentences still assert a ratio D37 falsified and D38 reversed, and they are pinned by
  the new test rather than fixed. Held pending review (`docs/STATUS.md` section 7).
- **Two gaps in the audit's instrument**, named in 22.10 and priced in 22.11: GRU run
  lengths were never recorded, and `lead_of` cannot separate a crossing at onset from an
  alarm already running. One bundle load together, 15 Class B and 1 Class A.

## [0.6.1] - 2026-09-01 - Phase 2 work item 9.5: the floor was wrong

A D8 correctness fix and the re-score it forced. Zero new capability; one published
claim falsified.

### Fixed - work item 9.5, 2026-09-01 - `baselines._rolling`

- **`_rolling` accumulated its prefix sums in float32 and lost the statistic it computed.**
  `baselines.py:40` now promotes to float64 before squaring and `np.cumsum` accumulates with
  `dtype=np.float64`. Against the flight rule the gap closes from **7.6584e+00 to
  1.8284e-08** and the spurious exact zeros go from 3,975 to none.
  `tests/test_rolling_precision.py` pins it against `numpy.nanstd` in five regimes, and the
  three work-item-9 tests that pinned the size of the divergence are inverted, as their own
  docstrings instructed.
- **The floor moved a long way.** Re-scored at **41 Class B and 3 Class A**, old artifacts
  preserved, no forecaster row recomputed and no run naming any other detector:

  ```
    m1-g8.9.10  rstd  F0.5  0.250 -> 0.676    MVGS  3/32 -> 25/32   lead -1,512 -> +0.0
    m1-ss5      rstd  F0.5  undefined -> 0.663   MVGS  0/31 -> 25/31
    m2-ss1      rstd  nominal-step FA  17.30% -> 0.003%    rare  84/424 -> 22/424
    m1-g3       rstd  F0.5  0.029 -> 0.351    MVGS  3/10 -> 8/10
  ```

- **The pre-registered falsification fired, and this is the finding.** `docs/MODELS.md` 21.4
  named 22/32 -- `gru-quantile`'s headline cell -- as the line past which the project's
  central claim would be "in serious question". The corrected floor reached **25/32**. A
  per-channel statistic finds twenty-five of the thirty-two cross-channel events, not three.
  The sentence this repository has quoted since work item 4 -- "a per-channel statistic
  finds three; a forecaster over the channel set finds twenty-eight" -- is about arithmetic,
  and is corrected everywhere it appears with the old figure beside it.
  **(!) CORRECTED 2026-09-02: that last clause was not true when it was written.** The
  sweep corrected every scorecard *table* and left the *prose claim* standing in nine
  sentences, including `Objective.md` 1.1's KEPT block, which is the passage
  `docs/INDEX.md` sends every reader to first. The restatement is held pending review at
  `docs/STATUS.md` section 7 on D39; `tests/test_documents_are_current.py` now pins the
  nine so the set cannot grow while it waits.
- **What survives.** On the gate metric D3 fixed before any of this was measured -- event-wise
  F0.5, never bare recall -- `gru-quantile` still clears the corrected floor **0.804 to
  0.676**, reaching comparable recall at **a third of the alarm rate** with precision 0.885
  against 0.661. D28's architecture gate never involved `rstd`. D25 and D29 are untouched,
  and D29's evidence is cleaner: on Mission 2 the corrected floor alarms on 0.003% of nominal
  time rather than a sixth, so the adoption number is now a comparison between two working
  detectors instead of one working detector and a broken one.
- **W1 was wrong and backwards.** It predicted the corrected threshold would fall; it rose
  from 3.34 to 13.67, because the scale divisor is the standard deviation of the spread
  series itself and correcting the numerator shrank the denominator more. Recorded Wrong.
- **A limit the fix does not remove.** `sqrt(S2/n - (S1/n)^2)` loses accuracy as the square
  of `|mean|/sigma` at any precision -- negligible at the ratios ESA-ADB's min-max scaling
  produces, total at 1e8, silently zero at 1e9. `baseline_reference.py` and therefore
  `flight/src/Baseline.cpp` share it exactly. The boundary is pinned by test and reported;
  making the form unconditionally stable is an algorithm change that would move the flight
  golden vectors, and was not taken here.
- Tests 493 -> 505.

## [0.6.0] - 2026-09-01 - Phase 2 work item 9 (tag wi9)

The F' component and the Level 1 safe-failure mode. Reviewed and checkpointed 2026-09-01.
Zero bucket operations throughout.

### Added - work item 9, 2026-09-01 - the F' component and Level 1

- **`Sentinel::Monitor` builds in F' v4.3.0's own Ref deployment**, which is work item 9's
  definition of done. Ref moved out of the framework root to `TestDeploymentsProject/Ref` at
  v4.3.0 and the move is not in the release notes, so the brief's target had to be found
  before it could be hit (`docs/MODELS.md` 20.2 correction 3). Two proofs: this project's own
  `fprime/SentinelRef` deployment, committed and rebuildable from a fresh clone, and F's Ref
  via `scripts/fprime_ref_patch.sh` -- 2,428,064 bytes against stock Ref's 2,352,560, carrying
  234 Sentinel symbols. Nothing is copied: `fprime/` is an F' library, so Ref consumes it the
  way a mission would, with one `library_locations` line.
- **Level 1 works, and was watched working.** All **11/11** refusal codes degrade to the
  statistical baseline with the code named in the event; **0/11** fail the topology; the
  component served 200 ticks after a refusal in test. Run for six seconds with no model file,
  the deployment emits `DegradedToBaseline: NO_MODEL_FILE` and carries on. Objective.md
  decision 10's Level 1 is resolved; Levels 2 and 3 stay open.
- **The loader has 11 refusal codes, not 16.** The 16 is the number of load *cases* in
  `flight/test/RefusalTests.cpp` -- 15 refusing, 1 accepting. `CHANGELOG.md`, `docs/STATUS.md`
  and `docs/MODELS.md` 19.8's prediction F7 were all loose the same way.
- **The Level 1 baseline transcribes the rule and not the implementation, deliberately** (D37).
  `baselines._rolling` runs `np.cumsum` on a float32 array and differences the result to
  recover a second moment, which is catastrophic cancellation: three independent
  implementations agree to 1.8e-08 and disagree with it by **7.6584e+00** on a true sigma of
  3.0, and on this project's own fixture it produces **3,975** exact zeros against float64's
  **1,123**. `flight/src/Baseline.cpp` matches `src/sentinel_models/baseline_reference.py`
  **exactly** -- 0.000e+00 over four tiers and 1,600 steps, flags exact. The harness repair is
  scoped as work item 9.5.
- **D32 to D37**: a passive component on a synchronous `Svc.Sched`; direct port wiring rather
  than a telemetry-path tap, which resolves Objective.md decision 3 and declines the tap on
  evidence; Level 1's constants as PrmDb-style parameters rather than model-file fields; a
  guarded types shim, amending D31 consequence 2; the whole-file read with the chunked reader
  deferred; and the `_rolling` finding.
- **The types shim swap, proven both ways.** One file differs between the freestanding build
  and the F' build, and it is the shim; `sizeof(Detector)` is still exactly 312,112 bytes under
  F' types. D31 consequence 2's literal wording is not achievable -- `Fw/FPrimeBasicTypes.hpp`
  needs a generated config header the Makefile build has no way to produce -- so the shim
  selects rather than replaces, and three tests hold the claim in place.
- **`FW_HAS_F64` does not exist in F' v4.3.0.** `docs/MODEL_FILE.md` 9 and the old `Types.hpp`
  both said F' treats F64 as switchable; `F64` is unconditional at `Fw/Types/BasicTypes.h:86`
  and the macro appears exactly once in the whole framework, in the documentation table this
  project read. The shim asserts the property instead. Both documents amended.
- **`clang-tidy` ran for the first time**: 170 findings, 16 fixed, 154 excluded with a written
  reason each, 0 remaining across three configurations -- ours, the framework's root config and
  the framework's release config for flight code. Two findings earned the exercise: an
  out-of-bounds access and a division by zero that the loader makes unreachable at load time
  but that a radiation bit-flip in RAM could reach afterwards. D31 consequence 5 was wrong
  about where clang-tidy comes from; it is Homebrew's llvm, not F'.
- **Footprint**: `sizeof(Sentinel::Monitor)` is **623,152 bytes** against 624,528 predicted
  before the component existed -- 1,376 B under. Of that, 312,112 is the `Detector`, 302,048
  the model-file buffer and 8,216 the `Baseline`.
- **Predictions C1 to C10 are re-tabulated in `docs/MODELS.md` 20.9; eight held and two were
  wrong.** C6 predicted fewer than 50 lint findings dominated by `readability-*` and got 170
  dominated by `misc-include-cleaner`; C9 predicted a 10-to-20-minute first F' build and got
  **12.4 seconds**, because F' builds only the modules the topology references.
- **Thirteen corrections to the work item's brief** are recorded in `docs/MODELS.md` 20.2 and
  four more things F' settled once code was being written in 20.10 -- among them that a
  component with parameters is required to carry command ports, and that a library's modules
  must be namespaced, which moved the component to `fprime/Sentinel/Monitor`.
- Tests 473 -> 493. `docs/FPRIME.md` records the toolchain and `scripts/fprime_setup.sh`
  rebuilds it from nothing.

## [0.5.0] - 2026-09-01 - Phase 2 work item 8 (tag wi8)

Phase 2's first work item: the flight inference core and the frozen model file. Reviewed
and checkpointed 2026-09-01. Zero bucket operations throughout.

### Added - work item 8, 2026-09-01 - the C++ inference core and the frozen `model.bin`

- **The format is frozen at version 1** (D30, `docs/MODEL_FILE.md`, normative). Plain
  little-endian float32 in `reference.Weights.arrays()` order with **both bias vectors
  unsummed**; a 64-byte self-protecting header naming the architecture and the gate order
  rather than leaving them to be inferred; a channel map; and a **separately-CRC'd
  parameter block** carrying the normalisation constants, the threshold, the EWMA span,
  `baseline_only` and the tier. A recalibration in orbit overwrites a fixed-size block and
  two header words, and never touches the 278.0 KiB of weights.
- **Objective.md 14.10's "quantized, self-describing FlatBuffer, TFLite-Micro compatible"
  is superseded and marked so, never deleted.** Nothing here consumes TFLite; a FlatBuffer
  parser is templated, allocating third-party code F' CPP-25 and CPP-1 exclude; and a fixed
  layout with a CRC is byte-inspectable by a review board. Quantization goes with it: the
  tolerance against `reference.py` is 1e-5 and int8 loses far more. Every requirement 14.10
  stated is met. Objective.md 14.2 is resolved.
- **F' pinned at v4.3.0** (D31), which resolves Objective.md 14.4. Reading F's own
  statement of its C/C++ rules corrected three this project had from memory: F' states no
  no-recursion rule (that is Power of Ten 1 and the JPL C standard, which F' cites at
  CPP-27); its no-heap rule is CPP-1, not Power of Ten 3; and CPP-3 forbids bare `float`
  and `double` outright, which was recorded nowhere and changes every declaration.
- `flight/`: the GRU forward pass and the frozen decision layer (D25) transcribed from
  `src/sentinel_models/reference.py`. Freestanding C++14 behind a types shim work item 9
  swaps for `Fw/FPrimeBasicTypes.hpp`. No exceptions, no RTTI, no STL, no allocation
  anywhere, no recursion, every loop bounded by a header field already checked.
  `sizeof(Detector)` is **312,112 bytes** against 312,642 predicted before the code existed.
- **The core matches the reference at 1.8e-07** worst case across seven weight sets --
  three seeded tiers and the four cached production fits -- over 504 steps spanning a chunk
  boundary and a reset, with the **crossing flag exact on every step**. The tolerance was
  1e-5, so the margin is roughly fifty-fold, and it sits where `docs/MODELS.md` 2 already
  measured the NumPy reference against torch (1.2e-07).
- `src/sentinel_export/` stops being a placeholder: `format.py`, `writer.py`, `reader.py`,
  standard library and numpy only, as its docstring has always promised. The reader returns
  a `Status` whose values are shared with the C++ `LoadStatus`, so one test asserts both
  sides refuse the same bytes for the same reason. Sixteen refusal cases; seven files
  round-trip Python to C++ to Python byte-identically.
- Golden vectors under `flight/test/vectors/`, committed and regenerable: deleting them all
  and rebuilding reproduces fourteen files byte-identically. **No vector uses real
  telemetry and none can** -- there is none on local disk -- so every input is the seeded
  fixture or a seeded generator, and the cached production weights supply the fourth tier.
- Determinism: the same 400-tick digest, `0xD66576B4`, twice in one process and again in a
  fresh one. Guaranteed by `-ffp-contract=off` and the absence of `-ffast-math`, both
  pre-registered. The build is **silent** at `-Wall -Wextra -Wpedantic -Wconversion
  -Wshadow -Werror`; `clang-tidy` is deferred to work item 9 with the F' toolchain.
- Tests 406 -> 473. `docs/MODELS.md` 19 is the pre-registration, committed before a line of
  C++, with its PREDICTED table re-tabulated in 19.8; all eight predictions held.

### Planned - Phase 2, and after

- After work item 9: the in-orbit threshold recalibration path, exercised end to end on the
  F' Ref. See docs/PHASE2.md and docs/STATUS.md section 7.
- Post-gate: injected-fault sensitivity study - controlled drifts and decouplings injected into
  real ESA-ADB telemetry, for a detection sensitivity curve and lead-time measurement. Never a
  headline number; see Objective.md section 13.

### Open decisions, and those resolved

| # | Decision | Deadline |
|---|---|---|
| 1 | Architecture selection - LSTM vs GRU vs TCN | **Resolved 2026-08-29: the GRU (D28)** |
| 2 | Model-file format freeze. The FlatBuffer/TFLite-Micro container above is **superseded by D30, not deleted**; the requirement it carried - normalisation constants and thresholds stored separately as PrmDb-style parameters (Objective.md 14.10) - stands and is met | **Resolved 2026-09-01: plain little-endian float32, version 1 (D30)**, specified in `docs/MODEL_FILE.md` |
| 3 | Channel-ingestion mechanism - telemetry-path tap vs direct port wiring | Early Phase 2 |
| 4 | Target F' version pin | **Resolved 2026-09-01: v4.3.0 (D31)** |
| 5 | Harness base - build on TimeEval or standalone | **Resolved 2026-08-25: standalone** (0.3.0) |
| 6 | R2 ingest sizing for 11.6 GB | **Resolved 2026-08-24: 11.53 GB in 234 objects** (0.2.0) |
| 7 | Second independent scoring set | **Resolved in practice and spent 2026-08-29**: Mission 2 the adoption number, Mission 1 group 3 the recall exam (RESULTS.md 6k) |
| 9 | SatNOGS as subsystem-prior corpus | Post-gate |
| 10 | Tiered capability architecture - Level 1 / 2 / 3, one loader, one file format. Level 1 is the loader's mandatory safe failure mode | Before Phase 2 |

Decision 8, normalisation policy, is **resolved**: identity. See Objective.md 14.

## [0.4.0] - 2026-08-29 - Phase 1 closed (tag wi7)

Phase 1 complete: the detection mathematics proven on the harness, the architecture gate passed, and the held-back transfer sets scored once. Work items 4 to 7; tags wi4 to wi7 and their Releases.

### Added - work item 4, 2026-08-25 to 2026-08-28 - telemanom reproduced with a multivariate LSTM

- `src/sentinel_models/lstm.py`, `reference.py`, `telemanom.py`, `detectors.py`: telemanom's
  detection method driven by one multivariate 2x80 LSTM over the channel set (91,640 parameters,
  358.0 KiB), trained in PyTorch and scored through a plain-NumPy reference held to 1e-5
  (4.1e-08 measured) - the Phase 2 C++ blueprint. Every deviation from the published
  configuration in MODELS.md section 1's ledger, ten rows.
- The floor cleared and the thesis held: 28/32 headline-cell events on `m1-g8.9.10` against
  `rstd`'s 3/32 (RESULTS.md 2). Fitting moved to a rented GPU, twelve fits in 15.9 minutes,
  certified by the equivalence assertion rather than a matching environment (D15, D16).
- D17: telemanom's published `min_delta = 3e-4` had silently disabled training for every fit in
  the project - one epoch kept, up to 8.8x better weights discarded. Replaced by a relative
  `min_improvement`; the forecast improved about fortyfold and the published threshold collapsed
  from 182 to 3,548 alarm ranges (RESULTS.md 6a). Thresholds belong to the model, not the method.
- The selection criterion measured over 5,684,580 reference windows: 92.6% chose the range
  minimum (D18, THRESHOLD.md). OS-CFAR pre-registered, run, retested and refuted (D20, D22).
  `lstm-whitened` built and measured - 2/48 rare-event false alarms at 21/32 (RESULTS.md 6d, D24).
- Lead time measured from a moment the detector could reach: the reported +26 was the batching
  latency counted backwards, the honest median is 0.0 (D21, RESULTS.md 6f); Objective.md 1.1
  retires the early-warning claim and keeps the cross-channel one.
- D25: the decision layer frozen as `lstm-quantile` - F0.5 0.838, recall 26/46, 21/32, 2/48,
  nominal-step 0.002%, honest lead +0.0 on `m1-g8.9.10` (RESULTS.md 6g), identical across every
  architecture at the gate. Telecommands wired as model inputs (`lstm-commanded`, D6, D7); the
  ablation on post-fix weights not yet run, so every figure remains telemanom-minus-commands.

### Added - work item 5, 2026-08-28

- `Hyper.cell`: the recurrent cell as a field of the LSTM's configuration, emitted only when
  not the default so no banked LSTM weight or published fingerprint moved (D26, pinned by test).
  One torch module builds `nn.LSTM` or `nn.GRU`; `train()` is shared verbatim.
- `reference.gru_cell` / `gru_layer`: the GRU tick beside the LSTM's, `GRU_GATES`, a one-vector
  state, and the cell of a weight file derived from its arrays and verified against the file's
  `cell` field. The third recurrent bias sits inside the reset product and cannot be folded --
  MODELS.md section 3 amended for the file format.
- Detectors `gru-telemanom`, `gru-quantile` (the gate arm) and `gru-smoke`.
- `scripts/fit_folds.py --determinism-check`; report written before the ledger; held-back sets
  refused. `scripts/head_to_head.py` records each event's reach beside the booleans.
- MODELS.md section 14: the `gru-quantile` pre-registration and its outcome. RESULTS.md 6h.

### Added - work item 6, 2026-08-29

- `Hyper.cell = "tcn"` with a TCN-only `kernel` field, emitted only for a TCN; `hidden` reused as
  the width of each residual block. `TelemanomTCN` (Bai et al. 2018 blocks, no weight norm) built
  by `build_model`, the one place the architecture is chosen; `train()` unchanged.
- `reference.ConvWeights`, `causal_conv1d`, `tcn_block`: the TCN blueprint, stateless by
  contract -- `forward` refuses a state and returns none. Receptive field 253, 91,670 parameters
  at the flown shape. Weight files carry `cell = "tcn"` with their own keys.
- Detectors `tcn-telemanom`, `tcn-quantile` (the gate arm) and `tcn-smoke`. D27.
- MODELS.md section 16: the `tcn-quantile` pre-registration and its outcome. RESULTS.md 6j: the
  TCN beside both cells -- MVGS 9/32 on the gate set, a floor above both incumbents' on every
  fold, one stalled fit. Three rows exist; the gate is a decision.
- MODELS.md section 17 and `scripts/combination_scope.py`: the LSTM+GRU combination scoped on
  cached weights, not built -- not nested; an OR reaches 25/32 at 2/48 on the gate set; a combined
  score recovers no gate-set solo event. Post-gate, its own decision if ever.

### Added - work item 7, 2026-08-29 - the gate and the held-back sets; Phase 1 closed

- D28: the architecture gate selects the GRU on Objective.md section 8's criteria; the LSTM stays
  the published baseline, the TCN rows the stateless answer. `lstm-gru-or`, the union as a
  detector, so the closure could score it through the one tested path.
- The held-back sets scored once, on a GO, with MODELS.md section 18's predictions committed
  first. `m2-ss1`: 4/424, 4/424, 6/424 rare-event false alarms and 0 nominal-step alarms for the
  LSTM, GRU and TCN; the floors 84/424 and 122/424. `m1-g3`: fold 0 clean; folds 1-2 a
  calibration collapse -- the noise floor fixed on the past sat under 87% of a later window.
- D29: `gru-quantile` flies alone; the union is not adopted (its cost failed on Mission 2, its
  edge evaporated on m1-g3); the calibration's transfer is what Phase 2 inherits. `docs/PHASE2.md`.

### Phase 1 work items, in order - all complete

- Reproduce telemanom's detection method with a multivariate LSTM forecaster (done).
- Train and score GRU (done, 2026-08-28; two stop-and-report rules fired, see RESULTS.md 6h).
- Train and score TCN (done, 2026-08-29; see RESULTS.md 6j).
- Pass the architecture selection gate, and score `m2-ss1` across LSTM, GRU, TCN, `rstd` and
  `mavg` together so the adoption number on an independent spacecraft is a comparison rather
  than a lone figure (done, 2026-08-29; D28, D29, RESULTS.md 6k). Phase 1 closed.

## [0.3.0] - 2026-08-25

The evaluation harness. Phase 1 work item 3 complete: the referee exists, it has
been checked against its own extremes, and the trivial baselines are scored.

### Added

- `src/sentinel_eval/` - the harness. `catalog` (manifest-only key resolution,
  typed dataclasses), `read` (streaming, checksum-verified, sharded-first),
  `labels` (events, taxonomy, the four categories), `grid` (zero-order hold with
  a staleness guard), `splits` (forward chaining, contamination reporting),
  `bundle` (fetch once, hold in memory, subset without re-reading), `metrics/`
  (event-wise F0.5, VUS-PR, false alarms, quarantined diagnostics), `harness`,
  `scorecard`, `tasks`, `ops`, `synthetic` and a CLI.
- `src/sentinel_models/` - the players. Trivial baselines plus the registry that
  work items 4-6 extend. The harness never imports a model; `tests/test_layering.py`
  enforces the direction.
- `src/sentinel_export/` - Phase 2 placeholder for the `model.bin` writer.
- `scripts/check_no_list.py` - the source-level LIST/glob ban, which the brief
  believed already existed. It did not.
- `docs/HARNESS.md`, `docs/RESULTS.md`.
- 136 tests, all offline against a generated fixture at zero R2 operations.

### Fixed

- `r2.fetch_ledger` caught bare `Exception` and returned a fresh ledger, so a
  transient failure reported the month's spend as zero. Only a genuinely absent
  ledger now starts fresh.
- The operations tripwire (1,000 per run) and monthly ceiling (50,000) were
  constants with no enforcement anywhere. Both now raise, at the point of
  spending, through the existing per-HTTP-attempt hook.
- The ops ledger was written with `new_ledger()`, erasing the month's history on
  every run. It is now read-modify-write.

### Decided

- **Normalisation is identity** (Objective.md 14.8). Cross-group spanning is
  acceptable; per-channel rescaling is refused because it erases the amplitude
  ratios ESA preserved within each group.
- **The gate number is event-wise F0.5**, never bare recall. Recall alone is
  satisfiable by carpet-bombing, which is how the trivial baseline first appeared
  to score 29/31.
- **`m1-g8.9.10` is the primary recall set**, promoted post-hoc on footprint
  evidence; `m1-ss5` is demoted, retained, and reported alongside it in every
  result. Partial runs are barred from RESULTS.md.
- **Standalone metrics, not TimeEval** (Objective.md 14.5): TimeEval requires
  Python <3.13 against this project's 3.14, pins `dask==2022.12.1` and needs
  Docker.

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

<!-- Versions 0.6.1 onward are untagged: work items 9.5 and later land on `dev` without a
     tag, so there is no tag pair to compare and no link is invented for them. The nine tags
     that exist are wi1..wi9. `wi9.5` was referenced here and never created. -->
[Unreleased]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi9...dev
[0.6.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi8...wi9
[0.5.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi7...wi8
[0.4.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi3...wi7
[0.3.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi2...wi3
[0.2.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi1...wi2
[0.1.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/releases/tag/wi1
