# The reorganisation plan

**Written 2026-09-09. RESEARCH AND PLAN ONLY -- nothing in this document has been
executed.** Every repair described below is proposed text. No file was moved, renamed
or deleted; no line of `src/`, `scripts/`, `tests/`, `flight/`, `fprime/`,
`third_party/`, `Objective.md`, `README.md`, `CHANGELOG.md` or any other `docs/` file
was changed; `main` was not touched; no history was rewritten; nothing was pushed.
Three files were written: this one, `docs/reorg_plan.json`, and two additive rows in
`docs/INDEX.md`.

**Cost: zero R2 operations.** Every check below is offline.

`docs/reorg_plan.json` carries the same content machine-readably, including the
**223-row file-by-file inventory** this document summarises rather than reprints.

---

## 0. The three most consequential proposals

**1. The flight core implements a decision layer the frozen pipeline does not use.**
D62 froze the pipeline on stage 4 -- `gru-telemanom` under telemanom's **published
dynamic threshold**. `flight/` transcribes **D25's static quantile alone**
(`flight/include/sentinel/Detector.hpp:3-7`, `flight/README.md:4`). Every headline
figure this project quotes -- 10 of 38 at 0.6820% (was 0.6838%; that is the commanded
arm's rate at 6/38, 26.18) -- comes from a rule the shipped core
does not implement. `docs/STATUS.md:271-274` records the gap; nothing in `flight/`
does, so a reader of the C++ has to infer it. This becomes **roadmap item 1**, **a
stated limitation in the public reading order** rather than a footnote, and **a
file-level comment** on `Detector.hpp` (drafted in full in the JSON's
`comment_and_doc_sweep`).

**2. Three guards pass while the thing they guard is wrong.** A guard that reports
clean on a defective subject is worse than no guard, because it buys false confidence.
All three are repaired in execution tranche 1, ahead of every cosmetic item. Details in
section 3.

**3. Nothing moves.** Measured, not assumed: 629 section citations across 601 sites
resolve today and every one cites a section *number*; all 25 scripts compute the
repository root as `Path(__file__).resolve().parents[1]`. Legibility is bought with
additions -- `scripts/README.md`, generated tables of contents, the dataset documents,
and a link-check guard -- not with a reorganisation that would break 630 references to
buy navigation a table of contents gives away. The physical alternatives are costed in
section 5 so the choice stays open.

---

## 1. What was measured

Every figure below was read from the repository this session, not from the brief.

| Fact | Value | Source |
|---|---|---|
| `dev` / `main` heads | `4136cab` / `439f400`; 152 vs 7 commits | `git rev-parse` |
| `main`'s tree | byte-identical to `dev` commit `9b2fb1f` -- exactly one commit behind | tree-hash compare |
| Tracked files / bytes | **223** / **3,380,754 = 3.22 MiB** of a **4 MiB** cap | `tests/test_no_local_persistence.py:141` |
| Tests | **606 collected** | `pytest --collect-only -q` |
| Operations ceiling | **50,000 Class A / 50,000 Class B per month**; tripwire 1,000 | `docs/DATA.md` 4; `src/sentinel_data/config.py:95-96` |
| `third_party/telemanom/` | **66.6 KiB**, 12 files, pinned `2e6c5b6c`, BSD 3-Clause | `git ls-files`; `LICENSE.txt` |
| `check_no_list.py` | **76 files clean** | run directly |
| ASCII | **24 tracked `.md`, 0 non-ASCII lines** | scripted |
| `runs/` locally | **438 MB** = 1,338 `.npz` (423.5 MB) + 144 `.json` (11.6 MB) | `du`, `find` |
| Decisions | **D1-D62, no gaps**; 7 still open | `docs/DECISIONS.md` |
| `CHANGELOG.md` | runs to **0.6.19**, no gaps | `CHANGELOG.md:17` |
| Section citations | **629 refs across 601 sites -- 0 unresolved** | scripted |
| Repository path citations | **110 distinct / 1,021 occurrences -- 0 true breaks** | scripted |
| `runs/` artifact citations | 98 concrete paths; **94 resolve, 3 are templates, 1 is broken** | scripted |
| Citations into the vendored source | **66** genuine distinct locations | scripted |

### Pre-publication safety sweep: nothing to stop and report

Every tracked file was scanned for AWS/R2 key patterns, private-key blocks,
`aws_secret` / `R2_*` / `KAGGLE_*` values, inline passwords, IP addresses and pod or
host references. **The only matches are variable names and empty placeholders**:
`.env.example` (four names, no values), `src/sentinel_data/config.py` (`os.environ[...]`
lookups), `docs/DATA.md` and `docs_gen.py` (`KEY` / `SECRET` in an illustrative
snippet), and `127.0.0.1` in `fprime/SentinelRef/README.md`, a loopback in a GDS
command line. No telemetry file, no licence-restricted artifact, no person's name, no
meeting, no reviewer, no acknowledgement, no pod address.

**One disclosure item, and it is not a defect.** 53 of the 152 commits on `dev` carry
the author's personal email address; the other 99, and all seven on `main`, use the
GitHub noreply address. Removing it would require rewriting `dev`, which is forbidden.
**It is disclosed in a note and the history is not rewritten** -- `dev` is the audit
trail this project's arguments rest on, and rewriting it to tidy a mail header would
cost more than the header is worth. The note is drafted in the JSON's
`branch_proposal.disclosure_note_author_email`.

### What this plan costs the size budget

`docs/reorg_plan.json` is about 313 KB and this document about 40 KB. Together they
move tracked content from **3.22 MiB to roughly 3.58 MiB of the 4 MiB cap** -- from 81%
to about 89%. That is real and it is stated rather than discovered later: the inventory
the brief asks for is a row per tracked file, and a row per tracked file is what it
costs. The remaining headroom is roughly 430 KB, which the additions in section 5 fit
inside, and `tests/test_no_local_persistence.py` will fail loudly if anything does not.

---

## 2. Where the brief and the repository disagree

The brief's own rule is that the repository wins. Twelve disagreements were found; the
JSON's `brief_corrections` carries all twelve with evidence. These five matter.

**The largest: R1/R2/R3 and the state of the freeze.** The brief says three arms are
queued and that *"D62's freeze is withdrawn pending them."* **No document names R1, R2
or R3, and no document withdraws D62.** `docs/MODELS.md:9523` reads *"Nothing is
registered. D62's freeze stands. `docs/MODELS.md` has no arm against this"*, and
`docs/STATUS.md:269` reads *"Nothing is registered against this yet."* Resolved in
favour of the repository: **the freeze stands and nothing was registered in this
session.** What this plan does instead is *describe the shape* of a reserved
`docs/MODELS.md` section 37, so that whoever registers the arms does not have to invent
the frame -- see roadmap item 2. The evidence justifying the reservation is committed at
`4136cab`.

**The OCaml integration layer does not exist in the repository.** The string "OCaml"
occurs nowhere in the tracked tree -- not in `Objective.md` 12's four-phase roadmap, not
in `docs/STATUS.md` 7, not in `docs/DECISIONS.md`, not in `CHANGELOG.md`. It is carried
as a real roadmap item **and given the provenance every other item here has**: a
proposed `docs/DECISIONS.md` entry recording the scope decision, grounded in F's own
language-integration precedent, in CPP-1 and CPP-25 as quoted in `docs/PHASE5.md` 3, and
in `Objective.md` 11 rule 5. Its zero-allocation capability is marked **UNVERIFIED**
until the mechanism is cited by file and version -- exactly the treatment
`docs/PHASE5.md` 4 already gives the Armadillo static-allocation claim.

**ESA-ADB is CC BY 3.0 IGO, not CC-BY 4.0.** Stated identically in five places, one of
them (`docs/manifest.snapshot.json:11`) written by the ingest itself. See section 6 for
the verification caveat that travels with it.

**Sixty-six citations point into `third_party/`, not "~78" -- and not the 64 the
repository itself advertises.** `docs/TELEMANOM_EXCERPTS.md:11` and `docs/INDEX.md:16`
both say 64, and the section 5 table has exactly 64 rows, under a header promising it is
*"Generated from the tree on 2026-09-09, not hand-maintained."* It has drifted by two
rows. See section 4.

**"Two tests pin its SHA and licence" is one test.**
`tests/test_layering.py:76-81` pins both; a different test enforces non-importability.
The protection is real but narrower than described -- it checks `PROVENANCE.md`'s text,
not the vendored files' content. Recorded, not changed.

---

## 3. The findings, and the order they are fixed in

### Tranche 1 -- the guards, and the sources describing a repaired defect

**3a. A guard blinded by a line break.**
`tests/test_documents_are_current.py::test_the_live_documents_state_the_real_test_count`
exists to stop a stale test count. It matches `r"(\d{3,4}) tests"` against raw file
text. In `docs/STATUS.md` the stale **605** is wrapped -- `**605` ends line 321,
`tests**,` begins line 322 -- so the regex, which needs a single space, never sees it.
Both live documents present as `{606}` and the guard passes on a document that
contradicts itself seventeen lines apart. Fix: normalise whitespace before matching.
**Applying it fails the suite until `docs/STATUS.md:321` reads 606**, which is the
point; the two land in the same commit.

**3b. A guard that looks for one phrasing of five.**
`test_no_new_live_occurrence_of_a_retired_claim` was run directly to confirm its state:
it returns exactly six sites and equals its `BASELINE`, so it is honest about what it
looks for. Its `lead` pattern is `\+26 timesteps|26 timesteps early`. In the tree, the
retired `+26` lead also appears as `26-timestep head start`
(`docs/RESULTS.md:90`), as a bare table cell (`:216`, `docs/DECISIONS.md:289`), as
*"a median of 26 timesteps ahead"* (`:226`) and as a sweep anchor (`:332`); and the
retired ratio appears as `28/32` at `:332`, which the `28 of 32` pattern misses. The
patterns are widened to all of them.

> **The consequence is stated rather than left to be discovered.** The test asserts
> set equality in **both** directions, so widening the patterns without re-pinning
> `BASELINE` in the same commit fails the build. Each newly caught site is classified
> first: `docs/RESULTS.md:216` and `docs/DECISIONS.md:289` are measured-at-the-time
> tables and are **pinned as records**; `:90`, `:226` and `:332` are live prose and are
> **corrected**. A record corrected in hindsight is not a record.

**3c. A generator that silently deletes recorded provenance.** `docs/DATA.md` is
overwritten wholesale by `docs_gen.write_data_md`
(`src/sentinel_data/docs_gen.py:201`, `path.write_text(text)`), destroying the
hand-added 56-line SMAP/MSL block. `docs/DATA.md:65` warns about this in prose and
nothing enforces it -- and D15's own words apply: the rule that was supposed to catch
this is a convention, and a convention has no test. Fix: **refuse rather than warn.**
Hand-authored regions are delimited, preserved verbatim, or the write raises and names
what it could not re-anchor.

**Three sources still describe a defect that has been repaired.** `baselines._rolling`
now promotes to float64 before accumulating (`baselines.py:51`) and sums at
`dtype=np.float64` (`:56`), with `tests/test_rolling_precision.py` pinning it. But:

| Site | Still says |
|---|---|
| `src/sentinel_models/baseline_reference.py:10-20` | *"It builds prefix sums with `np.cumsum` over a float32 array ... `baselines.py` is not changed here; that repair is scoped as `docs/MODELS.md` 21."* |
| `flight/include/sentinel/Baseline.hpp:19-27` | *"The harness repair is `docs/MODELS.md` 21 and **has not run yet**, so the two are knowingly different numbers today."* |
| `flight/src/Baseline.cpp:138-140` | *"Cancellation cannot drive this negative the way `_rolling`'s float32 prefix sums do (D37)"* |

All three are corrected **in the house form, with the old text kept in quotation** --
nothing overwritten, the superseded sentence beside the current one with its
D-reference. Full replacement text per site is in the JSON.

### Tranche 2 -- stale live figures and dangling citations

**Nine stale figures across three live documents.** `docs/STATUS.md` says "D1 to D57"
twice (the register runs to D62), 605 tests (606), "work items 9.5 to 9.14" (9.19) and
`check_no_list` "75 files" (76). `README.md` says "D1 to D57", Releases "`wi1` to
`wi7`" (nine tags exist) and the same work-item range twice. `docs/INDEX.md` says "64
cited locations" (66) and the same range.

**Two genuine dangling citations, both single-character-class typos.**
`docs/MODELS.md:9288` cites `runs/smap-msl/_forensics/2026-09-09T2138*`; its two
companions (`2240*`, `2245*`) resolve and nothing in that directory begins `T2138` --
the artifacts in that hour are `T213116Z` and `T213437Z`. And
`docs/TELEMANOM_EXCERPTS.md` section 5 has no row for `errors.py:337-339`, which is
cited four times and carries the load-bearing *"three-term test"* finding, nor for
`errors.py:62`; three further rows carry a wrong *Cited by* column.

**`CHANGELOG.md`'s link block is broken at both ends.** Definitions stop at `[0.6.1]`,
so 0.6.2 through 0.6.19 have none, and the two that exist point at a tag **`wi9.5`
that does not exist** -- the tags are `wi1`..`wi9`. The changelog *entries* are records
and are not touched; only the link block at the foot is repaired.

**A retired framing surviving as live text.** `docs/RESULTS.md:228` reads *"For a
component whose purpose is warning **hours** before a limit trips"* -- eighteen lines
after `:210` says no figure on this data may be expressed in hours, and nineteen before
`:247` says *"No wall-clock or 'hours' figure may be derived from anything in this
document."* It states a purpose, not a measured figure, so it is not a fabricated
number; it is the surviving instance of a framing `Objective.md` 0 and 1 struck on
2026-09-09. Corrected in the house form.

### What the audit found clean, and it is worth saying

Two of the six reference checks came back with **zero defects**: all 629 section
citations resolve, and all 110 cited repository paths exist, along with 29 relative
markdown links and 3 pytest node ids. No cited line into `third_party/` exceeds its
file. For a repository of this size and citation density that is a real result, and the
new link-check guard exists to keep it that way rather than to discover it.

Two conventions the guard must be taught, because a naive checker flags them wrongly:
`Objective.md 14.N` is a **table row**, not a heading (stated exactly once, at
`Objective.md:1187`); and five cited `docs/...` paths are upstream `nasa/fprime`
documentation that namespace-collides with this repository's `docs/`. One in reverse:
`detector.py:167-173` at `scripts/smap_rungs.py:1263` points into the **vendored**
`detector.py` (254 lines), not `src/sentinel_eval/detector.py` (163), and must not be
flagged.

---

## 4. The inventory

`docs/reorg_plan.json`'s `inventory` carries one row per tracked file -- 223 rows --
each with `path`, `bytes`, `purpose`, `referenced_by[]`, `produced_figures[]`,
`action`, `destination`, `reason`, `risk` and `breaks_if_actioned[]`. Purposes are read
from each file's own docstring or leading comment, not invented. The reference graph
runs over every tracked file **plus the full `git log` message corpus**, and is
augmented with an AST import graph (absolute and relative) so that a module reached only
through its package is not mis-reported as orphaned.

**Actions: 194 KEEP, 29 ANNOTATE, 0 MOVE, 0 RENAME, 0 SPLIT, 0 MERGE, 0 ARCHIVE,
0 DELETE.**

Nothing is deleted, and the rules that forbid it are not merely honoured but tested
against: every one of the 28 scripts has its `produced_figures[]` filled in, so the
never-delete rule has something to bind on. `docs/NARRATIVE.md` 11 records what deleting
a study script cost -- four pre-registered rungs became unreproducible -- and that is why
the column exists.

**Four files have no inbound reference of any kind, and each is a finding:**

| Path | Why it matters |
|---|---|
| `src/sentinel_data/ingest_esa_adb.py` | 342 lines, the largest module in `sentinel_data`. Imported by nothing, covered by no test, cited by no document -- and it is **how the primary dataset reached the bucket**. Never deleted; it gains a provenance header naming the work item, the run date and the manifest it wrote. |
| `scripts/bench_forecaster.py` | The only script behind no documented figure. It prices a run before an operation is spent, which is operationally live. Kept, annotated, and given a row in `scripts/README.md`. |
| `flight/README.md` | A good document that nothing points at and `docs/INDEX.md` does not list. Gains an INDEX row. |
| `third_party/telemanom/README.md`, `requirements.txt` | Expected and already explained: `docs/TELEMANOM_EXCERPTS.md:182-186` records that they are vendored because the tree is a copy of a commit rather than a selection from one. Not a defect. |

---

## 5. Structure and naming

### The convention, as rules

1. **`scripts/` stays flat.** Lifetime is recorded in `scripts/README.md`, never in the
   directory tree.
2. **A study script's work item is stated in the first line of its docstring.** Every
   script already does this.
3. **Test files are `tests/test_<subject>.py`**, except where the test *is* the claim.
4. **`runs/` artifacts** are `<task>/<detector>/<ISO8601-basic-Z>-<8-hex>.json`, or
   `<task>/_<study>/<ISO8601-basic-Z>-<label>.json`, with weights at
   `runs/_weights/<32-hex>.npz`. This is what the tree already does; the rule is written
   down because it never was.
5. **Documents never wrap a citable token across a line break** -- a `runs/` path, a
   repository path, a section number or a count. Measured cost of breaking this: **8 of
   the 77 cited `runs/` paths** are unresolvable as written for this reason alone, and
   `docs/STATUS.md`'s stale count is invisible to its own guard for the same reason.
6. **A generated document cannot be regenerated over hand-authored content.**
7. **A published section number is never renumbered.** It is the citation target for 629
   cross-references.
8. **`src/sentinel_*` module names describe the artifact, not the experiment.**

The twelve files that violate one of these today are listed in the JSON's
`naming_convention.violations`, with the note that **none is renamed** -- in every case
the reference cost exceeds the clarity gain, and `scripts/README.md` carries the mapping
instead.

### Why nothing moves, and what moving would cost

`scripts/` looks like the obvious candidate for an `ops/` and `studies/` split. It is a
**26-file edit with four distinct breakage classes**, two of which fail late rather than
at import:

- all **25** Python scripts compute the root as `Path(__file__).resolve().parents[1]`;
- `tests/test_no_list.py:9` and `tests/test_golden_vectors.py:23` put `ROOT/"scripts"`
  on `sys.path` and import by module name;
- `scripts/smap_forensics_38.py:50` loads `smap_rungs` by **file path** via
  `spec_from_file_location` -- this breaks at runtime, not at import;
- `tests/test_documents_are_current.py` pins the literal key
  `("scripts/threshold_sweep.py", "lead")` in its `BASELINE`.

The full edit list is in the JSON so the option stays open; the recommendation is to
revisit it **after** the link-check guard exists, because the guard is what makes the
move safe to attempt.

Regrouping `docs/` by audience costs 629 section citations and 110 path citations to
rewrite, plus a permanent cost that cannot be recovered: every `docs/` path in 152
commit messages becomes stale, and commit messages cannot be rewritten because `dev` is
never rewritten. **Grouping by audience is delivered by `docs/INDEX.md`'s reading order,
which already exists and costs nothing to reorder.**

### What is added instead (all additive, later commits)

```
  scripts/README.md                 one row per script: lifetime, work item,
                                    artifact written, document figure produced
  docs/datasets/ESA_ADB.md          obtainable without the bucket
  docs/datasets/SMAP_MSL.md         obtainable without the bucket
  docs/datasets/OTHERS.md           every dataset in docs/RESEARCH.md, used or not
  docs/datasets/REPRODUCING.md      what a reader can recompute without R2
  docs/PI_ENVELOPE.md               RESERVED AND EMPTY -- shape fixed, no numbers
  tests/test_references_resolve.py  the link-check guard
```

plus generated tables of contents for `docs/MODELS.md` (441 headings) and
`docs/DECISIONS.md`, sub-headings for `docs/NARRATIVE.md` section 6 (a 366-line
unsectioned block, 30% of that document), and fixes for the four structural defects a
table-of-contents generator trips on in `MODELS.md`: a duplicate heading `26.6`,
`33.7`/`33.8` before `33.6`, `34.8` before `34.7`, and OBSERVED blocks at `###`
everywhere except `35.7`. **No section number changes.**

---

## 6. Data, for readers who will never have the bucket

The JSON's `data_reference_docs` carries ten entries. Two are load-bearing.

**ESA-ADB.** Kotowski et al., *European Space Agency Benchmark for Anomaly Detection in
Satellite Telemetry*, `arXiv:2406.17826`, submitted 25 June 2024, revised 17 August
2025, *"87 pages, 24 figures, 19 tables"* -- **fetched and confirmed field by field this
session**. Data at Zenodo record `15237121`, DOI `10.5281/zenodo.15237121`, concept DOI
`10.5281/zenodo.12528695`. Licence **CC BY 3.0 IGO**, attribution to ESA required.

> **(!) The licence is verified against this repository only, and that is said rather
> than hidden.** `zenodo.org` was unreachable from this environment -- TLS *"unable to
> get issuer certificate"* -- on both the record URL and the DOI redirect. The string is
> identical in five internal places, one of which was written by the ingest itself at
> `2026-08-24T21:31:02Z`. It is marked `verified: "repository-only"` and **must be
> re-checked at the Zenodo record before publication.** The brief says CC-BY 4.0; the
> repository says 3.0 IGO; neither is asserted beyond what was actually checked.
>
> **(!) SETTLED 2026-09-11. The record was opened and the Rights field reads
> `CC BY 3.0 IGO`** -- record `15237121`, version v2, published 2025-04-17. **The
> repository was right and the brief was wrong.** This paragraph is kept as the record of
> what was and was not known on 2026-09-09; `docs/datasets/ESA_ADB.md` carries the
> verification.

**NASA SMAP/MSL.** Hundman et al., KDD 2018, `arXiv:1802.04431`; source at
`github.com/khundman/telemanom`, vendored here at `2e6c5b6c`. **BSD 3-Clause, verified
by reading `third_party/telemanom/LICENSE.txt` directly.** Clause 3, read verbatim at
line 11, is an obligation on this project and not only on its code:

> **No document in this repository may present this project as endorsed by, affiliated
> with, or produced by Caltech or the Jet Propulsion Laboratory.** Naming telemanom's
> authorship and citing the paper is description, not endorsement, and remains correct.

Recorded with it: some secondary sources say Apache-2.0 and are wrong; `docs/DATA.md`
did too until 2026-09-08; and the stored R2 manifest object `_manifest/smap_msl.json`
**still carries the wrong string.** Correcting it costs one Class A and is now approved
-- see section 9.

Both entries carry the full layout, channel counts, what was ingested and what was not,
the manifest key structure, and the caveats a reader must carry: ESA-ADB's anonymised
and scaled clock, its per-group min-max normalisation, its irregular per-channel
sampling and zero-order-hold requirement; SMAP/MSL's column 0 being telemetry with the
rest one-hot commands, and the labelling defect carried rather than silently
deduplicated -- channel `P-2` appears **twice** with conflicting spans, so the dataset
has **81 unique channels, not the 82 rows** its own label file implies.

**Reproducing without the bucket.** Recomputable by anyone, at zero cost and with no
credential: the full 606-test suite, `check_no_list.py`, `sentinel_eval selftest`,
`make -C flight test` and `lint` -- including the committed golden vectors `g1`/`g2` and
baseline vectors `b1`-`b4`, which are *in the repository* -- a scored run on the
synthetic fixture, all of `third_party/` and every one of the 66 citations into it, and
`docs/manifest.snapshot.json`. **Not** recomputable: any figure scored on ESA-ADB or
SMAP/MSL. What stands in: the 144 committed-by-citation scorecards under `runs/`, each
cited by path, each carrying its own provenance and measured operation counts. A reader
can verify that every number came from a named artifact; they cannot re-derive the
telemetry-scored numbers, and **no wording in the public material should suggest
otherwise.** The data itself is obtainable independently from Zenodo and from the Kaggle
mirror the telemanom README names -- the bucket is a convenience and an operation
budget, not a gate.

---

## 7. Branches

> **(!) SUPERSEDED IN PART 2026-09-10 by `docs/DECISIONS.md` D67, and kept.** This
> section proposes a `master` whose tree is **identical** to `dev`'s head. D67 decides
> instead that **`master` is curated** -- the component and the evidence it works, and
> nothing else -- with every omission stated and off-branch paths resolving on `dev` at
> a named commit. What survives here is the mechanics: branch from `main`, never rewrite
> `dev`, never force-push, leave the tags and Releases where they are.
> **D67.1 (2026-09-11) then added `docs/NARRATIVE.md` and `docs/PI_ENVELOPE.md` to the
> curated set**, as a rider rather than an edit to D67's lists.

**Current state, verified.** `dev` has 152 commits and the full history. `main` has 7,
one per approved checkpoint, and its tree is byte-identical to `dev` commit `9b2fb1f`.
Tags `wi1`..`wi9` **all point at commits reachable only from `dev`** -- none is on
`main` -- so their Releases link into `dev`'s history and will continue to whatever
happens elsewhere. GitHub's default branch is `main`.

**Recommended: keep `main`, add `master` branched from it, and make `master` the
GitHub default.**

```
  git branch master main          # master starts at 439f400 and inherits main's
                                  # ancestry, so its seven checkpoint commits are
                                  # visible in git log
  # then one snapshot commit on master, tree identical to dev's head
  # then change the default branch to master in repository settings
```

It is the only option that changes nothing existing: tags stay, Releases stay, no doc
SHA moves, no link breaks. Renaming `main` (option B) spends link integrity for
tidiness, and GitHub's redirect is silent, so a stale link appears to work while
pointing somewhere the reader did not choose. An orphan `master` (option C) gives
maximum content control and no auditability -- `git log` would show one commit and no
history -- which is the wrong trade for a repository whose whole argument is that its
record is inspectable.

**On making `master` the default, with the consequence stated both ways.** With the
change, new clones and the repository landing page get the curated branch -- which is
the point of having one -- and `README.md` must say plainly that `dev` carries the full
history. Without it, `master` is opt-in, nobody finds it, and the public branch does not
do the job it was made for. **The change is recommended.** `README.md`'s branch
paragraph is rewritten in the same commit; proposed text is in the JSON.

**Content.** `master`'s tree is identical to `dev`'s at the checkpoint it snapshots. **If
that ever ceases to be true the difference is documented file by file, because a public
branch that silently omits material is worse than one that includes it with an
explanation.** No file is omitted under this plan.

**What must never reach a public branch** -- eight categories, each with its verified
status, in the JSON's `branch_proposal.must_never_reach_public`: credentials, pod and
host addresses, telemetry, licence-restricted artifacts, personal names, any claim of
Caltech or JPL endorsement, any assistant attribution, and any retired claim restated as
live.

**The standing rules survive any choice.** `dev` is never rewritten. No force-push,
ever. One snapshot commit per user-approved checkpoint, tree identical to `dev`'s head.
Commits authored `GalacticDroid448` only. No assistant attribution anywhere. No person,
meeting or reviewer reference anywhere. Nothing is pushed unless the user says so.

> **No history rewrite is proposed or permitted**, by this plan or by any option in it.
> Option A requires none; B and C would not either, and are rejected on other grounds.

---

## 8. What the public branch must answer, and the roadmap

The reading order -- nine steps, about fifteen minutes -- is in the JSON's
`public_reading_order`, together with a claim-to-evidence table. Every engineering claim
was verified at its source this session:

| Claim | Evidence | Caveat |
|---|---|---|
| It warns before a limit trips | D46: 39 of 43 contextual anomalies stay inside their channel's historical range. `docs/MODELS.md` 26.18: 10 of 38 at **0.6820%** (was 0.6838%, the commanded arm's rate at 6/38) | It is 26%, on one dataset, with no floor to compare against at that rate |
| **The shipped core runs D25's static rule, not D62's frozen dynamic one** | `Detector.hpp:3-7`; `docs/STATUS.md:271-274` | **A stated limitation, not a footnote.** Roadmap item 1 |
| It is warn-only | `Monitor.fpp` declares **zero commands**; `cmdIn` exists only for F's autocoded `PARAM_SET`/`PARAM_SAVE`, said in situ at `Monitor.fpp:201-205` | None. Warn-only by interface, not by convention |
| It fails safe | 11 refusal codes in `Status.hpp`; 11/11 degrade to Level 1, 0/11 fail the topology (D32-D37) | None |
| It builds in F' v4.3.0 | `docs/FPRIME.md` 4: F's own Ref built from this toolchain in 12.4 s | `Ref` moved at v4.3.0 and it is not in the breaking-change notes |
| The C++ matches the Python to 1e-5 | `tests/test_reference_equivalence.py:36` `TOLERANCE = 1e-5`; **measured 1.8e-07** (D30) | Quote both -- the measurement is two orders better than the tolerance |
| Measured CPU, RAM and per-tick timing on a Pi | **None. Does not exist** | `docs/PI_ENVELOPE.md` is reserved and empty. No figure until measured, for inference *and* training, logged |
| What it catches, at what false-alarm rate | `docs/STATUS.md` 4's arm table; `RESULTS.md` 6k, 6l, 6m, each row citing its artifact | Every comparison at a matched nominal alarm rate (D41, D44); recall never alone; `n < 20` stamped UNDERPOWERED |

**The roadmap** is nine items with a definition of done and dependencies each, in the
JSON's `roadmap`. Item 1 is carrying the dynamic threshold into the C++ core, and it is
first because until it lands the flight component cannot reproduce the project's own
headline result. Item 2 is R1/R2/R3, stated as the repository has it -- **the freeze
stands, nothing is registered** -- with the shape of a reserved `docs/MODELS.md` section
37 described so that whoever registers the arms inherits a frame rather than inventing
one: three arms, one lever each (pruning's `p`; a derivative and horizon-disagreement
statistic; both), all measured at or below stage 4's 0.6820% (was 0.6838%) and scored against its own
38, with the standing bar being the 10 of 38 that nine arms have already failed to beat.
Then the Pi envelope, WI10, the toolkit, the Ref physics testbed, the OCaml layer,
Phase 5, and licence and release.

---

## 9. Execution order, and what is still open

Five tranches, each with its verification commands and its stop conditions, in the
JSON's `execution_order`. The ordering principle is that **a guard that passes while its
subject is wrong is repaired before anything cosmetic**, because every later tranche
relies on those guards to keep its work correct.

1. **The guards, and the sources describing a repaired defect** (section 3).
2. **Stale live figures and dangling citations** -- including this document's own:
   `0.6838%` was corrected to `0.6820%` on 2026-09-10 wherever it stood as the frozen
   arm's rate, in the house form. 0.6838% is the **commanded** arm's rate at 6/38;
   26.18's table gives the frozen `gru` arm **0.6820% at 10/38**. It had reached the
   public reading order's claim-to-evidence table, which is where it would have become
   public material.
3. **Additive legibility, and the link-check guard that keeps it.**
4. **The dataset documents and the public reading order.**
5. **The branch work.**

Stop conditions worth repeating here: if the widened guard catches a site nobody can
classify as either a record or live prose, **stop and report rather than pinning it to
make the suite green**; if adding a citation to `telemanom.py` would require reading a
line from memory, **stop** -- every one is read from `third_party/` and cited by file
and line, never recalled; if a licence cannot be verified, **mark it unverified rather
than stating it**; and nothing is pushed on this plan's authority.

**Six questions remain open**, in the JSON's `open_questions`. Four are now decided and
recorded there: the reserved section 37's shape is described now and written when the
arms are registered; `master` becomes the GitHub default; the author email is disclosed
in a note with **no history rewrite, ever**; and correcting the stored manifest's licence
string is **approved at one Class A** -- not executed here, because this task is scoped
at zero operations, but scoped as a standalone step that writes the artifact before the
ledger and commits the ledger inside its `try`/`except`, so a ledger failure reports
loudly instead of destroying the result.

Two stay open for later: whether a physical `scripts/` split is worth 26 edits once the
link-check guard exists, and whether `telemanom.py`'s dead `Z_LIMIT` and `window_ratios`
should be marked or removed. The plan recommends marking and does neither.
