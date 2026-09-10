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

[Unreleased]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi9.5...dev
[0.6.1]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi9...wi9.5
[0.6.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi8...wi9
[0.5.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi7...wi8
[0.4.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi3...wi7
[0.3.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi2...wi3
[0.2.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/compare/wi1...wi2
[0.1.0]: https://github.com/GalacticDroid448/fprime-DeepLearning-Sentinel/releases/tag/wi1
