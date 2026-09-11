# ESA-ADB, for a reader who will never have the bucket

> **Paths outside this branch resolve on `dev`.** `master` carries the component and the
> evidence it works, and nothing else (`docs/DECISIONS.md` D67). A citation here into
> `src/`, `scripts/`, `tests/`, `docs/MODELS.md` or `third_party/` points into the
> development branch at the commit this snapshot was taken from.
> `scripts/check_references.py --master` is what keeps that true rather than hoped.

**The primary dataset (D1), and the one this project's core claim needs.** It is the only
public set with real, time-synchronised, multivariate satellite telemetry labelled by the
operations engineers who flew the missions -- which is what a cross-channel claim has to be
measured on. **No telemetry from it is in this repository and none ever will be**
(`Objective.md` 11): it is staged to a private R2 bucket, read manifest-addressed, and
every figure derived from it is cited by artifact path.

## The source

| | |
|---|---|
| Paper | Kotowski, Haskamp, Andrzejewski, Ruszczak, Nalepa, Lakey, Collins, Kolmas, Bartesaghi, Martinez-Heras, De Canio, *European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry* |
| arXiv | `2406.17826`, submitted 25 June 2024, revised 17 August 2025 (v2), *"87 pages, 24 figures, 19 tables"* |
| Data | Zenodo record `15237121` |
| DOI | `10.5281/zenodo.15237121` (this version), `10.5281/zenodo.12528695` (concept) |
| Archives | `ESA-Mission1.zip` 3,776,246,073 B, `ESA-Mission2.zip` 4,098,539,932 B, `ESA-Mission3.zip` 3,734,403,444 B, MD5s in the manifest |

**The paper's metadata above was re-fetched and confirmed at `arxiv.org` on 2026-09-11**,
field by field: title, the eleven authors in order, both dates, and the comments string.

## (!) The licence is NOT verified at source, and that is stated rather than hidden

**This repository records the dataset licence as `CC BY 3.0 IGO`, with attribution to ESA
required in anything published from it.** The string appears identically in five places --
`README.md:226`, `Objective.md:672`, `docs/DATA.md`, `docs/REORG_PLAN.md`, and the stored
manifest, the last of which was **written by the ingest itself at
`2026-08-24T21:31:02Z`** and is therefore the closest thing here to a primary record.

**It has not been checked against the Zenodo record, and two attempts have now failed.**
`zenodo.org` is unreachable from this environment -- TLS *"unable to get issuer
certificate"* -- on the record URL, on the DOI redirect, and on the redirect's target, on
**2026-09-09 and again on 2026-09-11**. `arxiv.org` answers from the same environment, so
this is Zenodo-specific rather than a general lack of network.

> **It is marked `verified: "repository-only"` and MUST be re-checked at the Zenodo record
> before this repository is published.** An earlier brief said CC-BY **4.0**; this
> repository says **3.0 IGO**. **Neither is asserted beyond what was actually checked**,
> and a reader who needs the licence should read the record rather than this page.
>
> The reference *code* published with the benchmark is MIT (Airbus, KP Labs, Wenig &
> Schmidl) and is a **separate licence from the dataset's**. Nothing in this repository
> derives from that code.

## What was ingested

Three missions, staged under `esa-adb/v1/` and addressed only through
`_manifest/manifest.json`. **Never LIST, never glob** -- `scripts/check_no_list.py` refuses
code that would.

| Mission | Channels | Target channels | Telecommand series |
|---|---|---|---|
| `mission1` | 76 | 58 | yes |
| `mission2` | 100 | 47 | yes |
| `mission3` | 48 | 24 | no |

Annotations are separate: `esa-adb/v1/annotations/labels.parquet`, **17,342 rows**,
154,633 B, SHA-256 `8a619e35...`, with a channels table beside it. The label enum is
`NOMINAL 0, ANOMALY 1, RARE_EVENT 2, GAP 3, INVALID`.

## (!) Four caveats that will silently corrupt a result

Each of these is a property of the published data, not of this project's handling of it.
`docs/DATA.md` 5 is the normative statement; this is the same content for a reader who
does not have that file.

1. **Timestamps are anonymised mission time, not UTC.** Every mission's timeline was
   scaled by an **undisclosed factor greater than 1** and shifted to start 2000-01-01.
   Absolute dates are meaningless, elapsed durations are stretched by an unknown constant,
   and real sampling periods **cannot be recovered**. **This is why no figure in this
   repository is ever quoted in hours** -- every lead time is in timesteps, and a
   time-to-limit number can come only from the F' Ref physics testbed on a real clock.
2. **Values are min-max normalised to [0, 1] within each channel group.** Units are
   meaningless and the `Physical Unit` field is itself anonymised. Relative dependencies
   *within* a group survive -- which is exactly what a cross-channel method needs -- but
   absolute magnitudes are not physical.
3. **Sampling is irregular and per-channel.** Every channel has its own timestamp grid and
   they start and end at different times. **ESA prescribes zero-order hold**, never linear
   or Fourier interpolation. The manifest's `reference_hints` carries their per-mission
   resampling rules (`mission1` 30 s, `mission2` 18 s), the dominant sampling rates, and
   the ranges of **monotonic channels that must be differenced** -- `mission1` channels 4
   and 11, `mission2` channels 29 and 46. **These are magic numbers from ESA's own
   scripts and are derivable from no CSV.**
4. **Not every channel is numeric.** Some are categorical and stored as strings. Check
   `value_dtype` in the manifest before assuming arithmetic works.

## What it was used for here, and what it was not

**Used for:** the architecture gate (LSTM / GRU / TCN, D28), the decision layer's
calibration and its transfer failure (D29), the two held-back sets, and the sealed
transfer exam. **The finding that a threshold calibrated on one period of one spacecraft
does not transfer to a later period of the same spacecraft is ESA-ADB's**, and it is why
the threshold is recalibrated in orbit while the model stays frozen (D63).

**Not used for:** the 38 in-range contextual anomalies that carry the headline claim.
Those are SMAP/MSL's -- see `SMAP_MSL.md` -- because ESA-ADB does not have that population.

## Obtaining it yourself

Download the three archives from the Zenodo record, **read the licence there** (see
above), and stage them however you like. Nothing in this repository requires the bucket
layout: `REPRODUCING.md` lists exactly what can be recomputed without any dataset at all,
which is more than a reader expects.
