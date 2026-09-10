"""Regenerate docs/DATA.md and docs/manifest.snapshot.json from the manifest.

Both are written from the same manifest object that was just uploaded, as the
final step of every ingest, so they cannot drift from what is actually in R2.

DATA.md explains and points. It deliberately does NOT restate row counts,
checksums or byte sizes: those live in the manifest alone, because duplicating
them into prose guarantees drift and then nobody knows which copy is right.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import (
    ANNOTATIONS_PREFIX,
    ARCHIVE_PREFIX,
    DOCS_DIR,
    LEDGER_KEY,
    MANIFEST_KEY,
    OPS_CEILING,
    OPS_TRIPWIRE,
)
from .manifest import dumps

#: Delimiters for prose the generator must carry across a regeneration.
#: Anything between them is preserved verbatim and re-anchored to the generated
#: line that precedes it.
HAND_OPEN = "<!-- hand-authored -->"
HAND_CLOSE = "<!-- /hand-authored -->"

#: The footer line that records what the generator last produced. It is the
#: whole of the guard: if the file on disk no longer hashes to it, somebody
#: edited the file and the generator must not silently discard that edit.
DIGEST_PREFIX = "<!-- generated-digest: "
_DIGEST_RE = re.compile(r"^<!-- generated-digest: ([0-9a-f]{64}) -->$", re.M)


class HandAuthoredContentWouldBeLost(RuntimeError):
    """Raised instead of overwriting prose the generator did not write.

    `docs/DATA.md` carries hand-added prose inside a file whose own footer says
    not to edit it by hand -- section 1's SMAP/MSL block, and three `(!)` riders
    recording provenance corrections. `write_data_md` used to end in a bare
    `path.write_text(text)`, so the next ingest would have deleted all of it and
    said nothing. The file warned about that in prose, which is not a guard.

    Refusing is the whole behaviour: the artifact is not written, the operator is
    told which lines are at risk, and the fix is theirs to make -- wrap the prose
    in `HAND_OPEN`/`HAND_CLOSE` so it is carried across, or delete the file to
    ask for a clean generation on purpose.
    """


_HAND_BLOCK_RE = re.compile(
    re.escape(HAND_OPEN) + r".*?" + re.escape(HAND_CLOSE), re.S)


def _without_digest(text: str) -> str:
    """The document as it is written: digest line removed, tail trimmed.

    Hashing the raw text does not round-trip -- removing the footer leaves the
    blank line that preceded it -- so both the stamp and the check normalise
    here first.
    """
    return _DIGEST_RE.sub("", text).rstrip()


def _digest_body(text: str) -> str:
    """The **generated** part alone: what the digest actually covers.

    Delimited hand-authored regions are excluded, because those are the parts a
    person is invited to edit. If they counted towards the digest, fixing a typo
    inside one would make the next ingest refuse, and the lesson everyone would
    take from that is to delete the footer. The guard binds on the generated
    text and nothing else.
    """
    return _HAND_BLOCK_RE.sub("", _without_digest(text)).rstrip()


def _digest(text: str) -> str:
    """SHA-256 over the generated body."""
    return hashlib.sha256(_digest_body(text).encode("utf-8")).hexdigest()


def _stamped(text: str) -> str:
    """The full document -- hand-authored regions included -- with its footer."""
    return f"{_without_digest(text)}\n\n{DIGEST_PREFIX}{_digest(text)} -->\n"


def _hand_authored_blocks(text: str) -> list[tuple[str, str]]:
    """`(anchor, block)` for each delimited region, in order.

    The anchor is the last non-blank line before the region, which is what the
    region is re-attached to in the newly generated text.
    """
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        if lines[i].strip() == HAND_OPEN:
            j = i + 1
            while j < len(lines) and lines[j].strip() != HAND_CLOSE:
                j += 1
            if j >= len(lines):
                raise HandAuthoredContentWouldBeLost(
                    f"{HAND_OPEN} at line {i + 1} is never closed; refusing to write.")
            anchor = next((ln for ln in reversed(lines[:i]) if ln.strip()), "")
            out.append((anchor, "\n".join(lines[i:j + 1])))
            i = j + 1
        else:
            i += 1
    return out


def _carry_over(generated: str, existing: str) -> str:
    """Re-attach every delimited region to its anchor in the generated text."""
    for anchor, block in _hand_authored_blocks(existing):
        if not anchor or generated.count(anchor) != 1:
            raise HandAuthoredContentWouldBeLost(
                "a hand-authored region cannot be re-anchored: its preceding line "
                f"{anchor.strip()!r} appears {generated.count(anchor)} times in the "
                "newly generated text, not once. Refusing to write -- move the "
                "region under a line the generator emits exactly once.")
        generated = generated.replace(anchor, f"{anchor}\n\n{block}", 1)
    return generated


SNAPSHOT_HEADER = (
    "Snapshot for reference only. R2's _manifest/manifest.json is authoritative. "
    "Regenerated on every ingest."
)


def write_snapshot(manifest: dict, docs_dir: Path = DOCS_DIR) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    annotated = {"_comment": SNAPSHOT_HEADER, **manifest}
    path = docs_dir / "manifest.snapshot.json"
    path.write_bytes(dumps(annotated))
    return path


def write_data_md(manifest: dict, bucket: str, docs_dir: Path = DOCS_DIR) -> Path:
    ds = manifest["datasets"]["esa-adb"]
    missions = ds["missions"]
    docs_dir.mkdir(parents=True, exist_ok=True)

    mission_rows = "\n".join(
        f"| `{name}` | {m['channel_count']} | {m['target_channel_count']} | "
        f"{'yes' if m.get('telecommand_series') else 'no'} |"
        for name, m in sorted(missions.items())
    )
    tree = "\n".join(
        f"      {name}/ch_<channel_id>.parquet"
        + (f"\n      {name}/telecommands.parquet" if m.get("telecommand_series") else "")
        for name, m in sorted(missions.items())
    )

    text = f"""# Data

How to find and read the data this project runs on. If you have just cloned this
repository, start here.

**The bucket is `{bucket}`. `{MANIFEST_KEY}` is the single source of truth.**
Read it first, resolve every key from it, and never list the bucket.

---

## 1. The datasets and why

Roles are set in `Objective.md` section 9; this table is a pointer, not a
restatement.

| Dataset | Role | Why |
|---|---|---|
| **ESA-ADB** | **Primary** | Real, multivariate, time-synchronised, labelled by the operations engineers who flew the missions. The only public set that can prove a cross-channel claim |
| OPSSAT-AD | Real-data secondary | Univariate per segment, so it tests single-channel behaviour only |
| CATS | Synthetic stress test | Full root-cause metadata. Always mark as synthetic |
| SMAP/MSL | Legacy comparability only | Discredited (Wu & Keogh, TKDE 2023) and its channels are not synchronised with each other, so our core claim could not be demonstrated on it. See `Objective.md` section 9.2 |

Only ESA-ADB and SMAP/MSL are ingested so far. `opssat-ad/v1/` and `cats/v1/` are
later tasks.

## 2. Where it lives

```
r2://{bucket}/
  _manifest/
    manifest.json          <- the index. Read first, always.
    ops_ledger.json        <- running operation counts, month-keyed
  {ARCHIVE_PREFIX}/
{tree}
  {ANNOTATIONS_PREFIX}/
    labels.parquet         <- annotations, pre-joined to anomaly classes
    channels.parquet       <- channel descriptions: subsystem, group, target flag
    telecommands.parquet   <- telecommand descriptions (not the time series)
    events.parquet         <- mission plan (Mission2 only)
  {ARCHIVE_PREFIX.replace('/archive', '')}/working/   <- reserved, deliberately empty
```

| Mission | Channels | Target channels | Telecommand series |
|---|---|---|---|
{mission_rows}

## 3. Reading it

```python
import io, json, boto3, pandas as pd

s3 = boto3.client("s3", endpoint_url=f"https://{{ACCOUNT_ID}}.r2.cloudflarestorage.com",
                  aws_access_key_id=KEY, aws_secret_access_key=SECRET, region_name="auto")

BUCKET = "{bucket}"

# 1. the manifest is always the entry point
manifest = json.loads(s3.get_object(Bucket=BUCKET, Key="{MANIFEST_KEY}")["Body"].read())
esa = manifest["datasets"]["esa-adb"]

# 2. resolve a key from it -- never construct or discover one
channel = esa["missions"]["mission1"]["channels"][0]
keys = [s["key"] for s in channel["shards"]] if channel["shards"] else [channel["key"]]

# 3. read
df = pd.concat(
    pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read()))
    for k in keys
)
print(channel["channel_id"], df.shape, df.dtypes.to_dict())
```

A channel is either a single object (`key`) or several time-ordered shards
(`shards`), because no object may exceed 90 MiB. Always handle both.

## 4. Operating rules

- **Ceiling: {OPS_CEILING['class_a']:,} Class A and {OPS_CEILING['class_b']:,} Class B operations per calendar month.**
  Class A is writes and lists, Class B is reads. Egress is free.
- **Tripwire at {OPS_TRIPWIRE:,}.** Not a limit -- a signal that a script is shaped wrongly.
- **Never LIST, never glob.** `fsspec` and `s3fs` glob patterns trigger LIST silently.
  Resolve every key from the manifest.
- **Record what you spend.** `{LEDGER_KEY}` is month-keyed and rolls over
  automatically; increment it rather than starting your own count.
- Prefer few large objects to many small ones. Reading 800 objects costs 800
  Class B operations every single time.

## 5. Caveats that will silently corrupt your results

**Timestamps are anonymised mission time, not UTC.** Every mission's timeline was
scaled by an undisclosed factor greater than 1 and shifted to start on
2000-01-01. Absolute dates are meaningless, elapsed durations are stretched by an
unknown constant, and real sampling periods cannot be recovered. Do not compare
timestamps across missions, and do not interpret them as wall-clock time.

**Values are min-max normalised to [0,1] within each channel group.** Units are
meaningless and `Physical Unit` is itself an anonymised label. Relative
dependencies within a group are preserved, which is exactly what our
cross-channel method needs, but absolute magnitudes are not physical.

**Sampling is irregular and per-channel.** Each channel has its own timestamp
grid, and channels start and end at different times. Almost every algorithm needs
resampling first; ESA prescribes zero-order hold, never linear or Fourier.
`reference_hints` in the manifest carries their per-mission resampling rules and
the ranges of monotonic channels that must be differenced -- these are magic
numbers from ESA's own scripts, derivable from no CSV.

**Not every channel is numeric.** Some are categorical and stored as strings.
Check `value_dtype` in the manifest before assuming arithmetic works.

**Rare nominal events are not anomalies.** Manoeuvres, resets and calibrations are
labelled in `labels.parquet` with `Category == "Rare Event"`. They belong in
training, and they are how we prove Sentinel does not cry wolf at routine
operations. See `Objective.md` section 6.2.

## 6. Provenance

| | |
|---|---|
| Source | Zenodo record `{ds['source_url'].rsplit('/', 1)[-1]}`, DOI `{ds['source_doi']}` |
| Concept DOI | `{ds['concept_doi']}` |
| Dataset licence | **{ds['license']}** -- attribution to ESA required in anything published from it |
| Reference code licence | MIT (Airbus, KP Labs, Wenig & Schmidl) -- separate from the dataset licence |
| Ingested | {manifest['generated_utc']} |

Source archive checksums are in the manifest under `source_files`. Zenodo
publishes MD5 only; the SHA-256 values in the manifest are ours, computed over the
parquet objects we wrote.

## 7. What is deliberately absent

- **`working/` is reserved and empty.** It will hold pre-joined 8-12 channel
  subsystem bundles for the hot training path, once the target subsystem is
  chosen. The prefix is reserved; nothing is built.
- **No resampled or differenced data.** The archive is faithful to the source.
  Preprocessing belongs to the harness, not to the archive.
- **SMAP/MSL is here as of 2026-09-03** and is a legacy-comparability source, never a
  headline (Objective.md 9.2). See section 1. The line this replaces said it was not
  here and not planned, which section 1 had already contradicted.

---

*Generated from the manifest on every ingest. Do not edit by hand -- edit
`src/sentinel_data/docs_gen.py` instead.*
"""
    path = docs_dir / "DATA.md"

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        recorded = _DIGEST_RE.search(existing)
        if recorded is None:
            raise HandAuthoredContentWouldBeLost(
                f"{path} carries no generated-digest footer, so this generator "
                "cannot tell what of it is generated and what is hand-authored. "
                "Refusing to write. Wrap any hand-authored prose in "
                f"{HAND_OPEN} / {HAND_CLOSE} and add the footer, or delete the "
                "file to ask for a clean generation on purpose.")
        if recorded.group(1) != _digest(existing):
            raise HandAuthoredContentWouldBeLost(
                f"{path} has been edited since it was last generated: its content "
                "no longer matches its own digest. Refusing to write, because the "
                "next line would have discarded those edits in silence. Wrap them "
                f"in {HAND_OPEN} / {HAND_CLOSE} so they are carried across.")
        text = _carry_over(text, existing)

    path.write_text(_stamped(text), encoding="utf-8")
    return path


def regenerate(manifest: dict, bucket: str, docs_dir: Path = DOCS_DIR) -> list[Path]:
    return [write_data_md(manifest, bucket, docs_dir), write_snapshot(manifest, docs_dir)]
