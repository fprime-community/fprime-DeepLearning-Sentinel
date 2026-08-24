# Data

How to find and read the data this project runs on. If you have just cloned this
repository, start here.

**The bucket is `fprime-sentinel-data`. `_manifest/manifest.json` is the single source of truth.**
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

Only ESA-ADB is ingested so far. `opssat-ad/v1/` and `cats/v1/` are later tasks.

## 2. Where it lives

```
r2://fprime-sentinel-data/
  _manifest/
    manifest.json          <- the index. Read first, always.
    ops_ledger.json        <- running operation counts, month-keyed
  esa-adb/v1/archive/
      mission1/ch_<channel_id>.parquet
      mission1/telecommands.parquet
      mission2/ch_<channel_id>.parquet
      mission2/telecommands.parquet
      mission3/ch_<channel_id>.parquet
  esa-adb/v1/annotations/
    labels.parquet         <- annotations, pre-joined to anomaly classes
    channels.parquet       <- channel descriptions: subsystem, group, target flag
    telecommands.parquet   <- telecommand descriptions (not the time series)
    events.parquet         <- mission plan (Mission2 only)
  esa-adb/v1/working/   <- reserved, deliberately empty
```

| Mission | Channels | Target channels | Telecommand series |
|---|---|---|---|
| `mission1` | 76 | 58 | yes |
| `mission2` | 100 | 47 | yes |
| `mission3` | 48 | 24 | no |

## 3. Reading it

```python
import io, json, boto3, pandas as pd

s3 = boto3.client("s3", endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
                  aws_access_key_id=KEY, aws_secret_access_key=SECRET, region_name="auto")

BUCKET = "fprime-sentinel-data"

# 1. the manifest is always the entry point
manifest = json.loads(s3.get_object(Bucket=BUCKET, Key="_manifest/manifest.json")["Body"].read())
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

- **Ceiling: 50,000 Class A and 50,000 Class B operations per calendar month.**
  Class A is writes and lists, Class B is reads. Egress is free.
- **Tripwire at 1,000.** Not a limit -- a signal that a script is shaped wrongly.
- **Never LIST, never glob.** `fsspec` and `s3fs` glob patterns trigger LIST silently.
  Resolve every key from the manifest.
- **Record what you spend.** `_manifest/ops_ledger.json` is month-keyed and rolls over
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
| Source | Zenodo record `15237121`, DOI `10.5281/zenodo.15237121` |
| Concept DOI | `10.5281/zenodo.12528695` |
| Dataset licence | **CC BY 3.0 IGO** -- attribution to ESA required in anything published from it |
| Reference code licence | MIT (Airbus, KP Labs, Wenig & Schmidl) -- separate from the dataset licence |
| Ingested | 2026-08-24T21:31:02Z |

Source archive checksums are in the manifest under `source_files`. Zenodo
publishes MD5 only; the SHA-256 values in the manifest are ours, computed over the
parquet objects we wrote.

## 7. What is deliberately absent

- **`working/` is reserved and empty.** It will hold pre-joined 8-12 channel
  subsystem bundles for the hot training path, once the target subsystem is
  chosen. The prefix is reserved; nothing is built.
- **No resampled or differenced data.** The archive is faithful to the source.
  Preprocessing belongs to the harness, not to the archive.
- **SMAP/MSL is not here** and is not planned as a primary source. See section 1.

---

*Generated from the manifest on every ingest. Do not edit by hand -- edit
`src/sentinel_data/docs_gen.py` instead.*
