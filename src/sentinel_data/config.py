"""Configuration and immutable facts about the ESA-ADB source data.

Every constant here was verified against the Zenodo REST API or the official
kplabs-pl/ESA-ADB repository. Do not edit from memory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths. Everything transient lives under one scratch root that is deleted
# at the end of a successful ingest.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH = PROJECT_ROOT / "scratch"
RAW_DIR = SCRATCH / "raw"
PARQUET_DIR = SCRATCH / "parquet"
STATE_FILE = SCRATCH / "state.json"
PREFLIGHT_FILE = SCRATCH / "preflight.json"
DOCS_DIR = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------
# Source: Zenodo record 15237121 (ESA Anomaly Dataset), CC BY 3.0 IGO.
# Sizes and MD5s read live from https://zenodo.org/api/records/15237121
# --------------------------------------------------------------------------
ZENODO_RECORD = "15237121"
SOURCE_DOI = "10.5281/zenodo.15237121"
CONCEPT_DOI = "10.5281/zenodo.12528695"
DATASET_LICENSE = "CC BY 3.0 IGO"
ZENODO_FILE_URL = "https://zenodo.org/api/records/{record}/files/{key}/content"


@dataclass(frozen=True)
class SourceFile:
    key: str
    size: int
    md5: str
    mission: str

    @property
    def url(self) -> str:
        return ZENODO_FILE_URL.format(record=ZENODO_RECORD, key=self.key)


SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile("ESA-Mission1.zip", 3_776_246_073, "9770ad12ed730238f37c42d5c27ab436", "mission1"),
    SourceFile("ESA-Mission2.zip", 4_098_539_932, "bfc72012691427d9327eb41f726ce45e", "mission2"),
    SourceFile("ESA-Mission3.zip", 3_734_403_444, "d63943f09c81378acd9fc5e565ecc66e", "mission3"),
)

# Expected channel counts, from the paper's Table 1. Used as a completeness check.
EXPECTED_CHANNELS = {"mission1": 76, "mission2": 100, "mission3": 48}
EXPECTED_TARGETS = {"mission1": 58, "mission2": 47, "mission3": 24}
EXPECTED_TELECOMMANDS = {"mission1": 698, "mission2": 123, "mission3": 0}

# Knowledge that exists only in ESA's reference scripts and no CSV. Recorded in
# the manifest so the harness never has to re-derive it from the paper.
REFERENCE_HINTS = {
    "dominant_sampling_hz": {"mission1": 0.033, "mission2": 0.056, "mission3": 0.065},
    "reference_resampling": {"mission1": "30s", "mission2": "18s"},
    "monotonic_channels_requiring_diff": {"mission1": [4, 11], "mission2": [29, 46]},
    "annotation_label_enum": {"NOMINAL": 0, "ANOMALY": 1, "RARE_EVENT": 2, "GAP": 3, "INVALID": 4},
    "resampling_method": "zero-order hold (forward fill); never linear or Fourier",
}

ANONYMISATION = {
    "timestamps": (
        "Scaled by an undisclosed factor >1 and shifted to start 2000-01-01. "
        "These are anonymised mission time, NOT UTC."
    ),
    "values": "Min-max normalised to [0,1] within each channel group.",
    "names": "Missions, subsystems, channels, telecommands and units are renamed and renumbered.",
}

# --------------------------------------------------------------------------
# R2 target
# --------------------------------------------------------------------------
DATASET_PREFIX = "esa-adb/v1"
ARCHIVE_PREFIX = f"{DATASET_PREFIX}/archive"
ANNOTATIONS_PREFIX = f"{DATASET_PREFIX}/annotations"
MANIFEST_KEY = "_manifest/manifest.json"
LEDGER_KEY = "_manifest/ops_ledger.json"

MANIFEST_SCHEMA_VERSION = "1.1"

# Constraint 2: every object must be a single PutObject, so stay well under the
# point where any client would consider multipart.
MAX_OBJECT_BYTES = 90 * 1024 * 1024
MULTIPART_THRESHOLD = 100 * 1024 * 1024

# Constraint 1: the binding ceiling is per calendar month. 1000 is a tripwire
# that stops the run for a human, not a limit.
OPS_CEILING = {"class_a": 50_000, "class_b": 50_000}
OPS_TRIPWIRE = 1_000

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 9
PARQUET_ROW_GROUP_SIZE = 1_000_000


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


def load_r2_config() -> R2Config:
    """Read R2 credentials from the gitignored .env. Never committed."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
        if not os.environ.get(name)
    ]
    if missing:
        raise SystemExit(
            "Missing R2 credentials: "
            + ", ".join(missing)
            + f"\nCreate {PROJECT_ROOT / '.env'} using .env.example as the template."
        )
    return R2Config(
        account_id=os.environ["R2_ACCOUNT_ID"],
        access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        bucket=os.environ["R2_BUCKET"],
    )
