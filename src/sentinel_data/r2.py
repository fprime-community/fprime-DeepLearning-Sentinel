"""Cloudflare R2 client, with every billable operation counted.

Operation classes are taken from https://developers.cloudflare.com/r2/pricing/.
Counting hooks into botocore's `before-send`, which fires once per HTTP attempt,
so automatic retries are counted the way they are billed. A wrapper that counted
logical calls would quietly under-report them.

Uploads use put_object, never upload_file: put_object has no multipart code path
at all, so an upload is unconditionally exactly one Class A operation. The
transfer config is set as well, but nothing depends on it.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import (
    LEDGER_KEY,
    MULTIPART_THRESHOLD,
    OPS_CEILING,
    OPS_TRIPWIRE,
    R2Config,
)

CLASS_A = {
    "ListBuckets", "PutBucket", "CreateBucket", "ListObjects", "ListObjectsV2",
    "PutObject", "CopyObject", "CompleteMultipartUpload", "CreateMultipartUpload",
    "ListMultipartUploads", "UploadPart", "UploadPartCopy", "ListParts",
    "PutBucketEncryption", "PutBucketCors", "PutBucketLifecycleConfiguration",
}
CLASS_B = {
    "HeadBucket", "HeadObject", "GetObject", "UsageSummary", "GetBucketEncryption",
    "GetBucketLocation", "GetBucketCors", "GetBucketLifecycleConfiguration",
}
FREE = {"DeleteObject", "DeleteObjects", "DeleteBucket", "AbortMultipartUpload"}

FORBIDDEN = {"ListObjects", "ListObjectsV2", "ListBuckets", "ListMultipartUploads", "ListParts"}

TRANSFER_CONFIG = TransferConfig(multipart_threshold=MULTIPART_THRESHOLD)


@dataclass
class OpCounter:
    """Counts billable operations, and refuses the ones we have banned."""

    by_operation: Counter = field(default_factory=Counter)
    attempts: int = 0

    def record(self, operation: str) -> None:
        self.attempts += 1
        self.by_operation[operation] += 1
        if operation in FORBIDDEN:
            raise RuntimeError(
                f"{operation} is a LIST-class operation and is banned by constraint 4. "
                "Every key must be resolved from the manifest."
            )

    @property
    def class_a(self) -> int:
        return sum(n for op, n in self.by_operation.items() if op in CLASS_A)

    @property
    def class_b(self) -> int:
        return sum(n for op, n in self.by_operation.items() if op in CLASS_B)

    @property
    def unclassified(self) -> dict[str, int]:
        return {
            op: n
            for op, n in self.by_operation.items()
            if op not in CLASS_A and op not in CLASS_B and op not in FREE
        }

    def summary(self) -> str:
        a, b = self.class_a, self.class_b
        lines = [
            f"    Class A used:  {a:,}  /  {OPS_CEILING['class_a']:,}   "
            f"({100*a/OPS_CEILING['class_a']:.2f}%)",
            f"    Class B used:  {b:,}  /  {OPS_CEILING['class_b']:,}   "
            f"({100*b/OPS_CEILING['class_b']:.2f}%)",
        ]
        if self.unclassified:
            lines.append(f"    UNCLASSIFIED:  {self.unclassified}")
        return "\n".join(lines)


def _method_fallback(request) -> str:
    """Classify from the HTTP request when botocore does not name the event."""
    method = getattr(request, "method", "").upper()
    url = getattr(request, "url", "")
    if "list-type=" in url or url.rstrip("/").endswith("?"):
        return "ListObjectsV2"
    return {"PUT": "PutObject", "HEAD": "HeadObject", "GET": "GetObject",
            "DELETE": "DeleteObject", "POST": "CompleteMultipartUpload"}.get(method, method or "Unknown")


def make_client(cfg: R2Config, counter: OpCounter | None = None) -> tuple[object, OpCounter]:
    """Build an R2 client whose every HTTP attempt is counted.

    ``counter`` lets a caller supply a subclass -- sentinel_eval.ops.Budget adds
    the tripwire and ceiling enforcement -- without duplicating the operation
    classification tables above. Omitted, the behaviour is unchanged.
    """
    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        region_name="auto",
        config=Config(retries={"max_attempts": 3, "mode": "standard"},
                      s3={"addressing_style": "path"}),
    )
    counter = OpCounter() if counter is None else counter

    def _on_send(event_name=None, request=None, **_kw):
        if event_name:
            operation = event_name.rsplit(".", 1)[-1]
            if operation in ("s3", "*"):
                operation = _method_fallback(request)
        else:
            operation = _method_fallback(request)
        counter.record(operation)

    client.meta.events.register("before-send.s3", _on_send)
    return client, counter


# --------------------------------------------------------------------------
# Object helpers
# --------------------------------------------------------------------------
def put(client, bucket: str, key: str, path, md5_b64: str, metadata: dict[str, str]) -> None:
    """Exactly one PutObject. Content-MD5 makes R2 reject a corrupted write."""
    with open(path, "rb") as fh:
        client.put_object(
            Bucket=bucket, Key=key, Body=fh, ContentMD5=md5_b64,
            Metadata={k: str(v) for k, v in metadata.items()},
        )


def put_bytes(client, bucket: str, key: str, blob: bytes, md5_b64: str,
              metadata: dict[str, str] | None = None) -> None:
    client.put_object(
        Bucket=bucket, Key=key, Body=blob, ContentMD5=md5_b64,
        ContentType="application/json",
        Metadata={k: str(v) for k, v in (metadata or {}).items()},
    )


def head(client, bucket: str, key: str) -> dict:
    return client.head_object(Bucket=bucket, Key=key)


def get_bytes(client, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


# --------------------------------------------------------------------------
# Ops ledger -- month-keyed, because the ceiling resets per calendar month
# --------------------------------------------------------------------------
def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def new_ledger() -> dict:
    return {
        "current_month": current_month(),
        "months": {},
        "ceiling": dict(OPS_CEILING),
        "tripwire": OPS_TRIPWIRE,
    }


def roll_and_add(ledger: dict, class_a: int, class_b: int) -> dict:
    """Add this run's counts to the current month, rolling over if needed.

    Past months are kept as history. Budget checks only ever read the current
    month, so a ledger that has been running for a year cannot false-abort.
    """
    month = current_month()
    ledger.setdefault("months", {})
    ledger["current_month"] = month
    bucket = ledger["months"].setdefault(month, {"class_a": 0, "class_b": 0})
    bucket["class_a"] += class_a
    bucket["class_b"] += class_b
    ledger["ceiling"] = dict(OPS_CEILING)
    ledger["tripwire"] = OPS_TRIPWIRE
    return ledger


def fetch_ledger(client, bucket: str) -> dict:
    """Read the ledger, or start a fresh one. Costs one Class B op if present.

    Only a genuinely absent ledger starts a fresh one. Every other failure --
    a network blip, a permissions change, a truncated write -- is raised.
    Swallowing those would report a month's spend as zero, which is the one
    error a hard ceiling cannot survive.
    """
    try:
        return json.loads(get_bytes(client, bucket, LEDGER_KEY))
    except client.exceptions.NoSuchKey:
        return new_ledger()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NoSuchBucket"):
            return new_ledger()
        raise
