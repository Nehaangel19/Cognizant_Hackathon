"""
Where the trained model artefacts come from at runtime.

WHY THIS EXISTS
---------------
Locally, `models/*.joblib` sit in the repo and `Engine()` finds them by itself.
On AWS Lambda there are two workable options and this module supports both:

  1. BUNDLED (default) — the .joblib files are baked into the container image.
     Simplest, no IAM, no network call on cold start. The five artefacts total
     ~7.5 MB, which is nothing against Lambda's 10 GB image limit.

  2. S3 — set MODEL_S3_BUCKET (and optionally MODEL_S3_PREFIX) and the artefacts
     are pulled into /tmp on cold start instead. Decouples the model from the
     compute: retraining means re-uploading five files, not rebuilding and
     redeploying the whole image.

Option 2 is NOT needed to make this project fit in Lambda — it is a deliberate
architecture choice, and worth saying plainly rather than implying the image
would otherwise be too big. Default stays on option 1 because it has fewer
moving parts to fail on demo day.

/tmp is the only writable path on Lambda and gives 512 MB by default, so the
download target is fixed there. It persists for the life of the warm container,
so the download happens once per cold start, not once per request.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ARTEFACTS = (
    "failure_xgb.joblib",
    "anomaly_isolation_forest.joblib",
    "model_meta.joblib",
    "shap_explainer.joblib",
    "rootcause_ovr.joblib",
)

_TMP_MODEL_DIR = Path("/tmp/models")


def ensure_models() -> Path | None:
    """Return the directory Engine should load artefacts from.

    Returns None when no S3 bucket is configured, which tells the caller to let
    `Engine()` use its own default (the repo's models/ dir). Returns a concrete
    path when artefacts were fetched from S3.

    Never raises on a partial S3 failure: it falls back to the bundled copy and
    logs why, because a demo that runs on slightly-stale bundled models beats a
    demo that 500s because a bucket policy was wrong.
    """
    bucket = os.environ.get("MODEL_S3_BUCKET", "").strip()
    if not bucket:
        return None

    prefix = os.environ.get("MODEL_S3_PREFIX", "models").strip().strip("/")

    # Already downloaded by an earlier invocation on this warm container.
    if _TMP_MODEL_DIR.exists() and all((_TMP_MODEL_DIR / n).exists() for n in ARTEFACTS):
        print(f"[model_store] reusing warm /tmp copy of s3://{bucket}/{prefix}")
        return _TMP_MODEL_DIR

    try:
        import boto3

        s3 = boto3.client("s3")
        _TMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        total = 0
        for name in ARTEFACTS:
            key = f"{prefix}/{name}" if prefix else name
            dest = _TMP_MODEL_DIR / name
            s3.download_file(bucket, key, str(dest))
            total += dest.stat().st_size
        print(f"[model_store] pulled {len(ARTEFACTS)} artefacts "
              f"({total / 1e6:.1f} MB) from s3://{bucket}/{prefix} -> {_TMP_MODEL_DIR}")
        return _TMP_MODEL_DIR
    except Exception as exc:
        print(f"[model_store] S3 fetch failed ({type(exc).__name__}: {exc}); "
              f"falling back to bundled models/", file=sys.stderr)
        return None
