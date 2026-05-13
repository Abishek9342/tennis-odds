"""AWS Lambda entry points for all scheduled + API functions.

Each Lambda function points to a different handler here. The LAMBDA_FUNCTION
env var is also used by dispatch() for the shared-image pattern.

Functions deployed:
    tennis-predict-daily    → handler_predict
    tennis-clv-close        → handler_clv_close
    tennis-check-results    → handler_check_results
    tennis-injury-refresh   → handler_injury_refresh
    tennis-api              → handler_api  (API Gateway)
    tennis-retrain          → handler_retrain
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Lambda task root is /var/task — project files are there
TASK_ROOT = Path(os.environ.get("LAMBDA_TASK_ROOT", "/var/task"))
sys.path.insert(0, str(TASK_ROOT / "src"))

# All model artifacts live in S3; we cache them in /tmp (up to 10GB on Lambda)
S3_BUCKET = os.environ.get("S3_BUCKET", "tennis-odds-artifacts")
TMP = Path("/tmp/tennis")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── S3 artifact sync ──────────────────────────────────────────────────────────

def _sync_from_s3(force: bool = False) -> None:
    """Download model artifacts from S3 to /tmp if not already cached."""
    import boto3

    marker = TMP / ".synced"
    if marker.exists() and not force:
        log.info("S3 artifacts already cached in /tmp")
        return

    TMP.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    prefix = "data/processed/"

    log.info(f"Syncing artifacts from s3://{S3_BUCKET}/{prefix} ...")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local = TMP / key
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or local.stat().st_size != obj["Size"]:
                log.info(f"  Downloading {key}")
                s3.download_file(S3_BUCKET, key, str(local))

    marker.write_text("ok")
    log.info("S3 sync complete.")


def _push_to_s3(local_dir: Path, s3_prefix: str) -> None:
    """Upload updated artifacts back to S3 (used after retraining)."""
    import boto3
    s3 = boto3.client("s3")
    for f in local_dir.rglob("*"):
        if f.is_file():
            key = s3_prefix + str(f.relative_to(local_dir))
            log.info(f"  Uploading {key}")
            s3.upload_file(str(f), S3_BUCKET, key)


def _patch_paths() -> None:
    """Point DATA_DIR and MODEL_DIR at /tmp where artifacts live."""
    os.environ.setdefault("TENNIS_DATA_DIR", str(TMP / "data/processed"))
    os.environ.setdefault("TENNIS_MODEL_DIR", str(TMP / "data/processed/models"))


# ── Handler: daily predictions ────────────────────────────────────────────────

def handler_predict(event, context):
    """EventBridge trigger — run daily predictions and send alerts."""
    _sync_from_s3()
    _patch_paths()

    # Ensure writable output directories exist under /tmp
    (TMP / "out").mkdir(parents=True, exist_ok=True)
    (TMP / "logs").mkdir(parents=True, exist_ok=True)

    # Pull existing CSVs from S3 so history accumulates across container restarts
    import boto3
    s3 = boto3.client("s3")
    for s3_key, local_path in [
        ("out/predictions_log.csv",  TMP / "out/predictions_log.csv"),
        ("logs/paper_trades.csv",    TMP / "logs/paper_trades.csv"),
    ]:
        if not local_path.exists():
            try:
                s3.download_file(S3_BUCKET, s3_key, str(local_path))
                log.info(f"Pulled {s3_key} from S3")
            except s3.exceptions.ClientError:
                pass  # doesn't exist yet — first run

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "predict_upcoming", str(TASK_ROOT / "tools/predict_upcoming.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import sys
        sys.argv = ["predict_upcoming.py", "--days", "2", "--allow-stale-ranks"]
        mod.main()

        # Push outputs back to S3 so they survive container recycling
        for s3_key, local_path in [
            ("out/predictions_log.csv",  TMP / "out/predictions_log.csv"),
            ("logs/paper_trades.csv",    TMP / "logs/paper_trades.csv"),
        ]:
            if local_path.exists():
                s3.upload_file(str(local_path), S3_BUCKET, s3_key)
                log.info(f"Pushed {s3_key} to S3")

        return {"statusCode": 200, "body": "Predictions complete"}
    except Exception as e:
        log.exception("Prediction run failed")
        return {"statusCode": 500, "body": str(e)}


# ── Handler: CLV close ────────────────────────────────────────────────────────

def handler_clv_close(event, context):
    """EventBridge trigger — record closing Pinnacle prices."""
    _sync_from_s3()
    _patch_paths()

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "clv_close", str(TASK_ROOT / "tools/clv_close.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import sys
        sys.argv = ["clv_close.py"]
        mod.main()

        # Push updated CLV log back to S3
        log_path = TMP / "logs/clv_log.csv"
        if log_path.exists():
            import boto3
            boto3.client("s3").upload_file(str(log_path), S3_BUCKET, "logs/clv_log.csv")

        return {"statusCode": 200, "body": "CLV close complete"}
    except Exception as e:
        log.exception("CLV close failed")
        return {"statusCode": 500, "body": str(e)}


# ── Handler: check results ────────────────────────────────────────────────────

def handler_check_results(event, context):
    """EventBridge trigger — auto-fill actual match results."""
    _sync_from_s3()
    _patch_paths()

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_results", str(TASK_ROOT / "tools/check_results.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        import sys
        sys.argv = ["check_results.py", "--log", str(TMP / "out/predictions_log.csv")]
        mod.main()

        # Push updated log back to S3
        import boto3
        s3 = boto3.client("s3")
        for f in ["out/predictions_log.csv", "logs/paper_trades.csv"]:
            p = TMP / f
            if p.exists():
                s3.upload_file(str(p), S3_BUCKET, f)

        return {"statusCode": 200, "body": "Results check complete"}
    except Exception as e:
        log.exception("Check results failed")
        return {"statusCode": 500, "body": str(e)}


# ── Handler: injury refresh ───────────────────────────────────────────────────

def handler_injury_refresh(event, context):
    """EventBridge trigger — refresh injury/withdrawal data from Sofascore."""
    _sync_from_s3()
    _patch_paths()

    try:
        import injury_risk
        # Override cache path to /tmp
        injury_risk.CACHE = TMP / "data/processed/injury_log.parquet"
        n = injury_risk.refresh_atp_withdrawals(days_back=3)
        log.info(f"Added {n} injury events")

        # Push back to S3
        import boto3
        boto3.client("s3").upload_file(
            str(injury_risk.CACHE), S3_BUCKET, "data/processed/injury_log.parquet"
        )
        return {"statusCode": 200, "body": f"Injury refresh: {n} new events"}
    except Exception as e:
        log.exception("Injury refresh failed")
        return {"statusCode": 500, "body": str(e)}


# ── Handler: weekly retrain ───────────────────────────────────────────────────

def handler_retrain(event, context):
    """EventBridge trigger — full retraining pipeline.

    WARNING: This Lambda needs 15-min timeout + 3GB+ memory.
    Set in template.yaml: Timeout: 900, MemorySize: 3008
    """
    _sync_from_s3()
    _patch_paths()

    steps = [
        ["python", "main.py", "scrape"],
        ["python", "main.py", "features"],
        ["python", "main.py", "train"],
        ["python", "main.py", "train-surface"],
        ["python", "main.py", "tune-ensemble"],
        ["python", "main.py", "tune-thresholds"],
        ["python", "main.py", "calibrate", "--method", "isotonic"],
        ["python", "main.py", "sanity"],
    ]

    results = []
    for step in steps:
        log.info(f"Running: {' '.join(step)}")
        r = subprocess.run(
            step, capture_output=True, text=True, cwd=str(TASK_ROOT)
        )
        results.append({"cmd": step[2], "rc": r.returncode, "stdout": r.stdout[-2000:]})
        if r.returncode != 0:
            log.error(f"Step {step[2]} failed:\n{r.stderr}")
            return {"statusCode": 500, "body": json.dumps(results)}

    # Push updated model artifacts to S3
    log.info("Pushing updated artifacts to S3...")
    _push_to_s3(TMP / "data/processed", "data/processed/")

    return {"statusCode": 200, "body": json.dumps(results)}


# ── Handler: FastAPI (API Gateway) ────────────────────────────────────────────

def handler_api(event, context):
    """API Gateway HTTP trigger — serves the FastAPI app via Mangum."""
    # Short-circuit health checks before the expensive S3 sync
    path = (event.get("rawPath") or event.get("path") or "")
    if path == "/prod/health" or path == "/health":
        return {"statusCode": 200, "body": '{"status":"ok"}',
                "headers": {"Content-Type": "application/json"}}

    _sync_from_s3()
    _patch_paths()

    try:
        from mangum import Mangum
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "api", str(TASK_ROOT / "tools/api.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        handler = Mangum(mod.app, lifespan="off")
        return handler(event, context)
    except Exception as e:
        log.exception("API handler failed")
        return {"statusCode": 500, "body": str(e)}


# ── Dispatch (shared-image pattern) ──────────────────────────────────────────

_HANDLERS = {
    "predict":         handler_predict,
    "clv-close":       handler_clv_close,
    "check-results":   handler_check_results,
    "injury-refresh":  handler_injury_refresh,
    "retrain":         handler_retrain,
    "api":             handler_api,
}

def dispatch(event, context):
    """Single entry point — routes to the right handler via LAMBDA_FUNCTION env var."""
    fn = os.environ.get("LAMBDA_FUNCTION", "api")
    handler = _HANDLERS.get(fn)
    if not handler:
        return {"statusCode": 400, "body": f"Unknown LAMBDA_FUNCTION: {fn}"}
    return handler(event, context)
