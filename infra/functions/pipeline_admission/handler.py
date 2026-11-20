"""Persistently limits automatic pipeline starts without changing check cadence."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Callable, Dict, Optional, Tuple

LOCK_KEY = "manifests/control/pipeline-cadence.json"
MAX_CAS_ATTEMPTS = 5


class CadenceReservationConflict(RuntimeError):
    """The S3 compare-and-set reservation did not converge."""


def handler(event: Any, context: Any) -> Dict[str, Any]:
    fields = _correlation_fields(event, context)
    _emit_log("INFO", "pipeline_admission_started", "started", fields)
    try:
        import boto3

        result = admit_pipeline(
            event,
            s3_client=boto3.client("s3"),
            stepfunctions_client=boto3.client("stepfunctions"),
            bucket=os.environ["DATA_BUCKET"],
            state_machine_arn=os.environ["PIPELINE_STATE_MACHINE_ARN"],
            pipeline_enabled=os.environ.get("PIPELINE_ENABLED", "false").lower() == "true",
            interval_hours=int(os.environ.get("PIPELINE_INTERVAL_HOURS", "24")),
        )
    except Exception as exc:
        _emit_log(
            "ERROR",
            "pipeline_admission_failed",
            "failed",
            {**fields, "error_type": type(exc).__name__},
        )
        raise

    finished_fields = dict(fields)
    if isinstance(result.get("execution_name"), str):
        finished_fields["execution_name"] = result["execution_name"]
    _emit_log("INFO", "pipeline_admission_finished", str(result.get("status", "unknown")), finished_fields)
    return result


def admit_pipeline(
    event: Any,
    *,
    s3_client: Any,
    stepfunctions_client: Any,
    bucket: str,
    state_machine_arn: str,
    pipeline_enabled: bool,
    interval_hours: int,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Dict[str, Any]:
    """Reserve and start at most one automatic execution per minimum interval."""
    if not isinstance(event, dict) or not isinstance(event.get("available"), bool):
        raise ValueError("availability event must contain a boolean 'available' field")
    if interval_hours < 24:
        raise ValueError("pipeline interval must be at least 24 hours")
    if not event["available"]:
        return {"status": "unavailable", "started": False}
    if not pipeline_enabled:
        return {"status": "disabled", "started": False}

    trigger_id = event.get("check_id") or event.get("id")
    if not isinstance(trigger_id, str) or not trigger_id:
        raise ValueError("availability event has no stable check_id")
    request_bytes = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    request_hash = hashlib.sha256(request_bytes).hexdigest()
    execution_name = f"auto-{hashlib.sha256(trigger_id.encode('utf-8')).hexdigest()[:32]}"
    current_time = now()
    if current_time.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    current_time = current_time.astimezone(timezone.utc)
    interval_seconds = interval_hours * 60 * 60

    for _ in range(MAX_CAS_ATTEMPTS):
        existing, etag = _read_lock(s3_client, bucket)
        if existing is not None:
            if existing.get("trigger_id") == trigger_id:
                if existing.get("input_sha256") != request_hash:
                    raise ValueError("check_id was reused with a different availability payload")
                if existing.get("status") == "started":
                    return {"status": "already_started", "started": True, "execution_name": execution_name}
                if existing.get("status") != "reserved":
                    raise ValueError("cadence lock has an unknown state")
                reservation = existing
                response = {"ETag": etag}
            else:
                last_started = _parse_timestamp(existing.get("reserved_at"))
                if (current_time - last_started).total_seconds() < interval_seconds:
                    return {
                        "status": "interval_not_elapsed",
                        "started": False,
                        "next_eligible_at": (last_started.timestamp() + interval_seconds),
                    }
                reservation = {
                    "schema_version": 1,
                    "status": "reserved",
                    "reserved_at": current_time.isoformat(),
                    "trigger_id": trigger_id,
                    "execution_name": execution_name,
                    "input_sha256": request_hash,
                }
                try:
                    response = _write_lock(s3_client, bucket, reservation, etag)
                except Exception as exc:
                    if _error_code(exc) in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}:
                        continue
                    raise
        else:
            reservation = {
                "schema_version": 1,
                "status": "reserved",
                "reserved_at": current_time.isoformat(),
                "trigger_id": trigger_id,
                "execution_name": execution_name,
                "input_sha256": request_hash,
            }
            try:
                response = _write_lock(s3_client, bucket, reservation, etag)
            except Exception as exc:
                if _error_code(exc) in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}:
                    continue
                raise

        try:
            started = stepfunctions_client.start_execution(
                stateMachineArn=state_machine_arn,
                name=execution_name,
                input=request_bytes.decode("utf-8"),
            )
        except Exception as exc:
            if _error_code(exc) not in {"ExecutionAlreadyExists", "ExecutionAlreadyExistsException"}:
                # The durable reservation remains. EventBridge may retry this
                # same check_id, which safely reuses the same execution name.
                raise
            started = {"executionArn": None}

        reservation["status"] = "started"
        reservation["execution_arn"] = started.get("executionArn")
        try:
            _write_lock(s3_client, bucket, reservation, response.get("ETag"))
        except Exception as exc:
            if _error_code(exc) not in {"409", "412", "ConditionalRequestConflict", "PreconditionFailed"}:
                raise
        return {
            "status": "started",
            "started": True,
            "execution_name": execution_name,
            "execution_arn": started.get("executionArn"),
        }

    raise CadenceReservationConflict("could not reserve the pipeline interval after concurrent updates")


def _read_lock(client: Any, bucket: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # Existence is decided by ListObjectsV2 on the exact key, never by the error
    # of GetObject: without s3:ListBucket, S3 answers 403 (not 404) for a missing
    # key, so a GetObject error cannot tell "absent" from "forbidden". Any error of
    # the listing (including 403) propagates: permissions problems fail closed and
    # are never read as "no reservation yet".
    listing = client.list_objects_v2(Bucket=bucket, Prefix=LOCK_KEY, MaxKeys=5)
    # A key that merely shares the prefix (e.g. "<key>.bak") is not the lock.
    if not any(isinstance(item, dict) and item.get("Key") == LOCK_KEY for item in listing.get("Contents") or []):
        return None, None
    try:
        response = client.get_object(Bucket=bucket, Key=LOCK_KEY)
    except Exception as exc:
        if _error_code(exc) in {"NoSuchKey", "NotFound", "404"}:
            return None, None  # deleted between listing and read; creation stays conditional
        raise
    try:
        payload = json.loads(response["Body"].read().decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pipeline cadence lock is unreadable") from exc
    if not isinstance(payload, dict) or not response.get("ETag"):
        raise ValueError("pipeline cadence lock is malformed")
    return payload, response["ETag"]


def _write_lock(client: Any, bucket: str, payload: Dict[str, Any], etag: Optional[str]) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "Bucket": bucket,
        "Key": LOCK_KEY,
        "Body": json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        "ContentType": "application/json",
    }
    request["IfMatch" if etag else "IfNoneMatch"] = etag if etag else "*"
    return client.put_object(**request)


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("pipeline cadence lock has no reservation timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("pipeline cadence lock timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _error_code(exc: Exception) -> Optional[str]:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            return str(error.get("Code"))
    return None


def _correlation_fields(event: Any, context: Any) -> Dict[str, Any]:
    check_id = event.get("check_id") if isinstance(event, dict) else None
    if not isinstance(check_id, str) or not check_id or len(check_id) > 128:
        check_id = None
    request_id = getattr(context, "aws_request_id", None)
    seed = check_id or (request_id if isinstance(request_id, str) and request_id else None)
    fields: Dict[str, Any] = {"check_id": check_id}
    if seed is not None:
        fields["correlation_id"] = sha256(seed.encode("utf-8")).hexdigest()[:24]
    return fields


def _emit_log(level: str, message: str, status: str, fields: Dict[str, Any]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        "component": "pipeline-admission",
        "logger": __name__,
        "status": status,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))
