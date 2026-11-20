"""Synthetic tests for the persistent automatic pipeline cadence gate."""

import hashlib
import io
import json
import sys
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

import handler as admission


class ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """S3 operations modelled separately, including IAM-dependent answers.

    * ``ListObjectsV2`` succeeds only for a prefix in ``allowed_list_prefixes``
      (the ``s3:prefix`` condition of the real policy); otherwise 403.
    * ``GetObject`` of a missing key answers 404 only when the principal could
      list that key; without ListBucket it answers 403, as S3 does.
    """

    def __init__(self, *, allowed_list_prefixes=None, read_barrier=None, existing_barrier=None, enforce_conditions=True):
        self.objects = {}
        self.lock = threading.Lock()
        self.allowed_list_prefixes = (
            {admission.LOCK_KEY} if allowed_list_prefixes is None else set(allowed_list_prefixes)
        )
        self.read_barrier = read_barrier
        self.existing_barrier = existing_barrier
        self.enforce_conditions = enforce_conditions
        self.conditional_errors = []
        self.calls = []

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=1000, **_):
        self.calls.append(("list", Prefix))
        if Prefix not in self.allowed_list_prefixes:
            raise ClientError("403")
        with self.lock:
            keys = sorted(key for (bucket, key) in self.objects if bucket == Bucket and key.startswith(Prefix))
        if not any(key == admission.LOCK_KEY for key in keys) and self.read_barrier is not None:
            self.read_barrier.wait(timeout=5)
        return {"Contents": [{"Key": key} for key in keys[:MaxKeys]], "KeyCount": len(keys[:MaxKeys])}

    def get_object(self, *, Bucket, Key):
        self.calls.append(("get", Key))
        with self.lock:
            found = self.objects.get((Bucket, Key))
        if found is None:
            if Key not in self.allowed_list_prefixes:
                raise ClientError("403")
            raise ClientError("NoSuchKey")
        body, etag = found
        if Key == admission.LOCK_KEY and self.existing_barrier is not None:
            with self.lock:
                waits = self._existing_waits = getattr(self, "_existing_waits", 0) + 1
            if waits <= self.existing_barrier.parties:  # only the first read of each party
                self.existing_barrier.wait(timeout=5)  # both hold the same ETag before either writes
        return {"Body": io.BytesIO(body), "ETag": etag}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **_):
        with self.lock:
            existing = self.objects.get((Bucket, Key))
            if self.enforce_conditions and IfNoneMatch == "*" and existing is not None:
                self.conditional_errors.append("412")
                raise ClientError("412")
            if self.enforce_conditions and IfMatch is not None and (existing is None or existing[1] != IfMatch):
                self.conditional_errors.append("409")
                raise ClientError("409")
            if self.enforce_conditions and IfNoneMatch is None and IfMatch is None:
                raise AssertionError("cadence lock writes must use a conditional request")
            etag = '"' + hashlib.md5(Body).hexdigest() + '"'
            self.objects[(Bucket, Key)] = (Body, etag)
            return {"ETag": etag}


class ConflictOnceS3(FakeS3):
    def __init__(self):
        super().__init__()
        self.conflict_once = True

    def put_object(self, *, IfMatch=None, **kwargs):
        if IfMatch is not None and self.conflict_once:
            self.conflict_once = False
            self.conditional_errors.append("409")
            raise ClientError("409")
        return super().put_object(IfMatch=IfMatch, **kwargs)


class FakeStepFunctions:
    def __init__(self, *, fail_before_accept=False, ambiguous_after_accept=False):
        self.executions = {}
        self.calls = []
        self.fail_before_accept = fail_before_accept
        self.ambiguous_after_accept = ambiguous_after_accept
        self.lock = threading.Lock()

    def start_execution(self, *, stateMachineArn, name, input):
        with self.lock:
            self.calls.append((name, input))
            if name in self.executions:
                raise ClientError("ExecutionAlreadyExists")
            if self.fail_before_accept:
                self.fail_before_accept = False
                raise RuntimeError("temporary start failure")
            arn = f"{stateMachineArn}:execution:{name}"
            self.executions[name] = arn
            if self.ambiguous_after_accept:
                self.ambiguous_after_accept = False
                raise RuntimeError("ambiguous response")
            return {"executionArn": arn}


def event(check_id):
    return {"available": True, "check_id": check_id, "status_code": 200}


def call(s3, sfn, payload, now, interval_hours=24):
    return admission.admit_pipeline(
        payload,
        s3_client=s3,
        stepfunctions_client=sfn,
        bucket="fixture",
        state_machine_arn="arn:aws:states:region:account:stateMachine:pipeline",
        pipeline_enabled=True,
        interval_hours=interval_hours,
        now=lambda: now,
    )


def test_unavailable_or_disabled_checks_do_not_reserve_or_start():
    s3, sfn = FakeS3(), FakeStepFunctions()
    timestamp = datetime(2026, 11, 17, tzinfo=timezone.utc)

    unavailable = admission.admit_pipeline(
        {"available": False, "check_id": "check-0"},
        s3_client=s3,
        stepfunctions_client=sfn,
        bucket="fixture",
        state_machine_arn="pipeline",
        pipeline_enabled=True,
        interval_hours=24,
        now=lambda: timestamp,
    )
    disabled = admission.admit_pipeline(
        event("check-disabled"),
        s3_client=s3,
        stepfunctions_client=sfn,
        bucket="fixture",
        state_machine_arn="pipeline",
        pipeline_enabled=False,
        interval_hours=24,
        now=lambda: timestamp,
    )

    assert unavailable["status"] == "unavailable"
    assert disabled["status"] == "disabled"
    assert not s3.objects
    assert not sfn.executions


def test_availability_checks_continue_but_pipeline_is_limited_across_invocations():
    s3, sfn = FakeS3(), FakeStepFunctions()
    first = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)

    assert call(s3, sfn, event("check-1"), first)["status"] == "started"
    # A new invocation/process sees the same S3 reservation.
    result = call(s3, sfn, event("check-2"), first + timedelta(hours=12))
    assert result["status"] == "interval_not_elapsed"
    assert len(sfn.executions) == 1
    assert admission.LOCK_KEY in [key for _, key in s3.objects]


def test_duplicate_check_id_does_not_start_twice_or_renew_reservation():
    s3, sfn = FakeS3(), FakeStepFunctions()
    first = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    assert call(s3, sfn, event("duplicate"), first)["status"] == "started"
    original = json.loads(s3.objects[("fixture", admission.LOCK_KEY)][0])

    retry = call(s3, sfn, event("duplicate"), first + timedelta(hours=5))

    current = json.loads(s3.objects[("fixture", admission.LOCK_KEY)][0])
    assert retry["status"] == "already_started"
    assert current["reserved_at"] == original["reserved_at"]
    assert len(sfn.executions) == 1
    assert len(sfn.calls) == 1


def test_configured_elapsed_interval_reopens_after_failed_pipeline_window():
    s3, sfn = FakeS3(), FakeStepFunctions()
    first = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    assert call(s3, sfn, event("check-1"), first, interval_hours=48)["status"] == "started"

    assert call(s3, sfn, event("check-2"), first + timedelta(hours=47), interval_hours=48)[
        "status"
    ] == "interval_not_elapsed"
    assert call(s3, sfn, event("check-3"), first + timedelta(hours=48), interval_hours=48)[
        "status"
    ] == "started"
    assert len(sfn.executions) == 2


def test_failed_start_retries_same_reservation_and_does_not_consume_a_second_slot():
    s3, sfn = FakeS3(), FakeStepFunctions(fail_before_accept=True)
    timestamp = datetime(2026, 11, 17, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="temporary start failure"):
        call(s3, sfn, event("same-check"), timestamp)
    first_lock = json.loads(s3.objects[("fixture", admission.LOCK_KEY)][0])
    assert first_lock["status"] == "reserved"

    result = call(s3, sfn, event("same-check"), timestamp + timedelta(hours=5))
    assert result["status"] == "started"
    assert len(sfn.executions) == 1
    assert sfn.calls[0][0] == sfn.calls[1][0]
    second_lock = json.loads(s3.objects[("fixture", admission.LOCK_KEY)][0])
    assert second_lock["reserved_at"] == first_lock["reserved_at"]


def test_ambiguous_start_response_reuses_one_execution_name_on_retry():
    s3, sfn = FakeS3(), FakeStepFunctions(ambiguous_after_accept=True)
    timestamp = datetime(2026, 11, 17, tzinfo=timezone.utc)

    with pytest.raises(RuntimeError, match="ambiguous response"):
        call(s3, sfn, event("same-check"), timestamp)
    result = call(s3, sfn, event("same-check"), timestamp)

    assert result["status"] == "started"
    assert len(sfn.executions) == 1
    assert sfn.calls[0][0] == sfn.calls[1][0]


def test_concurrent_checks_allow_only_one_automatic_execution():
    barrier = threading.Barrier(2)
    s3, sfn = FakeS3(read_barrier=barrier), FakeStepFunctions()
    timestamp = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    results = []
    errors = []

    def worker(check_id):
        try:
            results.append(call(s3, sfn, event(check_id), timestamp))
        except Exception as exc:  # surfaced below with the thread result
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"check-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert sum(result["started"] for result in results) == 1
    assert len(sfn.executions) == 1
    assert len(sfn.calls) == 1
    assert "412" in s3.conditional_errors


def test_concurrent_checks_on_an_expired_reservation_use_if_match_and_start_only_one():
    barrier = threading.Barrier(2)
    s3, sfn = FakeS3(existing_barrier=barrier), FakeStepFunctions()
    now = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    old = {"schema_version": 1, "status": "started", "reserved_at": (now - timedelta(hours=48)).isoformat(),
           "trigger_id": "old", "execution_name": "auto-old", "input_sha256": "old"}
    s3.put_object(Bucket="fixture", Key=admission.LOCK_KEY, Body=json.dumps(old).encode(), IfNoneMatch="*")
    results, errors = [], []

    def worker(check_id):
        try:
            results.append(call(s3, sfn, event(check_id), now))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"renew-{i}",)) for i in range(2)]
    [t.start() for t in threads]
    [t.join(timeout=10) for t in threads]
    assert not any(t.is_alive() for t in threads) and not errors
    assert sum(r["started"] for r in results) == 1
    assert len(sfn.executions) == 1 and len(sfn.calls) == 1
    assert "409" in s3.conditional_errors


def test_get_object_of_a_missing_key_is_403_without_list_bucket_and_404_with_it():
    denied = FakeS3(allowed_list_prefixes=set())
    with pytest.raises(ClientError, match="403"):
        denied.get_object(Bucket="fixture", Key=admission.LOCK_KEY)
    allowed = FakeS3()
    with pytest.raises(ClientError, match="NoSuchKey"):
        allowed.get_object(Bucket="fixture", Key=admission.LOCK_KEY)
    assert allowed.list_objects_v2(Bucket="fixture", Prefix=admission.LOCK_KEY)["Contents"] == []
    with pytest.raises(ClientError, match="403"):
        allowed.list_objects_v2(Bucket="fixture", Prefix="manifests/control/")


def test_first_admission_creates_the_reservation_with_only_the_exact_prefix_authorized():
    s3, sfn = FakeS3(), FakeStepFunctions()
    result = call(s3, sfn, event("first-check"), datetime(2026, 11, 17, 15, tzinfo=timezone.utc))
    assert result["status"] == "started"
    assert s3.calls[0] == ("list", admission.LOCK_KEY)
    assert json.loads(s3.objects[("fixture", admission.LOCK_KEY)][0])["status"] == "started"


def test_listing_403_fails_closed_and_is_never_read_as_a_missing_reservation():
    s3, sfn = FakeS3(allowed_list_prefixes=set()), FakeStepFunctions()
    with pytest.raises(ClientError, match="403"):
        call(s3, sfn, event("first-check"), datetime(2026, 11, 17, 15, tzinfo=timezone.utc))
    assert s3.objects == {}
    assert sfn.executions == {} and sfn.calls == []


def test_a_key_that_only_shares_the_prefix_is_not_the_reservation():
    s3, sfn = FakeS3(), FakeStepFunctions()
    s3.objects[("fixture", admission.LOCK_KEY + ".bak")] = (b"{}", '"x"')
    listing = s3.list_objects_v2(Bucket="fixture", Prefix=admission.LOCK_KEY)
    assert [item["Key"] for item in listing["Contents"]] == [admission.LOCK_KEY + ".bak"]
    result = call(s3, sfn, event("first-check"), datetime(2026, 11, 17, 15, tzinfo=timezone.utc))
    assert result["status"] == "started"
    assert ("fixture", admission.LOCK_KEY) in s3.objects


def test_existing_exact_key_is_read_and_its_interval_respected():
    s3, sfn = FakeS3(), FakeStepFunctions()
    now = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    assert call(s3, sfn, event("a"), now)["status"] == "started"
    assert call(s3, sfn, event("b"), now + timedelta(hours=1))["status"] == "interval_not_elapsed"
    assert len(sfn.executions) == 1


def test_stale_if_match_receives_s3_409_and_if_none_match_receives_412():
    s3 = FakeS3()
    s3.put_object(Bucket="fixture", Key=admission.LOCK_KEY, Body=b"{}", IfNoneMatch="*")
    with pytest.raises(ClientError, match="412"):
        s3.put_object(Bucket="fixture", Key=admission.LOCK_KEY, Body=b"{}", IfNoneMatch="*")
    with pytest.raises(ClientError, match="409"):
        s3.put_object(Bucket="fixture", Key=admission.LOCK_KEY, Body=b"{}", IfMatch='"stale"')


def test_conditional_409_retries_reservation_using_latest_etag():
    s3, sfn = ConflictOnceS3(), FakeStepFunctions()
    first = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    old_lock = {
        "schema_version": 1,
        "status": "started",
        "reserved_at": (first - timedelta(hours=48)).isoformat(),
        "trigger_id": "old-check",
        "execution_name": "old-execution",
        "input_sha256": "old",
    }
    s3.put_object(
        Bucket="fixture",
        Key=admission.LOCK_KEY,
        Body=json.dumps(old_lock, sort_keys=True, separators=(",", ":")).encode(),
        IfNoneMatch="*",
    )
    result = call(s3, sfn, event("new-check"), first)
    assert result["status"] == "started"
    assert s3.conditional_errors == ["409"]
    assert len(sfn.executions) == 1


def test_mutant_without_conditional_headers_violates_concurrent_admission_invariant(monkeypatch):
    barrier = threading.Barrier(2)
    s3, sfn = FakeS3(read_barrier=barrier, enforce_conditions=False), FakeStepFunctions()
    timestamp = datetime(2026, 11, 17, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(
        admission,
        "_write_lock",
        lambda client, bucket, payload, etag: client.put_object(
            Bucket=bucket,
            Key=admission.LOCK_KEY,
            Body=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        ),
    )
    results = []
    errors = []

    def worker(check_id):
        try:
            results.append(call(s3, sfn, event(check_id), timestamp))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"mutant-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors
    assert sum(result["started"] for result in results) == 2
    assert len(sfn.executions) == 2


def test_admission_logs_structured_correlation_without_event_payload(monkeypatch, capsys):
    s3, sfn = FakeS3(), FakeStepFunctions()
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda service: s3 if service == "s3" else sfn),
    )
    monkeypatch.setenv("DATA_BUCKET", "fixture")
    monkeypatch.setenv("PIPELINE_STATE_MACHINE_ARN", "pipeline")
    monkeypatch.setenv("PIPELINE_ENABLED", "true")
    monkeypatch.setenv("PIPELINE_INTERVAL_HOURS", "24")

    admission.handler({**event("safe-check-id"), "sensitive": "must-not-log"}, SimpleNamespace(aws_request_id="request"))

    output = capsys.readouterr().out
    records = [json.loads(line) for line in output.splitlines()]
    assert [record["status"] for record in records] == ["started", "started"]
    assert all(record["component"] == "pipeline-admission" for record in records)
    assert all(record["check_id"] == "safe-check-id" for record in records)
    assert all(record["correlation_id"] for record in records)
    assert all("execution_name" in record for record in records[-1:])
    assert "sensitive" not in output


def test_admission_logs_and_reraises_failures(monkeypatch, capsys):
    s3 = FakeS3(allowed_list_prefixes=set())
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda service: s3 if service == "s3" else FakeStepFunctions()),
    )
    monkeypatch.setenv("DATA_BUCKET", "fixture")
    monkeypatch.setenv("PIPELINE_STATE_MACHINE_ARN", "pipeline")
    monkeypatch.setenv("PIPELINE_ENABLED", "true")
    monkeypatch.setenv("PIPELINE_INTERVAL_HOURS", "24")

    with pytest.raises(ClientError, match="403"):
        admission.handler(event("failed-check"), SimpleNamespace(aws_request_id="request"))

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["status"] for record in records] == ["started", "failed"]
    assert records[-1]["error_type"] == "ClientError"
    assert records[-1]["component"] == "pipeline-admission"


def test_conditional_write_sdk_is_pinned_for_the_admission_lambda():
    from pathlib import Path

    requirements = Path(__file__).parents[2] / "infra" / "functions" / "pipeline_admission" / "requirements.txt"
    assert requirements.read_text(encoding="utf-8").splitlines() == [
        "boto3==1.43.95",
        "botocore==1.43.95",
        "jmespath==1.1.0",
        "python-dateutil==2.9.0.post0",
        "s3transfer==0.19.2",
        "six==1.17.0",
        "urllib3==2.8.0",
    ]
