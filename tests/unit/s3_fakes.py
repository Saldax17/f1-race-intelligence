"""An in-memory stand-in for a boto3 S3 client.

It implements only the calls :class:`S3StorageBackend` makes, and fails the
way S3 does — ``ClientError``-shaped exceptions carrying the same error
codes — so the backend's error handling is exercised without boto3, network
access or AWS credentials. Listings are paginated with a deliberately tiny
page size, so code that forgets to follow continuation tokens is caught.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Iterator, List, Optional


class FakeClientError(Exception):
    """Shaped like botocore's ClientError: the code lives in ``response``."""

    def __init__(self, code: str, status: int, operation: str) -> None:
        super().__init__(f"An error occurred ({code}) when calling the {operation} operation")
        self.response = {"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _Paginator:
    def __init__(self, client: "FakeS3Client") -> None:
        self._client = client

    def paginate(self, **request: Any) -> Iterator[Dict[str, Any]]:
        token: Optional[str] = None
        while True:
            page = self._client.list_objects_v2(**request, **({"ContinuationToken": token} if token else {}))
            yield page
            if not page.get("IsTruncated"):
                return
            token = page["NextContinuationToken"]


class FakeS3Client:
    """Objects in a dict, per bucket."""

    def __init__(self, page_size: int = 2) -> None:
        self.objects: Dict[str, Dict[str, bytes]] = {}
        self.content_types: Dict[str, Optional[str]] = {}
        self.calls: List[str] = []
        self.page_size = page_size
        self.fail_with: Dict[str, FakeClientError] = {}
        self.fail_put_for_key: Dict[str, FakeClientError] = {}
        self.fail_put_on_call: Dict[int, FakeClientError] = {}
        self.fail_after_put_for_key: Dict[str, FakeClientError] = {}
        self.fail_after_put_on_call: Dict[int, FakeClientError] = {}
        self.put_call_count = 0
        self.conditional_write_successes: Dict[str, int] = {}
        self.conditional_write_attempts: Dict[str, int] = {}
        self.conditional_write_conflicts: Dict[str, int] = {}
        self.unconditional_write_keys: Dict[str, int] = {}
        self._lock = Lock()

    # -- helpers for tests ----------------------------------------------------

    def keys(self, bucket: str) -> List[str]:
        return sorted(self.objects.get(bucket, {}))

    def _maybe_fail(self, operation: str) -> None:
        self.calls.append(operation)
        if operation in self.fail_with:
            raise self.fail_with[operation]

    # -- the S3 API surface the backend uses ---------------------------------

    def head_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        self._maybe_fail("HeadObject")
        with self._lock:
            data = self.objects.get(Bucket, {}).get(Key)
        if data is None:
            raise FakeClientError("404", 404, "HeadObject")
        return {"ContentLength": len(data)}

    def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        self._maybe_fail("GetObject")
        with self._lock:
            data = self.objects.get(Bucket, {}).get(Key)
        if data is None:
            raise FakeClientError("NoSuchKey", 404, "GetObject")
        return {"Body": _Body(data), "ContentLength": len(data)}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: Optional[str] = None,
        IfNoneMatch: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._maybe_fail("PutObject")
        with self._lock:
            self.put_call_count += 1
            conditional = IfNoneMatch == "*"
            if conditional:
                self.conditional_write_attempts[Key] = self.conditional_write_attempts.get(Key, 0) + 1
            failure = self.fail_put_for_key.get(Key) or self.fail_put_on_call.get(self.put_call_count)
            if failure is not None:
                raise failure
            bucket = self.objects.setdefault(Bucket, {})
            if conditional and Key in bucket:
                self.conditional_write_conflicts[Key] = self.conditional_write_conflicts.get(Key, 0) + 1
                raise FakeClientError("PreconditionFailed", 412, "PutObject")
            bucket[Key] = bytes(Body)
            self.content_types[Key] = ContentType
            if conditional:
                self.conditional_write_successes[Key] = self.conditional_write_successes.get(Key, 0) + 1
            else:
                self.unconditional_write_keys[Key] = self.unconditional_write_keys.get(Key, 0) + 1
            after_failure = self.fail_after_put_for_key.get(Key) or self.fail_after_put_on_call.get(
                self.put_call_count
            )
            if after_failure is not None:
                raise after_failure
        return {"ETag": '"fake"'}

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        Delimiter: Optional[str] = None,
        MaxKeys: Optional[int] = None,
        ContinuationToken: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._maybe_fail("ListObjectsV2")
        with self._lock:
            bucket = self.objects.get(Bucket, {})
            keys = [key for key in sorted(bucket) if key.startswith(Prefix)]
            sizes = {key: len(bucket[key]) for key in keys}
        if Delimiter:
            keys = [key for key in keys if Delimiter not in key[len(Prefix) :]]
            sizes = {key: sizes[key] for key in keys}
        start = int(ContinuationToken) if ContinuationToken else 0
        size = min(MaxKeys or self.page_size, self.page_size)
        chunk = keys[start : start + size]
        page: Dict[str, Any] = {
            "Contents": [{"Key": key, "Size": sizes[key]} for key in chunk],
            "KeyCount": len(chunk),
            "IsTruncated": start + size < len(keys),
        }
        if not chunk:
            page.pop("Contents")
        if page["IsTruncated"]:
            page["NextContinuationToken"] = str(start + size)
        return page

    def get_paginator(self, operation: str) -> _Paginator:
        assert operation == "list_objects_v2", operation
        return _Paginator(self)
