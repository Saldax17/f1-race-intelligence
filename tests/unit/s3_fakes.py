"""An in-memory stand-in for a boto3 S3 client.

It implements only the calls :class:`S3StorageBackend` makes, and fails the
way S3 does — ``ClientError``-shaped exceptions carrying the same error
codes — so the backend's error handling is exercised without boto3, network
access or AWS credentials. Listings are paginated with a deliberately tiny
page size, so code that forgets to follow continuation tokens is caught.
"""

from __future__ import annotations

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
        data = self.objects.get(Bucket, {}).get(Key)
        if data is None:
            raise FakeClientError("404", 404, "HeadObject")
        return {"ContentLength": len(data)}

    def get_object(self, *, Bucket: str, Key: str) -> Dict[str, Any]:
        self._maybe_fail("GetObject")
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
        bucket = self.objects.setdefault(Bucket, {})
        if IfNoneMatch == "*" and Key in bucket:
            raise FakeClientError("PreconditionFailed", 412, "PutObject")
        bucket[Key] = bytes(Body)
        self.content_types[Key] = ContentType
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
        keys = [key for key in sorted(self.objects.get(Bucket, {})) if key.startswith(Prefix)]
        if Delimiter:
            keys = [key for key in keys if Delimiter not in key[len(Prefix) :]]
        start = int(ContinuationToken) if ContinuationToken else 0
        size = min(MaxKeys or self.page_size, self.page_size)
        chunk = keys[start : start + size]
        page: Dict[str, Any] = {
            "Contents": [{"Key": key, "Size": len(self.objects[Bucket][key])} for key in chunk],
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
