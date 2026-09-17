"""Amazon S3 storage.

Nothing in the pipeline imports ``boto3``: stages talk to
:class:`~f1_race_intelligence.storage.backends.base.StorageBackend`, and
this is the only module that knows S3 exists. ``boto3`` itself is an
optional dependency (``pip install ".[aws]"``) imported on first use, so a
local installation never needs it.

Credentials are never configured here. The client is created with the
default credential chain, which inside AWS resolves to the IAM role of the
task or job that runs the container.

How the two storage guarantees hold on S3:

* ``PutObject`` is atomic: readers see the previous object or the new one,
  never a partial upload. No temporary-file dance is needed.
* ``write_bytes_if_absent`` uses a conditional write (``If-None-Match: *``),
  so two concurrent writers of the same execution record cannot both win.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from f1_race_intelligence.storage.backends.base import StorageBackend, StorageError, join_key

_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}
# 412: the object already exists. 409: another conditional write to the same
# key is in flight; either way this name is taken.
_ALREADY_EXISTS_CODES = {"412", "PreconditionFailed", "409", "ConditionalRequestConflict"}


def _error_code(exc: BaseException) -> Optional[str]:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error") or {}
    code = error.get("Code")
    if code is None:
        status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        return str(status) if status is not None else None
    return str(code)


class S3StorageBackend(StorageBackend):
    """Stores objects under ``s3://<bucket>/<prefix>/``."""

    name = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: Any = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        """Create a backend.

        Args:
            bucket: Bucket name, without ``s3://``.
            prefix: Optional key prefix every object lives under, so one
                bucket can hold several environments (``dev``, ``prod``).
            client: A boto3-compatible S3 client. Tests inject a stub here;
                when omitted one is created lazily on first use.
            region_name: Passed to boto3 when it creates the client.
            endpoint_url: For S3-compatible endpoints (LocalStack, MinIO).
        """
        if not bucket or "/" in bucket or bucket.startswith("s3:"):
            raise ValueError(f"Invalid S3 bucket name: {bucket!r}")
        self._bucket = bucket
        self._prefix = join_key(prefix).strip("/")
        self._client = client
        self._region_name = region_name
        self._endpoint_url = endpoint_url

    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "The S3 storage backend needs boto3. Install the AWS extra: pip install '.[aws]'"
                ) from exc
            self._client = boto3.client("s3", region_name=self._region_name, endpoint_url=self._endpoint_url)
        return self._client

    # -- key mapping ----------------------------------------------------------

    def _object_key(self, key: str) -> str:
        logical = join_key(key).strip("/")
        segments = logical.split("/") if logical else []
        if any(segment in {".", ".."} for segment in segments) or (segments and segments[0].endswith(":")):
            raise ValueError(f"Invalid storage key for S3: {key!r}")
        return join_key(self._prefix, logical)

    def _logical_key(self, object_key: str) -> str:
        if not self._prefix:
            return object_key
        return object_key[len(self._prefix) + 1 :]

    def _listing_prefix(self, prefix: str) -> str:
        # "raw/laps" must not match "raw/laps_extra": list a directory, not a stem.
        object_prefix = self._object_key(prefix)
        return f"{object_prefix}/" if object_prefix else ""

    # -- primitives -----------------------------------------------------------

    def uri(self, key: str) -> str:
        object_key = self._object_key(key)
        return f"s3://{self._bucket}/{object_key}" if object_key else f"s3://{self._bucket}"

    def exists(self, key: str) -> bool:
        object_key = self._object_key(key)
        try:
            self.client.head_object(Bucket=self._bucket, Key=object_key)
        except Exception as exc:  # botocore is optional: classify by error code
            if _error_code(exc) in _NOT_FOUND_CODES:
                return False
            raise self._storage_error("head", key, exc) from exc
        return True

    def has_prefix(self, prefix: str) -> bool:
        listing_prefix = self._listing_prefix(prefix)
        try:
            response = self.client.list_objects_v2(Bucket=self._bucket, Prefix=listing_prefix, MaxKeys=1)
        except Exception as exc:
            raise self._storage_error("list", prefix, exc) from exc
        return bool(response.get("Contents"))

    def read_bytes(self, key: str) -> bytes:
        object_key = self._object_key(key)
        try:
            response = self.client.get_object(Bucket=self._bucket, Key=object_key)
            return response["Body"].read()
        except Exception as exc:
            if _error_code(exc) in _NOT_FOUND_CODES:
                raise FileNotFoundError(f"No object at {self.uri(key)}") from exc
            raise self._storage_error("read", key, exc) from exc

    def write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> str:
        self._put(key, data, content_type=content_type)
        return key

    def write_bytes_if_absent(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> bool:
        try:
            self._put(key, data, content_type=content_type, if_none_match=True)
        except StorageError as exc:
            if _error_code(exc.__cause__) in _ALREADY_EXISTS_CODES:
                return False
            raise
        return True

    def list(self, prefix: str, *, recursive: bool = True) -> List[str]:
        request = {"Bucket": self._bucket, "Prefix": self._listing_prefix(prefix)}
        if not recursive:
            request["Delimiter"] = "/"
        try:
            pages: Iterable[dict] = self.client.get_paginator("list_objects_v2").paginate(**request)
            keys = [
                self._logical_key(item["Key"])
                for page in pages
                for item in page.get("Contents", [])
                if not item["Key"].endswith("/") and not item["Key"].rsplit("/", 1)[-1].startswith(".")
            ]
        except Exception as exc:
            raise self._storage_error("list", prefix, exc) from exc
        return sorted(keys)

    def size(self, key: str) -> int:
        object_key = self._object_key(key)
        try:
            response = self.client.head_object(Bucket=self._bucket, Key=object_key)
        except Exception as exc:
            if _error_code(exc) in _NOT_FOUND_CODES:
                raise FileNotFoundError(f"No object at {self.uri(key)}") from exc
            raise self._storage_error("head", key, exc) from exc
        return int(response["ContentLength"])

    def describe(self) -> dict:
        return {"backend": self.name, "location": self.uri(""), "bucket": self._bucket, "prefix": self._prefix}

    # -- helpers --------------------------------------------------------------

    def _put(self, key: str, data: bytes, *, content_type: Optional[str], if_none_match: bool = False) -> None:
        request = {"Bucket": self._bucket, "Key": self._object_key(key), "Body": data}
        if content_type:
            request["ContentType"] = content_type
        if if_none_match:
            request["IfNoneMatch"] = "*"
        try:
            self.client.put_object(**request)
        except Exception as exc:
            raise self._storage_error("write", key, exc) from exc

    def _storage_error(self, operation: str, key: str, exc: BaseException) -> StorageError:
        code = _error_code(exc)
        detail = f" ({code})" if code else ""
        return StorageError(f"S3 {operation} failed for {self.uri(key)}{detail}: {exc}")
