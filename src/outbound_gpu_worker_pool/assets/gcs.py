"""Google Cloud Storage asset store (install the `gcs` extra).

Grants are exact-object, single-method, and short-lived: a read is a five-minute
signed GET, an output upload is a fifteen-minute signed PUT carrying
`ifGenerationMatch=0` so a worker can create its output exactly once and can
never overwrite or list anything. Which prefixes may be read or written is the
caller's policy, not this module's.
"""

import asyncio
from collections.abc import Iterable
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Protocol

import google.auth
from google.api_core.exceptions import PreconditionFailed
from google.auth.transport.requests import Request
from google.cloud import storage
from google.cloud.exceptions import NotFound

from outbound_gpu_worker_pool.contracts import (
    DEFAULT_OUTPUT_PREFIXES,
    DEFAULT_READ_PREFIXES,
    AssetDescriptor,
    AssetNotFound,
    AssetStorageUnavailable,
    AssetTooLarge,
)

READ_URL_LIFETIME = timedelta(minutes=5)
OUTPUT_UPLOAD_URL_LIFETIME = timedelta(minutes=15)
CREATE_ONCE_QUERY_PARAMETERS = {"ifGenerationMatch": "0"}


class _Blob(Protocol):
    size: int | None
    generation: int | None
    content_type: str | None

    def reload(self) -> None: ...

    def download_as_bytes(self, **kwargs: object) -> bytes: ...

    def upload_from_string(
        self, content: bytes, *, content_type: str, if_generation_match: int
    ) -> None: ...

    def generate_signed_url(self, **kwargs: object) -> str: ...


class _Bucket(Protocol):
    def blob(self, key: str) -> _Blob: ...


class _StorageClient(Protocol):
    def bucket(self, name: str) -> _Bucket: ...

    def list_blobs(self, bucket_name: str, *, max_results: int) -> Iterable[_Blob]: ...


class GcsAssetStore:
    def __init__(
        self,
        *,
        bucket: str,
        client: _StorageClient | None = None,
        precondition_failed: type[Exception] = PreconditionFailed,
        signing_service_account_email: str | None = None,
        signing_access_token: str | None = None,
        allowed_read_prefixes: tuple[str, ...] = DEFAULT_READ_PREFIXES,
        allowed_output_prefixes: tuple[str, ...] = DEFAULT_OUTPUT_PREFIXES,
    ) -> None:
        self._bucket = bucket
        self._client = client
        self._precondition_failed = precondition_failed
        self._signing_service_account_email = signing_service_account_email
        self._signing_access_token = signing_access_token
        self._allowed_read_prefixes = allowed_read_prefixes
        self._allowed_output_prefixes = allowed_output_prefixes
        self._ambient_credentials = None

    async def start(self) -> None:
        if self._client is None:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            await asyncio.to_thread(credentials.refresh, Request())
            self._ambient_credentials = credentials
            self._signing_access_token = credentials.token
            self._client = storage.Client(credentials=credentials)
        client = self._client
        try:
            await asyncio.to_thread(
                lambda: next(iter(client.list_blobs(self._bucket, max_results=1)), None)
            )
        except Exception as exc:
            raise RuntimeError(f"asset bucket is unavailable: {self._bucket}") from exc

    async def stop(self) -> None:
        """Drop the client and signing credentials this store created for itself."""
        if self._ambient_credentials is not None:
            self._client = None
            self._ambient_credentials = None
            self._signing_access_token = None

    async def create_read_url(self, key: str) -> str:
        object_key = _validate_object_key(key)
        _require_prefix(object_key, self._allowed_read_prefixes, "read")
        return await self._signed_url(
            object_key, method="GET", expiration=READ_URL_LIFETIME
        )

    async def create_output_upload_url(self, key: str, content_type: str) -> str:
        object_key = _validate_object_key(key)
        _require_prefix(object_key, self._allowed_output_prefixes, "output upload")
        return await self._signed_url(
            object_key,
            method="PUT",
            expiration=OUTPUT_UPLOAD_URL_LIFETIME,
            content_type=content_type,
            query_parameters=dict(CREATE_ONCE_QUERY_PARAMETERS),
        )

    async def describe(self, key: str) -> AssetDescriptor | None:
        object_key = _validate_object_key(key)
        blob = self._blob(object_key)
        try:
            await asyncio.to_thread(blob.reload)
        except (FileNotFoundError, NotFound):
            return None
        except Exception as exc:
            raise AssetStorageUnavailable("asset storage unavailable") from exc
        return AssetDescriptor(
            key=object_key, size=int(blob.size or 0), content_type=blob.content_type
        )

    async def read_limited(self, key: str, max_bytes: int) -> bytes:
        object_key = _validate_object_key(key)
        blob = self._blob(object_key)
        try:
            await asyncio.to_thread(blob.reload)
            if blob.size is not None and int(blob.size) > max_bytes:
                raise AssetTooLarge(object_key, max_bytes, int(blob.size))
            download_kwargs: dict[str, object] = {"start": 0, "end": max_bytes}
            if blob.generation is not None:
                download_kwargs["if_generation_match"] = blob.generation
            content = await asyncio.to_thread(blob.download_as_bytes, **download_kwargs)
        except (FileNotFoundError, NotFound) as exc:
            raise AssetNotFound(object_key) from exc
        except AssetTooLarge:
            raise
        except Exception as exc:
            raise AssetStorageUnavailable("asset storage unavailable") from exc
        if len(content) > max_bytes:
            raise AssetTooLarge(object_key, max_bytes, len(content))
        return content

    async def write_once(self, key: str, content: bytes, content_type: str) -> bool:
        blob = self._blob(_validate_object_key(key))
        try:
            await asyncio.to_thread(
                blob.upload_from_string,
                content,
                content_type=content_type,
                if_generation_match=0,
            )
        except self._precondition_failed:
            return False
        except Exception as exc:
            raise AssetStorageUnavailable("asset storage unavailable") from exc
        return True

    async def _signed_url(
        self, object_key: str, *, method: str, expiration: timedelta, **options: object
    ) -> str:
        try:
            if self._ambient_credentials is not None:
                await asyncio.to_thread(self._ambient_credentials.refresh, Request())
                self._signing_access_token = self._ambient_credentials.token
        except Exception as exc:
            raise AssetStorageUnavailable("asset storage unavailable") from exc
        if not self._signing_service_account_email or not self._signing_access_token:
            raise RuntimeError("IAM signing credentials are not configured")
        blob = self._blob(object_key)
        try:
            return await asyncio.to_thread(
                blob.generate_signed_url,
                version="v4",
                expiration=expiration,
                method=method,
                service_account_email=self._signing_service_account_email,
                access_token=self._signing_access_token,
                **options,
            )
        except Exception as exc:
            raise AssetStorageUnavailable("asset storage unavailable") from exc

    def _blob(self, object_key: str) -> _Blob:
        if self._client is None:
            raise RuntimeError("GCS asset store has not started")
        return self._client.bucket(self._bucket).blob(object_key)


def _validate_object_key(key: str) -> str:
    normalized = str(PurePosixPath(key))
    if (
        not key
        or key.startswith(("/", "gs://"))
        or normalized != key
        or ".." in PurePosixPath(key).parts
    ):
        raise ValueError("GCS object key must be a normalized relative path")
    return key


def _require_prefix(key: str, prefixes: tuple[str, ...], grant: str) -> None:
    if not key.startswith(prefixes):
        allowed = ", ".join(prefixes)
        raise ValueError(f"signed {grant} key must be in one of: {allowed}")
