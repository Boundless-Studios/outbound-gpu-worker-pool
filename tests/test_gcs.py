"""GCS asset store behavior, driven by a fake storage client.

The fake stands in for the Google client library only; every policy under test
(namespace allowlists, create-once preconditions, bounded reads, failure
normalization) belongs to this package.
"""

from datetime import timedelta

import pytest
from google.api_core.exceptions import NotFound

from outbound_gpu_worker_pool import (
    AssetNotFound,
    AssetStorageUnavailable,
    AssetTooLarge,
)
from outbound_gpu_worker_pool.assets.gcs import GcsAssetStore

BUCKET = "pool-assets"


class _Blob:
    def __init__(
        self, content: bytes | None = None, *, error: BaseException | None = None
    ) -> None:
        self.content = content
        self.error = error
        self.size = len(content) if content is not None else None
        self.generation: int | None = 7
        self.generation_match: int | None = None
        self.content_type: str | None = None
        self.signed_url_options: dict[str, object] | None = None
        self.download_call_count = 0
        self.last_download_kwargs: dict[str, object] | None = None
        self.reload_call_count = 0

    def reload(self) -> None:
        self.reload_call_count += 1
        if self.error is not None:
            raise self.error

    def download_as_bytes(self, **kwargs: object) -> bytes:
        self.download_call_count += 1
        self.last_download_kwargs = kwargs
        if self.error is not None:
            raise self.error
        if self.content is None:
            raise FileNotFoundError
        start = int(kwargs.get("start", 0) or 0)
        end = kwargs.get("end")
        if end is None:
            return self.content[start:]
        return self.content[start : int(end) + 1]

    def upload_from_string(
        self, content: bytes, *, content_type: str, if_generation_match: int
    ) -> None:
        self.generation_match = if_generation_match
        self.content_type = content_type
        if self.content is not None:
            raise _PreconditionFailed
        self.content = content

    def generate_signed_url(self, **kwargs: object) -> str:
        self.signed_url_options = kwargs
        assert kwargs["version"] == "v4"
        assert kwargs["method"] in {"GET", "PUT"}
        return "https://grant.invalid/signed"


class _Bucket:
    def __init__(self, blobs: dict[str, _Blob]) -> None:
        self.blobs = blobs

    def blob(self, key: str) -> _Blob:
        return self.blobs.setdefault(key, _Blob())


class _Client:
    def __init__(self, buckets: dict[str, _Bucket]) -> None:
        self.buckets = buckets

    def bucket(self, name: str) -> _Bucket:
        return self.buckets[name]

    def list_blobs(self, bucket_name: str, *, max_results: int):
        assert max_results == 1
        return iter(self.buckets[bucket_name].blobs.values())


class _PreconditionFailed(Exception):
    pass


class _UnavailableClient:
    def list_blobs(self, bucket_name: str, *, max_results: int):
        raise PermissionError(f"cannot list {bucket_name}")


def _store(bucket: _Bucket, **overrides: object) -> GcsAssetStore:
    options: dict[str, object] = {
        "client": _Client({BUCKET: bucket}),
        "bucket": BUCKET,
        "precondition_failed": _PreconditionFailed,
        "signing_service_account_email": "worker-pool@pool.invalid",
        "signing_access_token": "token",
    }
    options.update(overrides)
    return GcsAssetStore(**options)  # type: ignore[arg-type]


def test_construction_does_not_resolve_ambient_credentials(monkeypatch) -> None:
    def reject_client_construction():
        raise AssertionError("credentials resolved before startup")

    monkeypatch.setattr(
        "outbound_gpu_worker_pool.assets.gcs.storage.Client", reject_client_construction
    )

    GcsAssetStore(bucket=BUCKET)


async def test_start_validates_object_list_access_without_bucket_metadata() -> None:
    await _store(_Bucket({})).start()


async def test_start_normalizes_bucket_access_failures() -> None:
    store = _store(_Bucket({}), client=_UnavailableClient())

    with pytest.raises(
        RuntimeError, match=f"asset bucket is unavailable: {BUCKET}"
    ) as error:
        await store.start()

    assert isinstance(error.value.__cause__, PermissionError)


async def test_bounded_read_returns_the_object_bytes() -> None:
    store = _store(_Bucket({"inputs/pool/a.bin": _Blob(b"source")}))

    assert await store.read_limited("inputs/pool/a.bin", max_bytes=32) == b"source"


async def test_bounded_read_maps_a_missing_object_to_asset_not_found() -> None:
    store = _store(_Bucket({"inputs/pool/a.bin": _Blob(error=NotFound("missing"))}))

    with pytest.raises(AssetNotFound):
        await store.read_limited("inputs/pool/a.bin", max_bytes=32)


async def test_bounded_read_rejects_an_oversized_blob_from_metadata_alone() -> None:
    blob = _Blob(b"x" * 32)
    store = _store(_Bucket({"inputs/pool/a.bin": blob}))

    with pytest.raises(AssetTooLarge):
        await store.read_limited("inputs/pool/a.bin", max_bytes=16)

    assert blob.reload_call_count == 1
    assert blob.download_call_count == 0


async def test_bounded_read_downloads_at_most_max_bytes_plus_one_with_a_generation_pin() -> (
    None
):
    blob = _Blob(b"1234567890")
    blob.size = None
    store = _store(_Bucket({"inputs/pool/a.bin": blob}))

    with pytest.raises(AssetTooLarge):
        await store.read_limited("inputs/pool/a.bin", max_bytes=5)

    assert blob.download_call_count == 1
    assert blob.last_download_kwargs == {"start": 0, "end": 5, "if_generation_match": 7}


async def test_bounded_read_normalizes_a_non_runtime_provider_failure() -> None:
    store = _store(_Bucket({"inputs/pool/a.bin": _Blob(error=PermissionError("no"))}))

    with pytest.raises(AssetStorageUnavailable) as error:
        await store.read_limited("inputs/pool/a.bin", max_bytes=16)

    assert isinstance(error.value.__cause__, PermissionError)


async def test_describe_reports_size_and_content_type() -> None:
    blob = _Blob(b"echo")
    blob.content_type = "text/plain"
    store = _store(_Bucket({"outputs/pool/echo.txt": blob}))

    descriptor = await store.describe("outputs/pool/echo.txt")

    assert descriptor is not None
    assert descriptor.key == "outputs/pool/echo.txt"
    assert descriptor.size == 4
    assert descriptor.content_type == "text/plain"


async def test_describe_returns_none_for_a_missing_object() -> None:
    store = _store(_Bucket({"outputs/pool/echo.txt": _Blob(error=NotFound("gone"))}))

    assert await store.describe("outputs/pool/echo.txt") is None


async def test_describe_normalizes_a_provider_failure() -> None:
    store = _store(
        _Bucket({"outputs/pool/echo.txt": _Blob(error=PermissionError("no"))})
    )

    with pytest.raises(AssetStorageUnavailable):
        await store.describe("outputs/pool/echo.txt")


async def test_publishes_an_output_once_with_a_generation_precondition() -> None:
    bucket = _Bucket({})
    store = _store(bucket)

    first = await store.write_once("outputs/pool/echo.txt", b"echo", "text/plain")
    second = await store.write_once("outputs/pool/echo.txt", b"echo", "text/plain")

    assert first is True
    assert second is False
    assert bucket.blobs["outputs/pool/echo.txt"].generation_match == 0
    assert bucket.blobs["outputs/pool/echo.txt"].content_type == "text/plain"
    assert bucket.blobs["outputs/pool/echo.txt"].content == b"echo"


async def test_creates_a_create_once_signed_upload_for_an_output() -> None:
    bucket = _Bucket({})
    store = _store(bucket)

    url = await store.create_output_upload_url("outputs/pool/echo.txt", "text/plain")

    assert url == "https://grant.invalid/signed"
    options = bucket.blobs["outputs/pool/echo.txt"].signed_url_options
    assert options is not None
    assert options["method"] == "PUT"
    assert options["content_type"] == "text/plain"
    assert options["expiration"] == timedelta(minutes=15)
    assert options["query_parameters"] == {"ifGenerationMatch": "0"}
    assert options["service_account_email"] == "worker-pool@pool.invalid"
    assert options["access_token"] == "token"


async def test_signed_output_uploads_are_limited_to_the_output_prefixes() -> None:
    store = _store(_Bucket({}))

    with pytest.raises(ValueError, match="outputs/"):
        await store.create_output_upload_url("inputs/pool/echo.bin", "text/plain")


async def test_creates_a_short_lived_signed_read_url() -> None:
    bucket = _Bucket({})
    store = _store(bucket)

    url = await store.create_read_url("inputs/pool/echo.bin")

    assert url == "https://grant.invalid/signed"
    options = bucket.blobs["inputs/pool/echo.bin"].signed_url_options
    assert options is not None
    assert options["method"] == "GET"
    assert options["expiration"] == timedelta(minutes=5)


async def test_signed_reads_are_limited_to_the_read_prefixes() -> None:
    store = _store(_Bucket({}))

    with pytest.raises(ValueError, match="inputs/"):
        await store.create_read_url("secrets/pool/echo.bin")


async def test_custom_prefixes_replace_the_defaults() -> None:
    store = _store(
        _Bucket({}),
        allowed_read_prefixes=("shared/",),
        allowed_output_prefixes=("shared/out/",),
    )

    assert await store.create_read_url("shared/a.bin") == "https://grant.invalid/signed"
    assert (
        await store.create_output_upload_url("shared/out/b.bin", "text/plain")
        == "https://grant.invalid/signed"
    )
    with pytest.raises(ValueError, match="shared/"):
        await store.create_read_url("inputs/pool/a.bin")


async def test_signing_requires_configured_credentials() -> None:
    store = _store(_Bucket({}), signing_service_account_email=None)

    with pytest.raises(RuntimeError, match="IAM signing credentials"):
        await store.create_output_upload_url("outputs/pool/echo.txt", "text/plain")


async def test_requires_start_before_use() -> None:
    store = GcsAssetStore(bucket=BUCKET)

    with pytest.raises(RuntimeError, match="has not started"):
        await store.describe("outputs/pool/echo.txt")


@pytest.mark.parametrize(
    "key", ["gs://other-bucket/outputs/job.txt", "../job.txt", "/outputs/job.txt", ""]
)
async def test_rejects_nonlocal_or_unnormalized_keys(key: str) -> None:
    store = _store(_Bucket({}))

    with pytest.raises(ValueError, match="normalized relative path"):
        await store.write_once(key, b"echo", "text/plain")
