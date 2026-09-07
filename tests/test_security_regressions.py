"""Security regressions for #15-18. Tokens/identities are synthetic fixtures."""

import asyncio
import json
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY as ECHO,
    JobSubmission,
    MemoryAssetStore,
    MemoryAuditLog,
    MemoryJobStore,
    MemoryWorkerAuthenticator,
    MemoryWorkerRegistry,
    WorkerAuthBusy,
    WorkerAuthError,
    WorkerCapability,
    WorkerEnrollment,
    WorkerIdentity,
    WorkerIdentityMismatch,
    WorkerRegistration,
    WorkerStatus,
    WorkerTenantMismatch,
)
from outbound_gpu_worker_pool.auth import GoogleIdTokenWorkerAuthenticator
from outbound_gpu_worker_pool.coordinator import create_coordinator_app
from outbound_gpu_worker_pool.enrollment import enrollments_from_json, validate_enrollments
from outbound_gpu_worker_pool.plugins import DeterministicEchoPlugin, capability_schemas_from_plugins
from outbound_gpu_worker_pool.routes import create_pool_status_router
from outbound_gpu_worker_pool.service import RateLimited, RateLimiter, WorkerPoolService, WorkerRevoked

IDENTITY = WorkerIdentity("worker-a", "static:worker-a", "static")
CAPABILITIES = (WorkerCapability(ECHO, "deterministic-echo", "1"),)
REGISTRATION = WorkerRegistration("worker-a", CAPABILITIES, tenant_id="tenant-a")
SCHEMAS = capability_schemas_from_plugins((DeterministicEchoPlugin(),))


def service(*, enrollments=None, authenticator=None, **kwargs):
    return WorkerPoolService(
        MemoryJobStore(), MemoryAssetStore(), MemoryWorkerRegistry(), MemoryAuditLog(),
        authenticator or MemoryWorkerAuthenticator({"token-a": IDENTITY}), SCHEMAS,
        enrollments=enrollments, **kwargs,
    )


def approved_service(**kwargs):
    return service(enrollments={"worker-a": WorkerEnrollment(IDENTITY.subject, "tenant-a")}, **kwargs)


@pytest.mark.parametrize("proposed", ["tenant-b", None])
async def test_initial_tenant_assignment_is_authorized_before_any_row_or_grant(proposed):
    pool = approved_service()
    with pytest.raises(WorkerTenantMismatch):
        await pool.register_heartbeat(IDENTITY, replace(REGISTRATION, tenant_id=proposed))
    assert await pool.list_workers() == ()
    assert not pool._assets.assets
    await pool.register_heartbeat(IDENTITY, REGISTRATION)
    assert (await pool.list_workers())[0].tenant_id == "tenant-a"


async def test_house_membership_also_requires_explicit_approval():
    pool = service(enrollments={"worker-a": WorkerEnrollment(IDENTITY.subject, None)})
    with pytest.raises(WorkerTenantMismatch):
        await pool.register_heartbeat(IDENTITY, REGISTRATION)
    await pool.register_heartbeat(IDENTITY, replace(REGISTRATION, tenant_id=None))


async def test_valid_credential_without_enrollment_is_rejected():
    pool = service()
    with pytest.raises(WorkerAuthError):
        await pool.authenticate("Bearer token-a")
    with pytest.raises(WorkerAuthError):
        await pool.register_heartbeat(IDENTITY, REGISTRATION)
    assert await pool.list_workers() == ()


async def test_host_enrollment_is_idempotent_and_does_not_reset_telemetry_or_revocation():
    pool = service()
    enrollment = WorkerEnrollment(IDENTITY.subject, "tenant-a")
    await pool.enroll_worker("worker-a", enrollment)
    await pool.register_heartbeat(IDENTITY, REGISTRATION)
    await pool.set_worker_status("worker-a", WorkerStatus.REVOKED)
    before = (await pool.list_workers())[0]
    assert await pool.enroll_worker("worker-a", enrollment) == before
    with pytest.raises(WorkerRevoked):
        await pool.authenticate("Bearer token-a")
    with pytest.raises(WorkerTenantMismatch):
        await pool.enroll_worker("worker-a", WorkerEnrollment(IDENTITY.subject, "tenant-b"))
    with pytest.raises(WorkerIdentityMismatch):
        await pool.enroll_worker("worker-a", WorkerEnrollment("other-subject", "tenant-a"))


async def test_concurrent_first_heartbeats_do_not_allow_an_unauthorized_tenant_to_win():
    pool = approved_service()
    results = await asyncio.gather(
        pool.register_heartbeat(IDENTITY, replace(REGISTRATION, tenant_id="tenant-b")),
        pool.register_heartbeat(IDENTITY, REGISTRATION),
        return_exceptions=True,
    )
    assert isinstance(results[0], WorkerTenantMismatch)
    assert results[1].tenant_id == "tenant-a"


async def test_approved_worker_cannot_lease_another_tenants_input_assets():
    pool = approved_service()
    await pool.register_heartbeat(IDENTITY, REGISTRATION)
    job = JobSubmission(str(uuid4()), "private-job", ECHO, (), "outputs/private.bin", tenant_id="tenant-b")
    await pool.submit(job)
    assert await pool.lease(IDENTITY, (ECHO,)) is None
    assert (await pool.get(job.job_id)).attempts == 0


async def test_memory_registry_never_replaces_an_existing_subject_even_when_tenant_matches():
    registry = MemoryWorkerRegistry()
    before = await registry.upsert(REGISTRATION, identity_subject=IDENTITY.subject)
    with pytest.raises(WorkerIdentityMismatch):
        await registry.upsert(replace(REGISTRATION, draining=True), identity_subject="other-subject")
    assert await registry.get("worker-a") == before


@pytest.mark.parametrize("suffix,body", [
    ("heartbeat", {"worker_id": "worker-a", "capabilities": [{"capability_id": ECHO, "plugin_id": "deterministic-echo", "plugin_version": "1"}], "tenant_id": "tenant-a"}),
    ("lease", {"capability_ids": [ECHO]}),
    ("jobs/not-a-job/heartbeat", {"claim_token": "claim"}),
    ("jobs/not-a-job/fail", {"claim_token": "claim", "reason": "test", "retryable": True, "failure_code": "temporary_failure"}),
    ("jobs/not-a-job/release", {"claim_token": "claim", "reason": "test"}),
])
async def test_every_mutating_worker_route_rejects_a_full_identity_mismatch(suffix, body):
    pool = approved_service(authenticator=MemoryWorkerAuthenticator({
        "token-a": IDENTITY,
        "collision": WorkerIdentity("worker-a", "other-subject", "google_oidc"),
    }))
    await pool.register_heartbeat(IDENTITY, REGISTRATION)
    before = (await pool.list_workers())[0]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_coordinator_app(pool)), base_url="http://test") as client:
        response = await client.post("/worker/v1/" + suffix, json=body, headers={"Authorization": "Bearer collision"})
        assert response.status_code == 401
        response = await client.get("/worker/v1/capabilities/schema", headers={"Authorization": "Bearer collision"})
        assert response.status_code == 401
    assert (await pool.list_workers())[0] == before


async def test_complete_and_direct_service_methods_also_reject_a_colliding_identity():
    pool = approved_service()
    await pool.register_heartbeat(IDENTITY, REGISTRATION)
    collision = replace(IDENTITY, subject="other-subject")
    for operation in (
        pool.lease(collision, (ECHO,)),
        pool.job_heartbeat(collision, "not-a-job", "claim"),
        pool.complete(collision, "not-a-job", "claim", None),
        pool.release(collision, "not-a-job", "claim", "test"),
    ):
        with pytest.raises(WorkerIdentityMismatch):
            await operation


@pytest.mark.parametrize("email", [
    "gpu-worker-rig-01@otherproject.iam.gserviceaccount.com",
    "gpu-worker-rig-01@example.com",
    "gpu-worker-rig-01@project.iam.gserviceaccount.com.example.com",
])
async def test_oidc_auto_enrollment_is_an_exact_service_account_allowlist(email):
    auth = GoogleIdTokenWorkerAuthenticator(
        "audience", MemoryWorkerRegistry(), lambda *_: {"email": email, "email_verified": True},
        auto_enroll=True,
        enrollments={"rig-01": WorkerEnrollment("gpu-worker-rig-01@project.iam.gserviceaccount.com", None)},
    )
    with pytest.raises(WorkerAuthError):
        await auth.authenticate("Bearer synthetic")


async def test_oidc_collision_is_rejected_before_any_heartbeat_even_with_a_conflicting_allowlist():
    registry = MemoryWorkerRegistry()
    await registry.upsert(WorkerRegistration("rig-01", ()), identity_subject="gpu-worker-rig-01@project.iam.gserviceaccount.com")
    other = "gpu-worker-rig-01@otherproject.iam.gserviceaccount.com"
    auth = GoogleIdTokenWorkerAuthenticator(
        "audience", registry, lambda *_: {"email": other, "email_verified": True},
        auto_enroll=True, enrollments={"rig-01": WorkerEnrollment(other, None)},
    )
    with pytest.raises(WorkerAuthError):
        await auth.authenticate("Bearer synthetic")


def test_oidc_auto_enrollment_without_policy_fails_at_startup():
    with pytest.raises(ValueError, match="allowlist"):
        GoogleIdTokenWorkerAuthenticator("audience", MemoryWorkerRegistry(), auto_enroll=True)


async def test_cancelled_oidc_request_does_not_free_a_running_verifiers_slot():
    started, finish = threading.Event(), threading.Event()
    def verifier(*_):
        started.set()
        assert finish.wait(timeout=5)
        return {"email": "gpu-worker-rig-01@project.iam.gserviceaccount.com", "email_verified": True}
    auth = GoogleIdTokenWorkerAuthenticator(
        "audience", MemoryWorkerRegistry(), verifier,
        auto_enroll=True, max_concurrent_verifications=1,
        enrollments={"rig-01": WorkerEnrollment("gpu-worker-rig-01@project.iam.gserviceaccount.com", None)},
    )
    task = asyncio.create_task(auth.authenticate("Bearer synthetic"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(WorkerAuthBusy):
            await auth.authenticate("Bearer synthetic")
    finally:
        finish.set()
        await asyncio.gather(*auth._verification_tasks, return_exceptions=True)
    assert not auth._verification_slots.locked()


class CountingAuthenticator:
    def __init__(self):
        self.calls = 0
    async def authenticate(self, authorization):
        self.calls += 1
        if authorization == "Bearer token-a":
            return IDENTITY
        raise WorkerAuthError("invalid fixture credential")


async def test_invalid_flood_has_bounded_verification_and_audit_and_cannot_starve_other_peer():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    auth = CountingAuthenticator()
    pool = approved_service(
        authenticator=auth, clock=lambda: now[0], per_source_limit_per_minute=3,
        pre_auth_limit_per_minute=10, auth_audit_limit_per_minute=2,
    )
    for i in range(50):
        with pytest.raises(WorkerAuthError if i < 3 else RateLimited):
            await pool.authenticate("Bearer invalid", source="peer-a")
    assert auth.calls == 3
    assert len(pool._audit.events) == 2
    assert await pool.authenticate("Bearer token-a", source="peer-b") == IDENTITY
    now[0] += timedelta(seconds=60)
    assert await pool.authenticate("Bearer token-a", source="peer-a") == IDENTITY


async def test_pre_auth_global_limit_bounds_many_distinct_sources():
    auth = CountingAuthenticator()
    pool = approved_service(authenticator=auth, pre_auth_limit_per_minute=3)
    for i in range(30):
        with pytest.raises(WorkerAuthError if i < 3 else RateLimited):
            await pool.authenticate("Bearer invalid", source=f"peer-{i}")
    assert auth.calls == 3


async def test_concurrent_auth_requests_reject_without_building_a_waiter_queue():
    started, finish = asyncio.Event(), asyncio.Event()
    class BlockedAuthenticator:
        calls = 0
        async def authenticate(self, authorization):
            self.calls += 1
            started.set()
            await finish.wait()
            return IDENTITY
    auth = BlockedAuthenticator()
    pool = approved_service(authenticator=auth, max_auth_concurrency=1)
    first = asyncio.create_task(pool.authenticate("Bearer token-a"))
    await started.wait()
    with pytest.raises(RateLimited):
        await pool.authenticate("Bearer token-a")
    assert auth.calls == 1
    finish.set()
    assert await first == IDENTITY
    assert pool._active_auth == 0


def test_rate_limiter_is_bounded_does_not_evict_live_budgets_and_expires_idle_keys():
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    limiter = RateLimiter(1, lambda: now[0], max_keys=2)
    assert limiter.allow("one") and limiter.allow("two")
    assert not limiter.allow("three") and not limiter.allow("one")
    assert len(limiter._buckets) == 2
    now[0] += timedelta(seconds=61)
    assert limiter.allow("three")
    assert len(limiter._buckets) == 1


def test_forwarded_headers_do_not_choose_the_librarys_admission_source():
    auth = CountingAuthenticator()
    pool = approved_service(authenticator=auth, per_source_limit_per_minute=1)
    client = TestClient(create_coordinator_app(pool))
    for index, expected in enumerate((401, 429, 429)):
        response = client.get("/worker/v1/capabilities/schema", headers={
            "Authorization": "Bearer invalid", "X-Forwarded-For": f"203.0.113.{index}",
        })
        assert response.status_code == expected
    assert auth.calls == 1


@pytest.mark.parametrize("role,expected", [(None, 401), ("tenant-a", 403), ("admin", 200)])
@pytest.mark.parametrize("path", ["/pool/workers", "/pool/workers?tenant_id=tenant-b", "/pool/queue"])
def test_pool_status_requires_admin_even_for_authenticated_tenants(role, expected, path):
    def authorize_admin(x_role: str | None = Header(default=None)):
        # Fixture only. A production host checks its own authenticated principal.
        if x_role is None:
            raise HTTPException(401, "login required")
        return x_role == "admin"
    app = FastAPI()
    app.include_router(create_pool_status_router(approved_service(), authorize_admin=authorize_admin))
    response = TestClient(app).get(path, headers={} if role is None else {"X-Role": role})
    assert response.status_code == expected


def test_status_cannot_be_mounted_without_explicit_admin_authorization():
    with pytest.raises(TypeError):
        create_pool_status_router(approved_service())
    app = FastAPI()
    app.include_router(create_pool_status_router(approved_service(), authorize_admin=lambda: {"logged_in": True}))
    assert TestClient(app).get("/pool/workers").status_code == 403


@pytest.mark.parametrize("data", [
    [], {"worker-a": {"identity_subject": "subject"}},
    {"worker-a": {"identity_subject": "subject", "tenant_id": ""}},
    {"worker-a": {"identity_subject": "subject", "tenant_id": 12}},
    {"worker-a": {"identity_subject": "subject", "tenant_id": None, "extra": True}},
    {"Bad ID": {"identity_subject": "subject", "tenant_id": None}},
])
def test_enrollment_configuration_rejects_implicit_or_malformed_membership(data):
    with pytest.raises(ValueError, match="OGWP_WORKER_ENROLLMENTS"):
        enrollments_from_json(json.dumps(data))


def test_enrollment_configuration_rejects_duplicate_subjects_and_duplicate_json_keys():
    with pytest.raises(ValueError):
        validate_enrollments({"a": WorkerEnrollment("same", None), "b": WorkerEnrollment("same", None)})
    with pytest.raises(ValueError):
        enrollments_from_json('{"worker-a": {}, "worker-a": {}}')
    assert enrollments_from_json('{"worker-a":{"identity_subject":"static:worker-a","tenant_id":null}}') == {
        "worker-a": WorkerEnrollment("static:worker-a", None)
    }
