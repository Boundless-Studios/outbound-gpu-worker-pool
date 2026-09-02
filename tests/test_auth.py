"""Bearer credential resolution for the outbound GPU worker pool.

Both authenticators are exercised directly. The Google one runs against an
injected verifier so no test reaches the network or needs a real identity token,
and no test ever asserts on the credential itself.
"""

import hashlib

import pytest

from outbound_gpu_worker_pool import (
    MemoryWorkerRegistry,
    WorkerAuthError,
    WorkerIdentity,
    WorkerRegistration,
)
from outbound_gpu_worker_pool.auth import (
    GOOGLE_OIDC_AUTH_METHOD,
    STATIC_AUTH_METHOD,
    GoogleIdTokenWorkerAuthenticator,
    StaticTokenWorkerAuthenticator,
)

AUDIENCE = "https://coordinator.invalid"
ENROLLED_EMAIL = "worker-a@coordinator.invalid"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def test_static_tokens_resolve_only_an_enrolled_digest() -> None:
    authenticator = StaticTokenWorkerAuthenticator.from_env_value(
        f"worker-a:{_digest('token-a')}"
    )

    identity = await authenticator.authenticate("Bearer token-a")

    assert identity == WorkerIdentity(
        "worker-a", f"{STATIC_AUTH_METHOD}:worker-a", STATIC_AUTH_METHOD
    )
    with pytest.raises(WorkerAuthError):
        await authenticator.authenticate("Bearer token-b")


async def test_static_tokens_compare_against_every_enrolled_entry() -> None:
    authenticator = StaticTokenWorkerAuthenticator.from_env_value(
        f"worker-a:{_digest('token-a')},"
        f"worker-b:{_digest('token-b')},"
        f"worker-c:{_digest('token-c')}"
    )

    resolved = [
        (await authenticator.authenticate(f"Bearer {token}")).worker_id
        for token in ("token-a", "token-b", "token-c")
    ]

    assert resolved == ["worker-a", "worker-b", "worker-c"]


@pytest.mark.parametrize(
    "authorization", [None, "", "token-a", "Basic token-a", "Bearer ", "Bearer"]
)
async def test_a_missing_or_empty_bearer_credential_is_rejected(
    authorization: str | None,
) -> None:
    authenticator = StaticTokenWorkerAuthenticator.from_env_value(
        f"worker-a:{_digest('token-a')}"
    )

    with pytest.raises(WorkerAuthError):
        await authenticator.authenticate(authorization)


@pytest.mark.parametrize(
    "value",
    [
        "worker-a:not-a-digest",
        "worker-a",
        f"worker-a:{'A' * 64}",
        f":{_digest('token-a')}",
        "",
        "   ",
    ],
)
def test_static_token_parsing_rejects_a_malformed_entry(value: str) -> None:
    with pytest.raises(ValueError):
        StaticTokenWorkerAuthenticator.from_env_value(value)


def test_static_token_parsing_skips_blank_entries() -> None:
    authenticator = StaticTokenWorkerAuthenticator.from_env_value(
        f" worker-a:{_digest('token-a')} , "
    )

    assert authenticator is not None


async def test_google_identity_resolves_an_enrolled_verified_email() -> None:
    registry = MemoryWorkerRegistry()
    await registry.upsert(
        WorkerRegistration(worker_id="worker-a", capabilities=()),
        identity_subject=ENROLLED_EMAIL,
    )
    authenticator = GoogleIdTokenWorkerAuthenticator(
        audience=AUDIENCE,
        registry=registry,
        verifier=lambda _token, _audience: {
            "email": ENROLLED_EMAIL,
            "email_verified": True,
        },
    )

    identity = await authenticator.authenticate("Bearer id-token")

    assert identity == WorkerIdentity(
        "worker-a", ENROLLED_EMAIL, GOOGLE_OIDC_AUTH_METHOD
    )


async def test_google_identity_rejects_an_unverified_email() -> None:
    authenticator = GoogleIdTokenWorkerAuthenticator(
        audience=AUDIENCE,
        registry=MemoryWorkerRegistry(),
        verifier=lambda _token, _audience: {
            "email": ENROLLED_EMAIL,
            "email_verified": False,
        },
    )

    with pytest.raises(WorkerAuthError):
        await authenticator.authenticate("Bearer id-token")


async def test_google_identity_rejects_a_token_without_an_email_claim() -> None:
    authenticator = GoogleIdTokenWorkerAuthenticator(
        audience=AUDIENCE,
        registry=MemoryWorkerRegistry(),
        verifier=lambda _token, _audience: {"email_verified": True},
    )

    with pytest.raises(WorkerAuthError):
        await authenticator.authenticate("Bearer id-token")


async def test_google_identity_rejects_an_unenrolled_subject() -> None:
    authenticator = GoogleIdTokenWorkerAuthenticator(
        audience=AUDIENCE,
        registry=MemoryWorkerRegistry(),
        verifier=lambda _token, _audience: {
            "email": "stranger@coordinator.invalid",
            "email_verified": True,
        },
    )

    with pytest.raises(WorkerAuthError):
        await authenticator.authenticate("Bearer id-token")


async def test_a_failing_verifier_is_an_authentication_error() -> None:
    def _explode(_token: str, _audience: str) -> dict[str, object]:
        raise ValueError("token is not signed for this audience")

    authenticator = GoogleIdTokenWorkerAuthenticator(
        audience=AUDIENCE, registry=MemoryWorkerRegistry(), verifier=_explode
    )

    with pytest.raises(WorkerAuthError):
        await authenticator.authenticate("Bearer id-token")
