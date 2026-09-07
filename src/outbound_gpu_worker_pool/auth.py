"""Bearer credential resolvers for the outbound worker pool.

Both authenticators map one `Authorization: Bearer <credential>` header to a
`WorkerIdentity` or raise `WorkerAuthError`. Neither ever logs, stores, or returns
the credential itself.

The Google resolver imports `google.auth` inside its default verifier, so this
module — and therefore the coordinator router — imports without the `google-auth`
extra installed.
"""

import asyncio
import hashlib
import hmac
import re
from collections.abc import Callable, Mapping

from outbound_gpu_worker_pool.contracts import (
    WorkerAuthError,
    WorkerAuthBusy,
    WorkerIdentity,
    WorkerRegistry,
)

from outbound_gpu_worker_pool.enrollment import WorkerEnrollment, validate_enrollments

BEARER_PREFIX = "Bearer "
STATIC_AUTH_METHOD = "static"
GOOGLE_OIDC_AUTH_METHOD = "google_oidc"
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_WORKER_ACCOUNT_PREFIX = "gpu-worker-"
WORKER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")


def _bearer_credential(authorization: str | None) -> str:
    if authorization is None or not authorization.startswith(BEARER_PREFIX):
        raise WorkerAuthError("missing bearer credential")
    credential = authorization.removeprefix(BEARER_PREFIX)
    if not credential:
        raise WorkerAuthError("missing bearer credential")
    return credential


class StaticTokenWorkerAuthenticator:
    """Compares the sha256 of a shared token against enrolled digests."""

    def __init__(self, tokens: Mapping[str, str]) -> None:
        self._tokens = dict(tokens)

    @classmethod
    def from_env_value(cls, value: str) -> "StaticTokenWorkerAuthenticator":
        tokens: dict[str, str] = {}
        for entry in value.split(","):
            worker_id, separator, digest = entry.strip().partition(":")
            if not worker_id and not separator and not digest:
                continue
            if not worker_id or SHA256_HEX_PATTERN.match(digest) is None:
                raise ValueError("worker tokens must be worker-id:<sha256 hex> pairs")
            tokens[worker_id] = digest
        if not tokens:
            raise ValueError("worker tokens must enroll at least one worker")
        return cls(tokens)

    async def authenticate(self, authorization: str | None) -> WorkerIdentity:
        digest = hashlib.sha256(_bearer_credential(authorization).encode()).hexdigest()
        matched: str | None = None
        for worker_id, expected in self._tokens.items():
            if hmac.compare_digest(digest, expected):
                matched = worker_id
        if matched is None:
            raise WorkerAuthError("unknown worker credential")
        return WorkerIdentity(
            worker_id=matched,
            subject=f"{STATIC_AUTH_METHOD}:{matched}",
            method=STATIC_AUTH_METHOD,
        )


def _verify_google_id_token(token: str, audience: str) -> Mapping[str, object]:
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    transport = Request()

    def bounded_request(url: str, method: str = "GET", **kwargs: object):
        kwargs["timeout"] = 5
        return transport(url, method=method, **kwargs)

    return id_token.verify_oauth2_token(token, bounded_request, audience)


class GoogleIdTokenWorkerAuthenticator:
    """Resolves a Google-issued identity token to an enrolled worker row."""

    def __init__(
        self,
        audience: str,
        registry: WorkerRegistry,
        verifier: Callable[[str, str], Mapping[str, object]] | None = None,
        *,
        auto_enroll: bool = False,
        account_prefix: str = DEFAULT_WORKER_ACCOUNT_PREFIX,
        enrollments: Mapping[str, WorkerEnrollment] | None = None,
        max_concurrent_verifications: int = 8,
    ) -> None:
        """Resolve verified Google identities to workers.

        With auto_enroll off, a subject must already have a registry row. With
        it on, only an exact full identity in the server's enrollment allowlist
        may join; the allowlist, never the email's local part, chooses its id.
        External IAM admission remains required defense in depth.
        """
        self._audience = audience
        self._registry = registry
        self._verifier = verifier if verifier is not None else _verify_google_id_token
        self._auto_enroll = auto_enroll
        self._account_prefix = account_prefix
        self._enrollments = validate_enrollments(enrollments)
        if auto_enroll and not self._enrollments:
            raise ValueError("auto-enrollment requires an explicit enrollment allowlist")
        if max_concurrent_verifications < 1:
            raise ValueError("verification concurrency must be positive")
        self._verification_slots = asyncio.Semaphore(max_concurrent_verifications)
        self._verification_tasks: set[asyncio.Task] = set()

    async def authenticate(self, authorization: str | None) -> WorkerIdentity:
        credential = _bearer_credential(authorization)
        if self._verification_slots.locked():
            raise WorkerAuthBusy("identity verification is busy")
        await self._verification_slots.acquire()
        task = asyncio.create_task(asyncio.to_thread(self._verifier, credential, self._audience))
        self._verification_tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self._verification_tasks.discard(done)
            self._verification_slots.release()
            if not done.cancelled():
                done.exception()  # Retrieve exceptions even if the HTTP caller left.

        task.add_done_callback(finished)
        try:
            # Cancellation must not free capacity while a verifier thread still runs.
            claims = await asyncio.shield(task)
        except Exception as exc:
            # The verifier is foreign code reaching a foreign issuer; every way it
            # can fail means the same thing here, and none of them may leak out.
            raise WorkerAuthError("identity token verification failed") from exc
        if claims.get("email_verified") is not True:
            raise WorkerAuthError("identity token email is not verified")
        email = claims.get("email")
        if not isinstance(email, str) or not email:
            raise WorkerAuthError("identity token carries no email claim")
        record = await self._registry.find_by_identity_subject(email)
        if record is not None:
            worker_id = record.worker_id
        elif self._auto_enroll:
            worker_id_from_account(email, self._account_prefix)  # syntax, not admission
            worker_id = next(
                (worker_id for worker_id, enrollment in self._enrollments.items()
                 if enrollment.identity_subject == email),
                None,
            )
            if worker_id is None:
                raise WorkerAuthError("identity subject is not approved for enrollment")
            existing = await self._registry.get(worker_id)
            if existing is not None and existing.identity_subject != email:
                raise WorkerAuthError("worker identity binding does not match")
        else:
            raise WorkerAuthError("identity subject is not enrolled")
        return WorkerIdentity(
            worker_id=worker_id,
            subject=email,
            method=GOOGLE_OIDC_AUTH_METHOD,
        )


def worker_id_from_account(email: str, prefix: str = DEFAULT_WORKER_ACCOUNT_PREFIX) -> str:
    """``gpu-worker-<id>@project.iam.gserviceaccount.com`` → ``<id>``.

    Rejects anything that does not look like a per-machine service account so a
    human or unrelated identity can never be admitted as a worker by accident.
    """
    local_part, separator, domain = email.partition("@")
    if (
        not separator
        or not local_part.startswith(prefix)
        or re.fullmatch(r"[a-z][a-z0-9-]*\.iam\.gserviceaccount\.com", domain) is None
    ):
        raise WorkerAuthError("identity subject is not a worker account")
    worker_id = local_part.removeprefix(prefix)
    if WORKER_ID_PATTERN.fullmatch(worker_id) is None:
        raise WorkerAuthError("identity subject does not name a valid worker id")
    return worker_id
