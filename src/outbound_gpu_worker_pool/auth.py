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
    WorkerIdentity,
    WorkerRegistry,
)

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

    return id_token.verify_oauth2_token(token, Request(), audience)


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
    ) -> None:
        """Resolve verified Google identities to workers.

        With ``auto_enroll`` off (the default) a subject must already have a
        registry row. With it on, an unknown subject is admitted and its worker id
        is derived from the account name (``gpu-worker-<id>@...`` → ``<id>``); the
        first heartbeat then creates the row. Turn it on only when something in
        front of the coordinator already decides who may call it at all — for
        example Cloud Run IAM granting ``run.invoker`` per machine — so that the
        IAM grant is the admission decision and the registry tracks state.
        """
        self._audience = audience
        self._registry = registry
        self._verifier = verifier if verifier is not None else _verify_google_id_token
        self._auto_enroll = auto_enroll
        self._account_prefix = account_prefix

    async def authenticate(self, authorization: str | None) -> WorkerIdentity:
        credential = _bearer_credential(authorization)
        try:
            claims = await asyncio.to_thread(self._verifier, credential, self._audience)
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
            worker_id = worker_id_from_account(email, self._account_prefix)
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
    local_part, separator, _domain = email.partition("@")
    if not separator or not local_part.startswith(prefix):
        raise WorkerAuthError("identity subject is not a worker account")
    worker_id = local_part.removeprefix(prefix)
    if WORKER_ID_PATTERN.match(worker_id) is None:
        raise WorkerAuthError("identity subject does not name a valid worker id")
    return worker_id
