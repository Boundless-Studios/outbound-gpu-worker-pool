"""Bounds and canonical form for everything a host may submit.

`job_request_digest` is the contract a worker must be able to reproduce: the
coordinator hands the same five fields to the worker in its lease grant, and the
worker attests the digest back on completion.
"""

import hashlib
import json
import math
import re

from outbound_gpu_worker_pool.contracts import (
    MAX_ASSET_KEY_LENGTH,
    MAX_CAPABILITY_ID_LENGTH,
    MAX_EXECUTION_DEADLINE_SECONDS,
    MAX_JOB_ATTEMPT_BUDGET,
    MAX_JOB_ID_LENGTH,
    MAX_JOB_INPUT_KEYS,
    MAX_JOB_PAYLOAD_BYTES,
    MAX_JOB_PAYLOAD_DEPTH,
    MAX_JOB_PRIORITY,
    MIN_EXECUTION_DEADLINE_SECONDS,
    MIN_JOB_ATTEMPT_BUDGET,
    MIN_JOB_PRIORITY,
    JobPayload,
    JobPayloadValue,
    JobSubmission,
)

CAPABILITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_-]+)*\.v[0-9]+$")


def validate_capability_id(capability_id: str) -> str:
    if (
        not capability_id
        or len(capability_id) > MAX_CAPABILITY_ID_LENGTH
        or CAPABILITY_ID_PATTERN.match(capability_id) is None
    ):
        raise ValueError("capability_id must look like namespace.name.v1")
    return capability_id


def job_request_digest(submission: JobSubmission) -> str:
    """Canonical sha256 of the request a worker must hand back verbatim."""
    canonical = json.dumps(
        {
            "capability_id": submission.capability_id,
            "contract_version": submission.contract_version,
            "input_keys": list(submission.input_keys),
            "output_key": submission.output_key,
            "payload": submission.payload,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_job_submission(submission: JobSubmission) -> None:
    for label, value, maximum in (
        ("job_id", submission.job_id, MAX_JOB_ID_LENGTH),
        ("idempotency_key", submission.idempotency_key, MAX_JOB_ID_LENGTH),
        ("tenant_id", submission.tenant_id, MAX_JOB_ID_LENGTH),
        ("output_key", submission.output_key, MAX_ASSET_KEY_LENGTH),
    ):
        if value is not None and (not value or len(value) > maximum):
            raise ValueError(f"{label} must contain 1 to {maximum} characters")
    # A capability may take no inputs at all: text-to-video is prompt-only.
    if len(submission.input_keys) > MAX_JOB_INPUT_KEYS:
        raise ValueError(f"input_keys must hold at most {MAX_JOB_INPUT_KEYS} entries")
    for key in submission.input_keys:
        if not isinstance(key, str) or not key or len(key) > MAX_ASSET_KEY_LENGTH:
            raise ValueError(
                f"input_keys entries must contain 1 to {MAX_ASSET_KEY_LENGTH} characters"
            )
    if len(set(submission.input_keys)) != len(submission.input_keys):
        raise ValueError("input_keys must not repeat a key")
    if submission.contract_version < 1:
        raise ValueError("contract_version must be positive")
    validate_capability_id(submission.capability_id)
    for label, value, minimum, maximum in (
        ("priority", submission.priority, MIN_JOB_PRIORITY, MAX_JOB_PRIORITY),
        (
            "attempt_budget",
            submission.attempt_budget,
            MIN_JOB_ATTEMPT_BUDGET,
            MAX_JOB_ATTEMPT_BUDGET,
        ),
        (
            "execution_deadline_seconds",
            submission.execution_deadline_seconds,
            MIN_EXECUTION_DEADLINE_SECONDS,
            MAX_EXECUTION_DEADLINE_SECONDS,
        ),
    ):
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}")
    validate_job_payload(submission.payload)


def validate_job_payload(payload: JobPayload) -> None:
    _validate_json_tree(payload)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    if len(encoded) > MAX_JOB_PAYLOAD_BYTES:
        raise ValueError(f"job payload exceeds {MAX_JOB_PAYLOAD_BYTES} bytes")


def _validate_json_tree(root: JobPayloadValue) -> None:
    stack: list[tuple[JobPayloadValue, int]] = [(root, 1)]
    seen_containers: set[int] = set()
    while stack:
        value, depth = stack.pop()
        if depth > MAX_JOB_PAYLOAD_DEPTH:
            raise ValueError(f"job payload exceeds depth {MAX_JOB_PAYLOAD_DEPTH}")
        if isinstance(value, dict):
            identity = id(value)
            if identity in seen_containers:
                raise ValueError("job payload contains a repeated or cyclic container")
            seen_containers.add(identity)
            if any(not isinstance(key, str) for key in value):
                raise ValueError("job payload object keys must be strings")
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            identity = id(value)
            if identity in seen_containers:
                raise ValueError("job payload contains a repeated or cyclic container")
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("job payload numbers must be finite")
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("job payload contains an unsupported value")
