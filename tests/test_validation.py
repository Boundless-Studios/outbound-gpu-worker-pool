"""Payload, capability-id, bound, and request-digest validation.

These are pure-function tests: nothing here touches a store, a database, or a
network. They pin the canonical form of `job_request_digest` because a worker
must be able to recompute it byte-for-byte from the lease grant it receives.
"""

from dataclasses import replace

import pytest

from outbound_gpu_worker_pool import (
    DETERMINISTIC_ECHO_CAPABILITY,
    MAX_JOB_PAYLOAD_BYTES,
    MAX_JOB_PAYLOAD_DEPTH,
    JobSubmission,
    job_request_digest,
    validate_capability_id,
    validate_job_payload,
    validate_job_submission,
)


def _submission(**overrides: object) -> JobSubmission:
    base = JobSubmission(
        job_id="11111111-1111-4111-8111-111111111111",
        idempotency_key="pool:validation",
        capability_id=DETERMINISTIC_ECHO_CAPABILITY,
        input_keys=("inputs/pool/a.bin", "inputs/pool/b.bin"),
        output_key="outputs/pool/a.txt",
        payload={"seed": 7, "label": "a"},
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def test_capability_id_pattern_accepts_versioned_namespaces() -> None:
    for capability_id in (
        DETERMINISTIC_ECHO_CAPABILITY,
        "pool.v1",
        "vendor.image_to_3d.v2",
        "comfy.workflow.approved-id.v11",
    ):
        assert validate_capability_id(capability_id) == capability_id


def test_capability_id_pattern_rejects_unversioned_or_unsafe_ids() -> None:
    for capability_id in (
        "",
        "Not A Capability",
        "pool.thing",
        "pool.thing.v",
        ".pool.v1",
        "1pool.v1",
        "pool.thing.v1 ",
        "a" * 129 + ".v1",
    ):
        with pytest.raises(ValueError, match="capability_id"):
            validate_capability_id(capability_id)


def test_payload_accepts_a_finite_json_tree() -> None:
    validate_job_payload({})
    validate_job_payload(
        {
            "seed": 7,
            "scale": 1.5,
            "loop": True,
            "absent": None,
            "tags": ["a", "b"],
            "nested": {"depth": {"here": 1}},
        }
    )


def test_payload_rejects_non_finite_numbers() -> None:
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            validate_job_payload({"scale": value})


def test_payload_rejects_unsupported_values_and_non_string_keys() -> None:
    with pytest.raises(ValueError, match="unsupported value"):
        validate_job_payload({"when": object()})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="keys must be strings"):
        validate_job_payload({7: "seven"})  # type: ignore[dict-item]


def test_payload_rejects_trees_deeper_than_the_limit() -> None:
    deep: dict[str, object] = {"leaf": 1}
    for _ in range(MAX_JOB_PAYLOAD_DEPTH):
        deep = {"child": deep}

    with pytest.raises(ValueError, match=f"depth {MAX_JOB_PAYLOAD_DEPTH}"):
        validate_job_payload(deep)  # type: ignore[arg-type]


def test_payload_rejects_repeated_or_cyclic_containers() -> None:
    shared: dict[str, object] = {"a": 1}
    with pytest.raises(ValueError, match="repeated or cyclic"):
        validate_job_payload({"left": shared, "right": shared})  # type: ignore[arg-type]

    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="repeated or cyclic"):
        validate_job_payload(cyclic)  # type: ignore[arg-type]


def test_payload_rejects_oversized_encodings() -> None:
    with pytest.raises(ValueError, match=f"{MAX_JOB_PAYLOAD_BYTES} bytes"):
        validate_job_payload({"blob": "x" * (MAX_JOB_PAYLOAD_BYTES + 1)})


def test_submission_accepts_the_default_bounds() -> None:
    validate_job_submission(_submission())


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("capability_id", {"capability_id": "Not A Capability"}),
        ("capability_id", {"capability_id": "pool.thing"}),
        ("priority", {"priority": -1}),
        ("priority", {"priority": 1001}),
        ("attempt_budget", {"attempt_budget": 0}),
        ("attempt_budget", {"attempt_budget": 21}),
        ("execution_deadline_seconds", {"execution_deadline_seconds": 59}),
        ("execution_deadline_seconds", {"execution_deadline_seconds": 7201}),
        ("contract_version", {"contract_version": 0}),
        ("idempotency_key", {"idempotency_key": ""}),
        ("idempotency_key", {"idempotency_key": "k" * 256}),
        ("output_key", {"output_key": ""}),
        ("output_key", {"output_key": "o" * 1025}),
        ("tenant_id", {"tenant_id": ""}),
        ("input_keys", {"input_keys": ()}),
        ("input_keys", {"input_keys": ("",)}),
        ("input_keys", {"input_keys": tuple(f"inputs/{n}" for n in range(17))}),
    ],
)
def test_submission_rejects_out_of_bound_fields(
    label: str, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match=label):
        validate_job_submission(_submission(**overrides))


def test_submission_rejects_an_invalid_payload() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_job_submission(_submission(payload={"scale": float("inf")}))


def test_digest_is_stable_for_the_same_request() -> None:
    digest = job_request_digest(_submission())

    assert len(digest) == 64
    assert digest == job_request_digest(_submission())
    # A separate but equal payload object hashes identically.
    assert digest == job_request_digest(
        _submission(payload={"label": "a", "seed": 7})
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"capability_id": "pool.other.v1"},
        {"contract_version": 2},
        {"input_keys": ("inputs/pool/a.bin",)},
        {"output_key": "outputs/pool/b.txt"},
        {"payload": {"seed": 8, "label": "a"}},
        {"payload": {}},
    ],
)
def test_digest_changes_when_a_hashed_field_changes(
    overrides: dict[str, object],
) -> None:
    assert job_request_digest(_submission(**overrides)) != job_request_digest(
        _submission()
    )


def test_digest_depends_on_input_key_order() -> None:
    forward = _submission(input_keys=("inputs/pool/a.bin", "inputs/pool/b.bin"))
    reversed_keys = _submission(input_keys=("inputs/pool/b.bin", "inputs/pool/a.bin"))

    assert job_request_digest(forward) != job_request_digest(reversed_keys)


def test_digest_ignores_routing_and_identity_fields() -> None:
    digest = job_request_digest(_submission())

    for overrides in (
        {"job_id": "22222222-2222-4222-8222-222222222222"},
        {"idempotency_key": "pool:other"},
        {"tenant_id": "tenant-1"},
        {"priority": 10},
        {"attempt_budget": 2},
        {"execution_deadline_seconds": 600},
    ):
        assert job_request_digest(_submission(**overrides)) == digest
