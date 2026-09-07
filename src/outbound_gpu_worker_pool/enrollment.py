"""Server-controlled identity/tenant admission, never worker-supplied policy."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from outbound_gpu_worker_pool.contracts import MAX_TENANT_ID_LENGTH

WORKER_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,127}")


@dataclass(frozen=True)
class WorkerEnrollment:
    """An operator-approved full identity and tenant (None explicitly means house)."""

    identity_subject: str
    tenant_id: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identity_subject, str)
            or not 1 <= len(self.identity_subject) <= 255
            or self.identity_subject != self.identity_subject.strip()
        ):
            raise ValueError("enrollment needs a nonempty, bounded identity subject")
        if self.tenant_id is not None and (
            not isinstance(self.tenant_id, str)
            or not 1 <= len(self.tenant_id) <= MAX_TENANT_ID_LENGTH
            or self.tenant_id != self.tenant_id.strip()
        ):
            raise ValueError("enrollment tenant must be a bounded string or null")


def validate_enrollments(
    enrollments: Mapping[str, WorkerEnrollment] | None,
) -> dict[str, WorkerEnrollment]:
    result = dict(enrollments or {})
    subjects: set[str] = set()
    for worker_id, enrollment in result.items():
        if not isinstance(worker_id, str) or not WORKER_ID_PATTERN.fullmatch(worker_id):
            raise ValueError("invalid enrolled worker id")
        if not isinstance(enrollment, WorkerEnrollment):
            raise ValueError("expected WorkerEnrollment values")
        if enrollment.identity_subject in subjects:
            raise ValueError("an identity subject cannot enroll two workers")
        subjects.add(enrollment.identity_subject)
    return result


def enrollments_from_json(value: str | None) -> dict[str, WorkerEnrollment]:
    """Parse OGWP_WORKER_ENROLLMENTS; an explicit tenant key is mandatory."""
    if not value:
        return {}

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate enrollment configuration key")
            result[key] = item
        return result

    try:
        data = json.loads(value, object_pairs_hook=unique_object)
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        enrollments = {}
        for worker_id, entry in data.items():
            if not isinstance(entry, dict) or set(entry) != {"identity_subject", "tenant_id"}:
                raise ValueError("expected identity_subject and explicit tenant_id")
            enrollments[worker_id] = WorkerEnrollment(**entry)
        return validate_enrollments(enrollments)
    except (TypeError, ValueError) as exc:
        # Configuration values may contain identifiers: report no raw input.
        raise ValueError("invalid OGWP_WORKER_ENROLLMENTS configuration") from exc
