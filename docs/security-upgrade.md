# Upgrading the coordinator after security fixes #15–19

Apply this release to the coordinator and update host integrations before treating tenant
placement as an authorization boundary for outside workers. No database migration is required;
existing rows remain intact. No deployment is performed by merging these changes.

## Enrollment compatibility

Previously, a new credential could create its registry row with any tenant on its first
heartbeat. Now the worker must have either an existing registry binding or an explicit
server-approved enrollment. Audit the existing `pool_workers` identity/tenant assignments
before upgrading: preserving existing rows does not prove old self-selected memberships were
legitimate. Revoke any incorrect or uncertain binding before admitting untrusted workers.

For new workers, use the trusted `service.enroll_worker(worker_id, WorkerEnrollment(subject,
tenant_id))` call, or supply `OGWP_WORKER_ENROLLMENTS` as a JSON object on the coordinator:

```json
{
  "gpu-01": {
    "identity_subject": "static:gpu-01",
    "tenant_id": "tenant-a"
  }
}
```

`tenant_id: null` explicitly approves house-pool membership. An omitted tenant key is invalid.
Static credentials still come from `OGWP_WORKER_TOKENS`; enrollment metadata contains no token.
Existing approved registry rows need no new environment entry. On the agent, keep the same
`OGWP_WORKER_TENANT`; it must match the approved binding. No agent wire-protocol change is needed.

For Google OIDC, the approved subject is the full verified service-account email. With
`OGWP_WORKER_AUTO_ENROLL=false`, pre-enroll the row before first use. With it true, the coordinator
requires a nonempty `OGWP_WORKER_ENROLLMENTS` allowlist and admits only exact listed
`gpu-worker-…@<project>.iam.gserviceaccount.com` identities. The configured worker ID is
server-chosen; colliding local account names in different projects are not interchangeable.
Keep Cloud Run IAM admission narrowly scoped. A legacy true-without-allowlist deployment now
fails at startup rather than accepting identities by name prefix.

Neither enrollment nor heartbeat silently replaces a subject or tenant. For a new external
identity, enroll a new worker ID, verify it, then retire/revoke the old worker. Static-token
rotation under the same `static:<worker-id>` identity does not require reassignment.

## Status-router compatibility

Replace the old `create_pool_status_router(service, dependencies=[...])` integration with a
mandatory `authorize_admin` dependency. It must return literal `True` after authenticating and
authorizing a pool administrator, or raise the appropriate HTTP error. A generic login dependency
that returns a user object is deliberately insufficient.

```python
async def authorize_pool_admin(user=Depends(current_user)) -> bool:
    return user.is_pool_admin is True

app.include_router(
    create_pool_status_router(service, authorize_admin=authorize_pool_admin)
)
```

These routes are administrator-only, including filtered worker listings and queue aggregates.
For tenant self-service, keep host-owned routes that derive their tenant from the authenticated
principal and call `service.worker_views(tenant_id=...)`; do not mount these global admin views
for ordinary tenants. The optional `tenant_id` query on the admin router is only a filter.

## Admission and availability

Defaults per process are 120 admission attempts/minute per source, 1,200 total, and 16 concurrent
authentication operations. Google verification also has eight nonqueued slots and a five-second
certificate-fetch timeout; cancelling a request does not release its slot while its verifier
thread is still running. Rejected-auth/rate-limit audit writes share a 30/minute sampling budget.
Pre-admission overload is rejected without a database write. Audit rows are therefore samples,
not a complete event count; use ingress metrics for rejection rates.

A saturated peer cannot consume additional global tokens after exhausting its source budget;
other peers can continue while global capacity remains. A distributed flood can exhaust the
global budget, at which point all callers fail closed with 429 until capacity refills. There is
no identity bypass around verification. Source bucket state is bounded, idle entries expire,
and active entries are not evicted to reset budgets.

In standalone configuration, tune `OGWP_PRE_AUTH_LIMIT_PER_MINUTE`,
`OGWP_PER_SOURCE_LIMIT_PER_MINUTE`, `OGWP_AUTH_AUDIT_LIMIT_PER_MINUTE`, and
`OGWP_MAX_AUTH_CONCURRENCY`. Host applications can use the corresponding service constructor
parameters. Machines sharing a NAT/proxy share a source budget unless the ASGI server has a
correctly configured trusted-proxy policy. Never solve that by trusting arbitrary forwarded
headers. Configure external deployment-wide limits as well; replicas do not share these buckets.

## Repository controls

The new `Secrets` workflow checks pull requests, pushes to main, and a weekly scheduled full
scan, with an optional manual dispatch. It fetches branches, tags, and retained PR heads and
uses a checksum-pinned scanner with redacted metadata-only output. Keep this check required in
repository rules. Repository administrators must separately verify native GitHub secret
scanning, push protection, private vulnerability reporting, and prior alert history. Those
settings are not enabled or verified merely by merging this patch.

## Rollout checks

Before promoting a release, run the full test suite against PostgreSQL and confirm the secrets
check is green. Review existing worker bindings, provision any new explicit enrollment entries,
and update status-router consumers. After deploying, confirm an approved worker can heartbeat,
lease, and finish a canary job; a wrong tenant or identity is rejected; non-admin status access
is denied; and ingress source/rate settings match the deployment. Retain revocation and rollback
procedures. Do not restore permissive auto-enrollment as an availability workaround.
