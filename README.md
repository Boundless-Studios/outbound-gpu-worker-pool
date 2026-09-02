# outbound-gpu-worker-pool

Outbound-only, pull-based GPU worker pool. Workers on trusted machines open no ports and hold
no bucket or database credentials; they poll an authenticated coordinator, lease one job,
download inputs through job-scoped signed URLs, run an approved plugin, upload the output
through a create-once grant, and commit an attested result that the coordinator verifies
before the job is marked complete.

Status: alpha, slice 1 (durable capability leases, coordinator API, deterministic reference
plugin and agent). See [`docs/design.md`](docs/design.md) for the architecture, trust
boundaries, failure semantics, and rollout plan.

Library extraction in progress; see [`docs/plans/2026-09-02-extract-library.md`](docs/plans/2026-09-02-extract-library.md).
