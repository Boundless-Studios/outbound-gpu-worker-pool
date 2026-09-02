CREATE TABLE IF NOT EXISTS pool_jobs (
    job_id uuid PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    capability_id text NOT NULL,
    contract_version integer NOT NULL DEFAULT 1
        CONSTRAINT pool_jobs_contract_version_positive CHECK (contract_version > 0),
    status text NOT NULL DEFAULT 'queued'
        CONSTRAINT pool_jobs_status_known
        CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    input_keys jsonb NOT NULL DEFAULT '[]'::jsonb
        CONSTRAINT pool_jobs_input_keys_array CHECK (jsonb_typeof(input_keys) = 'array'),
    output_key text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb
        CONSTRAINT pool_jobs_payload_object CHECK (jsonb_typeof(payload) = 'object'),
    tenant_id text,
    attempts integer NOT NULL DEFAULT 0,
    attempt_budget integer NOT NULL DEFAULT 5
        CONSTRAINT pool_jobs_attempt_budget_range
        CHECK (attempt_budget BETWEEN 1 AND 20),
    priority integer NOT NULL DEFAULT 100
        CONSTRAINT pool_jobs_priority_range CHECK (priority BETWEEN 0 AND 1000),
    execution_deadline_seconds integer NOT NULL DEFAULT 1200
        CONSTRAINT pool_jobs_execution_deadline_range
        CHECK (execution_deadline_seconds BETWEEN 60 AND 7200),
    request_digest text NOT NULL DEFAULT '',
    claim_token text,
    leased_by text,
    lease_until timestamptz,
    output_content_type text,
    output_sha256 text,
    output_byte_length bigint,
    error text,
    failure_code text,
    failure_message text,
    retryable boolean NOT NULL DEFAULT false,
    progress_percent integer
        CONSTRAINT pool_jobs_progress_percent_range
        CHECK (progress_percent IS NULL OR progress_percent BETWEEN 0 AND 100),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    cancelled_at timestamptz
);

CREATE INDEX IF NOT EXISTS pool_jobs_leasable_idx
    ON pool_jobs (capability_id, priority, created_at, job_id)
    WHERE status IN ('queued', 'processing');

CREATE INDEX IF NOT EXISTS pool_jobs_tenant_idx
    ON pool_jobs (tenant_id, created_at, job_id)
    WHERE tenant_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS pool_workers (
    worker_id text PRIMARY KEY,
    identity_subject text NOT NULL,
    status text NOT NULL DEFAULT 'active'
        CONSTRAINT pool_workers_status_known
        CHECK (status IN ('active', 'draining', 'revoked')),
    capabilities jsonb NOT NULL DEFAULT '[]'::jsonb,
    gpu_model text,
    vram_mb integer,
    runtime_versions jsonb NOT NULL DEFAULT '{}'::jsonb,
    cost_class text NOT NULL DEFAULT 'local'
        CONSTRAINT pool_workers_cost_class_known
        CHECK (cost_class IN ('local', 'hosted', 'premium')),
    labels jsonb NOT NULL DEFAULT '[]'::jsonb,
    active_leases integer NOT NULL DEFAULT 0,
    last_heartbeat_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    revoked_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS pool_workers_identity_subject_idx
    ON pool_workers (identity_subject);

CREATE TABLE IF NOT EXISTS pool_audit_events (
    event_id bigserial PRIMARY KEY,
    event_type text NOT NULL,
    worker_id text,
    job_id uuid,
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pool_audit_events_job_idx
    ON pool_audit_events (job_id, event_id)
    WHERE job_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS pool_audit_events_worker_idx
    ON pool_audit_events (worker_id, event_id)
    WHERE worker_id IS NOT NULL;
