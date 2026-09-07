-- Add-only and replay-safe: nothing here drops a column a later start recreates.
ALTER TABLE pool_workers
    ADD COLUMN IF NOT EXISTS tenant_id text;

CREATE INDEX IF NOT EXISTS pool_workers_tenant_idx
    ON pool_workers (tenant_id, status);

ALTER TABLE pool_jobs
    ADD COLUMN IF NOT EXISTS requirements jsonb NOT NULL DEFAULT '{}'::jsonb
        CONSTRAINT pool_jobs_requirements_object
        CHECK (jsonb_typeof(requirements) = 'object');
