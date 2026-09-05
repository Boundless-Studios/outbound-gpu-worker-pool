ALTER TABLE pool_workers
    ADD COLUMN IF NOT EXISTS gpus jsonb NOT NULL DEFAULT '[]'::jsonb
        CONSTRAINT pool_workers_gpus_array CHECK (jsonb_typeof(gpus) = 'array'),
    ADD COLUMN IF NOT EXISTS busy_job_id uuid;
