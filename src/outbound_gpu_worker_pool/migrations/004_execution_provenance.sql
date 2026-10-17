ALTER TABLE pool_jobs ADD COLUMN IF NOT EXISTS executed_by text;
ALTER TABLE pool_jobs ADD COLUMN IF NOT EXISTS execution_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;
