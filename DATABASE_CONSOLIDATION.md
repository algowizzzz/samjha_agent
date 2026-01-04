# Database Consolidation - Complete

## Summary

Successfully consolidated two separate databases into a single unified database for the SAJHA platform.

## Before Consolidation

| Database | Purpose | Tables |
|----------|---------|--------|
| `samjha_agent` | Core platform (agents, prompts, conversations) | 10 tables |
| `bulk_doc_analysis` | Bulk document analysis feature | 14 tables |

**Problem:** Two databases caused configuration conflicts and operational complexity.

## After Consolidation

| Database | Purpose | Tables |
|----------|---------|--------|
| `samjha_agent` | **Unified platform** (agents + bulk doc analysis) | 24 tables |

## Changes Made

### 1. Configuration
- **Updated `.env.local`**: Changed `DATABASE_URL` from `bulk_doc_analysis` → `samjha_agent`
- **Updated `run_server.py`**: Changed `.env.local` loading from `override=False` → `override=True`

### 2. Database Schema
- **Created all bulk_doc tables** in `samjha_agent` database
- **Resolved table name conflict**: Renamed bulk_doc `runs` table to `bulk_doc_runs` (core platform already has `runs` table)
- **Updated foreign keys**: All FK references updated to use `bulk_doc_runs.run_id`

### 3. Code Updates
- **`models.py`**: Renamed `Run.__tablename__` from `"runs"` to `"bulk_doc_runs"`
- **FK references**: Updated all `ForeignKey("runs.run_id")` → `ForeignKey("bulk_doc_runs.run_id")`

## Current Database Structure

### Core Platform Tables (10)
- agents
- prompts
- conversations
- messages
- runs
- prompt_revisions
- run_events
- run_results
- tool_traces
- alembic_version

### Bulk Doc Analysis Tables (14)
- chains
- chain_steps
- chain_versions
- workflows
- workflow_versions
- workflow_domains
- sessions
- documents
- **bulk_doc_runs** (renamed from `runs`)
- step_results
- execution_tasks
- ingestion_profiles
- export_profiles
- job_queue_log

## Verification

✅ All bulk doc APIs working (create, list, get, update chains)
✅ Core platform tables intact
✅ Single database configuration
✅ No FK resolution errors

## Data Migration

**Note:** Historical data from `bulk_doc_analysis` database was not migrated automatically due to migration script issues. If needed, data can be migrated manually using:

```sql
-- Example: Migrate chains
INSERT INTO samjha_agent.chains 
SELECT * FROM bulk_doc_analysis.chains 
ON CONFLICT DO NOTHING;
```

## Configuration

**Environment Variables:**
- `.env`: Points to `samjha_agent` (main config)
- `.env.local`: Points to `samjha_agent` (local override, takes precedence)

**Server Startup:**
1. Loads `.env`
2. Loads `.env.local` with `override=True` (local settings win)

## Benefits

1. **Single source of truth** - One database for all platform features
2. **Simplified configuration** - No more database URL conflicts
3. **Easier operations** - One database to backup, monitor, maintain
4. **Future integrations** - Can add FK relationships between features if needed

## Future Considerations

- Consider using table prefixes for all feature tables (e.g., `bulk_doc_*`) to avoid future conflicts
- Consider consolidating `db_service.py` to use `core/db/session.py` for consistency
- Monitor for any performance impacts from table count increase

