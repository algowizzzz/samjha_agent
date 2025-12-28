# Local Setup Guide - Database & Redis

## Quick Start (SQLite + Redis)

**Easiest option** - SQLite doesn't require installation:

```bash
# Run setup script
./external/ai_bulk_doc_analysis/setup_local_sqlite.sh

# Load environment variables
source .env.local

# Start workers (separate terminal)
python external/ai_bulk_doc_analysis/workers/run_workers.py

# Start server
python run_server.py
```

## Full Setup (PostgreSQL + Redis)

**Production-like** - requires PostgreSQL installation:

```bash
# Run setup script
./external/ai_bulk_doc_analysis/setup_local_db.sh

# Load environment variables
source .env.local

# Start workers (separate terminal)
python external/ai_bulk_doc_analysis/workers/run_workers.py

# Start server
python run_server.py
```

## Manual Setup

### 1. Install Redis (Required for queues)

```bash
# macOS
brew install redis
brew services start redis

# Verify
redis-cli ping  # Should return "PONG"
```

### 2. Choose Database Option

#### Option A: SQLite (No installation needed)

Create `.env.local`:
```bash
DATABASE_URL=sqlite:///$(pwd)/data/ai_bulk_doc_analysis/db/bulk_doc.db
REDIS_URL=redis://localhost:6379/0
```

#### Option B: PostgreSQL

Install:
```bash
brew install postgresql@15
brew services start postgresql@15
```

Create database:
```bash
createdb bulk_doc_analysis
psql bulk_doc_analysis -f external/ai_bulk_doc_analysis/db_schema.sql
```

Create `.env.local`:
```bash
DATABASE_URL=postgresql://$(whoami)@localhost:5432/bulk_doc_analysis
REDIS_URL=redis://localhost:6379/0
```

### 3. Load Environment Variables

```bash
source .env.local
```

Or add to your shell profile (`~/.zshrc` or `~/.bashrc`):
```bash
export $(cat .env.local | xargs)
```

### 4. Start Workers (Required for async processing)

In a separate terminal:
```bash
cd /Users/saadahmed/Desktop/samjha_agent-1
source .venv/bin/activate
source .env.local  # Load env vars
python external/ai_bulk_doc_analysis/workers/run_workers.py
```

### 5. Start Server

```bash
python run_server.py
```

## Verification

1. **Check Redis**: `redis-cli ping` → Should return "PONG"
2. **Check Database**: Tables should be created automatically on first use
3. **Check Workers**: Should see "Starting RQ worker..." message

## Troubleshooting

- **Redis not running**: `brew services start redis`
- **Database connection error**: Check `DATABASE_URL` in `.env.local`
- **Workers not processing**: Ensure workers are running and Redis is accessible

