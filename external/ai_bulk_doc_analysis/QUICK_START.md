# Quick Start - Local Setup

## ✅ Already Configured

**Database**: PostgreSQL `bulk_doc_analysis` database created  
**Redis**: Running on `localhost:6379`  
**Environment**: `.env.local` file created with connection strings

## Environment Variables

Your `.env.local` file contains:
```
DATABASE_URL=postgresql://saadahmed@localhost:5432/bulk_doc_analysis
REDIS_URL=redis://localhost:6379/0
```

The server automatically loads this file on startup.

## Start Services

### 1. Start Workers (Required for async processing)

In **Terminal 1**:
```bash
cd /Users/saadahmed/Desktop/samjha_agent-1
source .venv/bin/activate
python external/ai_bulk_doc_analysis/workers/run_workers.py
```

You should see:
```
Starting RQ worker for queues: conversion, execution
Press Ctrl+C to stop
```

### 2. Start Server

In **Terminal 2**:
```bash
cd /Users/saadahmed/Desktop/samjha_agent-1
source .venv/bin/activate
python run_server.py
```

## How It Works

- **Without workers running**: System uses in-memory storage + synchronous execution (works, but slower)
- **With workers running**: PDF conversion and Claude execution happen async via Redis queues

## Verify Setup

1. **Database**: Tables already created (8 tables)
2. **Redis**: `redis-cli ping` → Should return "PONG"
3. **Workers**: Should see "Starting RQ worker..." message
4. **Server**: Access `http://localhost:8000/bulk-doc-analysis`

## Troubleshooting

- **Redis not running**: `brew services restart redis`
- **Database connection error**: Check PostgreSQL is running: `pg_isready`
- **Workers not processing**: Make sure workers are running in separate terminal

