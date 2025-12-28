#!/bin/bash
# Setup script for local PostgreSQL and Redis for AI Bulk Doc Analysis

set -e

echo "=========================================="
echo "Local Database & Redis Setup"
echo "=========================================="

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL not found. Installing via Homebrew..."
    brew install postgresql@15
    brew services start postgresql@15
    echo "✅ PostgreSQL installed and started"
else
    echo "✅ PostgreSQL found: $(psql --version)"
fi

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "Redis not found. Installing via Homebrew..."
    brew install redis
    brew services start redis
    echo "✅ Redis installed and started"
else
    echo "✅ Redis found: $(redis-server --version | head -1)"
fi

# Create database
DB_NAME="bulk_doc_analysis"
DB_USER="${USER}"
DB_PASSWORD=""

echo ""
echo "Creating database: $DB_NAME"

# Try to create database (ignore if exists)
psql postgres -c "CREATE DATABASE $DB_NAME;" 2>/dev/null || echo "Database already exists or using different connection"

# Get PostgreSQL port (default 5432)
PG_PORT=5432
if command -v psql &> /dev/null; then
    PG_PORT=$(psql -t -c "SHOW port;" 2>/dev/null | tr -d ' ' || echo "5432")
fi

# Create .env.local file
ENV_FILE=".env.local"
PROJECT_ROOT=$(pwd)

cat > "$ENV_FILE" << EOF
# Local Development Environment Variables
# AI Bulk Doc Analysis

# PostgreSQL Database
DATABASE_URL=postgresql://${DB_USER}@localhost:${PG_PORT}/${DB_NAME}

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage Base Path (relative to project root)
STORAGE_BASE=${PROJECT_ROOT}/data/ai_bulk_doc_analysis
EOF

echo ""
echo "✅ Created $ENV_FILE"
echo ""
echo "Configuration:"
echo "  DATABASE_URL=postgresql://${DB_USER}@localhost:${PG_PORT}/${DB_NAME}"
echo "  REDIS_URL=redis://localhost:6379/0"
echo ""

# Initialize database schema
SCHEMA_FILE="external/ai_bulk_doc_analysis/db_schema.sql"
if [ -f "$SCHEMA_FILE" ]; then
    echo "Initializing database schema..."
    psql -d "$DB_NAME" -f "$SCHEMA_FILE" 2>/dev/null || echo "Schema already initialized or manual setup needed"
    echo "✅ Database schema initialized"
else
    echo "⚠️  Schema file not found: $SCHEMA_FILE"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Load environment variables:"
echo "   source $ENV_FILE"
echo ""
echo "2. Start workers (in separate terminal):"
echo "   python external/ai_bulk_doc_analysis/workers/run_workers.py"
echo ""
echo "3. Start Flask server:"
echo "   python run_server.py"
echo ""

