#!/bin/bash
# Alternative: Setup with SQLite (no installation needed)

set -e

echo "=========================================="
echo "SQLite + Redis Setup (Simpler Alternative)"
echo "=========================================="

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "Redis not found. Installing via Homebrew..."
    brew install redis
    brew services start redis
    echo "✅ Redis installed and started"
else
    echo "✅ Redis found: $(redis-server --version | head -1)"
fi

# Create database directory
mkdir -p data/ai_bulk_doc_analysis/db
DB_PATH="data/ai_bulk_doc_analysis/db/bulk_doc.db"

# Create .env.local file
ENV_FILE=".env.local"
PROJECT_ROOT=$(pwd)

cat > "$ENV_FILE" << EOF
# Local Development Environment Variables (SQLite)
# AI Bulk Doc Analysis

# SQLite Database (file-based, no server needed)
DATABASE_URL=sqlite:///${PROJECT_ROOT}/${DB_PATH}

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage Base Path
STORAGE_BASE=${PROJECT_ROOT}/data/ai_bulk_doc_analysis
EOF

echo "✅ Created $ENV_FILE"
echo ""
echo "Configuration:"
echo "  DATABASE_URL=sqlite:///${PROJECT_ROOT}/${DB_PATH}"
echo "  REDIS_URL=redis://localhost:6379/0"
echo ""
echo "Note: SQLite doesn't need a server. Database will be created automatically."
echo ""

