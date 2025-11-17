#!/bin/bash
# Restart SAJHA MCP Server

cd "$(dirname "$0")"

echo "Stopping existing servers..."
pkill -9 -f "python.*run_server" 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting server..."
echo "=========================================="
echo "Server will be available at:"
echo "  http://localhost:8000/doc-review"
echo "=========================================="
echo ""

python3 run_server.py

