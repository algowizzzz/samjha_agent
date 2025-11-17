#!/bin/bash
# Start SAJHA MCP Server with virtual environment

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Kill any existing servers
pkill -f "python.*run_server" 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Wait a moment
sleep 1

# Start the server
echo "Starting SAJHA MCP Server..."
echo "Access the Document Review UI at: http://localhost:8000/doc-review"
echo ""
python3 run_server.py

