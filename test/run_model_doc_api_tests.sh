#!/bin/bash
# Script to run Model Documentation API endpoint tests

cd "$(dirname "$0")/.." || exit 1

echo "Model Documentation API Endpoint Tests"
echo "======================================"
echo ""

# Check if server is running
echo "Checking if server is running..."
SERVER_RUNNING=false
for endpoint in "http://localhost:5000/api/health" "http://localhost:5000/api/model_doc/templates" "http://localhost:5000/"; do
    if curl -s "$endpoint" > /dev/null 2>&1; then
        SERVER_RUNNING=true
        break
    fi
done

if [ "$SERVER_RUNNING" = true ]; then
    echo "✅ Server is running"
else
    echo "❌ Server is not running or not accessible"
    echo ""
    echo "Please start the server first:"
    echo "  python3 run_server.py"
    echo "or"
    echo "  python3 -m flask run"
    echo ""
    exit 1
fi

echo ""
echo "Running API endpoint tests..."
echo ""

# Run the test script
python3 test/test_model_doc_api_endpoints.py

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Tests completed"
else
    echo "❌ Tests failed with exit code $exit_code"
fi

exit $exit_code

