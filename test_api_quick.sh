#!/bin/bash
# Quick API Test Script

API_KEY="docreview_dev_key_12345"
BASE="http://localhost:8000/api/doc_review"

echo "============================================"
echo "Doc Review API Quick Test Report"
echo "============================================"
echo ""

# Test 1: List Documents
echo "[TEST 1] GET /api/doc_review/documents"
curl -s -H "X-API-Key: $API_KEY" "$BASE/documents" | python3 -m json.tool | head -n 30
echo ""

# Test 2: List Templates  
echo "[TEST 2] GET /api/doc_review/templates"
curl -s -H "X-API-Key: $API_KEY" "$BASE/templates" | head -n 20
echo ""

# Test 3: Upload endpoint (check if it exists)
echo "[TEST 3] POST /api/doc_review/upload (Testing availability)"
curl -s -X POST -H "X-API-Key: $API_KEY" "$BASE/upload" | head -n 10
echo ""

# Test 4: VFS Tree (with first document)
echo "[TEST 4] GET /api/doc_review/vfs/tree"
FILE_ID=$(curl -s -H "X-API-Key: $API_KEY" "$BASE/documents" | python3 -c "import json, sys; docs=json.load(sys.stdin); print(docs[0]['file_id'] if docs else '')")
if [ -n "$FILE_ID" ]; then
  echo "Using file_id: $FILE_ID"
  curl -s -H "X-API-Key: $API_KEY" "$BASE/vfs/tree?file_id=$FILE_ID" | python3 -m json.tool | head -n 40
else
  echo "No documents found"
fi
echo ""

echo "============================================"
echo "Test Complete"
echo "============================================"
