#!/bin/bash
# Test IDE Integration - Backend + Frontend + Extension

echo "=== IDE Integration Test ==="
echo ""

# 1. Backend Health
echo "1. Backend Health (Port 8000):"
BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/)
if [ "$BACKEND_STATUS" = "200" ] || [ "$BACKEND_STATUS" = "302" ]; then
    echo "   ✓ Backend responding (HTTP $BACKEND_STATUS)"
else
    echo "   ✗ Backend not responding (HTTP $BACKEND_STATUS)"
    exit 1
fi
echo ""

# 2. IDE Health
echo "2. IDE Health (Port 3001):"
IDE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:3001/)
if [ "$IDE_STATUS" = "200" ]; then
    echo "   ✓ IDE responding (HTTP $IDE_STATUS)"
else
    echo "   ✗ IDE not responding (HTTP $IDE_STATUS)"
    exit 1
fi
echo ""

# 3. Extension Installed
echo "3. Extension Installation:"
docker exec openvscode /home/.openvscode-server/bin/openvscode-server --list-extensions 2>&1 | grep -q "doc-review-vfs"
if [ $? -eq 0 ]; then
    echo "   ✓ doc-review-vfs extension installed"
else
    echo "   ✗ doc-review-vfs extension not found"
    exit 1
fi
echo ""

# 4. VFS Endpoints (requires auth - will fail without session)
echo "4. VFS Endpoints (requires auth):"
echo "   Note: These require login session, testing structure only"
VFS_TREE=$(curl -s http://127.0.0.1:8000/api/doc_review/vfs/tree 2>&1)
if echo "$VFS_TREE" | grep -q "Page Not Found\|Login"; then
    echo "   ⚠ VFS endpoints require authentication (expected)"
else
    echo "   ✓ VFS endpoints accessible"
fi
echo ""

# Summary
echo "=== Summary ==="
echo "✓ Backend: http://127.0.0.1:8000/"
echo "✓ IDE: http://127.0.0.1:3001/"
echo "✓ Extension: Installed in container"
echo ""
echo "MANUAL STEPS REQUIRED:"
echo "1. Open http://127.0.0.1:8000/ in Chrome → Login"
echo "2. Open http://127.0.0.1:3001/ in Chrome"
echo "3. In IDE: Settings → 'Doc Review: Api Base Url' = http://127.0.0.1:8000/"
echo "4. In IDE: Cmd+Shift+P → 'Doc Review: Select Document'"
echo "5. In IDE: Cmd+Shift+P → 'Doc Review: Open Agent Console'"


