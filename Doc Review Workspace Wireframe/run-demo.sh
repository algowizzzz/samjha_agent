#!/bin/bash

# Doc Review Demo Launcher
# This script starts both the backend and serves the demo HTML

echo "🚀 Starting Doc Review Demo..."
echo ""

# Check if backend is already running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Backend not running. Please start it first:"
    echo "   python web/app.py"
    echo ""
    read -p "Press Enter once backend is running..."
fi

# Serve the demo HTML
echo "✅ Backend is running!"
echo "🌐 Opening demo at: http://localhost:3001/demo.html"
echo ""
echo "📄 Document: CAR_Chapter_1_First_5_Pages (auto-loaded)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd build
python3 -m http.server 3001

