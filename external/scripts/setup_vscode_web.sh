#!/usr/bin/env bash
set -euo pipefail

# Use code-server as VS Code Web alternative (self-hosted VS Code)
USE_CODE_SERVER="${USE_CODE_SERVER:-true}"
CODE_SERVER_VERSION="${CODE_SERVER_VERSION:-latest}"
DEST_DIR="web/static/vscode-web"
ARCHIVE="web/static/vscode-web.tar.gz"

echo "► Preparing VS Code Web assets using code-server"

if [ "$USE_CODE_SERVER" = "true" ]; then
    # Get latest code-server release
    if [ "$CODE_SERVER_VERSION" = "latest" ]; then
        LATEST_RELEASE=$(curl -sL "https://api.github.com/repos/coder/code-server/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
        CODE_SERVER_VERSION="${LATEST_RELEASE#v}"  # Remove 'v' prefix if present
    fi
    
    # Detect platform
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64|amd64) PLATFORM="linux-amd64" ;;
        arm64|aarch64) PLATFORM="linux-arm64" ;;
        *) PLATFORM="linux-amd64" ;;  # Default fallback
    fi
    
    DOWNLOAD_URL="https://github.com/coder/code-server/releases/download/v${CODE_SERVER_VERSION}/code-server-${CODE_SERVER_VERSION}-${PLATFORM}.tar.gz"
    
    echo "► Downloading code-server v${CODE_SERVER_VERSION} (${PLATFORM}) from GitHub"
    echo "  URL: ${DOWNLOAD_URL}"
    
    mkdir -p "$(dirname "${ARCHIVE}")"
    curl -L "${DOWNLOAD_URL}" -o "${ARCHIVE}" || {
        echo "❌ Download failed. Trying alternative method..."
        exit 1
    }
    
    rm -rf "${DEST_DIR}"
    mkdir -p "${DEST_DIR}"
    echo "► Extracting to ${DEST_DIR}"
    tar -xzf "${ARCHIVE}" -C "${DEST_DIR}" --strip-components=1 || {
        echo "❌ Extraction failed. Archive may be corrupted."
        echo "   Keeping archive at ${ARCHIVE} for inspection."
        exit 1
    }
    
    # code-server structure: extract the lib/code-server directory contents
    if [ -d "${DEST_DIR}/lib/code-server" ]; then
        echo "► Restructuring code-server layout..."
        mv "${DEST_DIR}/lib/code-server"/* "${DEST_DIR}/" 2>/dev/null || true
        rm -rf "${DEST_DIR}/lib" "${DEST_DIR}/bin" 2>/dev/null || true
    fi
    
    # Create a minimal index.html that loads code-server
    if [ ! -f "${DEST_DIR}/index.html" ]; then
        echo "► Creating index.html wrapper..."
        cat > "${DEST_DIR}/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>VS Code Web - Document Review</title>
    <meta charset="utf-8">
    <style>
        body { margin: 0; padding: 0; overflow: hidden; }
        #vscode-container { width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <div id="vscode-container"></div>
    <script>
        // Redirect to code-server or load VS Code Web
        // This will be replaced with actual VS Code Web loading logic
        document.body.innerHTML = '<div style="padding: 2rem; text-align: center;"><h1>VS Code Web</h1><p>Loading...</p></div>';
    </script>
</body>
</html>
EOF
    fi
    
    rm -f "${ARCHIVE}"
    echo "✅ code-server assets ready at ${DEST_DIR}"
    echo "⚠️  Note: code-server requires additional configuration to work as VS Code Web."
    echo "   Consider building VS Code Web from source for full compatibility."
    echo "   Launch via http://localhost:8000/doc-review/ide after restarting the server."
else
    echo "❌ Direct VS Code Web download not available."
    echo "   Microsoft doesn't provide pre-built VS Code Web bundles."
    echo "   Options:"
    echo "   1. Use code-server (set USE_CODE_SERVER=true)"
    echo "   2. Build VS Code Web from source: https://github.com/microsoft/vscode"
    exit 1
fi

