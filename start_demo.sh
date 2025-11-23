#!/bin/bash
# Start server with demo mode enabled

cd "$(dirname "$0")"

# Try to initialize conda if available
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
fi

# Activate conda environment if conda is available
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'sajha'..."
    conda activate sajha
fi

# Set demo mode
export DEMO_MODE=true

echo "Starting server with DEMO_MODE=true..."
echo "Server will be available at http://localhost:8000"
echo ""

# Start the server
python run_server.py

