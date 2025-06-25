#!/bin/bash
set -e  # Exit on any error

CONFIG_PATH=$1

# Check if config path is provided
if [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 /path/to/config.yaml"
    exit 1
fi

echo "🚀 Setting up AI Toolkit environment..."

# Get absolute path of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get absolute path of the project root directory
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Path to AI Toolkit directory
AI_TOOLKIT_DIR="$PROJECT_ROOT/ai_toolkit"

# Check if HF_TOKEN is available in the environment
if [ -n "$HF_TOKEN" ]; then
    echo "🔑 Using HF_TOKEN from environment"
fi

# Change to toolkit directory
cd "$AI_TOOLKIT_DIR"

# Create and activate virtual environment
echo "🔧 Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install dependencies with uv
echo "📦 Installing dependencies..."
GIT_LFS_SKIP_SMUDGE=1 uv pip install --compile-bytecode -r requirements.txt

# Export the token to the virtual environment
if [ -n "$HF_TOKEN" ]; then
    # This explicitly passes HF_TOKEN to the virtual environment
    export HF_TOKEN="$HF_TOKEN"
    echo "🔐 HF_TOKEN exported to virtual environment"
fi

# Run the toolkit script
echo "🔄 Running AI Toolkit with config: $CONFIG_PATH"
python run.py "$CONFIG_PATH" || { echo "❌ AI Toolkit execution failed"; exit 1; }

# Deactivate virtual environment
deactivate

echo "✨ AI Toolkit execution complete." 