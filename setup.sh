#!/bin/bash
set -e  # Exit on any error

echo "🚀 Starting setup process..."

# Set cache directories in workspace
export PIP_CACHE_DIR="$(pwd)/.cache/pip"
export UV_CACHE_DIR="$(pwd)/.cache/uv"
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Install minimal system dependencies needed for uv
echo "📦 Installing minimal dependencies..."
apt-get update && apt-get install -y curl

# Install uv
echo "🔧 Installing uv package installer..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Source uv environment to update PATH
source $HOME/.local/bin/env

# Create and activate virtual environment
echo "🔧 Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install packages
echo "📦 Installing Python packages..."
uv pip install --upgrade pip
uv pip install -r base_requirements.txt

# Run the main Python install script
echo "📦 Running main installation script..."
python install.py

echo "✨ Setup complete!"
