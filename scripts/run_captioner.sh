#!/bin/bash
set -e  # Exit on any error

# Get absolute path of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get absolute path of the project root directory
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Path to LoRACaptioner directory
CAPTIONER_DIR="$PROJECT_ROOT/LoRACaptioner"

# Parse arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
INCLUDE_REFERENCE_IMAGE="$3"
PARTIAL_CAPTIONS="$4"

# Check if input directory is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Usage: $0 /path/to/input/directory /path/to/output/directory yes|no /path/to/partial_captions.json"
    exit 1
fi

# If output directory is not provided, use input directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$INPUT_DIR"
fi

# Convert to absolute paths
INPUT_DIR="$(cd "$(dirname "$INPUT_DIR")" && pwd)/$(basename "$INPUT_DIR")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"

# Convert partial_captions to absolute path if provided
if [ ! -z "$PARTIAL_CAPTIONS" ]; then
    PARTIAL_CAPTIONS="$(cd "$(dirname "$PARTIAL_CAPTIONS")" && pwd)/$(basename "$PARTIAL_CAPTIONS")"
    echo "📄 Partial captions file: $PARTIAL_CAPTIONS"
fi

echo "🔍 Input directory: $INPUT_DIR"
echo "📝 Output directory: $OUTPUT_DIR"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# Check for TOGETHER_API_KEY
if [ -z "$TOGETHER_API_KEY" ]; then
    echo "🔑 TOGETHER_API_KEY not found in environment"
    
    # Check if it's in .env file at project root
    if [ -f "$PROJECT_ROOT/.env" ]; then
        echo "📄 Found .env file in project root, loading variables"
        source "$PROJECT_ROOT/.env"
    fi
    
    # Check again after loading .env
    if [ -z "$TOGETHER_API_KEY" ]; then
        echo "❌ Error: TOGETHER_API_KEY is required for image captioning"
        exit 1
    fi
    
    echo "✅ TOGETHER_API_KEY loaded from .env file"
else
    echo "✅ TOGETHER_API_KEY found in environment"
fi

echo "🚀 Setting up LoRACaptioner environment..."

# Change to captioner directory
cd "$CAPTIONER_DIR"

# Create and activate virtual environment
echo "🔧 Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install --compile-bytecode -r requirements.txt

# Build the python command
PYTHON_CMD=(python main.py --input "$INPUT_DIR" --output "$OUTPUT_DIR")
if [ ! -z "$PARTIAL_CAPTIONS" ]; then
    PYTHON_CMD+=(--partial_captions "$PARTIAL_CAPTIONS")
fi
if [ "$INCLUDE_REFERENCE_IMAGE" = "yes" ]; then
    PYTHON_CMD+=(--reference_image original.png)
fi

TOGETHER_API_KEY="$TOGETHER_API_KEY" "${PYTHON_CMD[@]}"

# Deactivate virtual environment
deactivate

echo "✨ Image captioning complete."
echo "📂 Captions saved to: $OUTPUT_DIR" 