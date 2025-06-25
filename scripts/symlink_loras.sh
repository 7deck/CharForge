#!/bin/bash

SRC_DIR="./scratch"
DEST_DIR="./ComfyUI/models/loras"

mkdir -p "$DEST_DIR"

for folder in "$SRC_DIR"/*/; do
    folder_name=$(basename "$folder")
    src_file="$folder/char/char.safetensors"
    dest_link="$DEST_DIR/${folder_name}.safetensors"

    if [ -f "$src_file" ]; then
        ln -sf "$(realpath "$src_file")" "$dest_link"
        echo "Linked $src_file -> $dest_link"
    else
        echo "Warning: $src_file does not exist, skipping." 
    fi
done 