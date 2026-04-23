#!/bin/bash

# Render all .dot files in sources/graphviz/ to SVG in generated/
# Run from slideshow/assets/: ./sources/graphviz/render_diagrams.sh

echo "Rendering DOT files to SVG..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p generated

# Find all .dot files in the same directory as this script and convert to SVG
for dotfile in "$SCRIPT_DIR"/*.dot; do
    if [[ -f "$dotfile" ]]; then
        basename=$(basename "$dotfile")
        echo "Processing $basename..."
        # Use dot command to convert to SVG, output to generated/
        dot -Tsvg -o "generated/${basename%.dot}.svg" "$dotfile"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully created generated/${basename%.dot}.svg"
        else
            echo "✗ Failed to create generated/${basename%.dot}.svg"
        fi
    fi
done

echo "Done!"