#!/bin/bash

# Simple script to render all .dot files in the current directory to SVG
# Usage: ./render_diagrams.sh

echo "Rendering all .dot files to SVG..."

# Find all .dot files and convert them to SVG
for dotfile in *.dot; do
    if [[ -f "$dotfile" ]]; then
        echo "Processing $dotfile..."
        # Use dot command to convert to SVG
        dot -Tsvg -o "${dotfile%.dot}.svg" "$dotfile"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully created ${dotfile%.dot}.svg"
        else
            echo "✗ Failed to create ${dotfile%.dot}.svg"
        fi
    fi
done

echo "Done!"