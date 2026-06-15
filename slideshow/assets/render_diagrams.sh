#!/bin/bash

# Simple script to render all .dot files in the sources/graphviz directory to SVG
# Usage: ./render_diagrams.sh

echo "Rendering all .dot files to SVG..."

# Change to the sources/graphviz directory
cd sources/graphviz || exit 1

# Find all .dot files and convert them to SVG
for dotfile in *.dot; do
    if [[ -f "$dotfile" ]]; then
        echo "Processing $dotfile..."
        # Use dot command to convert to SVG
        dot -Tsvg -o "../../generated/${dotfile%.dot}.svg" "$dotfile"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully created ../../generated/${dotfile%.dot}.svg"
        else
            echo "✗ Failed to create ../../generated/${dotfile%.dot}.svg"
        fi
    fi
done

echo "Done!"