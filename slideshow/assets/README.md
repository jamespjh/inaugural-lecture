# Graphviz Diagrams

This directory contains Graphviz DOT files and a script to render them to SVG format.

## Files

- `example1.dot` - Network diagram using neato layout engine
- `example2.dot` - Hierarchical diagram using dot layout engine
- `render_diagrams.sh` - Shell script to convert all .dot files to SVG

## Usage

To render all diagrams:
```bash
chmod +x render_diagrams.sh
./render_diagrams.sh
```

This will generate SVG files for each DOT file in the directory.

## DOT File Features

Each DOT file contains:
- Layout engine specification using the `layout` attribute
- Styling information embedded directly in the file
- Clear node and edge definitions

The layout engine is specified in the graph attributes:
- `layout="neato"` for spring-based layouts
- `layout="dot"` for hierarchical layouts