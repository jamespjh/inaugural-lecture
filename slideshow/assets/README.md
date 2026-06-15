# Assets Directory Structure

This directory contains assets organized into three categories:

## Sources
- `sources/teachgrav/scenarios.yml` - Source configuration for teaching scenarios
- `sources/graphviz/` - Source Graphviz DOT files
- `sources/prompts/` - Source prompt files

## Generated
- Automatically generated files (SVGs, videos, etc.)

## Stored
- Permanently stored static files (images, etc.)

## DOT File Features

Each DOT file contains:
- Layout engine specification using the `layout` attribute
- Styling information embedded directly in the file
- Clear node and edge definitions

The layout engine is specified in the graph attributes:
- `layout="neato"` for spring-based layouts
- `layout="dot"` for hierarchical layouts