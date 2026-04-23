# Assets

This directory holds all slide assets, organised into three subdirectories:

- `generated/` — files produced by CI (videos, images, SVGs). Not committed; recreated on every build.
- `stored/` — permanently stored files (images, logos, placeholders) that are committed to the repo.
- `sources/` — source files organised by type, used to generate assets:
  - `sources/teachgrav/scenarios.yml` — scenario definitions for `generate-figures`
  - `sources/graphviz/` — Graphviz DOT files and `render_diagrams.sh`
  - `sources/ucl-logo-conversion.py` — script to regenerate `stored/ucl-logo.png` from `stored/ucl-logo-original.png`

## Generating assets

See `../README.md` for instructions.
