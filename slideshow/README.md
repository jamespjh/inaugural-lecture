# Slideshow

```bash
quarto check
quarto render slides.qmd --to revealjs
```

## UCL Logo Conversion

Run from repo root to regenerate `assets/ucl-logo.png` from `assets/ucl-logo-original.png`:

```bash
set -e
cd /Users/jamespjh/devel/inaugural/slideshow/assets
source /Users/jamespjh/devel/inaugural/demo/venv/bin/activate
python ucl-logo-conversion.py --input ucl-logo-original.png --output ucl-logo.png
```
