#!/bin/bash
# Build the NeurIPS paper.
# Run from paper/ directory: ./build.sh
set -e

# Download official NeurIPS style if not present.
if [ ! -f neurips_2025.sty ]; then
  echo "downloading NeurIPS 2025 style file..."
  curl -sLO https://raw.githubusercontent.com/gpleiss/latex_template/main/neurips_2025.sty
fi

# Compile (3 passes to resolve refs + bib).
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "done. open main.pdf"
