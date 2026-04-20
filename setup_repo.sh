#!/bin/bash
# setup_repo.sh
# Run this once after cloning the repo to push the full project structure.
#
# Usage:
#   git clone https://github.com/JullyL/makerlab-print-prediction.git
#   cd makerlab-print-prediction
#   bash setup_repo.sh

set -e

echo "Setting up makerlab-print-prediction repo..."

# Copy all files from the downloaded zip into the cloned repo
# (Adjust SOURCE_DIR to wherever you extracted the downloaded files)
SOURCE_DIR="$(dirname "$0")"

cp -r "$SOURCE_DIR"/. .

# Initialize git if not already a repo
if [ ! -d ".git" ]; then
  git init
  git remote add origin https://github.com/JullyL/makerlab-print-prediction.git
fi

git add .
git commit -m "feat: initial project structure

- Add README with full team role assignments and shared interfaces
- Add Owner A EDA notebook (01_eda.ipynb) with all 10 sections pre-built
- Add stub notebooks for Owners B, E, F
- Add stub src/ modules for all owners
- Add Streamlit app skeleton
- Add raw datasets (100, 1000, 5000 rows)
- Add .gitignore (excludes data/processed/, model weights)
- Add requirements.txt"

git push -u origin main

echo ""
echo "Done! Repo is live at https://github.com/JullyL/makerlab-print-prediction"
echo ""
echo "Next steps for Owner A:"
echo "  pip install -r requirements.txt"
echo "  jupyter notebook notebooks/01_eda.ipynb"
