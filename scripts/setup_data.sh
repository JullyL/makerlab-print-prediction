#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_data.sh /path/to/ChallengeData
#
# This script copies the large 5000-row dataset into data/raw/
# without tracking it in git.

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/setup_data.sh /path/to/ChallengeData"
  exit 1
fi

SOURCE_DIR="$1"
SOURCE_FILE="${SOURCE_DIR%/}/makerlab_dataset_5000_rows.csv"
TARGET_DIR="data/raw"
TARGET_FILE="${TARGET_DIR}/makerlab_dataset_5000_rows.csv"

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "Error: file not found: $SOURCE_FILE"
  exit 1
fi

mkdir -p "$TARGET_DIR"
cp "$SOURCE_FILE" "$TARGET_FILE"

echo "Copied:"
echo "  $SOURCE_FILE"
echo "to:"
echo "  $TARGET_FILE"
echo
echo "Note: $TARGET_FILE is intentionally gitignored (GitHub 100MB limit)."
