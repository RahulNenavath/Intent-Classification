#!/usr/bin/env zsh
set -euo pipefail

# setup_vectordb.sh
# Sets up a persistent Chroma vector database using src/vector_db/setup_db.py
# - Prepares input files in Data/
# - Installs dependencies from requirements.txt (if available)
# - Invokes the builder with configurable encoder + collection names

# Project root = directory of this script
ROOT_DIR=${0:A:h}
cd "$ROOT_DIR"

SRC_DIR="$ROOT_DIR/src"
DATA_DIR="$ROOT_DIR/Data"

# -------- Run setup_db builder (direct) --------
echo "[setup_vectordb] Building Chroma DB via setup_db.py..."
python3 "$SRC_DIR/vector_db/setup_db.py"

echo "[setup_vectordb] Done."
