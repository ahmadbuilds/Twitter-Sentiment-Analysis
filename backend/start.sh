#!/usr/bin/env bash
set -e

# HF Spaces automatically inject PORT (usually 7860)
PORT=${PORT:-7860}

if [ ! -d "./Models/chunk_model2" ]; then
  echo "Warning: ./Models/chunk_model2 not found. Make sure model files are included in the image or mounted at runtime."
fi

# Start FastAPI using uvicorn
exec uvicorn entry_main:app --host 0.0.0.0 --port ${PORT} --workers 1
