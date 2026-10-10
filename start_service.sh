#!/usr/bin/env bash
# 启动 REST 服务（Linux / macOS）
set -euo pipefail
cd "$(dirname "$0")"
export EMBEDDING_MODEL_DIR="${EMBEDDING_MODEL_DIR:-$PWD/Models/embeddinggemma-2-onnx}"
PYTHON_CMD="${PYTHON_EXE:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
echo "启动 EmbeddingGemma2 ONNX REST Service: http://localhost:${PORT:-8000}/docs"
echo "Embedding: $EMBEDDING_MODEL_DIR"
exec "$PYTHON_CMD" main.py
