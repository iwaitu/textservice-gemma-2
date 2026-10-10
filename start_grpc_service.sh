#!/usr/bin/env bash
# 启动 gRPC 服务（Linux / macOS）
set -euo pipefail
cd "$(dirname "$0")"
export EMBEDDING_MODEL_DIR="${EMBEDDING_MODEL_DIR:-$PWD/Models/embeddinggemma-2-onnx}"
export GRPC_PORT="${GRPC_PORT:-32688}"
PYTHON_CMD="${PYTHON_EXE:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
echo "启动 EmbeddingGemma2 ONNX gRPC Service: localhost:$GRPC_PORT"
echo "Embedding: $EMBEDDING_MODEL_DIR"
exec "$PYTHON_CMD" grpc_service.py
