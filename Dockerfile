# onnxruntime-gpu 1.31+ 需要 Python >= 3.11 和 CUDA 13 运行库。
# CUDA / cuDNN 运行库由 pip 的 onnxruntime-gpu[cuda,cudnn] 提供，GPU 驱动由宿主机的 NVIDIA 容器运行时注入，
# 因此不再依赖 nvidia/cuda 基础镜像（其 Ubuntu 22.04 自带 Python 3.10，装不了新版 onnxruntime-gpu）。
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV EMBEDDING_MODEL_DIR=/app/Models/embeddinggemma-2-onnx
ENV ONNX_EXECUTION_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
ENV MAX_CONCURRENT_INFERENCES=1
ENV IMAGE_MAX_SOFT_TOKENS=280
ENV EMBEDDING_MAX_BATCH_TOKENS=8192
ENV ORT_CUDA_GPU_MEM_LIMIT_MB=24576
ENV ORT_ENABLE_GPU_ARENA_SHRINKAGE=1

# 复制requirements.txt文件
COPY requirements.txt .

# 安装 Python 依赖。Docker 镜像只安装 GPU 版 ONNX Runtime，避免先装 CPU 版再卸载导致构建失败。
RUN grep -v "^onnxruntime\(\[.*\]\)\?>=" requirements.txt > requirements.docker.txt \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.docker.txt \
    && python -m pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]>=1.31.1"

# 复制运行 gRPC 服务所需的最小文件集
COPY gemma_engine.py ./
COPY image_utils.py ./
COPY grpc_service.py ./
COPY onnx_provider_utils.py ./
COPY text_service_pb2.py ./
COPY text_service_pb2_grpc.py ./
COPY Models/embeddinggemma-2-onnx ./Models/embeddinggemma-2-onnx

# 设置环境变量
ENV GRPC_PORT=32688

# 暴露端口
EXPOSE 32688

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc, text_service_pb2, text_service_pb2_grpc; channel = grpc.insecure_channel('localhost:32688'); stub = text_service_pb2_grpc.TextGrpcServiceStub(channel); response = stub.HealthCheck(text_service_pb2.HealthCheckRequest(), timeout=5); channel.close(); raise SystemExit(0 if response.status == 'healthy' else 1)" || exit 1

# 关联 GitHub 仓库（ghcr.io 包页面显示仓库链接，并沿用仓库的访问权限）
LABEL org.opencontainers.image.source="https://github.com/iwaitu/textservice-gemma-2"
LABEL org.opencontainers.image.description="EmbeddingGemma 2 ONNX text/image embedding, rerank and chunking gRPC service, compatible with TextServiceQwen3"
LABEL org.opencontainers.image.licenses="MIT"

# 启动gRPC服务
CMD ["python", "grpc_service.py"]
