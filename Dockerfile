# 使用NVIDIA CUDA基础镜像，支持GPU加速
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV EMBEDDING_MODEL_DIR=/app/Models/embeddinggemma-2-onnx
ENV ONNX_EXECUTION_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
ENV MAX_CONCURRENT_INFERENCES=1
ENV IMAGE_MAX_SOFT_TOKENS=280
ENV EMBEDDING_MAX_BATCH_TOKENS=8192
ENV ORT_CUDA_GPU_MEM_LIMIT_MB=24576
ENV ORT_ENABLE_GPU_ARENA_SHRINKAGE=1

# 安装 Python 和系统依赖。Ubuntu 22.04 默认提供 Python 3.10，直接使用 python3 包更稳妥。
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY requirements.txt .

# 安装 Python 依赖。Docker 镜像只安装 GPU 版 ONNX Runtime，避免先装 CPU 版再卸载导致构建失败。
RUN grep -v "^onnxruntime\(\[.*\]\)\?>=" requirements.txt > requirements.docker.txt \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.docker.txt \
    && python -m pip install --no-cache-dir "onnxruntime-gpu>=1.20.0"

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

# 启动gRPC服务
CMD ["python", "grpc_service.py"]
