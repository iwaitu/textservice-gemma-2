# textservice-gemma-2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![gRPC](https://img.shields.io/badge/gRPC-1.75%2B-4285F4.svg)](https://grpc.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20%2B-005CED.svg)](https://onnxruntime.ai/)
[![Model](https://img.shields.io/badge/Model-EmbeddingGemma%202-FF6F00.svg)](https://huggingface.co/google/embeddinggemma-2)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)

基于 Google **EmbeddingGemma 2** ONNX 模型的文本处理服务，提供文本嵌入、**图片嵌入**、重排序和分块功能。
REST API 与 gRPC 接口与 [TextServiceQwen3](https://github.com/iwaitu/TextServiceQwen3) **完全兼容**，可直接替换生产环境中的 TextServiceQwen3；图片向量以新增接口提供，不影响原有接口。

- 🔤 **文本嵌入**：EmbeddingGemma 2（740M 多模态模型中的 270M 文本部分），768 维，支持 MRL 截断到 512 / 256 / 128 维
- 🖼️ **图片嵌入**：与文本向量处于同一向量空间，可直接用文字检索图片（以文搜图）、以图搜图
- 🔄 **文本重排序**：默认使用同一嵌入模型的非对称检索前缀计算相关度；可选接入 Qwen3-Reranker cross-encoder
- ✂️ **文本分块**：基于 Gemma tokenizer 的 token 窗口切分
- 🌐 **双接口**：REST（FastAPI, 端口 8000）+ gRPC（端口 32688），原有消息与 TextServiceQwen3 完全一致
- 🐳 **Docker**：CUDA 镜像，默认启动 gRPC 服务

## 与 TextServiceQwen3 的兼容性

| 项目 | 是否一致 | 说明 |
| :-- | :-- | :-- |
| `protos/text_service.proto` 原有服务与消息 | ✅ 完全一致 | 在 TextServiceQwen3 的 proto 基础上只做了新增；用 TextServiceQwen3 生成的客户端代码可以直接调用本服务 |
| gRPC 服务名 / 方法 / 端口 32688 | ✅ | `TextGrpcService` 原有的 5 个 RPC 签名不变（另新增 `EmbedImage`） |
| REST 路径、请求体、响应体、状态码 | ✅ | `/embed_text` `/rerank` `/split_text` `/health`，错误同样返回 500（参数校验 422） |
| 排序语义 | ✅ | rerank 结果按 score 降序，`index` 为原始下标，score 在 [0, 1] |
| 向量维度 | ⚠️ 不同 | Qwen3 为 1024 维，本服务为 768 维（可设为 512/256/128） |
| 向量空间 | ⚠️ 不同 | **切换前必须用本服务重新生成全部向量**，不能与 Qwen3 的向量混用 |
| rerank 分数分布 | ⚠️ 近似 | 默认将余弦相似度经 sigmoid 校准映射到 [0,1]；若下游依赖 Qwen3 的分数阈值，请复核阈值或使用 cross-encoder 后端 |
| 分块边界 / `tokens_length` | ⚠️ 不同 | token 数基于 Gemma tokenizer 计算，与 Qwen3 tokenizer 不同 |
| `model` / `model_name` 字段值 | ⚠️ 不同 | 返回 `embeddinggemma-2-onnx` |
| 图片向量接口 | ➕ 新增 | REST `/embed_image`、`/embed_image/upload`，gRPC `EmbedImage`；proto 只新增了 RPC 与消息，原有消息未改动，旧客户端无需更新 |
| 服务端依赖版本 | ⚠️ 提高 | 重新生成的 gRPC 代码要求 `grpcio>=1.75.1`、`protobuf>=6.31.1`（TextServiceQwen3 为 1.73 / 6.31.0）；只影响服务端，客户端不受影响 |

用 `compat_check.py` 检查本服务（原有接口 + 图片向量，共 59 项），并可与线上 TextServiceQwen3 对比响应结构：

```bash
python compat_check.py --grpc 127.0.0.1:32688 --rest http://127.0.0.1:8000
python compat_check.py --grpc 127.0.0.1:32688 --reference-grpc <qwen3-host>:32688
python compat_check.py --rest http://127.0.0.1:8000 --reference-rest http://<qwen3-host>:8000
```

图片检查默认使用 `tests/fixtures/cats.jpg`（首次运行自动下载）；未启用图片向量时加 `--skip-image`。

## 快速开始

### 1. 安装依赖

```bash
cd textservice-gemma-2
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 下载模型

从 Hugging Face 的 [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) 下载文本模型和视觉编码器（不含音频编码器）：

```bash
python tools/download_model.py                       # fp32，文本 1.1 GB + 视觉 0.7 GB（默认，推荐）
python tools/download_model.py --variant quantized   # int8，文本 314 MB + 视觉 195 MB，CPU 部署可选
python tools/download_model.py --no-vision           # 只下载文本模型（不提供图片向量）
```

模型会放在 `Models/embeddinggemma-2-onnx/`：

```
Models/embeddinggemma-2-onnx/
├── config.json
├── config_sentence_transformers.json   # 任务前缀定义
├── tokenizer.json / tokenizer_config.json
├── processor_config.json
└── onnx/
    ├── model.onnx / model.onnx_data                   # 文本模型
    └── vision_encoder.onnx / vision_encoder.onnx_data # 视觉编码器（图片向量）
```

使用非 fp32 变体时设置 `EMBEDDING_MODEL_PATH` 和 `VISION_MODEL_PATH`，例如 `EMBEDDING_MODEL_PATH=Models/embeddinggemma-2-onnx/onnx/model_quantized.onnx`。

> 模型卡明确指出 EmbeddingGemma 2 的激活值会超出 float16 范围，不要用纯 fp16 推理；ONNX 的 `fp16` 变体请先自行验证质量。

### 3. 启动服务

```bash
python main.py            # REST:  http://localhost:8000/docs
python grpc_service.py    # gRPC:  localhost:32688
```

或使用 `start_service.sh` / `start_grpc_service.sh`（Windows 为对应的 `.bat`）。

### 4. Docker 部署

```bash
python tools/download_model.py
docker-compose up -d
```

镜像默认启动 gRPC 服务（端口 32688），需要 NVIDIA Container Toolkit。详见 [README-Docker.md](README-Docker.md)。

## API

### REST

```bash
POST /embed_text   {"input": ["文本1", "文本2"]}
POST /rerank       {"query": "查询", "documents": ["候选1", "候选2"], "instruction": "可选"}
POST /split_text   {"text": "长文本", "chunksize": 1000, "overlap_size": 100}
GET  /health

POST /embed_image         {"input": ["<base64>", "data:image/png;base64,..."]}    # 新增
POST /embed_image/upload  multipart/form-data，字段名 files，可多个               # 新增
```

### gRPC

`EmbedText`、`SimpleRerank`、`SplitTextIntoChunks`、`HealthCheck`、`GetModelInfo`，以及新增的 `EmbedImage`，定义见 [protos/text_service.proto](protos/text_service.proto)。

## 图片向量

图片向量和文本向量在**同一个 768 维向量空间**里（使用同一套 `EMBEDDING_DIMENSION` 截断），所以可以：

- **以文搜图**：图片入库用 `/embed_image`，查询用 `/embed_text`（建议加 `task: search result | query: ` 前缀），直接算余弦相似度；
- **以图搜图**：查询和入库都用 `/embed_image`；
- 图文混合检索：文本和图片向量放进同一个索引。

示例（`cats.jpg`，两只猫睡在沙发上）：

| 文本查询 | 与图片的相似度 |
| :-- | :-- |
| `task: search result \| query: two cats sleeping` | 0.748 |
| `task: search result \| query: 沙发上睡觉的猫` | 0.722 |
| `task: search result \| query: 一辆红色跑车` | 0.568 |
| `task: search result \| query: stock market report` | 0.507 |

返回结构与 `/embed_text` 完全相同（`data[].embedding / index / tokens_length`、`usage`、`model`），`tokens_length` 为该图片实际占用的视觉 token 数。

```bash
# 上传文件
curl -F "files=@cat.jpg" -F "files=@dog.png" http://localhost:8000/embed_image/upload

# JSON + base64
curl -X POST http://localhost:8000/embed_image -H "Content-Type: application/json" \
  -d "{\"input\": [\"$(base64 < cat.jpg | tr -d '\n')\"]}"
```

```python
# gRPC：images 为图片文件的原始字节
import grpc, text_service_pb2 as pb, text_service_pb2_grpc as pb_grpc

stub = pb_grpc.TextGrpcServiceStub(grpc.insecure_channel("localhost:32688"))
with open("cat.jpg", "rb") as f:
    resp = stub.EmbedImage(pb.EmbedImageRequest(images=[f.read()]))
vector = resp.data[0].embedding
```

说明：

- 支持 JPEG / PNG / WebP / GIF（取第一帧）/ BMP 等 Pillow 能解码的格式；会按 EXIF 方向自动摆正、转为 RGB。
- 预处理与 transformers 的 `Gemma4ImageProcessor` 逐像素一致：保持宽高比缩放到不超过 `IMAGE_MAX_SOFT_TOKENS × 9` 个 16×16 patch，默认每张图约 250–280 个视觉 token。
- 图片不加任务前缀（模型卡要求）。
- 输入错误（非图片、base64 无效、超过大小限制）时，REST 返回 400 并指出是第几项，gRPC 返回 `INVALID_ARGUMENT`；未加载视觉编码器时返回 500 / `INTERNAL`。
- gRPC 服务端默认允许 64 MB 的消息（`GRPC_MAX_MESSAGE_MB`），客户端发送大图时无需额外设置。
- 通过 URL 拉取图片默认关闭（避免 SSRF）。如需开启，设置 `ALLOW_IMAGE_URLS=1`；默认仍会拒绝解析到内网地址的 URL，可用 `ALLOW_PRIVATE_IMAGE_URLS=1` 放开。

## EmbeddingGemma 2 的任务前缀

模型训练时使用了任务前缀，加前缀可以提升质量，不加也能用：

| 场景 | 查询端前缀 | 文档端前缀 |
| :-- | :-- | :-- |
| 检索（SearchQuery） | `task: search result \| query: ` | `title: none \| text: `（有标题时 `title: {标题} \| text: `） |
| 问答（QuestionAnswering） | `task: question answering \| query: ` | 同上 |
| 代码检索（CodeRetrieval） | `task: code retrieval \| query: ` | 同上 |
| 分类 / 聚类 / 相似度 | `task: classification \| query: ` 等（两端相同） | — |

- **`/embed_text` 与 `EmbedText` 默认不加前缀**（与 TextServiceQwen3 一样直接编码原文），保证"什么都不改就能替换"。
- 如果调用方能区分查询和文档，建议在文本中自行带上前缀，例如入库文档用 `title: none | text: ...`、查询用 `task: search result | query: ...`。
- 也可以用环境变量 `EMBEDDING_PROMPT_NAME`（如 `Document`）或 `EMBEDDING_PROMPT`（原始字符串）给所有输入统一加前缀。
- `tokens_length` / `text_length` 统计的是原文的 token 数（不含前缀和特殊 token）。

## 重排序

EmbeddingGemma 2 只是嵌入模型，本身不带 reranker，所以提供两种后端：

1. **`RERANK_BACKEND=embedding`（默认）**：查询使用 `task: search result | query: ` 前缀，文档使用 `title: none | text: ` 前缀，计算余弦相似度后映射成分数：
   - `RERANK_SCORE_MODE=sigmoid`（默认）：`score = sigmoid((cos - 0.77) * 30)`。EmbeddingGemma 2 的余弦相似度分布很集中（实测相关文档约 0.80–0.90，不相关约 0.57–0.75），这样映射后更接近 Qwen3 reranker 的概率式分数。中心点和斜率可通过 `RERANK_SIGMOID_CENTER` / `RERANK_SIGMOID_SCALE` 调整。**这两个默认值只用 8 组样本校准过，上线前请用自己的数据复核。**
   - `RERANK_SCORE_MODE=cosine`：直接返回 `clip(cos, 0, 1)`。
   - REST `/rerank` 的 `instruction` 字段：可传入前缀名（如 `QuestionAnswering`、`CodeRetrieval`）或以 `task:` 开头的原始前缀；其他内容（如 Qwen3 的英文指令）会被忽略，回退到默认检索前缀。
2. **`RERANK_BACKEND=cross-encoder`**：设置 `RERANKER_MODEL_DIR` 指向 Qwen3-Reranker seq-cls ONNX 目录时自动启用，分数与 TextServiceQwen3 完全一致（嵌入仍使用 EmbeddingGemma 2）。

## 配置（环境变量）

| 变量 | 默认值 | 说明 |
| :-- | :-- | :-- |
| `EMBEDDING_MODEL_DIR` | `Models/embeddinggemma-2-onnx` | 模型目录（tokenizer 与配置文件） |
| `EMBEDDING_MODEL_PATH` | `$EMBEDDING_MODEL_DIR/onnx/model.onnx` | ONNX 文件路径 |
| `EMBEDDING_DIMENSION` | `768` | MRL 截断维度：768 / 512 / 256 / 128（截断后会重新归一化） |
| `EMBEDDING_PROMPT_NAME` / `EMBEDDING_PROMPT` | 空 | 嵌入时统一加的前缀 |
| `EMBEDDING_MAX_LENGTH` | `8192` | 最大输入 token 数（模型上限 8192） |
| `EMBEDDING_BATCH_SIZE` | `8` | 每批最多条数 |
| `EMBEDDING_MAX_BATCH_TOKENS` | `8192` | 每批 padded token 上限 |
| `RERANK_BACKEND` | `embedding` | `embedding` / `cross-encoder` |
| `RERANK_SCORE_MODE` | `sigmoid` | `sigmoid` / `cosine` |
| `RERANK_SIGMOID_CENTER` / `RERANK_SIGMOID_SCALE` | `0.77` / `30` | sigmoid 校准参数 |
| `RERANK_QUERY_PROMPT_NAME` / `RERANK_DOCUMENT_PROMPT_NAME` | `SearchQuery` / `Document` | rerank 使用的前缀 |
| `RERANKER_MODEL_DIR` / `RERANKER_MODEL_PATH` | 空 | cross-encoder 后端的模型 |
| `RERANK_MAX_LENGTH` / `RERANK_BATCH_SIZE` / `RERANK_MAX_BATCH_TOKENS` | `8192` / `4` / `8192` | cross-encoder 后端的批处理参数 |
| `ENABLE_IMAGE_EMBEDDING` | `auto` | `auto`：存在视觉编码器文件就加载；`true`：缺文件则启动失败；`false`：不加载 |
| `VISION_MODEL_PATH` | `$EMBEDDING_MODEL_DIR/onnx/vision_encoder.onnx` | 视觉编码器 ONNX 文件 |
| `IMAGE_MAX_SOFT_TOKENS` | `280` | 每张图的视觉 token 预算：70 / 140 / 280 / 560 / 1120，越大细节越多、越慢。**入库和查询必须使用相同值** |
| `IMAGE_BATCH_SIZE` | `4` | 视觉编码器每批图片数 |
| `IMAGE_MAX_BYTES` / `IMAGE_MAX_PIXELS` | 20 MB / 64M 像素 | 单张图片的大小限制 |
| `ALLOW_IMAGE_URLS` / `ALLOW_PRIVATE_IMAGE_URLS` / `IMAGE_URL_TIMEOUT` | 关 / 关 / `10` | REST 通过 URL 拉取图片 |
| `GRPC_MAX_MESSAGE_MB` | `64` | gRPC 单条消息上限 |
| `MAX_CONCURRENT_INFERENCES` | `1` | 同时执行的推理数 |
| `ONNX_EXECUTION_PROVIDERS` / `ONNX_PREFERRED_PROVIDER` | 自动 | Provider 选择，与 TextServiceQwen3 相同 |
| `ORT_CUDA_GPU_MEM_LIMIT_MB` / `ORT_ENABLE_GPU_ARENA_SHRINKAGE` | — | CUDA 显存控制，与 TextServiceQwen3 相同 |
| `GRPC_PORT` / `PORT` | `32688` / `8000` | 服务端口。gRPC 关闭了端口复用，同一端口重复启动会直接报错，避免多个进程悄悄分流请求 |

> macOS 上 CoreML EP 不支持文本模型的空多模态输入（`image_features` 等 0 行张量），视觉编码器在 CoreML 上也比 CPU 慢约 2.5 倍，因此服务会自动跳过 CoreML、改用 CPU。

## 从 TextServiceQwen3 迁移

1. 部署本服务（同样监听 32688 / 8000，服务端依赖 `grpcio>=1.75.1`、`protobuf>=6.31.1`），先用 `compat_check.py --reference-grpc` 对比两边的响应结构。
2. **向量库重建**：用本服务重新生成全部文档向量，按 768 维（或选定的 MRL 维度）新建 collection / index。
3. **复核 rerank 阈值**：如果业务代码按 `score > x` 过滤，用真实查询对比两边的分数，必要时调整 `RERANK_SIGMOID_*` 或使用 cross-encoder 后端。
4. **复核分块参数**：`chunk_size` 的单位是 Gemma token，同样的数值切出的文本长度会和 Qwen3 不同。
5. 切换流量。

## 基准测试

`benchmark_grpc.py`、`benchmark_service.py`、`memory_probe_grpc.py` 与 TextServiceQwen3 中的一致，可以直接使用：

```bash
python benchmark_grpc.py --target 127.0.0.1:32688
python benchmark_service.py --base-url http://127.0.0.1:8000
```

本地实测（Apple Silicon Mac，CPUExecutionProvider，fp32，3 轮）：

- Embedding：32 条文本平均 299 ms，吞吐量约 107 条/秒，向量维度 768
- Rerank：50 条候选平均 474 ms，吞吐量约 105 条/秒
- Rerank 准确性（5 个样本）：Top1 = 1.0，Top3 = 1.0，MRR = 1.0
- 单条 8190 token 输入约 24 s（CPU）
- 图片向量：约 2.1 s/张（CPU，280 视觉 token）；加载视觉编码器后进程内存约 2.9 GB。生产环境建议使用 GPU

图片向量与模型卡给出的参考结果一致：`cats.jpg` 与三条查询的相似度为 0.746 / 0.461 / 0.501（模型卡 q4 结果为 0.740 / 0.462 / 0.508）。

## 开发指南

### 生成 gRPC 代码

修改 `protos/text_service.proto` 后重新生成。为了让生成代码只要求 `protobuf>=6.31`，请使用 `grpcio-tools==1.75.1`（新版本生成的代码会要求 protobuf 7.x）：

```bash
python -m venv /tmp/grpc-gen && /tmp/grpc-gen/bin/pip install "grpcio-tools==1.75.1"
/tmp/grpc-gen/bin/python -m grpc_tools.protoc -I./protos --python_out=. --grpc_python_out=. ./protos/text_service.proto
```

proto 只能新增 RPC / 消息 / 字段（使用新的字段编号），不要修改或删除已有内容，以保持与 TextServiceQwen3 客户端的兼容。

## 项目结构

```
textservice-gemma-2/
├── gemma_engine.py         # 推理核心（REST 与 gRPC 共用）
├── image_utils.py          # 图片解码与视觉预处理
├── main.py                 # FastAPI 服务
├── grpc_service.py         # gRPC 服务
├── onnx_provider_utils.py  # ONNX Runtime Provider 选择
├── protos/text_service.proto
├── text_service_pb2.py / text_service_pb2_grpc.py
├── compat_check.py         # 接口兼容性 + 图片向量检查
├── tools/download_model.py # 下载 ONNX 模型
├── benchmark_grpc.py / benchmark_service.py / memory_probe_grpc.py
├── Dockerfile / docker-compose.yml
└── Models/                 # 模型文件（被 .gitignore 忽略）
```

## 更新日志

### v1.1.0
- 新增图片向量：REST `/embed_image`、`/embed_image/upload`，gRPC `EmbedImage`；与文本向量同一向量空间，支持以文搜图
- 视觉预处理与 transformers `Gemma4ImageProcessor` 逐像素一致，结果与模型卡参考值吻合
- `tools/download_model.py` 默认同时下载视觉编码器（`--no-vision` 跳过）
- gRPC 服务端消息上限提高到 64 MB，关闭端口复用
- `compat_check.py` 新增图片向量检查

### v1.0.0
- 初始版本：基于 EmbeddingGemma 2 ONNX，REST / gRPC 接口与 TextServiceQwen3 完全兼容
- 支持 MRL 维度截断、任务前缀、embedding / cross-encoder 两种 rerank 后端
- Docker 部署、兼容性检查脚本、模型下载脚本

## 许可证

代码采用 MIT 许可证，详见 [LICENSE](LICENSE)。EmbeddingGemma 2 模型采用 Apache 2.0 许可证，使用时须遵守 [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)。
