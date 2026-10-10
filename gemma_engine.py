"""EmbeddingGemma 2 ONNX 推理核心，供 REST (main.py) 与 gRPC (grpc_service.py) 共用。

接口语义与 TextServiceQwen3 保持一致：
- embed: 返回 L2 归一化向量、每条文本 token 数（不含特殊 token）、本次 padded 前的总 token 数
- rerank: 返回与 documents 顺序一致的 [0, 1] 分数
- split_text: 基于 embedding tokenizer 的 token 窗口切分
"""

import json
import logging
import os
import threading

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from onnx_provider_utils import (
    COREML_PROVIDER,
    CPU_PROVIDER,
    build_provider_chain,
    choose_execution_providers,
    provider_chain_to_string,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models")

EMBEDDING_DIR_DEFAULT = os.path.join(MODELS_DIR, "embeddinggemma-2-onnx")
EMBEDDING_DIR = os.environ.get("EMBEDDING_MODEL_DIR", EMBEDDING_DIR_DEFAULT)
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    os.path.join(EMBEDDING_DIR, "onnx", "model.onnx"),
)

# 可选：使用 cross-encoder（例如 Qwen3-Reranker ONNX）作为重排序后端，
# 以获得与 TextServiceQwen3 相同分布的 rerank 分数。
RERANKER_DIR = os.environ.get("RERANKER_MODEL_DIR", "").strip()
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    os.path.join(RERANKER_DIR, "model.onnx") if RERANKER_DIR else "",
)
RERANK_BACKEND = os.environ.get(
    "RERANK_BACKEND",
    "cross-encoder" if RERANKER_DIR else "embedding",
).strip().lower()

EMBED_MAX_LENGTH = int(os.environ.get("EMBEDDING_MAX_LENGTH", "8192"))
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "8192"))
EMBED_BATCH_SIZE = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "8")))
RERANK_BATCH_SIZE = max(1, int(os.environ.get("RERANK_BATCH_SIZE", "4")))
EMBED_MAX_BATCH_TOKENS = max(
    1,
    int(os.environ.get("EMBEDDING_MAX_BATCH_TOKENS", str(EMBED_MAX_LENGTH))),
)
RERANK_MAX_BATCH_TOKENS = max(
    1,
    int(os.environ.get("RERANK_MAX_BATCH_TOKENS", str(RERANK_MAX_LENGTH))),
)
_MAX_CONCURRENT = max(1, int(os.environ.get("MAX_CONCURRENT_INFERENCES", "1")))
_inference_semaphore = threading.Semaphore(_MAX_CONCURRENT)

# Matryoshka 截断维度：768（原生）/ 512 / 256 / 128
SUPPORTED_EMBEDDING_DIMENSIONS = (768, 512, 256, 128)
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))

# EmbeddingGemma 2 的任务前缀（见模型卡 Task Instruction Prefixes）。
# /embed_text 与 EmbedText 默认不加前缀，与 TextServiceQwen3 行为一致（原文直接编码）；
# 调用方可以在文本中自行带上前缀，或通过环境变量设置统一前缀。
EMBEDDING_PROMPT_NAME = os.environ.get("EMBEDDING_PROMPT_NAME", "").strip()
EMBEDDING_PROMPT = os.environ.get("EMBEDDING_PROMPT")
RERANK_QUERY_PROMPT_NAME = os.environ.get("RERANK_QUERY_PROMPT_NAME", "SearchQuery").strip()
RERANK_DOCUMENT_PROMPT_NAME = os.environ.get("RERANK_DOCUMENT_PROMPT_NAME", "Document").strip()

# embedding 后端的 rerank 分数映射：
#   sigmoid -> sigmoid((cos - center) * scale)（默认）。EmbeddingGemma 2 的余弦相似度分布很集中
#              （实测相关约 0.80~0.90、不相关约 0.57~0.75），映射后更接近 Qwen3 reranker
#              的概率分布，使下游基于分数阈值的逻辑不至于全部放行。
#   cosine  -> clip(cos, 0, 1)，原始余弦相似度。
RERANK_SCORE_MODE = os.environ.get("RERANK_SCORE_MODE", "sigmoid").strip().lower()
RERANK_SIGMOID_CENTER = float(os.environ.get("RERANK_SIGMOID_CENTER", "0.77"))
RERANK_SIGMOID_SCALE = float(os.environ.get("RERANK_SIGMOID_SCALE", "30.0"))

DEFAULT_PROMPTS = {
    "SearchQuery": "task: search result | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "FactChecking": "task: fact checking | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "Document": "title: none | text: ",
}

# Qwen3 cross-encoder reranker 模板（仅 RERANK_BACKEND=cross-encoder 时使用）
DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_PREFIX = (
    "<|im_start|>system\n"
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

LOGGER = logging.getLogger(__name__)


def resolve_model_name(model_dir, fallback_name):
    return os.path.basename(os.path.normpath(model_dir)) or fallback_name


def _get_execution_providers(excluded=()):
    providers, reason, available = choose_execution_providers()
    dropped = [provider for provider in providers if provider in excluded]
    if dropped:
        providers = [provider for provider in providers if provider not in excluded] or [CPU_PROVIDER]
        LOGGER.warning("Providers %s are not supported by this model, skipped", dropped)
    configured = build_provider_chain(providers)
    LOGGER.info(
        "Selected ONNX providers: %s | reason=%s | available=%s",
        provider_chain_to_string(configured),
        reason,
        available,
    )
    return configured


def _create_run_options():
    raw_value = os.environ.get("ORT_ENABLE_GPU_ARENA_SHRINKAGE", "").strip().lower()
    if raw_value not in {"1", "true", "yes", "on"}:
        return None

    run_options = ort.RunOptions()
    target = os.environ.get("ORT_GPU_ARENA_SHRINKAGE_TARGET", "gpu:0").strip() or "gpu:0"
    run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", target)
    LOGGER.info("Enabled ORT GPU arena shrinkage after each run: target=%s", target)
    return run_options


def _create_session(model_path, excluded_providers=()):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=_get_execution_providers(excluded_providers),
    )


def _normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-12, a_max=None)


def _mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask[..., None].astype(last_hidden_states.dtype)
    summed = (last_hidden_states * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    positive = values >= 0
    negative = ~positive
    result = np.empty_like(values, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    return result


def _load_prompts(model_dir):
    prompts = dict(DEFAULT_PROMPTS)
    config_path = os.path.join(model_dir, "config_sentence_transformers.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            prompts.update(json.load(handle).get("prompts") or {})
    except FileNotFoundError:
        pass
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", config_path, exc)
    return prompts


def _iter_token_budget_batches(
    values,
    tokenizer,
    tokenizer_lock,
    max_length,
    max_batch_size,
    max_batch_tokens,
):
    """Build batches without letting padding multiply one long input across the batch."""
    batch = []
    batch_lengths = []
    padded_length = 0

    for value in values:
        # Fast tokenizers temporarily mutate their Rust-side truncation/padding
        # configuration.  Every access to a shared tokenizer must use the same
        # lock or concurrent requests can fail with "Already borrowed".
        with tokenizer_lock:
            special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
            content_max_length = max(1, max_length - special_token_count)
            token_length_without_specials = len(tokenizer.encode(
                value,
                add_special_tokens=False,
                truncation=True,
                max_length=content_max_length,
                verbose=False,
            ))
        token_length = min(token_length_without_specials + special_token_count, max_length)
        next_padded_length = max(padded_length, token_length)
        exceeds_size = len(batch) >= max_batch_size
        exceeds_tokens = next_padded_length * (len(batch) + 1) > max_batch_tokens

        if batch and (exceeds_size or exceeds_tokens):
            yield batch, batch_lengths
            batch = []
            batch_lengths = []
            padded_length = 0

        batch.append(value)
        batch_lengths.append(token_length_without_specials)
        padded_length = max(padded_length, token_length)

    if batch:
        yield batch, batch_lengths


def split_text(text, chunk_size, overlap_size, tokenizer, tokenizer_lock):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap_size < 0:
        raise ValueError("overlap_size cannot be negative")
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be smaller than chunk_size")

    with tokenizer_lock:
        token_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            chunk_ids = token_ids[start:end]
            chunk_tokens = tokenizer.convert_ids_to_tokens(chunk_ids)
            chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
            if end >= len(token_ids):
                break
            start = end - overlap_size if overlap_size > 0 else end
    return chunks


class GemmaEmbeddingModel:
    """EmbeddingGemma 2 文本 ONNX 模型（onnx-community/embeddinggemma-2-ONNX 的 onnx/model*.onnx）。"""

    # 文本模型同时声明了多模态输入；纯文本推理时传入 0 行的空特征即可。
    _MULTIMODAL_FEATURE_INPUTS = ("image_features", "video_features", "audio_features")

    def __init__(self, model_dir=EMBEDDING_DIR, model_path=EMBEDDING_MODEL_PATH):
        if EMBEDDING_DIMENSION not in SUPPORTED_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"EMBEDDING_DIMENSION must be one of {SUPPORTED_EMBEDDING_DIMENSIONS}, "
                f"got {EMBEDDING_DIMENSION}"
            )

        self.model_name = resolve_model_name(model_dir, "embeddinggemma-2-onnx")
        self.tokenizer_lock = threading.RLock()
        # Gemma 使用 mean pooling，右侧 padding 即可（与 Qwen3 的 last-token pooling 不同）。
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            model_max_length=EMBED_MAX_LENGTH,
            padding_side="right",
        )
        self.prompts = _load_prompts(model_dir)
        self.default_prompt = self._resolve_default_prompt()

        # CoreML EP 不接受 0 行的动态形状输入（image/video/audio_features），只能退回 CPU。
        self.session = _create_session(model_path, excluded_providers=(COREML_PROVIDER,))
        self.run_options = _create_run_options()
        self.provider = self.session.get_providers()[0]
        self._input_names = {item.name for item in self.session.get_inputs()}
        self._output_names = [item.name for item in self.session.get_outputs()]
        self._feature_feed = self._build_empty_feature_feed()
        self.batch_size = self._resolve_batch_size(EMBED_BATCH_SIZE)

        LOGGER.info("Embedding session providers: %s", self.session.get_providers())
        LOGGER.info(
            "Embedding inputs=%s outputs=%s dimension=%s default_prompt=%r batch_size=%s",
            sorted(self._input_names),
            self._output_names,
            EMBEDDING_DIMENSION,
            self.default_prompt,
            self.batch_size,
        )

    def _resolve_default_prompt(self):
        if EMBEDDING_PROMPT is not None:
            return EMBEDDING_PROMPT
        if not EMBEDDING_PROMPT_NAME:
            return ""
        if EMBEDDING_PROMPT_NAME not in self.prompts:
            raise ValueError(
                f"Unknown EMBEDDING_PROMPT_NAME={EMBEDDING_PROMPT_NAME!r}; "
                f"available: {sorted(self.prompts)}"
            )
        return self.prompts[EMBEDDING_PROMPT_NAME]

    def prompt_for(self, name, fallback=""):
        return self.prompts.get(name, fallback)

    def _build_empty_feature_feed(self):
        feed = {}
        for item in self.session.get_inputs():
            if item.name not in self._MULTIMODAL_FEATURE_INPUTS:
                continue
            width = item.shape[-1] if isinstance(item.shape[-1], int) else 512
            feed[item.name] = np.zeros((0, width), dtype=np.float32)
        return feed

    def _resolve_batch_size(self, requested_batch_size):
        if requested_batch_size <= 1:
            return 1
        try:
            encoded, input_feed = self._build_inputs(["batch probe text 1", "batch probe text 2 longer"])
            self.session.run(None, input_feed)
            del encoded, input_feed
            return requested_batch_size
        except Exception as exc:
            LOGGER.warning(
                "Embedding ONNX model does not support true batching on current runtime; "
                "requested_batch_size=%s fallback_batch_size=1 error=%s",
                requested_batch_size,
                exc,
            )
            return 1

    def _build_inputs(self, texts):
        with self.tokenizer_lock:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="np",
                max_length=EMBED_MAX_LENGTH,
            )
        input_feed = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in self._input_names
        }
        input_feed.update(self._feature_feed)
        return encoded, input_feed

    def _forward(self, input_feed, attention_mask):
        with _inference_semaphore:
            outputs = self.session.run(None, input_feed, run_options=self.run_options)
        named = dict(zip(self._output_names, outputs))
        del outputs
        if "sentence_embedding" in named:
            embeddings = np.asarray(named["sentence_embedding"], dtype=np.float32)
        else:
            embeddings = _mean_pool(np.asarray(named[self._output_names[0]]), attention_mask)
        del named
        if EMBEDDING_DIMENSION < embeddings.shape[1]:
            embeddings = embeddings[:, :EMBEDDING_DIMENSION]
        return _normalize_embeddings(embeddings).astype(np.float32)

    def embed(self, texts, prompt=None):
        """返回 (embeddings[N, D], 每条原文 token 数, 实际计算的 token 总数)。"""
        if not texts:
            return np.empty((0, 0), dtype=np.float32), [], 0

        prompt = self.default_prompt if prompt is None else prompt
        model_inputs = [f"{prompt}{text}" for text in texts] if prompt else list(texts)

        embeddings = []
        input_lengths = []
        total_tokens = 0
        batches = _iter_token_budget_batches(
            model_inputs,
            self.tokenizer,
            self.tokenizer_lock,
            EMBED_MAX_LENGTH,
            self.batch_size,
            EMBED_MAX_BATCH_TOKENS,
        )
        for chunk_texts, chunk_lengths in batches:
            encoded, input_feed = self._build_inputs(chunk_texts)
            embeddings.extend(self._forward(input_feed, encoded["attention_mask"]))
            input_lengths.extend(chunk_lengths)
            total_tokens += int(np.asarray(encoded["attention_mask"]).sum())
            del encoded, input_feed

        text_lengths = input_lengths if not prompt else self.count_tokens(texts)
        return np.stack(embeddings, axis=0), text_lengths, total_tokens

    def count_tokens(self, texts):
        with self.tokenizer_lock:
            content_max_length = max(
                1,
                EMBED_MAX_LENGTH - self.tokenizer.num_special_tokens_to_add(pair=False),
            )
            return [
                len(self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=content_max_length,
                    verbose=False,
                ))
                for text in texts
            ]

    def split_text(self, text, chunk_size, overlap_size):
        return split_text(text, chunk_size, overlap_size, self.tokenizer, self.tokenizer_lock)


class EmbeddingReranker:
    """用 EmbeddingGemma 2 的非对称检索前缀做双塔重排序：score 由 query/document 余弦相似度得到。"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.model_name = embedding_model.model_name
        self.session = embedding_model.session
        self.query_prompt = embedding_model.prompt_for(
            RERANK_QUERY_PROMPT_NAME, DEFAULT_PROMPTS["SearchQuery"]
        )
        self.document_prompt = embedding_model.prompt_for(
            RERANK_DOCUMENT_PROMPT_NAME, DEFAULT_PROMPTS["Document"]
        )
        if RERANK_SCORE_MODE not in {"cosine", "sigmoid"}:
            raise ValueError("RERANK_SCORE_MODE must be 'cosine' or 'sigmoid'")

    def _query_prompt(self, instruction):
        """instruction 可为模型卡中的 prompt 名（如 QuestionAnswering）或以 'task:' 开头的原始前缀。"""
        if not instruction:
            return self.query_prompt
        instruction = instruction.strip()
        if instruction in self.embedding_model.prompts:
            return self.embedding_model.prompts[instruction]
        if instruction.lower().startswith("task:"):
            return instruction if instruction.endswith(" ") else f"{instruction} "
        return self.query_prompt

    def score(self, query, documents, instruction=None):
        if not documents:
            return np.array([], dtype=np.float32)
        query_embedding, _, _ = self.embedding_model.embed([query], prompt=self._query_prompt(instruction))
        document_embeddings, _, _ = self.embedding_model.embed(documents, prompt=self.document_prompt)
        cosine = (document_embeddings @ query_embedding[0]).astype(np.float32)
        if RERANK_SCORE_MODE == "sigmoid":
            return _sigmoid((cosine - RERANK_SIGMOID_CENTER) * RERANK_SIGMOID_SCALE)
        return np.clip(cosine, 0.0, 1.0)


class CrossEncoderReranker:
    """可选：Qwen3-Reranker seq-cls ONNX，分数分布与 TextServiceQwen3 完全一致。"""

    def __init__(self, model_dir=RERANKER_DIR, model_path=RERANKER_MODEL_PATH):
        if not model_dir:
            raise ValueError("RERANK_BACKEND=cross-encoder requires RERANKER_MODEL_DIR")
        self.model_name = resolve_model_name(model_dir, "qwen3-reranker-seq-cls-onnx")
        self.tokenizer_lock = threading.RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            model_max_length=RERANK_MAX_LENGTH,
            padding_side="left",
            trust_remote_code=True,
        )
        self.session = _create_session(model_path)
        self.run_options = _create_run_options()
        self._input_names = {item.name for item in self.session.get_inputs()}
        LOGGER.info("Reranker session providers: %s", self.session.get_providers())

    @staticmethod
    def _format_instruction(instruction, query, document):
        rerank_instruction = instruction or DEFAULT_RERANK_INSTRUCTION
        return (
            f"{RERANK_PREFIX}<Instruct>: {rerank_instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}{RERANK_SUFFIX}"
        )

    def _build_inputs(self, pairs):
        with self.tokenizer_lock:
            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="np",
                max_length=RERANK_MAX_LENGTH,
            )
        input_feed = {
            key: value.astype(np.int64)
            for key, value in encoded.items()
            if key in self._input_names
        }
        return encoded, input_feed

    def score(self, query, documents, instruction=None):
        if not documents:
            return np.array([], dtype=np.float32)

        pairs = [self._format_instruction(instruction, query, doc) for doc in documents]
        all_logits = []
        batches = _iter_token_budget_batches(
            pairs,
            self.tokenizer,
            self.tokenizer_lock,
            RERANK_MAX_LENGTH,
            RERANK_BATCH_SIZE,
            RERANK_MAX_BATCH_TOKENS,
        )
        for chunk, _ in batches:
            encoded, input_feed = self._build_inputs(chunk)
            with _inference_semaphore:
                outputs = self.session.run(None, input_feed, run_options=self.run_options)
            all_logits.append(np.asarray(outputs[0]).reshape(-1))
            del encoded, outputs, input_feed
        return _sigmoid(np.concatenate(all_logits))


class TextEngine:
    """服务对外暴露的全部推理能力。"""

    def __init__(self):
        self.embedding = GemmaEmbeddingModel()
        if RERANK_BACKEND == "cross-encoder":
            self.reranker = CrossEncoderReranker()
        elif RERANK_BACKEND == "embedding":
            self.reranker = EmbeddingReranker(self.embedding)
        else:
            raise ValueError("RERANK_BACKEND must be 'embedding' or 'cross-encoder'")

        LOGGER.info(
            "Inference limits: concurrency=%s embedding_batch_tokens=%s "
            "rerank_backend=%s rerank_score_mode=%s",
            _MAX_CONCURRENT,
            EMBED_MAX_BATCH_TOKENS,
            RERANK_BACKEND,
            RERANK_SCORE_MODE if RERANK_BACKEND == "embedding" else "sigmoid(logit)",
        )

    @property
    def embedding_model_name(self):
        return self.embedding.model_name

    @property
    def reranker_model_name(self):
        return self.reranker.model_name

    @property
    def provider(self):
        return self.embedding.provider

    def embed(self, texts):
        return self.embedding.embed(texts)

    def rerank(self, query, documents, instruction=None):
        """返回按分数降序排列的 (index, document, score) 列表。"""
        scores = self.reranker.score(query, documents, instruction)
        results = [
            (index, doc, float(score))
            for index, (doc, score) in enumerate(zip(documents, scores.tolist()))
        ]
        results.sort(key=lambda item: item[2], reverse=True)
        return results

    def split_text(self, text, chunk_size, overlap_size):
        return self.embedding.split_text(text, chunk_size, overlap_size)

    def close(self):
        self.embedding.session = None
        self.reranker.session = None
