from contextlib import asynccontextmanager
import gc
import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gemma_engine import (
    EMBEDDING_DIR,
    RERANK_BACKEND,
    RERANKER_DIR,
    TextEngine,
    resolve_model_name,
)


LOGGER = logging.getLogger(__name__)

engine: Optional[TextEngine] = None


def _ensure_ready():
    if engine is None:
        raise RuntimeError("ONNX sessions are not loaded")
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    LOGGER.info("Starting up: loading ONNX models...")
    try:
        engine = TextEngine()
    except Exception as exc:
        LOGGER.exception("Failed to load ONNX models")
        raise RuntimeError(f"Failed to load ONNX models: {exc}") from exc

    yield

    engine.close()
    engine = None
    gc.collect()
    LOGGER.info("Shutdown complete, GPU resources released.")


app = FastAPI(
    title="EmbeddingGemma2 Text Service",
    description="A FastAPI service for text embeddings and reranking using ONNX EmbeddingGemma 2 models",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

embedding_model_name = resolve_model_name(EMBEDDING_DIR, "embeddinggemma-2-onnx")
reranker_model_name = (
    resolve_model_name(RERANKER_DIR, "qwen3-reranker-seq-cls-onnx")
    if RERANK_BACKEND == "cross-encoder"
    else embedding_model_name
)


class EmbedTextRequest(BaseModel):
    input: List[str]


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    instruction: Optional[str] = None


class SplitTextRequest(BaseModel):
    text: str
    chunksize: int
    overlap_size: int = 0


class SplitTextResponse(BaseModel):
    chunks: List[str]
    total_chunks: int
    model: str = embedding_model_name


class RerankResult(BaseModel):
    document: str
    score: float
    index: int


class RerankResponse(BaseModel):
    results: List[RerankResult]
    model: str = reranker_model_name


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int
    tokens_length: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbedTextResponse(BaseModel):
    data: List[EmbeddingData]
    model: str = embedding_model_name
    object: str = "list"
    usage: Usage


@app.post("/embed_text", response_model=EmbedTextResponse, tags=["Embedding"])
async def embed_text(request: EmbedTextRequest):
    try:
        current = _ensure_ready()
        embeddings, text_lengths, total_tokens = current.embed(request.input)

        data = [
            EmbeddingData(
                embedding=embedding,
                index=index,
                tokens_length=text_length,
            )
            for index, (embedding, text_length) in enumerate(zip(embeddings.tolist(), text_lengths))
        ]
        usage = Usage(prompt_tokens=total_tokens, total_tokens=total_tokens)
        return EmbedTextResponse(data=data, usage=usage)
    except Exception as exc:
        LOGGER.exception("Error during embedding")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rerank", response_model=RerankResponse, tags=["Reranking"])
async def simple_rerank(request: RerankRequest):
    try:
        current = _ensure_ready()
        ranked = current.rerank(request.query, request.documents, request.instruction)
        results = [
            RerankResult(document=doc, score=score, index=index)
            for index, doc, score in ranked
        ]
        return RerankResponse(results=results)
    except Exception as exc:
        LOGGER.exception("Error during reranking")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def SplitText(text: str, chunksize: int, overlap_size: int) -> List[str]:
    if engine is None:
        raise RuntimeError("Embedding tokenizer not loaded")
    return engine.split_text(text, chunksize, overlap_size)


@app.post("/split_text", response_model=SplitTextResponse, tags=["Text Processing"])
async def split_text_api(request: SplitTextRequest):
    try:
        chunks = SplitText(request.text, request.chunksize, request.overlap_size)
        return SplitTextResponse(chunks=chunks, total_chunks=len(chunks))
    except Exception as exc:
        LOGGER.exception("Error during text splitting")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title + " - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@2.1.0/bundles/redoc.standalone.js",
    )


@app.get("/oauth2-redirect", include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy" if engine is not None else "loading",
        "embedding_model": embedding_model_name,
        "reranker_model": reranker_model_name,
        "device": engine.provider if engine is not None else None,
        "service": "EmbeddingGemma2 Text Service",
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
